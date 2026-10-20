#!/usr/bin/env python3
"""Cut clock-time intervals from an inbox WAV into labeled dataset clips."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


LABELS = ("siren", "explosion", "uav", "background", "unknown")
MANIFEST_FIELDS = (
    "source_file",
    "output_file",
    "label",
    "start_time",
    "end_time",
    "start_offset_seconds",
    "end_offset_seconds",
)


def recording_start(source: Path, timezone: ZoneInfo) -> datetime:
    metadata_path = source.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(
            f"Missing recorder sidecar: {metadata_path}. Use the updated recorder; "
            "the .wav and .json files must keep the same basename."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(metadata["started_at"])
    if started_at.tzinfo is None:
        raise ValueError(f"Recorder start time has no timezone: {metadata['started_at']}")
    return started_at.astimezone(timezone)


def parse_time(value: str, anchor: datetime, timezone: ZoneInfo) -> datetime:
    """Parse an @-prefixed file offset, local clock time, or ISO-8601 timestamp."""
    if value.startswith("@"):
        fields = value[1:].split(":")
        if len(fields) == 2:
            hours, minutes, seconds = 0, *map(int, fields)
        elif len(fields) == 3:
            hours, minutes, seconds = map(int, fields)
        else:
            raise ValueError(f"Invalid file offset {value!r}; use MM:SS or HH:MM:SS")
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError(f"Invalid file offset {value!r}; use MM:SS or HH:MM:SS")
        return anchor + timedelta(hours=hours, minutes=minutes, seconds=seconds)

    if "T" in value:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone)
        return parsed.astimezone(timezone)

    formats = ("%H:%M:%S", "%H:%M")
    clock = None
    for format_string in formats:
        try:
            clock = datetime.strptime(value, format_string).time()
            break
        except ValueError:
            continue
    if clock is None:
        raise ValueError(
            f"Invalid time {value!r}; use MM:SS or HH:MM:SS with -f, "
            "HH:MM[:SS] for local clock time, or ISO-8601"
        )

    result = datetime.combine(anchor.date(), clock, tzinfo=timezone)
    if result < anchor:
        result += timedelta(days=1)
    return result


def annotate(
    source: Path,
    segments: list[tuple[str, str, str]],
    timezone_name: str,
    dataset_root: Path | None = None,
) -> None:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"Recording not found: {source}")
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone_name}") from error

    all_relative = all(s.startswith("@") and e.startswith("@") for _, s, e in segments)
    if all_relative:
        # File-relative mode: no sidecar needed, use epoch as dummy anchor
        started_at = datetime(2000, 1, 1, tzinfo=timezone)
    else:
        started_at = recording_start(source, timezone)
    dataset_root = dataset_root or Path(__file__).resolve().parent
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "annotations.csv"
    outputs = []

    with wave.open(str(source), "rb") as input_wav:
        if input_wav.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files are supported")
        params = input_wav.getparams()
        duration = input_wav.getnframes() / params.framerate

        for label, start_text, end_text in segments:
            if label not in LABELS:
                raise ValueError(f"Unknown label {label!r}; choose from: {', '.join(LABELS)}")
            relative = start_text.startswith("@") and end_text.startswith("@")
            if start_text.startswith("@") != end_text.startswith("@"):
                raise ValueError("Both segment bounds must use the same time mode")
            start_at = parse_time(start_text, started_at, timezone)
            end_at = parse_time(end_text, started_at, timezone)
            if end_at <= start_at:
                if relative:
                    raise ValueError("File-relative segment end must be after its start")
                end_at += timedelta(days=1)
            start_offset = (start_at - started_at).total_seconds()
            end_offset = (end_at - started_at).total_seconds()
            start_frame = round(start_offset * params.framerate)
            end_frame = round(end_offset * params.framerate)
            if start_frame < 0 or end_frame > input_wav.getnframes() or end_frame <= start_frame:
                raise ValueError(
                    f"Interval {start_text}–{end_text} is outside the recording "
                    f"(0–{duration:.1f}s from its start at {started_at.isoformat(timespec='seconds')})"
                )

            output_name = (
                f"{source.stem}__{label}__{round(start_offset)}-"
                f"{round(end_offset)}s.wav"
            )
            output_path = dataset_root / label / output_name
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            input_wav.setpos(start_frame)
            frames = input_wav.readframes(end_frame - start_frame)
            with wave.open(str(output_path), "wb") as output_wav:
                output_wav.setparams(params)
                output_wav.writeframes(frames)

            outputs.append(
                {
                    "source_file": str(source.relative_to(dataset_root.parent)),
                    "output_file": str(output_path.relative_to(dataset_root.parent)),
                    "label": label,
                    "start_time": start_at.isoformat(timespec="seconds"),
                    "end_time": end_at.isoformat(timespec="seconds"),
                    "start_offset_seconds": f"{start_offset:.3f}",
                    "end_offset_seconds": f"{end_offset:.3f}",
                }
            )

    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        if manifest_file.tell() == 0:
            writer.writeheader()
        writer.writerows(outputs)

    for row in outputs:
        print(f"{row['start_time']}–{row['end_time']} [{row['label']}] -> {row['output_file']}")
    print(f"Annotation manifest: {manifest_path}")


def main() -> int:
    # argparse treats a token such as -f after --segment as a new option, so
    # rewrite the requested local syntax into an internal marker first.
    argv = sys.argv[1:]
    normalized_argv = []
    index = 0
    while index < len(argv):
        if argv[index] == "--segment" and index + 1 < len(argv) and argv[index + 1] == "-f":
            if index + 4 >= len(argv):
                raise SystemExit("--segment -f requires LABEL FROM TO")
            label, start, end = argv[index + 2:index + 5]
            normalized_argv.extend(("--segment", label, f"@{start}", f"@{end}"))
            index += 5
        else:
            normalized_argv.append(argv[index])
            index += 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="inbox WAV; its matching .json sidecar is required")
    parser.add_argument(
        "--timezone",
        default="Europe/Moscow",
        help="timezone used for the clock times you enter (default: Europe/Moscow)",
    )
    parser.add_argument(
        "--segment",
        nargs=3,
        action="append",
        metavar=("LABEL", "FROM", "TO"),
        required=True,
        help="clock interval, or file-relative interval with -f after --segment; repeatable",
    )
    args = parser.parse_args(normalized_argv)
    try:
        annotate(args.recording, [tuple(item) for item in args.segment], args.timezone)
    except (OSError, ValueError, wave.Error, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
