#!/usr/bin/env python3
"""
Prepare Label Studio import JSON from inbox WAV files + YAMNet windows.csv.

YAMNet predictions above --threshold are merged into segments and added as
pre-annotations so reviewers only need to confirm or correct them.

Usage:
  # all files currently in windows.csv
  python ml/ls_import.py

  # specific files
  python ml/ls_import.py --files dataset/inbox/foo.wav dataset/inbox/bar.wav

Output: label-studio/import.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_CSV = Path(__file__).resolve().parent / "results" / "own-dataset-yamnet-windows.csv"
OUTPUT = REPO_ROOT / "label-studio" / "import.json"

# URL prefix Label Studio uses when serving local files
AUDIO_BASE_URL = "http://localhost:8081"

SIREN_THRESHOLD = 0.10
MERGE_GAP_SECONDS = 2.0   # merge siren detections closer than this
CONTEXT_SECONDS = 0.5     # extend each segment by this on both sides


def load_windows(windows_csv: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Return {filename: [(time_s, score, top1_class), ...]}"""
    data: dict[str, list] = defaultdict(list)
    with windows_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["file"]].append(
                (float(row["window_time_s"]), float(row["siren_score"]), row["top1_class"])
            )
    return data


def merge_segments(
    windows: list[tuple[float, float, str]],
    threshold: float,
    gap: float,
    context: float,
    file_duration: float,
) -> list[tuple[float, float, float]]:
    """Return [(start, end, max_score)] merged siren segments."""
    hits = [(t, s) for t, s, _ in windows if s >= threshold]
    if not hits:
        return []

    segments: list[tuple[float, float, float]] = []
    seg_start, seg_end, seg_max = hits[0][0], hits[0][0] + 0.96, hits[0][1]

    for t, s in hits[1:]:
        if t - seg_end <= gap:
            seg_end = max(seg_end, t + 0.96)
            seg_max = max(seg_max, s)
        else:
            segments.append((seg_start, seg_end, seg_max))
            seg_start, seg_end, seg_max = t, t + 0.96, s
    segments.append((seg_start, seg_end, seg_max))

    # add context, clamp to file bounds
    return [
        (max(0.0, s - context), min(file_duration, e + context), m)
        for s, e, m in segments
    ]


def file_duration_seconds(path: Path) -> float:
    import soundfile as sf
    return sf.info(path).duration


def build_task(wav_path: Path, segments: list[tuple[float, float, float]]) -> dict:
    # path relative to dataset root (= Label Studio files root)
    rel = wav_path.relative_to(REPO_ROOT / "dataset")
    audio_url = AUDIO_BASE_URL + "/" + str(rel).replace(chr(92), "/")

    result = [
        {
            "from_name": "label",
            "to_name": "audio",
            "type": "labels",
            "value": {
                "start": round(start, 3),
                "end": round(end, 3),
                "labels": ["siren"],
                "score": round(score, 3),
            },
        }
        for start, end, score in segments
    ]

    task: dict = {"data": {"audio": audio_url, "filename": wav_path.name}}
    if result:
        task["predictions"] = [{"model_version": "yamnet", "result": result}]
    return task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="WAV files to include. Default: all files found in windows.csv.",
    )
    parser.add_argument("--threshold", type=float, default=SIREN_THRESHOLD)
    args = parser.parse_args()

    if not WINDOWS_CSV.exists():
        raise SystemExit(
            f"windows.csv not found at {WINDOWS_CSV}\n"
            "Run: ml/.venv/bin/python ml/score_dataset.py --labels inbox siren background"
        )

    windows_by_file = load_windows(WINDOWS_CSV)

    if args.files:
        wav_files = args.files
    else:
        wav_files = [
            REPO_ROOT / "dataset" / "inbox" / name
            for name in windows_by_file
            if (REPO_ROOT / "dataset" / "inbox" / name).exists()
        ]

    if not wav_files:
        raise SystemExit("No WAV files found.")

    tasks = []
    for wav_path in sorted(wav_files):
        if not wav_path.exists():
            print(f"  SKIP (not found): {wav_path}")
            continue
        fname = wav_path.name
        windows = windows_by_file.get(fname, [])
        duration = file_duration_seconds(wav_path)
        segments = merge_segments(windows, args.threshold, MERGE_GAP_SECONDS, CONTEXT_SECONDS, duration)
        tasks.append(build_task(wav_path, segments))
        print(f"  {fname}  →  {len(segments)} pre-annotated segment(s)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(tasks)} tasks → {OUTPUT}")
    print("Import this file into Label Studio: Projects → Import → JSON.")


if __name__ == "__main__":
    main()
