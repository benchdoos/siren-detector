#!/usr/bin/env python3
"""
Convert Label Studio export JSON to WAV segments + annotations.csv.

After annotating in Label Studio:
  Project → Export → JSON → save as label-studio/export.json

Then run:
  python ml/ls_export.py

Each labeled region is sliced from the source WAV and saved to dataset/<label>/.
New rows are appended to dataset/annotations.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import wave
from pathlib import Path
from urllib.parse import unquote

import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"
ANNOTATIONS_CSV = DATASET / "annotations.csv"
LABELS = ("siren", "explosion", "motorcycle", "background", "unknown")

AUDIO_BASE_URL = "http://localhost:8081"


def ls_url_to_path(url: str) -> Path:
    """Convert Label Studio audio URL back to an absolute path."""
    rel = unquote(url.removeprefix(AUDIO_BASE_URL + "/"))
    return DATASET / rel


def output_filename(source: Path, label: str, start: float, end: float) -> Path:
    stem = source.stem
    name = f"{stem}__{label}__{int(start)}-{int(end)}s.wav"
    return DATASET / label / name


def slice_wav(source: Path, start: float, end: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    s = int(start * rate)
    e = int(end * rate)
    chunk = audio[s:e]
    sf.write(str(dest), chunk, rate, subtype="PCM_16")


def load_existing_outputs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["output_file"] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        default=REPO_ROOT / "label-studio" / "export.json",
    )
    args = parser.parse_args()

    if not args.export.exists():
        raise SystemExit(
            f"Export file not found: {args.export}\n"
            "In Label Studio: Project → Export → JSON"
        )

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    existing = load_existing_outputs(ANNOTATIONS_CSV)

    new_rows: list[dict] = []
    skipped = 0

    for task in tasks:
        audio_url = task.get("data", {}).get("audio", "")
        if not audio_url.startswith(AUDIO_BASE_URL):
            continue
        source = ls_url_to_path(audio_url)
        if not source.exists():
            print(f"  SKIP (source not found): {source}")
            continue

        for annotation in task.get("annotations", []):
            for region in annotation.get("result", []):
                if region.get("type") != "labels":
                    continue
                val = region.get("value", {})
                labels = val.get("labels", [])
                if not labels:
                    continue
                label = labels[0]
                if label not in LABELS:
                    print(f"  SKIP unknown label '{label}'")
                    continue
                start = float(val["start"])
                end = float(val["end"])
                if end <= start:
                    continue

                dest = output_filename(source, label, start, end)
                rel_dest = str(dest.relative_to(REPO_ROOT))
                rel_src = str(source.relative_to(REPO_ROOT))

                if rel_dest in existing:
                    skipped += 1
                    continue

                slice_wav(source, start, end, dest)
                new_rows.append(
                    {
                        "source_file": rel_src,
                        "output_file": rel_dest,
                        "label": label,
                        "start_time": "",
                        "end_time": "",
                        "start_offset_seconds": f"{start:.3f}",
                        "end_offset_seconds": f"{end:.3f}",
                    }
                )
                existing.add(rel_dest)
                print(f"  {label:12s}  {start:.1f}–{end:.1f}s  →  {dest.name}")

    if new_rows:
        write_header = not ANNOTATIONS_CSV.exists()
        with ANNOTATIONS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"\nAdded {len(new_rows)} segments to {ANNOTATIONS_CSV}")
    else:
        print("\nNo new segments.")
    if skipped:
        print(f"Skipped {skipped} already-existing entries.")


if __name__ == "__main__":
    main()
