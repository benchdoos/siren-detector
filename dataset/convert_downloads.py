#!/usr/bin/env python3
"""Convert downloaded audio files to dataset format (16 kHz, mono, 5-sec WAV).

Drop files (any format ffmpeg can read) into dataset/inbox/downloads/,
then run:

    python3 dataset/convert_downloads.py explosion
    python3 dataset/convert_downloads.py uav

Each file is split into non-overlapping 5-second chunks and saved to
dataset/<label>/ with a unique name. Originals are moved to
dataset/inbox/downloads/done/ after processing.

Requires: ffmpeg
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parent
DOWNLOADS = DATASET / "inbox" / "downloads"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 5


def convert_and_split(src: Path, label: str) -> int:
    """Convert src to 16 kHz mono WAV chunks. Returns number of chunks created."""
    out_dir = DATASET / label
    out_dir.mkdir(exist_ok=True)

    # probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(src)],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        print(f"  skip {src.name}: can't read duration", file=sys.stderr)
        return 0

    stem = src.stem.replace(" ", "_")
    chunks = 0
    offset = 0.0

    while offset + 1.0 < duration:  # at least 1 second remaining
        length = min(CHUNK_SECONDS, duration - offset)
        out_name = f"dl_{stem}_{chunks:03d}.wav"
        out_path = out_dir / out_name

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{offset:.2f}",
            "-t", f"{length:.2f}",
            "-i", str(src),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr.strip()}", file=sys.stderr)
            break

        chunks += 1
        offset += CHUNK_SECONDS

    return chunks


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <label>", file=sys.stderr)
        print("  label: explosion, uav, siren, background, unknown", file=sys.stderr)
        return 1

    label = sys.argv[1]
    if not DOWNLOADS.exists():
        DOWNLOADS.mkdir(parents=True)
        print(f"Created {DOWNLOADS}")
        print(f"Drop audio files there, then re-run: python3 {sys.argv[0]} {label}")
        return 0

    files = sorted(
        f for f in DOWNLOADS.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.name != ".gitkeep"
    )
    if not files:
        print(f"No files in {DOWNLOADS}")
        return 0

    done_dir = DOWNLOADS / "done"
    done_dir.mkdir(exist_ok=True)

    total = 0
    for f in files:
        print(f"Processing {f.name}...")
        n = convert_and_split(f, label)
        if n > 0:
            shutil.move(str(f), done_dir / f.name)
            print(f"  {n} chunks → dataset/{label}/")
            total += n
        else:
            print(f"  skipped")

    print(f"\nTotal: {total} chunks saved to dataset/{label}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())