#!/usr/bin/env python3
"""
Interactive curation of ESC-50 siren clips.

Plays each clip and asks: keep (y), skip (n), quit (q).
Kept files are copied to dataset/external/esc50_curated/siren/.

Usage:
    python3 dataset/curate_esc50_sirens.py [--category siren]
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ESC50_AUDIO = REPO_ROOT / "dataset" / "external" / "esc50" / "audio"
ESC50_META  = REPO_ROOT / "dataset" / "external" / "esc50" / "meta" / "esc50.csv"
OUT_DIR     = REPO_ROOT / "dataset" / "external" / "esc50_curated"


def play(path: Path) -> subprocess.Popen:
    return subprocess.Popen(["afplay", str(path)])


def stop(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="siren",
                        help="ESC-50 category to curate (default: siren)")
    args = parser.parse_args()

    with open(ESC50_META) as f:
        files = [row["filename"] for row in csv.DictReader(f)
                 if row["category"] == args.category]

    if not files:
        print(f"No files found for category '{args.category}'")
        sys.exit(1)

    out_dir = OUT_DIR / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    already = {p.name for p in out_dir.glob("*.wav")}
    kept = len(already)

    print(f"\nCategory: {args.category}  |  {len(files)} files total  |  {kept} already kept")
    print("Controls: y = keep,  n = skip,  r = replay,  q = quit\n")

    for i, fname in enumerate(files, 1):
        src = ESC50_AUDIO / fname
        if not src.exists():
            print(f"[{i}/{len(files)}] MISSING: {fname}")
            continue

        if fname in already:
            print(f"[{i}/{len(files)}] already kept: {fname}")
            continue

        print(f"[{i}/{len(files)}] {fname}  ", end="", flush=True)
        proc = play(src)

        while True:
            key = input("keep? [y/n/r/q]: ").strip().lower()
            if key == "y":
                stop(proc)
                shutil.copy2(src, out_dir / fname)
                kept += 1
                print(f"  ✓ kept  ({kept} total)")
                break
            elif key == "n":
                stop(proc)
                print("  — skipped")
                break
            elif key == "r":
                stop(proc)
                proc = play(src)
            elif key == "q":
                stop(proc)
                print(f"\nDone. Kept {kept}/{i} files → {out_dir}")
                sys.exit(0)
            else:
                print("  y / n / r / q")

    print(f"\nDone. Kept {kept}/{len(files)} files → {out_dir}")


if __name__ == "__main__":
    main()
