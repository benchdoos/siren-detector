#!/usr/bin/env python3
"""Run YAMNet on every WAV in dataset/siren/ and dataset/background/.

Large files are processed in chunks to avoid loading them into memory at once.

Produces two CSVs in ml/results/:
  own-dataset-yamnet-windows.csv  — one row per YAMNet time window
  own-dataset-yamnet-summary.csv  — one row per file
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow_hub as hub


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
YAMNET_HOP_SECONDS = 0.48

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def resample(audio: np.ndarray, from_rate: int) -> np.ndarray:
    if from_rate == SAMPLE_RATE:
        return audio
    divisor = np.gcd(from_rate, SAMPLE_RATE)
    return scipy.signal.resample_poly(
        audio, SAMPLE_RATE // divisor, from_rate // divisor
    ).astype(np.float32)


def score_file(
    model,
    class_names: list[str],
    siren_index: int,
    path: Path,
    chunk_seconds: float,
    max_seconds: float | None,
) -> tuple[list[dict], dict]:
    """Return (window_rows, summary_row) for a single file."""
    info = sf.info(path)
    src_rate = info.samplerate
    total_duration = info.duration
    if max_seconds is not None:
        total_duration = min(total_duration, max_seconds)

    chunk_src_samples = int(chunk_seconds * src_rate)
    max_src_samples = int(total_duration * src_rate) if max_seconds else None

    window_rows: list[dict] = []
    all_siren_scores: list[float] = []
    all_top1: list[str] = []
    samples_read = 0

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=chunk_src_samples, dtype="float32", always_2d=True):
            if max_src_samples is not None and samples_read >= max_src_samples:
                break
            if max_src_samples is not None:
                remaining = max_src_samples - samples_read
                block = block[:remaining]

            chunk_offset_seconds = samples_read / src_rate
            samples_read += len(block)

            audio = block.mean(axis=1)
            audio = resample(audio, src_rate)
            audio = np.clip(audio, -1.0, 1.0)

            scores, _, _ = model(audio)
            scores_np = scores.numpy()

            siren_scores = scores_np[:, siren_index]
            top1_indices = np.argmax(scores_np, axis=1)

            for w_idx, (s_score, top1_idx) in enumerate(zip(siren_scores, top1_indices)):
                t = round(chunk_offset_seconds + w_idx * YAMNET_HOP_SECONDS, 3)
                top1_name = class_names[int(top1_idx)]
                window_rows.append(
                    {
                        "file": path.name,
                        "label": path.parent.name,
                        "window_time_s": t,
                        "siren_score": f"{s_score:.6f}",
                        "top1_class": top1_name,
                    }
                )
                all_siren_scores.append(float(s_score))
                all_top1.append(top1_name)

    siren_arr = np.array(all_siren_scores)
    top1_mode = Counter(all_top1).most_common(1)[0][0] if all_top1 else ""
    summary = {
        "file": path.name,
        "label": path.parent.name,
        "duration_s": f"{total_duration:.1f}",
        "windows": len(siren_arr),
        "siren_mean": f"{siren_arr.mean():.4f}",
        "siren_max": f"{siren_arr.max():.4f}",
        "siren_min": f"{siren_arr.min():.4f}",
        "windows_above_0.3": int(np.sum(siren_arr >= 0.3)),
        "windows_above_0.5": int(np.sum(siren_arr >= 0.5)),
        "top1_mode": top1_mode,
    }
    return window_rows, summary


def collect_files(labels: list[str]) -> list[Path]:
    files: list[Path] = []
    for label in labels:
        files.extend(sorted((DATASET_DIR / label).glob("*.wav")))
    return files


def print_summary(summary_rows: list[dict]) -> None:
    print("\n=== Per-file summary ===")
    col = f"{'label':10s}  {'mean':>6s}  {'max':>6s}  {'≥0.3':>6s}  {'≥0.5':>6s}  file"
    print(col)
    print("-" * len(col))
    for row in summary_rows:
        print(
            f"{row['label']:10s}  {row['siren_mean']:>6s}  {row['siren_max']:>6s}"
            f"  {row['windows_above_0.3']:>6}  {row['windows_above_0.5']:>6}  {row['file']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        metavar="N",
        help="Process files in chunks of N seconds (default: 60). Keeps memory flat.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        metavar="N",
        help="Stop each file after N seconds total.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["siren", "background"],
    )
    args = parser.parse_args()

    files = collect_files(args.labels)
    if not files:
        raise SystemExit("No WAV files found.")

    print(f"Found {len(files)} WAV files  |  chunk={args.chunk_seconds}s"
          + (f"  max={args.max_seconds}s" if args.max_seconds else ""))
    print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
    model = hub.load(MODEL_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    siren_index = next((i for i, n in enumerate(class_names) if n == "Siren"), None)
    if siren_index is None:
        raise SystemExit("YAMNet class map has no 'Siren' class.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    windows_path = RESULTS_DIR / "own-dataset-yamnet-windows.csv"
    summary_path = RESULTS_DIR / "own-dataset-yamnet-summary.csv"

    window_fieldnames = ["file", "label", "window_time_s", "siren_score", "top1_class"]
    summary_rows: list[dict] = []

    with windows_path.open("w", newline="", encoding="utf-8") as wf:
        window_writer = csv.DictWriter(wf, fieldnames=window_fieldnames)
        window_writer.writeheader()

        for idx, path in enumerate(files, start=1):
            info = sf.info(path)
            effective = min(info.duration, args.max_seconds) if args.max_seconds else info.duration
            print(
                f"[{idx}/{len(files)}] {path.parent.name:10s}  {path.name}"
                f"  ({effective:.0f}s)",
                flush=True,
            )
            window_rows, summary = score_file(
                model, class_names, siren_index, path,
                args.chunk_seconds, args.max_seconds,
            )
            window_writer.writerows(window_rows)
            summary_rows.append(summary)

    with summary_path.open("w", newline="", encoding="utf-8") as sf_out:
        writer = csv.DictWriter(sf_out, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print_summary(summary_rows)
    print(f"\nWindows: {windows_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
