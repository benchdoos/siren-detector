#!/usr/bin/env python3
"""Run pretrained YAMNet over project recordings and save file/frame scores."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow_hub as hub


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
FRAME_HOP_SECONDS = 0.48
FRAME_HOP_SAMPLES = round(SAMPLE_RATE * FRAME_HOP_SECONDS)
WINDOW_SAMPLES = round(SAMPLE_RATE * 0.96)
CHUNK_FRAME_COUNT = 120  # 57.6 seconds; keep memory bounded for long WAVs.
SCORE_LABELS = (
    "Civil defense siren",
    "Siren",
    "Alarm",
    "Explosion",
    "Motorcycle",
    "Vehicle",
    "Speech",
    "Rain",
    "Wind",
    "Engine",
    "Silence",
)
DEFAULT_CATEGORIES = ("siren", "background")


def load_mono_resampled(path: Path, target_start: int, target_end: int) -> tuple[np.ndarray, int]:
    """Read just one model chunk, resampling with guard samples at both edges."""
    with sf.SoundFile(path) as source:
        source_rate = source.samplerate
        total_source_frames = len(source)
        duration = total_source_frames / source_rate
        guard_source = max(32, round(source_rate * 0.05))
        start_seconds = target_start / SAMPLE_RATE
        end_seconds = target_end / SAMPLE_RATE
        source_start = max(0, int(start_seconds * source_rate) - guard_source)
        source_end = min(
            total_source_frames,
            int(np.ceil(end_seconds * source_rate)) + guard_source,
        )
        source.seek(source_start)
        audio = source.read(source_end - source_start, dtype="float32", always_2d=True)

    audio = audio.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        divisor = np.gcd(source_rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio, SAMPLE_RATE // divisor, source_rate // divisor
        ).astype(np.float32)

    chunk_start = round(source_start * SAMPLE_RATE / source_rate)
    trim_start = target_start - chunk_start
    wanted = target_end - target_start
    audio = audio[trim_start : trim_start + wanted]
    if len(audio) < wanted:
        audio = np.pad(audio, (0, wanted - len(audio)))
    return np.clip(audio, -1.0, 1.0), source_rate


def find_recordings(dataset: Path, requested: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    candidates = requested or [dataset / category for category in DEFAULT_CATEGORIES]
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if not candidate.exists():
            raise SystemExit(f"Input path does not exist: {candidate}")
        if candidate.is_file():
            if candidate.suffix.lower() == ".wav":
                paths.add(candidate)
            continue
        paths.update(path.resolve() for path in candidate.rglob("*.wav") if not path.name.startswith("."))
    if not paths:
        raise SystemExit("No WAV recordings found")
    return sorted(paths)


def category_for(path: Path, dataset: Path) -> str:
    try:
        relative = path.relative_to(dataset)
    except ValueError:
        return path.parent.name
    return relative.parts[0] if len(relative.parts) > 1 else "unlabeled"


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def score_file(
    path: Path,
    category: str,
    model,
    class_names: list[str],
    class_indices: dict[str, int],
    frame_writer: csv.DictWriter,
) -> dict[str, object]:
    info = sf.info(path)
    duration = info.frames / info.samplerate
    target_labels = [label for label in SCORE_LABELS if label in class_indices]
    target_indices = [class_indices[label] for label in target_labels]
    sums = np.zeros(len(class_names), dtype=np.float64)
    maxima = np.zeros(len(class_names), dtype=np.float32)
    target_frames = {label: [] for label in target_labels}
    frame_count = 0
    last_frame_index = -1
    total_target_samples = round(duration * SAMPLE_RATE)
    step_samples = CHUNK_FRAME_COUNT * FRAME_HOP_SAMPLES

    for chunk_number, target_start in enumerate(range(0, total_target_samples, step_samples), start=1):
        target_end = min(
            total_target_samples,
            target_start + step_samples + WINDOW_SAMPLES,
        )
        waveform, _ = load_mono_resampled(path, target_start, target_end)
        score_tensor, _, _ = model(waveform)
        chunk_scores = score_tensor.numpy()
        first_frame_index = round(target_start / FRAME_HOP_SAMPLES)

        for local_index, scores in enumerate(chunk_scores):
            frame_index = first_frame_index + local_index
            timestamp = frame_index * FRAME_HOP_SECONDS
            if frame_index <= last_frame_index or timestamp >= duration:
                continue
            last_frame_index = frame_index
            sums += scores
            maxima = np.maximum(maxima, scores)
            for label, class_index in zip(target_labels, target_indices):
                target_frames[label].append(float(scores[class_index]))
            top_index = int(np.argmax(scores))
            frame_row: dict[str, object] = {
                "file": str(path),
                "category": category,
                "window_start_seconds": f"{timestamp:.2f}",
                "top1_label": class_names[top_index],
                "top1_score": f"{scores[top_index]:.6f}",
            }
            frame_row.update(
                {slug(label): f"{scores[index]:.6f}" for label, index in zip(target_labels, target_indices)}
            )
            frame_writer.writerow(frame_row)
            frame_count += 1

        if chunk_number % 10 == 0:
            processed_seconds = min(duration, target_start / SAMPLE_RATE + step_samples / SAMPLE_RATE)
            print(f"  {path.name}: processed about {processed_seconds:.0f}/{duration:.0f}s", flush=True)

    if frame_count == 0:
        raise ValueError("YAMNet returned no frames for this recording")

    mean_scores = sums / frame_count
    best_indices = np.argsort(mean_scores)[-5:][::-1]
    result: dict[str, object] = {
        "file": str(path),
        "category": category,
        "duration_seconds": f"{duration:.3f}",
        "source_sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "frames_scored": frame_count,
        "top_mean_labels": "; ".join(
            f"{class_names[index]}={mean_scores[index]:.4f}" for index in best_indices
        ),
    }
    for label, class_index in zip(target_labels, target_indices):
        values = np.asarray(target_frames[label], dtype=np.float32)
        result[f"{slug(label)}_mean"] = f"{values.mean():.6f}"
        result[f"{slug(label)}_max"] = f"{values.max():.6f}"
        result[f"{slug(label)}_p95"] = f"{np.percentile(values, 95):.6f}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="optional WAV files or directories; by default scans dataset/siren and dataset/background",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="dataset root used for default folders and category names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "recordings-yamnet",
        help="directory for summary.csv and frame_scores.csv",
    )
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    recordings = find_recordings(dataset, args.paths)

    print(f"Loading pretrained YAMNet from {MODEL_URL} ...", flush=True)
    model = hub.load(MODEL_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as class_map_file:
        class_names = [row["display_name"] for row in csv.DictReader(class_map_file)]
    class_indices = {name: index for index, name in enumerate(class_names)}
    missing = [label for label in SCORE_LABELS if label not in class_indices]
    if missing:
        print(f"Note: YAMNet class map lacks: {', '.join(missing)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "frame_scores.csv"
    summary_path = output_dir / "summary.csv"
    target_labels = [label for label in SCORE_LABELS if label in class_indices]
    frame_fields = [
        "file",
        "category",
        "window_start_seconds",
        "top1_label",
        "top1_score",
        *(slug(label) for label in target_labels),
    ]
    summary_fields = [
        "file",
        "category",
        "duration_seconds",
        "source_sample_rate_hz",
        "channels",
        "frames_scored",
        "top_mean_labels",
        *(
            field
            for label in target_labels
            for field in (f"{slug(label)}_mean", f"{slug(label)}_max", f"{slug(label)}_p95")
        ),
        "error",
    ]

    print(f"Found {len(recordings)} WAV files; writing reports under {output_dir}", flush=True)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file, frame_path.open(
        "w", newline="", encoding="utf-8"
    ) as frame_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        frame_writer = csv.DictWriter(frame_file, fieldnames=frame_fields)
        summary_writer.writeheader()
        frame_writer.writeheader()
        for index, path in enumerate(recordings, start=1):
            category = category_for(path, dataset)
            print(f"[{index}/{len(recordings)}] {path}", flush=True)
            try:
                result = score_file(path, category, model, class_names, class_indices, frame_writer)
                result["error"] = ""
                summary_writer.writerow(result)
                summary_file.flush()
                frame_file.flush()
                print(f"  top mean labels: {result['top_mean_labels']}", flush=True)
            except Exception as error:  # Keep the batch moving; record each failed input.
                print(f"  ERROR: {error}", flush=True)
                summary_writer.writerow(
                    {
                        "file": str(path),
                        "category": category,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                summary_file.flush()

    print(f"Per-file summary: {summary_path}")
    print(f"Frame-by-frame scores: {frame_path}")
    print("Scores are pretrained YAMNet outputs, not calibrated probabilities or detector thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
