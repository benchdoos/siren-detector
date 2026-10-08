#!/usr/bin/env python3
"""Evaluate pretrained YAMNet on the curated ESC-50 subset, grouped by fold."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
CATEGORIES = {
    "siren": ("Siren", "Civil defense siren", "Ambulance (siren)", "Police car (siren)", "Fire engine, fire truck (siren)"),
    "rain": ("Rain", "Rain on surface"),
    "wind": ("Wind", "Wind noise (microphone)"),
    "washing_machine": (),  # YAMNet has no washing-machine class.
    "keyboard_typing": ("Typing", "Typewriter"),
    "engine": ("Engine", "Light engine (high frequency)", "Medium engine (mid frequency)", "Heavy engine (low frequency)", "Engine starting", "Engine knocking", "Aircraft engine", "Jet engine"),
    "fireworks": ("Fireworks",),
}


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = np.gcd(sample_rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return float("nan")
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def evaluate(dataset: Path, output: Path) -> int:
    metadata_path = dataset / "meta" / "esc50.csv"
    audio_dir = dataset / "audio"
    with metadata_path.open(newline="", encoding="utf-8") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["category"] in CATEGORIES and (audio_dir / row["filename"]).is_file()
        ]
    expected = len(CATEGORIES) * 40
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} selected files, found {len(rows)}")

    print(f"Loading pretrained YAMNet from {MODEL_URL} ...", flush=True)
    model = hub.load(MODEL_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as class_map_file:
        class_names = [row["display_name"] for row in csv.DictReader(class_map_file)]
    class_indices = {name: index for index, name in enumerate(class_names)}
    supported_labels = {
        category: [class_indices[name] for name in candidates if name in class_indices]
        for category, candidates in CATEGORIES.items()
    }
    if not supported_labels["siren"]:
        raise SystemExit("YAMNet class map does not contain a siren label")
    siren_index = class_indices["Siren"]

    results = []
    per_fold: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        waveform = load_audio(audio_dir / row["filename"])
        scores, _, _ = model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        predicted = class_names[int(np.argmax(mean_scores))]
        target_indices = supported_labels[row["category"]]
        siren_score = float(mean_scores[siren_index])
        is_siren = int(row["category"] == "siren")
        per_fold[row["fold"]].append((is_siren, siren_score))
        results.append(
            {
                "filename": row["filename"],
                "fold": row["fold"],
                "target": row["category"],
                "predicted_yamnet_top1": predicted,
                "target_label_top1_match": (
                    int(int(np.argmax(mean_scores)) in target_indices)
                    if target_indices
                    else ""
                ),
                "siren_score": f"{siren_score:.8f}",
            }
        )
        if index % 40 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} clips", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    scored = [row for row in results if row["target_label_top1_match"] != ""]
    exact = sum(row["target_label_top1_match"] for row in scored)
    print("\nYAMNet top-1 match against supported label groups:")
    for category in CATEGORIES:
        category_rows = [row for row in results if row["target"] == category]
        category_scored = [row for row in category_rows if row["target_label_top1_match"] != ""]
        if category_scored:
            correct = sum(row["target_label_top1_match"] for row in category_scored)
            print(f"  {category:18s} {correct:2d}/{len(category_scored):2d} ({correct / len(category_scored):.1%})")
        else:
            print(f"  {category:18s} n/a (no corresponding YAMNet class)")
    print(f"  {'all supported':18s} {exact:3d}/{len(scored):3d} ({exact / len(scored):.1%})")
    print("\nSiren-vs-other selected clips (YAMNet Siren score, ROC AUC):")
    for fold in sorted(per_fold, key=int):
        fold_rows = per_fold[fold]
        auc = roc_auc([item[0] for item in fold_rows], [item[1] for item in fold_rows])
        print(f"  fold {fold}: {auc:.3f} ({len(fold_rows)} clips)")
    all_rows = [item for fold_rows in per_fold.values() for item in fold_rows]
    print(f"  all folds: {roc_auc([item[0] for item in all_rows], [item[1] for item in all_rows]):.3f} ({len(all_rows)} clips)")
    print(f"\nPer-file scores: {output}")
    print("These are pretrained-model sanity checks, not performance estimates for the INMP441 device.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "external" / "esc50",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "esc50-yamnet.csv",
    )
    args = parser.parse_args()
    return evaluate(args.dataset, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
