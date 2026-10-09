#!/usr/bin/env python3
"""
Train a small MLP classifier on top of frozen YAMNet embeddings.

Architecture:
  WAV → YAMNet (frozen) → embeddings (1024-dim / window)
                               ↓
                          MLP head (256 → 128 → N classes)
                               ↓
                        siren / background / ...

Train/val split is by source recording, not by random windows.

Usage:
  ml/.venv/bin/python ml/train_classifier.py

Outputs:
  ml/results/classifier.keras
  ml/results/classifier.tflite
  ml/results/classifier_labels.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"
RESULTS = Path(__file__).resolve().parent / "results"

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
LABELS = ["siren", "background"]   # unknown/explosion/motorcycle have too few samples
VAL_SPLIT = 0.2
MAX_WINDOWS_PER_FILE = 300          # cap long files so one recording doesn't dominate
RANDOM_SEED = 42


# ── audio helpers ──────────────────────────────────────────────────────────────

def load_mono_16k(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = np.gcd(rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio, SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def source_name(filename: str) -> str:
    """'20261009-070807__siren__1918-2124s.wav' → '20261009-070807'"""
    return Path(filename).stem.split("__")[0]


# ── embedding extraction ────────────────────────────────────────────────────────

def extract_embeddings(
    yamnet,
    files: list[Path],
    label_idx: int,
    label_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (embeddings, labels, sources) for all windows across all files."""
    all_emb, all_lbl, all_src = [], [], []

    for path in files:
        audio = load_mono_16k(path)
        _, embeddings, _ = yamnet(audio)
        emb = embeddings.numpy()                        # (n_windows, 1024)

        if len(emb) > MAX_WINDOWS_PER_FILE:
            idx = np.random.choice(len(emb), MAX_WINDOWS_PER_FILE, replace=False)
            emb = emb[idx]

        src = source_name(path.name)
        all_emb.append(emb)
        all_lbl.extend([label_idx] * len(emb))
        all_src.extend([src] * len(emb))

        print(f"    {label_name:12s}  {path.name}  → {len(emb)} windows", flush=True)

    if not all_emb:
        return np.empty((0, 1024)), np.empty(0, dtype=int), []

    return np.vstack(all_emb), np.array(all_lbl), all_src


# ── train / val split by source ────────────────────────────────────────────────

def split_by_source(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    sources_arr = np.array(sources)
    unique_sources = list(set(sources))
    rng.shuffle(unique_sources)

    n_val = max(1, int(len(unique_sources) * val_frac))
    val_sources = set(unique_sources[:n_val])

    val_mask = np.array([s in val_sources for s in sources])
    train_mask = ~val_mask

    print(f"\n  Sources: {len(unique_sources)} total, "
          f"{len(unique_sources) - n_val} train, {n_val} val")
    print(f"  Windows: {train_mask.sum()} train, {val_mask.sum()} val")

    return (
        embeddings[train_mask], labels[train_mask],
        embeddings[val_mask],   labels[val_mask],
    )


# ── model ───────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="siren_classifier")


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
    yamnet = hub.load(MODEL_URL)

    # collect embeddings per label
    print("\nExtracting embeddings:")
    all_emb, all_lbl, all_src = [], [], []

    for idx, label in enumerate(LABELS):
        files = sorted((DATASET / label).glob("*.wav"))
        if not files:
            print(f"  {label}: no files, skipping")
            continue
        emb, lbl, src = extract_embeddings(yamnet, files, idx, label)
        if len(emb) == 0:
            continue
        all_emb.append(emb)
        all_lbl.append(lbl)
        all_src.extend(src)

    if not all_emb:
        raise SystemExit("No data found.")

    embeddings = np.vstack(all_emb)
    labels = np.concatenate(all_lbl)
    num_classes = len(LABELS)

    # class distribution
    print("\nClass distribution:")
    for i, name in enumerate(LABELS):
        n = int((labels == i).sum())
        print(f"  {name}: {n} windows")

    # train/val split
    print("\nSplitting by source recording:")
    X_train, y_train, X_val, y_val = split_by_source(
        embeddings, labels, all_src, VAL_SPLIT, RANDOM_SEED
    )

    # class weights to handle imbalance
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = {
        i: len(y_train) / (num_classes * max(c, 1))
        for i, c in enumerate(class_counts)
    }
    print(f"\n  Class weights: { {LABELS[i]: round(w, 2) for i, w in class_weights.items()} }")

    # build and train
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    print("\nTraining:")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # evaluate
    print("\nEvaluation on val set:")
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    for i, name in enumerate(LABELS):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        acc = (y_pred[mask] == i).mean()
        print(f"  {name:12s}  {mask.sum():5d} windows  acc={acc:.3f}")

    # confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_val, y_pred):
        cm[t, p] += 1
    header = "          " + "".join(f"{l[:8]:>10s}" for l in LABELS)
    print(header)
    for i, name in enumerate(LABELS):
        row = f"  {name[:8]:8s}" + "".join(f"{cm[i,j]:>10d}" for j in range(num_classes))
        print(row)

    # save
    keras_path = RESULTS / "classifier.keras"
    model.save(str(keras_path))
    print(f"\nSaved: {keras_path}")

    labels_path = RESULTS / "classifier_labels.json"
    labels_path.write_text(json.dumps(LABELS), encoding="utf-8")
    print(f"Saved: {labels_path}")

    # export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = RESULTS / "classifier.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Saved: {tflite_path}")


if __name__ == "__main__":
    main()
