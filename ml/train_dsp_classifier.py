#!/usr/bin/env python3
"""
Train a tiny Conv1D classifier on microphone-invariant DSP features.

Architecture:
  WAV → DSP features (5 channels × N frames)
    → Conv1D (16 filters, k=5) → Conv1D (32, k=5) → GlobalAvgPool
    → Dense(32) → Dense(4, softmax)

Total params: ~6K — fits on ESP32 with INT8 quantization (~6 KB).

Features per frame (20ms hop):
  [pitch_hz, log_rms, spectral_centroid_hz, hnr_db, zcr]

These features track *relative* signal changes and are largely invariant
to microphone frequency response, unlike raw spectrograms.

Usage:
  ml/.venv/bin/python ml/train_dsp_classifier.py
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from dsp_features import extract_features, load_mono_16k, LABELS, HOP_SIZE, SAMPLE_RATE


def add_deltas(features: np.ndarray) -> np.ndarray:
    """Append delta (rate of change) features. (N, 5) → (N, 10)"""
    deltas = np.zeros_like(features)
    deltas[1:] = features[1:] - features[:-1]
    return np.concatenate([features, deltas], axis=1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"
RESULTS = Path(__file__).resolve().parent / "results"
DSP_CACHE = Path(__file__).resolve().parent / "dsp_cache"

RANDOM_SEED = 42
VAL_SPLIT = 0.2
WINDOW_FRAMES = 250        # 5 seconds at 50 fps — captures full siren sweep cycles
WINDOW_HOP_FRAMES = 50     # 1 second hop between windows
N_FEATURES = 16            # 8 raw + 8 delta (rate of change)
MAX_FILES_PER_CLASS = 500   # cap files to avoid OOM
MIXUP_ALPHA = 0.0           # Beta(α, α) mixup; 0 disables
MAX_BG_TRAIN_WINDOWS = 3000
EPOCHS = 100
BATCH_SIZE = 64


# ── helpers ────────────────────────────────────────────────────────────────────

def source_name(filename: str) -> str:
    """Extract original source from filename, stripping aug prefix and hash suffix."""
    stem = Path(filename).stem
    if stem.startswith("aug_q_"):
        stem = stem[6:]
    elif stem.startswith("aug_"):
        stem = stem[4:]
    m = re.match(r"^(.+)_([0-9a-f]{8})$", stem)
    if m:
        stem = m.group(1)
    return stem.split("__")[0]


def is_augmented(filename: str) -> bool:
    return Path(filename).name.startswith("aug_")


# ── feature caching ────────────────────────────────────────────────────────────

def _cached_features(path: Path) -> np.ndarray:
    """Extract features with disk caching keyed by file mtime."""
    DSP_CACHE.mkdir(parents=True, exist_ok=True)
    cache_key = f"{path.stem}_{int(path.stat().st_mtime)}.npy"
    cache_file = DSP_CACHE / cache_key
    if cache_file.exists():
        return np.load(cache_file)
    audio = load_mono_16k(path, max_seconds=30)
    feats = add_deltas(extract_features(audio))
    np.save(cache_file, feats)
    return feats


# ── feature normalization stats ────────────────────────────────────────────────

def compute_norm_stats(all_features: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std for z-score normalization."""
    cat = np.vstack(all_features)
    return cat.mean(axis=0).astype(np.float32), cat.std(axis=0).astype(np.float32) + 1e-6


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


# ── dataset building ───────────────────────────────────────────────────────────

def collect_files() -> dict[str, list[Path]]:
    """Collect WAV files per label from dataset directory."""
    files_by_label: dict[str, list[Path]] = {}
    for label in LABELS:
        label_dir = DATASET / label
        if not label_dir.exists():
            continue
        files = sorted(label_dir.rglob("*.wav"))[:MAX_FILES_PER_CLASS]
        if files:
            files_by_label[label] = files
    return files_by_label


def extract_all_features(files_by_label: dict[str, list[Path]]) -> tuple[
    dict[str, list[tuple[np.ndarray, str, bool]]], dict[str, set[str]]
]:
    """Extract DSP features from all files.

    Returns:
        features_by_label: {label: [(features_array, source, is_aug), ...]}
        sources_by_label: {label: {source1, source2, ...}}
    """
    features_by_label: dict[str, list[tuple[np.ndarray, str, bool]]] = defaultdict(list)
    sources_by_label: dict[str, set[str]] = defaultdict(set)

    for label, files in files_by_label.items():
        print(f"  Extracting {label}: {len(files)} files...", end="", flush=True)
        for f in files:
            try:
                feats = _cached_features(f)
                if len(feats) < WINDOW_FRAMES:
                    continue
                src = source_name(f.name)
                aug = is_augmented(f.name)
                features_by_label[label].append((feats, src, aug))
                sources_by_label[label].add(src)
            except Exception as e:
                print(f"\n    skip {f.name}: {e}")
        print(f" {len(features_by_label[label])} ok")

    return dict(features_by_label), dict(sources_by_label)


def split_train_val(
    features_by_label: dict[str, list[tuple[np.ndarray, str, bool]]],
    sources_by_label: dict[str, set[str]],
) -> tuple[dict[str, list], dict[str, list], set[str]]:
    """Split by source recording (not random windows) for proper validation."""
    rng = random.Random(RANDOM_SEED)

    # Collect all unique sources
    all_sources = set()
    for sources in sources_by_label.values():
        all_sources |= sources

    val_sources = set(rng.sample(sorted(all_sources), max(1, int(len(all_sources) * VAL_SPLIT))))

    train_data: dict[str, list] = defaultdict(list)
    val_data: dict[str, list] = defaultdict(list)

    for label, items in features_by_label.items():
        for feats, src, aug in items:
            if src in val_sources:
                val_data[label].append((feats, src, aug))
            else:
                train_data[label].append((feats, src, aug))

    return dict(train_data), dict(val_data), val_sources


def make_windows(
    data: dict[str, list],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    cap_bg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice feature arrays into fixed-size windows.

    Returns (X, y, sample_weights) arrays.
    """
    windows = []
    labels_idx = []
    weights = []

    for label, items in data.items():
        label_i = LABELS.index(label)
        label_windows = []

        for feats, src, aug in items:
            feats_norm = normalize_features(feats, norm_mean, norm_std)
            # Sliding window
            for start in range(0, len(feats_norm) - WINDOW_FRAMES + 1, WINDOW_HOP_FRAMES):
                win = feats_norm[start:start + WINDOW_FRAMES]
                label_windows.append((win, 0.4 if aug else 1.0))

        # Cap background windows to prevent class imbalance
        if cap_bg and label == "background" and len(label_windows) > cap_bg:
            rng = random.Random(RANDOM_SEED)
            label_windows = rng.sample(label_windows, cap_bg)

        for win, w in label_windows:
            windows.append(win)
            labels_idx.append(label_i)
            weights.append(w)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels_idx, dtype=np.int32)
    w = np.array(weights, dtype=np.float32)

    # Shuffle
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(X))
    return X[idx], y[idx], w[idx]


def apply_mixup(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = MIXUP_ALPHA
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixup augmentation. Returns (X_mixed, y_onehot, w)."""
    if alpha <= 0:
        y_oh = np.eye(len(LABELS), dtype=np.float32)[y]
        return X, y_oh, w

    rng = np.random.RandomState(RANDOM_SEED + 1)
    n = len(X)
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    idx = rng.permutation(n)

    lam_x = lam[:, None, None]
    X_mixed = lam_x * X + (1 - lam_x) * X[idx]

    y_oh = np.eye(len(LABELS), dtype=np.float32)
    y_mixed = lam[:, None] * y_oh[y] + (1 - lam[:, None]) * y_oh[y[idx]]

    w_mixed = lam * w + (1 - lam) * w[idx]
    return X_mixed, y_mixed, w_mixed


# ── model ──────────────────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """Tiny Conv1D classifier on DSP features.

    Input: (batch, 100, 5) — 2 seconds of 5 features at 50fps
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_FRAMES, N_FEATURES)),
        tf.keras.layers.Conv1D(24, kernel_size=7, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(48, kernel_size=5, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(48, kernel_size=3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(48, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    return model


# ── training ───────────────────────────────────────────────────────────────────

def train():
    print("=== DSP Feature Classifier Training ===\n")

    # 1. Collect files
    print("1. Collecting files...")
    files_by_label = collect_files()
    for label, files in files_by_label.items():
        print(f"   {label}: {len(files)} files")

    # 2. Extract features
    print("\n2. Extracting DSP features...")
    features_by_label, sources_by_label = extract_all_features(files_by_label)

    # 3. Train/val split by source
    print("\n3. Splitting train/val by source...")
    train_data, val_data, val_sources = split_train_val(features_by_label, sources_by_label)
    print(f"   Val sources: {len(val_sources)}")
    for label in LABELS:
        n_train = len(train_data.get(label, []))
        n_val = len(val_data.get(label, []))
        print(f"   {label}: {n_train} train, {n_val} val files")

    # 4. Compute normalization stats from training data only
    print("\n4. Computing normalization stats...")
    train_feats = []
    for items in train_data.values():
        for feats, _, _ in items:
            train_feats.append(feats)
    norm_mean, norm_std = compute_norm_stats(train_feats)
    print(f"   Mean: {norm_mean}")
    print(f"   Std:  {norm_std}")

    # 5. Build windows
    print("\n5. Building windows...")
    X_train, y_train, w_train = make_windows(train_data, norm_mean, norm_std, cap_bg=MAX_BG_TRAIN_WINDOWS)
    X_val, y_val, w_val = make_windows(val_data, norm_mean, norm_std)
    print(f"   Train: {len(X_train)} windows")
    print(f"   Val:   {len(X_val)} windows")
    for i, label in enumerate(LABELS):
        n_tr = (y_train == i).sum()
        n_va = (y_val == i).sum()
        print(f"   {label}: {n_tr} train, {n_va} val windows")

    # 6. Merge class weights into sample weights
    class_counts = np.bincount(y_train, minlength=len(LABELS)).astype(float) + 1
    class_weight_map = {i: np.sqrt(class_counts.max() / c) for i, c in enumerate(class_counts)}
    print(f"\n6. Class weights: {class_weight_map}")
    for i, cw in class_weight_map.items():
        w_train[y_train == i] *= cw

    # 7. Apply mixup to training data
    print("\n7. Applying mixup...")
    X_train_mx, y_train_mx, w_train_mx = apply_mixup(X_train, y_train, w_train)
    print(f"   Mixup alpha={MIXUP_ALPHA}, y shape: {y_train_mx.shape}")

    # One-hot val labels for categorical_crossentropy
    y_val_oh = np.eye(len(LABELS), dtype=np.float32)[y_val]

    # 8. Build and train model
    print("\n8. Training model...")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    model.fit(
        X_train_mx, y_train_mx,
        sample_weight=w_train_mx,
        validation_data=(X_val, y_val_oh, w_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # class weights already baked into sample_weight
        callbacks=callbacks,
        verbose=1,
    )

    # 9. Evaluate
    print("\n9. Evaluation on val set...")
    y_val_probs = model.predict(X_val, verbose=0)
    y_pred = y_val_probs.argmax(axis=1)
    confusion_matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for truth, predicted in zip(y_val, y_pred):
        confusion_matrix[int(truth), int(predicted)] += 1

    class_metrics = []
    for i, label in enumerate(LABELS):
        mask = y_val == i
        if mask.sum() == 0:
            print(f"   {label:15s} — no val samples")
            continue
        precision = confusion_matrix[i, i] / max(1, confusion_matrix[:, i].sum())
        recall = confusion_matrix[i, i] / max(1, confusion_matrix[i, :].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        class_metrics.append({
            "label": label,
            "windows": int(mask.sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })
        print(f"   {label:15s} {mask.sum():5d} windows  "
              f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    overall_accuracy = float((y_pred == y_val).mean()) if len(y_val) else 0.0
    siren_idx = LABELS.index("siren")
    background_idx = LABELS.index("background")
    siren_mask = y_val == siren_idx
    siren_confident_background_rate = (
        float((y_val_probs[siren_mask, background_idx] >= 0.92).mean())
        if siren_mask.any() else None
    )
    siren_argmax_background_rate = (
        float((y_pred[siren_mask] == background_idx).mean())
        if siren_mask.any() else None
    )

    # Print confusion matrix
    print("\n   Confusion matrix (rows=true, cols=predicted):")
    print(f'           {"  ".join(f"{l:>10s}" for l in LABELS)}')
    for i, label in enumerate(LABELS):
        print(f'   {label:>10s}  {"  ".join(f"{v:10d}" for v in confusion_matrix[i])}')

    # 10. Save
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model_path = RESULTS / "dsp_classifier.keras"
    model.save(model_path)
    print(f"\n   Saved model: {model_path}")

    # Save normalization stats
    norm_path = RESULTS / "dsp_norm_stats.npz"
    np.savez(norm_path, mean=norm_mean, std=norm_std)
    print(f"   Saved norm stats: {norm_path}")

    # Save labels
    labels_path = RESULTS / "dsp_classifier_labels.json"
    labels_path.write_text(json.dumps(LABELS, indent=2))
    print(f"   Saved labels: {labels_path}")

    # Export TFLite (INT8 quantized)
    print("\n11. Exporting TFLite (INT8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(500, len(X_train))):
            yield [X_train[i:i + 1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = RESULTS / "dsp_classifier_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"   TFLite INT8: {tflite_path} ({len(tflite_model)} bytes)")

    dsp_metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": LABELS,
        "validation_windows": int(len(y_val)),
        "accuracy": round(overall_accuracy, 4),
        "classes": class_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "background_gate_threshold": 0.92,
        "siren_confident_background_rate": (
            round(siren_confident_background_rate, 4)
            if siren_confident_background_rate is not None else None
        ),
        "siren_argmax_background_rate": (
            round(siren_argmax_background_rate, 4)
            if siren_argmax_background_rate is not None else None
        ),
        "quantized_model_size_bytes": len(tflite_model),
    }
    metrics_path = RESULTS / "dsp_metrics.json"
    metrics_path.write_text(json.dumps(dsp_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   DSP metrics: {metrics_path}")

    # Also export float32 for comparison
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter_f32.convert()
    tflite_f32_path = RESULTS / "dsp_classifier_f32.tflite"
    tflite_f32_path.write_bytes(tflite_f32)
    print(f"   TFLite F32:  {tflite_f32_path} ({len(tflite_f32)} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    train()
