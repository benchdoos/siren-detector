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

import csv
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
EMB_CACHE = Path(__file__).resolve().parent / "emb_cache"
SAVED_EMB = RESULTS / "saved_embeddings.npz"

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
LABELS = ["siren", "background", "explosion", "motorcycle"]
VAL_SPLIT = 0.2
MAX_WINDOWS_PER_FILE = 300          # cap long files so one recording doesn't dominate
RANDOM_SEED = 42
# read at most this many seconds from any single file to avoid OOM on long recordings
MAX_SECONDS_PER_FILE = MAX_WINDOWS_PER_FILE * 0.5 + 5.0

ESC50_DIR = DATASET / "external" / "esc50"
ESC50_CURATED_DIR = DATASET / "external" / "esc50_curated"
FREESOUND_DIR = DATASET / "external" / "freesound"
ESC50_VAL_FOLD = 5
ESC50_LABEL_MAP: dict[str, str] = {
    # siren removed — use curated subset instead (ESC50_CURATED_DIR)
    "rain":            "background",
    "wind":            "background",
    "washing_machine": "background",
    "keyboard_typing": "background",
    "clock_alarm":     "background",
    "engine":          "motorcycle",
    "fireworks":       "background",
    "thunderstorm":    "background",
    "helicopter":      "background",
}  # ~155s


# ── audio helpers ──────────────────────────────────────────────────────────────

def load_mono_16k(path: Path, max_seconds: float | None = None) -> np.ndarray:
    info = sf.info(path)
    frames = int(max_seconds * info.samplerate) if max_seconds else -1
    audio, rate = sf.read(path, dtype="float32", always_2d=True, frames=frames)
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

def _cached_embeddings(yamnet, path: Path) -> np.ndarray:
    """Extract YAMNet embeddings, using a disk cache keyed by file mtime."""
    cache_file = EMB_CACHE / f"{path.stem}_{int(path.stat().st_mtime)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    audio = load_mono_16k(path, max_seconds=MAX_SECONDS_PER_FILE)
    _, embeddings, _ = yamnet(audio)
    emb = embeddings.numpy()
    np.save(cache_file, emb)
    return emb


def extract_embeddings(
    yamnet,
    files: list[Path],
    label_idx: int,
    label_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (embeddings, labels, sources) for all windows across all files."""
    all_emb, all_lbl, all_src = [], [], []

    for path in files:
        emb = _cached_embeddings(yamnet, path)          # (n_windows, 1024)

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
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    """Train/val split per class to ensure balanced val ratio across classes.

    Splitting globally causes lopsided splits when one class has few sources
    with many windows each (e.g. 2 long siren recordings dominate val).
    """
    sources_arr = np.array(sources)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)

    for lbl in sorted(set(labels.tolist())):
        lbl_idx = np.where(labels == lbl)[0]
        lbl_sources = list(set(sources_arr[lbl_idx].tolist()))
        rng = random.Random(seed + int(lbl))
        rng.shuffle(lbl_sources)
        n_val = max(1, int(len(lbl_sources) * val_frac))
        val_srcs = set(lbl_sources[:n_val])
        for i in lbl_idx:
            if sources_arr[i] in val_srcs:
                val_mask[i] = True
            else:
                train_mask[i] = True

    print(f"\n  Sources: {len(set(sources))} total")
    print(f"  Windows: {int(train_mask.sum())} train, {int(val_mask.sum())} val")

    train_src = [s for s, m in zip(sources, train_mask) if m]
    val_src = [s for s, m in zip(sources, val_mask) if m]

    return (
        embeddings[train_mask], labels[train_mask], train_src,
        embeddings[val_mask],   labels[val_mask], val_src,
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

    # ── собственные записи (split by source) ────────────────────────────────
    print("\nExtracting own-recording embeddings:")
    own_emb, own_lbl, own_src = [], [], []
    for idx, label in enumerate(LABELS):
        files = sorted((DATASET / label).glob("*.wav"))
        if not files:
            print(f"  {label}: no files, skipping")
            continue
        emb, lbl, src = extract_embeddings(yamnet, files, idx, label)
        if len(emb) == 0:
            continue
        own_emb.append(emb)
        own_lbl.append(lbl)
        own_src.extend(src)

    # ── Curated ESC-50 subsets (hand-picked, per label) ─────────────────────
    if ESC50_CURATED_DIR.exists():
        print("\nExtracting curated ESC-50 embeddings:")
        for label in LABELS:
            curated_files = sorted((ESC50_CURATED_DIR / label).glob("*.wav"))
            if not curated_files:
                continue
            idx = LABELS.index(label)
            emb, lbl, src = extract_embeddings(yamnet, curated_files, idx, f"curated/{label}")
            if len(emb):
                own_emb.append(emb)
                own_lbl.append(lbl)
                own_src.extend(src)

    # ── Freesound downloads (all treated as siren) ────────────────────────────
    freesound_audio = FREESOUND_DIR / "audio"
    if freesound_audio.exists():
        fs_files = sorted(
            p for p in freesound_audio.iterdir()
            if p.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac")
        )
        if fs_files:
            print(f"\nExtracting Freesound siren embeddings ({len(fs_files)} files):")
            siren_idx = LABELS.index("siren")
            emb, lbl, src = extract_embeddings(yamnet, fs_files, siren_idx, "freesound/siren")
            if len(emb):
                own_emb.append(emb)
                own_lbl.append(lbl)
                own_src.extend(src)

    if own_emb:
        print("\nSplitting own recordings by source:")
        X_own_tr, y_own_tr, own_tr_src, X_own_val, y_own_val, own_val_src = split_by_source(
            np.vstack(own_emb), np.concatenate(own_lbl), own_src,
            VAL_SPLIT, RANDOM_SEED,
        )
    else:
        X_own_tr = X_own_val = np.empty((0, 1024), dtype=np.float32)
        y_own_tr = y_own_val = np.empty(0, dtype=np.int64)
        own_tr_src, own_val_src = [], []

    # ── ESC-50 (split by fold 5) ─────────────────────────────────────────────
    esc_tr_emb: list[np.ndarray] = []
    esc_tr_lbl: list[np.ndarray] = []
    esc_tr_src: list[str] = []
    esc_val_emb: list[np.ndarray] = []
    esc_val_lbl: list[np.ndarray] = []
    esc_val_src: list[str] = []
    esc50_csv = ESC50_DIR / "meta" / "esc50.csv"
    if esc50_csv.exists():
        print("\nExtracting ESC-50 embeddings:")
        audio_dir = ESC50_DIR / "audio"
        esc_train_by_label: dict[int, list[Path]] = defaultdict(list)
        esc_val_by_label:   dict[int, list[Path]] = defaultdict(list)
        with esc50_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = row["category"]
                if cat not in ESC50_LABEL_MAP:
                    continue
                path = audio_dir / row["filename"]
                if not path.exists():
                    continue
                lbl_idx = LABELS.index(ESC50_LABEL_MAP[cat])
                if int(row["fold"]) == ESC50_VAL_FOLD:
                    esc_val_by_label[lbl_idx].append(path)
                else:
                    esc_train_by_label[lbl_idx].append(path)

        for lbl_idx, files in sorted(esc_train_by_label.items()):
            emb, lbl, src = extract_embeddings(
                yamnet, files, lbl_idx, f"esc50-train/{LABELS[lbl_idx]}"
            )
            if len(emb):
                esc_tr_emb.append(emb)
                esc_tr_lbl.append(lbl)
                esc_tr_src.extend(src)

        for lbl_idx, files in sorted(esc_val_by_label.items()):
            emb, lbl, src = extract_embeddings(
                yamnet, files, lbl_idx, f"esc50-val/{LABELS[lbl_idx]}"
            )
            if len(emb):
                esc_val_emb.append(emb)
                esc_val_lbl.append(lbl)
                esc_val_src.extend(src)
    else:
        print("\nESC-50 not found — training on own recordings only.")

    # ── загрузка ранее сохранённых эмбеддингов + merge с новыми ────────────────
    fresh_sources = set(own_tr_src) | set(own_val_src) | set(esc_tr_src) | set(esc_val_src)

    prev_tr_emb: list[np.ndarray] = []
    prev_tr_lbl: list[np.ndarray] = []
    prev_tr_src: list[str] = []
    prev_val_emb: list[np.ndarray] = []
    prev_val_lbl: list[np.ndarray] = []
    prev_val_src: list[str] = []

    if SAVED_EMB.exists():
        saved = np.load(SAVED_EMB, allow_pickle=True)
        saved_labels = json.loads(saved["labels"].item())
        if saved_labels == LABELS:
            s_tr_src = list(saved["sources_train"])
            s_val_src = list(saved["sources_val"])
            s_Xtr, s_ytr = saved["X_train"], saved["y_train"]
            s_Xval, s_yval = saved["X_val"], saved["y_val"]
            # keep only windows whose source is NOT in fresh data (avoid duplicates)
            tr_keep = [i for i, s in enumerate(s_tr_src) if s not in fresh_sources]
            val_keep = [i for i, s in enumerate(s_val_src) if s not in fresh_sources]
            if tr_keep:
                prev_tr_emb.append(s_Xtr[tr_keep])
                prev_tr_lbl.append(s_ytr[tr_keep])
                prev_tr_src.extend(s_tr_src[i] for i in tr_keep)
            if val_keep:
                prev_val_emb.append(s_Xval[val_keep])
                prev_val_lbl.append(s_yval[val_keep])
                prev_val_src.extend(s_val_src[i] for i in val_keep)
            n_reused = len(tr_keep) + len(val_keep)
            n_total = len(s_tr_src) + len(s_val_src)
            print(f"\nSaved embeddings: reused {n_reused}/{n_total} windows "
                  f"(skipped {n_total - n_reused} — fresh data available)")
        else:
            print(f"\nSaved embeddings have different labels {saved_labels} vs {LABELS} — skipping")

    # ── объединяем train и val ───────────────────────────────────────────────
    all_tr_src = prev_tr_src + own_tr_src + esc_tr_src
    all_val_src = prev_val_src + own_val_src + esc_val_src

    X_train = np.vstack([*prev_tr_emb, X_own_tr, *esc_tr_emb])
    y_train = np.concatenate([*prev_tr_lbl, y_own_tr, *esc_tr_lbl])
    X_val   = np.vstack([*prev_val_emb, X_own_val, *esc_val_emb])
    y_val   = np.concatenate([*prev_val_lbl, y_own_val, *esc_val_lbl])

    if len(X_train) == 0:
        raise SystemExit("No training data found.")

    num_classes = len(LABELS)

    # class distribution
    print("\nClass distribution (train):")
    for i, name in enumerate(LABELS):
        n = int((y_train == i).sum())
        print(f"  {name}: {n} windows")
    print("\nClass distribution (val):")
    for i, name in enumerate(LABELS):
        n = int((y_val == i).sum())
        print(f"  {name}: {n} windows")

    # class weights
    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
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

    # save embeddings for future incremental retraining
    np.savez_compressed(
        SAVED_EMB,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        sources_train=np.array(all_tr_src, dtype=object),
        sources_val=np.array(all_val_src, dtype=object),
        labels=json.dumps(LABELS),
    )
    print(f"Saved embeddings: {SAVED_EMB}  ({SAVED_EMB.stat().st_size // 1024 // 1024} MB)")

    # save
    keras_path = RESULTS / "classifier.keras"
    model.save(str(keras_path))
    print(f"\nSaved: {keras_path}")

    labels_path = RESULTS / "classifier_labels.json"
    labels_path.write_text(json.dumps(LABELS), encoding="utf-8")
    print(f"Saved: {labels_path}")

    # TFLite export skipped — broken in TF 2.16 (MLIR bug)
    # Export weights as numpy .npz for portable inference
    weights = model.get_weights()  # [W1, b1, W2, b2, W3, b3]
    npz_path = RESULTS / "classifier_weights.npz"
    np.savez(npz_path, W1=weights[0], b1=weights[1],
             W2=weights[2], b2=weights[3], W3=weights[4], b3=weights[5])
    print(f"Saved: {npz_path}  ({npz_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
