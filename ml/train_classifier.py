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
import re
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
VAL_SOURCES_FILE = RESULTS / "val_sources.json"

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16_000
LABELS = ["siren", "background", "explosion", "uav"]
VAL_SPLIT = 0.2
MAX_WINDOWS_PER_FILE = 300          # cap long files so one recording doesn't dominate
MAX_BG_TRAIN_WINDOWS = 2000         # undersample background: cap at 2-3× avg of other classes
RANDOM_SEED = 42
MIXUP_ALPHA = 0.1                    # Beta(α, α) for mixup; 0 disables mixup
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
    "engine":          "background",
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
    """Extract original source from filename, stripping aug prefix and hash suffix.

    '20261009-070807__siren__1918-2124s.wav' → '20261009-070807'
    'aug_freesound_133989_244ff033.wav'      → 'freesound_133989'
    'aug_q_1-54084-A-42_04986324.wav'        → '1-54084-A-42'
    """
    stem = Path(filename).stem
    if stem.startswith("aug_q_"):
        stem = stem[6:]
    elif stem.startswith("aug_"):
        stem = stem[4:]
    # Strip trailing _XXXXXXXX hex hash added by augment.py
    m = re.match(r"^(.+)_([0-9a-f]{8})$", stem)
    if m:
        stem = m.group(1)
    return stem.split("__")[0]


def is_augmented(filename: str) -> bool:
    """Check if file is augmented (starts with aug_ or aug_q_)."""
    return Path(filename).name.startswith("aug_")


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


YAMNET_WIN_SAMPLES = int(0.96 * SAMPLE_RATE)   # 15360
YAMNET_HOP_SAMPLES = int(0.48 * SAMPLE_RATE)   # 7680
ZERO_WINDOW_THRESHOLD = 0.5  # drop window if >50% samples are exactly 0.0


def _dead_window_mask(path: Path, n_windows: int) -> np.ndarray:
    """Return boolean mask: True for windows where >50% samples are zero (UDP drops)."""
    try:
        audio = load_mono_16k(path, max_seconds=MAX_SECONDS_PER_FILE)
    except Exception:
        return np.zeros(n_windows, dtype=bool)
    mask = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * YAMNET_HOP_SAMPLES
        end = start + YAMNET_WIN_SAMPLES
        if end > len(audio):
            break
        chunk = audio[start:end]
        zero_frac = np.mean(chunk == 0.0)
        if zero_frac > ZERO_WINDOW_THRESHOLD:
            mask[i] = True
    return mask


def extract_embeddings(
    yamnet,
    files: list[Path],
    label_idx: int,
    label_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    """Return (embeddings, labels, sources, is_aug) for all windows across all files."""
    all_emb, all_lbl, all_src, all_aug = [], [], [], []

    for path in files:
        emb = _cached_embeddings(yamnet, path)          # (n_windows, 1024)

        # Filter out dead windows (UDP packet drops → zeros)
        dead = _dead_window_mask(path, len(emb))
        n_dead = int(dead.sum())
        if n_dead > 0:
            emb = emb[~dead]

        if len(emb) > MAX_WINDOWS_PER_FILE:
            idx = np.random.choice(len(emb), MAX_WINDOWS_PER_FILE, replace=False)
            emb = emb[idx]

        src = source_name(path.name)
        aug = is_augmented(path.name)
        all_emb.append(emb)
        all_lbl.extend([label_idx] * len(emb))
        all_src.extend([src] * len(emb))
        all_aug.extend([aug] * len(emb))

        dead_info = f" ({n_dead} dead dropped)" if n_dead else ""
        aug_tag = " [aug]" if aug else ""
        print(f"    {label_name:12s}  {path.name}  → {len(emb)} windows{dead_info}{aug_tag}", flush=True)

    if not all_emb:
        return np.empty((0, 1024)), np.empty(0, dtype=int), [], []

    return np.vstack(all_emb), np.array(all_lbl), all_src, all_aug


# ── train / val split by source ────────────────────────────────────────────────

def _load_val_sources() -> dict[str, list[str]] | None:
    """Load fixed val sources from file. Returns {label: [source, ...]} or None."""
    if VAL_SOURCES_FILE.exists():
        return json.loads(VAL_SOURCES_FILE.read_text(encoding="utf-8"))
    return None


def _save_val_sources(val_sources: dict[str, list[str]]) -> None:
    VAL_SOURCES_FILE.write_text(
        json.dumps(val_sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Saved val sources: {VAL_SOURCES_FILE}")


def split_by_source(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sources: list[str],
    is_aug: list[bool],
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[bool],
           np.ndarray, np.ndarray, list[str], list[bool]]:
    """Train/val split per class with fixed val sources.

    On first run: picks val sources randomly (val_frac per class), saves to
    val_sources.json. On subsequent runs: reuses saved val sources, new data
    goes to train by default.
    """
    fixed = _load_val_sources()
    sources_arr = np.array(sources)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    new_val_sources: dict[str, list[str]] = {}

    for lbl in sorted(set(labels.tolist())):
        lbl_name = LABELS[int(lbl)]
        lbl_idx = np.where(labels == lbl)[0]
        lbl_sources = sorted(set(sources_arr[lbl_idx].tolist()))

        if fixed and lbl_name in fixed:
            val_srcs = set(fixed[lbl_name]) & set(lbl_sources)
        else:
            rng = random.Random(seed + int(lbl))
            rng.shuffle(lbl_sources)
            n_val = max(1, int(len(lbl_sources) * val_frac))
            val_srcs = set(lbl_sources[:n_val])

        new_val_sources[lbl_name] = sorted(val_srcs)

        for i in lbl_idx:
            if sources_arr[i] in val_srcs:
                val_mask[i] = True
            else:
                train_mask[i] = True

    if not fixed:
        _save_val_sources(new_val_sources)
        print("  (first run — val sources fixed for future runs)")
    else:
        _save_val_sources(new_val_sources)

    print(f"\n  Sources: {len(set(sources))} total")
    print(f"  Windows: {int(train_mask.sum())} train, {int(val_mask.sum())} val")

    train_src = [s for s, m in zip(sources, train_mask) if m]
    val_src = [s for s, m in zip(sources, val_mask) if m]
    train_aug = [a for a, m in zip(is_aug, train_mask) if m]
    val_aug = [a for a, m in zip(is_aug, val_mask) if m]

    return (
        embeddings[train_mask], labels[train_mask], train_src, train_aug,
        embeddings[val_mask],   labels[val_mask], val_src, val_aug,
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
    own_emb, own_lbl, own_src, own_aug = [], [], [], []
    for idx, label in enumerate(LABELS):
        files = sorted((DATASET / label).rglob("*.wav"))
        if not files:
            print(f"  {label}: no files, skipping")
            continue
        emb, lbl, src, aug = extract_embeddings(yamnet, files, idx, label)
        if len(emb) == 0:
            continue
        own_emb.append(emb)
        own_lbl.append(lbl)
        own_src.extend(src)
        own_aug.extend(aug)

    # ── Curated ESC-50 subsets (hand-picked, per label) ─────────────────────
    if ESC50_CURATED_DIR.exists():
        print("\nExtracting curated ESC-50 embeddings:")
        for label in LABELS:
            curated_files = sorted((ESC50_CURATED_DIR / label).glob("*.wav"))
            if not curated_files:
                continue
            idx = LABELS.index(label)
            emb, lbl, src, aug = extract_embeddings(yamnet, curated_files, idx, f"curated/{label}")
            if len(emb):
                own_emb.append(emb)
                own_lbl.append(lbl)
                own_src.extend(src)
                own_aug.extend(aug)

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
            emb, lbl, src, aug = extract_embeddings(yamnet, fs_files, siren_idx, "freesound/siren")
            if len(emb):
                own_emb.append(emb)
                own_lbl.append(lbl)
                own_src.extend(src)
                own_aug.extend(aug)

    if own_emb:
        print("\nSplitting own recordings by source:")
        (X_own_tr, y_own_tr, own_tr_src, own_tr_aug,
         X_own_val, y_own_val, own_val_src, own_val_aug) = split_by_source(
            np.vstack(own_emb), np.concatenate(own_lbl), own_src, own_aug,
            VAL_SPLIT, RANDOM_SEED,
        )
    else:
        X_own_tr = X_own_val = np.empty((0, 1024), dtype=np.float32)
        y_own_tr = y_own_val = np.empty(0, dtype=np.int64)
        own_tr_src, own_val_src = [], []
        own_tr_aug, own_val_aug = [], []

    # ── ESC-50 (split by fold 5) ─────────────────────────────────────────────
    esc_tr_emb: list[np.ndarray] = []
    esc_tr_lbl: list[np.ndarray] = []
    esc_tr_src: list[str] = []
    esc_tr_aug: list[bool] = []  # ESC-50 treated as external (weight like aug)
    esc_val_emb: list[np.ndarray] = []
    esc_val_lbl: list[np.ndarray] = []
    esc_val_src: list[str] = []
    esc_val_aug: list[bool] = []
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
            emb, lbl, src, _aug = extract_embeddings(
                yamnet, files, lbl_idx, f"esc50-train/{LABELS[lbl_idx]}"
            )
            if len(emb):
                esc_tr_emb.append(emb)
                esc_tr_lbl.append(lbl)
                esc_tr_src.extend(src)
                esc_tr_aug.extend([True] * len(src))  # external data → low weight

        for lbl_idx, files in sorted(esc_val_by_label.items()):
            emb, lbl, src, _aug = extract_embeddings(
                yamnet, files, lbl_idx, f"esc50-val/{LABELS[lbl_idx]}"
            )
            if len(emb):
                esc_val_emb.append(emb)
                esc_val_lbl.append(lbl)
                esc_val_src.extend(src)
                esc_val_aug.extend([True] * len(src))
    else:
        print("\nESC-50 not found — training on own recordings only.")

    # ── загрузка ранее сохранённых эмбеддингов + merge с новыми ────────────────
    fresh_sources = set(own_tr_src) | set(own_val_src) | set(esc_tr_src) | set(esc_val_src)

    prev_tr_emb: list[np.ndarray] = []
    prev_tr_lbl: list[np.ndarray] = []
    prev_tr_src: list[str] = []
    prev_tr_aug: list[bool] = []
    prev_val_emb: list[np.ndarray] = []
    prev_val_lbl: list[np.ndarray] = []
    prev_val_src: list[str] = []
    prev_val_aug: list[bool] = []

    if SAVED_EMB.exists():
        try:
            saved = np.load(SAVED_EMB)
        except ValueError:
            print(f"\nSaved embeddings use legacy format (allow_pickle) — "
                  f"delete {SAVED_EMB} and re-run to rebuild")
            saved = None
        if saved is not None and json.loads(saved["labels"].item()) == LABELS:
            s_tr_src = json.loads(saved["sources_train"].item())
            s_val_src = json.loads(saved["sources_val"].item())
            s_Xtr, s_ytr = saved["X_train"], saved["y_train"]
            s_Xval, s_yval = saved["X_val"], saved["y_val"]
            # keep only windows whose source is NOT in fresh data (avoid duplicates)
            tr_keep = [i for i, s in enumerate(s_tr_src) if s not in fresh_sources]
            val_keep = [i for i, s in enumerate(s_val_src) if s not in fresh_sources]
            # Load is_aug if saved, otherwise infer from source name
            if "is_aug_train" in saved:
                s_tr_aug = list(saved["is_aug_train"])
                s_val_aug = list(saved["is_aug_val"])
            else:
                s_tr_aug = [s.startswith("aug_") for s in s_tr_src]
                s_val_aug = [s.startswith("aug_") for s in s_val_src]
            if tr_keep:
                prev_tr_emb.append(s_Xtr[tr_keep])
                prev_tr_lbl.append(s_ytr[tr_keep])
                prev_tr_src.extend(s_tr_src[i] for i in tr_keep)
                prev_tr_aug.extend(s_tr_aug[i] for i in tr_keep)
            if val_keep:
                prev_val_emb.append(s_Xval[val_keep])
                prev_val_lbl.append(s_yval[val_keep])
                prev_val_src.extend(s_val_src[i] for i in val_keep)
                prev_val_aug.extend(s_val_aug[i] for i in val_keep)
            n_reused = len(tr_keep) + len(val_keep)
            n_total = len(s_tr_src) + len(s_val_src)
            print(f"\nSaved embeddings: reused {n_reused}/{n_total} windows "
                  f"(skipped {n_total - n_reused} — fresh data available)")
        elif saved is not None:
            print(f"\nSaved embeddings have different labels — skipping")

    # ── объединяем train и val ───────────────────────────────────────────────
    all_tr_src = prev_tr_src + own_tr_src + esc_tr_src
    all_val_src = prev_val_src + own_val_src + esc_val_src
    all_tr_aug = prev_tr_aug + own_tr_aug + esc_tr_aug
    all_val_aug = prev_val_aug + own_val_aug + esc_val_aug

    X_train = np.vstack([*prev_tr_emb, X_own_tr, *esc_tr_emb])
    y_train = np.concatenate([*prev_tr_lbl, y_own_tr, *esc_tr_lbl])
    X_val   = np.vstack([*prev_val_emb, X_own_val, *esc_val_emb])
    y_val   = np.concatenate([*prev_val_lbl, y_own_val, *esc_val_lbl])

    if len(X_train) == 0:
        raise SystemExit("No training data found.")

    num_classes = len(LABELS)

    # undersample background in train
    bg_idx_val = LABELS.index("background")
    bg_mask = y_train == bg_idx_val
    n_bg = int(bg_mask.sum())
    if n_bg > MAX_BG_TRAIN_WINDOWS:
        bg_indices = np.where(bg_mask)[0]
        keep_bg = np.random.choice(bg_indices, MAX_BG_TRAIN_WINDOWS, replace=False)
        non_bg = np.where(~bg_mask)[0]
        keep_all = np.sort(np.concatenate([non_bg, keep_bg]))
        print(f"\nUndersample background: {n_bg} → {MAX_BG_TRAIN_WINDOWS} windows")
        X_train = X_train[keep_all]
        y_train = y_train[keep_all]
        all_tr_src = [all_tr_src[i] for i in keep_all]
        all_tr_aug = [all_tr_aug[i] for i in keep_all]

    # class distribution
    aug_arr = np.array(all_tr_aug)
    print("\nClass distribution (train):")
    for i, name in enumerate(LABELS):
        mask = y_train == i
        n = int(mask.sum())
        n_aug = int((mask & aug_arr).sum())
        n_real = n - n_aug
        print(f"  {name}: {n} windows ({n_real} real, {n_aug} aug/external)")
    print("\nClass distribution (val):")
    val_aug_arr = np.array(all_val_aug)
    for i, name in enumerate(LABELS):
        mask = y_val == i
        n = int(mask.sum())
        n_aug = int((mask & val_aug_arr).sum())
        n_real = n - n_aug
        print(f"  {name}: {n} windows ({n_real} real, {n_aug} aug/external)")

    # combined sample weights = class_weight × source_weight
    # source weight: real=1.0, augmented/external=0.4
    AUG_SAMPLE_WEIGHT = 0.4
    source_weights = np.where(aug_arr, AUG_SAMPLE_WEIGHT, 1.0).astype(np.float32)

    # class weights — sqrt formula from TRAINING_GUIDE
    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    n_total = len(y_train)
    class_weight_map = {
        i: float(np.sqrt(n_total / max(c, 1)))
        for i, c in enumerate(class_counts)
    }
    class_weight_arr = np.array([class_weight_map[int(y)] for y in y_train], dtype=np.float32)

    # Keras doesn't allow class_weight + sample_weight simultaneously,
    # so we multiply them into a single sample_weight array
    sample_weights = source_weights * class_weight_arr

    print(f"\n  Source weights: real=1.0, aug/external={AUG_SAMPLE_WEIGHT}")
    print(f"  Class weights: { {LABELS[i]: round(w, 2) for i, w in class_weight_map.items()} }")
    print(f"  Effective train size: {sample_weights.sum():.0f} "
          f"(from {len(sample_weights)} windows)")

    # build and train
    model = build_model(num_classes)
    model.summary()

    # ── mixup on embeddings ────────────────────────────────────────────────
    # Convert labels to one-hot (required for soft labels after mixing)
    y_train_oh = tf.one_hot(y_train.astype(np.int32), num_classes).numpy().astype(np.float32)
    y_val_oh = tf.one_hot(y_val.astype(np.int32), num_classes).numpy().astype(np.float32)

    batch_size = 256

    X_train_f = X_train.astype(np.float32)

    if MIXUP_ALPHA > 0:
        print(f"\nMixup enabled: α={MIXUP_ALPHA} (re-shuffled each epoch)")

        def apply_mixup(X, y_oh, weights, alpha, rng):
            """Mix pairs of samples with λ ~ Beta(α, α)."""
            n = len(X)
            idx2 = rng.permutation(n)
            lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
            lam = np.maximum(lam, 1.0 - lam)  # keep dominant sample
            lam_col = lam[:, None]
            X_m = (lam_col * X + (1 - lam_col) * X[idx2]).astype(np.float32)
            y_m = (lam_col * y_oh + (1 - lam_col) * y_oh[idx2]).astype(np.float32)
            w_m = (lam * weights + (1 - lam) * weights[idx2]).astype(np.float32)
            return X_m, y_m, w_m

        class MixupCallback(tf.keras.callbacks.Callback):
            """Re-generate mixup pairs at the start of each epoch."""
            def __init__(self):
                super().__init__()
                self.rng = np.random.RandomState(RANDOM_SEED)
                self.X_mixed = None
                self.y_mixed = None
                self.w_mixed = None

            def on_epoch_begin(self, epoch, logs=None):
                self.X_mixed, self.y_mixed, self.w_mixed = apply_mixup(
                    X_train_f, y_train_oh, sample_weights, MIXUP_ALPHA, self.rng,
                )

        mixup_cb = MixupCallback()
        mixup_cb.on_epoch_begin(0)  # prepare first epoch

        def train_generator():
            while True:
                X_m, y_m, w_m = mixup_cb.X_mixed, mixup_cb.y_mixed, mixup_cb.w_mixed
                indices = np.random.permutation(len(X_m))
                for start in range(0, len(X_m), batch_size):
                    idx = indices[start:start + batch_size]
                    yield X_m[idx], y_m[idx], w_m[idx]

        # Include the final partial batch so every training window is used.
        steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
        train_ds = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 1024), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        ).prefetch(tf.data.AUTOTUNE)
    else:
        print("\nMixup disabled")
        mixup_cb = None
        train_ds = None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
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
    if mixup_cb:
        callbacks.append(mixup_cb)

    print("\nTraining:")
    if train_ds is not None:
        model.fit(
            train_ds,
            validation_data=(X_val.astype(np.float32), y_val_oh),
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            # train_generator already permutes samples before yielding batches.
            shuffle=False,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        model.fit(
            X_train_f, y_train_oh,
            validation_data=(X_val.astype(np.float32), y_val_oh),
            epochs=100,
            batch_size=batch_size,
            sample_weight=sample_weights,
            callbacks=callbacks,
            verbose=1,
        )

    # evaluate
    print("\nEvaluation on val set:")
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    val_metrics = []
    for i, name in enumerate(LABELS):
        mask = y_val == i
        if mask.sum() == 0:
            continue
        acc = float((y_pred[mask] == i).mean())
        val_metrics.append({"label": name, "windows": int(mask.sum()), "accuracy": round(acc, 3)})
        print(f"  {name:12s}  {mask.sum():5d} windows  acc={acc:.3f}")

    # confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_val, y_pred):
        cm[t, p] += 1

    # save val metrics for web UI
    from datetime import datetime, timezone
    metrics_path = RESULTS / "val_metrics.json"
    metrics_path.write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": val_metrics,
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
    }), encoding="utf-8")
    print(f"Saved: {metrics_path}")
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
        sources_train=json.dumps(all_tr_src),
        sources_val=json.dumps(all_val_src),
        is_aug_train=np.array(all_tr_aug),
        is_aug_val=np.array(all_val_aug),
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
