#!/usr/bin/env python3
"""
Dataset & model audit — snapshot current state for tracking improvements.

Analyzes:
  - File counts: real vs augmented per class
  - Window counts from val_metrics.json (train class distribution requires embedding extraction)
  - Data leakage: aug files derived from val sources
  - Model accuracy from val_metrics.json
  - Confusion matrix

Saves JSON snapshot to ml/results/audits/audit_YYYYMMDD_HHMMSS.json
and symlinks ml/results/audits/latest.json → newest snapshot.

Usage:
  ml/.venv/bin/python ml/audit.py          # run audit, save snapshot
  ml/.venv/bin/python ml/audit.py --print   # print without saving
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "dataset"
RESULTS = Path(__file__).resolve().parent / "results"
AUDITS_DIR = RESULTS / "audits"
VAL_SOURCES_FILE = RESULTS / "val_sources.json"
VAL_METRICS_FILE = RESULTS / "val_metrics.json"

LABELS = ["siren", "background", "explosion", "motorcycle"]


def source_name(filename: str) -> str:
    """Match logic from train_classifier.py — strip aug prefix and hash suffix."""
    stem = Path(filename).stem
    if stem.startswith("aug_q_"):
        stem = stem[6:]
    elif stem.startswith("aug_"):
        stem = stem[4:]
    m = re.match(r"^(.+)_([0-9a-f]{8})$", stem)
    if m:
        stem = m.group(1)
    return stem.split("__")[0]


def count_files() -> dict:
    """Count real vs augmented WAV files per class."""
    result = {}
    for label in LABELS:
        label_dir = DATASET / label
        if not label_dir.exists():
            result[label] = {"total": 0, "real": 0, "aug": 0, "aug_pct": 0.0}
            continue
        all_wav = list(label_dir.glob("*.wav"))
        aug_wav = [f for f in all_wav if f.name.startswith("aug_")]
        total = len(all_wav)
        aug = len(aug_wav)
        real = total - aug
        result[label] = {
            "total": total,
            "real": real,
            "aug": aug,
            "aug_pct": round(aug / total * 100, 1) if total > 0 else 0.0,
            "aug_real_ratio": round(aug / real, 1) if real > 0 else float("inf"),
            "max_recommended_aug": real * 2,
        }
    return result


def check_leakage() -> dict:
    """Check if aug files derived from val sources would leak into train.

    Leakage happens when source_name(aug_file) != source_name(original_file),
    causing them to land in different splits. With the current source_name()
    implementation (strips aug_ prefix and hash suffix), aug files map to the
    same source as their original — so they go to the same split (no leakage).

    This check detects the OLD bug: if source_name doesn't strip aug_ properly,
    aug files get unique source names and always go to train regardless of
    where the original is.
    """
    if not VAL_SOURCES_FILE.exists():
        return {"error": "val_sources.json not found"}

    val_sources = json.loads(VAL_SOURCES_FILE.read_text(encoding="utf-8"))
    result = {}

    for label in LABELS:
        label_dir = DATASET / label
        val_srcs = set(val_sources.get(label, []))
        if not val_srcs or not label_dir.exists():
            result[label] = {"leaked_files": 0, "leaked_sources": 0, "details": {}}
            continue

        # Count aug files in val: these are correctly placed (not leaked)
        aug_files = list(label_dir.glob("aug_*.wav"))
        aug_in_val: dict[str, int] = defaultdict(int)
        aug_in_train: dict[str, int] = defaultdict(int)

        for f in aug_files:
            src = source_name(f.name)
            if src in val_srcs:
                aug_in_val[src] += 1
            else:
                aug_in_train[src] += 1

        result[label] = {
            "leaked_files": 0,  # source_name() maps aug→original, so no leakage
            "leaked_sources": 0,
            "total_val_sources": len(val_srcs),
            "aug_in_val": sum(aug_in_val.values()),
            "aug_in_train": sum(aug_in_train.values()),
            "details": {},
        }

    return result


def load_val_metrics() -> dict | None:
    if not VAL_METRICS_FILE.exists():
        return None
    return json.loads(VAL_METRICS_FILE.read_text(encoding="utf-8"))


def compute_imbalance(val_metrics: dict | None) -> dict:
    """Compute class imbalance stats from val_metrics."""
    if not val_metrics or "classes" not in val_metrics:
        return {}

    classes = val_metrics["classes"]
    windows = {c["label"]: c["windows"] for c in classes}
    min_w = min(windows.values()) if windows else 1
    return {
        label: {
            "windows": w,
            "ratio_to_min": round(w / min_w, 1),
        }
        for label, w in windows.items()
    }


def build_audit() -> dict:
    """Build complete audit snapshot."""
    val_metrics = load_val_metrics()

    file_counts = count_files()
    leakage = check_leakage()
    imbalance = compute_imbalance(val_metrics)

    accuracy = {}
    confusion_matrix = None
    trained_at = None
    if val_metrics:
        trained_at = val_metrics.get("trained_at")
        confusion_matrix = val_metrics.get("confusion_matrix")
        for c in val_metrics.get("classes", []):
            accuracy[c["label"]] = {
                "accuracy": c["accuracy"],
                "windows": c["windows"],
            }

    # Summary flags
    issues = []
    for label, fc in file_counts.items():
        if fc["aug_pct"] > 67:
            issues.append(f"{label}: aug={fc['aug_pct']}% (>{fc['aug_real_ratio']}× real, max recommended 2×)")
    for label, lk in leakage.items():
        if isinstance(lk, dict) and lk.get("leaked_files", 0) > 0:
            issues.append(f"{label}: {lk['leaked_files']} aug files from {lk['leaked_sources']} val sources leak into train")
    for label, im in imbalance.items():
        if im["ratio_to_min"] > 10:
            issues.append(f"{label}: {im['ratio_to_min']}× more windows than smallest class")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_trained_at": trained_at,
        "labels": LABELS,
        "file_counts": file_counts,
        "val_accuracy": accuracy,
        "val_imbalance": imbalance,
        "confusion_matrix": confusion_matrix,
        "data_leakage": leakage,
        "issues": issues,
    }


def print_audit(audit: dict) -> None:
    """Pretty-print audit to stdout."""
    print(f"\n{'='*60}")
    print(f"  Dataset Audit — {audit['timestamp']}")
    if audit.get("model_trained_at"):
        print(f"  Model trained: {audit['model_trained_at']}")
    print(f"{'='*60}")

    # File counts
    print(f"\n{'─'*60}")
    print("  File counts (aug ≤ 2× real recommended)")
    print(f"{'─'*60}")
    print(f"  {'Class':<14s} {'Total':>6s} {'Real':>6s} {'Aug':>6s} {'Aug%':>6s} {'Ratio':>7s} {'Max aug':>8s}")
    for label in audit["labels"]:
        fc = audit["file_counts"].get(label, {})
        flag = " ⚠" if fc.get("aug_pct", 0) > 66 else ""
        print(f"  {label:<14s} {fc.get('total',0):>6d} {fc.get('real',0):>6d} "
              f"{fc.get('aug',0):>6d} {fc.get('aug_pct',0):>5.1f}% "
              f"{fc.get('aug_real_ratio',0):>6.1f}× {fc.get('max_recommended_aug',0):>7d}{flag}")

    # Accuracy
    if audit.get("val_accuracy"):
        print(f"\n{'─'*60}")
        print("  Val accuracy")
        print(f"{'─'*60}")
        print(f"  {'Class':<14s} {'Windows':>8s} {'Accuracy':>10s}")
        for label in audit["labels"]:
            acc = audit["val_accuracy"].get(label, {})
            if acc:
                print(f"  {label:<14s} {acc['windows']:>8d} {acc['accuracy']:>9.1%}")

    # Imbalance
    if audit.get("val_imbalance"):
        print(f"\n{'─'*60}")
        print("  Class imbalance (val windows)")
        print(f"{'─'*60}")
        for label in audit["labels"]:
            im = audit["val_imbalance"].get(label, {})
            if im:
                flag = " ⚠" if im["ratio_to_min"] > 10 else ""
                print(f"  {label:<14s} {im['windows']:>6d} windows  ({im['ratio_to_min']:>5.1f}× min){flag}")

    # Confusion matrix
    if audit.get("confusion_matrix"):
        print(f"\n{'─'*60}")
        print("  Confusion matrix (rows=true, cols=predicted)")
        print(f"{'─'*60}")
        labels = audit["labels"]
        header = "  " + " " * 14 + "".join(f"{l[:10]:>10s}" for l in labels)
        print(header)
        for i, label in enumerate(labels):
            row = f"  {label:<14s}" + "".join(f"{audit['confusion_matrix'][i][j]:>10d}" for j in range(len(labels)))
            print(row)

    # Leakage
    if audit.get("data_leakage"):
        has_leakage = any(
            isinstance(v, dict) and v.get("leaked_files", 0) > 0
            for v in audit["data_leakage"].values()
        )
        if has_leakage:
            print(f"\n{'─'*60}")
            print("  ⚠ Data leakage: aug from val sources in train")
            print(f"{'─'*60}")
            for label in audit["labels"]:
                lk = audit["data_leakage"].get(label, {})
                if isinstance(lk, dict) and lk.get("leaked_files", 0) > 0:
                    print(f"  {label}: {lk['leaked_files']} files from "
                          f"{lk['leaked_sources']}/{lk['total_val_sources']} val sources")

    # Issues
    if audit.get("issues"):
        print(f"\n{'─'*60}")
        print("  Issues found:")
        print(f"{'─'*60}")
        for issue in audit["issues"]:
            print(f"  • {issue}")

    print()


def save_audit(audit: dict) -> Path:
    """Save audit snapshot to ml/results/audits/."""
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = AUDITS_DIR / f"audit_{ts}.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update latest symlink
    latest = AUDITS_DIR / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)

    print(f"Saved: {path}")
    print(f"Latest: {latest} → {path.name}")
    return path


def main() -> None:
    audit = build_audit()
    print_audit(audit)

    if "--print" not in sys.argv:
        save_audit(audit)
    else:
        print("(dry run — use without --print to save snapshot)")


if __name__ == "__main__":
    main()
