#!/usr/bin/env python3
"""
Compare two dataset audit snapshots to track improvements.

Usage:
  ml/.venv/bin/python ml/audit_compare.py                           # latest vs previous
  ml/.venv/bin/python ml/audit_compare.py audit_A.json audit_B.json # specific snapshots
  ml/.venv/bin/python ml/audit_compare.py audit_B.json              # latest vs audit_B
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AUDITS_DIR = Path(__file__).resolve().parent / "results" / "audits"


def load_audit(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_audits() -> list[Path]:
    """Find all audit files sorted by name (= by timestamp)."""
    if not AUDITS_DIR.exists():
        return []
    return sorted(AUDITS_DIR.glob("audit_*.json"))


def delta_str(old: float, new: float, fmt: str = ".1f", higher_is_better: bool = True) -> str:
    """Format a delta with arrow and color hint."""
    diff = new - old
    if abs(diff) < 0.001:
        return "  ─"
    arrow = "▲" if diff > 0 else "▼"
    good = (diff > 0) == higher_is_better
    marker = "✓" if good else "✗"
    return f" {arrow}{abs(diff):{fmt}} {marker}"


def compare(old: dict, new: dict) -> None:
    labels = new.get("labels", old.get("labels", []))

    print(f"\n{'='*70}")
    print(f"  Audit comparison")
    print(f"  OLD: {old['timestamp']}")
    print(f"  NEW: {new['timestamp']}")
    print(f"{'='*70}")

    # File counts
    print(f"\n{'─'*70}")
    print("  File counts")
    print(f"{'─'*70}")
    print(f"  {'Class':<12s} {'Old total':>10s} {'New total':>10s} "
          f"{'Old aug%':>9s} {'New aug%':>9s} {'Change':>10s}")
    for label in labels:
        o = old.get("file_counts", {}).get(label, {})
        n = new.get("file_counts", {}).get(label, {})
        ot, nt = o.get("total", 0), n.get("total", 0)
        oa, na = o.get("aug_pct", 0), n.get("aug_pct", 0)
        d = delta_str(oa, na, ".1f", higher_is_better=False)
        print(f"  {label:<12s} {ot:>10d} {nt:>10d} {oa:>8.1f}% {na:>8.1f}%{d}")

    # Accuracy
    old_acc = old.get("val_accuracy", {})
    new_acc = new.get("val_accuracy", {})
    if old_acc or new_acc:
        print(f"\n{'─'*70}")
        print("  Val accuracy")
        print(f"{'─'*70}")
        print(f"  {'Class':<12s} {'Old':>10s} {'New':>10s} {'Delta':>12s}  "
              f"{'Old win':>8s} {'New win':>8s}")
        for label in labels:
            oa = old_acc.get(label, {}).get("accuracy", 0)
            na = new_acc.get(label, {}).get("accuracy", 0)
            ow = old_acc.get(label, {}).get("windows", 0)
            nw = new_acc.get(label, {}).get("windows", 0)
            d = delta_str(oa, na, ".3f", higher_is_better=True)
            print(f"  {label:<12s} {oa:>9.1%} {na:>9.1%} {d:>12s}  {ow:>8d} {nw:>8d}")

        # Weighted average accuracy
        def weighted_acc(acc_dict):
            total_w, total_correct = 0, 0.0
            for c in acc_dict.values():
                w = c.get("windows", 0)
                total_w += w
                total_correct += w * c.get("accuracy", 0)
            return total_correct / total_w if total_w else 0

        owa = weighted_acc(old_acc)
        nwa = weighted_acc(new_acc)
        d = delta_str(owa, nwa, ".3f", higher_is_better=True)
        print(f"  {'WEIGHTED':<12s} {owa:>9.1%} {nwa:>9.1%} {d:>12s}")

    # Imbalance
    old_imb = old.get("val_imbalance", {})
    new_imb = new.get("val_imbalance", {})
    if old_imb or new_imb:
        print(f"\n{'─'*70}")
        print("  Class imbalance (ratio to min class)")
        print(f"{'─'*70}")
        print(f"  {'Class':<12s} {'Old ratio':>10s} {'New ratio':>10s} {'Change':>10s}")
        for label in labels:
            o_r = old_imb.get(label, {}).get("ratio_to_min", 0)
            n_r = new_imb.get(label, {}).get("ratio_to_min", 0)
            d = delta_str(o_r, n_r, ".1f", higher_is_better=False)
            print(f"  {label:<12s} {o_r:>9.1f}× {n_r:>9.1f}×{d}")

    # Leakage
    old_leak = old.get("data_leakage", {})
    new_leak = new.get("data_leakage", {})
    if old_leak or new_leak:
        print(f"\n{'─'*70}")
        print("  Data leakage (aug files from val sources)")
        print(f"{'─'*70}")
        print(f"  {'Class':<12s} {'Old leaked':>11s} {'New leaked':>11s} {'Change':>10s}")
        for label in labels:
            o_l = old_leak.get(label, {}).get("leaked_files", 0) if isinstance(old_leak.get(label), dict) else 0
            n_l = new_leak.get(label, {}).get("leaked_files", 0) if isinstance(new_leak.get(label), dict) else 0
            d = delta_str(o_l, n_l, ".0f", higher_is_better=False)
            print(f"  {label:<12s} {o_l:>11d} {n_l:>11d}{d}")

    # Issues
    old_issues = set(old.get("issues", []))
    new_issues = set(new.get("issues", []))
    resolved = old_issues - new_issues
    added = new_issues - old_issues

    if resolved or added:
        print(f"\n{'─'*70}")
        print("  Issue changes")
        print(f"{'─'*70}")
        for issue in sorted(resolved):
            print(f"  ✓ RESOLVED: {issue}")
        for issue in sorted(added):
            print(f"  ✗ NEW: {issue}")

    print()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]

    if len(args) == 2:
        old_path = Path(args[0]) if "/" in args[0] else AUDITS_DIR / args[0]
        new_path = Path(args[1]) if "/" in args[1] else AUDITS_DIR / args[1]
    elif len(args) == 1:
        latest = AUDITS_DIR / "latest.json"
        if not latest.exists():
            print("No latest.json found. Run audit.py first.")
            sys.exit(1)
        new_path = latest.resolve()
        old_path = Path(args[0]) if "/" in args[0] else AUDITS_DIR / args[0]
    else:
        # Latest vs previous
        audits = find_audits()
        if len(audits) < 2:
            print(f"Need at least 2 audit snapshots to compare (found {len(audits)}).")
            print("Run `make audit` after making changes to create a new snapshot.")
            sys.exit(1)
        old_path = audits[-2]
        new_path = audits[-1]

    if not old_path.exists():
        print(f"Not found: {old_path}")
        sys.exit(1)
    if not new_path.exists():
        print(f"Not found: {new_path}")
        sys.exit(1)

    compare(load_audit(old_path), load_audit(new_path))


if __name__ == "__main__":
    main()
