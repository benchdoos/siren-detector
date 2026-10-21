#!/usr/bin/env python3
"""Package the current retrain outputs for a GitHub Release."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "ml" / "results"
ARCHIVE_ROOT = RESULTS / "retrain_archive"

REQUIRED_ASSETS = (
    "classifier_weights.npz",
    "classifier_labels.json",
    "model_card.json",
    "val_sources.json",
    "saved_embeddings.npz",
)
OPTIONAL_ASSETS = (
    # Keep the Keras models for restoring or continuing training.
    "classifier.keras",
    "val_metrics.json",
    "dsp_classifier.keras",
    "dsp_classifier_f32.tflite",
    "dsp_classifier_int8.tflite",
    "dsp_classifier_labels.json",
    "dsp_norm_stats.npz",
    "dsp_metrics.json",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_metrics(labels: list[str], matrix: list[list[int]]) -> tuple[dict, dict]:
    rows = [sum(row) for row in matrix]
    total = sum(rows)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    per_class = {}
    recalls = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        recall = tp / rows[i] if rows[i] else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        per_class[label] = {
            "correct": tp,
            "windows": rows[i],
            "accuracy_recall": recall,
            "precision": precision,
            "f1": f1,
        }
    summary = {
        "correct_windows": correct,
        "total_windows": total,
        "overall_accuracy": correct / total if total else 0.0,
        "macro_class_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
    }
    return summary, per_class


def make_release_notes(metadata: dict, card: dict) -> str:
    metrics = metadata["validation"]
    lines = [
        f"# {metadata['release_title']}",
        "",
        f"**Release tag:** `{metadata['release_tag']}`  ",
        f"**Source commit:** `{metadata['source_commit']}`  ",
        f"**Source commit message:** {metadata['source_subject']}",
        "",
        "## Validation results",
        "",
        f"- Overall accuracy: **{metrics['correct_windows']:,}/{metrics['total_windows']:,} = {metrics['overall_accuracy']:.2%}**.",
        f"- Macro-average class accuracy: **{metrics['macro_class_accuracy']:.2%}**.",
        f"- Labels: {', '.join(f'`{label}`' for label in metadata['labels'])}.",
        "",
        "| Class | Correct / total | Accuracy / recall | Precision | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in metadata["labels"]:
        item = metrics["per_class"][label]
        lines.append(
            f"| `{label}` | {item['correct']:,} / {item['windows']:,} "
            f"| {item['accuracy_recall']:.2%} | {item['precision']:.2%} | {item['f1']:.2%} |"
        )

    counts = card.get("file_counts", {})
    lines += ["", "## Training data inventory", "", "| Class | Total files | Original | Augmented |", "|---|---:|---:|---:|"]
    for label in metadata["labels"]:
        item = counts.get(label, {})
        lines.append(
            f"| `{label}` | {item.get('total', '—')} | {item.get('real', '—')} | {item.get('aug', '—')} |"
        )
    lines += [
        "",
        "These metrics come from the validation confusion matrix in `model_card.json`. Validation sources may change between retrains, so results across releases are not necessarily a strict apples-to-apples comparison.",
    ]

    dsp = card.get("dsp_metrics") or {}
    if dsp.get("classes"):
        lines += [
            "",
            "## Separate DSP evaluation",
            "",
            f"Accuracy: **{dsp.get('accuracy', 0):.2%}** ({dsp.get('validation_windows', 0):,} windows). This is a separate evaluation set; do not combine it with the validation results above.",
            "",
            "| Class | Windows | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in dsp["classes"]:
            lines.append(
                f"| `{item['label']}` | {item['windows']:,} | {item['precision']:.2%} "
                f"| {item['recall']:.2%} | {item['f1']:.2%} |"
            )

    lines += ["", "## Release assets", ""]
    for filename in metadata["asset_files"]:
        lines.append(f"- `{filename}`")
    emb = metadata["files"]["saved_embeddings.npz"]
    lines += [
        "",
        f"`saved_embeddings.npz` is included for continuing training ({emb['bytes']:,} bytes; SHA-256 `{emb['sha256']}`).",
        "",
        "When publishing after history cleanup, target the release tag at the cleaned `main` commit. The source retrain commit is recorded for provenance; tagging it would keep the old history reachable.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    missing = [name for name in REQUIRED_ASSETS if not (RESULTS / name).is_file()]
    if missing:
        raise SystemExit("Missing retrain output(s): " + ", ".join(missing))

    embeddings = RESULTS / "saved_embeddings.npz"
    with embeddings.open("rb") as stream:
        npz_header = stream.read(4)
    if embeddings.stat().st_size < 4 or npz_header != b"PK\x03\x04":
        raise SystemExit("saved_embeddings.npz is missing or is not a materialized NPZ file")

    card = json.loads((RESULTS / "model_card.json").read_text(encoding="utf-8"))
    labels = card.get("labels", [])
    matrix = card.get("confusion_matrix")
    if not labels or not matrix or len(matrix) != len(labels):
        raise SystemExit("model_card.json has no usable validation confusion matrix")
    validation, per_class = class_metrics(labels, matrix)

    commit = git("rev-parse", "HEAD")
    subject = git("show", "-s", "--format=%s", "HEAD")
    branch = git("branch", "--show-current")
    created = datetime.now(timezone.utc).replace(microsecond=0)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    short = commit[:10]
    slug = f"siren-dsp-retrain-{stamp}-{short}"
    release_tag = f"retrain-{stamp}-{short}"
    release_title = f"DSP retrain {created.strftime('%Y-%m-%d %H:%M UTC')} ({short})"

    assets = [name for name in REQUIRED_ASSETS + OPTIONAL_ASSETS if (RESULTS / name).is_file()]
    files = {
        name: {"bytes": (RESULTS / name).stat().st_size, "sha256": sha256(RESULTS / name)}
        for name in assets
    }
    metadata = {
        "schema_version": 1,
        "release_title": release_title,
        "release_tag": release_tag,
        "created_at": created.isoformat(),
        "source_commit": commit,
        "source_branch": branch,
        "source_subject": subject,
        "labels": labels,
        "validation": {**validation, "per_class": per_class},
        "asset_files": assets,
        "files": files,
    }

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    final_dir = ARCHIVE_ROOT / slug
    final_zip = ARCHIVE_ROOT / f"{release_tag}.zip"
    if final_dir.exists() or final_zip.exists():
        raise SystemExit(f"Refusing to overwrite existing package: {slug}")

    with tempfile.TemporaryDirectory(prefix=".retrain-package-", dir=ARCHIVE_ROOT) as temp:
        stage = Path(temp) / slug
        stage.mkdir()
        for name in assets:
            shutil.copy2(RESULTS / name, stage / name)
        (stage / "release-manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = (
            f"# {release_title}\n\n"
            f"- **Release tag:** `{release_tag}`\n"
            f"- **Source commit:** `{commit}` ({branch or 'detached HEAD'})\n"
            f"- **Overall validation accuracy:** {validation['overall_accuracy']:.2%}\n"
            f"- **Macro class accuracy:** {validation['macro_class_accuracy']:.2%}\n\n"
            "This package contains the classifier and training artifacts from the current `make retrain` run. "
            "`saved_embeddings.npz` is included so training can continue from the saved embeddings. "
            "See [`RELEASE_NOTES.md`](RELEASE_NOTES.md) for per-class metrics and the asset list.\n\n"
            "Validation sources can change across retrains, so metrics are not always directly comparable. "
            "After repository history cleanup, point the GitHub Release tag at the cleaned `main` commit.\n"
        )
        (stage / "README.md").write_text(readme, encoding="utf-8")
        notes = make_release_notes(metadata, card)
        (stage / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")
        metadata["documentation"] = {
            "README.md": hashlib.sha256(readme.encode()).hexdigest(),
            "RELEASE_NOTES.md": hashlib.sha256(notes.encode()).hexdigest(),
        }
        (stage / "release-manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        staged_zip = Path(temp) / f"{release_tag}.zip"
        with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(stage.iterdir()):
                archive.write(path, arcname=f"{slug}/{path.name}")

        shutil.move(str(stage), final_dir)
        shutil.move(str(staged_zip), final_zip)

    print(f"Retrain package created: {final_dir}")
    print(f"GitHub Release title: {release_title}")
    print(f"GitHub Release tag:   {release_tag}")
    print(f"Upload asset:         {final_zip}")
    print(f"Release description:  {final_dir / 'RELEASE_NOTES.md'}")


if __name__ == "__main__":
    main()
