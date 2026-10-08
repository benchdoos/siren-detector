#!/usr/bin/env python3
"""Download a small, provenance-preserving ESC-50 subset for local experiments."""

from __future__ import annotations

import csv
import io
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DESTINATION = Path(__file__).resolve().parent / "external" / "esc50"
CATEGORIES = {
    "siren",
    "rain",
    "wind",
    "washing_machine",
    "keyboard_typing",
    "engine",
    "fireworks",
}


def main() -> int:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        print(f"Refusing to overwrite non-empty directory: {DESTINATION}", file=sys.stderr)
        return 2

    print("Downloading the official ESC-50 archive (~600 MB)...", flush=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "siren-detector-dataset-setup"})
    try:
        with tempfile.TemporaryDirectory(prefix="esc50-") as temporary_directory:
            archive_path = Path(temporary_directory) / "esc50.zip"
            with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            return install_subset(archive_path)
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1


def install_subset(archive_path: Path) -> int:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            csv_name = next(name for name in names if name.endswith("/meta/esc50.csv"))
            root = csv_name.rsplit("/meta/esc50.csv", 1)[0]
            metadata = list(csv.DictReader(io.TextIOWrapper(archive.open(csv_name), encoding="utf-8")))
            selected = [row["filename"] for row in metadata if row["category"] in CATEGORIES]
            DESTINATION.mkdir(parents=True, exist_ok=True)
            (DESTINATION / "audio").mkdir()
            (DESTINATION / "meta").mkdir()
            for filename in selected:
                source = f"{root}/audio/{filename}"
                with archive.open(source) as src, (DESTINATION / "audio" / filename).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            for relative in ("LICENSE", "meta/esc50.csv"):
                source = f"{root}/{relative}"
                target = DESTINATION / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(source))
            categories = ", ".join(sorted(CATEGORIES))
            source_note = (
                "# ESC-50 source record\n\n"
                f"- Retrieved (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}\n"
                f"- Upstream: {URL}\n"
                "- Imported categories: " + categories + "\n"
                f"- Imported clips: {len(selected)}\n"
                "- Audio license: see the adjacent upstream LICENSE; ESC-50 is "
                "CC BY-NC 3.0 as a whole.\n"
                "- Metadata: the full upstream meta/esc50.csv is retained; only "
                "selected audio files are extracted.\n"
            )
            (DESTINATION / "SOURCE.md").write_text(source_note, encoding="utf-8")
    except Exception as error:
        shutil.rmtree(DESTINATION, ignore_errors=True)
        print(f"Could not extract ESC-50 subset: {error}", file=sys.stderr)
        return 1

    print(f"Installed {len(selected)} selected clips in {DESTINATION}")
    print("Review external/esc50/LICENSE and the source metadata before reuse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
