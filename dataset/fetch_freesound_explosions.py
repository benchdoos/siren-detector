#!/usr/bin/env python3
"""Download explosion / blast clips from Freesound for explosion class training data.

Usage:
    FREESOUND_API_KEY=<your_key> python3 dataset/fetch_freesound_explosions.py

Get an API key at https://freesound.org/apiv2/apply/
Only sounds with permissive licenses (CC0, CC BY, CC BY-NC) are downloaded.
Audio is converted to 16 kHz mono WAV and saved to dataset/explosion/.
Already-downloaded Freesound IDs (dl_<id>__* files) are skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://freesound.org/apiv2"
DATASET_DIR = Path(__file__).resolve().parent / "explosion"
SAMPLE_RATE = 16_000
MAX_DURATION = 30  # skip clips longer than 30 s

QUERIES = [
    "explosion",
    "bomb explosion",
    "blast",
    "detonation",
    "grenade explosion",
    "artillery explosion",
    "mortar explosion",
]

ALLOWED_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/",
    "http://creativecommons.org/licenses/by-nc/3.0/",
}


def get_existing_ids(directory: Path) -> set[int]:
    """Scan directory for dl_<id>__* files and return set of Freesound IDs."""
    ids: set[int] = set()
    for f in directory.iterdir():
        if f.name.startswith("dl_"):
            parts = f.stem.split("__")
            try:
                ids.add(int(parts[0].removeprefix("dl_")))
            except ValueError:
                pass
    return ids


def api_get(path: str, params: dict, token: str) -> dict:
    params = dict(params, token=token)
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "siren-detector-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def search_sounds(query: str, token: str, page_size: int = 100) -> list[dict]:
    data = api_get(
        "/search/text/",
        {
            "query": query,
            "fields": "id,name,username,license,previews,duration,tags",
            "page_size": page_size,
            "sort": "score",
            "filter": f"duration:[0 TO {MAX_DURATION}]",
        },
        token,
    )
    return data.get("results", [])


def download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "siren-detector-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert audio to 16 kHz mono WAV using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-sample_fmt", "s16",
                str(dst),
            ],
            capture_output=True, timeout=30, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"    ffmpeg failed: {exc}", file=sys.stderr)
        return False


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip().replace(" ", "-").lower()[:60]


def main() -> int:
    token = os.environ.get("FREESOUND_API_KEY", "").strip()
    if not token:
        print(
            "Set FREESOUND_API_KEY environment variable.\n"
            "Get a key at https://freesound.org/apiv2/apply/",
            file=sys.stderr,
        )
        return 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = get_existing_ids(DATASET_DIR)
    print(f"Already have {len(existing_ids)} Freesound IDs in {DATASET_DIR}")

    print("Searching Freesound for explosion sounds...")
    seen_ids: set[int] = set()
    candidates: list[dict] = []

    for query in QUERIES:
        print(f"  query: {query!r}", flush=True)
        try:
            results = search_sounds(query, token)
        except Exception as exc:
            print(f"  search failed: {exc}", file=sys.stderr)
            continue
        for sound in results:
            sid = sound["id"]
            if sid not in seen_ids and sid not in existing_ids:
                seen_ids.add(sid)
                candidates.append(sound)
        time.sleep(0.5)

    # filter licenses
    allowed = [s for s in candidates if s.get("license", "") in ALLOWED_LICENSES]
    print(f"\nFound {len(candidates)} new unique candidates, {len(allowed)} with allowed license.")

    if not allowed:
        print("Nothing to download.", file=sys.stderr)
        return 0

    downloaded = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for sound in allowed:
            previews = sound.get("previews", {})
            preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
            if not preview_url:
                skipped += 1
                continue

            slug = sanitize(sound["name"])
            sid = sound["id"]
            username = sanitize(sound["username"])
            mp3_path = Path(tmpdir) / f"{sid}.mp3"
            wav_name = f"dl_{sid}__{username}__{slug}.wav"
            wav_path = DATASET_DIR / wav_name

            print(f"  [{downloaded + 1}] {sound['name']} (id={sid}, {sound['duration']:.1f}s)", flush=True)

            try:
                download_file(preview_url, mp3_path)
            except Exception as exc:
                print(f"    download failed: {exc}", file=sys.stderr)
                skipped += 1
                continue

            if not convert_to_wav(mp3_path, wav_path):
                skipped += 1
                continue

            downloaded += 1
            time.sleep(0.3)

    print(f"\nDownloaded and converted {downloaded} clips to {DATASET_DIR}/")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
