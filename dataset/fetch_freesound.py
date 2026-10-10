#!/usr/bin/env python3
"""Download civil-defence siren clips from Freesound for siren class training data.

Usage:
    FREESOUND_API_KEY=<your_key> python3 dataset/fetch_freesound.py

Get an API key at https://freesound.org/apiv2/apply/
Only sounds with permissive licenses (CC0, CC BY, CC BY-NC) are downloaded.
Audio files are saved to dataset/external/freesound/audio/ (git-ignored).
Metadata and provenance are saved alongside and ARE committed.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://freesound.org/apiv2"
DESTINATION = Path(__file__).resolve().parent / "external" / "freesound"
MAX_RESULTS = 150  # cap total candidates before dedup

# Civil defence / air raid sirens only — NOT emergency vehicle sirens.
# Queries are run in order; results are merged and deduplicated by sound ID.
QUERIES = [
    "civil defense siren",
    "civil defence siren",
    "air raid siren",
    "air raid warning",
    "ГО сирена",
]

# Only download sounds under these licenses.
ALLOWED_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by/3.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/",
    "http://creativecommons.org/licenses/by-nc/3.0/",
}


def api_get(path: str, params: dict, token: str) -> dict:
    params = dict(params, token=token)
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "siren-detector-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def search_sounds(query: str, token: str, page_size: int = 50) -> list[dict]:
    """Return up to page_size results for a text query."""
    data = api_get(
        "/search/text/",
        {
            "query": query,
            "fields": "id,name,username,license,previews,duration,description",
            "page_size": page_size,
            "sort": "score",
        },
        token,
    )
    return data.get("results", [])


def download_file(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "siren-detector-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def main() -> int:
    token = os.environ.get("FREESOUND_API_KEY", "").strip()
    if not token:
        print(
            "Set FREESOUND_API_KEY environment variable.\n"
            "Get a key at https://freesound.org/apiv2/apply/",
            file=sys.stderr,
        )
        return 1

    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        print(f"Refusing to overwrite non-empty directory: {DESTINATION}", file=sys.stderr)
        print("Delete it first if you want a fresh download.", file=sys.stderr)
        return 2

    print("Searching Freesound for civil-defence siren sounds...")
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
            if sound["id"] not in seen_ids:
                seen_ids.add(sound["id"])
                candidates.append(sound)
        time.sleep(0.5)  # be polite to the API

    print(f"Found {len(candidates)} unique candidates.")

    allowed = [s for s in candidates if s.get("license", "") in ALLOWED_LICENSES]
    print(f"{len(allowed)} sounds have an allowed license (CC0 / CC BY / CC BY-NC).")

    if not allowed:
        print("Nothing to download. Check your queries or license filter.", file=sys.stderr)
        return 1

    audio_dir = DESTINATION / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_records: list[dict] = []
    downloaded = 0
    skipped = 0

    for sound in allowed:
        previews = sound.get("previews", {})
        preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not preview_url:
            skipped += 1
            continue

        filename = f"freesound_{sound['id']}.mp3"
        dest = audio_dir / filename
        print(f"  [{downloaded + 1}/{len(allowed)}] {sound['name']} (id={sound['id']}, {sound['duration']:.1f}s)", flush=True)
        try:
            download_file(preview_url, dest)
        except Exception as exc:
            print(f"    download failed: {exc}", file=sys.stderr)
            skipped += 1
            continue

        metadata_records.append(
            {
                "id": sound["id"],
                "filename": filename,
                "name": sound["name"],
                "username": sound["username"],
                "license": sound["license"],
                "duration_s": sound["duration"],
                "freesound_url": f"https://freesound.org/s/{sound['id']}/",
                "description": sound.get("description", "")[:200],
            }
        )
        downloaded += 1
        time.sleep(0.3)

    if not downloaded:
        print("No files downloaded.", file=sys.stderr)
        return 1

    metadata_path = DESTINATION / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_note = (
        "# Freesound source record\n\n"
        f"- Retrieved (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}\n"
        f"- Queries: {', '.join(repr(q) for q in QUERIES)}\n"
        f"- Downloaded clips: {downloaded}\n"
        f"- Skipped: {skipped}\n"
        "- Licenses: CC0, CC BY 3.0/4.0, CC BY-NC 3.0/4.0\n"
        "- Audio format: Freesound HQ preview MP3 (~128 kbps)\n"
        "- Audio files are git-ignored; metadata.json is committed.\n"
        "- These are CIVIL DEFENCE / AIR RAID sirens — not emergency vehicle sirens.\n"
        "- Do not mix into dataset/siren/ — use dataset/external/freesound/ only.\n"
        "- Individual licenses and attribution are in metadata.json.\n"
    )
    (DESTINATION / "SOURCE.md").write_text(source_note, encoding="utf-8")

    print(f"\nDownloaded {downloaded} clips to {DESTINATION}/audio/")
    print(f"Metadata saved to {DESTINATION}/metadata.json")
    print("Review individual licenses in metadata.json before reuse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
