"""File-backed clip storage adapter used by the current detector runtime."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from ..config import CLIP_DIR
from .contracts import ClipSaveRequest


class WavClipStorage:
    """Write confirmed-event audio as WAV with a JSON metadata sidecar."""

    def __init__(self, clip_dir: Path = CLIP_DIR) -> None:
        self._clip_dir = clip_dir

    def save_clip(self, request: ClipSaveRequest) -> str:
        filename = Path(request.filename).name
        if filename != request.filename or not filename.endswith(".wav"):
            raise ValueError("Clip filename must be a simple .wav filename")

        self._clip_dir.mkdir(parents=True, exist_ok=True)
        path = self._clip_dir / filename
        sf.write(
            str(path), request.samples, request.sample_rate_hz, subtype="PCM_16",
        )
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(dict(request.metadata)), encoding="utf-8")
        return f"/clips/{filename}"
