"""WAV file input adapter for repeatable local inference and evaluation."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from ..runtime.contracts import AudioFrame


class FileAudioAdapter:
    """Publish mono 16 kHz WAV audio as normalized frames.

    Files with a different sample rate or multiple channels are rejected rather
    than silently resampled or downmixed. Playback is immediate by default;
    ``realtime=True`` paces frames according to their audio duration.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: int = 1000,
        chunk_samples: int = 1600,
        realtime: bool = False,
    ) -> None:
        if source_id < 0:
            raise ValueError("source_id must be non-negative")
        if chunk_samples < 1:
            raise ValueError("chunk_samples must be positive")
        self.path = Path(path)
        self.source_id = source_id
        self.chunk_samples = chunk_samples
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, publish_frame: Callable[[AudioFrame], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        samples, sample_rate = sf.read(self.path, dtype="float32", always_2d=True)
        if sample_rate != 16_000:
            raise ValueError(f"File input requires 16000 Hz WAV, got {sample_rate} Hz")
        if samples.shape[1] != 1:
            raise ValueError(f"File input requires mono WAV, got {samples.shape[1]} channels")
        mono = np.ascontiguousarray(samples[:, 0], dtype=np.float32)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(publish_frame, mono), daemon=True,
            name=f"audio-file-{self.source_id}",
        )
        self._thread.start()

    def _run(self, publish_frame: Callable[[AudioFrame], None], mono: np.ndarray) -> None:
        sequence = 0
        for offset in range(0, len(mono), self.chunk_samples):
            if self._stop.is_set():
                return
            chunk = mono[offset:offset + self.chunk_samples].copy()
            publish_frame(AudioFrame(
                source_id=self.source_id,
                sequence=sequence,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                sample_rate_hz=16_000,
                samples=chunk,
                metadata={"transport": "file/wav", "path": self.path.name},
            ))
            sequence += 1
            if self.realtime and self._stop.wait(len(chunk) / sample_rate):
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
