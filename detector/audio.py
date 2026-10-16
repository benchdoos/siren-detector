"""Audio helpers: I2S conversion, ring buffer, clip buffer."""

from __future__ import annotations

import collections
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


def i2s32_to_float32(payload: bytes) -> np.ndarray:
    n = len(payload) // 4
    samples = np.frombuffer(payload[: n * 4], dtype="<i4")
    return (samples >> 16).astype(np.float32) / 32768.0


class AudioRingBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: collections.deque = collections.deque(maxlen=capacity)
        self._pending = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(samples.tolist())
            self._pending += len(samples)

    def consume(self, hop: int) -> np.ndarray | None:
        with self._lock:
            if self._pending < hop:
                return None
            self._pending = 0
            return np.array(self._buf, dtype=np.float32)


class ClipBuffer:
    """Rolling buffer for saving audio clips around events."""

    def __init__(self, seconds: float) -> None:
        self._buf: collections.deque = collections.deque(maxlen=int(seconds * SAMPLE_RATE))
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(samples.tolist())

    def save(self, path: Path, last_seconds: float | None = None) -> None:
        with self._lock:
            audio = np.array(self._buf, dtype=np.float32)
        if last_seconds is not None:
            max_samples = int(last_seconds * SAMPLE_RATE)
            if len(audio) > max_samples:
                audio = audio[-max_samples:]
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
