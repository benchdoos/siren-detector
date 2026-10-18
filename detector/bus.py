"""In-process event bus replacing module-level globals.

One ``EventBus`` instance is created in ``main.py`` and passed to all
modules.  Thread-safe: the inference thread calls ``broadcast()`` /
``broadcast_audio()``, and the asyncio event loop delivers messages to
WebSocket clients.
"""

from __future__ import annotations

import asyncio
import collections
import threading
import time
from typing import Callable

import numpy as np
import soundfile as sf
from fastapi import WebSocket

from .audio import ClipBuffer
from .config import CLIP_DIR, CLIP_SECONDS, DATASET_DIR, SAMPLE_RATE, VALID_LABELS


class _CalibrationRecorder:
    """Accumulates audio samples for a specific source_id."""

    def __init__(self, source_id: int, max_samples: int) -> None:
        self.source_id = source_id
        self.max_samples = max_samples
        self._buf: list[float] = []
        self._done = threading.Event()

    def push(self, samples: np.ndarray) -> None:
        remaining = self.max_samples - len(self._buf)
        if remaining <= 0:
            self._done.set()
            return
        chunk = samples[:remaining]
        self._buf.extend(chunk.tolist())
        if len(self._buf) >= self.max_samples:
            self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def progress(self) -> float:
        return len(self._buf) / self.max_samples if self.max_samples > 0 else 1.0

    def get_audio(self) -> np.ndarray:
        return np.array(self._buf, dtype=np.float32)


class EventBus:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started_at: float = time.time()

        # WebSocket client queues
        self.client_queues: dict[WebSocket, asyncio.Queue] = {}
        # (queue, source_id filter or None for all)
        self.audio_queues: dict[WebSocket, tuple[asyncio.Queue, int | None]] = {}

        # State
        self.event_history: collections.deque = collections.deque(maxlen=100)
        self.last_score: dict = {"type": "score", "top": ""}
        self.feedback: dict[str, str] = {}  # filename → label

        # Per-source clip buffers (source_id → ClipBuffer)
        self.clip_bufs: dict[int, ClipBuffer] = {}

        # Source names (source_id → name), populated by engine
        self.source_names: dict[int, str] = {}

        # Calibration recorder (at most one active at a time)
        self._calibration: _CalibrationRecorder | None = None
        self._calibration_lock = threading.Lock()

        # Integration subscribers (called from asyncio loop context)
        self._subscribers: list[Callable[[dict], None]] = []

    @property
    def clip_buf(self) -> ClipBuffer | None:
        """Default clip buffer (source 0) for backwards compatibility."""
        return self.clip_bufs.get(0)

    @clip_buf.setter
    def clip_buf(self, value: ClipBuffer | None) -> None:
        if value is not None:
            self.clip_bufs[0] = value
        else:
            self.clip_bufs.pop(0, None)

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Register a callback for all broadcast messages (events + scores)."""
        self._subscribers.append(callback)

    # ── broadcast ────────────────────────────────────────────────────────────

    def _enqueue_all(self, msg: dict) -> None:
        for q in self.client_queues.values():
            q.put_nowait(msg)
        for cb in self._subscribers:
            try:
                cb(msg)
            except Exception:
                pass

    def _enqueue_audio(self, data: bytes, source_id: int) -> None:
        dead = []
        for ws, (q, sid_filter) in self.audio_queues.items():
            if sid_filter is not None and sid_filter != source_id:
                continue
            if q.qsize() < 50:
                q.put_nowait(data)
            else:
                dead.append(ws)
        for ws in dead:
            self.audio_queues.pop(ws, None)

    def broadcast(self, msg: dict) -> None:
        if msg.get("type") == "event":
            self.event_history.append(msg)
        elif msg.get("type") == "score":
            self.last_score = msg
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._enqueue_all, msg)

    def broadcast_audio(self, samples: np.ndarray, source_id: int = 0) -> None:
        """Send raw float32 PCM to all listening audio WebSocket clients."""
        # Feed calibration recorder if active for this source
        cal = self._calibration
        if cal is not None and cal.source_id == source_id and not cal.done:
            cal.push(samples)

        if not self.audio_queues or not self.loop or self.loop.is_closed():
            return
        data = samples.astype(np.float32).tobytes()
        self.loop.call_soon_threadsafe(self._enqueue_audio, data, source_id)

    # ── calibration ─────────────────────────────────────────────────────────

    def start_calibration(self, source_id: int, duration_seconds: int) -> bool:
        """Begin recording background audio from *source_id*.

        Returns False if a calibration is already in progress.
        """
        with self._calibration_lock:
            if self._calibration is not None and not self._calibration.done:
                return False
            self._calibration = _CalibrationRecorder(
                source_id, max_samples=duration_seconds * SAMPLE_RATE,
            )
        print(f"[calibrate] started for source {source_id}, {duration_seconds}s", flush=True)
        return True

    def calibration_status(self) -> dict | None:
        """Return current calibration progress, or None if idle."""
        cal = self._calibration
        if cal is None:
            return None
        return {
            "source_id": cal.source_id,
            "progress": round(cal.progress, 2),
            "done": cal.done,
            "duration": cal.max_samples // SAMPLE_RATE if SAMPLE_RATE else 0,
        }

    def cancel_calibration(self) -> None:
        """Discard current calibration recording."""
        with self._calibration_lock:
            self._calibration = None
        print("[calibrate] cancelled", flush=True)

    def finish_calibration(self, source_name: str) -> str | None:
        """Save recorded audio as background WAV. Returns the file path or None."""
        with self._calibration_lock:
            cal = self._calibration
            if cal is None:
                return None
            self._calibration = None

        audio = cal.get_audio()
        if len(audio) < SAMPLE_RATE:
            return None

        out_dir = DATASET_DIR / "background" / source_name
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone as tz
        ts = datetime.now(tz.utc).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"calibration_{ts}.wav"
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
        secs = len(audio) / SAMPLE_RATE
        print(f"[calibrate] saved {path} ({secs:.1f}s)", flush=True)
        return str(path)
