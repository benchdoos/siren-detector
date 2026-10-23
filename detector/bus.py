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
import uuid
from pathlib import Path
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
        self._buf: list[np.ndarray] = []
        self._sample_count = 0
        self._done = threading.Event()
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            remaining = self.max_samples - self._sample_count
            if remaining <= 0:
                self._done.set()
                return
            chunk = samples[:remaining]
            if len(chunk):
                self._buf.append(np.asarray(chunk, dtype=np.float32).copy())
                self._sample_count += len(chunk)
            if self._sample_count >= self.max_samples:
                self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def progress(self) -> float:
        with self._lock:
            return self._sample_count / self.max_samples if self.max_samples > 0 else 1.0

    def get_audio(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._buf) if self._buf else np.empty(0, dtype=np.float32)


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

        # Active calibration recorders, keyed by source_id.
        self._calibrations: dict[int, _CalibrationRecorder] = {}
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
            # Slow one-way clients must not break delivery to other subscribers.
            # Bounded queues (such as SSE) favor the freshest state update.
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
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
        cal = self._calibrations.get(source_id)
        if cal is not None and not cal.done:
            cal.push(samples)

        if not self.audio_queues or not self.loop or self.loop.is_closed():
            return
        data = samples.astype(np.float32).tobytes()
        self.loop.call_soon_threadsafe(self._enqueue_audio, data, source_id)

    # ── calibration ─────────────────────────────────────────────────────────

    def start_calibration(self, source_ids: list[int], duration_seconds: int) -> bool:
        """Begin synchronized background recording for selected sources."""
        with self._calibration_lock:
            if any(not c.done for c in self._calibrations.values()):
                return False
            self._calibrations = {
                sid: _CalibrationRecorder(sid, max_samples=duration_seconds * SAMPLE_RATE)
                for sid in source_ids
            }
        print(f"[calibrate] started for sources {source_ids}, {duration_seconds}s", flush=True)
        return True

    def calibration_status(self) -> dict | None:
        """Return progress per source, or None if idle."""
        cals = self._calibrations
        if not cals:
            return None
        sources = {sid: {"progress": round(c.progress, 2), "done": c.done}
                   for sid, c in cals.items()}
        first = next(iter(cals.values()))
        return {
            "source_ids": list(cals), "sources": sources,
            "progress": min((x["progress"] for x in sources.values()), default=0),
            "done": all(c.done for c in cals.values()),
            "duration": first.max_samples // SAMPLE_RATE if SAMPLE_RATE else 0,
        }

    def cancel_calibration(self) -> None:
        """Discard current calibration recording."""
        with self._calibration_lock:
            self._calibrations = {}
        print("[calibrate] cancelled", flush=True)

    def finish_calibration(self) -> dict[int, str]:
        """Save each recorded source as a background WAV; return paths by source."""
        with self._calibration_lock:
            cals = dict(self._calibrations)
        from datetime import datetime, timezone as tz
        # Microseconds prevent a second calibration from overwriting an
        # existing recording when it is saved in the same second.
        ts = datetime.now(tz.utc).strftime("%Y%m%d_%H%M%S_%f")
        paths = {}
        discard_ids: list[int] = []
        tmp_path: Path | None = None
        saving_sid: int | None = None
        try:
            for sid, cal in cals.items():
                saving_sid = sid
                audio = cal.get_audio()
                if len(audio) < SAMPLE_RATE:
                    discard_ids.append(sid)
                    continue
                source_name = self.source_names.get(sid, f"source-{sid}")
                out_dir = DATASET_DIR / "background" / source_name
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"calibration_{ts}.wav"
                # Publish only a complete WAV.  A spectrogram request can race
                # with saving immediately after the stop action, so write to a
                # sibling temporary file and atomically rename it into place.
                tmp_path = out_dir / f".{path.stem}.{uuid.uuid4().hex}.wav"
                sf.write(
                    str(tmp_path), audio, SAMPLE_RATE,
                    subtype="PCM_16",
                )
                tmp_path.replace(path)
                paths[sid] = str(path)
                print(f"[calibrate] saved {path} ({len(audio) / SAMPLE_RATE:.1f}s)", flush=True)
        except Exception as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            print(f"[calibrate] failed to save source {saving_sid}: {exc}", flush=True)
            raise
        finally:
            # Do not discard a recording until its WAV is safely published.
            # If saving fails, the active recorder remains available for a
            # retry through the same Stop & Save action.
            with self._calibration_lock:
                for sid in [*discard_ids, *paths]:
                    if self._calibrations.get(sid) is cals[sid]:
                        self._calibrations.pop(sid)
        return paths
