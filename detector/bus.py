"""In-process event bus replacing module-level globals.

One ``EventBus`` instance is created in ``main.py`` and passed to all
modules.  Thread-safe: the inference thread calls ``broadcast()`` /
``broadcast_audio()``, and the asyncio event loop delivers messages to
WebSocket clients.
"""

from __future__ import annotations

import asyncio
import collections
from typing import Callable

import numpy as np
from fastapi import WebSocket

from .audio import ClipBuffer
from .config import CLIP_DIR, CLIP_SECONDS, DATASET_DIR, SAMPLE_RATE, VALID_LABELS


class EventBus:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None

        # WebSocket client queues
        self.client_queues: dict[WebSocket, asyncio.Queue] = {}
        self.audio_queues: dict[WebSocket, asyncio.Queue] = {}

        # State
        self.event_history: collections.deque = collections.deque(maxlen=100)
        self.last_score: dict = {"type": "score", "top": ""}
        self.feedback: dict[str, str] = {}  # filename → label
        self.clip_buf: ClipBuffer | None = None

        # Integration subscribers (called from asyncio loop context)
        self._subscribers: list[Callable[[dict], None]] = []

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

    def _enqueue_audio(self, data: bytes) -> None:
        dead = []
        for ws, q in self.audio_queues.items():
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

    def broadcast_audio(self, samples: np.ndarray) -> None:
        """Send raw float32 PCM to all listening audio WebSocket clients."""
        if not self.audio_queues or not self.loop or self.loop.is_closed():
            return
        data = samples.astype(np.float32).tobytes()
        self.loop.call_soon_threadsafe(self._enqueue_audio, data)
