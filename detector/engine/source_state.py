"""Per-source inference state for multi-microphone support."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from ..audio import AudioRingBuffer, ClipBuffer
from .smoother import TemporalSmoother


@dataclass
class SourceState:
    """Holds all per-source buffers and inference state.

    Each audio source (ESP32 mic, USB mic, etc.) gets its own
    SourceState with independent buffers and smoothers.
    """

    source_id: int
    name: str
    inference_buf: AudioRingBuffer
    clip_buf: ClipBuffer
    smoothers: dict[str, TemporalSmoother] = field(default_factory=dict)
    peaks: dict[str, dict] = field(default_factory=dict)
    last_scores: dict[str, float] = field(default_factory=dict)
    possible_start: dict[str, float | None] = field(default_factory=dict)
    window_scores: dict[str, list[dict]] = field(default_factory=dict)
    dsp_feature_buf: collections.deque = field(default_factory=lambda: collections.deque(maxlen=250))
    dsp_last_log: float = 0.0
