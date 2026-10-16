"""Dynamic settings: load, save, apply to running smoothers and clip buffer."""

from __future__ import annotations

import collections
import json
from typing import TYPE_CHECKING

from .config import CLIP_SECONDS, DEFAULT_SETTINGS, SAMPLE_RATE, SETTINGS_PATH

if TYPE_CHECKING:
    from .bus import EventBus
    from .engine.smoother import TemporalSmoother


# Module-level mutable state, initialised by main.py
settings: dict = {}
smoothers: dict[str, TemporalSmoother] = {}


def load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    result = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for k in DEFAULT_SETTINGS:
                if k in saved:
                    result[k] = type(DEFAULT_SETTINGS[k])(saved[k])
        except Exception as exc:
            print(f"Settings load error: {exc}", flush=True)
    return result


def save_settings() -> None:
    """Persist current settings to JSON file."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def apply_clip_seconds(bus: EventBus) -> None:
    """Resize ClipBuffer when clip_seconds setting changes."""
    if bus.clip_buf is None:
        return
    new_seconds = settings.get("clip_seconds", CLIP_SECONDS)
    new_maxlen = int(new_seconds * SAMPLE_RATE)
    if bus.clip_buf._buf.maxlen != new_maxlen:
        bus.clip_buf._buf = collections.deque(bus.clip_buf._buf, maxlen=new_maxlen)
        print(f"[settings] clip buffer resized to {new_seconds}s", flush=True)


def apply_settings_to_smoothers() -> None:
    """Push current settings into running TemporalSmoother instances."""
    for name, smoother in smoothers.items():
        thr_key = f"{name}_threshold"
        if thr_key in settings:
            smoother._threshold = settings[thr_key]
        smoother._possible = settings["possible_triggers"]
        smoother._confirmed = settings["confirmed_triggers"]
        smoother._cooldown_seconds = settings["cooldown_seconds"]
        smoother._refractory_seconds = settings["refractory_seconds"]
        new_maxlen = int(settings["smoothing_window_seconds"] / 0.5) + 1
        if smoother._scores.maxlen != new_maxlen:
            smoother._scores = collections.deque(smoother._scores, maxlen=new_maxlen)
