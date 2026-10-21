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
            # Keep versioned extension blocks (for example engine_configs and
            # engine_routes) intact across upgrades that do not know their fields.
            for k, value in saved.items():
                if k not in DEFAULT_SETTINGS:
                    result[k] = value
        except Exception as exc:
            print(f"Settings load error: {exc}", flush=True)
    return result


def save_settings() -> None:
    """Persist current settings to JSON file."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def apply_clip_seconds(bus: EventBus) -> None:
    """Resize all ClipBuffers when clip_seconds setting changes."""
    new_seconds = max(
        settings.get("clip_seconds", CLIP_SECONDS),
        settings.get("user_event_clip_seconds", 15),
    )
    new_maxlen = int(new_seconds * SAMPLE_RATE)
    for source_id, clip_buf in bus.clip_bufs.items():
        if clip_buf._buf.maxlen != new_maxlen:
            clip_buf._buf = collections.deque(clip_buf._buf, maxlen=new_maxlen)
            print(f"[settings] clip buffer (source {source_id}) resized to {new_seconds}s", flush=True)


def apply_settings_to_smoothers() -> None:
    """Push current settings into running TemporalSmoother instances.

    Smoother keys may be plain detector names (``siren``) or
    ``source_id/detector`` (``0/siren``) for multi-source setups.
    """
    for key, smoother in smoothers.items():
        # Extract detector name: "0/siren" → "siren", "siren" → "siren"
        det_name = key.split("/")[-1] if "/" in key else key
        thr_key = f"{det_name}_threshold"
        if thr_key in settings:
            smoother._threshold = settings[thr_key]
        smoother._possible = settings["possible_triggers"]
        smoother._confirmed = settings["confirmed_triggers"]
        smoother._cooldown_seconds = settings["cooldown_seconds"]
        smoother._refractory_seconds = settings["refractory_seconds"]
        inference_interval = max(0.25, min(2.0, float(settings.get("inference_interval_seconds", 0.5))))
        new_maxlen = int(settings["smoothing_window_seconds"] / inference_interval) + 1
        if smoother._scores.maxlen != new_maxlen:
            smoother._scores = collections.deque(smoother._scores, maxlen=new_maxlen)
