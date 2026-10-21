"""Minimal third-party-style engine used to demonstrate the component API.

This is an amplitude threshold demo, not a siren classifier. It publishes its
score through the existing ``siren`` UI slot only so the example works with the
current single-engine UI.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections import deque
from datetime import datetime, timezone

import numpy as np

from ..runtime.contracts import ClipSaveRequest, RuntimeMessage
from ..runtime.manifest import EngineManifest

logger = logging.getLogger(__name__)

ENGINE_MANIFEST = EngineManifest(
    engine_id="energy-threshold-example",
    name="Energy threshold example",
    version="1.0.0",
    description=(
        "Demonstration engine that maps frame RMS energy to the existing siren "
        "score slot. It does not classify sirens."
    ),
    input_contracts=("audio-frame.v1",),
    output_contracts=("score", "event"),
    classes=("siren",),
    settings_schema={
        "type": "object",
        "properties": {
            "energy_event_threshold": {
                "type": "number", "minimum": 0.001, "maximum": 1.0,
                "default": 0.08,
            },
            "energy_release_threshold": {
                "type": "number", "minimum": 0.001, "maximum": 1.0,
                "default": 0.05,
            },
            "energy_clip_seconds": {
                "type": "number", "minimum": 1, "maximum": 60,
                "default": 5,
            },
        },
        "additionalProperties": True,
    },
    resources={"accelerators": ["cpu"], "cpu_cores": 0.1, "memory_mb": 16},
    launch={
        "kind": "python-class",
        "entrypoint": "detector.examples.energy_threshold_engine.EnergyThresholdEngine",
        "selector": "--engine detector.examples.energy_threshold_engine.EnergyThresholdEngine",
    },
)


class EnergyThresholdEngine:
    """Publish a visible demo score/event for normalized audio frames."""

    ENGINE_MANIFEST = ENGINE_MANIFEST

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--energy-event-threshold", type=float, default=0.08)
        parser.add_argument("--energy-release-threshold", type=float, default=0.05)
        parser.add_argument("--energy-clip-seconds", type=float, default=5.0)

    def start(self, runtime, settings: dict, args: argparse.Namespace) -> None:
        event_threshold = float(settings.get(
            "energy_event_threshold", args.energy_event_threshold,
        ))
        release_threshold = float(settings.get(
            "energy_release_threshold", args.energy_release_threshold,
        ))
        clip_seconds = float(settings.get("energy_clip_seconds", args.energy_clip_seconds))
        if event_threshold <= 0 or release_threshold < 0 or release_threshold > event_threshold:
            raise ValueError(
                "Energy thresholds must satisfy 0 <= release threshold <= event threshold"
            )
        if clip_seconds <= 0:
            raise ValueError("energy_clip_seconds must be positive")

        states: dict[int, bool] = {}
        buffers: dict[int, deque[np.ndarray]] = {}
        buffered_samples: dict[int, int] = {}
        lock = threading.Lock()
        source_names: dict[int, str] = {0: args.source}
        for mapping in getattr(args, "source_name", []) or []:
            if "=" in mapping:
                sid, name = mapping.split("=", 1)
                source_names[int(sid)] = name

        def on_audio_frame(frame) -> None:
            samples = frame.samples
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
            score = min(1.0, rms / max(event_threshold * 2.0, 1e-12))
            source_id = frame.source_id
            source_name = source_names.get(source_id, f"source-{source_id}")
            runtime.event_bus.source_names[source_id] = source_name

            with lock:
                clip_buffer = buffers.setdefault(source_id, deque())
                clip_buffer.append(samples.copy())
                buffered_samples[source_id] = buffered_samples.get(source_id, 0) + samples.size
                max_samples = max(1, int(clip_seconds * frame.sample_rate_hz))
                while clip_buffer and buffered_samples[source_id] - clip_buffer[0].size >= max_samples:
                    buffered_samples[source_id] -= clip_buffer.popleft().size

                was_active = states.get(source_id, False)
                is_active = rms >= (release_threshold if was_active else event_threshold)
                states[source_id] = is_active
                emit_event = is_active != was_active
                clip = (
                    np.concatenate(tuple(clip_buffer))[-max_samples:].astype(np.float32, copy=False)
                    if emit_event and is_active else None
                )

            state = "confirmed" if is_active else "idle"
            runtime.publish(RuntimeMessage.from_legacy({
                "type": "score",
                "top": "Energy threshold demo",
                "source": source_name,
                "source_id": source_id,
                "source_names": {sid: name for sid, name in source_names.items()},
                "siren_score": round(score, 4),
                "siren_state": state,
                "siren_triggers": int(is_active),
                "siren_possible_needed": 1,
                "siren_confirmed_needed": 1,
                "siren_cooldown": 0,
                "siren_refractory": 0,
            }))

            if emit_event:
                timestamp = frame.timestamp or datetime.now(timezone.utc).isoformat()
                clip_url = None
                if is_active and clip is not None and clip.size:
                    filename = f"energy-demo-{source_id}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}.wav"
                    try:
                        clip_url = runtime.save_clip(ClipSaveRequest(
                            filename=filename,
                            sample_rate_hz=frame.sample_rate_hz,
                            samples=clip,
                            metadata={
                                "event": "siren", "confidence": round(score, 4),
                                "timestamp": timestamp, "source": source_name,
                                "source_id": source_id,
                                "engine": ENGINE_MANIFEST.engine_id,
                                "threshold": event_threshold,
                                "demo_only": True,
                            },
                        ))
                    except Exception:
                        logger.exception("Could not save example engine event clip")
                runtime.publish(RuntimeMessage.from_legacy({
                    "type": "event",
                    "event": "siren",
                    "state": state,
                    "confidence": round(score, 4),
                    "timestamp": timestamp,
                    "source": source_name,
                    "source_id": source_id,
                    "engine": ENGINE_MANIFEST.engine_id,
                    "clip_url": clip_url,
                    "threshold": event_threshold,
                    "triggers": int(is_active),
                    "triggers_needed": 1,
                    "demo_only": True,
                }))

        runtime.subscribe_audio(
            on_audio_frame, subscriber_id=ENGINE_MANIFEST.engine_id,
        )
        logger.info(
            "Started energy threshold example (event %.3f, release %.3f)",
            event_threshold, release_threshold,
        )
