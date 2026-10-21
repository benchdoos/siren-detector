"""Versioned messages exchanged between processing and runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

import numpy as np
from jsonschema import Draft202012Validator


MessageKind = Literal["score", "event"]
DETECTOR_EVENT_CONTRACT = "detector-event.v1"
DETECTOR_EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["contract", "type", "event", "state", "confidence", "timestamp",
                 "source", "source_id", "engine_id"],
    "properties": {
        "contract": {"const": DETECTOR_EVENT_CONTRACT},
        "type": {"const": "event"},
        "event": {"type": "string", "minLength": 1},
        "state": {"enum": ["idle", "possible", "confirmed"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "timestamp": {"type": "string", "minLength": 1},
        "source": {"type": "string"},
        "source_id": {"type": ["integer", "null"], "minimum": 0},
        "engine_id": {"type": "string", "minLength": 1},
    },
    "additionalProperties": True,
}
_DETECTOR_EVENT_VALIDATOR = Draft202012Validator(DETECTOR_EVENT_SCHEMA)


def normalize_detector_event(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy event payload to the versioned public event contract."""
    event = dict(payload)
    if event.get("contract", DETECTOR_EVENT_CONTRACT) != DETECTOR_EVENT_CONTRACT:
        raise ValueError(f"unsupported event contract: {event.get('contract')!r}")
    event.setdefault("contract", DETECTOR_EVENT_CONTRACT)
    event.setdefault("state", "confirmed")
    event.setdefault("confidence", 0.0)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    event.setdefault("source", "unknown")
    event.setdefault("source_id", None)
    event.setdefault("engine_id", event.get("engine") or "legacy")
    error = next(iter(_DETECTOR_EVENT_VALIDATOR.iter_errors(event)), None)
    if error is not None:
        path = ".".join(map(str, error.absolute_path)) or "event"
        raise ValueError(f"invalid {DETECTOR_EVENT_CONTRACT} at {path}: {error.message}")
    return event


@dataclass(frozen=True)
class AudioFrame:
    """Normalized mono PCM frame emitted by an input adapter."""

    source_id: int
    sequence: int
    timestamp: str
    sample_rate_hz: int
    samples: np.ndarray
    metadata: Mapping[str, Any]
    contract: Literal["audio-frame.v1"] = "audio-frame.v1"
    channels: int = 1
    sample_format: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        if self.contract != "audio-frame.v1":
            raise ValueError(f"unsupported audio frame contract: {self.contract!r}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.channels != 1:
            raise ValueError("audio-frame.v1 currently supports mono audio only")
        if self.sample_format != "float32":
            raise ValueError(f"unsupported audio sample format: {self.sample_format!r}")
        if (not isinstance(self.samples, np.ndarray) or self.samples.ndim != 1
                or self.samples.dtype != np.float32):
            raise ValueError("samples must be a one-dimensional float32 array")


@dataclass(frozen=True)
class InferenceResult:
    """Raw model output routed from an inference stage to event policy."""

    source_id: int
    engine_id: str
    stage_id: str
    timestamp: str
    scores: Mapping[str, float]
    metadata: Mapping[str, Any]
    contract: Literal["inference-result.v1"] = "inference-result.v1"

    def __post_init__(self) -> None:
        if self.contract != "inference-result.v1":
            raise ValueError(f"unsupported inference result contract: {self.contract!r}")
        if not self.engine_id.strip():
            raise ValueError("engine_id must not be empty")
        if not self.stage_id.strip():
            raise ValueError("stage_id must not be empty")


@dataclass(frozen=True)
class ClipSaveRequest:
    """Internal request for storing audio and metadata for a confirmed event."""

    filename: str
    sample_rate_hz: int
    samples: np.ndarray
    metadata: Mapping[str, Any]
    contract: Literal["clip-save-request.v1"] = "clip-save-request.v1"


@dataclass(frozen=True)
class RuntimeMessage:
    """A score update or versioned detector event ready for runtime routing.

    Scores retain their legacy-compatible shape. Events are normalized to
    ``detector-event.v1`` while retaining extra fields for existing clients.
    """

    kind: MessageKind
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.payload.get("type") != self.kind:
            raise ValueError("Runtime message kind must match payload['type']")

    @classmethod
    def from_legacy(cls, payload: Mapping[str, Any]) -> "RuntimeMessage":
        """Wrap a current engine message and reject unsupported message types."""
        kind = payload.get("type")
        if kind not in ("score", "event"):
            raise ValueError(f"Unsupported runtime message type: {kind!r}")
        normalized = normalize_detector_event(payload) if kind == "event" else dict(payload)
        return cls(kind=kind, payload=normalized)

    def as_legacy(self) -> dict[str, Any]:
        """Return a copy in the existing output format."""
        return dict(self.payload)
