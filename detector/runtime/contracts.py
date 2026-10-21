"""Versioned messages exchanged between processing and runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np


MessageKind = Literal["score", "event"]


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
    """A score update or detector event ready for runtime routing.

    ``payload`` temporarily retains the existing WebSocket/MQTT message shape.
    This lets the runtime boundary be introduced without breaking current
    consumers while the public component schemas are being finalized.
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
        return cls(kind=kind, payload=dict(payload))

    def as_legacy(self) -> dict[str, Any]:
        """Return a copy in the existing output format."""
        return dict(self.payload)
