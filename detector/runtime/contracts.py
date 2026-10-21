"""Versioned messages exchanged between processing and runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


MessageKind = Literal["score", "event"]


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
