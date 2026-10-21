"""Runtime Hub: route processing results to the existing output bus."""

from __future__ import annotations

from typing import Protocol

from ..bus import EventBus
from .contracts import RuntimeMessage


class OutputAdapter(Protocol):
    """Receives normalized runtime messages for a concrete output layer."""

    def publish(self, message: RuntimeMessage) -> None:
        ...


class EventBusOutput:
    """Compatibility adapter for current WebSocket and integration outputs."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    def publish(self, message: RuntimeMessage) -> None:
        self._event_bus.broadcast(message.as_legacy())


class RuntimeHub:
    """Receive component outputs and route them to configured output adapters.

    The first migration step registers the existing EventBus as the output
    adapter. Audio ingress and pipeline routing remain in the legacy engine
    until those boundaries are extracted in a later step.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._outputs: list[OutputAdapter] = [EventBusOutput(event_bus)]

    def add_output(self, output: OutputAdapter) -> None:
        """Register an output adapter."""
        self._outputs.append(output)

    def broadcast_audio(self, samples, source_id: int = 0) -> None:
        """Forward raw audio to current live-listening clients."""
        self.event_bus.broadcast_audio(samples, source_id)

    def publish(self, message: RuntimeMessage) -> None:
        """Route a score or event from a processing component to outputs."""
        for output in self._outputs:
            output.publish(message)
