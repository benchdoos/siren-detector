"""Runtime Hub: route normalized input frames and processing results."""

from __future__ import annotations

import logging
from typing import Protocol

from ..bus import EventBus
from .contracts import AudioFrame, RuntimeMessage

logger = logging.getLogger(__name__)


class AudioConsumer(Protocol):
    """Receives normalized audio frames from the runtime router."""

    def __call__(self, frame: AudioFrame) -> None:
        ...


class AudioInputAdapter(Protocol):
    """Produces normalized frames and can be started/stopped by the runtime."""

    def start(self, publish_frame: AudioConsumer) -> None:
        ...

    def stop(self) -> None:
        ...


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
        self._audio_consumers: list[AudioConsumer] = []
        self._inputs: list[AudioInputAdapter] = []
        self._started_inputs: set[int] = set()

    def add_output(self, output: OutputAdapter) -> None:
        """Register an output adapter."""
        self._outputs.append(output)

    def add_input(self, input_adapter: AudioInputAdapter) -> None:
        """Register an input adapter; it starts when a processor subscribes."""
        self._inputs.append(input_adapter)
        if self._audio_consumers:
            self._start_inputs()

    def broadcast_audio(self, samples, source_id: int = 0) -> None:
        """Forward raw audio to current live-listening clients."""
        self.event_bus.broadcast_audio(samples, source_id)

    def subscribe_audio(self, consumer: AudioConsumer) -> None:
        """Register a processing component for normalized input frames."""
        self._audio_consumers.append(consumer)
        self._start_inputs()

    def _start_inputs(self) -> None:
        for input_adapter in self._inputs:
            adapter_id = id(input_adapter)
            if adapter_id in self._started_inputs:
                continue
            input_adapter.start(self.publish_audio_frame)
            self._started_inputs.add(adapter_id)

    def stop_inputs(self) -> None:
        """Stop all registered input adapters."""
        for input_adapter in self._inputs:
            adapter_id = id(input_adapter)
            if adapter_id in self._started_inputs:
                input_adapter.stop()
                self._started_inputs.remove(adapter_id)

    def publish_audio_frame(self, frame: AudioFrame) -> None:
        """Route an input frame to processors and the live-audio output."""
        for consumer in tuple(self._audio_consumers):
            try:
                consumer(frame)
            except Exception:
                logger.exception(
                    "Audio consumer failed for source %s sequence %s",
                    frame.source_id, frame.sequence,
                )
        self.broadcast_audio(frame.samples, frame.source_id)

    def publish(self, message: RuntimeMessage) -> None:
        """Route a score or event from a processing component to outputs."""
        for output in self._outputs:
            output.publish(message)
