"""Runtime Hub: route normalized input frames and processing results."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Protocol

from ..bus import EventBus
from .contracts import AudioFrame, ClipSaveRequest, InferenceResult, RuntimeMessage
from .policies import EventPolicy

logger = logging.getLogger(__name__)


class AudioConsumer(Protocol):
    """Receives normalized audio frames from the runtime router."""

    def __call__(self, frame: AudioFrame) -> None:
        ...


_STOP = object()


@dataclass
class AudioSubscription:
    """Isolated bounded audio queue and worker for one processing component."""

    contract: str = field(default="audio-subscription.v1", init=False)
    subscriber_id: str
    consumer: AudioConsumer
    max_queue_size: int = 8
    overflow: str = "drop_oldest"
    _queue: queue.Queue = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _thread: threading.Thread = field(init=False, repr=False)
    state: str = field(default="starting", init=False)
    frames_received: int = field(default=0, init=False)
    frames_processed: int = field(default=0, init=False)
    frames_dropped: int = field(default=0, init=False)
    sequence_gaps: int = field(default=0, init=False)
    callback_errors: int = field(default=0, init=False)
    last_error: str | None = field(default=None, init=False)
    _last_sequence: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.subscriber_id.strip():
            raise ValueError("subscriber_id must not be empty")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")
        if self.overflow not in ("drop_oldest", "drop_newest"):
            raise ValueError("overflow must be 'drop_oldest' or 'drop_newest'")
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._thread = threading.Thread(
            target=self._run, name=f"audio-{self.subscriber_id}", daemon=True,
        )
        self._thread.start()

    def submit(self, frame: AudioFrame) -> None:
        """Enqueue without blocking the input adapter."""
        with self._lock:
            if self.state in ("stopping", "stopped"):
                return
            self.frames_received += 1
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                if self.overflow == "drop_newest":
                    self.frames_dropped += 1
                    return
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self.frames_dropped += 1
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    self.frames_dropped += 1

    def snapshot(self) -> dict:
        """Return a thread-safe diagnostic snapshot for this subscription."""
        with self._lock:
            return {
                "contract": self.contract,
                "subscriber_id": self.subscriber_id,
                "state": self.state,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self.max_queue_size,
                "overflow": self.overflow,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "sequence_gaps": self.sequence_gaps,
                "callback_errors": self.callback_errors,
                "last_error": self.last_error,
            }

    def stop(self, join_timeout: float = 2.0) -> None:
        """Discard pending frames and request worker shutdown."""
        with self._lock:
            if self.state in ("stopping", "stopped"):
                return
            self.state = "stopping"
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.frames_dropped += 1
            self._queue.put_nowait(_STOP)
        self._thread.join(timeout=join_timeout)
        with self._lock:
            self.state = "stopped" if not self._thread.is_alive() else "stopping"

    def _run(self) -> None:
        with self._lock:
            self.state = "ready"
        while True:
            frame = self._queue.get()
            if frame is _STOP:
                break
            with self._lock:
                previous = self._last_sequence.get(frame.source_id)
                if previous is not None and frame.sequence > previous + 1:
                    self.sequence_gaps += frame.sequence - previous - 1
                self._last_sequence[frame.source_id] = frame.sequence
            try:
                self.consumer(frame)
            except Exception as exc:
                logger.exception(
                    "Audio subscriber %s failed for source %s sequence %s",
                    self.subscriber_id, frame.source_id, frame.sequence,
                )
                with self._lock:
                    self.callback_errors += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self.state = "degraded"
            else:
                with self._lock:
                    self.frames_processed += 1
                    if self.state == "degraded":
                        self.state = "ready"
        with self._lock:
            self.state = "stopped"


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


class ClipStorageAdapter(Protocol):
    """Stores an audio clip and its event metadata, returning its public URL."""

    def save_clip(self, request: ClipSaveRequest) -> str:
        ...


class EventBusOutput:
    """Compatibility adapter for current WebSocket and integration outputs."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    def publish(self, message: RuntimeMessage) -> None:
        self._event_bus.broadcast(message.as_legacy())


class RuntimeHub:
    """Route normalized audio, inference results, and component outputs.

    The current migration registers the existing EventBus as an output adapter
    and routes inference results to the current event policy.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._outputs: list[OutputAdapter] = [EventBusOutput(event_bus)]
        self._audio_subscriptions: dict[str, AudioSubscription] = {}
        self._next_subscription_id = 1
        self._subscriptions_lock = threading.RLock()
        self._event_policy: EventPolicy | None = None
        self._clip_storage: ClipStorageAdapter | None = None
        self._inputs: list[AudioInputAdapter] = []
        self._started_inputs: set[int] = set()

    def add_output(self, output: OutputAdapter) -> None:
        """Register an output adapter."""
        self._outputs.append(output)

    def set_clip_storage(self, storage: ClipStorageAdapter) -> None:
        """Register the adapter used to persist event clips."""
        self._clip_storage = storage

    def save_clip(self, request: ClipSaveRequest) -> str:
        """Route a clip artifact to the configured storage adapter."""
        if self._clip_storage is None:
            raise RuntimeError("No clip storage adapter is configured")
        return self._clip_storage.save_clip(request)

    def add_input(self, input_adapter: AudioInputAdapter) -> None:
        """Register an input adapter; it starts when a processor subscribes."""
        self._inputs.append(input_adapter)
        with self._subscriptions_lock:
            has_subscribers = bool(self._audio_subscriptions)
        if has_subscribers:
            self._start_inputs()

    def broadcast_audio(self, samples, source_id: int = 0) -> None:
        """Forward raw audio to current live-listening clients."""
        self.event_bus.broadcast_audio(samples, source_id)

    def subscribe_audio(
        self,
        consumer: AudioConsumer,
        subscriber_id: str | None = None,
        *,
        max_queue_size: int = 8,
        overflow: str = "drop_oldest",
    ) -> AudioSubscription:
        """Register an isolated consumer with a bounded, non-blocking queue.

        ``drop_oldest`` favors fresh audio for real-time inference. Sequence
        gaps and all queue/callback errors remain visible in ``audio_status``.
        """
        with self._subscriptions_lock:
            if subscriber_id is None:
                subscriber_id = f"audio-subscriber-{self._next_subscription_id}"
                self._next_subscription_id += 1
            if subscriber_id in self._audio_subscriptions:
                raise ValueError(f"Audio subscriber {subscriber_id!r} already exists")
            subscription = AudioSubscription(
                subscriber_id=subscriber_id,
                consumer=consumer,
                max_queue_size=max_queue_size,
                overflow=overflow,
            )
            self._audio_subscriptions[subscriber_id] = subscription
        self._start_inputs()
        return subscription

    def audio_status(self) -> list[dict]:
        """Return per-subscriber lifecycle and queue diagnostics."""
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        return [sub.snapshot() for sub in subscriptions]

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

    def stop(self) -> None:
        """Stop audio producers first, then all subscriber workers."""
        self.stop_inputs()
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        for subscription in subscriptions:
            subscription.stop()

    def publish_audio_frame(self, frame: AudioFrame) -> None:
        """Route an input frame to processors and the live-audio output."""
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        for subscription in subscriptions:
            subscription.submit(frame)
        self.broadcast_audio(frame.samples, frame.source_id)

    def set_event_policy(self, policy: EventPolicy) -> None:
        """Select the event policy that handles inference results."""
        self._event_policy = policy

    def publish_inference(self, result: InferenceResult) -> None:
        """Route model output to the selected event policy."""
        if self._event_policy is None:
            raise RuntimeError("No event policy is configured")
        self._event_policy.process(result)

    def publish(self, message: RuntimeMessage) -> None:
        """Route a score or event from a processing component to outputs."""
        for output in self._outputs:
            output.publish(message)
