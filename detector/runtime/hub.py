"""Runtime Hub: route normalized input frames and processing results."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from ..bus import EventBus
from .contracts import AudioFrame, ClipSaveRequest, InferenceResult, RuntimeMessage
from .policies import EventPolicy
from .engine_graph import route_for_source, validate_graph

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
    source_ids: frozenset[int] | None = None
    engine_id: str | None = None
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
                "source_ids": sorted(self.source_ids) if self.source_ids is not None else None,
                "engine_id": self.engine_id,
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
        self._outputs: list[OutputAdapter] = []
        self._output_stats: list[dict] = []
        self._output_stats_lock = threading.Lock()
        self._audio_subscriptions: dict[str, AudioSubscription] = {}
        self._next_subscription_id = 1
        self._subscriptions_lock = threading.RLock()
        self._components: dict[str, dict] = {}
        self._event_policies: dict[str, EventPolicy] = {}
        self._engine_graph: dict | None = None
        self._clip_storage: ClipStorageAdapter | None = None
        self._inputs: list[AudioInputAdapter] = []
        self._started_inputs: set[int] = set()
        self.add_output(EventBusOutput(event_bus))

    def add_output(self, output: OutputAdapter) -> None:
        """Register an output adapter."""
        self._outputs.append(output)
        adapter_name = getattr(output, "name", type(output).__name__)
        self._output_stats.append({
            "adapter_id": f"{adapter_name}:{len(self._output_stats) + 1}",
            "name": adapter_name,
            "messages_delivered": 0,
            "failures": 0,
            "last_error": None,
        })

    def output_status(self) -> list[dict]:
        """Return delivery counters plus adapter-specific diagnostics."""
        with self._output_stats_lock:
            snapshots = [dict(item) for item in self._output_stats]
        for output, snapshot in zip(self._outputs, snapshots):
            get_status = getattr(output, "snapshot", None)
            if callable(get_status):
                snapshot.update(get_status())
        return snapshots

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
        source_ids: set[int] | frozenset[int] | None = None,
        engine_id: str | None = None,
    ) -> AudioSubscription:
        """Register an isolated consumer with a bounded, non-blocking queue.

        ``drop_oldest`` favors fresh audio for real-time inference. Sequence
        gaps and all queue/callback errors remain visible in ``audio_status``.
        """
        with self._subscriptions_lock:
            if subscriber_id is None:
                subscriber_id = f"audio-subscriber-{self._next_subscription_id}"
                self._next_subscription_id += 1
            if (engine_id is None and self._engine_graph is not None
                    and subscriber_id in self._engine_graph["engine_configs"]):
                engine_id = subscriber_id
            if subscriber_id in self._audio_subscriptions:
                raise ValueError(f"Audio subscriber {subscriber_id!r} already exists")
            subscription = AudioSubscription(
                subscriber_id=subscriber_id,
                consumer=consumer,
                max_queue_size=max_queue_size,
                overflow=overflow,
                source_ids=frozenset(source_ids) if source_ids is not None else None,
                engine_id=engine_id,
            )
            self._audio_subscriptions[subscriber_id] = subscription
            component = self._components.get(subscriber_id)
            if component is not None:
                component["state"] = "ready"
                component["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._start_inputs()
        return subscription

    def audio_status(self) -> list[dict]:
        """Return per-subscriber lifecycle and queue diagnostics."""
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        return [sub.snapshot() for sub in subscriptions]

    def register_component(
        self, component_id: str, name: str, version: str | None = None,
    ) -> None:
        """Register an engine lifecycle record before starting its thread."""
        if not component_id.strip() or not name.strip():
            raise ValueError("component_id and name must not be empty")
        with self._subscriptions_lock:
            if component_id in self._components:
                raise ValueError(f"Component {component_id!r} already exists")
            now = datetime.now(timezone.utc).isoformat()
            self._components[component_id] = {
                "component_id": component_id,
                "name": name,
                "version": version,
                "state": "starting",
                "started_at": now,
                "updated_at": now,
                "error_count": 0,
                "last_error": None,
            }

    def set_component_state(
        self, component_id: str, state: str, error: BaseException | str | None = None,
    ) -> None:
        """Update lifecycle state and retain the latest diagnostic."""
        allowed = {"starting", "ready", "degraded", "error", "stopping", "stopped"}
        if state not in allowed:
            raise ValueError(f"Unsupported component state {state!r}")
        with self._subscriptions_lock:
            component = self._components.get(component_id)
            if component is None:
                return
            component["state"] = state
            component["updated_at"] = datetime.now(timezone.utc).isoformat()
            if error is not None:
                component["error_count"] += 1
                component["last_error"] = (
                    f"{type(error).__name__}: {error}"
                    if isinstance(error, BaseException) else str(error)
                )

    def component_status(self) -> list[dict]:
        """Return snapshots of registered engine lifecycle records."""
        with self._subscriptions_lock:
            return [dict(component) for component in self._components.values()]

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
        with self._subscriptions_lock:
            for component in self._components.values():
                if component["state"] not in ("error", "stopped"):
                    component["state"] = "stopping"
                    component["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.stop_inputs()
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        for subscription in subscriptions:
            subscription.stop()
        for output in self._outputs:
            stop = getattr(output, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    logger.exception("Output adapter %s failed to stop", type(output).__name__)
        with self._subscriptions_lock:
            for component in self._components.values():
                if component["state"] != "error":
                    component["state"] = "stopped"
                    component["updated_at"] = datetime.now(timezone.utc).isoformat()

    def publish_audio_frame(self, frame: AudioFrame) -> None:
        """Route an input frame to processors and the live-audio output."""
        with self._subscriptions_lock:
            subscriptions = tuple(self._audio_subscriptions.values())
        for subscription in subscriptions:
            source_allowed = (
                subscription.source_ids is None or frame.source_id in subscription.source_ids
            )
            route = (
                self._engine_graph is None or subscription.engine_id is None
                or subscription.engine_id in route_for_source(self._engine_graph, frame.source_id)
            )
            if source_allowed and route:
                subscription.submit(frame)
        self.broadcast_audio(frame.samples, frame.source_id)

    def configure_engine_graph(self, graph: dict) -> None:
        """Apply source-to-engine routing before any engine subscribes."""
        self._engine_graph = validate_graph(graph)

    @property
    def engine_graph_active(self) -> bool:
        """Whether startup is using a saved engine graph instead of legacy CLI."""
        return self._engine_graph is not None

    def set_event_policy(self, policy: EventPolicy, engine_id: str = "default") -> None:
        """Select the policy for one engine's inference results."""
        if not engine_id.strip():
            raise ValueError("engine_id must not be empty")
        self._event_policies[engine_id] = policy

    def publish_inference(self, result: InferenceResult) -> None:
        """Route model output to the selected event policy."""
        policy = self._event_policies.get(result.engine_id) or self._event_policies.get("default")
        if policy is None:
            raise RuntimeError("No event policy is configured")
        policy.process(result)

    def publish(self, message: RuntimeMessage) -> None:
        """Route a message, isolating failures and tracking every output."""
        for index, (output, status) in enumerate(zip(self._outputs, self._output_stats)):
            try:
                output.publish(message)
            except Exception as exc:
                logger.exception("Output adapter %s failed", status["adapter_id"])
                with self._output_stats_lock:
                    current = self._output_stats[index]
                    current["failures"] += 1
                    current["last_error"] = f"{type(exc).__name__}: {exc}"
            else:
                with self._output_stats_lock:
                    self._output_stats[index]["messages_delivered"] += 1
