"""Prometheus text exposition for Runtime Hub status snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _sample(name: str, labels: Mapping[str, Any], value: int | float) -> str:
    rendered_labels = ",".join(
        f'{key}="{_escape_label(val)}"' for key, val in sorted(labels.items())
    )
    suffix = f"{{{rendered_labels}}}" if rendered_labels else ""
    return f"{name}{suffix} {value}"


def render_runtime_metrics(status: Mapping[str, Any]) -> str:
    """Format runtime diagnostics as Prometheus 0.0.4 text exposition."""
    lines: list[str] = []

    def metric(name: str, help_text: str, kind: str, samples: list[str]) -> None:
        lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *samples))

    components = status.get("components", [])
    metric("detector_engine_state", "Engine lifecycle state; one state label is 1.", "gauge", [
        _sample("detector_engine_state", {
            "engine_id": component.get("component_id", "unknown"),
            "state": component.get("state", "unknown"),
        }, 1)
        for component in components
    ])
    metric("detector_engine_errors_total", "Engine startup and runtime error count.", "counter", [
        _sample("detector_engine_errors_total", {
            "engine_id": component.get("component_id", "unknown"),
        }, component.get("error_count", 0))
        for component in components
    ])

    subscribers = status.get("subscribers", [])
    for suffix, field, help_text in (
        ("frames_received_total", "frames_received", "Audio frames accepted by a subscriber."),
        ("frames_processed_total", "frames_processed", "Audio frames processed by a subscriber."),
        ("frames_dropped_total", "frames_dropped", "Audio frames dropped by a subscriber."),
        ("sequence_gaps_total", "sequence_gaps", "Missing audio sequence numbers observed."),
        ("callback_errors_total", "callback_errors", "Audio callback failures."),
    ):
        name = f"detector_audio_{suffix}"
        metric(name, help_text, "counter", [
            _sample(name, {
                "subscriber_id": sub.get("subscriber_id", "unknown"),
                "engine_id": sub.get("engine_id") or "unknown",
            }, sub.get(field, 0))
            for sub in subscribers
        ])
    metric("detector_audio_queue_depth", "Current queued audio frames.", "gauge", [
        _sample("detector_audio_queue_depth", {
            "subscriber_id": sub.get("subscriber_id", "unknown"),
        }, sub.get("queue_depth", 0))
        for sub in subscribers
    ])

    outputs = status.get("outputs", [])
    for suffix, field, help_text, kind in (
        ("messages_delivered_total", "messages_delivered", "Messages accepted by an output adapter.", "counter"),
        ("failures_total", "failures", "Output adapter delivery failures.", "counter"),
        ("webhook_deliveries_total", "sent", "Webhook requests delivered successfully.", "counter"),
        ("webhook_dropped_total", "dropped", "Webhook events dropped from the queue.", "counter"),
        ("webhook_queue_depth", "queue_depth", "Events waiting in the webhook queue.", "gauge"),
    ):
        name = f"detector_{suffix}"
        metric(name, help_text, kind, [
            _sample(name, {"adapter_id": output.get("adapter_id", "unknown")}, output[field])
            for output in outputs if field in output
        ])

    return "\n".join(lines) + "\n"
