"""Engine entry point and current Runtime Hub message contract.

Engines implement ``start(runtime, settings, args)``. The current YAMNet engine
subscribes to normalized audio frames through ``RuntimeHub`` and owns buffering
and orchestration. Its inference stage returns results through the Hub to an
event policy; the policy publishes score/event output through the Hub.

During the compatibility period, an engine publishes the existing message
shape through ``RuntimeMessage.from_legacy``::

    runtime.publish(RuntimeMessage.from_legacy({
        "type": "score", "top": "Siren", "siren_score": 0.82,
    }))

Event messages use ``type="event"`` and include ``event``, ``state``,
``confidence``, ``timestamp``, ``source_id``, and ``engine``. WebSocket and
MQTT consumers still receive the unchanged legacy dictionary payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import argparse

    from detector.runtime.hub import RuntimeHub
    from detector.runtime.manifest import EngineManifest


class EngineInterface(Protocol):
    """Contract that every detection engine must satisfy."""

    def start(self, runtime: RuntimeHub, settings: dict, args: argparse.Namespace) -> None:
        """Start the engine, normally in a background thread.

        The engine loads its processing components, receives audio, and sends
        score/event messages through ``runtime.publish()``. Raw audio for the
        existing browser player can be forwarded with ``runtime.broadcast_audio``.

        During migration, ``runtime.event_bus`` remains available for legacy
        source and clip state. New engines should avoid using it directly.
        """
        ...


class ManifestedEngineInterface(EngineInterface, Protocol):
    """Engine contract for named registrations with discovery metadata."""

    ENGINE_MANIFEST: EngineManifest
