"""Engine interface protocol.

To create a custom detection engine:

1. Implement a class with a ``start(bus, settings, args)`` method.
2. Pass its dotted path via ``--engine``, e.g.::

       python -m detector.main --engine mypackage.my_engine.MyEngine

   Or register it in ``main.py``'s ``ENGINE_REGISTRY``.

Message formats
---------------

Your engine must push messages via ``bus.broadcast(msg)`` in these formats.

**Score message** (sent every inference hop, ~0.5s):

.. code-block:: python

    {
        "type": "score",
        "top": "Speech",                  # YAMNet top-1 class name (or "")
        "<name>_score": 0.42,             # effective score per detector (float 0..1)
        "<name>_state": "possible",       # "idle" | "possible" | "confirmed"
        "<name>_triggers": 5,             # trigger count in current window
        "<name>_possible_needed": 4,      # triggers needed for "possible"
        "<name>_confirmed_needed": 8,     # triggers needed for "confirmed"
        "<name>_cooldown": 0.0,           # cooldown remaining (seconds)
        "<name>_refractory": 0.0,         # refractory remaining (seconds)
        "audio_alive": True,              # UDP transport alive?
        "audio_ago": 0.1,                 # seconds since last packet
        "audio_pps": 31.2,               # packets per second
        # optional:
        "yamnet": 0.05,                   # raw YAMNet siren score
        "mlp": 0.42,                      # MLP siren score (or None)
        "yamnet_top3": [{"name": "Speech", "score": 0.8}, ...],
        "mlp_top3": [{"name": "siren", "score": 0.42}, ...],
    }

Where ``<name>`` is each key in ``config.DETECTORS`` (siren, explosion, motorcycle).

**Event message** (sent on state transitions to possible/confirmed):

.. code-block:: python

    {
        "type": "event",
        "event": "siren",                 # detector name
        "state": "confirmed",             # "possible" | "confirmed"
        "confidence": 0.87,               # peak effective score
        "timestamp": "2026-10-16T12:00:00+00:00",
        "source": "siren-detector-01",    # --source arg
        "engine": "yamnet+mlp",           # engine identifier string
        "clip_url": "/clips/20261016-120000.wav",  # or None if not confirmed
        "yamnet": 0.15,                   # peak raw YAMNet score
        "mlp": 0.87,                      # peak MLP score (or None)
        "yamnet_top": "Siren",            # YAMNet top-1 at peak
        "yamnet_top3": [...],
        "mlp_top3": [...],
        "threshold": 0.10,
        "triggers": 8,
        "triggers_needed": 8,
    }

Audio streaming
---------------

Call ``bus.broadcast_audio(samples)`` with float32 numpy arrays to enable
live listening in the Web UI via ``/ws/audio``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import argparse

    from detector.bus import EventBus


class EngineInterface(Protocol):
    """Contract that every detection engine must satisfy."""

    def start(self, bus: EventBus, settings: dict, args: argparse.Namespace) -> None:
        """Start the engine (called from a background thread).

        The engine should:
        1. Set up transport (e.g. UDP receiver) to receive audio.
        2. Run inference on incoming audio.
        3. Push score and event messages via ``bus.broadcast()``.
        4. Push raw audio via ``bus.broadcast_audio()`` for live listening.
        """
        ...
