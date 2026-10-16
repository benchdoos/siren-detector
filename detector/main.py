#!/usr/bin/env python3
"""
Sound Detector — modular entry point.

Wires together: EventBus, Engine (YAMNet by default), Server (FastAPI),
and Integrations (MQTT).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import threading

import uvicorn

from .bus import EventBus
from .config import WEB_PORT
from .integrations.mqtt import mqtt_connect
from .server.app import create_app
from .settings import load_settings

# Built-in engines: name → module.ClassName
ENGINE_REGISTRY = {
    "yamnet": "detector.engine.yamnet_engine.YAMNetEngine",
}


def _load_engine(name: str):
    """Load engine class by registry name or dotted path."""
    dotted = ENGINE_REGISTRY.get(name, name)
    module_path, _, class_name = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--source", default="siren-detector-01")
    parser.add_argument("--classifier-weights", default="/models/classifier_weights.npz")
    parser.add_argument("--classifier-labels", default="/models/classifier_labels.json")
    parser.add_argument("--web-port", type=int, default=WEB_PORT)
    parser.add_argument(
        "--engine", default="yamnet",
        help="Engine name (from registry) or dotted path to Engine class, "
             "e.g. 'yamnet' or 'mypackage.engine.MyEngine'",
    )
    args = parser.parse_args()

    bus = EventBus()
    settings = load_settings()

    # MQTT (non-blocking, works without broker)
    mqtt_connect()

    # Engine in background thread
    engine_cls = _load_engine(args.engine)
    engine = engine_cls()
    threading.Thread(target=engine.start, args=(bus, settings, args), daemon=True).start()

    # FastAPI in main thread (captures the event loop)
    app = create_app(bus)
    config = uvicorn.Config(app, host="0.0.0.0", port=args.web_port, log_level="warning")
    server = uvicorn.Server(config)

    async def serve():
        bus.loop = asyncio.get_running_loop()
        await server.serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
