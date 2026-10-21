#!/usr/bin/env python3
"""
Sound Detector — modular entry point.

Wires together: RuntimeHub/EventBus, Engine (YAMNet by default), Server
(FastAPI), and Integrations (MQTT).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import threading

import uvicorn

from .bus import EventBus
from .config import WEB_PORT
from .integrations.mqtt import mqtt_connect, mqtt_subscribe
from .runtime.clip_storage import WavClipStorage
from .runtime.hub import RuntimeHub
from .server.app import create_app
from .settings import load_settings
from .transport.udp import UDPSourceAdapter

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
    # Phase 1: parse --engine first to let engine add its own args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine", default="yamnet")
    pre_args, _ = pre.parse_known_args()

    engine_cls = _load_engine(pre_args.engine)

    # Phase 2: full parser with shared + engine-specific args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--source", default="siren-detector-01")
    parser.add_argument("--web-port", type=int, default=WEB_PORT)
    parser.add_argument(
        "--engine", default="yamnet",
        help="Engine name (from registry) or dotted path to Engine class, "
             "e.g. 'yamnet' or 'mypackage.engine.MyEngine'",
    )
    if hasattr(engine_cls, "add_arguments"):
        engine_cls.add_arguments(parser)
    args = parser.parse_args()

    bus = EventBus()
    runtime = RuntimeHub(bus)
    runtime.set_clip_storage(WavClipStorage())
    runtime.add_input(UDPSourceAdapter(args.bind, args.port))
    settings = load_settings()

    # MQTT (non-blocking, works without broker)
    mqtt_connect()
    mqtt_subscribe(bus)

    # Engine in background thread
    engine = engine_cls()
    threading.Thread(target=engine.start, args=(runtime, settings, args), daemon=True).start()

    # FastAPI in main thread (captures the event loop)
    app = create_app(bus)
    config = uvicorn.Config(app, host="0.0.0.0", port=args.web_port, log_level="warning")
    server = uvicorn.Server(config)

    async def serve():
        bus.loop = asyncio.get_running_loop()
        try:
            await server.serve()
        finally:
            runtime.stop_inputs()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
