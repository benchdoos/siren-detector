#!/usr/bin/env python3
"""
Sound Detector — modular entry point.

Wires together: RuntimeHub/EventBus, Engine (YAMNet by default), Server
(FastAPI), and Integrations (MQTT).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading

import uvicorn

from .bus import EventBus
from .config import WEB_PORT
from .engine.registry import engine_registry
from .integrations.mqtt import mqtt_connect, mqtt_subscribe
from .runtime.clip_storage import WavClipStorage
from .runtime.hub import RuntimeHub
from .server.app import create_app
from .settings import load_settings
from .transport.udp import UDPSourceAdapter


def main() -> None:
    # Phase 1: parse --engine first to let engine add its own args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine", default="yamnet")
    pre.add_argument("--show-engine-manifest", action="store_true")
    pre.add_argument("--list-engines", action="store_true")
    pre_args, _ = pre.parse_known_args()

    if pre_args.list_engines:
        print(json.dumps(engine_registry.selectors(), indent=2))
        return

    try:
        engine_cls, manifest = engine_registry.resolve(pre_args.engine)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        pre.error(str(exc))
    if pre_args.show_engine_manifest:
        if manifest is None:
            pre.error(f"Engine {pre_args.engine!r} does not declare ENGINE_MANIFEST")
        print(json.dumps(manifest.to_dict(), indent=2))
        return

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
            runtime.stop()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
