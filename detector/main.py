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
import sys
import threading

import uvicorn

from .bus import EventBus
from .config import WEB_PORT
from .engine.registry import engine_registry
from .integrations.mqtt import mqtt_connect, mqtt_subscribe
from .runtime.clip_storage import WavClipStorage
from .runtime.engine_graph import validate_graph, validate_settings_schema
from .runtime.hub import RuntimeHub
from .runtime.lifecycle import has_explicit_engine_option, run_engine_component
from .server.app import create_app
from . import settings as settings_mod
from .settings import load_settings
from .transport.udp import UDPSourceAdapter


def main() -> None:
    settings = load_settings()
    stored_graph = None
    if settings.get("engine_configs"):
        try:
            stored_graph = validate_graph({
                "contract": settings.get("engine_graph_contract", "engine-graph.v1"),
                "engine_configs": settings.get("engine_configs", {}),
                "engine_routes": settings.get("engine_routes", {"default": [], "by_source": {}}),
            })
        except ValueError as exc:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.error(f"Invalid saved engine graph: {exc}")

    # Phase 1: parse --engine first to let engine add its own args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine", default="yamnet")
    pre.add_argument("--show-engine-manifest", action="store_true")
    pre.add_argument("--list-engines", action="store_true")
    pre_args, _ = pre.parse_known_args()

    if pre_args.list_engines:
        print(json.dumps(engine_registry.selectors(), indent=2))
        return

    graph_engines: list[tuple[str, type, object, dict]] = []
    graph_engine_args: dict[str, argparse.Namespace] = {}
    if stored_graph is not None:
        if has_explicit_engine_option(sys.argv):
            pre.error("A saved engine graph is active; remove --engine or clear the graph first")
        if pre_args.show_engine_manifest:
            pre.error("--show-engine-manifest requires --engine and cannot be used with a saved graph")
        routes = stored_graph["engine_routes"]
        routed_ids = list(dict.fromkeys(
            [*routes["default"], *(engine_id for route in routes["by_source"].values()
                                    for engine_id in route)]
        ))
        implementation_ids = set()
        for engine_id in routed_ids:
            config = stored_graph["engine_configs"][engine_id]
            try:
                engine_cls, manifest = engine_registry.resolve(config["selector"])
            except (AttributeError, ImportError, TypeError, ValueError) as exc:
                pre.error(f"Engine {engine_id!r}: {exc}")
            if manifest is None or manifest.engine_id != engine_id:
                pre.error(f"Engine {engine_id!r} selector must resolve to a matching manifest")
            try:
                validate_settings_schema(config["settings"], manifest.settings_schema)
            except ValueError as exc:
                pre.error(f"Engine {engine_id!r} settings: {exc}")
            implementation = (engine_cls.__module__, engine_cls.__qualname__)
            if implementation in implementation_ids:
                pre.error(f"Engine implementation {implementation!r} is assigned more than once")
            implementation_ids.add(implementation)
            graph_engines.append((engine_id, engine_cls, manifest, config))
    else:
        try:
            engine_cls, manifest = engine_registry.resolve(pre_args.engine)
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            pre.error(str(exc))
    if pre_args.show_engine_manifest and stored_graph is None:
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
    if stored_graph is None:
        parser.add_argument(
            "--engine", default="yamnet",
            help="Engine name (from registry) or dotted path to Engine class, "
                 "e.g. 'yamnet' or 'mypackage.engine.MyEngine'",
        )
        if hasattr(engine_cls, "add_arguments"):
            engine_cls.add_arguments(parser)
    args = parser.parse_args()

    if stored_graph is not None:
        for engine_id, cls, _manifest, engine_config in graph_engines:
            engine_parser = argparse.ArgumentParser(add_help=False)
            if hasattr(cls, "add_arguments"):
                cls.add_arguments(engine_parser)
            engine_parser.set_defaults(**engine_config["settings"])
            engine_args = vars(engine_parser.parse_args([]))
            engine_args.update(vars(args))
            # Shared arguments take precedence over engine parser defaults.
            graph_engine_args[engine_id] = argparse.Namespace(**engine_args)

    bus = EventBus()
    runtime = RuntimeHub(bus)
    if stored_graph is not None:
        runtime.configure_engine_graph(stored_graph)
    runtime.set_clip_storage(WavClipStorage())
    runtime.add_input(UDPSourceAdapter(args.bind, args.port))
    settings_mod.settings.clear()
    settings_mod.settings.update(settings)

    # MQTT (non-blocking, works without broker)
    mqtt_connect()
    mqtt_subscribe(bus)

    # Start the legacy CLI engine or each implementation selected by the graph.
    if stored_graph is None:
        engines_to_start = [(manifest.engine_id if manifest else pre_args.engine,
                             engine_cls, manifest, settings, args)]
    else:
        engines_to_start = []
        for engine_id, cls, selected_manifest, config in graph_engines:
            scoped_settings = {**settings, **config["settings"]}
            engines_to_start.append((engine_id, cls, selected_manifest,
                                     scoped_settings, graph_engine_args[engine_id]))

    for component_id, cls, selected_manifest, component_settings, component_args in engines_to_start:
        runtime.register_component(
            component_id, selected_manifest.name if selected_manifest else component_id,
            selected_manifest.version if selected_manifest else None,
        )

        threading.Thread(
            target=run_engine_component,
            args=(runtime, component_id, cls, component_settings, component_args),
            daemon=True,
            name=f"engine-{component_id}",
        ).start()

    # FastAPI in main thread (captures the event loop)
    app = create_app(bus, runtime)
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
