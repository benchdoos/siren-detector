"""Entry-point startup checks with model and network adapters isolated."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types
import unittest
from argparse import ArgumentParser
from unittest.mock import patch

from detector.runtime.manifest import EngineManifest


ENGINE_ID = "startup-probe"
ENGINE_SETTINGS_SCHEMA = {
    "type": "object",
    "properties": {"probe_value": {"type": "number", "default": 0.1}},
    "additionalProperties": False,
}
MANIFEST = EngineManifest(
    engine_id=ENGINE_ID,
    name="Startup probe",
    version="1.0.0",
    description="A lightweight engine used by the startup test.",
    input_contracts=("audio-frame.v1",),
    output_contracts=("inference-result.v1",),
    classes=("siren",),
    settings_schema=ENGINE_SETTINGS_SCHEMA,
    resources={},
    launch={},
)


class _FakeEngine:
    started = threading.Event()
    settings = None
    args = None

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument("--probe-value", type=float, default=0.1)

    def start(self, runtime, settings, args) -> None:
        type(self).settings = settings
        type(self).args = args
        type(self).started.set()


class _FakeEventBus:
    def __init__(self):
        self.loop = None


class _FakeRuntimeHub:
    last_instance = None

    def __init__(self, bus):
        self.bus = bus
        self.graph = None
        self.components = {}
        self.input = None
        type(self).last_instance = self

    def configure_engine_graph(self, graph):
        self.graph = graph

    def set_clip_storage(self, storage):
        self.storage = storage

    def add_input(self, adapter):
        self.input = adapter

    def add_output(self, adapter):
        self.output = adapter

    def register_component(self, component_id, name, version=None):
        self.components[component_id] = {"name": name, "version": version, "state": "starting"}

    def set_component_state(self, component_id, state, error=None):
        self.components[component_id].update({"state": state, "error": error})

    def stop(self):
        pass


class _FakeUDPSourceAdapter:
    def __init__(self, bind, port):
        self.bind = bind
        self.port = port


class _FakeWavClipStorage:
    pass


class _FakeUvicornConfig:
    def __init__(self, app, **kwargs):
        self.app = app
        self.kwargs = kwargs


class _FakeUvicornServer:
    def __init__(self, config):
        self.config = config

    async def serve(self):
        return None


def _load_main_with_lightweight_adapters():
    bus_module = types.ModuleType("detector.bus")
    bus_module.EventBus = _FakeEventBus

    mqtt_module = types.ModuleType("detector.integrations.mqtt")
    mqtt_module.mqtt_connect = lambda: None
    mqtt_module.mqtt_subscribe = lambda bus: None

    clip_storage_module = types.ModuleType("detector.runtime.clip_storage")
    clip_storage_module.WavClipStorage = _FakeWavClipStorage

    hub_module = types.ModuleType("detector.runtime.hub")
    hub_module.RuntimeHub = _FakeRuntimeHub

    app_module = types.ModuleType("detector.server.app")
    app_module.create_app = lambda bus, runtime: {"bus": bus, "runtime": runtime}

    udp_module = types.ModuleType("detector.transport.udp")
    udp_module.UDPSourceAdapter = _FakeUDPSourceAdapter

    uvicorn_module = types.ModuleType("uvicorn")
    uvicorn_module.Config = _FakeUvicornConfig
    uvicorn_module.Server = _FakeUvicornServer

    soundfile_module = types.ModuleType("soundfile")
    adapters = {
        "detector.bus": bus_module,
        "detector.integrations.mqtt": mqtt_module,
        "detector.runtime.clip_storage": clip_storage_module,
        "detector.runtime.hub": hub_module,
        "detector.server.app": app_module,
        "detector.transport.udp": udp_module,
        "soundfile": soundfile_module,
        "uvicorn": uvicorn_module,
    }
    with patch.dict(sys.modules, adapters):
        sys.modules.pop("detector.main", None)
        module = importlib.import_module("detector.main")
    return module


main_module = _load_main_with_lightweight_adapters()


class MainStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeEngine.started.clear()
        _FakeEngine.settings = None
        _FakeEngine.args = None
        _FakeRuntimeHub.last_instance = None
        self.graph_settings = {
            "engine_graph_contract": "engine-graph.v1",
            "engine_configs": {
                ENGINE_ID: {
                    "selector": "startup-probe",
                    "enabled": True,
                    "settings": {"probe_value": 0.73},
                },
            },
            "engine_routes": {"default": [ENGINE_ID], "by_source": {"12": []}},
        }
        self.patches = [
            patch.object(main_module, "load_settings", return_value=self.graph_settings),
            patch.object(
                main_module.engine_registry,
                "resolve",
                return_value=(_FakeEngine, MANIFEST),
            ),
            patch.object(sys, "argv", ["detector", "--port", "9012"]),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_graph_startup_builds_runtime_and_starts_manifest_engine(self) -> None:
        main_module.main()

        self.assertTrue(_FakeEngine.started.wait(1))
        runtime = _FakeRuntimeHub.last_instance
        deadline = time.monotonic() + 1
        while runtime.components[ENGINE_ID]["state"] == "starting" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(runtime.graph["engine_routes"]["by_source"], {"12": []})
        self.assertEqual(runtime.input.bind, "0.0.0.0")
        self.assertEqual(runtime.input.port, 9012)
        self.assertEqual(runtime.components[ENGINE_ID]["name"], MANIFEST.name)
        self.assertEqual(runtime.components[ENGINE_ID]["state"], "ready")
        self.assertEqual(_FakeEngine.settings["probe_value"], 0.73)
        self.assertEqual(_FakeEngine.args.probe_value, 0.73)
        self.assertEqual(_FakeEngine.args.port, 9012)

    def test_graph_startup_rejects_explicit_legacy_engine_flag(self) -> None:
        with patch.object(sys, "argv", ["detector", "--engine", "yamnet"]):
            with self.assertRaises(SystemExit) as raised:
                main_module.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
