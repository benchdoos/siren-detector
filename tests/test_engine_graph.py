"""Regression coverage for the engine graph migration boundary."""

from __future__ import annotations

import json
import importlib
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from detector.engine.registry import engine_registry
from detector.runtime.manifest import EngineManifest
from detector.runtime.engine_graph import route_for_source, validate_graph, validate_settings_schema
from detector.runtime.contracts import AudioFrame
from detector.runtime.engine_settings import settings_for_engine
from detector.runtime.lifecycle import has_explicit_engine_option, run_engine_component
from detector.settings import load_settings
from detector import settings as settings_module


class _FakeEventBus:
    def __init__(self):
        self.audio = []

    def broadcast_audio(self, samples, source_id=0):
        self.audio.append(source_id)

    def broadcast(self, message):
        pass


_bus_stub = types.ModuleType("detector.bus")
_bus_stub.EventBus = _FakeEventBus
with patch.dict(sys.modules, {"detector.bus": _bus_stub}):
    RuntimeHub = importlib.import_module("detector.runtime.hub").RuntimeHub
sys.modules.pop("detector.runtime.hub", None)


ENERGY_SCHEMA = {
    "type": "object",
    "properties": {"energy_event_threshold": {"type": "number", "minimum": 0, "maximum": 1}},
    "additionalProperties": False,
}
YAMNET_SCHEMA = {
    "type": "object",
    "properties": {"siren_threshold": {"type": "number", "minimum": 0, "maximum": 1}},
    "additionalProperties": False,
}


class EngineGraphTests(unittest.TestCase):
    def test_legacy_settings_migrate_without_losing_graph_extensions(self) -> None:
        saved = {
            "siren_threshold": 0.27,
            "engine_graph_contract": "engine-graph.v1",
            "engine_configs": {
                "energy-threshold-example": {
                    "selector": "energy-threshold-example",
                    "enabled": True,
                    "settings": {"energy_event_threshold": 0.12},
                    "vendor_extension": {"kept": True},
                },
            },
            "engine_routes": {"default": ["energy-threshold-example"], "by_source": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            with patch("detector.settings.SETTINGS_PATH", path):
                loaded = load_settings()

        self.assertEqual(loaded["siren_threshold"], 0.27)
        self.assertEqual(loaded["engine_configs"]["energy-threshold-example"]["settings"],
                         {"energy_event_threshold": 0.12})
        self.assertTrue(loaded["engine_configs"]["energy-threshold-example"]["vendor_extension"]["kept"])

    def test_legacy_dotted_engine_selector_still_resolves(self) -> None:
        engine_class, manifest = engine_registry.resolve(
            "detector.examples.energy_threshold_engine.EnergyThresholdEngine"
        )
        self.assertEqual(engine_class.__name__, "EnergyThresholdEngine")
        self.assertEqual(manifest.engine_id, "energy-threshold-example")

    def test_saved_graph_rejects_explicit_legacy_engine_option(self) -> None:
        self.assertFalse(has_explicit_engine_option(["detector", "--port", "9000"]))
        self.assertTrue(has_explicit_engine_option(["detector", "--engine", "yamnet"]))
        self.assertTrue(has_explicit_engine_option(["detector", "--engine=yamnet"]))

    def test_default_source_override_and_unknown_source_routes(self) -> None:
        graph = validate_graph({
            "contract": "engine-graph.v1",
            "engine_configs": {
                "yamnet": {"selector": "yamnet", "enabled": True, "settings": {}},
                "energy-threshold-example": {
                    "selector": "energy-threshold-example", "enabled": True, "settings": {},
                },
            },
            "engine_routes": {
                "default": ["yamnet"],
                "by_source": {"7": ["energy-threshold-example"]},
            },
        })
        self.assertEqual(route_for_source(graph, 2), ("yamnet",))
        self.assertEqual(route_for_source(graph, 7), ("energy-threshold-example",))
        self.assertEqual(route_for_source(graph, 999), ("yamnet",))

    def test_runtime_hub_delivers_only_to_routed_engine(self) -> None:
        graph = {
            "engine_configs": {
                "yamnet": {"selector": "yamnet", "enabled": True, "settings": {}},
                "energy-threshold-example": {
                    "selector": "energy-threshold-example", "enabled": True, "settings": {},
                },
            },
            "engine_routes": {
                "default": ["yamnet"],
                "by_source": {"7": ["energy-threshold-example"]},
            },
        }
        runtime = RuntimeHub(_FakeEventBus())
        runtime.configure_engine_graph(graph)
        received = {"yamnet": [], "energy-threshold-example": []}
        events = {engine_id: threading.Event() for engine_id in received}
        for engine_id in received:
            def consume(frame, key=engine_id):
                received[key].append(frame.source_id)
                events[key].set()

            runtime.subscribe_audio(consume, subscriber_id=engine_id)
        try:
            for source_id in (2, 7, 999):
                runtime.publish_audio_frame(AudioFrame(
                    source_id=source_id,
                    sequence=source_id,
                    timestamp="2026-10-21T00:00:00Z",
                    sample_rate_hz=16_000,
                    samples=np.zeros(4, dtype=np.float32),
                    metadata={},
                ))
            self.assertTrue(events["yamnet"].wait(1))
            self.assertTrue(events["energy-threshold-example"].wait(1))
            deadline = time.monotonic() + 1
            while len(received["yamnet"]) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(received["yamnet"], [2, 999])
            self.assertEqual(received["energy-threshold-example"], [7])
        finally:
            runtime.stop()

    def test_one_engine_failure_does_not_change_peer_state(self) -> None:
        class BrokenEngine:
            def start(self, runtime, settings, args):
                raise RuntimeError("engine startup failed")

        class HealthyEngine:
            def start(self, runtime, settings, args):
                return None

        runtime = RuntimeHub(_FakeEventBus())
        runtime.register_component("broken", "Broken")
        runtime.register_component("healthy", "Healthy")
        run_engine_component(runtime, "broken", BrokenEngine, {}, Namespace())
        run_engine_component(runtime, "healthy", HealthyEngine, {}, Namespace())
        states = {item["component_id"]: item for item in runtime.component_status()}
        self.assertEqual(states["broken"]["state"], "error")
        self.assertEqual(states["broken"]["error_count"], 1)
        self.assertIn("engine startup failed", states["broken"]["last_error"])
        self.assertEqual(states["healthy"]["state"], "ready")

    def test_disabled_engine_cannot_be_routed(self) -> None:
        with self.assertRaisesRegex(ValueError, "route references disabled engine"):
            validate_graph({
                "engine_configs": {
                    "yamnet": {"selector": "yamnet", "enabled": False, "settings": {}},
                },
                "engine_routes": {"default": ["yamnet"], "by_source": {}},
            })

    def test_distinct_engines_accept_independent_manifest_settings(self) -> None:
        energy_manifest = EngineManifest(
            engine_id="energy", name="Energy", version="1.0.0", description="test energy engine",
            input_contracts=("audio-frame.v1",), output_contracts=("inference-result.v1",),
            classes=("siren",), settings_schema=ENERGY_SCHEMA, resources={}, launch={},
        )
        yamnet_manifest = EngineManifest(
            engine_id="yamnet", name="YAMNet", version="1.0.0", description="test YAMNet engine",
            input_contracts=("audio-frame.v1",), output_contracts=("inference-result.v1",),
            classes=("siren",), settings_schema=YAMNET_SCHEMA, resources={}, launch={},
        )
        validate_settings_schema({"energy_event_threshold": 0.11}, energy_manifest.settings_schema)
        validate_settings_schema({"siren_threshold": 0.22}, yamnet_manifest.settings_schema)
        with self.assertRaisesRegex(ValueError, "unknown engine settings"):
            validate_settings_schema(
                {"energy_event_threshold": 0.11},
                {"type": "object", "properties": {}, "additionalProperties": False},
            )

    def test_graph_engine_settings_do_not_mutate_legacy_process_settings(self) -> None:
        legacy_defaults = {
            "siren_threshold": 0.1,
            "inference_interval_seconds": 0.5,
            "dsp_source_overrides": {"3": {"dsp_enabled": 1}},
        }
        with patch.dict(settings_module.settings, legacy_defaults, clear=True):
            graph_runtime = type("Runtime", (), {"engine_graph_active": True})()
            graph_values = settings_for_engine(
                graph_runtime,
                {"siren_threshold": 0.8, "energy_event_threshold": 0.2},
            )
            self.assertEqual(graph_values["siren_threshold"], 0.8)
            graph_values["dsp_source_overrides"]["3"]["dsp_enabled"] = 0
            self.assertEqual(settings_module.settings, legacy_defaults)

            legacy_runtime = type("Runtime", (), {"engine_graph_active": False})()
            legacy_values = settings_for_engine(legacy_runtime, {"siren_threshold": 0.4})
            self.assertIs(legacy_values, settings_module.settings)
            self.assertEqual(settings_module.settings["siren_threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
