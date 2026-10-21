"""Integration checks for the real UDP input, Runtime Hub, and FastAPI routes."""

from __future__ import annotations

import os
import importlib
import sys
import queue
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

_previous_auth_enabled = os.environ.get("AUTH_ENABLED")
os.environ["AUTH_ENABLED"] = "false"
from detector import auth
from detector.bus import EventBus
from detector.config import HEADER, MAGIC
from detector.runtime.hub import RuntimeHub
from detector.runtime.manifest import EngineManifest
from detector.server import routes_clips
from detector.server.app import create_app
from detector.transport.udp import UDPSourceAdapter

if _previous_auth_enabled is None:
    os.environ.pop("AUTH_ENABLED", None)
else:
    os.environ["AUTH_ENABLED"] = _previous_auth_enabled


class RuntimeIntegrationTests(unittest.TestCase):
    def test_real_udp_frames_reach_runtime_and_api_metrics(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(("127.0.0.1", 0))
            except PermissionError:
                self.skipTest("sandbox does not allow opening a UDP loopback socket")
            udp_port = probe.getsockname()[1]

        bus = EventBus()
        runtime = RuntimeHub(bus)
        received = queue.Queue()
        runtime.register_component("integration-probe", "Integration probe", "1.0.0")
        runtime.subscribe_audio(
            received.put, subscriber_id="integration-probe", engine_id="integration-probe",
        )
        runtime.add_input(UDPSourceAdapter("127.0.0.1", udp_port))
        app = create_app(bus, runtime)

        try:
            with tempfile.TemporaryDirectory() as clips_dir, \
                    patch.object(routes_clips, "CLIP_DIR", Path(clips_dir)), \
                    patch.object(auth, "AUTH_ENABLED", False), \
                    TestClient(app) as client:
                payload = bytes(16)
                packet = HEADER.pack(MAGIC, 1, len(payload), 321) + payload
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    sender.sendto(packet, ("127.0.0.1", udp_port))

                frame = received.get(timeout=2)
                self.assertEqual(frame.source_id, 321)
                self.assertEqual(frame.sequence, 1)
                self.assertEqual(frame.samples.dtype, np.float32)

                status = client.get("/api/runtime/status")
                self.assertEqual(status.status_code, 200)
                subscription = next(
                    item for item in status.json()["subscribers"]
                    if item["subscriber_id"] == "integration-probe"
                )
                self.assertGreaterEqual(subscription["frames_received"], 1)
                self.assertEqual(subscription["engine_id"], "integration-probe")

                metrics = client.get("/metrics")
                self.assertEqual(metrics.status_code, 200)
                self.assertIn("detector_audio_frames_received_total", metrics.text)
                self.assertIn("integration-probe", metrics.text)

                with client.stream("GET", "/events") as response:
                    self.assertEqual(response.status_code, 200)
                    lines = iter(response.iter_lines())
                    self.assertEqual(next(lines), "event: session")
                    self.assertTrue(next(lines).startswith("data: {"))
                    self.assertEqual(next(lines), "")
                    self.assertEqual(next(lines), "event: score")
        finally:
            runtime.stop()

    def test_main_starts_registered_engine_udp_and_real_http_api(self) -> None:
        def free_port(kind: int) -> int:
            with socket.socket(socket.AF_INET, kind) as probe:
                probe.bind(("127.0.0.1", 0))
                return probe.getsockname()[1]

        try:
            udp_port = free_port(socket.SOCK_DGRAM)
            web_port = free_port(socket.SOCK_STREAM)
        except PermissionError:
            self.skipTest("sandbox does not allow local network sockets")

        frames = queue.Queue()
        engine_id = "main-startup-probe"
        manifest = EngineManifest(
            engine_id=engine_id,
            name="Main startup probe",
            version="1.0.0",
            description="Integration engine used to verify main startup wiring.",
            input_contracts=("audio-frame.v1",),
            output_contracts=("inference-result.v1",),
            classes=("siren",),
            settings_schema={"type": "object", "additionalProperties": False},
            resources={},
            launch={},
        )

        class ProbeEngine:
            def start(self, runtime, _settings, _args):
                runtime.subscribe_audio(
                    frames.put, subscriber_id=engine_id, engine_id=engine_id,
                )

        sys.modules.pop("detector.main", None)
        main_module = importlib.import_module("detector.main")
        settings = {
            "engine_graph_contract": "engine-graph.v1",
            "engine_configs": {
                engine_id: {"selector": engine_id, "enabled": True, "settings": {}},
            },
            "engine_routes": {"default": [engine_id], "by_source": {}},
        }
        server_ref = {}
        errors = []
        real_serve = main_module.uvicorn.Server.serve

        async def tracked_serve(server, *args, **kwargs):
            server_ref["server"] = server
            await real_serve(server, *args, **kwargs)

        def run_main():
            try:
                main_module.main()
            except BaseException as exc:
                errors.append(exc)

        with patch.object(main_module, "load_settings", return_value=settings), \
                patch.object(main_module.engine_registry, "resolve", return_value=(ProbeEngine, manifest)), \
                patch.object(main_module, "mqtt_connect"), \
                patch.object(main_module, "mqtt_subscribe"), \
                patch.object(main_module.WebhookOutput, "from_environment", return_value=None), \
                patch.object(main_module.uvicorn.Server, "serve", tracked_serve), \
                patch.object(auth, "AUTH_ENABLED", False), \
                patch.object(sys, "argv", ["detector", "--port", str(udp_port), "--web-port", str(web_port)]):
            thread = threading.Thread(target=run_main, name="integration-main", daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 8
                response = None
                while time.monotonic() < deadline:
                    if errors:
                        raise errors[0]
                    try:
                        response = httpx.get(f"http://127.0.0.1:{web_port}/api/runtime/status", timeout=0.25)
                        break
                    except httpx.HTTPError:
                        time.sleep(0.025)
                self.assertIsNotNone(response, "FastAPI server did not become ready")
                self.assertEqual(response.status_code, 200)
                status = response.json()
                self.assertEqual(status["components"][0]["component_id"], engine_id)

                payload = bytes(16)
                packet = HEADER.pack(MAGIC, 1, len(payload), 654) + payload
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    sender.sendto(packet, ("127.0.0.1", udp_port))
                frame = frames.get(timeout=3)
                self.assertEqual(frame.source_id, 654)
                self.assertEqual(frame.sequence, 1)

                metrics = httpx.get(f"http://127.0.0.1:{web_port}/metrics", timeout=1)
                self.assertEqual(metrics.status_code, 200)
                self.assertIn(engine_id, metrics.text)
            finally:
                if "server" in server_ref:
                    server_ref["server"].should_exit = True
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive(), "main process did not shut down")
                if errors:
                    raise errors[0]


if __name__ == "__main__":
    unittest.main()
