"""MQTT compatibility payload tests."""

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch


class MqttEventTests(unittest.TestCase):
    def test_event_stream_preserves_engine_and_source_ids(self) -> None:
        bus_module = types.ModuleType("detector.bus")
        bus_module.EventBus = type("EventBus", (), {})
        with patch.dict(sys.modules, {"detector.bus": bus_module}):
            mqtt = importlib.import_module("detector.integrations.mqtt")
            client = Mock()
            previous_client = mqtt._mqtt_client
            mqtt._mqtt_client = client
            try:
                mqtt._on_bus_message({
                    "type": "event",
                    "event": "siren",
                    "state": "confirmed",
                    "confidence": 0.81234,
                    "source": "front-mic",
                    "source_id": 12,
                    "engine_id": "yamnet",
                    "timestamp": "2026-10-21T10:15:30Z",
                })
            finally:
                mqtt._mqtt_client = previous_client

        event_call = next(
            call for call in client.publish.call_args_list
            if call.args[0] == "sound_detector/events"
        )
        payload = json.loads(event_call.args[1])
        self.assertEqual(payload["engine_id"], "yamnet")
        self.assertEqual(payload["source_id"], 12)
        self.assertEqual(payload["confidence"], 0.812)
        self.assertFalse(event_call.kwargs["retain"])
