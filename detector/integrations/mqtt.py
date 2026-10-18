"""MQTT integration: Home Assistant discovery and state publishing."""

from __future__ import annotations

import json
import os

from ..bus import EventBus
from ..config import DETECTORS

MQTT_HOST = os.environ.get("MQTT_HOST", "")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")

_mqtt_client = None  # paho.mqtt.client.Client | None


def mqtt_connect() -> None:
    """Connect to MQTT broker and publish HA discovery configs."""
    global _mqtt_client
    if not MQTT_HOST:
        print("MQTT_HOST not set — MQTT disabled.", flush=True)
        return
    try:
        import paho.mqtt.client as paho_mqtt
    except ImportError:
        print("paho-mqtt not installed — MQTT disabled.", flush=True)
        return

    client = paho_mqtt.Client(
        paho_mqtt.CallbackAPIVersion.VERSION2,
        client_id="siren-detector",
        protocol=paho_mqtt.MQTTv5,
    )
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    client.will_set("sound_detector/availability", "offline", retain=True)

    def on_connect(c, _ud, _flags, rc, _props=None):
        print(f"MQTT connected to {MQTT_HOST}:{MQTT_PORT}", flush=True)
        c.publish("sound_detector/availability", "online", retain=True)
        _publish_discovery(c)
        for name in DETECTORS:
            c.publish(f"sound_detector/{name}/state", "OFF", retain=True)
            c.publish(f"sound_detector/{name}/status", "idle", retain=True)
            c.publish(f"sound_detector/{name}/confidence", "0", retain=True)

    def on_disconnect(c, _ud, _flags, rc, _props=None):
        print(f"MQTT disconnected: rc={rc}", flush=True)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    try:
        client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        client.loop_start()
        _mqtt_client = client
    except Exception as exc:
        print(f"MQTT connection error: {exc}", flush=True)


def _publish_discovery(client) -> None:
    """Publish Home Assistant MQTT Discovery configs for each detector."""
    device = {
        "identifiers": ["sound_detector_01"],
        "name": "Sound Detector",
        "manufacturer": "DIY",
        "model": "ESP32-S3 + YAMNet",
    }
    availability = {"topic": "sound_detector/availability"}

    for name in DETECTORS:
        bs_config = {
            "name": f"Sound {name}",
            "unique_id": f"sound_detector_{name}",
            "device_class": "sound",
            "state_topic": f"sound_detector/{name}/state",
            "payload_on": "ON",
            "payload_off": "OFF",
            "availability": availability,
            "device": device,
        }
        client.publish(
            f"homeassistant/binary_sensor/sound_detector_{name}/config",
            json.dumps(bs_config),
            retain=True,
        )

        state_sensor_config = {
            "name": f"Sound {name} status",
            "unique_id": f"sound_detector_{name}_status",
            "state_topic": f"sound_detector/{name}/status",
            "icon": "mdi:ear-hearing",
            "availability": availability,
            "device": device,
        }
        client.publish(
            f"homeassistant/sensor/sound_detector_{name}_status/config",
            json.dumps(state_sensor_config),
            retain=True,
        )

        sensor_config = {
            "name": f"Sound {name} confidence",
            "unique_id": f"sound_detector_{name}_confidence",
            "state_topic": f"sound_detector/{name}/confidence",
            "unit_of_measurement": "",
            "icon": "mdi:volume-high",
            "availability": availability,
            "device": device,
        }
        client.publish(
            f"homeassistant/sensor/sound_detector_{name}_confidence/config",
            json.dumps(sensor_config),
            retain=True,
        )
    print("MQTT discovery configs published.", flush=True)


def _mqtt_publish_event(name: str, state: str, confidence: float, source: str = "") -> None:
    """Publish detector state change to MQTT.

    When *source* is non-empty an additional per-source topic is published
    so Home Assistant can distinguish which microphone triggered the event.
    """
    if _mqtt_client is None:
        return
    _mqtt_client.publish(f"sound_detector/{name}/status", state, retain=True)
    if state == "confirmed":
        _mqtt_client.publish(f"sound_detector/{name}/state", "ON", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", str(round(confidence, 3)), retain=True)
        if source:
            _mqtt_client.publish(f"sound_detector/{name}/source", source, retain=True)
    elif state == "idle":
        _mqtt_client.publish(f"sound_detector/{name}/state", "OFF", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", "0", retain=True)


def _on_bus_message(msg: dict) -> None:
    """Bus subscriber callback: forward events to MQTT."""
    if msg.get("type") != "event":
        return
    name = msg.get("event", "")
    state = msg.get("state", "")
    confidence = msg.get("confidence", 0.0)
    source = msg.get("source", "")
    if state in ("idle", "possible", "confirmed"):
        _mqtt_publish_event(name, state, confidence, source)


def mqtt_subscribe(bus: EventBus) -> None:
    """Subscribe to EventBus so all engines get MQTT publishing for free."""
    bus.subscribe(_on_bus_message)
