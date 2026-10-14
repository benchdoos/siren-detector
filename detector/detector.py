#!/usr/bin/env python3
"""
Streaming siren detector with Web UI.

Receives UDP audio from ESP32, runs YAMNet + MLP classifier,
saves audio clips on confirmed events, streams events via WebSocket.

Web UI: http://raspi5.local:8080
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow_hub as hub
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

SAMPLE_RATE = 16_000
PACKET_MAX = 4096 + 12
HEADER = struct.Struct("!IIHH")
MAGIC = 0x53445231

MODEL_URL = "https://tfhub.dev/google/yamnet/1"
ANALYSIS_SECONDS = 2.0
INFERENCE_HOP_SAMPLES = SAMPLE_RATE // 2

SMOOTHING_WINDOW_SECONDS = 30
POSSIBLE_TRIGGERS = 4
CONFIRMED_TRIGGERS = 8
COOLDOWN_SECONDS = 60

# Each detector: event name → (YAMNet class names, threshold, clip prefix)
DETECTORS = {
    "siren":      (("Siren",), 0.10, ""),
    "explosion":  (("Explosion", "Burst, pop", "Gunshot, gunfire"), 0.15, "exp-"),
    "motorcycle": (("Motorcycle",), 0.15, "moto-"),
}

# When MLP is used for siren, require YAMNet to also show at least this much
# to filter out false positives (e.g. Speech classified as siren by MLP)
YAMNET_MIN_GATE = 0.01

CLIP_SECONDS = 10       # default; overridden by settings["clip_seconds"]
CLIP_DIR = Path("/clips")
WEB_PORT = 8080

# ── UDP packet stats ─────────────────────────────────────────────────────────

_udp_stats_lock = threading.Lock()
_udp_last_packet_time: float = 0.0  # monotonic
_udp_packet_count: int = 0
_udp_packet_count_prev: int = 0
_udp_pps: float = 0.0
_udp_pps_last_calc: float = 0.0


def _udp_stats_update() -> None:
    """Called on each received UDP packet."""
    global _udp_last_packet_time, _udp_packet_count
    with _udp_stats_lock:
        _udp_last_packet_time = time.monotonic()
        _udp_packet_count += 1


def _udp_stats_snapshot() -> dict:
    """Return current UDP stats for broadcasting to UI."""
    global _udp_pps, _udp_pps_last_calc, _udp_packet_count_prev
    now = time.monotonic()
    with _udp_stats_lock:
        last = _udp_last_packet_time
        count = _udp_packet_count
    # recalculate pps every ~2 seconds
    dt = now - _udp_pps_last_calc
    if dt >= 2.0:
        _udp_pps = (count - _udp_packet_count_prev) / dt
        _udp_packet_count_prev = count
        _udp_pps_last_calc = now
    ago = round(now - last, 1) if last > 0 else -1
    alive = last > 0 and (now - last) < 5.0
    return {"audio_alive": alive, "audio_ago": ago, "audio_pps": round(_udp_pps, 1)}

# ── Dynamic settings ─────────────────────────────────────────────────────────

SETTINGS_PATH = CLIP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "siren_threshold": 0.10,
    "explosion_threshold": 0.15,
    "motorcycle_threshold": 0.15,
    "possible_triggers": 4,
    "confirmed_triggers": 8,
    "cooldown_seconds": 60,
    "refractory_seconds": 15,
    "smoothing_window_seconds": 30,
    "yamnet_min_gate": 0.01,
    "clip_seconds": 10,
    "clip_from_possible": 0,
}

_settings: dict = {}
_smoothers: dict[str, TemporalSmoother] = {}
_clip_buf: ClipBuffer | None = None


def _load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            for k in DEFAULT_SETTINGS:
                if k in saved:
                    settings[k] = type(DEFAULT_SETTINGS[k])(saved[k])
        except Exception as exc:
            print(f"Settings load error: {exc}", flush=True)
    return settings


def _save_settings() -> None:
    """Persist current settings to JSON file."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(_settings, indent=2), encoding="utf-8")


def _apply_clip_seconds() -> None:
    """Resize ClipBuffer when clip_seconds setting changes."""
    global _clip_buf
    if _clip_buf is None:
        return
    new_seconds = _settings.get("clip_seconds", CLIP_SECONDS)
    new_maxlen = int(new_seconds * SAMPLE_RATE)
    if _clip_buf._buf.maxlen != new_maxlen:
        _clip_buf._buf = collections.deque(_clip_buf._buf, maxlen=new_maxlen)
        print(f"[settings] clip buffer resized to {new_seconds}s", flush=True)


def _apply_settings_to_smoothers() -> None:
    """Push current settings into running TemporalSmoother instances."""
    for name, smoother in _smoothers.items():
        thr_key = f"{name}_threshold"
        if thr_key in _settings:
            smoother._threshold = _settings[thr_key]
        smoother._possible = _settings["possible_triggers"]
        smoother._confirmed = _settings["confirmed_triggers"]
        smoother._cooldown_seconds = _settings["cooldown_seconds"]
        smoother._refractory_seconds = _settings["refractory_seconds"]
        new_maxlen = int(_settings["smoothing_window_seconds"] / 0.5) + 1
        if smoother._scores.maxlen != new_maxlen:
            smoother._scores = collections.deque(smoother._scores, maxlen=new_maxlen)

# ── MQTT / Home Assistant ─────────────────────────────────────────────────────

MQTT_HOST = os.environ.get("MQTT_HOST", "")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")

_mqtt_client = None  # paho.mqtt.client.Client | None


def _mqtt_connect() -> None:
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
        _mqtt_publish_discovery(c)
        # reset all states on connect to avoid stale retained messages
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


def _mqtt_publish_discovery(client) -> None:
    """Publish Home Assistant MQTT Discovery configs for each detector."""
    device = {
        "identifiers": ["sound_detector_01"],
        "name": "Sound Detector",
        "manufacturer": "DIY",
        "model": "ESP32-S3 + YAMNet",
    }
    availability = {"topic": "sound_detector/availability"}

    for name in DETECTORS:
        # binary_sensor: ON when confirmed
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

        # sensor: detector state (idle/possible/confirmed)
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

        # sensor: confidence 0..1
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


def mqtt_publish_event(name: str, state: str, confidence: float) -> None:
    """Publish detector state change to MQTT."""
    if _mqtt_client is None:
        return
    _mqtt_client.publish(f"sound_detector/{name}/status", state, retain=True)
    if state == "confirmed":
        _mqtt_client.publish(f"sound_detector/{name}/state", "ON", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", str(round(confidence, 3)), retain=True)
    elif state == "idle":
        _mqtt_client.publish(f"sound_detector/{name}/state", "OFF", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", "0", retain=True)


# ── Admin auth ────────────────────────────────────────────────────────────────

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() not in ("0", "false", "no")
ADMIN_TOKEN: str = ""
_TOKEN_HASH: bytes = b""

if AUTH_ENABLED:
    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
    if not ADMIN_TOKEN:
        ADMIN_TOKEN = secrets.token_urlsafe(32)
        print(f"Generated ADMIN_TOKEN: {ADMIN_TOKEN}", flush=True)
        print("Set ADMIN_TOKEN env var to use a fixed token.", flush=True)
    _TOKEN_HASH = hashlib.sha256(ADMIN_TOKEN.encode()).digest()
else:
    print("AUTH_ENABLED=false — admin auth disabled, all users have full access.", flush=True)


def _check_token(token: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(token.encode()).digest(),
        _TOKEN_HASH,
    )


async def require_admin(authorization: str = Header("")):
    if not AUTH_ENABLED:
        return
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not _check_token(token):
        raise HTTPException(status_code=401, detail="Admin token required")


async def require_viewer(authorization: str = Header("")):
    """Read-level auth: same token, but separate dependency for clarity."""
    if not AUTH_ENABLED:
        return
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not _check_token(token):
        raise HTTPException(status_code=401, detail="Authentication required")


def _ws_check_token(ws: WebSocket) -> bool:
    """Check token from WebSocket query param ?token=..."""
    if not AUTH_ENABLED:
        return True
    token = ws.query_params.get("token", "")
    return bool(token) and _check_token(token)


# ── audio helpers ──────────────────────────────────────────────────────────────

def i2s32_to_float32(payload: bytes) -> np.ndarray:
    n = len(payload) // 4
    samples = np.frombuffer(payload[: n * 4], dtype="<i4")
    return (samples >> 16).astype(np.float32) / 32768.0


class AudioRingBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: collections.deque = collections.deque(maxlen=capacity)
        self._pending = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(samples.tolist())
            self._pending += len(samples)

    def consume(self, hop: int) -> np.ndarray | None:
        with self._lock:
            if self._pending < hop:
                return None
            self._pending = 0
            return np.array(self._buf, dtype=np.float32)


class ClipBuffer:
    """Rolling buffer for saving audio clips around events."""
    def __init__(self, seconds: float) -> None:
        self._buf: collections.deque = collections.deque(maxlen=int(seconds * SAMPLE_RATE))
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.extend(samples.tolist())

    def save(self, path: Path, last_seconds: float | None = None) -> None:
        with self._lock:
            audio = np.array(self._buf, dtype=np.float32)
        if last_seconds is not None:
            max_samples = int(last_seconds * SAMPLE_RATE)
            if len(audio) > max_samples:
                audio = audio[-max_samples:]
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ── detection ──────────────────────────────────────────────────────────────────

class TemporalSmoother:
    def __init__(self, window_seconds, hop_seconds, threshold, possible, confirmed,
                 cooldown_seconds, refractory_seconds):
        maxlen = int(window_seconds / hop_seconds) + 1
        self._scores: collections.deque = collections.deque(maxlen=maxlen)
        self._threshold = threshold
        self._possible = possible
        self._confirmed = confirmed
        self._cooldown_seconds = cooldown_seconds
        self._refractory_seconds = refractory_seconds
        self._cooldown_until = 0.0
        self._refractory_until = 0.0
        self._state = "idle"
        self._last_trigger_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def trigger_count(self) -> int:
        return self._last_trigger_count

    @property
    def cooldown_remaining(self) -> float:
        r = self._cooldown_until - time.monotonic()
        return max(0.0, r)

    @property
    def refractory_remaining(self) -> float:
        r = self._refractory_until - time.monotonic()
        return max(0.0, r)

    @property
    def possible_needed(self) -> int:
        return self._possible

    @property
    def confirmed_needed(self) -> int:
        return self._confirmed

    def push(self, score: float) -> str | None:
        now = time.monotonic()
        if now < self._cooldown_until:
            if score >= self._threshold:
                self._cooldown_until = now + self._cooldown_seconds
            return None
        if self._state == "confirmed":
            self._state = "idle"
            self._refractory_until = now + self._refractory_seconds
            self._scores.clear()
            return "idle"
        # refractory: time-based, but ends early if score drops below threshold
        if now < self._refractory_until:
            if score < self._threshold:
                self._refractory_until = 0.0  # quiet → end early
            else:
                self._scores.clear()
                return None
        prev = self._state
        self._scores.append(score)
        triggers = sum(1 for s in self._scores if s >= self._threshold)
        self._last_trigger_count = triggers
        if triggers >= self._confirmed:
            self._state = "confirmed"
            self._cooldown_until = now + self._cooldown_seconds
            self._scores.clear()
        elif triggers >= self._possible:
            self._state = "possible"
        else:
            self._state = "idle"
        return self._state if self._state != prev else None


# ── WebSocket broadcast ────────────────────────────────────────────────────────

_client_queues: dict[WebSocket, asyncio.Queue] = {}
_audio_queues: dict[WebSocket, asyncio.Queue] = {}
_loop: asyncio.AbstractEventLoop | None = None
_event_history: collections.deque = collections.deque(maxlen=100)
_last_score: dict = {"type": "score", "top": ""}


def _enqueue_all(msg: dict) -> None:
    for q in _client_queues.values():
        q.put_nowait(msg)


def _enqueue_audio(data: bytes) -> None:
    dead = []
    for ws, q in _audio_queues.items():
        if q.qsize() < 50:  # drop if client is too slow
            q.put_nowait(data)
        else:
            dead.append(ws)
    for ws in dead:
        _audio_queues.pop(ws, None)


def broadcast(msg: dict) -> None:
    global _last_score
    if msg.get("type") == "event":
        _event_history.append(msg)
    elif msg.get("type") == "score":
        _last_score = msg
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_enqueue_all, msg)


def broadcast_audio(samples: np.ndarray) -> None:
    """Send raw float32 PCM to all listening audio WebSocket clients."""
    if not _audio_queues or not _loop or _loop.is_closed():
        return
    data = samples.astype(np.float32).tobytes()
    _loop.call_soon_threadsafe(_enqueue_audio, data)


# ── FastAPI app ────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(application: FastAPI):
    await _restore_clips()
    yield


app = FastAPI(lifespan=_lifespan)
_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


@app.get("/clips/{filename}")
async def serve_clip(filename: str, token: str = "", authorization: str = Header("")):
    if AUTH_ENABLED:
        hdr_token = authorization.removeprefix("Bearer ").strip()
        if not ((hdr_token and _check_token(hdr_token)) or (token and _check_token(token))):
            raise HTTPException(status_code=401, detail="Authentication required")
    path = (CLIP_DIR / filename).resolve()
    if not path.is_relative_to(CLIP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(str(path), media_type="audio/wav")


DATASET_DIR = Path("/dataset")
VALID_LABELS = {"siren", "background", "motorcycle", "explosion", "unknown"}
_feedback: dict[str, str] = {}  # filename → label


@app.get("/api/settings", dependencies=[Depends(require_viewer)])
async def get_settings():
    return {**_settings, "defaults": DEFAULT_SETTINGS}


@app.get("/api/auth/status")
async def auth_status():
    return {"auth_enabled": AUTH_ENABLED}


@app.post("/api/auth/check")
async def auth_check(body: dict):
    if not AUTH_ENABLED:
        return {"ok": True}
    token = body.get("token", "")
    if _check_token(token):
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/api/settings", dependencies=[Depends(require_admin)])
async def post_settings(body: dict):
    SETTING_RANGES = {
        "siren_threshold": (0.0, 1.0),
        "explosion_threshold": (0.0, 1.0),
        "motorcycle_threshold": (0.0, 1.0),
        "possible_triggers": (1, 60),
        "confirmed_triggers": (1, 60),
        "cooldown_seconds": (0, 600),
        "refractory_seconds": (0, 120),
        "smoothing_window_seconds": (5, 120),
        "yamnet_min_gate": (0.0, 1.0),
        "clip_seconds": (5, 60),
        "clip_from_possible": (0, 1),
    }
    updated = {}
    for k, default_val in DEFAULT_SETTINGS.items():
        if k in body:
            try:
                val = type(default_val)(body[k])
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid value for {k}")
            lo, hi = SETTING_RANGES.get(k, (None, None))
            if lo is not None and not (lo <= val <= hi):
                raise HTTPException(status_code=400, detail=f"{k} must be between {lo} and {hi}")
            updated[k] = val
    if not updated:
        raise HTTPException(status_code=400, detail="No valid settings provided")
    _settings.update(updated)
    _save_settings()
    _apply_settings_to_smoothers()
    _apply_clip_seconds()
    print(f"[settings] updated: {updated}", flush=True)
    return _settings


@app.post("/api/settings/reset", dependencies=[Depends(require_admin)])
async def reset_settings():
    _settings.clear()
    _settings.update(DEFAULT_SETTINGS)
    _save_settings()
    _apply_settings_to_smoothers()
    _apply_clip_seconds()
    print("[settings] reset to defaults", flush=True)
    return _settings


@app.get("/api/metrics", dependencies=[Depends(require_viewer)])
async def get_metrics():
    metrics_path = Path("/models/val_metrics.json")
    if not metrics_path.exists():
        return {"error": "no metrics available"}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/feedback", dependencies=[Depends(require_admin)])
async def feedback(body: dict):
    import shutil
    filename = body.get("filename", "")
    label = body.get("label", "")
    if not filename or label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid filename or label")
    src = (CLIP_DIR / filename).resolve()
    if not src.is_relative_to(CLIP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not src.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    dst_dir = DATASET_DIR / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / src.name))
    _feedback[filename] = label
    print(f"[feedback] {filename} → {label}", flush=True)
    return {"ok": True}


def _save_clip_meta(clip_name: str, meta: dict) -> None:
    """Save event metadata as JSON sidecar next to the WAV clip."""
    meta_path = CLIP_DIR / (clip_name.rsplit(".", 1)[0] + ".json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _load_clip_meta(clip_name: str) -> dict | None:
    """Load event metadata from JSON sidecar, if it exists."""
    meta_path = CLIP_DIR / (clip_name.rsplit(".", 1)[0] + ".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


async def _restore_clips():
    """Scan /clips on startup to restore event history from surviving WAV files."""
    if not CLIP_DIR.exists():
        return
    wavs = sorted(CLIP_DIR.glob("*.wav"))
    for path in wavs:
        name = path.name
        meta = _load_clip_meta(name)
        if meta:
            event_type = meta["event"]
            ts = meta["timestamp"]
        else:
            # fallback: detect event type from prefix
            if name.startswith("exp-"):
                event_type = "explosion"
                date_part = name[4:].replace(".wav", "")
            elif name.startswith("moto-"):
                event_type = "motorcycle"
                date_part = name[5:].replace(".wav", "")
            elif name.startswith("manual-"):
                event_type = "unknown"
                date_part = name[7:].replace(".wav", "")
            else:
                event_type = "siren"
                date_part = name.replace(".wav", "")
            # parse YYYYMMDD-HHMMSS → ISO timestamp
            try:
                ts = datetime.strptime(date_part, "%Y%m%d-%H%M%S").replace(
                    tzinfo=timezone.utc
                ).isoformat(timespec="seconds")
            except ValueError:
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # check if feedback label exists in dataset
        for label in VALID_LABELS:
            if (DATASET_DIR / label / name).exists():
                _feedback[name] = label
                break
        ev = {
            "type": "event", "event": event_type, "state": "confirmed",
            "confidence": meta.get("confidence", 0.0) if meta else 0.0,
            "timestamp": ts,
            "source": meta.get("source", "restored") if meta else "restored",
            "engine": meta.get("engine", "unknown") if meta else "unknown",
            "clip_url": f"/clips/{name}",
            "yamnet": meta.get("yamnet", 0.0) if meta else 0.0,
            "mlp": meta.get("mlp") if meta else None,
            "yamnet_top": meta.get("yamnet_top", "") if meta else "",
            "yamnet_top3": meta.get("yamnet_top3", []) if meta else [],
            "mlp_top3": meta.get("mlp_top3", []) if meta else [],
            "threshold": meta.get("threshold") if meta else None,
            "triggers": meta.get("triggers") if meta else None,
            "triggers_needed": meta.get("triggers_needed") if meta else None,
        }
        _event_history.append(ev)
    if wavs:
        print(f"Restored {len(wavs)} clips from /clips/", flush=True)


@app.websocket("/ws/audio")
async def ws_audio_endpoint(ws: WebSocket):
    """Stream raw float32 PCM audio to the browser for live listening."""
    if not _ws_check_token(ws):
        await ws.close(code=4401, reason="Authentication required")
        return
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    _audio_queues[ws] = queue
    try:
        async def sender():
            while True:
                data = await queue.get()
                await ws.send_bytes(data)

        async def receiver():
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    return

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception:
        pass
    finally:
        _audio_queues.pop(ws, None)


@app.post("/api/manual-event", dependencies=[Depends(require_admin)])
async def manual_event():
    """Save the current clip buffer as a manual (user-triggered) event."""
    if _clip_buf is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    ts = datetime.now(timezone.utc)
    clip_name = f"manual-{ts.strftime('%Y%m%d-%H%M%S')}.wav"
    _clip_buf.save(CLIP_DIR / clip_name)
    clip_url = f"/clips/{clip_name}"
    ts_iso = ts.isoformat(timespec="seconds")
    meta = {
        "event": "unknown", "confidence": 0.0,
        "timestamp": ts_iso, "source": "manual", "engine": "user",
    }
    _save_clip_meta(clip_name, meta)
    ev = {
        "type": "event", "event": "unknown", "state": "confirmed",
        "confidence": 0.0, "timestamp": ts_iso,
        "source": "manual", "engine": "user", "clip_url": clip_url,
        "yamnet": 0.0, "mlp": None, "yamnet_top": "",
        "yamnet_top3": [], "mlp_top3": [],
    }
    broadcast(ev)
    print(f"[{ts_iso}] manual event saved: {clip_url}", flush=True)
    return {"ok": True, "clip_url": clip_url}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not _ws_check_token(ws):
        await ws.close(code=4401, reason="Authentication required")
        return
    await ws.accept()
    await ws.send_json(_last_score)
    for ev in list(_event_history):
        clip_url = ev.get("clip_url")
        if not clip_url:
            continue
        filename = clip_url.split("/")[-1]
        resolved = (CLIP_DIR / filename).resolve()
        if not resolved.is_relative_to(CLIP_DIR.resolve()) or not resolved.exists():
            continue
        ev_out = dict(ev)
        if filename in _feedback:
            ev_out["label"] = _feedback[filename]
        await ws.send_json(ev_out)

    queue: asyncio.Queue = asyncio.Queue()
    _client_queues[ws] = queue

    async def sender():
        while True:
            msg = await queue.get()
            await ws.send_json(msg)

    async def receiver():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

    try:
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    except Exception:
        pass
    finally:
        _client_queues.pop(ws, None)


# ── UDP receiver ───────────────────────────────────────────────────────────────

def udp_receiver(
    inference_buf: AudioRingBuffer,
    clip_buf: ClipBuffer,
    bind: str,
    port: int,
    stop: threading.Event,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))
    sock.settimeout(1.0)
    expected_seq: int | None = None
    last_recv = time.monotonic()
    print(f"Listening on {bind}:{port}", flush=True)
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(PACKET_MAX)
        except socket.timeout:
            # reset seq tracking after 5s of silence so reconnect works
            if expected_seq is not None and time.monotonic() - last_recv > 5.0:
                print("UDP: no packets for 5s, resetting sequence tracking", flush=True)
                expected_seq = None
            continue
        if len(data) < HEADER.size:
            continue
        magic, seq, payload_size, _ = HEADER.unpack_from(data)
        if magic != MAGIC or payload_size > len(data) - HEADER.size:
            continue
        if expected_seq is None:
            expected_seq = seq
            print(f"UDP: source connected, seq={seq}", flush=True)
        if seq < expected_seq:
            # seq jumped backwards — source restarted, accept new sequence
            if expected_seq - seq > 100:
                print(f"UDP: seq reset detected ({expected_seq} → {seq}), re-syncing", flush=True)
                expected_seq = seq
            else:
                continue
        gap = seq - expected_seq
        if gap:
            silence = np.zeros(payload_size // 4 * gap, dtype=np.float32)
            inference_buf.push(silence)
            clip_buf.push(silence)
        payload = data[HEADER.size : HEADER.size + payload_size]
        samples = i2s32_to_float32(payload)
        inference_buf.push(samples)
        clip_buf.push(samples)
        broadcast_audio(samples)
        _udp_stats_update()
        last_recv = time.monotonic()
        expected_seq = seq + 1
    sock.close()


# ── inference loop ─────────────────────────────────────────────────────────────

def inference_loop(args: argparse.Namespace) -> None:
    global _loop, _settings, _smoothers

    _settings.update(_load_settings())
    print(f"Settings: {_settings}", flush=True)

    print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
    yamnet = hub.load(MODEL_URL)

    with open(yamnet.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    # resolve YAMNet indices for each detector
    detector_indices: dict[str, list[int]] = {}
    for name, (yamnet_classes, _, _) in DETECTORS.items():
        indices = [i for i, n in enumerate(class_names) if n in yamnet_classes]
        detector_indices[name] = indices
        print(f"Detector '{name}' YAMNet classes: {[class_names[i] for i in indices]}", flush=True)

    clf = None
    engine = "yamnet"
    weights_path = Path(args.classifier_weights)
    labels_path = Path(args.classifier_labels)
    if weights_path.exists() and labels_path.exists():
        from classifier_infer import SirenClassifier
        clf = SirenClassifier(weights_path, labels_path)
        engine = "yamnet+mlp"
        print(f"Classifier loaded: {clf.labels}", flush=True)
    else:
        print("No classifier weights — using raw YAMNet siren score.", flush=True)

    inference_buf = AudioRingBuffer(capacity=int(ANALYSIS_SECONDS * SAMPLE_RATE))
    global _clip_buf
    _clip_buf = ClipBuffer(seconds=_settings.get("clip_seconds", CLIP_SECONDS))
    clip_buf = _clip_buf
    smoothers: dict[str, TemporalSmoother] = {}
    for name in DETECTORS:
        thr_key = f"{name}_threshold"
        smoothers[name] = TemporalSmoother(
            window_seconds=_settings["smoothing_window_seconds"],
            hop_seconds=0.5,
            threshold=_settings.get(thr_key, 0.15),
            possible=_settings["possible_triggers"],
            confirmed=_settings["confirmed_triggers"],
            cooldown_seconds=_settings["cooldown_seconds"],
            refractory_seconds=_settings["refractory_seconds"],
        )
    _smoothers = smoothers

    # Track possible start time and per-window scores for clip_from_possible mode
    possible_start: dict[str, float | None] = {name: None for name in DETECTORS}
    window_scores: dict[str, list[dict]] = {name: [] for name in DETECTORS}

    stop = threading.Event()
    threading.Thread(
        target=udp_receiver,
        args=(inference_buf, clip_buf, args.bind, args.port, stop),
        daemon=True,
    ).start()

    print("Ready.", flush=True)

    _last_heartbeat = 0.0
    _empty_peak = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": "", "yamnet_top3": [], "mlp_top3": []}
    peaks: dict[str, dict] = {name: dict(_empty_peak) for name in DETECTORS}
    while True:
        waveform = inference_buf.consume(INFERENCE_HOP_SAMPLES)
        if waveform is None:
            time.sleep(0.05)
            now = time.monotonic()
            if now - _last_heartbeat >= 1.0:
                heartbeat: dict = {"type": "score", "top": ""}
                for name in DETECTORS:
                    heartbeat[f"{name}_score"] = 0.0
                    heartbeat[f"{name}_state"] = smoothers[name].state
                    heartbeat[f"{name}_triggers"] = smoothers[name].trigger_count
                    heartbeat[f"{name}_possible_needed"] = smoothers[name].possible_needed
                    heartbeat[f"{name}_confirmed_needed"] = smoothers[name].confirmed_needed
                    heartbeat[f"{name}_cooldown"] = round(smoothers[name].cooldown_remaining, 1)
                    heartbeat[f"{name}_refractory"] = round(smoothers[name].refractory_remaining, 1)
                heartbeat.update(_udp_stats_snapshot())
                broadcast(heartbeat)
                _last_heartbeat = now
            continue

        waveform = np.clip(waveform, -1.0, 1.0)
        scores, embeddings, _ = yamnet(waveform)
        scores_np = scores.numpy()  # (n_windows, 521)
        top_idx = int(scores_np.mean(axis=0).argmax())
        yamnet_top = class_names[top_idx]

        # compute raw YAMNet score for each detector
        yamnet_scores: dict[str, float] = {}
        for name, indices in detector_indices.items():
            yamnet_scores[name] = float(scores_np[:, indices].max()) if indices else 0.0

        # MLP overrides siren score if available
        mlp_score = None
        mlp_top1 = None
        if clf is not None and "siren" in clf.labels:
            probs = clf.predict(embeddings.numpy())
            mean_probs = probs.mean(axis=0)
            siren_idx = clf.labels.index("siren")
            mlp_score = float(mean_probs[siren_idx])
            mlp_top1 = clf.labels[int(mean_probs.argmax())]

        # effective score: MLP for siren (if available), YAMNet for the rest
        # gate: if YAMNet sees no siren at all, don't trust MLP alone
        # top-1 filter: if MLP top-1 is not the target class and score < 50%,
        #               MLP itself thinks it's something else — don't trigger
        effective: dict[str, float] = {}
        for name in DETECTORS:
            if name == "siren" and mlp_score is not None:
                if yamnet_scores["siren"] >= _settings["yamnet_min_gate"]:
                    if mlp_top1 == name or mlp_score >= 0.5:
                        effective[name] = mlp_score
                    else:
                        effective[name] = 0.0
                else:
                    effective[name] = 0.0
            else:
                effective[name] = yamnet_scores[name]

        # push through smoothers
        state_changes: dict[str, str | None] = {}
        for name, smoother in smoothers.items():
            state_changes[name] = smoother.push(effective[name])

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # top-3 YAMNet classes
        mean_scores = scores_np.mean(axis=0)
        top3_idx = mean_scores.argsort()[::-1][:3]
        cur_yamnet_top3 = [
            {"name": class_names[i], "score": round(float(mean_scores[i]), 3)}
            for i in top3_idx
        ]

        # top-3 MLP classes
        cur_mlp_top3 = []
        if clf is not None:
            mean_probs = probs.mean(axis=0) if mlp_score is not None else clf.predict(embeddings.numpy()).mean(axis=0)
            mlp_top3_idx = mean_probs.argsort()[::-1][:3]
            cur_mlp_top3 = [
                {"name": clf.labels[i], "score": round(float(mean_probs[i]), 3)}
                for i in mlp_top3_idx
            ]

        # track peaks per detector
        for name in DETECTORS:
            if smoothers[name].state == "idle":
                peaks[name] = dict(_empty_peak)
            elif effective[name] > peaks[name]["score"]:
                peaks[name] = {
                    "score": effective[name],
                    "yamnet": yamnet_scores[name],
                    "mlp": mlp_score if name == "siren" else None,
                    "top": yamnet_top,
                    "yamnet_top3": cur_yamnet_top3,
                    "mlp_top3": cur_mlp_top3,
                }

        # track possible start and per-window scores for clip_from_possible
        for name in DETECTORS:
            st = smoothers[name].state
            if st == "possible" and possible_start[name] is None:
                possible_start[name] = time.monotonic()
                window_scores[name] = []
            if st in ("possible", "confirmed"):
                window_scores[name].append({
                    "t": round(time.monotonic() - (possible_start[name] or time.monotonic()), 2),
                    "score": round(effective[name], 3),
                })
            if st == "idle":
                possible_start[name] = None
                window_scores[name] = []

        # push scores to UI
        score_msg: dict = {"type": "score", "top": yamnet_top}
        for name in DETECTORS:
            score_msg[f"{name}_score"] = round(effective[name], 3)
            score_msg[f"{name}_state"] = smoothers[name].state
            score_msg[f"{name}_triggers"] = smoothers[name].trigger_count
            score_msg[f"{name}_possible_needed"] = smoothers[name].possible_needed
            score_msg[f"{name}_confirmed_needed"] = smoothers[name].confirmed_needed
            score_msg[f"{name}_cooldown"] = round(smoothers[name].cooldown_remaining, 1)
            score_msg[f"{name}_refractory"] = round(smoothers[name].refractory_remaining, 1)
        score_msg["yamnet"] = round(yamnet_scores.get("siren", 0.0), 3)
        score_msg["mlp"] = round(mlp_score, 3) if mlp_score is not None else None
        score_msg["yamnet_top3"] = cur_yamnet_top3
        if cur_mlp_top3:
            score_msg["mlp_top3"] = cur_mlp_top3

        score_msg.update(_udp_stats_snapshot())
        broadcast(score_msg)
        _last_heartbeat = time.monotonic()

        # emit events for each detector
        for name, (_, _, prefix) in DETECTORS.items():
            new_state = state_changes[name]
            if new_state == "idle":
                mqtt_publish_event(name, "idle", 0.0)
            if new_state not in ("possible", "confirmed"):
                continue
            clip_url = None
            if new_state == "confirmed":
                clip_name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                # clip_from_possible: save from possible start + 2s padding
                clip_duration = None
                if _settings.get("clip_from_possible") and possible_start[name] is not None:
                    elapsed = time.monotonic() - possible_start[name] + 2.0
                    clip_seconds = _settings.get("clip_seconds", CLIP_SECONDS)
                    clip_duration = min(elapsed, clip_seconds)
                clip_buf.save(CLIP_DIR / clip_name, last_seconds=clip_duration)
                clip_url = f"/clips/{clip_name}"
                print(f"[{ts}] {name} clip saved: {clip_url}", flush=True)
                # save event metadata as JSON sidecar for restore after restart
                meta = {
                    "event": name, "confidence": round(pk["score"], 3),
                    "timestamp": ts, "source": args.source, "engine": det_engine,
                    "yamnet": round(pk["yamnet"], 3),
                    "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                    "yamnet_top": pk["top"],
                    "yamnet_top3": pk.get("yamnet_top3", []),
                    "mlp_top3": pk.get("mlp_top3", []),
                    "threshold": _settings.get(thr_key, 0.15),
                    "triggers": smoothers[name].trigger_count,
                    "triggers_needed": _settings["confirmed_triggers"],
                }
                if window_scores[name]:
                    meta["window_scores"] = window_scores[name]
                _save_clip_meta(clip_name, meta)
                possible_start[name] = None
                window_scores[name] = []

            pk = peaks[name]
            det_engine = engine if (name == "siren" and clf is not None) else "yamnet"
            thr_key = f"{name}_threshold"
            event = {
                "type": "event", "event": name, "state": new_state,
                "confidence": round(pk["score"], 3), "timestamp": ts,
                "source": args.source, "engine": det_engine, "clip_url": clip_url,
                "yamnet": round(pk["yamnet"], 3),
                "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                "yamnet_top": pk["top"],
                "yamnet_top3": pk.get("yamnet_top3", []),
                "mlp_top3": pk.get("mlp_top3", []),
                "threshold": _settings.get(thr_key, 0.15),
                "triggers": smoothers[name].trigger_count,
                "triggers_needed": _settings["confirmed_triggers"] if new_state == "confirmed" else _settings["possible_triggers"],
            }
            broadcast(event)
            mqtt_publish_event(name, new_state, pk["score"])
            print(f"EVENT {json.dumps(event)}", flush=True)
            if new_state == "confirmed":
                peaks[name] = dict(_empty_peak)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global _loop

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--source", default="siren-detector-01")
    parser.add_argument("--classifier-weights", default="/models/classifier_weights.npz")
    parser.add_argument("--classifier-labels", default="/models/classifier_labels.json")
    parser.add_argument("--web-port", type=int, default=WEB_PORT)
    args = parser.parse_args()

    # connect to MQTT (non-blocking, works without broker too)
    _mqtt_connect()

    # run inference in background thread
    threading.Thread(target=inference_loop, args=(args,), daemon=True).start()

    # run FastAPI in main thread (captures the event loop)
    config = uvicorn.Config(app, host="0.0.0.0", port=args.web_port, log_level="warning")
    server = uvicorn.Server(config)

    async def serve():
        global _loop
        _loop = asyncio.get_running_loop()
        await server.serve()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
