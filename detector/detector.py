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
import json
import os
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

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

CLIP_SECONDS = 30       # audio saved before confirmed event
CLIP_DIR = Path("/clips")
WEB_PORT = 8080

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
    if state == "confirmed":
        _mqtt_client.publish(f"sound_detector/{name}/state", "ON", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", str(round(confidence, 3)), retain=True)
    elif state == "idle":
        _mqtt_client.publish(f"sound_detector/{name}/state", "OFF", retain=True)
        _mqtt_client.publish(f"sound_detector/{name}/confidence", "0", retain=True)


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

    def save(self, path: Path) -> None:
        with self._lock:
            audio = np.array(self._buf, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


# ── detection ──────────────────────────────────────────────────────────────────

class TemporalSmoother:
    def __init__(self, window_seconds, hop_seconds, threshold, possible, confirmed, cooldown_seconds):
        maxlen = int(window_seconds / hop_seconds) + 1
        self._scores: collections.deque = collections.deque(maxlen=maxlen)
        self._threshold = threshold
        self._possible = possible
        self._confirmed = confirmed
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until = 0.0
        self._state = "idle"

    @property
    def state(self) -> str:
        return self._state

    def push(self, score: float) -> str | None:
        now = time.monotonic()
        if now < self._cooldown_until:
            return None
        if self._state == "confirmed":
            self._state = "idle"
        prev = self._state
        self._scores.append(score)
        triggers = sum(1 for s in self._scores if s >= self._threshold)
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
_loop: asyncio.AbstractEventLoop | None = None
_event_history: collections.deque = collections.deque(maxlen=100)
_last_score: dict = {"type": "score", "top": ""}


def _enqueue_all(msg: dict) -> None:
    for q in _client_queues.values():
        q.put_nowait(msg)


def broadcast(msg: dict) -> None:
    global _last_score
    if msg.get("type") == "event":
        _event_history.append(msg)
    elif msg.get("type") == "score":
        _last_score = msg
    if _loop and not _loop.is_closed():
        _loop.call_soon_threadsafe(_enqueue_all, msg)


# ── FastAPI app ────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(application: FastAPI):
    await _restore_clips()
    yield


app = FastAPI(lifespan=_lifespan)
_static = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


@app.get("/clips/{filename}")
async def serve_clip(filename: str):
    path = CLIP_DIR / filename
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(str(path), media_type="audio/wav")


DATASET_DIR = Path("/dataset")
VALID_LABELS = {"siren", "background", "motorcycle", "explosion", "unknown"}
_feedback: dict[str, str] = {}  # filename → label


@app.get("/api/metrics")
async def get_metrics():
    metrics_path = Path("/models/val_metrics.json")
    if not metrics_path.exists():
        return {"error": "no metrics available"}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/feedback")
async def feedback(body: dict):
    from fastapi import HTTPException
    import shutil
    filename = body.get("filename", "")
    label = body.get("label", "")
    if not filename or label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail="Invalid filename or label")
    src = CLIP_DIR / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    dst_dir = DATASET_DIR / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_dir / filename))
    _feedback[filename] = label
    print(f"[feedback] {filename} → {label}", flush=True)
    return {"ok": True}


async def _restore_clips():
    """Scan /clips on startup to restore event history from surviving WAV files."""
    if not CLIP_DIR.exists():
        return
    wavs = sorted(CLIP_DIR.glob("*.wav"))
    for path in wavs:
        name = path.name
        # detect event type from prefix
        if name.startswith("exp-"):
            event_type = "explosion"
            date_part = name[4:].replace(".wav", "")
        elif name.startswith("moto-"):
            event_type = "motorcycle"
            date_part = name[5:].replace(".wav", "")
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
            "confidence": 0.0, "timestamp": ts,
            "source": "restored", "engine": "unknown", "clip_url": f"/clips/{name}",
            "yamnet": 0.0, "mlp": None, "yamnet_top": "",
        }
        _event_history.append(ev)
    if wavs:
        print(f"Restored {len(wavs)} clips from /clips/", flush=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json(_last_score)
    for ev in list(_event_history):
        clip_url = ev.get("clip_url")
        if not clip_url:
            continue
        filename = clip_url.split("/")[-1]
        if not (CLIP_DIR / filename).exists():
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
    print(f"Listening on {bind}:{port}", flush=True)
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(PACKET_MAX)
        except socket.timeout:
            continue
        if len(data) < HEADER.size:
            continue
        magic, seq, payload_size, _ = HEADER.unpack_from(data)
        if magic != MAGIC or payload_size > len(data) - HEADER.size:
            continue
        if expected_seq is None:
            expected_seq = seq
        if seq < expected_seq:
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
        expected_seq = seq + 1
    sock.close()


# ── inference loop ─────────────────────────────────────────────────────────────

def inference_loop(args: argparse.Namespace) -> None:
    global _loop

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
    clip_buf = ClipBuffer(seconds=CLIP_SECONDS)
    smoothers: dict[str, TemporalSmoother] = {}
    for name, (_, threshold, _) in DETECTORS.items():
        smoothers[name] = TemporalSmoother(
            window_seconds=SMOOTHING_WINDOW_SECONDS,
            hop_seconds=0.5,
            threshold=threshold,
            possible=POSSIBLE_TRIGGERS,
            confirmed=CONFIRMED_TRIGGERS,
            cooldown_seconds=COOLDOWN_SECONDS,
        )

    stop = threading.Event()
    threading.Thread(
        target=udp_receiver,
        args=(inference_buf, clip_buf, args.bind, args.port, stop),
        daemon=True,
    ).start()

    print("Ready.", flush=True)

    _last_heartbeat = 0.0
    peaks: dict[str, dict] = {name: {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": ""} for name in DETECTORS}
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
        if clf is not None and "siren" in clf.labels:
            probs = clf.predict(embeddings.numpy())
            siren_idx = clf.labels.index("siren")
            mlp_score = float(probs[:, siren_idx].max())

        # effective score: MLP for siren (if available), YAMNet for the rest
        # gate: if YAMNet sees no siren at all, don't trust MLP alone
        effective: dict[str, float] = {}
        for name in DETECTORS:
            if name == "siren" and mlp_score is not None:
                if yamnet_scores["siren"] >= YAMNET_MIN_GATE:
                    effective[name] = mlp_score
                else:
                    effective[name] = 0.0
            else:
                effective[name] = yamnet_scores[name]

        # push through smoothers
        state_changes: dict[str, str | None] = {}
        for name, smoother in smoothers.items():
            state_changes[name] = smoother.push(effective[name])

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # track peaks per detector
        for name in DETECTORS:
            if smoothers[name].state == "idle":
                peaks[name] = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": ""}
            elif effective[name] > peaks[name]["score"]:
                peaks[name] = {
                    "score": effective[name],
                    "yamnet": yamnet_scores[name],
                    "mlp": mlp_score if name == "siren" else None,
                    "top": yamnet_top,
                }

        # push scores to UI
        score_msg: dict = {"type": "score", "top": yamnet_top}
        for name in DETECTORS:
            score_msg[f"{name}_score"] = round(effective[name], 3)
            score_msg[f"{name}_state"] = smoothers[name].state
        score_msg["yamnet"] = round(yamnet_scores.get("siren", 0.0), 3)
        score_msg["mlp"] = round(mlp_score, 3) if mlp_score is not None else None

        # top-3 YAMNet classes
        mean_scores = scores_np.mean(axis=0)
        top3_idx = mean_scores.argsort()[::-1][:3]
        score_msg["yamnet_top3"] = [
            {"name": class_names[i], "score": round(float(mean_scores[i]), 3)}
            for i in top3_idx
        ]

        # top-3 MLP classes
        if clf is not None:
            mean_probs = probs.mean(axis=0) if mlp_score is not None else clf.predict(embeddings.numpy()).mean(axis=0)
            mlp_top3_idx = mean_probs.argsort()[::-1][:3]
            score_msg["mlp_top3"] = [
                {"name": clf.labels[i], "score": round(float(mean_probs[i]), 3)}
                for i in mlp_top3_idx
            ]

        broadcast(score_msg)
        _last_heartbeat = time.monotonic()

        # emit events for each detector
        for name, (_, _, prefix) in DETECTORS.items():
            new_state = state_changes[name]
            if new_state not in ("possible", "confirmed"):
                continue
            clip_url = None
            if new_state == "confirmed":
                clip_name = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                clip_buf.save(CLIP_DIR / clip_name)
                clip_url = f"/clips/{clip_name}"
                print(f"[{ts}] {name} clip saved: {clip_url}", flush=True)

            pk = peaks[name]
            det_engine = engine if (name == "siren" and clf is not None) else "yamnet"
            event = {
                "type": "event", "event": name, "state": new_state,
                "confidence": round(pk["score"], 3), "timestamp": ts,
                "source": args.source, "engine": det_engine, "clip_url": clip_url,
                "yamnet": round(pk["yamnet"], 3),
                "mlp": round(pk["mlp"], 3) if pk["mlp"] is not None else None,
                "yamnet_top": pk["top"],
            }
            broadcast(event)
            mqtt_publish_event(name, new_state, pk["score"])
            print(f"EVENT {json.dumps(event)}", flush=True)
            if new_state == "confirmed":
                peaks[name] = {"score": 0.0, "yamnet": 0.0, "mlp": None, "top": ""}


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
