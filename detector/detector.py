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

SIREN_THRESHOLD = 0.10
SMOOTHING_WINDOW_SECONDS = 30
POSSIBLE_TRIGGERS = 4
CONFIRMED_TRIGGERS = 8
COOLDOWN_SECONDS = 60

CLIP_SECONDS = 30       # audio saved before confirmed event
CLIP_DIR = Path("/clips")
WEB_PORT = 8080


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

_ws_clients: set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None
_event_history: collections.deque = collections.deque(maxlen=100)


async def _broadcast(msg: dict) -> None:
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients -= dead


def broadcast(msg: dict) -> None:
    if msg.get("type") == "event":
        _event_history.append(msg)
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _loop)


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI()
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    for ev in list(_event_history):
        ev_out = dict(ev)
        filename = (ev.get("clip_url") or "").split("/")[-1]
        if filename in _feedback:
            ev_out["label"] = _feedback[filename]
        await ws.send_json(ev_out)
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)


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
        with open(yamnet.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
            class_names = [row["display_name"] for row in csv.DictReader(f)]
        siren_index = next(i for i, n in enumerate(class_names) if n == "Siren")
        print("No classifier weights — using raw YAMNet siren score.", flush=True)

    inference_buf = AudioRingBuffer(capacity=int(ANALYSIS_SECONDS * SAMPLE_RATE))
    clip_buf = ClipBuffer(seconds=CLIP_SECONDS)
    smoother = TemporalSmoother(
        window_seconds=SMOOTHING_WINDOW_SECONDS,
        hop_seconds=0.5,
        threshold=SIREN_THRESHOLD,
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

    while True:
        waveform = inference_buf.consume(INFERENCE_HOP_SAMPLES)
        if waveform is None:
            time.sleep(0.05)
            continue

        waveform = np.clip(waveform, -1.0, 1.0)
        scores, embeddings, _ = yamnet(waveform)

        if clf is not None:
            probs = clf.predict(embeddings.numpy())
            siren_idx = clf.labels.index("siren")
            max_score = float(probs[:, siren_idx].max())
        else:
            max_score = float(scores.numpy()[:, siren_index].max())

        new_state = smoother.push(max_score)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # push score to UI every cycle
        broadcast({"type": "score", "score": round(max_score, 3), "state": smoother.state})

        if new_state in ("possible", "confirmed"):
            clip_url = None
            if new_state == "confirmed":
                clip_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.wav"
                clip_buf.save(CLIP_DIR / clip_name)
                clip_url = f"/clips/{clip_name}"
                print(f"[{ts}] Clip saved: {clip_url}", flush=True)

            event = {
                "type": "event",
                "event": "siren",
                "state": new_state,
                "confidence": round(max_score, 3),
                "timestamp": ts,
                "source": args.source,
                "engine": engine,
                "clip_url": clip_url,
            }
            broadcast(event)
            print(f"EVENT {json.dumps(event)}", flush=True)


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
