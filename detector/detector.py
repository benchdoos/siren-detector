#!/usr/bin/env python3
"""
Streaming siren detector.

Receives UDP audio from ESP32 (same packet format as recorder),
runs YAMNet every 0.5 s, applies temporal smoothing,
and prints JSON events to stdout.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import socket
import struct
import threading
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow_hub as hub


SAMPLE_RATE = 16_000
PACKET_MAX = 4096 + 12          # header(12) + max payload
HEADER = struct.Struct("!IIHH") # magic, seq, payload_size, reserved
MAGIC = 0x53445231              # "SDR1"

MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# How much audio we feed to YAMNet each call (gives ~4 windows at 0.48s hop)
ANALYSIS_SECONDS = 2.0
# How many new samples we wait for before running inference
INFERENCE_HOP_SAMPLES = SAMPLE_RATE // 2  # 0.5 s

# Detection thresholds
SIREN_THRESHOLD = 0.10
SMOOTHING_WINDOW_SECONDS = 30
POSSIBLE_TRIGGERS = 2   # → "possible" state
CONFIRMED_TRIGGERS = 4  # → fires an event
COOLDOWN_SECONDS = 60   # silence after firing


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
        """Return snapshot when >= hop new samples arrived, else None."""
        with self._lock:
            if self._pending < hop:
                return None
            self._pending = 0
            return np.array(self._buf, dtype=np.float32)


class TemporalSmoother:
    """Sliding-window trigger counter with cooldown."""

    def __init__(
        self,
        window_seconds: float,
        hop_seconds: float,
        threshold: float,
        possible: int,
        confirmed: int,
        cooldown_seconds: float,
    ) -> None:
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
        """Add score; return new state name if state changed, else None."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        # Coming out of cooldown
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


def udp_receiver(buf: AudioRingBuffer, bind: str, port: int, stop: threading.Event) -> None:
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
            buf.push(np.zeros(payload_size // 4 * gap, dtype=np.float32))
        payload = data[HEADER.size : HEADER.size + payload_size]
        buf.push(i2s32_to_float32(payload))
        expected_seq = seq + 1
    sock.close()


def run(args: argparse.Namespace) -> None:
    print(f"Loading YAMNet from {MODEL_URL} ...", flush=True)
    model = hub.load(MODEL_URL)
    with open(model.class_map_path().numpy().decode(), newline="", encoding="utf-8") as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    siren_index = next(i for i, n in enumerate(class_names) if n == "Siren")
    print("YAMNet ready.", flush=True)

    buf = AudioRingBuffer(capacity=int(ANALYSIS_SECONDS * SAMPLE_RATE))
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
        target=udp_receiver, args=(buf, args.bind, args.port, stop), daemon=True
    ).start()

    while True:
        waveform = buf.consume(INFERENCE_HOP_SAMPLES)
        if waveform is None:
            time.sleep(0.05)
            continue

        scores, _, _ = model(np.clip(waveform, -1.0, 1.0))
        max_score = float(scores.numpy()[:, siren_index].max())

        new_state = smoother.push(max_score)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if max_score >= SIREN_THRESHOLD:
            print(
                f"[{ts}] score={max_score:.3f}  state={smoother.state}",
                flush=True,
            )

        if new_state in ("possible", "confirmed"):
            event = {
                "event": "siren",
                "state": new_state,
                "confidence": round(max_score, 3),
                "timestamp": ts,
                "source": args.source,
                "engine": "yamnet",
            }
            print(f"EVENT {json.dumps(event)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--source", default="siren-detector-01")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
