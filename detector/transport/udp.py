"""UDP receiver: accepts SDR1 packets from ESP32 / Hub."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from ..audio import AudioRingBuffer, ClipBuffer, i2s32_to_float32
from ..bus import EventBus
from ..config import HEADER, MAGIC, PACKET_MAX


# ── UDP packet stats ─────────────────────────────────────────────────────────

_udp_stats_lock = threading.Lock()
_udp_last_packet_time: float = 0.0
_udp_packet_count: int = 0
_udp_packet_count_prev: int = 0
_udp_pps: float = 0.0
_udp_pps_last_calc: float = 0.0


def _udp_stats_update() -> None:
    global _udp_last_packet_time, _udp_packet_count
    with _udp_stats_lock:
        _udp_last_packet_time = time.monotonic()
        _udp_packet_count += 1


def udp_stats_snapshot() -> dict:
    """Return current UDP stats for broadcasting to UI."""
    global _udp_pps, _udp_pps_last_calc, _udp_packet_count_prev
    now = time.monotonic()
    with _udp_stats_lock:
        last = _udp_last_packet_time
        count = _udp_packet_count
    dt = now - _udp_pps_last_calc
    if dt >= 2.0:
        _udp_pps = (count - _udp_packet_count_prev) / dt
        _udp_packet_count_prev = count
        _udp_pps_last_calc = now
    ago = round(now - last, 1) if last > 0 else -1
    alive = last > 0 and (now - last) < 5.0
    return {"audio_alive": alive, "audio_ago": ago, "audio_pps": round(_udp_pps, 1)}


def udp_receiver(
    inference_buf: AudioRingBuffer,
    clip_buf: ClipBuffer,
    bus: EventBus,
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
        bus.broadcast_audio(samples)
        _udp_stats_update()
        last_recv = time.monotonic()
        expected_seq = seq + 1
    sock.close()
