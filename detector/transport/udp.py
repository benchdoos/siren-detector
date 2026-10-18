"""UDP receiver: accepts SDR1 packets from ESP32 / Hub, demuxes by source_id."""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable

import numpy as np

from ..audio import AudioRingBuffer, ClipBuffer, i2s32_to_float32
from ..bus import EventBus
from ..config import HEADER, MAGIC, PACKET_MAX


# ── Per-source tracking ─────────────────────────────────────────────────────

class _SourceTracker:
    """Sequence and stats tracking for one audio source."""

    __slots__ = ("expected_seq", "last_recv", "packet_count",
                 "packet_count_prev", "pps", "pps_last_calc")

    def __init__(self) -> None:
        self.expected_seq: int | None = None
        self.last_recv: float = 0.0
        self.packet_count: int = 0
        self.packet_count_prev: int = 0
        self.pps: float = 0.0
        self.pps_last_calc: float = 0.0


_stats_lock = threading.Lock()
_source_trackers: dict[int, _SourceTracker] = {}


def _get_tracker(source_id: int) -> _SourceTracker:
    if source_id not in _source_trackers:
        _source_trackers[source_id] = _SourceTracker()
    return _source_trackers[source_id]


def udp_stats_snapshot(source_names: dict[int, str] | None = None) -> dict:
    """Return aggregated + per-source UDP stats for broadcasting to UI."""
    now = time.monotonic()
    with _stats_lock:
        trackers = dict(_source_trackers)

    if not trackers:
        return {"audio_alive": False, "audio_ago": -1, "audio_pps": 0.0}

    names = source_names or {}
    total_pps = 0.0
    best_ago = -1.0
    any_alive = False
    per_source: dict[int, dict] = {}

    for sid, t in trackers.items():
        dt = now - t.pps_last_calc
        if dt >= 2.0:
            t.pps = (t.packet_count - t.packet_count_prev) / dt
            t.packet_count_prev = t.packet_count
            t.pps_last_calc = now
        ago = round(now - t.last_recv, 1) if t.last_recv > 0 else -1
        alive = t.last_recv > 0 and (now - t.last_recv) < 5.0
        total_pps += t.pps
        if alive:
            any_alive = True
        if ago >= 0 and (best_ago < 0 or ago < best_ago):
            best_ago = ago
        per_source[sid] = {
            "name": names.get(sid, f"source-{sid}"),
            "alive": alive, "ago": ago, "pps": round(t.pps, 1),
            "packets": t.packet_count,
        }

    result: dict = {
        "audio_alive": any_alive,
        "audio_ago": best_ago,
        "audio_pps": round(total_pps, 1),
    }
    if len(per_source) > 1:
        result["audio_sources"] = per_source
    return result


# ── UDP receiver ─────────────────────────────────────────────────────────────

def udp_receiver(
    get_buffers: Callable[[int], tuple[AudioRingBuffer, ClipBuffer] | None],
    bus: EventBus,
    bind: str,
    port: int,
    stop: threading.Event,
) -> None:
    """Receive SDR1 packets, demux by source_id, push to per-source buffers.

    ``get_buffers(source_id)`` returns ``(inference_buf, clip_buf)`` for a
    known source, or creates a new source on-the-fly.  Return ``None`` to
    reject unknown sources.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))
    sock.settimeout(1.0)
    print(f"Listening on {bind}:{port}", flush=True)

    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(PACKET_MAX)
        except socket.timeout:
            # Check for stale sources
            now = time.monotonic()
            with _stats_lock:
                for sid, t in _source_trackers.items():
                    if (t.expected_seq is not None
                            and t.last_recv > 0
                            and now - t.last_recv > 5.0):
                        print(f"UDP: source {sid} no packets for 5s, resetting sequence tracking", flush=True)
                        t.expected_seq = None
            continue

        if len(data) < HEADER.size:
            continue
        magic, seq, payload_size, source_id = HEADER.unpack_from(data)
        if magic != MAGIC or payload_size > len(data) - HEADER.size:
            continue

        bufs = get_buffers(source_id)
        if bufs is None:
            continue
        inference_buf, clip_buf = bufs

        with _stats_lock:
            tracker = _get_tracker(source_id)

        if tracker.expected_seq is None:
            tracker.expected_seq = seq
            print(f"UDP: source {source_id} connected, seq={seq}", flush=True)

        if seq < tracker.expected_seq:
            if tracker.expected_seq - seq > 100:
                print(f"UDP: source {source_id} seq reset ({tracker.expected_seq} → {seq})", flush=True)
                tracker.expected_seq = seq
            else:
                continue

        gap = seq - tracker.expected_seq
        if gap:
            silence = np.zeros(payload_size // 4 * gap, dtype=np.float32)
            inference_buf.push(silence)
            clip_buf.push(silence)

        payload = data[HEADER.size : HEADER.size + payload_size]
        samples = i2s32_to_float32(payload)
        inference_buf.push(samples)
        clip_buf.push(samples)
        bus.broadcast_audio(samples, source_id)

        with _stats_lock:
            tracker.last_recv = time.monotonic()
            tracker.packet_count += 1
        tracker.expected_seq = seq + 1

    sock.close()
