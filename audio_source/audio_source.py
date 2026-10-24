#!/usr/bin/env python3
"""Capture audio from a local microphone and stream SDR1 packets via UDP.

Designed for USB microphones on Raspberry Pi (or any host with PortAudio).
Captures at the device's native sample rate, resamples to 16 kHz, and packs
into SDR1 frames identical to those sent by the ESP32 firmware.

Usage:
    python audio_source.py --device default --source-id 2 --target 127.0.0.1:9000

Or via Docker (see Dockerfile in this directory).
"""

from __future__ import annotations

import argparse
import queue
import socket
import struct
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd

# SDR1 protocol constants (must match detector/config.py)
MAGIC = 0x53445231  # "SDR1"
HEADER = struct.Struct("!IIHH")  # magic(4) + seq(4) + payload_size(2) + source_id(2)

TARGET_RATE = 16_000
CHUNK_DURATION = 0.032  # ~32ms per packet (matches ESP32 default)


def list_devices() -> None:
    """Print available audio devices and exit."""
    print(sd.query_devices())


def _query_native_rate(device) -> int:
    """Get the device's default sample rate."""
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


def _build_resampler(native_rate: int, target_rate: int):
    """Return a resampling function (native_rate → target_rate).

    Uses simple linear interpolation — good enough for 48→16 kHz speech/audio.
    Returns None if rates already match.
    """
    if native_rate == target_rate:
        return None
    ratio = target_rate / native_rate

    def resample(audio: np.ndarray) -> np.ndarray:
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        idx_floor = indices.astype(np.intp)
        frac = (indices - idx_floor).astype(np.float32)
        idx_floor = np.clip(idx_floor, 0, len(audio) - 1)
        idx_ceil = np.clip(idx_floor + 1, 0, len(audio) - 1)
        return audio[idx_floor] * (1.0 - frac) + audio[idx_ceil] * frac

    return resample


def run(
    device: int | str | None,
    source_id: int,
    target_host: str,
    target_port: int,
    gain: float,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # UDP packets are disposable audio frames.  Never let a full kernel
    # transmit buffer stall the capture pipeline; the sender drops that frame
    # and immediately continues with the next one.
    sock.setblocking(False)
    target = (target_host, target_port)

    native_rate = _query_native_rate(device)
    resample = _build_resampler(native_rate, TARGET_RATE)
    chunk_samples = int(CHUNK_DURATION * native_rate)

    print(f"Audio source: device={device}, source_id={source_id}, "
          f"target={target_host}:{target_port}, gain={gain}", flush=True)
    print(f"Device rate: {native_rate} Hz → target: {TARGET_RATE} Hz, "
          f"chunk: {chunk_samples} samples", flush=True)

    # Keep PortAudio's real-time callback small. Processing, UDP I/O and even
    # writing to Docker's log driver can take longer than one audio block.
    audio_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=32)
    input_overflows = 0
    queue_drops = 0
    send_drops = 0
    metrics_stop = threading.Event()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal input_overflows, queue_drops
        if status:
            # This is PortAudio's real-time callback.  Do not log here: a
            # synchronous write to Docker logs can itself cause more input
            # overflows.  The reporter below emits an aggregate instead.
            input_overflows += 1
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.ravel().copy()
        try:
            audio_q.put_nowait(audio)
        except queue.Full:
            # Fresh capture takes priority over waiting.  Dropping this block
            # keeps the PortAudio callback within its deadline.
            queue_drops += 1

    def sender() -> None:
        nonlocal send_drops
        seq = 0
        while True:
            chunk = audio_q.get()
            if chunk is None:
                return
            try:
                audio = chunk.astype(np.float32) * gain
                if resample is not None:
                    audio = resample(audio)
                i32 = (audio * 32768.0).clip(-32768, 32767).astype(np.int32) << 16
                payload = i32.tobytes()
                header = HEADER.pack(MAGIC, seq, len(payload), source_id)
                sock.sendto(header + payload, target)
            except (OSError, ValueError, OverflowError):
                # UDP either accepts the complete datagram or rejects it.
                # Skip a rejected/bad chunk and let the receiver reconstruct
                # the sequence gap as silence.
                send_drops += 1
            except Exception:
                # A malformed block must not silently kill this consumer: if
                # it died, capture would only fill and discard the queue.
                send_drops += 1
            finally:
                # Advance even when the packet was dropped.  Reusing its
                # sequence number would conceal the gap at the receiver.
                seq = (seq + 1) & 0xFFFFFFFF

    def report_metrics() -> None:
        """Log capture health outside PortAudio's real-time callback."""
        previous = (0, 0, 0)
        while not metrics_stop.wait(5.0):
            current = (input_overflows, queue_drops, send_drops)
            delta = tuple(now - then for now, then in zip(current, previous))
            previous = current
            if any(delta):
                print(
                    "Audio drops in last 5s: "
                    f"input_overflow={delta[0]}, queue_full={delta[1]}, "
                    f"send_failed={delta[2]}",
                    file=sys.stderr,
                    flush=True,
                )

    sender_thread = threading.Thread(target=sender, name="audio-sender", daemon=True)
    sender_thread.start()
    metrics_thread = threading.Thread(target=report_metrics, name="audio-metrics", daemon=True)
    metrics_thread.start()

    try:
        while True:
            try:
                with sd.InputStream(
                    device=device,
                    samplerate=native_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=chunk_samples,
                    latency="high",
                    callback=audio_callback,
                ) as stream:
                    print("Streaming... (Ctrl+C to stop)", flush=True)
                    while stream.active:
                        time.sleep(1.0)
                    print("Audio stream stopped; reconnecting in 2s...", file=sys.stderr, flush=True)
            except sd.PortAudioError as exc:
                print(f"Audio error: {exc}; reconnecting in 2s...", file=sys.stderr, flush=True)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        metrics_stop.set()
        # Make room for the shutdown marker so shutdown cannot hang when the
        # consumer has stopped and the bounded queue is full.
        while True:
            try:
                audio_q.put_nowait(None)
                break
            except queue.Full:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    pass
        sender_thread.join(timeout=2.0)
        metrics_thread.join(timeout=2.0)
        sock.close()


def record(
    device: int | str | None,
    output_path: str,
    duration: int | None,
    gain: float,
) -> None:
    """Record audio from microphone to WAV file (16 kHz, 16-bit mono)."""
    native_rate = _query_native_rate(device)
    resample = _build_resampler(native_rate, TARGET_RATE)
    chunk_samples = int(CHUNK_DURATION * native_rate)

    wf = wave.open(output_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(TARGET_RATE)

    total_samples = 0
    start = time.monotonic()

    print(f"Recording: device={device}, output={output_path}, gain={gain}", flush=True)
    print(f"Device rate: {native_rate} Hz → WAV: {TARGET_RATE} Hz", flush=True)
    if duration:
        print(f"Duration: {duration}s (Ctrl+C to stop early)", flush=True)
    else:
        print("Duration: unlimited (Ctrl+C to stop)", flush=True)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal total_samples
        if status:
            print(f"Audio status: {status}", file=sys.stderr, flush=True)

        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        audio = audio.astype(np.float32) * gain

        if resample is not None:
            audio = resample(audio)

        pcm16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
        total_samples += len(pcm16)

    try:
        with sd.InputStream(
            device=device,
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            latency="high",
            callback=audio_callback,
        ):
            while True:
                time.sleep(1.0)
                elapsed = time.monotonic() - start
                if duration and elapsed >= duration:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        wf.close()
        secs = total_samples / TARGET_RATE
        print(f"\nSaved {output_path} ({secs:.1f}s, {total_samples} samples)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream audio from a local microphone as SDR1 packets.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Audio device (index or name substring). Default: system default.",
    )
    parser.add_argument(
        "--source-id", type=int, default=1,
        help="SDR1 source_id (uint16). Default: 1. Use 0 for ESP32 compat.",
    )
    parser.add_argument(
        "--target", default="127.0.0.1:9000",
        help="Target host:port for UDP packets. Default: 127.0.0.1:9000",
    )
    parser.add_argument(
        "--gain", type=float, default=1.0,
        help="Audio gain multiplier. Default: 1.0",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--record", metavar="FILE",
        help="Record to WAV file instead of streaming UDP.",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Recording duration in seconds (with --record). Default: unlimited.",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Parse device: "default" → None, integer index, or string name
    device = args.device
    if device is None or device == "default":
        device = None
    else:
        try:
            device = int(device)
        except ValueError:
            pass  # Keep as string (name substring match)

    if args.record:
        record(device, args.record, args.duration, args.gain)
        return

    host, _, port_str = args.target.rpartition(":")
    target_host = host or "127.0.0.1"
    target_port = int(port_str) if port_str else 9000

    run(device, args.source_id, target_host, target_port, args.gain)


if __name__ == "__main__":
    main()
