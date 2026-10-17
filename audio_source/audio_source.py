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
import socket
import struct
import sys
import time

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
    target = (target_host, target_port)
    seq = 0

    native_rate = _query_native_rate(device)
    resample = _build_resampler(native_rate, TARGET_RATE)
    chunk_samples = int(CHUNK_DURATION * native_rate)

    print(f"Audio source: device={device}, source_id={source_id}, "
          f"target={target_host}:{target_port}, gain={gain}", flush=True)
    print(f"Device rate: {native_rate} Hz → target: {TARGET_RATE} Hz, "
          f"chunk: {chunk_samples} samples", flush=True)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal seq
        if status:
            print(f"Audio status: {status}", file=sys.stderr, flush=True)

        # Convert to mono float32
        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        audio = audio.astype(np.float32) * gain

        # Resample to 16 kHz if needed
        if resample is not None:
            audio = resample(audio)

        # Convert to I2S-like 32-bit int (matches INMP441 format)
        # The detector's i2s32_to_float32() does: (i32 >> 16) / 32768.0
        # So we need: float → int32 with data in upper 16 bits
        i32 = (audio * 32768.0).clip(-32768, 32767).astype(np.int32) << 16

        payload = i32.tobytes()
        header = HEADER.pack(MAGIC, seq, len(payload), source_id)
        sock.sendto(header + payload, target)
        seq = (seq + 1) & 0xFFFFFFFF

    try:
        with sd.InputStream(
            device=device,
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_callback,
        ):
            print("Streaming... (Ctrl+C to stop)", flush=True)
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except sd.PortAudioError as exc:
        print(f"Audio error: {exc}", file=sys.stderr, flush=True)
        print("\nAvailable devices:", file=sys.stderr, flush=True)
        list_devices()
        sys.exit(1)
    finally:
        sock.close()


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
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    host, _, port_str = args.target.rpartition(":")
    target_host = host or "127.0.0.1"
    target_port = int(port_str) if port_str else 9000

    # Parse device: "default" → None, integer index, or string name
    device = args.device
    if device is None or device == "default":
        device = None
    else:
        try:
            device = int(device)
        except ValueError:
            pass  # Keep as string (name substring match)

    run(device, args.source_id, target_host, target_port, args.gain)


if __name__ == "__main__":
    main()
