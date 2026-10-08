#!/usr/bin/env python3
"""Receive raw ESPHome INMP441 UDP packets and save them as a WAV file."""

from __future__ import annotations

import argparse
import socket
import struct
import time
import wave
from datetime import datetime
from pathlib import Path


SAMPLE_RATE = 16_000
PACKET_SIZE = 4096
HEADER = struct.Struct("!IIHH")
MAGIC = 0x53445231  # "SDR1"
LABELS = ("siren", "explosion", "motorcycle", "background", "unknown", "inbox")


def convert_i2s_32_to_pcm16(data: bytes) -> bytes:
    """Convert signed 32-bit I2S slots to signed 16-bit PCM."""
    usable_size = len(data) - (len(data) % 4)
    samples = struct.unpack(f"<{usable_size // 4}i", data[:usable_size])
    pcm16 = bytearray()
    for sample in samples:
        pcm16.extend(struct.pack("<h", max(-32768, min(32767, sample >> 16))))
    return bytes(pcm16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--label", choices=LABELS, default="inbox")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="recording length in seconds; 0 records until Ctrl-C",
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("--duration must be 0 or greater")

    output_dir = args.output_dir / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.wav"

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind((args.bind, args.port))
    udp.settimeout(5.0)

    started_at = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    print(f"Started at {started_at.isoformat(timespec='seconds')}")
    print(f"Label: {args.label}")
    print(f"Duration: {args.duration}s" if args.duration else "Duration: until Ctrl-C")
    print(f"Listening on {args.bind}:{args.port}")
    print(f"Writing {output_path}")
    print("Press Ctrl-C to stop and close the WAV file.")

    packets = 0
    dropped_packets = 0
    invalid_packets = 0
    expected_sequence = None
    last_sequence = None
    frames = 0
    stop_reason = "Ctrl-C"
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        try:
            while True:
                elapsed = time.monotonic() - started_monotonic
                if args.duration and elapsed >= args.duration:
                    stop_reason = "duration reached"
                    break
                timeout = 5.0
                if args.duration:
                    timeout = min(timeout, max(0.1, args.duration - elapsed))
                udp.settimeout(timeout)
                try:
                    data, address = udp.recvfrom(PACKET_SIZE + HEADER.size)
                except socket.timeout:
                    elapsed = time.monotonic() - started_monotonic
                    if args.duration and elapsed >= args.duration:
                        stop_reason = "duration reached"
                        break
                    print(
                        f"No packets for 5s; wall_seconds={elapsed:.1f}, "
                        f"packets={packets}, last_seq={last_sequence}"
                    )
                    continue
                if len(data) < HEADER.size:
                    invalid_packets += 1
                    print(f"Ignoring short UDP packet: bytes={len(data)}")
                    continue

                magic, sequence, payload_size, _ = HEADER.unpack_from(data)
                if magic != MAGIC or payload_size > len(data) - HEADER.size:
                    invalid_packets += 1
                    print(
                        f"Ignoring invalid UDP packet: bytes={len(data)}, "
                        f"magic=0x{magic:08x}, payload={payload_size}"
                    )
                    continue

                if expected_sequence is None:
                    expected_sequence = sequence
                if sequence < expected_sequence:
                    continue

                missing = sequence - expected_sequence
                if missing:
                    dropped_packets += missing
                    missing_payload = payload_size * missing
                    wav.writeframes(b"\x00" * (missing_payload // 2))
                    frames += missing_payload // 4

                payload = data[HEADER.size:HEADER.size + payload_size]
                pcm16 = convert_i2s_32_to_pcm16(payload)
                wav.writeframes(pcm16)
                packets += 1
                frames += len(pcm16) // 2
                expected_sequence = sequence + 1
                last_sequence = sequence
                if packets % 100 == 0:
                    elapsed = time.monotonic() - started_monotonic
                    print(
                        f"wall_seconds={elapsed:.1f}, packets={packets}, "
                        f"audio_seconds={frames / SAMPLE_RATE:.1f}, "
                        f"dropped={dropped_packets}, invalid={invalid_packets}, "
                        f"last_seq={last_sequence}, source={address[0]}"
                    )
                if args.duration and time.monotonic() - started_monotonic >= args.duration:
                    stop_reason = "duration reached"
                    break
        except KeyboardInterrupt:
            pass
        finally:
            stopped_at = datetime.now().astimezone()
            elapsed = time.monotonic() - started_monotonic
            print(f"Stopped at {stopped_at.isoformat(timespec='seconds')} ({stop_reason})")
            print(f"Wall-clock duration: {elapsed:.1f}s")
            print(
                f"Saved {frames / SAMPLE_RATE:.1f}s to {output_path}; "
                f"dropped packets: {dropped_packets}; invalid packets: {invalid_packets}"
            )
            udp.close()


if __name__ == "__main__":
    main()
