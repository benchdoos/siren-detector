#!/usr/bin/env python3
"""
UDP fan-out hub.

Receives audio packets from ESP32 on one port and forwards every packet
to all registered subscribers. Subscribers are configured statically via
--forward host:port (repeatable). Dynamic registration is a future extension.

Usage:
  python hub.py --port 9000 --forward 127.0.0.1:9001 --forward 127.0.0.1:9002
"""

from __future__ import annotations

import argparse
import socket
import time

PACKET_MAX = 4096 + 12  # SDR1 header(12) + max payload


def parse_addr(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    return host or "127.0.0.1", int(port)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--forward",
        dest="forwards",
        action="append",
        default=[],
        metavar="HOST:PORT",
        help="Forward packets to this address (repeatable).",
    )
    args = parser.parse_args()

    if not args.forwards:
        parser.error("Specify at least one --forward HOST:PORT")

    subscribers = [parse_addr(f) for f in args.forwards]

    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in.bind((args.bind, args.port))

    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Hub listening on {args.bind}:{args.port}", flush=True)
    for addr in subscribers:
        print(f"  → forwarding to {addr[0]}:{addr[1]}", flush=True)

    packets = 0
    last_report = time.monotonic()

    while True:
        data, source = sock_in.recvfrom(PACKET_MAX)
        for addr in subscribers:
            sock_out.sendto(data, addr)
        packets += 1
        now = time.monotonic()
        if now - last_report >= 60:
            print(f"packets_forwarded={packets}  source={source[0]}", flush=True)
            last_report = now


if __name__ == "__main__":
    main()
