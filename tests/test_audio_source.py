"""Fast tests for the standalone microphone audio source."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sounddevice")


_AUDIO_SOURCE_PATH = Path(__file__).parents[1] / "audio_source" / "audio_source.py"
_SPEC = importlib.util.spec_from_file_location("audio_source", _AUDIO_SOURCE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
audio_source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audio_source)


def test_failed_udp_chunk_is_skipped_and_next_chunk_is_sent(monkeypatch):
    """One UDP send error must not kill the sender or reuse its sequence."""

    class FakeSocket:
        def __init__(self):
            self.attempts = 0
            self.sent_packets = []
            self.closed = False

        def setblocking(self, enabled):
            assert enabled is False

        def sendto(self, packet, target):
            self.attempts += 1
            if self.attempts == 1:
                raise BlockingIOError("simulated full UDP send buffer")
            self.sent_packets.append((packet, target))

        def close(self):
            self.closed = True

    fake_socket = FakeSocket()
    monkeypatch.setattr(audio_source.socket, "socket", lambda *_args: fake_socket)
    monkeypatch.setattr(audio_source, "_query_native_rate", lambda _device: 16_000)

    class FakeStream:
        active = True

        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    stream_holder = {}

    def input_stream(**kwargs):
        stream = FakeStream(**kwargs)
        stream_holder["stream"] = stream
        return stream

    monkeypatch.setattr(audio_source.sd, "InputStream", input_stream)

    def fast_sleep(_seconds):
        # Deliver two blocks without an audio device or actual waiting.  The
        # main loop then raises KeyboardInterrupt to run ordinary cleanup.
        if "stream" not in stream_holder:
            return
        stream = stream_holder["stream"]
        if stream.active:
            block = np.ones((8, 1), dtype=np.float32)
            stream.callback(block, len(block), None, None)
            stream.callback(block, len(block), None, None)
            stream.active = False
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(audio_source.time, "sleep", fast_sleep)

    audio_source.run(None, 7, "127.0.0.1", 9000, 1.0)

    assert fake_socket.attempts == 2
    assert len(fake_socket.sent_packets) == 1
    packet, target = fake_socket.sent_packets[0]
    magic, sequence, payload_size, source_id = struct.unpack_from("!IIHH", packet)
    assert (magic, sequence, payload_size, source_id) == (
        audio_source.MAGIC,
        1,  # Sequence 0 failed; the successful packet must expose the gap.
        len(packet) - audio_source.HEADER.size,
        7,
    )
    assert target == ("127.0.0.1", 9000)
    assert fake_socket.closed

