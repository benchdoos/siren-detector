"""Webhook output queue and failure isolation tests."""

from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

from detector.runtime.contracts import RuntimeMessage
from detector.runtime.webhook import WebhookOutput, webhook_urls_from_environment


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"ok"


class WebhookTests(unittest.TestCase):
    def test_event_is_posted_to_all_targets_and_failures_are_counted(self) -> None:
        done = threading.Event()
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            if request.full_url.endswith("/down"):
                done.set()
                raise OSError("recipient unavailable")
            return _Response()

        output = WebhookOutput(
            ["https://example.test/up", "https://example.test/down"],
            timeout_seconds=0.25,
            token="secret-token",
        )
        try:
            with patch("detector.runtime.webhook.urlopen", side_effect=fake_urlopen):
                output.publish(RuntimeMessage.from_legacy({
                    "type": "event", "event": "siren", "state": "confirmed",
                    "confidence": 0.8, "source_id": 4, "engine_id": "yamnet",
                }))
                self.assertTrue(done.wait(1))
                # Wait briefly for the worker to update its final counters.
                end = time.monotonic() + 1
                while output.snapshot()["failures"] == 0 and time.monotonic() < end:
                    time.sleep(0.005)
            snapshot = output.snapshot()
            self.assertEqual(snapshot["sent"], 1)
            self.assertEqual(snapshot["failures"], 1)
            self.assertEqual(len(requests), 2)
            payload = json.loads(requests[0][0].data)
            self.assertEqual(payload["contract"], "detector-event.v1")
            self.assertEqual(payload["engine_id"], "yamnet")
            self.assertEqual(requests[0][0].get_header("Authorization"), "Bearer secret-token")
        finally:
            output.stop()

    def test_environment_requires_http_urls(self) -> None:
        with patch.dict("os.environ", {"WEBHOOK_URLS": "file:///tmp/event.json"}):
            with self.assertRaisesRegex(ValueError, r"absolute HTTP\(S\) URLs"):
                webhook_urls_from_environment()

    def test_non_finite_timeout_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "queue capacity and timeout"):
            WebhookOutput(["https://example.test/hook"], timeout_seconds=float("inf"))


if __name__ == "__main__":
    unittest.main()
