"""Asynchronous HTTP webhook output for versioned detector events."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .contracts import RuntimeMessage

logger = logging.getLogger(__name__)
_STOP = object()


def webhook_urls_from_environment() -> list[str]:
    """Parse comma-separated webhook endpoints from ``WEBHOOK_URLS``."""
    raw = os.environ.get("WEBHOOK_URLS", "")
    urls = [item.strip() for item in raw.split(",") if item.strip()]
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("WEBHOOK_URLS entries must be absolute HTTP(S) URLs")
    return urls


class WebhookOutput:
    """Deliver event JSON from a bounded queue without blocking inference."""

    name = "webhook"

    def __init__(
        self,
        urls: list[str],
        *,
        queue_capacity: int = 128,
        timeout_seconds: float = 2.0,
        token: str = "",
    ) -> None:
        if not urls:
            raise ValueError("at least one webhook URL is required")
        if queue_capacity < 1 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("webhook queue capacity and timeout must be positive")
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("webhook URLs must be absolute HTTP(S) URLs")
        self._urls = tuple(urls)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        self._timeout = timeout_seconds
        self._token = token
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sent = 0
        self._dropped = 0
        self._failures = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="webhook-output", daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_environment(cls) -> "WebhookOutput | None":
        """Create an output only when ``WEBHOOK_URLS`` is configured."""
        urls = webhook_urls_from_environment()
        if not urls:
            return None
        timeout = float(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "2"))
        return cls(
            urls,
            timeout_seconds=timeout,
            token=os.environ.get("WEBHOOK_TOKEN", ""),
        )

    def publish(self, message: RuntimeMessage) -> None:
        """Queue event messages, dropping the oldest pending event on overflow."""
        if message.kind != "event":
            return
        payload = json.dumps(message.as_legacy(), separators=(",", ":"), allow_nan=False).encode()
        with self._lock:
            if self._stop.is_set():
                return
            try:
                self._queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                else:
                    self._dropped += 1
                try:
                    self._queue.put_nowait(payload)
                except queue.Full:
                    self._dropped += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "targets": len(self._urls),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "sent": self._sent,
                "dropped": self._dropped,
                "failures": self._failures,
                "last_error": self._last_error,
                "state": "stopping" if self._stop.is_set() else "ready",
            }

    def stop(self) -> None:
        """Discard pending events and stop after any current request times out."""
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
                    self._dropped += 1
            self._queue.put_nowait(_STOP)
        self._thread.join(timeout=self._timeout * len(self._urls) + 0.5)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            for url in self._urls:
                headers = {"Content-Type": "application/json"}
                if self._token:
                    headers["Authorization"] = f"Bearer {self._token}"
                request = Request(url, data=item, headers=headers, method="POST")
                try:
                    with urlopen(request, timeout=self._timeout) as response:
                        response.read()
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    logger.warning("Webhook delivery to %s failed: %s", url, exc)
                    with self._lock:
                        self._failures += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"
                else:
                    with self._lock:
                        self._sent += 1
            self._queue.task_done()
