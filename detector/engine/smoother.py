"""Temporal smoother: sliding window state machine (idle → possible → confirmed)."""

from __future__ import annotations

import collections
import time


class TemporalSmoother:
    def __init__(self, window_seconds, hop_seconds, threshold, possible, confirmed,
                 cooldown_seconds, refractory_seconds):
        maxlen = int(window_seconds / hop_seconds) + 1
        self._scores: collections.deque = collections.deque(maxlen=maxlen)
        self._threshold = threshold
        self._possible = possible
        self._confirmed = confirmed
        self._cooldown_seconds = cooldown_seconds
        self._refractory_seconds = refractory_seconds
        self._cooldown_until = 0.0
        self._refractory_until = 0.0
        self._state = "idle"
        self._last_trigger_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def trigger_count(self) -> int:
        return self._last_trigger_count

    @property
    def cooldown_remaining(self) -> float:
        r = self._cooldown_until - time.monotonic()
        return max(0.0, r)

    @property
    def refractory_remaining(self) -> float:
        r = self._refractory_until - time.monotonic()
        return max(0.0, r)

    @property
    def possible_needed(self) -> int:
        return self._possible

    @property
    def confirmed_needed(self) -> int:
        return self._confirmed

    def push(self, score: float) -> str | None:
        now = time.monotonic()
        if now < self._cooldown_until:
            if score >= self._threshold:
                self._cooldown_until = now + self._cooldown_seconds
            return None
        if self._state == "confirmed":
            self._state = "idle"
            self._refractory_until = now + self._refractory_seconds
            self._scores.clear()
            return "idle"
        # refractory: time-based, but ends early if score drops below threshold
        if now < self._refractory_until:
            if score < self._threshold:
                self._refractory_until = 0.0  # quiet → end early
            else:
                self._scores.clear()
                return None
        prev = self._state
        self._scores.append(score)
        triggers = sum(1 for s in self._scores if s >= self._threshold)
        self._last_trigger_count = triggers
        if triggers >= self._confirmed:
            self._state = "confirmed"
            self._cooldown_until = now + self._cooldown_seconds
            self._scores.clear()
        elif triggers >= self._possible:
            self._state = "possible"
        else:
            self._state = "idle"
        return self._state if self._state != prev else None
