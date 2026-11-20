"""Per-API-key, per-minute rate limiting (in-memory sliding window).

LiteLLM-style gateways enforce RPM limits per virtual key before the
request ever reaches a model — the same idea here, minus Redis: a
sliding window of request timestamps per key, kept in memory. Limits
restart with the process (documented; for a multi-worker deployment
this would need a shared store such as Redis).
"""
from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> list of request timestamps (seconds, last 60s window)
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str, limit_per_minute: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds).

        ``allowed`` is False when the key already used its whole minute's
        allowance. ``retry_after_seconds`` is 0 when allowed, otherwise
        how long until the oldest hit in the window expires.
        """
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > window_start]
            if len(hits) >= limit_per_minute:
                retry_after = int(hits[0] + 60.0 - now) + 1
                self._hits[key] = hits
                return False, max(1, retry_after)
            hits.append(now)
            self._hits[key] = hits
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
