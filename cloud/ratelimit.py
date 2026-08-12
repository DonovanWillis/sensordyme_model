"""Per-IP rate limiting for endpoints reachable before (or instead of) auth.

`/v1/login`, `/v1/signup` and `/v1/devices/claim` all take a guessable secret
(a password, an email/password pair, an 8-character claim code) from an
unauthenticated caller, with nothing else in front of them — no proxy, no
WAF, no account lockout. Without a limit here, an attacker with network
reachability to the backend can make unlimited guesses at whatever cost of
their own request rate, which is exactly the "brute force it" case rate
limiting exists for.

In memory, the same design as cloud/notify.py's NotificationGate and for the
same reason: a rate limit is a statement about the last few minutes, not
something that needs to survive a restart, and a restart at worst grants
every caller one fresh window.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class RateLimiter:
    """Sliding-window attempt limiter, keyed by an arbitrary string (a client IP)."""

    def __init__(self, max_attempts: int, window_s: float) -> None:
        self._max_attempts = max_attempts
        self._window_s = window_s
        self._attempts: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """Whether `key` may make another attempt right now.

        Records the attempt (counts toward the window) only when allowed —
        a caller already being throttled must not keep extending their own
        window by continuing to hammer the endpoint.
        """
        now = time.time() if now is None else now
        with self._lock:
            attempts = self._attempts.setdefault(key, deque())
            cutoff = now - self._window_s
            while attempts and attempts[0] < cutoff:
                attempts.popleft()
            if len(attempts) >= self._max_attempts:
                return False
            attempts.append(now)
            return True

    def reset(self, key: Optional[str] = None) -> None:
        """Forget rate-limit state — for tests."""
        with self._lock:
            if key is None:
                self._attempts.clear()
            else:
                self._attempts.pop(key, None)


# 10 attempts per 5 minutes per IP. Generous enough that a real person
# mistyping a password or a claim code a few times in a row is never
# affected; tight enough that an online guessing script is not viable.
LOGIN_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)
SIGNUP_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)
CLAIM_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)
