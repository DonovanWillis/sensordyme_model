"""Per-IP rate limiting for endpoints reachable before (or instead of) auth.

`/v1/login`, `/v1/signup` and `/v1/devices/claim` all take a guessable secret
(a password, an email/password pair, a 12-hex-digit claim code) from an
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
from collections import OrderedDict, deque
from typing import Optional

#: Most distinct keys a limiter remembers at once.
#:
#: Keys are client addresses, and nothing used to forget one: every address
#: that ever touched login, signup or claim kept its (empty) entry for the life
#: of the process -- measured at ~860 bytes each, so ~630k distinct callers
#: filled the deployment's 512 MB VM, and signup with an invalid email is a
#: cheap way to be a new caller. Keys whose window has emptied are now dropped,
#: and past this many the least recently used go first. 50k keys is ~43 MB.
#: Evicting under pressure means a caller with tens of thousands of distinct
#: addresses could push their own entry out; that is far beyond what keying
#: IPv6 by /64 (cloud/app.py's `_client_ip`) leaves to a single host.
MAX_KEYS = 50_000


class RateLimiter:
    """Sliding-window attempt limiter, keyed by an arbitrary string (a client IP)."""

    def __init__(self, max_attempts: int, window_s: float, max_keys: int = MAX_KEYS) -> None:
        self._max_attempts = max_attempts
        self._window_s = window_s
        self._max_keys = max_keys
        self._attempts: "OrderedDict[str, deque[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._next_sweep = 0.0

    # ------------------------------------------------------------ internals
    def _current(self, key: str, now: float) -> Optional[deque]:
        """`key`'s attempts still inside the window (None if there are none).
        Caller holds the lock."""
        attempts = self._attempts.get(key)
        if attempts is None:
            return None
        cutoff = now - self._window_s
        while attempts and attempts[0] < cutoff:
            attempts.popleft()
        if not attempts:
            del self._attempts[key]
            return None
        return attempts

    def _sweep(self, now: float) -> None:
        """Forget every key whose window has emptied. At most once per window,
        so the O(keys) pass is amortised over everything in between."""
        if now < self._next_sweep:
            return
        self._next_sweep = now + self._window_s
        cutoff = now - self._window_s
        for key in [k for k, a in self._attempts.items() if not a or a[-1] < cutoff]:
            del self._attempts[key]

    def _record(self, key: str, now: float) -> None:
        attempts = self._attempts.get(key)
        if attempts is None:
            attempts = deque()
            self._attempts[key] = attempts
            while len(self._attempts) > self._max_keys:
                self._attempts.popitem(last=False)  # least recently used
        else:
            self._attempts.move_to_end(key)
        attempts.append(now)

    # ------------------------------------------------------------ public
    def allow(self, key: str, now: Optional[float] = None) -> bool:
        """Whether `key` may make another attempt right now.

        Records the attempt (counts toward the window) only when allowed —
        a caller already being throttled must not keep extending their own
        window by continuing to hammer the endpoint.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._sweep(now)
            attempts = self._current(key, now)
            if attempts is not None and len(attempts) >= self._max_attempts:
                return False
            self._record(key, now)
            return True

    def blocked(self, key: str, now: Optional[float] = None) -> bool:
        """Whether `key` is over its limit, WITHOUT counting this as an attempt.

        For limits that count only failures (see LOGIN_FAILURE_LIMITER): the
        check happens before the outcome is known, the recording after.
        """
        now = time.time() if now is None else now
        with self._lock:
            attempts = self._current(key, now)
            return attempts is not None and len(attempts) >= self._max_attempts

    def record(self, key: str, now: Optional[float] = None) -> None:
        """Count one attempt against `key` unconditionally (a failure)."""
        now = time.time() if now is None else now
        with self._lock:
            self._sweep(now)
            self._current(key, now)
            self._record(key, now)

    def __len__(self) -> int:
        with self._lock:
            return len(self._attempts)

    def reset(self, key: Optional[str] = None) -> None:
        """Forget rate-limit state — for tests."""
        with self._lock:
            if key is None:
                self._attempts.clear()
            else:
                self._attempts.pop(key, None)


class ConcurrencyLimiter:
    """At most `max_in_flight` concurrent operations per key.

    A sliding window bounds how OFTEN someone may start something; this bounds
    how much of the server they can hold at once. The two differ for anything
    slow: model calls block a worker thread for up to a minute each, so a
    handful started together by one account occupies the whole thread pool and
    every other account's requests -- including the devices' ingest -- wait.
    """

    def __init__(self, max_in_flight: int) -> None:
        self._max = max_in_flight
        self._in_flight: dict[str, int] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> bool:
        with self._lock:
            current = self._in_flight.get(key, 0)
            if current >= self._max:
                return False
            self._in_flight[key] = current + 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            remaining = self._in_flight.get(key, 0) - 1
            if remaining > 0:
                self._in_flight[key] = remaining
            else:
                self._in_flight.pop(key, None)


# 10 attempts per 5 minutes per IP. Generous enough that a real person
# mistyping a password or a claim code a few times in a row is never
# affected; tight enough that an online guessing script is not viable.
LOGIN_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)
SIGNUP_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)
CLAIM_LIMITER = RateLimiter(max_attempts=10, window_s=300.0)

# Failed claims per ACCOUNT (M8), whatever address they come from: the per-IP
# limit above is blind to an account guessing claim codes from many addresses
# (an IPv4 botnet, or a trusted proxy header naming a different client each
# time). Only failures count -- an unknown code, or a device another account
# owns -- so pairing a real device is never slowed. 20 an hour is far above
# anyone mistyping a label, and holds one account to under 500 guesses a day
# against a 48-bit code space.
CLAIM_FAILURE_LIMITER = RateLimiter(max_attempts=20, window_s=3600.0)

# Failed logins per ACCOUNT (normalised email), whatever address they come
# from. The per-address limit above cannot see a guesser who rotates addresses
# -- one IPv6 host has a /64 to rotate through, and a botnet has thousands of
# IPv4s -- so it was the only thing between a known email and unlimited online
# guessing. Only FAILURES count, so the owner signing in normally never uses
# it up. The cost, stated plainly: someone who knows an email can keep that
# account from signing in on a NEW phone for the window. Existing sessions are
# unaffected (they hold tokens), which is the trade every lockout makes.
LOGIN_FAILURE_LIMITER = RateLimiter(max_attempts=20, window_s=900.0)

# Model calls, per ACCOUNT (they are authenticated). The app asks for a preview
# when someone stops typing, so a few a minute is normal use; 30 is far above
# it. Two in flight lets a new preview start while the last is finishing.
AI_RATE_LIMITER = RateLimiter(max_attempts=30, window_s=60.0)
AI_CONCURRENCY = ConcurrencyLimiter(max_in_flight=2)

# Model calls in flight across EVERY account. The per-account bound above stops
# one account holding the thread pool; it did nothing about twenty accounts
# (two addresses' worth of signups) holding two each, which is the whole
# default pool of 40 -- and devices' ingest waits behind it. 8 leaves most of
# the pool for everything else. One key, so `acquire(AI_GLOBAL_KEY)`.
AI_GLOBAL_KEY = "*"
AI_GLOBAL_CONCURRENCY = ConcurrencyLimiter(max_in_flight=8)
