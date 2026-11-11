"""A deterministic clock for driving the virtual device (and backend) in tests.

The device side reads `time.monotonic()` throughout (frame ages, heartbeats,
trackers, episode timing). The existing deterministic tests replace it
process-wide (tests/test_m01_role_floors.py's `clock` fixture); SimClock does
the same, explicitly and reversibly. threading's own waits are unaffected:
threading captured the real monotonic at import.

The backend reads wall time (`time.time()`, last_seen and the 30 s staleness
window). `wall_clock_module()` is a stand-in for a module's `time` import whose
time() follows this clock, so a test can age the backend's view of the device
without waiting 30 s.
"""

from __future__ import annotations

import time as _time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional


class SimClock:
    def __init__(self, start: float = 1000.0, wall_start: Optional[float] = None) -> None:
        self._start = float(start)
        self._now = float(start)
        self._wall_start = _time.time() if wall_start is None else float(wall_start)

    def monotonic(self) -> float:
        return self._now

    def wall(self) -> float:
        return self._wall_start + (self._now - self._start)

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("a clock does not go backwards")
        self._now = round(self._now + seconds, 6)
        return self._now

    def wall_clock_module(self) -> "_WallTime":
        return _WallTime(self)

    @contextmanager
    def installed(self) -> Iterator["SimClock"]:
        """time.monotonic() follows this clock until the block exits."""
        original = _time.monotonic
        _time.monotonic = self.monotonic
        try:
            yield self
        finally:
            _time.monotonic = original


class _WallTime:
    """Looks like the `time` module, with time() on the sim clock."""

    def __init__(self, clock: SimClock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.wall()

    def __getattr__(self, name: str):
        return getattr(_time, name)
