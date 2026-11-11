"""Shared primitives for behaviour reasoning: explicit unknowns and honest time.

Two rules every component in core/behavior/ follows, and these helpers are
how they follow them:

  * **Unknown is a state, not a default.** A component whose inputs are
    missing or stale reports ``UNKNOWN`` with a reason. It never collapses
    "could not tell" into "safe" (no event) or "unsafe" (an event): those are
    claims, and a missing input supports neither. ``Assessment`` carries the
    state and the reason together so the reason cannot be dropped on the way
    to telemetry.
  * **Unobserved time is not evidence.** ``FrameGapGuard`` turns successive
    frame timestamps into *credited* seconds. A gap longer than ``max_gap_s``
    (a camera stall, a blocked loop, a dropped run of frames) credits nothing
    at all -- stricter than modules/base.EvidenceClock, which still credits
    ``max_gap_s`` across a stall. Sustain timers built on credited time cannot
    be satisfied by time nobody watched.

Pure: standard library only, no clock is read. Callers pass frame time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

#: The state every component reports when its inputs cannot support a claim.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Assessment:
    """A component's current state plus why, when the why matters.

    ``reason`` is required (non-empty) for ``UNKNOWN`` and optional otherwise,
    where it explains a non-obvious known state (e.g. "outside driver region").
    """

    state: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state == UNKNOWN and not self.reason:
            raise ValueError("an UNKNOWN assessment must say why")

    @property
    def known(self) -> bool:
        return self.state != UNKNOWN

    def to_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason}


def unknown(reason: str) -> Assessment:
    return Assessment(UNKNOWN, reason)


class FrameGapGuard:
    """Credited seconds between consecutive observations.

    ``step(t)`` returns ``(credited_dt, stale)``. The first call credits 0.
    A gap above ``max_gap_s`` -- or a timestamp that went backwards -- credits
    0 and is reported stale, so nothing downstream accrues across it.
    """

    def __init__(self, max_gap_s: float = 0.5) -> None:
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        self.max_gap_s = float(max_gap_s)
        self._last: Optional[float] = None
        self.stale_gaps = 0
        self.uncredited_s = 0.0

    def step(self, t: float) -> tuple[float, bool]:
        last, self._last = self._last, float(t)
        if last is None:
            return 0.0, False
        gap = float(t) - last
        if gap < 0 or gap > self.max_gap_s:
            self.stale_gaps += 1
            self.uncredited_s += max(gap, 0.0)
            return 0.0, True
        return gap, False

    def reset(self) -> None:
        self._last = None


class KOfN:
    """k-of-n persistence over discrete observations (e.g. detector ticks).

    ``update(True/False)`` records one observation; ``present`` is True while
    at least ``k`` of the last ``n`` observations were True. Only call it when
    an observation was actually made -- a frame where the detector did not run
    is not a "False".
    """

    def __init__(self, k: int, n: int) -> None:
        if not (1 <= k <= n):
            raise ValueError(f"need 1 <= k <= n, got k={k} n={n}")
        self.k, self.n = int(k), int(n)
        self._window: deque[bool] = deque(maxlen=self.n)

    def update(self, observed: bool) -> bool:
        self._window.append(bool(observed))
        return self.present

    @property
    def present(self) -> bool:
        return sum(self._window) >= self.k

    @property
    def hits(self) -> int:
        return sum(self._window)

    def reset(self) -> None:
        self._window.clear()


Box = tuple[float, float, float, float]


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def intersection_over_area(subject: Box, other: Box) -> float:
    """Fraction of ``subject``'s area covered by ``other`` (NOT IoU).

    For hand-on-phone this is the right measure: a hand box is usually far
    larger than the phone it holds, so IoU stays small even when the phone is
    entirely inside the hand -- the phone's own coverage is what "held" means.
    """
    area = box_area(subject)
    return intersection_area(subject, other) / area if area > 0 else 0.0


def box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_contains_point(box: Box, point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
