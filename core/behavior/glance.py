"""Glance analysis: where the driver is looking, for how long, and how often.

The shipped ``frequent_glancing`` counts rising edges of ``|yaw| >
yaw_threshold_deg`` only (modules/driver_attention.py GlanceCounter): pitch is
ignored entirely (a driver texting in their lap never "glances"), every mirror
check counts the same as a look at the passenger, and nothing measures how
much total time the eyes were off the road.

Here the calibrated gaze (yaw, pitch -- relative to the driver's own neutral
"road" pose, see AttentionCalibrator) is classified into cabin ZONES, and
three things are measured over glance EPISODES (contiguous off-road time):

  * long_glance              one episode reaching ``long_glance_s`` off road
  * glance_budget_exceeded   credited off-road seconds in a sliding window
                             above ``budget_s``
  * frequent_glancing        ``repeat_min_count`` counted episodes in a window

A glance that stays entirely within LEGITIMATE zones (mirrors, instrument
cluster by default) and ends within ``legit_glance_max_s`` is part of driving:
it counts toward none of the three. One that overstays, or wanders into any
other zone, loses that status and is charged from its start.

Defaults follow the shape of NHTSA's visual-manual guidance (2 s single
glance) but NOT its values for total time: NHTSA's 12 s is a per-TASK total,
not a per-window budget. ``budget_s = 6 s per 30 s`` is an unvalidated
engineering default. Zone geometry assumes a left-hand-drive car, an
unmirrored image and a camera roughly in front of the driver (+yaw = toward
the frame's right = the driver's left). All of it needs in-cabin validation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.behavior.evidence import Assessment, FrameGapGuard, unknown

ROAD = "road"
OTHER = "other"

#: (name, (yaw_min, yaw_max, pitch_min, pitch_max)) in calibrated degrees;
#: first match wins, anything unmatched is OTHER.
DEFAULT_ZONES: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    (ROAD, (-20.0, 20.0, -8.0, 12.0)),
    ("cluster", (-15.0, 15.0, -25.0, -8.0)),
    ("infotainment", (-45.0, -15.0, -40.0, -8.0)),
    ("mirror_rear", (-45.0, -15.0, 5.0, 30.0)),
    ("mirror_left", (30.0, 80.0, -20.0, 15.0)),
    ("mirror_right", (-90.0, -45.0, -20.0, 15.0)),
    ("lap", (-35.0, 35.0, -90.0, -25.0)),
)
DEFAULT_LEGIT_ZONES = frozenset({"mirror_left", "mirror_right", "mirror_rear", "cluster"})

LONG_GLANCE = "long_glance"
GLANCE_BUDGET_EXCEEDED = "glance_budget_exceeded"
FREQUENT_GLANCING = "frequent_glancing"


@dataclass(frozen=True)
class GlanceConfig:
    zones: tuple[tuple[str, tuple[float, float, float, float]], ...] = DEFAULT_ZONES
    legit_zones: frozenset[str] = DEFAULT_LEGIT_ZONES
    legit_glance_max_s: float = 1.6
    long_glance_s: float = 2.0
    budget_window_s: float = 30.0
    budget_s: float = 6.0
    repeat_window_s: float = 45.0
    repeat_min_count: int = 5
    min_glance_s: float = 0.15  # shorter off-road blips are not counted as glances
    release_s: float = 0.2  # back on the road this long ends a glance
    unknown_end_s: float = 1.0  # gaze unknown this long ends the open glance (no time credited)
    max_gap_s: float = 0.5


def classify_zone(yaw: float, pitch: float, zones=DEFAULT_ZONES) -> str:
    for name, (y0, y1, p0, p1) in zones:
        if y0 <= yaw <= y1 and p0 <= pitch <= p1:
            return name
    return OTHER


@dataclass
class _Episode:
    start_t: float
    off_s: float = 0.0
    legit: bool = True
    charged: bool = False
    long_fired: bool = False
    road_since: Optional[float] = None
    dwell: dict[str, float] = field(default_factory=dict)

    @property
    def dominant_zone(self) -> str:
        return max(self.dwell, key=self.dwell.get) if self.dwell else OTHER


@dataclass(frozen=True)
class GlanceEvent:
    type: str
    duration_s: float
    detail: str


class GlanceAnalyzer:
    """Zone classification + long-glance / budget / repeated-glance events."""

    def __init__(self, config: Optional[GlanceConfig] = None) -> None:
        self.config = config or GlanceConfig()
        self._gap = FrameGapGuard(self.config.max_gap_s)
        self._episode: Optional[_Episode] = None
        self._unknown_since: Optional[float] = None
        self._budget: deque[tuple[float, float]] = deque()
        self._budget_sum = 0.0
        self._budget_warned = False
        self._repeats: deque[float] = deque()
        self._repeat_warned = False
        self.assessment: Assessment = unknown("no frame yet")
        self.last_glance: Optional[dict] = None

    # ----------------------------------------------------------- accounting
    def _charge(self, t: float, seconds: float) -> None:
        if seconds > 0:
            self._budget.append((t, seconds))
            self._budget_sum += seconds

    def _trim(self, now: float) -> None:
        c = self.config
        while self._budget and self._budget[0][0] < now - c.budget_window_s:
            self._budget_sum -= self._budget.popleft()[1]
        if not self._budget:
            self._budget_sum = 0.0
        while self._repeats and self._repeats[0] < now - c.repeat_window_s:
            self._repeats.popleft()

    def _close_episode(self, reason: str) -> None:
        ep = self._episode
        if ep is not None:
            self.last_glance = {
                "zone": ep.dominant_zone, "duration_s": round(ep.off_s, 2),
                "counted": ep.charged, "ended": reason,
            }
        self._episode = None

    @property
    def budget_used_s(self) -> float:
        return max(0.0, self._budget_sum)

    @property
    def repeat_count(self) -> int:
        return len(self._repeats)

    @property
    def current_glance_s(self) -> float:
        return self._episode.off_s if self._episode is not None else 0.0

    # ----------------------------------------------------------- update
    def update(self, t: float, gaze: Optional[tuple[float, float]], unknown_reason: str = "gaze unknown"
               ) -> list[GlanceEvent]:
        c = self.config
        dt, _stale = self._gap.step(t)
        events: list[GlanceEvent] = []
        self._trim(t)

        if gaze is None:
            self.assessment = unknown(unknown_reason)
            if self._unknown_since is None:
                self._unknown_since = t
            if self._episode is not None and t - self._unknown_since > c.unknown_end_s:
                self._close_episode("gaze unknown")
            return self._rearm(events)
        self._unknown_since = None

        zone = classify_zone(gaze[0], gaze[1], c.zones)
        self.assessment = Assessment(zone)
        if zone == ROAD:
            ep = self._episode
            if ep is not None:
                if ep.road_since is None:
                    ep.road_since = t
                if t - ep.road_since >= c.release_s:
                    self._close_episode("back on road")
            return self._rearm(events)

        ep = self._episode
        if ep is None:
            ep = self._episode = _Episode(start_t=t)
        ep.road_since = None
        ep.off_s += dt
        ep.dwell[zone] = ep.dwell.get(zone, 0.0) + dt
        if zone not in c.legit_zones:
            ep.legit = False
        if ep.legit and ep.off_s > c.legit_glance_max_s:
            ep.legit = False
        if ep.charged:
            self._charge(t, dt)
        elif not ep.legit and ep.off_s >= c.min_glance_s:
            ep.charged = True
            self._charge(ep.start_t, ep.off_s)  # charged from its start, incl. any legit-looking lead-in
            self._repeats.append(ep.start_t)
        if ep.charged and not ep.long_fired and ep.off_s >= c.long_glance_s:
            ep.long_fired = True
            events.append(GlanceEvent(LONG_GLANCE, ep.off_s,
                                      f"single off-road glance {ep.off_s:.1f}s (zone {ep.dominant_zone})"))
        self._trim(t)
        self._rearm(events)
        if self._budget_sum > c.budget_s and not self._budget_warned:
            self._budget_warned = True
            events.append(GlanceEvent(
                GLANCE_BUDGET_EXCEEDED, self._budget_sum,
                f"{self._budget_sum:.1f}s off road in {c.budget_window_s:.0f}s (budget {c.budget_s:.1f}s)"))
        if len(self._repeats) >= c.repeat_min_count and not self._repeat_warned:
            self._repeat_warned = True
            events.append(GlanceEvent(
                FREQUENT_GLANCING, 0.0,
                f"{len(self._repeats)} off-road glances in {c.repeat_window_s:.0f}s (yaw and pitch)"))
        return events

    def _rearm(self, events: list[GlanceEvent]) -> list[GlanceEvent]:
        c = self.config
        if self._budget_warned and self._budget_sum <= c.budget_s * 0.5:
            self._budget_warned = False
        if self._repeat_warned and len(self._repeats) < c.repeat_min_count:
            self._repeat_warned = False
        return events
