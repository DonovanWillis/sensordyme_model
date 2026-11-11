"""Unit tests for core/behavior/ components (synthetic scenarios, deterministic time).

Every scenario here is SYNTHETIC: hand-built boxes, angles and timestamps.
Passing them shows the logic does what it says; it is not a measured accuracy.
"""

from __future__ import annotations

import pytest

from core.behavior.evidence import UNKNOWN
from core.behavior.glance import (
    FREQUENT_GLANCING,
    GLANCE_BUDGET_EXCEEDED,
    LONG_GLANCE,
    GlanceAnalyzer,
    classify_zone,
)

STEP = 0.05



def _glance(analyzer, t0, seconds, gaze):
    events = []
    n = round(seconds / STEP)
    for i in range(n):
        events += analyzer.update(t0 + i * STEP, gaze)
    return events, t0 + n * STEP


MIRROR_LEFT = (50.0, 0.0)
MIRROR_RIGHT = (-60.0, 0.0)
CLUSTER = (0.0, -15.0)
INFOTAINMENT = (-30.0, -20.0)
LAP = (0.0, -35.0)
ROAD = (0.0, 0.0)


@pytest.mark.parametrize(("gaze", "zone"), [
    (ROAD, "road"), (MIRROR_LEFT, "mirror_left"), (MIRROR_RIGHT, "mirror_right"), (CLUSTER, "cluster"),
    (INFOTAINMENT, "infotainment"), (LAP, "lap"), ((-30.0, 15.0), "mirror_rear"), ((25.0, 40.0), "other"),
])
def test_default_zone_geometry(gaze, zone):
    assert classify_zone(*gaze) == zone


def test_repeated_short_mirror_checks_are_not_distraction():
    a = GlanceAnalyzer()
    events, t = [], 0.0
    for gaze in (MIRROR_LEFT, MIRROR_RIGHT) * 5:  # ten 0.8 s mirror checks in ~38 s
        e, t = _glance(a, t, 0.8, gaze)
        events += e
        e, t = _glance(a, t, 3.0, ROAD)
        events += e
    assert events == []
    assert a.budget_used_s == 0.0 and a.repeat_count == 0


def test_a_one_second_instrument_glance_is_nothing():
    a = GlanceAnalyzer()
    e1, t = _glance(a, 0.0, 1.0, CLUSTER)
    e2, _ = _glance(a, t, 2.0, ROAD)
    assert e1 + e2 == []


def test_a_mirror_glance_that_overstays_is_charged_from_its_start():
    a = GlanceAnalyzer()
    events, _ = _glance(a, 0.0, 2.5, MIRROR_LEFT)
    assert [e.type for e in events] == [LONG_GLANCE]
    assert a.budget_used_s == pytest.approx(2.45, abs=0.06)


def test_single_long_off_road_glance_fires_long_glance_once():
    a = GlanceAnalyzer()
    events, t = _glance(a, 0.0, 2.5, INFOTAINMENT)
    events += _glance(a, t, 1.0, ROAD)[0]
    assert [e.type for e in events] == [LONG_GLANCE]
    assert "infotainment" in events[0].detail


def test_four_short_glances_exceed_the_off_road_budget():
    a = GlanceAnalyzer()
    events, t = [], 0.0
    for _ in range(4):
        e, t = _glance(a, t, 1.8, LAP)
        events += e
        e, t = _glance(a, t, 2.0, ROAD)
        events += e
    types = [e.type for e in events]
    assert GLANCE_BUDGET_EXCEEDED in types and LONG_GLANCE not in types
    assert types.count(GLANCE_BUDGET_EXCEEDED) == 1


def test_pitch_only_glances_count_toward_frequent_glancing():
    a = GlanceAnalyzer()
    events, t = [], 0.0
    for _ in range(5):  # looking down into the lap, never turning the head sideways
        e, t = _glance(a, t, 0.8, LAP)
        events += e
        e, t = _glance(a, t, 4.0, ROAD)
        events += e
    assert [e.type for e in events] == [FREQUENT_GLANCING]


def test_gaze_unknown_is_reported_and_accrues_nothing():
    a = GlanceAnalyzer()
    _glance(a, 0.0, 1.0, LAP)
    used = a.budget_used_s
    for i in range(40):
        a.update(1.0 + i * STEP, None, "face not detected")
    assert a.assessment.state == UNKNOWN and a.assessment.reason == "face not detected"
    assert a.budget_used_s == pytest.approx(used)


def test_stale_frames_do_not_accrue_off_road_time():
    a = GlanceAnalyzer()
    events = []
    for i in range(10):  # off-road frames 1 s apart: ten seconds of wall time, zero credited
        events += a.update(i * 1.0, LAP)
    assert events == [] and a.budget_used_s == 0.0

