"""fatigue_pattern: what it is, and why it is not an alert.

Blink rate is a real fatigue correlate, but it also moves with reading, talking,
screens and dry air, so on its own it is too confounded to interrupt a driver.
The decision (modules/driver_attention_max.py, BlinkRateTracker): it stays
log-only, it feeds the FATIGUED state — which config always claimed and the
code never did — and it corroborates the PERCLOS "fatigue" alert. And it is
judged against the person's OWN baseline, which was measured and never read.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from modules.driver_attention import PerclosTracker
from modules.driver_attention_max import BlinkRateTracker, DriverAttentionMaxModule


def _tracker(**overrides):
    params = {"window_s": 120.0, "baseline_seconds": 300.0, "low_per_min": 8.0, "high_per_min": 30.0, "confirm_s": 30.0}
    params.update(overrides)
    return BlinkRateTracker(**params)


def _blink_at_rate(tracker, per_min, start, end, step=0.5):
    """Feed blinks at `per_min` and call update every `step` seconds; return fires."""
    fired, next_blink, t = [], start, start
    interval = 60.0 / per_min if per_min > 0 else float("inf")
    while t < end:
        if t >= next_blink:
            tracker.record_blink(t)
            next_blink += interval
        if tracker.update(t):
            fired.append(t)
        t += step
    return fired


def test_a_naturally_slow_blinker_is_not_permanently_fatigued():
    """7 blinks/min is below the 8/min population floor, and was this person's
    normal for the whole session — before, that alone was a permanent anomaly."""
    tracker = _tracker()
    fired = _blink_at_rate(tracker, per_min=7.0, start=0.0, end=1200.0)
    assert tracker.baseline_rate_per_min == pytest.approx(7.0, abs=0.5)
    assert fired == []
    assert not tracker.is_active


def test_a_real_drop_from_the_persons_own_baseline_fires_with_a_confidence():
    tracker = _tracker()
    _blink_at_rate(tracker, per_min=16.0, start=0.0, end=300.0)   # baseline ~16/min
    fired = _blink_at_rate(tracker, per_min=3.0, start=300.0, end=900.0)
    assert fired, "a drop from 16 to 3 blinks/min, sustained, is the pattern"
    assert tracker.is_active
    assert tracker.last_confidence is not None and tracker.last_confidence > 0.9


def test_a_slow_blinker_still_fires_when_they_drop_far_below_their_own_normal():
    tracker = _tracker()
    _blink_at_rate(tracker, per_min=7.0, start=0.0, end=301.0)  # past the 300 s baseline window
    low, _ = tracker.effective_band
    assert low == pytest.approx(3.5, abs=0.3), "their band is half their own baseline"
    assert _blink_at_rate(tracker, per_min=1.0, start=301.0, end=900.0)


def test_a_brief_dip_does_not_fire():
    tracker = _tracker()
    _blink_at_rate(tracker, per_min=16.0, start=0.0, end=300.0)
    assert _blink_at_rate(tracker, per_min=2.0, start=300.0, end=315.0) == []


def _state(**flags):
    module: Any = DriverAttentionMaxModule.__new__(DriverAttentionMaxModule)
    module._calibrator = types.SimpleNamespace(is_calibrated=True)
    module._vlm_confirm = types.SimpleNamespace(active_state=None)
    defaults: dict[str, Any] = dict.fromkeys(("_drowsy_active", "_phone_use_distracted_active", "_fatigue_critical_active", "_fatigue_active", "_off_road_active", "_eyes_down_active", "_excessive_glancing_active"), False)
    defaults.update(flags)
    for name, value in defaults.items():
        setattr(module, name, value)
    module._blink_rate_tracker = types.SimpleNamespace(is_active=flags.get("blink", False))
    return module._resolve_state()


def test_the_blink_rate_anomaly_feeds_the_fatigued_state():
    assert _state() == "ATTENTIVE"
    assert _state(blink=True) == "FATIGUED"
    assert _state(blink=True, _drowsy_active=True) == "DROWSY", "a stronger state still wins"


def test_fatigue_pattern_is_deliberately_not_an_alert():
    import main as main_module

    assert "fatigue_pattern" not in main_module.ALERT_EVENT_TYPES_MAC_MAX


def test_perclos_counts_independent_closures_not_frames():
    """PERCLOS confidence rests on the number of distinct closures: 30 frames of
    one long closure are one observation, not thirty."""
    tracker = PerclosTracker(window_s=60.0, min_window_s=1.0)
    t = 0.0
    for closed in [False] * 10 + [True] * 30 + [False] * 10 + [True] * 5 + [False] * 5:
        tracker.update(closed, t)
        t += 0.05
    assert tracker.episodes == 2
