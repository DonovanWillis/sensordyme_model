"""Tests for the baby monitoring module's reasoning layer (no camera, no YOLO)."""

import numpy as np

from modules.baby_monitoring import BabyMonitoringModule

ZONES = {
    "crib": {"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]], "kind": "safe"},
    "stairs": {"polygon": [[0.0, 0.5], [0.15, 0.5], [0.15, 1.0], [0.0, 1.0]], "kind": "restricted"},
}

W, H = 1000, 1000
IN_CRIB = (400, 300, 600, 800)      # feet at (0.5, 0.8) -> inside crib
ON_STAIRS = (20, 600, 120, 950)     # feet at (0.07, 0.95) -> inside stairs
ELSEWHERE = (850, 300, 980, 800)    # feet at (0.915, 0.8) -> outside both


def _module(**overrides):
    config = {
        "confidence_threshold": 0.4, "still_warning_s": 5.0, "still_max_motion": 0.015,
        "absence_warning_s": 2.0, "expect_presence": False, "zone_confirm_s": 1.0,
        "zones": ZONES,
    }
    config.update(overrides)
    return BabyMonitoringModule(config)


def _run(module, boxes, motion, seconds, start=0.0, step=0.1):
    events = []
    t = start
    while t < start + seconds:
        events += module._reason(list(boxes), motion, W, H, t)
        t += step
    return events, t


def test_restricted_zone_entry_confirms_after_persistence():
    module = _module()
    events, _ = _run(module, [ON_STAIRS], motion=0.05, seconds=3.0)
    types = [e.type for e in events]
    assert "zone_violation" in types
    violation = next(e for e in events if e.type == "zone_violation")
    assert "stairs" in violation.contributing_signals
    assert violation.duration >= 1.0  # not a single-frame alarm
    assert module.is_alert_active()


def test_safe_zone_exit_alerts_and_staying_in_crib_does_not():
    module = _module()
    events, t = _run(module, [IN_CRIB], motion=0.05, seconds=3.0)
    assert "safe_zone_exit" not in [e.type for e in events]  # in the crib: all good

    events, _ = _run(module, [ELSEWHERE], motion=0.05, seconds=3.0, start=t)
    assert "safe_zone_exit" in [e.type for e in events]  # left the crib


def test_unusual_stillness_needs_the_full_duration():
    module = _module()
    events, t = _run(module, [IN_CRIB], motion=0.001, seconds=3.0)
    assert "unusual_stillness" not in [e.type for e in events]  # only 3s of 5s
    events, _ = _run(module, [IN_CRIB], motion=0.001, seconds=4.0, start=t)
    assert "unusual_stillness" in [e.type for e in events]  # crossed still_warning_s


def test_normal_motion_never_triggers_stillness():
    module = _module()
    events, _ = _run(module, [IN_CRIB], motion=0.05, seconds=10.0)
    assert "unusual_stillness" not in [e.type for e in events]


def test_person_absent_only_when_expected():
    relaxed = _module()
    events, _ = _run(relaxed, [], motion=0.0, seconds=5.0)
    assert "person_absent" not in [e.type for e in events]

    strict = _module(expect_presence=True)
    events, _ = _run(strict, [], motion=0.0, seconds=5.0)
    assert "person_absent" in [e.type for e in events]


def test_count_change_is_logged():
    module = _module()
    _run(module, [IN_CRIB], motion=0.05, seconds=1.0)
    events, _ = _run(module, [IN_CRIB, ELSEWHERE], motion=0.05, seconds=0.2, start=1.0)
    assert "person_count_change" in [e.type for e in events]


def test_motion_energy_computation():
    module = _module()
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    white = np.full((240, 320, 3), 255, dtype=np.uint8)
    assert module._motion_energy(black) == 0.0       # first frame: no reference
    assert module._motion_energy(black) < 0.001      # identical frame: still
    assert module._motion_energy(white) > 0.5        # full change: huge motion


def test_annotate_draws_without_camera(tmp_path):
    module = _module()
    module._last_boxes = [IN_CRIB]
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    out = module.annotate(frame)
    assert out.shape == frame.shape
    assert out.sum() > 0  # zones + box + status text actually drawn
