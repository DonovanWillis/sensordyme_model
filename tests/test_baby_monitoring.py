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


def _run(module, boxes, motion, seconds, start=0.0, step=0.1, bottle_boxes=(), object_boxes=None):
    events = []
    t = start
    while t < start + seconds:
        events += module._reason(list(boxes), motion, W, H, t, list(bottle_boxes), object_boxes)
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


# --------------------------------------------------------------- per-frame detection view
def test_get_detections_exposes_people_with_boxes(monkeypatch):
    """Zone and count rules need boxes and counts, which Events don't carry."""
    module = _module()
    monkeypatch.setattr(module, "_detect_people", lambda frame: [IN_CRIB])
    module.process(np.zeros((H, W, 3), dtype=np.uint8))

    frame = module.get_detections()
    assert frame.size == (W, H)
    assert frame.count("person") == 1
    assert len(frame.boxes("person")) == 1


def test_get_detections_defaults_confidence_for_stubbed_boxes(monkeypatch):
    """Boxes from an override that reports no confidence count as certain, not dropped."""
    module = _module()
    monkeypatch.setattr(module, "_detect_people", lambda frame: [IN_CRIB])
    module.process(np.zeros((H, W, 3), dtype=np.uint8))
    assert module.get_detections().mean_confidence("person") == 1.0


def test_get_detections_is_empty_before_any_frame():
    module = _module()
    frame = module.get_detections()
    assert frame.count("person") == 0
    assert frame.size == (0, 0)


# ------------------------------------------------------------------- holding_bottle
BOTTLE_IN_CRIB = (460, 350, 520, 500)     # overlaps IN_CRIB
BOTTLE_ELSEWHERE = (860, 300, 920, 500)   # does not overlap IN_CRIB


def test_bottle_overlapping_the_child_confirms_holding_bottle():
    module = _module(holding_bottle_confirm_s=1.0)
    events, _ = _run(module, [IN_CRIB], motion=0.05, seconds=1.5, bottle_boxes=[BOTTLE_IN_CRIB])
    assert "holding_bottle" in [e.type for e in events]


def test_bare_bottle_away_from_the_child_does_not_confirm():
    """'Bottle detected' alone is explicitly not meaningful -- only overlap is."""
    module = _module(holding_bottle_confirm_s=1.0)
    events, _ = _run(module, [IN_CRIB], motion=0.05, seconds=1.5, bottle_boxes=[BOTTLE_ELSEWHERE])
    assert "holding_bottle" not in [e.type for e in events]


def test_no_bottle_at_all_does_not_confirm():
    module = _module(holding_bottle_confirm_s=1.0)
    events, _ = _run(module, [IN_CRIB], motion=0.05, seconds=1.5)
    assert "holding_bottle" not in [e.type for e in events]


def test_holding_bottle_resolves_once_the_bottle_moves_away():
    module = _module(holding_bottle_confirm_s=1.0)
    events, t = _run(module, [IN_CRIB], motion=0.05, seconds=1.5, bottle_boxes=[BOTTLE_IN_CRIB])
    assert "holding_bottle" in [e.type for e in events]

    events, _ = _run(module, [IN_CRIB], motion=0.05, seconds=3.0, start=t, bottle_boxes=[BOTTLE_ELSEWHERE])
    assert "holding_bottle_resolved" in [e.type for e in events]


# ------------------------------------------------------------------- holding_object
# Before this, only "bottle" was ever requested from the object detector at all --
# a phone, cup, or food item held up in frame was invisible to this module
# regardless of confidence threshold, because the class was never in the YOLO
# call's `classes` filter to begin with. These exercise the same reasoning path
# holding_bottle already had, generalized to config.yaml's object_classes.
PHONE_IN_CRIB = (460, 350, 520, 500)      # overlaps IN_CRIB
PHONE_ELSEWHERE = (860, 300, 920, 500)    # does not overlap IN_CRIB


def test_the_default_object_vocabulary_is_not_bottle_only():
    """The literal regression: only 'bottle' reached the detector's class filter."""
    module = _module()
    from core.accel import COCO_NAMES
    labels = {COCO_NAMES[cid] for cid in module._object_class_ids}
    for expected in ("cell phone", "cup", "banana", "apple"):
        assert expected in labels, f"{expected} should be requested from the detector by default"


def test_a_phone_overlapping_the_child_confirms_holding_object():
    module = _module(holding_object_confirm_s=1.0)
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.5,
        object_boxes={"phone": [PHONE_IN_CRIB]},
    )
    assert "holding_object" in [e.type for e in events]
    fired = next(e for e in events if e.type == "holding_object")
    assert "object=phone" in fired.contributing_signals


def test_a_phone_elsewhere_in_frame_does_not_confirm():
    """Same honest 'overlap, not bare presence' rule holding_bottle already applies."""
    module = _module(holding_object_confirm_s=1.0)
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.5,
        object_boxes={"phone": [PHONE_ELSEWHERE]},
    )
    assert "holding_object" not in [e.type for e in events]


def test_holding_object_resolves_once_the_object_moves_away():
    module = _module(holding_object_confirm_s=1.0)
    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.5,
        object_boxes={"phone": [PHONE_IN_CRIB]},
    )
    assert "holding_object" in [e.type for e in events]

    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=3.0, start=t,
        object_boxes={"phone": [PHONE_ELSEWHERE]},
    )
    assert "holding_object_resolved" in [e.type for e in events]


def test_holding_bottle_and_holding_object_are_independent():
    """A bottle AND a held phone must both be reportable -- one shouldn't mask the other."""
    module = _module(holding_bottle_confirm_s=1.0, holding_object_confirm_s=1.0)
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.5,
        bottle_boxes=[BOTTLE_IN_CRIB], object_boxes={"phone": [PHONE_IN_CRIB]},
    )
    types = [e.type for e in events]
    assert "holding_bottle" in types
    assert "holding_object" in types


def test_keys_are_honestly_not_in_the_default_vocabulary():
    """COCO has no 'keys' class -- claiming one would be exactly the kind of
    faked capability this module's own docstring refuses to ship."""
    from modules.baby_monitoring import DEFAULT_OBJECT_CLASSES
    assert "keys" not in DEFAULT_OBJECT_CLASSES
    assert "key" not in DEFAULT_OBJECT_CLASSES
