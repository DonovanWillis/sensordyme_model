"""Tests for the baby monitoring module's reasoning layer (no camera, no YOLO)."""

import numpy as np
import pytest

from core.activity import ActivityState, Attribute
from core.hand_perception import HandPerceptionResult, UNKNOWN_OBJECT
from core.tracking import TrackedBox
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


# --------------------------------------------------------- the reported cooldown bug
# Before the per-label fix, a single shared "holding_object" engine key meant putting
# an object down (which resolves after resolve_after_s=2.0 and then serves a 30s-
# default cooldown) could silently swallow a COMPLETELY DIFFERENT object picked up
# moments later, because the cooldown was keyed by the rule name, not by what was
# actually being held.
CUP_IN_CRIB = (460, 350, 520, 500)  # overlaps IN_CRIB, same box as PHONE_IN_CRIB


def test_putting_down_one_object_and_picking_up_a_different_one_is_not_suppressed():
    module = _module(holding_object_confirm_s=0.5, holding_cooldown_s=3.0)

    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.0, object_boxes={"phone": [PHONE_IN_CRIB]}
    )
    assert "holding_object" in [e.type for e in events]

    # Phone put down (moves out of overlap) -- resolves after resolve_after_s=2.0.
    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=2.5, start=t, object_boxes={"phone": [PHONE_ELSEWHERE]}
    )
    assert "holding_object_resolved" in [e.type for e in events]

    # Immediately (well within the old 30s-default / even within this test's own
    # 3s holding_cooldown_s) a DIFFERENT class is picked up -- must be detected,
    # not swallowed by the phone's own cooldown.
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.0, start=t, object_boxes={"cup": [CUP_IN_CRIB]}
    )
    fired = [e for e in events if e.type == "holding_object"]
    assert fired, "a different object class must confirm immediately, not be suppressed by phone's cooldown"
    assert "object=cup" in fired[0].contributing_signals


def test_holding_cooldown_s_is_configurable_and_applies_per_label():
    """The SAME class picked back up quickly still respects its own cooldown
    (this was never the reported bug -- flapping on one object is legitimately
    debounced) -- but with a short cooldown configured, it re-confirms well
    inside a test, proving the cooldown is real and per-label, not removed."""
    module = _module(holding_object_confirm_s=0.3, holding_cooldown_s=0.5)

    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=0.6, object_boxes={"phone": [PHONE_IN_CRIB]}
    )
    assert "holding_object" in [e.type for e in events]

    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=2.5, start=t, object_boxes={"phone": [PHONE_ELSEWHERE]}
    )
    assert "holding_object_resolved" in [e.type for e in events]

    # Within the cooldown: picking the SAME class back up must not re-confirm yet.
    events, t = _run(
        module, [IN_CRIB], motion=0.05, seconds=0.3, start=t, object_boxes={"phone": [PHONE_IN_CRIB]}
    )
    assert "holding_object" not in [e.type for e in events]

    # Past the (short, configured) cooldown: it can re-confirm.
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.0, start=t + 0.5, object_boxes={"phone": [PHONE_IN_CRIB]}
    )
    assert "holding_object" in [e.type for e in events]


def test_two_different_objects_held_at_once_are_both_reported():
    """Per-label keys mean a phone AND a cup held simultaneously are two
    independent, simultaneously-active detections -- not a single
    'whichever was found first' winner."""
    module = _module(holding_object_confirm_s=0.5)
    events, _ = _run(
        module, [IN_CRIB], motion=0.05, seconds=1.0,
        object_boxes={"phone": [PHONE_IN_CRIB], "cup": [CUP_IN_CRIB]},
    )
    labels = {
        e.contributing_signals.split("object=")[1].split()[0]
        for e in events if e.type == "holding_object"
    }
    assert labels == {"phone", "cup"}


# ------------------------------------------------------------------------- flailing
def _flailing_state(present=True, confidence=0.8):
    return ActivityState(flailing=Attribute("flailing", present, confidence), ready=True)


def test_flailing_confirms_after_persistence():
    module = _module(flailing_confirm_s=0.5)
    events = []
    t = 0.0
    while t < 1.0:
        events += module._reason(
            [IN_CRIB], 0.05, W, H, t, activity_state=_flailing_state(),
        )
        t += 0.1
    assert "flailing" in [e.type for e in events]
    assert module.is_alert_active()


def test_flailing_does_not_fire_on_a_brief_burst():
    module = _module(flailing_confirm_s=0.5)
    events = []
    t = 0.0
    while t < 0.3:  # under flailing_confirm_s
        events += module._reason([IN_CRIB], 0.05, W, H, t, activity_state=_flailing_state())
        t += 0.1
    assert "flailing" not in [e.type for e in events]


def test_flailing_is_false_when_activity_is_not_ready():
    module = _module(flailing_confirm_s=0.2)
    not_ready = ActivityState(ready=False)
    events = []
    t = 0.0
    while t < 1.0:
        events += module._reason([IN_CRIB], 0.05, W, H, t, activity_state=not_ready)
        t += 0.1
    assert "flailing" not in [e.type for e in events]


def test_flailing_defaults_to_false_with_no_activity_state_at_all():
    """process()-less callers (and the old _reason() signature) must not crash
    or spuriously flag flailing when no pose data is supplied."""
    module = _module(flailing_confirm_s=0.2)
    events = []
    t = 0.0
    while t < 1.0:
        events += module._reason([IN_CRIB], 0.05, W, H, t)  # no activity_state kwarg at all
        t += 0.1
    assert "flailing" not in [e.type for e in events]


# ------------------------------------------------------------- human-object interaction
def test_an_object_moving_toward_the_child_fires_object_approaching_before_it_overlaps():
    """Trajectory, not proximity: a tracked item closing in on the child, still
    short of any overlap, is a real signal a single-frame overlap test can't
    see at all -- the anticipatory half of "about to be picked up"."""
    module = _module()
    t = 0.0
    events = []
    x = 750  # object box never overlaps IN_CRIB=(400,300,600,800) at any step below
    for _ in range(10):
        events += module._reason(
            [IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [(x, 530, x + 60, 590)]},
        )
        x -= 20  # closing in on the child every frame
        t += 0.1
    fired = [e for e in events if e.type == "object_approaching"]
    assert fired, "a phone tracked moving toward the child must fire object_approaching"
    assert "object=phone" in fired[0].contributing_signals


def test_a_stationary_object_never_fires_approaching():
    module = _module()
    events = []
    t = 0.0
    for _ in range(10):
        events += module._reason(
            [IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [(750, 530, 810, 590)]},
        )
        t += 0.1
    assert "object_approaching" not in [e.type for e in events]


def test_an_object_moving_away_never_fires_approaching():
    module = _module()
    events = []
    t = 0.0
    x = 650
    for _ in range(10):
        events += module._reason(
            [IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [(x, 530, x + 60, 590)]},
        )
        x += 20  # receding
        t += 0.1
    assert "object_approaching" not in [e.type for e in events]


def test_holding_confidence_is_fused_not_a_flat_constant():
    """Confidence at the moment holding CONFIRMS reflects how long that exact
    item had already been reliably tracked beforehand (TrackedBox.stability),
    not a hardcoded 0.7 regardless of track history: an item tracked nearby
    for a while before it starts overlapping the child must confirm with
    higher confidence than one that appears already overlapping with no
    track history at all."""
    brief = _module(holding_object_confirm_s=0.1)
    events = []
    t = 0.0
    for _ in range(4):  # crosses min_frames=3 with no prior track history
        events += brief._reason([IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [PHONE_IN_CRIB]})
        t += 0.1
    brief_conf = next(e.confidence for e in events if e.type == "holding_object")

    established = _module(holding_object_confirm_s=0.1)
    events = []
    t = 0.0
    NEAR_CRIB_NOT_OVERLAPPING = (620, 350, 680, 500)  # close enough to stay the SAME track
    # once it moves into PHONE_IN_CRIB below (within BoxTracker's match gate),
    # but not yet overlapping IN_CRIB=(400,300,600,800) (x1=620 > x2=600)
    for _ in range(20):  # tracked nearby (NOT overlapping) for ~2s first
        events += established._reason(
            [IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [NEAR_CRIB_NOT_OVERLAPPING]},
        )
        t += 0.1
    for _ in range(4):  # then moves into overlap -- same track, now well-established
        events += established._reason([IN_CRIB], 0.05, W, H, t, object_boxes={"phone": [PHONE_IN_CRIB]})
        t += 0.1
    established_conf = next(e.confidence for e in events if e.type == "holding_object")

    assert established_conf > brief_conf


def test_enable_pose_false_never_touches_mediapipe_pose(monkeypatch):
    """A deployment can fully opt out of the extra pose inference; process()
    must not construct or call a Pose model at all in that case."""
    module = _module(enable_pose=False)
    assert module._pose is None
    monkeypatch.setattr(module, "_detect_people", lambda frame: [IN_CRIB])
    events = module.process(np.zeros((H, W, 3), dtype=np.uint8))  # must not raise
    assert isinstance(events, list)


# ------------------------------------------------------------- hand-anchored perception
# _reason()'s new `hand_result` kwarg is fed directly with a synthetic
# HandPerceptionResult here (core/hand_perception.py's own 28 tests cover the
# engine's internal reasoning -- hand ROI sizing, association scoring,
# adaptive scheduling); these test the INTEGRATION layer: what
# modules/baby_monitoring.py does with an association once it has one.
def _hand_result(track_id=1, label="phone", side="right", score=0.9, box=(460, 350, 520, 500)):
    track = TrackedBox(
        track_id=track_id, label=label, box=box, confidence=score,
        first_seen=0.0, last_seen=0.0, hits=5,
    )
    return HandPerceptionResult(hands=[], objects=[track], associations={f"{side}_hand": (track, score)})


def _run_hand(module, boxes, motion, seconds, hand_result_fn, start=0.0, step=0.1,
              bottle_boxes=(), object_boxes=None):
    events = []
    t = start
    while t < start + seconds:
        events += module._reason(
            list(boxes), motion, W, H, t, list(bottle_boxes), object_boxes,
            hand_result=hand_result_fn(t),
        )
        t += step
    return events, t


def test_hand_associated_object_confirms_holding_object_and_fires_picked_up():
    """A hand-anchored association alone -- with NO full-frame bottle/object
    box at all -- must be able to drive holding_object AND the new, more
    specific object_picked_up event. This is the whole point of hand-anchored
    perception: catching what the full-frame pass misses."""
    module = _module(holding_object_confirm_s=1.0, hand_hold_confirm_s=1.0)
    events, _ = _run_hand(module, [IN_CRIB], 0.05, 1.5, lambda t: _hand_result())
    types = [e.type for e in events]
    assert "holding_object" in types
    assert "object_picked_up" in types
    picked = next(e for e in events if e.type == "object_picked_up")
    assert "object=phone" in picked.contributing_signals
    assert "hand=right" in picked.contributing_signals


def test_hand_associated_bottle_confirms_holding_bottle():
    module = _module(holding_bottle_confirm_s=1.0, hand_hold_confirm_s=1.0)
    events, _ = _run_hand(
        module, [IN_CRIB], 0.05, 1.5, lambda t: _hand_result(label="bottle", side="left"),
    )
    assert "holding_bottle" in [e.type for e in events]


def test_object_put_down_fires_when_the_hand_association_ends():
    module = _module(hand_hold_confirm_s=0.5)
    events, t = _run_hand(module, [IN_CRIB], 0.05, 1.0, lambda tt: _hand_result())
    assert "object_picked_up" in [e.type for e in events]

    events, _ = _run_hand(module, [IN_CRIB], 0.05, 3.0, lambda tt: None, start=t)
    types = [e.type for e in events]
    assert "object_put_down" in types
    put_down = next(e for e in events if e.type == "object_put_down")
    assert "object=phone" in put_down.contributing_signals
    assert "hand=right" in put_down.contributing_signals


def test_unknown_object_in_hand_is_reported_honestly_not_hallucinated_into_a_class():
    """UNKNOWN_OBJECT must surface as its own honest event -- and must NOT be
    silently folded into holding_object/holding_bottle for some configured
    label just because the engine found *something* in the hand."""
    module = _module(hand_hold_confirm_s=0.5)
    events, _ = _run_hand(
        module, [IN_CRIB], 0.05, 1.0, lambda t: _hand_result(label=UNKNOWN_OBJECT),
    )
    types = [e.type for e in events]
    assert "object_picked_up" in types
    picked = next(e for e in events if e.type == "object_picked_up")
    assert "object=unknown_object" in picked.contributing_signals
    assert "holding_object" not in types
    assert "holding_bottle" not in types


def test_object_transferred_between_hands_within_the_window():
    module = _module(hand_hold_confirm_s=0.3, object_transfer_window_s=2.0)
    events, t = _run_hand(module, [IN_CRIB], 0.05, 0.6, lambda tt: _hand_result(track_id=7, side="left"))
    assert "object_picked_up" in [e.type for e in events]

    # Left hand releases it -- resolve_after_s=2.0 for object_in_hand.
    events, t = _run_hand(module, [IN_CRIB], 0.05, 2.2, lambda tt: None, start=t)
    assert "object_put_down" in [e.type for e in events]

    # The SAME track id confirms in the OTHER hand shortly after: a transfer,
    # not an unrelated pickup.
    events, _ = _run_hand(
        module, [IN_CRIB], 0.05, 0.6, lambda tt: _hand_result(track_id=7, side="right"), start=t,
    )
    types = [e.type for e in events]
    assert "object_transferred" in types
    assert "object_picked_up" not in types
    transferred = next(e for e in events if e.type == "object_transferred")
    assert "left->right" in transferred.contributing_signals
    assert "object=phone" in transferred.contributing_signals


def test_transfer_outside_the_window_is_two_separate_pickups_not_a_transfer():
    module = _module(hand_hold_confirm_s=0.3, object_transfer_window_s=0.5)
    events, t = _run_hand(module, [IN_CRIB], 0.05, 0.6, lambda tt: _hand_result(track_id=9, side="left"))
    assert "object_picked_up" in [e.type for e in events]

    events, t = _run_hand(module, [IN_CRIB], 0.05, 2.2, lambda tt: None, start=t)
    assert "object_put_down" in [e.type for e in events]

    t += 1.0  # well past the configured 0.5s transfer window
    events, _ = _run_hand(
        module, [IN_CRIB], 0.05, 0.6, lambda tt: _hand_result(track_id=9, side="right"), start=t,
    )
    types = [e.type for e in events]
    assert "object_picked_up" in types
    assert "object_transferred" not in types


def test_both_hands_holding_different_objects_are_independently_reported():
    module = _module(hand_hold_confirm_s=0.3, holding_object_confirm_s=0.3)

    def hands(t):
        left = TrackedBox(track_id=1, label="cup", box=(300, 300, 340, 340), confidence=0.9,
                           first_seen=0.0, last_seen=t, hits=5)
        right = TrackedBox(track_id=2, label="phone", box=(700, 300, 740, 340), confidence=0.9,
                            first_seen=0.0, last_seen=t, hits=5)
        return HandPerceptionResult(
            hands=[], objects=[left, right],
            associations={"left_hand": (left, 0.9), "right_hand": (right, 0.9)},
        )

    events, _ = _run_hand(module, [IN_CRIB], 0.05, 0.6, hands)
    picked = {e.contributing_signals for e in events if e.type == "object_picked_up"}
    assert any("object=cup" in s and "hand=left" in s for s in picked)
    assert any("object=phone" in s and "hand=right" in s for s in picked)
    held_labels = {
        e.contributing_signals.split("object=")[1].split()[0]
        for e in events if e.type == "holding_object"
    }
    assert held_labels == {"cup", "phone"}


def test_full_frame_corroboration_boosts_hand_association_confidence():
    # Deliberately NOT overlapping IN_CRIB, so the legacy overlap-with-child
    # fallback stays silent and this isolates the corroboration bonus itself.
    box = (850, 300, 910, 500)
    module = _module(holding_object_confirm_s=0.3)
    events, _ = _run_hand(module, [IN_CRIB], 0.05, 0.6, lambda t: _hand_result(score=0.6, box=box))
    base_conf = next(e.confidence for e in events if e.type == "holding_object")
    assert base_conf == pytest.approx(0.6)

    module2 = _module(holding_object_confirm_s=0.3)
    events2, _ = _run_hand(
        module2, [IN_CRIB], 0.05, 0.6, lambda t: _hand_result(score=0.6, box=box),
        object_boxes={"phone": [box]},
    )
    corroborated_conf = next(e.confidence for e in events2 if e.type == "holding_object")
    assert corroborated_conf == pytest.approx(0.75)


def test_no_hand_result_falls_back_to_the_legacy_overlap_signal():
    """hand_result=None (pose disabled, or the caller predates this feature)
    must not disable holding_object/holding_bottle -- the legacy full-frame
    overlap test alone must still be able to confirm them."""
    module = _module(holding_object_confirm_s=1.0)
    events, _ = _run_hand(
        module, [IN_CRIB], 0.05, 1.5, lambda t: None,
        object_boxes={"phone": [PHONE_IN_CRIB]},
    )
    assert "holding_object" in [e.type for e in events]
    assert "object_picked_up" not in [e.type for e in events]  # no hand data -> no hand-level event
