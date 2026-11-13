"""M8.4 behaviour engine units: typed evidence, abstention, phone evidence levels,
driver association, temporal reasoning in shadow, evaluation metrics.

Pure: no MediaPipe, no module. Synthetic inputs; says nothing about real cabins.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from core.behavior import signals as sig
from core.behavior.association import (
    AMBIGUOUS,
    ASSUMED_LARGEST,
    CONFIGURED_SEAT,
    FACE_ANCHORED,
    NO_PERSON,
    associate_driver,
)
from core.behavior.engine import (
    ATTENTION_LOST,
    DISTRACTION,
    DROWSINESS,
    EYES_DOWN,
    FRAME_BLOCKED,
    FRAME_DARK,
    FRAME_OK,
    PHONE_USE,
    BehaviorEngine,
    EngineConfig,
    FrameEvidence,
    frame_quality,
)
from core.behavior.evaluation import aggregate, score_scenario, wilson
from core.behavior.phone_evidence import (
    ACTIVE,
    HELD,
    INSUFFICIENT,
    INTERACTION,
    MOUNTED,
    NEAR_DRIVER,
    NO_PHONE,
    VISIBLE,
    PhoneEvidenceInput,
    classify_phone,
    phone_risk_for,
)

DT = 0.1
NO_PHONE_INPUT = PhoneEvidenceInput(detector_ok=True)


def _frame(t: float, *, face: bool = True, eye: Optional[str] = "open", eye_reason: str = "eyes occluded",
           gaze: Optional[tuple[float, float]] = (0.0, 0.0), gaze_reason: str = "face not detected",
           person: Optional[int] = 1, quality: str = FRAME_OK,
           phone: PhoneEvidenceInput = NO_PHONE_INPUT) -> FrameEvidence:
    signals = {
        sig.FACE: sig.observed(sig.FACE, face, t),
        sig.FRAME: sig.observed(sig.FRAME, quality, t),
        sig.EYE_STATE: (sig.observed(sig.EYE_STATE, eye, t) if face and eye is not None
                        else sig.missing(sig.EYE_STATE, eye_reason if face else "face not detected", t)),
        sig.GAZE: (sig.observed(sig.GAZE, gaze, t, calibrated=True) if face and gaze is not None
                   else sig.missing(sig.GAZE, gaze_reason, t)),
        sig.PERSON: (sig.observed(sig.PERSON, person, t) if person is not None
                     else sig.missing(sig.PERSON, "object detection failed", t, capability_valid=False)),
    }
    return FrameEvidence(t=t, signals=signals, phone=phone)


def _run(engine: BehaviorEngine, start: float, seconds: float, **kwargs) -> tuple[float, list[str]]:
    started: list[str] = []
    t = start
    for _ in range(round(seconds / DT)):
        engine.update(_frame(t, **kwargs))
        started += engine.started
        t = round(t + DT, 6)
    return t, started


# ============================================================ evidence representation
def test_missing_evidence_must_say_why_and_is_never_a_value():
    with pytest.raises(ValueError, match="must say why"):
        sig.SignalEvidence(sig.EYE_STATE, None, 0.0)
    with pytest.raises(ValueError, match="has a value and a missing_reason"):
        sig.SignalEvidence(sig.EYE_STATE, "open", 0.0, missing_reason="x")
    ev = sig.missing(sig.EYE_STATE, "face not detected", 5.0, observed_at=4.0)
    assert ev.missing and not ev.usable()
    assert ev.why_unusable() == "face not detected"
    assert ev.freshness_s == pytest.approx(1.0)
    assert ev.source == "face"


def test_staleness_calibration_and_capability_make_evidence_unusable():
    old = sig.SignalEvidence(sig.PERSON, 1, now=10.0, observed_at=8.0)
    assert old.usable() and not old.usable(max_age_s=1.0)
    assert "stale" in old.why_unusable(1.0)
    assert not sig.observed(sig.GAZE, (0.0, 0.0), 0.0, calibrated=False).usable()
    assert not sig.observed(sig.PERSON, 1, 0.0, capability_valid=False).usable()
    assert sig.missing(sig.GAZE, "x", 0.0).freshness_s == math.inf
    d = sig.observed(sig.GAZE, (1.234, -5.0), 2.0, confidence=0.5, uncertainty=1.0).to_dict()
    assert d["value"] == [1.2, -5.0] and d["confidence"] == 0.5 and d["missing"] is False


def test_an_abstention_must_say_why():
    with pytest.raises(ValueError, match="must say why"):
        sig.Verdict(DROWSINESS, sig.ABSTAIN)
    with pytest.raises(ValueError, match="unknown verdict state"):
        sig.Verdict(DROWSINESS, "maybe", "x")


def test_continuity_restarts_on_change_and_credits_nothing_across_a_stall():
    c = sig.Continuity(max_gap_s=0.5)
    assert c.update(0.0, "open") == 0.0
    assert c.update(0.1, "open") == pytest.approx(0.1)
    assert c.update(3.0, "open") == pytest.approx(0.1)  # 2.9 s stall: held, not grown
    assert c.update(3.1, "closed") == 0.0


# ============================================================ unavailable is never negative
def test_a_lost_face_is_not_drowsy_and_not_alert_it_is_abstention():
    engine = BehaviorEngine()
    _run(engine, 0.0, 1.0)
    engine.update(_frame(1.0, face=False, person=1))
    v = engine.verdicts
    assert v[DROWSINESS].state == sig.ABSTAIN and "face not detected" in v[DROWSINESS].reason
    assert v[DISTRACTION].state == sig.ABSTAIN and v[EYES_DOWN].state == sig.ABSTAIN
    assert set(engine.abstentions()) >= {DROWSINESS, DISTRACTION, EYES_DOWN}


def test_occluded_eyes_abstain_for_drowsiness_for_the_whole_time():
    engine = BehaviorEngine()
    _, started = _run(engine, 0.0, 10.0, eye=None, eye_reason="eyes occluded (sunglasses or obstruction)")
    assert DROWSINESS not in started
    assert engine.verdicts[DROWSINESS].state == sig.ABSTAIN
    assert engine.comparison()[DROWSINESS]["abstain_rate"] == 1.0


def test_a_phone_detection_is_not_phone_use():
    engine = BehaviorEngine()
    visible = PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="unknown",
                                 eyes_off_road=False)
    _, started = _run(engine, 0.0, 10.0, phone=visible)
    assert PHONE_USE not in started
    assert engine.phone_evidence.level == NEAR_DRIVER
    assert engine.verdicts[PHONE_USE].state == sig.NOT_SUPPORTED


def test_one_downward_pose_is_not_distraction_and_a_glance_is_not_sustained():
    engine = BehaviorEngine()
    engine.update(_frame(0.0))
    engine.update(_frame(0.1, gaze=(0.0, -40.0)))
    engine.update(_frame(0.2))
    assert engine.verdicts[EYES_DOWN].state == sig.NOT_SUPPORTED
    _t, started = _run(engine, 0.3, 1.5, gaze=(0.0, -30.0))  # a 1.5 s dashboard glance
    assert EYES_DOWN not in started and engine.verdicts[EYES_DOWN].reason.startswith("glance")


# ============================================================ temporal reasoning (same bars as production)
def test_sustained_conditions_fire_at_the_shipped_bars():
    cfg = EngineConfig()
    engine = BehaviorEngine(cfg)
    t, _ = _run(engine, 0.0, 1.0)
    on = t
    started_at = None
    while started_at is None and t < on + 10:
        engine.update(_frame(t, gaze=(45.0, 0.0)))
        if DISTRACTION in engine.started:
            started_at = t
        t = round(t + DT, 6)
    assert started_at is not None and started_at - on == pytest.approx(cfg.distraction_seconds, abs=1e-6)


def test_a_wide_mirror_check_gets_the_rearview_patience():
    engine = BehaviorEngine()
    _, started = _run(engine, 0.0, 4.0, gaze=(60.0, 0.0))  # > 3 s but < 5 s at a mirror angle
    assert DISTRACTION not in started
    _, started = _run(engine, 4.0, 1.5, gaze=(60.0, 0.0))
    assert DISTRACTION in started


def test_brief_returns_inside_the_release_grace_do_not_restart_the_clock():
    engine = BehaviorEngine()
    t, _ = _run(engine, 0.0, 2.0, gaze=(45.0, 0.0))
    engine.update(_frame(t))  # one noisy on-road frame
    t, started = _run(engine, round(t + DT, 6), 1.2, gaze=(45.0, 0.0))
    assert DISTRACTION in started


def test_a_stall_credits_no_time_and_missing_gaze_coasts_only_briefly():
    engine = BehaviorEngine()
    t, _ = _run(engine, 0.0, 2.0, gaze=(45.0, 0.0))
    _, started = _run(engine, t + 4.0, 0.5, gaze=(45.0, 0.0))  # 4 s stall: no frames, no credit
    assert DISTRACTION not in started
    engine2 = BehaviorEngine()
    t, _ = _run(engine2, 0.0, 2.5, gaze=(45.0, 0.0))
    t, _ = _run(engine2, t, 2.0, face=True, gaze=None, gaze_reason="head pose fit failed")
    assert engine2.verdicts[DISTRACTION].state == sig.ABSTAIN  # coast exhausted: progress dropped
    _, started = _run(engine2, t, 1.0, gaze=(45.0, 0.0))
    assert DISTRACTION not in started


def test_a_motionless_offroad_pose_for_long_reads_as_a_moved_camera():
    engine = BehaviorEngine(EngineConfig(misalignment_s=10.0))
    _, started = _run(engine, 0.0, 15.0, gaze=(40.0, 0.0))
    assert DISTRACTION in started  # at 3 s a moved camera and a turned head are indistinguishable
    assert engine.verdicts[DISTRACTION].state == sig.ABSTAIN
    assert "camera may have moved" in engine.verdicts[DISTRACTION].reason
    assert engine.misalignment_suspected


def test_eye_closure_survives_a_brief_face_loss_but_not_a_long_one():
    engine = BehaviorEngine()
    t, _ = _run(engine, 0.0, 1.5, eye="closed")
    t, _ = _run(engine, t, 0.5, face=False)
    _, started = _run(engine, t, 0.7, eye="closed")
    assert DROWSINESS in started
    engine2 = BehaviorEngine()
    t, _ = _run(engine2, 0.0, 1.5, eye="closed")
    t, _ = _run(engine2, t, 3.0, face=False)
    _, started = _run(engine2, t, 0.7, eye="closed")
    assert DROWSINESS not in started


def test_attention_lost_names_its_cause_and_abstains_on_an_empty_vehicle():
    engine = BehaviorEngine(EngineConfig(empty_after_s=10.0))
    t, started = _run(engine, 0.0, 3.0, face=False, person=0)
    assert ATTENTION_LOST in started
    assert engine.verdicts[ATTENTION_LOST].reason == "no face and no person"
    _run(engine, t, 10.0, face=False, person=0)
    assert engine.verdicts[ATTENTION_LOST].state == sig.ABSTAIN
    assert "vehicle may be empty" in engine.verdicts[ATTENTION_LOST].reason
    blocked = BehaviorEngine()
    _, started = _run(blocked, 0.0, 3.0, face=False, person=0, quality=FRAME_BLOCKED)
    assert ATTENTION_LOST in started and "blocked" in blocked.verdicts[ATTENTION_LOST].reason
    turned = BehaviorEngine()
    _, started = _run(turned, 0.0, 5.0, face=False, person=1)
    assert ATTENTION_LOST not in started  # the patient bar while a body is seen


def test_shadow_vs_production_comparison_counts_and_times():
    engine = BehaviorEngine()
    engine.update(_frame(0.0))
    _run(engine, 0.1, 3.5, gaze=(45.0, 0.0))
    engine.record_production(["distraction"], t=3.1)
    engine.record_production(["phone_use"], t=3.1)
    comp = engine.comparison()
    assert comp[DISTRACTION]["matched"] == 1 and comp[DISTRACTION]["mean_shadow_minus_production_s"] == 0.0
    assert comp[PHONE_USE]["production_only"] == 1
    engine.reset_stats()
    assert engine.comparison()[DISTRACTION]["production"] == 0


def test_frame_quality():
    assert frame_quality(3.0, 1.0) == FRAME_BLOCKED
    assert frame_quality(25.0, 12.0) == FRAME_DARK
    assert frame_quality(120.0, 30.0) == FRAME_OK
    assert frame_quality(float("nan"), 1.0) == FRAME_BLOCKED


# ============================================================ phone evidence (decision B)
@pytest.mark.parametrize(("inp", "level"), [
    (PhoneEvidenceInput(detector_ok=False, detector_reason="failed"), INSUFFICIENT),
    (PhoneEvidenceInput(detector_ok=True), NO_PHONE),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_known=False), INSUFFICIENT),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=False), VISIBLE),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="not_held",
                        eyes_off_road=True, held_off_road_s=9.0), MOUNTED),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="unknown",
                        eyes_off_road=True, held_off_road_s=9.0), NEAR_DRIVER),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held",
                        eyes_off_road=False), HELD),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held",
                        eyes_off_road=True, gaze_toward_phone=True), INTERACTION),
    (PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held",
                        eyes_off_road=True, held_off_road_s=2.0), ACTIVE),
])
def test_phone_evidence_levels(inp, level):
    assert classify_phone(inp).level == level


def test_a_visible_mounted_or_unheld_phone_never_contributes_risk():
    for level_input in (
        PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=False),
        PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="not_held"),
        PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="unknown"),
        PhoneEvidenceInput(detector_ok=False),
    ):
        evidence = classify_phone(level_input)
        for off_road in (True, False, None):
            assert phone_risk_for(evidence, off_road) == 0.0, (evidence, off_road)
    held = classify_phone(PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held",
                                             eyes_off_road=False))
    assert phone_risk_for(held, False) == 0.0  # held, eyes on the road: no attention evidence
    assert phone_risk_for(held, True) == 0.6
    active = classify_phone(PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held",
                                               eyes_off_road=True, held_off_road_s=3.0))
    assert phone_risk_for(active, None) == 1.0


def test_engine_phone_use_needs_a_held_phone_and_abstains_when_holding_is_unknown():
    engine = BehaviorEngine()
    held = PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="held", eyes_off_road=True)
    _, started = _run(engine, 0.0, 2.5, gaze=(0.0, -30.0), phone=held)
    assert PHONE_USE in started and engine.phone_evidence.level == ACTIVE
    unknown = BehaviorEngine()
    near = PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="unknown",
                              eyes_off_road=True)
    _, started = _run(unknown, 0.0, 5.0, gaze=(0.0, -30.0), phone=near)
    assert PHONE_USE not in started
    assert unknown.verdicts[PHONE_USE].state == sig.ABSTAIN
    mounted = BehaviorEngine()
    mount = PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold="not_held",
                               eyes_off_road=True)
    _, started = _run(mounted, 0.0, 5.0, gaze=(0.0, -30.0), phone=mount)
    assert PHONE_USE not in started and mounted.verdicts[PHONE_USE].state == sig.NOT_SUPPORTED
    assert EYES_DOWN in started  # still eyes off the road, by its own rule


# ============================================================ driver association (A4)
DRIVER = (200.0, 60.0, 460.0, 470.0)
PASSENGER = (0.0, 120.0, 190.0, 470.0)
LEANING = (0.0, 0.0, 300.0, 480.0)
FACE = (274.0, 190.0, 366.0, 308.0)


def test_driver_association_states_its_method():
    assert associate_driver([]).method == NO_PERSON
    assert associate_driver([DRIVER, PASSENGER]).method == ASSUMED_LARGEST
    anchored = associate_driver([DRIVER, LEANING], FACE)
    assert anchored.method == FACE_ANCHORED and anchored.driver_box == DRIVER
    assert anchored.agrees_with_largest is False  # the shipped largest-person proxy would pick the passenger
    assert associate_driver([DRIVER, PASSENGER], FACE).agrees_with_largest is True
    seat = associate_driver([DRIVER, LEANING], None, seat_region=(160.0, 0.0, 640.0, 480.0))
    assert seat.method == CONFIGURED_SEAT and seat.driver_box == DRIVER
    overlapping = associate_driver([DRIVER, (180.0, 40.0, 480.0, 480.0)], FACE)
    assert overlapping.method == AMBIGUOUS and overlapping.driver_box is None
    assert anchored.to_dict()["accuracy"] == "unevaluated"


# ============================================================ evaluation metrics (A6)
def test_wilson_interval():
    assert wilson(0, 0) is None
    low, high = wilson(8, 10)  # type: ignore[misc]
    assert 0.44 < low < 0.5 and 0.94 < high < 0.97


def test_event_level_scoring():
    events = [{"t": 2.5, "type": "drowsiness"}, {"t": 9.0, "type": "phone_use"},
              {"t": 1.0, "type": "attention_lost"}, {"t": 1.0, "type": "phone_detected"}]
    scored = score_scenario(events, [("drowsiness", 0.5, 3.0), ("eyes_down", 4.0, 6.0)], ["attention_lost"])
    assert (scored["tp"], scored["fp"], scored["fn"]) == (1, 1, 1)
    assert scored["delays_s"] == [2.0] and scored["false_positives"] == ["phone_use@9.0"]
    total = aggregate([scored], [3600.0])
    assert total["precision"] == 0.5 and total["recall"] == 0.5 and total["false_alerts_per_hour"] == 1.0
