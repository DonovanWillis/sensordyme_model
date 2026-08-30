"""Phase 1 safety regression tests for modules/baby_monitoring.py.

Like tests/test_phase1_safety_fall.py, every test here is a MISSED-DETECTION
case. The choking-hazard tests in particular assert against the numbers
HARDWARE TESTING actually produced (config.yaml records that a genuine held
object near the face peaks around 0.38 association) rather than against the
optimistic 0.85-0.9 association the existing suite uses -- the gap between
those two numbers is precisely what made the alert arithmetically
unreachable in the field while every test passed.
"""

from __future__ import annotations

import pytest

from core.detectors import upper_region
from core.hand_perception import HandPerceptionResult
from core.tracking import TrackedBox
from modules.baby_monitoring import (
    MOUTH_ZONE_FRACTION,
    BabyMonitoringModule,
    _assess_choking_hazard,
    _infer_baby_box,
)

W, H = 1000, 1000
IN_CRIB = (400, 300, 600, 800)
ZONES = {"crib": {"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]], "kind": "safe"}}

#: config.yaml:476-482 records this from hardware testing: a hand near the
#: FACE produces a smaller pose-derived ROI than a hand at chest/lap height,
#: and a REAL held object there peaked at this association score. risk_score
#: multiplies this value directly, so it is the hard ceiling on choking risk.
MEASURED_ASSOCIATION_CEILING = 0.38

# For IN_CRIB with MOUTH_ZONE_FRACTION=0.3: upper_region -> (400, 300, 600, 450).
MOUTH_CENTER = (500.0, 375.0)


def _module(**overrides):
    """Deliberately NOT passing the tuning knobs under test: this module is
    built from its own in-code defaults so these tests measure what a
    deployment that never edits config.yaml actually gets."""
    config = {"zones": ZONES, "zone_confirm_s": 1.0, "still_max_motion": 0.015}
    config.update(overrides)
    return BabyMonitoringModule(config)


def _converging(start=(760.0, 660.0), end=MOUTH_CENTER, duration=0.6, half_size=20,
                label="bottle", score=MEASURED_ASSOCIATION_CEILING,
                hand_box=(440, 330, 560, 460)):
    """An object carried on a straight line into the mouth region and held
    there -- a realistic multi-frame closing approach, with the association
    score pinned to whatever the caller says the hardware really produces."""
    trail: list[tuple[float, float, float]] = []

    def hand_result_fn(t):
        frac = min(1.0, t / duration) if duration > 0 else 1.0
        x = start[0] + (end[0] - start[0]) * frac
        y = start[1] + (end[1] - start[1]) * frac
        trail.append((t, x, y))
        if len(trail) > 15:
            trail.pop(0)
        box = (int(x - half_size), int(y - half_size), int(x + half_size), int(y + half_size))
        obj = TrackedBox(track_id=1, label=label, box=box, confidence=score,
                         first_seen=trail[0][0], last_seen=t, hits=5, trail=list(trail))
        hand = TrackedBox(track_id=100, label="right_hand", box=hand_box, confidence=0.9,
                          first_seen=0.0, last_seen=t, hits=5)
        return HandPerceptionResult(hands=[hand], objects=[obj],
                                    associations={"right_hand": (obj, score)})

    return hand_result_fn


def _run(module, boxes, motion, seconds, start=0.0, step=0.1, hand_result_fn=None,
         object_boxes=None):
    events = []
    t = start
    while t < start + seconds - 1e-9:
        events += module._reason(
            list(boxes), motion, W, H, t, (), object_boxes,
            hand_result=hand_result_fn(t) if hand_result_fn else None,
        )
        t = round(t + step, 6)
    return events, t


# ------------------------------------------------------------------ BUG 4
def test_choking_hazard_is_reachable_at_the_measured_association_ceiling():
    """The headline regression: risk_score = association * size * proximity *
    trajectory, and every factor after the first is <= 1.0, so risk can never
    exceed the association score. With hardware measuring a genuine held
    object near the face at ~0.38, a 0.4 risk threshold and EventEngine's
    default 0.50 min_confidence made this critical alert arithmetically
    impossible to fire, in every deployment, for every real hazard.
    """
    module = _module()
    events, _ = _run(module, [IN_CRIB], 0.05, 1.8, hand_result_fn=_converging())

    types = [e.type for e in events]
    assert "choking_hazard" in types, (
        "a small object genuinely held and genuinely closing on the mouth, at the "
        "association score hardware actually produces, must be able to alert"
    )
    fired = next(e for e in events if e.type == "choking_hazard")
    assert "object=bottle" in fired.contributing_signals
    assert module.is_alert_active()


def test_measured_risk_at_the_ceiling_case_clears_the_configured_gate():
    """Pins the ARITHMETIC, not just the event: the risk actually produced by
    the measured-ceiling scenario must sit above the module's own threshold.
    Running these numbers is exactly the step that was skipped when the
    unreachable gate was introduced."""
    module = _module()
    _run(module, [IN_CRIB], 0.05, 1.2, hand_result_fn=_converging())
    right = module._last_choking_assessments["right"]
    assert right.approaching is True
    assert right.size_factor == 1.0
    assert right.risk_score == pytest.approx(MEASURED_ASSOCIATION_CEILING, abs=0.02)
    assert right.risk_score >= module._choking_hazard_risk_threshold


def test_large_far_object_at_the_measured_ceiling_still_does_not_alert():
    """The paired negative for BUG 4: lowering the gate to a reachable number
    must not make an ordinary held toy, nowhere near the mouth, alert."""
    module = _module()
    far = _converging(start=(900.0, 900.0), end=(880.0, 890.0), duration=1.0,
                      half_size=120, label="object")
    events, _ = _run(module, [IN_CRIB], 0.05, 2.0, hand_result_fn=far)
    assert "choking_hazard" not in [e.type for e in events]


# ------------------------------------------------------------------ BUG 5
def test_a_baby_that_stops_being_detected_raises_person_absent_by_default():
    """A baby covered by a blanket, face-down in bedding, or rolled where the
    detector loses the small low-contrast box produces NO person boxes. With
    `expect_presence` defaulting to false the person_absent rule was never
    even registered, so the default and play_area presets raised ZERO events
    for ten minutes of a baby that had simply vanished from the frame."""
    module = _module(absence_warning_s=2.0)
    events, _ = _run(module, [IN_CRIB], 0.05, 1.0)
    events2, _ = _run(module, [], 0.0, 5.0, start=1.0)
    assert "person_absent" in [e.type for e in events2]


def test_person_absent_can_still_be_turned_off_explicitly():
    """The paired negative: a deployment that genuinely does not expect a
    person in frame must still be able to opt out."""
    module = _module(absence_warning_s=2.0, expect_presence=False)
    events, _ = _run(module, [], 0.0, 5.0)
    assert "person_absent" not in [e.type for e in events]


# ------------------------------------------------------------------ BUG 6
def test_unusual_stillness_does_not_fire_on_a_demonstrably_active_baby():
    """A baby moving 1s in every 3.5s emitted BOTH activity_burst AND
    unusual_stillness: EventEngine's occlusion_grace_s bridged the "not
    still" frames without resetting the held-time clock. For a stillness
    rule, "absent" means THE SUBJECT MOVED -- contradicting evidence, not
    missing evidence."""
    module = _module(still_warning_s=10.0)
    events = []
    t = 0.0
    while t < 60.0:
        # 1.0s of unmistakable movement (8x still_max_motion) out of every
        # 3.5s; the other 2.5s the baby holds position.
        moving = round(t % 3.5, 6) < 1.0
        events += module._reason([IN_CRIB], 0.12 if moving else 0.001, W, H, t)
        t = round(t + 0.1, 6)
    types = [e.type for e in events]
    assert "activity_burst" in types, "the movement must still be observed"
    assert "unusual_stillness" not in types


def test_unusual_stillness_still_fires_through_a_brief_detection_dropout():
    """The paired positive: a genuinely motionless baby must still confirm
    even though the detector loses the box for a moment -- BUG 6's fix must
    reset stillness on observed MOTION only, never on missing evidence."""
    module = _module(still_warning_s=5.0)
    events, t = _run(module, [IN_CRIB], 0.001, 3.0)
    events2, t = _run(module, [], 0.001, 0.5, start=t)  # box lost briefly
    events3, _ = _run(module, [IN_CRIB], 0.001, 6.0, start=t)
    assert "unusual_stillness" in [e.type for e in events + events2 + events3]


# ===================================================================== PHASE 2
# PINNED KNOWN-WRONG BEHAVIOUR -- a red line for Phase 2, NOT a fix.
#
# _infer_baby_box picks the SMALLEST person box as the baby's. When a
# marginal-confidence infant box falls below the detector's threshold while a
# caregiver is in frame, it does not return None -- it returns the CAREGIVER's
# box, and choking-hazard risk is then scored against an adult's mouth region
# and body diagonal. That is a silent WRONG-TARGET failure, materially worse
# than a missed detection, and it is not covered anywhere else in the suite.
#
# These two tests PASS today. They exist to pin the current behaviour with
# real measured numbers so a Phase 2 fix has something to break, and so the
# limitation cannot be quietly forgotten. Fixing it needs the infant-box
# recall measurement (labelled crib-cam footage, confidence sweep) that has
# not been done -- guessing at the detector threshold instead of measuring it
# is the exact failure this phase exists to correct, so it is deliberately
# NOT fixed here.
INFANT_IN_CRIB = (500, 400, 700, 850)
#: Realistic caregiver placements, all LARGER in area than the infant box.
CAREGIVER_LEANING_OVER_CRIB = (450, 250, 800, 980)
CAREGIVER_ACROSS_THE_ROOM = (30, 500, 300, 950)
CAREGIVER_FAR_CORNER = (40, 600, 190, 880)


def _risk_scored_against(baby_box, association=MEASURED_ASSOCIATION_CEILING):
    """Risk for ONE fixed real hazard -- a small object carried to and held at
    the INFANT's mouth -- scored against whatever box is passed as the baby's.
    Only the target box varies; the hazard itself is identical every time."""
    mouth = upper_region(INFANT_IN_CRIB, MOUTH_ZONE_FRACTION)
    x, y = (mouth[0] + mouth[2]) / 2.0, (mouth[1] + mouth[3]) / 2.0
    trail = [(0.0, x + 250, y + 250), (1.0, x, y)]  # a genuine closing approach
    obj = TrackedBox(track_id=1, label="bottle", box=(int(x - 20), int(y - 20), int(x + 20), int(y + 20)),
                     confidence=association, first_seen=0.0, last_seen=1.0, hits=5, trail=trail)
    hand = TrackedBox(track_id=100, label="right_hand", box=(440, 330, 560, 460),
                      confidence=0.9, first_seen=0.0, last_seen=1.0, hits=5)
    return _assess_choking_hazard("right", (obj, association), hand, baby_box, 1.0).risk_score


def test_pins_infer_baby_box_returning_the_caregiver_when_the_infant_box_is_lost():
    """PINS KNOWN-WRONG BEHAVIOUR. Correct behaviour would be to return None
    (or otherwise refuse to score) rather than silently substitute an adult."""
    both = _infer_baby_box([CAREGIVER_ACROSS_THE_ROOM, INFANT_IN_CRIB])
    assert both == INFANT_IN_CRIB, "with both visible, the smallest box is the infant's -- correct"

    only_adult = _infer_baby_box([CAREGIVER_ACROSS_THE_ROOM])
    assert only_adult == CAREGIVER_ACROSS_THE_ROOM, (
        "WRONG-TARGET: the infant box was lost, so the caregiver's box is now "
        "'the baby'. Phase 2: this should not silently return an adult."
    )
    assert _infer_baby_box([]) is None  # no people at all is handled honestly


def test_pins_choking_risk_outcome_depending_on_where_the_caregiver_stands():
    """PINS KNOWN-WRONG BEHAVIOUR, and the part that actually matters: with
    the infant box lost, whether ONE fixed real hazard is caught depends on
    where the CAREGIVER happens to be standing -- the detection becomes
    non-deterministic with respect to the hazard itself. Measured values, gate
    0.30."""
    gate = _module()._choking_hazard_risk_threshold
    assert gate == pytest.approx(0.30)

    correct = _risk_scored_against(INFANT_IN_CRIB)
    assert correct == pytest.approx(0.380, abs=0.005)
    assert correct >= gate, "scored against the right target, the hazard is caught"

    # Same hazard, same frame, only the caregiver's position differs.
    leaning = _risk_scored_against(CAREGIVER_LEANING_OVER_CRIB)
    across = _risk_scored_against(CAREGIVER_ACROSS_THE_ROOM)
    corner = _risk_scored_against(CAREGIVER_FAR_CORNER)

    assert leaning == pytest.approx(0.380, abs=0.005)
    assert leaning >= gate, "caught, but only by luck of geometry -- not because it was scored right"
    assert across == pytest.approx(0.198, abs=0.005)
    assert across < gate, "REAL HAZARD MISSED"
    assert corner == pytest.approx(0.0, abs=0.005)
    assert corner < gate, "REAL HAZARD MISSED ENTIRELY -- risk scores literally zero"
