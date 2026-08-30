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


# ============================================== PHASE 2 PINS, FIXED IN PHASE 3
# These two tests were written in Phase 1 as CHARACTERIZATION PINS: they
# asserted the then-current, known-WRONG behaviour of `_infer_baby_box`, which
# returned the SMALLEST person box as "the baby" and therefore always returned
# something. When a marginal-confidence infant box fell below the detector
# threshold while a caregiver was in frame, it handed back the CAREGIVER's box
# and choking-hazard risk was scored against an adult's mouth region and body
# diagonal -- a silent wrong-target failure, materially worse than a missed
# detection. They were pinned rather than fixed because the fix needed positive
# identification, not a threshold guess.
#
# Phase 3 built that (modules/baby_monitoring.py's InfantIdentifier: crib-zone
# containment, temporal identity, size/position stability, relative scale, and
# an explicit reported "I do not know" that DECLINES to score). `_infer_baby_box`
# no longer exists. So these two now assert the CORRECT behaviour instead,
# keeping every measured number from the pinned version as the record of what
# was actually wrong. The full Phase 3 coverage lives in
# tests/test_phase3_baby_infant_identification.py; these two stay here as the
# Phase 1 red line, closed.
#
# Renamed from test_pins_infer_baby_box_returning_the_caregiver_when_the_infant_
# box_is_lost and test_pins_choking_risk_outcome_depending_on_where_the_
# caregiver_stands: a name beginning "test_pins_" that asserts the opposite of
# what it pins would be worse than no name at all.
INFANT_IN_CRIB = (500, 400, 700, 850)
#: Realistic caregiver placements. NOT all larger in area than the infant box:
#: the far-corner one is 42,000px to the infant's 90,000px, which is how
#: "smallest box wins" targeted an adult with the baby plainly in the crib.
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


def _hazard_at_the_infants_mouth():
    """The same fixed hazard as _risk_scored_against, but fed through the module
    frame by frame so the identifier -- not the test -- chooses the target."""
    mouth = upper_region(INFANT_IN_CRIB, MOUTH_ZONE_FRACTION)
    end = ((mouth[0] + mouth[2]) / 2.0, (mouth[1] + mouth[3]) / 2.0)
    return _converging(start=(end[0] + 250, end[1] + 250), end=end, duration=0.6)


def test_infant_identification_declines_instead_of_returning_the_caregiver_when_the_infant_box_is_lost():
    """FIXED IN PHASE 3 (was a characterization pin).

    With both visible the infant is identified, which the old smallest-box
    heuristic also managed. The difference is the case it got wrong: with the
    infant's box lost and only a caregiver in frame, identification must FAIL
    and report that failure -- `identified=False`, `box=None`, a stated reason
    -- rather than silently substituting an adult.

    Note this is NOT the `if len(boxes) == 1: return None` shortcut the Phase 1
    audit rejected. Two boxes where neither is the infant declines too, and one
    box that IS the infant identifies fine; the count of boxes is not what
    decides it.
    """
    module = _module()
    _run(module, [CAREGIVER_ACROSS_THE_ROOM, INFANT_IN_CRIB], 0.05, 1.0)
    both = module._last_infant_identification
    assert both.identified is True
    assert both.box == INFANT_IN_CRIB, "with both visible, the infant is identified"

    lost = _module()
    _run(lost, [CAREGIVER_ACROSS_THE_ROOM], 0.05, 1.0)
    only_adult = lost._last_infant_identification
    assert only_adult.identified is False, (
        "the infant box was lost, so there is no infant to score against -- the "
        "caregiver must NOT become 'the baby'"
    )
    assert only_adult.box is None
    assert only_adult.people_in_frame == 1, "and it is not being confused with an empty frame"
    assert only_adult.reason

    # Not a box-count rule: two non-infants decline as well.
    neither = _module()
    _run(neither, [CAREGIVER_ACROSS_THE_ROOM, CAREGIVER_FAR_CORNER], 0.05, 1.0)
    assert neither._last_infant_identification.identified is False

    empty = _module()
    _run(empty, [], 0.0, 1.0)
    assert empty._last_infant_identification.box is None
    assert empty._last_infant_identification.people_in_frame == 0


def test_choking_risk_outcome_no_longer_depends_on_where_the_caregiver_stands():
    """FIXED IN PHASE 3 (was a characterization pin), and the part that actually
    mattered: with the infant's box lost, whether ONE fixed real hazard was
    caught used to depend on where the CAREGIVER happened to be standing, which
    made the detection non-deterministic with respect to the hazard itself.

    The risk FUNCTION is still position-dependent, and cannot be otherwise --
    it measures distance to a mouth region and normalizes by a body diagonal,
    so pointing it at a different body genuinely does give a different number.
    The measured spread below is kept exactly as pinned, because it is the
    reason target selection is safety-critical rather than a detail. What
    changed is that a non-infant box can no longer REACH the function: the
    module declines instead, so the OUTCOME is invariant. Gate 0.30.
    """
    gate = _module()._choking_hazard_risk_threshold
    assert gate == pytest.approx(0.30)

    # --- unchanged, and unchangeable: the arithmetic of scoring the wrong body
    correct = _risk_scored_against(INFANT_IN_CRIB)
    assert correct == pytest.approx(0.380, abs=0.005)
    assert correct >= gate, "scored against the right target, the hazard is caught"
    assert _risk_scored_against(CAREGIVER_LEANING_OVER_CRIB) == pytest.approx(0.380, abs=0.005)
    assert _risk_scored_against(CAREGIVER_ACROSS_THE_ROOM) == pytest.approx(0.198, abs=0.005)
    assert _risk_scored_against(CAREGIVER_FAR_CORNER) == pytest.approx(0.0, abs=0.005)

    # --- what changed: the same hazard, swept over the same caregiver
    # placements, through the real module. One outcome, every time.
    outcomes = {}
    for name, caregiver in (
        ("leaning", CAREGIVER_LEANING_OVER_CRIB),
        ("across", CAREGIVER_ACROSS_THE_ROOM),
        ("corner", CAREGIVER_FAR_CORNER),
    ):
        module = _module()
        events, _ = _run(module, [caregiver], 0.05, 2.4,
                         hand_result_fn=_hazard_at_the_infants_mouth())
        outcomes[name] = (
            "choking_hazard" in [e.type for e in events],
            module._last_choking_assessments["right"].risk_score,
            module._last_infant_identification.identified,
        )
    assert set(outcomes.values()) == {(False, 0.0, False)}, (
        f"outcome still varies with where the adult stands: {outcomes}"
    )

    # ...and the true positive is intact: put the infant back in frame and the
    # identical hazard fires, at the correctly-targeted 0.380, from every one
    # of those caregiver positions.
    for name, caregiver in (
        ("leaning", CAREGIVER_LEANING_OVER_CRIB),
        ("across", CAREGIVER_ACROSS_THE_ROOM),
        ("corner", CAREGIVER_FAR_CORNER),
    ):
        module = _module()
        events, _ = _run(module, [caregiver, INFANT_IN_CRIB], 0.05, 2.4,
                         hand_result_fn=_hazard_at_the_infants_mouth())
        assert "choking_hazard" in [e.type for e in events], f"real hazard missed ({name})"
        assert module._last_infant_identification.box == INFANT_IN_CRIB
