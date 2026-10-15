"""Phase 3: positive infant identification in modules/baby_monitoring.py.

The defect these cover, measured in Phase 1 by execution: `_infer_baby_box`
returned the SMALLEST person box as "the baby" and therefore ALWAYS returned
something. When a marginal-confidence infant box fell below the detector
threshold while a caregiver was in frame, the CAREGIVER became "the baby" and
choking-hazard risk was scored against an adult's mouth region and body
diagonal. One byte-identical real hazard either fired or was missed purely
according to where the adult happened to be standing:

    caregiver leaning over the crib .......... 0.380  FIRES
    caregiver close to camera, filling frame . 0.347  FIRES
    caregiver standing beside the crib ....... 0.333  FIRES
    caregiver across the room, left .......... 0.233  MISSED
    caregiver across the room, seated ........ 0.198  MISSED
    caregiver in the far corner, small ....... 0.000  MISSED ENTIRELY

The headline test here is test_choking_risk_outcome_is_invariant_to_where_the_
caregiver_stands, which sweeps the same six placements against the same fixed
hazard and requires ONE outcome for all of them.

Most tests deliberately drive BabyMonitoringModule end to end (boxes in, events
out) rather than poking InfantIdentifier directly: the defect was in what the
product DID, and a test that only exercised the new helper would have passed
just as happily on the old code path.
"""

from __future__ import annotations

import pytest

from core.detectors import upper_region
from core.hand_perception import HandPerceptionResult
from core.tracking import TrackedBox
from core.zones import load_zones
from modules.baby_monitoring import (
    INFANT_NO_ZONE_MAX_CONFIDENCE,
    MOUTH_ZONE_FRACTION,
    BabyMonitoringModule,
    InfantIdentifier,
    _zone_containment,
)

W, H = 1000, 1000
CRIB = {"crib": {"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]], "kind": "safe"}}

#: config.yaml records this from hardware testing: a REAL held object near the
#: face peaks around this association score, and risk_score multiplies it
#: directly, so it is the hard ceiling on choking risk. Every hazard below is
#: built at this value -- an optimistic 0.85 would prove nothing about the field.
MEASURED_ASSOCIATION_CEILING = 0.38
RISK_GATE = 0.30

INFANT_IN_CRIB = (500, 400, 700, 850)
#: Six realistic caregiver placements. Note that these are NOT all larger than
#: the infant's box: the far-corner one is 42,000px to the infant's 90,000px,
#: which is exactly how "the smallest box is the baby" put a real hazard at
#: 0.000 with the infant plainly visible in the crib.
CAREGIVER_PLACEMENTS = [
    ("leaning over the crib", (450, 250, 800, 980)),
    ("close to camera, filling frame", (150, 60, 880, 1000)),
    ("standing beside the crib", (700, 180, 900, 940)),
    ("across the room, left", (30, 480, 330, 980)),
    ("across the room, seated", (30, 500, 300, 950)),
    ("in the far corner, small", (40, 600, 190, 880)),
]

MOUTH = upper_region(INFANT_IN_CRIB, MOUTH_ZONE_FRACTION)
INFANT_MOUTH = ((MOUTH[0] + MOUTH[2]) / 2.0, (MOUTH[1] + MOUTH[3]) / 2.0)


def _module(**overrides):
    """Built from the module's own in-code defaults except where a test says
    otherwise, so these measure what a real deployment actually gets."""
    config = {"zones": CRIB, "zone_confirm_s": 1.0, "still_max_motion": 0.015}
    config.update(overrides)
    return BabyMonitoringModule(config)


def _hazard(target=INFANT_MOUTH, duration=0.6, half_size=20, label="bottle",
            score=MEASURED_ASSOCIATION_CEILING, hand_box=(440, 330, 560, 460), t0=0.0):
    """ONE small object, genuinely held, carried on a straight closing line
    into `target` and held there. Identical in every sweep row.

    `t0` is when the approach begins, for tests whose run starts partway along
    the timeline -- without it the object would be parked at the target from
    the first frame with no trail, which is a non-hazard (no observed closing
    trajectory) and would prove nothing."""
    trail: list[tuple[float, float, float]] = []
    start = (target[0] + 250, target[1] + 250)

    def hand_result_fn(t):
        t = t - t0
        frac = min(1.0, t / duration) if duration > 0 else 1.0
        x = start[0] + (target[0] - start[0]) * frac
        y = start[1] + (target[1] - start[1]) * frac
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


def _run(module, boxes, seconds, start=0.0, step=0.1, hand_result_fn=None, motion=0.05):
    """Returns (events, end_time, peak right-hand risk over the run). Peak, not
    the last frame's value: the gate sees every frame, and after the object
    arrives its trajectory factor legitimately decays."""
    events = []
    peak = 0.0
    t = start
    while t < start + seconds - 1e-9:
        events += module._reason(
            list(boxes), motion, W, H, t, (), None,
            hand_result=hand_result_fn(t) if hand_result_fn else None,
        )
        assessment = module._last_choking_assessments.get("right")
        if assessment is not None:
            peak = max(peak, assessment.risk_score)
        t = round(t + step, 6)
    return events, t, peak


# ============================================================ THE HEADLINE
@pytest.mark.parametrize(("where", "caregiver"), CAREGIVER_PLACEMENTS)
def test_choking_risk_outcome_is_invariant_to_where_the_caregiver_stands(where, caregiver):
    """THE fix, stated as the property that was violated: with the infant's box
    lost and a caregiver in frame, ONE fixed real hazard must produce ONE
    outcome regardless of where the adult is standing.

    Before: 0.380/0.347/0.333/0.233/0.198/0.000 across these six placements --
    three fired, three were missed, and the three that fired were not scored
    correctly, they merely had an adult's head near where the infant's was.
    After: 0.000 and no alert in all six, because no person in frame is
    identifiably the infant and the module declines to score rather than
    scoring an adult.
    """
    module = _module()
    events, _, peak = _run(module, [caregiver], 2.4, hand_result_fn=_hazard())

    assert "choking_hazard" not in [e.type for e in events], (
        f"caregiver {where}: risk was scored against an adult and alerted"
    )
    assert peak == 0.0, f"caregiver {where}: risk {peak} was scored against a non-infant"
    assert module._last_infant_identification.identified is False
    assert module._last_infant_identification.box is None


@pytest.mark.parametrize(("where", "caregiver"), CAREGIVER_PLACEMENTS)
def test_a_real_hazard_at_the_infants_mouth_fires_wherever_the_caregiver_stands(where, caregiver):
    """The other half of invariance, and the true-positive guard: with the
    infant visible alongside the caregiver, the same fixed hazard must fire at
    the same correct score every time.

    This is not merely the old behaviour preserved. The old heuristic MISSED
    this outright for the far-corner caregiver (0.000): that adult's box is
    smaller in area than the infant's, so "smallest box wins" targeted the
    adult with the baby plainly visible in the crib.
    """
    module = _module()
    events, _, peak = _run(module, [caregiver, INFANT_IN_CRIB], 2.4, hand_result_fn=_hazard())

    assert "choking_hazard" in [e.type for e in events], f"caregiver {where}: real hazard missed"
    assert peak == pytest.approx(MEASURED_ASSOCIATION_CEILING, abs=0.005)
    assert peak >= RISK_GATE
    assert module._last_infant_identification.box == INFANT_IN_CRIB, (
        f"caregiver {where}: risk was scored against the wrong person"
    )


def test_an_object_at_a_caregivers_mouth_does_not_fire():
    """The false-positive guard. An adult across the room putting the same
    small object to their OWN mouth is not a choking hazard to the child, and
    must not be reported as one -- the risk is scored against the identified
    infant's geometry, so distance does the work."""
    caregiver = (30, 500, 300, 950)
    caregiver_mouth_region = upper_region(caregiver, MOUTH_ZONE_FRACTION)
    caregiver_mouth = (
        (caregiver_mouth_region[0] + caregiver_mouth_region[2]) / 2.0,
        (caregiver_mouth_region[1] + caregiver_mouth_region[3]) / 2.0,
    )
    module = _module()
    events, _, peak = _run(
        module, [caregiver, INFANT_IN_CRIB], 2.4,
        hand_result_fn=_hazard(target=caregiver_mouth, hand_box=(60, 520, 180, 640)),
    )
    assert "choking_hazard" not in [e.type for e in events]
    assert peak < RISK_GATE
    # ...and it is silent for the RIGHT reason: the infant was identified fine,
    # the object simply was not near the infant.
    assert module._last_infant_identification.box == INFANT_IN_CRIB


# =================================================== declining is not silence
def test_declining_to_score_raises_an_operator_visible_event():
    """Declining that nobody can observe is just a quieter version of the bug.
    People in frame with none of them identifiable as the infant must produce
    an actual event saying choking-hazard scoring is paused."""
    module = _module(expect_presence=True, absence_warning_s=30.0)
    events, t, _ = _run(module, [INFANT_IN_CRIB], 5.0)
    events2, _, _ = _run(module, [(450, 250, 800, 980)], 40.0, start=t)

    fired = [e for e in events + events2 if e.type == "infant_unidentified"]
    assert len(fired) == 1, "the operator must be told monitoring is degraded"
    assert "choking-hazard scoring is paused" in fired[0].contributing_signals


def test_person_absent_does_not_cover_the_declined_case():
    """Why the rejected `if len(boxes) == 1: return None` one-liner is not a
    fix: a caregiver in frame means the frame is NOT person-absent, so that
    rule stays silent throughout and the operator would be told nothing at
    all."""
    module = _module(expect_presence=True, absence_warning_s=5.0)
    events, t, _ = _run(module, [INFANT_IN_CRIB], 5.0)
    events2, _, _ = _run(module, [(450, 250, 800, 980)], 40.0, start=t)

    types = [e.type for e in events + events2]
    assert "person_absent" not in types, "a caregiver is in frame -- this is not absence"
    assert "infant_unidentified" in types, "so THIS is the rule that has to speak"


def test_the_decline_and_its_reasoning_are_in_the_debug_state():
    """"Why did/didn't this fire" needs the breakdown, not a bare 0.0 -- a zero
    risk score used to be indistinguishable from a risk score computed against
    the wrong person."""
    module = _module()
    _run(module, [(30, 500, 300, 950)], 2.0, hand_result_fn=_hazard())
    state = module.get_debug_state()["infant_identification"]

    assert state["identified"] is False
    assert state["box"] is None
    assert state["people_in_frame"] == 1
    assert state["eligible_candidates"] == 0
    assert state["reason"]
    assert any("safe zone" in r for r in state["rejected"]), state["rejected"]
    # ...and the hand's own assessment says it declined, not "no baby in frame".
    assert "declined to score" in module._last_choking_assessments["right"].reason


def test_telemetry_carries_whether_the_infant_was_identified_at_all():
    """The single number that distinguishes "no hazard happened" from "nothing
    was ever scored" in a logged session."""
    module = _module()
    _run(module, [INFANT_IN_CRIB], 1.0)
    assert module.get_telemetry_snapshot()["infant_identified"] is True

    lost = _module()
    _run(lost, [(30, 500, 300, 950)], 1.0)
    snapshot = lost.get_telemetry_snapshot()
    assert snapshot["infant_identified"] is False
    assert snapshot["infant_identification_confidence"] == 0.0


# =========================================================== the signals
def test_zone_containment_separates_lying_in_the_crib_from_reaching_into_it():
    """The strongest available signal, and why it is an AREA fraction rather
    than core/zones.py's feet-point test: an infant IN the crib is entirely
    inside it; an adult leaning INTO the crib is not, even though a large part
    of them overlaps it."""
    zones = load_zones(CRIB)
    assert _zone_containment(INFANT_IN_CRIB, zones, W, H) == 1.0
    leaning = _zone_containment((450, 250, 800, 980), zones, W, H)
    assert 0.5 < leaning < 0.85, f"the leaning adult partly overlaps the crib ({leaning})"
    assert _zone_containment((40, 600, 190, 880), zones, W, H) == 0.0


def test_a_caregiver_walking_in_never_takes_over_the_infants_identity():
    """Temporal identity: a person seen in frame ALONGSIDE the identified
    infant is definitively somebody else, and stays somebody else after the
    infant's box drops out. Without this, losing the infant's box hands the
    label straight to whoever is left standing."""
    module = _module(zones={})  # no crib zone: temporal identity is all there is
    _, t, _ = _run(module, [INFANT_IN_CRIB], 5.0)
    _, t, _ = _run(module, [INFANT_IN_CRIB, (450, 250, 800, 980)], 5.0, start=t)
    assert module._last_infant_identification.box == INFANT_IN_CRIB

    _, _, peak = _run(module, [(450, 250, 800, 980)], 2.4, start=t,
                      hand_result_fn=_hazard(t0=t))
    assert module._last_infant_identification.identified is False
    assert peak == 0.0


def test_a_caregivers_box_grafted_onto_the_infants_track_breaks_the_identity():
    """core/tracking.py's matcher is nearest-centroid within 1.2 box diagonals,
    and is read-only from here. Measured: it re-matches every one of the six
    caregiver placements onto the INFANT's own track the frame the infant's box
    is lost -- the far-corner one, 500px away, still lands at 1.01 diagonals.
    Left unchecked the caregiver inherits the infant's track id AND its
    accumulated age, and temporal identity silently hands the wrong-target bug
    back. A box that changes size or position faster than a subject plausibly
    can must therefore break the identity claim."""
    module = _module(zones={})
    _, t, _ = _run(module, [INFANT_IN_CRIB], 6.0)
    assert module._last_infant_identification.identified is True

    # The infant's box vanishes and the caregiver's appears in the same frame,
    # close enough for the tracker to call it the same subject.
    _, _, peak = _run(module, [(450, 250, 800, 980)], 2.4, start=t,
                      hand_result_fn=_hazard(t0=t))
    assert module._last_infant_identification.identified is False, (
        "the caregiver inherited the infant's track"
    )
    assert peak == 0.0


def test_two_indistinguishable_candidates_decline_rather_than_guess():
    """Two people equally consistent with being the infant means the module
    does not know which is which. Saying so is the point; picking one is the
    bug."""
    twin_a = (300, 400, 500, 850)
    twin_b = (550, 400, 750, 850)
    module = _module()
    _run(module, [twin_a, twin_b], 1.0)
    identification = module._last_infant_identification
    assert identification.identified is False
    assert identification.eligible == 2
    assert "equally consistent" in identification.reason


def test_a_spurious_second_person_box_does_not_change_the_outcome():
    """Phase 2 measured the Hailo backend emitting spurious extra person boxes
    on 12.7% of single-person images (CPU: 7.2%) -- so moving to the NPU makes
    the multi-person path MORE common, not less. An established infant must
    keep its identity when a phantom box appears beside it, and the hazard must
    still fire."""
    module = _module()
    _, t, _ = _run(module, [INFANT_IN_CRIB], 3.0)
    assert module._last_infant_identification.box == INFANT_IN_CRIB

    spurious = (520, 430, 660, 700)  # a partial duplicate, also inside the crib
    events, _, peak = _run(
        module, [INFANT_IN_CRIB, spurious], 2.4, start=t, hand_result_fn=_hazard(t0=t),
    )
    assert module._last_infant_identification.box == INFANT_IN_CRIB
    assert "choking_hazard" in [e.type for e in events]
    assert peak == pytest.approx(MEASURED_ASSOCIATION_CEILING, abs=0.005)


# ================================================== the no-safe-zone default
def test_with_no_safe_zone_a_lone_established_infant_is_still_identified():
    """config.yaml ships `zones: {}`, so the fallback is the common case, not
    an edge case. It must still work -- after observing the subject long enough
    to have earned the claim."""
    module = _module(zones={})
    _run(module, [INFANT_IN_CRIB], 1.0)
    assert module._last_infant_identification.identified is False, "not yet observed enough"

    _run(module, [INFANT_IN_CRIB], 5.0, start=1.0)
    identification = module._last_infant_identification
    assert identification.identified is True
    assert identification.box == INFANT_IN_CRIB


def test_identification_confidence_is_capped_without_a_safe_zone():
    """Honest about its own evidence: with no crib zone this rests on
    persistence and stability alone, and the reported confidence has to say
    so rather than claiming the certainty crib geometry would give."""
    zoned = _module()
    _run(zoned, [INFANT_IN_CRIB], 6.0)
    unzoned = _module(zones={})
    _run(unzoned, [INFANT_IN_CRIB], 6.0)

    assert unzoned._last_infant_identification.confidence <= INFANT_NO_ZONE_MAX_CONFIDENCE
    assert zoned._last_infant_identification.confidence > INFANT_NO_ZONE_MAX_CONFIDENCE
    assert "safe zone" in unzoned._last_infant_identification.reason


def test_no_people_in_frame_is_reported_differently_from_cannot_identify():
    """Two genuinely different situations that a bare `None` conflated."""
    module = _module()
    _run(module, [], 1.0)
    empty = module._last_infant_identification
    assert empty.identified is False and empty.people_in_frame == 0
    assert "no person detected" in empty.reason

    _run(module, [(30, 500, 300, 950)], 1.0, start=1.0)
    unidentified = module._last_infant_identification
    assert unidentified.identified is False and unidentified.people_in_frame == 1
    assert "could be positively identified" in unidentified.reason


def test_the_identifier_never_returns_a_box_it_did_not_identify():
    """The structural invariant behind all of the above: `box` is non-None if
    and only if `identified` is true. The old helper's whole failure was
    returning a box while having identified nothing."""
    identifier = InfantIdentifier(load_zones(CRIB))
    for now, boxes in enumerate([
        [], [INFANT_IN_CRIB], [(30, 500, 300, 950)], [INFANT_IN_CRIB, (450, 250, 800, 980)],
        [(40, 600, 190, 880)], [],
    ]):
        result = identifier.update(boxes, [], W, H, float(now))
        assert (result.box is not None) == result.identified
