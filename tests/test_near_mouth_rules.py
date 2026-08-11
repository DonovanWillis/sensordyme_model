"""The `near_mouth` predicate: an object overlapping a person's face region.

This is the evidence upgrade behind eating/drinking rules. `holding` fires on
any box overlap — which includes a bottle carried at the hip, or standing next
to one on a table edge that clips the person's box. `near_mouth` requires the
object to intersect the TOP of the person's box, where a standing or seated
person's face is, so it fires when the object is raised toward the mouth and
not before. Still an approximation (a box has no actual mouth in it), and the
prompt says so in every rule description it writes — but the distinction
between "has a drink" and "raised it to drink" is real and testable.
"""

from __future__ import annotations

from core.detections import Detection, DetectionFrame
from core.rules import RuleEngine, _upper_region


def _frame(detections: list[tuple[str, tuple]], size=(640, 480)) -> DetectionFrame:
    return DetectionFrame(
        width=size[0], height=size[1],
        detections=tuple(Detection(label=label, confidence=0.9, box=box)
                         for label, box in detections),
    )


def _rule(predicate: str, secondary: str) -> dict:
    return {
        "target": "person", "predicate": predicate, "secondary": secondary,
        "description": "test", "zone": "", "persist_s": 1.0, "min_count": 1,
        "confidence": 0.2, "severity": "info", "cooldown_s": 30.0,
        "schedule": "", "notify": "push",
    }


PERSON = (200.0, 100.0, 400.0, 460.0)          # standing person, head at the top
BOTTLE_AT_FACE = (330.0, 110.0, 380.0, 180.0)  # overlaps the top of the person box
BOTTLE_AT_HIP = (330.0, 300.0, 380.0, 380.0)   # overlaps the person, low
BOTTLE_ACROSS_ROOM = (500.0, 300.0, 550.0, 380.0)


def _drive(engine: RuleEngine, frame: DetectionFrame, seconds=2.0, step=0.5, start=1000.0):
    alerts = []
    for i in range(int(seconds / step) + 1):
        alerts += engine.observe_frame(frame, now=start + i * step)
    return alerts


class TestNearMouth:
    def test_object_at_face_height_fires(self):
        engine = RuleEngine([_rule("near_mouth", "bottle")])
        alerts = _drive(engine, _frame([("person", PERSON), ("bottle", BOTTLE_AT_FACE)]))
        assert len(alerts) == 1

    def test_object_at_hip_height_does_not_fire(self):
        """The distinction from `holding`: carried is not raised-to-drink."""
        engine = RuleEngine([_rule("near_mouth", "bottle")])
        alerts = _drive(engine, _frame([("person", PERSON), ("bottle", BOTTLE_AT_HIP)]))
        assert alerts == []

    def test_holding_still_fires_at_hip_height(self):
        """`holding` keeps its broader any-overlap meaning — near_mouth is an
        additional predicate, not a redefinition of the old one."""
        engine = RuleEngine([_rule("holding", "bottle")])
        alerts = _drive(engine, _frame([("person", PERSON), ("bottle", BOTTLE_AT_HIP)]))
        assert len(alerts) == 1

    def test_object_away_from_the_person_does_not_fire(self):
        engine = RuleEngine([_rule("near_mouth", "bottle")])
        alerts = _drive(engine, _frame([("person", PERSON), ("bottle", BOTTLE_ACROSS_ROOM)]))
        assert alerts == []

    def test_no_secondary_never_fires(self):
        """Fails closed, matching near/holding — and jupiter/schema.py refuses
        to accept such a rule in the first place."""
        engine = RuleEngine([_rule("near_mouth", "")])
        alerts = _drive(engine, _frame([("person", PERSON), ("bottle", BOTTLE_AT_FACE)]))
        assert alerts == []

    def test_missing_person_never_fires(self):
        engine = RuleEngine([_rule("near_mouth", "bottle")])
        alerts = _drive(engine, _frame([("bottle", BOTTLE_AT_FACE)]))
        assert alerts == []


class TestUpperRegion:
    def test_upper_region_is_the_top_of_the_box(self):
        x1, y1, x2, y2 = _upper_region((0.0, 100.0, 50.0, 300.0))
        assert (x1, y1, x2) == (0.0, 100.0, 50.0)
        assert y2 == 100.0 + 200.0 * 0.35
