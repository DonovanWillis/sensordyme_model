"""Per-frame detection view, and rules evaluating against whole frames.

Covers the link that makes zone and count rules possible at all: `Event` carries
no boxes and no counts, so custom rules read `DetectionFrame` instead.
"""

from core.detections import Detection, DetectionFrame
from core.rules import RuleEngine
from core.zones import Zone

FRAME_W, FRAME_H = 1000, 1000
POOL = Zone(name="pool", polygon=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))
IN_POOL = (600.0, 600.0, 700.0, 900.0)     # feet at (0.65, 0.9)
OUT_OF_POOL = (10.0, 10.0, 110.0, 200.0)   # feet at (0.06, 0.2)


def frame(*detections):
    return DetectionFrame(width=FRAME_W, height=FRAME_H, detections=tuple(detections))


def person(box=None, confidence=0.9):
    return Detection(label="person", confidence=confidence, box=box)


def rule(**overrides):
    base = {
        "detection": "person", "description": "someone is here", "zone": "",
        "persist_s": 2.0, "min_count": 1, "confidence": 0.5, "severity": "warning",
        "cooldown_s": 30.0, "schedule": "", "notify": "push",
    }
    base.update(overrides)
    return base


def drive_frames(engine, detection_frame, seconds=5.0, start=1000.0, step=0.5):
    alerts = []
    for i in range(int(seconds / step) + 1):
        alerts += engine.observe_frame(detection_frame, now=start + i * step)
    return alerts


# --------------------------------------------------------------------- DetectionFrame
def test_counts_and_boxes_by_label():
    f = frame(person(IN_POOL), person(OUT_OF_POOL), Detection("dog", 0.7))
    assert f.count("person") == 2
    assert f.count("dog") == 1
    assert f.count("cat") == 0
    assert f.labels() == {"person", "dog"}
    assert len(f.boxes("person")) == 2


def test_mean_confidence_not_max():
    """One confident flicker among weak detections must not read as confident."""
    f = frame(person(confidence=1.0), person(confidence=0.2))
    assert f.mean_confidence("person") == 0.6
    assert f.mean_confidence("absent") == 0.0


def test_detections_without_boxes_are_allowed():
    f = frame(Detection("smoke", 0.8))
    assert f.boxes("smoke") == []
    assert f.count("smoke") == 1


# --------------------------------------------------------------------- frame-driven rules
def test_frame_with_person_confirms_a_presence_rule():
    engine = RuleEngine([rule()])
    assert len(drive_frames(engine, frame(person(OUT_OF_POOL)))) == 1


def test_empty_frames_never_confirm():
    engine = RuleEngine([rule()])
    assert drive_frames(engine, frame()) == []


def test_count_rule_uses_the_frame_count():
    engine = RuleEngine([rule(min_count=3, description="more than three people")])
    assert drive_frames(engine, frame(person(), person())) == []
    three = frame(person(), person(), person())
    assert len(drive_frames(engine, three, start=9000.0)) == 1


def test_zone_rule_uses_the_frame_boxes():
    engine = RuleEngine([rule(zone="pool")], zones=[POOL])
    assert drive_frames(engine, frame(person(OUT_OF_POOL))) == []
    alerts = drive_frames(engine, frame(person(IN_POOL)), start=9000.0)
    assert len(alerts) == 1 and alerts[0].zone == "pool"


def test_one_person_in_zone_among_several_outside_still_fires():
    engine = RuleEngine([rule(zone="pool")], zones=[POOL])
    mixed = frame(person(OUT_OF_POOL), person(OUT_OF_POOL), person(IN_POOL))
    assert len(drive_frames(engine, mixed)) == 1


def test_absence_resolves_a_confirmed_rule_so_it_can_fire_again():
    """Frames must report absence, not just presence, or events never resolve."""
    engine = RuleEngine([rule(cooldown_s=5.0)])
    assert len(drive_frames(engine, frame(person()), seconds=4.0, start=1000.0)) == 1
    for i in range(40):  # person leaves; event resolves, cooldown elapses
        engine.observe_frame(frame(), now=1010.0 + i)
    assert len(drive_frames(engine, frame(person()), seconds=4.0, start=1100.0)) == 1


def test_labels_with_no_rules_are_ignored():
    engine = RuleEngine([rule(detection="person")])
    assert drive_frames(engine, frame(Detection("dog", 0.9))) == []


# --------------------------------------------------------------------- module contract
def test_base_module_opts_out_by_default():
    """A module that hasn't implemented the view must report None, not a guess."""
    import numpy as np

    from core.events import Event
    from modules.base import VisionModule

    class Bare(VisionModule):
        name = "bare"

        def process(self, frame) -> list[Event]:
            return []

    assert Bare({}).get_detections() is None
    assert np.asarray([1]).size == 1  # numpy import is real, not decorative
