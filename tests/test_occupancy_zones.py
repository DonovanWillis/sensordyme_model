"""Tests for zone security events in the occupancy (security) module."""

import numpy as np
import pytest

import modules.occupancy as occupancy_module
from core.accel import COCO_NAMES
from modules.occupancy import OccupancyModule

_NAME_TO_ID = {name: class_id for class_id, name in COCO_NAMES.items()}


class _FakeBox:
    def __init__(self, xyxy, conf=0.9, cls=0):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.float32(conf)]
        self.cls = [np.float32(cls)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self):
        self.boxes_to_return = []

    def predict(self, frame, **kwargs):
        # Each item is a plain xyxy 4-tuple (defaults to person), or an
        # (xyxy, class_id) pair for a specific class.
        boxes = []
        for item in self.boxes_to_return:
            if len(item) == 2 and len(item[0]) == 4:
                xyxy, cls = item
            else:
                xyxy, cls = item, 0
            boxes.append(_FakeBox(xyxy, cls=cls))
        return [_FakeResult(boxes)]


@pytest.fixture()
def module(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(occupancy_module, "get_yolo_model", lambda weights: fake)
    mod = OccupancyModule({
        "confidence_threshold": 0.5, "confirm_frames": 2,
        "zones": {"entry": {"polygon": [[0.6, 0.2], [0.95, 0.2], [0.95, 0.9], [0.6, 0.9]],
                            "kind": "restricted"}},
        "zone_confirm_s": 0.5, "linger_s": 3.0,
    })
    mod._fake = fake
    return mod


FRAME = np.zeros((1000, 1000, 3), dtype=np.uint8)
IN_ENTRY = (700, 300, 900, 800)   # feet (0.8, 0.8) -> inside entry zone
OUTSIDE = (100, 300, 300, 800)    # feet (0.2, 0.8) -> outside


def _run(module, boxes, frames):
    events = []
    module._fake.boxes_to_return = boxes
    for _ in range(frames):
        events += module.process(FRAME)
    return events


def test_person_outside_zone_only_counts(module):
    events = _run(module, [OUTSIDE], 20)
    types = {e.type for e in events}
    assert "occupancy_change" in types
    assert "zone_violation" not in types
    assert not module.is_alert_active()


def test_zone_violation_confirms_then_lingering(module, monkeypatch):
    # Zone events use wall-clock persistence; step monotonic time deterministically.
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    events = []
    for _ in range(50):  # 5 simulated seconds inside the zone
        module._fake.boxes_to_return = [IN_ENTRY]
        events += module.process(FRAME)
        t["now"] += 0.1
    types = [e.type for e in events]
    assert "zone_violation" in types      # confirmed after zone_confirm_s
    assert "person_lingering" in types    # confirmed after linger_s
    assert module.is_alert_active()

    for _ in range(80):  # zone empties -> resolution
        module._fake.boxes_to_return = []
        events += module.process(FRAME)
        t["now"] += 0.1
    assert "zone_violation_resolved" in [e.type for e in events]
    assert not module.is_alert_active()


def test_vehicle_in_zone_confirms_and_resolves(module, monkeypatch):
    """Vehicle detection reuses the same zone engine as zone_violation, gated
    on the SAME live capability-registry vocabulary Jupiter uses (core.
    taxonomy.py used to mark this 'not wired into the security event set')."""
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    car_id = _NAME_TO_ID["car"]
    events = []
    for _ in range(10):
        module._fake.boxes_to_return = [(IN_ENTRY, car_id)]
        events += module.process(FRAME)
        t["now"] += 0.1
    assert "vehicle_in_zone" in [e.type for e in events]
    assert "zone_violation" not in [e.type for e in events]  # a car is not a person
    assert module.is_alert_active()

    for _ in range(60):
        module._fake.boxes_to_return = []
        events += module.process(FRAME)
        t["now"] += 0.1
    assert "vehicle_in_zone_resolved" in [e.type for e in events]


def test_animal_in_zone_is_independent_of_vehicle_in_zone(module, monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    dog_id = _NAME_TO_ID["dog"]
    events = []
    for _ in range(10):
        module._fake.boxes_to_return = [(IN_ENTRY, dog_id)]
        events += module.process(FRAME)
        t["now"] += 0.1
    types = [e.type for e in events]
    assert "animal_in_zone" in types
    assert "vehicle_in_zone" not in types


def test_vehicle_outside_the_zone_does_not_confirm(module, monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    car_id = _NAME_TO_ID["car"]
    events = []
    for _ in range(10):
        module._fake.boxes_to_return = [(OUTSIDE, car_id)]
        events += module.process(FRAME)
        t["now"] += 0.1
    assert "vehicle_in_zone" not in [e.type for e in events]


def test_two_different_people_in_sequence_dont_sum_into_one_lingering(module, monkeypatch):
    """The bug per-person tracking fixes: an aggregate "has the zone been
    occupied" clock read person A leaving and person B arriving moments later
    as one continuous visit. linger_s=3.0 here; each visit lasts 1.8s (under
    the threshold), separated by a 3.0s empty gap -- longer than
    PersonTracker's own lost_grace_s (2.0s), so the second visit is a
    genuinely different tracked identity, not a re-matched continuation of
    the first."""
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    events = []

    for _ in range(18):  # person A, 1.8s
        module._fake.boxes_to_return = [IN_ENTRY]
        events += module.process(FRAME)
        t["now"] += 0.1

    for _ in range(30):  # A leaves; empty long enough to retire the track
        module._fake.boxes_to_return = []
        events += module.process(FRAME)
        t["now"] += 0.1

    for _ in range(18):  # person B, 1.8s
        module._fake.boxes_to_return = [IN_ENTRY]
        events += module.process(FRAME)
        t["now"] += 0.1

    assert "person_lingering" not in [e.type for e in events]


def test_one_person_lingering_the_full_duration_still_fires(module, monkeypatch):
    """The positive case for the same fix: one continuous stay past linger_s
    must still confirm -- per-person tracking narrows lingering, it must not
    silently disable it."""
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    events = []
    for _ in range(40):  # 4.0s, one continuous person, over linger_s=3.0
        module._fake.boxes_to_return = [IN_ENTRY]
        events += module.process(FRAME)
        t["now"] += 0.1
    assert "person_lingering" in [e.type for e in events]


def test_person_lingering_state_is_pruned_after_a_visitor_leaves_for_good(module, monkeypatch):
    """Regression guard for the leak EventEngine.remove_rule exists to prevent:
    without it, every track id that ever entered the zone keeps its own
    RuleSpec sitting in the engine for the life of the process."""
    t = {"now": 0.0}
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: t["now"])
    for _ in range(15):
        module._fake.boxes_to_return = [IN_ENTRY]
        module.process(FRAME)
        t["now"] += 0.1
    for _ in range(30):  # long enough for the tracker to retire, then the engine to drain
        module._fake.boxes_to_return = []
        module.process(FRAME)
        t["now"] += 0.1
    assert module._known_track_ids == set()
    assert not any(key.startswith("person_lingering:") for key in module._engine._rules)


def test_occupancy_change_debounce_still_works(module):
    events = _run(module, [OUTSIDE], 3)
    assert sum(1 for e in events if e.type == "occupancy_change") == 1  # confirmed once
    events = _run(module, [], 1)
    assert not any(e.type == "occupancy_change" for e in events)  # 1 frame < confirm_frames


# --------------------------------------------------------------- per-frame detection view
def test_get_detections_exposes_boxes_for_custom_rules(module):
    """AI rules need boxes and counts, which Events don't carry."""
    _run(module, [IN_ENTRY, OUTSIDE], 1)
    frame = module.get_detections()

    assert frame.size == (1000, 1000)
    assert frame.count("person") == 2
    assert len(frame.boxes("person")) == 2
    assert 0.0 < frame.mean_confidence("person") <= 1.0


def test_get_detections_reports_an_empty_frame_when_nobody_is_present(module):
    """Absence must be reported, not omitted, or confirmed rules never resolve."""
    _run(module, [IN_ENTRY], 1)
    _run(module, [], 1)
    frame = module.get_detections()
    assert frame.count("person") == 0
    assert frame.labels() == set()


def test_get_detections_boxes_land_in_the_right_zone(module):
    """The exposed geometry must agree with the module's own zone verdict."""
    from core.zones import Zone

    entry = Zone(name="entry", polygon=((0.6, 0.2), (0.95, 0.2), (0.95, 0.9), (0.6, 0.9)))
    _run(module, [IN_ENTRY], 1)
    boxes = module.get_detections().boxes("person")
    assert any(entry.contains_box(box, 1000, 1000) for box in boxes)

    _run(module, [OUTSIDE], 1)
    boxes = module.get_detections().boxes("person")
    assert not any(entry.contains_box(box, 1000, 1000) for box in boxes)
