"""Regression tests for the bottle/banana class-mapping investigation.

Traces the exact path a raw model class id takes to become the label a
custom AI rule (and Recent Activity) sees: modules/occupancy.py's
`process()` reads `box.cls[0]`, looks it up in `core.accel.COCO_NAMES`, and
stores the result as `Detection.label` via `get_detections()`. This is the
SAME lookup `set_requested_labels()` uses in reverse (name -> id) when a
custom rule asks the detector to also watch for something beyond "person".

See tests/test_accel.py for the COCO_NAMES pin itself (bottle=39,
banana=46, etc.) — these tests cover the id -> label -> Detection.label
round trip through the actual module code, plus the two failure modes the
product asked to be proven safe: a class id the model returns that
COCO_NAMES doesn't know about, and a detector shim with no `.cls` at all.
"""

import numpy as np
import pytest

import modules.occupancy as occupancy_module
from core.accel import COCO_NAMES
from modules.occupancy import OccupancyModule


class _FakeBoxWithClass:
    def __init__(self, xyxy, cls_id, conf=0.9):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.float32(conf)]
        self.cls = [np.float32(cls_id)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self):
        self.boxes_to_return = []  # list of (xyxy, cls_id)

    def predict(self, frame, **kwargs):
        return [_FakeResult([_FakeBoxWithClass(xyxy, cls_id) for xyxy, cls_id in self.boxes_to_return])]


@pytest.fixture()
def module(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(occupancy_module, "get_yolo_model", lambda weights: fake)
    mod = OccupancyModule({"confidence_threshold": 0.5, "confirm_frames": 1})
    mod._fake = fake
    return mod


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BOX = (100, 100, 200, 200)

# class_id: expected_label, matching tests/test_accel.py's pinned set.
_CASES = {39: "bottle", 46: "banana", 0: "person", 67: "cell phone", 56: "chair", 2: "car"}


@pytest.mark.parametrize("class_id,expected_label", sorted(_CASES.items()))
def test_class_id_resolves_to_the_correct_label(module, class_id, expected_label):
    """The exact table the product asked for: raw_class_id -> SensorDyme label."""
    module.set_requested_labels(set(_CASES.values()))
    module._fake.boxes_to_return = [(BOX, class_id)]
    module.process(FRAME)
    frame = module.get_detections()
    assert frame.labels() == {expected_label} or expected_label in frame.labels(), (
        f"class id {class_id} resolved to {frame.labels()!r}, expected {expected_label!r}"
    )


def test_bottle_and_banana_are_never_the_same_label(module):
    """The exact reported symptom, reproduced as a regression guard: request
    both, show both in the same frame, confirm they never collapse into one."""
    module.set_requested_labels({"bottle", "banana"})
    module._fake.boxes_to_return = [(BOX, 39), ((300, 300, 400, 400), 46)]
    module.process(FRAME)
    frame = module.get_detections()
    assert frame.count("bottle") == 1
    assert frame.count("banana") == 1
    bottle_boxes = frame.boxes("bottle")
    banana_boxes = frame.boxes("banana")
    assert bottle_boxes != banana_boxes


def test_an_unknown_class_id_is_labeled_safely_not_crashed(module):
    """A class id the model returns that COCO_NAMES doesn't recognize (a
    future model with more classes, a corrupted output) must not crash
    detection — it degrades to a string of the id itself, same as
    COCO_NAMES.get(class_id, str(class_id)) already does."""
    module.set_requested_labels({"person"})
    module._fake.boxes_to_return = [(BOX, 9999)]
    module.process(FRAME)  # must not raise
    frame = module.get_detections()
    assert frame.labels() == {"9999"}
    assert frame.count("person") == 0  # the unknown-id box must not be miscounted as a person


def test_set_requested_labels_drops_unknown_names_without_raising():
    """A stray/misspelled label from a stale AI-configured rule must not take
    the camera down — the capability layer already refused it upstream, so
    this is belt-and-braces, not the primary defense."""
    mod = OccupancyModule.__new__(OccupancyModule)  # avoid loading a real model
    mod._extra_labels = set()
    mod._requested_class_ids = [0]
    OccupancyModule.set_requested_labels(mod, {"bottle", "not_a_real_coco_class", "banana"})
    assert mod._extra_labels == {"bottle", "banana"}
    name_to_id = {name: cid for cid, name in COCO_NAMES.items()}
    assert mod._requested_class_ids == sorted({0, name_to_id["bottle"], name_to_id["banana"]})


def test_a_detector_shim_with_no_cls_attribute_still_yields_people(module):
    """Older/CPU-fallback shims may not expose `.cls` at all — process() must
    fall back to treating every box as a person rather than raising."""
    class _NoClassBox:
        def __init__(self, xyxy, conf=0.9):
            self.xyxy = [np.array(xyxy, dtype=np.float32)]
            self.conf = [np.float32(conf)]
            # deliberately no .cls

    class _ShimModel:
        def predict(self, frame, **kwargs):
            return [_FakeResult([_NoClassBox(BOX)])]

    module._model = _ShimModel()
    events = module.process(FRAME)  # must not raise AttributeError
    assert module.get_detections().labels() == {"person"}
