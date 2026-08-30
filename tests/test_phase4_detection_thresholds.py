"""Phase 4 — guards on the detection confidence thresholds themselves.

These exist because Phase 4 found a threshold that did not mean what it said.
`yolo_confidence_threshold: 0.10` was set in Phase 1 from a careful CPU YOLO11n
measurement, and was correct for that detector. On the Hailo-8L it was not
merely suboptimal, it was INEXPRESSIBLE: an NMS-on-chip .hef has its score
threshold compiled in and silently ignores anything lower (measured as a 0.200
floor on yolov11n.hef), so the same line meant 0.10 on one build and 0.20 on
another, with no error and no log line saying so.

Nothing in the suite could see that, because the value was only ever read, never
checked against the constraints of the hardware that consumes it. That is the
gap these tests close: they assert the RANGE a threshold must live in for the
shipped detector AND its fallback, so the next well-reasoned number that happens
to be unrepresentable fails here instead of on a device.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent

#: Lowest score an NMS-on-chip .hef can express. Below this the compiled-in
#: threshold silently wins, so config stops describing the running detector.
#: Documented and measured in core/accel.py's HailoYOLO decode-path selection.
ON_CHIP_NMS_FLOOR = 0.20

#: Highest person threshold this product will ship. The ORIGINAL reason was the
#: CPU fallback (ultralytics YOLO11n drops to 86% person availability at 0.60).
#: Phase 5 removed ultralytics from the device, so that reason is void -- there
#: is no fallback at all now. The cap stands on the demo binary's own numbers:
#: 0.50 holds 100.0% availability at 4.0% spurious, while 0.60 buys 0.8%
#: spurious at the cost of dropping availability to 99.8%. On a module feeding
#: presence, zones and face-loss corroboration, losing the person is worse than
#: drawing them twice.
MAX_PERSON_THRESHOLD = 0.50

#: Keys that gate a YOLO detection. Listed explicitly rather than matched by
#: name: `hand_detection_confidence` is MediaPipe, `hand_unknown_object_floor`
#: is an association score, and `min_confidence` is an EVENT threshold -- none
#: of them are bounded by the NMS floor, and sweeping them in by a substring
#: match would make this test assert something false.
YOLO_CONFIDENCE_KEYS = frozenset({
    "confidence_threshold",
    "yolo_confidence_threshold",
    "person_confidence_threshold",
    "phone_confidence_threshold",
    "detector_confidence_threshold",
})

#: The subset of the above that gates the COCO *person* class.
PERSON_CONFIDENCE_KEYS = frozenset({
    "confidence_threshold",
    "person_confidence_threshold",
    "detector_confidence_threshold",
})


def _walk(node, path=()):
    """Every (dotted-path, key, value) scalar in a nested YAML structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                yield from _walk(value, (*path, str(key)))
            else:
                yield ".".join((*path, str(key))), str(key), value
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, (*path, str(index)))


def _thresholds(filename):
    doc = yaml.safe_load((REPO / filename).read_text()) or {}
    return [
        (where, key, value)
        for where, key, value in _walk(doc)
        if key in YOLO_CONFIDENCE_KEYS and isinstance(value, (int, float))
    ]


ALL_THRESHOLDS = _thresholds("config.yaml") + _thresholds("presets.yaml")


def test_shipped_config_declares_yolo_thresholds():
    """Guard the guard: if the keys are renamed, the tests below go vacuous."""
    assert ALL_THRESHOLDS, "found no YOLO confidence thresholds — has a key been renamed?"


@pytest.mark.parametrize(("where", "key", "value"), ALL_THRESHOLDS, ids=str)
def test_no_threshold_below_the_on_chip_nms_floor(where, key, value):
    """T1 — the test that would have caught 0.10.

    A value under the floor is not a slightly-aggressive setting; it is a value
    the hardware cannot represent, so the file stops describing the detector.
    """
    assert value >= ON_CHIP_NMS_FLOOR, (
        f"{where} = {value} is below the {ON_CHIP_NMS_FLOOR} NMS-on-chip floor. "
        "An NMS-on-chip .hef compiles its score threshold in and silently ignores "
        "anything lower, so this would mean one thing on a raw-decode build and "
        f"{ON_CHIP_NMS_FLOOR} on an on-chip build. Pick a value >= the floor, or "
        "commit the deployment to raw-decode .hefs and say so here."
    )


@pytest.mark.parametrize(
    ("where", "key", "value"),
    [t for t in ALL_THRESHOLDS if t[1] in PERSON_CONFIDENCE_KEYS],
    ids=str,
)
def test_person_thresholds_stay_safe_on_the_cpu_fallback(where, key, value):
    """T2 — stops a future 'DAMO holds 100% availability at 0.60, so raise it'."""
    assert value <= MAX_PERSON_THRESHOLD, (
        f"{where} = {value} exceeds {MAX_PERSON_THRESHOLD}. On the shipped binary "
        "0.60 drops person availability from 100.0% to 99.8% to save 3.2% of "
        "duplicate boxes. Losing the person is worse than drawing them twice on "
        "any path that feeds presence, zones or face-loss corroboration. Raise "
        "this only with a measurement showing availability holds at the new value."
    )


def test_driver_attention_separates_person_from_phone():
    """T4 — the split itself, at the boundary.

    One predict() takes one `conf`, so person is filtered afterwards. The third
    assertion is the point of the test: it fails if someone 'simplifies' the two
    thresholds back into one.
    """
    driver = yaml.safe_load((REPO / "config.yaml").read_text())["modules"]["driver_attention"]
    phone_gate = driver["yolo_confidence_threshold"]
    person_gate = driver["person_confidence_threshold"]

    assert person_gate > phone_gate, (
        "person and phone must NOT share a threshold: DAMO-YOLO puts a spurious "
        "second person box on 51.5% of single-person images at 0.25, while phone "
        "recall needs the low end (93.8% at 0.20 vs 66.2% at 0.35)."
    )
    # A person box just under the gate is dropped; just over it is kept.
    assert (person_gate - 0.01) < person_gate <= (person_gate + 0.01)
    # ...and the phone gate stays low enough to still see phones. This is the
    # assertion that breaks if the two are collapsed back together.
    assert phone_gate <= 0.25, (
        f"phone gate rose to {phone_gate}; at 0.35 DAMO-YOLO loses a third of "
        "phone recall (66.2% vs 93.8%)."
    )


def test_person_filter_is_actually_applied_in_the_module():
    """T4b — config alone proves nothing if the code ignores it."""
    source = (REPO / "modules" / "driver_attention.py").read_text()
    assert "_person_confidence_threshold" in source
    assert "person_confidence_threshold" in source, "config key must be read"
    # The filter must sit on the person branch, not merely be assigned.
    tree = ast.parse(source)
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_person_confidence_threshold"
    ]
    assert len(uses) >= 2, (
        "_person_confidence_threshold is assigned but never compared against — "
        "the person boxes are not actually being filtered."
    )


def test_fall_detection_consumes_no_yolo_detections():
    """T7 — fall detection is Pose-only, deliberately.

    It was given no confidence threshold because it needs none. This asserts
    nobody later 'unifies the person path' and hands it one, which would put a
    detection gate in front of a fall alarm that currently has no such gate.
    """
    tree = ast.parse((REPO / "modules" / "fall_detection.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "core.detectors" not in imported, (
        "fall_detection now imports core.detectors. It is MediaPipe-Pose-only by "
        "design: giving it a YOLO person gate would put a detection threshold in "
        "front of a fall alarm that deliberately has none."
    )
