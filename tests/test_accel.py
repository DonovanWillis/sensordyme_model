"""Tests for accelerator detection, backend selection, and the ultralytics shims."""

import numpy as np

from core import accel
from core.accel import AcceleratorInfo, _ShimBox, _ShimResult


def teardown_function():
    accel.configure({}, info=None)  # reset module state between tests


def test_detect_on_dev_machine_reports_no_hailo():
    info = accel.detect()
    assert info.present is False  # no /dev/hailo0 on a Mac
    assert info.usable is False
    assert info.kind == "none"


def test_no_hef_configured_falls_back_to_cpu():
    accel.configure({"enabled": True, "yolo_hef": None})
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    healthy, detail = accel.status()
    assert healthy  # CPU-only with no HAT is the intended mode
    assert "no Hailo device" in detail


def test_disabled_accelerator_is_healthy_and_never_selected():
    accel.configure({"enabled": False, "yolo_hef": "/some/model.hef"})
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    healthy, detail = accel.status()
    assert healthy and "disabled" in detail


def test_usable_hat_without_hef_reports_degraded():
    fake = AcceleratorInfo(present=True, runtime_available=True, kind="hailo", detail="ok")
    accel.configure({"enabled": True, "yolo_hef": None}, info=fake)
    healthy, detail = accel.status()
    assert not healthy
    assert "no compiled model" in detail


def test_missing_hef_file_records_reason_and_falls_back():
    fake = AcceleratorInfo(present=True, runtime_available=True, kind="hailo", detail="ok")
    accel.configure({"enabled": True, "yolo_hef": "/nope/model.hef"}, info=fake)
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    healthy, detail = accel.status()
    assert not healthy
    assert "missing file" in detail


def test_shims_match_the_ultralytics_surface_modules_use():
    box = _ShimBox(np.array([10.0, 20.0, 110.0, 220.0]), cls_id=67, conf=0.9)
    result = _ShimResult([box])
    # Exactly the access patterns in modules/occupancy.py and driver_attention.py:
    for r in [result]:
        for b in r.boxes:
            assert [int(v) for v in b.xyxy[0].tolist()] == [10, 20, 110, 220]
            assert int(b.cls[0]) == 67
            assert float(b.conf[0]) > 0.89
    assert accel.COCO_NAMES[67] == "cell phone"
    assert accel.COCO_NAMES[0] == "person"


# The exact class ids the bottle/banana investigation depended on. Standard
# COCO order (https://cocodataset.org, and identical in the Ultralytics and
# Hailo Model Zoo COCO checkpoints this repo actually loads) — pinned here so
# a future edit to COCO_NAMES that shifts an index is caught immediately,
# rather than silently mislabeling whatever detector runs on top of it. This
# is the single mapping every consumer (core/accel.py's HailoYOLO.names,
# modules/occupancy.py's get_detections, modules/driver_attention.py's
# yolo_classes config) reads from — verified correct here once rather than
# trusted separately in each.
_KNOWN_COCO_IDS = {
    0: "person", 2: "car", 39: "bottle", 46: "banana", 56: "chair", 67: "cell phone",
}


def test_coco_names_pins_the_classes_the_bottle_banana_investigation_checked():
    for class_id, name in _KNOWN_COCO_IDS.items():
        assert accel.COCO_NAMES[class_id] == name, (
            f"COCO_NAMES[{class_id}] is {accel.COCO_NAMES.get(class_id)!r}, expected {name!r} — "
            "a shifted index here mislabels every detection built on top of it"
        )
    # bottle and banana specifically: 7 apart, not adjacent — a real class-id
    # confusion between them would have to be a genuine indexing bug, not a
    # simple off-by-one, which is worth asserting explicitly given that was
    # the exact reported symptom.
    assert accel.COCO_NAMES[39] == "bottle"
    assert accel.COCO_NAMES[46] == "banana"
    assert accel.COCO_NAMES[39] != accel.COCO_NAMES[46]


def test_coco_names_is_the_full_standard_80_class_set_with_no_gaps_or_dupes():
    assert set(accel.COCO_NAMES.keys()) == set(range(80)), "COCO_NAMES must cover ids 0..79 with no gaps"
    names = list(accel.COCO_NAMES.values())
    assert len(names) == len(set(names)), "no two class ids should share the same name"
