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
