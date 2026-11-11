"""Calibrated head-pose intrinsics are ACCEPTED (values come from hardware calibration, not here).

Synthetic: landmarks are projected through a known camera, then solved back.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from core.behavior.camera_model import CameraIntrinsics, validate_intrinsics
from core.config_schema import validate
from modules.driver_attention import (
    _MODEL_POINTS_3D,
    AttentionCalibrator,
    _camera_matrix,
    estimate_head_pose_deg,
)

REPO = Path(__file__).resolve().parent.parent
W, H = 640, 480
_R_NEUTRAL = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

#: A wide lens (~102 deg HFOV, like Camera Module 3 Wide) with barrel distortion,
#: specified at 2x the processed resolution.
WIDE = {"width": 1280, "height": 960, "fx": 518.0, "fy": 518.0, "cx": 640.0, "cy": 480.0,
        "distortion": [-0.12, 0.03, 0.0, 0.0, 0.0]}


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def _project(rotation, k, dist):
    rvec, _ = cv2.Rodrigues(rotation)
    pts, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, np.array([60.0, 0.0, 450.0]), k, dist)
    return pts.reshape(-1, 2)


def test_null_keeps_the_historical_matrix_exactly():
    assert CameraIntrinsics.from_config(None) is None
    assert np.array_equal(_camera_matrix((H, W)), _camera_matrix((H, W), None, None))
    assert _camera_matrix((H, W))[0, 0] == W


def test_intrinsics_rescale_to_the_processed_frame():
    intr = CameraIntrinsics.from_config(WIDE)
    k = intr.matrix((H, W))
    assert k[0, 0] == pytest.approx(259.0) and k[0, 2] == pytest.approx(320.0) and k[1, 2] == pytest.approx(240.0)
    assert intr.distortion_coefficients().ravel().tolist() == WIDE["distortion"]
    with pytest.raises(ValueError, match="aspect ratio"):
        intr.matrix((360, 640))


def test_calibrated_intrinsics_recover_a_turn_the_default_assumption_misreads():
    intr = CameraIntrinsics.from_config(WIDE)
    k, dist = intr.matrix((H, W)), intr.distortion_coefficients()
    neutral = estimate_head_pose_deg(_project(_R_NEUTRAL, k, dist), (H, W), intrinsics=intr)
    turned = estimate_head_pose_deg(_project(_rot_y(35.0) @ _R_NEUTRAL, k, dist), (H, W), intrinsics=intr)
    calibrated_delta = abs(turned[0] - neutral[0])
    assert calibrated_delta == pytest.approx(35.0, abs=1.0)
    n0 = estimate_head_pose_deg(_project(_R_NEUTRAL, k, dist), (H, W))
    t0 = estimate_head_pose_deg(_project(_rot_y(35.0) @ _R_NEUTRAL, k, dist), (H, W))
    assert abs(abs(t0[0] - n0[0]) - 35.0) > abs(calibrated_delta - 35.0)


@pytest.mark.parametrize(("value", "expected"), [
    ("wide", "must be null or a mapping"),
    ({**WIDE, "fx": 0}, "fx:` must be a positive number"),
    ({k: v for k, v in WIDE.items() if k != "cy"}, "cy:` is required"),
    ({**WIDE, "distortion": [0.1]}, "list of 4, 5 or 8 numbers"),
    ({**WIDE, "skew": 1}, "unknown keys"),
])
def test_bad_intrinsics_are_errors(value, expected):
    assert any(expected in e for e in validate_intrinsics(value))


def test_schema_rejects_both_hfov_and_intrinsics():
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert config["modules"]["driver_attention"]["camera_intrinsics"] is None
    assert validate(config)[0] == []
    both = copy.deepcopy(config)
    both["modules"]["driver_attention"].update(camera_intrinsics=WIDE, camera_hfov_deg=102)
    assert any("not both" in e for e in validate(both)[0])
    only = copy.deepcopy(config)
    only["modules"]["driver_attention"]["camera_intrinsics"] = WIDE
    assert validate(only)[0] == []


def _calibrate(path, intrinsics=None):
    cal = AttentionCalibrator(path, 1.0, 0.75, 0.2, False, camera_intrinsics=intrinsics)
    for i in range(30):
        cal.observe(0.4, _R_NEUTRAL, None, i * 0.05)
    assert cal.is_calibrated
    return cal


def test_calibration_file_is_unchanged_without_intrinsics_and_bound_to_them_with(tmp_path):
    path = tmp_path / "calibration.json"
    _calibrate(path)
    assert "camera_intrinsics" not in json.loads(path.read_text())
    assert AttentionCalibrator(path, 1.0, 0.75, 0.2, False).is_calibrated  # still loads
    intr = CameraIntrinsics.from_config(WIDE)
    assert not AttentionCalibrator(path, 1.0, 0.75, 0.2, False, camera_intrinsics=intr).is_calibrated
    _calibrate(path, intr)
    assert AttentionCalibrator(path, 1.0, 0.75, 0.2, False, camera_intrinsics=intr).is_calibrated
    other = CameraIntrinsics.from_config({**WIDE, "fx": 600.0, "fy": 600.0})
    assert not AttentionCalibrator(path, 1.0, 0.75, 0.2, False, camera_intrinsics=other).is_calibrated
    assert not AttentionCalibrator(path, 1.0, 0.75, 0.2, False).is_calibrated
