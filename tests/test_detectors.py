"""Unit tests for the shared detector utilities (IoU box math, ROI-crop inference)."""

import numpy as np
import pytest

from core.detectors import iou, predict_roi

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.cls = [np.float32(cls)]
        self.conf = [np.float32(conf)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Records the crop it was actually called with, and returns scripted
    boxes in CROP-relative coordinates (matching a real detector's contract:
    it never sees the rest of the frame)."""

    def __init__(self, boxes_in_crop_coords):
        self._boxes = boxes_in_crop_coords
        self.last_crop_shape = None

    def predict(self, frame, **kwargs):
        self.last_crop_shape = frame.shape
        return [_FakeResult([_FakeBox(xyxy, cls, conf) for xyxy, cls, conf in self._boxes])]


def test_predict_roi_translates_crop_coordinates_to_frame_coordinates():
    model = _FakeModel([((5, 5, 25, 25), 0, 0.9)])
    [(box, cls, conf)] = predict_roi(model, FRAME, roi=(100, 100, 200, 200))
    assert box == (105, 105, 125, 125)
    assert cls == 0
    assert conf == pytest.approx(0.9, abs=1e-4)  # round-tripped through float32


def test_predict_roi_only_sees_the_cropped_region_not_the_full_frame():
    model = _FakeModel([])
    predict_roi(model, FRAME, roi=(100, 100, 180, 160))
    assert model.last_crop_shape[:2] == (60, 80)  # (height, width) of the crop, not (480, 640)


def test_predict_roi_clamps_the_roi_to_frame_bounds():
    model = _FakeModel([((0, 0, 10, 10), 0, 0.9)])
    # Runs off the left/top edge of the frame -- must clamp, not crash or wrap.
    [(box, _cls, conf)] = predict_roi(model, FRAME, roi=(-50, -50, 50, 50))
    assert box == (0, 0, 10, 10)
    assert conf == pytest.approx(0.9, abs=1e-4)


def test_predict_roi_returns_nothing_for_a_degenerate_roi():
    model = _FakeModel([((0, 0, 10, 10), 0, 0.9)])
    assert predict_roi(model, FRAME, roi=(700, 700, 720, 720)) == []  # entirely off-frame
    assert predict_roi(model, FRAME, roi=(100, 100, 100, 100)) == []  # zero area


def test_predict_roi_passes_through_multiple_boxes():
    model = _FakeModel([((0, 0, 10, 10), 1, 0.5), ((20, 20, 30, 30), 2, 0.8)])
    results = predict_roi(model, FRAME, roi=(50, 50, 150, 150))
    assert len(results) == 2
    assert {r[1] for r in results} == {1, 2}


def test_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    # intersection: (5,5)-(10,10) = 25; union = 100 + 100 - 25 = 175
    assert abs(iou(a, b) - 25 / 175) < 1e-9


def test_iou_one_box_fully_inside_another():
    outer = (0, 0, 20, 20)
    inner = (5, 5, 10, 10)
    # intersection = area of inner = 25; union = 400 + 25 - 25 = 400
    assert abs(iou(outer, inner) - 25 / 400) < 1e-9
