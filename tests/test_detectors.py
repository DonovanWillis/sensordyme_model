"""Unit tests for the shared detector utilities (IoU box math)."""

from core.detectors import iou


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
