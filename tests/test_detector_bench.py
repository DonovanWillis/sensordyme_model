"""Mission 4 detector benchmark: decoder parity with the device, and the scoring rules.

The benchmark's "production" DAMO-YOLO variant must decode exactly as the device
does (core/accel.py HailoYOLO._decode_raw); otherwise the offset/multilabel
comparisons would measure the benchmark, not the product.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.accel import HailoYOLO, letterbox_transform
from research.detector_bench import models as dm
from research.detector_bench import oid_eval as oe


def _heads(seed: int):
    rng = np.random.default_rng(seed)
    heads, raw, scales = [], {}, []
    for stride in (8, 16, 32):
        g = 640 // stride
        box = rng.normal(0, 2, (g, g, 68)).astype(np.float32)
        cls = (rng.random((g, g, 81)) ** 8).astype(np.float32)  # mostly low, a few confident
        heads.append((box, cls, stride))
        raw[f"b{stride}"], raw[f"c{stride}"] = box[None], cls[None]
        scales.append({"box": f"b{stride}", "cls": f"c{stride}", "grid": g, "stride": stride})
    return heads, raw, scales


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_production_variant_decodes_exactly_like_the_device(seed):
    heads, raw, scales = _heads(seed)
    device = HailoYOLO.__new__(HailoYOLO)
    device._scales, device._nms_iou = scales, 0.50
    (result,) = device._decode_raw(raw, 1.0, 0, 0, 640, 640, None, 0.3)
    device_top = sorted(result.boxes, key=lambda b: -float(b.conf[0]))[: dm.MAX_DETS]  # the bench keeps the top 100
    expected = sorted((int(b.cls[0]), round(float(b.conf[0]), 4), *(round(float(v), 2) for v in b.xyxy[0]))
                      for b in device_top)
    boxes, scores, classes = dm.decode_dfl(heads, conf=0.3, offset=0.5, multilabel=False, nms_iou=0.50)
    boxes = np.clip(boxes, 0, 640)
    got = sorted((int(c), round(float(s), 4), *(round(float(v), 2) for v in b))
                 for b, s, c in zip(boxes, scores, classes, strict=True))
    assert len(expected) > 5 and len(got) == len(expected)
    assert [g[:2] for g in got] == [e[:2] for e in expected]  # same classes and scores
    assert np.allclose([g[2:] for g in got], [e[2:] for e in expected], atol=0.02)  # same boxes (float32 on device)


def test_offset_zero_shifts_every_box_by_half_a_stride():
    heads, _, _ = _heads(3)
    a, _, _ = dm.decode_dfl(heads, conf=0.5, offset=0.5, multilabel=False, nms_iou=1.0)
    b, _, _ = dm.decode_dfl(heads, conf=0.5, offset=0.0, multilabel=False, nms_iou=1.0)
    shift = a - b
    assert set(np.round(np.unique(shift[:, 0]), 3)) <= {4.0, 8.0, 16.0}


def test_multilabel_keeps_a_class_that_loses_the_argmax():
    box = np.zeros((20, 20, 68), np.float32)
    cls = np.zeros((20, 20, 81), np.float32)
    cls[5, 5, 0], cls[5, 5, 67] = 0.9, 0.6  # person beats phone at the same anchor
    heads = [(box, cls, 32)]
    _, _, argmax = dm.decode_dfl(heads, conf=0.2, offset=0.0, multilabel=False, nms_iou=0.5)
    _, _, multi = dm.decode_dfl(heads, conf=0.2, offset=0.0, multilabel=True, nms_iou=0.5)
    assert list(argmax) == [0] and sorted(multi) == [0, 67]


def test_centered_letterbox_matches_the_device_geometry():
    from PIL import Image

    image = Image.new("RGB", (640, 480), (10, 20, 30))
    canvas, scale, left, top = dm.letterbox(image, 640, centered=True, pad=114)
    assert (scale, left, top) == letterbox_transform(640, 480, 640, 640)
    assert canvas[0, 0].tolist() == [114, 114, 114] and canvas[320, 320].tolist() == [10, 20, 30]


def _truth(gts, verified=True, negative=False, image="i"):
    return oe.ImageTruth(image, "g-" + image, [oe.GroundTruth(b, group_of=g) for b, g in gts], verified, negative)


def test_unverified_images_are_not_scored():
    truth = _truth([], verified=False)
    assert oe.match_image([(0.9, (0.1, 0.1, 0.2, 0.2))], truth) == []


def test_duplicates_are_false_positives_and_group_boxes_count_once():
    truth = _truth([((0.1, 0.1, 0.3, 0.3), False), ((0.5, 0.5, 0.9, 0.9), True)])
    scored = oe.match_image([(0.9, (0.1, 0.1, 0.3, 0.3)), (0.8, (0.11, 0.1, 0.3, 0.3)),
                            (0.7, (0.6, 0.6, 0.7, 0.7)), (0.6, (0.7, 0.7, 0.8, 0.8)),
                            (0.5, (0.0, 0.8, 0.05, 0.85))], truth)
    assert [(s.score, s.tp) for s in scored] == [(0.9, True), (0.8, False), (0.7, True), (0.5, False)]


def test_average_precision_and_operating_point():
    truths = [_truth([((0.1, 0.1, 0.3, 0.3), False)], image="a"),
              _truth([((0.1, 0.1, 0.3, 0.3), False)], image="b"),
              _truth([], negative=True, image="c")]
    preds = {"a": [(0.9, (0.1, 0.1, 0.3, 0.3))], "c": [(0.8, (0.4, 0.4, 0.5, 0.5))],
             "b": [(0.3, (0.1, 0.1, 0.3, 0.3))]}
    result = oe.evaluate_class(preds, truths)
    # ranked: TP(0.9) FP(0.8) TP(0.3): precision 1, 0.5, 0.667 at recall .5, .5, 1
    assert oe.average_precision(result.scored, result.npos) == pytest.approx(0.5 * 1 + 0.5 * (2 / 3))
    op = oe.operating_point(result, 0.5)
    assert (op["tp"], op["fp"], op["recall"], op["fp_per_negative_image"]) == (1, 1, 0.5, 1.0)
    assert op["negative_images_with_a_false_box"] == 1.0 and op["fp_background"] == 1
