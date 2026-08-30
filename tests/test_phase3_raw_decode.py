"""Off-chip decoding for .hef files that ship without on-chip NMS.

WHY THIS PATH EXISTS. A Hailo Model Zoo .hef either does NMS on-chip or emits raw
per-scale tensors. The distinction is not a detail — it decides whether the
detector can see a phone:

    yolov11n.hef   on-chip NMS, score threshold 0.200 COMPILED IN
                   -> phone recall 25.4%, and `conf` below 0.200 is a silent no-op
    damoyolo.hef   no NMS, six raw tensors, threshold entirely ours
                   -> phone recall 93.8% at the same measured operating point

Both measured on a real Pi 5 + Hailo-8L against the same 600 subject-disjoint
images. The raw path costs ~8 ms of host CPU per frame (23.6 ms vs 15.2 ms) and
buys back 68 points of recall.

WHAT IS PINNED HERE. The decode is hand-written, which is exactly where silent
accuracy bugs live: a wrong stride, a transposed DFL layout or an off-by-half
anchor produces boxes that look plausible and are wrong. These tests pin the
geometry against hand-computed expectations rather than against the
implementation's own output.

The layout was established EMPIRICALLY against the real damoyolo_tinynasL20_T.hef
(not assumed from the architecture paper): 68-channel heads carry raw DFL logits
(observed range -3.3..9.6, mean ~0); 81-channel heads carry already-sigmoid'd
probabilities (observed 0..1); channel 0 lit at 0.85 on a real driver image,
confirming channels map directly to COCO ids with `person` at 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.accel import COCO_NAMES, HailoYOLO, _ShimResult, _nms


# ----------------------------------------------------------------- NMS
def test_nms_keeps_the_higher_scoring_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 105, 105]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep = _nms(boxes, scores, iou_threshold=0.5)

    assert keep == [0]


def test_nms_keeps_boxes_that_do_not_overlap():
    boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    assert sorted(_nms(boxes, scores, iou_threshold=0.5)) == [0, 1]


def test_nms_threshold_is_actually_honoured():
    """Two boxes at a known IoU must be kept or merged depending on the threshold."""
    # 100x100 boxes offset by 50 in x: intersection 50x100, union 150x100 -> IoU = 1/3.
    boxes = np.array([[0, 0, 100, 100], [50, 0, 150, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    assert len(_nms(boxes, scores, iou_threshold=0.5)) == 2, "IoU 0.33 < 0.5: both kept"
    assert len(_nms(boxes, scores, iou_threshold=0.3)) == 1, "IoU 0.33 > 0.3: merged"


def test_nms_does_not_divide_by_zero_on_a_degenerate_box():
    """A zero-area box must not produce NaN and silently keep or drop detections."""
    boxes = np.array([[10, 10, 10, 10], [0, 0, 100, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    keep = _nms(boxes, scores, iou_threshold=0.5)

    assert len(keep) == 2 and not any(np.isnan(scores[k]) for k in keep)


def test_nms_on_nothing_returns_nothing():
    assert _nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), 0.5) == []


# ----------------------------------------------------- scale/head grouping
class _Info:
    """Stands in for a HailoRT vstream info; only .name and .shape are read."""

    def __init__(self, name, shape):
        self.name, self.shape = name, shape


def _yolo(**attrs) -> HailoYOLO:
    """A HailoYOLO without __init__ — constructing one needs a real Hailo device."""
    model = HailoYOLO.__new__(HailoYOLO)
    model._input_h = model._input_w = 640
    model._nms_iou = 0.5
    model.names = dict(COCO_NAMES)
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


def test_heads_are_paired_by_shape_not_by_layer_name():
    """Layer names are vendor-specific; 4*17 channels is an architectural fact."""
    infos = [
        _Info("m/conv53", (80, 80, 68)), _Info("m/conv54", (80, 80, 81)),
        _Info("m/conv65", (40, 40, 68)), _Info("m/conv66", (40, 40, 81)),
        _Info("m/conv76", (20, 20, 68)), _Info("m/conv77", (20, 20, 81)),
    ]

    scales = _yolo()._group_raw_outputs(infos)

    assert [s["stride"] for s in scales] == [8, 16, 32], "strides must be input/grid"
    assert [s["grid"] for s in scales] == [80, 40, 20]
    for s in scales:
        assert s["box"].endswith(("53", "65", "76")), "68-channel head is the box head"
        assert s["cls"].endswith(("54", "66", "77"))


def test_a_head_layout_this_decoder_does_not_understand_is_refused():
    """Guessing at an unknown layout would silently produce wrong boxes."""
    infos = [_Info("m/a", (80, 80, 81)), _Info("m/b", (40, 40, 81))]  # no box heads

    with pytest.raises(RuntimeError, match="missing a box or class head"):
        _yolo()._group_raw_outputs(infos)


def test_a_non_square_grid_is_refused():
    with pytest.raises(RuntimeError, match="non-square"):
        _yolo()._group_raw_outputs([_Info("m/a", (80, 40, 68)), _Info("m/b", (80, 40, 81))])


# ----------------------------------------------------------- DFL decode
def _heads(grid: int, *, cls_id: int, cell: tuple[int, int], distance_bin: int,
           score: float, bins: int = 17):
    """One scale's raw tensors with a single object planted at a known cell.

    The box head is built so the DFL softmax collapses onto `distance_bin`: one
    bin gets a large logit, the rest zero, so the expectation is that bin's index
    and the decoded distance is exactly `distance_bin * stride`. That makes the
    expected box hand-computable, which is the point — it checks the geometry
    rather than agreeing with whatever the code happens to do.
    """
    box = np.zeros((grid, grid, 4 * bins), dtype=np.float32)
    cls = np.zeros((grid, grid, 81), dtype=np.float32)
    row, col = cell
    for side in range(4):
        box[row, col, side * bins + distance_bin] = 50.0   # dominates the softmax
    cls[row, col, cls_id] = score
    return box, cls


def test_dfl_decode_places_the_box_where_the_arithmetic_says_it_should():
    """One scale, one object, hand-computed expected box."""
    grid, stride, bins, dist_bin = 20, 32, 17, 4
    row = col = 10
    box, cls = _heads(grid, cls_id=0, cell=(row, col), distance_bin=dist_bin, score=0.9)

    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": stride, "box": "b", "cls": "c"}])
    result = model._decode_raw(
        {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)},
        scale=1.0, left=0, top=0, w=640, h=640, classes=None, conf=0.25,
    )[0]

    assert len(result.boxes) == 1
    x1, y1, x2, y2 = (float(v) for v in result.boxes[0].xyxy[0])
    # anchor centre = (col+0.5)*stride ; distance = dist_bin * stride on every side
    cx = (col + 0.5) * stride
    cy = (row + 0.5) * stride
    d = dist_bin * stride
    assert x1 == pytest.approx(cx - d, abs=0.5)
    assert y1 == pytest.approx(cy - d, abs=0.5)
    assert x2 == pytest.approx(cx + d, abs=0.5)
    assert y2 == pytest.approx(cy + d, abs=0.5)
    assert int(result.boxes[0].cls[0]) == 0
    assert float(result.boxes[0].conf[0]) == pytest.approx(0.9, abs=1e-5)


def test_the_letterbox_is_undone_so_boxes_land_in_original_frame_pixels():
    """A detection must come back in the caller's coordinates, not the model's."""
    grid, stride, bins, dist_bin = 20, 32, 17, 2
    row = col = 10
    box, cls = _heads(grid, cls_id=0, cell=(row, col), distance_bin=dist_bin, score=0.9)

    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": stride, "box": "b", "cls": "c"}])
    scale, left, top = 0.5, 40, 80
    result = model._decode_raw(
        {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)},
        scale=scale, left=left, top=top, w=1000, h=1000, classes=None, conf=0.25,
    )[0]

    x1, y1, _x2, _y2 = (float(v) for v in result.boxes[0].xyxy[0])
    cx, cy = (col + 0.5) * stride, (row + 0.5) * stride
    d = dist_bin * stride
    assert x1 == pytest.approx((cx - d - left) / scale, abs=0.5)
    assert y1 == pytest.approx((cy - d - top) / scale, abs=0.5)


def test_scores_below_conf_are_dropped():
    """`conf` is the ONLY gate on this path — that is the whole reason it exists."""
    grid = 20
    box, cls = _heads(grid, cls_id=0, cell=(5, 5), distance_bin=3, score=0.30)
    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": 32, "box": "b", "cls": "c"}])
    raw = {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)}

    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, None, conf=0.25)[0].boxes) == 1
    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, None, conf=0.35)[0].boxes) == 0


def test_conf_is_honoured_far_below_the_on_chip_floor():
    """The point of the raw path: 0.05 must actually mean 0.05.

    An on-chip build with a 0.200 threshold returns identical results for every
    conf <= 0.200 — that flat curve is what capped phone recall at 25.4%.
    """
    grid = 20
    box, cls = _heads(grid, cls_id=67, cell=(7, 7), distance_bin=2, score=0.08)
    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": 32, "box": "b", "cls": "c"}])
    raw = {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)}

    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, None, conf=0.05)[0].boxes) == 1, (
        "a 0.08-confidence detection was dropped at conf=0.05 — the raw path has "
        "acquired a floor it must not have"
    )
    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, None, conf=0.20)[0].boxes) == 0


def test_class_filtering_matches_the_on_chip_path():
    grid = 20
    box, cls = _heads(grid, cls_id=67, cell=(5, 5), distance_bin=3, score=0.9)
    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": 32, "box": "b", "cls": "c"}])
    raw = {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)}

    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, [67], 0.25)[0].boxes) == 1
    assert len(model._decode_raw(raw, 1.0, 0, 0, 640, 640, [0], 0.25)[0].boxes) == 0


def test_padding_channels_beyond_coco_are_never_emitted_as_detections():
    """The class head has 81 channels; COCO has 80. The extra is padding.

    Letting it through would invent detections for a class that does not exist —
    and `int(cls) == 80` would then index past COCO_NAMES in every consumer.
    """
    grid = 20
    box = np.zeros((grid, grid, 68), dtype=np.float32)
    cls = np.zeros((grid, grid, 81), dtype=np.float32)
    for side in range(4):
        box[5, 5, side * 17 + 3] = 50.0
    cls[5, 5, 80] = 0.99                      # the padding channel, lit hard

    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": 32, "box": "b", "cls": "c"}])
    result = model._decode_raw(
        {"b": np.expand_dims(box, 0), "c": np.expand_dims(cls, 0)},
        1.0, 0, 0, 640, 640, None, 0.25,
    )[0]

    assert result.boxes == [], "a padding channel was emitted as a class-80 detection"


def test_an_empty_frame_returns_an_empty_result_not_an_error():
    grid = 20
    model = _yolo(_raw_outputs=True,
                  _scales=[{"grid": grid, "stride": 32, "box": "b", "cls": "c"}])
    result = model._decode_raw(
        {"b": np.zeros((1, grid, grid, 68), dtype=np.float32),
         "c": np.zeros((1, grid, grid, 81), dtype=np.float32)},
        1.0, 0, 0, 640, 640, None, 0.25,
    )

    assert isinstance(result[0], _ShimResult) and result[0].boxes == []


def test_detections_from_every_scale_are_merged():
    """Three FPN levels, one object each — all three must survive to the caller."""
    scales, raw = [], {}
    for i, (grid, stride) in enumerate([(80, 8), (40, 16), (20, 32)]):
        box, cls = _heads(grid, cls_id=i, cell=(grid // 2, grid // 2),
                          distance_bin=2, score=0.9)
        raw[f"b{i}"] = np.expand_dims(box, 0)
        raw[f"c{i}"] = np.expand_dims(cls, 0)
        scales.append({"grid": grid, "stride": stride, "box": f"b{i}", "cls": f"c{i}"})

    result = _yolo(_raw_outputs=True, _scales=scales)._decode_raw(
        raw, 1.0, 0, 0, 640, 640, None, 0.25
    )[0]

    assert sorted(int(b.cls[0]) for b in result.boxes) == [0, 1, 2]
