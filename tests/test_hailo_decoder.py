"""Mission 5: the DAMO-YOLO decoder candidate (core/accel.py HailoYOLO, decoder="reference").

First the two shipped defects are reproduced on hand-built head tensors, then the
candidate is checked against an independent transcription of DAMO-YOLO's own
decode (damo/base_models/heads/zero_head.py: get_single_level_center_priors,
Integral, distance2bbox; damo/utils/boxes.py: multiclass_nms selection), and the
default is checked to be unchanged ("legacy").
"""

# ruff: noqa: F811 - pytest fixture imported from another module

from __future__ import annotations

import numpy as np
import pytest
from test_hailo_lifecycle import hailo  # noqa: F401 - fixture

from core import accel
from core.accel import HailoYOLO
from core.config_schema import validate

BINS = 17
STRIDES = (8, 16, 32)


def _empty_heads(input_size: int = 640):
    heads = {}
    for stride in STRIDES:
        g = input_size // stride
        heads[stride] = (np.full((g, g, 4 * BINS), -20.0, np.float32), np.zeros((g, g, 81), np.float32))
    return heads


def _peak(box_head, gy, gx, dists):
    """Make the DFL distribution at (gy, gx) a sharp peak at the given bin per side."""
    for side, d in enumerate(dists):
        box_head[gy, gx, side * BINS + d] = 20.0


def _device(heads, decoder: str, nms_iou: float = 0.5) -> tuple[HailoYOLO, dict]:
    device = HailoYOLO.__new__(HailoYOLO)
    device.decoder, device._nms_iou = decoder, nms_iou
    device._scales, raw = [], {}
    for stride, (box, cls) in heads.items():
        g = box.shape[0]
        raw[f"b{stride}"], raw[f"c{stride}"] = box[None], cls[None]
        device._scales.append({"box": f"b{stride}", "cls": f"c{stride}", "grid": g, "stride": stride})
    return device, raw


def _decode(heads, decoder, conf=0.3, scale=1.0, left=0, top=0, w=640, h=640, nms_iou=0.5):
    device, raw = _device(heads, decoder, nms_iou)
    (result,) = device._decode_raw(raw, scale, left, top, w, h, None, conf)
    return sorted((int(b.cls[0]), round(float(b.conf[0]), 4), *(round(float(v), 3) for v in b.xyxy[0]))
                  for b in result.boxes)


def _reference(heads, conf, scale=1.0, left=0, top=0, w=640, h=640):
    """Independent numpy transcription of DAMO-YOLO's decode (no suppression)."""
    out = []
    for stride, (box, cls) in heads.items():
        g = box.shape[0]
        x = np.tile(np.arange(g) * stride, g)          # x_range.repeat(h, 1).flatten()
        y = np.repeat(np.arange(g) * stride, g)        # y_range.unsqueeze(-1).repeat(1, w).flatten()
        logits = box.reshape(-1, 4, BINS).astype(np.float64)
        prob = np.exp(logits - logits.max(axis=2, keepdims=True))
        prob /= prob.sum(axis=2, keepdims=True)
        dist = prob @ np.linspace(0, BINS - 1, BINS) * stride   # Integral(...) * priors[..., 2]
        scores = cls.reshape(-1, cls.shape[-1])[:, :80]
        for a, c in zip(*np.nonzero(scores >= conf), strict=True):   # every (anchor, class) pair
            x1, y1 = x[a] - dist[a, 0], y[a] - dist[a, 1]              # distance2bbox
            x2, y2 = x[a] + dist[a, 2], y[a] + dist[a, 3]
            mapped = [min(max((v - off) / scale, 0.0), lim)
                      for v, off, lim in ((x1, left, w), (y1, top, h), (x2, left, w), (y2, top, h))]
            out.append((int(c), round(float(scores[a, c]), 4), *(round(v, 3) for v in mapped)))
    return sorted(out)


# -- the two shipped defects, reproduced -----------------------------------------------

def test_legacy_shifts_every_box_by_half_a_stride():
    heads = _empty_heads()
    box, cls = heads[16]
    _peak(box, gy=10, gx=12, dists=(2, 3, 4, 5))
    cls[10, 12, 0] = 0.9  # a person
    truth = [(0, 0.9, 12 * 16 - 2 * 16, 10 * 16 - 3 * 16, 12 * 16 + 4 * 16, 10 * 16 + 5 * 16)]
    assert _reference(heads, 0.3) == truth
    legacy = _decode(heads, "legacy")
    assert legacy == [(0, 0.9, *(v + 8 for v in truth[0][2:]))], "legacy box is 8 px (stride/2) right and down"
    assert _decode(heads, "reference") == truth


def test_legacy_drops_a_phone_that_shares_an_anchor_with_a_stronger_class():
    heads = _empty_heads()
    box, cls = heads[32]
    _peak(box, gy=5, gx=5, dists=(1, 1, 1, 1))
    cls[5, 5, 0], cls[5, 5, 67] = 0.9, 0.6  # person beats cell phone at the same anchor
    assert [d[0] for d in _decode(heads, "legacy")] == [0], "the phone is discarded"
    assert sorted(d[0] for d in _decode(heads, "reference")) == [0, 67]


# -- the candidate equals the reference ------------------------------------------------

@pytest.mark.parametrize(("seed", "scale", "left", "top"), [(0, 1.0, 0, 0), (1, 1.0, 0, 80), (2, 0.5, 0, 80)])
def test_the_candidate_matches_damo_yolos_own_decode(seed, scale, left, top):
    rng = np.random.default_rng(seed)
    heads = {}
    for stride in STRIDES:
        g = 640 // stride
        heads[stride] = (rng.normal(0, 2, (g, g, 4 * BINS)).astype(np.float32),
                         (rng.random((g, g, 81)) ** 200).astype(np.float32))
    w, h = (1280, 960) if scale == 0.5 else (640, 480 if top else 640)
    expected = _reference(heads, 0.9, scale, left, top, w, h)
    assert 5 < len(expected) < HailoYOLO._REFERENCE_MAX_DETECTIONS
    got = _decode(heads, "reference", 0.9, scale, left, top, w, h, nms_iou=1.0)  # 1.0 = no suppression
    assert [g[:2] for g in got] == [e[:2] for e in expected]
    assert np.allclose([g[2:] for g in got], [e[2:] for e in expected], atol=0.02)


def test_padding_channels_are_never_a_class():
    heads = _empty_heads()
    box, cls = heads[8]
    _peak(box, 3, 3, (1, 1, 1, 1))
    cls[3, 3, 80] = 0.99  # channel 81 is padding
    assert _decode(heads, "reference") == []


# -- nothing changes unless configured -------------------------------------------------

def test_the_shipped_default_is_legacy(hailo):
    model = accel.maybe_hailo_yolo("yolo11n.pt")
    assert model is not None and model.decoder == "legacy"


def test_the_candidate_is_selected_only_by_config(hailo):
    accel.configure({"yolo_hef": str(hailo.path), "yolo_hef_sha256": hailo.sha, "yolo_decoder": "reference"},
                    info=accel.AcceleratorInfo(True, True, "hailo", "fake HAT"))
    from core import detectors

    detectors.release_all()
    model = accel.maybe_hailo_yolo("yolo11n.pt")
    assert model is not None and model.decoder == "reference"


def test_an_unknown_decoder_is_refused():
    with pytest.raises(ValueError, match="unknown decoder"):
        HailoYOLO.__init__(HailoYOLO.__new__(HailoYOLO), "x.hef", decoder="fast")
    errors, _ = validate({"accelerator": {"yolo_decoder": "fast"}})
    assert any("yolo_decoder" in e for e in errors)
