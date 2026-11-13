"""M8.3: the DEVICE's own inference path, end to end, without a Pi.

core.accel.HailoYOLO is built for real against a stand-in `hailo_platform`
(scripts/hw_validation/fake_hailo.py) whose .hef has the layout
`hailortcli parse-hef` reports for damoyolo_tinynasL20_T: six NHWC float32
outputs, grids 80/40/20, a 68-channel box head (DFL logits) and an 81-channel
class head (80 COCO sigmoid probabilities + 1 padding channel) per grid, in an
order that is not finest-first. Everything above `pipeline.infer` is the
production code: HEF parsing and head pairing, the UINT8 letterboxed RGB input,
decode, class-wise NMS, class mapping and the map back to frame pixels.

Expected values never come from core.accel itself. They come from:
  * hand arithmetic (a planted object must come back where it was planted);
  * scripts/hw_validation/hwv_decode.py, a separately written numpy decoder;
  * a deliberately naive per-anchor reference in this file (loops, explicit
    IoU matrix), used to check hwv_decode on sparse inputs.

Small synthetic fixtures only. The full float-ONNX comparison on Open Images val
runs on the fleet: research/m83/inference_path_val.py.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
HWV = REPO / "scripts" / "hw_validation"
sys.path.insert(0, str(HWV))

import fake_hailo  # noqa: E402
import hwv_decode as hd  # noqa: E402

from core import accel, detectors  # noqa: E402
from core.accel import HailoYOLO  # noqa: E402

PERSON, PHONE, CUP = 0, 67, 41


@pytest.fixture
def fake(monkeypatch):
    """Install the fake chip; yield a helper that builds HailoYOLO on it."""
    monkeypatch.setattr(accel, "record_inference", lambda: None)

    def build(backend=fake_hailo.zero_backend, outputs=None, decoder="legacy", keep_inputs=False):
        fake_hailo.install(backend, outputs=outputs, keep_inputs=keep_inputs)
        return HailoYOLO("fake.hef", sha256="0" * 64, decoder=decoder)

    yield build
    fake_hailo.uninstall()


def _rows(boxes, scores, classes) -> np.ndarray:
    rows = np.column_stack([np.asarray(scores, float), np.asarray(classes, float),
                            np.asarray(boxes, float).reshape(-1, 4)])
    return rows[np.lexsort((rows[:, 2], rows[:, 1], -rows[:, 0]))] if len(rows) else rows.reshape(0, 6)


def _shim_rows(result) -> np.ndarray:
    b = result.boxes
    return _rows([x.xyxy[0] for x in b], [float(x.conf[0]) for x in b], [int(x.cls[0]) for x in b])


def _random_heads(seed: int, hot: float = 12.0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, (g, _, c) in fake_hailo.OUTPUTS:
        if c == 68:
            out[name] = rng.normal(0, 2, (1, g, g, c)).astype(np.float32)
        else:  # mostly tiny probabilities, a few dozen above 0.2
            out[name] = (rng.random((1, g, g, c)) ** hot).astype(np.float32)
    return out


def _backend(heads: dict):
    def run(batch):
        fake_hailo.zero_backend(batch)
        return {k: v.copy() for k, v in heads.items()}
    return run


# ------------------------------------------------------------------ layout
def test_the_parse_hef_layout_pairs_into_strides_8_16_32_whatever_the_output_order(fake):
    expected = [(80, 8, 81), (40, 16, 81), (20, 32, 81)]
    for perm in list(itertools.permutations(fake_hailo.OUTPUTS))[::97]:  # a spread of orders
        model = fake(outputs=perm)
        assert model._raw_outputs is True
        got = [(s["grid"], s["stride"], s["num_classes"]) for s in model._scales]
        assert got == expected
        for s in model._scales:
            assert dict(perm)[s["box"]][2] == 68 and dict(perm)[s["cls"]][2] == 81
        model.close()


def test_the_input_is_nhwc_640_and_the_names_are_coco(fake):
    model = fake()
    assert (model._input_h, model._input_w) == (640, 640)
    assert model.names[PERSON] == "person" and model.names[PHONE] == "cell phone"
    assert len(model.names) == 80 and sorted(model.names) == list(range(80))
    model.close()


def test_every_class_the_driver_module_configures_exists_on_the_device_detector(fake):
    """driver_attention maps config names -> class ids through model.names. A
    misspelt name would silently never map; a name past COCO would never fire."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    model = fake()
    names = model.names
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "yolo_classes" in node and isinstance(node["yolo_classes"], dict):
                found.append(node["yolo_classes"])
            for value in node.values():
                walk(value)

    walk(config)
    assert found, "no yolo_classes block in config.yaml"
    for class_map in found:
        ids = {cid: class_map[n] for cid, n in names.items() if n in class_map}  # driver_attention.py:1522
        assert set(class_map) == {names[c] for c in ids}, "a configured class name is not a COCO name"
        assert ids.get(PHONE) == "phone_detected"
        assert all(0 <= c < 80 for c in ids)
    model.close()


# ------------------------------------------------------------------ preprocessing
def test_a_camera_frame_reaches_the_chip_as_an_rgb_letterbox_with_114_bars(fake):
    """640x480 BGR (core/camera.py) -> scale 1, 80-px bars top and bottom, RGB."""
    model = fake(keep_inputs=True)
    frame = np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    model.predict(frame, conf=0.2)
    (sent,) = fake_hailo.state.inputs_seen
    expected = np.full((1, 640, 640, 3), 114, dtype=np.uint8)
    expected[0, 80:560] = frame[..., ::-1]  # BGR -> RGB, no resampling at scale 1
    assert sent.dtype == np.uint8 and sent.shape == (1, 640, 640, 3)
    assert np.array_equal(sent, expected)
    model.close()


@pytest.mark.parametrize(("h", "w", "scale", "left", "top"), [
    (720, 1280, 0.5, 0, 140),      # 16:9 camera mode, downscaled
    (640, 480, 1.0, 80, 0),        # portrait: bars left and right
    (150, 200, 3.2, 0, 80),        # a hand-ROI crop (predict_roi), upscaled
])
def test_other_geometries_use_the_documented_scale_and_offsets(fake, h, w, scale, left, top):
    model = fake(keep_inputs=True)
    frame = np.random.default_rng(h + w).integers(0, 256, (h, w, 3), dtype=np.uint8)
    model.predict(frame, conf=0.2)
    (sent,) = fake_hailo.state.inputs_seen
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)[..., ::-1]
    assert np.array_equal(sent[0, top:top + new_h, left:left + new_w], resized)
    bars = np.ones((640, 640), bool)
    bars[top:top + new_h, left:left + new_w] = False
    assert (sent[0][bars] == 114).all()
    model.close()


# ------------------------------------------------------------------ planted objects
def test_a_planted_phone_comes_back_at_its_frame_coordinates(fake):
    """Hand arithmetic: a phone at frame (300,200)-(364,328) is input (300,280)-(364,408)."""
    frame_box = (300.0, 200.0, 364.0, 328.0)
    top = 80
    planted = [(PHONE, 0.9, frame_box[0], frame_box[1] + top, frame_box[2], frame_box[3] + top, 16)]
    model = fake(backend=fake_hailo.planted_backend(planted))
    (result,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2)
    assert len(result.boxes) == 1
    box = result.boxes[0]
    assert int(box.cls[0]) == PHONE and model.names[int(box.cls[0])] == "cell phone"
    assert float(box.conf[0]) == pytest.approx(0.9, abs=1e-6)
    assert np.allclose(box.xyxy[0], frame_box, atol=1e-3)
    model.close()


def test_the_damo_reference_convention_places_the_same_tensors_half_a_stride_up_left(fake):
    """The known, documented legacy/reference difference (docs/DECODER_CANDIDATE.md):
    on identical tensors the DAMO decode (anchor at cell*stride) lands every box
    exactly stride/2 up and left of the shipped decode. Measured, not changed."""
    planted = [(PHONE, 0.9, 300, 280, 364, 408, 16), (PERSON, 0.8, 100, 120, 420, 600, 32),
               (CUP, 0.7, 500, 300, 540, 340, 8)]
    heads = fake_hailo.planted_heads(planted)
    scales = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    legacy = hd.decode(heads, scales, hd.LEGACY, 0.2)
    ref = hd.decode(heads, scales, hd.REFERENCE, 0.2)
    for i, cls in enumerate(legacy.classes):
        j = int(np.nonzero(ref.classes == cls)[0][0])
        shift = legacy.boxes[i] - ref.boxes[j]
        assert np.allclose(shift, legacy.strides[i] / 2, atol=1e-3)


def test_a_phone_beside_a_person_survives_nms_and_same_class_duplicates_do_not(fake):
    planted = [
        (PERSON, 0.95, 120, 100, 520, 620, 32),
        (PHONE, 0.60, 300, 280, 364, 408, 16),   # inside the person box: different class, kept
        (PHONE, 0.50, 304, 284, 368, 412, 8),    # same phone again on P3: IoU ~0.89 > 0.5, suppressed
        (PHONE, 0.40, 40, 40, 104, 168, 16),     # a second, separate phone: kept
    ]
    model = fake(backend=fake_hailo.planted_backend(planted))
    (result,) = model.predict(np.zeros((640, 640, 3), np.uint8), conf=0.2)
    got = sorted((int(b.cls[0]), round(float(b.conf[0]), 2)) for b in result.boxes)
    assert got == [(PERSON, 0.95), (PHONE, 0.4), (PHONE, 0.6)]
    model.close()


def test_nms_boundary_iou_exactly_at_the_threshold_keeps_both():
    """IoU == 0.50 is kept (greedy `iou <= threshold` survives), > 0.50 is not."""
    # Two 96x96 boxes offset by dx: IoU = (96-dx)/(96+dx); dx=32 is exactly 0.5 in floats.
    a = np.array([[0.0, 0.0, 96.0, 96.0]])
    for dx, n in ((32.0, 2), (31.0, 1)):
        boxes = np.vstack([a, a + np.array([dx, 0.0, dx, 0.0])])
        assert len(accel._nms(boxes, np.array([0.9, 0.8]), 0.5)) == n, dx


# ------------------------------------------------------------------ confidence semantics
def test_conf_is_inclusive_and_the_classes_filter_is_applied(fake):
    planted = [(PHONE, 0.25, 300, 280, 364, 408, 16), (PERSON, 0.2499, 100, 120, 420, 600, 32)]
    model = fake(backend=fake_hailo.planted_backend(planted))
    frame = np.zeros((640, 640, 3), np.uint8)
    (r,) = model.predict(frame, conf=0.25)
    assert [int(b.cls[0]) for b in r.boxes] == [PHONE], "score == conf must be kept, below dropped"
    (r,) = model.predict(frame, conf=0.2, classes=[PERSON])
    assert [int(b.cls[0]) for b in r.boxes] == [PERSON]
    (r,) = model.predict(frame, conf=0.2, classes=[CUP])
    assert r.boxes == []
    model.close()


def test_the_padding_channel_is_never_a_class_even_when_it_is_the_largest(fake):
    heads = fake_hailo.planted_heads([(PHONE, 0.3, 300, 280, 364, 408, 16)])
    for name, (_g, _, c) in fake_hailo.OUTPUTS:
        if c == 81:
            heads[name][..., 80] = 0.99  # the .hef's padding channel, lit everywhere
    model = fake(backend=_backend(heads))
    (r,) = model.predict(np.zeros((640, 640, 3), np.uint8), conf=0.2)
    assert all(int(b.cls[0]) < 80 for b in r.boxes)
    # Legacy takes the argmax over the 80 REAL classes only: the phone still wins its anchor.
    assert [int(b.cls[0]) for b in r.boxes] == [PHONE]
    model.close()


# ------------------------------------------------------------------ parity with independent decoders
def _naive_legacy(heads: dict, conf: float, nms_iou: float = 0.5):
    """Per-anchor loops and an explicit IoU matrix: slow, obvious, independent."""
    dets = []
    for name, (g, _, c) in fake_hailo.OUTPUTS:
        if c != 68:
            continue
        stride = 640 // g
        cls_name = next(n for n, (gg, _, cc) in fake_hailo.OUTPUTS if gg == g and cc == 81)
        box, cls = heads[name][0], heads[cls_name][0]
        for row in range(g):
            for col in range(g):
                probs = cls[row, col, :80]
                k = int(np.argmax(probs))
                if probs[k] < conf:
                    continue
                d = []
                for side in range(4):
                    z = box[row, col, side * 17:(side + 1) * 17].astype(np.float64)
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    d.append(float((p * np.arange(17)).sum()))
                px, py = (col + 0.5) * stride, (row + 0.5) * stride
                dets.append((float(probs[k]), k, px - d[0] * stride, py - d[1] * stride,
                             px + d[2] * stride, py + d[3] * stride))
    dets.sort(key=lambda t: -t[0])
    kept = []
    for det in dets:
        ok = True
        for other in kept:
            if other[1] != det[1]:
                continue
            ix = max(0.0, min(det[4], other[4]) - max(det[2], other[2]))
            iy = max(0.0, min(det[5], other[5]) - max(det[3], other[3]))
            inter = ix * iy
            union = ((det[4] - det[2]) * (det[5] - det[3]) + (other[4] - other[2]) * (other[5] - other[3])
                     - inter)
            if union > 0 and inter / union > nms_iou:
                ok = False
                break
        if ok:
            kept.append(det)
    return _rows([k[2:] for k in kept], [k[0] for k in kept], [k[1] for k in kept])


def test_hwv_decode_agrees_with_the_naive_reference_on_a_sparse_scene():
    heads = _random_heads(11, hot=40.0)  # sparse: a handful of anchors over 0.2
    scales = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    d = hd.decode(heads, scales, hd.LEGACY, 0.2)
    naive = _naive_legacy(heads, 0.2)
    assert len(naive) > 3
    assert np.allclose(_rows(d.boxes, d.scores, d.classes), naive, atol=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("h", "w"), [(480, 640), (720, 1280), (640, 480), (150, 200)])
def test_predict_equals_the_independent_decoder_in_frame_pixels(fake, seed, h, w):
    heads = _random_heads(seed)
    model = fake(backend=_backend(heads))
    (result,) = model.predict(np.zeros((h, w, 3), np.uint8), conf=0.2)
    scale, left, top = accel.letterbox_transform(w, h, 640, 640)
    scales = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    ref = hd.unmap(hd.decode(heads, scales, hd.LEGACY, 0.2),
                   {"mode": "letterbox", "scale": scale, "left": left, "top": top, "frame_w": w, "frame_h": h})
    assert len(ref) > 5
    assert np.allclose(_shim_rows(result), _rows(ref.boxes, ref.scores, ref.classes), atol=1e-3)
    model.close()


def test_the_reference_decoder_equals_the_independent_reference_semantics(fake):
    """The opt-in candidate decoder, against hwv_decode's DAMO semantics with the
    product's IoU (0.5), 500 cap and inclusive threshold."""
    heads = _random_heads(5)
    model = fake(backend=_backend(heads), decoder="reference")
    (result,) = model.predict(np.zeros((640, 640, 3), np.uint8), conf=0.2)
    spec = hd.DecoderSpec("candidate", anchor_offset=0.0, multilabel=True, nms_iou=0.5, max_det=500)
    scales = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    ref = hd.unmap(hd.decode(heads, scales, spec, 0.2),
                   {"mode": "letterbox", "scale": 1.0, "left": 0, "top": 0, "frame_w": 640, "frame_h": 640})
    assert np.allclose(_shim_rows(result), _rows(ref.boxes, ref.scores, ref.classes), atol=1e-3)
    model.close()


def test_the_default_decoder_is_still_legacy_and_reference_stays_opt_in():
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert config["accelerator"]["yolo_decoder"] == "legacy"
    import inspect

    assert inspect.signature(HailoYOLO.__init__).parameters["decoder"].default == "legacy"


# ------------------------------------------------------------------ failure states: explicit, never silent-empty
def _drop(heads, name):
    return {k: v for k, v in heads.items() if k != name}


@pytest.mark.parametrize("decoder", ["legacy", "reference"])
@pytest.mark.parametrize("fault", ["missing", "nchw", "no_batch", "wrong_grid", "nan_cls", "inf_box", "nan_box"])
def test_a_malformed_device_result_raises_instead_of_returning_no_detections(fake, decoder, fault):
    heads = fake_hailo.planted_heads([(PHONE, 0.9, 300, 280, 364, 408, 16)])
    box16 = "damoyolo_tinynasL20_T/conv65"
    cls16 = "damoyolo_tinynasL20_T/conv66"
    row, col = round(344 / 16 - 0.5), round(332 / 16 - 0.5)  # the planted phone's anchor
    assert heads[cls16][0, row, col, PHONE] == pytest.approx(0.9)
    if fault == "missing":
        heads = _drop(heads, cls16)
    elif fault == "nchw":
        heads[box16] = np.ascontiguousarray(heads[box16].transpose(0, 3, 1, 2))
    elif fault == "no_batch":
        heads[cls16] = heads[cls16][0]
    elif fault == "wrong_grid":
        heads[cls16] = np.zeros((1, 20, 20, 81), np.float32)
    elif fault == "nan_cls":
        heads[cls16][0, 3, 3, 5] = np.nan
    elif fault == "inf_box":
        heads[box16][0, row, col, 0] = np.inf   # the planted phone's own anchor
    elif fault == "nan_box":
        heads[box16][0, row, col, 20] = np.nan
    model = fake(backend=_backend(heads), decoder=decoder)
    with pytest.raises(RuntimeError, match=r"Hailo output|non-finite"):
        model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2)
    model.close()


def test_all_zero_outputs_are_a_valid_empty_answer(fake):
    model = fake()
    (r,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2)
    assert r.boxes == []
    model.close()


@pytest.mark.parametrize("outputs", [
    # a class head too narrow for COCO: the class ids would silently be wrong
    tuple((n, (g, g, 79 if c == 81 else c)) for n, (g, _, c) in fake_hailo.OUTPUTS),
    # two box heads at one grid: one would silently overwrite the other
    (*fake_hailo.OUTPUTS, ("damoyolo_tinynasL20_T/conv99", (40, 40, 68))),
    # a grid that does not divide the input
    tuple((n, (21 if g == 20 else g, 21 if g == 20 else g, c)) for n, (g, _, c) in fake_hailo.OUTPUTS),
])
def test_a_hef_with_an_unexpected_head_layout_is_refused_at_load(fake, outputs):
    with pytest.raises(RuntimeError):
        fake(outputs=outputs)
    fake_hailo.state.vdevice_open = False


def test_a_non_uint8_frame_is_refused_not_reinterpreted(fake):
    model = fake()
    with pytest.raises(TypeError, match="uint8"):
        model.predict(np.zeros((480, 640, 3), np.float32), conf=0.2)
    model.close()


# ------------------------------------------------------------------ through the product's loader
def test_get_yolo_model_returns_the_verified_hailo_model_and_counts_inferences(tmp_path, monkeypatch):
    hef = tmp_path / "damoyolo_tinynasL20_T.hef"
    hef.write_bytes(b"not a real hef " * 100)
    sha = accel.file_sha256(str(hef))
    fake_hailo.install(fake_hailo.planted_backend([(PHONE, 0.9, 300, 280, 364, 408, 16)]))
    detectors.release_all()
    try:
        accel.configure({"yolo_hef": str(hef), "yolo_hef_sha256": sha, "yolo_decoder": "legacy"},
                        info=accel.AcceleratorInfo(True, True, "hailo", "fake HAT (simulated)"))
        before = accel.inference_stats()["count"]
        model = detectors.get_yolo_model("yolo11n.pt")
        assert detectors.require_backend("yolo11n.pt") == "hailo"
        (r,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2, classes=[PERSON, PHONE])
        assert [model.names[int(b.cls[0])] for b in r.boxes] == ["cell phone"]
        assert accel.inference_stats()["count"] == before + 1
        assert (detectors.model_identity("yolo11n.pt") or {}).get("sha256") == sha
        # A wrong pin is refused at load, with the reason surfaced.
        detectors.release_all()
        accel.configure({"yolo_hef": str(hef), "yolo_hef_sha256": "f" * 64},
                        info=accel.AcceleratorInfo(True, True, "hailo", "fake HAT (simulated)"))
        assert accel.maybe_hailo_yolo("yolo11n.pt") is None
        assert "integrity check FAILED" in (accel.failure_reason() or "")
    finally:
        detectors.release_all()
        accel.configure({}, info=accel.AcceleratorInfo(False, False, "none", "no HAT"))
        fake_hailo.uninstall()


def test_dfl_encoding_used_by_the_fixtures_is_exact():
    for d in (0.0, 0.25, 3.5, 7.999, 16.0):
        logits = fake_hailo.encode_distance(d)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        assert math.isclose(float((p * np.arange(17)).sum()), d, abs_tol=1e-5)
