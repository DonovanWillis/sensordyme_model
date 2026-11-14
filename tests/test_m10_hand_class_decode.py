"""M10: opt-in extra-class decoding (`accelerator.yolo_extra_classes`), e.g. {80: hand}.

Both HailoYOLO decoders read the 80 COCO class channels and drop channel 80, so a
3-class device-layout export (research/m84/device_export.py: person -> 0,
phone -> 67, hand -> 80 of an 81-channel class head) reports person and phone
but never hand. These tests pin, on deterministic synthetic head tensors in the
device layout (strides 8/16/32 -> grids 80/40/20, 68-channel DFL box heads,
81-channel class heads), that:

  * with the map, both decoders report channel 80 as "hand" with exact scores
    and boxes after decode + class-wise NMS (expected values by hand arithmetic,
    never from core.accel);
  * without it (the default), channel 80 is still dropped, bit for bit;
  * a .hef that cannot carry the mapped channel, an on-chip-NMS build, or the
    COCO-80 shipping artifact is refused at load, before the chip is opened;
  * config validation rejects malformed maps.

No model, no hardware. Nothing here wires hand detections into any product rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import fake_hailo  # noqa: E402

from core import accel, detectors  # noqa: E402
from core.accel import COCO80_ONLY_HEF_SHA256, HailoYOLO  # noqa: E402
from core.config_schema import parse_yolo_extra_classes, validate  # noqa: E402

BINS = 17
PERSON, PHONE, HAND = 0, 67, 80
HAND_MAP = {80: "hand"}
PRODUCTION_SHA = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"

# (class channel, score, stride, (gy, gx), DFL bin per side l,t,r,b). Integer bins
# make the softmax expectation exactly the bin index, so boxes are hand-computable.
PLANTS = (
    (PERSON, 0.91, 32, (8, 10), (3, 4, 3, 5)),
    (PHONE, 0.77, 16, (20, 15), (1, 2, 1, 2)),
    (HAND, 0.55, 16, (20, 15), (1, 2, 1, 2)),   # a hand sharing the phone's anchor, weaker
    (HAND, 0.66, 8, (50, 30), (2, 2, 3, 3)),
    (HAND, 0.50, 8, (50, 31), (3, 2, 2, 3)),    # the same box from the next cell: NMS removes it
)


def _box(stride, cell, dists, offset):
    """Hand arithmetic of DFL distance decoding: anchor (cell + offset) * stride, sides * stride."""
    gy, gx = cell
    px, py = (gx + offset) * stride, (gy + offset) * stride
    left, top, right, bottom = (d * stride for d in dists)
    return (px - left, py - top, px + right, py + bottom)


# Reference: anchor offset 0, every (anchor, class) >= conf is a candidate.
REF_PERSON = (PERSON, 0.91, 224.0, 128.0, 416.0, 416.0)
REF_PHONE = (PHONE, 0.77, 224.0, 288.0, 256.0, 352.0)
REF_HAND_AT_PHONE = (HAND, 0.55, 224.0, 288.0, 256.0, 352.0)
REF_HAND = (HAND, 0.66, 224.0, 384.0, 264.0, 424.0)
# Legacy: anchor offset +0.5 (boxes +stride/2), argmax class per anchor only.
LEG_PERSON = (PERSON, 0.91, 240.0, 144.0, 432.0, 432.0)
LEG_PHONE = (PHONE, 0.77, 232.0, 296.0, 264.0, 360.0)
LEG_HAND = (HAND, 0.66, 228.0, 388.0, 268.0, 428.0)


def test_the_hand_arithmetic_matches_the_planted_geometry():
    assert _box(32, (8, 10), (3, 4, 3, 5), 0.0) == REF_PERSON[2:]
    assert _box(16, (20, 15), (1, 2, 1, 2), 0.0) == REF_PHONE[2:]
    assert _box(8, (50, 30), (2, 2, 3, 3), 0.0) == REF_HAND[2:]
    assert _box(8, (50, 31), (3, 2, 2, 3), 0.0) == REF_HAND[2:]  # the duplicate is the identical box
    assert _box(32, (8, 10), (3, 4, 3, 5), 0.5) == LEG_PERSON[2:]
    assert _box(16, (20, 15), (1, 2, 1, 2), 0.5) == LEG_PHONE[2:]
    assert _box(8, (50, 30), (2, 2, 3, 3), 0.5) == LEG_HAND[2:]


# ------------------------------------------------------------------ fixtures
def _device_heads(plants=PLANTS, class_channels: int = 81) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """stride -> (box (G, G, 68) DFL logits, class (G, G, C) probabilities), device layout."""
    heads = {}
    for stride in (8, 16, 32):
        g = 640 // stride
        heads[stride] = (np.full((g, g, 4 * BINS), -20.0, np.float32), np.zeros((g, g, class_channels), np.float32))
    for channel, score, stride, (gy, gx), dists in plants:
        box, cls = heads[stride]
        box[gy, gx, :] = -20.0
        for side, d in enumerate(dists):
            box[gy, gx, side * BINS + d] = 20.0
        cls[gy, gx, channel] = score
    return heads


def _decoder(heads, decoder: str, extra_classes=None) -> tuple[HailoYOLO, dict]:
    """A HailoYOLO with no device, holding exactly the state the decoders read."""
    device = HailoYOLO.__new__(HailoYOLO)
    device.decoder, device._nms_iou = decoder, 0.5
    device._set_extra_classes(extra_classes, sha256="0" * 64)
    device._scales, raw = [], {}
    for stride, (box, cls) in heads.items():
        raw[f"b{stride}"], raw[f"c{stride}"] = box[None], cls[None]
        device._scales.append({"box": f"b{stride}", "cls": f"c{stride}", "grid": box.shape[0], "stride": stride,
                               "num_classes": cls.shape[-1]})
    return device, raw


def _decode(decoder, extra_classes=None, conf=0.3, classes=None, heads=None):
    device, raw = _decoder(heads or _device_heads(), decoder, extra_classes)
    (result,) = device._decode_raw(raw, 1.0, 0, 0, 640, 640, classes, conf)
    return sorted((int(b.cls[0]), round(float(b.conf[0]), 4), *(round(float(v), 3) for v in b.xyxy[0]))
                  for b in result.boxes)


# ------------------------------------------------------------------ decode with the map
def test_reference_decoder_reports_the_hand_channel_when_mapped():
    assert _decode("reference", HAND_MAP) == sorted([REF_PERSON, REF_PHONE, REF_HAND_AT_PHONE, REF_HAND])


def test_legacy_decoder_reports_the_hand_channel_when_mapped():
    # argmax per anchor: the weaker hand under the phone is not a candidate on the legacy path
    assert _decode("legacy", HAND_MAP) == sorted([LEG_PERSON, LEG_PHONE, LEG_HAND])


def test_legacy_argmax_includes_the_mapped_channel():
    """A hand stronger than a COCO class at the same anchor wins the legacy argmax."""
    plants = ((PHONE, 0.40, 16, (20, 15), (1, 2, 1, 2)), (HAND, 0.80, 16, (20, 15), (1, 2, 1, 2)))
    heads = _device_heads(plants)
    assert _decode("legacy", HAND_MAP, heads=heads) == [(HAND, 0.8, 232.0, 296.0, 264.0, 360.0)]
    assert _decode("legacy", None, heads=heads) == [(PHONE, 0.4, 232.0, 296.0, 264.0, 360.0)]


@pytest.mark.parametrize("decoder", ["legacy", "reference"])
def test_classes_filter_selects_the_mapped_channel(decoder):
    got = _decode(decoder, HAND_MAP, classes=[HAND])
    assert {d[0] for d in got} == {HAND}
    assert (REF_HAND if decoder == "reference" else LEG_HAND) in got
    assert {d[0] for d in _decode(decoder, HAND_MAP, classes=[PERSON, PHONE])} == {PERSON, PHONE}


@pytest.mark.parametrize("decoder", ["legacy", "reference"])
def test_the_conf_gate_applies_to_the_mapped_channel(decoder):
    assert HAND not in {d[0] for d in _decode(decoder, HAND_MAP, conf=0.7)}


# ------------------------------------------------------------------ default unchanged
@pytest.mark.parametrize("extra", [None, {}])
def test_the_default_still_drops_channel_80(extra):
    assert _decode("reference", extra) == sorted([REF_PERSON, REF_PHONE])
    assert _decode("legacy", extra) == sorted([LEG_PERSON, LEG_PHONE])


@pytest.mark.parametrize("decoder", ["legacy", "reference"])
def test_the_default_is_bit_for_bit_the_pre_m10_decode(decoder):
    """On random heads (lots of candidates, channel 80 hot), no map == the original 80-column slice."""
    rng = np.random.default_rng(10)
    heads = {}
    for stride in (8, 16, 32):
        g = 640 // stride
        cls = (rng.random((g, g, 81)) ** 30).astype(np.float32)
        cls[..., 80] = 0.99  # channel 80 everywhere: it must never appear
        heads[stride] = (rng.normal(0, 2, (g, g, 68)).astype(np.float32), cls)
    device, raw = _decoder(heads, decoder, None)
    (got,) = device._decode_raw(raw, 0.8, 0, 64, 800, 512, None, 0.5)
    sliced = {s: (b, c[..., :80].copy()) for s, (b, c) in heads.items()}
    pre, raw80 = _decoder(sliced, decoder, None)
    (want,) = pre._decode_raw(raw80, 0.8, 0, 64, 800, 512, None, 0.5)
    assert len(got.boxes) == len(want.boxes) > 5
    for a, b in zip(got.boxes, want.boxes, strict=True):
        assert int(a.cls[0]) == int(b.cls[0]) != HAND
        assert a.conf.tobytes() == b.conf.tobytes() and a.xyxy.tobytes() == b.xyxy.tobytes()


# ------------------------------------------------------------------ end to end on the fake chip
@pytest.fixture
def fake_chip(tmp_path, monkeypatch):
    """The fake hailo_platform (device layout) + a .hef file on disk, configured through accel."""
    monkeypatch.setattr(accel, "record_inference", lambda: None)
    hef = tmp_path / "exp08_heads81.hef"
    hef.write_bytes(b"not a real hef: the fake reads only shapes " * 100)
    sha = accel.file_sha256(str(hef))
    detectors.release_all()

    def load(backend, decoder="reference", extra=HAND_MAP, outputs=None):
        detectors.release_all()
        fake_hailo.install(backend, outputs=outputs)
        cfg = {"yolo_hef": str(hef), "yolo_hef_sha256": sha, "yolo_decoder": decoder}
        if extra is not None:
            cfg["yolo_extra_classes"] = extra
        accel.configure(cfg, info=accel.AcceleratorInfo(True, True, "hailo", "fake HAT"))
        return accel.maybe_hailo_yolo("yolo11n.pt")

    yield load
    detectors.release_all()
    fake_hailo.uninstall()


def _hef_backend(heads):
    """Planted (G, G, C) heads -> the fake .hef's six named NHWC outputs."""
    by_key = {(b.shape[0], "box"): b for b, _ in heads.values()} | {(c.shape[0], "cls"): c for _, c in heads.values()}

    def run(batch):
        fake_hailo.zero_backend(batch)
        return {name: by_key[(g, "box" if c == 68 else "cls")][None].copy() for name, (g, _, c) in fake_hailo.OUTPUTS}
    return run


@pytest.mark.parametrize(("decoder", "hand"), [("reference", REF_HAND), ("legacy", LEG_HAND)])
def test_a_640x480_frame_reports_the_hand_in_frame_pixels(fake_chip, decoder, hand):
    model = fake_chip(_hef_backend(_device_heads()), decoder=decoder)
    assert model is not None and accel.failure_reason() is None
    assert model.names[HAND] == "hand" and model.names[PHONE] == "cell phone" and len(model.names) == 81
    (res,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.3, classes=[HAND])
    got = [(int(b.cls[0]), round(float(b.conf[0]), 4), *(round(float(v), 3) for v in b.xyxy[0])) for b in res.boxes]
    # letterbox 640x480 -> 640x640: scale 1, 80 px of padding on top; boxes come back minus 80 in y
    want = (HAND, hand[1], hand[2], hand[3] - 80, hand[4], hand[5] - 80)
    assert want in got and {g[0] for g in got} == {HAND}


def test_without_the_map_the_same_chip_never_reports_a_hand(fake_chip):
    model = fake_chip(_hef_backend(_device_heads()), extra=None)
    assert model is not None and HAND not in model.names
    (res,) = model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.3)
    assert {int(b.cls[0]) for b in res.boxes} == {PERSON, PHONE}


# ------------------------------------------------------------------ fail closed at load
def test_a_class_head_too_narrow_for_the_map_is_refused(fake_chip):
    narrow = tuple((n, (g, w, 80 if c == 81 else c)) for n, (g, w, c) in fake_hailo.OUTPUTS)
    heads = _device_heads(PLANTS[:2], class_channels=80)
    assert fake_chip(_hef_backend(heads), outputs=narrow) is None
    reason = accel.failure_reason() or ""
    assert "yolo_extra_classes" in reason and "at least 81" in reason
    assert not fake_hailo.state.vdevice_open, "refused before the exclusive chip was opened"


def test_a_map_beyond_the_head_width_is_refused(fake_chip):
    assert fake_chip(_hef_backend(_device_heads()), extra={81: "hand"}) is None
    assert "at least 82" in (accel.failure_reason() or "")
    assert not fake_hailo.state.vdevice_open


def test_an_on_chip_nms_build_is_refused(fake_chip):
    assert fake_chip(fake_hailo.zero_backend, outputs=(("nms", (80, 5, 100)),)) is None
    assert "on-chip NMS" in (accel.failure_reason() or "")


def test_the_shipping_coco80_hef_is_refused_with_a_map():
    assert PRODUCTION_SHA in COCO80_ONLY_HEF_SHA256
    with pytest.raises(ValueError, match="background slot"):
        HailoYOLO("damoyolo_tinynasL20_T.hef", sha256=PRODUCTION_SHA, extra_classes=HAND_MAP)


def test_the_shipping_hef_without_a_map_is_unaffected(fake_chip):
    """The refusal is keyed on the map: no map, the decoder is constructed as before."""
    fake_hailo.install(fake_hailo.zero_backend)
    model = HailoYOLO("damoyolo_tinynasL20_T.hef", sha256=PRODUCTION_SHA)
    try:
        assert model._extra_classes == {} and len(model.names) == 80
    finally:
        model.close()


# ------------------------------------------------------------------ config validation
@pytest.mark.parametrize(("value", "want"), [
    (None, {}), ({}, {}), ({80: "hand"}, {80: "hand"}), ({"80": " hand "}, {80: "hand"}),
    ({81: "glove", 80: "hand"}, {80: "hand", 81: "glove"}),
])
def test_valid_maps_parse(value, want):
    assert parse_yolo_extra_classes(value) == want
    errors, _ = validate({"module": "x", "accelerator": {"yolo_extra_classes": value}})
    assert not [e for e in errors if "yolo_extra_classes" in e]


@pytest.mark.parametrize(("value", "fragment"), [
    ([80], "must be a mapping"),
    ({0: "hand"}, "is a COCO class"),
    ({67: "phone_in_hand"}, "is a COCO class"),
    ({"x": "hand"}, "not an integer"),
    ({80.0: "hand"}, "not an integer"),
    ({True: "hand"}, "not an integer"),
    ({80: ""}, "non-empty class name"),
    ({80: 3}, "non-empty class name"),
    ({80: "person"}, "collides with a COCO class"),
    ({80: "hand", 81: "hand"}, "more than one channel"),
    ({80: "hand", "80": "palm"}, "mapped twice"),
])
def test_malformed_maps_are_config_errors(value, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_yolo_extra_classes(value)
    errors, _ = validate({"module": "x", "accelerator": {"yolo_extra_classes": value}})
    assert any("yolo_extra_classes" in e and fragment in e for e in errors)


def test_the_shipped_config_has_no_map():
    import yaml

    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    assert "yolo_extra_classes" not in cfg["accelerator"]
    assert cfg["accelerator"]["yolo_decoder"] == "legacy"
    assert cfg["accelerator"]["yolo_hef_sha256"] == PRODUCTION_SHA
