"""M10.1 Priority 4: hardware accuracy-gate readiness (no hardware here).

* `hwv_stage.py parity --decoder legacy|reference|both`: the shipping .hef's INT8
  tensors vs the float model's expectations under EACH decoder (the reference
  decoder runs offline on the same device tensors; the production decoder is
  never switched).
* The reference-decoder float expectations (known_good/expected_float_reference.json)
  were made on the fleet by research/m101/known_good_decoders.py, in the same job
  that reproduced the committed legacy file exactly (job 20261114-142158-db510c,
  repeatability.json: every image's input tensor, head statistics and detections
  identical).
* Hand-class support in the diagnostic tools: hwv_decode's offline decoder reads
  a mapped extra class channel (e.g. 80:hand) exactly as core.accel does with
  accelerator.yolo_extra_classes, and stays bit-for-bit unchanged without one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
HWV = REPO / "scripts" / "hw_validation"
sys.path.insert(0, str(HWV))

import hwv_decode as hd  # noqa: E402
import hwv_stage as hs  # noqa: E402
from test_m10_hand_class_decode import (  # noqa: E402
    HAND_MAP,
    LEG_HAND,
    LEG_PERSON,
    LEG_PHONE,
    REF_HAND,
    REF_HAND_AT_PHONE,
    REF_PERSON,
    REF_PHONE,
    _decode,
    _device_heads,
)

PY = sys.executable
KG = HWV / "known_good"


def _run(args: list[str], cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603


# ------------------------------------------------------------------ expectations
def test_reference_expectations_match_the_legacy_file_image_for_image():
    legacy = json.loads((KG / "expected_float.json").read_text())
    reference = json.loads((KG / "expected_float_reference.json").read_text())
    assert legacy["decoder"] == "legacy" and reference["decoder"] == "reference"
    assert reference["conf"] == legacy["conf"] == 0.05
    assert reference["provenance"]["job"] == "20261114-142158-db510c"
    assert reference["provenance"]["onnx_sha256"] == "02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657"
    by_file = {im["file"]: im for im in legacy["images"]}
    assert [im["file"] for im in reference["images"]] == [im["file"] for im in legacy["images"]]
    for im in reference["images"]:
        # Same image, same input tensor, same heads: only the decoder differs.
        assert im["sha256"] == by_file[im["file"]]["sha256"]
        assert im["input_tensor_sha256"] == by_file[im["file"]]["input_tensor_sha256"]
        other = by_file[im["file"]]["heads"]
        assert im["heads"].keys() == other.keys()
        for name, entry in im["heads"].items():  # float sums: equal to ~1e-16
            for key in ("min", "max", "mean", "std"):
                assert entry[key] == pytest.approx(other[name][key], abs=1e-9), (name, key)
    grey = next(im for im in reference["images"] if im["file"] == "kg_flat_grey.png")
    assert grey["detections"] == []


def test_every_decoder_has_an_expectation_file():
    for decoder, path in hs.EXPECTED_FLOAT_BY_DECODER.items():
        assert path.exists(), path
        assert json.loads(path.read_text())["decoder"] == decoder


# ------------------------------------------------------------------ parity per decoder
@pytest.fixture(scope="module")
def fake_bundle(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("parity")
    bundle = tmp / "bundle"
    r = _run([PY, str(HWV / "hwv_stage.py"), "capture", "--fake", "--bundle", str(bundle), "--out", str(tmp)], tmp)
    assert (bundle / "manifest.json").exists(), r.stdout + r.stderr
    return bundle


@pytest.mark.parametrize("decoder", ["legacy", "reference", "both"])
def test_parity_stage_runs_per_decoder_and_never_claims_a_pass(fake_bundle, tmp_path, decoder):
    r = _run([PY, str(HWV / "hwv_stage.py"), "parity", "--fake", "--decoder", decoder, "--bundle", str(fake_bundle),
              "--out", str(tmp_path)], tmp_path)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out
    assert r.returncode == hs.SIMULATED, out  # a --fake run is never a PASS
    wanted = ("legacy", "reference") if decoder == "both" else (decoder,)
    for dec in ("legacy", "reference"):
        assert (f"[{dec}] every known-good image captured" in out) == (dec in wanted), out
    assert "production decoder == offline legacy decoder on the device tensors" in out
    report = json.loads(next(tmp_path.glob("parity_*.json")).read_text())
    assert set(report["values"]["decoders"]) == set(wanted)


def test_parity_blocks_without_the_decoders_expectations(fake_bundle, tmp_path, monkeypatch):
    monkeypatch.setitem(hs.EXPECTED_FLOAT_BY_DECODER, "reference", tmp_path / "absent.json")
    rep = hs.Report("parity", fake=True)
    args = type("A", (), {"decoder": "reference", "bundle": fake_bundle, "fake": True})()
    assert hs.stage_parity(rep, args) == hs.BLOCKED


# ------------------------------------------------------------------ hand class in the offline decoder
def _offline(spec: hd.DecoderSpec, extra=None, conf: float = 0.3, classes=None) -> list[tuple]:
    heads = _device_heads()
    outputs, shapes = {}, {}
    for stride, (box, cls) in heads.items():
        outputs[f"b{stride}"], outputs[f"c{stride}"] = box[None], cls[None]
        shapes[f"b{stride}"], shapes[f"c{stride}"] = box.shape, cls.shape
    scales = hd.group_heads(shapes, 640)
    dets = hd.decode(outputs, scales, spec, conf, extra_classes=extra, classes=classes)
    return sorted((int(c), round(float(s), 4), *(round(float(v), 3) for v in b))
                  for b, s, c in zip(dets.boxes, dets.scores, dets.classes, strict=True))


def test_offline_reference_reads_the_mapped_hand_like_the_device():
    want = sorted([REF_PERSON, REF_PHONE, REF_HAND_AT_PHONE, REF_HAND])
    assert _offline(hd.REFERENCE, HAND_MAP) == want
    assert _decode("reference", HAND_MAP) == want  # the device decoder, same heads


def test_offline_legacy_reads_the_mapped_hand_like_the_device():
    want = sorted([LEG_PERSON, LEG_PHONE, LEG_HAND])
    assert _offline(hd.LEGACY, HAND_MAP) == want
    assert _decode("legacy", HAND_MAP) == want


def test_without_a_map_the_hand_channel_is_still_dropped():
    assert _offline(hd.REFERENCE) == sorted([REF_PERSON, REF_PHONE])
    assert _offline(hd.LEGACY) == sorted([LEG_PERSON, LEG_PHONE])


def test_class_filter_uses_channel_ids():
    assert _offline(hd.REFERENCE, HAND_MAP, classes=[80]) == sorted([REF_HAND_AT_PHONE, REF_HAND])


def test_unchanged_bit_for_bit_without_a_map_on_random_heads():
    rng = np.random.default_rng(3)
    shapes = {f"b{s}": (640 // s, 640 // s, 68) for s in (8, 16, 32)}
    shapes |= {f"c{s}": (640 // s, 640 // s, 81) for s in (8, 16, 32)}
    outputs = {k: rng.random((1, *v)).astype(np.float32) * (0.3 if k.startswith("c") else 8) for k, v in shapes.items()}
    scales = hd.group_heads(shapes, 640)
    for spec in (hd.LEGACY, hd.REFERENCE):
        a = hd.decode(outputs, scales, spec, 0.25)
        b = hd.decode(outputs, scales, spec, 0.25, extra_classes=None)
        assert np.array_equal(a.boxes, b.boxes) and np.array_equal(a.scores, b.scores)
        assert np.array_equal(a.classes, b.classes) and int(a.classes.max(initial=0)) < 80


def test_a_head_too_narrow_for_the_map_is_refused():
    heads = _device_heads()
    outputs, shapes = {}, {}
    for stride, (box, cls) in heads.items():
        outputs[f"b{stride}"], outputs[f"c{stride}"] = box[None], cls[None]
        shapes[f"b{stride}"], shapes[f"c{stride}"] = box.shape, cls.shape
    scales = hd.group_heads(shapes, 640)
    with pytest.raises(ValueError, match="need class heads with 82 channels"):
        hd.decode(outputs, scales, hd.LEGACY, 0.3, extra_classes={81: "x"})


@pytest.mark.parametrize(("text", "expected"), [("80:hand", {80: "hand"}), (" 80:hand , 81:glove ", {80: "hand", 81: "glove"}),
                                                ("", {}), (None, {})])
def test_parse_extra_classes(text, expected):
    assert hd.parse_extra_classes(text) == expected


@pytest.mark.parametrize("bad", ["67:phone", "80", "x:hand", "80:", "80:hand,80:glove", "80:hand,81:hand"])
def test_parse_extra_classes_refuses(bad):
    with pytest.raises(ValueError, match="extra class"):
        hd.parse_extra_classes(bad)


def test_arch_refuses_a_malformed_second_class_map(tmp_path):
    second = tmp_path / "candidate.hef"
    second.write_bytes(b"not a real hef")
    r = _run([PY, str(HWV / "hwv_stage.py"), "arch", "--fake", "--second-hef", str(second),
              "--second-extra-classes", "67:phone", "--out", str(tmp_path / "out")], tmp_path)
    assert r.returncode == hs.BLOCKED, r.stdout + r.stderr
    assert "COCO channel" in r.stdout + r.stderr


# ------------------------------------------------------------------ the offline spec IS the device decoder
def _random_device_heads(seed: int):
    rng = np.random.default_rng(seed)
    heads = {}
    for stride in (8, 16, 32):
        g = 640 // stride
        box = rng.normal(0, 3, (g, g, 68)).astype(np.float32)
        cls = (rng.random((g, g, 81)) ** 6).astype(np.float32)  # mostly low, a few high scores
        heads[stride] = (box, cls)
    return heads


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("extra", [None, HAND_MAP])
def test_device_reference_spec_equals_the_products_reference_decoder(seed, extra):
    """Same tensors through core.accel's reference decoder and hwv_decode.DEVICE_REFERENCE:
    the same boxes, scores and classes (the parity stage's reference arm decodes with it)."""
    heads = _random_device_heads(seed)
    device = _decode("reference", extra, conf=0.35, heads=heads)
    outputs, shapes = {}, {}
    for stride, (box, cls) in heads.items():
        outputs[f"b{stride}"], outputs[f"c{stride}"] = box[None], cls[None]
        shapes[f"b{stride}"], shapes[f"c{stride}"] = box.shape, cls.shape
    dets = hd.decode(outputs, hd.group_heads(shapes, 640), hd.DEVICE_REFERENCE, 0.35, extra_classes=extra)
    # The device maps boxes back to the frame and clips them to it (here 640 x 640,
    # no letterbox); the raw offline decode clips later, in unmap().
    clipped = np.clip(dets.boxes, 0.0, 640.0)
    offline = sorted((int(c), round(float(s), 4), *(round(float(v), 3) for v in b))
                     for b, s, c in zip(clipped, dets.scores, dets.classes, strict=True))
    assert len(device) > 20
    assert len(offline) == len(device)
    for got, want in zip(offline, device, strict=True):  # float32 on the device, float64 offline
        assert got[:2] == want[:2], (got, want)
        assert np.allclose(got[2:], want[2:], atol=0.01), (got, want)


def test_the_upstream_reference_spec_differs_from_the_device_one():
    """Why the parity stage must not use hd.REFERENCE: strict threshold, NMS 0.7, 100 boxes."""
    assert (hd.REFERENCE.strict_threshold, hd.REFERENCE.nms_iou, hd.REFERENCE.max_det) == (True, 0.70, 100)
    assert (hd.DEVICE_REFERENCE.strict_threshold, hd.DEVICE_REFERENCE.nms_iou, hd.DEVICE_REFERENCE.max_det) == (
        False, 0.50, 500)


# ------------------------------------------------------------------ threshold calibration procedure
def test_a_threshold_derived_at_a_precision_reproduces_its_operating_point_on_the_device():
    """research/m8/analysis.r_at_p (the procedure behind every R@P0.80 threshold in
    M8-M10) returns the score of the LAST detection it keeps; operating_point keeps
    score >= threshold, and so do both device decoders (legacy and the product's
    reference, core.accel `>= conf`). So the derived threshold, set in config,
    reproduces the measured recall/precision rather than dropping the box that
    defined it. Hand-computable curve."""
    from research.detector_bench import oid_eval as oe
    from research.m8.analysis import r_at_p

    hits = [(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.5, False), (0.4, False)]
    result = oe.ClassResult(scored=[oe.Scored(f"i{k}", s, tp, k if tp else None) for k, (s, tp) in enumerate(hits)],
                            npos=4, images=6)
    op = r_at_p(result.scored, result.npos, 0.75)
    assert op == {"recall": 0.75, "threshold": 0.6}
    applied = oe.operating_point(result, op["threshold"])
    assert (applied["recall"], applied["precision"]) == (0.75, 0.75)
    # Both device decoders keep a box scored exactly at the threshold.
    assert not hd.LEGACY.strict_threshold and not hd.DEVICE_REFERENCE.strict_threshold
    heads = fake_heads_with_one_box(score=0.6)
    for spec in (hd.LEGACY, hd.DEVICE_REFERENCE):
        assert len(hd.decode(heads[0], heads[1], spec, 0.6)) == 1, spec.name
    assert len(hd.decode(heads[0], heads[1], hd.REFERENCE, 0.6)) == 0  # upstream DAMO keeps > only


def fake_heads_with_one_box(score: float):
    outputs, shapes = {}, {}
    for stride in (8, 16, 32):
        g = 640 // stride
        box = np.full((g, g, 68), -20.0, np.float32)
        cls = np.zeros((g, g, 81), np.float32)
        if stride == 16:
            for side in range(4):
                box[10, 10, side * 17 + 2] = 20.0
            cls[10, 10, 67] = np.float32(score)
        outputs[f"b{stride}"], outputs[f"c{stride}"] = box[None], cls[None]
        shapes[f"b{stride}"], shapes[f"c{stride}"] = box.shape, cls.shape
    # Thresholds are compared in float32 on the device as here.
    return outputs, hd.group_heads(shapes, 640)
