"""M10 Task 2: the parts of research/m10/hailo_compile that run without the Hailo SDK.

The compile itself has never run (no Dataflow Compiler on the fleet, M8.4 job 20261113-090503-dc1833).
What is pinned here: the calibration tensor is byte-identical to what HailoYOLO.predict hands the chip,
the calibration set refuses < 1,024 / duplicate / wrong-size frames, the model-script and parse-hef guards,
and that the production .hef can never be the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import fake_hailo  # noqa: E402

from core import accel  # noqa: E402
from core.accel import HailoYOLO  # noqa: E402
from research.m10.hailo_compile import calib as hc_calib  # noqa: E402
from research.m10.hailo_compile import compile as hc_compile  # noqa: E402


def test_the_calibration_tensor_is_the_tensor_predict_sends(monkeypatch):
    monkeypatch.setattr(accel, "record_inference", lambda: None)
    fake_hailo.install(fake_hailo.zero_backend, keep_inputs=True)
    try:
        model = HailoYOLO("fake.hef", sha256="0" * 64)
        rng = np.random.default_rng(3)
        for shape in ((480, 640, 3), (720, 1280, 3)):
            frame = rng.integers(0, 256, shape, dtype=np.uint8)
            model.predict(frame, conf=0.5)
            sent = fake_hailo.state.inputs_seen[-1][0]
            assert np.array_equal(hc_calib.device_tensor(frame), sent)
        model.close()
    finally:
        fake_hailo.uninstall()


def test_calibration_collect_refuses_duplicates_and_wrong_sizes(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    img = np.full((480, 640, 3), 50, np.uint8)
    cv2.imwrite(str(frames / "a.png"), img)
    tensors, rows = hc_calib.collect(frames, allow_any_size=False)
    assert len(tensors) == 1 and tensors[0].shape == (640, 640, 3) and rows[0]["size"] == [640, 480]
    assert (tensors[0][:80] == 114).all() and (tensors[0][-80:] == 114).all()   # letterbox bands
    cv2.imwrite(str(frames / "b.png"), img)
    with pytest.raises(SystemExit, match="duplicate frame"):
        hc_calib.collect(frames, allow_any_size=False)
    (frames / "b.png").unlink()
    cv2.imwrite(str(frames / "c.png"), np.zeros((720, 1280, 3), np.uint8))
    with pytest.raises(SystemExit, match="expected the device camera"):
        hc_calib.collect(frames, allow_any_size=False)
    assert len(hc_calib.collect(frames, allow_any_size=True)[0]) == 2


def test_calibration_set_needs_1024_frames(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    cv2.imwrite(str(frames / "a.png"), np.full((480, 640, 3), 7, np.uint8))
    with pytest.raises(SystemExit, match="at least 1024"):
        hc_calib.main(["--frames", str(frames), "--out", str(tmp_path / "o")])
    assert hc_calib.main(["--frames", str(frames), "--out", str(tmp_path / "o"), "--allow-small"]) == 0
    check = hc_compile.check_calib(tmp_path / "o")
    assert check["frames"] == 1
    assert any("release grade" in p for p in check["problems"]) and any("< 1024" in p for p in check["problems"])


def test_model_script_guards():
    assert hc_compile.check_script("normalization1 = normalization([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])\n") == []
    assert hc_compile.check_script("# nms_postprocess(...) commented out\n") == []
    (p,) = hc_compile.check_script("nms_postprocess('cfg.json', meta_arch=yolov8)\n")
    assert "raw heads" in p
    (p,) = hc_compile.check_script("normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])")
    assert "raw 0..255" in p


def test_the_production_hef_can_never_be_the_output(tmp_path):
    with pytest.raises(SystemExit, match=r"production \.hef.s name"):
        hc_compile.refuse_production_path(tmp_path / "damoyolo_tinynasL20_T.hef")
    with pytest.raises(SystemExit, match="models/ directory"):
        hc_compile.refuse_production_path(hc_compile.ROOT / "models" / "candidate.hef")
    hc_compile.refuse_production_path(tmp_path / "exp08_heads81_hailo8l.hef")  # an ordinary path is fine


GOOD_PARSE_HEF = """Architecture HEF was compiled for: HAILO8L
Network group name: sensordyme_exp08_heads81
        VStream infos:
            Input  sensordyme_exp08_heads81/input_layer1 UINT8, NHWC(640x640x3)
            Output sensordyme_exp08_heads81/conv53 UINT8, FCR(80x80x68)
            Output sensordyme_exp08_heads81/conv54 UINT8, NHWC(80x80x81)
            Output sensordyme_exp08_heads81/conv65 UINT8, FCR(40x40x68)
            Output sensordyme_exp08_heads81/conv66 UINT8, NHWC(40x40x81)
            Output sensordyme_exp08_heads81/conv76 UINT8, FCR(20x20x68)
            Output sensordyme_exp08_heads81/conv77 UINT8, NHWC(20x20x81)
"""


def test_parse_hef_text_contract():
    """Synthetic text in the shape of parse-hef output (the real format is version-dependent)."""
    assert hc_compile.parse_hef_text(GOOD_PARSE_HEF)["ok"]
    nms = GOOD_PARSE_HEF + "            Output sensordyme/nms HAILO NMS(number of classes: 80)\n"
    assert not hc_compile.parse_hef_text(nms)["ok"]
    narrow = GOOD_PARSE_HEF.replace("80x80x81", "80x80x80")
    assert any("six heads" in p for p in hc_compile.parse_hef_text(narrow)["problems"])
