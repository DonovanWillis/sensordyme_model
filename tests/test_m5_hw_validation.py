"""Pure logic of the hardware-validation package (scripts/hw_validation/).

Nothing here touches a Pi, a Hailo chip, a camera or a GPIO. What IS pinned:

  * the offline "legacy" decoder is the production decoder (same boxes as
    core.accel.HailoYOLO._decode_raw on the same tensors), so a Pi bundle
    decoded on a worker describes what the device does;
  * the capture script's letterbox is byte-identical to what predict() feeds
    the Hailo pipeline;
  * the reference decoder's three differences (offset 0, multilabel, NMS/cap)
    behave as specified, and the agreement statistics expose a pure anchor
    offset as exactly +stride/2;
  * the safety gates refuse (service active/unknown, pin mismatch/missing,
    output inside the installed tree);
  * the analysis helpers (parity, ONNX layout, stage residuals, health rows,
    latency, buzzer timelines, intrinsics/FOV, file manifests) compute what
    their docstrings say.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

HWV = Path(__file__).resolve().parent.parent / "scripts" / "hw_validation"
sys.path.insert(0, str(HWV))

import buzzer_check  # noqa: E402
import calibrate_intrinsics as ci  # noqa: E402
import compare_float_vs_int8 as cmp  # noqa: E402
import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402
import integrity_manifest as im  # noqa: E402
import stage_timing as st  # noqa: E402

from core import accel  # noqa: E402
from core.accel import COCO_NAMES, HailoYOLO  # noqa: E402

SHAPES = {"m/conv53": (80, 80, 68), "m/conv54": (80, 80, 81), "m/conv65": (40, 40, 68),
          "m/conv66": (40, 40, 81), "m/conv76": (20, 20, 68), "m/conv77": (20, 20, 81)}


class _Info:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


def _production_model() -> HailoYOLO:
    model = HailoYOLO.__new__(HailoYOLO)
    model._input_h = model._input_w = 640
    model._nms_iou = 0.5
    model.names = dict(COCO_NAMES)
    model._raw_outputs = True
    model._scales = model._group_raw_outputs([_Info(n, s) for n, s in SHAPES.items()])
    return model


def _random_outputs(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    outs = {}
    for name, (g, _, c) in SHAPES.items():
        if c == 68:
            outs[name] = rng.normal(0, 2, (1, g, g, c)).astype(np.float32)
        else:  # mostly small scores, a few dozen above 0.25
            outs[name] = (rng.random((1, g, g, c)) ** 12).astype(np.float32)
    return outs


def _rows(dets: hd.Detections) -> np.ndarray:
    rows = np.column_stack([dets.scores, dets.classes, dets.boxes])
    return rows[np.lexsort((rows[:, 2], -rows[:, 0]))]


# ------------------------------------------------------------------ decoder parity
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_offline_legacy_decoder_is_the_production_decoder(seed):
    outs = _random_outputs(seed)
    geometry = {"mode": "letterbox", "scale": 1.0, "left": 0, "top": 80,
                "frame_w": 640, "frame_h": 480}
    prod = _production_model()._decode_raw(outs, 1.0, 0, 80, 640, 480, None, 0.25)[0].boxes
    scales = hd.group_heads(SHAPES, 640)
    mine = hd.unmap(hd.decode(outs, scales, hd.LEGACY, 0.25), geometry)

    assert len(prod) == len(mine) > 5
    prod_rows = np.array([[float(b.conf[0]), float(b.cls[0]), *map(float, b.xyxy[0])] for b in prod])
    prod_rows = prod_rows[np.lexsort((prod_rows[:, 2], -prod_rows[:, 0]))]
    np.testing.assert_allclose(_rows(mine), prod_rows, atol=1e-3)


def test_nms_copy_matches_production_nms():
    rng = np.random.default_rng(3)
    xy = rng.random((60, 2)) * 300
    boxes = np.hstack([xy, xy + rng.random((60, 2)) * 80 + 5]).astype(np.float32)
    scores = rng.random(60).astype(np.float32)
    for thr in (0.3, 0.5, 0.7):
        assert hd.nms(boxes, scores, thr) == accel._nms(boxes, scores, thr)


def test_head_grouping_matches_production():
    prod = _production_model()._scales
    mine = hd.group_heads(SHAPES, 640)
    assert [(s["grid"], s["stride"], s["box"], s["cls"]) for s in prod] == \
        [(s.grid, s.stride, s.box, s.cls) for s in mine]


def _planted(cell=(10, 12), dist_bin=3, scores=None):
    """Stride-32 grid only, one anchor lit; other grids silent."""
    outs = {n: np.zeros((1, *s), np.float32) for n, s in SHAPES.items()}
    row, col = cell
    for side in range(4):
        outs["m/conv76"][0, row, col, side * 17 + dist_bin] = 50.0
    for cls_id, score in (scores or {0: 0.9}).items():
        outs["m/conv77"][0, row, col, cls_id] = score
    return outs


def test_reference_anchor_is_cell_corner_and_legacy_is_cell_centre():
    scales = hd.group_heads(SHAPES, 640)
    outs = _planted(cell=(10, 12), dist_bin=3)
    leg = hd.decode(outs, scales, hd.LEGACY, 0.25)
    ref = hd.decode(outs, scales, hd.REFERENCE, 0.25)
    d = 3 * 32
    np.testing.assert_allclose(ref.boxes[0], [12 * 32 - d, 10 * 32 - d, 12 * 32 + d, 10 * 32 + d],
                               atol=1e-3)
    np.testing.assert_allclose(leg.boxes[0] - ref.boxes[0], [16, 16, 16, 16], atol=1e-3)

    agreement = hd.Agreement(iou_threshold=0.5)
    agreement.add(leg, ref)
    shift = agreement.summary()["centre_shift_a_minus_b_by_stride"]["32"]
    assert shift["dx_mean"] == pytest.approx(16.0, abs=1e-3)
    assert shift["dy_mean"] == pytest.approx(16.0, abs=1e-3)


def test_reference_is_multilabel_and_legacy_keeps_only_the_argmax():
    scales = hd.group_heads(SHAPES, 640)
    outs = _planted(scores={0: 0.9, 67: 0.6})
    assert sorted(hd.decode(outs, scales, hd.REFERENCE, 0.25).classes.tolist()) == [0, 67]
    assert hd.decode(outs, scales, hd.LEGACY, 0.25).classes.tolist() == [0]


def test_threshold_semantics_and_padding_channel_is_ignored():
    scales = hd.group_heads(SHAPES, 640)
    at_threshold = _planted(scores={5: 0.25})
    assert len(hd.decode(at_threshold, scales, hd.LEGACY, 0.25)) == 1, "production keeps >="
    assert len(hd.decode(at_threshold, scales, hd.REFERENCE, 0.25)) == 0, "DAMO keeps >"
    padding = _planted(scores={80: 0.99})
    assert len(hd.decode(padding, scales, hd.REFERENCE, 0.25)) == 0


def test_max_det_cap_and_overrides():
    spec = hd.with_overrides(hd.REFERENCE, nms_iou=0.6, max_det=2)
    assert (spec.nms_iou, spec.max_det) == (0.6, 2)
    outs = _random_outputs(5)
    dets = hd.decode(outs, hd.group_heads(SHAPES, 640), spec, 0.25)
    assert len(dets) == 2 and dets.scores[0] >= dets.scores[1]


def test_stretch_unmap_returns_frame_pixels():
    dets = hd.Detections(np.array([[0.0, 0.0, 640.0, 640.0]]), np.array([0.9]),
                         np.array([0]), np.array([8]))
    geom = {"mode": "stretch", "input_w": 640, "input_h": 640, "frame_w": 640, "frame_h": 480}
    np.testing.assert_allclose(hd.unmap(dets, geom).boxes[0], [0, 0, 640, 480])


# ------------------------------------------------------------------ preprocessing parity
@pytest.mark.parametrize(("h", "w"), [(480, 640), (333, 500)])
def test_capture_letterbox_is_exactly_what_predict_feeds_the_chip(monkeypatch, h, w):
    captured = {}

    class _Pipeline:
        def infer(self, feed):
            captured.update(feed)
            return {}

    model = _production_model()
    model._scales = []
    model._input_name = "in"
    model._lock = threading.Lock()
    model._ensure_ready = _Pipeline
    monkeypatch.setattr(accel, "record_inference", lambda: None)
    frame = np.random.default_rng(7).integers(0, 256, (h, w, 3), dtype=np.uint8)

    model.predict(frame, conf=0.25)
    rgb, geometry = hc.letterbox_bgr_to_rgb(frame, 640, 640)

    assert captured["in"].shape == (1, 640, 640, 3)
    np.testing.assert_array_equal(captured["in"][0], rgb)
    assert geometry["mode"] == "letterbox" and geometry["frame_w"] == w


def test_stretch_is_plain_resize_to_rgb():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0] = 255  # blue in BGR
    rgb, geometry = hc.stretch_bgr_to_rgb(frame, 640, 640)
    assert rgb.shape == (640, 640, 3) and rgb[0, 0, 2] == 255 and rgb[0, 0, 0] == 0
    assert geometry == {"mode": "stretch", "frame_w": 640, "frame_h": 480, "input_w": 640,
                        "input_h": 640, "interpolation": "INTER_LINEAR"}


# ------------------------------------------------------------------ safety gates
@pytest.mark.parametrize("state", ["active", "activating", "reloading", "deactivating"])
def test_service_gate_refuses_while_the_service_may_hold_the_device(state):
    ok, why = hc.service_gate(state)
    assert not ok and state in why


def test_service_gate_allows_stopped_and_fails_closed_on_unknown():
    assert hc.service_gate("inactive")[0]
    assert hc.service_gate("failed")[0]
    assert not hc.service_gate(None)[0]
    assert hc.service_gate(None, allow_unknown=True)[0]


def test_operator_commands_are_printed_reversibly_and_never_run():
    text = hc.operator_commands()
    assert "sudo systemctl stop sensordyme.service" in text
    assert "sudo systemctl start sensordyme.service" in text


def test_pin_check():
    assert hc.verify_pin("ab" * 32, "AB" * 32)[0]
    assert not hc.verify_pin("ab" * 32, "cd" * 32)[0]
    assert not hc.verify_pin("ab" * 32, None)[0]
    assert not hc.verify_pin("ab" * 32, "  ")[0]


def test_output_must_be_outside_the_installed_tree(tmp_path):
    install = tmp_path / "sensordyme"
    install.mkdir()
    assert not hc.ensure_outside(install, install)[0]
    assert not hc.ensure_outside(install / "data" / "x", install)[0]
    assert hc.ensure_outside(tmp_path / "hwval" / "x", install)[0]


def test_bundle_round_trip(tmp_path):
    inp = np.zeros((1, 640, 640, 3), np.uint8)
    outs = [np.full((1, 20, 20, 68), 1.5, np.float32), np.full((1, 20, 20, 81), 0.5, np.float32)]
    hc.save_sample(tmp_path / "s.npz", inp, outs)
    sample = {"file": "s.npz", "outputs": [{"name": "a/box"}, {"name": "a/cls"}]}
    got_in, got_out = hc.load_sample(tmp_path, sample)
    assert got_in.shape == inp.shape
    assert set(got_out) == {"a/box", "a/cls"} and got_out["a/cls"].shape == (1, 20, 20, 81)
    hc.write_json(tmp_path / "manifest.json", {"format": "other"})
    with pytest.raises(ValueError, match="not a"):
        hc.load_bundle(tmp_path)


# ------------------------------------------------------------------ parity statistics
def test_parity_accumulator_equals_one_shot_statistics():
    rng = np.random.default_rng(11)
    a = [rng.random((1, 4, 4, 81)) for _ in range(3)]
    b = [x + rng.normal(0, 0.01, x.shape) for x in a]
    acc = hc.ParityAccumulator()
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        acc.add(x, y, f"s{i}")
    one = hc.tensor_parity(np.concatenate([x.ravel() for x in a]), np.concatenate([y.ravel() for y in b]))
    got = acc.summary()
    for key in ("mean_abs_err", "max_abs_err", "rmse", "cosine", "pearson"):
        assert got[key] == pytest.approx(one[key], rel=1e-9)
    chans = acc.per_channel(qp_scale=0.004)
    assert len(chans) == 81 and chans[0]["mean_abs_err_lsb"] == pytest.approx(chans[0]["mean_abs_err"] / 0.004)


def test_parity_compares_common_channels_when_counts_differ():
    acc = hc.ParityAccumulator()
    acc.add(np.ones((1, 2, 2, 81)), np.ones((1, 2, 2, 80)))
    assert acc.summary()["max_abs_err"] == 0.0 and len(acc.per_channel()) == 80


def test_range_verdicts():
    assert hc.range_verdict(0.0, 1.0, "cls").startswith("ok")
    assert hc.range_verdict(-0.2, 1.0, "cls").startswith("FAIL")
    assert hc.range_verdict(-3.3, 9.6, "box").startswith("ok")
    assert hc.range_verdict(0.0, 0.9, "box").startswith("SUSPECT")


@pytest.mark.parametrize("shape", [(1, 68, 20, 20), (1, 20, 20, 68), (1, 68, 400), (1, 400, 68)])
def test_onnx_head_layouts_become_nhwc(shape):
    assert cmp.to_nhwc(np.zeros(shape)).shape == (1, 20, 20, 68)


def test_onnx_nchw_transpose_keeps_values_in_place():
    nchw = np.arange(2 * 3 * 3).reshape(1, 2, 3, 3)
    nhwc = cmp.to_nhwc(nchw)
    assert nhwc[0, 1, 2, 1] == nchw[0, 1, 1, 2]


def test_head_mapping_pairs_by_grid_and_role():
    scales = hd.group_heads(SHAPES, 640)
    onnx = {"Mul_224": (1, 80, 80, 68), "Sigmoid_244": (1, 80, 80, 80), "Mul_264": (1, 40, 40, 68),
            "Sigmoid_284": (1, 40, 40, 80), "Mul_304": (1, 20, 20, 68), "Sigmoid_324": (1, 20, 20, 80)}
    rows = cmp.map_heads(scales, onnx)
    assert {(r["hef"], r["onnx_node"]) for r in rows} == {
        ("m/conv53", "Mul_224"), ("m/conv54", "Sigmoid_244"), ("m/conv65", "Mul_264"),
        ("m/conv66", "Sigmoid_284"), ("m/conv76", "Mul_304"), ("m/conv77", "Sigmoid_324")}
    del onnx["Mul_304"]
    with pytest.raises(ValueError, match="no ONNX head"):
        cmp.map_heads(scales, onnx)


def test_onnx_input_is_nchw_float_with_normalisation():
    batch = np.full((1, 4, 4, 3), 100, np.uint8)
    batch[..., 0] = 200
    x = cmp.normalise_input(batch, mean=[0, 0, 0], std=[1, 1, 1])
    assert x.shape == (1, 3, 4, 4) and x.dtype == np.float32 and x[0, 0, 0, 0] == 200
    assert cmp.normalise_input(batch, [0, 0, 0], [1, 1, 1], swap_rb=True)[0, 2, 0, 0] == 200


def test_production_parity_check():
    dets = hd.Detections(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0.5]), np.array([0]), np.array([8]))
    same = [{"xyxy": [1.0, 2.0, 3.0, 4.0], "cls": 0, "score": 0.5}]
    assert cmp.compare_to_production(same, dets)["max_coord_diff"] == 0.0
    assert not cmp.compare_to_production([], dets)["same_count"]


def test_agreement_counts_unmatched_and_score_deltas():
    a = hd.Detections(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float), np.array([0.9, 0.8]),
                      np.array([0, 0]), np.array([8, 8]))
    b = hd.Detections(np.array([[0, 0, 10, 10]], float), np.array([0.7]), np.array([0]), np.array([8]))
    agreement = hd.Agreement()
    agreement.add(a, b)
    s = agreement.summary()
    assert (s["matched"], s["unmatched_a"], s["unmatched_b"]) == (1, 1, 0)
    assert s["score_delta_b_minus_a"]["mean"] == pytest.approx(-0.2)
    assert s["matched_iou"]["p50"] == pytest.approx(1.0)


# ------------------------------------------------------------------ timing / health / latency
class _FakeHailo:
    def __init__(self):
        self.pipeline = type("P", (), {"infer": lambda _self, feed: {"x": 1}})()

    def _ensure_ready(self):
        return self.pipeline

    def _decode_raw(self, raw):
        return ["boxes"]

    def predict(self, frame):
        raw = self._ensure_ready().infer({"in": frame})
        return self._decode_raw(raw)


def test_detector_instrumentation_times_infer_and_decode_without_changing_results():
    rec = st.StageRecorder()
    model = _FakeHailo()
    assert st.instrument_detector(model, rec) is True
    assert st.instrument_detector(model, rec) is True  # idempotent
    assert model.predict("frame") == ["boxes"]
    rec.add("process_total", 100.0)
    frame = rec.end_frame()
    assert {"detector_total", "hailo_infer", "decode", "preprocess", "module_logic"} <= set(frame)
    assert frame["module_logic"] == pytest.approx(100.0 - frame["detector_total"])


def test_stage_recorder_residual_and_fps():
    rec = st.StageRecorder()
    for _ in range(4):
        for stage, ms in (("capture", 5.0), ("facemesh", 10.0), ("hands", 5.0),
                          ("process_total", 40.0), ("frame_total", 50.0)):
            rec.add(stage, ms)
        rec.end_frame()
    s = rec.summary()
    assert s["module_logic"]["p50"] == pytest.approx(25.0)
    assert s["sequential_fps"] == pytest.approx(20.0)
    assert s["processing_bound_fps"] == pytest.approx(25.0)


def test_health_rows_turn_processed_frames_into_fps():
    payload = {"device": {
        "camera": {"metrics": {"fps": 30.0}},
        "inference": {"metrics": {"processed_frames": 120, "processed_p95_ms": 40.0,
                                  "processed_stalls_500ms": 1}},
        "accelerator": {"metrics": {"inferences": 1100}},
        "system": {"metrics": {"temperature_c": 61.0}},
        "monitoring": {"state": "ready"}, "monitoring_active": True}}
    first = st.health_row(payload, 0.0, None)
    payload["device"]["accelerator"]["metrics"]["inferences"] = 1160
    second = st.health_row(payload, 5.0, first)
    assert "processed_fps" not in first
    assert second["processed_fps"] == pytest.approx(24.0)
    assert second["hailo_infer_per_s"] == pytest.approx(12.0)
    summary = st.summarize_health([first, second])
    assert summary["windows"] == 1 and summary["stalls_500ms_total"] == 1
    assert summary["states_seen"] == ["ready"]


def test_latency_analysis():
    trials = st.analyse_trials([{"stimulus_ms": "1000", "alert_onset_ms": "3200", "designed_delay_ms": "2000"},
                                {"stimulus_ms": "0", "alert_onset_ms": "2400", "designed_delay_ms": "2000"}])
    assert trials["stimulus_to_alert_ms"]["mean"] == pytest.approx(2300.0)
    assert trials["minus_designed_delay_ms"]["mean"] == pytest.approx(300.0)
    glass = st.analyse_glass([{"displayed_ms": "1000", "captured_ms": "1095", "offset_ms": "15"}])
    assert glass["glass_to_capture_ms"]["mean"] == pytest.approx(80.0)


# ------------------------------------------------------------------ buzzer timelines
def test_buzzer_expected_vs_commanded_timeline():
    from core.alerts import PATTERNS

    expected = buzzer_check.expected_timeline(PATTERNS["double"].steps)
    assert expected == [(0.0, 300.0), (500.0, 800.0)]
    events = [("on", 10.0), ("on", 10.1), ("off", 10.305), ("on", 10.5), ("off", 10.8)]
    observed = buzzer_check.commanded_timeline(events)
    assert len(observed) == 2 and observed[0][1] == pytest.approx(305.0)
    result = buzzer_check.compare(expected, observed)
    assert result["ok"] and result["max_abs_error_ms"] == pytest.approx(5.0)
    assert not buzzer_check.compare(expected, observed[:1])["ok"]


def test_buzzer_audio_annotation_analysis_and_missing_patterns():
    timeline = {"patterns": {"double": {"expected_ms": [[0, 300], [500, 800]]},
                             "chirp": {"expected_ms": [[0, 150]]}}}
    rows = [{"pattern": "double", "step": "1", "onset_ms": "2520", "offset_ms": "2810"},
            {"pattern": "double", "step": "0", "onset_ms": "2000", "offset_ms": "2310"}]
    result = buzzer_check.analyse(timeline, rows)
    assert result["not_heard"] == ["chirp"]
    assert result["audible_patterns"]["double"]["onset_error_ms"] == [0.0, 20.0]


# ------------------------------------------------------------------ intrinsics
def test_fov_of_a_distortion_free_pinhole():
    w, h, fx = 640, 480, 320.0
    k = np.array([[fx, 0, (w - 1) / 2], [0, fx, (h - 1) / 2], [0, 0, 1]])
    fov = ci.fields_of_view("pinhole", k, np.zeros(5), (w, h))
    assert fov["edge_hfov_deg"] == pytest.approx(90.0, abs=1e-6)
    assert fov["paraxial_hfov_deg"] == pytest.approx(90.0, abs=1e-6)
    assert fov["edge_vfov_deg"] == pytest.approx(2 * np.degrees(np.arctan(240 / 320)), abs=1e-6)


def test_pinhole_calibration_recovers_known_intrinsics_and_recommends_paraxial_hfov():
    import cv2

    w, h = 640, 480
    k_true = np.array([[300.0, 0, 322.0], [0, 300.0, 238.0], [0, 0, 1]])
    board = ci.board_points(9, 6, 25.0)
    rng = np.random.default_rng(0)
    obj, img = [], []
    for _ in range(15):
        rvec = rng.normal(0, 0.3, 3)
        tvec = np.array([rng.uniform(-120, 0), rng.uniform(-90, 0), rng.uniform(280, 420)])
        pts, _ = cv2.projectPoints(board, rvec, tvec, k_true, np.zeros(5))
        obj.append(board)
        img.append(pts.astype(np.float32))
    result = ci.calibrate_pinhole(obj, img, (w, h))
    assert result["rms"] < 0.05
    assert result["camera_matrix"][0][0] == pytest.approx(300.0, rel=0.01)
    rec = ci.recommend({"pinhole_rational": result}, (w, h))
    assert rec["camera_hfov_deg"] == pytest.approx(np.degrees(2 * np.arctan(320 / 300.0)), abs=0.5)
    assert rec["residuals_not_modelled_by_config"]["principal_point_offset_px"][0] == pytest.approx(2.0, abs=0.5)
    assert "NOT applied" in rec["warning"]


# ------------------------------------------------------------------ integrity manifest
@pytest.mark.parametrize(("rel", "excluded"), [
    ("core/accel.py", False), ("config.yaml", True), ("models/x.hef", True),
    (".venv/lib/a.py", True), ("data/events.csv", True), ("core/__pycache__/a.pyc", True),
    ("cloud/.jwt_secret", True), ("benchmarks/results/x.json", True),
    ("benchmarks/run_benchmarks.py", False), ("scripts/hw_validation/ms_clock.html", False),
])
def test_manifest_excludes_mirror_make_deploy(rel, excluded):
    assert im.should_exclude(rel) is excluded


def test_manifest_diff_classifies_drift():
    lf, crlf = im.content_hashes(b"a\nb\n"), im.content_hashes(b"a\r\nb\r\n")
    device = {"same.py": lf, "eol.py": crlf, "changed.py": im.content_hashes(b"x"), "stale.py": lf}
    ref = {"same.py": lf, "eol.py": lf, "changed.py": im.content_hashes(b"y"), "new.py": lf}
    d = im.diff_manifests(device, ref)
    assert d["identical"] == 1 and d["differ"] == ["changed.py"]
    assert d["differ_only_line_endings"] == ["eol.py"]
    assert d["only_on_device"] == ["stale.py"] and d["only_in_revision"] == ["new.py"]
    assert not d["exact_match"]


def test_tree_manifest_prunes_excluded_dirs(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "a.py").write_text("x")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "b.py").write_text("y")
    (tmp_path / "config.yaml").write_text("token: secret")
    assert set(im.tree_manifest(tmp_path)) == {"core/a.py"}


# ------------------------------------------------------------------ compare script, end to end
_HEADS = {"Mul_224": (8, "box"), "Sigmoid_244": (8, "cls"), "Mul_264": (16, "box"),
          "Sigmoid_284": (16, "cls"), "Mul_304": (32, "box"), "Sigmoid_324": (32, "cls")}
_HEF_NAME = {(8, "box"): "m/conv53", (8, "cls"): "m/conv54", (16, "box"): "m/conv65",
             (16, "cls"): "m/conv66", (32, "box"): "m/conv76", (32, "cls"): "m/conv77"}


def _fake_float_heads(x_nchw: np.ndarray) -> dict[str, np.ndarray]:
    """A deterministic stand-in for the float ONNX: pool, 1x1 'conv', sigmoid. NCHW out."""
    rng = np.random.default_rng(42)
    w_box, w_cls = rng.normal(0, 0.02, (68, 3)), rng.normal(0, 0.02, (80, 3))
    out = {}
    for node, (stride, role) in _HEADS.items():
        g = 640 // stride
        pooled = x_nchw.reshape(1, 3, g, stride, g, stride).mean(axis=(3, 5)) / 255.0 * 50
        if role == "box":
            out[node] = np.einsum("oc,nchw->nohw", w_box, pooled).astype(np.float32)
        else:
            logits = np.einsum("oc,nchw->nohw", w_cls, pooled) - 3.5
            out[node] = (1 / (1 + np.exp(-logits))).astype(np.float32)
    return out


class _FakeSession:
    def run(self, fetch, feed):
        heads = _fake_float_heads(feed["images"])
        return [heads[name.removesuffix(":0")] for name in fetch]


def _write_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "samples").mkdir(parents=True)
    frame = np.random.default_rng(9).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    outputs_meta = [
        {"name": name, "shape": [640 // stride, 640 // stride, 68 if role == "box" else 81],
         "quant_info": {"qp_zp": 0.0, "qp_scale": 1 / 255} if role == "cls" else {}}
        for (stride, role), name in _HEF_NAME.items()]
    names = [o["name"] for o in outputs_meta]
    samples = []
    for i, mode in enumerate(("letterbox", "stretch")):
        rgb, geometry = hc.PREPROCESSORS[mode](frame, 640, 640)
        batch = rgb[None]
        heads = _fake_float_heads(batch.astype(np.float32).transpose(0, 3, 1, 2))
        dev = {}
        for node, key in _HEADS.items():
            nhwc = heads[node].transpose(0, 2, 3, 1)
            if key[1] == "cls":  # int8-like: quantised, plus a padding channel
                nhwc = np.concatenate([np.round(nhwc * 255) / 255, np.zeros((*nhwc.shape[:3], 1))],
                                      axis=-1)
            else:
                nhwc = nhwc + np.random.default_rng(i).normal(0, 0.01, nhwc.shape)
            dev[_HEF_NAME[key]] = nhwc.astype(np.float32)
        production = None
        if mode == "letterbox":
            boxes = _production_model()._decode_raw(
                dev, geometry["scale"], geometry["left"], geometry["top"], 640, 480, None, 0.25)[0].boxes
            production = [{"xyxy": [float(v) for v in b.xyxy[0]], "cls": int(b.cls[0]),
                           "score": float(b.conf[0])} for b in boxes]
        fname = f"samples/{i:05d}_{mode}.npz"
        hc.save_sample(bundle / fname, batch, [dev[n] for n in names])
        samples.append({"id": f"{i:05d}_{mode}", "image_id": "img0", "mode": mode, "file": fname,
                        "geometry": geometry, "production_detections": production,
                        "outputs": [{"index": j, "name": n} for j, n in enumerate(names)]})
    hc.write_json(bundle / "manifest.json", {
        "format": hc.BUNDLE_FORMAT, "status": "complete", "created_utc": "t",
        "model": {"sha256": "ab" * 32}, "config": {"sha256": "cd" * 32},
        "decode": {"conf": 0.25}, "modes": ["letterbox", "stretch"],
        "hef": {"input": {"shape": [640, 640, 3]}, "outputs": outputs_meta}, "samples": samples})
    return bundle


def test_compare_script_end_to_end_with_a_stand_in_float_model(tmp_path, monkeypatch):
    """Everything in compare_float_vs_int8.main except the ONNX graph surgery
    (needs the `onnx` package; covered by the next test where it is installed)."""
    import json

    bundle = _write_bundle(tmp_path)
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"not parsed: build_session is replaced")
    monkeypatch.setattr(cmp, "build_session", lambda data, end_nodes: (
        _FakeSession(), {e: f"{e}:0" for e in end_nodes}, {"input_name": "images"}))
    out = tmp_path / "report"

    assert cmp.main(["--bundle", str(bundle), "--onnx", str(onnx_file), "--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text())
    assert len(report["tensor_mapping"]) == 6
    for t in report["tensors"]:
        assert t["cosine"] > 0.99, t["hef"]
        assert t["range_verdict_device"].startswith("ok"), t
    cls8 = next(t for t in report["tensors"] if t["hef"] == "m/conv54")
    assert len(cls8["per_channel"]) == 80
    assert cls8["per_channel"][0]["max_abs_err_lsb"] <= 0.5 + 1e-3
    assert report["production_decoder_parity"] == {
        **report["production_decoder_parity"], "all_match": True, "samples": 1}
    assert set(report["int8_vs_float"]) == {f"{v}/{m}" for v in hd.VARIANTS
                                            for m in ("letterbox", "stretch")}
    assert report["int8_vs_float"]["legacy/letterbox"]["n_a"] > 0, "parity check must see boxes"
    assert report["production_vs_reference_pipeline"]["device"]["images"] == 1
    assert (out / "summary.md").read_text().startswith("# Hailo int8 vs float ONNX")


def test_onnx_graph_surgery_exposes_intermediate_heads():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("final", TensorProto.FLOAT, None)
    nodes = [helper.make_node("Relu", ["images"], ["mid"], name="Mul_224"),
             helper.make_node("Sigmoid", ["mid"], ["final"], name="Sigmoid_244")]
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y]),
                              opset_imports=[helper.make_opsetid("", 13)])
    sess, tensors, meta = cmp.build_session(model.SerializeToString(), ["Mul_224", "Sigmoid_244"])
    feed = np.random.default_rng(0).normal(size=(1, 3, 8, 8)).astype(np.float32)
    mid, final = sess.run([tensors["Mul_224"], tensors["Sigmoid_244"]], {meta["input_name"]: feed})
    np.testing.assert_allclose(mid, np.maximum(feed, 0), atol=1e-6)
    assert final.shape == (1, 3, 8, 8)
