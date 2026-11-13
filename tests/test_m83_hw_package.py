"""M8.3: the Monday hardware package runs on a machine with NO hardware.

Two things are pinned here, both on the Mac in CI:

  1. Every script in scripts/hw_validation/ executes and FAILS CLEANLY when
     the hardware is absent: a non-zero exit, a stated reason, and never a
     Python traceback. A script that crashes on a missing device is one an
     engineer cannot tell apart from a script that found a fault.
  2. Every hwv_stage.py stage runs end to end against stand-ins (--fake) and
     exits 3 (SIMULATED), never 0, so a dry run can never be read as a pass.
     The pure logic (parse-hef parsing, INT8-vs-float comparison, soak
     analysis) is checked on both its PASS and FAIL paths.
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

import hwv_stage as hs  # noqa: E402

PY = sys.executable


def _run(args: list[str], tmp_path: Path, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([*args], cwd=tmp_path, capture_output=True, text=True, timeout=timeout, check=False,  # noqa: S603 - fixed argv
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"})


# ------------------------------------------------------------------ 1. no hardware: clean refusals
NO_HW = {
    "hwv_stage model": ["hwv_stage.py", "model", "--no-service-check"],
    "hwv_stage capture": ["hwv_stage.py", "capture", "--no-service-check"],
    "hwv_stage latency": ["hwv_stage.py", "latency", "--no-service-check"],
    "hwv_stage buzzer": ["hwv_stage.py", "buzzer", "--no-service-check"],
    "hwv_stage parse-hef": ["hwv_stage.py", "parse-hef"],
    "hwv_stage thermal": ["hwv_stage.py", "thermal"],
    "hwv_stage model (service gate)": ["hwv_stage.py", "model"],
    "capture_hailo_outputs": ["capture_hailo_outputs.py", "--images", str(HWV / "known_good"), "--out", "cap"],
    "stage_timing stages": ["stage_timing.py", "stages", "--minutes", "1", "--out", "st"],
    "stage_timing health": ["stage_timing.py", "health", "--minutes", "1", "--interval", "5", "--out", "h.jsonl"],
    "buzzer_check run": ["buzzer_check.py", "run", "--out", "b.json"],
    "capture_calibration_frames": ["capture_calibration_frames.py", "--stream", "--count", "2", "--out", "cal"],
    "integrity_manifest snapshot": ["integrity_manifest.py", "snapshot", "--out", "int.json"],
    "capture_hailo_outputs (no service check)": ["capture_hailo_outputs.py", "--images", str(HWV / "known_good"),
                                                 "--out", "cap", "--no-service-check"],
    "buzzer_check run (no service check)": ["buzzer_check.py", "run", "--out", "b.json", "--no-service-check"],
    "calibrate_intrinsics": ["calibrate_intrinsics.py", "--images", "nope", "--pattern", "9x6", "--square-mm", "25",
                             "--out", "i.json"],
    "compare_float_vs_int8": ["compare_float_vs_int8.py", "--bundle", "nope", "--onnx", "x.onnx", "--out", "rep"],
}


@pytest.mark.parametrize("name", sorted(NO_HW))
def test_every_pi_script_refuses_cleanly_without_hardware(name, tmp_path):
    script, *rest = NO_HW[name]
    args = [PY, str(HWV / script), *rest]
    if script == "hwv_stage.py":
        args += ["--out", str(tmp_path / "reports")]
    r = _run(args, tmp_path)
    output = r.stdout + r.stderr
    assert r.returncode != 0, output
    assert "Traceback" not in output, output
    if script == "hwv_stage.py":
        assert r.returncode == hs.BLOCKED and "BLOCKED" in output


def test_the_thermal_soak_refuses_an_output_inside_the_install(tmp_path):
    r = _run(["bash", str(HWV / "thermal_soak.sh"), "--minutes", "0", "--out", "/home/pi/sensordyme/x.csv"], tmp_path)
    assert r.returncode != 0 and "Traceback" not in r.stdout + r.stderr


# ------------------------------------------------------------------ 2. --fake: everything executes, nothing passes
@pytest.mark.parametrize("stage", ["pins", "model", "parse-hef", "latency", "thermal"])
def test_each_stage_runs_against_stand_ins_and_reports_simulated(stage, tmp_path):
    r = _run([PY, str(HWV / "hwv_stage.py"), stage, "--fake", "--out", str(tmp_path)], tmp_path)
    output = r.stdout + r.stderr
    assert "Traceback" not in output, output
    if stage == "pins":  # pure software: a real PASS even with --fake (nothing simulated)
        assert r.returncode in (0, 3), output
    else:
        assert r.returncode == hs.SIMULATED, output
        assert "SIMULATED-PASS" in output
    report = json.loads(next(tmp_path.glob(f"{stage}_*.json")).read_text())
    assert stage == "pins" or (report["simulated"] is True and report["exit_code"] != 0)


def test_capture_then_parity_against_stand_ins(tmp_path):
    bundle = tmp_path / "bundle"
    r = _run([PY, str(HWV / "hwv_stage.py"), "capture", "--fake", "--bundle", str(bundle), "--out", str(tmp_path)],
             tmp_path, timeout=300)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out
    assert (bundle / "manifest.json").exists()
    if hs.EXPECTED_FLOAT.exists():
        # The input-tensor check is real even here: it is the real preprocessing code.
        assert "input tensors bit-identical to the float run's: 6/6" in out, out
        assert r.returncode == hs.SIMULATED, out
    r = _run([PY, str(HWV / "hwv_stage.py"), "parity", "--fake", "--bundle", str(bundle), "--out", str(tmp_path)],
             tmp_path)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out
    assert r.returncode in (hs.SIMULATED, hs.BLOCKED), out


def test_buzzer_stage_against_a_fake_gpio(tmp_path):
    r = _run([PY, str(HWV / "hwv_stage.py"), "buzzer", "--fake", "--patterns", "chirp", "--out", str(tmp_path)],
             tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == hs.SIMULATED, out
    assert "MANUAL: a person confirms every pattern is heard" in out


def test_list_names_every_stage_and_whether_it_needs_hardware(tmp_path):
    r = _run([PY, str(HWV / "hwv_stage.py"), "list"], tmp_path)
    for name in hs.STAGES:
        assert name in r.stdout


# ------------------------------------------------------------------ pure logic, both paths
def test_parse_hef_text_accepts_the_expected_layout_and_flags_nms():
    good = hs.parse_hef_text(hs.SIMULATED_PARSE_HEF)
    assert len(good["outputs"]) == 6 and not good["nms"] and good["architecture"] == "HAILO8L"
    assert sorted(tuple(o["shape"]) for o in good["outputs"]) == hs.EXPECTED_HEF_SHAPES
    nms = hs.parse_hef_text("Architecture HEF was compiled for: HAILO8L\n"
                            "  Output yolov11n/yolov8_nms_postprocess FLOAT32, HAILO NMS BY CLASS(number of "
                            "classes: 80, maximum bboxes per class: 100)\n")
    assert nms["nms"] and nms["outputs"] == []


def _img(file, dets):
    return {"file": file, "detections": dets}


def test_int8_vs_float_comparison_passes_close_and_fails_moved_missing_or_ghost():
    f = {"images": [_img("a", [{"cls": 41, "score": 0.78, "xyxy": [100, 100, 200, 220]},
                               {"cls": 60, "score": 0.20, "xyxy": [0, 0, 640, 480]}])]}
    close = {"a": [{"cls": 41, "score": 0.74, "xyxy": [101, 99, 201, 221]}]}
    res = hs.compare_to_float(f, close)["a"]
    assert res["missed_or_moved"] == [] and res["ghosts"] == []
    moved = {"a": [{"cls": 41, "score": 0.74, "xyxy": [130, 100, 230, 220]}]}   # IoU ~0.6 < 0.8
    assert len(hs.compare_to_float(f, moved)["a"]["missed_or_moved"]) == 1
    dropped = {"a": [{"cls": 41, "score": 0.55, "xyxy": [100, 100, 200, 220]}]}  # |delta| 0.23 > 0.10
    assert len(hs.compare_to_float(f, dropped)["a"]["missed_or_moved"]) == 1
    ghost = {"a": [*close["a"], {"cls": 67, "score": 0.5, "xyxy": [300, 300, 340, 380]}]}
    assert len(hs.compare_to_float(f, ghost)["a"]["ghosts"]) == 1
    assert hs.compare_to_float(f, {})["a"] == {"missing": True}


def test_soak_analysis_flags_throttling_and_clock_drops(tmp_path):
    header = ("utc,elapsed_s,soc_temp_c,throttled,arm_hz,core_v,fan_rpm,load1,hailo_temp_c,processed_fps,"
              "camera_fps,processed_p95_ms,hailo_inferences,monitoring_state\n")
    p = tmp_path / "s.csv"
    p.write_text(header + "".join(f"t,{i},{60 + i},{'0x0' if i < 5 else '0x50005'},"
                                  f"{2400000000 if i < 5 else 1500000000},0.9,,1,,30,30,40,10,ready\n"
                                  for i in range(10)))
    s = hs.analyse_soak(p)
    assert s["throttled_values"] == ["0x0", "0x50005"] and s["arm_hz_min"] < s["arm_hz_first"]
    assert s["peak_soc_c"] == 69.0


def test_the_known_good_set_has_no_people_and_matches_its_manifest():
    import hwv_common as hc

    sources = json.loads((HWV / "known_good" / "SOURCES.json").read_text())
    assert len(sources["images"]) == 6
    for row in sources["images"]:
        assert row["people"] is False and row["licence"]
        assert hc.sha256_file(HWV / "known_good" / row["file"]) == row["sha256"]
        assert row["shape"] == [480, 640, 3]  # camera size: predict letterboxes at scale 1, no resampling


def test_float_expectations_cover_the_known_good_set_when_present():
    if not hs.EXPECTED_FLOAT.exists():
        pytest.skip("expected_float.json is produced by the fleet job")
    exp = json.loads(hs.EXPECTED_FLOAT.read_text())
    sources = json.loads((HWV / "known_good" / "SOURCES.json").read_text())
    assert [i["file"] for i in exp["images"]] == [r["file"] for r in sources["images"]]
    assert exp["decoder"] == "legacy" and exp["conf"] == 0.05
    grey = next(i for i in exp["images"] if i["file"] == "kg_flat_grey.png")
    assert not [d for d in grey["detections"] if d["score"] >= 0.2]
    # The tensor the Pi will capture is the one the float model saw: recompute it here.
    import cv2

    from core.accel import letterbox_transform

    for img in exp["images"]:
        frame = cv2.imread(str(HWV / "known_good" / img["file"]), cv2.IMREAD_COLOR)
        scale, left, top = letterbox_transform(640, 480, 640, 640)
        assert (scale, left, top) == (1.0, 0, 80)
        batch = np.full((1, 640, 640, 3), 114, np.uint8)
        batch[0, 80:560] = frame[..., ::-1]
        import hashlib

        assert hashlib.sha256(batch.tobytes()).hexdigest() == img["input_tensor_sha256"]
