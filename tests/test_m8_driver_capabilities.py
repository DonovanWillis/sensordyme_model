"""Mission 8: Distracted Driving reports only what it actually computes.

Per frame (the shared Observation's face signals) and per capability (the
monitoring report the app reads): a signal is present only when it was computed
on that frame, a capability is available only when the thing it names is
running, and the experimental behaviours are reported as experimental and off.

SYNTHETIC: geometric landmarks and the virtual device. Nothing here measures
accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from test_driver_attention_scenarios import (
    _CONFIG,
    H,
    W,
    _FakeFaceMeshResult,
    clock,  # noqa: F401 - fixture
    make_fake_landmarks,
)

from core.monitoring_state import STARTING
from modules.driver_attention import DriverAttentionModule

REPO = Path(__file__).resolve().parents[1]


def _fresh(tmp_path, **overrides) -> Any:
    return DriverAttentionModule({**_CONFIG, "calibration_path": str(tmp_path / "cal.json"), **overrides})


def _step(module, clock, landmarks, seconds, step=0.05):  # noqa: F811
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for _ in range(max(1, round(seconds / step))):
        module.process(frame)
        clock["now"] += step


def _closed(landmarks):
    """The same face with both eyes shut (lids together, the iris covered)."""
    out = list(landmarks)
    for upper, lower in ((160, 144), (158, 153), (385, 380), (387, 373)):
        mid = (out[upper].y + out[lower].y) / 2.0
        out[upper] = type(out[upper])(out[upper].x, mid - 0.5 / H)
        out[lower] = type(out[lower])(out[lower].x, mid + 0.5 / H)
    return out


# ---------------------------------------------------------------------- module level
def test_attention_is_starting_until_this_drivers_forward_gaze_is_calibrated(tmp_path, clock):  # noqa: F811
    module = _fresh(tmp_path)
    neutral = make_fake_landmarks()
    _step(module, clock, neutral, 0.3)
    caps = module.capability_status()
    assert caps["attention"].state == STARTING and "calibrating" in caps["attention"].reason
    signals = module.perception_signals()
    assert signals["head_pose"].calibrated is False, "raw angles, labelled as such"
    assert signals["eye_state"].closed is None, "no personal threshold to judge an open eye by yet"

    _step(module, clock, neutral, 1.5)  # past calibration_duration_s (1.0 in this config)
    assert module.capability_status()["attention"].available
    signals = module.perception_signals()
    assert signals["head_pose"].calibrated is True and signals["eye_state"].closed is False
    assert "gaze" in signals


def test_gaze_events_need_the_calibrated_gaze_and_eye_events_do_not(tmp_path):
    reqs = _fresh(tmp_path).event_requirements()
    for event in ("distraction", "eyes_down", "eyes_up", "frequent_glancing", "phone_use"):
        assert "attention" in reqs[event], event
    for event in ("drowsiness", "fatigue_warning", "attention_lost"):
        assert reqs[event] == frozenset({"face"}), event
    assert reqs["phone_use"] >= {"face", "objects", "attention"}


def test_closed_eyes_carry_no_gaze_and_a_lost_face_carries_no_face_signal(tmp_path, clock):  # noqa: F811
    module = _fresh(tmp_path)
    neutral = make_fake_landmarks()
    _step(module, clock, neutral, 1.5)
    _step(module, clock, _closed(neutral), 0.5)
    signals = module.perception_signals()
    assert "gaze" not in signals, "the iris behind a shut lid is a guess, not a gaze"
    assert signals["eye_state"].closed is True

    _step(module, clock, None, 0.2)
    assert module.perception_signals() == {}, "no face: no eye state, no head pose, no gaze"


def test_with_hand_tracking_off_no_hands_capability_is_claimed(tmp_path):
    module = _fresh(tmp_path, enable_hands=False)
    assert "hands" not in module.capability_status()
    assert "erratic_hand_movement" not in module.event_requirements()
    assert "hands" in _fresh(tmp_path / "on").capability_status()


def test_the_head_pose_lens_model_in_use_is_stated_not_implied(tmp_path):
    """The shipped config has no Wide-lens intrinsics yet (hardware calibration,
    docs/HAILO_HARDWARE_VALIDATION.md Q6): the module says it is assuming, and
    calibrated intrinsics win when present."""
    assumed = _fresh(tmp_path).get_debug_state()["head_pose_intrinsics"]
    assert assumed.startswith("assumed") and "uncalibrated" in assumed
    assert _fresh(tmp_path / "h", camera_hfov_deg=97.0).head_pose_intrinsics.startswith("camera_hfov_deg 97")
    calibrated = _fresh(tmp_path / "k", camera_intrinsics={
        "width": 640, "height": 480, "fx": 280.0, "fy": 280.0, "cx": 320.0, "cy": 240.0})
    assert calibrated.head_pose_intrinsics == "calibrated camera_intrinsics"


# ---------------------------------------------------------------------- device level
pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

from sim.harness import SimStack  # noqa: E402
from sim.scenario import ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment  # noqa: E402

C = CALIBRATION_S


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path / "stack") as s:
        yield s


def test_first_boot_reports_starting_while_calibrating_then_ready(stack):
    stack.claim()
    device = stack.boot(Scenario("steady", "steady"))
    device.advance(5.0)
    status = stack.status()
    assert status["monitoring_state"] == "starting", status
    assert "attention: starting" in status["reasons"]
    assert "attention" in status["capabilities"]["starting"]
    assert status["monitoring_active"] is False, "not claimed as protecting before gaze alerts can fire"

    device.advance(C)
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting"
    assert "attention" in status["capabilities"]["available"]

    # A restart loads this driver's calibration: no second calibration window.
    device = stack.restart(Scenario("steady", "steady"))
    device.advance(4.0)
    assert stack.status()["monitoring_state"] == "ready"


def test_a_failing_detector_makes_the_phone_events_unavailable_never_silently_green(stack):
    stack.claim()
    fail = Faults(detector_failing=True)
    device = stack.boot(Scenario("det", "det", (Segment(0.0), Segment(C + 2, ATTENTIVE, fail))))
    device.advance(C + 8)
    status = stack.status()
    assert status["monitoring_state"] == "degraded" and status["protection"] == "protecting"
    assert "objects" in {c["name"] for c in status["capabilities"]["impaired"]}
    policy = next(p for p in status["policies"] if p["id"].startswith("preset:driver_safety"))
    assert {"phone_use", "phone_detected", "eating_drinking"} <= set(policy["unavailable_events"])
    assert not {"drowsiness", "distraction", "eyes_down", "attention_lost"} & set(policy["unavailable_events"])
    assert device.runtime.module._phone_hold.state == "none", "no boxes carried past a failure"


def test_experimental_driver_behaviours_are_reported_off_and_unvalidated(stack):
    stack.claim()
    device = stack.boot(Scenario("steady", "steady"))
    device.advance(C)
    node = next(n for n in stack.status()["perception"]["nodes"] if n["id"] == "driver_safety")
    assert node["validation"] == "synthetic_tests" and node["experimental"] is False
    assert node["state"] == "active"
    features = {f["id"]: f for f in device.runtime.monitoring_report().to_dict()["perception"]["nodes"][0]
                ["experimental_features"]}
    assert features["behavior_v2"]["enabled"] is False
    assert "not validated" in features["behavior_v2"]["status"]
    assert "hand-phone interaction" in features["behavior_v2"]["includes"]


def test_shipped_config_keeps_the_production_detector_and_loads_no_experiment():
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    accel = config["accelerator"]
    assert accel["yolo_decoder"] == "legacy"
    assert accel["yolo_hef"] == "models/damoyolo_tinynasL20_T.hef"
    driver = config["modules"]["driver_attention"]
    assert driver.get("behavior_v2") is False
    assert not [k for k in driver if "checkpoint" in k or k.endswith(("_onnx", "_pt_experimental"))]
    assert driver["yolo_weights"] == "yolo11n.pt", "a cache key; the device runs the pinned .hef"
