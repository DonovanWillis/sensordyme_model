"""Tests for the vendored openpilot driver-monitoring module.

Covers the two things we own: the ONNX output parsing (exact struct layout,
sigmoid/REG_SCALE/exp post-processing) and that the vendored DriverMonitoring
policy reproduces openpilot's real escalation timeline (pre ~3s, prompt ~5s,
terminal ~11s of continuous distraction at 20Hz). No model, camera, or network.
"""

import ctypes
import math
from types import SimpleNamespace

import numpy as np

from modules.driver_attention_openpilot import (
    DT_DMON,
    OUTPUT_SIZE,
    DMonitoringModelResult,
    DriverMonitoring,
    EventName,
    parse_driverstate,
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# --- output parsing ---------------------------------------------------------------


def test_parse_driverstate_struct_layout_and_postprocessing():
    raw = DMonitoringModelResult()
    lhd = raw.driver_state_lhd
    lhd.face_orientation[0] = 0.4  # -> * REG_SCALE = 0.1
    lhd.face_orientation_std[0] = math.log(0.05)  # -> exp = 0.05
    lhd.face_prob = _logit(0.9)
    lhd.left_blink_prob = _logit(0.8)
    lhd.not_ready_prob[0] = _logit(0.3)
    raw.wheel_on_right_prob = _logit(0.2)

    flat = np.frombuffer(bytes(raw), dtype=np.float32).copy()
    assert flat.shape == (OUTPUT_SIZE,)

    ds = parse_driverstate(flat)
    assert abs(ds.leftDriverData.faceOrientation[0] - 0.1) < 1e-6
    assert abs(ds.leftDriverData.faceOrientationStd[0] - 0.05) < 1e-6
    assert abs(ds.leftDriverData.faceProb - 0.9) < 1e-5
    assert abs(ds.leftDriverData.leftBlinkProb - 0.8) < 1e-5
    assert abs(ds.leftDriverData.notReadyProb[0] - 0.3) < 1e-5
    assert abs(ds.wheelOnRightProb - 0.2) < 1e-5


def test_struct_size_matches_openpilot_output_size():
    assert ctypes.sizeof(DMonitoringModelResult) == OUTPUT_SIZE * ctypes.sizeof(ctypes.c_float)


# --- vendored policy: openpilot's real escalation timeline -------------------------


def _driver_state(pitch_net=0.0, blink=0.01, not_ready=0.01, face=0.99):
    dd = SimpleNamespace(
        faceOrientation=[pitch_net, 0.0, 0.0],
        facePosition=[0.0, 0.0],
        faceOrientationStd=[0.05, 0.05, 0.05],
        facePositionStd=[0.05, 0.05],
        faceProb=face,
        leftEyeProb=0.9,
        rightEyeProb=0.9,
        leftBlinkProb=blink,
        rightBlinkProb=blink,
        sunglassesProb=0.05,
        occludedProb=0.05,
        readyProb=[0.9] * 4,
        notReadyProb=[not_ready] * 2,
    )
    return SimpleNamespace(poorVisionProb=0.1, wheelOnRightProb=0.1, leftDriverData=dd, rightDriverData=dd)


def _sm(state, engaged=True, wheel=False):
    return {
        "modelV2": SimpleNamespace(
            meta=SimpleNamespace(disengagePredictions=SimpleNamespace(brakeDisengageProbs=[0.0]))
        ),
        "driverStateV2": state,
        "liveCalibration": SimpleNamespace(rpyCalib=[0.0, 0.0, 0.0]),
        "carState": SimpleNamespace(
            vEgo=25.0, steeringPressed=wheel, gasPressed=False, standstill=False, gearShifter="drive"
        ),
        "controlsState": SimpleNamespace(enabled=engaged),
    }


def test_attentive_driver_never_alerts():
    dm = DriverMonitoring()
    for _ in range(600):  # 30 simulated seconds
        dm.run_step(_sm(_driver_state()))
        assert dm.current_events.events == []
    assert dm.awareness == 1.0


def test_openpilot_escalation_timeline_pre_prompt_terminal():
    dm = DriverMonitoring()
    for _ in range(40):
        dm.run_step(_sm(_driver_state()))

    firsts = {}
    distracted = _driver_state(pitch_net=-0.7)  # head down, well past the natural threshold
    for i in range(1, 301):
        dm.run_step(_sm(distracted))
        for e in dm.current_events.events:
            firsts.setdefault(e, i * DT_DMON)

    # openpilot's real numbers: pre at 11-8=3s, prompt at 11-6=5s, terminal at 11s of
    # awareness decay, each plus ~0.3s for the distraction filter to cross 0.63.
    assert 3.0 < firsts[EventName.preDriverDistracted] < 4.0
    assert 5.0 < firsts[EventName.promptDriverDistracted] < 6.0
    assert 11.0 < firsts[EventName.driverDistracted] < 12.0


def test_recovering_attention_stops_the_countdown():
    dm = DriverMonitoring()
    distracted = _driver_state(pitch_net=-0.7)
    for _ in range(80):  # 4s distracted -> past pre, before prompt
        dm.run_step(_sm(distracted))
    assert dm.awareness < 1.0
    for _ in range(200):  # 10s attentive
        dm.run_step(_sm(_driver_state()))
    assert dm.awareness == 1.0
    assert dm.current_events.events == []


def test_face_lost_escalates_on_the_passive_timeline():
    dm = DriverMonitoring()
    for _ in range(40):
        dm.run_step(_sm(_driver_state()))

    no_face = _driver_state(face=0.1)
    seen = set()
    for _ in range(int(35 / DT_DMON)):  # face gone for 35 simulated seconds
        dm.run_step(_sm(no_face))
        seen |= set(dm.current_events.events)
    # face lost -> passive (unresponsive) alert chain, terminal within _AWARENESS_TIME=30s
    assert EventName.promptDriverUnresponsive in seen
    assert EventName.driverUnresponsive in seen


def test_instant_distraction_chime_fires_once_on_certainly_distracted(monkeypatch):
    """The additive instant chime: fires ~0.3s after detection (openpilot's own
    certainly_distracted signal), on the rising edge only, and never on attentive."""
    import modules.driver_attention_openpilot as op

    states = {"current": _driver_state()}

    class _StubModel:
        def __init__(self, path):
            pass

        def run(self, frame):
            return states["current"], 0.0

    monkeypatch.setattr(op, "_OnnxDMModel", _StubModel)
    now = [1000.0]
    monkeypatch.setattr(op.time, "monotonic", lambda: now[0])

    module = op.DriverAttentionOpenpilotModule(
        {"instant_distraction_sound": True, "phone_detection": False}
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    collected = []
    for _ in range(6):  # ~2.5 simulated seconds attentive
        now[0] += 0.5
        collected += module.process(frame)
    assert [e.type for e in collected] == []

    states["current"] = _driver_state(pitch_net=-0.7)
    for _ in range(6):  # 1.5 simulated seconds distracted
        now[0] += 0.25
        collected += module.process(frame)
    types = [e.type for e in collected]
    assert types.count("distractionDetected") == 1  # fires once, on the rising edge
    assert "instant" in next(e for e in collected if e.type == "distractionDetected").contributing_signals


def _make_stub_module(monkeypatch, states, config=None):
    import modules.driver_attention_openpilot as op

    class _StubModel:
        def __init__(self, path):
            pass

        def run(self, frame):
            return states["current"], 0.0

    monkeypatch.setattr(op, "_OnnxDMModel", _StubModel)
    now = [1000.0]
    monkeypatch.setattr(op.time, "monotonic", lambda: now[0])
    module = op.DriverAttentionOpenpilotModule(
        {"instant_distraction_sound": False, "phone_detection": False, **(config or {})}
    )
    return module, now


def test_eyes_closed_over_two_seconds_flags_distracted_driving(monkeypatch):
    states = {"current": _driver_state()}
    module, now = _make_stub_module(monkeypatch, states)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    collected = []
    for _ in range(4):  # attentive, eyes open
        now[0] += 0.5
        collected += module.process(frame)
    assert "eyesClosedDetected" not in [e.type for e in collected]

    states["current"] = _driver_state(blink=0.99)  # eyes closed
    for _ in range(6):  # 3 simulated seconds
        now[0] += 0.5
        collected += module.process(frame)
    types = [e.type for e in collected]
    assert types.count("eyesClosedDetected") == 1
    event = next(e for e in collected if e.type == "eyesClosedDetected")
    assert event.duration >= 2.0  # fired only after the sustained threshold


def test_brief_blink_does_not_flag_eyes_closed(monkeypatch):
    states = {"current": _driver_state()}
    module, now = _make_stub_module(monkeypatch, states)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    collected = []
    for _ in range(3):
        now[0] += 0.5
        collected += module.process(frame)
    states["current"] = _driver_state(blink=0.99)
    now[0] += 0.4  # 0.4s closure -- a long blink, under the 2s bar
    collected += module.process(frame)
    states["current"] = _driver_state()
    for _ in range(4):
        now[0] += 0.5
        collected += module.process(frame)
    assert "eyesClosedDetected" not in [e.type for e in collected]


def test_phone_in_use_sustained_flags_distracted_driving(monkeypatch):
    states = {"current": _driver_state()}
    module, now = _make_stub_module(monkeypatch, states)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    collected = []
    now[0] += 0.5
    collected += module.process(frame)

    module._phone_in_use = True  # what the YOLO+hands detector sets when a held phone is seen
    for _ in range(4):  # 2 simulated seconds of sustained phone use
        now[0] += 0.5
        collected += module.process(frame)
    types = [e.type for e in collected]
    assert types.count("phoneUseDetected") == 1
    assert module.is_alert_active()  # the flag drives the red alert flash

    module._phone_in_use = False  # phone put down -> flag clears, tracker re-arms
    now[0] += 0.5
    collected += module.process(frame)
    assert not module._phone_use_tracker.is_active


def test_disengaging_resets_awareness():
    dm = DriverMonitoring()
    distracted = _driver_state(pitch_net=-0.7)
    for _ in range(120):  # 6s distracted -> orange prompt territory
        dm.run_step(_sm(distracted))
    assert dm.awareness <= dm.threshold_prompt
    dm.run_step(_sm(distracted, engaged=False))  # driver disengages openpilot
    assert dm.awareness == 1.0
