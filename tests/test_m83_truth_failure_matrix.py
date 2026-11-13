"""M8.3 failure matrix (docs/FAILURE_MATRIX.md): the rows this stream proves on the
virtual device, each asserting every layer's observable behaviour, with the
truth InvariantMonitor attached so no single cycle may contradict itself.

Deterministic: the sim clock drives the device and the backend; no sleeps.
Rows owned elsewhere (event order and duplicates, stream liveness, BLE
provisioning) are referenced from the matrix, not repeated here.
"""

from __future__ import annotations

import pytest

from core import device_health, state_model
from core import monitoring_state as ms
from sim.harness import SimStack
from sim.invariants import InvariantMonitor, report_violations
from sim.scenario import ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

C = CALIBRATION_S
ANALYSIS = {"face", "objects", "hands", "pose", "attention"}


def _timeline(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), ATTENTIVE, f) for t, f in segments)))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _boot(stack, scenario) -> tuple:
    monitor = InvariantMonitor(stack)
    device = stack.boot(scenario)
    stack.claim()
    return monitor, device


def _user(stack) -> str:
    return state_model.derive(stack.row(), stack.clock.wall()).state


# ------------------------------------------------------------------ F1 camera disconnected while monitoring
def test_f1_camera_lost_while_monitoring_is_faulted_with_nothing_detecting_then_recovers(stack):
    monitor, device = _boot(stack, _timeline("cam", (C + 5, Faults(camera="frozen")), (C + 25, Faults())))
    device.advance(C + 4.0)
    assert _user(stack) == state_model.READY
    device.advance(12.0)
    status = stack.status()
    assert status["monitoring_state"] == "failed" and status["protection"] == "not_protecting"
    assert any("camera" in r for r in status["reasons"]), status["reasons"]
    assert not ANALYSIS & set(status["capabilities"]["available"]), status["capabilities"]
    assert _user(stack) == state_model.FAULTED
    device.advance(C + 20.0)
    assert stack.status()["monitoring_state"] == "ready" and _user(stack) == state_model.READY
    monitor.assert_clean()


# ------------------------------------------------------------------ F2 accelerator / detector unavailable
def test_f2_detector_failing_is_degraded_with_objects_impaired_and_still_protecting(stack):
    monitor, device = _boot(stack, _timeline("det", (C + 5, Faults(detector_failing=True)), (C + 45, Faults())))
    device.advance(C + 20.0)
    status = stack.status()
    assert status["monitoring_state"] == "degraded" and status["protection"] == "protecting"
    assert "objects" in {i["name"] for i in status["capabilities"]["impaired"]}
    assert "objects" not in status["capabilities"]["available"]
    assert _user(stack) == state_model.DEGRADED
    device.advance(60.0)
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


# ------------------------------------------------------------------ F3 process crash, F4 device restart
@pytest.mark.parametrize("how", ["crash", "restart"])
def test_f3_f4_a_crash_or_restart_is_offline_then_a_new_boot_that_comes_back_protecting(stack, how):
    scenario = _timeline("attentive")
    monitor, device = _boot(stack, scenario)
    device.advance(C + 4.0)
    boot_before = stack.status()["health_freshness"]["boot_id"]
    stack.protection("stop", confirm=True)
    device.advance(6.0)
    assert stack.status()["protection"] == "stopped"
    if how == "crash":
        stack.power_cut()
    else:
        stack.power_off()
    stack.clock.advance(31.0)  # nothing runs while the device is down
    row = stack.row()
    assert row["online"] is False and row["status"]["protection"] == "unknown" and row["health"] is None
    assert state_model.derive(row, stack.clock.wall()).state == state_model.OFFLINE
    device = stack.boot(scenario)
    device.advance(C + 6.0)
    status = stack.status()
    assert status["health_freshness"]["boot_id"] != boot_before
    # A stop is not persisted: a device that restarts comes back protecting
    # (services/runtime.py _stopped), and says so.
    assert status["protection"] == "protecting" and status["monitoring_state"] == "ready"
    monitor.assert_clean()


# ------------------------------------------------------------------ F5-F7 backend unreachable, Wi-Fi drop, network return
def test_f5_f7_an_outage_goes_offline_in_the_app_while_the_device_keeps_protecting_then_returns(stack):
    monitor, device = _boot(stack, _timeline("net", (C + 5, Faults(network_down=True)), (C + 65, Faults())))
    device.advance(C + 4.0)
    assert _user(stack) == state_model.READY
    device.advance(40.0)
    row = stack.row()
    assert row["online"] is False and row["status"]["stale"] is True and row["status"]["protection"] == "unknown"
    assert _user(stack) == state_model.OFFLINE
    assert device.runtime.protection_state()["state"] == "protecting", "the device protects offline"
    device.advance(30.0)
    assert stack.row()["online"] is True and _user(stack) == state_model.READY
    monitor.assert_clean()


# ------------------------------------------------------------------ F8 delayed telemetry
def test_f8_a_sample_that_arrives_old_is_stale_on_arrival_never_ready(stack):
    monitor, device = _boot(stack, _timeline("attentive"))
    device.advance(C + 4.0)
    device.state.fault_overrides["network_down"] = True
    device.advance(10.0)
    # The held sample ages on the device's monotonic clock: hold the next one back 40 s.
    uplink = device.runtime.uplink
    real_set_health = uplink.set_health
    uplink.set_health = lambda report: None
    device.advance(40.0)
    device.state.fault_overrides["network_down"] = False
    device.advance(2.5)
    row = stack.row()
    assert row["online"] is True and row["status"]["health_stale"] is True
    assert row["status"]["monitoring_state"] == "unknown" and _user(stack) == state_model.STALE
    uplink.set_health = real_set_health
    device.advance(6.0)
    assert _user(stack) == state_model.READY, "stale telemetry replaced by fresh is live again"
    monitor.assert_clean()


# ------------------------------------------------------------------ F13 video continues while inference stopped
def test_f13_frames_keep_arriving_while_inference_is_dead_is_faulted_and_not_detecting(stack):
    monitor, device = _boot(stack, _timeline("attentive"))
    device.advance(C + 4.0)
    runtime = device.runtime
    real_pass = runtime._inference_pass
    runtime._inference_pass = lambda *args, **kwargs: None  # the loop wedges; the camera keeps publishing
    device.advance(14.0)
    status = stack.status()
    assert "camera" in status["capabilities"]["available"], "video continues"
    assert "inference" in {i["name"] for i in status["capabilities"]["impaired"]}
    assert not ANALYSIS & set(status["capabilities"]["available"]), status["capabilities"]
    assert status["monitoring_state"] == "failed" and status["protection"] == "not_protecting"
    assert _user(stack) == state_model.FAULTED
    runtime._inference_pass = real_pass
    device.advance(C + 10.0)
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


# ------------------------------------------------------------------ F14 inference alive but detections stale
def test_f14_a_model_with_no_fresh_result_is_impaired_while_the_loop_is_alive(stack):
    monitor, device = _boot(stack, _timeline("attentive"))
    device.advance(C + 4.0)
    runtime = device.runtime
    real = runtime.perception.status

    def stale_face(*args, **kwargs):
        statuses = dict(real(*args, **kwargs))
        face = statuses["face_landmarks"]
        statuses["face_landmarks"] = type(face)(**{**face.__dict__, "state": "stale", "last_run_age_s": 9.0,
                                                   "reason": "no result for 9.0s"})
        return statuses

    runtime.perception.status = stale_face
    device.advance(4.0)
    status = stack.status()
    assert "inference" in status["capabilities"]["available"], "the loop is alive"
    assert "face" not in status["capabilities"]["available"]
    assert status["monitoring_state"] == "failed", "face is required on a driver unit"
    runtime.perception.status = real
    device.advance(6.0)
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


# ------------------------------------------------------------------ F20 sustained high SoC temperature
def test_f20_sustained_heat_is_degraded_with_the_reason_and_clears_with_hysteresis(stack):
    monitor, device = _boot(stack, _timeline("hot", (C + 5, Faults(soc_temp_c=84.0)),
                                             (C + 65, Faults(soc_temp_c=78.0)), (C + 125, Faults(soc_temp_c=60.0))))
    device.advance(C + 4.0)
    assert stack.status()["monitoring_state"] == "ready"
    device.advance(30.0)  # one thermal reading interval (20 s) after the heat started
    status = stack.status()
    assert status["monitoring_state"] == "degraded", status
    assert status["protection"] == "protecting", "a hot device still watches: degraded, never failed"
    assert any("84.0°C" in r and "slower" in r for r in status["reasons"]), status["reasons"]
    assert stack.row()["health"]["system"]["state"] == "warning"
    assert _user(stack) == state_model.DEGRADED
    device.advance(60.0)  # 78 °C: below the 80 °C throttle point, above the 75 °C clear point
    assert stack.status()["monitoring_state"] == "degraded", "no flapping at the limit"
    device.advance(60.0)  # 60 °C
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


def test_f20_a_live_throttle_bit_degrades_even_below_the_temperature_limit(stack):
    monitor, device = _boot(stack, _timeline("throttle", (C + 5, Faults(soc_temp_c=70.0,
                                                                         throttle=("under-voltage", "throttled")))))
    device.advance(C + 30.0)
    status = stack.status()
    assert status["monitoring_state"] == "degraded"
    assert any("under-voltage and throttled now" in r for r in status["reasons"]), status["reasons"]
    monitor.assert_clean()


def test_thermal_condition_uses_the_published_limits_unchanged():
    assert device_health.thermal_condition(None, [], False) is None
    assert device_health.thermal_condition(79.9, [], False) is None
    assert device_health.thermal_condition(80.0, [], False) is not None
    assert device_health.thermal_condition(76.0, [], True) is not None, "held until below 75 °C"
    assert device_health.thermal_condition(74.9, [], True) is None
    assert device_health.thermal_condition(50.0, ["throttled"], False) is not None
    evaluation = ms.evaluate(stopped=None, capabilities={"camera": ms.Capability(), "inference": ms.Capability()},
                             policies=[ms.Policy("p", "preset", "p", {"e": frozenset({"camera", "inference"})})],
                             conditions=("SoC at 84.0°C",))
    assert evaluation.state is ms.MonitoringState.DEGRADED and evaluation.reasons == ("device: SoC at 84.0°C",)
    stopped = ms.evaluate(stopped="stopped", capabilities={}, policies=[], conditions=("hot",))
    assert stopped.state is ms.MonitoringState.STOPPED, "a condition never overrides an explicit stop"


# ------------------------------------------------------------------ the device's own sample is coherent
def test_every_sample_the_device_sends_is_coherent_across_a_fault_tour(stack):
    """One evaluation feeds both the monitoring and protection blocks (M8.3)."""
    monitor, device = _boot(stack, _timeline("tour", (C + 5, Faults(camera="frozen")), (C + 20, Faults()),
                                             (C + 40, Faults(detector_failing=True)), (C + 60, Faults())))
    for _ in range(45):
        device.advance(2.0)
        sample = device.runtime.uplink._health
        assert report_violations(sample) == [], report_violations(sample)
        assert sample["protection"]["monitoring_state"] in (sample["monitoring"]["state"], "stopped")
    monitor.assert_clean()
