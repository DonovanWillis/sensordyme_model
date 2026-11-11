# ruff: noqa: F811 - pytest fixtures imported from test_m01_role_floors are redefined as arguments
"""M0.2 — truthful monitoring state, role-aware health, actuator readiness.

Fault injection is deterministic and SIMULATED: a detector stub that raises, a
camera that stops publishing, an inference loop that stops stamping its
heartbeat, an alert output that raises or could not be set up. None of this is
a verified physical hardware failure, and no test here can tell whether a
buzzer is audible: "ready" means the software command path works.
"""

from __future__ import annotations

import itertools
import threading
import time

import numpy as np
import pytest
from test_m01_role_floors import (  # noqa: F401 - clock/isolated_state/backend are fixtures
    DRIVER_REPORT,
    PHONE_RULE,
    _drive,
    _driver_runtime,
    _ingest,
    backend,
    clock,
    isolated_state,
)

from core import monitoring_state as ms
from core import roles
from core.alerts import AlertManager, ConsoleSink, UnavailableSink
from core.logger import EventLogger
from modules.base import UnavailableModule, VisionModule
from services.runtime import DeviceRuntime

READY, DEGRADED, FAILED = ms.MonitoringState.READY, ms.MonitoringState.DEGRADED, ms.MonitoringState.FAILED
STARTING_STATE, STOPPED = ms.MonitoringState.STARTING, ms.MonitoringState.STOPPED


# ====================================================================== the pure state model
def _caps(**states):
    return {name: ms.Capability(state, None if state == ms.AVAILABLE else f"{name} {state}")
            for name, state in states.items()}


_DRIVER_POLICY = ms.Policy("preset:driver_safety", "preset", "driver_safety", events={
    "drowsiness": ms.BASE_REQUIREMENTS | {"face"},
    "phone_use": ms.BASE_REQUIREMENTS | {"face", "objects"},
})
_ALL_UP = {"camera": ms.AVAILABLE, "inference": ms.AVAILABLE, "face": ms.AVAILABLE, "objects": ms.AVAILABLE}
_BUZZER = ms.Actuator("gpio_buzzer", "buzzer", audible=True, ready=True)
_DRIVER_REQ = roles.ROLES["driver_safety"].required_capabilities


def _eval(caps, *, stopped=None, required=_DRIVER_REQ, actuators=(_BUZZER,), policies=(_DRIVER_POLICY,)):
    return ms.evaluate(stopped=stopped, capabilities=_caps(**caps), policies=policies,
                       role_required=required, actuators=actuators)


def test_state_precedence_stopped_failed_starting_degraded_ready():
    assert _eval(_ALL_UP).state is READY
    assert _eval({**_ALL_UP, "objects": ms.IMPAIRED}).state is DEGRADED
    assert _eval({**_ALL_UP, "objects": ms.STARTING}).state is STARTING_STATE
    # A known failure outranks "still starting".
    assert _eval({**_ALL_UP, "objects": ms.STARTING, "face": ms.IMPAIRED}).state is FAILED
    # An explicit stop outranks everything, failures included.
    assert _eval({**_ALL_UP, "camera": ms.IMPAIRED}, stopped="stopped by app").state is STOPPED


def test_a_role_required_capability_is_failed_anything_else_degraded():
    face_down = _eval({**_ALL_UP, "face": ms.IMPAIRED})
    assert face_down.state is FAILED and "face" in face_down.reasons[0]
    objects_down = _eval({**_ALL_UP, "objects": ms.IMPAIRED})
    assert objects_down.state is DEGRADED
    policy = objects_down.policies[0]
    assert policy.can_evaluate and not policy.operational
    assert policy.unavailable_events == ("phone_use",)


def test_without_a_role_requirement_failure_means_nothing_can_be_evaluated():
    assert _eval({**_ALL_UP, "objects": ms.IMPAIRED}, required=()).state is DEGRADED
    face_down = _eval({**_ALL_UP, "face": ms.IMPAIRED}, required=())  # both events need the face
    assert face_down.state is FAILED and face_down.reasons[0] == "nothing can be monitored"
    nothing = _eval({**_ALL_UP, "camera": ms.IMPAIRED}, required=())
    assert nothing.state is FAILED and nothing.reasons[0] == "nothing can be monitored"
    assert _eval(_ALL_UP, required=(), policies=()).state is FAILED  # no policy at all


def test_actuator_readiness_is_role_aware():
    dead = ms.Actuator("gpio_buzzer", "buzzer", audible=True, ready=False, detail="not running on a Pi")
    console = ms.Actuator("console", "console", audible=False, ready=True)
    driver = _eval(_ALL_UP, actuators=(dead, console))
    assert driver.state is FAILED and "audible_alert" in driver.reasons[0]
    general = _eval(_ALL_UP, required=(), actuators=(dead, console))
    assert general.state is DEGRADED and "gpio_buzzer" in general.reasons[0]
    assert all(not a.physically_verified for a in (dead, console))


def test_a_policy_blocked_for_a_non_capability_reason_is_reported():
    rule = ms.Policy("rule:r1", "rule", "person in driveway", events={"r1": ms.BASE_REQUIREMENTS | {"objects"}},
                     blocked={"r1": "area 'driveway' is not drawn yet"})
    result = _eval(_ALL_UP, policies=(_DRIVER_POLICY, rule))
    assert result.state is DEGRADED
    assert any("driveway" in r for r in result.reasons)


# ====================================================================== runtime, real driver module
def _tick(runtime):
    """The health monitor's 2 s tick (it runs on its own thread in production)."""
    return runtime._monitoring_check()


def _report(runtime):
    return runtime.monitoring_report()


def _impaired(actuator):
    return [p for p in actuator.played if p[2] == "impaired"]


def _freeze_camera(runtime, clock, seconds, step=0.5):
    """No frames arrive; the loop is alive (it stamps its heartbeat on each timeout)."""
    for _ in range(round(seconds / step)):
        runtime._heartbeat = clock["now"]
        _tick(runtime)
        clock["now"] = round(clock["now"] + step, 6)


def _hang_inference(runtime, module, frame, clock, seconds, step=0.5):
    """Frames keep arriving, but the inference loop is wedged (no heartbeat, nothing analysed)."""
    for _ in range(round(seconds / step)):
        runtime.bus.publish(frame)
        _tick(runtime)
        clock["now"] = round(clock["now"] + step, 6)


def test_a_working_driver_device_is_ready_and_reports_everything(tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    _drive(runtime, module, frame, clock, 1.0)
    healthy, _detail = _tick(runtime)
    report = _report(runtime).to_dict()
    assert healthy and report["state"] == "ready" and report["monitoring"] is True
    assert {"camera", "inference", "face", "objects", "audible_alert"} <= set(report["available"])
    assert report["impaired"] == []
    assert report["policies"][0]["operational"] is True
    assert report["role"] == "driver_safety" and report["active_preset"] == "driver_safety"
    assert report["last_frame_age_s"] is not None and report["last_frame_age_s"] < 0.2
    assert report["last_inference_age_s"] is not None and report["last_inference_age_s"] < 0.2
    assert report["versions"]["software"] and "model" in report["versions"]
    assert report["actuators"][0]["physically_verified"] is False
    health = runtime._collect_health("test")
    assert health["monitoring"]["state"] == "ready" and health["monitoring_active"] is True
    runtime.logger.close()


def test_object_detection_failure_degrades_and_face_monitoring_keeps_alerting(
        tmp_path, clock, monkeypatch, isolated_state):
    """The spec's example: object detection fails, face-based monitoring continues,
    the device says exactly which events are unavailable, and recovers by itself."""
    closed = {"value": False}
    runtime, module, frame, actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    module.stub_detector.failing = True
    _drive(runtime, module, frame, clock, 1.0)
    _tick(runtime)
    report = _report(runtime)
    assert report.state is DEGRADED
    assert report.capabilities["objects"].impaired and report.capabilities["face"].available
    assert "phone_use" in report.policies[0].unavailable_events
    assert "drowsiness" not in report.policies[0].unavailable_events
    protection = runtime.protection_state()
    assert protection["state"] == "protecting" and protection["degraded"]
    assert runtime._collect_health("test")["monitoring_active"] is True

    closed["value"] = True
    _drive(runtime, module, frame, clock, 3.0)
    assert any(p[1] == "drowsiness" for p in actuator.played), "face monitoring stopped with the detector"
    assert _impaired(actuator) == [], "DEGRADED must not sound the impaired alarm"

    closed["value"] = False
    module.stub_detector.failing = False
    _drive(runtime, module, frame, clock, 5.0)
    _tick(runtime)
    assert _report(runtime).state is READY, _report(runtime).reasons
    runtime.logger.close()


def test_a_frozen_camera_is_failed_signalled_bounded_and_recovers_without_duplicates(
        tmp_path, clock, monkeypatch, isolated_state):
    closed = {"value": False}
    runtime, module, frame, actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    closed["value"] = True  # after calibration, like a driver falling asleep
    _drive(runtime, module, frame, clock, 3.0)
    assert "drowsiness" in runtime.alert_manager.open_keys("module")

    _freeze_camera(runtime, clock, 3.0)
    report = _report(runtime)
    assert report.state is FAILED
    assert any("no new frame" in r for r in report.reasons)
    assert runtime.alert_manager.open_keys("module") == set(), "stale episodes survived the outage"
    assert len(_impaired(actuator)) == 1
    assert runtime._collect_health("test")["monitoring_active"] is False

    _freeze_camera(runtime, clock, 600.0, step=1.0)  # ten minutes dark
    impaired = _impaired(actuator)
    assert 2 <= len(impaired) <= 21, f"{len(impaired)} impaired alerts in 10 min"
    gaps = [b[0] - a[0] for a, b in itertools.pairwise(impaired)]
    assert min(gaps) >= 15.0 - 1e-6

    played_before = len(actuator.played)
    _drive(runtime, module, frame, clock, 0.5)  # frames again; the driver is still asleep
    _tick(runtime)
    assert _report(runtime).state is READY
    assert runtime.alert_manager.open_keys("health") == set(), "impaired episode did not end"
    new = [p for p in actuator.played[played_before:] if p[1] == "drowsiness"]
    assert len(new) == 1, "a still-sleeping driver must alert once on recovery, not zero or twice"
    runtime.logger.close()


def test_a_hung_inference_loop_is_failed_and_signalled_from_the_health_thread(
        tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    _drive(runtime, module, frame, clock, 1.0)
    _hang_inference(runtime, module, frame, clock, 12.0)
    report = _report(runtime)
    assert report.state is FAILED
    assert any("unresponsive" in r for r in report.reasons)
    assert len(_impaired(actuator)) == 1
    runtime.logger.close()


def test_stop_and_resume_with_the_driver_still_asleep(tmp_path, clock, monkeypatch, isolated_state):
    """Explicit stop is STOPPED (no impaired alarm); after resume a driver still
    asleep alerts again at once (M0.1 review: 0 actuations for 105 s)."""
    closed = {"value": False}
    runtime, module, frame, actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    closed["value"] = True
    _drive(runtime, module, frame, clock, 3.0)
    runtime._apply_protection({"action": "stop", "confirm": True}, source="test")
    _freeze_camera(runtime, clock, 5.0)  # stopped AND no frames: still STOPPED, never FAILED
    assert _report(runtime).state is STOPPED
    _drive(runtime, module, frame, clock, 30.0)  # frames flow; nothing is analysed while stopped
    _tick(runtime)
    assert _report(runtime).state is STOPPED
    assert _impaired(actuator) == []
    assert runtime._collect_health("test")["monitoring_active"] is False

    played = len(actuator.played)
    runtime._apply_protection({"action": "resume"}, source="test")
    assert _report(runtime).state is STARTING_STATE  # nothing analysed since resume yet
    _drive(runtime, module, frame, clock, 0.1)
    assert any(p[1] == "drowsiness" for p in actuator.played[played:]), "no alert after resume"
    _tick(runtime)
    assert _report(runtime).state is READY
    runtime.logger.close()


def test_an_unavailable_buzzer_fails_a_driver_device_but_only_degrades_a_general_one(
        tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    runtime.alert_manager._sinks = [UnavailableSink("gpio_buzzer", "buzzer", True, "not running on a Raspberry Pi"),
                                    ConsoleSink()]
    _drive(runtime, module, frame, clock, 1.0)
    report = _report(runtime)
    assert report.state is FAILED
    assert any("audible_alert" in r and "not running on a Raspberry Pi" in r for r in report.reasons)
    assert runtime.protection_state()["state"] == "not_protecting"

    runtime.role = roles.ROLES["general"]
    report = _report(runtime)
    assert report.state is DEGRADED
    assert any("gpio_buzzer" in r for r in report.reasons)
    runtime.logger.close()


def test_a_failing_alert_command_makes_the_output_not_ready(tmp_path, clock, monkeypatch, isolated_state):
    class _BrokenBuzzer:
        name, kind, audible = "gpio_buzzer", "buzzer", True

        def play(self, pattern, event, category):
            raise OSError("lgpio: GPIO busy (simulated)")

    closed = {"value": False}
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.alert_manager._sinks = [_BrokenBuzzer(), ConsoleSink()]
    _drive(runtime, module, frame, clock, 1.0)
    assert _report(runtime).state is READY  # nothing has failed yet: it was never commanded
    closed["value"] = True
    _drive(runtime, module, frame, clock, 3.0)
    buzzer = next(a for a in runtime.alert_manager.actuators() if a.name == "gpio_buzzer")
    assert not buzzer.ready and buzzer.commands_failed >= 1 and "GPIO busy" in buzzer.detail
    assert _report(runtime).state is FAILED
    runtime.logger.close()


def test_a_rule_whose_detector_is_down_cannot_be_evaluated(tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    runtime.uplink.rules = [PHONE_RULE]
    runtime._apply_pending_commands()
    module.stub_detector.failing = True
    _drive(runtime, module, frame, clock, 1.0)
    rule = next(p for p in _report(runtime).policies if p.kind == "rule")
    assert not rule.can_evaluate and "objects" in rule.reasons[0]
    assert _report(runtime).state is DEGRADED
    runtime.logger.close()


def test_a_property_device_requires_object_detection(tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    runtime.role = roles.ROLES["property"]
    module.stub_detector.failing = True
    _drive(runtime, module, frame, clock, 1.0)
    report = _report(runtime)
    assert report.state is FAILED and "objects" in report.reasons[0]
    runtime.logger.close()


def test_an_unbuildable_module_reports_failed_with_its_reason(tmp_path, isolated_state):
    module = UnavailableModule("driver_attention", "mediapipe graph failed to load")
    runtime = DeviceRuntime(
        config={"modules": {"driver_attention": {}}}, profile="pi", module=module,
        module_name="driver_attention", alert_manager=AlertManager(sinks=[ConsoleSink()]), alert_cfg={},
        alert_event_types=set(), alert_types_for=lambda n, p=None: set(), presets={}, active_preset=None,
        display=False, debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    report = _report(runtime)
    assert report.state is FAILED
    assert any("could not be built: mediapipe graph failed to load" in r for r in report.reasons)
    runtime.logger.close()


def test_after_a_service_restart_the_device_is_starting_then_ready(tmp_path, clock, monkeypatch, isolated_state):
    runtime, module, frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed={"value": False})
    runtime.logger.close()
    clock["now"] += 5.0  # systemd restarts the process
    from test_m01_role_floors import _repo_config, alert_types_for, build_alert_manager

    config = _repo_config()
    restarted = DeviceRuntime(
        config=config, profile="pi", module=module, module_name="driver_attention",
        alert_manager=build_alert_manager({**config["alerts"], "sinks": []}, False), alert_cfg=config["alerts"],
        alert_event_types=runtime.alert_event_types, alert_types_for=alert_types_for, presets=runtime.presets,
        active_preset="driver_safety", display=False, debug=False,
    )
    restarted.alert_manager._sinks = runtime.alert_manager._sinks
    restarted.logger = EventLogger(str(tmp_path / "events2.csv"))
    assert _report(restarted).state is STARTING_STATE
    _drive(restarted, module, frame, clock, 1.0)
    assert _report(restarted).state is READY
    restarted.logger.close()


# ====================================================================== loop gating (audit F05)
class _FaceAndObjects(VisionModule):
    """Declares one face-only event and one object event; counts frames processed."""

    name = "m02_face_and_objects"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.processed = 0

    def process(self, frame):
        self.processed += 1
        return []

    def annotate(self, frame):
        return frame

    def event_requirements(self):
        return {"looked_away": frozenset({"face"}), "phone": frozenset({"objects"})}


class _ObjectsOnly(_FaceAndObjects):
    def event_requirements(self):
        return {"intruder": frozenset({"objects"})}


@pytest.mark.parametrize(("module_cls", "expect_processing"), [(_FaceAndObjects, True), (_ObjectsOnly, False)])
def test_a_missing_detector_stops_only_what_needs_it(tmp_path, isolated_state, module_cls, expect_processing):
    """Audit F05: with the detector missing at boot the loop ran NO module, so
    face-only drowsiness was off too. Now a module that can still do something
    keeps running; one that cannot still idles honestly."""
    module = module_cls({})
    runtime = DeviceRuntime(
        config={"modules": {module.name: {}}}, profile="pi", module=module, module_name=module.name,
        alert_manager=AlertManager(sinks=[]), alert_cfg={}, alert_event_types=set(),
        alert_types_for=lambda n, p=None: set(), presets={}, active_preset=None, display=False, debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    runtime.telemetry = runtime.logger
    runtime._detection_degraded = "accelerator.yolo_hef points to a missing file (simulated)"
    runtime._retry_detection_backend = lambda: None
    stop = threading.Event()
    loop = threading.Thread(target=runtime._inference_loop, args=(stop,), daemon=True)
    loop.start()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(10):
        runtime.bus.publish(frame)
        time.sleep(0.03)
    stop.set()
    loop.join(2)
    assert (module.processed > 0) is expect_processing
    report = _report(runtime)
    assert report.capabilities["objects"].impaired
    assert report.state is (DEGRADED if expect_processing else FAILED)
    runtime.logger.close()


# ====================================================================== backend convergence
def test_protection_converges_on_the_request_id_not_on_a_matching_state(backend):
    """M0.1 review: a resume sent while a stop was still in flight was 'converged'
    by a report taken before the stop landed, and the device then stayed stopped."""
    client, headers = backend
    _ingest(client, {**DRIVER_REPORT, "protection": {**DRIVER_REPORT["protection"], "last_command_id": None}})
    stop_id = client.post("/v1/devices/1/protection", json={"action": "stop", "confirm": True},
                          headers=headers).json()["command_id"]
    assert _ingest(client, DRIVER_REPORT)["set_protection"]["id"] == stop_id
    resume_id = client.post("/v1/devices/1/protection", json={"action": "resume"}, headers=headers).json()["command_id"]
    # The device reports "protecting" before it has even applied the stop:
    early = {**DRIVER_REPORT, "protection": {**DRIVER_REPORT["protection"], "last_command_id": None}}
    assert _ingest(client, early)["set_protection"]["id"] == resume_id, "the resume was dropped"
    applied = {**DRIVER_REPORT, "protection": {**DRIVER_REPORT["protection"], "last_command_id": resume_id}}
    assert (_ingest(client, applied) or {}).get("set_protection") is None


def test_firmware_without_stop_support_is_told_so_and_never_nagged(backend):
    client, headers = backend
    old_firmware = {"active": False}  # pre-M0.1 monitoring report: no role, no protection
    _ingest(client, old_firmware)
    resp = client.post("/v1/devices/1/protection", json={"action": "stop", "confirm": True}, headers=headers)
    assert resp.status_code == 409 and "does not support" in resp.json()["detail"]


def test_a_stored_preset_the_role_forbids_is_dropped_not_resent_forever(backend):
    import importlib

    client, _headers = backend
    app_module = importlib.import_module("cloud.app")
    _ingest(client, DRIVER_REPORT)
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET desired_preset = 'baby_monitoring' WHERE id = 1")
    assert "set_preset" not in (_ingest(client, DRIVER_REPORT) or {})
    with app_module.db() as conn:
        assert conn.execute("SELECT desired_preset FROM devices WHERE id = 1").fetchone()[0] is None

