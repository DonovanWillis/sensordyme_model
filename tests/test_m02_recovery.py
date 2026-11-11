"""M0.2 — failure recovery: detector retry, fresh probes, stall evidence, restarts.

Each test reproduces a confirmed defect from the 2026-11-10 audit (or one found
while fixing them) and pins the corrected behaviour. Every failure here is
SIMULATED (a fake detector, a fake HAT, a jumped clock); none of it is a
verified physical hardware failure.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from test_driver_attention_scenarios import _FakeFaceMeshResult, _feed, _make_module, make_fake_landmarks
from test_hailo_lifecycle import hailo  # noqa: F401 - pytest fixture
from test_health import _drive_checks
from test_phase1_driver_fixes import _FakeBox, _FakeResult

import main
import modules.driver_attention as driver_attention_module
from core import accel
from core.health import HealthMonitor
from services.runtime import DeviceRuntime


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    return t


def _closed_eyes(monkeypatch, closed: dict) -> None:
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio",
                        lambda *pts: 0.05 if closed["value"] else 0.40)


def _step(module, frame, clock, seconds: float, step: float = 0.05) -> list[str]:
    fired = []
    for _ in range(round(seconds / step)):
        fired += [e.type for e in module.process(frame)]
        clock["now"] = round(clock["now"] + step, 6)
    return fired


# ------------------------------------------------- R4: a stall is not evidence
def test_a_frame_stall_does_not_count_as_eye_closure(tmp_path, clock, monkeypatch):
    """Audit F4: two closed-eye frames ~2.2 s apart fired `drowsiness`, because the
    whole unobserved gap counted as closed time. A camera freeze must not become
    evidence of sleep."""
    closed = {"value": False}
    _closed_eyes(monkeypatch, closed)
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    face = make_fake_landmarks()
    module._face_mesh.process = lambda rgb, lm=face: _FakeFaceMeshResult(lm)
    _step(module, frame, clock, 1.0)

    closed["value"] = True
    fired = _step(module, frame, clock, 0.5)           # 0.5 s observed closed
    clock["now"] += 3.0                                # camera frozen for 3 s: no frames at all
    fired += _step(module, frame, clock, 0.3)          # 0.3 s observed closed
    assert "drowsiness" not in fired, "unobserved time was counted as eye closure"

    fired = _step(module, frame, clock, 1.5)           # the closure keeps being OBSERVED
    assert "drowsiness" in fired, "a real, observed closure must still alert"


def test_a_frame_stall_does_not_count_as_looking_down(tmp_path, clock):
    """Audit F4 case C: two pitch-down frames 2.6 s apart fired `eyes_down`."""
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=-35.0, seconds=0.3)
    clock["now"] += 2.6
    fired += _feed(module, frame, clock, pitch_deg=-35.0, seconds=0.2)
    assert "eyes_down" not in fired


# ---------------------------------- R1: a detector error is no longer permanent
class _FlakyDetector:
    """A detector that raises while `failing` is set, else finds a phone."""

    names: ClassVar[dict[int, str]] = {0: "person", 67: "cell phone"}

    def __init__(self) -> None:
        self.failing = False
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        if self.failing:
            raise RuntimeError("HAILO_STREAM_ABORTED_BY_USER")
        return [_FakeResult([_FakeBox(0, (200, 60, 460, 470), 0.9), _FakeBox(67, (300, 300, 360, 380), 0.9)])]


def _detector_module(tmp_path, clock, monkeypatch):
    detector = _FlakyDetector()
    monkeypatch.setattr(driver_attention_module, "get_yolo_model", lambda weights: detector)
    module, frame = _make_module(tmp_path, clock, yolo_every_n_frames=1, enable_hands=False)
    return module, frame, detector


def test_a_transient_detector_error_recovers_after_backoff(tmp_path, clock, monkeypatch):
    """Audit F6: one exception disabled object detection for the rest of the run."""
    module, frame, detector = _detector_module(tmp_path, clock, monkeypatch)
    _step(module, frame, clock, 0.2)
    assert module._last_phone_boxes, "precondition: the detector sees the phone"

    detector.failing = True
    _step(module, frame, clock, 0.2)
    assert module._last_phone_boxes == [], "stale boxes must not outlive the detector"
    assert module.capability_status()["objects"].state == "impaired"

    detector.failing = False
    _step(module, frame, clock, 5.0)
    assert module._last_phone_boxes, "object detection never came back"
    assert module.capability_status()["objects"].state == "available"


def test_a_persistent_detector_error_is_retried_with_bounded_backoff(tmp_path, clock, monkeypatch):
    """Retries continue (so recovery is automatic) but at a bounded rate."""
    module, frame, detector = _detector_module(tmp_path, clock, monkeypatch)
    _step(module, frame, clock, 0.2)
    detector.failing = True
    detector.calls = 0
    _step(module, frame, clock, 600.0, step=0.25)       # ten minutes, 2,400 frames
    assert 5 <= detector.calls <= 25, f"{detector.calls} detector attempts in 10 min"
    status = module.capability_status()["objects"]
    assert status.state == "impaired" and "HAILO_STREAM_ABORTED_BY_USER" in status.reason


def test_face_monitoring_continues_while_object_detection_is_down(tmp_path, clock, monkeypatch):
    closed = {"value": False}
    _closed_eyes(monkeypatch, closed)
    module, frame, detector = _detector_module(tmp_path, clock, monkeypatch)
    module._face_mesh.process = lambda rgb, lm=make_fake_landmarks(): _FakeFaceMeshResult(lm)
    module._iris_veto = False
    detector.failing = True
    _step(module, frame, clock, 1.0)
    closed["value"] = True
    assert "drowsiness" in _step(module, frame, clock, 3.0)
    assert module.capability_status()["face"].state == "available"


# ------------------------------------------- R2: a fresh probe on every retry
def test_a_hat_that_appears_after_boot_is_used_without_a_restart(hailo, monkeypatch):  # noqa: F811
    """accel.py:286 reused the boot probe: a HAT that enumerated late (or was
    reseated) was never used until the service restarted."""
    accel.configure({"yolo_hef": str(hailo.path), "yolo_hef_sha256": hailo.sha},
                    info=accel.AcceleratorInfo(False, False, "none", "no Hailo device"))
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    monkeypatch.setattr(accel, "detect", lambda: accel.AcceleratorInfo(True, True, "hailo", "fake HAT"))
    assert accel.maybe_hailo_yolo("yolo11n.pt") is not None


# --------------------------- R5: "staying up degraded" must keep the watchdog fed
def _degraded_device(tmp_path, monkeypatch, health_cfg: dict, reboot_state: dict) -> DeviceRuntime:
    device = DeviceRuntime.__new__(DeviceRuntime)
    device.config = {"health": health_cfg}
    device._REBOOT_STATE_PATH = tmp_path / ".reboot_state.json"
    device._REBOOT_STATE_PATH.write_text(json.dumps(reboot_state))
    monkeypatch.setattr("services.runtime.nmcli_available", lambda: True)
    monkeypatch.setattr("services.runtime.subprocess.run", lambda *a, **k: pytest.fail("rebooted"))
    monitor = HealthMonitor()
    monitor.register("camera", critical=True, check=lambda: (False, "no frame for 60s"))
    _drive_checks(monitor, "camera", 3)
    assert monitor._critical_component_down()
    device.health = monitor
    return device


def test_a_suppressed_reboot_releases_the_watchdog_hold(tmp_path, monkeypatch):
    """The log said 'staying up degraded' while the ping stayed withheld, so systemd
    killed the process every ~45 s for a fault no restart can fix (and dropped an
    explicit stop each time)."""
    import time as _time

    device = _degraded_device(tmp_path, monkeypatch,
                              {"allow_reboot": True, "reboot_max_total": 2},
                              {"count": 2, "last_at": _time.time() - 60, "process_restarts": {}})
    device._last_resort("camera")
    assert not device.health._critical_component_down(), "watchdog still withheld"


def test_with_reboot_disabled_one_process_restart_then_stay_up(tmp_path, monkeypatch):
    """allow_reboot: false used to withhold the ping for ever: an unbounded process
    restart loop (StartLimitBurst never trips at one restart per ~45 s)."""
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setenv("WATCHDOG_USEC", "45000000")
    device = _degraded_device(tmp_path, monkeypatch, {"allow_reboot": False},
                              {"count": 0, "last_at": 0.0, "process_restarts": {}})
    device._last_resort("camera")
    assert device.health._critical_component_down(), "first time: let the watchdog restart us"
    device._last_resort("camera")                    # the restarted process, down again
    assert not device.health._critical_component_down(), "second time: stay up, keep reporting"


def test_a_released_hold_is_rearmed_once_the_component_recovers():
    monitor = HealthMonitor()
    healthy = {"value": False}
    monitor.register("camera", critical=True, check=lambda: (healthy["value"], "x"))
    _drive_checks(monitor, "camera", 3)
    monitor.release_watchdog_hold("camera")
    assert not monitor._critical_component_down()
    healthy["value"] = True
    _drive_checks(monitor, "camera", 1)
    healthy["value"] = False
    _drive_checks(monitor, "camera", 3)
    assert monitor._critical_component_down(), "a NEW outage must withhold the ping again"


# ------------------------------- R6: config and module failures never reboot-loop
@pytest.mark.parametrize("text", [
    pytest.param("module: driver_attention\n  bad: [indent\n", id="malformed-yaml"),
    pytest.param("- just\n- a list\n", id="not-a-mapping"),
    pytest.param("", id="empty"),
])
def test_an_unparseable_config_exits_with_ex_config(tmp_path, text):
    """D12: a YAML error crashed before the exit-78 guard: exit 1, Restart=always,
    StartLimitAction=reboot -- a reboot loop over a file no reboot can fix."""
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit) as exited:
        main.run(["--config", str(path)])
    assert exited.value.code == main.CONFIG_ERROR_EXIT


def test_a_missing_config_exits_with_ex_config(tmp_path):
    with pytest.raises(SystemExit) as exited:
        main.run(["--config", str(tmp_path / "nope.yaml")])
    assert exited.value.code == main.CONFIG_ERROR_EXIT


def test_a_module_that_cannot_be_built_boots_failed_instead_of_crashing(tmp_path, monkeypatch):
    """D12: module construction raised out of main.run (exit 1 -> reboot loop).
    Now the runtime starts with a placeholder that reports FAILED and why, so
    health, the dashboard, the app and BLE still work."""
    from test_m01_role_floors import _repo_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(_repo_config()))
    built = {}

    class _Broken:
        def __init__(self, config):
            raise RuntimeError("mediapipe graph failed to load")

    class _Runtime:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def run(self):
            pass

    monkeypatch.setattr(main, "get_module_class", lambda name: _Broken)
    monkeypatch.setattr("services.runtime.DeviceRuntime", _Runtime)
    monkeypatch.chdir(tmp_path)
    main.run(["--config", str(config_path), "--no-display"])
    module = built["module"]
    assert module.process(None) == []
    status = module.capability_status()["module"]
    assert status.state == "impaired" and "mediapipe graph failed to load" in status.reason
