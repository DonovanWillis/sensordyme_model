"""Regression tests for the runtime/service-layer review findings.

Each test names the finding it pins. Every one of them failed against the code
as it was before the fix (see the docstrings for the observed failure).
"""

from __future__ import annotations

import importlib.util
import os
import stat
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from core import device_health
from core.camera import Camera, CameraError
from core.frame_bus import FrameBus
from core.health import DEGRADED, DOWN, HealthMonitor
from services import supervisor as supervisor_module
from services.capture import CaptureService
from services.supervisor import Service

REPO = Path(__file__).resolve().parent.parent


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class _GlitchyCamera:
    """Produces frames; fails the next `fail_next` reads on demand."""

    fps = 30.0

    def __init__(self) -> None:
        self.fail_next = 0
        self.reopens = 0
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            if self.fail_next > 0:
                self.fail_next -= 1
                raise CameraError("transient glitch")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def reopen(self):
        self.reopens += 1

    def release(self):
        pass


# =============================================================== F2: push reports
def test_f2_a_push_triggered_restart_never_runs_on_the_reporting_thread():
    """It ran inline on the reporter -- the capture pump -- whose restart then
    joined itself. Now it is queued for the monitor thread."""
    ran_on: list[threading.Thread] = []
    monitor = HealthMonitor()
    monitor.register(
        "worker", critical=True, restart=lambda: ran_on.append(threading.current_thread())
    )

    reporter = threading.Thread(
        target=lambda: [monitor.report_failure("worker", "boom") for _ in range(3)]
    )
    reporter.start()
    reporter.join(5)
    assert ran_on == [], "the restart ran on the thread that reported the failure"

    monitor._run_pending_actions()  # what the monitor thread does each wake
    assert ran_on == [threading.current_thread()]


def test_f2_b_push_failures_on_a_checked_component_never_escalate():
    """Three failed camera reads (~100 ms) used to count as three failed health
    checks: DOWN (watchdog withheld) and a component restart."""
    restarts: list[int] = []
    monitor = HealthMonitor()
    monitor.register(
        "camera", critical=True, check=lambda: (True, "fresh"), restart=lambda: restarts.append(1)
    )
    for _ in range(5):
        monitor.report_failure("camera", "read failed")
    monitor._run_pending_actions()

    comp = monitor.snapshot()["components"]["camera"]
    assert restarts == []
    assert comp["state"] == DEGRADED and comp["consecutive_failures"] == 0
    assert comp["detail"] == "read failed"
    assert monitor._critical_component_down() is False


def test_f2_c_a_push_ok_does_not_overrule_a_failing_check():
    monitor = HealthMonitor()
    monitor.register("camera", critical=True, check=lambda: (False, "stale"))
    comp = monitor._components["camera"]
    for _ in range(3):
        comp.last_checked = 0
        monitor._run_check(comp)
    assert comp.state == DOWN
    monitor.report_ok("camera", "reads recovered")
    assert comp.state == DOWN and comp.consecutive_failures == 3


def test_f2_d_transient_read_glitches_are_recovered_in_place_without_restarts_or_reboot():
    """End to end, wired like services/runtime.py. Before: each 3-read glitch
    killed the pump ("cannot join current thread"), spent two restarts, and the
    second glitch inside the window fired the last resort (a Pi reboot)."""
    last_resort: list[str] = []
    monitor = HealthMonitor(check_interval_s=0.1, on_last_resort=last_resort.append)
    bus, camera = FrameBus(), _GlitchyCamera()
    capture = CaptureService(camera, bus)
    boot = time.monotonic()

    def camera_check():
        age = bus.age_s()
        if age == float("inf"):
            return time.monotonic() - boot < 10.0, "waiting for first frame"
        return age < 2.0, f"last frame {age:.1f}s ago"

    monitor.register(
        "camera", critical=True, check=camera_check, restart=capture.restart, check_interval_s=0.2
    )
    capture.start(monitor)
    monitor.start()
    try:
        assert _wait_until(lambda: bus.sequence > 5)
        first_thread = capture._thread
        for glitch in range(2):
            seq = bus.sequence
            camera.fail_next = 3
            assert _wait_until(lambda g=glitch: camera.reopens >= g + 1)
            assert _wait_until(lambda s=seq: bus.sequence > s + 5), "frames did not resume"
            assert capture._thread is first_thread and capture.is_running, "the pump was killed"
        time.sleep(0.5)
        snap = monitor.snapshot()["components"]["camera"]
        assert snap["total_restarts"] == 0
        assert last_resort == []
    finally:
        monitor.stop()
        capture.stop()


def test_f2_e_capture_restart_from_the_pump_thread_refuses_and_keeps_pumping():
    outcome: dict = {}
    bus = FrameBus()

    class _SelfRestartingCamera(_GlitchyCamera):
        def read(self):
            if "error" not in outcome:
                try:
                    service.restart(stop_timeout=0.2)
                    outcome["error"] = None
                except RuntimeError as exc:
                    outcome["error"] = exc
            return super().read()

    service = CaptureService(_SelfRestartingCamera(), bus)
    service.start()
    try:
        assert _wait_until(lambda: "error" in outcome)
        assert isinstance(outcome["error"], RuntimeError)
        seq = bus.sequence
        assert _wait_until(lambda: bus.sequence > seq + 3), "the pump stopped after a self-restart"
        assert service.is_running
    finally:
        service.stop()


# ======================================================= F1: wedged restart
class _WedgedService(Service):
    name = "wedged"

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.started = threading.Event()

    def run(self, stop: threading.Event) -> None:
        self.started.set()
        self.release.wait()  # ignores `stop`: a native hang


def test_f1_restart_of_a_wedged_service_raises_instead_of_reporting_success():
    """Returning normally told HealthMonitor the restart worked; with the
    runtime's heartbeat reset, a wedged inference loop then read as healthy and
    the systemd watchdog was fed for ever."""
    service = _WedgedService()
    service.start()
    try:
        assert service.started.wait(2)
        thread = service._thread
        with pytest.raises(RuntimeError):
            service.restart(stop_timeout=0.1)
        assert service._thread is thread, "a second generation was started beside the wedged one"
    finally:
        service.release.set()
        service.stop()


def test_f1_a_raising_restart_leaves_the_component_failing():
    """The monitor must not reset the failure count for a restart that raised."""
    monitor = HealthMonitor()

    def refused():
        raise RuntimeError("wedged")

    monitor.register("inference", critical=True, check=lambda: (False, "stale"), restart=refused)
    comp = monitor._components["inference"]
    for _ in range(3):
        comp.last_checked = 0
        monitor._run_check(comp)
    assert comp.total_restarts == 1
    assert comp.consecutive_failures == 3 and comp.state == DOWN


def test_f1_service_restart_from_its_own_thread_raises_and_keeps_running():
    outcome: dict = {}

    class _SelfRestarter(Service):
        name = "self-restarter"

        def run(self, stop):
            try:
                self.restart(stop_timeout=0.1)
            except RuntimeError as exc:
                outcome["error"] = exc
            stop.wait(5)

    service = _SelfRestarter()
    service.start()
    try:
        assert _wait_until(lambda: "error" in outcome)
        time.sleep(0.1)
        assert service.is_running, "a refused self-restart still stopped the service"
    finally:
        service.stop()


# =================================================== F10: crash budget decays
def test_f10_crash_budget_resets_after_a_long_healthy_run(monkeypatch):
    """Three crashes spread over days ended the thread on the fourth."""
    monkeypatch.setattr(supervisor_module, "_RESTART_BACKOFF_S", [0.01, 0.01, 0.01])
    monkeypatch.setattr(supervisor_module, "_CRASH_BUDGET_RESET_AFTER_S", 0.05)

    class _SlowCrasher(Service):
        name = "slow-crasher"
        attempts = 0

        def run(self, stop):
            type(self).attempts += 1
            if type(self).attempts >= 8:
                stop.wait(5)
                return
            time.sleep(0.08)  # a healthy stretch longer than the reset window
            raise RuntimeError("occasional crash")

    service = _SlowCrasher()
    service.start()
    try:
        assert _wait_until(lambda: _SlowCrasher.attempts >= 8, timeout=5), (
            f"gave up after {_SlowCrasher.attempts} attempts"
        )
    finally:
        service.stop()


def test_f10_a_tight_crash_loop_still_gives_up(monkeypatch):
    monkeypatch.setattr(supervisor_module, "_RESTART_BACKOFF_S", [0.01, 0.01, 0.01])

    class _FastCrasher(Service):
        name = "fast-crasher"
        attempts = 0

        def run(self, stop):
            type(self).attempts += 1
            raise RuntimeError("boom")

    service = _FastCrasher()
    service.start()
    assert _wait_until(lambda: not service.is_running)
    assert _FastCrasher.attempts == 1 + len(supervisor_module._RESTART_BACKOFF_S)


# ===================================================== F4: unopened camera
class _FakeCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((6, 8, 3), dtype=np.uint8)

    def set(self, *_args):
        return True

    def release(self):
        self.released = True


def test_f4_a_camera_can_be_built_unopened_and_opened_later(monkeypatch):
    monkeypatch.setattr("core.camera.detect_pi", lambda: False)
    camera = Camera({"fps": 15, "width": 8, "height": 6}, open_now=False)
    assert camera.fps == 15.0
    assert camera.is_open is False
    with pytest.raises(CameraError):
        camera.read()

    def _open():
        camera._cap = _FakeCapture()

    monkeypatch.setattr(camera, "_start_cv2_capture", _open)
    camera.reopen()
    assert camera.is_open
    assert camera.read().shape == (6, 8, 3)


def test_f4_the_capture_pump_opens_an_unopened_camera_through_its_reopen_path(monkeypatch):
    monkeypatch.setattr("core.camera.detect_pi", lambda: False)
    camera = Camera({"fps": 60}, open_now=False)
    monkeypatch.setattr(
        camera, "_start_cv2_capture", lambda: setattr(camera, "_cap", _FakeCapture())
    )
    bus = FrameBus()
    service = CaptureService(camera, bus)
    service.start()
    try:
        assert _wait_until(lambda: bus.sequence >= 3), "the pump never opened the camera"
    finally:
        service.stop()


def test_f4_a_verifying_read_can_be_bounded_and_abandons_a_hung_handle():
    """A driver hang in the boot-time verifying read held the whole service
    start until TimeoutStartSec, with no health, dashboard or provisioning."""
    hang = threading.Event()

    class _HungCapture(_FakeCapture):
        def read(self):
            hang.wait(10)
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

    camera = Camera.__new__(Camera)
    camera.is_pi = False
    camera._picam = None
    handle = _HungCapture()
    camera._cap = handle
    camera._rotate_180 = False
    try:
        started = time.monotonic()
        with pytest.raises(CameraError):
            camera.read(timeout_s=0.2)
        assert time.monotonic() - started < 2.0
        assert camera.is_open is False
        camera.release()  # must not touch (and so cannot block on) the hung handle
        assert handle.released is False
    finally:
        hang.set()


# ================================================= F8: monitoring_active
class _Bus:
    sequence = 100

    def __init__(self, age: float) -> None:
        self._age = age

    def age_s(self):
        return self._age

    def interval_stats(self, caller="default"):
        return {}


class _Capture:
    def __init__(self, running: bool) -> None:
        self.is_running = running


@pytest.fixture()
def _fresh_fps(monkeypatch):
    monkeypatch.setattr(device_health, "_last_fps_sample", {})


def _collect(**kwargs):
    defaults = {
        "wifi": None,
        "module_name": "fall_detection",
        "detector_required": False,
        "caller": "t",
    }
    defaults.update(kwargs)
    return device_health.collect(**defaults)


@pytest.mark.usefixtures("_fresh_fps")
def test_f8_a_stalled_inference_loop_is_not_monitoring():
    report = _collect(bus=_Bus(0.05), capture=_Capture(True), heartbeat_age_s=60.0)
    assert report["inference"]["state"] == device_health.FAILED
    assert report["monitoring_active"] is False


@pytest.mark.usefixtures("_fresh_fps")
def test_f8_a_stopped_capture_service_is_not_monitoring():
    report = _collect(bus=_Bus(30.0), capture=_Capture(False), heartbeat_age_s=0.1)
    assert report["camera"]["detail"] == "capture service stopped"
    assert report["monitoring_active"] is False


@pytest.mark.usefixtures("_fresh_fps")
def test_f8_boot_states_still_read_as_monitoring():
    """The two ordinary boot states must not flip it."""
    no_frame_yet = _collect(bus=_Bus(float("inf")), capture=_Capture(True), heartbeat_age_s=None)
    assert no_frame_yet["monitoring_active"] is True
    healthy = _collect(bus=_Bus(0.05), capture=_Capture(True), heartbeat_age_s=0.2)
    assert healthy["monitoring_active"] is True


# ========================================== F11: processed-frame intervals
def test_f11_processed_stalls_are_visible_while_capture_cadence_is_fine():
    bus = FrameBus()
    bus.interval_stats("reader")
    bus.processed_interval_stats("reader")  # register both windows

    stop = threading.Event()

    def publisher():
        while not stop.is_set():
            bus.publish(np.zeros((2, 2, 3), dtype=np.uint8))
            time.sleep(0.02)

    thread = threading.Thread(target=publisher, daemon=True)
    thread.start()
    try:
        bus.record_processed()
        time.sleep(0.05)
        bus.record_processed()
        time.sleep(1.1)  # the inference loop stalls; capture carries on
        bus.record_processed()
    finally:
        stop.set()
        thread.join(2)

    capture = bus.interval_stats("reader")
    processed = bus.processed_interval_stats("reader")
    assert capture["evidence_losing_stalls_1s"] == 0
    assert processed["frames"] == 2
    assert processed["evidence_losing_stalls_1s"] == 1
    assert set(processed) == set(capture), "processed stats must keep interval_stats' shape"
    assert bus.processed_interval_stats("reader") == {}, "the window drains per caller"


@pytest.mark.usefixtures("_fresh_fps")
def test_f11_processed_stats_reach_the_health_report():
    bus = FrameBus()
    bus.publish(np.zeros((2, 2, 3), dtype=np.uint8))
    device_health.collect(bus=bus, heartbeat_age_s=0.1, caller="r", detector_required=False)
    bus.record_processed()
    bus.record_processed()
    report = device_health.collect(
        bus=bus, heartbeat_age_s=0.1, caller="r", detector_required=False
    )
    assert report["inference"]["metrics"]["processed_frames"] == 1
    assert "processed_p95_ms" in report["inference"]["metrics"]


# ================================================= F12: monitor shutdown
def test_f12_stop_waits_for_an_in_flight_restart():
    """stop() joined for check_interval+1 s, so shutdown raced ahead while the
    monitor was still restarting the services shutdown was stopping."""
    in_restart, done = threading.Event(), threading.Event()

    def slow_restart():
        in_restart.set()
        time.sleep(1.5)
        done.set()

    monitor = HealthMonitor(check_interval_s=0.05)
    monitor.register(
        "camera",
        critical=True,
        check=lambda: (False, "dead"),
        restart=slow_restart,
        check_interval_s=0.0,
    )
    monitor.start()
    assert in_restart.wait(5)
    monitor.stop()
    assert done.is_set(), "stop() returned while a restart was still running"
    assert not monitor._thread.is_alive()


def test_f12_no_recovery_action_runs_once_stop_has_been_called():
    restarts: list[int] = []
    monitor = HealthMonitor()
    monitor.register(
        "camera", critical=True, check=lambda: (False, "dead"), restart=lambda: restarts.append(1)
    )
    comp = monitor._components["camera"]
    monitor._stop.set()
    for _ in range(4):
        comp.last_checked = 0
        monitor._run_check(comp)
    assert restarts == []


# ========================================================= F5 / F7: boot
def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "sensordyme_preflight_review", REPO / "scripts" / "preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PreflightCamera:
    instances: ClassVar[list] = []

    def __init__(self, config, fail: bool = False) -> None:
        self.released = False
        self.fail = fail
        _PreflightCamera.instances.append(self)

    def read(self, timeout_s=None):
        if self.fail:
            raise CameraError("no frame")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_f5_preflight_survives_an_unwritable_data_directory(tmp_path, monkeypatch, capsys):
    """It crashed with PermissionError/ENOSPC writing data/preflight.json -- a
    non-zero ExecStartPre, i.e. a restart loop and then a reboot loop."""
    monkeypatch.setattr("core.camera.Camera", _PreflightCamera)
    preflight = _load_preflight()
    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.chdir(tmp_path)
    try:
        rc = preflight.main()
    finally:
        os.chmod(data, stat.S_IRWXU)
    assert rc in (0, 1)
    assert "could not write" in capsys.readouterr().out


def test_preflight_releases_the_camera_when_the_verifying_read_fails(tmp_path, monkeypatch):
    _PreflightCamera.instances = []
    monkeypatch.setattr("core.camera.Camera", lambda config: _PreflightCamera(config, fail=True))
    preflight = _load_preflight()
    monkeypatch.chdir(tmp_path)
    preflight.main()
    assert _PreflightCamera.instances and _PreflightCamera.instances[0].released


def test_f7_a_failing_preflight_or_bad_config_cannot_drive_a_reboot_loop():
    unit = (REPO / "sensordyme.service").read_text()
    lines = [line.strip() for line in unit.splitlines() if not line.strip().startswith("#")]
    pre = [line for line in lines if line.startswith("ExecStartPre=")]
    assert pre and all(line.startswith("ExecStartPre=-") for line in pre), (
        "ExecStartPre failure under Restart=always counts toward StartLimitAction=reboot"
    )
    assert "RestartPreventExitStatus=78" in lines
