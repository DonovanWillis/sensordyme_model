"""Tests for CaptureService: pumping, read-failure recovery, reopen."""

import time

import numpy as np

from core.camera import CameraError
from core.frame_bus import FrameBus
from services.capture import CaptureService


class FakeCamera:
    """Scriptable camera: yields frames, fails on demand, counts reopens."""

    fps = 100.0

    def __init__(self):
        self.reads = 0
        self.reopens = 0
        self.fail_reads = 0  # fail the next N reads

    def read(self):
        self.reads += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise CameraError("simulated read failure")
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def reopen(self):
        self.reopens += 1

    def release(self):
        pass


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_frames_flow_to_the_bus():
    camera, bus = FakeCamera(), FrameBus()
    service = CaptureService(camera, bus)
    service.start()
    try:
        assert _wait_until(lambda: bus.latest()[1] >= 3)
    finally:
        service.stop()


def test_read_failures_trigger_reopen_and_recovery():
    camera, bus = FakeCamera(), FrameBus()
    camera.fail_reads = 5  # more than the 3-failure reopen threshold
    service = CaptureService(camera, bus)
    service.start()
    try:
        assert _wait_until(lambda: camera.reopens >= 1)  # reopened in place
        assert _wait_until(lambda: bus.latest()[1] >= 1)  # and frames flow again
    finally:
        service.stop()


def test_service_restart_reopens_the_camera():
    camera, bus = FakeCamera(), FrameBus()
    service = CaptureService(camera, bus)
    service.start()
    try:
        assert _wait_until(lambda: bus.latest()[1] >= 1)
        service.restart()
        assert camera.reopens == 1
        seq_after_restart = bus.latest()[1]
        assert _wait_until(lambda: bus.latest()[1] > seq_after_restart)
    finally:
        service.stop()


def test_start_refuses_to_double_run_while_already_alive():
    """Found in a repo-wide audit: start() previously had no guard against
    running twice, unlike every Service subclass -- two pump threads on one
    non-thread-safe Camera object is a real hazard, not a theoretical one."""
    camera, bus = FakeCamera(), FrameBus()
    service = CaptureService(camera, bus)
    service.start()
    try:
        first_thread = service._thread
        assert _wait_until(lambda: bus.latest()[1] >= 1)
        service.start()  # must be a no-op: still alive
        assert service._thread is first_thread
    finally:
        service.stop()


def test_a_crash_inside_run_is_contained_not_silent():
    """Found in a repo-wide audit: run() was the raw thread target with no
    crash containment (unlike every Service subclass's _guarded_run), so any
    exception other than CameraError killed the pump thread silently."""

    class ExplodingCamera(FakeCamera):
        def read(self):
            raise RuntimeError("not a CameraError -- simulates a genuine bug")

    reported = []

    class _FakeHealth:
        def report_failure(self, name, detail):
            reported.append((name, detail))

    camera, bus = ExplodingCamera(), FrameBus()
    service = CaptureService(camera, bus, health=_FakeHealth())
    service.start()
    try:
        assert _wait_until(lambda: len(reported) >= 1)
        assert reported[0][0] == "camera"
    finally:
        service.stop()


def test_restart_does_not_reopen_a_camera_a_wedged_read_still_holds():
    """A driver-level hang can keep the pump inside read() past stop()'s join
    timeout. restart() used to reopen the camera anyway — under a thread that
    was still reading it — and then start() refused to run, leaving a reopened
    camera with no pump at all. Leave the device alone; a wedged thread is the
    process-level recovery's job (the systemd watchdog), not an in-place one."""
    import threading

    release = threading.Event()

    class WedgedCamera(FakeCamera):
        def read(self):
            release.wait()  # a driver hang: never returns until released
            return super().read()

    import pytest

    camera, bus = WedgedCamera(), FrameBus()
    service = CaptureService(camera, bus)
    service.start()
    try:
        assert _wait_until(lambda: service.is_running)
        # Raises, so the health monitor counts a FAILED restart: returning
        # normally read as success and reset its failure count.
        with pytest.raises(RuntimeError):
            service.restart(stop_timeout=0.2)
        assert camera.reopens == 0, "reopened the camera under a thread still reading it"
        assert service.is_running, "the wedged thread is still the one holding the camera"
    finally:
        release.set()
        service.stop()
