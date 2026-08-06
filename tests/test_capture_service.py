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
