"""Tests for the camera's boot-time open path (no camera hardware required)."""

from typing import ClassVar

import pytest

from core.camera import CameraError
from services import runtime as runtime_module
from services.runtime import DeviceRuntime


class _FakeCamera:
    """A camera that opens successfully but whose verifying read can fail.

    This is the real boot failure, not a hypothetical: `Camera()` starts the
    sensor, and the `read()` that proves it produces frames is a separate call
    that can fail on its own -- so "open succeeded, verification failed" is a
    state that actually happens while the Pi's camera stack settles.
    """

    #: Shared registry of every handle this test opened, so a test can assert
    #: that the failed ones were released rather than leaked.
    opened: ClassVar[list["_FakeCamera"]] = []

    def __init__(self, config, budget: list[int]):
        #: Shared across constructions, deliberately. A per-instance countdown
        #: would make every attempt fail exactly once and succeed on its own
        #: retry, which is not the failure being modelled.
        self._budget = budget
        self.released = False
        _FakeCamera.opened.append(self)

    def read(self):
        if self._budget[0] > 0:
            self._budget[0] -= 1
            raise CameraError("device or resource busy")
        return object()

    def release(self):
        self.released = True


def _camera_failing_reads(count: int):
    budget = [count]
    return lambda config: _FakeCamera(config, budget)


def _runtime_with(config) -> DeviceRuntime:
    # _open_camera reads nothing but self.config, so a fully built runtime (which
    # wants a camera, a model and a network) is not needed to exercise it.
    rt = DeviceRuntime.__new__(DeviceRuntime)
    rt.config = config
    return rt


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(runtime_module.time, "sleep", lambda _s: None)
    _FakeCamera.opened = []


def test_a_camera_that_fails_verification_is_released_before_the_next_attempt(monkeypatch):
    """The retry loop must not leak the handle it just proved broken.

    `Camera()` can succeed and the verifying `read()` then fail, which left a
    live handle on /dev/video* behind on *every* attempt -- so attempts 2-5 were
    contending with the fds attempt 1 still owned, making each one strictly
    likelier to fail than the last. Five failures inside StartLimitIntervalSec
    reach systemd's StartLimitAction=reboot, so a transient glitch at boot could
    turn itself into a reboot cycle. Releasing between attempts is what makes the
    retries independent, and therefore what makes them worth having.
    """
    monkeypatch.setattr(runtime_module, "Camera", _camera_failing_reads(1))
    camera = _runtime_with({})._open_camera()

    assert len(_FakeCamera.opened) == 2, "should have taken a second attempt"
    first, second = _FakeCamera.opened
    assert first.released is True, "the failed attempt's handle was leaked"
    assert second.released is False, "the working camera must be handed back open"
    assert camera is second


def test_exhausting_the_retries_reports_the_last_cause_and_leaks_nothing(monkeypatch):
    """Giving up is allowed; giving up while still holding five sensors is not."""
    monkeypatch.setattr(runtime_module, "Camera", _camera_failing_reads(99))
    with pytest.raises(CameraError) as caught:
        _runtime_with({})._open_camera()

    assert "device or resource busy" in str(caught.value), "the real cause must survive"
    assert len(_FakeCamera.opened) == runtime_module._CAMERA_OPEN_RETRIES
    assert all(c.released for c in _FakeCamera.opened), "a handle was left open"


def test_a_camera_that_cannot_even_be_constructed_still_retries(monkeypatch):
    """The original failure mode -- Camera() itself raising -- must keep working.

    Worth pinning separately: the release-between-attempts fix added a branch
    that runs with `camera` still unbound on this path, and reaching it would
    have raised NameError over the top of the real CameraError.
    """
    attempts: list[int] = []

    def _explode(config):
        attempts.append(1)
        raise CameraError("no cameras available")

    monkeypatch.setattr(runtime_module, "Camera", _explode)
    with pytest.raises(CameraError) as caught:
        _runtime_with({})._open_camera()
    assert len(attempts) == runtime_module._CAMERA_OPEN_RETRIES
    assert "no cameras available" in str(caught.value)


def test_a_half_built_picamera_is_dropped_without_raising():
    """core.camera._release_picamera_quietly runs on the failure path only.

    A half-open sensor still holds /dev/video*, and the teardown calls on a
    broken handle can themselves raise -- which would replace the real error
    with a teardown error, the one outcome that makes the log useless.
    """
    from core.camera import Camera

    class _Broken:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True
            raise RuntimeError("already stopped")

        def close(self):
            raise RuntimeError("and this fails too")

    cam = Camera.__new__(Camera)
    broken = _Broken()
    cam._picam = broken
    cam._release_picamera_quietly()  # must not raise

    assert broken.stopped is True, "teardown should still have been attempted"
    assert cam._picam is None, "the dead handle must not be reachable again"
    cam._release_picamera_quietly()  # idempotent: safe to call twice
