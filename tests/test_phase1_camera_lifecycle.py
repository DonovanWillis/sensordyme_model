"""Releasing the camera must actually release the device, not just stop the stream.

`Camera.release()` used to call `Picamera2.stop()` and drop the reference. stop()
halts the stream; close() releases the underlying media device. So `reopen()` --
step 1 of ARCHITECTURE.md's recovery ladder, the thing that is supposed to recover
a camera that stopped producing frames -- constructed a fresh Picamera2 against a
/dev/video node the previous instance still held, leaking the object and its file
descriptors on every attempt.

That path cannot be executed on this machine: picamera2 is Pi-only and is not
installed here. These tests therefore drive `release()`/`reopen()` against a fake
that records the calls, which is enough to pin the CONTRACT ("close is called, in
order, exactly once per release") without needing the hardware. The behaviour of
real Picamera2 against a real camera remains UNVERIFIED and is listed in
docs/HARDWARE_VERIFICATION.md as a required on-device check.
"""

from __future__ import annotations

from core.camera import Camera


class _FakePicam:
    """Records lifecycle calls in order, the way Picamera2 would receive them."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.close_raises: BaseException | None = None

    def stop(self) -> None:
        self.calls.append("stop")

    def close(self) -> None:
        self.calls.append("close")
        if self.close_raises is not None:
            raise self.close_raises


def _camera_with(picam) -> Camera:
    """A Camera wired to `picam` without touching __init__ (which opens hardware)."""
    camera = Camera.__new__(Camera)
    camera._picam = picam
    camera._cap = None
    return camera


def test_release_closes_the_device_not_just_the_stream():
    """The regression: close() was never called, so the device stayed held."""
    picam = _FakePicam()

    _camera_with(picam).release()

    assert picam.calls == ["stop", "close"], (
        f"expected stop() then close(), got {picam.calls}. Without close() the media "
        "device stays held and reopen() races the instance it is replacing."
    )


def test_release_drops_the_reference_so_a_second_release_is_a_no_op():
    """release() is called from the health monitor's restart path; it must be idempotent."""
    picam = _FakePicam()
    camera = _camera_with(picam)

    camera.release()
    camera.release()

    assert picam.calls == ["stop", "close"], "the second release re-touched a released device"
    assert camera._picam is None


def test_a_failing_close_does_not_propagate_out_of_release():
    """Some Picamera2 versions close on stop(); a double close must not crash recovery.

    This is deliberately NOT a blanket suppression: the camera reference is still
    cleared, the failure is logged, and stop() has already run. Turning a
    recoverable camera fault into an exception inside the health monitor's restart
    path would escalate a degraded camera into a process restart.
    """
    picam = _FakePicam()
    picam.close_raises = RuntimeError("Camera already closed")
    camera = _camera_with(picam)

    camera.release()  # must not raise

    assert picam.calls == ["stop", "close"]
    assert camera._picam is None


def test_stop_failing_still_clears_the_reference():
    """If stop() throws, release must not leave a half-released camera behind --
    and must not raise either. reopen() calls release(), and the capture pump
    only expects CameraError from it: a "device busy" from stop() used to escape
    and kill the pump thread, so the one fault reopen() exists for stopped the
    recovery instead."""
    class _StopFails(_FakePicam):
        def stop(self) -> None:
            self.calls.append("stop")
            raise RuntimeError("stream already stopped")

    picam = _StopFails()
    camera = _camera_with(picam)

    camera.release()  # must not raise

    assert camera._picam is None, (
        "the reference survived a failing stop(), so a later reopen() would race it"
    )
    assert picam.calls == ["stop", "close"], "close() must still run when stop() fails"


def test_reopen_releases_before_it_reopens():
    """reopen() must fully release first -- opening over a held device is the bug."""
    picam = _FakePicam()
    camera = _camera_with(picam)
    camera.is_pi = True

    opened: list[str] = []
    camera._start_picamera = lambda: opened.append("start")  # type: ignore[method-assign]

    camera.reopen()

    assert picam.calls == ["stop", "close"], "reopen() did not fully release the old device"
    assert opened == ["start"]
