"""CaptureService: the single owner of the camera device.

Opens the camera exactly once, reads at the configured fps, and publishes each
frame to the FrameBus for however many consumers exist (inference, streaming).
Failed reads never propagate: after a few consecutive failures the device is
re-opened in place, and the health monitor sees staleness through
`FrameBus.age_s()` plus this service's own reports.
"""

from __future__ import annotations

import threading
import time

from core.camera import Camera, CameraError
from core.frame_bus import FrameBus
from core.health import HealthMonitor
from core.logging_setup import get_logger

log = get_logger("capture")

_READ_FAILURES_BEFORE_REOPEN = 3
_REOPEN_BACKOFF_S = 2.0


class CaptureService:
    """Camera -> FrameBus pump with in-place reopen recovery.

    Not a supervisor-thread Service subclass by accident: it is one — see
    services/supervisor.py — but kept import-light so tests can drive `run`
    with a fake camera.
    """

    name = "camera"

    def __init__(self, camera: Camera, bus: FrameBus, health: HealthMonitor | None = None) -> None:
        self._camera = camera
        self._bus = bus
        self._health = health
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ Service lifecycle
    def start(self, health: HealthMonitor | None = None) -> None:
        if health is not None:
            self._health = health
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, args=(self._stop,), daemon=True, name=self.name)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def restart(self) -> None:
        """Full recovery: stop the pump, reopen the device, start again."""
        self.stop()
        try:
            self._camera.reopen()
        except CameraError as exc:
            log.error("camera reopen during restart failed: %s", exc)
        self.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ pump loop
    def run(self, stop: threading.Event) -> None:
        interval = 1.0 / self._camera.fps if self._camera.fps > 0 else 0.0
        consecutive_failures = 0
        last_read = 0.0

        while not stop.is_set():
            # FPS throttle (moved here from Camera.frames so the bus timing is ours).
            elapsed = time.monotonic() - last_read
            if elapsed < interval:
                if stop.wait(interval - elapsed):
                    break
            last_read = time.monotonic()

            try:
                frame = self._camera.read()
            except CameraError as exc:
                consecutive_failures += 1
                if self._health is not None:
                    self._health.report_failure(self.name, str(exc))
                if consecutive_failures >= _READ_FAILURES_BEFORE_REOPEN:
                    log.warning(
                        "%d consecutive read failures — reopening the camera", consecutive_failures
                    )
                    try:
                        self._camera.reopen()
                        consecutive_failures = 0
                    except CameraError as reopen_exc:
                        log.error("camera reopen failed: %s", reopen_exc)
                        if stop.wait(_REOPEN_BACKOFF_S):
                            break
                continue

            if consecutive_failures:
                consecutive_failures = 0
                if self._health is not None:
                    self._health.report_ok(self.name, "reads recovered")
            self._bus.publish(frame)
