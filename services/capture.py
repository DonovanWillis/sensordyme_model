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
        # Refuse to double-run -- mirrors services/supervisor.py's Service.start(),
        # which this class duck-types but (found in a repo-wide audit) had
        # reimplemented without this guard. A wedged read (real CameraError
        # aside, e.g. a driver-level hang) can leave the old thread still
        # inside a blocking read() past stop()'s join timeout; starting a
        # second pump thread on the same non-thread-safe Camera object is
        # exactly the "two loops on one camera" failure this guard exists to
        # prevent everywhere else in the codebase.
        if self._thread is not None and self._thread.is_alive():
            log.warning("capture service is still running; not starting a second pump thread")
            return
        if health is not None:
            self._health = health
        # A FRESH Event per generation, never a cleared shared one -- an
        # abandoned thread from a previous generation still polls the OLD
        # event object, which stays set, so it exits at its next check
        # instead of resuming forever alongside a new one (same reasoning as
        # Service.start()).
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._guarded_run, args=(self._stop,), daemon=True, name=self.name
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            if self._thread is threading.current_thread():
                # Stop requested from inside the pump itself: the event is set
                # and the loop exits at its next check. join() would raise.
                return
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Keep the handle: start() must be able to see the thread is
                # still alive and refuse to double-run. Dropping it here was
                # how a wedged pump became two concurrent ones.
                log.warning("capture service did not stop within %.1fs", timeout)
                return
        self._thread = None

    def _guarded_run(self, stop: threading.Event) -> None:
        """Crash containment for the pump loop -- CaptureService predates
        services/supervisor.py's Service base class and (found in a repo-wide
        audit) was never wrapped the way every other supervised thread is:
        `run()` was the raw thread target, so any exception NOT already
        handled inside it (anything but CameraError -- e.g. a FrameBus bug)
        killed the pump thread silently, discoverable only later via
        FrameBus staleness. Not a Service subclass (kept import-light so
        tests can drive `run` directly with a fake camera), so this is a
        thin, local equivalent of Service._guarded_run rather than
        inheritance."""
        try:
            self.run(stop)
        except Exception as exc:
            log.error("capture service crashed: %s", exc, exc_info=True)
            if self._health is not None:
                self._health.report_failure(self.name, f"crashed: {exc}")

    def restart(self, stop_timeout: float = 5.0) -> None:
        """Full recovery: stop the pump, reopen the device, start again.

        Raises RuntimeError when it cannot do that, so the health monitor counts
        a failed restart rather than a successful one:

          * called from the pump thread itself. That used to happen through
            HealthMonitor.report_failure, which ran the camera's restart on the
            thread reporting the read failure: stop() then joined itself, raised,
            and -- having already set the pump's stop event -- killed the pump.
          * the pump is wedged in read() past `stop_timeout`.
        """
        if self._thread is not None and self._thread is threading.current_thread():
            raise RuntimeError("capture restart called from the pump thread itself; refusing")
        self.stop(timeout=stop_timeout)
        if self.is_running:
            # The pump is wedged inside read() (a driver-level hang). Reopening
            # the device under a thread still reading it is the one thing that
            # must not happen, and start() would refuse anyway — leaving a
            # reopened camera with no pump. In-place recovery cannot help here;
            # the health monitor's escalation (watchdog → process restart) can.
            log.error("capture restart skipped: the pump thread is wedged in a camera read")
            raise RuntimeError("capture pump is wedged in a camera read; not restarted")
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
            if elapsed < interval and stop.wait(interval - elapsed):
                break
            last_read = time.monotonic()

            try:
                frame = self._camera.read()
            except CameraError as exc:
                consecutive_failures += 1
                if self._health is not None:
                    # Informational: the camera component is judged by its
                    # pull check (frame age), and this loop is already doing
                    # the in-place reopen. Counting each failed read toward the
                    # restart threshold made three bad reads (~100 ms) trigger
                    # a component restart that pre-empted the reopen below.
                    self._health.report_failure(self.name, str(exc), escalate=False)
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
