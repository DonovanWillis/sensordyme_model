"""Service base class and the Supervisor that owns them.

A Service is a named daemon thread with a stop event and crash containment: an
exception escaping `run()` is logged and reported to the health monitor instead
of taking the process down, and the supervisor restarts the thread with backoff
(bounded — beyond that, recovery belongs to the health monitor's escalation).

The Supervisor starts services in registration order (startup dependencies:
capture before inference before provisioning) and stops them in reverse.
Every service can also be restarted individually — the health monitor's
`restart` hooks call `supervisor.restart(name)` so one wedged subsystem never
requires bouncing the rest.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from core.health import HealthMonitor
from core.logging_setup import get_logger

log = get_logger("supervisor")

_RESTART_BACKOFF_S = [1.0, 3.0, 10.0]  # per-service crash restart delays, then give up
#: A run() that lasted this long before crashing was a healthy stretch, not a
#: crash loop, so the in-thread restart budget starts over. Without it the
#: budget was for the generation's whole life: three crashes spread over days
#: ended the thread on the fourth, exactly like three in three seconds.
_CRASH_BUDGET_RESET_AFTER_S = 300.0


class Service:
    """A supervised worker thread. Subclasses implement run(stop)."""

    name = "service"

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._crashes = 0
        self._health: Optional[HealthMonitor] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self, health: Optional[HealthMonitor] = None) -> None:
        # Refuse to double-run. A wedged thread that outlived stop()'s join
        # timeout is still executing; starting a replacement beside it would
        # put two loops on one module — MediaPipe graphs are not thread-safe,
        # and interleaved tracker updates at double cadence read as detection
        # flapping. Better to stay wedged (visibly, for the health monitor's
        # escalation) than to run twice.
        if self._thread is not None and self._thread.is_alive():
            log.warning("service %s is still running; not starting a second instance", self.name)
            return
        self._health = health if health is not None else self._health
        # A FRESH stop event per generation, never a cleared shared one: an
        # abandoned thread from a previous generation still polls the old
        # event, which stays set — so it exits at its next check instead of
        # resuming forever alongside the new thread.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._guarded_run, daemon=True, name=self.name)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            if self._thread is threading.current_thread():
                # Asked to stop from inside its own run(): the event is set, so
                # the loop exits at its next check. Joining here would raise
                # "cannot join current thread"; keep the handle, it is alive.
                return
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Keep the handle: start() must be able to see the thread is
                # still alive and refuse to double-run. Dropping it here was
                # how a wedged loop became two concurrent loops.
                log.warning("service %s did not stop within %.1fs", self.name, timeout)
                return
        self._thread = None

    def restart(self, stop_timeout: float = 5.0) -> None:
        """Stop this service and start a fresh generation.

        Raises RuntimeError when no fresh generation could be started: when
        called from the service's own thread, or when the old thread is still
        alive after stop()'s timeout (wedged). Returning normally in that case
        told the health monitor the restart had worked -- it then reset the
        failure count, and a wedged loop could read as recovered while nothing
        was running at all.
        """
        if self._thread is not None and self._thread is threading.current_thread():
            raise RuntimeError(f"service {self.name} cannot restart itself from its own thread")
        log.info("restarting service %s", self.name)
        self.stop(timeout=stop_timeout)
        if self.is_running:
            raise RuntimeError(
                f"service {self.name} is still running after the stop timeout (wedged); "
                "not started again"
            )
        self._crashes = 0
        self.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ crash containment
    def _guarded_run(self) -> None:
        # Bind THIS generation's stop event. The attribute is replaced on every
        # start(), and a thread that read the live attribute could be steered
        # by a successor generation's event instead of its own.
        stop = self._stop
        while not stop.is_set():
            started = time.monotonic()
            try:
                self.run(stop)
                return  # clean exit
            except Exception as exc:
                if time.monotonic() - started >= _CRASH_BUDGET_RESET_AFTER_S:
                    self._crashes = 0  # a long healthy run, not a crash loop
                log.error("service %s crashed: %s", self.name, exc, exc_info=True)
                if self._health is not None:
                    self._health.report_failure(self.name, f"crashed: {exc}")
                if self._crashes >= len(_RESTART_BACKOFF_S):
                    log.error(
                        "service %s exceeded in-thread restart budget; leaving recovery to the health monitor",
                        self.name,
                    )
                    return
                delay = _RESTART_BACKOFF_S[self._crashes]
                self._crashes += 1
                if stop.wait(delay):
                    return

    def run(self, stop: threading.Event) -> None:
        """Service body: loop until `stop` is set. Overridden by subclasses."""
        raise NotImplementedError


class Supervisor:
    """Ordered lifecycle owner for all services."""

    def __init__(self, health: Optional[HealthMonitor] = None) -> None:
        self._services: list[Service] = []
        self._health = health

    def add(self, service: Service) -> Service:
        """Register a service; registration order is start order (stop is reversed)."""
        self._services.append(service)
        return service

    def start_all(self) -> None:
        for service in self._services:
            log.info("starting %s", service.name)
            service.start(self._health)

    def stop_all(self) -> None:
        for service in reversed(self._services):
            log.info("stopping %s", service.name)
            service.stop()

    def restart(self, name: str) -> bool:
        """Restart one service by name (the health monitor's recovery hook)."""
        for service in self._services:
            if service.name == name:
                service.restart()
                return True
        return False

    def get(self, name: str) -> Optional[Service]:
        return next((s for s in self._services if s.name == name), None)
