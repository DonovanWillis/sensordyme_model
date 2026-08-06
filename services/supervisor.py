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
        self._health = health if health is not None else self._health
        self._stop.clear()
        self._thread = threading.Thread(target=self._guarded_run, daemon=True, name=self.name)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("service %s did not stop within %.1fs", self.name, timeout)
        self._thread = None

    def restart(self) -> None:
        log.info("restarting service %s", self.name)
        self.stop()
        self._crashes = 0
        self.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ crash containment
    def _guarded_run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run(self._stop)
                return  # clean exit
            except Exception as exc:  # noqa: BLE001 - a service crash must never kill the device
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
                if self._stop.wait(delay):
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
