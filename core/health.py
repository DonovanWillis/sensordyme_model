"""Centralized health monitoring and automatic recovery.

Every subsystem (camera, inference, accelerator, dashboard, uplink, network,
provisioning, hardware) registers here. The monitor thread periodically runs
each component's check, tracks failures with timestamps, and recovers in
escalating order:

  1. component-level restart (the registered `restart` callable), retried with
     a bounded budget (`max_restarts` within `restart_window_s`);
  2. if a *critical* component stays down after the budget is spent, the
     `on_last_resort` callback fires (services/runtime.py wires that to a
     rate-limited reboot on the Pi — and systemd's own StartLimitAction is the
     backstop behind even that);
  3. the systemd watchdog: while any critical component is DOWN, the monitor
     stops sending WATCHDOG keep-alives, so a wedged process is restarted by
     systemd even if Python itself is still technically running.

`snapshot()` is the single source of truth the dashboard serves at /api/health
and the mobile app displays.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.logging_setup import get_logger

log = get_logger("health")

# Component states, in increasing severity.
OK = "ok"                # last check passed
DEGRADED = "degraded"    # failing, but not yet past the failure threshold (or non-critical)
DOWN = "down"            # past the threshold; restarts exhausted or unavailable

_FAILURES_BEFORE_RESTART = 3  # consecutive check failures before a restart is attempted


@dataclass
class _Component:
    name: str
    critical: bool = False
    check: Optional[Callable[[], tuple[bool, str]]] = None
    restart: Optional[Callable[[], None]] = None
    check_interval_s: float = 5.0
    max_restarts: int = 3
    restart_window_s: float = 300.0

    state: str = OK
    detail: str = ""
    last_ok: float = field(default_factory=time.monotonic)
    last_failure: float = 0.0
    last_checked: float = 0.0
    consecutive_failures: int = 0
    total_restarts: int = 0
    restart_times: deque = field(default_factory=lambda: deque(maxlen=32))
    last_resort_fired: bool = False


class HealthMonitor:
    """Registry + periodic checker + recovery escalator for every subsystem."""

    def __init__(
        self,
        check_interval_s: float = 2.0,
        on_last_resort: Optional[Callable[[str], None]] = None,
        watchdog_ping: Optional[Callable[[], object]] = None,
    ) -> None:
        """Create the monitor.

        Args:
            check_interval_s: How often the monitor thread wakes to run due checks.
            on_last_resort: Called with the component name when a critical
                component stays down after its restart budget is spent.
            watchdog_ping: Called every wake while no critical component is DOWN
                (services/runtime.py passes core.sdnotify.notify_watchdog).
        """
        self._components: dict[str, _Component] = {}
        self._lock = threading.RLock()
        self._interval = check_interval_s
        self._on_last_resort = on_last_resort
        self._watchdog_ping = watchdog_ping
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = time.monotonic()

    # ------------------------------------------------------------------ registration
    def register(
        self,
        name: str,
        *,
        critical: bool = False,
        check: Optional[Callable[[], tuple[bool, str]]] = None,
        restart: Optional[Callable[[], None]] = None,
        check_interval_s: float = 5.0,
        max_restarts: int = 3,
        restart_window_s: float = 300.0,
    ) -> None:
        """Register a component.

        Args:
            name: Component name ("camera", "inference", "wifi", ...).
            critical: Critical components gate the systemd watchdog and may
                escalate to on_last_resort; non-critical ones only ever DEGRADE.
            check: Callable returning (healthy, detail). Omit for components
                that push their state via report_ok/report_failure instead.
            restart: Callable that attempts recovery (re-open camera, restart
                service thread, reconnect Wi-Fi). Omit if not restartable.
            check_interval_s / max_restarts / restart_window_s: recovery tuning.
        """
        with self._lock:
            self._components[name] = _Component(
                name=name,
                critical=critical,
                check=check,
                restart=restart,
                check_interval_s=check_interval_s,
                max_restarts=max_restarts,
                restart_window_s=restart_window_s,
            )

    # ------------------------------------------------------------------ push-style reporting
    def report_ok(self, name: str, detail: str = "") -> None:
        """Mark a component healthy (for components without a check callable)."""
        with self._lock:
            comp = self._components.get(name)
            if comp is None:
                return
            self._mark_ok(comp, detail)

    def report_failure(self, name: str, error: str) -> None:
        """Record a failure observed by the component itself (e.g. a thread crash)."""
        with self._lock:
            comp = self._components.get(name)
            if comp is None:
                return
            self._mark_failed(comp, error)
            self._maybe_recover(comp)

    # ------------------------------------------------------------------ monitor thread
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="health-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                due = [
                    c for c in self._components.values()
                    if c.check is not None and now - c.last_checked >= c.check_interval_s
                ]
            for comp in due:
                self._run_check(comp)
            if self._watchdog_ping is not None and not self._critical_component_down():
                self._watchdog_ping()
            self._stop.wait(self._interval)

    def _run_check(self, comp: _Component) -> None:
        comp.last_checked = time.monotonic()
        try:
            healthy, detail = comp.check()  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a failure
            healthy, detail = False, f"check raised: {exc}"
        with self._lock:
            if healthy:
                self._mark_ok(comp, detail)
            else:
                self._mark_failed(comp, detail)
                self._maybe_recover(comp)

    # ------------------------------------------------------------------ state transitions
    def _mark_ok(self, comp: _Component, detail: str) -> None:
        if comp.state != OK and comp.consecutive_failures > 0:
            log.info("%s recovered (%s)", comp.name, detail or "check passing again")
        comp.state = OK
        comp.detail = detail
        comp.last_ok = time.monotonic()
        comp.consecutive_failures = 0
        comp.last_resort_fired = False

    def _mark_failed(self, comp: _Component, error: str) -> None:
        comp.consecutive_failures += 1
        comp.last_failure = time.monotonic()
        comp.detail = error
        if comp.consecutive_failures == 1:
            log.warning("%s failing: %s", comp.name, error)
        threshold_passed = comp.consecutive_failures >= _FAILURES_BEFORE_RESTART
        comp.state = DOWN if (threshold_passed and comp.critical) else DEGRADED

    def _maybe_recover(self, comp: _Component) -> None:
        """Attempt a restart once the failure threshold is crossed; escalate when spent."""
        if comp.consecutive_failures < _FAILURES_BEFORE_RESTART:
            return

        now = time.monotonic()
        recent = [t for t in comp.restart_times if now - t <= comp.restart_window_s]
        if comp.restart is not None and len(recent) < comp.max_restarts:
            comp.restart_times.append(now)
            comp.total_restarts += 1
            log.warning(
                "restarting %s (attempt %d/%d in window): %s",
                comp.name, len(recent) + 1, comp.max_restarts, comp.detail,
            )
            try:
                comp.restart()
                # Don't mark OK here — the next check passing is what proves recovery.
                comp.consecutive_failures = _FAILURES_BEFORE_RESTART - 1
            except Exception as exc:  # noqa: BLE001
                log.error("restart of %s raised: %s", comp.name, exc)
            return

        if comp.critical and not comp.last_resort_fired and self._on_last_resort is not None:
            comp.last_resort_fired = True
            log.error("%s is down and its restart budget is spent — escalating to last resort", comp.name)
            try:
                self._on_last_resort(comp.name)
            except Exception as exc:  # noqa: BLE001
                log.error("last-resort handler raised: %s", exc)

    def _critical_component_down(self) -> bool:
        with self._lock:
            return any(c.critical and c.state == DOWN for c in self._components.values())

    # ------------------------------------------------------------------ reporting
    def snapshot(self) -> dict:
        """Full health picture: overall state plus per-component details (for /api/health)."""
        with self._lock:
            components = {}
            worst = OK
            for c in self._components.values():
                components[c.name] = {
                    "state": c.state,
                    "critical": c.critical,
                    "detail": c.detail,
                    "seconds_since_ok": round(time.monotonic() - c.last_ok, 1),
                    "consecutive_failures": c.consecutive_failures,
                    "total_restarts": c.total_restarts,
                }
                if c.state == DOWN or (c.state == DEGRADED and worst != DOWN):
                    worst = c.state
            return {
                "state": worst,
                "uptime_s": round(time.monotonic() - self._started_at, 1),
                "components": components,
            }
