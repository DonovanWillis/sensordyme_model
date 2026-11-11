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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from core.logging_setup import get_logger

log = get_logger("health")

# Component states, in increasing severity.
OK = "ok"                # last check passed
DEGRADED = "degraded"    # failing, but not yet past the failure threshold (or non-critical)
DOWN = "down"            # past the threshold; restarts exhausted or unavailable

_FAILURES_BEFORE_RESTART = 3  # consecutive check failures before a restart is attempted
#: How long stop() waits, beyond one check interval, for an in-flight check or
#: recovery action to finish. A camera restart is a 5 s join plus a reopen; a
#: short join let shutdown race ahead while the monitor was still restarting
#: the very services shutdown was stopping.
_STOP_JOIN_GRACE_S = 30.0


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
    #: The last-resort policy decided to stay up degraded rather than restart:
    #: this outage no longer withholds the watchdog ping (see release_watchdog_hold).
    watchdog_released: bool = False


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
        #: Recovery actions decided by a push report, run by the monitor thread.
        #: Never on the reporter's own thread: that thread may be the very one
        #: the restart has to stop and join.
        self._pending_actions: deque = deque()

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
        """Mark a component healthy.

        For a component with a pull `check`, a push "ok" only clears a
        push-reported degradation; it never overrules a check that is failing,
        whose own next pass is what proves recovery.
        """
        with self._lock:
            comp = self._components.get(name)
            if comp is None:
                return
            if comp.check is not None and comp.consecutive_failures > 0:
                return
            self._mark_ok(comp, detail)

    def report_failure(self, name: str, error: str, *, escalate: bool = True) -> None:
        """Record a failure observed by the component itself (e.g. a thread crash).

        Escalation (restart, last resort) happens only for push-only components
        -- those registered without a `check` -- and only when `escalate` is
        true. A component with a pull check is recovered by the monitor thread
        on the strength of that check; a push report about it is informational
        (detail, timestamp, OK -> DEGRADED). Two reasons:

          * the reporter is often the component's own worker thread, and its
            restart has to stop and join that thread. Run inline, the camera
            pump joined itself, raised, and died with its stop event set.
          * push reports arrive far faster than checks (every failed camera
            read), so counting them toward the restart threshold turned ~100 ms
            of bad reads into a component restart that pre-empted the pump's own
            in-place reopen and spent two of three restarts per glitch.

        Any action decided here is queued and run by the monitor thread.
        """
        with self._lock:
            comp = self._components.get(name)
            if comp is None:
                return
            if not escalate or comp.check is not None:
                self._note_failure(comp, error)
                return
            self._mark_failed(comp, error)
            action = self._maybe_recover(comp)
            if action is not None:
                self._pending_actions.append(action)

    # ------------------------------------------------------------------ monitor thread
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="health-monitor")
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor, waiting for an in-flight check or restart to finish.

        After this returns no further recovery action runs: the loop checks
        the stop flag before every check and every action. The join is bounded
        (generously) so a hung action cannot hold shutdown for ever.
        """
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._interval + _STOP_JOIN_GRACE_S)
            if self._thread.is_alive():
                log.warning("health monitor did not stop: a check or restart is still running")

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                due = [
                    c for c in self._components.values()
                    if c.check is not None and now - c.last_checked >= c.check_interval_s
                ]
            for comp in due:
                if self._stop.is_set():
                    return
                self._run_check(comp)
            self._run_pending_actions()
            if self._stop.is_set():
                return
            if self._watchdog_ping is not None and not self._critical_component_down():
                self._watchdog_ping()
            self._stop.wait(self._interval)

    def _run_pending_actions(self) -> None:
        """Run recovery actions queued by push reports, on this (monitor) thread."""
        while not self._stop.is_set():
            with self._lock:
                if not self._pending_actions:
                    return
                action = self._pending_actions.popleft()
            action()

    def _run_check(self, comp: _Component) -> None:
        comp.last_checked = time.monotonic()
        try:
            healthy, detail = comp.check()  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a failure
            healthy, detail = False, f"check raised: {exc}"
        action = None
        with self._lock:
            if healthy:
                self._mark_ok(comp, detail)
            else:
                self._mark_failed(comp, detail)
                action = self._maybe_recover(comp)
        # Outside the lock. A restart can take seconds (joining a thread, reopening
        # a camera); held under the lock it blocked /api/health, every component's
        # own report_failure, and -- through _critical_component_down -- the
        # systemd watchdog ping, for that whole time.
        #
        # Not once stop() has been called: a restart started during shutdown
        # would bring back a service that shutdown is in the middle of stopping.
        if action is not None and not self._stop.is_set():
            action()

    # ------------------------------------------------------------------ state transitions
    def _mark_ok(self, comp: _Component, detail: str) -> None:
        if comp.state != OK and comp.consecutive_failures > 0:
            log.info("%s recovered (%s)", comp.name, detail or "check passing again")
        comp.state = OK
        comp.detail = detail
        comp.last_ok = time.monotonic()
        comp.consecutive_failures = 0
        comp.last_resort_fired = False
        comp.watchdog_released = False  # a new outage withholds the ping again

    def _note_failure(self, comp: _Component, error: str) -> None:
        """Record an informational (push) failure without touching the
        consecutive-failure count that drives restarts and DOWN."""
        comp.last_failure = time.monotonic()
        comp.detail = error
        if comp.state == OK:
            log.warning("%s reported a failure: %s", comp.name, error)
            comp.state = DEGRADED

    def _mark_failed(self, comp: _Component, error: str) -> None:
        comp.consecutive_failures += 1
        comp.last_failure = time.monotonic()
        comp.detail = error
        if comp.consecutive_failures == 1:
            log.warning("%s failing: %s", comp.name, error)
        threshold_passed = comp.consecutive_failures >= _FAILURES_BEFORE_RESTART
        comp.state = DOWN if (threshold_passed and comp.critical) else DEGRADED

    def _maybe_recover(self, comp: _Component):
        """Decide on a restart (or escalation) once the failure threshold is crossed.

        Called under the lock, so the bookkeeping is consistent; returns the
        action to run AFTER the lock is released, or None.
        """
        if comp.consecutive_failures < _FAILURES_BEFORE_RESTART:
            return None

        now = time.monotonic()
        recent = [t for t in comp.restart_times if now - t <= comp.restart_window_s]
        if comp.restart is not None and len(recent) < comp.max_restarts:
            comp.restart_times.append(now)
            comp.total_restarts += 1
            log.warning(
                "restarting %s (attempt %d/%d in window): %s",
                comp.name, len(recent) + 1, comp.max_restarts, comp.detail,
            )
            restart = comp.restart

            def _restart() -> None:
                try:
                    restart()
                except Exception as exc:  # noqa: BLE001
                    log.error("restart of %s raised: %s", comp.name, exc)
                    return
                # Don't mark OK here — the next check passing is what proves recovery.
                with self._lock:
                    comp.consecutive_failures = _FAILURES_BEFORE_RESTART - 1

            return _restart

        if comp.critical and not comp.last_resort_fired and self._on_last_resort is not None:
            comp.last_resort_fired = True
            log.error("%s is down and its restart budget is spent — escalating to last resort", comp.name)
            on_last_resort = self._on_last_resort

            def _escalate() -> None:
                try:
                    on_last_resort(comp.name)
                except Exception as exc:  # noqa: BLE001
                    log.error("last-resort handler raised: %s", exc)

            return _escalate
        return None

    def _critical_component_down(self) -> bool:
        with self._lock:
            return any(
                c.critical and c.state == DOWN and not c.watchdog_released
                for c in self._components.values()
            )

    def release_watchdog_hold(self, name: str) -> None:
        """Keep pinging the watchdog through `name`'s current outage.

        Withholding the ping is how a DOWN critical component gets a process
        restart from systemd. Once the last-resort policy has decided that a
        restart will not help and the device should stay up degraded (reboot
        suppressed, or a process restart already tried for this outage), the
        hold must go: otherwise systemd kills the process every WatchdogSec for
        a fault no restart fixes, and each restart drops the device's reports
        and any explicit stop. The hold re-arms when the component recovers.
        """
        with self._lock:
            comp = self._components.get(name)
            if comp is not None and not comp.watchdog_released:
                comp.watchdog_released = True
                log.warning("%s stays down; keeping the watchdog fed and reporting degraded", name)

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
