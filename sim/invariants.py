"""Product invariants, checked at every transition of a virtual device run.

    with SimStack(tmp_path) as stack:
        monitor = InvariantMonitor(stack)        # watches every device stack.boot()s
        device = stack.boot(scenario)
        stack.claim()
        device.advance(60.0)
        monitor.assert_clean()

The monitor hooks the device's own seams (no production code changes): after
every monitoring check and every command application it checks what the DEVICE
says, and after every uplink cycle it checks what the APP sees (GET /v1/devices).
End-state assertions can miss a state that was wrong for one cycle; this cannot.

Invariants (docs/VIRTUAL_DEVICE.md, "Reliability invariants"):

  failed_not_protecting   a FAILED device never reports protecting / monitoring_active
  stopped_not_ready       a stopped device never reports ready or protecting
  stale_not_healthy       a stale device is unknown, offline, health withheld
  rejected_rule_inactive  a rule set the device refused is never the one running
  no_duplicate_effect     one command id = one effect; one alert = one stored event
  driver_floor            a driver_safety device that says protecting runs driver_attention,
                          and custom rules never replace it
  queue_bounded           the uplink backlog never exceeds its capacity
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Optional

PROTECTING_STATES = frozenset({"ready", "degraded"})


@dataclass
class Violation:
    invariant: str
    sim_time: float
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.invariant}] t={self.sim_time:.2f}s {self.where}: {self.detail}"


@dataclass
class InvariantMonitor:
    stack: Any
    #: Check the backend view after every uplink cycle (one GET per cycle).
    check_backend: bool = True
    violations: list[Violation] = field(default_factory=list)
    #: Device-side checks run / backend-side checks run (proves the monitor ran).
    device_checks: int = 0
    backend_checks: int = 0
    #: Every (state, protection) the device reported, de-duplicated consecutively.
    transitions: list[tuple[str, str]] = field(default_factory=list)
    #: Effects per protection command id: how many times each stop/resume took effect.
    protection_effects: collections.Counter = field(default_factory=collections.Counter)
    #: The largest uplink backlog seen.
    max_queue: int = 0

    def __post_init__(self) -> None:
        boot = self.stack.boot

        def watched_boot(*args, **kwargs):
            device = boot(*args, **kwargs)
            self.attach(device)
            return device

        self.stack.boot = watched_boot
        if getattr(self.stack, "device", None) is not None:
            self.attach(self.stack.device)

    # ------------------------------------------------------------------ hooks
    def attach(self, device) -> None:
        runtime = device.runtime
        check, apply, sync = runtime._monitoring_check, runtime._apply_pending_commands, device.sync
        apply_protection = runtime._apply_protection

        def monitoring_check():
            result = check()
            self.check_device(device, "monitoring_check")
            return result

        def apply_pending():
            try:
                return apply()
            finally:
                self.check_device(device, "commands")

        def protection(request, source):
            before = runtime._stopped is not None
            apply_protection(request, source)
            if (runtime._stopped is not None) != before:
                self.protection_effects[str((request or {}).get("id"))] += 1

        def synced():
            sync()
            self.check_device(device, "sync")
            if self.check_backend and getattr(self.stack, "device_id", None) is not None:
                self.check_app(device)

        runtime._monitoring_check = monitoring_check
        runtime._apply_pending_commands = apply_pending
        runtime._apply_protection = protection
        device.sync = synced

    # ------------------------------------------------------------------ device side
    def _now(self) -> float:
        return float(self.stack.clock.monotonic())

    def _fail(self, invariant: str, where: str, detail: str) -> None:
        self.violations.append(Violation(invariant, self._now(), where, detail))

    def check_device(self, device, where: str) -> None:
        runtime = device.runtime
        self.device_checks += 1
        protection = runtime.protection_state()
        state, prot = protection["monitoring_state"], protection["state"]
        if not self.transitions or self.transitions[-1] != (state, prot):
            self.transitions.append((state, prot))
        if prot == "protecting" and state not in PROTECTING_STATES:
            self._fail("failed_not_protecting", f"device/{where}", f"protecting while {state}")
        if runtime._stopped is not None and (prot != "stopped" or state != "stopped"):
            self._fail("stopped_not_ready", f"device/{where}", f"stopped but reports {state}/{prot}")
        role = runtime.role
        if role.has_floor and prot == "protecting":
            if runtime.module_name not in role.mandatory_modules:
                self._fail("driver_floor", f"device/{where}",
                           f"protecting on role {role.name} while running {runtime.module_name}")
            if runtime.mode.value == "ai_custom":
                self._fail("driver_floor", f"device/{where}", "custom rules replaced the role's module")
        if runtime._custom_active():
            problems = runtime._rule_problems(runtime.monitoring.rules, runtime.module,
                                              runtime._preset_module or runtime.module_name)
            if problems:
                self._fail("rejected_rule_inactive", f"device/{where}",
                           f"running a rule set this device refuses: {problems}")
        uplink = runtime.uplink
        if uplink is not None:
            size, cap = len(uplink._queue), uplink._queue.maxlen
            self.max_queue = max(self.max_queue, size)
            if cap is not None and size > cap:
                self._fail("queue_bounded", f"device/{where}", f"backlog {size} > capacity {cap}")
        for command_id, count in self.protection_effects.items():
            if command_id != "None" and count > 1:
                self._fail("no_duplicate_effect", f"device/{where}",
                           f"protection command {command_id} took effect {count} times")

    # ------------------------------------------------------------------ app side
    def check_app(self, device) -> None:
        self.backend_checks += 1
        rows = [d for d in self.stack.devices() if d["id"] == self.stack.device_id]
        if not rows:
            return
        row = rows[0]
        status = row["status"]
        state, prot, active = status["monitoring_state"], status["protection"], status["monitoring_active"]
        if state == "failed" and (prot == "protecting" or active is True):
            self._fail("failed_not_protecting", "app", f"failed but protection={prot} active={active}")
        if prot == "stopped" and (state != "stopped" or active is not False):
            self._fail("stopped_not_ready", "app", f"stopped but state={state} active={active}")
        if state == "stopped" and prot == "protecting":
            self._fail("stopped_not_ready", "app", "state stopped but protection protecting")
        healthy = (state, prot, active) != ("unknown", "unknown", None) or row["online"] or row["health"] is not None
        if status["stale"] and healthy:
            self._fail("stale_not_healthy", "app",
                       f"stale but state={state} protection={prot} active={active} online={row['online']}")
        rejected = (status.get("rules") or {}).get("rejected")
        if rejected is not None and device.runtime._custom_active():
            running_version = device.runtime.monitoring.version
            if running_version and running_version == rejected.get("version"):
                self._fail("rejected_rule_inactive", "app", f"rejected version {running_version} is running")
        if status.get("role") == "driver_safety" and status.get("mode") == "ai_custom":
            self._fail("driver_floor", "app", "driver_safety device reports ai_custom (floor replaced)")

    def check_events(self) -> None:
        """One alert, one stored row: no (type, timestamp) appears twice."""
        if getattr(self.stack, "device_id", None) is None:
            return
        seen = collections.Counter((e["type"], e.get("timestamp")) for e in self.stack.events(limit=500))
        for key, count in seen.items():
            if count > 1:
                self._fail("no_duplicate_effect", "app/events", f"{key} stored {count} times")

    # ------------------------------------------------------------------ results
    def assert_clean(self, events: bool = True) -> None:
        if events:
            self.check_events()
        assert self.device_checks > 0, "the monitor never ran"
        assert not self.violations, "\n".join(str(v) for v in self.violations[:20])

    def summary(self) -> dict[str, Optional[int]]:
        return {"device_checks": self.device_checks, "backend_checks": self.backend_checks,
                "transitions": len(self.transitions), "violations": len(self.violations),
                "max_queue": self.max_queue}
