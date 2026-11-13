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

M8.3 truth invariants (docs/STATE_MODEL.md), as pure functions any test can
reuse on a device sample (`report_violations`) or an app row
(`status_violations`), and checked here at every transition:

  stale_withholds         stale: no state, protection, capability, perception or command echo
  stale_definition        stale == link_stale or health_stale; health_stale == freshness "stale";
                          old verified health is never fresh, whatever `online` says
  active_matches_state    monitoring_active is exactly state in {ready, degraded}
  protecting_requires_monitoring  protecting only while ready/degraded and active
  stopped_coherent        protection stopped <=> monitoring_state stopped
  paused_not_analysing    stopped: no analysis capability available, no component ok
  pipeline_down_not_detecting  camera or inference impaired: no analysis capability available,
                          and never ready/degraded
  model_missing_not_available  a failed/unavailable component's capability is not available
  capability_lists_disjoint    a capability is in at most one list
  ack_before_success      a confirmed stop/resume was applied (echoed) with the requested state
  state_transition        consecutive user-facing states follow core/state_model.ALLOWED, and
                          stopped is entered / left only with an applied command or a restart
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Optional

from core import state_model

PROTECTING_STATES = frozenset({"ready", "degraded"})
#: Capabilities that are analysis results (services/runtime.py, cloud/app.py).
ANALYSIS_CAPABILITIES = frozenset({"face", "objects", "hands", "pose", "attention"})
COMPONENT_CAPABILITY = {"object_detector": "objects", "face_landmarks": "face",
                        "hand_landmarks": "hands", "pose": "pose"}


def _capability_lists(caps: dict) -> dict[str, list[str]]:
    impaired = [i.get("name") for i in caps.get("impaired") or [] if isinstance(i, dict)]
    return {"available": list(caps.get("available") or []), "impaired": impaired,
            "starting": list(caps.get("starting") or []), "paused": list(caps.get("paused") or [])}


def _monitoring_truth(state: Optional[str], protection: Optional[str], active: Optional[bool],
                      lists: Optional[dict[str, list[str]]], components: dict) -> list[tuple[str, str]]:
    """Invariants shared by the device's sample and the backend's status."""
    out: list[tuple[str, str]] = []
    if protection == "protecting" and (state not in PROTECTING_STATES or active is False):
        out.append(("protecting_requires_monitoring", f"protecting while {state} (active={active})"))
    if (protection == "stopped") != (state == "stopped") and "unknown" not in (state, protection):
        out.append(("stopped_coherent", f"protection={protection} but monitoring_state={state}"))
    if state == "stopped":
        if lists is not None and (ANALYSIS_CAPABILITIES | {"inference"}) & set(lists["available"]):
            analysing = sorted((ANALYSIS_CAPABILITIES | {"inference"}) & set(lists["available"]))
            out.append(("paused_not_analysing", f"stopped but {analysing} available"))
        running = sorted(n for n, c in components.items() if isinstance(c, dict) and c.get("state") == "ok")
        if running:
            out.append(("paused_not_analysing", f"stopped but components {running} ok"))
    if lists is not None:
        seen = collections.Counter(n for names in lists.values() for n in names)
        twice = sorted(n for n, k in seen.items() if k > 1)
        if twice:
            out.append(("capability_lists_disjoint", f"{twice} in more than one list"))
        down = [p for p in ("camera", "inference") if p in lists["impaired"]]
        if down:
            detecting = sorted(ANALYSIS_CAPABILITIES & set(lists["available"]))
            if detecting:
                out.append(("pipeline_down_not_detecting", f"{down} impaired but {detecting} available"))
            if state in PROTECTING_STATES:
                out.append(("pipeline_down_not_detecting", f"{state} while {down} impaired"))
        for name, comp in components.items():
            kind = COMPONENT_CAPABILITY.get(name)
            if kind and isinstance(comp, dict) and comp.get("state") in ("failed", "unavailable") \
                    and kind in lists["available"]:
                out.append(("model_missing_not_available", f"{name} {comp.get('state')} but {kind} available"))
    return out


def report_violations(health: dict) -> list[tuple[str, str]]:
    """Contradictions inside one device health sample (what the device sends)."""
    monitoring = health.get("monitoring") if isinstance(health.get("monitoring"), dict) else {}
    protection = health.get("protection") if isinstance(health.get("protection"), dict) else {}
    state = monitoring.get("state")
    caps = monitoring.get("capabilities")
    lists = None
    if isinstance(caps, dict):
        lists = {k: [] for k in ("available", "impaired", "starting", "paused")}
        for name, cap in caps.items():
            key = (cap or {}).get("state")
            if key in lists:
                lists[key].append(name)
    components = ((monitoring.get("perception") or {}).get("components") or {})
    out = _monitoring_truth(state, protection.get("state"), monitoring.get("monitoring"), lists, components)
    if protection.get("monitoring_state") not in (None, state):
        out.append(("protecting_requires_monitoring",
                    f"protection block evaluated {protection.get('monitoring_state')}, monitoring block {state}"))
    if health.get("monitoring_active") is True and state not in PROTECTING_STATES:
        out.append(("active_matches_state", f"monitoring_active while {state}"))
    return out


def status_violations(row: dict) -> list[tuple[str, str]]:
    """Contradictions in one `GET /v1/devices` row: what the app is told."""
    status = row.get("status")
    if not isinstance(status, dict):
        return []
    out: list[tuple[str, str]] = []
    state, prot, active = status.get("monitoring_state"), status.get("protection"), status.get("monitoring_active")
    freshness = status.get("health_freshness") if isinstance(status.get("health_freshness"), dict) else {}
    link_stale, health_stale = status.get("link_stale"), status.get("health_stale")
    if link_stale is not None and status.get("stale") != bool(link_stale or health_stale):
        out.append(("stale_definition", f"stale={status.get('stale')} link={link_stale} health={health_stale}"))
    if freshness and health_stale != (freshness.get("state") == "stale"):
        out.append(("stale_definition", f"health_stale={health_stale} freshness={freshness.get('state')}"))
    age, threshold = freshness.get("age_s"), freshness.get("threshold_s")
    if freshness.get("verified") and isinstance(age, (int, float)) and isinstance(threshold, (int, float)) \
            and age >= threshold + 0.05 and not status.get("stale"):
        # +0.05: the backend decides on the exact age (fresh iff age < threshold) and
        # reports age_s rounded to 0.1 s, so 29.96 s is fresh and reads "30.0" (M8.4 soak).
        out.append(("stale_definition", f"health {age}s old (>= {threshold}s) but not stale; online={row.get('online')}"))
    if row.get("online") and link_stale:
        out.append(("stale_definition", "online but link_stale"))
    if status.get("stale"):
        claims = {k: status.get(k) for k in ("capabilities", "perception", "policies", "actuators",
                                             "last_command_id") if status.get(k) is not None}
        if (state, prot, active) != ("unknown", "unknown", None) or claims or row.get("health") is not None \
                or (status.get("rules") or {}).get("active") is not None:
            held = "set" if row.get("health") else None
            out.append(("stale_withholds", (f"stale but state={state} protection={prot} active={active} "
                                            f"claims={sorted(claims)} health={held}")))
    if state == "unknown" and active is not None:
        out.append(("active_matches_state", f"unknown but active={active}"))
    if state not in (None, "unknown") and active != (state in PROTECTING_STATES):
        out.append(("active_matches_state", f"{state} but active={active}"))
    caps = status.get("capabilities")
    lists = _capability_lists(caps) if isinstance(caps, dict) else None
    components = ((status.get("perception") or {}).get("components") or {})
    out += _monitoring_truth(state, prot, active, lists, components)
    command = status.get("protection_command")
    if isinstance(command, dict) and command.get("outcome") == "confirmed":
        applied = command.get("reported_state")
        wanted = applied == "stopped" if command.get("action") == "stop" else applied not in (None, "stopped")
        if command.get("applied_at") is None or not wanted:
            out.append(("ack_before_success", (f"{command.get('action')} confirmed with applied_at="
                                               f"{command.get('applied_at')} reported_state={applied}")))
    return out


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
    #: M8.3: the user-facing state (core/state_model.py) at every backend check,
    #: de-duplicated consecutively, and the evidence counters its guarded edges need.
    user_states: list[str] = field(default_factory=list)
    stops_applied: int = 0
    resumes_applied: int = 0
    boots: int = 0
    _evidence_at_last_state: tuple[int, int, int] = (0, 0, 0)

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
        self.boots += 1
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
                if runtime._stopped is not None:
                    self.stops_applied += 1
                else:
                    self.resumes_applied += 1

        def synced():
            sync()
            self.check_device(device, "sync")
            # M8.3: the sample the device actually sent must not contradict itself.
            sample = getattr(runtime.uplink, "_health", None) if runtime.uplink is not None else None
            if isinstance(sample, dict):
                for invariant, detail in report_violations(sample):
                    self._fail(invariant, "device/sample", detail)
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
        # `online` is link-only (M8.1): a polling device with stale health is online and stale.
        link_up = row["online"] and status.get("link_stale", True)
        healthy = (state, prot, active) != ("unknown", "unknown", None) or link_up or row["health"] is not None
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
        # M8.3: the whole row against the truth invariants, and the state model's edges.
        for invariant, detail in status_violations(row):
            self._fail(invariant, "app", detail)
        self.observe_user_state(state_model.derive(row, self.stack.clock.wall()).state)

    def observe_user_state(self, current: str) -> None:
        """Record one user-facing state; check the edge from the previous one."""
        evidence = (self.stops_applied, self.resumes_applied, self.boots)
        if self.user_states and self.user_states[-1] != current:
            stops, resumes, boots = (now - then for now, then in zip(evidence, self._evidence_at_last_state, strict=True))
            problem = state_model.transition_problem(self.user_states[-1], current, stop_applied=stops > 0,
                                                     resume_applied=resumes > 0, restarted=boots > 0)
            if problem:
                self._fail("state_transition", "app", problem)
        if not self.user_states or self.user_states[-1] != current:
            self.user_states.append(current)
            self._evidence_at_last_state = evidence

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
                "transitions": len(self.transitions), "user_states": len(self.user_states),
                "violations": len(self.violations),
                "max_queue": self.max_queue}
