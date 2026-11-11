"""The device's authoritative monitoring state: what it can actually watch for now.

"The process is running" is not "the device is monitoring". This module turns
observed facts -- capability health, the active policies and what each needs,
the role's mandatory capabilities, the alert outputs -- into ONE state with its
reasons. It is pure (no I/O, no clock): the runtime gathers the facts, this
decides, and the same inputs always give the same answer.

States, in precedence order (the first that applies wins):

  STOPPED    the owner explicitly stopped monitoring (a confirmed command)
  FAILED     a capability the device's role requires is impaired, or nothing
             any active policy watches for can be evaluated at all
  STARTING   boot (or a mode switch) is still bringing a needed capability up,
             and nothing has failed outright; bounded by the caller's grace
  DEGRADED   some policy, event or alert output cannot work, but every
             role-required capability is available and something is watched
  READY      every active policy can evaluate every event it declares, and
             every configured alert output is ready

Capabilities share the vocabulary of core/capabilities.py (`kind`): "objects",
"pose", "face", "hands", plus the runtime's own "camera", "inference",
"alert_output" (any ready alert output) and "audible_alert" (a ready output
that makes sound). A capability is AVAILABLE, IMPAIRED (with a reason), or
STARTING (not yet initialised; never a failure on its own).

Nothing here measures accuracy or confidence. "Available" means the component
is producing results now, not that the results are right.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MonitoringState(str, Enum):  # noqa: UP042 - str() of members is part of the wire format
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


AVAILABLE = "available"
IMPAIRED = "impaired"
STARTING = "starting"

#: Every policy needs frames and a running inference loop, whatever it watches.
BASE_REQUIREMENTS = frozenset({"camera", "inference"})


@dataclass(frozen=True)
class Capability:
    """One capability's state, with the reason when it is not available."""

    state: str = AVAILABLE
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.state == AVAILABLE

    @property
    def impaired(self) -> bool:
        return self.state == IMPAIRED

    def to_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason}


@dataclass(frozen=True)
class Policy:
    """An active monitoring policy (a preset's module, or one custom rule) and
    what each of its events needs. An event mapped to None cannot be evaluated
    for a reason that is not a capability (e.g. its zone is not drawn yet)."""

    id: str
    kind: str  # "preset" | "rule"
    name: str
    events: Mapping[str, frozenset[str]]
    blocked: Mapping[str, str] = field(default_factory=dict)  # event -> non-capability reason


@dataclass(frozen=True)
class PolicyHealth:
    id: str
    kind: str
    name: str
    can_evaluate: bool  # at least one of its events can be evaluated now
    operational: bool  # every one of its events can
    unavailable_events: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name,
            "can_evaluate": self.can_evaluate, "operational": self.operational,
            "unavailable_events": list(self.unavailable_events), "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Actuator:
    """One configured alert output, as far as software can tell.

    `ready` is software-level readiness (initialised, last command did not
    fail). It is never a claim that anything was physically heard or seen:
    `physically_verified` stays False unless real feedback instrumentation
    exists, which no SensorDyme output has today.
    """

    name: str
    kind: str  # buzzer | speaker | console | led | haptic | app
    audible: bool
    ready: bool
    detail: str = ""
    commands_ok: int = 0
    commands_failed: int = 0
    physically_verified: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "audible": self.audible, "ready": self.ready,
            "detail": self.detail, "commands_ok": self.commands_ok,
            "commands_failed": self.commands_failed, "physically_verified": self.physically_verified,
        }


@dataclass(frozen=True)
class Evaluation:
    state: MonitoringState
    reasons: tuple[str, ...]
    policies: tuple[PolicyHealth, ...]
    capabilities: Mapping[str, Capability]


def actuator_capabilities(actuators: Iterable[Actuator]) -> dict[str, Capability]:
    """`alert_output` and `audible_alert` from the configured outputs."""
    actuators = list(actuators)
    ready = [a for a in actuators if a.ready]
    out = {
        "alert_output": Capability() if ready else Capability(
            IMPAIRED, "no alert output is ready" if actuators else "no alert output is configured"),
    }
    if any(a.audible and a.ready for a in actuators):
        out["audible_alert"] = Capability()
    else:
        audible = [a for a in actuators if a.audible]
        detail = "; ".join(f"{a.name}: {a.detail}" for a in audible if a.detail)
        out["audible_alert"] = Capability(
            IMPAIRED,
            f"no audible alert output is ready ({detail})" if detail
            else "no audible alert output is configured",
        )
    return out


def _policy_health(policy: Policy, caps: Mapping[str, Capability]) -> PolicyHealth:
    unavailable: list[str] = []
    reasons: list[str] = []
    for event, needs in sorted(policy.events.items()):
        if event in policy.blocked:
            unavailable.append(event)
            reasons.append(f"{event}: {policy.blocked[event]}")
            continue
        missing = sorted(n for n in needs if caps.get(n, Capability(IMPAIRED, "not provided")).impaired)
        if missing:
            unavailable.append(event)
            reasons.append(f"{event}: needs {', '.join(missing)}")
    total = len(policy.events)
    return PolicyHealth(
        id=policy.id, kind=policy.kind, name=policy.name,
        can_evaluate=total > len(unavailable), operational=not unavailable,
        unavailable_events=tuple(unavailable), reasons=tuple(reasons),
    )


def evaluate(
    *,
    stopped: Optional[str],
    capabilities: Mapping[str, Capability],
    policies: Iterable[Policy],
    role_required: Iterable[str] = (),
    actuators: Iterable[Actuator] = (),
) -> Evaluation:
    """The device's monitoring state from observed facts. See the module docstring.

    `stopped` is the reason monitoring was explicitly stopped, or None.
    """
    caps = dict(capabilities)
    actuators = list(actuators)
    caps.update(actuator_capabilities(actuators))
    policy_health = tuple(_policy_health(p, caps) for p in policies)

    if stopped is not None:
        return Evaluation(MonitoringState.STOPPED, (stopped,), policy_health, caps)

    def _why(name: str) -> str:
        cap = caps.get(name, Capability(IMPAIRED, "not provided by this device"))
        return f"{name}: {cap.reason}" if cap.reason else name

    required = sorted(set(role_required))
    failed = [_why(n) for n in required if caps.get(n, Capability(IMPAIRED)).impaired]
    if failed:
        return Evaluation(
            MonitoringState.FAILED, tuple(f"required capability unavailable — {r}" for r in failed),
            policy_health, caps,
        )
    if not policy_health or not any(p.can_evaluate for p in policy_health):
        why = [r for p in policy_health for r in p.reasons] or ["no monitoring policy is active"]
        return Evaluation(MonitoringState.FAILED, ("nothing can be monitored", *why), policy_health, caps)

    needed = set(required) | {n for p in policies for needs in p.events.values() for n in needs}
    if any(caps.get(n, Capability()).state == STARTING for n in needed):
        pending = sorted(n for n in needed if caps.get(n, Capability()).state == STARTING)
        return Evaluation(MonitoringState.STARTING, tuple(f"{n}: starting" for n in pending),
                          policy_health, caps)

    reasons = [f"{p.name}: {r}" for p in policy_health for r in p.reasons]
    reasons += [f"alert output {a.name} unavailable: {a.detail}" for a in actuators if not a.ready]
    if reasons:
        return Evaluation(MonitoringState.DEGRADED, tuple(reasons), policy_health, caps)
    return Evaluation(MonitoringState.READY, (), policy_health, caps)
