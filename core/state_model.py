"""The user-facing state model (docs/STATE_MODEL.md), derived from the backend contract.

One device is in exactly ONE of these states at a time, decided from the fields
of `GET /v1/devices` (docs/APP_STATUS_CONTRACT.md) and nothing else. This is
not a second status system: every input is a contract field the device itself
caused, and the app's own mapping (ios/Sources/InstrumentState.swift) is the
same table. Pure: no I/O, no clock except the `now` passed in.

    state          when (first rule that applies wins)
    offline        `online` is false: no poll in the live window
    stale          online, but `status.stale` (the health evidence is old)
    unknown        live, but no monitoring state (no status, or `unknown`)
    resuming       stopped with a resume not yet applied, or starting right
                   after a resume the device confirmed
    stopped        `protection` / `monitoring_state` stopped
    faulted        `monitoring_state` failed
    activation_pending  `monitoring_state` starting (boot, mode switch, calibration)
    degraded       `monitoring_state` degraded
    ready          `monitoring_state` ready

`protecting` (the "Active" claim) is true only in ready / degraded with
`protection == "protecting"`; None whenever the state cannot be known.
Capabilities have their own state (`capability_state`), where `paused` lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

OFFLINE = "offline"
STALE = "stale"
UNKNOWN = "unknown"
RESUMING = "resuming"
STOPPED = "stopped"
FAULTED = "faulted"
ACTIVATION_PENDING = "activation_pending"
DEGRADED = "degraded"
READY = "ready"

STATES = (OFFLINE, STALE, UNKNOWN, RESUMING, STOPPED, FAULTED, ACTIVATION_PENDING, DEGRADED, READY)
#: States in which nothing current is known about monitoring.
NOT_LIVE = frozenset({OFFLINE, STALE, UNKNOWN})
#: States in which the device may be watching (monitoring_active true).
WATCHING = frozenset({READY, DEGRADED})

#: How long after a confirmed resume `starting` reads as resuming rather than
#: activation (the device's 30 s start-up grace, plus one link window).
RESUME_SETTLE_S = 60.0

#: Capability vocabulary of the contract (`capabilities.*` lists).
CAPABILITY_STATES = ("available", "starting", "paused", "impaired", "not_reported")


@dataclass(frozen=True)
class DerivedState:
    state: str
    #: True only when the device says every required safeguard runs on live
    #: evidence; None when that cannot be known.
    protecting: Optional[bool]
    #: The contract fields that justify the state, by name.
    evidence: dict[str, Any] = field(default_factory=dict)


def derive(row: dict, now: Optional[float] = None) -> DerivedState:
    """The one user-facing state of a `GET /v1/devices` row."""
    status = row.get("status") if isinstance(row.get("status"), dict) else None
    evidence: dict[str, Any] = {"online": row.get("online")}
    if not row.get("online"):
        if status is not None:
            evidence["link_stale"] = status.get("link_stale")
        return DerivedState(OFFLINE, None, evidence)
    if status is None:
        return DerivedState(UNKNOWN, None, {**evidence, "status": None})
    evidence.update(stale=status.get("stale"), health_stale=status.get("health_stale"))
    if status.get("stale"):
        return DerivedState(STALE, None, evidence)
    state, protection = status.get("monitoring_state"), status.get("protection")
    evidence.update(monitoring_state=state, protection=protection)
    command = status.get("protection_command") if isinstance(status.get("protection_command"), dict) else {}
    if command:
        evidence["protection_command"] = {k: command.get(k) for k in ("action", "outcome", "id")}
    if state in (None, "unknown"):
        return DerivedState(UNKNOWN, None, evidence)
    resume_unapplied = command.get("action") == "resume" and command.get("outcome") in ("pending", "timed_out")
    if state == "stopped" or protection == "stopped":
        return DerivedState(RESUMING if resume_unapplied else STOPPED, False, evidence)
    if state == "starting":
        applied = command.get("applied_at")
        just_resumed = (command.get("action") == "resume" and command.get("outcome") == "confirmed"
                        and isinstance(applied, (int, float)) and now is not None
                        and now - applied <= RESUME_SETTLE_S)
        return DerivedState(RESUMING if just_resumed else ACTIVATION_PENDING, False, evidence)
    if state == "failed":
        return DerivedState(FAULTED, False, evidence)
    protecting = protection == "protecting"
    if state == "degraded":
        return DerivedState(DEGRADED, protecting, evidence)
    if state == "ready":
        return DerivedState(READY, protecting, evidence)
    return DerivedState(UNKNOWN, None, evidence)


def capability_state(status: Optional[dict], name: str) -> str:
    """One capability's state from `status.capabilities`; `not_reported` when
    stale or absent (never assumed available)."""
    caps = (status or {}).get("capabilities")
    if not isinstance(caps, dict):
        return "not_reported"
    if name in (caps.get("available") or []):
        return "available"
    if name in (caps.get("starting") or []):
        return "starting"
    if name in (caps.get("paused") or []):
        return "paused"
    if any(isinstance(i, dict) and i.get("name") == name for i in caps.get("impaired") or []):
        return "impaired"
    return "not_reported"


#: Allowed transitions between consecutive observations of one device. Losing
#: or regaining liveness is always allowed (a cable can be pulled at any time);
#: the constraints are on entering and leaving `stopped`, which only a person
#: (or a restart) can cause. `guarded` names the evidence a guarded edge needs.
ALLOWED: dict[str, frozenset[str]] = {
    OFFLINE: frozenset(STATES),
    STALE: frozenset(STATES),
    UNKNOWN: frozenset(STATES),
    READY: frozenset({OFFLINE, STALE, UNKNOWN, READY, DEGRADED, FAULTED, ACTIVATION_PENDING, STOPPED}),
    DEGRADED: frozenset({OFFLINE, STALE, UNKNOWN, READY, DEGRADED, FAULTED, ACTIVATION_PENDING, STOPPED}),
    FAULTED: frozenset({OFFLINE, STALE, UNKNOWN, READY, DEGRADED, FAULTED, ACTIVATION_PENDING, STOPPED}),
    ACTIVATION_PENDING: frozenset({OFFLINE, STALE, UNKNOWN, READY, DEGRADED, FAULTED, ACTIVATION_PENDING,
                                   STOPPED}),
    STOPPED: frozenset({OFFLINE, STALE, UNKNOWN, STOPPED, RESUMING, READY, DEGRADED, FAULTED,
                        ACTIVATION_PENDING}),
    RESUMING: frozenset({OFFLINE, STALE, UNKNOWN, STOPPED, RESUMING, READY, DEGRADED, FAULTED,
                         ACTIVATION_PENDING}),
}
#: Edges allowed only with evidence: into stopped needs an applied stop; out of
#: stopped into a running state needs an applied resume or a device restart.
GUARDED = {
    "enter_stopped": "a stop applied by the device (app command echoed, or the device's own dashboard)",
    "leave_stopped": "a resume applied by the device, or a restart (a new boot id)",
}


def transition_problem(previous: str, current: str, *, stop_applied: bool = False,
                       resume_applied: bool = False, restarted: bool = False) -> Optional[str]:
    """Why `previous -> current` is not allowed with this evidence, or None."""
    if current not in ALLOWED.get(previous, frozenset()):
        return f"{previous} -> {current} is not a transition of the state model"
    running = {READY, DEGRADED, FAULTED, ACTIVATION_PENDING}
    if previous in running and current == STOPPED and not stop_applied:
        return f"{previous} -> stopped without {GUARDED['enter_stopped']}"
    if previous in (STOPPED, RESUMING) and current in running and not (resume_applied or restarted):
        return f"{previous} -> {current} without {GUARDED['leave_stopped']}"
    return None
