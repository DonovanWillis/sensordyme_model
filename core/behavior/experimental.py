"""Owner decision A (M8.4): experimental alert families are observations, not alerts.

``high_risk``, ``elevated_risk`` (the RiskFusion composite score) and
``erratic_hand_movement`` (MediaPipe Hands trajectories) have no in-cabin
evidence behind them. They keep being computed and logged -- they are research
signals -- but by default they:

  * never actuate an in-cabin output (buzzer, speaker, visual alert flash):
    core/alerts.AlertManager refuses them before any sink is commanded;
  * are tagged severity ``log`` on the device (services/runtime.py) and stored
    as ``log`` by the backend, which never pushes or texts them (cloud/app.py);
  * carry ``experimental: true`` on the backend's event rows, so the app labels
    them "Experimental observation", never as an alert.

Nothing in config, presets, variants, owner settings, migration or a config
reset re-enables them: the refusal is in code and reads ONE explicit key,

    alerts:
      experimental_alerts_audible: true   # owner opt-in; anything else = false

which must be the boolean ``true`` (a string "true", 1 or "yes" is not an
opt-in). A deployed config whose ``alerts.loud_types``/``soft_types`` still
list these types (every unit configured before M8.4; scripts/migrate_config.py
never removes list members) is therefore safe without editing it: the lists are
read through ``gate_alert_lists`` and the policy refuses the types anyway.

Standard library only: the backend imports this module.
"""

from __future__ import annotations

from typing import Any

#: The experimental alert families (owner decision A, 2026-11-13).
EXPERIMENTAL_ALERT_TYPES: frozenset[str] = frozenset({"high_risk", "elevated_risk", "erratic_hand_movement"})

#: The one config key (under ``alerts:``) that makes them audible again.
OPT_IN_KEY = "experimental_alerts_audible"

#: How the app must present one of these events.
EXPERIMENTAL_LABEL = "Experimental observation"

#: Prefix of the event's contributing-signals text, so a human reading the log,
#: the dashboard or a notification-less timeline row sees it too.
EXPERIMENTAL_DETAIL_PREFIX = "EXPERIMENTAL observation (unvalidated, not an alert)"

_ALERT_LIST_KEYS = ("loud_types", "soft_types", "critical_types")


def is_experimental(event_type: object) -> bool:
    return isinstance(event_type, str) and event_type in EXPERIMENTAL_ALERT_TYPES


def opted_in(alert_cfg: Any) -> bool:
    """True only for an explicit boolean ``true`` under ``alerts.experimental_alerts_audible``."""
    return isinstance(alert_cfg, dict) and alert_cfg.get(OPT_IN_KEY) is True


def gate_alert_lists(alert_cfg: Any) -> tuple[dict, list[str]]:
    """``alert_cfg`` with the experimental types removed from its loud/soft/critical
    lists unless the owner opted in, plus the (sorted, unique) types removed.

    Never mutates its argument. A non-dict comes back as an empty dict.
    """
    if not isinstance(alert_cfg, dict):
        return {}, []
    if opted_in(alert_cfg):
        return dict(alert_cfg), []
    out = dict(alert_cfg)
    removed: set[str] = set()
    for key in _ALERT_LIST_KEYS:
        value = alert_cfg.get(key)
        if isinstance(value, list):
            kept = [t for t in value if not is_experimental(t)]
            removed.update(t for t in value if is_experimental(t))
            out[key] = kept
    return out, sorted(removed)


def observation_detail(detail: str) -> str:
    """The contributing-signals text for an experimental observation."""
    return f"{EXPERIMENTAL_DETAIL_PREFIX}: {detail}" if detail else EXPERIMENTAL_DETAIL_PREFIX


__all__ = [
    "EXPERIMENTAL_ALERT_TYPES",
    "EXPERIMENTAL_DETAIL_PREFIX",
    "EXPERIMENTAL_LABEL",
    "OPT_IN_KEY",
    "gate_alert_lists",
    "is_experimental",
    "observation_detail",
    "opted_in",
]
