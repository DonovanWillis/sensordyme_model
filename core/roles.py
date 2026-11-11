"""Deployment roles and their safety floors.

A device declares ONE role (`role:` in config.yaml). A role says which
safeguards must keep running while the device reports that it is protecting,
and which changes may never weaken them. Roles are data: adding one never
touches the platform code, and roles other than driver_safety carry none of
its restrictions.

What a role can require:

  mandatory_modules     at least one of these must be the running module
                        whenever the device reports "protecting"
  allowed_presets       the only presets that may be selected (None = any)
  mandatory_alert_types event types that must stay alertable and loud
  critical_types        event types that must use the critical alert category
  parameter_floors      module settings that may only move in the safe direction
                        (e.g. drowsy_seconds may be lowered, never raised past
                        its floor)
  requires_audible_sink at least one audible actuator (buzzer / sound) configured
  additive_custom_rules custom rules run BESIDE the mandatory module instead of
                        replacing it
  required_capabilities perception/alert capabilities (core/monitoring_state.py)
                        whose loss makes the device FAILED rather than DEGRADED

Enforcement lives at every channel that can change behaviour: config load
(core/config_schema.py), preset restore (main.py), preset switches and custom
rules from the dashboard, the uplink and the backend (services/runtime.py,
cloud/app.py). Each refusal carries a human-readable reason.

The only way to stop a protecting device is the explicit, confirmed stop
command, which changes the reported state to "stopped" (services/runtime.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: The role a config without `role:` gets. It imposes nothing, which keeps
#: every existing non-driver deployment and development config working.
DEFAULT_ROLE = "general"


@dataclass(frozen=True)
class ParameterFloor:
    """A module setting that may only move in the safe direction."""

    module: str
    key: str
    max: Optional[float] = None  # the value may be lowered, never raised past this
    min: Optional[float] = None  # the value may be raised, never lowered past this

    def violation(self, value) -> Optional[str]:
        """A reason when `value` breaks the floor, else None."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"`{self.module}.{self.key}` must be a number, got {value!r}"
        if self.max is not None and value > self.max:
            return f"`{self.module}.{self.key}` {value} exceeds the safety floor of {self.max}"
        if self.min is not None and value < self.min:
            return f"`{self.module}.{self.key}` {value} is below the safety floor of {self.min}"
        return None


@dataclass(frozen=True)
class Role:
    """What a deployment role requires. Empty requirements = no floor."""

    name: str
    title: str
    mandatory_modules: frozenset[str] = frozenset()
    allowed_presets: Optional[frozenset[str]] = None
    mandatory_alert_types: frozenset[str] = frozenset()
    critical_types: frozenset[str] = frozenset()
    parameter_floors: tuple[ParameterFloor, ...] = ()
    requires_audible_sink: bool = False
    additive_custom_rules: bool = False
    #: The alert categories that must keep repeating (bounded) for this role.
    repeating_categories: frozenset[str] = field(default_factory=frozenset)
    #: Capabilities without which this role cannot do its job at all: losing one
    #: is FAILED. Anything else an active policy needs is DEGRADED when lost.
    required_capabilities: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_floor(self) -> bool:
        return bool(self.mandatory_modules)

    def describe(self) -> dict:
        """What the app and backend are told about this device's role."""
        return {
            "name": self.name,
            "title": self.title,
            "mandatory_modules": sorted(self.mandatory_modules),
            "allowed_presets": None if self.allowed_presets is None else sorted(self.allowed_presets),
            "mandatory_alert_types": sorted(self.mandatory_alert_types),
            "additive_custom_rules": self.additive_custom_rules,
            "required_capabilities": sorted(self.required_capabilities),
        }


#: Every role needs frames and a running inference loop.
_VISION = frozenset({"camera", "inference"})

_DRIVER_FLOORS = tuple(
    ParameterFloor("driver_attention", key, max=limit)
    for key, limit in (
        # Each cap is today's shipped default (config.yaml), so no threshold
        # changes; presets may make them stricter (lower), never looser.
        ("drowsy_seconds", 2.0),
        ("distraction_seconds", 3.0),
        ("eyes_down_seconds", 2.5),
        ("eyes_up_seconds", 2.5),
        ("face_lost_seconds", 2.5),
        ("face_lost_with_body_seconds", 8.0),
        ("phone_use_seconds", 2.0),
        ("rearview_tolerance_seconds", 5.0),
    )
)

ROLES: dict[str, Role] = {
    role.name: role for role in (
        Role("general", "General purpose"),
        Role(
            "driver_safety", "Driver safety",
            mandatory_modules=frozenset({"driver_attention"}),
            allowed_presets=frozenset({"driver_safety"}),
            mandatory_alert_types=frozenset({
                "drowsiness", "attention_lost", "distraction", "eyes_down", "eyes_up",
                "phone_use", "high_risk",
            }),
            critical_types=frozenset({"drowsiness", "attention_lost"}),
            parameter_floors=_DRIVER_FLOORS,
            requires_audible_sink=True,
            additive_custom_rules=True,
            repeating_categories=frozenset({"critical", "warning"}),
            # Face landmarks carry drowsiness, distraction and attention-lost;
            # without them, or without a way to sound an alert in the cabin,
            # the device cannot protect a driver. Losing object detection only
            # loses phone/eating detection: DEGRADED.
            required_capabilities=_VISION | {"face", "audible_alert"},
        ),
        # Product roles with no preset floor defined yet (none inherits driver
        # rules). Their health requirements are the minimum each one's purpose
        # implies; refine them when each role's preset is validated.
        Role("child_monitoring", "Child monitoring", required_capabilities=_VISION),
        Role("occupancy", "Occupancy monitoring", required_capabilities=_VISION | {"objects"}),
        Role("property", "Property monitoring", required_capabilities=_VISION | {"objects"}),
        Role("workplace", "Workplace safety", required_capabilities=_VISION),
    )
}

#: Sinks that make sound (core/alerts.py). Console is not audible in a car.
AUDIBLE_SINKS = frozenset({"sound", "gpio_buzzer"})


def role_named(name: Optional[str]) -> Role:
    """The role for `name` (default when None). Raises KeyError for unknown names."""
    return ROLES[name or DEFAULT_ROLE]


def role_for_config(config: dict) -> Role:
    """The role a validated config declares."""
    return role_named(config.get("role") if isinstance(config, dict) else None)


# ---------------------------------------------------------------------- checks
def config_violations(config: dict) -> list[str]:
    """Every way `config` weakens its role's floor (empty when it does not)."""
    name = config.get("role", DEFAULT_ROLE)
    if not isinstance(name, str) or name not in ROLES:
        return [f"`role:` must be one of {sorted(ROLES)}, got {name!r}"]
    role = ROLES[name]
    if not role.has_floor or config.get("profile", "pi") == "mac_max":
        # mac_max is a developer profile; config_schema warns that the role is
        # not enforced there rather than pretending it protects anyone.
        return []
    errors: list[str] = []
    module = config.get("module")
    if module not in role.mandatory_modules:
        errors.append(
            f"role '{role.name}' requires `module:` to be one of {sorted(role.mandatory_modules)}, "
            f"got {module!r}"
        )
    raw_modules = config.get("modules")
    modules_cfg: dict = raw_modules if isinstance(raw_modules, dict) else {}
    for floor in role.parameter_floors:
        section = modules_cfg.get(floor.module)
        if isinstance(section, dict) and floor.key in section:
            reason = floor.violation(section[floor.key])
            if reason:
                errors.append(f"role '{role.name}': {reason}")
    raw_alerts = config.get("alerts")
    alerts: dict = raw_alerts if isinstance(raw_alerts, dict) else {}
    errors += _alert_violations(role, alerts)
    return errors


def _alert_violations(role: Role, alerts: dict) -> list[str]:
    errors: list[str] = []
    sinks = alerts.get("sinks", ["console"])
    if role.requires_audible_sink and isinstance(sinks, list) and not AUDIBLE_SINKS & set(sinks):
        errors.append(
            f"role '{role.name}' requires an audible alert sink ({' or '.join(sorted(AUDIBLE_SINKS))}) "
            f"in `alerts.sinks:`, got {sinks!r}"
        )
    loud = set(alerts.get("loud_types") or []) | set(alerts.get("critical_types") or [])
    soft = set(alerts.get("soft_types") or [])
    for event_type in sorted(role.mandatory_alert_types):
        if event_type not in loud:
            errors.append(f"role '{role.name}' requires `{event_type}` in alerts.loud_types or critical_types")
        if event_type in soft:
            errors.append(f"role '{role.name}' forbids `{event_type}` in alerts.soft_types (it must stay loud)")
    critical = set(alerts.get("critical_types") or [])
    for event_type in sorted(role.critical_types - critical):
        errors.append(f"role '{role.name}' requires `{event_type}` in alerts.critical_types")
    categories = alerts.get("categories") or {}
    if isinstance(categories, dict):
        for name in sorted(role.repeating_categories):
            override = categories.get(name)
            if isinstance(override, dict) and (
                override.get("repeat_s", 1) == 0 or override.get("max_repeats", 1) == 0
            ):
                errors.append(f"role '{role.name}' requires the '{name}' alert category to keep repeating")
    return errors


def preset_violation(role: Role, name: str, presets: dict) -> Optional[str]:
    """Why selecting preset `name` ("preset" or "preset:variant") would weaken `role`, or None."""
    if not role.has_floor or name == "":
        return None
    from core.presets import PresetError, parse_settings, split_settings

    try:
        head = split_settings(name)[0]
    except PresetError as exc:
        return f"preset '{name}': {exc.reason}"
    base, _, variant_name = head.partition(":")
    if role.allowed_presets is not None and base not in role.allowed_presets:
        return (
            f"this device's role is '{role.name}': preset '{base}' would stop "
            f"{', '.join(sorted(role.mandatory_modules))}; allowed: {', '.join(sorted(role.allowed_presets))}"
        )
    preset = presets.get(base) or {}
    if preset.get("module") not in role.mandatory_modules:
        return f"preset '{base}' does not run the module role '{role.name}' requires"
    overrides = dict(((preset.get("variants") or {}).get(variant_name) or {}).get("overrides") or {})
    # Owner settings ("preset@key=value", core/presets.py) are checked exactly
    # like a variant's overrides: a floor cannot be loosened by either. Settings
    # that cannot be read are refused here too -- fail closed, never unchecked.
    try:
        overrides.update(parse_settings(name, presets))
    except PresetError as exc:
        return f"preset '{name}': {exc.reason}"
    for floor in role.parameter_floors:
        if floor.key in overrides:
            reason = floor.violation(overrides[floor.key])
            if reason:
                return f"preset '{name}': {reason}"
    return None
