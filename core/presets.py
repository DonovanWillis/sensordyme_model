"""The mode registry, the names modes used to have, and the preset catalogue.

Deliberately dependency-free — yaml and the standard library, nothing else.
Both the device and the backend need to agree on what a preset name means, and
the backend runs on machines that have never had mediapipe or picamera2
installed. Putting this in main.py made it unreachable from cloud/app.py, and
the two ended up comparing names by exact string equality instead, which is
what let a renamed mode silently never converge.

A preset name has up to three parts:

    driver_safety                         the mode
    driver_safety:strict                  the mode, a named variant
    driver_safety:strict@drowsy_seconds=1.5,phone_use_seconds=1
                                          ... and per-device settings

The settings suffix is how a setting the owner chose in the app reaches the
device: it rides the existing `desired_preset` -> `set_preset` -> `active_preset`
delivery, so convergence, persistence across a restart (core/device_state.py)
and the role check all work on it unchanged. Settings are validated against the
preset's catalogue schema (`settings:` in presets.yaml) on BOTH sides — the
backend refuses an invalid one with a reason, the device refuses one it cannot
apply — and a role's parameter floors (core/roles.py) still win over any of them.

The catalogue half (`catalog_entries`, `device_availability`) describes each
preset for the app: what it needs, what it watches for and for how long, what
the owner may tune, and how far the models behind it have been validated. That
validation status is never written in presets.yaml: it is derived from
core/model_registry.py (the weakest status along the preset's capability chain)
so a preset can never claim more than its weakest model has earned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "presets.yaml"

#: Separates a preset name from its settings ("mode:variant@key=value,...").
SETTINGS_SEPARATOR = "@"

#: What this software can do with policies beyond what older firmware could.
#: The device reports these (services/runtime.py `_policy_state`); the backend
#: only sends a feature to a device that reported it, because an older device
#: would refuse or misread it for ever.
#:   preset_settings  a `@key=value` settings suffix on a preset name
#:   dwell_in_zone    per-subject dwell rules (`dwell_s` on a zone rule)
DEVICE_POLICY_FEATURES = ("dwell_in_zone", "preset_settings")

SETTING_TYPES = ("number", "bool", "enum")
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OPTION = re.compile(r"^[a-z0-9_]{1,64}$")

#: Capability kinds a preset may name (core/monitoring_state.py's vocabulary).
CAPABILITY_KINDS = frozenset({
    "camera", "inference", "face", "objects", "hands", "pose", "alert_output", "audible_alert",
})
_OUTPUT_KINDS = frozenset({"alert_output", "audible_alert"})
_TEMPORAL_KEYS = frozenset({"event", "description", "sustain_s", "window_s", "setting", "experimental", "config_key"})
_PATTERN_KEYS = frozenset({"tone", "events", "description"})

#: Availability codes for one preset on one device (docs/PRESET_LIBRARY.md).
CODE_ROLE_FORBIDS = "role_forbids_preset"
CODE_IMPAIRED = "capability_impaired"
CODE_UNSUPPORTED = "capability_unsupported"
CODE_STALE = "device_stale"
CODE_PLANNED = "planned"


class PresetError(ValueError):
    """A preset name (or its settings) this software cannot apply. `key` names
    the offending setting when there is one."""

    def __init__(self, reason: str, key: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.key = key


# ---------------------------------------------------------------------- loading
def _load(path: Union[str, Path, None]) -> dict:
    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_presets(path: Union[str, Path, None] = None) -> dict:
    """The modes on offer: name -> {title, module, alert_types?, variants?, ...}.

    Only presets a device can run. Planned catalogue entries live under
    `planned:` and are never returned here, so nothing that switches modes can
    be handed one.
    """
    return _load(path).get("presets", {}) or {}


def load_preset_aliases(path: Union[str, Path, None] = None) -> dict:
    """Superseded mode names -> their replacement."""
    return _load(path).get("aliases", {}) or {}


def load_planned(path: Union[str, Path, None] = None) -> dict:
    """Catalogue entries for product modes no device can run yet, with why."""
    return _load(path).get("planned", {}) or {}


def catalog_version(path: Union[str, Path, None] = None) -> str:
    return str(_load(path).get("catalog_version") or "")


# ---------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Setting:
    """One owner-tunable value, mapped 1:1 onto an existing module config key."""

    key: str
    title: str
    type: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    options: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key, "title": self.title, "type": self.type, "default": self.default,
            "min": self.min, "max": self.max, "step": self.step, "unit": self.unit or None,
            "options": list(self.options) if self.type == "enum" else None,
            "description": self.description or None,
        }

    def coerce(self, value: Any) -> Any:
        """`value` as this setting's type, or PresetError naming the key.

        Strict: a bool is not a number, "1.5" (a string) is not a number, and a
        number outside [min, max] or off the step grid is refused rather than
        clamped — the owner must see the value that will run.
        """
        if self.type == "bool":
            if not isinstance(value, bool):
                raise PresetError(f"setting '{self.key}' must be true or false, got {value!r}", self.key)
            return value
        if self.type == "enum":
            if not isinstance(value, str) or value not in self.options:
                raise PresetError(
                    f"setting '{self.key}' must be one of {', '.join(self.options)}, got {value!r}", self.key)
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PresetError(f"setting '{self.key}' must be a number, got {value!r}", self.key)
        if self.min is not None and value < self.min:
            raise PresetError(f"setting '{self.key}' {value} is below the minimum of {self.min}", self.key)
        if self.max is not None and value > self.max:
            raise PresetError(f"setting '{self.key}' {value} is above the maximum of {self.max}", self.key)
        if self.step:
            base = self.min if self.min is not None else 0.0
            steps = (value - base) / self.step
            if abs(steps - round(steps)) > 1e-6:
                raise PresetError(f"setting '{self.key}' must move in steps of {self.step}", self.key)
        return _number(value, integral=bool(self.step) and float(self.step).is_integer())

    def parse_text(self, text: str) -> Any:
        """A value written in a preset-name suffix, back to its type."""
        if self.type == "bool":
            if text not in ("true", "false"):
                raise PresetError(f"setting '{self.key}' must be true or false, got {text!r}", self.key)
            return text == "true"
        if self.type == "enum":
            return self.coerce(text)
        try:
            value = float(text)
        except ValueError:
            raise PresetError(f"setting '{self.key}' must be a number, got {text!r}", self.key) from None
        return self.coerce(value)


def _number(value: float, *, integral: bool) -> Union[int, float]:
    if integral and float(value).is_integer():
        return int(value)
    return float(value)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else format(float(value), ".6g")
    return str(value)


def _setting(raw: Any, where: str) -> Setting:
    if not isinstance(raw, dict):
        raise PresetError(f"{where}: a setting must be a mapping")
    key = str(raw.get("key") or "")
    kind = str(raw.get("type") or "")
    if not _KEY.match(key):
        raise PresetError(f"{where}: invalid setting key {key!r}")
    if kind not in SETTING_TYPES:
        raise PresetError(f"{where}.{key}: type must be one of {SETTING_TYPES}")
    options = tuple(str(o) for o in raw.get("options") or ())
    if kind == "enum" and (not options or not all(_OPTION.match(o) for o in options)):
        raise PresetError(f"{where}.{key}: an enum needs options of [a-z0-9_]")

    def bound(name: str) -> Optional[float]:
        value = raw.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PresetError(f"{where}.{key}: {name} must be a number")
        return float(value)

    setting = Setting(
        key=key, title=str(raw.get("title") or key), type=kind, default=raw.get("default"),
        min=bound("min"), max=bound("max"), step=bound("step"), unit=str(raw.get("unit") or ""),
        options=options, description=str(raw.get("description") or ""),
    )
    if kind == "number" and (setting.min is None or setting.max is None or setting.min > setting.max):
        raise PresetError(f"{where}.{key}: a number setting needs min <= max")
    setting.coerce(setting.default)  # the default itself must be a legal value
    return setting


def settings_schema(preset: Optional[dict]) -> dict[str, Setting]:
    """key -> Setting for one preset (empty when it defines none). Raises
    PresetError for a malformed schema."""
    schema: dict[str, Setting] = {}
    for raw in (preset or {}).get("settings") or []:
        setting = _setting(raw, "settings")
        if setting.key in schema:
            raise PresetError(f"settings: duplicate key {setting.key!r}")
        schema[setting.key] = setting
    return schema


def validate_settings(preset: dict, settings: Any) -> dict:
    """`settings` checked against `preset`'s schema: the cleaned values, or
    PresetError naming the first bad key (unknown keys included)."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise PresetError("settings must be an object of key -> value")
    schema = settings_schema(preset)
    cleaned = {}
    for key in sorted(settings):
        setting = schema.get(key)
        if setting is None:
            known = ", ".join(sorted(schema)) or "none"
            raise PresetError(f"'{key}' is not a setting of this mode (settings: {known})", str(key))
        cleaned[key] = setting.coerce(settings[key])
    return cleaned


# ---------------------------------------------------------------------- names
def split_settings(name: str) -> tuple[str, dict[str, str]]:
    """"mode:variant@a=1,b=true" -> ("mode:variant", {"a": "1", "b": "true"}).

    The values stay text: only the preset's schema knows their types. Raises
    PresetError for a suffix that is not key=value pairs.
    """
    base, sep, suffix = (name or "").partition(SETTINGS_SEPARATOR)
    if not sep:
        return base, {}
    pairs: dict[str, str] = {}
    for part in suffix.split(","):
        key, eq, value = part.partition("=")
        if not eq or not _KEY.match(key) or not value or key in pairs:
            raise PresetError(f"malformed settings in preset name {name!r}")
        pairs[key] = value
    return base, pairs


def encode_settings(name: str, settings: Optional[dict]) -> str:
    """The canonical name carrying `settings` (sorted keys; none -> bare name)."""
    base = (name or "").partition(SETTINGS_SEPARATOR)[0]
    if not settings:
        return base
    return base + SETTINGS_SEPARATOR + ",".join(f"{k}={_format(settings[k])}" for k in sorted(settings))


def parse_settings(name: str, presets: Optional[dict] = None) -> dict:
    """The typed settings a preset name carries (validated; PresetError if not)."""
    presets = load_presets() if presets is None else presets
    base, raw = split_settings(name)
    if not raw:
        return {}
    preset = presets.get(base.partition(":")[0])
    if preset is None:
        raise PresetError(f"unknown preset '{base}'")
    schema = settings_schema(preset)
    values = {}
    for key, text in raw.items():
        setting = schema.get(key)
        if setting is None:
            raise PresetError(f"'{key}' is not a setting of '{base}'", key)
        values[key] = setting.parse_text(text)
    return values


def resolve_preset(name: str, presets: Optional[dict] = None,
                   aliases: Optional[dict] = None) -> str:
    """Map a possibly-superseded preset name onto one that exists.

    Renaming a mode strands every device already holding the old name: an
    unknown preset is a device stuck in a mode it cannot switch out of, and the
    name can be anywhere — a deployed config.yaml, a `desired_preset` sitting in
    the backend, a phone that has not been updated.

    A variant suffix ("driver_distraction:teen_driver") is carried across, since
    a variant belongs to the preset rather than to the name it used to have —
    but only when the replacement actually defines it, because a variant that no
    longer exists would fail later, a long way from the rename that caused it.
    A settings suffix is carried across unchanged (it is validated when applied).
    """
    if not name:
        return name
    presets = load_presets() if presets is None else presets
    head, sep, settings = name.partition(SETTINGS_SEPARATOR)
    base, _, variant = head.partition(":")
    if base in presets:
        return name
    target = (load_preset_aliases() if aliases is None else aliases).get(base)
    if target is None or target not in presets:
        return name
    if variant and variant in (presets[target].get("variants") or {}):
        resolved = f"{target}:{variant}"
    else:
        resolved = target
    return f"{resolved}{sep}{settings}" if sep else resolved


def canonical_preset(name: str, presets: Optional[dict] = None) -> str:
    """`name` resolved, with its settings (if any) in canonical order and format.

    Falls back to the resolved name when the settings cannot be parsed, so a
    comparison never raises; applying such a name fails later, with a reason.
    """
    presets = load_presets() if presets is None else presets
    resolved = resolve_preset(name, presets)
    try:
        return encode_settings(resolved, parse_settings(resolved, presets))
    except PresetError:
        return resolved


def same_preset(a: Optional[str], b: Optional[str], presets: Optional[dict] = None) -> bool:
    """Whether two preset names refer to the same mode once aliases are applied.

    The comparison that convergence depends on. A device handed a superseded
    name applies — and correctly reports — the current one, so comparing the raw
    strings never matches: the backend concludes the command was never obeyed
    and re-sends it on every poll, fighting any mode the user picks locally.
    Settings are compared by value, not spelling ("1.50" and "1.5" are one value).
    """
    if not a or not b:
        return a == b
    presets = load_presets() if presets is None else presets
    return canonical_preset(a, presets) == canonical_preset(b, presets)


def preset_overrides(name: str, presets: Optional[dict] = None) -> tuple[str, dict, dict]:
    """(canonical name, preset, module overrides) for a name a device should run.

    Overrides are the variant's, with the name's settings on top. Raises
    PresetError for an unknown preset or variant and for any setting the
    preset's schema refuses — the device must refuse such a switch with the
    reason rather than run something nobody asked for. Role floors are checked
    separately (core/roles.py preset_violation), and they win.
    """
    presets = load_presets() if presets is None else presets
    resolved = resolve_preset(name, presets)
    head = resolved.partition(SETTINGS_SEPARATOR)[0]
    base, _, variant_name = head.partition(":")
    preset = presets.get(base)
    if preset is None:
        raise PresetError(f"unknown preset '{base}'")
    overrides: dict = {}
    if variant_name:
        variant = (preset.get("variants") or {}).get(variant_name)
        if variant is None:
            raise PresetError(f"preset '{base}' has no variant '{variant_name}'")
        overrides = dict(variant.get("overrides") or {})
    settings = parse_settings(resolved, presets)
    overrides.update(settings)
    return encode_settings(head, settings), preset, overrides


# ---------------------------------------------------------------------- catalogue
def _strings(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _capabilities(value: Any, where: str) -> list[str]:
    kinds = _strings(value)
    unknown = sorted(set(kinds) - CAPABILITY_KINDS)
    if unknown:
        raise PresetError(f"{where}: unknown capability kind(s) {unknown}")
    return kinds


def _validation(registry: Any, name: str, preset: dict, presets: dict) -> dict:
    """{status, models, evidence} from the registry: never from presets.yaml.

    Runnable presets use the registry's own chain (core/model_registry.py
    preset_view); a planned entry with a module uses that module's chain; one
    with no module has nothing to evaluate ("not_applicable").
    """
    from core import model_registry as mr

    module = preset.get("module")
    if not module or module not in registry.module_map:
        return {"status": "not_applicable", "validated": False, "models": [], "evidence": []}
    if name in presets:
        view = mr.preset_view(registry, name, presets)
    else:
        use = registry.module_map[module]
        view = mr.chain_for(registry, f"preset:{name}", ("camera", "inference", *use.uses), use.optional)
    weakest = view.weakest
    evidence = []
    for model_id in view.models:
        card = registry.card(model_id)
        if card is None:
            continue
        for item in card.evaluation.evidence:
            evidence.append({
                "model": model_id, "kind": item.kind, "report_id": item.report_id,
                "date": item.date, "granularity": item.granularity or None,
                "supports": item.level.value if item.level is not None and item.shipped_config else None,
                "shipped_config": item.shipped_config,
            })
    return {
        "status": weakest.value if weakest is not None else mr.EvaluationStatus.UNEVALUATED.value,
        "validated": view.validated,
        "models": list(view.models),
        "evidence": evidence,
    }


def _entry(registry: Any, name: str, preset: dict, presets: dict, *, planned: bool) -> dict:
    where = f"{'planned' if planned else 'presets'}.{name}"
    schema = settings_schema(preset)
    variants = preset.get("variants") or {}
    return {
        "id": name,
        "version": int(preset.get("version") or 1),
        "title": str(preset.get("title") or name),
        "description": str(preset.get("description") or ""),
        "product_mode": preset.get("product_mode"),
        "module": preset.get("module"),
        "availability": "planned" if planned else "available",
        "planned_reason": str(preset.get("reason") or "") if planned else None,
        "served_by": _strings(preset.get("served_by")) if planned else [],
        "variants": [{"id": vid, "title": str((v or {}).get("title") or vid)} for vid, v in variants.items()],
        "required_capabilities": _capabilities(preset.get("required_capabilities"), where),
        "optional_capabilities": _capabilities(preset.get("optional_capabilities"), where),
        "hardware": dict(preset.get("hardware") or {}),
        "alert_types": _strings(preset.get("alert_types")),
        "temporal_policies": [dict(p) for p in preset.get("temporal_policies") or [] if isinstance(p, dict)],
        "alert_patterns": [dict(p) for p in preset.get("alert_patterns") or [] if isinstance(p, dict)],
        "settings": [s.to_dict() for s in schema.values()],
        "compatibility": {
            "exclusive_group": "vision_module",
            "module": preset.get("module"),
            **dict(preset.get("compatibility") or {}),
        },
        "validation": {
            **_validation(registry, name, preset, presets),
            "references": _strings(preset.get("evidence")),
        },
        "limitations": _strings(preset.get("limitations")),
        # Monitoring nodes the preset runs (core/nodes.py), when it is built from them.
        "nodes": [str(n.get("node")) for n in preset.get("nodes") or [] if isinstance(n, dict) and n.get("node")],
    }


def catalog_entries(path: Union[str, Path, None] = None, registry: Any = None) -> list[dict]:
    """Every catalogue entry (runnable presets first, then planned), as the API
    serves them. Raises PresetError for a malformed catalogue."""
    if registry is None:
        from core.model_registry import load_registry

        registry = load_registry()
    presets = load_presets(path)
    entries = [_entry(registry, n, p or {}, presets, planned=False) for n, p in presets.items()]
    entries += [_entry(registry, n, p or {}, presets, planned=True) for n, p in load_planned(path).items()]
    return entries


def catalog_problems(path: Union[str, Path, None] = None, registry: Any = None) -> list[str]:
    """Everything wrong with the catalogue (empty = consistent). For tests/CI.

    Checks the claims presets.yaml makes against the sources that own the
    facts: the registry (which capabilities each module consumes) and every
    setting default against its own schema. Validation status is not checked
    because presets.yaml cannot state one.
    """
    if registry is None:
        from core.model_registry import load_registry

        registry = load_registry()
    problems: list[str] = []
    presets = load_presets(path)
    planned = load_planned(path)
    for name in set(presets) & set(planned):
        problems.append(f"{name}: is both a preset and a planned entry")
    for name, raw in [*presets.items(), *planned.items()]:
        preset = raw or {}
        try:
            entry = _entry(registry, name, preset, presets, planned=name in planned)
        except PresetError as exc:
            problems.append(f"{name}: {exc.reason}")
            continue
        if "validation_status" in preset or "status" in preset:
            problems.append(f"{name}: validation status is derived from the registry, never declared")
        module = preset.get("module")
        if name in presets or module:
            use = registry.module_map.get(module)
            if use is None:
                problems.append(f"{name}: module {module!r} is not described in core/model_registry.yaml")
                continue
            declared = set(entry["required_capabilities"]) | set(entry["optional_capabilities"])
            expected = {"camera", "inference", *use.uses, *use.optional}
            if declared - _OUTPUT_KINDS != expected:
                problems.append(
                    f"{name}: capabilities {sorted(declared - _OUTPUT_KINDS)} do not match what module "
                    f"{module} consumes per the registry {sorted(expected)}")
            if not {"camera", "inference"} <= set(entry["required_capabilities"]):
                problems.append(f"{name}: camera and inference must be required")
        overlap = set(entry["required_capabilities"]) & set(entry["optional_capabilities"])
        if overlap:
            problems.append(f"{name}: {sorted(overlap)} both required and optional")
        if name in planned and not entry["planned_reason"]:
            problems.append(f"{name}: a planned entry must say why it is not available")
        schema = {s["key"]: s for s in entry["settings"]}
        for policy in entry["temporal_policies"]:
            if not set(policy) <= _TEMPORAL_KEYS or "event" not in policy:
                problems.append(f"{name}: temporal policy {policy} has unknown keys or no event")
            setting = schema.get(policy.get("setting")) if policy.get("setting") else None
            if policy.get("setting") and setting is None:
                problems.append(f"{name}: temporal policy {policy['event']} names unknown setting")
            elif setting is not None and policy.get("sustain_s") != setting["default"]:
                problems.append(f"{name}: {policy['event']} sustain_s disagrees with {setting['key']}'s default")
        for pattern in entry["alert_patterns"]:
            if not set(pattern) <= _PATTERN_KEYS or pattern.get("tone") not in ("critical", "loud", "soft"):
                problems.append(f"{name}: alert pattern {pattern} is malformed")
        for variant in (preset.get("variants") or {}).values():
            for key, value in ((variant or {}).get("overrides") or {}).items():
                setting = settings_schema(preset).get(key)
                if setting is not None:
                    try:
                        setting.coerce(value)
                    except PresetError as exc:
                        problems.append(f"{name}: variant override {exc.reason}")
    return problems


# ---------------------------------------------------------------------- availability
def effective_settings(preset: dict, variant: Optional[str] = None,
                       settings: Optional[dict] = None) -> dict:
    """key -> the value that runs: default, then variant override, then settings."""
    schema = settings_schema(preset)
    overrides = ((preset.get("variants") or {}).get(variant) or {}).get("overrides") or {} if variant else {}
    values = {key: s.default for key, s in schema.items()}
    for key, value in {**overrides, **(settings or {})}.items():
        if key in values:
            values[key] = value
    return values


def device_availability(entry: dict, *, role: Any = None, capabilities: Optional[dict] = None,
                        stale: bool, presets: Optional[dict] = None) -> tuple[Optional[str], Optional[str]]:
    """(code, reason) why `entry` cannot be selected on a device, or (None, None).

    `capabilities` is the app status block's {available, impaired: [{name,
    reason}], starting}; None when stale or never reported. Precedence:
    planned > role_forbids_preset > device_stale > capability_unsupported >
    capability_impaired. A capability the device does not report at all is not
    held against a preset: a device reports only what its current module uses.
    """
    if entry.get("availability") == "planned":
        return CODE_PLANNED, entry.get("planned_reason") or "this mode is not available yet"
    if role is not None:
        from core.roles import preset_violation

        presets = load_presets() if presets is None else presets
        refusal = preset_violation(role, entry["id"], presets)
        if refusal:
            return CODE_ROLE_FORBIDS, refusal
    if stale:
        return CODE_STALE, ("the device's monitoring status is not current (no recent report, or its health has not "
                            "been updated), so what it can run now is unknown")
    impaired = {item.get("name"): item.get("reason") for item in (capabilities or {}).get("impaired") or []
                if isinstance(item, dict)}
    for kind in entry.get("required_capabilities") or []:
        if kind in impaired:
            reason = impaired[kind] or "impaired"
            # "no audible alert output is configured", "not provided by this
            # device": nothing to repair, the device does not have it.
            if re.search(r"\bnot?\b", str(reason)) and re.search(r"\b(configured|provided)\b", str(reason)):
                return CODE_UNSUPPORTED, f"{kind} is not available on this device ({reason})"
    for kind in entry.get("required_capabilities") or []:
        if kind in impaired:
            return CODE_IMPAIRED, f"{kind} is impaired on this device ({impaired[kind] or 'no reason given'})"
    return None, None


# ---------------------------------------------------------------------- monitoring nodes (M7)
def load_companion_nodes() -> list[dict]:
    """The companion monitoring nodes this software can run (core/nodes.py), as
    the API describes them: id, title, required capabilities, settings schema,
    events, limitations. The backend validates `POST .../nodes` against this.

    Imported lazily: core.nodes pulls in the rule engine, which this module's
    dependency-free callers may not want at import time.
    """
    from core.nodes import COMPANIONS

    return [cls.spec.to_dict() for cls in COMPANIONS.values()]


def load_nodes() -> list[dict]:
    """Alias of load_companion_nodes() (the name the backend also looks for)."""
    return load_companion_nodes()
