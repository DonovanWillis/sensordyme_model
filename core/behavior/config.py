"""Config for behaviour reasoning: ``modules.driver_attention.behavior_v2`` and ``.behavior``.

    behavior_v2: false        # the only switch; false = shipped behaviour, untouched
    behavior:                 # optional overrides, all keys optional
      max_frame_gap_s: 0.5
      phone: {presence_k: 3, presence_n: 5, hold_overlap: 0.3, ...}
      glance: {long_glance_s: 2.0, budget_s: 6.0, zones: {road: [-20, 20, -8, 12], ...}, ...}
      eye: {down_head_pitch_deg: -10, occlusion_check: true, ...}
      face_loss_coast_s: 1.0
      driver_region: null     # [x1, y1, x2, y2] in 0..1 frame coordinates

A nested mapping rather than flat keys on purpose: presets override flat
driver_attention keys, and nothing in this block is validated enough to be a
preset knob (docs/BEHAVIOUR_REASONING.md, "Before exposing to presets").

``validate_behavior(section)`` returns error strings for core/config_schema.py;
``parse_behavior(section)`` builds the typed config (call validate first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

from core.behavior.eye_state import EyeConfig
from core.behavior.glance import DEFAULT_ZONES, GlanceConfig
from core.behavior.phone import PhoneConfig


@dataclass(frozen=True)
class BehaviorConfig:
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    glance: GlanceConfig = field(default_factory=GlanceConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    eye_occlusion_check: bool = True
    face_loss_coast_s: float = 1.0
    drowsy_seconds: float = 2.0
    eye_release_frames: int = 2
    max_frame_gap_s: float = 0.5
    driver_region: Optional[tuple[float, float, float, float]] = None  # normalised 0..1


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_INT_KEYS = {"presence_k", "presence_n", "hold_k", "hold_n", "repeat_min_count"}
_FRACTION_KEYS = {"hold_overlap", "near_face_overlap", "static_tolerance"}
_NONNEG_KEYS = {"face_margin_face_widths", "reach_above_face_widths", "gaze_min_deg", "in_line_face_widths",
                "release_s", "min_glance_s", "unknown_end_s", "occlusion_max_mean", "occlusion_max_std",
                "iris_closed"}
_SIGNED_KEYS = {"down_head_pitch_deg", "down_iris_pitch_deg"}


def _validate_group(name: str, group: Any, cls, errors: list[str], extra: frozenset[str] = frozenset()) -> None:
    where = f"modules.driver_attention.behavior.{name}"
    if not isinstance(group, dict):
        errors.append(f"`{where}:` must be a mapping, got {group!r}")
        return
    known = {f.name for f in fields(cls)} | extra
    for key, value in group.items():
        if key not in known:
            errors.append(f"`{where}.{key}:` unknown key (known: {sorted(known)})")
            continue
        if key in ("zones", "legit_zones"):
            continue  # validated by the caller
        if key in _INT_KEYS:
            if not (_is_int(value) and value >= 1):
                errors.append(f"`{where}.{key}:` must be a positive integer, got {value!r}")
        elif key in _FRACTION_KEYS:
            if not (_is_num(value) and 0 < value <= 1):
                errors.append(f"`{where}.{key}:` must be a number in (0, 1], got {value!r}")
        elif key in _SIGNED_KEYS:
            if not (_is_num(value) and -90 <= value <= 90):
                errors.append(f"`{where}.{key}:` must be degrees in [-90, 90], got {value!r}")
        elif key in _NONNEG_KEYS:
            if not (_is_num(value) and value >= 0):
                errors.append(f"`{where}.{key}:` must be a non-negative number, got {value!r}")
        elif not (_is_num(value) and value > 0):
            errors.append(f"`{where}.{key}:` must be a positive number, got {value!r}")


def _zones_errors(zones: Any, errors: list[str]) -> None:
    where = "modules.driver_attention.behavior.glance.zones"
    if not isinstance(zones, dict) or not zones:
        errors.append(f"`{where}:` must be a non-empty mapping of zone -> [yaw_min, yaw_max, pitch_min, pitch_max]")
        return
    if "road" not in zones:
        errors.append(f"`{where}:` must define a `road` zone")
    for name, box in zones.items():
        if not (isinstance(name, str) and isinstance(box, list) and len(box) == 4 and all(_is_num(v) for v in box)
                and box[0] < box[1] and box[2] < box[3]):
            errors.append(f"`{where}.{name}:` must be [yaw_min, yaw_max, pitch_min, pitch_max] "
                          f"with min < max, got {box!r}")


def validate_behavior(module_cfg: dict) -> list[str]:
    """Errors for ``behavior_v2`` / ``behavior`` in a driver_attention config section."""
    errors: list[str] = []
    if "behavior_v2" in module_cfg and not isinstance(module_cfg["behavior_v2"], bool):
        errors.append(f"`modules.driver_attention.behavior_v2:` must be true or false, "
                      f"got {module_cfg['behavior_v2']!r}")
    section = module_cfg.get("behavior")
    if section is None:
        return errors
    if not isinstance(section, dict):
        return [*errors, f"`modules.driver_attention.behavior:` must be a mapping, got {section!r}"]
    top = {"phone", "glance", "eye", "max_frame_gap_s", "face_loss_coast_s", "driver_region", "eye_occlusion_check"}
    for key in section:
        if key not in top:
            errors.append(f"`modules.driver_attention.behavior.{key}:` unknown key (known: {sorted(top)})")
    for key in ("max_frame_gap_s", "face_loss_coast_s"):
        if key in section and not (_is_num(section[key]) and section[key] > 0):
            errors.append(f"`modules.driver_attention.behavior.{key}:` must be a positive number, "
                          f"got {section[key]!r}")
    if "eye_occlusion_check" in section and not isinstance(section["eye_occlusion_check"], bool):
        errors.append("`modules.driver_attention.behavior.eye_occlusion_check:` must be true or false")
    region = section.get("driver_region")
    if region is not None and not (
        isinstance(region, list) and len(region) == 4 and all(_is_num(v) and 0 <= v <= 1 for v in region)
        and region[0] < region[2] and region[1] < region[3]
    ):
        errors.append("`modules.driver_attention.behavior.driver_region:` must be null or "
                      f"[x1, y1, x2, y2] in 0..1 with x1 < x2, y1 < y2, got {region!r}")
    if "phone" in section:
        _validate_group("phone", section["phone"], PhoneConfig, errors)
        phone = section["phone"] if isinstance(section["phone"], dict) else {}
        for k, n in (("presence_k", "presence_n"), ("hold_k", "hold_n")):
            kv, nv = phone.get(k, getattr(PhoneConfig, k)), phone.get(n, getattr(PhoneConfig, n))
            if _is_int(kv) and _is_int(nv) and kv > nv:
                errors.append(f"`modules.driver_attention.behavior.phone:` {k} ({kv}) exceeds {n} ({nv})")
    if "glance" in section:
        _validate_group("glance", section["glance"], GlanceConfig, errors, frozenset({"zones", "legit_zones"}))
        glance = section["glance"] if isinstance(section["glance"], dict) else {}
        if "zones" in glance:
            _zones_errors(glance["zones"], errors)
        if "legit_zones" in glance:
            legit = glance["legit_zones"]
            if not (isinstance(legit, list) and all(isinstance(z, str) for z in legit)):
                errors.append("`modules.driver_attention.behavior.glance.legit_zones:` must be a list of zone names")
            elif "road" in legit:
                errors.append("`modules.driver_attention.behavior.glance.legit_zones:` cannot include `road`")
    if "eye" in section:
        _validate_group("eye", section["eye"], EyeConfig, errors)
    return errors


def parse_behavior(module_cfg: dict) -> BehaviorConfig:
    """The typed config. Defaults for anything absent; shipped keys feed the
    sustain times so enabling v2 does not silently change them."""
    section = module_cfg.get("behavior") or {}
    phone_over = dict(section.get("phone") or {})
    phone_over.setdefault("use_sustain_s", float(module_cfg.get("phone_use_seconds", 2.0)))
    glance_over = dict(section.get("glance") or {})
    zones = glance_over.pop("zones", None)
    legit = glance_over.pop("legit_zones", None)
    glance = replace(GlanceConfig(), **glance_over)
    if zones is not None:
        glance = replace(glance, zones=tuple((name, tuple(float(v) for v in box)) for name, box in zones.items()))
    if legit is not None:
        glance = replace(glance, legit_zones=frozenset(legit))
    max_gap = float(section.get("max_frame_gap_s", module_cfg.get("max_evidence_gap_s", 0.5)))
    phone_over.setdefault("max_gap_s", max_gap)
    glance = replace(glance, max_gap_s=glance_over.get("max_gap_s", max_gap))
    region = section.get("driver_region")
    return BehaviorConfig(
        phone=replace(PhoneConfig(), **phone_over),
        glance=glance,
        eye=replace(EyeConfig(), **dict(section.get("eye") or {})),
        eye_occlusion_check=bool(section.get("eye_occlusion_check", True)),
        face_loss_coast_s=float(section.get("face_loss_coast_s", 1.0)),
        drowsy_seconds=float(module_cfg.get("drowsy_seconds", 2.0)),
        eye_release_frames=int(module_cfg.get("eye_closure_release_frames", 2)),
        max_frame_gap_s=max_gap,
        driver_region=None if region is None else tuple(float(v) for v in region),
    )


__all__ = ["DEFAULT_ZONES", "BehaviorConfig", "parse_behavior", "validate_behavior"]
