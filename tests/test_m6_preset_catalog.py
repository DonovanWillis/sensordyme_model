"""M6: the versioned preset catalogue (presets.yaml + core/presets.py).

Pins backward compatibility with every existing reader, the settings schema,
and that a preset's validation status is DERIVED from core/model_registry.py
(the weakest model along its capability chain) and can never be declared.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from core import model_registry as mr
from core import presets as P  # noqa: N812 - short alias used throughout
from core.roles import ROLES, preset_violation

REPO = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((REPO / "config.yaml").read_text())
FIVE_PRODUCT_MODES = {"driver_safety", "child_monitoring", "occupancy", "property_monitoring", "workplace_safety"}


@pytest.fixture(scope="module")
def registry():
    return mr.load_registry()


@pytest.fixture(scope="module")
def entries(registry):
    return {e["id"]: e for e in P.catalog_entries(registry=registry)}


# ---------------------------------------------------------------------- back-compat
def test_existing_readers_see_exactly_the_runnable_presets():
    presets = P.load_presets()
    assert set(presets) == {"driver_safety", "elderly_monitoring", "security_monitoring", "baby_monitoring"}
    for preset in presets.values():
        assert preset["module"] and preset["title"]  # the fields every older reader uses
    assert "occupancy" not in presets and "workplace_safety" not in presets


def test_aliases_and_variants_still_resolve():
    assert P.resolve_preset("driver_distraction") == "driver_safety"
    assert P.resolve_preset("driver_distraction:strict") == "driver_safety:strict"
    assert P.resolve_preset("driver_distraction:strict@drowsy_seconds=1.5") == "driver_safety:strict@drowsy_seconds=1.5"
    assert P.same_preset("driver_distraction_openpilot", "driver_safety")
    assert P.same_preset("driver_safety@drowsy_seconds=1.50", "driver_safety@drowsy_seconds=1.5")
    assert not P.same_preset("driver_safety@drowsy_seconds=1.5", "driver_safety")


def test_the_catalogue_is_internally_consistent(registry):
    assert P.catalog_problems(registry=registry) == []
    assert P.catalog_version()


def test_the_five_product_modes_are_all_present_honestly(entries):
    by_mode = {e["product_mode"]: e for e in entries.values() if e["product_mode"]}
    assert set(by_mode) == FIVE_PRODUCT_MODES
    assert by_mode["driver_safety"]["id"] == "driver_safety"
    assert by_mode["child_monitoring"]["id"] == "baby_monitoring"
    assert by_mode["property_monitoring"]["id"] == "security_monitoring"
    for planned in ("occupancy", "workplace_safety"):
        entry = by_mode[planned]
        assert entry["availability"] == "planned" and entry["planned_reason"]
        assert set(entry["served_by"]) <= set(P.load_presets())


def test_every_entry_carries_the_contract_fields(entries):
    required = {"id", "version", "title", "description", "variants", "required_capabilities",
                "optional_capabilities", "hardware", "temporal_policies", "alert_patterns", "settings",
                "compatibility", "validation", "limitations"}
    for entry in entries.values():
        assert required <= set(entry), entry["id"]
        assert entry["compatibility"]["exclusive_group"] == "vision_module"
        assert {"status", "evidence"} <= set(entry["validation"])
        for setting in entry["settings"]:
            assert setting["type"] in ("number", "bool", "enum")


def test_setting_defaults_are_what_config_yaml_runs():
    """A setting maps onto an existing module key, and its default is that key's shipped value."""
    for name, preset in P.load_presets().items():
        module_cfg = CONFIG["modules"][preset["module"]]
        for setting in P.settings_schema(preset).values():
            assert setting.key in module_cfg, f"{name}: {setting.key} is not a {preset['module']} config key"
            assert module_cfg[setting.key] == setting.default, f"{name}.{setting.key}"


# ---------------------------------------------------------------------- validation status
def test_status_is_the_registry_weakest_link(registry, entries):
    for name in P.load_presets():
        expected = mr.preset_view(registry, name).weakest
        assert entries[name]["validation"]["status"] == expected.value
    # Today: only the detector has offline evidence; face/pose/hands have none.
    assert entries["security_monitoring"]["validation"]["status"] == "benchmarked_offline"
    assert entries["driver_safety"]["validation"]["status"] == "unevaluated"
    assert entries["workplace_safety"]["validation"]["status"] == "not_applicable"
    assert not any(e["validation"]["validated"] for e in entries.values())


def test_status_moves_only_when_the_registry_does(registry):
    detector = registry.provider_of("objects")
    promoted = dataclasses.replace(
        detector, evaluation=dataclasses.replace(detector.evaluation,
                                                 status=mr.EvaluationStatus.EVALUATED_IN_DOMAIN))
    changed = dataclasses.replace(
        registry, models=tuple(promoted if m.id == detector.id else m for m in registry.models))
    entries = {e["id"]: e for e in P.catalog_entries(registry=changed)}
    assert entries["security_monitoring"]["validation"]["status"] == "evaluated_in_domain"
    # Baby still needs pose, which has no evidence: the weakest link holds it down.
    assert entries["baby_monitoring"]["validation"]["status"] == "unevaluated"


def test_a_declared_status_is_refused(tmp_path, registry):
    data = yaml.safe_load((REPO / "presets.yaml").read_text())
    data["presets"]["security_monitoring"]["validation_status"] = "validated_on_device"
    path = tmp_path / "presets.yaml"
    path.write_text(yaml.safe_dump(data))
    assert any("derived from the registry" in p for p in P.catalog_problems(path, registry=registry))


def test_capabilities_must_match_what_the_module_consumes(tmp_path, registry):
    data = yaml.safe_load((REPO / "presets.yaml").read_text())
    data["presets"]["elderly_monitoring"]["optional_capabilities"] = ["objects"]
    path = tmp_path / "presets.yaml"
    path.write_text(yaml.safe_dump(data))
    assert any("elderly_monitoring: capabilities" in p for p in P.catalog_problems(path, registry=registry))


# ---------------------------------------------------------------------- settings
SECURITY = P.load_presets()["security_monitoring"]
DRIVER_PRESET = P.load_presets()["driver_safety"]


@pytest.mark.parametrize(("settings", "key"), [
    ({"linger_s": 1.0}, "linger_s"),            # below min
    ({"linger_s": 9999}, "linger_s"),           # above max
    ({"linger_s": 12.5}, "linger_s"),           # off the 5 s step
    ({"linger_s": "30"}, "linger_s"),           # a string is not a number
    ({"crowd_threshold": True}, "crowd_threshold"),  # a bool is not a number
    ({"drowsy_seconds": 1.5}, "drowsy_seconds"),     # another mode's setting
])
def test_invalid_settings_name_the_key(settings, key):
    with pytest.raises(P.PresetError) as exc:
        P.validate_settings(SECURITY, settings)
    assert exc.value.key == key and key in exc.value.reason


def test_valid_settings_are_typed_and_encoded_canonically():
    cleaned = P.validate_settings(SECURITY, {"linger_s": 60, "crowd_threshold": 6.0})
    assert cleaned == {"crowd_threshold": 6, "linger_s": 60.0}
    name = P.encode_settings("security_monitoring:away", cleaned)
    assert name == "security_monitoring:away@crowd_threshold=6,linger_s=60"
    assert P.parse_settings(name) == cleaned
    canonical, _preset, overrides = P.preset_overrides(name)
    assert canonical == name
    assert overrides == {"confidence_threshold": 0.4, "confirm_frames": 2, "crowd_threshold": 6, "linger_s": 60.0}


def test_bool_and_enum_settings():
    preset = {"settings": [
        {"key": "loud", "title": "Loud", "type": "bool", "default": True},
        {"key": "tone", "title": "Tone", "type": "enum", "default": "beep", "options": ["beep", "chime"]},
    ]}
    assert P.validate_settings(preset, {"loud": False, "tone": "chime"}) == {"loud": False, "tone": "chime"}
    for bad in ({"loud": 1}, {"tone": "siren"}):
        with pytest.raises(P.PresetError):
            P.validate_settings(preset, bad)
    with pytest.raises(P.PresetError):
        P.settings_schema({"settings": [{"key": "tone", "type": "enum", "default": "x", "options": []}]})


def test_role_floors_win_over_settings():
    presets = P.load_presets()
    driver = ROLES["driver_safety"]
    # The schema allows up to 3.0 s; the driver role's floor is 2.0 s.
    assert P.validate_settings(DRIVER_PRESET, {"drowsy_seconds": 2.5})
    assert "safety floor" in preset_violation(driver, "driver_safety@drowsy_seconds=2.5", presets)
    assert preset_violation(driver, "driver_safety:strict@drowsy_seconds=1.2", presets) is None
    # A general device has no floor.
    assert preset_violation(ROLES["general"], "driver_safety@drowsy_seconds=2.5", presets) is None
    # Unreadable settings are refused, never waved through unchecked.
    assert preset_violation(driver, "driver_safety@drowsy_seconds", presets)


# ---------------------------------------------------------------------- availability
def test_availability_codes_and_precedence(entries):
    driver = ROLES["driver_safety"]
    caps = {"available": ["camera", "inference"], "impaired": [
        {"name": "face", "reason": "no face landmarks for 10 s"},
        {"name": "audible_alert", "reason": "no audible alert output is configured"},
    ], "starting": []}
    assert P.device_availability(entries["occupancy"], role=driver, capabilities=None, stale=True)[0] == "planned"
    assert P.device_availability(entries["baby_monitoring"], role=driver, capabilities=None,
                                 stale=True)[0] == "role_forbids_preset"
    assert P.device_availability(entries["driver_safety"], role=driver, capabilities=None,
                                 stale=True)[0] == "device_stale"
    code, reason = P.device_availability(entries["driver_safety"], role=None, capabilities=caps, stale=False)
    assert code == "capability_unsupported" and "audible_alert" in reason
    caps["impaired"] = caps["impaired"][:1]
    code, reason = P.device_availability(entries["driver_safety"], role=None, capabilities=caps, stale=False)
    assert code == "capability_impaired" and "face" in reason
    # A capability the device does not report at all is not held against a preset.
    assert P.device_availability(entries["elderly_monitoring"], role=None, capabilities=caps,
                                 stale=False) == (None, None)


def test_effective_settings_layer_default_variant_and_owner_values():
    values = P.effective_settings(DRIVER_PRESET, "strict", {"phone_use_seconds": 0.5})
    assert values["drowsy_seconds"] == 1.5      # the variant
    assert values["phone_use_seconds"] == 0.5   # the owner's setting
    assert values["eyes_down_seconds"] == 2.5   # the default
