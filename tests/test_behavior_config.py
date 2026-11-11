"""Config parsing/validation for behaviour reasoning (modules.driver_attention.behavior_v2 / .behavior)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from core.behavior.config import BehaviorConfig, parse_behavior, validate_behavior
from core.behavior.glance import DEFAULT_ZONES
from core.config_schema import validate

REPO = Path(__file__).resolve().parent.parent


def _repo_config() -> dict:
    return yaml.safe_load((REPO / "config.yaml").read_text())


def test_shipped_config_keeps_behavior_v2_off_and_validates():
    config = _repo_config()
    driver = config["modules"]["driver_attention"]
    assert driver.get("behavior_v2", False) is False
    assert validate(config)[0] == []


def test_defaults_inherit_the_shipped_sustain_times():
    parsed = parse_behavior({"phone_use_seconds": 1.5, "drowsy_seconds": 1.8, "max_evidence_gap_s": 0.3})
    assert parsed.phone.use_sustain_s == 1.5
    assert parsed.drowsy_seconds == 1.8
    assert parsed.max_frame_gap_s == 0.3
    assert parsed.phone.max_gap_s == 0.3 and parsed.glance.max_gap_s == 0.3
    assert parse_behavior({}).glance.zones == DEFAULT_ZONES
    assert isinstance(parse_behavior({}), BehaviorConfig)


def test_overrides_are_applied():
    parsed = parse_behavior({"behavior": {
        "glance": {"budget_s": 12.0, "zones": {"road": [-10, 10, -5, 5], "lap": [-40, 40, -90, -20]},
                   "legit_zones": ["lap"]},
        "phone": {"presence_k": 2, "presence_n": 3},
        "eye": {"down_head_pitch_deg": -12.0},
        "driver_region": [0.2, 0.0, 0.9, 1.0],
        "eye_occlusion_check": False,
    }})
    assert parsed.glance.budget_s == 12.0
    assert dict(parsed.glance.zones)["road"] == (-10.0, 10.0, -5.0, 5.0)
    assert parsed.glance.legit_zones == frozenset({"lap"})
    assert (parsed.phone.presence_k, parsed.phone.presence_n) == (2, 3)
    assert parsed.eye.down_head_pitch_deg == -12.0
    assert parsed.driver_region == (0.2, 0.0, 0.9, 1.0)
    assert parsed.eye_occlusion_check is False


@pytest.mark.parametrize(("section", "expected"), [
    ({"behavior_v2": "yes"}, "behavior_v2:` must be true or false"),
    ({"behavior": []}, "behavior:` must be a mapping"),
    ({"behavior": {"typo": 1}}, "behavior.typo:` unknown key"),
    ({"behavior": {"phone": {"presence_k": 6, "presence_n": 5}}}, "presence_k (6) exceeds presence_n (5)"),
    ({"behavior": {"phone": {"hold_overlap": 1.5}}}, "hold_overlap:` must be a number in (0, 1]"),
    ({"behavior": {"phone": {"no_such": 1}}}, "phone.no_such:` unknown key"),
    ({"behavior": {"glance": {"budget_s": -1}}}, "budget_s:` must be a positive number"),
    ({"behavior": {"glance": {"repeat_min_count": 0}}}, "repeat_min_count:` must be a positive integer"),
    ({"behavior": {"glance": {"zones": {"lap": [0, 1, 2, 3]}}}}, "must define a `road` zone"),
    ({"behavior": {"glance": {"zones": {"road": [10, -10, 0, 1]}}}}, "with min < max"),
    ({"behavior": {"glance": {"legit_zones": ["road"]}}}, "cannot include `road`"),
    ({"behavior": {"eye": {"down_head_pitch_deg": -120}}}, "must be degrees in [-90, 90]"),
    ({"behavior": {"driver_region": [0.5, 0, 0.2, 1]}}, "driver_region:` must be null or"),
    ({"behavior": {"face_loss_coast_s": 0}}, "face_loss_coast_s:` must be a positive number"),
])
def test_bad_values_are_errors(section, expected):
    errors = validate_behavior(section)
    assert any(expected in e for e in errors), errors


def test_config_schema_reports_behavior_errors():
    config = copy.deepcopy(_repo_config())
    config["modules"]["driver_attention"]["behavior"] = {"glance": {"long_glance_s": "2"}}
    errors, _ = validate(config)
    assert any("long_glance_s" in e for e in errors), errors


def test_a_valid_enabled_config_passes_the_schema():
    config = copy.deepcopy(_repo_config())
    config["modules"]["driver_attention"]["behavior_v2"] = True
    config["modules"]["driver_attention"]["behavior"] = {"glance": {"budget_s": 8.0}}
    assert validate(config)[0] == []
