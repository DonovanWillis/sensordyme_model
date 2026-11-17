"""`network.direct_mode:` in config.yaml and the schema: off by default, never open."""

from __future__ import annotations

import copy

import pytest

from core.config_schema import validate
from core.direct_mode import DirectModeConfig
from main import load_config


def _base(**direct):
    config = {"module": "m", "modules": {"m": {}}, "network": {"provisioning": {"method": "ble"}}}
    if direct:
        config["network"]["direct_mode"] = direct
    return config


def test_the_shipped_config_has_direct_mode_off_and_is_valid():
    config = load_config("config.yaml")
    errors, warnings = validate(config)
    assert errors == []
    direct = config["network"]["direct_mode"]
    assert direct["enabled"] is False
    assert direct["fallback_enabled"] is False
    assert not any("direct_mode" in w for w in warnings)


def test_the_block_passes_the_release_credential_gate():
    """No key under direct_mode reads as a filled-in credential to the transfer package's gate."""
    from pathlib import Path

    from scripts.release.build_transfer_package import config_credential_check

    text = (Path(__file__).resolve().parent.parent / "config.yaml").read_text()
    assert not [f for f in config_credential_check(text) if "direct_mode" in f]


def test_the_shipped_values_are_the_code_defaults():
    """Deleting the block from config.yaml changes nothing."""
    shipped = DirectModeConfig.from_dict(load_config("config.yaml")["network"]["direct_mode"])
    assert shipped == DirectModeConfig()
    assert DirectModeConfig.from_dict(None) == DirectModeConfig()
    assert DirectModeConfig().enabled is False and DirectModeConfig().fallback_enabled is False


def test_a_disabled_block_validates_exactly_like_no_block():
    repo = load_config("config.yaml")
    without = copy.deepcopy(repo)
    del without["network"]["direct_mode"]
    assert validate(repo) == validate(without)


def test_only_true_enables_it():
    for value in ("yes", 1, "true"):
        assert DirectModeConfig.from_dict({"enabled": value}).enabled is False
        errors, _ = validate(_base(enabled=value))
        assert any("network.direct_mode.enabled" in e for e in errors)


@pytest.mark.parametrize(("key", "value"), [
    ("security", "open"), ("security", "none"), ("security", "wep"), ("security", ""),
    ("channel", 0), ("channel", 13), ("channel", "6"), ("channel", True),
    ("ap_timeout_s", 1), ("ap_timeout_s", "30"), ("fallback_after_s", 5), ("return_check_s", 0),
    ("interface", ""), ("interface", "wlan0; reboot"), ("interface", "x" * 16),
    ("fallback_enabled", "no"), ("passphrase_path", ""), ("state_path", 7),
])
def test_bad_values_are_errors_naming_the_key(key, value):
    errors, _ = validate(_base(**{key: value}))
    assert any(f"network.direct_mode.{key}" in e for e in errors), errors


@pytest.mark.parametrize("key", ["psk", "password", "passphrase"])
def test_a_passphrase_in_config_yaml_is_refused(key):
    errors, _ = validate(_base(**{key: "correct horse battery"}))
    assert any(f"network.direct_mode.{key}" in e and "never set in config.yaml" in e for e in errors)
    assert not any("correct horse battery" in e for e in errors), "the value is never echoed"


def test_the_mapping_itself_must_be_a_mapping():
    config = _base()
    config["network"]["direct_mode"] = "on"
    errors, _ = validate(config)
    assert any("network.direct_mode" in e and "mapping" in e for e in errors)


def test_a_typo_is_a_warning():
    _, warnings = validate(_base(enabled=False, fallback_enable=True))
    assert any("network.direct_mode.fallback_enable" in w for w in warnings)


def test_enabled_without_bluetooth_or_a_token_warns_about_what_will_not_work():
    config = _base(enabled=True)
    config["network"]["provisioning"]["method"] = "hotspot"
    errors, warnings = validate(config)
    assert errors == []
    assert any("recovery path" in w for w in warnings)
    assert any("dashboard.token" in w for w in warnings)
    config["network"]["provisioning"]["method"] = "ble"
    config["dashboard"] = {"token": "t", "host": "0.0.0.0"}  # noqa: S104 - a config value, not a bind
    _, warnings = validate(config)
    assert not any("network.direct_mode" in w for w in warnings)
