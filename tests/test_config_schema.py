"""Tests for config.yaml validation."""

from core.config_schema import validate
from main import load_config


def test_repo_config_is_valid():
    errors, warnings = validate(load_config("config.yaml"))
    assert errors == []


def test_bad_types_are_reported_with_paths():
    errors, _ = validate(
        {
            "module": "driver_attention",
            "camera": {"fps": "fast", "width": 640, "height": 480},
            "dashboard": {"port": 99999, "jpeg_quality": 200},
            "accelerator": {"yolo_hef": 42},
            "network": {"provisioning": {"method": "carrier-pigeon"}},
            "health": {"check_interval_s": -1},
            "modules": {"driver_attention": {}},
        }
    )
    joined = "\n".join(errors)
    assert "camera.fps" in joined
    assert "dashboard.port" in joined
    assert "dashboard.jpeg_quality" in joined
    assert "accelerator.yolo_hef" in joined
    assert "network.provisioning.method" in joined
    assert "health.check_interval_s" in joined


def test_unknown_section_warns_but_does_not_error():
    errors, warnings = validate({"module": "m", "modules": {"m": {}}, "dashbord": {}})
    assert errors == []
    assert any("dashbord" in w for w in warnings)


def test_missing_module_section_errors():
    errors, _ = validate({"module": "ghost", "modules": {"real": {}}})
    assert any("modules.ghost" in e for e in errors)


def test_token_with_lan_host_warns():
    _, warnings = validate(
        {"module": "m", "modules": {"m": {}},
         "dashboard": {"host": "0.0.0.0", "token": "secret"}}
    )
    assert any("token" in w for w in warnings)


# ------------------------------------------------------------------ cloud:
# Previously entirely unvalidated (found in a repo-wide audit) despite holding
# the security-relevant TLS settings -- a typo'd/wrong-typed key produced
# neither an error nor a warning at boot, just a device that silently never
# connected.
def test_cloud_bad_types_are_reported():
    errors, _ = validate({
        "module": "m", "modules": {"m": {}},
        "cloud": {"enabled": "yes", "tls_insecure_skip_verify": "nah", "poll_interval_s": -1},
    })
    joined = "\n".join(errors)
    assert "cloud.enabled" in joined
    assert "cloud.tls_insecure_skip_verify" in joined
    assert "cloud.poll_interval_s" in joined


def test_cloud_enabled_requires_an_http_base_url():
    errors, _ = validate({
        "module": "m", "modules": {"m": {}},
        "cloud": {"enabled": True, "base_url": "not-a-url"},
    })
    assert any("cloud.base_url" in e for e in errors)


def test_cloud_disabled_does_not_require_a_base_url():
    errors, _ = validate({"module": "m", "modules": {"m": {}}, "cloud": {"enabled": False}})
    assert errors == []


def test_cloud_missing_entirely_is_fine():
    errors, _ = validate({"module": "m", "modules": {"m": {}}})
    assert errors == []


def test_cloud_insecure_tls_over_https_warns():
    _, warnings = validate({
        "module": "m", "modules": {"m": {}},
        "cloud": {"enabled": True, "base_url": "https://example.com",
                  "tls_insecure_skip_verify": True},
    })
    assert any("tls_insecure_skip_verify" in w for w in warnings)


def test_cloud_must_be_a_mapping():
    errors, _ = validate({"module": "m", "modules": {"m": {}}, "cloud": "nope"})
    assert any("cloud:" in e for e in errors)


def test_shapes_that_used_to_pass_validation_and_crash_boot_are_errors():
    """Each of these validated clean and then crashed startup (AttributeError,
    ValueError, TypeError in `{**None}`) or broke the uplink silently."""
    base = {"module": "m", "modules": {"m": {}}}
    cases = {
        "logging:": {"logging": None},
        "alerts:": {"alerts": None},
        "alerts.cooldown_s": {"alerts": {"cooldown_s": "fifteen"}},
        "modules.m": {"modules": {"m": None}},
        "health.reboot_max_total": {"health": {"reboot_max_total": "five"}},
        "cloud.queue_max": {"cloud": {"queue_max": 5000}},
    }
    for expected, patch in cases.items():
        errors, _ = validate({**base, **patch})
        assert any(expected in e for e in errors), f"{patch} was not reported: {errors}"


def test_an_enabled_uplink_with_no_key_is_called_out():
    _, warnings = validate({"module": "m", "modules": {"m": {}},
                            "cloud": {"enabled": True, "base_url": "http://x", "device_key": ""}})
    assert any("device_key" in w for w in warnings)
