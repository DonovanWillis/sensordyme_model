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
