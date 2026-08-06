"""Unit tests for config.yaml loading."""

from pathlib import Path

from main import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_config_reads_repo_config():
    config = load_config(REPO_ROOT / "config.yaml")
    assert config["module"] == "driver_attention"
    assert "modules" in config
    assert "driver_attention" in config["modules"]


def test_driver_attention_config_has_the_new_thresholds():
    config = load_config(REPO_ROOT / "config.yaml")
    da_config = config["modules"]["driver_attention"]
    for key in (
        "ear_threshold_ratio",
        "drowsy_seconds",
        "perclos_threshold",
        "distraction_seconds",
        "glance_min_count",
        "pitch_threshold_deg",
        "eyes_down_seconds",
        "face_lost_seconds",
        "yolo_every_n_frames",
        "yolo_classes",
        "phone_use_seconds",
        "enable_hands",
        "risk_ema_alpha",
        "risk_elevated_threshold",
        "risk_high_threshold",
        "risk_confirm_seconds",
    ):
        assert key in da_config, f"missing config.yaml key: modules.driver_attention.{key}"


def test_driver_attention_object_classes_cover_phone_and_a_broad_range_of_food_drink():
    config = load_config(REPO_ROOT / "config.yaml")
    yolo_classes = config["modules"]["driver_attention"]["yolo_classes"]
    assert yolo_classes["cell phone"] == "phone_detected"
    food_drink_classes = {name for name, event in yolo_classes.items() if event == "eating_drinking"}
    # A real driver could be holding any number of things -- require broad coverage,
    # not just "cup" and call it done.
    assert len(food_drink_classes) >= 10
    assert {"cup", "bottle", "sandwich"}.issubset(food_drink_classes)


def test_alerts_config_has_the_escalation_type_map():
    config = load_config(REPO_ROOT / "config.yaml")
    alerts_config = config["alerts"]
    assert "drowsiness" in alerts_config["loud_types"]
    assert "phone_use" in alerts_config["loud_types"]
    assert "attention_lost" in alerts_config["loud_types"]
    assert "high_risk" in alerts_config["loud_types"]
    assert "fatigue_warning" in alerts_config["soft_types"]
    assert "frequent_glancing" in alerts_config["soft_types"]
    assert "elevated_risk" in alerts_config["soft_types"]


def test_logging_config_has_telemetry_settings():
    config = load_config(REPO_ROOT / "config.yaml")
    logging_config = config["logging"]
    assert "telemetry_csv_path" in logging_config
    assert "telemetry_interval_s" in logging_config


def test_load_config_parses_arbitrary_yaml(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "module: occupancy\n"
        "display: false\n"
        "modules:\n"
        "  occupancy:\n"
        "    confidence_threshold: 0.6\n"
    )
    config = load_config(config_path)
    assert config == {
        "module": "occupancy",
        "display": False,
        "modules": {"occupancy": {"confidence_threshold": 0.6}},
    }
