"""Light tests for the vendored skinnemoen FaceMesh DMS (no camera; FaceMesh on
synthetic frames finds no face, which must be handled exactly like the original —
frame skipped, no events, no crash)."""

import numpy as np

from modules.base import available_modules, get_module_class


def test_module_is_registered():
    assert "driver_attention_skinnemoen" in available_modules()


def test_blank_frames_are_skipped_without_events():
    module = get_module_class("driver_attention_skinnemoen")({"phone_detection": False})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        assert module.process(frame) == []
    # annotate must not crash with no face ever seen
    out = module.annotate_debug(frame.copy())
    assert out.shape == frame.shape
    snapshot = module.get_telemetry_snapshot()
    assert set(snapshot) >= {"ear", "yaw_deg", "pitch_deg", "perclos", "risk_score"}


def test_phone_use_fires_only_when_looking_at_it_and_once_per_episode():
    module = get_module_class("driver_attention_skinnemoen")({"phone_detection": False})

    module._phone_visible = True
    module._distracted = False  # phone in frame, eyes forward (mounted phone) -> silent
    assert module._check_phone_use() is None

    module._distracted = True  # now looking at it -> fires immediately
    event = module._check_phone_use()
    assert event is not None and event.type == "skinnemoen_phone_use"
    assert module._check_phone_use() is None  # rising edge only

    module._distracted = False  # looks back at the road -> re-arms
    assert module._check_phone_use() is None
    module._distracted = True
    assert module._check_phone_use().type == "skinnemoen_phone_use"


def test_every_skinnemoen_alert_type_is_audible():
    import main as main_module
    import yaml
    from pathlib import Path

    alert_set = main_module.ALERT_EVENT_TYPES_SKINNEMOEN
    for event_type in ("drowsiness", "skinnemoen_distracted", "skinnemoen_phone_use"):
        assert event_type in alert_set

    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    loud = set(config["alerts"]["loud_types"])
    soft = set(config["alerts"]["soft_types"])
    assert {"drowsiness", "skinnemoen_distracted", "skinnemoen_phone_use"} <= loud
    assert "skinnemoen_calibration_complete" in soft
