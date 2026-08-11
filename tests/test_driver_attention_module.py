"""Runs DriverAttentionModule against a synthetic frame — no real camera needed."""

import time

import numpy as np

import modules.driver_attention as driver_attention_module
from modules.base import CorroboratedGraceTracker
from modules.driver_attention import DriverAttentionModule

_CONFIG = {
    "ear_threshold_ratio": 0.75,
    "ear_threshold_fallback": 0.20,
    "calibration_duration_s": 10.0,
    "drowsy_seconds": 2.0,
    "perclos_window_s": 60.0,
    "perclos_threshold": 0.15,
    "yaw_threshold_deg": 35,
    "distraction_seconds": 3.0,
    "glance_window_s": 45.0,
    "glance_min_count": 5,
    "pitch_threshold_deg": -20,
    "eyes_down_seconds": 2.5,
    "face_lost_seconds": 2.5,
    "yolo_weights": "yolo11n.pt",
    # Never a multiple of the single process() call below -> the YOLO/Hands lazy
    # load never triggers, so this test needs no network access.
    "yolo_every_n_frames": 1_000_000,
    "yolo_confidence_threshold": 0.5,
    "yolo_classes": {"cell phone": "phone_detected"},
    "phone_use_seconds": 2.0,
    "enable_hands": True,
    "hand_phone_iou_threshold": 0.2,
    "risk_ema_alpha": 0.3,
    "risk_elevated_threshold": 0.4,
    "risk_high_threshold": 0.75,
    "risk_confirm_seconds": 1.0,
}


def test_process_blank_frame_emits_no_events(tmp_path):
    config = {**_CONFIG, "calibration_path": str(tmp_path / "calibration.json")}
    module = DriverAttentionModule(config)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = module.process(blank_frame)
    assert events == []
    assert module.is_alert_active() is False


def test_telemetry_snapshot_has_the_expected_schema(tmp_path):
    config = {**_CONFIG, "calibration_path": str(tmp_path / "calibration.json")}
    module = DriverAttentionModule(config)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    module.process(blank_frame)

    snapshot = module.get_telemetry_snapshot()
    assert snapshot is not None
    for key in ("ear", "yaw_deg", "pitch_deg", "perclos", "risk_score", "phone_visible", "hand_count", "eating_visible"):
        assert key in snapshot


def test_repeated_blank_frames_eventually_fire_attention_lost(tmp_path):
    """A driver whose face the camera can't see at all for face_lost_seconds should
    fire attention_lost -- this is the fix for naive logic silently treating a fully
    turned-away/slumped-over driver as 'nothing to report'."""
    config = {**_CONFIG, "calibration_path": str(tmp_path / "calibration.json"), "face_lost_seconds": 0.05}
    module = DriverAttentionModule(config)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fired_types = []
    for _ in range(20):
        fired_types += [e.type for e in module.process(blank_frame)]
        time.sleep(0.01)

    assert "attention_lost" in fired_types
    assert module.is_alert_active() is True


def test_face_lost_with_a_person_still_detected_does_not_fire_at_the_short_threshold(tmp_path, monkeypatch):
    """A turned head the object detector still finds a person for (checking a
    mirror/blind spot) must not be treated with the same urgency as a driver
    who has vanished entirely -- see face_lost_with_body_seconds."""
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    config = {
        **_CONFIG, "calibration_path": str(tmp_path / "calibration.json"),
        "face_lost_seconds": 2.5, "face_lost_with_body_seconds": 8.0,
    }
    module = DriverAttentionModule(config)
    module._last_person_visible = True  # bypass real YOLO; this test is about the wiring
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    fired_types = []
    for _ in range(35):  # 4.375s: well past face_lost_seconds, short of face_lost_with_body_seconds
        fired_types += [e.type for e in module.process(blank_frame)]
        t["now"] += 0.125
    assert "attention_lost" not in fired_types
    assert module.is_alert_active() is False

    for _ in range(35):  # continue on to 8.75s: past face_lost_with_body_seconds
        fired_types += [e.type for e in module.process(blank_frame)]
        t["now"] += 0.125
    assert "attention_lost" in fired_types
    assert module.is_alert_active() is True


def test_person_disappearing_partway_through_face_loss_fires_promptly_not_late(tmp_path, monkeypatch):
    """The regression this design exists to prevent: if the corroborating body
    detection ALSO drops out partway through an episode, the alert must not be
    slower than face_lost_seconds alone would have been."""
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    config = {
        **_CONFIG, "calibration_path": str(tmp_path / "calibration.json"),
        "face_lost_seconds": 2.5, "face_lost_with_body_seconds": 8.0,
    }
    module = DriverAttentionModule(config)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    module._last_person_visible = True
    for _ in range(32):  # 4.0s corroborated -- already past face_lost_seconds, nothing fired yet
        module.process(blank_frame)
        t["now"] += 0.125
    assert module.is_alert_active() is False

    module._last_person_visible = False  # body detection drops out too, at t=4.0s
    fired_types = [e.type for e in module.process(blank_frame)]
    assert "attention_lost" in fired_types  # fires on this exact frame, not 2.5s further out
    assert module.is_alert_active() is True


def test_distraction_tracker_reads_both_configured_thresholds(tmp_path):
    """Construction-level check that the two-threshold config keys actually
    reach the tracker (yaw-based end-to-end behavior is covered by
    TestCorroboratedGraceTracker in test_driver_signal_conditioning.py, the
    same tracker class driving this one)."""
    config = {
        **_CONFIG, "calibration_path": str(tmp_path / "calibration.json"),
        "distraction_seconds": 3.0, "rearview_tolerance_seconds": 5.0, "mirror_check_yaw_deg": 55.0,
    }
    module = DriverAttentionModule(config)
    assert isinstance(module._distract_tracker, CorroboratedGraceTracker)
    assert module._distract_tracker._short_s == 3.0
    assert module._distract_tracker._long_s == 5.0
    assert module._mirror_check_yaw_deg == 55.0
