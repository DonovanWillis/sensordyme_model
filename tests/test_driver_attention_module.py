"""Runs DriverAttentionModule against a synthetic frame — no real camera needed."""

import time

import numpy as np

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
