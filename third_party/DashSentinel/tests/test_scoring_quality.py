"""Regression tests for DashSentinel scoring gates."""

from src.profile import DriverProfile
from src.scoring import AdaptiveScorer


BASE_FEATURES = {
    "ear": 0.30,
    "mar": 0.05,
    "blink_rate": 15.0,
    "closed_frames_norm": 0.0,
    "yawn_flag": 0.0,
    "yawn_count": 0,
    "roll_deg": 0.0,
    "yaw_ratio": 0.0,
    "pitch_ratio": 0.58,
    "posture_flag": 0.0,
    "bad_pose_norm": 0.0,
    "look_away_norm": 0.0,
    "head_tilt_norm": 0.0,
    "head_back_norm": 0.0,
    "eyes_visible": 1.0,
    "mouth_visible": 1.0,
    "visibility_score": 1.0,
    "frame_usable": 1.0,
    "frame_quality": 1.0,
    "frame_quality_reason": "ok",
    "face_count": 1,
    "model_drowsy_score": None,
    "model_confidence": 0.0,
    "model_label": "unavailable",
}


def make_scorer(tmp_path):
    """Helper to create a scorer with a profile that has seen enough updates"""
    profile = DriverProfile(str(tmp_path / "profile.json"))
    profile.total_updates = 200
    scorer = AdaptiveScorer(
        profile,
        calibration_seconds=0,
        low_quality_hold_frames=1,
        status_hold_frames=1,
    )
    return scorer


def test_low_quality_frame_reports_vision_degraded(tmp_path):
    """Test that a low quality frame triggers the vision degraded status."""
    scorer = make_scorer(tmp_path)
    features = dict(BASE_FEATURES)
    features.update({
        "frame_usable": 0.0,
        "frame_quality": 0.1,
        "frame_quality_reason": "unclear/blurry frame",
    })

    score = scorer.score(features)

    assert score.status == "VISION DEGRADED"
    assert "unclear" in score.reason


def test_eye_occlusion_does_not_trigger_eye_closure_score(tmp_path):
    """Test that eye occlusion does not trigger a high drowsiness score."""
    scorer = make_scorer(tmp_path)
    features = dict(BASE_FEATURES)
    features.update({"eyes_visible": 0.0, "closed_frames_norm": 1.0, "ear": 0.01})

    score = scorer.score(features)

    assert score.status == "ALERT"
    assert score.drowsy_score < 0.20


def test_local_model_can_raise_warning_when_confident(tmp_path):
    """Test that a confident local model can raise a warning."""
    scorer = make_scorer(tmp_path)
    features = dict(BASE_FEATURES)
    features.update({"model_drowsy_score": 1.0, "model_confidence": 0.95, "model_label": "drowsy"})

    score = scorer.score(features)

    assert score.status in {"ALERT", "WARNING"}
    assert score.drowsy_score > 0.25
