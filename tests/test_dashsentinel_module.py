"""Tests for the vendored DashSentinel adapter (no camera needed — FaceMesh finds
no face in synthetic frames, which exercises their NO FACE path end-to-end)."""

import numpy as np

from modules.base import available_modules, get_module_class


def _make_module(tmp_path):
    return get_module_class("driver_attention_dashsentinel")(
        {"profile_path": str(tmp_path / "profile.json"), "save_profile_every_seconds": 9999}
    )


def test_module_is_registered():
    assert "driver_attention_dashsentinel" in available_modules()


def test_no_face_transition_emits_one_event(tmp_path):
    module = _make_module(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    events = module.process(frame)
    assert [e.type for e in events] == ["dashsentinel_no_face"]  # transition into NO FACE
    for _ in range(4):
        assert module.process(frame) == []  # steady state: no repeat events

    out = module.annotate_debug(frame.copy())
    assert out.shape == frame.shape
    assert not module.is_alert_active()


def test_vendored_scorer_escalation_behavior(tmp_path):
    """Drive DashSentinel's real AdaptiveScorer directly with synthetic features.

    Documents a real property of their weighting: pure sustained eye closure sums
    to at most 0.28 (ear) + 0.20 (closed frames) + 0.08 (low blink) = 0.56, which
    is UNDER their 0.58 DROWSY threshold — closure alone only ever reaches
    WARNING; DROWSY additionally requires a yawn, bad head pose, or model score."""
    module = _make_module(tmp_path)
    scorer = module._scorer
    scorer.start_time -= 1000  # skip their 25s wall-clock calibration window
    scorer.profile.total_updates = 200  # and their min-updates calibration gate

    def features(closed, yawning=False):
        return {
            "ear": 0.10 if closed else 0.30,
            "mar": 0.70 if yawning else 0.16,
            "blink_rate": 2.0 if closed else 15.0,
            "closed_frames_norm": 1.0 if closed else 0.0,
            "yawn_flag": 1.0 if yawning else 0.0,
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
            "selected_face_index": 0,
        }

    for _ in range(10):
        out = scorer.score(features(closed=False))
    assert out.status == "ALERT"

    # sustained eye closure alone: WARNING, and provably never DROWSY (score ceiling 0.56)
    closure_statuses = [scorer.score(features(closed=True)).status for _ in range(60)]
    assert "WARNING" in closure_statuses
    assert "DROWSY" not in closure_statuses
    assert closure_statuses[-1] == "WARNING"

    # closure + yawning pushes the score past 0.58 -> DROWSY after their hysteresis
    drowsy_statuses = [scorer.score(features(closed=True, yawning=True)).status for _ in range(60)]
    assert "DROWSY" in drowsy_statuses
    assert drowsy_statuses[-1] == "DROWSY"
