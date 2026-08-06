"""Unit tests for RiskFusion: cross-signal reasoning, corroboration, smoothing, hysteresis."""

from modules.driver_attention import RiskFusion


def test_all_signals_zero_gives_zero_score():
    fusion = RiskFusion()
    score, event = fusion.update(eye_risk=0.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=0.0)
    assert score == 0.0
    assert event is None


def test_single_full_eye_risk_alone_does_not_cross_high_threshold_at_steady_state():
    """Weighted alone, even a fully-fired eye signal (weight 0.42) shouldn't reach the 0.75
    'high' bar once the EMA has converged -- that's a deliberate design choice: full atomic
    events (drowsiness, distraction, ...) already fire their own dedicated alert; the fused
    "high_risk" event exists for *combinations* no single signal's own threshold would catch."""
    fusion = RiskFusion()
    score = 0.0
    for i in range(30):  # let the EMA converge to its steady-state value
        score, _ = fusion.update(
            eye_risk=1.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=float(i)
        )
    assert score < 0.75


def test_corroboration_bonus_applies_when_eye_and_gaze_risk_are_both_elevated():
    """Mild drowsiness + mild distraction together should score higher than the sum of
    treating them independently would suggest -- this is the whole point of fusion."""
    fusion_combined = RiskFusion()
    combined_score, _ = fusion_combined.update(
        eye_risk=0.6, gaze_risk=0.6, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=0.0
    )

    fusion_eye_only = RiskFusion()
    eye_only_score, _ = fusion_eye_only.update(
        eye_risk=0.6, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=0.0
    )
    fusion_gaze_only = RiskFusion()
    gaze_only_score, _ = fusion_gaze_only.update(
        eye_risk=0.0, gaze_risk=0.6, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=0.0
    )

    # The combined score must exceed simply adding the two isolated contributions --
    # that gap is the corroboration bonus modeling compound danger.
    assert combined_score > eye_only_score + gaze_only_score


def test_score_is_smoothed_not_instantaneous():
    fusion = RiskFusion(ema_alpha=0.3)
    score_1, _ = fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=0.0)
    # A single frame of maximum risk shouldn't instantly snap the smoothed score to 1.0.
    assert 0.0 < score_1 < 1.0
    score_2, _ = fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=1.0)
    assert score_2 > score_1  # keeps climbing while the inputs stay maxed out


def test_high_risk_event_requires_sustained_high_score():
    # elevated_threshold set unreachably high so only high_risk's own confirm-delay
    # logic is under test here (elevated_risk has no confirm delay and would otherwise
    # fire immediately, masking what this test is checking).
    fusion = RiskFusion(ema_alpha=1.0, elevated_threshold=1.5, high_threshold=0.75, confirm_seconds=1.0)
    _, event = fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=0.0)
    assert event is None  # first instant above threshold -- not sustained yet
    _, event = fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=1.1)
    assert event == "high_risk"


def test_elevated_risk_fires_once_then_rearms_after_dropping():
    fusion = RiskFusion(ema_alpha=1.0, elevated_threshold=0.4, high_threshold=0.99)
    _, event = fusion.update(eye_risk=1.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=0.0)
    assert event == "elevated_risk"
    # Doesn't refire every frame while still elevated.
    _, event = fusion.update(eye_risk=1.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=1.0)
    assert event is None
    # Drops well below the threshold -- rearms.
    fusion.update(eye_risk=0.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=2.0)
    _, event = fusion.update(eye_risk=1.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=3.0)
    assert event == "elevated_risk"


def test_face_lost_risk_alone_can_reach_elevated():
    fusion = RiskFusion(ema_alpha=1.0, elevated_threshold=0.35)
    score, event = fusion.update(
        eye_risk=0.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=1.0, now=0.0
    )
    assert score >= 0.35
    assert event == "elevated_risk"


def test_is_high_risk_active_tracks_the_confirm_tracker():
    fusion = RiskFusion(ema_alpha=1.0, high_threshold=0.75, confirm_seconds=1.0)
    assert fusion.is_high_risk_active is False
    fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=0.0)
    assert fusion.is_high_risk_active is False
    fusion.update(eye_risk=1.0, gaze_risk=1.0, phone_risk=1.0, eating_risk=1.0, face_lost_risk=0.0, now=1.1)
    assert fusion.is_high_risk_active is True
    fusion.update(eye_risk=0.0, gaze_risk=0.0, phone_risk=0.0, eating_risk=0.0, face_lost_risk=0.0, now=1.2)
    assert fusion.is_high_risk_active is False
