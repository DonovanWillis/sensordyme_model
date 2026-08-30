"""Gestures (hand_raised, waving) and resting, from synthetic pose sequences.

These attributes exist to back three Jupiter compositions honestly:

  * "alert me if someone waves"        -> waving label, direct
  * "alert me if someone raises a hand" -> hand_raised label, direct
  * "alert me if I'm sleeping"          -> resting label + long persist_s

The false-positive boundaries matter more than the positives, and each has a
named nearest-innocent-motion it must not fire on: a raised hand is not a wave
(no oscillation); reaching out and pulling back is not a wave (two swings, not
three); a person sitting and gesturing is not resting (wrists moving); a person
standing still is not resting (not settled).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.activity import (
    FLAIL_MIN_REVERSALS,
    HAND_RAISED_MARGIN,
    MIN_SAMPLES,
    POSE_LOST_GRACE_S,
    WAVE_MIN_SWINGS,
    ActivityRecognizer,
)


@dataclass
class _Landmark:
    x: float
    y: float
    visibility: float = 1.0


def _make_landmarks(overrides: dict[int, tuple[float, float]]) -> list[_Landmark]:
    """A 33-landmark list, everything hidden except what the test places."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    for index, (x, y) in overrides.items():
        marks[index] = _Landmark(x, y)
    return marks


# Shared body geometry (normalised coords, y grows downward).
_STANDING = {
    0: (0.50, 0.10),   # nose
    11: (0.45, 0.20), 12: (0.55, 0.20),   # shoulders
    23: (0.45, 0.50), 24: (0.55, 0.50),   # hips
    27: (0.45, 0.90), 28: (0.55, 0.90),   # ankles (leg span > torso span)
}
_SITTING = {
    0: (0.50, 0.25),
    11: (0.45, 0.35), 12: (0.55, 0.35),
    23: (0.45, 0.55), 24: (0.55, 0.55),
    27: (0.45, 0.65), 28: (0.55, 0.65),   # folded legs: leg span < torso span
}


def _feed(recognizer: ActivityRecognizer, frames: list[dict], dt: float = 0.1):
    state = None
    for step, overrides in enumerate(frames):
        state = recognizer.update(_make_landmarks(overrides), now=step * dt)
    return state


def _repeat(base: dict, count: int) -> list[dict]:
    return [dict(base) for _ in range(count)]


class TestHandRaised:
    def test_wrist_held_above_shoulder_is_hand_raised(self):
        frames = _repeat({**_STANDING, 16: (0.60, 0.08)}, 12)  # right wrist above shoulder
        state = _feed(ActivityRecognizer(), frames)
        assert state.hand_raised.present
        assert not state.waving.present

    def test_wrist_at_hip_is_not_hand_raised(self):
        frames = _repeat({**_STANDING, 16: (0.60, 0.55)}, 12)
        state = _feed(ActivityRecognizer(), frames)
        assert not state.hand_raised.present

    def test_a_momentary_raise_does_not_count(self):
        """The wrist must be up for the WHOLE window — a hand tossed up while
        talking passes through the raised region without holding it."""
        frames = _repeat({**_STANDING, 16: (0.60, 0.55)}, 9)
        frames += _repeat({**_STANDING, 16: (0.60, 0.08)}, 3)  # up only at the end
        state = _feed(ActivityRecognizer(), frames)
        assert not state.hand_raised.present


class TestWaving:
    def _oscillate(self, count: int, amplitude: float = 0.12) -> list[dict]:
        """Raised right wrist swinging left-right once per 3 frames."""
        frames = []
        for step in range(count):
            x = 0.60 + amplitude * (1 if (step // 3) % 2 else -1)
            frames.append({**_STANDING, 16: (x, 0.08)})
        return frames

    def test_oscillating_raised_wrist_is_waving(self):
        state = _feed(ActivityRecognizer(), self._oscillate(12))
        assert state.waving.present

    def test_waving_reports_as_waving_not_hand_raised(self):
        """The more specific claim wins — a wave IS a raised hand, and
        reporting both would make a hand_raised rule fire on every wave."""
        state = _feed(ActivityRecognizer(), self._oscillate(12))
        assert not state.hand_raised.present

    def test_reach_out_and_back_is_not_waving(self):
        """Two wide swings — out, back — is the nearest innocent motion."""
        frames = _repeat({**_STANDING, 16: (0.48, 0.08)}, 4)
        frames += _repeat({**_STANDING, 16: (0.72, 0.08)}, 4)   # out
        frames += _repeat({**_STANDING, 16: (0.48, 0.08)}, 4)   # back
        state = _feed(ActivityRecognizer(), frames)
        assert not state.waving.present
        assert WAVE_MIN_SWINGS == 3  # the boundary this test encodes

    def test_oscillation_at_hip_height_is_not_waving(self):
        """Wiping a counter: same motion, wrong height."""
        frames = []
        for step in range(12):
            x = 0.60 + 0.12 * (1 if (step // 3) % 2 else -1)
            frames.append({**_STANDING, 16: (x, 0.55)})
        state = _feed(ActivityRecognizer(), frames)
        assert not state.waving.present

    def test_jitter_sized_oscillation_is_not_waving(self):
        """Landmark noise produces direction changes with near-zero travel."""
        frames = []
        for step in range(12):
            x = 0.60 + 0.005 * (1 if step % 2 else -1)
            frames.append({**_STANDING, 16: (x, 0.08)})
        state = _feed(ActivityRecognizer(), frames)
        assert not state.waving.present
        assert state.hand_raised.present  # still up, just not moving


class TestResting:
    def test_sitting_still_is_resting(self):
        """The chair/desk case — the whole reason resting is not lying_down."""
        frames = _repeat({**_SITTING, 15: (0.42, 0.50), 16: (0.58, 0.50)}, 12)
        state = _feed(ActivityRecognizer(), frames)
        assert state.sitting.present
        assert state.resting.present

    def test_standing_still_is_not_resting(self):
        """Standing is not a settled posture, however still."""
        frames = _repeat(_STANDING, 12)
        state = _feed(ActivityRecognizer(), frames)
        assert state.standing.present
        assert not state.resting.present

    def test_sitting_with_moving_hands_is_not_resting(self):
        """Typing, knitting, eating: seated and still at the hips, active at
        the wrists. Settled means the whole body has stopped."""
        frames = []
        for step in range(12):
            x = 0.40 + 0.15 * (step % 2)
            frames.append({**_SITTING, 16: (x, 0.50)})
        state = _feed(ActivityRecognizer(), frames)
        assert state.sitting.present
        assert not state.resting.present

    def test_head_droop_raises_resting_confidence(self):
        """Asleep at a desk: nose sunk to shoulder height."""
        upright = _repeat({**_SITTING, 0: (0.50, 0.25)}, 12)
        drooped = _repeat({**_SITTING, 0: (0.50, 0.34)}, 12)
        upright_state = _feed(ActivityRecognizer(), upright)
        drooped_state = _feed(ActivityRecognizer(), drooped)
        assert upright_state.resting.present and drooped_state.resting.present
        assert drooped_state.resting.confidence > upright_state.resting.confidence
        assert drooped_state.signals.get("head_drooped") is True

    def test_resting_needs_a_full_window(self):
        state = _feed(ActivityRecognizer(), _repeat(_SITTING, MIN_SAMPLES - 2))
        assert not state.ready

    def test_new_labels_flow_through_labels(self):
        """get_detections() consumes active() — the new attributes must be in it."""
        frames = _repeat(_SITTING, 12)
        state = _feed(ActivityRecognizer(), frames)
        assert "resting" in state.labels()


class TestHandRaisedMargin:
    def test_margin_is_measured_from_the_shoulder_not_the_head(self):
        """A drooped head must not turn a normal wrist position into 'raised':
        the margin is against the shoulder line, which droop does not move."""
        assert HAND_RAISED_MARGIN > 0


class TestFlailing:
    def _flail(self, count: int, amplitude: float = 0.15, height: float = 0.30) -> list[dict]:
        """A wrist reversing direction EVERY frame -- fast and chaotic, unlike
        a wave's period-3 rhythmic swing. height=0.30 sits BELOW the shoulder
        line (0.20) on purpose: flailing must not require a raised hand."""
        frames = []
        for step in range(count):
            x = 0.55 + amplitude * (1 if step % 2 else -1)
            frames.append({**_STANDING, 16: (x, height)})
        return frames

    def test_fast_chaotic_wrist_motion_at_any_height_is_flailing(self):
        state = _feed(ActivityRecognizer(), self._flail(12))
        assert state.flailing.present
        assert not state.hand_raised.present  # height=0.30 is below the shoulder line

    def test_flailing_reports_kinematic_signals(self):
        state = _feed(ActivityRecognizer(), self._flail(12))
        assert state.signals["right_wrist_peak_speed"] > 0
        assert state.signals["right_wrist_reversals"] >= FLAIL_MIN_REVERSALS

    def test_a_single_fast_reach_with_no_reversal_is_not_flailing(self):
        """High speed alone (one direction, no chaos) is the nearest innocent
        motion this must not fire on -- reaching quickly for something."""
        frames = _repeat({**_STANDING, 16: (0.40, 0.30)}, 3)
        frames += _repeat({**_STANDING, 16: (0.85, 0.30)}, 9)  # one fast, sustained move
        state = _feed(ActivityRecognizer(), frames)
        assert not state.flailing.present

    def test_slow_deliberate_back_and_forth_is_not_flailing(self):
        """Low speed, few reversals -- ordinary hand movement, not distress."""
        frames = []
        for step in range(12):
            x = 0.55 + 0.06 * (1 if (step // 4) % 2 else -1)
            frames.append({**_STANDING, 16: (x, 0.30)})
        state = _feed(ActivityRecognizer(), frames)
        assert not state.flailing.present

    def test_deliberate_wave_is_not_automatically_flailing(self):
        """A greeting wave (period-3, wide swings, 3 reversals) sits below
        flailing's higher reversal-count bar (FLAIL_MIN_REVERSALS=4) -- the two
        are related but distinct motions, not the same detector twice."""
        state = _feed(ActivityRecognizer(), TestWaving()._oscillate(12))
        assert state.waving.present
        assert not state.flailing.present

    def test_jitter_sized_oscillation_is_not_flailing(self):
        """Landmark noise produces direction changes with near-zero travel --
        must not cross the per-reversal amplitude floor."""
        frames = []
        for step in range(12):
            x = 0.55 + 0.005 * (1 if step % 2 else -1)
            frames.append({**_STANDING, 16: (x, 0.30)})
        state = _feed(ActivityRecognizer(), frames)
        assert not state.flailing.present

    def test_both_arms_flailing_boosts_confidence_over_one_arm(self):
        one_arm = self._flail(12)  # left wrist (15) stays unset/invisible -- one arm only
        both_arms = []
        for step in range(12):
            x = 0.55 + 0.15 * (1 if step % 2 else -1)
            lx = 0.40 + 0.15 * (1 if step % 2 else -1)
            both_arms.append({**_STANDING, 16: (x, 0.30), 15: (lx, 0.30)})
        one_state = _feed(ActivityRecognizer(), one_arm)
        both_state = _feed(ActivityRecognizer(), both_arms)
        assert one_state.flailing.present and both_state.flailing.present
        assert both_state.flailing.confidence > one_state.flailing.confidence

    def test_flailing_thresholds_are_configurable(self):
        """The product requirement is a configurable threshold, not a fixed
        constant -- a stricter override must reject the same motion the
        default thresholds accept."""
        strict = ActivityRecognizer(flail_min_reversals=50)
        state = _feed(strict, self._flail(12))
        assert not state.flailing.present

    def test_a_brief_pose_dropout_mid_flail_does_not_reset_the_window(self):
        """Fast, chaotic motion is exactly what causes MediaPipe to lose lock
        for a frame or two (motion blur) -- the same motion flailing exists to
        catch. A one-frame dropout must not wipe the rolling window, or the
        more genuinely distressed the motion, the less likely it is caught."""
        recognizer = ActivityRecognizer()
        frames = self._flail(12)
        for step, overrides in enumerate(frames[:6]):
            recognizer.update(_make_landmarks(overrides), now=step * 0.08)
        recognizer.pose_lost(now=6 * 0.08 + 0.05)  # one missed frame, well under the grace period
        state = None
        for step, overrides in enumerate(frames[6:], start=7):
            state = recognizer.update(_make_landmarks(overrides), now=step * 0.08)
        assert state.flailing.present

    def test_a_sustained_pose_loss_still_clears_the_window(self):
        """A dropout that OUTLASTS the grace period is the subject actually
        being gone (or a new person arriving) -- stale flailing evidence must
        not carry over to whoever the camera sees next."""
        recognizer = ActivityRecognizer()
        state = _feed(recognizer, self._flail(12))
        assert state.flailing.present  # sanity: baseline behavior still works

        last_t = 11 * 0.1
        recognizer.pose_lost(now=last_t + 0.1)          # loss begins
        recognizer.pose_lost(now=last_t + 0.1 + POSE_LOST_GRACE_S + 0.1)  # outlasts the grace period

        state = None
        for step in range(3):  # a few calm frames for whoever appears next
            t = last_t + 0.1 + POSE_LOST_GRACE_S + 0.1 + step * 0.1
            state = recognizer.update(_make_landmarks(_STANDING), now=t)
        assert not state.flailing.present
