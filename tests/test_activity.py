"""Activity recognition, including the cases that must NOT fire.

Half of these tests are negative. An activity engine that finds running
everywhere is worse than none: every alert it raises trains the owner to ignore
the next one, and the one that matters is always the next one.
"""

import pytest

from core import posture
from core.activity import (
    JUMP_ANKLE_RISE,
    MIN_SAMPLES,
    RUN_SPEED,
    ActivityRecognizer,
    ActivityState,
)


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def pose(hip_x=0.5, hip_y=0.55, ankle_y=0.95, shoulder_y=0.30, visibility=1.0):
    """33 landmarks with the ones this module reads set meaningfully."""
    marks = [Landmark(hip_x, hip_y) for _ in range(33)]
    marks[posture.LEFT_SHOULDER] = Landmark(hip_x - 0.05, shoulder_y, visibility)
    marks[posture.RIGHT_SHOULDER] = Landmark(hip_x + 0.05, shoulder_y, visibility)
    marks[posture.LEFT_HIP] = Landmark(hip_x - 0.04, hip_y, visibility)
    marks[posture.RIGHT_HIP] = Landmark(hip_x + 0.04, hip_y, visibility)
    marks[posture.LEFT_KNEE] = Landmark(hip_x, (hip_y + ankle_y) / 2, visibility)
    marks[posture.RIGHT_KNEE] = Landmark(hip_x, (hip_y + ankle_y) / 2, visibility)
    marks[posture.LEFT_ANKLE] = Landmark(hip_x, ankle_y, visibility)
    marks[posture.RIGHT_ANKLE] = Landmark(hip_x, ankle_y, visibility)
    return marks


def drive(recognizer, frames, step=0.05):
    """Feed a list of (hip_x, hip_y, ankle_y) through the recognizer."""
    state = None
    for index, (hip_x, hip_y, ankle_y) in enumerate(frames):
        state = recognizer.update(pose(hip_x=hip_x, hip_y=hip_y, ankle_y=ankle_y),
                                  now=1000.0 + index * step)
    return state


# Torso here is |shoulder_y - hip_y| = 0.25, so a displacement of 0.25 is one
# torso length. Every scenario below is written in those terms.
TORSO = 0.25


def still(n=12):
    return [(0.5, 0.55, 0.95)] * n


def walking(n=12, per_frame=0.012):
    """~0.24 torso/s at 20fps — deliberately under the run threshold."""
    return [(0.5 + i * per_frame, 0.55, 0.95) for i in range(n)]


def running(n=12, per_frame=0.05):
    """Fast, with the vertical bounce a run has and a walk does not."""
    return [
        (0.5 + i * per_frame, 0.55 - (0.03 if i % 2 else 0.0), 0.95)
        for i in range(n)
    ]


def jumping(n=12):
    """Still, then hips and both ankles rise together, then land."""
    frames = [(0.5, 0.55, 0.95)] * 4
    frames += [(0.5, 0.55 - 0.06 * k, 0.95 - 0.05 * k) for k in (1, 2, 3)]
    frames += [(0.5, 0.55, 0.95)] * 5
    return frames[:n]


# --------------------------------------------------------------------- readiness
def test_nothing_is_claimed_before_enough_frames():
    recognizer = ActivityRecognizer()
    state = drive(recognizer, still(MIN_SAMPLES - 2))
    assert not state.ready
    assert not state.running.present and not state.jumping.present


def test_an_invisible_torso_reports_not_ready_rather_than_guessing():
    """Without the torso there is no scale reference, so nothing is measurable."""
    recognizer = ActivityRecognizer()
    state = recognizer.update(pose(visibility=0.1), now=1000.0)
    assert not state.ready
    assert "torso" in state.signals.get("reason", "")


# --------------------------------------------------------------------- positives
def test_running_is_detected_from_speed_and_bounce():
    state = drive(ActivityRecognizer(), running())
    assert state.running.present
    assert state.running.confidence >= 0.6
    assert not state.walking.present


def test_walking_is_detected_and_is_not_running():
    state = drive(ActivityRecognizer(), walking())
    assert state.walking.present
    assert not state.running.present


def test_jumping_is_detected_from_hips_and_ankles_rising_together():
    state = drive(ActivityRecognizer(), jumping())
    assert state.jumping.present
    assert state.jumping.confidence >= 0.5


def test_standing_still_is_standing():
    state = drive(ActivityRecognizer(), still())
    assert state.standing.present
    assert not state.walking.present and not state.running.present


# --------------------------------------------------------------------- negatives
def test_walking_normally_does_not_trigger_running():
    """Test 11 from the brief."""
    state = drive(ActivityRecognizer(), walking())
    assert not state.running.present


def test_standing_still_does_not_trigger_jumping():
    state = drive(ActivityRecognizer(), still())
    assert not state.jumping.present


def test_arm_movement_does_not_trigger_a_jump():
    """Test 12: the arms are not read at all, so waving cannot register."""
    recognizer = ActivityRecognizer()
    state = None
    for index in range(12):
        marks = pose()
        marks[15] = Landmark(0.2 + index * 0.05, 0.3)   # left wrist flailing
        marks[16] = Landmark(0.8 - index * 0.05, 0.3)   # right wrist
        state = recognizer.update(marks, now=1000.0 + index * 0.05)
    assert not state.jumping.present
    assert not state.running.present


def test_rising_hips_alone_are_not_a_jump():
    """Standing up quickly raises the hips as fast as a jump does. Requiring the
    ankles to leave the ground too is what separates them."""
    frames = [(0.5, 0.55, 0.95)] * 4 + [(0.5, 0.55 - 0.06 * k, 0.95) for k in (1, 2, 3)]
    frames += [(0.5, 0.37, 0.95)] * 5
    state = drive(ActivityRecognizer(), frames)
    assert not state.jumping.present


def test_speed_without_bounce_is_not_running():
    """A person on a scooter, or a panning camera, moves fast without gait."""
    frames = [(0.5 + i * 0.05, 0.55, 0.95) for i in range(12)]  # no vertical motion
    state = drive(ActivityRecognizer(), frames)
    assert not state.running.present


def test_bounce_without_speed_is_not_running():
    """Bobbing in place is not a run."""
    frames = [(0.5, 0.55 - (0.03 if i % 2 else 0.0), 0.95) for i in range(12)]
    state = drive(ActivityRecognizer(), frames)
    assert not state.running.present


# --------------------------------------------------------------------- scale invariance
def test_the_same_motion_reads_the_same_near_and_far():
    """A person across the room moves a fraction of the pixels doing exactly the
    same thing. Measuring in torso lengths is what removes distance."""
    near = drive(ActivityRecognizer(), running())

    # Half the size, half the displacement — the same gait, further away.
    far_frames = [
        (0.5 + i * 0.025, 0.55 - (0.015 if i % 2 else 0.0), 0.75) for i in range(12)
    ]
    recognizer = ActivityRecognizer()
    state = None
    for index, (hip_x, hip_y, ankle_y) in enumerate(far_frames):
        marks = pose(hip_x=hip_x, hip_y=hip_y, ankle_y=ankle_y)
        marks[posture.LEFT_SHOULDER] = Landmark(hip_x - 0.025, hip_y - 0.125)
        marks[posture.RIGHT_SHOULDER] = Landmark(hip_x + 0.025, hip_y - 0.125)
        state = recognizer.update(marks, now=1000.0 + index * 0.05)

    assert near.running.present
    assert state.running.present, "the same gait at distance must still read as running"


# --------------------------------------------------------------------- multi-attribute
def test_attributes_are_independent_not_one_label():
    """The brief's requirement: a person can be several things at once."""
    state = drive(ActivityRecognizer(), running())
    payload = state.as_dict()
    assert set(payload) >= {"standing", "walking", "running", "jumping", "lying_down"}
    assert payload["running"]["present"] is True
    assert payload["jumping"]["present"] is False


def test_moving_upright_is_not_reported_as_standing():
    state = drive(ActivityRecognizer(), running())
    assert state.running.present
    assert not state.standing.present, "standing means still"


def test_signals_explain_the_verdict():
    """Evidence travels with the event, so an alert can say why."""
    state = drive(ActivityRecognizer(), running())
    assert "speed_torso_per_s" in state.signals
    assert "bounce_torso" in state.signals


def test_reset_forgets_history():
    recognizer = ActivityRecognizer()
    drive(recognizer, running())
    recognizer.reset()
    state = recognizer.update(pose(), now=2000.0)
    assert not state.ready
