"""Posture classification from pose landmarks.

The safety-critical distinctions here are "upright vs on the ground" and
"lying on the floor vs lying in bed". Both are alert-worthy or not depending
entirely on getting them right, so they are tested at the boundaries.
"""

import pytest

from core import posture
from core.posture import LYING_DOWN, SITTING, STANDING, UNKNOWN


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def body(shoulder_y=0.3, hip_y=0.55, ankle_y=0.95, x=0.5, hip_x=None, visibility=1.0):
    """33 landmarks with only the ones posture reads set meaningfully."""
    marks = [Landmark(x, 0.1) for _ in range(33)]
    hip_x = x if hip_x is None else hip_x
    marks[posture.LEFT_SHOULDER] = Landmark(x - 0.05, shoulder_y, visibility)
    marks[posture.RIGHT_SHOULDER] = Landmark(x + 0.05, shoulder_y, visibility)
    marks[posture.LEFT_HIP] = Landmark(hip_x - 0.04, hip_y, visibility)
    marks[posture.RIGHT_HIP] = Landmark(hip_x + 0.04, hip_y, visibility)
    marks[posture.LEFT_KNEE] = Landmark(hip_x, (hip_y + ankle_y) / 2, visibility)
    marks[posture.RIGHT_KNEE] = Landmark(hip_x, (hip_y + ankle_y) / 2, visibility)
    marks[posture.LEFT_ANKLE] = Landmark(hip_x, ankle_y, visibility)
    marks[posture.RIGHT_ANKLE] = Landmark(hip_x, ankle_y, visibility)
    return marks


# --------------------------------------------------------------------- torso angle
def test_upright_torso_is_zero_degrees():
    angle = posture.torso_angle_from_vertical(body(shoulder_y=0.2, hip_y=0.6))
    assert angle == pytest.approx(0.0, abs=1.0)


def test_horizontal_torso_is_ninety_degrees():
    marks = body()
    marks[posture.LEFT_SHOULDER] = Landmark(0.2, 0.5)
    marks[posture.RIGHT_SHOULDER] = Landmark(0.2, 0.55)
    marks[posture.LEFT_HIP] = Landmark(0.7, 0.5)
    marks[posture.RIGHT_HIP] = Landmark(0.7, 0.55)
    assert posture.torso_angle_from_vertical(marks) == pytest.approx(90.0, abs=8.0)


def test_angle_is_none_when_the_torso_is_not_visible():
    marks = body(visibility=0.1)
    assert posture.torso_angle_from_vertical(marks) is None


def test_aspect_ratio_is_corrected():
    """Landmarks normalise per-axis, so a wide frame flattens angles unless corrected."""
    # A torso at a moderate diagonal: flat inputs saturate both readings at 90°
    # and would hide the correction entirely.
    marks = body()
    marks[posture.LEFT_SHOULDER] = Landmark(0.40, 0.30)
    marks[posture.RIGHT_SHOULDER] = Landmark(0.40, 0.32)
    marks[posture.LEFT_HIP] = Landmark(0.55, 0.60)
    marks[posture.RIGHT_HIP] = Landmark(0.55, 0.62)
    square = posture.torso_angle_from_vertical(marks, aspect=1.0)
    wide = posture.torso_angle_from_vertical(marks, aspect=16 / 9)
    assert 0 < square < 45
    assert wide > square, "a wide frame stretches horizontal offsets, tilting the torso further"


# --------------------------------------------------------------------- classification
def test_standing_is_upright_with_extended_legs():
    result = posture.classify(body(shoulder_y=0.2, hip_y=0.5, ankle_y=0.95))
    assert result.label == STANDING
    assert result.confidence >= 0.7


def test_sitting_is_upright_with_folded_legs():
    result = posture.classify(body(shoulder_y=0.3, hip_y=0.6, ankle_y=0.72))
    assert result.label == SITTING


def test_lying_down_is_detected_from_a_horizontal_torso():
    marks = body()
    marks[posture.LEFT_SHOULDER] = Landmark(0.2, 0.8)
    marks[posture.RIGHT_SHOULDER] = Landmark(0.2, 0.84)
    marks[posture.LEFT_HIP] = Landmark(0.7, 0.8)
    marks[posture.RIGHT_HIP] = Landmark(0.7, 0.84)
    result = posture.classify(marks)
    assert result.label == LYING_DOWN
    assert result.confidence >= 0.6


def test_leaning_is_reported_as_unknown_not_guessed():
    """A person bent over a counter is neither standing nor lying; guessing here
    is what produces an alert that flaps."""
    marks = body()
    marks[posture.LEFT_SHOULDER] = Landmark(0.35, 0.45)
    marks[posture.RIGHT_SHOULDER] = Landmark(0.35, 0.48)
    marks[posture.LEFT_HIP] = Landmark(0.60, 0.62)
    marks[posture.RIGHT_HIP] = Landmark(0.60, 0.65)
    assert posture.classify(marks).label == UNKNOWN


def test_invisible_torso_is_unknown_with_zero_confidence():
    result = posture.classify(body(visibility=0.1))
    assert result.label == UNKNOWN
    assert result.confidence == 0.0


def test_upright_without_visible_legs_prefers_standing_but_says_it_is_partial():
    marks = body()
    for index in (posture.LEFT_ANKLE, posture.RIGHT_ANKLE, posture.LEFT_KNEE, posture.RIGHT_KNEE):
        marks[index] = Landmark(0.5, 0.9, visibility=0.1)
    result = posture.classify(marks)
    assert result.label == STANDING
    assert result.confidence <= 0.5
    assert "not visible" in result.detail


# --------------------------------------------------------------------- on the floor
def test_lying_low_in_frame_is_on_the_floor():
    lying = posture.Posture(LYING_DOWN, 0.9)
    assert posture.is_on_floor(lying, hips_y=0.85)


def test_lying_high_in_frame_is_not_the_floor():
    """Someone in bed is horizontal too. An elderly alert that cannot tell a bed
    from a floor would fire every single night."""
    lying = posture.Posture(LYING_DOWN, 0.9)
    assert not posture.is_on_floor(lying, hips_y=0.35)


def test_standing_is_never_on_the_floor():
    assert not posture.is_on_floor(posture.Posture(STANDING, 0.9), hips_y=0.9)


def test_unknown_hip_position_answers_false_rather_than_guessing():
    assert not posture.is_on_floor(posture.Posture(LYING_DOWN, 0.9), hips_y=None)


# --------------------------------------------------------------------- capability wiring
def test_postures_are_registered_as_labels_rules_can_ask_for():
    from core.capabilities import default_registry

    registry = default_registry()
    if "lying_down" not in registry.labels():
        pytest.skip("mediapipe not installed in this environment")
    assert registry.assess("lying_down", "present").ok
    assert registry.assess("on_floor", "present").ok
