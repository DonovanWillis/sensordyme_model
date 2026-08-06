"""Unit tests for the Eye Aspect Ratio (EAR) math, using synthetic landmark fixtures."""

import math

from modules.driver_attention import eye_aspect_ratio


def test_ear_open_eye_greater_than_closed_eye():
    open_eye = ((0.0, 0.0), (2.0, -5.0), (8.0, -5.0), (10.0, 0.0), (8.0, 5.0), (2.0, 5.0))
    closed_eye = ((0.0, 0.0), (2.0, -0.5), (8.0, -0.5), (10.0, 0.0), (8.0, 0.5), (2.0, 0.5))
    assert eye_aspect_ratio(*open_eye) > eye_aspect_ratio(*closed_eye)


def test_ear_matches_hand_computed_value():
    points = ((0.0, 0.0), (2.0, -3.0), (8.0, -3.0), (10.0, 0.0), (8.0, 3.0), (2.0, 3.0))
    ear = eye_aspect_ratio(*points)
    expected = (6.0 + 6.0) / (2 * 10.0)
    assert math.isclose(ear, expected, rel_tol=1e-9)


def test_ear_zero_horizontal_distance_returns_zero():
    p1 = p4 = (5.0, 5.0)
    ear = eye_aspect_ratio(p1, (5.0, 4.0), (5.0, 4.5), p4, (5.0, 5.5), (5.0, 6.0))
    assert ear == 0.0
