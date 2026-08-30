"""Posture from body-pose landmarks: standing, sitting, lying down, on the floor.

The fall-detection module already runs MediaPipe Pose on every frame and reads
four `.y` values out of the 33 landmarks it gets back, discarding the rest. The
information needed to answer "is this person upright or on the ground" is
therefore already being computed and thrown away every frame — this module reads
it instead. No new model, no extra inference cost.

What it can and cannot tell you
-------------------------------
Torso angle from vertical separates upright from horizontal reliably, because it
is a large geometric difference. Sitting is genuinely harder: a seated torso is
near-vertical, so it is distinguished by how compressed the body is
(hip-to-ankle span against shoulder-to-hip span), which a heavily occluded lower
body will get wrong. That is why `sitting` reports lower confidence than
`lying_down`, and why the capability is registered as an approximation.

Crucially this is posture, not events. "On the floor for 30 seconds" is this
plus the temporal engine; "fell" is this plus the drop-velocity detector, since
lying down deliberately and collapsing look identical in a single frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

# MediaPipe Pose landmark indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

STANDING = "standing"
SITTING = "sitting"
LYING_DOWN = "lying_down"
UNKNOWN = "unknown"

# Torso angle from vertical. Below the first the body is upright; above the
# second it is horizontal. The gap between them is deliberately wide — a person
# leaning over a counter is neither standing nor lying, and guessing at 45° would
# produce exactly the flapping a safety alert must not do.
_UPRIGHT_MAX_DEG = 35.0
_HORIZONTAL_MIN_DEG = 60.0

# A landmark below this visibility is a guess by MediaPipe, not an observation.
_MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class Posture:
    """One frame's posture verdict."""

    label: str
    confidence: float
    torso_angle_deg: Optional[float] = None
    detail: str = ""


def _point(landmarks: Sequence, index: int) -> Optional[tuple[float, float]]:
    """(x, y) for a landmark, or None when it is not actually visible."""
    try:
        landmark = landmarks[index]
    except (IndexError, TypeError):
        return None
    visibility = getattr(landmark, "visibility", 1.0)
    if visibility is not None and visibility < _MIN_VISIBILITY:
        return None
    return float(landmark.x), float(landmark.y)


def _midpoint(a: Optional[tuple], b: Optional[tuple]) -> Optional[tuple[float, float]]:
    """Midpoint of two points, or whichever single one is visible.

    Falling back to one side matters: a person lying on their side occludes half
    their body, which is exactly when posture is most worth knowing.
    """
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


def torso_angle_from_vertical(landmarks: Sequence, aspect: float = 1.0) -> Optional[float]:
    """Degrees between the shoulder→hip axis and vertical. 0° is upright.

    `aspect` is width/height of the frame. Landmarks are normalised per-axis, so
    without correcting for it a 16:9 frame reports a lying body as less
    horizontal than it is.
    """
    shoulders = _midpoint(_point(landmarks, LEFT_SHOULDER), _point(landmarks, RIGHT_SHOULDER))
    hips = _midpoint(_point(landmarks, LEFT_HIP), _point(landmarks, RIGHT_HIP))
    if shoulders is None or hips is None:
        return None
    dx = (hips[0] - shoulders[0]) * aspect
    dy = hips[1] - shoulders[1]
    if dx == 0 and dy == 0:
        return None
    # atan2(|dx|, |dy|): 0 when the torso runs straight up-down, 90 when flat.
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def classify(landmarks: Sequence, aspect: float = 1.0) -> Posture:
    """Posture for one frame of pose landmarks."""
    angle = torso_angle_from_vertical(landmarks, aspect)
    if angle is None:
        return Posture(UNKNOWN, 0.0, None, "torso landmarks not visible")

    if angle >= _HORIZONTAL_MIN_DEG:
        # Confidence rises the closer to truly flat, so a borderline 60° reads as
        # less certain than an unambiguous 85°.
        confidence = min(1.0, 0.6 + (angle - _HORIZONTAL_MIN_DEG) / 60.0)
        return Posture(LYING_DOWN, round(confidence, 2), round(angle, 1),
                       "torso is horizontal")

    if angle > _UPRIGHT_MAX_DEG:
        return Posture(UNKNOWN, 0.3, round(angle, 1),
                       "torso is between upright and horizontal (leaning?)")

    # Upright: standing or sitting. Sitting compresses the legs relative to the
    # torso, which is the only cue available without depth.
    shoulders = _midpoint(_point(landmarks, LEFT_SHOULDER), _point(landmarks, RIGHT_SHOULDER))
    hips = _midpoint(_point(landmarks, LEFT_HIP), _point(landmarks, RIGHT_HIP))
    ankles = _midpoint(_point(landmarks, LEFT_ANKLE), _point(landmarks, RIGHT_ANKLE))
    knees = _midpoint(_point(landmarks, LEFT_KNEE), _point(landmarks, RIGHT_KNEE))
    lower = ankles or knees

    if lower is None or shoulders is None or hips is None:
        # Upright with the legs out of shot: standing is the safer call, and the
        # confidence says it is a partial observation.
        return Posture(STANDING, 0.5, round(angle, 1), "upright; lower body not visible")

    torso_span = abs(hips[1] - shoulders[1])
    leg_span = abs(lower[1] - hips[1])
    if torso_span <= 1e-6:
        return Posture(STANDING, 0.5, round(angle, 1), "upright")

    ratio = leg_span / torso_span
    # Standing legs are roughly as long as the torso or longer; sitting folds
    # them toward the camera and collapses that ratio.
    if ratio < 0.85:
        return Posture(SITTING, 0.6, round(angle, 1), "upright with folded legs")
    return Posture(STANDING, 0.8, round(angle, 1), "upright with extended legs")


def is_on_floor(posture: Posture, hips_y: Optional[float]) -> bool:
    """Lying down AND low in the frame.

    Lying alone is not being on the floor — someone in bed is horizontal too, and
    an elderly-monitoring alert that cannot tell a bed from a floor would fire
    every night. Frames where the hip position is unknown answer False rather
    than guessing.
    """
    if posture.label != LYING_DOWN or hips_y is None:
        return False
    return hips_y > 0.6  # lower 40% of the frame


def hip_height(landmarks: Sequence) -> Optional[float]:
    """Normalised y of the hips (0 top, 1 bottom), or None if not visible."""
    hips = _midpoint(_point(landmarks, LEFT_HIP), _point(landmarks, RIGHT_HIP))
    return None if hips is None else hips[1]
