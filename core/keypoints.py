"""Body keypoints, model-agnostic.

The point of this module is that nothing above it knows which pose model
produced the skeleton. Today that is MediaPipe Pose on the CPU; the moment a
Hailo-compiled pose model is available it becomes an adapter that fills the same
`Skeleton`, and posture, activity, tracking and the rule engine keep working
untouched.

That indirection is the whole reason to write this now rather than after the
model swap. MediaPipe cannot run on the Hailo accelerator — no Hailo build
exists, and the Dataflow Compiler needs an ahead-of-time `.hef` — so moving pose
onto the HAT eventually means changing models. Code that reads
`landmark[23].y` is code that has to be rewritten when that happens; code that
asks for `skeleton.get(Part.LEFT_HIP)` does not.

Occlusion is handled by admitting it. A keypoint below its confidence threshold
is *absent*, not zero, and every geometric helper returns `None` when its inputs
are missing rather than computing an angle from a landmark the model was
guessing at. A missing part means "cannot say", never "not there".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Part(str, Enum):
    """Named body parts. Values are stable strings so they survive storage."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT = "left_foot_index"
    RIGHT_FOOT = "right_foot_index"


#: MediaPipe Pose index -> Part. Only the parts SensorDyme reasons about are
#: mapped; the rest of the 33 are real but unused, and mapping them would imply
#: a precision nothing downstream consumes.
MEDIAPIPE_POSE_MAP: dict[int, Part] = {
    0: Part.NOSE,
    2: Part.LEFT_EYE, 5: Part.RIGHT_EYE,
    7: Part.LEFT_EAR, 8: Part.RIGHT_EAR,
    11: Part.LEFT_SHOULDER, 12: Part.RIGHT_SHOULDER,
    13: Part.LEFT_ELBOW, 14: Part.RIGHT_ELBOW,
    15: Part.LEFT_WRIST, 16: Part.RIGHT_WRIST,
    23: Part.LEFT_HIP, 24: Part.RIGHT_HIP,
    25: Part.LEFT_KNEE, 26: Part.RIGHT_KNEE,
    27: Part.LEFT_ANKLE, 28: Part.RIGHT_ANKLE,
    29: Part.LEFT_HEEL, 30: Part.RIGHT_HEEL,
    31: Part.LEFT_FOOT, 32: Part.RIGHT_FOOT,
}

#: Bones, for drawing a skeleton and for reasoning about limb geometry.
SKELETON: tuple[tuple[Part, Part], ...] = (
    (Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER),
    (Part.LEFT_SHOULDER, Part.LEFT_ELBOW), (Part.LEFT_ELBOW, Part.LEFT_WRIST),
    (Part.RIGHT_SHOULDER, Part.RIGHT_ELBOW), (Part.RIGHT_ELBOW, Part.RIGHT_WRIST),
    (Part.LEFT_SHOULDER, Part.LEFT_HIP), (Part.RIGHT_SHOULDER, Part.RIGHT_HIP),
    (Part.LEFT_HIP, Part.RIGHT_HIP),
    (Part.LEFT_HIP, Part.LEFT_KNEE), (Part.LEFT_KNEE, Part.LEFT_ANKLE),
    (Part.RIGHT_HIP, Part.RIGHT_KNEE), (Part.RIGHT_KNEE, Part.RIGHT_ANKLE),
    (Part.LEFT_ANKLE, Part.LEFT_HEEL), (Part.LEFT_HEEL, Part.LEFT_FOOT),
    (Part.RIGHT_ANKLE, Part.RIGHT_HEEL), (Part.RIGHT_HEEL, Part.RIGHT_FOOT),
)

#: Below this a keypoint is treated as absent. MediaPipe reports a position for
#: occluded joints with low visibility — using those silently is how an arm
#: behind a back becomes a confident elbow angle.
MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Keypoint:
    """One body part in normalised (0..1) frame coordinates."""

    part: Part
    x: float
    y: float
    confidence: float = 1.0

    @property
    def visible(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def distance_to(self, other: "Keypoint", aspect: float = 1.0) -> float:
        return math.hypot((self.x - other.x) * aspect, self.y - other.y)


@dataclass
class Skeleton:
    """One person's keypoints, plus the geometry worth deriving from them.

    Every accessor returns `None` rather than a default when the parts it needs
    are not visible. A safety system that computes a knee angle from a knee it
    cannot see produces confident nonsense, which is worse than silence.
    """

    keypoints: dict[Part, Keypoint] = field(default_factory=dict)
    aspect: float = 1.0

    # ------------------------------------------------------------------ access
    def get(self, part: Part) -> Optional[Keypoint]:
        keypoint = self.keypoints.get(part)
        return keypoint if keypoint and keypoint.visible else None

    def has(self, *parts: Part) -> bool:
        return all(self.get(p) is not None for p in parts)

    def visible_parts(self) -> list[Part]:
        return [p for p, k in self.keypoints.items() if k.visible]

    @property
    def coverage(self) -> float:
        """Fraction of known parts currently visible — an occlusion measure."""
        if not self.keypoints:
            return 0.0
        return round(len(self.visible_parts()) / len(self.keypoints), 2)

    def midpoint(self, a: Part, b: Part) -> Optional[tuple[float, float]]:
        """Midpoint of two parts, or the single visible one.

        Falling back to one side matters: a person lying on their side occludes
        half their body, which is exactly when this is most worth knowing.
        """
        first, second = self.get(a), self.get(b)
        if first and second:
            return ((first.x + second.x) / 2, (first.y + second.y) / 2)
        only = first or second
        return (only.x, only.y) if only else None

    # ------------------------------------------------------------------ landmarks
    def shoulders(self) -> Optional[tuple[float, float]]:
        return self.midpoint(Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER)

    def hips(self) -> Optional[tuple[float, float]]:
        return self.midpoint(Part.LEFT_HIP, Part.RIGHT_HIP)

    def ankles(self) -> Optional[tuple[float, float]]:
        return self.midpoint(Part.LEFT_ANKLE, Part.RIGHT_ANKLE)

    def torso_length(self) -> Optional[float]:
        """Shoulder-to-hip distance: the scale reference everything else uses.

        Measuring motion in torso lengths rather than pixels is what makes a
        threshold mean the same thing near and far from the camera.
        """
        shoulders, hips = self.shoulders(), self.hips()
        if shoulders is None or hips is None:
            return None
        length = math.hypot((hips[0] - shoulders[0]) * self.aspect, hips[1] - shoulders[1])
        return length if length > 1e-4 else None

    def center_of_mass(self) -> Optional[tuple[float, float]]:
        """Approximate CoM, weighted toward the torso where most mass sits."""
        shoulders, hips = self.shoulders(), self.hips()
        if shoulders is None or hips is None:
            return None
        return ((shoulders[0] + hips[0]) / 2, shoulders[1] * 0.4 + hips[1] * 0.6)

    def height_in_frame(self) -> Optional[float]:
        """Head-to-ankle extent. Collapses when someone lies down, which is a
        signal in itself."""
        head = self.get(Part.NOSE)
        ankles = self.ankles()
        if head is None or ankles is None:
            return None
        return abs(ankles[1] - head.y)

    # ------------------------------------------------------------------ angles
    def joint_angle(self, a: Part, b: Part, c: Part) -> Optional[float]:
        """Interior angle at `b`, in degrees. 180 is a straight limb."""
        first, middle, last = self.get(a), self.get(b), self.get(c)
        if not (first and middle and last):
            return None
        v1 = ((first.x - middle.x) * self.aspect, first.y - middle.y)
        v2 = ((last.x - middle.x) * self.aspect, last.y - middle.y)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return round(math.degrees(math.acos(cosine)), 1)

    def left_knee_angle(self) -> Optional[float]:
        return self.joint_angle(Part.LEFT_HIP, Part.LEFT_KNEE, Part.LEFT_ANKLE)

    def right_knee_angle(self) -> Optional[float]:
        return self.joint_angle(Part.RIGHT_HIP, Part.RIGHT_KNEE, Part.RIGHT_ANKLE)

    def left_elbow_angle(self) -> Optional[float]:
        return self.joint_angle(Part.LEFT_SHOULDER, Part.LEFT_ELBOW, Part.LEFT_WRIST)

    def right_elbow_angle(self) -> Optional[float]:
        return self.joint_angle(Part.RIGHT_SHOULDER, Part.RIGHT_ELBOW, Part.RIGHT_WRIST)

    def torso_angle_from_vertical(self) -> Optional[float]:
        """0 when upright, 90 when horizontal. The core posture signal."""
        shoulders, hips = self.shoulders(), self.hips()
        if shoulders is None or hips is None:
            return None
        dx = (hips[0] - shoulders[0]) * self.aspect
        dy = hips[1] - shoulders[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return round(math.degrees(math.atan2(abs(dx), abs(dy))), 1)

    def stance_width(self) -> Optional[float]:
        """Distance between the ankles, in torso lengths."""
        left, right = self.get(Part.LEFT_ANKLE), self.get(Part.RIGHT_ANKLE)
        torso = self.torso_length()
        if not (left and right and torso):
            return None
        return round(left.distance_to(right, self.aspect) / torso, 2)

    def geometry(self) -> dict:
        """Every derived measurement that is currently computable.

        Absent keys mean "not visible enough to say" — the caller can tell the
        difference between an unmeasurable angle and a zero one.
        """
        values = {
            "torso_angle_deg": self.torso_angle_from_vertical(),
            "left_knee_deg": self.left_knee_angle(),
            "right_knee_deg": self.right_knee_angle(),
            "left_elbow_deg": self.left_elbow_angle(),
            "right_elbow_deg": self.right_elbow_angle(),
            "stance_width_torso": self.stance_width(),
            "height_in_frame": self.height_in_frame(),
            "coverage": self.coverage,
        }
        return {k: v for k, v in values.items() if v is not None}

    def as_dict(self) -> dict:
        return {
            "keypoints": {
                part.value: {"x": round(k.x, 4), "y": round(k.y, 4),
                             "confidence": round(k.confidence, 2)}
                for part, k in self.keypoints.items()
            },
            "geometry": self.geometry(),
        }


def from_mediapipe(landmarks, aspect: float = 1.0) -> Skeleton:
    """Adapt MediaPipe Pose landmarks into a Skeleton.

    The only place in the codebase that knows MediaPipe's indices. A Hailo pose
    model gets its own adapter here and nothing downstream changes.
    """
    keypoints: dict[Part, Keypoint] = {}
    for index, part in MEDIAPIPE_POSE_MAP.items():
        try:
            landmark = landmarks[index]
        except (IndexError, TypeError):
            continue
        keypoints[part] = Keypoint(
            part=part,
            x=float(landmark.x),
            y=float(landmark.y),
            confidence=float(getattr(landmark, "visibility", 1.0)),
        )
    return Skeleton(keypoints=keypoints, aspect=aspect)


def from_detection_box(box: tuple[float, float, float, float], aspect: float = 1.0) -> Skeleton:
    """Coarse Skeleton for a plain detector box, no pose model involved.

    core/tracking.py's PersonTracker matches on `center_of_mass()` and
    `torso_length()`, which only ever read the shoulder and hip keypoints. A
    module with nothing but object-detector boxes (no MediaPipe -- see
    modules/occupancy.py, which deliberately doesn't run pose to stay the
    cheap always-on detector) can still reuse that same torso-length-
    normalized matcher instead of a second implementation of it, by fabricating
    just those two landmark pairs from ordinary standing-body proportions
    (shoulders ~15% down from the top of the box, hips ~50% down).

    Every other Part is left unset on purpose. `Skeleton.get()` already treats
    an unset part as "cannot say" rather than a default, so a knee angle or
    anything else pose-specific correctly comes back None here -- this
    fabricates only the two measurements it was built to supply, not a body.
    """
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return Skeleton(aspect=aspect)
    center_x = (x1 + x2) / 2.0
    shoulder_y = y1 + height * 0.15
    hip_y = y1 + height * 0.50
    keypoints = {
        Part.LEFT_SHOULDER: Keypoint(Part.LEFT_SHOULDER, center_x, shoulder_y),
        Part.RIGHT_SHOULDER: Keypoint(Part.RIGHT_SHOULDER, center_x, shoulder_y),
        Part.LEFT_HIP: Keypoint(Part.LEFT_HIP, center_x, hip_y),
        Part.RIGHT_HIP: Keypoint(Part.RIGHT_HIP, center_x, hip_y),
    }
    return Skeleton(keypoints=keypoints, aspect=aspect)


def from_coco_keypoints(points: Iterable, aspect: float = 1.0,
                        order: Optional[list[Part]] = None) -> Skeleton:
    """Adapt a COCO-17 keypoint array — the format Hailo pose models emit.

    Present so the Hailo path is a drop-in when a compiled pose model lands,
    rather than a refactor discovered at that moment.
    """
    order = order or [
        Part.NOSE, Part.LEFT_EYE, Part.RIGHT_EYE, Part.LEFT_EAR, Part.RIGHT_EAR,
        Part.LEFT_SHOULDER, Part.RIGHT_SHOULDER, Part.LEFT_ELBOW, Part.RIGHT_ELBOW,
        Part.LEFT_WRIST, Part.RIGHT_WRIST, Part.LEFT_HIP, Part.RIGHT_HIP,
        Part.LEFT_KNEE, Part.RIGHT_KNEE, Part.LEFT_ANKLE, Part.RIGHT_ANKLE,
    ]
    keypoints: dict[Part, Keypoint] = {}
    for part, row in zip(order, points):
        try:
            x, y = float(row[0]), float(row[1])
            confidence = float(row[2]) if len(row) > 2 else 1.0
        except (TypeError, ValueError, IndexError):
            continue
        keypoints[part] = Keypoint(part=part, x=x, y=y, confidence=confidence)
    return Skeleton(keypoints=keypoints, aspect=aspect)
