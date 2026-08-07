"""Activity recognition from pose sequences: standing, walking, running, jumping.

Built on the MediaPipe Pose landmarks the fall detector already computes every
frame, so this adds temporal reasoning rather than another model. No new
inference cost, which is the only way it fits alongside vision on a Pi.

Three ideas carry the whole module:

**Scale invariance.** All motion is measured in *torso lengths per second*, never
pixels or normalised units. A person across the room moves a fraction of the
pixels of someone near the camera while doing exactly the same thing, so a
pixel threshold means "running, if you stand here". Dividing by the
shoulder-to-hip distance removes distance and body size together.

**Attributes, not a single state.** A person can be running and wearing glasses
and about to fall. `ActivityState` reports every attribute independently with
its own confidence, because forcing one label per frame throws away exactly the
combinations that matter.

**Sequences, not frames.** Walking and running look identical in a still image;
what separates them is cadence and vertical oscillation over ~1 second. Jumping
is only visible as a trajectory. So the classifier reads a rolling window and
refuses to answer until it has one.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from core import posture

# MediaPipe Pose indices this module reads.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# How much history to reason over. About a second at 15-30fps: long enough for
# two footfalls, short enough that the answer still describes *now*.
WINDOW_S = 1.2
MIN_SAMPLES = 6

# Thresholds in torso-lengths per second, from the geometry of human gait
# rather than tuning against one recording. A walking hip travels roughly half a
# torso length per second; running is markedly faster and adds vertical bounce.
WALK_SPEED = 0.35
RUN_SPEED = 1.10
RUN_BOUNCE = 0.10        # vertical hip oscillation amplitude, torso lengths
JUMP_RISE = 0.18         # upward hip displacement that is not gait bounce
JUMP_ANKLE_RISE = 0.12   # both ankles leaving their baseline together

_MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class Attribute:
    """One thing believed about a person, with how strongly."""

    name: str
    present: bool
    confidence: float

    def __bool__(self) -> bool:
        return self.present


@dataclass
class ActivityState:
    """Everything believed about one person this instant.

    Attributes are independent on purpose: running and jumping are both false
    while standing, and running and glasses can both be true at once.
    """

    standing: Attribute = field(default_factory=lambda: Attribute("standing", False, 0.0))
    sitting: Attribute = field(default_factory=lambda: Attribute("sitting", False, 0.0))
    walking: Attribute = field(default_factory=lambda: Attribute("walking", False, 0.0))
    running: Attribute = field(default_factory=lambda: Attribute("running", False, 0.0))
    jumping: Attribute = field(default_factory=lambda: Attribute("jumping", False, 0.0))
    lying_down: Attribute = field(default_factory=lambda: Attribute("lying_down", False, 0.0))
    #: Why the classifier believes it — carried into alerts as evidence.
    signals: dict = field(default_factory=dict)
    ready: bool = False

    def active(self) -> list[Attribute]:
        return [a for a in (self.standing, self.sitting, self.walking, self.running,
                            self.jumping, self.lying_down) if a.present]

    def labels(self) -> list[str]:
        return [a.name for a in self.active()]

    def as_dict(self) -> dict:
        return {
            a.name: {"present": a.present, "confidence": round(a.confidence, 2)}
            for a in (self.standing, self.sitting, self.walking, self.running,
                      self.jumping, self.lying_down)
        }


@dataclass
class _Sample:
    at: float
    hip_x: float
    hip_y: float
    ankle_y: float
    torso: float


def _point(landmarks: Sequence, index: int) -> Optional[tuple[float, float]]:
    try:
        landmark = landmarks[index]
    except (IndexError, TypeError):
        return None
    if getattr(landmark, "visibility", 1.0) < _MIN_VISIBILITY:
        return None
    return float(landmark.x), float(landmark.y)


def _mid(a, b):
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


class ActivityRecognizer:
    """Rolling-window activity classifier over pose landmarks."""

    def __init__(self, window_s: float = WINDOW_S, aspect: float = 1.0) -> None:
        self.window_s = window_s
        self.aspect = aspect
        self._samples: deque[_Sample] = deque()
        self._baseline_ankle: Optional[float] = None

    def reset(self) -> None:
        """Forget history — the subject left, or a new one arrived."""
        self._samples.clear()
        self._baseline_ankle = None

    def update(self, landmarks: Sequence, now: Optional[float] = None,
               aspect: Optional[float] = None) -> ActivityState:
        """Add one frame of landmarks and classify."""
        now = time.monotonic() if now is None else now
        if aspect is not None:
            self.aspect = aspect

        shoulders = _mid(_point(landmarks, LEFT_SHOULDER), _point(landmarks, RIGHT_SHOULDER))
        hips = _mid(_point(landmarks, LEFT_HIP), _point(landmarks, RIGHT_HIP))
        ankles = _mid(_point(landmarks, LEFT_ANKLE), _point(landmarks, RIGHT_ANKLE))

        if shoulders is None or hips is None:
            # Losing the torso means losing the scale reference, so nothing here
            # is measurable. Report not-ready rather than guess.
            return ActivityState(ready=False, signals={"reason": "torso not visible"})

        torso = math.hypot((hips[0] - shoulders[0]) * self.aspect, hips[1] - shoulders[1])
        if torso < 1e-4:
            return ActivityState(ready=False, signals={"reason": "degenerate torso"})

        self._samples.append(_Sample(
            at=now, hip_x=hips[0] * self.aspect, hip_y=hips[1],
            ankle_y=ankles[1] if ankles else hips[1], torso=torso,
        ))
        while self._samples and now - self._samples[0].at > self.window_s:
            self._samples.popleft()

        return self._classify(landmarks, now)

    # ------------------------------------------------------------------ classification
    def _classify(self, landmarks: Sequence, now: float) -> ActivityState:
        samples = list(self._samples)
        state = ActivityState()

        # Posture is a per-frame property and is answerable immediately.
        stance = posture.classify(landmarks, aspect=self.aspect)
        if stance.label == posture.LYING_DOWN:
            state.lying_down = Attribute("lying_down", True, stance.confidence)
        elif stance.label == posture.SITTING:
            state.sitting = Attribute("sitting", True, stance.confidence)
        elif stance.label == posture.STANDING:
            state.standing = Attribute("standing", True, stance.confidence)

        if len(samples) < MIN_SAMPLES:
            state.ready = False
            state.signals["reason"] = f"collecting motion ({len(samples)}/{MIN_SAMPLES} frames)"
            return state

        elapsed = samples[-1].at - samples[0].at
        if elapsed <= 0:
            state.ready = False
            return state

        torso = sum(s.torso for s in samples) / len(samples)

        # Horizontal hip travel, in torso lengths per second. Total path rather
        # than net displacement, so pacing back and forth still reads as motion.
        path = sum(
            abs(b.hip_x - a.hip_x) for a, b in zip(samples, samples[1:])
        ) / torso
        speed = path / elapsed

        # Vertical oscillation: the bounce that separates a run from a walk.
        heights = [s.hip_y for s in samples]
        bounce = (max(heights) - min(heights)) / torso

        state.signals.update({
            "speed_torso_per_s": round(speed, 2),
            "bounce_torso": round(bounce, 2),
            "window_s": round(elapsed, 2),
        })
        state.ready = True

        # --- jumping ------------------------------------------------------
        # A jump is the hips rising sharply while both ankles leave their
        # baseline. Requiring the ankles too is what separates a jump from
        # standing up quickly, which raises the hips just as fast.
        rise = (max(heights) - heights[-1]) / torso  # y grows downward
        hip_rise = (heights[0] - min(heights)) / torso
        ankle_baseline = max(s.ankle_y for s in samples)
        ankle_rise = (ankle_baseline - min(s.ankle_y for s in samples)) / torso
        state.signals["hip_rise_torso"] = round(hip_rise, 2)
        state.signals["ankle_rise_torso"] = round(ankle_rise, 2)

        if hip_rise >= JUMP_RISE and ankle_rise >= JUMP_ANKLE_RISE:
            confidence = min(0.99, 0.55 + (hip_rise - JUMP_RISE) * 1.5)
            state.jumping = Attribute("jumping", True, round(confidence, 2))

        # --- running / walking --------------------------------------------
        # Running needs both speed AND bounce. Speed alone is satisfied by a
        # camera pan or a person on a scooter; bounce alone by someone bobbing
        # in place. Requiring both is the main false-positive control here.
        if speed >= RUN_SPEED and bounce >= RUN_BOUNCE:
            confidence = min(0.99, 0.6 + (speed - RUN_SPEED) * 0.25)
            state.running = Attribute("running", True, round(confidence, 2))
        elif speed >= WALK_SPEED:
            confidence = min(0.95, 0.5 + (speed - WALK_SPEED) * 0.5)
            state.walking = Attribute("walking", True, round(confidence, 2))

        # Moving and upright is not "standing": standing means still.
        if (state.walking or state.running or state.jumping) and state.standing.present:
            state.standing = Attribute("standing", False, 0.0)
        return state
