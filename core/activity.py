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
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Optional

from core import posture

# MediaPipe Pose indices this module reads.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
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

# --- gestures -----------------------------------------------------------
# A raised hand is a wrist held above its own shoulder. Measured against the
# SHOULDER rather than the head, because the head can droop (someone asleep at
# a desk still has a head) while a wrist above the shoulder is a deliberate arm
# position almost nowhere else in ordinary life.
HAND_RAISED_MARGIN = 0.10   # wrist above shoulder by this many torso lengths
# A wave is a raised hand OSCILLATING sideways: direction reversals, not just
# motion. A wave needs at least three wide swings within the window
# (out-back-out) — two would also match reaching toward something and pulling
# straight back, which is the nearest innocent motion this must not fire on.
# Each swing must be wide enough that landmark jitter can't fake it.
WAVE_MIN_SWINGS = 3
WAVE_MIN_SWING = 0.25       # horizontal wrist travel per swing, torso lengths

# Flailing: fast, chaotic wrist motion -- no "raised above the shoulder"
# precondition (unlike waving; distress/flailing happens at any height), judged
# on velocity, acceleration, and how often the wrist reverses direction rather
# than on total displacement in one direction. A deliberate wave is a WIDE,
# roughly rhythmic swing (few reversals, each one big); flailing is CHAOTIC —
# many short, fast reversals with no consistent rhythm. Requiring both a speed
# floor AND a reversal-count floor is what keeps a single fast reach (one big
# motion, zero reversals) and an ordinary gesture (a couple of slow reversals)
# from crossing this bar.
FLAIL_MIN_PEAK_SPEED = 1.6      # torso-lengths/s, peak instantaneous wrist speed
FLAIL_MIN_PEAK_ACCEL = 40.0     # torso-lengths/s^2, peak instantaneous wrist acceleration --
# an alternative gate to peak speed: a sharp, jerky direction change can flag
# flailing even in a window where peak speed alone is borderline. Measured on
# velocity VECTORS: a reversal at speed v within time T is 2v/T. The old value
# (6.0) was tuned against differences of scalar SPEEDS, which read a reversal
# at constant speed as zero; on vectors 6.0 would be cleared by any reversal
# at all. 40 torso/s^2 (~20 m/s^2, ~2 g, for a ~0.5 m torso) sits above an
# ordinary hand reversal (a 1 Hz, 10 cm wave peaks near 8) and is reached by
# a snap reversal at ~1.3 torso/s on 15 fps samples -- "borderline" speed.
FLAIL_MIN_REVERSALS = 4         # direction reversals within the window -- a higher bar than a
# wave's 3, because flailing is chaotic (many quick reversals) rather than a
# rhythmic swing
FLAIL_MIN_REVERSAL_AMPLITUDE = 0.10  # torso-lengths of 2D travel per reversal leg -- smaller
# than a wave's 0.25 (flailing is short, fast segments, not one wide swing), but
# well above landmark jitter

# --- stillness (the evidence "resting" is built from) -------------------
# Below this hip speed (torso lengths/s) across the whole window the person is
# not going anywhere; the wrist bound is looser because people shift their
# hands while dozing without being "active".
STILL_MAX_HIP_SPEED = 0.08
STILL_MAX_WRIST_SPEED = 0.30
# Head droop: the nose sinking toward shoulder height. Upright and alert, the
# nose sits well above the shoulder line; asleep at a desk it approaches or
# passes it. Expressed as nose-above-shoulders distance in torso lengths.
HEAD_DROOP_MAX_CLEARANCE = 0.18

_MIN_VISIBILITY = 0.5

# How long a momentary pose-tracking dropout is tolerated before the rolling
# window is actually cleared, matching core/tracking.py's lost_grace_s
# reasoning (occlusion is handled by patience, not an instant reset). Fast,
# chaotic motion -- exactly what flailing detection exists to catch -- is also
# a common cause of MediaPipe Pose losing lock for a frame or two (motion
# blur), so clearing the window on the very first missing frame would make it
# least reliable exactly when genuine distress motion needs it most.
POSE_LOST_GRACE_S = 0.5


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
    hand_raised: Attribute = field(default_factory=lambda: Attribute("hand_raised", False, 0.0))
    waving: Attribute = field(default_factory=lambda: Attribute("waving", False, 0.0))
    #: Fast, chaotic arm motion (flailing/frantic movement) -- see FLAIL_* above.
    #: Independent of hand_raised/waving: it fires at any arm height and does
    #: not require the rhythmic swing a wave does.
    flailing: Attribute = field(default_factory=lambda: Attribute("flailing", False, 0.0))
    # Resting: a settled body (sitting OR lying) that is also still. THE point
    # of this attribute is that it is not a posture — someone asleep in a
    # chair, at a desk, or in bed all read as resting, where `lying_down`
    # alone only ever caught the bed. Sustained duration is the rule engine's
    # job (persist_s), so this answers only "is the body settled right now".
    resting: Attribute = field(default_factory=lambda: Attribute("resting", False, 0.0))
    #: Why the classifier believes it — carried into alerts as evidence.
    signals: dict = field(default_factory=dict)
    ready: bool = False

    def _all(self) -> tuple[Attribute, ...]:
        return (self.standing, self.sitting, self.walking, self.running,
                self.jumping, self.lying_down, self.hand_raised, self.waving,
                self.flailing, self.resting)

    def active(self) -> list[Attribute]:
        return [a for a in self._all() if a.present]

    def labels(self) -> list[str]:
        return [a.name for a in self.active()]

    def as_dict(self) -> dict:
        return {
            a.name: {"present": a.present, "confidence": round(a.confidence, 2)}
            for a in self._all()
        }


@dataclass
class _Sample:
    at: float
    hip_x: float
    hip_y: float
    #: None when neither ankle is visible. Never substituted with hip_y: that
    #: made the jump test's "both ankles leave their baseline" gate identical
    #: to the hip-rise gate it exists to corroborate, so standing up from a
    #: chair with the feet out of frame read as a jump.
    ankle_y: Optional[float]
    torso: float
    # Wrist positions in aspect-corrected units, or None when not visible.
    # Kept per-side: a wave is one arm's motion, and averaging a moving wrist
    # with a stationary one would halve the signal.
    left_wrist: Optional[tuple[float, float]] = None
    right_wrist: Optional[tuple[float, float]] = None
    shoulder_y: float = 0.0
    nose_y: Optional[float] = None


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

    def __init__(
        self,
        window_s: float = WINDOW_S,
        aspect: float = 1.0,
        flail_min_peak_speed: float = FLAIL_MIN_PEAK_SPEED,
        flail_min_peak_accel: float = FLAIL_MIN_PEAK_ACCEL,
        flail_min_reversals: int = FLAIL_MIN_REVERSALS,
        flail_min_reversal_amplitude: float = FLAIL_MIN_REVERSAL_AMPLITUDE,
        pose_lost_grace_s: float = POSE_LOST_GRACE_S,
    ) -> None:
        self.window_s = window_s
        self.aspect = aspect
        self._flail_min_peak_speed = flail_min_peak_speed
        self._flail_min_peak_accel = flail_min_peak_accel
        self._flail_min_reversals = flail_min_reversals
        self._flail_min_reversal_amplitude = flail_min_reversal_amplitude
        self._pose_lost_grace_s = pose_lost_grace_s
        self._samples: deque[_Sample] = deque()
        self._baseline_ankle: Optional[float] = None
        self._lost_since: Optional[float] = None

    def reset(self) -> None:
        """Forget history — the subject left, or a new one arrived."""
        self._samples.clear()
        self._baseline_ankle = None
        self._lost_since = None

    def pose_lost(self, now: Optional[float] = None) -> None:
        """Pose landmarks were unavailable this frame.

        Tolerant of a brief dropout (up to `pose_lost_grace_s`), unlike
        `reset()`: a caller expected to call this every frame pose is missing,
        and only a dropout that OUTLASTS the grace period actually clears the
        window (see POSE_LOST_GRACE_S for why a single missing frame must not).
        """
        now = time.monotonic() if now is None else now
        if self._lost_since is None:
            self._lost_since = now
        elif now - self._lost_since >= self._pose_lost_grace_s:
            self.reset()

    def update(self, landmarks: Sequence, now: Optional[float] = None,
               aspect: Optional[float] = None) -> ActivityState:
        """Add one frame of landmarks and classify."""
        now = time.monotonic() if now is None else now
        self._lost_since = None  # pose is back, whatever streak was building ends
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

        left_wrist = _point(landmarks, LEFT_WRIST)
        right_wrist = _point(landmarks, RIGHT_WRIST)
        nose = _point(landmarks, NOSE)
        self._samples.append(_Sample(
            at=now, hip_x=hips[0] * self.aspect, hip_y=hips[1],
            ankle_y=ankles[1] if ankles else None, torso=torso,
            left_wrist=(left_wrist[0] * self.aspect, left_wrist[1]) if left_wrist else None,
            right_wrist=(right_wrist[0] * self.aspect, right_wrist[1]) if right_wrist else None,
            shoulder_y=shoulders[1],
            nose_y=nose[1] if nose else None,
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
            abs(b.hip_x - a.hip_x) for a, b in pairwise(samples)
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
        hip_rise = (heights[0] - min(heights)) / torso
        state.signals["hip_rise_torso"] = round(hip_rise, 2)
        # The ankle half of the test needs ankles actually observed: only
        # samples with visible ankles count, and too few of them means "cannot
        # say", not "the hips rose, so the ankles must have too".
        ankle_heights = [s.ankle_y for s in samples if s.ankle_y is not None]
        ankle_rise = None
        if len(ankle_heights) >= MIN_SAMPLES:
            ankle_rise = (max(ankle_heights) - min(ankle_heights)) / torso
            state.signals["ankle_rise_torso"] = round(ankle_rise, 2)

        if hip_rise >= JUMP_RISE and ankle_rise is not None and ankle_rise >= JUMP_ANKLE_RISE:
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

        self._classify_gestures(samples, torso, state)
        self._classify_resting(samples, torso, speed, state)
        return state

    def _classify_gestures(self, samples: list[_Sample], torso: float,
                           state: ActivityState) -> None:
        """Raised hand and waving, from wrist trajectories.

        Both are decided per arm and OR-ed: one raised hand is a raised hand.
        A wave requires the hand to already be raised — a wrist oscillating at
        hip height is scrubbing a counter, not greeting anyone.
        """
        for side in ("left_wrist", "right_wrist"):
            track = [(s.at, getattr(s, side), s.shoulder_y) for s in samples
                     if getattr(s, side) is not None]
            if len(track) < MIN_SAMPLES:
                continue

            # Raised: the wrist above the shoulder line for the whole window,
            # not one frame — a hand tossed up mid-gesture must not count.
            clearances = [(shoulder_y - wrist[1]) / torso for _, wrist, shoulder_y in track]
            raised_throughout = all(c >= HAND_RAISED_MARGIN for c in clearances)
            if raised_throughout:
                lowest = min(clearances)
                confidence = min(0.95, 0.6 + (lowest - HAND_RAISED_MARGIN) * 1.5)
                if confidence > state.hand_raised.confidence:
                    state.hand_raised = Attribute("hand_raised", True, round(confidence, 2))

                # Waving = raised AND oscillating: count completed wide swings
                # (runs of same-direction horizontal travel). Jitter produces
                # direction changes with near-zero travel, so a swing only
                # counts once the wrist has moved WAVE_MIN_SWING torso-lengths
                # in one direction.
                xs = [wrist[0] for _, wrist, _ in track]
                swings = 0
                swing = 0.0
                direction = 0
                for previous, current in pairwise(xs):
                    step = current - previous
                    if abs(step) < 1e-6:
                        continue
                    step_dir = 1 if step > 0 else -1
                    if direction == 0:
                        direction = step_dir
                        swing = abs(step) / torso
                    elif step_dir == direction:
                        swing += abs(step) / torso
                    else:
                        if swing >= WAVE_MIN_SWING:
                            swings += 1
                        direction = step_dir
                        swing = abs(step) / torso
                if swing >= WAVE_MIN_SWING:
                    swings += 1  # the in-flight swing counts once it is wide enough
                state.signals[f"{side}_swings"] = swings
                if swings >= WAVE_MIN_SWINGS:
                    confidence = min(0.95, 0.45 + 0.1 * swings)
                    if confidence > state.waving.confidence:
                        state.waving = Attribute("waving", True, round(confidence, 2))

            # --- flailing: fast, chaotic 2D wrist motion, no "raised" precondition ---
            # Deliberately reads the FULL track (not just frames where the wrist
            # was raised): distress/flailing motion happens at any height.
            self._classify_flailing(side, samples, torso, state)

        # A wave is a moving raised hand; report the more specific of the two.
        if state.waving.present and state.hand_raised.present:
            state.hand_raised = Attribute("hand_raised", False, 0.0)

        # Two arms flailing together is more clearly distress than one -- a real
        # signal boost, not just the max of independent per-arm confidences.
        if state.signals.get("left_wrist_flailing") and state.signals.get("right_wrist_flailing"):
            state.flailing = Attribute("flailing", True, round(min(0.99, state.flailing.confidence + 0.15), 2))

    def _classify_flailing(self, side: str, samples: list[_Sample], torso: float,
                           state: ActivityState) -> None:
        """Peak speed/acceleration + direction-reversal frequency for one wrist.

        Velocity and acceleration alone can't tell a single fast reach from
        flailing (both can hit a high peak speed) -- requiring several direction
        reversals too is what rejects the single-motion case. Reversal count
        alone can't tell a slow, deliberate back-and-forth from flailing --
        requiring speed/acceleration too is what rejects that case. Together
        they need BOTH chaos (reversals) AND intensity (speed or acceleration).
        """
        track = [(s.at, getattr(s, side)) for s in samples if getattr(s, side) is not None]
        if len(track) < MIN_SAMPLES:
            return

        # Per-step velocity VECTORS (torso-lengths/s), not just speeds: a sharp
        # reversal at constant speed is a large acceleration (the velocity
        # flips sign) but a zero change in speed, and catching exactly that
        # "jerky direction change" is what the acceleration gate is for.
        velocities: list[tuple[float, float, float]] = []  # (vx, vy, dt)
        for (t0, p0), (t1, p1) in pairwise(track):
            dt = t1 - t0
            if dt <= 0:
                continue
            velocities.append(((p1[0] - p0[0]) / torso / dt, (p1[1] - p0[1]) / torso / dt, dt))
        if not velocities:
            return
        peak_speed = max(math.hypot(vx, vy) for vx, vy, _ in velocities)

        peak_accel = 0.0
        for (vx0, vy0, _), (vx1, vy1, dt1) in pairwise(velocities):
            if dt1 > 0:
                peak_accel = max(peak_accel, math.hypot(vx1 - vx0, vy1 - vy0) / dt1)

        # Direction reversals over the full 2D displacement vector (flailing
        # isn't purely horizontal, unlike the wave counter above): accumulate
        # travel in roughly one direction (dot product with the running
        # direction stays positive) and count a reversal each time the wrist
        # commits to a new direction after covering at least
        # flail_min_reversal_amplitude torso-lengths in the old one.
        reversals = 0
        seg_dx = seg_dy = 0.0
        prev_dir: Optional[tuple[float, float]] = None
        for (_, p0), (_, p1) in pairwise(track):
            dx, dy = (p1[0] - p0[0]) / torso, (p1[1] - p0[1]) / torso
            step = math.hypot(dx, dy)
            if step < 1e-6:
                continue
            direction = (dx / step, dy / step)
            if prev_dir is None:
                seg_dx, seg_dy = dx, dy
            elif direction[0] * prev_dir[0] + direction[1] * prev_dir[1] > 0.3:
                seg_dx += dx
                seg_dy += dy
            else:
                if math.hypot(seg_dx, seg_dy) >= self._flail_min_reversal_amplitude:
                    reversals += 1
                seg_dx, seg_dy = dx, dy
            prev_dir = direction
        if math.hypot(seg_dx, seg_dy) >= self._flail_min_reversal_amplitude:
            reversals += 1

        state.signals[f"{side}_peak_speed"] = round(peak_speed, 2)
        state.signals[f"{side}_peak_accel"] = round(peak_accel, 2)
        state.signals[f"{side}_reversals"] = reversals

        intense = peak_speed >= self._flail_min_peak_speed or peak_accel >= self._flail_min_peak_accel
        if intense and reversals >= self._flail_min_reversals:
            confidence = min(0.95, 0.5
                              + (peak_speed - self._flail_min_peak_speed) * 0.1
                              + 0.05 * (reversals - self._flail_min_reversals))
            state.signals[f"{side}_flailing"] = True
            if confidence > state.flailing.confidence:
                state.flailing = Attribute("flailing", True, round(max(confidence, 0.5), 2))

    def _classify_resting(self, samples: list[_Sample], torso: float,
                          speed: float, state: ActivityState) -> None:
        """A settled body: sitting or lying, and still.

        This is deliberately NOT called "sleeping". Vision cannot confirm sleep
        — it can observe that a body has settled and stopped moving, which
        sustained over minutes (the rule's persist_s) is honest evidence of
        rest. Head droop, when the nose is visible, raises confidence for the
        asleep-at-a-desk case without being required for the in-bed one.
        """
        if not (state.sitting.present or state.lying_down.present):
            return
        if speed > STILL_MAX_HIP_SPEED:
            return

        wrist_speed = self._max_wrist_speed(samples, torso)
        if wrist_speed is not None and wrist_speed > STILL_MAX_WRIST_SPEED:
            state.signals["wrist_speed_torso_per_s"] = round(wrist_speed, 2)
            return

        posture_confidence = max(state.sitting.confidence, state.lying_down.confidence)
        confidence = min(0.9, 0.5 + posture_confidence * 0.3)

        drooped = self._head_drooped(samples, torso)
        if drooped:
            confidence = min(0.95, confidence + 0.15)
            state.signals["head_drooped"] = True
        state.resting = Attribute("resting", True, round(confidence, 2))

    def _max_wrist_speed(self, samples: list[_Sample], torso: float) -> Optional[float]:
        """Fastest wrist, torso-lengths/second, or None with no wrist data."""
        fastest = None
        for side in ("left_wrist", "right_wrist"):
            track = [(s.at, getattr(s, side)) for s in samples if getattr(s, side) is not None]
            if len(track) < 2:
                continue
            elapsed = track[-1][0] - track[0][0]
            if elapsed <= 0:
                continue
            path = sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for (_, a), (_, b) in pairwise(track)
            ) / torso
            speed = path / elapsed
            fastest = speed if fastest is None else max(fastest, speed)
        return fastest

    def _head_drooped(self, samples: list[_Sample], torso: float) -> bool:
        """Whether the nose sits near or below shoulder height, sustained."""
        clearances = [
            (s.shoulder_y - s.nose_y) / torso
            for s in samples if s.nose_y is not None
        ]
        if len(clearances) < MIN_SAMPLES:
            return False
        return max(clearances) <= HEAD_DROOP_MAX_CLEARANCE
