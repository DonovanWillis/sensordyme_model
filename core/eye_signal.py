"""Eye-signal conditioning for driver monitoring: making one noisy frame unable to lie.

Three small, pure components that sit between raw MediaPipe landmarks and the
drowsiness/distraction thresholds in modules/driver_attention.py. They exist
because of a specific failure mode: **glasses**. Frames, glare and lens
refraction degrade FaceMesh's eyelid-contour landmarks, which turns the raw
EAR (eye aspect ratio) from a clean signal into a spiky one — and every
downstream consumer assumed clean:

  * a single spiky "open" frame in the middle of a real 2-second closure used
    to reset the drowsiness timer to zero (SustainedConditionTracker resets on
    one False frame), so the sleepier the signal, the harder it became to
    detect;
  * a single spiky "closed" frame inflated PERCLOS and could seed a glance
    "episode", so noise alone manufactured fatigue and frequent-glancing
    warnings;
  * a threshold calibrated bare-faced sat wrong the day glasses went on,
    because glasses shift the resting EAR distribution.

None of this needs a new model. It needs the signal treated as what it is —
a noisy measurement — before any boolean is derived from it.

`SignalSmoother`     rolling median: single-frame spikes cannot pass at all.
`AdaptiveBaseline`   a slow rolling open-eye reference (upper percentile), so
                     the effective threshold follows the face in front of the
                     camera today — with glasses or without — instead of the
                     face that calibrated last month. Banded so a genuinely
                     degrading driver cannot drag the threshold down with them.
`iris_openness`      a second, independent eye-openness measure from the iris
                     landmarks FaceMesh already computes (refine_landmarks=True
                     has always been on; the landmarks were simply unused).
                     The iris is fit from the pupil/limbus contrast, which
                     survives spectacle frames better than the eyelid contour,
                     and the ratio is normalised by the iris's own diameter so
                     it is distance- and lens-magnification-invariant.
`eye_gaze_offset_deg` eye-IN-HEAD deviation (yaw, pitch), from the same iris
                     landmarks, independent of head pose -- what lets a driver-
                     attention threshold tell "head forward, eyes on the road"
                     apart from "head forward, eyes in the driver's lap": whole-
                     head yaw/pitch (solvePnP on 6 landmarks) cannot see the
                     eyes move without the head moving. Added to head yaw/pitch
                     this produces a fused gaze vector -- the same approach
                     modules/driver_attention_max.py's `_compute_gaze_components`
                     already validated on the Mac-only profile, reimplemented
                     independently here (that module imports FROM this one, so
                     the reverse would be circular) rather than shared code.

Everything here is arithmetic on a handful of floats per frame — nothing runs
inference, so the Pi's frame budget is untouched.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Optional

Point = tuple[float, float]

# MediaPipe FaceMesh iris landmark indices (present when refine_landmarks=True).
# Each eye: center, then right/top/left/bottom of the iris contour.
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)

# Eyelid landmarks re-used from the 6-point EAR sets: the upper/lower pairs.
# (Same indices modules/driver_attention.py already tracks.)
RIGHT_EYE_VERTICALS = ((160, 144), (158, 153))
LEFT_EYE_VERTICALS = ((385, 380), (387, 373))

# Below this fraction of the eye-corner distance, an iris "diameter" is a
# degenerate fit (eye closed, or the mesh lost the iris) — not a measurement.
_MIN_IRIS_FRACTION = 0.08


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def iris_openness(points: Sequence[Point]) -> Optional[float]:
    """Eyelid gap over iris diameter, averaged over both usable eyes.

    Scale-free by construction: the iris diameter is a fixed anatomical length
    (~11.7mm across adults, one of the least variable human measurements), so
    dividing the eyelid gap by it cancels camera distance AND spectacle-lens
    magnification — the two things that make an absolute pixel threshold wrong.

    Typical values: comfortably open ≈ 0.7-1.1, closed < 0.2. Returns None
    when neither eye yields a usable iris (landmarks missing or degenerate),
    which callers must treat as "no opinion", never as "closed".
    """
    ratios = []
    for iris, verticals, corners in (
        (RIGHT_IRIS, RIGHT_EYE_VERTICALS, (33, 133)),
        (LEFT_IRIS, LEFT_EYE_VERTICALS, (362, 263)),
    ):
        try:
            _center_r, right_r, _top, left_r, _bottom = (points[i] for i in iris)
            corner_a, corner_b = points[corners[0]], points[corners[1]]
        except (IndexError, TypeError):
            continue
        diameter = _dist(right_r, left_r)
        eye_width = _dist(corner_a, corner_b)
        if eye_width <= 0 or diameter < eye_width * _MIN_IRIS_FRACTION:
            continue  # degenerate iris fit — closed lid or lost track
        gap = sum(_dist(points[up], points[down]) for up, down in verticals) / len(verticals)
        ratios.append(gap / diameter)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


# Eye "socket" landmarks -- the same 6-point EAR box modules/driver_attention.py's
# LEFT_EYE/RIGHT_EYE already use, duplicated here (not imported) because these are
# stable, independent MediaPipe FaceMesh index constants and this module already
# holds its own eye-landmark knowledge (see RIGHT_EYE_VERTICALS/LEFT_EYE_VERTICALS
# above, themselves subsets of the same 6 points).
_LEFT_EYE_SOCKET = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE_SOCKET = (33, 160, 158, 133, 153, 144)

# Extreme iris deflection to the edge of its socket reads as roughly this many
# degrees of eye-in-head rotation. An approximation (irises don't sweep through a
# literal angle the way a pinhole-camera model would compute) -- the SAME
# approximation already validated on the mac_max profile
# (modules/driver_attention_max.py's EYE_GAZE_SCALE_DEG), reused here rather than
# re-derived from nothing.
EYE_GAZE_SCALE_DEG = 40.0

# Nominal open-eye palpebral aperture as a fraction of the corner-to-corner eye
# width (~10mm over ~30mm; MediaPipe's 6-point open-eye EAR of ~0.3 says the
# same). The vertical iris offset is normalised by THIS fixed fraction of the
# width rather than by the measured lid aperture: the aperture is exactly what
# changes when the lids droop, squint or close, and dividing by it turned a
# closing eye into "looking up" (the lids' centroid sinks toward the lower lid
# while the iris stays put -- a still head with closed eyes read +24 deg up).
_NOMINAL_APERTURE_FRACTION = 0.35


def eye_gaze_offset_deg(
    points: Sequence[Point], scale_deg: float = EYE_GAZE_SCALE_DEG
) -> Optional[tuple[float, float]]:
    """Eye-in-head deviation (yaw, pitch), in pseudo-degrees, from iris position
    within each eye's own socket -- averaged across both eyes, independent of
    head pose.

    The socket reference is the two eye CORNERS (canthi), not the eyelids: the
    corners are fixed to the skull, while the lids move with blinks, squints
    and drowsiness. Both axes are normalised by the corner-to-corner width
    (vertical by a nominal fraction of it, see _NOMINAL_APERTURE_FRACTION), so
    the lids opening or closing does not move the reading at all. The static
    offset of the iris from the corner line is removed by calibration (the
    pi profile calibrates an eye-offset baseline alongside the head pose).

    Returns None when neither eye's socket geometry is usable (degenerate
    width -- an extreme angle or a landmark dropout), never a wrong number.
    """

    def _eye_component(socket: tuple[int, ...], iris_index: int) -> Optional[tuple[float, float]]:
        try:
            corner_a, corner_b = points[socket[0]], points[socket[3]]
            iris = points[iris_index]
        except (IndexError, TypeError):
            return None
        width = math.hypot(corner_b[0] - corner_a[0], corner_b[1] - corner_a[1])
        if width <= 0:
            return None
        center_x = (corner_a[0] + corner_b[0]) / 2.0
        center_y = (corner_a[1] + corner_b[1]) / 2.0
        return (
            (iris[0] - center_x) / width,
            (iris[1] - center_y) / (width * _NOMINAL_APERTURE_FRACTION),
        )

    right = _eye_component(_RIGHT_EYE_SOCKET, RIGHT_IRIS[0])
    left = _eye_component(_LEFT_EYE_SOCKET, LEFT_IRIS[0])
    components = [c for c in (right, left) if c is not None]
    if not components:
        return None
    h_ratio = sum(c[0] for c in components) / len(components)
    v_ratio = sum(c[1] for c in components) / len(components)
    # v_ratio is POSITIVE when the iris sits toward the LOWER eyelid (larger
    # image y = physically looking down) -- but this module's pitch convention
    # (shared with modules/driver_attention.py's estimate_head_pose_deg, whose
    # own docstring documents the same head-pose sign fix) is NEGATIVE = down,
    # so the vertical component is negated to fuse consistently with head
    # pitch: "head level, eyes cast down" must ADD toward eyes_down (more
    # negative), not partially cancel it. Verified against
    # estimate_head_pose_deg's own already-confirmed sign
    # (tests/test_eye_gaze.py), not assumed. h_ratio needs no such flip: image-
    # right already matches this module's "positive yaw = turned toward the
    # frame's right" convention directly.
    return h_ratio * scale_deg, -v_ratio * scale_deg


class SignalSmoother:
    """Rolling median over the last `window` samples.

    A median, not a mean, because the enemy is the outlier: one glare frame
    that reads EAR=0.02 in an open eye must contribute NOTHING, not a diluted
    fraction. With window 5, no single frame — and no two consecutive frames —
    can move the output past the majority.
    """

    def __init__(self, window: int = 5) -> None:
        self._window = max(1, int(window))
        self._samples: deque[float] = deque(maxlen=self._window)

    def update(self, value: float) -> float:
        self._samples.append(float(value))
        ordered = sorted(self._samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def reset(self) -> None:
        """Forget history — the face was lost; stale samples describe nobody."""
        self._samples.clear()


class AdaptiveBaseline:
    """A slow rolling estimate of this session's open-eye signal level.

    The calibrated baseline answers "what does this user's open eye measure" —
    once, on whatever face was present that day. Put glasses on and the answer
    changes; the stored number does not. This tracks the upper percentile of
    the recent signal (default: P90 over 60s), on the observation that even a
    drowsy driver opens their eyes most of the time, so the high end of the
    recent distribution is a live open-eye reference.

    Two guards keep it honest:

      * **The band.** When a calibrated baseline exists, adaptation is clamped
        to [band_low, band_high] x that baseline. A driver whose eyes are
        genuinely closing for minutes cannot drag the reference — and with it
        the closure threshold — down to meet them. Glasses shift a baseline by
        tens of percent; pathology would need to move it much further.
      * **Warmup.** With no calibration at all, nothing is reported until
        `min_samples` observations exist, so the first seconds of a session
        cannot define "open".
    """

    def __init__(
        self,
        window_s: float = 60.0,
        percentile: float = 0.90,
        band_low: float = 0.6,
        band_high: float = 1.4,
        min_samples: int = 30,
    ) -> None:
        self._window_s = window_s
        self._percentile = percentile
        self._band_low = band_low
        self._band_high = band_high
        self._min_samples = min_samples
        self._samples: deque[tuple[float, float]] = deque()

    def observe(self, value: float, now: float) -> None:
        self._samples.append((now, float(value)))
        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def reference(self, calibrated: Optional[float]) -> Optional[float]:
        """The open-eye reference to threshold against right now.

        With a calibrated baseline: the session estimate, clamped to the band
        around it (so it can follow glasses, not pathology). Without one: the
        raw session estimate once warmed up, else None.
        """
        if len(self._samples) < self._min_samples:
            return calibrated
        ordered = sorted(value for _, value in self._samples)
        index = min(len(ordered) - 1, int(len(ordered) * self._percentile))
        session = ordered[index]
        if calibrated is None:
            return session
        return max(self._band_low * calibrated, min(self._band_high * calibrated, session))

    def reset(self) -> None:
        self._samples.clear()
