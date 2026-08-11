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

Everything here is arithmetic on a handful of floats per frame — nothing runs
inference, so the Pi's frame budget is untouched.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

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

    Typical values: comfortably open ≈ 0.7–1.1, closed < 0.2. Returns None
    when neither eye yields a usable iris (landmarks missing or degenerate),
    which callers must treat as "no opinion", never as "closed".
    """
    ratios = []
    for iris, verticals, corners in (
        (RIGHT_IRIS, RIGHT_EYE_VERTICALS, (33, 133)),
        (LEFT_IRIS, LEFT_EYE_VERTICALS, (362, 263)),
    ):
        try:
            center_r, right_r, _top, left_r, _bottom = (points[i] for i in iris)
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
        to [band_low, band_high] × that baseline. A driver whose eyes are
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
