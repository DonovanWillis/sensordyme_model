"""What an event's `confidence` means, and the only ways one is produced.

DEFINITION. `Event.confidence` is the system's certainty, in [0, 1], that the
condition the event type names really occurred. It is never a raw measurement
and never a threshold. `None` means the event has no graded evidence behind it
(a count crossing a limit, a mode change), and is stated as absent rather than
invented. Before this module, five unrelated quantities shared the field —
|pitch|/90, PERCLOS, 1-EAR, fixed tiers, model scores — so a confirmed eyes-down
alert reached a phone as "22% confidence".

There are exactly three sources, and every event uses one of them:

  model        A detector's own score (YOLO box confidence, a VLM verdict, or
               EventEngine's EMA of per-frame model scores). Passed through: it
               is the model's certainty, uncalibrated but genuinely its own.

  measurement  A rule over a continuous measurement (head-pose angle, EAR,
               PERCLOS, blink rate). The value is the probability that the TRUE
               signal is beyond the threshold, given the measurement's noise:
               Phi(margin / sigma). 0.5 means the reading sits exactly on the
               threshold and could be noise either way; values near 1 mean it is
               past the threshold by many noise widths. sigma is measured from
               the signal itself (see NoiseEstimator), not assumed.

  none         No graded evidence: confidence is None.

What `measurement` does NOT claim: that the criterion perfectly captures the
human state. "Head pitched 30 degrees down for 2 s" can be certain while the
driver is reading a mounted map. It quantifies whether the criterion was met,
which is the claim the event makes.

Conjunctions of independent conditions ("phone visible AND eyes off road")
multiply, which is the probability that both held under independence.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

#: Consistent rounding for everything that leaves the device.
_DECIMALS = 3

#: Median-filter efficiency: the standard error of a median of k samples of
#: Gaussian noise is ~sqrt(pi/2) * sigma / sqrt(k) — 1.2533x the mean's.
_MEDIAN_EFFICIENCY = math.sqrt(math.pi / 2.0)

#: Scale from a median absolute deviation to a Gaussian standard deviation.
_MAD_TO_SIGMA = 1.4826


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def clamp_unit(value: Optional[float]) -> Optional[float]:
    """A model score made safe for the field: finite, within [0, 1], rounded.

    Non-finite input is evidence of a fault upstream, not of certainty, so it
    becomes None rather than 0 or 1.
    """
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(min(1.0, max(0.0, value)), _DECIMALS)


def margin_confidence(margin: float, sigma: float) -> float:
    """P(true signal is past the threshold) for a reading `margin` beyond it.

    `margin` is signed so that positive means "on the event's side of the
    threshold". `sigma` is the noise of the value the decision was made on.
    """
    if not math.isfinite(margin):
        return 0.5
    if sigma <= 0 or not math.isfinite(sigma):
        return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    return round(_phi(margin / sigma), _DECIMALS)


def proportion_confidence(observed: float, threshold: float, n_effective: float) -> float:
    """P(the true proportion exceeds `threshold`), for a fraction like PERCLOS.

    The standard error is the binomial one, sqrt(p(1-p)/n), with n the number
    of INDEPENDENT observations. Frames are not independent — one closure
    spans many of them — so callers pass the number of distinct episodes, which
    is what actually varies from one window to the next.
    """
    n = max(1.0, float(n_effective))
    p = min(1.0 - 1e-6, max(1e-6, float(observed)))
    return margin_confidence(observed - threshold, math.sqrt(p * (1.0 - p) / n))


def rate_confidence(count: int, minutes: float, low: float, high: float) -> float:
    """P(the true event rate is outside [low, high] per minute), Poisson counts.

    For blink rate: the count over a window is approximately Poisson, so the
    rate's standard error is sqrt(count) / minutes (floored at one event, so an
    empty window is uncertain rather than infinitely precise).
    """
    if minutes <= 0:
        return 0.5
    rate = count / minutes
    sigma = math.sqrt(max(1, count)) / minutes
    below = margin_confidence(low - rate, sigma)
    above = margin_confidence(rate - high, sigma)
    return max(below, above)


def joint(*confidences: Optional[float]) -> Optional[float]:
    """Certainty that ALL of several independent conditions held."""
    if any(c is None for c in confidences):
        return None
    product = 1.0
    for c in confidences:
        product *= float(c)  # type: ignore[arg-type]  # None excluded above
    return round(product, _DECIMALS)


class NoiseEstimator:
    """The frame-to-frame noise of one signal, measured from the signal itself.

    A fixed sigma would be a guess, and a wrong one: head-pose jitter differs
    with glasses, lighting and distance from the camera. Consecutive frames of a
    real head move little, so most of the difference between them is noise;
    the median absolute first difference, scaled to a Gaussian sigma and
    divided by sqrt(2) (a difference holds two samples' noise), estimates it
    robustly — a genuine head turn is a few large differences, and the median
    ignores them.

    `floor` is the sensor's resolution: below it, a quiet signal would claim
    certainty it does not have. `smoothing_window` is the median filter the
    decision value passes through; the reported sigma is that filtered value's.
    """

    def __init__(self, floor: float, smoothing_window: int = 1, window: int = 90) -> None:
        self._floor = float(floor)
        self._k = max(1, int(smoothing_window))
        self._diffs: deque[float] = deque(maxlen=max(8, int(window)))
        self._last: Optional[float] = None

    def observe(self, raw: Optional[float]) -> None:
        if raw is None or not math.isfinite(raw):
            # A gap (face lost) breaks continuity: the next sample's difference
            # would span the gap and be motion, not noise.
            self._last = None
            return
        if self._last is not None:
            self._diffs.append(abs(raw - self._last))
        self._last = float(raw)

    #: Before there is enough evidence to measure, assume this many floors of
    #: noise: uncertain, not the quietest the sensor could possibly be.
    _PRIOR_FLOORS = 3.0

    @property
    def sigma(self) -> float:
        """Noise of the median-filtered value the decision is made on, never
        below the sensor's resolution."""
        if len(self._diffs) < 8:
            return self._floor * self._PRIOR_FLOORS
        ordered = sorted(self._diffs)
        mad = ordered[len(ordered) // 2]
        raw_sigma = _MAD_TO_SIGMA * mad / math.sqrt(2.0)
        if self._k > 1:
            raw_sigma *= _MEDIAN_EFFICIENCY / math.sqrt(self._k)
        # The floor is the resolution of the value DECIDED ON, so it applies
        # after the filter's reduction, not before it.
        return max(self._floor, raw_sigma)

    def reset(self) -> None:
        self._diffs.clear()
        self._last = None
