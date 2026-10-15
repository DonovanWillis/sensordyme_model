"""What `Event.confidence` means, pinned (core/confidence.py).

Confidence is the certainty that an event's condition really occurred — never a
raw measurement, never a threshold, and None when there is no graded evidence.
These tests pin each of the three sources, and the boundary behaviour that
makes the number interpretable: a reading exactly on its threshold is 0.5.
"""

from __future__ import annotations

import math
import random

import pytest

from core.confidence import (
    NoiseEstimator,
    clamp_unit,
    joint,
    margin_confidence,
    proportion_confidence,
    rate_confidence,
)


# ------------------------------------------------------------------ measurement
def test_a_reading_exactly_on_its_threshold_is_a_coin_flip():
    assert margin_confidence(0.0, 2.0) == 0.5


def test_confidence_grows_with_the_margin_in_noise_units_and_saturates():
    sigma = 2.0
    values = [margin_confidence(m, sigma) for m in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values), "more margin must never mean less certainty"
    assert margin_confidence(2 * sigma, sigma) == pytest.approx(0.977, abs=0.001)  # Phi(2)
    assert margin_confidence(10 * sigma, sigma) == 1.0
    assert margin_confidence(-2 * sigma, sigma) == pytest.approx(0.023, abs=0.001)


def test_the_same_margin_is_less_certain_on_a_noisier_signal():
    assert margin_confidence(3.0, sigma=1.0) > margin_confidence(3.0, sigma=3.0)


def test_a_threshold_never_becomes_the_confidence():
    """The bug this replaced: eyes_down at its -20 degree threshold reported
    |pitch|/90 = 0.22. The pitch, the threshold and the confidence are three
    different numbers now; a confirmed condition is never below 0.5."""
    threshold, pitch, sigma = -20.0, -20.5, 1.0
    confidence = margin_confidence(threshold - pitch, sigma)
    assert confidence > 0.5
    assert confidence != pytest.approx(abs(pitch) / 90.0, abs=0.05)


def test_a_degenerate_sigma_does_not_divide_by_zero():
    assert margin_confidence(1.0, 0.0) == 1.0
    assert margin_confidence(-1.0, 0.0) == 0.0
    assert margin_confidence(0.0, 0.0) == 0.5
    assert margin_confidence(float("nan"), 1.0) == 0.5


# ------------------------------------------------------------------ noise
def test_the_noise_estimator_measures_known_gaussian_noise():
    rng = random.Random(7)
    true_sigma = 2.0
    estimator = NoiseEstimator(floor=0.01, window=400)
    for _ in range(400):
        estimator.observe(10.0 + rng.gauss(0.0, true_sigma))
    assert estimator.sigma == pytest.approx(true_sigma, rel=0.15)


def test_a_head_turn_does_not_inflate_the_noise_estimate():
    """The MEDIAN of first differences ignores the few large steps a genuine
    movement makes; a mean would read a turn as noise and lower confidence."""
    rng = random.Random(3)
    estimator = NoiseEstimator(floor=0.01, window=200)
    value = 0.0
    for i in range(200):
        if i in (60, 61, 62):
            value += 15.0  # a real, fast head turn
        estimator.observe(value + rng.gauss(0.0, 1.0))
    assert estimator.sigma == pytest.approx(1.0, rel=0.2)


def test_a_median_filtered_value_is_less_noisy_than_one_frame():
    rng = random.Random(11)
    raw, filtered = NoiseEstimator(floor=0.01), NoiseEstimator(floor=0.01, smoothing_window=5)
    for _ in range(200):
        sample = rng.gauss(0.0, 2.0)
        raw.observe(sample)
        filtered.observe(sample)
    # Standard error of a median of 5: sqrt(pi/2) / sqrt(5) of one sample's.
    assert filtered.sigma == pytest.approx(raw.sigma * math.sqrt(math.pi / 2) / math.sqrt(5), rel=1e-6)


def test_the_floor_stops_a_quiet_signal_claiming_infinite_certainty():
    estimator = NoiseEstimator(floor=1.0)
    for _ in range(50):
        estimator.observe(5.0)  # perfectly still: zero measured noise
    assert estimator.sigma == pytest.approx(1.0)
    assert margin_confidence(0.5, estimator.sigma) < 0.7


def test_too_few_samples_assume_a_noisy_prior_not_the_floor():
    """Before there is evidence the signal is uncertain; the floor would be the
    MOST confident possible answer — the opposite of what is known."""
    estimator = NoiseEstimator(floor=1.5, smoothing_window=5)
    estimator.observe(0.0)
    estimator.observe(0.0)
    assert estimator.sigma == pytest.approx(4.5)


def test_the_floor_holds_after_median_filtering():
    """A still signal through a median of 5 must not claim sub-resolution precision."""
    estimator = NoiseEstimator(floor=1.0, smoothing_window=5)
    for _ in range(40):
        estimator.observe(2.0)
    assert estimator.sigma == pytest.approx(1.0)


def test_a_gap_is_not_counted_as_noise():
    """A lost face then a returned one is a jump across time, not jitter."""
    estimator = NoiseEstimator(floor=0.01, window=50)
    for _ in range(40):
        estimator.observe(0.0)
    estimator.observe(None)
    estimator.observe(80.0)  # face back, head now somewhere else entirely
    for _ in range(9):
        estimator.observe(80.0)
    assert estimator.sigma == pytest.approx(0.01)


# ------------------------------------------------------------------ proportions / rates
def test_perclos_just_over_threshold_with_few_closures_is_uncertain():
    shaky = proportion_confidence(0.16, threshold=0.15, n_effective=3)
    solid = proportion_confidence(0.30, threshold=0.15, n_effective=40)
    assert 0.5 < shaky < 0.6
    # sigma = sqrt(0.3 * 0.7 / 40) = 0.0725, z = 2.07 -> Phi = 0.981
    assert solid == pytest.approx(0.981, abs=0.002)


def test_zero_episodes_is_treated_as_one_not_as_a_division_by_zero():
    assert 0.0 <= proportion_confidence(0.2, 0.15, n_effective=0) <= 1.0


def test_blink_rate_confidence_follows_poisson_counts():
    # 4 blinks in 2 minutes = 2/min against a floor of 8/min: clearly low.
    assert rate_confidence(count=4, minutes=2.0, low=8.0, high=30.0) > 0.99
    # 15/min is comfortably inside the band.
    assert rate_confidence(count=30, minutes=2.0, low=8.0, high=30.0) < 0.01
    # Exactly on the floor: a coin flip.
    assert rate_confidence(count=16, minutes=2.0, low=8.0, high=30.0) == pytest.approx(0.5, abs=0.01)


# ------------------------------------------------------------------ model / joint / none
def test_a_model_score_passes_through_made_safe():
    assert clamp_unit(0.8234) == 0.823
    assert clamp_unit(1.7) == 1.0
    assert clamp_unit(-0.2) == 0.0
    assert clamp_unit(None) is None
    assert clamp_unit(float("nan")) is None, "a fault upstream is unknown, not certain"


def test_a_conjunction_is_never_more_certain_than_its_weakest_part():
    both = joint(0.9, 0.8)
    assert both == pytest.approx(0.72)
    assert both is not None and both <= min(0.9, 0.8)
    assert joint(0.9, None) is None, "unknown evidence for one part makes the whole unknown"


def test_phone_calibration_measures_score_against_labelled_truth():
    """benchmarks/accuracy_state_farm.py: a phone event's confidence is the
    detector's score, so the benchmark measures whether that score behaves as a
    probability on labelled frames (reliability bins + ECE)."""
    from benchmarks.accuracy_state_farm import phone_calibration

    per_class = {
        # 10 phone frames scored 0.9; 9 of them real phones -> rate 0.9 (calibrated)
        "c1": {"phone_scores": [0.9] * 9 + [0.0]},
        "c0": {"phone_scores": [0.9] + [0.3] * 4 + [0.0] * 5},
        # 4 frames at 0.3, none of them phones -> rate 0.0 (over-confident by 0.3)
    }
    cal = phone_calibration(per_class)
    by_band = {tuple(b["band"]): b for b in cal["bins"]}
    assert by_band[(0.8, 1.0)]["phone_rate"] == pytest.approx(0.9)
    assert by_band[(0.2, 0.4)]["phone_rate"] == 0.0
    assert cal["frames_with_a_score"] == 14, "frames with no phone box carry no score"
    # ECE: 10/14 * |0.9-0.9| + 4/14 * |0.3-0.0|
    assert cal["ece"] == pytest.approx(4 / 14 * 0.3, abs=0.001)
    assert cal["base_rate"] == 0.5
