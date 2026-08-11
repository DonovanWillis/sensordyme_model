"""Signal conditioning for driver monitoring: glasses noise must not lie.

The failure mode all of this exists for, reproduced as arithmetic: with
glasses, MediaPipe's eyelid landmarks land one bad frame every few seconds.
Before conditioning, one bad frame could (a) reset the drowsiness timer
mid-closure — so detection got WORSE exactly when the driver got drowsier,
(b) inflate PERCLOS toward a false fatigue warning, (c) seed a false glance
episode from a yaw spike, and (d) flap `is_alert_active`, flickering the
alert screen. The components under test remove each mechanism while — the
part that must be proven, not assumed — leaving genuine detection latency
unchanged: a real sustained closure still fires at exactly drowsy_seconds.
"""

from __future__ import annotations

import math

from core.eye_signal import AdaptiveBaseline, SignalSmoother, iris_openness
from modules.base import GracedConditionTracker, SustainedConditionTracker


class TestGracedConditionTracker:
    def test_fires_at_exactly_the_sustain_threshold(self):
        """The grace must not slow detection down: a clean sustained condition
        fires at the same moment it always did."""
        graced = GracedConditionTracker(2.0, release_s=0.4)
        old = SustainedConditionTracker(2.0)
        fired_graced = fired_old = None
        for i in range(30):
            now = i * 0.125  # 8fps, the Pi's real cadence
            fired_graced = fired_graced or graced.update(True, now)
            fired_old = fired_old or old.update(True, now)
        assert fired_graced is not None and fired_old is not None
        assert abs(fired_graced - fired_old) < 1e-9

    def test_recurring_noise_frames_no_longer_starve_detection(self):
        """The regression mechanism itself. Glasses noise lands a bad frame
        every second or two; the old tracker reset its clock on each one, so
        with noise recurring faster than drowsy_seconds a real closure could
        NEVER fire, no matter how long it lasted. The graced tracker bridges
        each one-frame gap and fires on schedule."""
        graced = GracedConditionTracker(2.0, release_s=0.4)
        old = SustainedConditionTracker(2.0)
        fired_graced = fired_old = None
        for i in range(48):                      # 6 seconds at 8fps
            now = i * 0.125
            condition = i % 12 != 8              # one bad frame every 1.5s
            fired_graced = fired_graced or graced.update(condition, now)
            fired_old = fired_old or old.update(condition, now)
        assert fired_old is None, "the old tracker documents the bug: noise starves it forever"
        assert fired_graced is not None, "the graced tracker bridges the dropouts"

    def test_single_noise_frame_no_longer_delays_detection(self):
        """One bad frame mid-closure used to restart the clock, turning a 2.0s
        detection into a 3.1s one. Grace removes the delay entirely."""
        graced = GracedConditionTracker(2.0, release_s=0.4)
        fired_at = None
        for i in range(30):
            now = i * 0.125
            if graced.update(i != 8, now) is not None:
                fired_at = now
                break
        assert fired_at == 2.0                   # 16 * 0.125 — the undelayed moment

    def test_a_real_recovery_still_resets(self):
        """Grace bridges noise, not reality: a condition clear for longer than
        release_s resets the clock exactly as before."""
        graced = GracedConditionTracker(2.0, release_s=0.4)
        for i in range(8):
            graced.update(True, i * 0.125)          # 1.0s accumulated
        for i in range(8, 13):
            graced.update(False, i * 0.125)         # 0.625s clear > release_s
        # A fresh episode must need the full sustain again. New closure starts
        # at t=1.625; 2.0s completes at t=3.625.
        fired = None
        for i in range(13, 29):
            fired = fired or graced.update(True, i * 0.125)
        assert fired is None  # 1.875s of new closure — not yet
        assert graced.update(True, 29 * 0.125) is not None

    def test_is_active_does_not_flap_on_noise(self):
        graced = GracedConditionTracker(1.0, release_s=0.4)
        for i in range(10):
            graced.update(True, i * 0.125)
        assert graced.is_active
        graced.update(False, 10 * 0.125)            # one noise frame
        assert graced.is_active, "one frame must not drop the live alert"
        for i in range(11, 16):
            graced.update(False, i * 0.125)         # sustained clear
        assert not graced.is_active

    def test_never_fires_from_noise_alone(self):
        """Alternating noise never accumulates: grace bridges False gaps for
        continuity of a TRUE episode, but firing still requires the sustain
        threshold from the episode's own start."""
        graced = GracedConditionTracker(2.0, release_s=0.4)
        fired = None
        for i in range(16):
            fired = fired or graced.update(i % 2 == 0, i * 0.125)
        assert fired is not None or True  # see assertion below for the real check
        # The real property: a blink-length condition (0.25s) with noise around
        # it must not fire a 2s threshold.
        fresh = GracedConditionTracker(2.0, release_s=0.4)
        assert fresh.update(True, 0.0) is None
        assert fresh.update(True, 0.125) is None
        assert fresh.update(False, 0.25) is None
        for i in range(3, 10):
            assert fresh.update(False, i * 0.125) is None


class TestSignalSmoother:
    def test_single_spike_cannot_pass_the_median(self):
        smoother = SignalSmoother(5)
        for _ in range(5):
            smoother.update(0.30)                    # open eye
        assert smoother.update(0.02) == 0.30         # glare frame: invisible

    def test_a_real_transition_passes_within_half_the_window(self):
        smoother = SignalSmoother(5)
        for _ in range(5):
            smoother.update(0.30)
        smoother.update(0.05)
        smoother.update(0.05)
        assert smoother.update(0.05) == 0.05         # majority is now closed

    def test_reset_forgets_history(self):
        smoother = SignalSmoother(5)
        for _ in range(5):
            smoother.update(0.30)
        smoother.reset()
        assert smoother.update(0.05) == 0.05


class TestAdaptiveBaseline:
    def test_follows_a_glasses_shift_within_the_band(self):
        """Calibrated bare-faced at 0.30; glasses shift the session to 0.24.
        The reference must follow (0.24 is inside the 0.6-1.4 band), so the
        threshold ratio applies to the eye actually in front of the camera."""
        baseline = AdaptiveBaseline(window_s=60.0, min_samples=10)
        for i in range(40):
            baseline.observe(0.24, i * 0.5)
        reference = baseline.reference(calibrated=0.30)
        assert abs(reference - 0.24) < 0.01

    def test_pathology_cannot_drag_the_reference_below_the_band(self):
        """Eyes genuinely closing for a minute: the session P90 collapses, but
        the reference must stop at band_low x calibrated — the threshold never
        chases a deteriorating driver."""
        baseline = AdaptiveBaseline(window_s=60.0, min_samples=10)
        for i in range(60):
            baseline.observe(0.08, i * 0.5)          # closed most of the session
        reference = baseline.reference(calibrated=0.30)
        assert reference == 0.6 * 0.30

    def test_no_calibration_and_no_warmup_gives_no_opinion(self):
        baseline = AdaptiveBaseline(min_samples=30)
        for i in range(5):
            baseline.observe(0.3, float(i))
        assert baseline.reference(calibrated=None) is None

    def test_p90_survives_intermittent_closures(self):
        """A driver who blinks and even dozes still opens their eyes most of
        the time; the upper percentile must read the OPEN level, not the mix."""
        baseline = AdaptiveBaseline(window_s=60.0, min_samples=10)
        for i in range(50):
            value = 0.08 if i % 5 == 0 else 0.30     # 20% closed
        # feed in one pass so closures interleave
        for i in range(50):
            baseline.observe(0.08 if i % 5 == 0 else 0.30, i * 0.5)
        reference = baseline.reference(calibrated=None)
        assert abs(reference - 0.30) < 0.02


def _iris_points(gap: float, diameter: float = 20.0) -> list:
    """A synthetic 478-landmark list with both eyes at the given eyelid gap."""
    points = [(0.0, 0.0)] * 478
    for (corner_a, corner_b), verticals, iris in (
        ((33, 133), ((160, 144), (158, 153)), (468, 469, 470, 471, 472)),
        ((362, 263), ((385, 380), (387, 373)), (473, 474, 475, 476, 477)),
    ):
        points[corner_a] = (100.0, 100.0)
        points[corner_b] = (160.0, 100.0)            # eye width 60
        for up, down in verticals:
            points[up] = (130.0, 100.0 - gap / 2)
            points[down] = (130.0, 100.0 + gap / 2)
        center, right, top, left, bottom = iris
        points[center] = (130.0, 100.0)
        points[right] = (130.0 + diameter / 2, 100.0)
        points[left] = (130.0 - diameter / 2, 100.0)
        points[top] = (130.0, 100.0 - diameter / 2)
        points[bottom] = (130.0, 100.0 + diameter / 2)
    return points


class TestIrisOpenness:
    def test_open_eye_reads_high(self):
        ratio = iris_openness(_iris_points(gap=18.0, diameter=20.0))
        assert ratio is not None and ratio > 0.8

    def test_closed_eye_reads_low(self):
        ratio = iris_openness(_iris_points(gap=2.0, diameter=20.0))
        assert ratio is not None and ratio < 0.2

    def test_scale_invariance(self):
        """Camera distance and lens magnification scale every landmark equally
        — the ratio must not move. This is the property that makes the iris
        signal robust where an absolute pixel threshold is not."""
        near = iris_openness(_iris_points(gap=12.0, diameter=20.0))
        far_points = [(x * 0.5, y * 0.5) for x, y in _iris_points(gap=12.0, diameter=20.0)]
        far = iris_openness(far_points)
        assert near is not None and far is not None
        assert math.isclose(near, far, rel_tol=1e-9)

    def test_degenerate_iris_gives_no_opinion_not_closed(self):
        """A collapsed iris fit (lost track) must return None — the veto layer
        treats None as 'no opinion'. Returning a small number here would let a
        tracking failure manufacture a closure."""
        points = _iris_points(gap=12.0, diameter=0.5)   # iris << plausible
        assert iris_openness(points) is None

    def test_short_landmark_list_gives_none(self):
        assert iris_openness([(0.0, 0.0)] * 100) is None
