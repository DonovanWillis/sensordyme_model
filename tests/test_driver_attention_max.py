"""Unit tests for the mac_max profile's fusion logic, via synthetic signal timelines.

These test the underlying trackers/classes directly rather than the full
DriverAttentionMaxModule, which eagerly loads MediaPipe and (on its very first
process() call, since mac_max runs detection every frame with no stride) lazily
loads a ~109MB YOLO11-x checkpoint -- exactly the same synthetic-timeline approach
the pi profile's tests (test_drowsiness.py, test_distraction.py) already use, so
this suite still runs on a Mac with no camera, no GPU, and no network.
"""

import json
from pathlib import Path

import test_driver_attention_scenarios as das
import yaml

import main as main_module
from modules.base import SustainedConditionTracker
from modules.driver_attention import GlanceCounter, angle_delta_deg, estimate_head_pose_deg
from modules.driver_attention_max import (
    BlinkRateTracker,
    CumulativeConditionTracker,
    DriverAttentionMaxModule,
    GazeCalibrator,
    RollingMedianFilter,
)

MICROSLEEP_SECONDS = 0.5
DROWSINESS_SECONDS = 1.5
GAZE_OFF_ROAD_SECONDS = 2.0
GAZE_YAW_THRESHOLD_DEG = 20.0
EXCESSIVE_GLANCE_WINDOW_S = 12.0
EXCESSIVE_GLANCE_BUDGET_S = 4.5


# --- 20 normal blinks -> zero events ------------------------------------------------


def test_twenty_normal_blinks_produce_zero_microsleep_events():
    """20 blinks of ~200ms each -- well under the 500ms 'ignore' cutoff -- must never
    sustain long enough to cross the 0.5s microsleep threshold."""
    tracker = SustainedConditionTracker(MICROSLEEP_SECONDS)
    fired_count = 0
    t = 0.0
    for _ in range(20):
        assert tracker.update(True, t) is None
        t += 0.2  # eyes closed for ~200ms
        fired = tracker.update(True, t)
        if fired is not None:
            fired_count += 1
        t += 0.05
        tracker.update(False, t)  # reopen
        t += 0.3  # gap before the next blink
    assert fired_count == 0


def test_twenty_normal_blinks_never_reach_drowsiness_either():
    tracker = SustainedConditionTracker(DROWSINESS_SECONDS)
    fired_count = 0
    t = 0.0
    for _ in range(20):
        tracker.update(True, t)
        t += 0.2
        fired = tracker.update(True, t)
        if fired is not None:
            fired_count += 1
        t += 0.05
        tracker.update(False, t)
        t += 0.3
    assert fired_count == 0


# --- 1.2s mirror check -> zero events -----------------------------------------------


def test_1_2s_mirror_check_produces_zero_eyes_off_road_events():
    """A 1.2s glance away (a mirror check) is well under the 2.0s NHTSA
    single-glance threshold and must never fire eyes_off_road."""
    tracker = SustainedConditionTracker(GAZE_OFF_ROAD_SECONDS)
    series = [(0.0, 0.0), (0.1, 30.0), (1.3, 30.0), (1.4, 0.0), (2.0, 0.0)]
    fired_count = 0
    for now, gaze_yaw in series:
        fired = tracker.update(abs(gaze_yaw) > GAZE_YAW_THRESHOLD_DEG, now)
        if fired is not None:
            fired_count += 1
    assert fired_count == 0


# --- 3 short glances totaling 5s in a 12s window -> excessive_glancing fires --------


def test_three_short_glances_totaling_5s_in_12s_window_fires_excessive_glancing():
    """Three glances of ~1.7s each (5s total), none individually crossing the 2.0s
    single-glance threshold, but cumulatively exceeding the 4.5s/12s budget -- this
    is the texting pattern excessive_glancing exists to catch, which single-glance
    logic alone would miss entirely."""
    single_glance_tracker = SustainedConditionTracker(GAZE_OFF_ROAD_SECONDS)
    cumulative_tracker = CumulativeConditionTracker(EXCESSIVE_GLANCE_WINDOW_S)

    segments = [
        (0.0, True),
        (1.7, False),
        (3.0, True),
        (4.7, False),
        (6.0, True),
        (7.7, False),
        (12.0, False),
    ]
    single_glance_fired = 0
    excessive_glancing_fired = False
    for now, off_road in segments:
        fired = single_glance_tracker.update(off_road, now)
        if fired is not None:
            single_glance_fired += 1
        cumulative_seconds = cumulative_tracker.update(off_road, now)
        if cumulative_seconds > EXCESSIVE_GLANCE_BUDGET_S:
            excessive_glancing_fired = True

    assert single_glance_fired == 0  # no single glance ever reached 2.0s
    assert excessive_glancing_fired is True  # but the ~5s cumulative total crossed 4.5s


def test_cumulative_tracker_does_not_overestimate_before_a_full_window_elapses():
    """Regression guard: converting a *fraction* of elapsed history into seconds by
    multiplying by the full window size overestimates cumulative time before a full
    window's worth of history exists (e.g. 2 seconds in, entirely off-road, is 2
    cumulative seconds -- not 12). CumulativeConditionTracker must return the former,
    not the latter."""
    tracker = CumulativeConditionTracker(window_s=12.0)
    tracker.update(True, now=0.0)
    cumulative = tracker.update(True, now=2.0)  # only 2s have elapsed, all off-road
    assert cumulative == 2.0  # not 12.0


def test_cumulative_tracker_forgets_time_outside_the_window():
    tracker = CumulativeConditionTracker(window_s=12.0)
    tracker.update(True, now=0.0)
    tracker.update(False, now=2.0)
    # Big gap -- the early off-road segment ages out of the 12s window.
    cumulative = tracker.update(True, now=20.0)
    assert cumulative < 2.0


# --- phone with eyes forward -> no audible alert ------------------------------------


def test_phone_alone_event_type_is_not_alert_worthy_for_mac_max():
    """phone visible, eyes forward, no hand overlap -> log-only per spec (mounted
    GPS is legitimate). The event type the module emits for this ("phone_detected")
    must not be in the set that reaches the AlertManager."""
    assert "phone_detected" not in main_module.ALERT_EVENT_TYPES_MAC_MAX


def test_phone_in_hand_and_phone_use_while_distracted_are_alert_worthy():
    """The two tiers that DO involve a hand (and, for the loudest tier, distraction)
    must reach the AlertManager -- only the "phone alone" tier is silent."""
    assert "phone_in_hand" in main_module.ALERT_EVENT_TYPES_MAC_MAX
    assert "phone_use_while_distracted" in main_module.ALERT_EVENT_TYPES_MAC_MAX


def test_confidence_stacking_matches_config():
    """The three phone-fusion tiers must have strictly increasing confidence,
    matching the 'confidence stacking, not single triggers' design."""
    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    da_config = config["mac_max"]["driver_attention"]
    assert da_config["phone_alone_confidence"] < da_config["phone_in_hand_confidence"]
    assert da_config["phone_in_hand_confidence"] < da_config["phone_use_distracted_confidence"]


# --- GazeCalibrator ------------------------------------------------------------------


def test_gaze_calibrator_uses_fallback_and_zero_baseline_before_calibrating(tmp_path):
    calibrator = GazeCalibrator(
        calibration_path=tmp_path / "gaze_calibration.json",
        duration_s=3.0,
        ear_threshold_ratio=0.75,
        ear_threshold_fallback=0.20,
        force_recalibrate=False,
    )
    assert calibrator.is_calibrated is False
    assert calibrator.ear_threshold == 0.20
    assert calibrator.baseline_gaze_yaw == 0.0
    assert calibrator.seconds_remaining == 3.0


def test_gaze_calibrator_computes_and_persists_baseline_after_duration(tmp_path):
    path = tmp_path / "gaze_calibration.json"
    calibrator = GazeCalibrator(path, duration_s=3.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False)

    calibrator.observe(0.30, gaze_yaw=15.0, gaze_pitch=-2.0, now=0.0)
    calibrator.observe(0.32, gaze_yaw=17.0, gaze_pitch=-4.0, now=1.5)
    assert calibrator.is_calibrated is False

    calibrator.observe(0.28, gaze_yaw=19.0, gaze_pitch=-3.0, now=3.0)
    assert calibrator.is_calibrated is True

    expected_yaw_baseline = (15.0 + 17.0 + 19.0) / 3
    assert calibrator.baseline_gaze_yaw == expected_yaw_baseline

    data = json.loads(path.read_text())
    assert data["baseline_gaze_yaw"] == expected_yaw_baseline
    assert "baseline_gaze_pitch" in data


def test_gaze_calibrator_loads_persisted_state(tmp_path):
    path = tmp_path / "gaze_calibration.json"
    path.write_text(
        json.dumps({"baseline_ear": 0.3, "ear_threshold": 0.225, "baseline_gaze_yaw": 12.0, "baseline_gaze_pitch": -5.0})
    )
    calibrator = GazeCalibrator(path, duration_s=3.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False)
    assert calibrator.is_calibrated is True
    assert calibrator.baseline_gaze_yaw == 12.0
    assert calibrator.baseline_gaze_pitch == -5.0


def test_gaze_calibrator_force_recalibrate_ignores_existing_file(tmp_path):
    path = tmp_path / "gaze_calibration.json"
    path.write_text(
        json.dumps({"baseline_ear": 0.3, "ear_threshold": 0.225, "baseline_gaze_yaw": 12.0, "baseline_gaze_pitch": -5.0})
    )
    calibrator = GazeCalibrator(path, duration_s=3.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=True)
    assert calibrator.is_calibrated is False


# --- BlinkRateTracker ----------------------------------------------------------------


def test_blink_rate_tracker_no_anomaly_before_baseline_established():
    tracker = BlinkRateTracker(window_s=120.0, baseline_seconds=300.0, low_per_min=8.0, high_per_min=30.0, confirm_s=30.0)
    tracker.record_blink(now=0.0)
    assert tracker.update(now=1.0) is None  # baseline window (300s) hasn't elapsed


def test_blink_rate_tracker_flags_a_sustained_low_rate_after_baseline():
    tracker = BlinkRateTracker(window_s=120.0, baseline_seconds=10.0, low_per_min=8.0, high_per_min=30.0, confirm_s=5.0)
    # Establish a normal baseline: one blink every 3s for the first 10s (~20/min).
    for t in (0.0, 3.0, 6.0, 9.0):
        tracker.record_blink(t)
    assert tracker.update(now=10.0) is None  # baseline just established, no anomaly check yet this call

    # Now go quiet for a long stretch -- rate drops well under 8/min.
    fired_at = None
    for t in range(11, 90):
        result = tracker.update(now=float(t))
        if result == "fatigue_pattern":
            fired_at = t
            break
    assert fired_at is not None


def test_blink_rate_tracker_flags_a_sustained_high_rate_after_baseline():
    tracker = BlinkRateTracker(window_s=120.0, baseline_seconds=10.0, low_per_min=8.0, high_per_min=30.0, confirm_s=5.0)
    for t in (0.0, 3.0, 6.0, 9.0):
        tracker.record_blink(t)
    tracker.update(now=10.0)

    # Blink very rapidly (well above 30/min) for a sustained stretch.
    fired_at = None
    t = 10.5
    while t < 60.0:
        tracker.record_blink(t)
        result = tracker.update(now=t)
        if result == "fatigue_pattern":
            fired_at = t
            break
        t += 1.0
    assert fired_at is not None


# --- RollingMedianFilter --------------------------------------------------------------


def test_rolling_median_filter_smooths_a_single_flicker():
    filt = RollingMedianFilter(size=5)
    for value in (10.0, 10.0, 10.0):
        smoothed = filt.update(value)
    smoothed = filt.update(100.0)  # a single flickered outlier
    assert smoothed == 10.0  # median of [10,10,10,100] is 10
    smoothed = filt.update(10.0)
    assert smoothed == 10.0


def test_rolling_median_filter_as_a_boolean_majority_vote():
    filt = RollingMedianFilter(size=5)
    votes = [1.0, 1.0, 0.0, 1.0, 0.0]  # 3 True out of 5
    result = 0.0
    for v in votes:
        result = filt.update(v)
    assert result >= 0.5  # majority True


# --- GlanceCounter reused for hand-to-face repeated-approach detection --------------


def test_glance_counter_reused_for_hand_to_face_approach_pattern():
    """Same reusable episode-counting class as the pi profile's frequent_glancing,
    repurposed here for 'hand approached the face >= 2 times in 30s' (the eating
    pattern this profile's eating_drinking fusion looks for)."""
    counter = GlanceCounter(window_s=30.0, min_count=2)
    now = 0.0
    assert counter.update(True, now) is False  # first approach
    now += 0.5
    counter.update(False, now)
    now += 5.0
    assert counter.update(True, now) is True  # second approach within 30s -> fires


# --- _compute_gaze_components sign convention ---------------------------------------
# Found while porting the same fusion to the pi profile (core/eye_signal.py's
# eye_gaze_offset_deg, tests/test_eye_gaze.py): the un-negated formula pushes pitch
# the WRONG way relative to estimate_head_pose_deg's own verified "down = negative"
# convention. _compute_gaze_components has no `self.*` access at all (pure function
# of `points`), so it's called unbound, without constructing the full module (which
# would require MPS + eagerly load a ~109MB YOLO checkpoint on first process()).
def _shift_iris(landmarks, dy_px: float):
    out = list(landmarks)
    for idx in (468, 469, 470, 471, 472, 473, 474, 475, 476, 477):
        out[idx] = das._FakeLandmark(out[idx].x, out[idx].y + dy_px / das.H)
    return out


def test_compute_gaze_components_pitch_matches_head_pitchs_down_is_negative_convention():
    pose_neutral = estimate_head_pose_deg(das._pose_pixel_points(0.0, 0.0), (das.H, das.W))
    pose_down = estimate_head_pose_deg(das._pose_pixel_points(10.0, 0.0), (das.H, das.W))
    assert angle_delta_deg(pose_down[1], pose_neutral[1]) < 0  # verified "looking down" rotation

    landmarks = _shift_iris(das.make_fake_landmarks(), dy_px=8.0)  # iris toward the lower lid
    points = [(lm.x * das.W, lm.y * das.H) for lm in landmarks]
    _, eye_pitch = DriverAttentionMaxModule._compute_gaze_components(None, points)
    assert eye_pitch < 0, (
        "iris shifted toward the lower lid must produce a NEGATIVE pitch component, "
        "matching estimate_head_pose_deg's own 'down = negative' convention -- the "
        "same bug found and fixed in the pi profile's eye_gaze_offset_deg"
    )


def test_compute_gaze_components_is_zero_for_a_centered_iris():
    points = [(lm.x * das.W, lm.y * das.H) for lm in das.make_fake_landmarks()]
    yaw, pitch = DriverAttentionMaxModule._compute_gaze_components(None, points)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
