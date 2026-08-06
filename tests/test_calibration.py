"""Unit tests for per-user EAR + neutral-head-pose calibration and its persistence."""

import json

from modules.driver_attention import AttentionCalibrator


def test_uses_fallback_threshold_and_zero_baseline_pose_before_calibration_completes(tmp_path):
    calibrator = AttentionCalibrator(
        calibration_path=tmp_path / "calibration.json",
        duration_s=10.0,
        ear_threshold_ratio=0.75,
        ear_threshold_fallback=0.20,
        force_recalibrate=False,
    )
    assert calibrator.is_calibrated is False
    assert calibrator.ear_threshold == 0.20
    assert calibrator.baseline_yaw == 0.0
    assert calibrator.baseline_pitch == 0.0


def test_computes_and_persists_ear_and_pose_baseline_once_duration_elapses(tmp_path):
    path = tmp_path / "calibration.json"
    calibrator = AttentionCalibrator(
        path, duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )

    calibrator.observe(0.30, yaw=10.0, pitch=-5.0, now=0.0)
    calibrator.observe(0.32, yaw=12.0, pitch=-7.0, now=5.0)
    assert calibrator.is_calibrated is False  # duration hasn't elapsed yet

    calibrator.observe(0.28, yaw=14.0, pitch=-6.0, now=10.0)  # duration elapsed -> calibrates on this call
    assert calibrator.is_calibrated is True

    expected_ear_baseline = (0.30 + 0.32 + 0.28) / 3
    expected_yaw_baseline = (10.0 + 12.0 + 14.0) / 3
    expected_pitch_baseline = (-5.0 + -7.0 + -6.0) / 3
    assert calibrator.ear_threshold == expected_ear_baseline * 0.75
    assert calibrator.baseline_yaw == expected_yaw_baseline
    assert calibrator.baseline_pitch == expected_pitch_baseline

    data = json.loads(path.read_text())
    assert data["baseline_ear"] == expected_ear_baseline
    assert data["baseline_yaw"] == expected_yaw_baseline
    assert data["baseline_pitch"] == expected_pitch_baseline


def test_off_center_camera_mount_calibrates_a_nonzero_neutral_yaw(tmp_path):
    """A camera mounted to the side sees the driver's forward gaze at a nonzero yaw --
    calibration should learn that offset rather than assuming 0."""
    path = tmp_path / "calibration.json"
    calibrator = AttentionCalibrator(
        path, duration_s=5.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )
    # Driver looking straight at the road, but the off-center camera reads it as ~25 deg yaw.
    for t in (0.0, 2.0, 5.0):
        calibrator.observe(0.30, yaw=25.0, pitch=0.0, now=t)
    assert calibrator.is_calibrated is True
    assert calibrator.baseline_yaw == 25.0


def test_missing_pose_samples_fall_back_to_zero_baseline(tmp_path):
    path = tmp_path / "calibration.json"
    calibrator = AttentionCalibrator(
        path, duration_s=5.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )
    calibrator.observe(0.30, yaw=None, pitch=None, now=0.0)
    calibrator.observe(0.30, yaw=None, pitch=None, now=5.0)
    assert calibrator.is_calibrated is True
    assert calibrator.baseline_yaw == 0.0
    assert calibrator.baseline_pitch == 0.0


def test_loads_persisted_calibration_on_next_run(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"baseline_ear": 0.30, "ear_threshold": 0.225, "baseline_yaw": 18.0, "baseline_pitch": -3.0})
    )

    calibrator = AttentionCalibrator(
        path, duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )
    assert calibrator.is_calibrated is True
    assert calibrator.ear_threshold == 0.225
    assert calibrator.baseline_yaw == 18.0
    assert calibrator.baseline_pitch == -3.0

    # Loaded calibration means further observations are ignored.
    calibrator.observe(0.05, yaw=0.0, pitch=0.0, now=0.0)
    assert calibrator.ear_threshold == 0.225


def test_force_recalibrate_ignores_existing_file(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"baseline_ear": 0.30, "ear_threshold": 0.225, "baseline_yaw": 18.0, "baseline_pitch": -3.0})
    )

    calibrator = AttentionCalibrator(
        path, duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=True
    )
    assert calibrator.is_calibrated is False
    assert calibrator.ear_threshold == 0.20
    assert calibrator.baseline_yaw == 0.0


def test_corrupt_calibration_file_falls_back_gracefully(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ not valid json")

    calibrator = AttentionCalibrator(
        path, duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )
    assert calibrator.is_calibrated is False
    assert calibrator.ear_threshold == 0.20


def test_old_schema_calibration_file_is_treated_as_uncalibrated(tmp_path):
    """A calibration.json from before yaw/pitch baselines existed lacks required keys --
    it should be treated as uncalibrated (triggering fresh calibration) rather than crashing."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"baseline_ear": 0.30, "threshold": 0.225}))  # old key name: "threshold"

    calibrator = AttentionCalibrator(
        path, duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.20, force_recalibrate=False
    )
    assert calibrator.is_calibrated is False
