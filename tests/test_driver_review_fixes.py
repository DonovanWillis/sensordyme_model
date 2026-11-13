"""Regression tests for the driver-attention review findings F1-F18.

Each test reproduces a defect found by tracing the pi-profile pipeline and
fails against the code as it stood before the fix. They share
test_driver_attention_scenarios.py's harness: the real DriverAttentionModule,
real calibration, trackers and fusion, with only the inference boundaries
(FaceMesh landmarks, YOLO, MediaPipe Hands) replaced by synthetic but
geometrically real stand-ins.
"""

import json
import math
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest
import test_driver_attention_scenarios as das
import test_driver_hand_motion as hm
import test_hand_perception as thp

import modules.driver_attention as driver_attention_module
from core.activity import ActivityRecognizer
from core.eye_signal import eye_gaze_offset_deg
from core.hand_perception import (
    HandPerceptionEngine,
    detect_hand_anchors,
    hand_roi_box,
    torso_length_from_landmarks,
)
from modules.driver_attention import (
    _MODEL_POINTS_3D,
    AttentionCalibrator,
    GlanceCounter,
    RiskFusion,
    estimate_head_rotation,
    head_rotation_from_yaw_pitch_deg,
    relative_yaw_pitch_deg,
)

W, H = das.W, das.H
clock = das.clock
_make_module = das._make_module
_feed = das._feed


def _rx(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _run(module, frame, clock, landmarks, seconds, step=0.05):
    module._face_mesh.process = lambda rgb, lm=landmarks: das._FakeFaceMeshResult(lm)
    events = []
    for _ in range(max(1, round(seconds / step))):
        events += module.process(frame)
        clock["now"] += step
    return events


def _types(events):
    return [e.type for e in events]


# ------------------------------------------------------------- F1: mount coupling
def _mounted_pose_points(elev_deg, side_deg=0.0):
    """_pose_pixel_points for a camera `elev_deg` BELOW the driver's eye line
    (negative = above) and `side_deg` to the side, looking at the face. The
    head turns about the BODY's vertical axis, which is tilted in this camera's
    frame -- the physical situation of any dash-mounted camera."""
    mount = _rx(-elev_deg) @ _ry(side_deg)

    def pose_px(pitch_deg=0.0, yaw_deg=0.0):
        rotation = mount @ _ry(-yaw_deg) @ _rx(pitch_deg) @ das._R_NEUTRAL
        rvec, _ = cv2.Rodrigues(rotation)
        pts, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, das._TVEC, das._K, das._DIST)
        return pts.reshape(-1, 2)

    return pose_px


@pytest.mark.parametrize("elev_deg", [15.0, 20.0, 25.0])
def test_f1_a_blind_spot_check_with_a_low_dash_camera_is_not_eyes_up(tmp_path, clock, monkeypatch, elev_deg):
    """Camera below the eye line; a pure 70 deg head turn held 4 s -- inside
    rearview_tolerance_seconds (5 s). Per-axis Euler subtraction read it as
    +23..+29 deg pitch and fired a LOUD eyes_up at 2.5 s."""
    monkeypatch.setattr(das, "_pose_pixel_points", _mounted_pose_points(elev_deg))
    module, frame = _make_module(tmp_path, clock)
    events = _feed(module, frame, clock, yaw_deg=70.0, seconds=4.0)
    assert "eyes_up" not in events
    assert "eyes_down" not in events
    assert abs(module._last_pitch) < 3.0
    assert abs(module._last_yaw) == pytest.approx(70.0, abs=3.0)


def test_f1_turns_away_from_an_off_centre_low_camera_keep_their_true_angle(tmp_path, clock, monkeypatch):
    """Camera 20 deg low AND 20 deg to the side (the README's off-centre dash
    mount): a 55 deg turn away from it read (-46, +33) -- eyes_up, and never
    reaching mirror_check_yaw_deg (55)."""
    monkeypatch.setattr(das, "_pose_pixel_points", _mounted_pose_points(20.0, 20.0))
    module, frame = _make_module(tmp_path, clock)
    _feed(module, frame, clock, yaw_deg=-60.0, seconds=1.0)
    assert abs(module._last_yaw) == pytest.approx(60.0, abs=3.0)
    assert abs(module._last_pitch) < 3.0


def test_f1_camera_above_the_eye_line_does_not_turn_head_turns_into_eyes_down(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(das, "_pose_pixel_points", _mounted_pose_points(-20.0))
    module, frame = _make_module(tmp_path, clock)
    events = _feed(module, frame, clock, yaw_deg=70.0, seconds=4.0)
    assert "eyes_down" not in events
    assert abs(module._last_pitch) < 3.0


def test_f1_a_real_nod_still_reads_as_pitch_with_a_low_camera(tmp_path, clock, monkeypatch):
    """The fix must not lose real pitch: looking down 30 deg still fires eyes_down."""
    monkeypatch.setattr(das, "_pose_pixel_points", _mounted_pose_points(20.0))
    module, frame = _make_module(tmp_path, clock)
    events = _feed(module, frame, clock, pitch_deg=30.0, seconds=4.0)
    assert "eyes_down" in events
    assert module._last_pitch == pytest.approx(-30.0, abs=2.0)


def test_f1_relative_rotation_recovers_head_angles_for_any_mount():
    for mount in (_rx(-25.0), _rx(-20.0) @ _ry(20.0), _rx(20.0)):
        neutral = mount @ das._R_NEUTRAL
        for yaw, nod in ((70.0, 0.0), (-55.0, 0.0), (0.0, 25.0), (40.0, 20.0)):
            rotation = mount @ _ry(-yaw) @ _rx(nod) @ das._R_NEUTRAL
            got_yaw, got_pitch = relative_yaw_pitch_deg(neutral, rotation)
            assert got_yaw == pytest.approx(yaw, abs=1e-6)
            assert got_pitch == pytest.approx(-nod, abs=1e-6)


def test_f1_calibration_version_bump_rejects_old_files(tmp_path):
    """User-visible: every device recalibrates once after this upgrade."""
    assert AttentionCalibrator.FORMAT_VERSION == 5
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"version": 4, "baseline_ear": 0.3, "ear_threshold": 0.225,
                                "baseline_yaw": 3.0, "baseline_pitch": -176.0}))
    calibrator = AttentionCalibrator(path, 10.0, 0.75, 0.20, force_recalibrate=False)
    assert not calibrator.is_calibrated


# --------------------------------------------- F2: lid closure read as looking up
_UPPER_LIDS = (160, 158, 385, 387)


def _lower_upper_lids(landmarks, px):
    out = list(landmarks)
    for idx in _UPPER_LIDS:
        out[idx] = das._FakeLandmark(out[idx].x, out[idx].y + px / H)
    return out


def test_f2_eye_gaze_offset_does_not_move_when_the_lids_close():
    base = das.make_fake_landmarks()
    open_pts = [(lm.x * W, lm.y * H) for lm in base]
    for drop in (3, 6, 9, 11):
        pts = [(lm.x * W, lm.y * H) for lm in _lower_upper_lids(base, drop)]
        assert eye_gaze_offset_deg(pts) == pytest.approx(eye_gaze_offset_deg(open_pts), abs=1e-9)


def test_f2_closed_eyes_with_a_still_head_are_drowsiness_not_eyes_up(tmp_path, clock):
    """Measured before the fix: a 4 s closure with the head perfectly still
    fired drowsiness AND a loud eyes_up (fused pitch +24 deg)."""
    module, frame = _make_module(tmp_path, clock)
    events = _types(_run(module, frame, clock, _lower_upper_lids(das.make_fake_landmarks(), 11), 4.0))
    assert "drowsiness" in events
    assert "eyes_up" not in events
    assert abs(module._last_pitch) < 3.0
    assert module._last_gaze_offset is None, "a closed eye must fall back to head pose only"


# ------------------------------------------- F3: phone_use flag lost on one miss
def test_f3_one_missed_phone_tick_does_not_drop_the_live_phone_use_state(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    phone = [(280, 300, 360, 420)]
    module._last_phone_score = 0.8
    landmarks = das.make_fake_landmarks(pitch_deg=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: das._FakeFaceMeshResult(lm)
    fired_at = None
    for i in range(120):
        module._last_phone_boxes = [] if i == 70 else phone  # one detector miss after phone_use fired
        if "phone_use" in _types(module.process(frame)) and fired_at is None:
            fired_at = i
        clock["now"] += 0.05
    assert fired_at is not None and fired_at < 70
    assert module._phone_use_tracker.is_active
    assert module._phone_use_active is True
    assert module._state_label() == "PHONE_USE" or module._eyes_down_active


# ----------------------------------- F4: phone_use frozen across a face loss
def test_f4_phone_use_does_not_fire_from_time_spent_without_a_face(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    module._last_phone_boxes = [(500, 380, 600, 470)]
    module._last_phone_score = 0.7
    assert "phone_use" not in _feed(module, frame, clock, pitch_deg=30.0, seconds=1.0)

    module._face_mesh.process = lambda rgb: das._FakeFaceMeshResult(None)
    module._last_person_visible = False  # the driver stepped out: nothing to coast on
    for _ in range(int(120 / 0.05)):
        module.process(frame)
        clock["now"] += 0.05

    landmarks = das.make_fake_landmarks(pitch_deg=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: das._FakeFaceMeshResult(lm)
    first_back = _types(module.process(frame))
    assert "phone_use" not in first_back, "fired from 2 minutes nobody observed"


def test_f4_phone_use_coasts_like_the_gaze_it_depends_on(tmp_path, clock):
    """Face lost mid-glance with the body still visible: phone_use keeps
    accumulating on the coasted gaze (extrapolated confidence), consistent
    with eyes_down's own coast."""
    module, frame = _make_module(tmp_path, clock)
    module._last_phone_boxes = [(500, 380, 600, 470)]
    module._last_phone_score = 0.7
    module._hand_overlaps_phone = True  # M8.5: phone_use needs a handled phone; this one is in a hand
    _feed(module, frame, clock, pitch_deg=30.0, seconds=1.0)
    module._last_person_visible = True
    module._update_object_detections = lambda frame: None
    module._face_mesh.process = lambda rgb: das._FakeFaceMeshResult(None)
    events = []
    for _ in range(40):
        events += module.process(frame)
        clock["now"] += 0.05
    phone = [e for e in events if e.type == "phone_use"]
    assert phone and phone[0].confidence is not None and phone[0].confidence <= 0.5


# ------------------------------------------------------ F5: blink flurry
def _ear_timeline(monkeypatch, fn):
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio", lambda *pts: fn())


@pytest.mark.parametrize("fps", [15, 30])
def test_f5_a_flurry_of_ordinary_blinks_is_not_drowsiness(tmp_path, clock, monkeypatch, fps):
    """0.25 s closed / 0.30 s open, repeated: no closure longer than a blink.
    Before: the 0.4 s grace bridged every reopening and the clock counted it,
    so loud drowsiness fired at ~2.3 s with the eyes closed half the time."""
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    start = clock["now"]
    _ear_timeline(monkeypatch, lambda: 0.05 if (clock["now"] - start) % 0.55 < 0.25 else 0.40)
    events = _types(_run(module, frame, clock, das.make_fake_landmarks(), 4.0, step=1.0 / fps))
    assert "drowsiness" not in events


def test_f5_a_real_sustained_closure_still_fires_on_time(tmp_path, clock, monkeypatch):
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    _ear_timeline(monkeypatch, lambda: 0.05)
    events = _run(module, frame, clock, das.make_fake_landmarks(), 3.0, step=1.0 / 30)
    drowsy = [e for e in events if e.type == "drowsiness"]
    assert drowsy and drowsy[0].duration == pytest.approx(2.0, abs=0.05)


def test_f5_a_one_frame_glitch_mid_closure_does_not_restart_the_clock(tmp_path, clock, monkeypatch):
    """The grace must still do its original job: a single bad frame inside a
    real closure neither resets nor much delays it."""
    module, frame = _make_module(tmp_path, clock, iris_veto=False, signal_median_window=1)
    start = clock["now"]
    _ear_timeline(monkeypatch, lambda: 0.40 if abs((clock["now"] - start) - 1.0) < 0.01 else 0.05)
    events = _run(module, frame, clock, das.make_fake_landmarks(), 2.4, step=1.0 / 30)
    assert "drowsiness" in _types(events)


# ---------------------------------------------- F6: ROI-as-hand association
def test_f6_a_still_object_at_the_edge_of_the_search_region_is_not_held():
    """Before: association scored overlap against the 0.55-torso ROI, which
    every crop detection overlaps by construction -- a still bottle 0.6 torso
    lengths from a still wrist scored 0.627 (held)."""
    engine = HandPerceptionEngine(thp.OBJECT_LABELS)
    landmarks = thp._standing_with_hands(right_wrist=(0.65, 0.45))
    roi = hand_roi_box(detect_hand_anchors(landmarks)[0], torso_length_from_landmarks(landmarks), W, H, 1.0)
    rw, rh = roi[2] - roi[0], roi[3] - roi[1]
    model = thp._FakeYoloModel()
    model.responses = [[((rw - 30, rh - 30, rw - 2, rh - 2), 39, 0.8)]] * 12
    result = None
    for i in range(12):
        result = engine.update(landmarks, thp.FRAME, model, now=i * 0.1, aspect=1.0)
    assert result.objects, "test setup: the ROI crop did find the bottle"
    assert result.associations == {}


def test_f6_an_object_at_a_still_hand_is_still_held():
    engine = HandPerceptionEngine(thp.OBJECT_LABELS)
    landmarks = thp._standing_with_hands(right_wrist=(0.65, 0.45))
    roi = hand_roi_box(detect_hand_anchors(landmarks)[0], torso_length_from_landmarks(landmarks), W, H, 1.0)
    cx, cy = 0.65 * W - roi[0], 0.45 * H - roi[1]  # the wrist, in crop coordinates
    model = thp._FakeYoloModel()
    model.responses = [[((cx - 12, cy - 12, cx + 12, cy + 12), 39, 0.8)]] * 12
    result = None
    for i in range(12):
        result = engine.update(landmarks, thp.FRAME, model, now=i * 0.1, aspect=1.0)
    assert "right_hand" in result.associations


# ------------------------------------------- F7: calibration trailing window
def test_f7_an_early_glance_ages_out_instead_of_blocking_and_biasing_calibration(tmp_path):
    calibrator = AttentionCalibrator(
        tmp_path / "c.json", duration_s=10.0, ear_threshold_ratio=0.75, ear_threshold_fallback=0.2,
        force_recalibrate=True, max_pose_spread_deg=20.0, max_duration_s=30.0,
    )
    road = head_rotation_from_yaw_pitch_deg(0.0, 0.0)
    phone = head_rotation_from_yaw_pitch_deg(25.0, -40.0)
    t, calibrated_at = 0.0, None
    while t < 40.0 and calibrated_at is None:
        calibrator.observe(0.30, phone if 2.0 <= t < 5.0 else road, None, t)
        if calibrator.is_calibrated:
            calibrated_at = t
        t = round(t + 0.1, 3)
    # Before: never settled (the glance stayed in the spread forever), waited
    # the full 30 s and persisted a neutral of (2.4, -3.75).
    assert calibrated_at is not None and calibrated_at < 16.0
    yaw, pitch = relative_yaw_pitch_deg(road, calibrator.baseline_rotation)
    assert abs(yaw) < 0.01 and abs(pitch) < 0.01


# ------------------------------------------------------- F8: PERCLOS depth
def test_f8_squinting_is_not_charged_as_perclos_closure(tmp_path, clock, monkeypatch):
    """2 s squints (EAR 60% of open: below the 0.75 drowsiness ratio, far from
    P80 closure) 40% of the time. Before: PERCLOS ~0.4 -> fatigue_warning."""
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    start = clock["now"]
    _ear_timeline(monkeypatch, lambda: 0.24 if (clock["now"] - start) % 5.0 < 2.0 else 0.40)
    events = _types(_run(module, frame, clock, das.make_fake_landmarks(), 25.0))
    assert "fatigue_warning" not in events
    assert module._last_perclos == pytest.approx(0.0, abs=0.01)


def test_f8_deep_closures_still_count_toward_perclos(tmp_path, clock, monkeypatch):
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    start = clock["now"]
    _ear_timeline(monkeypatch, lambda: 0.05 if (clock["now"] - start) % 5.0 < 1.5 else 0.40)
    events = _types(_run(module, frame, clock, das.make_fake_landmarks(), 25.0))
    assert "fatigue_warning" in events
    assert module._last_perclos > 0.15


# ------------------------------------------- F9: pixel boxes and the aspect ratio
def _oscillate(axis, amplitude_px, count=24, centre=(0.5, 0.6)):
    cx, cy = centre
    out = []
    for i in range(count):
        s = 1 if i % 2 else -1
        if axis == "x":
            out.append((cx + s * amplitude_px / W, cy))
        else:
            out.append((cx, cy + s * amplitude_px / H))
    return out


def test_f9_horizontal_and_vertical_hand_motion_are_judged_alike(tmp_path, clock, monkeypatch):
    """12.5 px oscillation: 1.76 box-diagonals/s, under the 2.0 gate, in BOTH
    directions. Before, pixel boxes were multiplied by w/h horizontally, so
    the horizontal version read 2.35 and fired erratic_hand_movement."""
    results = {}
    for axis in ("x", "y"):
        module, frame = hm._hand_module(tmp_path / axis, clock, monkeypatch)
        results[axis] = "erratic_hand_movement" in hm._feed_hand_positions(
            module, frame, clock, _oscillate(axis, 12.5)
        )
    assert results == {"x": False, "y": False}


def test_f9_hand_perception_tracks_pixel_motion_isotropically():
    """The same 32 px/tick wrist motion, horizontally vs vertically, in a 4:3
    frame. The hand track's pixel speed sizes the search ROI; with the frame
    aspect applied to pixel boxes, horizontal motion read 1.33x faster and got
    a bigger ROI than the identical vertical motion."""
    sizes = {}
    for axis in ("x", "y"):
        engine = HandPerceptionEngine(thp.OBJECT_LABELS)
        result = None
        for i in range(5):
            wx = 0.5 + (i * 32 / W if axis == "x" else 0.0)
            wy = 0.45 + (i * 32 / H if axis == "y" else 0.0)
            landmarks = thp._standing_with_hands(right_wrist=(wx, wy))
            result = engine.update(landmarks, thp.FRAME, thp._FakeYoloModel(), now=i * 0.1, aspect=W / H)
        box = result.hands[0].box
        sizes[axis] = math.hypot(box[2] - box[0], box[3] - box[1])
    assert sizes["x"] == pytest.approx(sizes["y"], rel=0.02)


# --------------------------------------- F10: erratic hands stuck after an error
def test_f10_a_detector_error_clears_the_erratic_hands_signal(tmp_path, clock, monkeypatch):
    module, frame = hm._hand_module(tmp_path, clock, monkeypatch)
    hm._feed_hand_positions(module, frame, clock, hm._oscillation(0.12))
    assert module._erratic_hands_active

    class _Broken:
        names: ClassVar[dict[int, str]] = {0: "cell phone"}

        def predict(self, *args, **kwargs):
            raise RuntimeError("hailo: device lost")

    module._yolo_model = _Broken()
    module._frame_index = module._yolo_every_n_frames - 1
    module.process(frame)
    for _ in range(30):
        module.process(frame)
        clock["now"] += 0.1
    assert module._object_detection_error is not None
    assert module._erratic_hands_active is False
    assert module._state_label() != "ERRATIC_HANDS"


# ----------------------------------------------- F11: glance-counter chatter
def test_f11_one_glance_hovering_at_the_threshold_counts_once(tmp_path, clock):
    """A single glance whose angle dips back under 35 deg for 0.2 s at a time
    (enough to pass the 5-frame median, well under release_s) was counted once
    per dip: 5 dips -> frequent_glancing from ONE glance."""
    module, frame = _make_module(tmp_path, clock)
    events = []
    for _ in range(5):
        events += _feed(module, frame, clock, yaw_deg=38.0, seconds=0.5)
        events += _feed(module, frame, clock, yaw_deg=31.0, seconds=0.2)  # 4 frames: passes the median
    assert module._glance_counter.count == 1
    assert "frequent_glancing" not in events


def test_f11_separate_glances_still_count_separately():
    counter = GlanceCounter(window_s=45.0, min_count=3, release_s=0.4)
    fired = False
    t = 0.0
    for _ in range(3):
        for off in (True,) * 10 + (False,) * 20:  # 0.5 s away, 1 s back on the road
            fired |= counter.update(off, t)
            t += 0.05
    assert counter.count == 3 and fired


# ------------------------------------------------------ F12: iris veto cap
def test_f12_a_genuinely_drooping_eye_is_not_vetoed_by_a_fixed_iris_cutoff(tmp_path, clock):
    """EAR 0.25 against a 0.30 threshold, eyes ~60% open. iris-openness 0.62 >=
    the old fixed 0.5 -> vetoed, whatever ear_threshold_ratio said."""
    module, frame = _make_module(tmp_path, clock)
    events = _types(_run(module, frame, clock, _lower_upper_lids(das.make_fake_landmarks(), 4.5), 4.0))
    assert "drowsiness" in events


def test_f12_the_veto_still_catches_corrupted_eye_corners(tmp_path, clock):
    """The case the veto exists for: glare moves the eye-corner landmarks EAR
    divides by, so EAR drops while the lid gap (and iris-openness) is normal."""
    module, frame = _make_module(tmp_path, clock)
    landmarks = list(das.make_fake_landmarks())
    for idx, dx in ((133, 22.0), (362, -22.0)):  # inner corners pushed toward the nose
        landmarks[idx] = das._FakeLandmark(landmarks[idx].x + dx / W, landmarks[idx].y)
    events = _types(_run(module, frame, clock, landmarks, 4.0))
    assert module._last_ear < module._last_ear_threshold, "test setup: EAR must read closed"
    assert "drowsiness" not in events


# ---------------------------------------- F13: implausible pose rejected, not clamped
def test_f13_an_impossible_pose_burst_drives_no_gaze_alert(tmp_path, clock):
    """A head 'nodded' 120 deg -- no face does that; solvePnP degenerating does.
    Clamped to +-90 it still crossed every threshold and fired eyes_down/
    distraction; rejected it changes nothing."""
    module, frame = _make_module(tmp_path, clock)
    events = _feed(module, frame, clock, pitch_deg=120.0, seconds=4.0)
    assert not {"eyes_down", "eyes_up", "distraction"} & set(events)
    assert abs(module._last_pitch) < 1.0 and abs(module._last_yaw) < 1.0


# ---------------------------------------------- F14: calibration provenance
@pytest.mark.parametrize(
    ("override", "stored_value"),
    [("gaze_fused", False), ("eye_gaze_scale_deg", 30.0), ("camera_hfov_deg", 102.0)],
)
def test_f14_a_baseline_learned_on_a_different_signal_is_rejected(tmp_path, override, stored_value):
    path = tmp_path / "calibration.json"
    data = {
        "version": AttentionCalibrator.FORMAT_VERSION, "baseline_ear": 0.3, "ear_threshold": 0.225,
        "baseline_rotation": [float(v) for v in head_rotation_from_yaw_pitch_deg(0.0, 0.0).ravel()],
        "baseline_eye_offset": [0.0, 0.0], "baseline_yaw": 0.0, "baseline_pitch": 0.0,
        "gaze_fused": True, "eye_gaze_scale_deg": 40.0, "camera_hfov_deg": None,
    }
    path.write_text(json.dumps(data))
    assert AttentionCalibrator(path, 10.0, 0.75, 0.2, force_recalibrate=False).is_calibrated
    data[override] = stored_value
    path.write_text(json.dumps(data))
    assert not AttentionCalibrator(path, 10.0, 0.75, 0.2, force_recalibrate=False).is_calibrated


def test_f14_toggling_enable_gaze_makes_the_module_recalibrate(tmp_path, clock):
    module, _ = _make_module(tmp_path, clock)
    assert module._calibrator.is_calibrated
    again = driver_attention_module.DriverAttentionModule(
        {**das._CONFIG, "calibration_path": str(tmp_path / "calibration.json"), "enable_gaze": False}
    )
    assert not again._calibrator.is_calibrated


# ------------------------------------------------------- F15: jump without ankles
def _pose_landmarks(points):
    out = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for i, (x, y) in points.items():
        out[i] = SimpleNamespace(x=x, y=y, visibility=0.99)
    return out


def test_f15_standing_up_with_the_feet_out_of_frame_is_not_a_jump():
    recognizer = ActivityRecognizer()
    state = None
    for k in range(10):
        hip_y = 0.80 - 0.08 * min(1.0, k / 6)
        shoulder_y = hip_y - 0.30
        state = recognizer.update(_pose_landmarks({
            11: (0.45, shoulder_y), 12: (0.55, shoulder_y), 23: (0.46, hip_y), 24: (0.54, hip_y),
        }), now=k / 15.0, aspect=4 / 3)
    assert state.ready
    assert not state.jumping.present
    assert "ankle_rise_torso" not in state.signals


def test_f15_a_jump_with_visible_ankles_is_still_a_jump():
    recognizer = ActivityRecognizer()
    state = None
    for k in range(10):
        lift = 0.08 * min(1.0, k / 6)
        hip_y, ankle_y = 0.60 - lift, 0.95 - lift
        state = recognizer.update(_pose_landmarks({
            11: (0.45, hip_y - 0.30), 12: (0.55, hip_y - 0.30), 23: (0.46, hip_y), 24: (0.54, hip_y),
            27: (0.46, ankle_y), 28: (0.54, ankle_y),
        }), now=k / 15.0, aspect=4 / 3)
    assert state.jumping.present


# ------------------------------------------------ F16: flail acceleration on vectors
def test_f16_a_sharp_reversal_at_borderline_speed_trips_the_acceleration_gate():
    """A wrist snapping back and forth at 1.4 torso-lengths/s (under the 1.6
    speed gate) at 15 fps: every reversal is a ~42 torso/s^2 acceleration, but
    differencing scalar SPEEDS read it as 0."""
    recognizer = ActivityRecognizer()
    torso = 0.30
    step = 1.4 * torso / 15 / (4 / 3)  # normalised-x travel per frame for 1.4 torso/s
    state = None
    for k in range(18):
        # two frames out, two frames back: each leg (0.19 torso) clears the
        # reversal-amplitude floor, and the velocity flips between samples
        offset = (0.0, step, 2 * step, step)[k % 4]
        state = recognizer.update(_pose_landmarks({
            11: (0.45, 0.40), 12: (0.55, 0.40), 23: (0.46, 0.70), 24: (0.54, 0.70),
            15: (0.40 + offset, 0.55),
        }), now=k / 15.0, aspect=4 / 3)
    signals = state.signals
    assert signals["left_wrist_peak_speed"] < 1.6
    assert signals["left_wrist_peak_accel"] > 40.0
    assert state.flailing.present


def test_f16_a_smooth_fast_reach_has_no_acceleration():
    recognizer = ActivityRecognizer()
    state = None
    for k in range(12):
        state = recognizer.update(_pose_landmarks({
            11: (0.45, 0.40), 12: (0.55, 0.40), 23: (0.46, 0.70), 24: (0.54, 0.70),
            15: (0.20 + 0.03 * k, 0.55),
        }), now=k / 15.0, aspect=4 / 3)
    assert state.signals["left_wrist_peak_accel"] == pytest.approx(0.0, abs=1e-6)
    assert not state.flailing.present


# ---------------------------------------------------- F17: frame-rate-free EMA
def test_f17_risk_smoothing_takes_the_same_wall_clock_time_at_any_frame_rate():
    def score_after(fps, seconds):
        fusion = RiskFusion(ema_alpha=0.3)
        t, score = 0.0, 0.0
        for _ in range(int(fps * 1.0)):  # 1 s of nothing
            score, _ = fusion.update(eye_risk=0, gaze_risk=0, phone_risk=0, eating_risk=0, face_lost_risk=0, now=t)
            t += 1.0 / fps
        for _ in range(round(fps * seconds)):
            score, _ = fusion.update(eye_risk=1, gaze_risk=1, phone_risk=0, eating_risk=0, face_lost_risk=0, now=t)
            t += 1.0 / fps
        return score

    # Before: 0.25 s of maximal input reached 0.92 at 30 fps but 0.49 at 8 fps.
    assert score_after(8, 0.25) == pytest.approx(score_after(30, 0.25), abs=0.06)


# ------------------------------------------------------ F18: camera intrinsics
def _project(rotation, fx, tvec):
    k = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    pts, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, tvec, k, np.zeros((4, 1)))
    return pts.reshape(-1, 2)


def test_f18_a_configured_wide_lens_recovers_the_true_turn_angle():
    fx = (W / 2) / math.tan(math.radians(102.0) / 2)  # Camera Module 3 Wide
    tvec = np.array([650.0 * math.tan(math.radians(20.0)), 0.0, 650.0])  # face 20 deg off-axis
    neutral_pts = _project(das._R_NEUTRAL, fx, tvec)

    def yaw(turn, hfov):
        turned_pts = _project(_ry(-turn) @ das._R_NEUTRAL, fx, tvec)
        neutral = estimate_head_rotation(neutral_pts, (H, W), hfov)
        return relative_yaw_pitch_deg(neutral, estimate_head_rotation(turned_pts, (H, W), hfov))[0]

    for turn in (35.0, -35.0):
        assert yaw(turn, 102.0) == pytest.approx(turn, abs=0.5)
    # Documents what the default (f = width) costs on this lens: the turn
    # toward the camera's side under-reads by ~3.4 deg.
    assert abs(yaw(-35.0, None) - (-35.0)) > 2.0


def test_f18_the_default_intrinsics_are_unchanged(tmp_path, clock):
    module, _ = _make_module(tmp_path, clock)
    assert module._camera_hfov_deg is None
    k = driver_attention_module._camera_matrix((H, W), None)
    assert k[0, 0] == W and k[1, 1] == W

