"""End-to-end scenario tests for DriverAttentionModule, driven by SYNTHETIC but
GEOMETRICALLY REAL face landmarks -- not hand-picked yaw/pitch numbers.

Real MediaPipe FaceMesh inference needs an actual photographed face, which
this environment cannot produce. The next best thing, and considerably more
rigorous than asserting against made-up numbers, is what's done here: the
same synthetic-landmark generator validated in test_head_pose_geometry.py
(known 3D rotation -> cv2.projectPoints -> 2D landmarks) produces the 6
head-pose landmarks for a specific, physically-described pose ("head pitched
down 25 degrees"), the eye landmarks are set to a fixed, clearly-open shape
(these tests are about gaze, not drowsiness), and module._face_mesh.process
is monkeypatched to return them in place of real inference. Everything
downstream of that -- calibration, smoothing, the CorroboratedGraceTracker
and GracedConditionTracker instances, RiskFusion, event firing -- is the
real, unmodified module code.
"""

import time

import cv2
import numpy as np
import pytest

import modules.driver_attention as driver_attention_module
from modules.driver_attention import _MODEL_POINTS_3D, POSE_LANDMARK_IDS, DriverAttentionModule

W, H = 640, 480
_FOCAL = W
_CENTER = (W / 2.0, H / 2.0)
_K = np.array([[_FOCAL, 0, _CENTER[0]], [0, _FOCAL, _CENTER[1]], [0, 0, 1]], dtype=np.float64)
_DIST = np.zeros((4, 1))
_TVEC = np.array([0.0, 0.0, 600.0])
_R_NEUTRAL = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)  # rot_x(180) -- see test_head_pose_geometry.py

_POSE_ORDER = ["nose_tip", "chin", "left_eye_corner", "right_eye_corner", "left_mouth", "right_mouth"]


def _rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _pose_pixel_points(pitch_deg=0.0, yaw_deg=0.0):
    """2D pixel positions of the 6 head-pose landmarks for a described pose.

    Composition order matches test_head_pose_geometry.py's combined-pose test:
    pitch, then yaw, applied on top of the upright-facing-camera extrinsic.
    """
    rotation = _rot_x(pitch_deg) @ _rot_y(yaw_deg) @ _R_NEUTRAL
    rvec, _ = cv2.Rodrigues(rotation)
    pts2d, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, _TVEC, _K, _DIST)
    return pts2d.reshape(-1, 2)


class _FakeLandmark:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeFaceProto:
    def __init__(self, landmark):
        self.landmark = landmark


class _FakeFaceMeshResult:
    def __init__(self, landmark):
        self.multi_face_landmarks = [_FakeFaceProto(landmark)] if landmark is not None else None


def _open_eye_points(x0, y0):
    """Pixel offsets (relative to (x0, y0)) for one clearly-open eye: EAR ~0.4,
    iris-openness ~1.0 -- see core/eye_signal.py for the geometry these ratios
    come from. Returns {local_offset_name: (dx, dy)}; callers map onto the
    real landmark indices for each of the two eyes (they use the same shape,
    mirrored only in which indices they're assigned to, not in geometry).
    """
    return {
        "corner_a": (x0, y0), "corner_b": (x0 + 30, y0),
        "upper_a": (x0 + 8, y0 - 6), "lower_a": (x0 + 8, y0 + 6),
        "upper_b": (x0 + 22, y0 - 6), "lower_b": (x0 + 22, y0 + 6),
        "iris_center": (x0 + 15, y0), "iris_right": (x0 + 21, y0), "iris_left": (x0 + 9, y0),
    }


def make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0):
    """A full ~478-entry MediaPipe-shaped landmark list for a described head pose.

    Everything not explicitly a head-pose or eye landmark sits at frame
    center -- harmless, since nothing in this module reads those indices.

    Landmarks 33 and 263 are BOTH a head-pose corner (POSE_LANDMARK_IDS) AND
    an EAR eye corner (RIGHT_EYE/LEFT_EYE) -- the real MediaPipe topology
    reuses the same physical eye-corner point for both, and this must too:
    each eye's shape is built as small offsets FROM that already-rotated
    corner position, never as independent fixed coordinates, or the two
    writes collide and the second silently overwrites the pose-rotated point
    solvePnP needs with a static one, corrupting the pose estimate.
    """
    points = [[0.5, 0.5] for _ in range(478)]

    pose_px = _pose_pixel_points(pitch_deg, yaw_deg)
    for key, (px, py) in zip(_POSE_ORDER, pose_px):
        idx = POSE_LANDMARK_IDS[key]
        points[idx] = [px / W, py / H]

    # Right eye (subject's right; MediaPipe indices 33/160/158/133/153/144 + iris 468-472).
    # Anchored on landmark 33's own (already pose-rotated) position, not a fixed pixel.
    r0x, r0y = points[POSE_LANDMARK_IDS["left_eye_corner"]][0] * W, points[POSE_LANDMARK_IDS["left_eye_corner"]][1] * H
    r = _open_eye_points(r0x, r0y)
    for idx, key in ((133, "corner_b"), (160, "upper_a"), (144, "lower_a"),
                      (158, "upper_b"), (153, "lower_b")):
        px, py = r[key]
        points[idx] = [px / W, py / H]
    for idx, key in ((468, "iris_center"), (469, "iris_right"), (471, "iris_left")):
        px, py = r[key]
        points[idx] = [px / W, py / H]
    points[470] = points[468]  # iris top/bottom: unused by iris_openness's ratio, center is a safe fill
    points[472] = points[468]

    # Left eye (subject's left; indices 362/385/387/263/373/380 + iris 473-477).
    # Landmark 263 is this eye's corner_b, not corner_a -- anchor accordingly
    # (corner_b sits at x0+30 in _open_eye_points, so back out x0 = 263's x - 30).
    l1x = points[POSE_LANDMARK_IDS["right_eye_corner"]][0] * W - 30
    l1y = points[POSE_LANDMARK_IDS["right_eye_corner"]][1] * H
    lft = _open_eye_points(l1x, l1y)
    for idx, key in ((362, "corner_a"), (385, "upper_a"), (380, "lower_a"),
                      (387, "upper_b"), (373, "lower_b")):
        px, py = lft[key]
        points[idx] = [px / W, py / H]
    for idx, key in ((473, "iris_center"), (474, "iris_right"), (476, "iris_left")):
        px, py = lft[key]
        points[idx] = [px / W, py / H]
    points[475] = points[473]
    points[477] = points[473]

    return [_FakeLandmark(x, y) for x, y in points]


_CONFIG = {
    "ear_threshold_ratio": 0.75,
    "ear_threshold_fallback": 0.20,
    "calibration_duration_s": 1.0,
    "calibration_max_pose_spread_deg": 5.0,
    "calibration_max_duration_s": 2.0,
    "drowsy_seconds": 2.0,
    "perclos_window_s": 60.0,
    "perclos_threshold": 0.15,
    "yaw_threshold_deg": 35,
    "distraction_seconds": 3.0,
    "glance_window_s": 45.0,
    "glance_min_count": 5,
    "pitch_threshold_deg": -20,
    "eyes_down_seconds": 2.5,
    "pitch_up_threshold_deg": 20,
    "eyes_up_seconds": 2.5,
    "face_lost_seconds": 2.5,
    "yolo_weights": "yolo11n.pt",
    "yolo_every_n_frames": 1_000_000,  # keep real YOLO from ever loading in these tests
    "yolo_confidence_threshold": 0.5,
    "yolo_classes": {"cell phone": "phone_detected"},
    "phone_use_seconds": 2.0,
    "enable_hands": True,
    "hand_phone_iou_threshold": 0.2,
    "risk_ema_alpha": 0.3,
    "risk_elevated_threshold": 0.4,
    "risk_high_threshold": 0.75,
    "risk_confirm_seconds": 1.0,
}


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    return t


def _make_module(tmp_path, clock, **overrides):
    """A calibrated module: runs the calibration window on NEUTRAL pose first,
    exactly like a driver looking at the road when the device boots, then
    hands back control at t=0 equivalent for the scenario itself.
    """
    config = {**_CONFIG, "calibration_path": str(tmp_path / "calibration.json"), **overrides}
    module = DriverAttentionModule(config)
    neutral = make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # A bit past calibration_duration_s, all neutral -> settles immediately.
    steps = int(_CONFIG["calibration_duration_s"] / 0.05) + 3
    for _ in range(steps):
        module._face_mesh.process = lambda rgb, lm=neutral: _FakeFaceMeshResult(lm)
        module.process(frame)
        clock["now"] += 0.05
    assert module._calibrator.is_calibrated
    return module, frame


def _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=0.0, seconds=0.0, step=0.05):
    """Drive `seconds` worth of frames at a described pose; returns fired event types."""
    landmarks = make_fake_landmarks(pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    fired = []
    steps = max(1, int(seconds / step))
    for _ in range(steps):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += step
    return fired


# ------------------------------------------------------------------ scenarios
def test_looking_directly_forward_stays_focused(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=0.0, seconds=4.0)
    assert "eyes_down" not in fired
    assert "eyes_up" not in fired
    assert "distraction" not in fired
    assert module.is_alert_active() is False


def test_calibration_window_noise_does_not_carry_a_head_start_into_the_calibrated_period(tmp_path, clock):
    """A second, distinct bug this suite's rigor caught: before calibration
    completes, baseline_yaw/baseline_pitch are the 0.0 placeholder, not a
    real reading -- and for this face model, an ordinary upright pose's raw
    pitch sits near the +/-180 wrap (see AttentionCalibrator's docstring), so
    a delta computed against 0.0 during the calibration window can swing
    close to 180 degrees on pure noise, easily crossing the eyes_down/
    distraction thresholds and starting a tracker's clock at t=0 -- before
    the driver has done anything. Left uncaught, that tracker keeps running
    underneath calibration, and a driver who looks at the road for the
    ENTIRE calibrated period afterward could still see a false "eyes_down"
    shortly after calibration finishes, credited from time that was never
    real. `_make_module` already runs a full calibration window; this
    confirms nothing carried through it.
    """
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=0.0, seconds=4.0)
    assert "eyes_down" not in fired
    assert "distraction" not in fired
    assert "eyes_up" not in fired
    assert module._eyes_down_tracker.elapsed(clock["now"]) == 0.0
    assert module._distract_tracker.elapsed(clock["now"]) == 0.0


def test_a_brief_glance_down_does_not_fire(tmp_path, clock):
    """The temporal-tolerance requirement: momentary is not the same as sustained."""
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=30.0, seconds=0.8)  # well under eyes_down_seconds=2.5
    assert "eyes_down" not in fired
    assert module.is_alert_active() is False


def test_looking_down_for_over_five_seconds_definitely_triggers(tmp_path, clock):
    """The user's explicit bar: sustained gaze outside the focused zone for
    ~5 seconds must trigger. eyes_down_seconds=2.5 in this config clears that
    bar with margin; fed for 5s to match the exact scenario asked for."""
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=30.0, seconds=5.0)
    assert "eyes_down" in fired
    assert module.is_alert_active() is True


def test_looking_significantly_left_triggers_distraction(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, yaw_deg=45.0, seconds=4.0)
    assert "distraction" in fired
    assert module.is_alert_active() is True


def test_looking_significantly_right_triggers_distraction(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, yaw_deg=-45.0, seconds=4.0)
    assert "distraction" in fired
    assert module.is_alert_active() is True


def test_looking_all_the_way_up_triggers(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=-30.0, seconds=4.0)
    assert "eyes_up" in fired
    assert module.is_alert_active() is True


def test_an_anatomically_impossible_pose_still_clamps_to_a_plausible_gaze_angle(tmp_path, clock):
    """Found live: a fast head turn broke FaceMesh lock mid-motion, and
    solvePnP on the degrading landmarks briefly produced pitch=-92deg,
    yaw=-76.51deg -- angles no real gaze delta from a windshield-mounted
    camera can reach. Those values then got HELD by the face-loss coast path
    (see MAX_PLAUSIBLE_GAZE_DEG's own comment) and fed straight into
    eyes_down/distraction confidence. An anatomically extreme synthetic pose
    (well past any real head rotation) exercises the same solvePnP-extremity
    path; regardless of what raw angle it produces, self._last_pitch/_last_yaw
    must never carry a value outside the plausible range."""
    module, frame = _make_module(tmp_path, clock)
    _feed(module, frame, clock, pitch_deg=-150.0, yaw_deg=170.0, seconds=0.5)
    assert abs(module._last_pitch) <= driver_attention_module.MAX_PLAUSIBLE_GAZE_DEG
    assert abs(module._last_yaw) <= driver_attention_module.MAX_PLAUSIBLE_GAZE_DEG


def test_looking_down_at_a_phone_for_over_five_seconds_is_very_high_confidence(tmp_path, clock):
    """Phone + downward gaze + sustained duration -- the exact combination the
    product brief calls out as needing a hard, non-averaged trigger."""
    module, frame = _make_module(tmp_path, clock)
    module._last_phone_boxes = [(280, 300, 360, 420)]
    module._hand_overlaps_phone = True
    fired = _feed(module, frame, clock, pitch_deg=30.0, seconds=5.0)
    assert "eyes_down" in fired
    assert "phone_use" in fired
    assert module.is_alert_active() is True


def test_phone_use_confidence_is_high_when_a_hand_is_on_it(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock)
    module._last_phone_boxes = [(280, 300, 360, 420)]
    module._hand_overlaps_phone = True
    landmarks = make_fake_landmarks(pitch_deg=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    events = []
    for _ in range(50):  # 2.5s @ 0.05s step
        events += module.process(frame)
        clock["now"] += 0.05
    phone_events = [e for e in events if e.type == "phone_use"]
    assert phone_events, "phone_use should have fired"
    assert phone_events[0].confidence >= 0.9


def test_face_temporarily_occluded_does_not_instantly_reset_progress(tmp_path, clock):
    """A one-frame tracking dropout mid-episode must not restart the eyes_down
    timer from zero -- release_s grace, exercised here through the real module
    rather than the tracker in isolation."""
    module, frame = _make_module(tmp_path, clock, signal_release_s=0.4)
    fired = _feed(module, frame, clock, pitch_deg=30.0, seconds=2.0)  # 2.0s of 2.5s needed
    assert "eyes_down" not in fired

    # One frame with no face at all (well under release_s=0.4s duration)
    module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(None)
    module.process(frame)
    clock["now"] += 0.05

    fired += _feed(module, frame, clock, pitch_deg=30.0, seconds=1.0)  # resumes; total sustained > 2.5s
    assert "eyes_down" in fired, "a single dropout frame must not have restarted the sustain timer"


def test_debug_state_reports_gaze_and_duration_while_looking_down(tmp_path, clock):
    """get_debug_state()'s fields, exercised through a real scenario rather
    than asserted against in isolation."""
    module, frame = _make_module(tmp_path, clock)
    _feed(module, frame, clock, pitch_deg=30.0, seconds=3.0)
    state = module.get_debug_state()
    assert state["gaze"] == "Down"
    assert state["focused"] is False
    assert state["eyes_down_state"] == "CONFIRMED"
    assert state["eyes_down_duration_s"] >= 2.5
    assert state["alert_active"] is True
