"""Ground-truth check for estimate_head_pose_deg's yaw/pitch sign convention.

This is the regression test for the actual root cause behind driver_attention
never alerting on a driver visibly looking down at a phone: solvePnP's raw
Rodrigues decomposition returned a pitch that was the OPPOSITE sign of what
the rest of the module assumed ("negative pitch = looking down", stated in
this function's own docstring and relied on by eyes_down_condition). Every
existing test in this codebase drove the module with pre-computed yaw/pitch
numbers or blank frames -- none of them ever checked that a KNOWN physical
head rotation produces the pitch/yaw sign the module's threshold comparisons
actually assume, so the inversion shipped invisibly.

The ground truth here is independent of solvePnP: for a given synthetic 3D
rotation applied to the module's own face model and projected through the
module's own camera intrinsics, "looking down" is defined directly as the
face's local +Z axis (the direction the nose points, i.e. where the face/eyes
are oriented -- see _MODEL_POINTS_3D's comment) having a positive Y component
in camera space after that SAME rotation is applied (camera Y grows downward
in the image, so a gaze vector with positive Y points toward the floor). That
definition never touches estimate_head_pose_deg -- only cv2.projectPoints and
plain vector rotation -- so it cannot be circular.
"""

import cv2
import numpy as np

from modules.driver_attention import (
    _MODEL_POINTS_3D,
    angle_delta_deg,
    estimate_head_pose_deg,
)

IMAGE_SIZE = (480, 640)  # (h, w)
_H, _W = IMAGE_SIZE
_FOCAL = _W
_CENTER = (_W / 2.0, _H / 2.0)
_K = np.array([[_FOCAL, 0, _CENTER[0]], [0, _FOCAL, _CENTER[1]], [0, 0, 1]], dtype=np.float64)
_DIST = np.zeros((4, 1))
_TVEC = np.array([0.0, 0.0, 600.0])  # 600mm in front of the camera -- a plausible driver distance

_NOSE, _CHIN, _LEYE, _REYE, _LMOUTH, _RMOUTH = range(6)
_GAZE_LOCAL = np.array([0.0, 0.0, 1.0])  # local +Z: the direction the nose/face points


def _project(rotation_matrix: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    pts2d, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, _TVEC, _K, _DIST)
    return pts2d.reshape(-1, 2)


def _rot_x(deg: float) -> np.ndarray:
    """Rotation about the camera's X (left-right) axis -- pitch."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    """Rotation about the camera's Y (up-down) axis -- yaw."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# "Upright, facing the camera" extrinsic: _MODEL_POINTS_3D authors +Y as
# anatomically up (eyes y=+32.7 > nose y=0 > mouth y=-28.9 > chin y=-63.6) and
# +Z as the direction the nose protrudes (nose z=0 is the max among the 6
# points). OpenCV camera space is X-right, Y-DOWN, Z-forward-into-the-scene.
# rot_x(180) maps local (x,y,z) -> (x,-y,-z): flips Z so the nose becomes the
# CLOSEST point to the camera (nose protrudes toward the viewer) and flips Y
# so anatomical "up" becomes image "up" (smaller pixel y). Both are verified
# below, not assumed.
_R_NEUTRAL = _rot_x(180)


def test_the_neutral_extrinsic_actually_represents_an_upright_face_facing_the_camera():
    """Sanity check on the test's own ground truth, before trusting anything built on it."""
    cam_points = (_R_NEUTRAL @ _MODEL_POINTS_3D.T).T + _TVEC
    assert cam_points[_NOSE][2] < cam_points[_CHIN][2], "nose should be closer to the camera than the chin"
    assert cam_points[_LEYE][1] < cam_points[_CHIN][1], "eyes should render above the chin (smaller pixel y)"


def test_pitch_is_negative_when_the_head_is_genuinely_rotated_to_look_down():
    """The literal bug. Ground truth (gaze vector's camera-Y > 0 == looking down)
    is computed independently of estimate_head_pose_deg, by rotating the SAME
    gaze vector with the SAME rotation used to generate the synthetic image
    points -- not by asking the function under test what it thinks happened."""
    pose_neutral = estimate_head_pose_deg(_project(_R_NEUTRAL), IMAGE_SIZE)
    assert pose_neutral is not None

    for delta_deg in (10, 20, 30):
        rotation = _rot_x(delta_deg) @ _R_NEUTRAL
        gaze_camera_space = rotation @ _GAZE_LOCAL
        assert gaze_camera_space[1] > 0, "test setup error: this rotation should geometrically be looking down"

        pose = estimate_head_pose_deg(_project(rotation), IMAGE_SIZE)
        assert pose is not None
        delta_pitch = angle_delta_deg(pose[1], pose_neutral[1])
        assert delta_pitch < 0, (
            f"rotating {delta_deg} deg toward a geometrically-verified downward gaze produced "
            f"delta_pitch={delta_pitch:+.1f} -- eyes_down_condition (pitch < threshold) can only "
            f"ever fire on the OPPOSITE motion if this is positive"
        )


def test_pitch_is_positive_when_the_head_is_genuinely_rotated_to_look_up():
    pose_neutral = estimate_head_pose_deg(_project(_R_NEUTRAL), IMAGE_SIZE)
    for delta_deg in (10, 20, 30):
        rotation = _rot_x(-delta_deg) @ _R_NEUTRAL
        gaze_camera_space = rotation @ _GAZE_LOCAL
        assert gaze_camera_space[1] < 0, "test setup error: this rotation should geometrically be looking up"

        pose = estimate_head_pose_deg(_project(rotation), IMAGE_SIZE)
        delta_pitch = angle_delta_deg(pose[1], pose_neutral[1])
        assert delta_pitch > 0, f"looking up produced delta_pitch={delta_pitch:+.1f}, expected positive"


def test_yaw_is_positive_when_the_head_is_genuinely_turned_toward_the_frames_right():
    """estimate_head_pose_deg's own docstring: "Positive yaw = turned toward
    the frame's right." Not load-bearing for today's distraction threshold
    (abs(yaw) > threshold is sign-agnostic), but get_debug_state()'s gaze
    label and any future asymmetric zone logic depend on this being true."""
    pose_neutral = estimate_head_pose_deg(_project(_R_NEUTRAL), IMAGE_SIZE)
    for delta_deg in (10, 20, 30):
        rotation = _rot_y(delta_deg) @ _R_NEUTRAL
        gaze_camera_space = rotation @ _GAZE_LOCAL
        turned_right = gaze_camera_space[0] > 0
        pose = estimate_head_pose_deg(_project(rotation), IMAGE_SIZE)
        delta_yaw = angle_delta_deg(pose[0], pose_neutral[0])
        if turned_right:
            assert delta_yaw > 0, f"turned toward frame-right produced delta_yaw={delta_yaw:+.1f}"
        else:
            assert delta_yaw < 0, f"turned toward frame-left produced delta_yaw={delta_yaw:+.1f}"


def test_pitch_sign_is_consistent_across_a_realistic_combined_yaw_and_pitch_pose():
    """A driver glancing down-and-slightly-sideways at a phone in their lap --
    not pure pitch in isolation -- must still read as looking down."""
    pose_neutral = estimate_head_pose_deg(_project(_R_NEUTRAL), IMAGE_SIZE)
    rotation = _rot_x(25) @ _rot_y(15) @ _R_NEUTRAL
    gaze_camera_space = rotation @ _GAZE_LOCAL
    assert gaze_camera_space[1] > 0, "test setup error: combined pose should geometrically be looking down"

    pose = estimate_head_pose_deg(_project(rotation), IMAGE_SIZE)
    delta_pitch = angle_delta_deg(pose[1], pose_neutral[1])
    assert delta_pitch < 0, f"combined down+sideways pose produced delta_pitch={delta_pitch:+.1f}"
