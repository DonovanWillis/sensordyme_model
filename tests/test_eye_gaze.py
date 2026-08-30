"""Tests for eye-in-head gaze tracking, independent of head pose.

Covers three things end to end, all against the REAL module code (synthetic
but geometrically real landmarks, same rigor as test_driver_attention_scenarios.py
and test_head_pose_geometry.py -- not hand-picked numbers):

  1. core/eye_signal.py's eye_gaze_offset_deg sign convention, cross-checked
     against estimate_head_pose_deg's own already-verified sign (a small head
     pitch and an equivalent-direction iris shift must push the SAME way).
  2. The reported bug this exists to fix: head forward + eyes cast down (no
     head movement at all) now fires "eyes_down" on the pi profile, where
     before it structurally could not (eyes_down was pitch-only).
  3. The "significant head turn -> FaceMesh loses lock -> reverts to the far
     more patient face_lost path" bug: distraction now keeps accumulating
     through a landmark-loss gap when a body still corroborates presence and
     the last known gaze was already past threshold.
"""

import pytest
import test_driver_attention_scenarios as das

from core.eye_signal import eye_gaze_offset_deg
from modules.driver_attention import estimate_head_pose_deg

W, H = das.W, das.H

# Re-exported fixtures/helpers from the existing, geometrically-rigorous harness.
clock = das.clock
_make_module = das._make_module
_feed = das._feed
_FakeFaceMeshResult = das._FakeFaceMeshResult
make_fake_landmarks = das.make_fake_landmarks


def _shift_iris(landmarks, dy_px: float = 0.0, dx_px: float = 0.0):
    """Move both eyes' iris landmarks by (dx_px, dy_px), leaving every other
    landmark (including the eyelid/EAR shape) untouched -- isolates "the
    eyeball rotated within a stationary socket" from "the head moved"."""
    out = list(landmarks)
    for idx in (468, 469, 470, 471, 472, 473, 474, 475, 476, 477):
        out[idx] = das._FakeLandmark(out[idx].x + dx_px / W, out[idx].y + dy_px / H)
    return out


# ------------------------------------------------------------------ sign convention
def test_eye_gaze_offset_is_zero_for_a_centered_iris():
    points = [(lm.x * W, lm.y * H) for lm in make_fake_landmarks()]
    yaw, pitch = eye_gaze_offset_deg(points)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)


def test_iris_toward_the_lower_lid_matches_head_pitchs_own_down_is_negative_convention():
    """Cross-check against estimate_head_pose_deg's independently-verified sign
    (test_head_pose_geometry.py): a small physical head-pitch-down rotation
    must produce a NEGATIVE delta, and an iris shifted toward the lower lid
    (the same physical direction, but from eye rotation, not head rotation)
    must produce a pitch component that pushes the SAME way -- otherwise
    fusing the two would partially cancel a real "head level, eyes down"
    instead of reinforcing it."""
    pose_neutral = estimate_head_pose_deg(das._pose_pixel_points(0.0, 0.0), (H, W))
    pose_down = estimate_head_pose_deg(das._pose_pixel_points(10.0, 0.0), (H, W))
    from modules.driver_attention import angle_delta_deg
    head_delta_pitch = angle_delta_deg(pose_down[1], pose_neutral[1])
    assert head_delta_pitch < 0, "test setup: pitch_deg=10 should already be a verified 'looking down' rotation"

    points = [(lm.x * W, lm.y * H) for lm in _shift_iris(make_fake_landmarks(), dy_px=8.0)]
    _, eye_pitch = eye_gaze_offset_deg(points)
    assert eye_pitch < 0, (
        "iris shifted toward the lower lid must produce a NEGATIVE pitch component, "
        "matching estimate_head_pose_deg's own 'down = negative' convention"
    )


# ------------------------------------------------------------------ the reported bug
def test_head_forward_eyes_down_fires_eyes_down(tmp_path, clock):
    """The exact reported scenario: face/head facing forward, hands on the
    wheel, eyes looking down. Before gaze fusion this was structurally
    invisible (eyes_down read head pitch only)."""
    module, frame = _make_module(tmp_path, clock)

    landmarks = _shift_iris(make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0), dy_px=8.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    fired = []
    for _ in range(80):  # 4s at 0.05s/step, comfortably past eyes_down_seconds=2.5
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "eyes_down" in fired
    debug = module.get_debug_state()
    assert debug["gaze"] == "Down"


def test_head_forward_eyes_sideways_fires_distraction(tmp_path, clock):
    """The horizontal counterpart to the eyes-down scenario: head/face facing
    forward, eyes deviated sideways within the socket -- must fire
    'distraction' via the same fused-gaze yaw, not require the head to turn."""
    module, frame = _make_module(tmp_path, clock)

    landmarks = _shift_iris(make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0), dx_px=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    fired = []
    for _ in range(80):  # 4s, comfortably past distraction_seconds=3.0
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "distraction" in fired
    debug = module.get_debug_state()
    assert debug["gaze"] in ("Left", "Right")


def test_head_forward_eyes_sideways_does_not_fire_with_gaze_disabled(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock, enable_gaze=False)

    landmarks = _shift_iris(make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0), dx_px=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    fired = []
    for _ in range(80):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "distraction" not in fired


def test_head_forward_eyes_down_does_not_fire_with_gaze_disabled(tmp_path, clock):
    """enable_gaze: false must restore the OLD (head-pose-only) behavior
    exactly -- the same eyes-cast-down-only motion must NOT fire eyes_down."""
    module, frame = _make_module(tmp_path, clock, enable_gaze=False)

    landmarks = _shift_iris(make_fake_landmarks(pitch_deg=0.0, yaw_deg=0.0), dy_px=8.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    fired = []
    for _ in range(80):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "eyes_down" not in fired


def test_eyes_forward_head_forward_stays_focused_with_gaze_enabled(tmp_path, clock):
    """Gaze fusion must not manufacture false positives on ordinary forward
    driving -- a centered iris contributes ~0 regardless."""
    module, frame = _make_module(tmp_path, clock)
    fired = _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=0.0, seconds=4.0)
    assert "eyes_down" not in fired
    assert "eyes_up" not in fired
    assert "distraction" not in fired
    assert module.is_alert_active() is False


# ------------------------------------------------------------------ coasting through landmark loss
def test_distraction_keeps_accumulating_through_an_ordinary_turn_that_loses_face_lock(tmp_path, clock):
    """The other reported bug: turning significantly to the side produced no
    alert. Simulates the real MediaPipe failure mode directly (FaceMesh
    returns no landmarks partway through the turn) rather than assuming the
    synthetic geometry generator would reproduce it on its own -- and asserts
    "distraction" fires well before face_lost_with_body_seconds=8.0 would
    have, because the last known gaze (already past yaw_threshold_deg) is
    carried forward while a body still corroborates presence. yaw=45 is
    ordinary (uncorroborated -- below mirror_check_yaw_deg=55), so this
    demonstrates the fix against distraction_seconds=3.0, not the wide-turn
    variant's more patient 5.0s -- see the mirror-check-shaped test below."""
    module, frame = _make_module(tmp_path, clock)
    module._last_person_visible = True  # object detector still sees the driver's torso

    fired = _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=45.0, seconds=1.5)
    assert "distraction" not in fired  # not sustained long enough yet (1.5s of 3.0s)

    # FaceMesh loses lock entirely (the real failure mode at a wide angle) --
    # module._last_person_visible stays True (set by the object-detector stride,
    # untouched by the face-mesh path) for the remainder of this test.
    module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(None)
    for _ in range(int(2.0 / 0.05)):  # 1.5s + 2.0s = 3.5s total, past distraction_seconds=3.0
        fired += [e.type for e in module.process(frame)]  # but well under face_lost_with_body_seconds=8.0
        clock["now"] += 0.05

    assert "distraction" in fired, (
        "a turn that breaks landmark tracking must still accumulate toward "
        "distraction using the last known gaze -- not silently reset and wait "
        "for the much more patient face_lost/attention_lost path"
    )
    assert "attention_lost" not in fired, "distraction should fire well before face_lost's own threshold"


def test_mirror_check_shaped_turn_that_loses_face_lock_still_beats_face_lost(tmp_path, clock):
    """A wide (>mirror_check_yaw_deg), fast turn -- exactly what "turned my
    face significantly to the side" describes -- is corroborated as
    mirror-check-shaped from the first frame and so needs the longer 5.0s
    (rearview_tolerance_seconds), not 3.0s. Coasting through the landmark-loss
    gap still gets it there well before face_lost_with_body_seconds=8.0."""
    module, frame = _make_module(tmp_path, clock)
    module._last_person_visible = True

    fired = _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=60.0, seconds=1.0)
    assert "distraction" not in fired

    module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(None)
    for _ in range(int(4.5 / 0.05)):  # 1.0s + 4.5s = 5.5s total, past the corroborated 5.0s
        fired += [e.type for e in module.process(frame)]  # but well under face_lost_with_body_seconds=8.0
        clock["now"] += 0.05

    assert "distraction" in fired
    assert "attention_lost" not in fired


def test_coasting_does_not_claim_distraction_when_the_last_known_gaze_was_forward(tmp_path, clock):
    """Losing the face while the driver was looking straight ahead a moment
    ago must NOT be treated as distraction -- only an already-observed
    off-center condition is carried forward, never invented."""
    module, frame = _make_module(tmp_path, clock)
    module._last_person_visible = True

    _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=0.0, seconds=0.5)  # neutral, then...
    module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(None)  # ...camera suddenly can't find the face
    fired = []
    for _ in range(int(3.0 / 0.05)):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "distraction" not in fired
    assert "eyes_down" not in fired
    assert "eyes_up" not in fired


def test_coasting_does_not_apply_without_a_corroborating_body(tmp_path, clock):
    """If the object detector ALSO can't find a person, coasting must not
    apply -- the driver may genuinely be gone, not merely turned away, and
    the module should fall back to the (uncorroborated, faster) face_lost
    path rather than asserting a distraction it can't support."""
    module, frame = _make_module(tmp_path, clock)
    module._last_person_visible = False

    _feed(module, frame, clock, pitch_deg=0.0, yaw_deg=60.0, seconds=1.0)
    module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(None)
    fired = []
    for _ in range(int(3.0 / 0.05)):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.05

    assert "distraction" not in fired
    assert "attention_lost" in fired, "uncorroborated face loss should still reach its own (short) threshold"
