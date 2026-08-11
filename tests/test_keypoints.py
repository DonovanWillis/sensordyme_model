"""Body keypoints, geometry, occlusion, and multi-person tracking.

The property that matters most here is that missing data stays missing. A safety
system computing a knee angle from a knee it cannot see produces confident
nonsense, and confident nonsense is what a person acts on.
"""

import pytest

from core.keypoints import (
    MEDIAPIPE_POSE_MAP,
    SKELETON,
    Keypoint,
    Part,
    Skeleton,
    from_coco_keypoints,
    from_detection_box,
    from_mediapipe,
)
from core.tracking import PersonTracker


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def upright(x=0.5, visibility=1.0) -> Skeleton:
    """A standing person, torso length 0.25."""
    return Skeleton(keypoints={
        Part.NOSE: Keypoint(Part.NOSE, x, 0.15, visibility),
        Part.LEFT_SHOULDER: Keypoint(Part.LEFT_SHOULDER, x - 0.06, 0.30, visibility),
        Part.RIGHT_SHOULDER: Keypoint(Part.RIGHT_SHOULDER, x + 0.06, 0.30, visibility),
        Part.LEFT_ELBOW: Keypoint(Part.LEFT_ELBOW, x - 0.10, 0.45, visibility),
        Part.RIGHT_ELBOW: Keypoint(Part.RIGHT_ELBOW, x + 0.10, 0.45, visibility),
        Part.LEFT_WRIST: Keypoint(Part.LEFT_WRIST, x - 0.10, 0.60, visibility),
        Part.RIGHT_WRIST: Keypoint(Part.RIGHT_WRIST, x + 0.10, 0.60, visibility),
        Part.LEFT_HIP: Keypoint(Part.LEFT_HIP, x - 0.05, 0.55, visibility),
        Part.RIGHT_HIP: Keypoint(Part.RIGHT_HIP, x + 0.05, 0.55, visibility),
        Part.LEFT_KNEE: Keypoint(Part.LEFT_KNEE, x - 0.05, 0.75, visibility),
        Part.RIGHT_KNEE: Keypoint(Part.RIGHT_KNEE, x + 0.05, 0.75, visibility),
        Part.LEFT_ANKLE: Keypoint(Part.LEFT_ANKLE, x - 0.05, 0.95, visibility),
        Part.RIGHT_ANKLE: Keypoint(Part.RIGHT_ANKLE, x + 0.05, 0.95, visibility),
    })


# --------------------------------------------------------------------- adapters
def test_mediapipe_landmarks_become_named_parts():
    landmarks = [Landmark(0.5, 0.5) for _ in range(33)]
    landmarks[11] = Landmark(0.4, 0.3, 0.95)
    skeleton = from_mediapipe(landmarks)
    shoulder = skeleton.get(Part.LEFT_SHOULDER)
    assert shoulder is not None
    assert shoulder.x == pytest.approx(0.4)
    assert shoulder.confidence == pytest.approx(0.95)


def test_the_mediapipe_map_is_the_only_place_indices_appear():
    """Downstream code names parts; index knowledge lives here alone, so a
    Hailo pose model is an adapter rather than a refactor."""
    assert MEDIAPIPE_POSE_MAP[23] is Part.LEFT_HIP
    assert MEDIAPIPE_POSE_MAP[28] is Part.RIGHT_ANKLE


def test_coco_keypoints_adapt_for_the_hailo_path():
    """Hailo pose models emit COCO-17; this is the drop-in for that day."""
    points = [(0.5, 0.1, 0.9)] + [(0.5, 0.2, 0.9)] * 16
    skeleton = from_coco_keypoints(points)
    assert skeleton.get(Part.NOSE) is not None
    assert skeleton.get(Part.RIGHT_ANKLE) is not None


def test_a_short_landmark_list_does_not_raise():
    assert from_mediapipe([Landmark(0.5, 0.5)]).get(Part.LEFT_HIP) is None


def test_detection_box_yields_a_trackable_torso_and_nothing_else():
    """Only the two measurements PersonTracker's matcher reads (center_of_mass,
    torso_length) come from a box; every part with no box-derived counterpart
    (a knee, an elbow) must stay absent rather than implied."""
    skeleton = from_detection_box((700, 300, 900, 800))  # x1,y1,x2,y2
    assert skeleton.center_of_mass() is not None
    assert skeleton.torso_length() is not None
    assert skeleton.get(Part.LEFT_KNEE) is None
    assert skeleton.left_knee_angle() is None


def test_detection_box_degenerate_size_does_not_raise():
    assert from_detection_box((10, 10, 10, 10)).torso_length() is None


def test_two_box_only_people_still_get_separate_stable_ids():
    """The exact reuse modules/occupancy.py relies on: PersonTracker matching
    coarse box-derived skeletons the same way it matches real pose skeletons."""
    tracker = PersonTracker()
    far_apart = [(50, 100, 150, 500), (700, 100, 800, 500)]
    first = tracker.update([from_detection_box(b) for b in far_apart], now=100.0)
    assert len({t.track_id for t in first}) == 2

    second = tracker.update([from_detection_box(b) for b in far_apart], now=100.1)
    assert [t.track_id for t in first] == [t.track_id for t in second]


# --------------------------------------------------------------------- occlusion
def test_a_low_confidence_keypoint_is_absent_not_zero():
    """MediaPipe reports a position for occluded joints. Using those silently is
    how an arm behind a back becomes a confident elbow angle."""
    skeleton = upright(visibility=0.2)
    assert skeleton.get(Part.LEFT_ELBOW) is None
    assert skeleton.left_elbow_angle() is None


def test_geometry_omits_what_it_cannot_measure():
    skeleton = upright()
    del skeleton.keypoints[Part.LEFT_KNEE]
    geometry = skeleton.geometry()
    assert "left_knee_deg" not in geometry
    assert "right_knee_deg" in geometry


def test_coverage_reports_how_occluded_a_person_is():
    full = upright()
    assert full.coverage == 1.0
    half = upright()
    for part in (Part.LEFT_ANKLE, Part.RIGHT_ANKLE, Part.LEFT_KNEE, Part.RIGHT_KNEE):
        half.keypoints[part] = Keypoint(part, 0.5, 0.9, 0.1)
    assert half.coverage < 1.0


def test_midpoint_falls_back_to_one_visible_side():
    """Someone lying on their side occludes half their body — exactly when
    posture matters most."""
    skeleton = upright()
    skeleton.keypoints[Part.RIGHT_HIP] = Keypoint(Part.RIGHT_HIP, 0.55, 0.55, 0.1)
    assert skeleton.hips() is not None


def test_nothing_visible_yields_no_geometry():
    empty = Skeleton()
    assert empty.torso_length() is None
    assert empty.center_of_mass() is None
    assert empty.geometry() == {"coverage": 0.0}


# --------------------------------------------------------------------- geometry
def test_upright_torso_reads_near_zero_degrees():
    assert upright().torso_angle_from_vertical() == pytest.approx(0.0, abs=2.0)


def test_horizontal_torso_reads_near_ninety():
    lying = upright()
    lying.keypoints[Part.LEFT_SHOULDER] = Keypoint(Part.LEFT_SHOULDER, 0.2, 0.80)
    lying.keypoints[Part.RIGHT_SHOULDER] = Keypoint(Part.RIGHT_SHOULDER, 0.2, 0.86)
    lying.keypoints[Part.LEFT_HIP] = Keypoint(Part.LEFT_HIP, 0.6, 0.80)
    lying.keypoints[Part.RIGHT_HIP] = Keypoint(Part.RIGHT_HIP, 0.6, 0.86)
    assert lying.torso_angle_from_vertical() > 70


def test_a_straight_leg_is_about_180_degrees():
    angle = upright().left_knee_angle()
    assert angle is not None and angle > 160


def test_a_bent_knee_is_detected():
    crouch = upright()
    crouch.keypoints[Part.LEFT_KNEE] = Keypoint(Part.LEFT_KNEE, 0.30, 0.75)
    angle = crouch.left_knee_angle()
    assert angle is not None and angle < 150


def test_torso_length_is_the_scale_reference():
    near = upright().torso_length()
    far = Skeleton(keypoints={
        Part.LEFT_SHOULDER: Keypoint(Part.LEFT_SHOULDER, 0.50, 0.40),
        Part.RIGHT_SHOULDER: Keypoint(Part.RIGHT_SHOULDER, 0.53, 0.40),
        Part.LEFT_HIP: Keypoint(Part.LEFT_HIP, 0.50, 0.52),
        Part.RIGHT_HIP: Keypoint(Part.RIGHT_HIP, 0.53, 0.52),
    }).torso_length()
    assert near is not None and far is not None
    assert far < near, "a distant person has a shorter torso in frame"


def test_center_of_mass_sits_between_shoulders_and_hips():
    com = upright().center_of_mass()
    assert com is not None
    assert 0.30 < com[1] < 0.55


def test_skeleton_bones_reference_only_known_parts():
    for a, b in SKELETON:
        assert isinstance(a, Part) and isinstance(b, Part)


def test_skeleton_serialises_for_the_debug_view():
    payload = upright().as_dict()
    assert "left_hip" in payload["keypoints"]
    assert "torso_angle_deg" in payload["geometry"]


# --------------------------------------------------------------------- tracking
def test_one_person_keeps_one_id_across_frames():
    tracker = PersonTracker()
    first = tracker.update([upright(x=0.50)], now=100.0)
    second = tracker.update([upright(x=0.52)], now=100.1)
    assert first[0].track_id == second[0].track_id
    assert second[0].hits == 2


def test_two_people_get_separate_ids():
    tracker = PersonTracker()
    tracked = tracker.update([upright(x=0.25), upright(x=0.75)], now=100.0)
    assert len({t.track_id for t in tracked}) == 2


def test_two_people_keep_their_ids_across_frames():
    tracker = PersonTracker()
    first = tracker.update([upright(x=0.25), upright(x=0.75)], now=100.0)
    second = tracker.update([upright(x=0.27), upright(x=0.73)], now=100.1)
    assert [t.track_id for t in first] == [t.track_id for t in second]


def test_a_person_who_jumps_across_the_frame_is_treated_as_new():
    """Beyond the gate it is not the same person — better a new id than a track
    that teleports and corrupts every timer attached to it."""
    tracker = PersonTracker()
    first = tracker.update([upright(x=0.1)], now=100.0)
    second = tracker.update([upright(x=0.9)], now=100.1)
    assert first[0].track_id != second[0].track_id


def test_a_briefly_hidden_person_keeps_their_id():
    """Walking behind a sofa is not leaving; retiring the track would restart
    every timer that matters."""
    tracker = PersonTracker()
    first = tracker.update([upright()], now=100.0)
    tracker.update([], now=100.5)          # occluded
    resumed = tracker.update([upright()], now=101.0)
    assert resumed[0].track_id == first[0].track_id


def test_a_person_gone_long_enough_is_retired():
    tracker = PersonTracker(lost_grace_s=1.0)
    first = tracker.update([upright()], now=100.0)
    tracker.update([], now=105.0)
    later = tracker.update([upright()], now=105.1)
    assert later[0].track_id != first[0].track_id


def test_a_skeleton_without_a_centroid_is_not_tracked():
    """Nothing to match on; inventing a position would produce a phantom."""
    assert PersonTracker().update([Skeleton()], now=100.0) == []


def test_velocity_is_none_until_there_is_history():
    tracker = PersonTracker()
    tracked = tracker.update([upright(x=0.5)], now=100.0)
    assert tracked[0].velocity() is None


def test_velocity_is_measured_in_torso_lengths():
    tracker = PersonTracker()
    tracker.update([upright(x=0.50)], now=100.0)
    tracked = tracker.update([upright(x=0.60)], now=100.5)
    speed = tracked[0].velocity()
    assert speed is not None and speed > 0


def test_reset_clears_every_track():
    tracker = PersonTracker()
    tracker.update([upright()], now=100.0)
    tracker.reset()
    assert tracker.tracks == []
