"""Tests for core/tracking.py's BoxTracker -- the object-identity counterpart to
PersonTracker, for modules that reason about held/nearby items (phone, bottle, cup)
where no pose skeleton exists to supply a torso-length scale reference."""

import pytest

from core.tracking import BoxTracker, TrackedBox


def test_a_new_detection_becomes_a_new_track():
    tracker = BoxTracker()
    tracks = tracker.update([("phone", (100, 100, 140, 160), 0.9)], now=0.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].label == "phone"
    assert tracks[0].hits == 1


def test_a_nearby_same_class_box_next_frame_matches_the_same_track():
    tracker = BoxTracker()
    tracker.update([("phone", (100, 100, 140, 160), 0.9)], now=0.0)
    tracks = tracker.update([("phone", (104, 102, 144, 162), 0.9)], now=0.1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 2


def test_matching_is_class_scoped_not_just_distance():
    """A cup appearing exactly where a phone just was must NOT be absorbed
    into the phone's track -- class match is a stronger identity signal than
    centroid distance alone provides for arbitrary objects. The phone's own
    track is untouched (not in this frame's detections, but not retired
    either -- same "seen this frame vs still alive" contract PersonTracker
    already uses: `.tracks` is everything alive, update()'s return is only
    what matched or was created THIS frame)."""
    tracker = BoxTracker()
    tracker.update([("phone", (100, 100, 140, 160), 0.9)], now=0.0)
    tracks = tracker.update([("cup", (101, 101, 141, 161), 0.9)], now=0.1)
    assert len(tracks) == 1 and tracks[0].label == "cup"
    labels = {t.label: t.track_id for t in tracker.tracks}
    assert labels.keys() == {"phone", "cup"}
    assert labels["phone"] != labels["cup"]


def test_two_simultaneous_same_class_objects_get_distinct_ids():
    tracker = BoxTracker()
    tracks = tracker.update(
        [("cup", (0, 0, 40, 40), 0.8), ("cup", (400, 400, 440, 440), 0.8)], now=0.0
    )
    assert len({t.track_id for t in tracks}) == 2


def test_a_track_survives_brief_occlusion_within_the_grace_window():
    tracker = BoxTracker(lost_grace_s=1.0)
    tracker.update([("bottle", (100, 100, 140, 220), 0.9)], now=0.0)
    tracker.update([], now=0.5)  # briefly gone (e.g. occluded, or a stride miss)
    tracks = tracker.update([("bottle", (102, 101, 142, 221), 0.9)], now=0.9)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1  # same identity preserved through the gap
    assert tracks[0].hits == 2


def test_a_track_retires_after_the_grace_window_expires():
    tracker = BoxTracker(lost_grace_s=1.0)
    tracker.update([("bottle", (100, 100, 140, 220), 0.9)], now=0.0)
    tracker.update([], now=1.5)  # past lost_grace_s with no match
    tracks = tracker.update([("bottle", (100, 100, 140, 220), 0.9)], now=2.0)
    assert tracks[0].track_id == 2  # a NEW track, not a resurrected id-1


def test_velocity_reports_box_diagonals_per_second():
    tracker = BoxTracker()
    tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)
    tracker.update([("phone", (40, 0, 80, 40), 0.9)], now=0.5)  # moved one full width
    track = tracker.update([("phone", (80, 0, 120, 40), 0.9)], now=1.0)[0]
    # diagonal ~56.6px, moved 80px total over 1.0s -> a bit over 1 diagonal/s
    assert track.velocity() > 1.0


def test_velocity_is_none_with_only_one_sample():
    tracker = BoxTracker()
    track = tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)[0]
    assert track.velocity() is None


def test_peak_speed_sees_oscillation_that_net_displacement_cannot():
    """The defect peak_speed exists for: a box snapping back and forth ends up
    where it started, so velocity() (net displacement / elapsed) reports it as
    barely moving no matter how violent the motion was. Both readings are
    correct answers to different questions -- this pins that they differ."""
    tracker = BoxTracker(max_distance=5.0)
    t = 0.0
    for step in range(9):  # odd count -> ends one step from where it started
        x = 100 + 60 * (1 if step % 2 else -1)
        track = tracker.update([("hand", (x, 80, x + 40, 120), 0.9)], now=t)[0]
        t += 0.1
    # 120px of travel every 0.1s against a ~56.6px diagonal -> >20 diagonals/s.
    assert track.peak_speed() > 20.0
    # ...while the net displacement over the whole 0.8s trail is one step.
    assert track.velocity() < 3.0


def test_peak_speed_matches_velocity_for_steady_one_direction_travel():
    """No oscillation to hide: the fastest single step and the whole-trail
    average agree, so peak_speed isn't systematically inflating anything."""
    tracker = BoxTracker()
    track = _move_across(tracker, "phone", [0, 20, 40, 60, 80], dt=0.2)
    assert track.peak_speed() == pytest.approx(track.velocity(), rel=0.02)


def test_peak_speed_is_none_with_only_one_sample():
    tracker = BoxTracker()
    track = tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)[0]
    assert track.peak_speed() is None


def test_peak_speed_is_none_when_no_time_passed_between_samples():
    """Two detections stamped with the same instant carry no speed information;
    reporting 0.0 would be a claim the trail cannot support."""
    tracker = BoxTracker()
    tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)
    track = tracker.update([("phone", (40, 0, 80, 40), 0.9)], now=0.0)[0]
    assert track.peak_speed() is None


def _move_across(tracker, label, xs, y=80, size=40, dt=0.2, conf=0.9):
    """Feed a sequence of small, matchable steps (each within BoxTracker's
    default match gate) rather than one large jump, so the track's identity
    is preserved across the whole path -- the same way real frame-to-frame
    motion would arrive."""
    track = None
    t = 0.0
    for x in xs:
        track = tracker.update([(label, (x, y, x + size, y + size), conf)], now=t)[0]
        t += dt
    return track


def test_approaching_detects_motion_toward_a_point_not_just_proximity():
    tracker = BoxTracker()
    point = (500.0, 100.0)
    track = _move_across(tracker, "phone", [0, 20, 40, 60, 80])
    assert track.approaching(point) is True


def test_approaching_is_false_for_motion_heading_away():
    tracker = BoxTracker()
    point = (0.0, 100.0)
    track = _move_across(tracker, "phone", [200, 220, 240, 260, 280])
    assert track.approaching(point) is False


def test_approaching_is_none_for_a_stationary_track():
    tracker = BoxTracker()
    tracker.update([("phone", (0, 80, 40, 120), 0.9)], now=0.0)
    track = tracker.update([("phone", (0, 80, 40, 120), 0.9)], now=1.0)[0]
    assert track.approaching((500.0, 100.0)) is None


def test_stability_rises_with_hits_and_age_but_starts_below_one():
    """update() mutates and returns the SAME TrackedBox instance held
    internally (matching PersonTracker's own pattern), so the fresh
    stability must be snapshotted as a plain float before further updates
    mutate that object in place."""
    tracker = BoxTracker()
    fresh = tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)[0]
    fresh_stability = fresh.stability
    assert 0.0 < fresh_stability < 1.0
    established = fresh
    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        established = tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=t)[0]
    assert established.stability > fresh_stability


def test_as_dict_is_json_shaped():
    tracker = BoxTracker()
    track = tracker.update([("cup", (0, 0, 10, 10), 0.5)], now=0.0)[0]
    d = track.as_dict()
    assert d["label"] == "cup" and d["track_id"] == 1 and "stability" in d


def _oscillate(tracker, label, count, amplitude=20, y=80, size=40, dt=0.1, conf=0.9):
    """A box reversing direction every step -- the generic-track counterpart
    to core/activity.py's wave/flail oscillation helpers."""
    track = None
    t = 0.0
    for step in range(count):
        x = 100 + amplitude * (1 if step % 2 else -1)
        track = tracker.update([(label, (x, y, x + size, y + size), conf)], now=t)[0]
        t += dt
    return track


def test_reversals_counts_a_chaotic_back_and_forth_track():
    tracker = BoxTracker()
    track = _oscillate(tracker, "hand", count=10, amplitude=25)
    assert track.reversals(min_amplitude=0.1) >= 4


def test_reversals_is_zero_for_a_single_smooth_reach():
    tracker = BoxTracker()
    track = _move_across(tracker, "hand", [0, 20, 40, 60, 80, 100])
    assert track.reversals(min_amplitude=0.1) == 0


def test_reversals_ignores_jitter_sized_direction_flips():
    tracker = BoxTracker()
    track = _oscillate(tracker, "hand", count=10, amplitude=0.5)  # near-zero amplitude
    assert track.reversals(min_amplitude=0.1) == 0


def test_reversals_is_zero_with_fewer_than_three_samples():
    tracker = BoxTracker()
    track = tracker.update([("hand", (0, 0, 40, 40), 0.9)], now=0.0)[0]
    assert track.reversals(min_amplitude=0.1) == 0


def test_reset_clears_tracks_and_id_counter():
    tracker = BoxTracker()
    tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=0.0)
    tracker.reset()
    tracks = tracker.update([("phone", (0, 0, 40, 40), 0.9)], now=1.0)
    assert tracks[0].track_id == 1  # counter restarted, not continuing from 2
