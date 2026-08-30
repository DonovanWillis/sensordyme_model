"""Tests for Driver Mode's erratic/frantic hand-movement detection (hands
leaving the wheel or moving chaotically) -- the trajectory-based counterpart
to Baby Mode's flailing detector, built on the same core.tracking.BoxTracker
kinematics (peak speed + direction-reversal count) already unit-tested in
test_tracking.py. These focus on the debounce/event-wiring layer in
modules/driver_attention.py's process(), using the same real-module,
synthetic-but-geometrically-grounded harness as test_driver_attention_scenarios.py.

EVERY threshold here is the SHIPPED one (config.yaml's erratic_hand_min_speed
2.0, erratic_hand_min_reversals 3, erratic_hands_seconds 1.0 -- which are also
the module's own defaults, so passing no override IS the shipped configuration).
That is deliberate and load-bearing: this suite previously overrode
erratic_hand_min_speed to 0.5, four times below the shipped 2.0, and passed
green while the feature could not fire at all on the device -- the speed gate
was fed net displacement, which oscillation cancels to nearly zero by
definition, so "fast" and "reversing" were near-mutually-exclusive. A test that
invents its own threshold cannot see that. The hand geometry is likewise sized
to a real hand in a 640x480 dash-camera frame (see _hand_box_points) rather
than to whatever happens to squeeze inside the tracker's match gate, because
the match gate being too tight for a fast hand was the second half of the same
defect.
"""

import test_driver_attention_scenarios as das

import modules.driver_attention as driver_attention_module

W, H = das.W, das.H
clock = das.clock
_make_module = das._make_module
_FakeLandmark = das._FakeLandmark

#: A hand's size as a fraction of a 640x480 driver-facing frame: ~90x110 px,
#: which is what a hand on the wheel actually subtends at dash-camera framing.
#: Its box diagonal (~142 px) is the unit every speed and distance below is
#: measured in, so getting it right is what makes "2.0 box-diagonals/second"
#: mean a real hand speed rather than an arbitrary number.
_HAND_W, _HAND_H = 0.14, 0.23

#: Stride spacing: yolo_every_n_frames=3 at 30fps is one hand sample per 0.1s.
_STRIDE_S = 0.1


class _FakeYoloModel:
    """Just enough of the ultralytics surface for _ensure_object_detectors_loaded
    to succeed without a real (network-fetched) model -- same pattern
    test_occupancy_zones.py's _FakeModel uses."""

    names = {0: "cell phone"}

    def predict(self, frame, **kwargs):
        return []


class _FakeHandLandmarks:
    def __init__(self, points):
        self.landmark = [_FakeLandmark(x, y) for x, y in points]


class _FakeHandsResult:
    def __init__(self, hands):
        self.multi_hand_landmarks = hands or None


def _hand_box_points(cx, cy):
    """21 MediaPipe-Hands-shaped points spanning a REAL-sized hand centred on
    (cx, cy), in normalized [0,1] coords. The module turns these into the
    min/max bounding box it tracks, so only the extremes matter -- but the
    extremes have to be a hand's extremes, since every threshold downstream is
    expressed in that box's own diagonal."""
    return [
        (cx + (i % 3 - 1) * _HAND_W / 2.0, cy + (i % 5 - 2) * _HAND_H / 4.0)
        for i in range(21)
    ]


def _hand_module(tmp_path, clock, monkeypatch, **overrides):
    monkeypatch.setattr(driver_attention_module, "get_yolo_model", lambda weights: _FakeYoloModel())
    module, frame = _make_module(tmp_path, clock, enable_hands=True, **overrides)
    module._ensure_object_detectors_loaded()  # real (but network-free) YOLO + Hands construction
    return module, frame


def _feed_hand_positions(module, frame, clock, positions, step=_STRIDE_S):
    """Drive one stride tick per position through the REAL
    _run_object_detections code path (only module._hands.process is
    monkeypatched) -- BoxTracker feed, peak-speed/reversal kinematics, and
    process()'s debounce are all the real, unmodified code."""
    fired = []
    for cx, cy in positions:
        module._hands.process = lambda rgb, pts=_hand_box_points(cx, cy): _FakeHandsResult(
            [_FakeHandLandmarks(pts)]
        )
        # Force this frame to be a stride tick regardless of yolo_every_n_frames.
        module._frame_index = module._yolo_every_n_frames - 1
        fired += [e.type for e in module.process(frame)]
        clock["now"] += step
    return fired


def _oscillation(amplitude, count=24, cy=0.6):
    """A hand snapping between two points `2 * amplitude` apart every sample."""
    return [(0.5 + amplitude * (1 if i % 2 else -1), cy) for i in range(count)]


def test_moderate_oscillation_fires_erratic_at_shipped_thresholds(tmp_path, clock, monkeypatch):
    """A hand shaking back and forth over ~half its own width per sample --
    agitated, not violent. At the shipped 2.0 diagonals/second and 3 reversals
    this must fire; it did not before peak speed replaced net displacement,
    because oscillation cancels net displacement to nearly nothing."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    fired = _feed_hand_positions(module, frame, clock, _oscillation(0.04))
    assert "erratic_hand_movement" in fired
    assert module.is_alert_active()
    assert module._state_label() == "ERRATIC_HANDS"


def test_violent_oscillation_fires_erratic_at_shipped_thresholds(tmp_path, clock, monkeypatch):
    """The worst case, and the one that used to fail hardest: a hand covering
    more than one of its own diagonals between samples never matched its own
    track at the old gate, so every tick started a fresh track with an empty
    trail and the trajectory this feature reasons about never existed."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    fired = _feed_hand_positions(module, frame, clock, _oscillation(0.12))
    assert "erratic_hand_movement" in fired


def test_violent_motion_stays_one_tracked_hand_not_a_string_of_new_tracks(tmp_path, clock, monkeypatch):
    """The association half of the same defect, asserted directly: a fast hand
    must remain ONE identity with a real trail. Without this the kinematics
    above are unreachable no matter how they are computed."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    _feed_hand_positions(module, frame, clock, _oscillation(0.12))
    assert len(module._hand_tracker.tracks) == 1
    assert len(module._hand_tracker.tracks[0].trail) > 5


def test_a_single_smooth_reach_does_not_fire_erratic(tmp_path, clock, monkeypatch):
    """High speed, zero reversals -- reaching for the gearshift or a
    cupholder, the nearest innocent motion this must not fire on. Peak speed
    makes the speed gate EASIER to clear, so this boundary matters more now,
    not less: the reversal count is the only thing rejecting it."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    positions = [(0.15 + 0.03 * i, 0.6) for i in range(24)]  # one smooth direction
    fired = _feed_hand_positions(module, frame, clock, positions)
    assert "erratic_hand_movement" not in fired


def test_a_slow_deliberate_back_and_forth_does_not_fire_erratic(tmp_path, clock, monkeypatch):
    """The mirror-image false positive: plenty of reversals, no intensity --
    an unhurried hand moving between the wheel and the climate controls. The
    speed gate is the only thing rejecting this one, so it is what proves the
    gate still gates after being made reachable."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    fired = _feed_hand_positions(module, frame, clock, _oscillation(0.04, count=12), step=0.6)
    assert "erratic_hand_movement" not in fired


def test_stationary_hands_never_fire_erratic(tmp_path, clock, monkeypatch):
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    fired = _feed_hand_positions(module, frame, clock, [(0.5, 0.6)] * 24)  # steady on the wheel
    assert "erratic_hand_movement" not in fired
    assert not module.is_alert_active()


def test_erratic_hand_thresholds_are_configurable(tmp_path, clock, monkeypatch):
    """A stricter override must reject the same motion the shipped defaults
    accept -- the product requirement is a configurable threshold. Overriding
    UPWARD is the safe direction to test in: it cannot mask a feature that
    fails to fire."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch, erratic_hand_min_reversals=50)
    fired = _feed_hand_positions(module, frame, clock, _oscillation(0.12))
    assert "erratic_hand_movement" not in fired

    module, frame = _hand_module(tmp_path, clock, monkeypatch, erratic_hand_min_speed=500.0)
    fired = _feed_hand_positions(module, frame, clock, _oscillation(0.12))
    assert "erratic_hand_movement" not in fired


def test_the_erratic_alert_clears_within_the_trail_window_after_hands_settle(
    tmp_path, clock, monkeypatch
):
    """The tail of a peak-based measure, pinned rather than left implicit.

    A peak over the trail necessarily remembers the fastest step still in that
    trail, so the raw signal outlives the motion by up to the trail's own span
    (30 samples, one per stride tick) plus the tracker's release grace. That
    is bounded and intended -- the alternative, a peak over the last two
    samples, is the two-frame position delta this feature was built to stop
    being -- but "bounded" is only true if something checks. An alert that
    never cleared would be worse than one that never fired."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    _feed_hand_positions(module, frame, clock, _oscillation(0.12))
    assert module._erratic_hands_active

    settled_at = clock["now"]
    cleared_at = None
    for _ in range(60):
        _feed_hand_positions(module, frame, clock, [(0.5, 0.6)])
        if not module._erratic_hands_active and cleared_at is None:
            cleared_at = clock["now"]
    assert cleared_at is not None
    assert cleared_at - settled_at < 4.0


def test_two_hands_keep_separate_identities_under_the_loosened_match_gate(
    tmp_path, clock, monkeypatch
):
    """The cost side of widening the hand match gate, pinned down.

    Both hands carry the same "hand" label, and at the wheel they sit closer
    together than the widened gate, so a naive matcher could hand one hand's
    detection to the other's track. Greedy nearest-first assignment is what
    prevents it: with both hands detected, the two closest pairings win and
    each detection is consumed once. Two hands shaking in place must stay two
    tracks with stable ids, not swap or collapse into one."""
    module, frame = _hand_module(tmp_path, clock, monkeypatch)
    for i in range(24):
        wobble = 0.01 * (1 if i % 2 else -1)
        left = _hand_box_points(0.30 + wobble, 0.62)
        right = _hand_box_points(0.70 + wobble, 0.62)
        module._hands.process = lambda rgb, l=left, r=right: _FakeHandsResult(
            [_FakeHandLandmarks(l), _FakeHandLandmarks(r)]
        )
        module._frame_index = module._yolo_every_n_frames - 1
        module.process(frame)
        clock["now"] += _STRIDE_S

    tracks = module._hand_tracker.tracks
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}
    # Each track followed ONE hand the whole way: their trails stay on
    # opposite sides of the frame rather than crossing into each other.
    left_track, right_track = sorted(tracks, key=lambda t: t.centroid[0])
    assert all(x < 0.5 * W for _, x, _ in left_track.trail)
    assert all(x > 0.5 * W for _, x, _ in right_track.trail)


# ------------------------------------------------------------------ debounce layer
# These drive module._hands_erratic_raw directly rather than through
# _run_object_detections -- isolating process()'s debounce/event-wiring from
# the upstream kinematics (already covered above and in test_tracking.py).
def test_erratic_hands_debounce_layer_is_independent_of_face_state(tmp_path, clock):
    """The raw signal is judged every frame regardless of whether a face was
    found this frame -- a driver's hands leaving the wheel matters whether or
    not their face is currently resolvable."""
    module, frame = _make_module(tmp_path, clock, erratic_hands_seconds=0.2)
    module._face_mesh.process = lambda rgb: das._FakeFaceMeshResult(None)  # no face at all
    module._hands_erratic_raw = True
    fired = []
    for _ in range(6):
        fired += [e.type for e in module.process(frame)]
        clock["now"] += 0.1
    assert "erratic_hand_movement" in fired


def test_erratic_hands_resolves_when_motion_settles(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock, erratic_hands_seconds=0.2, signal_release_s=0.1)
    module._hands_erratic_raw = True
    for _ in range(6):
        module.process(frame)
        clock["now"] += 0.1
    assert module.is_alert_active()

    module._hands_erratic_raw = False
    for _ in range(5):
        module.process(frame)
        clock["now"] += 0.1
    assert module._erratic_hands_active is False
