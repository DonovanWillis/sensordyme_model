"""Tests for Driver Mode's erratic/frantic hand-movement detection (hands
leaving the wheel or moving chaotically) -- the trajectory-based counterpart
to Baby Mode's flailing detector, built on the same core.tracking.BoxTracker
kinematics (velocity + direction-reversal count) already unit-tested in
test_tracking.py. These focus on the debounce/event-wiring layer in
modules/driver_attention.py's process(), using the same real-module,
synthetic-but-geometrically-grounded harness as test_driver_attention_scenarios.py.
"""

import modules.driver_attention as driver_attention_module
import test_driver_attention_scenarios as das

W, H = das.W, das.H
clock = das.clock
_make_module = das._make_module
_FakeLandmark = das._FakeLandmark


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


def _hand_box_points(cx, cy, size=0.08):
    """A tiny cluster of 21 MediaPipe-Hands-shaped points around (cx, cy) in
    normalized [0,1] coords -- sized so that a realistic per-step displacement
    (a fraction of the box's own diagonal) stays within BoxTracker's match
    gate and so the SAME track persists across the whole trajectory, the way
    real frame-to-frame hand motion would."""
    return [(cx + (i % 3 - 1) * size * 0.3, cy + (i % 5 - 2) * size * 0.3) for i in range(21)]


def _hand_module(tmp_path, clock, monkeypatch, **overrides):
    monkeypatch.setattr(driver_attention_module, "get_yolo_model", lambda weights: _FakeYoloModel())
    module, frame = _make_module(tmp_path, clock, enable_hands=True, **overrides)
    module._ensure_object_detectors_loaded()  # real (but network-free) YOLO + Hands construction
    return module, frame


def _feed_hand_positions(module, frame, clock, positions, step=0.1):
    """Drive one stride tick per position through the REAL
    _run_object_detections code path (only module._hands.process is
    monkeypatched) -- BoxTracker feed, velocity/reversal kinematics, and
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


def test_hands_oscillating_rapidly_fires_erratic_hand_movement(tmp_path, clock, monkeypatch):
    module, frame = _hand_module(
        tmp_path, clock, monkeypatch, erratic_hands_seconds=0.3,
        erratic_hand_min_speed=0.5, erratic_hand_min_reversals=3,
    )
    # A hand snapping between two nearby points every stride tick -- fast AND
    # chaotic, but each step stays within the tracker's match gate so it's
    # genuinely the SAME hand being followed, not a string of brand-new tracks.
    positions = [(0.5 + 0.02 * (1 if i % 2 else -1), 0.6) for i in range(14)]
    fired = _feed_hand_positions(module, frame, clock, positions)
    assert "erratic_hand_movement" in fired
    assert module.is_alert_active()
    assert module._state_label() == "ERRATIC_HANDS"


def test_a_single_smooth_reach_does_not_fire_erratic(tmp_path, clock, monkeypatch):
    """High speed, zero reversals -- reaching for the gearshift or a
    cupholder, the nearest innocent motion this must not fire on."""
    module, frame = _hand_module(
        tmp_path, clock, monkeypatch, erratic_hands_seconds=0.3,
        erratic_hand_min_speed=0.5, erratic_hand_min_reversals=3,
    )
    positions = [(0.3 + 0.05 * i, 0.6) for i in range(10)]  # one smooth direction
    fired = _feed_hand_positions(module, frame, clock, positions)
    assert "erratic_hand_movement" not in fired


def test_stationary_hands_never_fire_erratic(tmp_path, clock, monkeypatch):
    module, frame = _hand_module(
        tmp_path, clock, monkeypatch, erratic_hands_seconds=0.3,
        erratic_hand_min_speed=0.5, erratic_hand_min_reversals=3,
    )
    positions = [(0.5, 0.6)] * 10  # hands steady on the wheel
    fired = _feed_hand_positions(module, frame, clock, positions)
    assert "erratic_hand_movement" not in fired
    assert not module.is_alert_active()


def test_erratic_hand_thresholds_are_configurable(tmp_path, clock, monkeypatch):
    """A stricter min_reversals override must reject the same motion the
    default accepts -- the product requirement is a configurable threshold."""
    module, frame = _hand_module(
        tmp_path, clock, monkeypatch, erratic_hands_seconds=0.3,
        erratic_hand_min_speed=0.5, erratic_hand_min_reversals=50,
    )
    positions = [(0.5 + 0.02 * (1 if i % 2 else -1), 0.6) for i in range(14)]
    fired = _feed_hand_positions(module, frame, clock, positions)
    assert "erratic_hand_movement" not in fired


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
