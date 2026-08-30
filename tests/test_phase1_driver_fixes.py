"""Regression tests for five defects in Driver Mode that had no test able to
see them, each written to FAIL against the code as it shipped.

They share test_driver_attention_scenarios.py's harness: real
DriverAttentionModule, real calibration, real trackers and fusion, with only
the inference boundaries (FaceMesh landmarks, the YOLO model) replaced by
synthetic-but-geometrically-real stand-ins. Thresholds are the SHIPPED ones
unless a test's whole point is a threshold override.

The defects:
  * PERCLOS reported a fatigue percentage 33 milliseconds into a session,
    from a "window" two samples wide (bug 2 below).
  * The frame size every AI-configured zone rule divides by was only ever
    written on an object-detector tick, so with detection off or errored the
    rules silently could not fire (bug 3).
  * rearview_tolerance_seconds could not apply to any real head turn (bug 4).
  * eating_drinking fired on a bottle anywhere in the frame, cupholder
    included, with no spatial relation to the driver (bug 6).
  * Two config values were unguarded divisors inside the frame loop (bug 7).
"""

import numpy as np
import pytest
import test_driver_attention_scenarios as das

import modules.driver_attention as driver_attention_module
from core.zones import Zone
from modules.driver_attention import PerclosTracker

W, H = das.W, das.H
clock = das.clock
_make_module = das._make_module


# ---------------------------------------------------------------- bug 2: PERCLOS
# PERCLOS is defined as the PERCENTAGE OF TIME the eyes are closed OVER A
# WINDOW. Below its window it is not a noisy estimate of anything -- it is
# undefined, and the old code answered it anyway by dividing closed time by
# the span of however many samples it happened to be holding.
def test_perclos_is_zero_until_a_real_window_has_accumulated():
    """The exact shipped failure: one closed frame, 33ms into the session,
    read as 100% of time closed -- 6.7x the 0.15 fatigue threshold."""
    tracker = PerclosTracker(window_s=60.0)
    tracker.update(True, 0.0)
    assert tracker.update(True, 0.033) == 0.0
    assert tracker.update(True, 0.2) == 0.0
    assert tracker.update(True, 0.5) == 0.0
    assert tracker.update(True, 5.0) == 0.0
    assert tracker.update(True, 9.9) == 0.0


def test_perclos_reports_once_its_window_is_actually_populated():
    """The guard must delay the metric, not disable it."""
    tracker = PerclosTracker(window_s=60.0)
    tracker.update(True, 0.0)
    assert tracker.update(True, 12.0) == pytest.approx(1.0)


def test_perclos_minimum_window_cannot_exceed_a_shorter_configured_window():
    """A deployment that configures a window shorter than the default minimum
    must still get a metric, not a permanent zero -- the minimum is a floor on
    'enough evidence', and a 5s window IS all the evidence that config asks for."""
    tracker = PerclosTracker(window_s=5.0)
    tracker.update(True, 0.0)
    assert tracker.update(True, 1.0) == 0.0
    assert tracker.update(True, 6.0) == pytest.approx(1.0)


def test_no_fatigue_warning_in_the_opening_seconds_of_a_session(tmp_path, clock, monkeypatch):
    """End-to-end statement of the same defect: a driver whose very first
    measured frames read closed used to get "fatigue_warning" almost
    immediately, because PERCLOS went straight to 1.0 against a 0.15 bar."""
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    # Closed eyes at the EAR seam. The EAR geometry itself is covered by
    # test_head_pose_geometry.py / the eye_signal suite; what is under test
    # here is what PERCLOS does with a sustained closed reading.
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio", lambda *pts: 0.05)
    fired = das._feed(module, frame, clock, seconds=5.0)
    assert "fatigue_warning" not in fired
    assert module._last_perclos == 0.0


def test_fatigue_warning_still_fires_once_the_window_is_real(tmp_path, clock, monkeypatch):
    """The other half: the guard must not have quietly killed fatigue detection."""
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio", lambda *pts: 0.05)
    fired = das._feed(module, frame, clock, seconds=20.0)
    assert "fatigue_warning" in fired
    assert module._last_perclos > 0.15


# ------------------------------------------------- bug 3: frame size for zone rules
# core/zones.py normalizes a pixel box by the frame size reported through
# get_detections(). A reported size of 0 makes max(frame_w, 1) == 1, so
# foot_x becomes the raw pixel value -- never inside a 0..1 polygon, for any
# polygon, silently, forever.
def test_frame_size_is_reported_even_when_object_detection_never_ticks(tmp_path, clock):
    """The harness config sets yolo_every_n_frames to a million, which is
    exactly the shipped "detection is off / erroring" situation: process()
    runs, the detector never does, and the frame size stayed (0, 0)."""
    module, frame = _make_module(tmp_path, clock)
    module.process(frame)
    detections = module.get_detections()
    assert (detections.width, detections.height) == (W, H)


def test_a_zone_rule_can_actually_match_with_object_detection_off(tmp_path, clock):
    """The consequence, stated in the terms the AI-configure feature uses: a
    normalized polygon over the lower half of the frame must contain a box
    whose feet are in the lower half. With width/height reported as 0 this
    test's assertion is unreachable no matter how the polygon is drawn."""
    module, frame = _make_module(tmp_path, clock)
    module.process(frame)
    detections = module.get_detections()
    lower_half = Zone(name="footwell", polygon=((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)))
    box = (280.0, 380.0, 360.0, 440.0)  # feet at y=440 of 480 -> normalized 0.92
    assert lower_half.contains_box(box, detections.width, detections.height)


# ------------------------------------------- bug 4: mirror-check tolerance was dead
def _sweep_to(module, frame, clock, target_yaw, ramp_s=0.6, hold_s=8.0, step=0.05):
    """A head turning to `target_yaw` at a human rate and holding there.

    The ramp is the whole point. A distraction episode opens the instant
    |yaw| passes yaw_threshold_deg (35), which is necessarily BELOW
    mirror_check_yaw_deg (55) -- a head does not teleport past 55, it passes
    through 35 on the way. Returns the elapsed time distraction first fired,
    or None.
    """
    start = clock["now"]
    fired_at = None
    while clock["now"] - start < ramp_s + hold_s:
        elapsed = clock["now"] - start
        yaw = min(target_yaw, target_yaw * elapsed / ramp_s) if ramp_s > 0 else target_yaw
        landmarks = das.make_fake_landmarks(yaw_deg=yaw)
        module._face_mesh.process = lambda rgb, lm=landmarks: das._FakeFaceMeshResult(lm)
        types = [e.type for e in module.process(frame)]
        if "distraction" in types and fired_at is None:
            fired_at = elapsed
        clock["now"] += step
    return fired_at


def test_a_sustained_blind_spot_check_gets_the_rearview_tolerance(tmp_path, clock):
    """A 70-degree turn is shaped like a blind-spot check, so it is supposed to
    get rearview_tolerance_seconds (5.0) rather than distraction_seconds (3.0).
    Measured before the fix: it fired at 3.45s and never corroborated once --
    the setting could not apply to any real head turn, in any configuration."""
    module, frame = _make_module(tmp_path, clock)
    assert module._distract_tracker._short_s == 3.0
    assert module._distract_tracker._long_s == 5.0
    fired_at = _sweep_to(module, frame, clock, target_yaw=70.0)
    assert fired_at is not None
    assert fired_at >= 5.0
    assert module._distract_tracker.corroborated is True


def test_an_ordinary_off_center_glance_still_fires_at_the_short_threshold(tmp_path, clock):
    """The safety boundary the fix must not erode: a 45-degree turn is past
    the distraction threshold but nowhere near blind-spot magnitude, so it
    gets no extra patience at all."""
    module, frame = _make_module(tmp_path, clock)
    fired_at = _sweep_to(module, frame, clock, target_yaw=45.0)
    assert fired_at is not None
    assert fired_at < 4.0
    assert module._distract_tracker.corroborated is False


def test_a_turn_that_never_reaches_blind_spot_magnitude_is_not_corroborated_by_a_later_episode(
    tmp_path, clock
):
    """Corroboration accumulates WITHIN an episode and must not leak across
    them: a wide turn followed by an ordinary glance must put the ordinary
    glance back on the short threshold."""
    module, frame = _make_module(tmp_path, clock)
    _sweep_to(module, frame, clock, target_yaw=70.0, hold_s=6.0)
    das._feed(module, frame, clock, yaw_deg=0.0, seconds=2.0)  # back to the road; episode ends
    fired_at = _sweep_to(module, frame, clock, target_yaw=45.0)
    assert fired_at is not None
    assert fired_at < 4.0
    assert module._distract_tracker.corroborated is False


# ------------------------------------- bug 6: eating/drinking had no spatial gate
class _FakeBox:
    def __init__(self, class_id, xyxy, conf):
        self.cls = [class_id]
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    """A detector returning a fixed set of boxes -- the seam is the model, so
    everything downstream (class mapping, the spatial gate, the edge event)
    is the real module code."""

    names = {0: "person", 39: "bottle", 67: "cell phone"}

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, frame, **kwargs):
        return [_FakeResult(self._boxes)]


#: A seated driver filling most of the frame, as a detector would report them.
_DRIVER_BOX = (200, 60, 460, 470)
#: The top 35% of that box (core/detectors.upper_region's default) is y 60..203.
_BOTTLE_AT_FACE = (330, 110, 380, 200)
_BOTTLE_IN_CUPHOLDER = (300, 395, 345, 465)


def _detection_module(tmp_path, clock, monkeypatch, boxes):
    monkeypatch.setattr(
        driver_attention_module, "get_yolo_model",
        lambda weights: _FakeYoloModel(boxes),
    )
    module, frame = _make_module(
        tmp_path, clock,
        enable_hands=False,
        yolo_classes={"cell phone": "phone_detected", "bottle": "eating_drinking"},
    )
    module._ensure_object_detectors_loaded()
    module._frame_index = module._yolo_every_n_frames - 1
    return module, frame


def test_a_bottle_in_the_cupholder_is_not_drinking(tmp_path, clock, monkeypatch):
    """The shipped behaviour fired eating_drinking on ANY box of a food/drink
    class ANYWHERE in the frame. A bottle sitting in a cupholder was
    indistinguishable from a bottle being drunk from."""
    module, frame = _detection_module(
        tmp_path, clock, monkeypatch,
        [_FakeBox(0, _DRIVER_BOX, 0.9), _FakeBox(39, _BOTTLE_IN_CUPHOLDER, 0.8)],
    )
    fired = [e.type for e in module.process(frame)]
    assert "eating_drinking" not in fired
    assert module._last_eating_boxes == []


def test_a_bottle_raised_to_the_face_is_drinking(tmp_path, clock, monkeypatch):
    """The other half: the gate must not have simply turned the feature off."""
    module, frame = _detection_module(
        tmp_path, clock, monkeypatch,
        [_FakeBox(0, _DRIVER_BOX, 0.9), _FakeBox(39, _BOTTLE_AT_FACE, 0.8)],
    )
    fired = [e.type for e in module.process(frame)]
    assert "eating_drinking" in fired
    assert module._last_eating_boxes == [tuple(int(v) for v in _BOTTLE_AT_FACE)]


def test_an_ungated_bottle_is_still_reported_as_a_detection_for_custom_rules(
    tmp_path, clock, monkeypatch
):
    """The gate belongs to this module's own eating_drinking CONCLUSION, not
    to the detector's observations. get_detections() feeds user-authored zone
    rules that target the THING ("a bottle in this region"), so suppressing
    the box there would break a feature that is working correctly."""
    module, frame = _detection_module(
        tmp_path, clock, monkeypatch,
        [_FakeBox(0, _DRIVER_BOX, 0.9), _FakeBox(39, _BOTTLE_IN_CUPHOLDER, 0.8)],
    )
    module.process(frame)
    labels = [d.label for d in module.get_detections().detections]
    assert "bottle" in labels


# ------------------------------------------ bug 7: unguarded divisors in the loop
def test_a_zero_perclos_threshold_does_not_crash_the_frame_loop(tmp_path, clock):
    """perclos_threshold and glance_min_count are plain config numbers divided
    by inside RiskFusion's input calculation with no guard. Either set to 0 --
    a plausible "turn this contribution off" edit -- was a ZeroDivisionError
    raised once per frame, taking the whole video loop down."""
    module, frame = _make_module(tmp_path, clock, perclos_threshold=0.0)
    das._feed(module, frame, clock, seconds=1.0)  # must not raise
    assert module._last_risk_score >= 0.0


def test_a_zero_glance_min_count_does_not_crash_the_frame_loop(tmp_path, clock):
    module, frame = _make_module(tmp_path, clock, glance_min_count=0)
    das._feed(module, frame, clock, seconds=1.0)  # must not raise
    assert module._last_risk_score >= 0.0
