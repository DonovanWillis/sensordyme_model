"""behavior_v2 regression scenarios through the REAL DriverAttentionModule.

SYNTHETIC SCENARIOS ONLY. Face landmarks come from the geometric generator in
tests/test_driver_attention_scenarios.py (known 3-D pose -> projected points),
detector output from a fake model returning chosen boxes, hands from a fake
MediaPipe Hands, time from a deterministic clock. Everything between those
seams -- calibration, smoothing, the shipped trackers, the v2 components,
event emission -- is the real code. Passing shows the logic behaves as
designed on constructed inputs; it is NOT a measured accuracy and says
nothing about real cabins, faces, lenses or detectors.

Where useful a scenario also runs with behavior_v2 OFF to pin the shipped
behaviour it changes (the audit-confirmed defect), so a reader can see both.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from test_driver_attention_scenarios import (
    _CONFIG,
    H,
    W,
    _FakeFaceMeshResult,
    _FakeLandmark,
    make_fake_landmarks,
)
from test_phase1_driver_fixes import _FakeBox, _FakeResult

import modules.driver_attention as driver_attention_module
from modules.driver_attention import DriverAttentionModule

STEP = 0.05
DRIVER_BOX = (200, 60, 460, 470)
# Face (fake landmarks, neutral) spans roughly x 274..366, y 190..308: width ~92, centre x 320.
MOUNT_PHONE = (470, 300, 520, 380)  # dash mount, frame right, inside driver reach
LAP_PHONE = (295, 370, 345, 440)  # in the driver's lap, below the face
LAP_HAND = (285, 385, 355, 460)
PASSENGER_PHONE = (20, 300, 70, 380)  # far frame left: outside driver reach
PASSENGER_HAND = (10, 310, 80, 400)


# ---------------------------------------------------------------------------- fakes
class _FakeYolo:
    names: ClassVar[dict[int, str]] = {0: "person", 67: "cell phone"}

    def __init__(self) -> None:
        self.phones: list[tuple] = []
        self.fail = False

    def predict(self, frame, **kwargs):
        if self.fail:
            raise RuntimeError("accelerator timeout")
        boxes = [_FakeBox(0, DRIVER_BOX, 0.9)] + [_FakeBox(67, b, 0.8) for b in self.phones]
        return [_FakeResult(boxes)]


class _FakeHandLandmarks:
    def __init__(self, box) -> None:
        x1, y1, x2, y2 = box
        self.landmark = [_FakeLandmark(x / W, y / H) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]


class _FakeHands:
    def __init__(self) -> None:
        self.boxes: list[tuple] = []

    def process(self, rgb):
        class _R:
            multi_hand_landmarks = [_FakeHandLandmarks(b) for b in self.boxes] or None

        return _R()

    def close(self) -> None:
        pass


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    return t


@pytest.fixture
def yolo(monkeypatch):
    model = _FakeYolo()
    monkeypatch.setattr(driver_attention_module, "get_yolo_model", lambda weights: model)
    return model


def _textured_frame() -> np.ndarray:
    """A frame with texture everywhere, so no eye patch reads as a dark lens."""
    return np.random.default_rng(7).integers(60, 200, (H, W, 3), dtype=np.uint8)


def landmarks(yaw=0.0, pitch=0.0, lid=6.0, iris_dy=0.0, lid_left=None):
    """Fake landmarks in the MODULE's convention: +yaw toward the frame's right,
    -pitch down. `lid` is the eyelid half-gap in px (6 = open, EAR 0.40, iris-
    openness 1.0; 4 = lowered, EAR 0.27; 1 = shut, iris-openness 0.17);
    `iris_dy` moves both irises down the image (eyes cast down)."""
    lms = make_fake_landmarks(pitch_deg=-pitch, yaw_deg=-yaw)
    pts = [[p.x * W, p.y * H] for p in lms]
    for anchor, uppers, lowers, iris, half in (
        (33, (160, 158), (144, 153), (468, 469, 470, 471, 472), lid),
        (362, (385, 387), (380, 373), (473, 474, 475, 476, 477), lid if lid_left is None else lid_left),
    ):
        x0, y0 = pts[anchor]
        for idx, dx in zip(uppers, (8, 22), strict=True):
            pts[idx] = [x0 + dx, y0 - half]
        for idx, dx in zip(lowers, (8, 22), strict=True):
            pts[idx] = [x0 + dx, y0 + half]
        c, r, top, left, bottom = iris
        pts[c] = [x0 + 15, y0 + iris_dy]
        pts[r] = [x0 + 21, y0 + iris_dy]
        pts[left] = [x0 + 9, y0 + iris_dy]
        pts[top] = pts[c]
        pts[bottom] = pts[c]
    return [_FakeLandmark(x / W, y / H) for x, y in pts]


def _paint_lenses(frame: np.ndarray, lms, eyes=("right", "left")) -> np.ndarray:
    """Uniform dark rectangles over the chosen eyes: a sunglasses lens to a visible-light camera."""
    out = frame.copy()
    for eye in eyes:
        anchor = 33 if eye == "right" else 362
        x0, y0 = lms[anchor].x * W, lms[anchor].y * H
        out[int(y0 - 15):int(y0 + 15), int(x0 - 10):int(x0 + 40)] = 10
    return out


def _make(tmp_path, clock, *, v2=True, hands=None, frame=None, **overrides):
    config = {
        **_CONFIG, "calibration_path": str(tmp_path / f"calibration_{v2}.json"),
        "yolo_every_n_frames": 1, "behavior_v2": v2, **overrides,
    }
    module: Any = DriverAttentionModule(config)
    module._hands = hands if hands is not None else _FakeHands()
    frame = _textured_frame() if frame is None else frame
    neutral = landmarks()
    module._face_mesh.process = lambda rgb, lm=neutral: _FakeFaceMeshResult(lm)
    for _ in range(int(_CONFIG["calibration_duration_s"] / STEP) + 3):
        module.process(frame)
        clock["now"] += STEP
    assert module._calibrator.is_calibrated
    return module, frame


def _feed(module, frame, clock, seconds, lms=None, step=STEP):
    """Drive `seconds` of frames with the given landmarks (None = face lost); returns events."""
    module._face_mesh.process = lambda rgb, lm=lms: _FakeFaceMeshResult(lm)
    events = []
    for _ in range(max(1, round(seconds / step))):
        events += module.process(frame)
        clock["now"] += step
    return events


def _types(events) -> list[str]:
    return [e.type for e in events]


# ============================================================ default behaviour unchanged
def test_flag_off_builds_nothing_and_adds_no_fields(tmp_path, clock, yolo):
    module, _frame = _make(tmp_path, clock, v2=False)
    assert module._behavior is None
    assert module.behavior_uncertainty() == {}
    assert not any(k.startswith("behavior") for k in module.get_telemetry_snapshot())
    assert "behavior" not in module.get_debug_state()
    assert "long_glance" not in module.event_requirements()


def test_flag_on_attentive_driver_raises_nothing(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    events = _feed(module, frame, clock, 5.0, landmarks())
    assert events == []
    assert module.get_telemetry_snapshot()["behavior_eye_state"] == "open"
    assert module.get_debug_state()["behavior"]["behavior_gaze_zone"] == "road"
    assert set(module.event_requirements()) >= {"long_glance", "glance_budget_exceeded"}


# ============================================================ phone
def test_mounted_phone_and_attentive_driver_is_not_phone_use(tmp_path, clock, yolo):
    yolo.phones = [MOUNT_PHONE]
    module, frame = _make(tmp_path, clock)
    events = _feed(module, frame, clock, 6.0, landmarks())
    assert "phone_use" not in _types(events)
    assert module._behavior.phone.assessment.state == "visible_only"


def test_mounted_phone_while_looking_into_the_lap(tmp_path, clock, yolo):
    """Shipped rule: any phone box + eyes off road = phone_use. v2: no hand, no relation -> not use."""
    yolo.phones = [MOUNT_PHONE]
    v1, frame = _make(tmp_path, clock, v2=False)
    assert "phone_use" in _types(_feed(v1, frame, clock, 4.0, landmarks(pitch=-30.0)))
    v2, frame = _make(tmp_path, clock)
    assert "phone_use" not in _types(_feed(v2, frame, clock, 4.0, landmarks(pitch=-30.0)))


def test_passenger_holding_a_phone_is_not_phone_use(tmp_path, clock, yolo):
    yolo.phones = [PASSENGER_PHONE]
    hands = _FakeHands()
    hands.boxes = [PASSENGER_HAND]
    v1, frame = _make(tmp_path, clock, v2=False, hands=hands)
    assert "phone_use" in _types(_feed(v1, frame, clock, 4.0, landmarks(yaw=-40.0)))
    v2, frame = _make(tmp_path, clock, hands=hands)
    assert "phone_use" not in _types(_feed(v2, frame, clock, 4.0, landmarks(yaw=-40.0)))
    assert v2._behavior.phone.assessment.reason == "outside driver reach region"


def test_driver_holding_and_looking_at_phone_is_phone_use_once(tmp_path, clock, yolo):
    yolo.phones = [LAP_PHONE]
    hands = _FakeHands()
    hands.boxes = [LAP_HAND]
    module, frame = _make(tmp_path, clock, hands=hands)
    events = _feed(module, frame, clock, 3.0, landmarks(pitch=-30.0))
    phone = [e for e in events if e.type == "phone_use"]
    assert len(phone) == 1
    assert phone[0].contributing_signals.startswith("behavior_v2: hand-held")
    assert phone[0].confidence == pytest.approx(0.8)
    assert module.is_alert_active()
    assert "phone_use" in module.active_alert_conditions()


def test_phone_detected_needs_persistence_not_one_detection(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    yolo.phones = [MOUNT_PHONE]
    first = _feed(module, frame, clock, STEP, landmarks())  # one tick with a phone
    yolo.phones = []
    rest = _feed(module, frame, clock, 1.0, landmarks())
    assert "phone_detected" not in _types(first + rest)
    yolo.phones = [MOUNT_PHONE]
    assert _types(_feed(module, frame, clock, 1.0, landmarks())).count("phone_detected") == 1


def test_detector_interruption_makes_phone_state_unknown_not_none(tmp_path, clock, yolo):
    yolo.phones = [LAP_PHONE]
    module, frame = _make(tmp_path, clock)
    yolo.fail = True
    events = _feed(module, frame, clock, 1.0, landmarks())
    assert module._last_phone_boxes == []  # the shipped path now sees "no phone"
    phone = module._behavior.phone.assessment
    assert phone.state == "unknown" and "accelerator timeout" in phone.reason
    assert module.behavior_uncertainty()["phone"] == phone.reason
    assert not {"phone_use", "phone_detected"} & set(_types(events))


# ============================================================ glances
def test_repeated_mirror_checks_are_not_distraction(tmp_path, clock, yolo):
    def drive(module, frame):
        events = []
        for yaw in (50.0, -60.0) * 3:  # six 0.8 s mirror checks in ~23 s
            events += _feed(module, frame, clock, 0.8, landmarks(yaw=yaw))
            events += _feed(module, frame, clock, 3.0, landmarks())
        return _types(events)

    v1, frame = _make(tmp_path, clock, v2=False)
    assert "frequent_glancing" in drive(v1, frame)  # shipped: every mirror check counts
    v2, frame = _make(tmp_path, clock)
    fired = drive(v2, frame)
    assert not {"distraction", "frequent_glancing", "long_glance", "glance_budget_exceeded"} & set(fired)
    assert v2._behavior.glance.budget_used_s == 0.0


def test_one_second_instrument_glance_is_nothing(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    events = _feed(module, frame, clock, 1.0, landmarks(pitch=-15.0))
    events += _feed(module, frame, clock, 2.0, landmarks())
    assert events == []


def test_single_long_off_road_glance_fires_long_glance(tmp_path, clock, yolo):
    """Toward the centre console: under the shipped yaw/pitch thresholds, so v1 is silent."""
    v1, frame = _make(tmp_path, clock, v2=False)
    assert _feed(v1, frame, clock, 2.5, landmarks(yaw=-30.0, pitch=-20.0)) == []
    v2, frame = _make(tmp_path, clock)
    events = _feed(v2, frame, clock, 2.5, landmarks(yaw=-30.0, pitch=-20.0))
    assert _types(events) == ["long_glance"]
    assert "infotainment" in events[0].contributing_signals
    assert events[0].confidence is None


def test_four_short_off_road_glances_exceed_the_budget(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    events = []
    for _ in range(4):  # 4 x 1.8 s within ~15 s
        events += _feed(module, frame, clock, 1.8, landmarks(yaw=-30.0, pitch=-20.0))
        events += _feed(module, frame, clock, 2.0, landmarks())
    assert _types(events) == ["glance_budget_exceeded"]


# ============================================================ eyes
def test_looking_down_at_the_lap_is_not_drowsiness(tmp_path, clock, yolo):
    """Lids lowered ~33% by a downward look: the shipped 0.75 x reference threshold reads it as closure."""
    pose = landmarks(pitch=-30.0, lid=4.0)
    v1, frame = _make(tmp_path, clock, v2=False)
    assert "drowsiness" in _types(_feed(v1, frame, clock, 3.0, pose))
    v2, frame = _make(tmp_path, clock)
    fired = _types(_feed(v2, frame, clock, 3.0, pose))
    assert "drowsiness" not in fired
    assert v2._behavior.eye_state.state == "looking_down"


def test_eyes_cast_down_with_head_level_is_not_drowsiness(tmp_path, clock, yolo):
    pose = landmarks(lid=4.0, iris_dy=3.0)
    v2, frame = _make(tmp_path, clock)
    assert "drowsiness" not in _types(_feed(v2, frame, clock, 3.0, pose))
    assert v2._behavior.eye_state.reason == "EAR drop with iris shifted down, iris visible"


def test_real_closure_still_fires_drowsiness_even_while_nodding(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    events = _feed(module, frame, clock, 3.0, landmarks(pitch=-20.0, lid=1.0))
    assert _types(events).count("drowsiness") == 1
    assert module.is_alert_active()


def test_sunglasses_make_eye_state_unknown_without_a_false_alarm(tmp_path, clock, yolo):
    pose = landmarks(lid=1.0)  # landmarks guessed behind dark lenses read "shut"
    v1, frame = _make(tmp_path, clock, v2=False)
    shaded = _paint_lenses(frame, pose)
    assert "drowsiness" in _types(_feed(v1, shaded, clock, 4.0, pose))  # shipped: false alarm
    v2, frame = _make(tmp_path, clock)
    events = _feed(v2, _paint_lenses(frame, pose), clock, 4.0, pose)
    assert "drowsiness" not in _types(events)
    assert v2._behavior.eye_state.state == "unknown"
    assert v2.get_telemetry_snapshot()["behavior_eyes_occluded"] == 2
    assert "occluded" in v2.behavior_uncertainty()["eye_state"]
    assert v2.get_telemetry_snapshot()["behavior_eye_state"] == "unknown"


def test_partial_occlusion_defers_to_the_visible_eye(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    open_left = landmarks(lid=1.0, lid_left=6.0)  # right eye covered (garbage lids), left eye open
    covered = _paint_lenses(frame, open_left, eyes=("right",))
    assert _feed(module, covered, clock, 3.0, open_left) == []
    assert module._behavior.eye_state.state == "open"
    assert module.get_telemetry_snapshot()["behavior_eyes_occluded"] == 1
    assert "eye_state" not in module.behavior_uncertainty()
    both_shut = landmarks(lid=1.0)  # now the visible eye closes too
    assert "drowsiness" in _types(_feed(module, _paint_lenses(frame, both_shut, eyes=("right",)),
                                        clock, 3.0, both_shut))


def test_brief_face_loss_mid_closure_still_fires_drowsiness_once(tmp_path, clock, yolo):
    def drive(module, frame):
        shut = landmarks(lid=1.0)
        return _types(_feed(module, frame, clock, 1.2, shut) + _feed(module, frame, clock, 0.5, None)
                      + _feed(module, frame, clock, 1.5, shut))

    v1, frame = _make(tmp_path, clock, v2=False)
    assert "drowsiness" not in drive(v1, frame)  # shipped: face loss reset the closure clock
    v2, frame = _make(tmp_path, clock)
    assert drive(v2, frame).count("drowsiness") == 1


def test_long_face_loss_is_attention_lost_and_reported_uncertain(tmp_path, clock, yolo):
    module, frame = _make(tmp_path, clock)
    _feed(module, frame, clock, 1.2, landmarks(lid=1.0))
    yolo.phones = []
    events = _feed(module, frame, clock, 10.0, None)
    assert "attention_lost" in _types(events)
    assert "drowsiness" not in _types(events)
    uncertain = module.behavior_uncertainty()
    assert uncertain["eye_state"] == "face not detected"
    assert uncertain["glance"] == "face not detected"
    assert "drowsiness" in uncertain
    assert module._behavior.drowsiness.closed_s == 0.0
    assert module.get_debug_state()["behavior"]["uncertain"] == uncertain


def test_stale_frames_accrue_no_evidence(tmp_path, clock, yolo):
    """Frames 1 s apart: EvidenceClock credits 0.5 s per gap; v2 credits nothing."""
    shut = landmarks(lid=1.0)
    v1, frame = _make(tmp_path, clock, v2=False)
    assert "drowsiness" in _types(_feed(v1, frame, clock, 8.0, shut, step=1.0))
    v2, frame = _make(tmp_path, clock)
    events = _feed(v2, frame, clock, 8.0, shut, step=1.0)
    events += _feed(v2, frame, clock, 8.0, landmarks(yaw=-30.0, pitch=-20.0), step=1.0)
    assert not {"drowsiness", "long_glance", "glance_budget_exceeded"} & set(_types(events))
    assert v2._behavior.drowsiness.closed_s == 0.0
    assert v2._behavior.glance.budget_used_s == 0.0
