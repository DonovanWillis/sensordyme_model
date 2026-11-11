"""Unit tests for core/behavior/ components (synthetic scenarios, deterministic time).

Every scenario here is SYNTHETIC: hand-built boxes, angles and timestamps.
Passing them shows the logic does what it says; it is not a measured accuracy.
"""

from __future__ import annotations

from core.behavior.evidence import UNKNOWN
from core.behavior.phone import (
    DETECTOR_FAILED,
    HELD,
    IN_USE,
    NEAR_FACE,
    NONE,
    VISIBLE_ONLY,
    PhoneDetection,
    PhoneFrame,
    PhoneInteraction,
    gaze_toward,
)

STEP = 0.05



FACE = (280.0, 200.0, 360.0, 300.0)  # face width 80, centre (320, 250)
LAP_PHONE = (300.0, 380.0, 340.0, 440.0)  # below the face, within reach
LAP_HAND = (290.0, 390.0, 360.0, 460.0)
MOUNT_PHONE = (470.0, 300.0, 520.0, 380.0)  # dash mount, frame right, within reach
PASSENGER_PHONE = (20.0, 300.0, 70.0, 380.0)  # far frame left: outside reach (320 - 2.5*80 = 120)
PASSENGER_HAND = (10.0, 310.0, 80.0, 400.0)
LOOK_DOWN = (0.0, -30.0)  # calibrated (yaw, pitch): pitch -30 = down
LOOK_ROAD = (0.0, 0.0)


def _run_phone(machine, seconds, *, t0, phone_boxes, hand_boxes=(), gaze=LOOK_ROAD, face=FACE,
               detector="ok", hands_available=True, jitter=0.0):
    updates = []
    n = round(seconds / STEP)
    for i in range(n):
        t = t0 + i * STEP
        boxes = tuple((b[0] + jitter * (i % 3), b[1], b[2] + jitter * (i % 3), b[3]) for b in phone_boxes)
        det = PhoneDetection(t=t, phone_boxes=boxes, hand_boxes=tuple(hand_boxes),
                             hands_available=hands_available, phone_score=0.8)
        updates.append(machine.update(PhoneFrame(t=t, detector=detector, detection=det, face_box=face, gaze=gaze)))
    return updates, t0 + n * STEP


def test_mounted_phone_with_attentive_driver_never_becomes_in_use():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 6.0, t0=0.0, phone_boxes=[MOUNT_PHONE])
    states = {u.assessment.state for u in ups}
    assert IN_USE not in states and not any(u.use_started for u in ups)
    assert ups[-1].assessment.state == VISIBLE_ONLY  # static for >= static_s, no hand
    assert "mounted" in ups[-1].assessment.reason
    # before stillness is established it is undecidable, not "safe"
    assert any(u.assessment.state == UNKNOWN for u in ups[:30])


def test_mounted_phone_looked_at_without_a_hand_is_still_not_phone_use():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 6.0, t0=0.0, phone_boxes=[MOUNT_PHONE], gaze=(30.0, -10.0))
    assert not any(u.use_started for u in ups)


def test_passenger_holding_a_phone_is_visible_only():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 4.0, t0=0.0, phone_boxes=[PASSENGER_PHONE], hand_boxes=[PASSENGER_HAND],
                        gaze=(-40.0, -10.0))  # even with the driver glancing that way
    assert not any(u.use_started for u in ups)
    assert ups[-1].assessment.state == VISIBLE_ONLY
    assert "outside driver" in ups[-1].assessment.reason


def test_driver_holding_and_looking_at_phone_for_the_sustain_becomes_in_use_once():
    m = PhoneInteraction()
    ups, t = _run_phone(m, 1.5, t0=0.0, phone_boxes=[LAP_PHONE], hand_boxes=[LAP_HAND], gaze=LOOK_DOWN)
    assert ups[-1].assessment.state == HELD and not any(u.use_started for u in ups)
    ups2, _ = _run_phone(m, 2.0, t0=t, phone_boxes=[LAP_PHONE], hand_boxes=[LAP_HAND], gaze=LOOK_DOWN)
    assert ups2[-1].assessment.state == IN_USE
    assert sum(u.use_started for u in ups + ups2) == 1
    assert m.in_use_active


def test_held_phone_with_eyes_on_road_is_held_not_in_use():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 5.0, t0=0.0, phone_boxes=[LAP_PHONE], hand_boxes=[LAP_HAND], gaze=LOOK_ROAD)
    assert ups[-1].assessment.state == HELD
    assert not any(u.use_started for u in ups)


def test_phone_at_the_ear_is_near_face_without_a_hand():
    m = PhoneInteraction()
    ear_phone = (355.0, 210.0, 385.0, 270.0)
    ups, _ = _run_phone(m, 3.0, t0=0.0, phone_boxes=[ear_phone], gaze=LOOK_ROAD)
    assert ups[-1].assessment.state == NEAR_FACE


def test_moving_phone_with_no_hand_found_is_unknown_not_safe():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 3.0, t0=0.0, phone_boxes=[LAP_PHONE], gaze=LOOK_DOWN, jitter=15.0)
    assert ups[-1].assessment.state == UNKNOWN
    assert "no hand" in ups[-1].assessment.reason


def test_hands_unavailable_is_unknown():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 1.0, t0=0.0, phone_boxes=[LAP_PHONE], hands_available=False, jitter=15.0)
    assert ups[-1].assessment.state == UNKNOWN and "hand model" in ups[-1].assessment.reason


def test_detector_failure_and_stale_results_are_unknown_not_none():
    m = PhoneInteraction()
    _run_phone(m, 1.0, t0=0.0, phone_boxes=[])
    assert m.assessment.state == NONE
    up = m.update(PhoneFrame(t=1.0, detector=DETECTOR_FAILED, detector_reason="hailo timeout"))
    assert up.assessment.state == UNKNOWN and "hailo" in up.assessment.reason
    m2 = PhoneInteraction()
    _run_phone(m2, 0.5, t0=0.0, phone_boxes=[])
    up = m2.update(PhoneFrame(t=2.0, detection=None, face_box=FACE, gaze=LOOK_ROAD))  # no tick for 1.55 s
    assert up.assessment.state == UNKNOWN and "stale" in up.assessment.reason


def test_face_unavailable_with_a_phone_present_is_unknown():
    m = PhoneInteraction()
    ups, _ = _run_phone(m, 1.0, t0=0.0, phone_boxes=[LAP_PHONE], hand_boxes=[LAP_HAND], face=None)
    assert ups[-1].assessment.state == UNKNOWN and "face" in ups[-1].assessment.reason


def test_one_frame_phone_flicker_never_becomes_present():
    m = PhoneInteraction()
    m.update(PhoneFrame(t=0.0, detection=PhoneDetection(t=0.0, phone_boxes=(MOUNT_PHONE,)), face_box=FACE))
    for i in range(1, 10):
        up = m.update(PhoneFrame(t=i * STEP, detection=PhoneDetection(t=i * STEP), face_box=FACE))
        assert not up.presence_started
    assert m.assessment.state == NONE


def test_stale_frames_do_not_accrue_phone_use_time():
    m = PhoneInteraction()
    _run_phone(m, 0.5, t0=0.0, phone_boxes=[LAP_PHONE], hand_boxes=[LAP_HAND], gaze=LOOK_DOWN)
    started = []
    for i in range(10):  # ten frames 0.9 s apart: the detector stays fresh but each gap is a stall
        t = 1.0 + i * 0.9
        det = PhoneDetection(t=t, phone_boxes=(LAP_PHONE,), hand_boxes=(LAP_HAND,))
        started.append(m.update(PhoneFrame(t=t, detection=det, face_box=FACE, gaze=LOOK_DOWN)).use_started)
    assert not any(started)
    assert m.toward_s < 0.5


def test_gaze_toward_uses_direction_in_the_image_plane():
    from core.behavior.phone import PhoneConfig

    c = PhoneConfig()
    assert gaze_toward((0.0, -30.0), LAP_PHONE, FACE, c)  # phone below, looking down
    assert not gaze_toward((40.0, 0.0), LAP_PHONE, FACE, c)  # looking sideways
    assert gaze_toward((35.0, -5.0), MOUNT_PHONE, FACE, c)  # phone at frame right, +yaw
    assert not gaze_toward((-35.0, -5.0), MOUNT_PHONE, FACE, c)

