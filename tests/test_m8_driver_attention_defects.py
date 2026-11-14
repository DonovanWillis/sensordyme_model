"""Mission 8 driver-attention defects, reproduced and fixed in the shipped path.

SYNTHETIC SCENARIOS ONLY: the real DriverAttentionModule (calibration, trackers,
fusion, event emission) with FaceMesh, the detector and MediaPipe Hands replaced
by geometric stand-ins (tests/test_behavior_integration.py's harness). Passing
says the logic does what it is designed to on constructed inputs, not that it
is accurate in a real cabin.

  D1  phone_use needed no hand: a phone anywhere + eyes off road was phone use.
  D2  looking down lowers the lids enough to read as closure: drowsiness fired
      (the critical alarm) for a look into the lap.
  D3  each camera stall credited max_evidence_gap_s (0.5 s) of evidence it never
      observed, so closures seen in short bursts fired early.
"""

from __future__ import annotations

import pytest
from test_behavior_integration import (
    DRIVER_BOX,
    LAP_HAND,
    LAP_PHONE,
    STEP,
    _FakeHands,
    _make,
    _types,
    clock,  # noqa: F401 - fixture
    landmarks,
    yolo,  # noqa: F401 - fixture
)
from test_driver_attention_scenarios import _FakeFaceMeshResult

from modules.driver_attention import (
    PHONE_HELD,
    PHONE_HOLD_UNKNOWN,
    PHONE_NOT_HELD,
    PhoneHoldEvidence,
    StallAwareEvidenceClock,
)

#: A phone in a dash/windscreen mount IN FRONT of the driver: inside the driver's
#: person box (DRIVER_BOX x 200..460), below the face, nobody holding it.
MOUNTED_IN_VIEW = (360, 330, 410, 400)
#: Both hands on the wheel, visible to the hand model, nowhere near that phone.
WHEEL_HANDS = [(215, 405, 275, 460), (395, 415, 455, 465)]

LOOK_DOWN = {"pitch": -30.0}


def _run(module, frame, clock, seconds, lms):  # noqa: F811 - clock is the fixture value
    module._face_mesh.process = lambda rgb, lm=lms: _FakeFaceMeshResult(lm)
    events = []
    for _ in range(max(1, round(seconds / STEP))):
        events += module.process(frame)
        clock["now"] += STEP
    return events


# ============================================================ D1 phone held vs present
def test_d1_a_mounted_phone_with_hands_on_the_wheel_is_not_phone_use(tmp_path, clock, yolo):  # noqa: F811
    """Repro (pre-fix): phone_use fired at 2.0 s for a glance into the lap with a
    phone sitting in a mount. Now: eyes_down (the real finding) fires, phone_use does not."""
    yolo.phones = [MOUNTED_IN_VIEW]
    hands = _FakeHands()
    hands.boxes = list(WHEEL_HANDS)
    module, frame = _make(tmp_path, clock, v2=False, hands=hands)
    _run(module, frame, clock, 3.0, landmarks())  # the phone sits still while the driver watches the road
    assert module._phone_hold.state == PHONE_NOT_HELD
    events = _run(module, frame, clock, 4.0, landmarks(**LOOK_DOWN))
    assert "eyes_down" in _types(events)
    assert "phone_use" not in _types(events)
    assert module.get_telemetry_snapshot()["phone_visible"] is True, "still seen and logged, just not 'in use'"


def test_d1_a_held_phone_in_the_lap_is_phone_use_and_says_a_hand_holds_it(tmp_path, clock, yolo):  # noqa: F811
    yolo.phones = [LAP_PHONE]
    hands = _FakeHands()
    hands.boxes = [LAP_HAND]
    module, frame = _make(tmp_path, clock, v2=False, hands=hands)
    events = _run(module, frame, clock, 3.0, landmarks(**LOOK_DOWN))
    phone = [e for e in events if e.type == "phone_use"]
    assert len(phone) == 1 and phone[0].contributing_signals == "hand overlapping the phone (experimental hand tracking)"
    assert phone[0].duration == pytest.approx(2.0, abs=0.1), "the configured phone_use_seconds"
    assert module._phone_hold.state == PHONE_HELD and module.is_alert_active()


class _FlickeringGrip(_FakeHands):
    """The hand model finds the gripping hand on one detector tick in three; the
    other hand (on the wheel) every tick -- MediaPipe loses a gripping hand often."""

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def process(self, rgb):
        self.ticks += 1
        self.boxes = [WHEEL_HANDS[0]] + ([LAP_HAND] if self.ticks % 3 == 0 else [])
        return super().process(rgb)


def test_d1_a_grip_seen_on_one_tick_in_three_still_counts_as_held(tmp_path, clock, yolo):  # noqa: F811
    yolo.phones = [LAP_PHONE]
    module, frame = _make(tmp_path, clock, v2=False, hands=_FlickeringGrip())
    events = _run(module, frame, clock, 3.0, landmarks(**LOOK_DOWN))
    assert _types(events).count("phone_use") == 1
    assert module._phone_hold.state == PHONE_HELD


def test_d1_without_hand_tracking_a_still_mount_phone_leaves_the_look_to_eyes_down(
        tmp_path, clock, yolo):  # noqa: F811
    """M8.5 (owner-requested; was: phone_use, "judged as before"). Hand tracking
    off: holding is unobserved, the phone has not moved and sits neither at the
    face nor in the lap zone, so it is not handled -- the look is eyes_down."""
    yolo.phones = [MOUNTED_IN_VIEW]
    module, frame = _make(tmp_path, clock, v2=False, enable_hands=False)
    module._hands = None  # enable_hands: false -- the owner turned hand tracking off
    _run(module, frame, clock, 3.0, landmarks())
    events = _run(module, frame, clock, 3.0, landmarks(**LOOK_DOWN))
    assert "phone_use" not in _types(events)
    assert "eyes_down" in _types(events)
    assert module._phone_hold.state == PHONE_HOLD_UNKNOWN


def test_d1_no_hand_found_anywhere_on_a_still_mount_phone_leaves_the_look_to_eyes_down(
        tmp_path, clock, yolo):  # noqa: F811
    """M8.5 (owner-requested; was: phone_use). The hand model ran and found
    nothing: still weak evidence of "no hand", so the phone is never called set
    down on it -- but a still phone outside the face / lap zone shows no handling
    either, and the look is judged by eyes_down, not phone_use."""
    yolo.phones = [MOUNTED_IN_VIEW]
    module, frame = _make(tmp_path, clock, v2=False, hands=_FakeHands())
    _run(module, frame, clock, 3.0, landmarks())
    events = _run(module, frame, clock, 3.0, landmarks(**LOOK_DOWN))
    assert "phone_use" not in _types(events)
    assert "eyes_down" in _types(events)
    assert module._phone_hold.state == PHONE_HOLD_UNKNOWN


def test_d1_no_hand_found_on_a_lap_phone_under_a_downward_gaze_is_still_phone_use(
        tmp_path, clock, yolo):  # noqa: F811
    """Recall kept where the evidence supports handling: a phone in the lap zone
    (below the chin, inside the face's span) looked down at, no hand found."""
    yolo.phones = [LAP_PHONE]
    module, frame = _make(tmp_path, clock, v2=False, hands=_FakeHands())
    _run(module, frame, clock, 3.0, landmarks())
    events = _run(module, frame, clock, 3.0, landmarks(**LOOK_DOWN))
    phone = [e for e in events if e.type == "phone_use"]
    assert len(phone) == 1
    assert phone[0].contributing_signals.startswith("phone visible; no hand confirmed on it; possible handling")
    assert "unvalidated" in phone[0].contributing_signals


def test_d1_hold_evidence_unlatches_when_the_phone_moves_or_a_hand_reaches_it():
    hold = PhoneHoldEvidence(hold_ticks=3, static_s=2.0)
    phone = [(100, 100, 150, 180)]
    hands = [(300, 300, 360, 360)]
    for i in range(25):  # 2.5 s of ticks, still, hands elsewhere
        hold.update(i * 0.1, phone, hands, hands_ran=True, overlap=False)
    assert hold.state == PHONE_NOT_HELD
    moved = [(160, 100, 210, 180)]
    assert hold.update(2.6, moved, hands, hands_ran=True, overlap=False) == PHONE_HOLD_UNKNOWN
    assert hold.update(2.7, moved, hands, hands_ran=True, overlap=True) == PHONE_HELD
    assert hold.update(2.8, [], hands, hands_ran=True, overlap=False) == "none"


# ============================================================ D2 looking down vs closing
def test_d2_a_look_into_the_lap_that_lowers_the_lids_is_eyes_down_not_drowsiness(
        tmp_path, clock, yolo):  # noqa: F811
    """Repro (pre-fix): lids lowered to EAR 0.27 (0.67 x open) by a 30 degree look
    down crossed the 0.75 x closure threshold: drowsiness at 2.0 s, before eyes_down."""
    module, frame = _make(tmp_path, clock, v2=False)
    events = _run(module, frame, clock, 3.0, landmarks(pitch=-30.0, lid=4.0))
    assert "drowsiness" not in _types(events)
    assert "eyes_down" in _types(events)
    assert module.perception_signals()["eye_state"].closed is False, "reported as open, looking down"


def test_d2_lids_over_the_iris_with_the_head_down_are_still_drowsiness(tmp_path, clock, yolo):  # noqa: F811
    module, frame = _make(tmp_path, clock, v2=False)
    events = _run(module, frame, clock, 3.0, landmarks(pitch=-20.0, lid=1.0))
    assert _types(events).count("drowsiness") == 1, "a nodding-off head pitches down too"


def test_d2_a_partial_closure_with_the_head_level_is_still_drowsiness(tmp_path, clock, yolo):  # noqa: F811
    module, frame = _make(tmp_path, clock, v2=False)
    events = _run(module, frame, clock, 3.0, landmarks(lid=4.0))
    assert _types(events).count("drowsiness") == 1


# ============================================================ D3 stalls are not evidence
def test_d3_closures_seen_in_bursts_between_stalls_need_real_observed_closure(
        tmp_path, clock, yolo):  # noqa: F811
    """Repro (pre-fix): 10-frame bursts (0.45 s observed) separated by 3 s camera
    stalls fired drowsiness after 1.1 s of observed closure (measured), each stall adding
    0.5 s. Now a stall credits one frame interval: it fires on ~2 s observed."""
    module, frame = _make(tmp_path, clock, v2=False)
    shut = landmarks(lid=1.0)
    module._face_mesh.process = lambda rgb, lm=shut: _FakeFaceMeshResult(lm)
    observed, fired_after = 0.0, None
    for _burst in range(8):
        for i in range(10):
            if "drowsiness" in _types(module.process(frame)) and fired_after is None:
                fired_after = observed
            clock["now"] += STEP
            if i < 9:
                observed += STEP
        clock["now"] += 3.0  # the camera stalls
    assert fired_after is not None, "a real closure still fires"
    assert fired_after >= 2.0 - 0.3, f"fired after {fired_after:.2f} s of observed closure"
    assert module._evidence_clock.gaps >= 4 and module._evidence_clock.unobserved_s > 4 * 2.9


def test_d3_a_steadily_slow_camera_still_alerts(tmp_path, clock, yolo):  # noqa: F811
    """Every interval above the gap cap (0.6 s frames throughout): the stream's own
    interval is the cap, so evidence still accrues (later in wall time), never stops."""
    module, frame = _make(tmp_path, clock, v2=False)
    shut = landmarks(lid=1.0)
    module._face_mesh.process = lambda rgb, lm=shut: _FakeFaceMeshResult(lm)
    fired = []
    for _ in range(40):
        fired += _types(module.process(frame))
        clock["now"] += 0.6
    assert "drowsiness" in fired


def test_d3_the_clock_credits_one_frame_interval_for_a_stall():
    evidence = StallAwareEvidenceClock(max_gap_s=0.5)
    t = 0.0
    for _ in range(20):
        evidence.advance(t)
        t += 0.05
    before = evidence.now
    assert evidence.advance(t + 4.0) == pytest.approx(before + 0.05), "a 4 s stall is one frame"
    assert evidence.gaps == 1 and evidence.unobserved_s == pytest.approx(4.05 - 0.05)
    t += 4.0
    last = evidence.now
    for _ in range(30):  # then a stream that is slow throughout
        t += 1.0
        last = evidence.now
        evidence.advance(t)
    assert evidence.now - last == pytest.approx(0.5), "a slow stream keeps the base cap"


def test_d3_ordinary_frame_intervals_are_credited_unchanged():
    evidence = StallAwareEvidenceClock(max_gap_s=0.5)
    for t in (0.0, 0.033, 0.07, 0.1, 0.45, 0.9):
        evidence.advance(t)
    assert evidence.now == pytest.approx(0.9) and evidence.gaps == 0


def test_d1_the_driver_box_gate_needs_a_person_box_to_relate_to():
    from modules.driver_attention import phones_for_driver

    inside, outside = (300, 300, 340, 360), (20, 300, 60, 360)
    assert phones_for_driver([inside, outside], [DRIVER_BOX]) == [inside]
    assert phones_for_driver([inside, outside], []) == [inside, outside]
