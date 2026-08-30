"""Phase 1 safety regression tests for modules/fall_detection.py.

Every test here is a MISSED-DETECTION case: a real fall (or a real
unusual-stillness episode) that the shipped state machine silently swallowed.
The existing test_fall_detection.py suite is almost entirely negative
assertions ("does not confirm"), which is exactly how these regressions got
in -- so each test below asserts that the alert DOES fire under the adverse
condition, and is paired with the false-positive case it must not drag back
in with it.

Same fake-pose approach as test_fall_detection.py / test_fall_detection_zones.py:
`module._pose` is replaced so the state machine is exercised without MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import modules.fall_detection as fall_detection_module
from modules.fall_detection import FallDetectionModule

FRAME_H, FRAME_W = 480, 640
FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


@dataclass
class _Landmark:
    x: float
    y: float
    visibility: float = 1.0


class _FakeLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _FakeResults:
    def __init__(self, landmarks):
        self.pose_landmarks = _FakeLandmarkList(landmarks) if landmarks is not None else None


class _FakePose:
    def __init__(self):
        self.next_landmarks = None

    def process(self, rgb):
        return _FakeResults(self.next_landmarks)


def _blank() -> list[_Landmark]:
    return [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]


def _standing(x: float = 0.3, y_hip: float = 0.3) -> list[_Landmark]:
    """Fully visible upright pose -- posture.classify reads STANDING @ 0.8."""
    marks = _blank()
    marks[0] = _Landmark(x, y_hip - 0.35)
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.25), _Landmark(x + 0.05, y_hip - 0.25)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.2), _Landmark(x + 0.05, y_hip + 0.2)
    marks[27], marks[28] = _Landmark(x - 0.05, y_hip + 0.4), _Landmark(x + 0.05, y_hip + 0.4)
    return marks


def _upright_legs_occluded(x: float = 0.3, y_hip: float = 0.975) -> list[_Landmark]:
    """Near-vertical torso with knees AND ankles invisible.

    This is not a contrived pose: core/posture.py:128-131 returns exactly
    Posture(STANDING, 0.5, "upright; lower body not visible") for it, and it
    is what a camera actually sees mid-fall behind a sofa/bed, or for a fall
    along the camera axis (the torso stays near-vertical in projection until
    the body settles). The default y_hip puts the torso low in frame, i.e.
    the person is already most of the way to the floor.
    """
    marks = _blank()
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.25), _Landmark(x + 0.05, y_hip - 0.25)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    return marks


def _lying(x: float = 0.3, y_center: float = 0.85) -> list[_Landmark]:
    """Horizontal torso low in frame -- posture.classify reads LYING_DOWN."""
    marks = _blank()
    marks[0] = _Landmark(x - 0.4, y_center)
    marks[11], marks[12] = _Landmark(x - 0.2, y_center - 0.02), _Landmark(x - 0.2, y_center + 0.02)
    marks[23], marks[24] = _Landmark(x + 0.2, y_center - 0.02), _Landmark(x + 0.2, y_center + 0.02)
    marks[25], marks[26] = _Landmark(x + 0.35, y_center - 0.02), _Landmark(x + 0.35, y_center + 0.02)
    marks[27], marks[28] = _Landmark(x + 0.5, y_center - 0.02), _Landmark(x + 0.5, y_center + 0.02)
    return marks


def _config(**overrides) -> dict:
    """The SHIPPED config.yaml fall_detection values, so these tests measure
    the product's real behavior rather than a permissive test-only tuning."""
    config = {
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5,
        "stillness_velocity_threshold": 0.02, "still_max_motion": 0.01,
        "unusual_stillness_s": 600.0,
        "fall_pose_recovery_max_gap_s": 2.5, "fall_upright_recency_s": 0.5,
        "fall_pending_timeout_s": 6.0, "fall_resolve_after_s": 4.0,
    }
    config.update(overrides)
    return config


@pytest.fixture()
def make_module(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(fall_detection_module.time, "monotonic", lambda: t["now"])

    def _make(**overrides):
        mod = FallDetectionModule(_config(**overrides))
        mod._pose = _FakePose()
        mod._t = t
        return mod

    return _make


def _feed(module, landmarks, frames: int, dt: float = 0.1) -> list:
    events = []
    module._pose.next_landmarks = landmarks
    for _ in range(frames):
        events += module.process(FRAME)
        module._t["now"] += dt
    return events


def _feed_sequence(module, landmark_frames, dt: float = 0.1) -> list:
    events = []
    for landmarks in landmark_frames:
        module._pose.next_landmarks = landmarks
        events += module.process(FRAME)
        module._t["now"] += dt
    return events


def _drop_and_settle(module, drop_pose=None, settle_frames: int = 25) -> list:
    """Establish "recently upright", take one violent single-frame drop, then
    stay down and motionless -- the canonical confirmable fall."""
    events = _feed(module, _standing(), 5)
    events += _feed(module, drop_pose if drop_pose is not None else _lying(), 1)
    events += _feed(module, _lying(), settle_frames)
    return events


# ------------------------------------------------------------------ BUG 1
def test_fall_with_occluded_lower_body_is_still_detected(make_module):
    """A violent drop whose peak-velocity frame reads as core/posture.py's
    "upright; lower body not visible" sentinel (STANDING @ exactly 0.5) --
    a fall behind a sofa, or along the camera axis -- must still confirm.

    Before the fix the cancellation branch fired in the SAME process() call
    that set the pending state (0.5 >= 0.5 passed the confidence gate), so
    the pending window was erased on the drop frame itself and nothing could
    ever confirm afterwards, no matter how violent the drop or how long the
    person then lay motionless.
    """
    module = make_module()
    events = _feed(module, _standing(), 5)
    events += _feed(module, _upright_legs_occluded(), 1)  # ~11x the drop threshold
    events += _feed(module, _lying(), 25)  # settles horizontal, motionless

    assert "fall" in [e.type for e in events]
    assert module.is_alert_active()


def test_confident_full_body_standing_still_cancels_a_pending_fall(make_module):
    """The other half of BUG 1's fix: a CONFIDENT return to standing (full
    body visible, posture confidence 0.8) must still abandon a pending fall,
    so tightening the gate above did not simply disable the cancellation."""
    module = make_module()
    events = _feed(module, _standing(), 5)
    events += _feed(module, _upright_legs_occluded(), 1)  # pending opens
    events += _feed(module, _standing(y_hip=0.3), 3)  # caught themselves, fully visible
    events += _feed(module, _standing(y_hip=0.3), 40)  # stands still afterwards
    assert "fall" not in [e.type for e in events]
    assert module._fall_pending_since is None


# ------------------------------------------------------------------ BUG 2
def test_fall_followed_by_a_long_struggle_then_stillness_is_detected(make_module):
    """Someone who falls, tries to get up for ten seconds, then stops moving.

    Before the fix `fall_pending_timeout_s` counted WALL CLOCK from the drop
    with no re-entry path, so struggling for longer than 6s guaranteed the
    pending state was thrown away before the stillness that would have
    confirmed it -- detection probability inverted against severity (a 2s
    struggle was detected, a 10s one never was).
    """
    module = make_module()
    events = _feed(module, _standing(), 5)
    events += _feed(module, _lying(), 1)  # the drop itself

    # 10 seconds of struggling: still down (LYING_DOWN every frame), but
    # moving far too much to count as "still" (240 px/s vs the 9.6 px/s
    # stillness bar), and never fast enough to re-trigger the drop
    # threshold (288 px/s) and open a fresh pending window.
    struggle = []
    for i in range(100):
        struggle.append(_lying(y_center=0.85 if i % 2 == 0 else 0.80))
    events += _feed_sequence(module, struggle)
    assert "fall" not in [e.type for e in events], "must not confirm while still moving"

    events += _feed(module, _lying(), 25)  # loses consciousness, goes still
    assert "fall" in [e.type for e in events]
    assert module.is_alert_active()


def test_pending_fall_still_expires_when_the_person_is_not_down(make_module):
    """BUG 2's fix must not disable the backstop: a pending state opened by a
    velocity spike that is NOT followed by an observed lying-down posture
    must still be abandoned, so it cannot confirm against unrelated stillness
    minutes later."""
    module = make_module()
    events = _feed(module, _standing(), 5)
    events += _feed(module, _upright_legs_occluded(), 1)  # spike, never reads lying_down

    # 8s of an ambiguous, never-lying-down posture, well past the 6s backstop.
    events += _feed(module, _upright_legs_occluded(), 80)
    assert module._fall_pending_since is None
    assert "fall" not in [e.type for e in events]


# ------------------------------------------------------------------ BUG 3
def test_confirmed_fall_resolves_when_pose_tracking_is_lost(make_module):
    """A CONFIRMED fall must not latch forever when MediaPipe stops returning
    landmarks.

    The pose-loss early-return fed an explicit False to the zone/stillness
    engine but never touched the fall engine at all, and EventEngine only
    advances a CONFIRMED track on an explicit update -- so the alert stayed
    active indefinitely and every later real fall was absorbed into it
    silently.
    """
    module = make_module()
    events = _drop_and_settle(module)
    assert "fall" in [e.type for e in events]
    assert module.is_alert_active()

    events2 = _feed(module, None, 100)  # 10s with no pose at all
    assert "fall_resolved" in [e.type for e in events2]
    assert not module.is_alert_active()


def test_a_second_fall_after_a_pose_dropout_is_still_reported(make_module):
    """The consequence that actually harms someone: while the first fall was
    latched, a second real fall produced NO new alert."""
    module = make_module()
    events = _drop_and_settle(module)
    assert [e.type for e in events].count("fall") == 1

    _feed(module, None, 100)  # tracking lost long enough to resolve
    events2 = _drop_and_settle(module)  # an entirely separate, real fall
    assert "fall" in [e.type for e in events2]


def test_a_fall_that_happens_during_a_pose_dropout_is_still_detected(make_module):
    """BUG 3's fix must not break the tracking-loss recovery path: upright,
    tracking lost, reacquired lying down still has to confirm."""
    module = make_module()
    events = _feed(module, _standing(), 5)
    events += _feed(module, None, 3)  # the violent phase, unseen
    events += _feed(module, _lying(), 25)
    fired = [e for e in events if e.type == "fall"]
    assert fired
    assert "tracking-loss recovery" in fired[0].contributing_signals


# ------------------------------------------------------------------ BUG 7
def test_a_second_fall_within_the_old_cooldown_is_reported(make_module):
    """Repeat falls (syncope, a failed attempt to stand) are the clinically
    most alarming pattern. EventEngine's cooldown gate returns BEFORE a track
    is created, so a 20s fall cooldown meant the second fall never entered
    the lifecycle at all and produced exactly one alert for two falls.
    """
    module = make_module()
    events = _drop_and_settle(module)
    assert "fall" in [e.type for e in events]

    events += _feed(module, _standing(), 60)  # gets up; resolves after 4s
    assert "fall_resolved" in [e.type for e in events]

    events2 = _drop_and_settle(module)  # falls again, ~12s after the first
    assert "fall" in [e.type for e in events2]
    assert module.is_alert_active()


# ------------------------------------------------------------------ BUG 6
def test_unusual_stillness_does_not_fire_on_a_demonstrably_moving_subject(make_module):
    """occlusion_grace_s bridges ABSENCE without resetting the held-time
    clock -- but for a stillness rule "absent" means THE SUBJECT MOVED, which
    is contradicting evidence, not missing evidence. A subject moving for 1s
    out of every 3.5s for minutes on end must never read as unusually still.
    """
    module = make_module(unusual_stillness_s=10.0)
    frames = []
    cycle = 0
    while len(frames) < 600:  # 60s at dt=0.1
        # 1.0s of real, unambiguous movement (0.02 normalized per frame,
        # twice still_max_motion), then 2.5s of holding position.
        for i in range(10):
            frames.append(_standing(x=0.30 + 0.02 * i))
        frames += [_standing(x=0.48)] * 25
        cycle += 1
    events = _feed_sequence(module, frames)
    assert "unusual_stillness" not in [e.type for e in events]


def test_unusual_stillness_still_fires_through_a_brief_pose_dropout(make_module):
    """The paired positive: a genuinely motionless person (the unconscious/
    collapsed case this rule exists for) must still confirm even though pose
    tracking blinks out for a moment -- BUG 6's fix must reset stillness on
    observed MOTION only, never on missing evidence."""
    module = make_module(unusual_stillness_s=5.0)
    events = _feed(module, _standing(x=0.5), 20)  # 2s motionless
    events += _feed(module, None, 5)  # 0.5s dropout: no evidence either way
    events += _feed(module, _standing(x=0.5), 60)  # motionless again
    assert "unusual_stillness" in [e.type for e in events]
