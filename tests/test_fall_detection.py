"""Tests for fall_detection.py's actual fall-trigger state machine: rapid
drop, pose-loss-during-the-fall recovery, posture corroboration, and the
resolve/cooldown lifecycle -- the behavior test_fall_detection_zones.py does
NOT cover (that file exercises only zones/stillness on a static pose).

Same fake-pose approach as test_fall_detection_zones.py: `module._pose` is
replaced with a fake whose `.process()` returns exactly the landmarks a test
wants, isolating the state machine under test from MediaPipe inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import modules.fall_detection as fall_detection_module
from modules.fall_detection import FallDetectionModule


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


def _standing(x: float = 0.3, y_hip: float = 0.3) -> list[_Landmark]:
    """A 33-landmark standing pose (legs extended, upright torso)."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    marks[0] = _Landmark(x, y_hip - 0.35)                                    # nose
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.25), _Landmark(x + 0.05, y_hip - 0.25)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.2), _Landmark(x + 0.05, y_hip + 0.2)
    marks[27] = _Landmark(x - 0.05, y_hip + 0.4)
    marks[28] = _Landmark(x + 0.05, y_hip + 0.4)
    return marks


def _sitting(x: float = 0.3, y_hip: float = 0.5) -> list[_Landmark]:
    """Upright torso, legs folded toward the camera (posture.classify's
    "sitting" case: leg_span/torso_span < 0.85)."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    marks[0] = _Landmark(x, y_hip - 0.25)
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.2), _Landmark(x + 0.05, y_hip - 0.2)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.1), _Landmark(x + 0.05, y_hip + 0.1)
    marks[27] = _Landmark(x - 0.05, y_hip + 0.15)
    marks[28] = _Landmark(x + 0.05, y_hip + 0.15)
    return marks


def _lying(x: float = 0.3, y_center: float = 0.85) -> list[_Landmark]:
    """Horizontal torso, low in frame (posture.classify's "lying_down" case:
    shoulder->hip axis close to horizontal, i.e. small y-delta / large x-delta)."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    marks[0] = _Landmark(x - 0.4, y_center)
    marks[11], marks[12] = _Landmark(x - 0.2, y_center - 0.02), _Landmark(x - 0.2, y_center + 0.02)
    marks[23], marks[24] = _Landmark(x + 0.2, y_center - 0.02), _Landmark(x + 0.2, y_center + 0.02)
    marks[25], marks[26] = _Landmark(x + 0.35, y_center - 0.02), _Landmark(x + 0.35, y_center + 0.02)
    marks[27] = _Landmark(x + 0.5, y_center - 0.02)
    marks[28] = _Landmark(x + 0.5, y_center + 0.02)
    return marks


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def module(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(fall_detection_module.time, "monotonic", lambda: t["now"])
    mod = FallDetectionModule({
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5, "stillness_velocity_threshold": 0.02,
        "still_max_motion": 0.01, "unusual_stillness_s": 600.0,
        "fall_pose_recovery_max_gap_s": 2.5, "fall_upright_recency_s": 0.5,
        "fall_pending_timeout_s": 6.0, "fall_resolve_after_s": 2.0, "fall_cooldown_s": 5.0,
    })
    mod._pose = _FakePose()
    mod._t = t
    return mod


def _feed(module, landmarks, frames: int, dt: float = 0.1) -> list:
    """Advance `frames` frames with a fixed set of landmarks (or None)."""
    events = []
    module._pose.next_landmarks = landmarks
    for _ in range(frames):
        events += module.process(FRAME)
        module._t["now"] += dt
    return events


def _feed_sequence(module, landmark_frames, dt: float = 0.1) -> list:
    """Advance one frame per entry in `landmark_frames` (each may be a
    landmark list or None), for building an actual multi-frame trajectory."""
    events = []
    for landmarks in landmark_frames:
        module._pose.next_landmarks = landmarks
        events += module.process(FRAME)
        module._t["now"] += dt
    return events


def _interpolated_fall(steps: int = 20):
    """A SLOW, continuous standing -> lying transition: every landmark is
    linearly interpolated between a standing pose and a lying pose over
    `steps` frames, so per-step displacement (and therefore velocity) stays
    small throughout even though the net change is large -- the way an
    intentional lie-down (or a slow, controlled movement to the floor)
    actually looks, as opposed to an abrupt single-frame jump."""
    start, end = _standing(), _lying()
    frames = []
    for i in range(steps + 1):
        frac = i / steps
        frames.append([
            _Landmark(
                a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac,
                visibility=max(a.visibility, b.visibility),
            )
            for a, b in zip(start, end, strict=True)
        ])
    return frames


# --------------------------------------------------------------------- true positives
def test_rapid_fall_then_stillness_confirms_a_high_confidence_fall(module):
    events = _feed(module, _standing(), 5)  # establish "recently upright"
    events += _feed(module, _lying(), 1)  # one-frame violent drop: crosses the velocity threshold
    events += _feed(module, _lying(), 20)  # remains down and still

    types = [e.type for e in events]
    assert "fall" in types
    fired = next(e for e in events if e.type == "fall")
    assert fired.confidence >= 0.6
    assert "measured rapid drop" in fired.contributing_signals
    assert module.is_alert_active()


def test_pose_lost_mid_fall_still_confirms(module):
    """The headline regression case: MediaPipe loses lock entirely during the
    violent part of the fall (motion blur/extreme pose) and only reacquires
    once the person is already down -- must still confirm, which the
    original velocity-only implementation could not do (see module
    docstring's missed-fall cause #1)."""
    events = _feed(module, _standing(), 5)  # establish "recently upright"
    events += _feed(module, None, 3)  # tracking drops out entirely for 0.3s -- the fall itself
    events += _feed(module, _lying(), 20)  # reacquired already on the floor, and stays there

    types = [e.type for e in events]
    assert "fall" in types
    fired = next(e for e in events if e.type == "fall")
    assert "tracking-loss recovery" in fired.contributing_signals
    assert module.is_alert_active()


def test_fall_confirms_once_not_every_frame_it_stays_down(module):
    events = _feed(module, _standing(), 5)
    events += _feed(module, _lying(), 1)
    events += _feed(module, _lying(), 40)  # stays down well past confirmation
    assert [e.type for e in events].count("fall") == 1


def test_getting_up_quickly_resolves_and_does_not_repeat(module):
    events = _feed(module, _standing(), 5)
    events += _feed(module, _lying(), 1)
    events += _feed(module, _lying(), 20)  # confirms
    assert "fall" in [e.type for e in events]
    assert module.is_alert_active()

    # Gets back up and stays up long enough to resolve (fall_resolve_after_s=2.0).
    events2 = _feed(module, _standing(), 30)
    assert "fall_resolved" in [e.type for e in events2]
    assert not module.is_alert_active()
    assert [e.type for e in (events + events2)].count("fall") == 1


# -------------------------------------------------------------------- false positives
def test_slow_intentional_lie_down_never_confirms(module):
    events = _feed_sequence(module, _interpolated_fall(steps=20))
    events += _feed(module, _lying(y_center=0.85), 20)  # then rests, still, in bed
    assert "fall" not in [e.type for e in events]


def test_sitting_down_quickly_does_not_confirm(module):
    """A fast hip drop that ends in a SITTING posture (legs folded, torso
    still upright) must not read as a fall -- only a genuinely horizontal
    end state should."""
    events = _feed(module, _standing(), 5)
    events += _feed(module, _sitting(), 1)  # fast drop in hip height
    events += _feed(module, _sitting(), 20)  # stays seated and still
    assert "fall" not in [e.type for e in events]


def test_bending_over_and_recovering_does_not_confirm(module):
    """A quick bend that never reads as lying_down and is followed by a
    confident return to standing must not linger and confirm later."""
    events = _feed(module, _standing(), 5)
    events += _feed(module, _sitting(), 1)  # closest available "leaning-ish" fast dip
    events += _feed(module, _standing(), 5)  # immediately back up
    events += _feed(module, _standing(), 20)  # stands still afterward (reading, etc.)
    assert "fall" not in [e.type for e in events]


def test_ordinary_standing_still_never_confirms_a_fall(module):
    """No drop, no gap, no pending state -- just an ordinary period of
    standing still (e.g. at a counter) -- must never be mistaken for a fall."""
    events = _feed(module, _standing(), 40)
    assert "fall" not in [e.type for e in events]


def test_lying_in_bed_from_the_start_never_confirms(module):
    """No prior upright context at all (module starts fresh with the subject
    already lying down, e.g. already asleep when the camera boots) must not
    itself read as a fall -- there is no "was upright" trigger to fire off."""
    events = _feed(module, _lying(), 30)
    assert "fall" not in [e.type for e in events]


def test_get_debug_state_reports_the_required_fields(module):
    events = _feed(module, _standing(), 5)
    events += _feed(module, _lying(), 1)
    events += _feed(module, _lying(), 20)
    assert "fall" in [e.type for e in events]

    state = module.get_debug_state()
    assert state is not None
    assert state["posture"] == "lying_down"
    assert state["fall_trigger"] == "measured_rapid_drop"
    assert state["fall_pending"] is True
    assert 0.0 <= state["fall_confidence"] <= 1.0
    assert "fall" in state["active_alerts"]
