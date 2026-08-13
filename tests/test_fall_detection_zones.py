"""Elderly-mode zone and stillness detection, layered onto fall_detection.py's
existing pose landmarks (no extra model, no fabricated bounding box -- see
module docstrings for why).

MediaPipe Pose itself isn't exercised here (that's `test_skinnemoen_module.py`'s
approach, and it works because a blank frame genuinely finds no face/pose).
Finding a *specific* pose reliably needs a real photo, which this environment
doesn't have -- so `module._pose` is replaced with a fake whose `.process()`
returns exactly the landmarks a test wants, isolating the code actually under
test (zone containment + stillness thresholding) from MediaPipe inference.
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


def _standing(x: float = 0.3, y_hip: float = 0.5, ankles_visible: bool = True) -> list[_Landmark]:
    """A 33-landmark standing pose at a given hip position; everything else hidden."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    marks[0] = _Landmark(x, y_hip - 0.35)                                    # nose
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.25), _Landmark(x + 0.05, y_hip - 0.25)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.2), _Landmark(x + 0.05, y_hip + 0.2)
    ankle_vis = 1.0 if ankles_visible else 0.0
    marks[27] = _Landmark(x - 0.05, y_hip + 0.4, visibility=ankle_vis)
    marks[28] = _Landmark(x + 0.05, y_hip + 0.4, visibility=ankle_vis)
    return marks


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
# x in [0, 0.6], y in [0.6, 1.0] -- a standing pose at x=0.3 puts the ankles
# (y_hip + 0.4) at y=0.9, inside; x=0.9 puts them outside.
_ZONE_CONFIG = {"stairs": {"polygon": [[0.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]], "kind": "restricted"}}


@pytest.fixture()
def module(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(fall_detection_module.time, "monotonic", lambda: t["now"])
    mod = FallDetectionModule({
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5, "stillness_velocity_threshold": 0.02,
        "zones": _ZONE_CONFIG, "zone_confirm_s": 0.5,
        "still_max_motion": 0.01, "unusual_stillness_s": 1.0,
    })
    mod._pose = _FakePose()
    mod._t = t
    return mod


def _run(module, landmarks, frames: int, dt: float = 0.1) -> list:
    events = []
    module._pose.next_landmarks = landmarks
    for _ in range(frames):
        events += module.process(FRAME)
        module._t["now"] += dt
    return events


# ------------------------------------------------------------------- zones
def test_person_in_restricted_zone_confirms_zone_violation(module):
    events = _run(module, _standing(x=0.3), 10)
    assert "zone_violation" in [e.type for e in events]
    assert module.is_alert_active()


def test_person_outside_zone_does_not_confirm(module):
    events = _run(module, _standing(x=0.9), 10)
    assert "zone_violation" not in [e.type for e in events]


def test_invisible_ankles_skip_the_zone_check_without_crashing(module):
    """Occluded feet are not evidence the zone was left -- they're no evidence
    at all, which is different from a False."""
    events = _run(module, _standing(x=0.3, ankles_visible=False), 10)
    assert "zone_violation" not in [e.type for e in events]


def test_person_leaving_resolves_a_confirmed_zone_violation(module):
    events = _run(module, _standing(x=0.3), 10)
    assert "zone_violation" in [e.type for e in events]

    events = _run(module, None, 60)  # gone entirely, long enough to resolve
    assert "zone_violation_resolved" in [e.type for e in events]
    assert not module.is_alert_active()


# ------------------------------------------------------------ bed monitoring
_BED_ZONE_CONFIG = {"bed": {"polygon": [[0.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]], "kind": "safe"}}


@pytest.fixture()
def bed_module(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(fall_detection_module.time, "monotonic", lambda: t["now"])
    mod = FallDetectionModule({
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5, "stillness_velocity_threshold": 0.02,
        "zones": _BED_ZONE_CONFIG, "zone_confirm_s": 0.5,
        "still_max_motion": 0.01, "unusual_stillness_s": 600.0,
    })
    mod._pose = _FakePose()
    mod._t = t
    return mod


def test_staying_in_bed_does_not_confirm_safe_zone_exit(bed_module):
    events = _run(bed_module, _standing(x=0.3), 10)  # ankles at (0.3, 0.9) -- inside "bed"
    assert "safe_zone_exit" not in [e.type for e in events]


def test_getting_out_of_bed_confirms_safe_zone_exit(bed_module):
    events = _run(bed_module, _standing(x=0.9), 10)  # ankles outside "bed"
    assert "safe_zone_exit" in [e.type for e in events]
    assert bed_module.is_alert_active()


def test_leaving_the_frame_entirely_does_not_itself_confirm_safe_zone_exit(bed_module):
    """Matches baby_monitoring.py's convention: gone-entirely is a different
    concern from detected-but-outside-the-safe-zone."""
    events = _run(bed_module, None, 10)
    assert "safe_zone_exit" not in [e.type for e in events]


def test_returning_to_bed_resolves_a_confirmed_safe_zone_exit(bed_module):
    events = _run(bed_module, _standing(x=0.9), 10)
    assert "safe_zone_exit" in [e.type for e in events]

    events = _run(bed_module, _standing(x=0.3), 60)  # back in bed, long enough to resolve
    assert "safe_zone_exit_resolved" in [e.type for e in events]
    assert not bed_module.is_alert_active()


# --------------------------------------------------------------- stillness
def test_unusual_stillness_confirms_after_the_configured_duration(module):
    events = _run(module, _standing(x=0.9), 20)  # outside the zone; isolates stillness
    assert "unusual_stillness" in [e.type for e in events]


def test_ordinary_movement_does_not_confirm_stillness(module):
    events = []
    for i in range(20):
        events += _run(module, _standing(x=0.9 - i * 0.02), 1)
    assert "unusual_stillness" not in [e.type for e in events]


def test_person_leaving_resolves_stillness_too(module):
    events = _run(module, _standing(x=0.9), 20)
    assert "unusual_stillness" in [e.type for e in events]

    events = _run(module, None, 150)  # occlusion_grace_s=3.0 + resolve_after_s=10.0
    assert "unusual_stillness_resolved" in [e.type for e in events]


def test_close_releases_the_pose_graph_without_raising():
    """Found in a repo-wide audit: fall_detection is one of the two modules
    _apply_monitoring_mode/_switch_preset hot-swap between on every mode
    change, and close() was previously a no-op here (base class default),
    leaking a live MediaPipe Pose graph on every swap away from this module."""
    mod = FallDetectionModule({
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5, "stillness_velocity_threshold": 0.02,
    })
    mod.close()
    mod.close()  # closing twice must be harmless
