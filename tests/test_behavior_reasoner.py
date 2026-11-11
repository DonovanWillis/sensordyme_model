"""Unit tests for core/behavior/ components (synthetic scenarios, deterministic time).

Every scenario here is SYNTHETIC: hand-built boxes, angles and timestamps.
Passing them shows the logic does what it says; it is not a measured accuracy.
"""

from __future__ import annotations

from core.behavior.reasoner import BehaviorFrame, BehaviorReasoner

STEP = 0.05



def test_reasoner_reports_every_unknown_component_with_its_reason():
    r = BehaviorReasoner()
    r.update(BehaviorFrame(t=0.0, face_present=False, detector="not_started",
                           detector_reason="object detector not started"))
    u = r.uncertainty()
    assert set(u) == {"eye_state", "drowsiness", "glance", "phone"}
    assert u["eye_state"] == "face not detected"
    assert u["phone"] == "object detector not started"
    snap = r.snapshot()
    assert snap["behavior_uncertain"] == "drowsiness;eye_state;glance;phone"
