"""Unit tests for core/behavior/ components (synthetic scenarios, deterministic time).

Every scenario here is SYNTHETIC: hand-built boxes, angles and timestamps.
Passing them shows the logic does what it says; it is not a measured accuracy.
"""

from __future__ import annotations

import numpy as np

from core.behavior.evidence import (
    UNKNOWN,
    Assessment,
)
from core.behavior.eye_state import (
    CLOSED,
    LOOKING_DOWN,
    OPEN,
    DrowsinessEvidence,
    EyeFrame,
    EyeObservation,
    classify_eye_state,
    eye_patch_occluded,
)

STEP = 0.05



def _eyes(ear_left, ear_right=None, occluded=(False, False)):
    ear_right = ear_left if ear_right is None else ear_right
    return (EyeObservation(ear_left, occluded[0]), EyeObservation(ear_right, occluded[1]))


def test_eye_closure_vs_downward_glance():
    base = {"face_present": True, "closure_threshold": 0.30, "deep_threshold": 0.16}
    assert classify_eye_state(EyeFrame(eyes=_eyes(0.40), iris_openness=1.0, **base)).state == OPEN
    # lid lowered with the head pitched down and the iris still visible: looking down
    down = classify_eye_state(EyeFrame(eyes=_eyes(0.26), iris_openness=0.65, head_pitch_deg=-25.0, **base))
    assert down.state == LOOKING_DOWN
    # same lid drop, eyes cast down with the head level (iris downshift)
    assert classify_eye_state(EyeFrame(eyes=_eyes(0.26), iris_openness=0.65, head_pitch_deg=0.0,
                                       eye_pitch_deg=-12.0, **base)).state == LOOKING_DOWN
    # same lid drop, no downward gaze: a closure
    assert classify_eye_state(EyeFrame(eyes=_eyes(0.26), iris_openness=0.65, head_pitch_deg=0.0,
                                       eye_pitch_deg=0.0, **base)).state == CLOSED
    # lid over the iris while nodding off (head down): still a closure
    assert classify_eye_state(EyeFrame(eyes=_eyes(0.07), iris_openness=0.15, head_pitch_deg=-25.0,
                                       **base)).state == CLOSED


def test_occluded_eyes_are_unknown_and_one_occluded_eye_defers_to_the_other():
    base = {"face_present": True, "closure_threshold": 0.30, "iris_openness": 0.9}
    both = classify_eye_state(EyeFrame(eyes=_eyes(0.05, occluded=(True, True)), **base))
    assert both.state == UNKNOWN and "occluded" in both.reason
    one = classify_eye_state(EyeFrame(eyes=_eyes(0.05, 0.40, occluded=(True, False)), **base))
    assert one.state == OPEN
    assert classify_eye_state(EyeFrame(face_present=False)).state == UNKNOWN


def test_eye_patch_occlusion_heuristic():
    rng = np.random.default_rng(0)
    assert eye_patch_occluded(np.full((12, 30), 12.0), 40.0, 8.0) is True
    assert eye_patch_occluded(rng.integers(0, 255, (12, 30)).astype(float), 40.0, 8.0) is False
    assert eye_patch_occluded(np.zeros((0, 0)), 40.0, 8.0) is None


def _drowse(ev, t0, seconds, state):
    fired = []
    n = round(seconds / STEP)
    for i in range(n):
        r = ev.update(t0 + i * STEP, state)
        if r is not None:
            fired.append(r)
    return fired, t0 + n * STEP


CLOSED_A = Assessment(CLOSED)
OPEN_A = Assessment(OPEN)
LOST = Assessment(UNKNOWN, "face not detected")


def test_brief_face_loss_mid_closure_does_not_reset_drowsiness():
    ev = DrowsinessEvidence(sustain_s=2.0, coast_s=1.0)
    f1, t = _drowse(ev, 0.0, 1.2, CLOSED_A)
    f2, t = _drowse(ev, t, 0.5, LOST)
    assert ev.assessment.state == "coasting"
    f3, t = _drowse(ev, t, 1.2, CLOSED_A)
    assert len(f1 + f2 + f3) == 1


def test_long_face_loss_drops_the_evidence_and_reports_unknown():
    ev = DrowsinessEvidence(sustain_s=2.0, coast_s=1.0)
    _, t = _drowse(ev, 0.0, 1.2, CLOSED_A)
    _, t = _drowse(ev, t, 2.0, LOST)
    assert ev.assessment.state == UNKNOWN and ev.closed_s == 0.0
    f, _ = _drowse(ev, t, 1.2, CLOSED_A)
    assert f == []


def test_looking_down_is_not_drowsiness_and_open_eyes_reset():
    ev = DrowsinessEvidence(sustain_s=2.0)
    f, t = _drowse(ev, 0.0, 3.0, Assessment(LOOKING_DOWN, "x"))
    assert f == []
    _, t = _drowse(ev, t, 1.5, CLOSED_A)
    _, t = _drowse(ev, t, 0.5, OPEN_A)
    f, _ = _drowse(ev, t, 1.5, CLOSED_A)
    assert f == []


def test_stale_frames_do_not_accrue_closure():
    ev = DrowsinessEvidence(sustain_s=2.0, max_gap_s=0.5)
    assert all(ev.update(i * 1.0, CLOSED_A) is None for i in range(10))
    assert ev.closed_s == 0.0

