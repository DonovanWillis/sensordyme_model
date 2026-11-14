"""Mission 10 (temporal stream): defects in the shipped temporal reasoning, each
reproduced here BEFORE its fix (docs/TEMPORAL_REASONING.md has the before/after).

SYNTHETIC SCENARIOS ONLY: the real DriverAttentionModule with its shipped
configuration, fed constructed landmark timelines at injected frame times
(tests/m10_temporal_harness.py). Passing says the temporal logic does what it is
designed to on constructed inputs, not that it is accurate in a real cabin.

  T1  Gaze time nobody observed was credited: with the face found but the head-
      pose fit failing (solvePnP: no rotation), the distraction / eyes_down /
      eyes_up trackers were simply not updated, so the first readable frame
      afterwards was credited the WHOLE unreadable stretch -- distraction fired
      on that frame at full confidence, even when the driver had looked back at
      the road meanwhile. The same frames fed phone_use "eyes on the road", so a
      live phone_use alert dropped and re-fired on a 1 s pose failure.
  T2  PERCLOS charged time it never observed: a face loss between two open
      samples counted as open time (diluting a heavy-blinking driver's PERCLOS
      below the threshold) and between two closed samples as closed time (two
      blinks around a 20 s look away read as fatigue).
  T3  Drowsiness raised before calibration reported its margin confidence with
      no mark, while the same frame's eye_state said closure could not be judged
      (closed=None) without a calibrated threshold.
  T4  The shadow engine (core/behavior/engine.py) credits nothing across a frame
      gap -- correctly -- but then said NOT_SUPPORTED ("closed 0.0s") about eyes
      closed on every frame of a slow stream: "unobserved" silently became "no".
"""

from __future__ import annotations

from typing import Any

import pytest
from m10_temporal_harness import OPEN, SHUT, Seg, face, run, shipped_config
from test_behavior_integration import LAP_HAND, LAP_PHONE, _textured_frame, landmarks
from test_driver_attention_scenarios import _FakeFaceMeshResult
from test_m84_behavior_engine import _frame as engine_frame
from test_m84_behavior_engine import _run as engine_run

from core.behavior import signals as sig
from core.behavior.engine import DROWSINESS, BehaviorEngine
from modules.driver_attention import DriverAttentionModule, PerclosTracker

FPS = (30.0, 10.0, 5.0)


# ============================================================ T1 unreadable gaze is not evidence
@pytest.mark.parametrize("fps", FPS)
def test_t1_a_head_pose_failure_mid_turn_credits_no_time(tmp_path, monkeypatch, fps):
    """Repro (pre-fix): 1.5 s turned, 5 s unreadable pose (face found), then the
    turn read again: distraction on that first frame, duration 6.4 s, confidence 1.0."""
    r = run(tmp_path, monkeypatch, [
        Seg(1.0, face()), Seg(1.5, face(yaw=45)), Seg(5.0, face(yaw=45), pose_fail=True),
        Seg(3.0, face(yaw=45)), Seg(1.0, face())], fps=fps)
    fired = [e for e in r.events if e.type == "distraction"]
    assert len(fired) == 1, r.types()
    resumed = r.seg_starts[3]
    assert fired[0].t >= resumed + 1.3, f"fired {fired[0].t - resumed:.2f}s after the pose came back"
    assert fired[0].duration < 3.6, f"reported {fired[0].duration:.2f}s: unobserved time was credited"


@pytest.mark.parametrize("fps", FPS)
def test_t1_a_look_back_to_the_road_nobody_could_measure_is_not_a_distraction(tmp_path, monkeypatch, fps):
    """Repro (pre-fix): 1.5 s turned, 5 s unreadable (the driver was in fact facing
    the road), then 0.4 s turned again: distraction at confidence 1.0."""
    r = run(tmp_path, monkeypatch, [
        Seg(1.0, face()), Seg(1.5, face(yaw=45)), Seg(5.0, face(), pose_fail=True),
        Seg(0.4, face(yaw=45)), Seg(2.0, face())], fps=fps)
    assert "distraction" not in r.types()


@pytest.mark.parametrize("fps", FPS)
def test_t1_eyes_down_does_not_fire_on_unobserved_time(tmp_path, monkeypatch, fps):
    r = run(tmp_path, monkeypatch, [
        Seg(1.0, face()), Seg(1.0, face(pitch=-30)), Seg(5.0, face(pitch=-30), pose_fail=True),
        Seg(0.5, face(pitch=-30)), Seg(2.0, face())], fps=fps)
    assert "eyes_down" not in r.types()


def test_t1_a_live_phone_use_alert_holds_through_a_brief_pose_failure(tmp_path, monkeypatch):
    """Repro (pre-fix): the unreadable frames fed phone_use 'eyes on the road', so
    after 0.4 s its alert condition dropped and a second phone_use fired 2 s later."""
    look_down = face(pitch=-30)
    r = run(tmp_path, monkeypatch, [
        Seg(1.0, face(), phones=[LAP_PHONE], hands=[LAP_HAND]),
        Seg(3.0, look_down), Seg(1.0, look_down, pose_fail=True), Seg(3.0, look_down), Seg(1.0, face())],
        fps=30.0)
    assert r.count("phone_use") == 1, r.types()
    held = r.active_during("phone_use", r.fired("phone_use").t, r.seg_starts[4])
    assert held and all(held), "the live phone_use condition dropped during the unreadable second"


# ============================================================ T2 PERCLOS over observed time only
def _long_blinks(seconds: float, closed: float = 0.5, period: float = 2.5) -> list[Seg]:
    segs: list[Seg] = []
    for _ in range(int(seconds / period)):
        segs += [Seg(closed, face(lid=SHUT)), Seg(period - closed, face())]
    return segs


@pytest.mark.parametrize("fps", FPS)
def test_t2_a_face_loss_is_not_open_eye_time_in_perclos(tmp_path, monkeypatch, fps):
    """Repro (pre-fix): 12.5 s of 0.5 s closures every 2.5 s (~20 % closed), a 30 s
    look away, 5 s more of the same read PERCLOS 0.061 / 0.045 / 0.031 at 30 / 10 /
    5 fps against 0.147 / 0.101 / 0.057 for the same 17.5 s of blinking without the
    look away: the 30 s nobody saw were counted as open eyes. (The value falling
    with the frame rate is the frame-count median filter, a separate open item.)"""
    seen = run(tmp_path / "seen", monkeypatch, [Seg(1.0, face()), *_long_blinks(17.5)], fps=fps)
    away = run(tmp_path / "away", monkeypatch,
               [Seg(1.0, face()), *_long_blinks(12.5), Seg(30.0, None), *_long_blinks(5.0)], fps=fps)
    observed = seen.module.get_telemetry_snapshot()["perclos"]
    with_loss = away.module.get_telemetry_snapshot()["perclos"]
    assert observed > 0.05
    assert with_loss >= 0.85 * observed, f"diluted: {with_loss:.3f} vs {observed:.3f} without the face loss"


@pytest.mark.parametrize("fps", FPS)
def test_t2_two_blinks_around_a_long_look_away_are_not_fatigue(tmp_path, monkeypatch, fps):
    """Repro (pre-fix): a blink as the face was lost and another as it came back
    charged the whole 20 s between them as closure: fatigue_warning."""
    r = run(tmp_path, monkeypatch, [
        Seg(15.0, face()), Seg(0.25, face(lid=SHUT)), Seg(20.0, None), Seg(0.25, face(lid=SHUT)),
        Seg(5.0, face())], fps=fps)
    assert "fatigue_warning" not in r.types()
    assert r.module.get_telemetry_snapshot()["perclos"] < 0.1


def test_t2_perclos_tracker_semantics_are_unchanged_for_a_steady_stream():
    """The tracker itself still integrates sample-and-hold over its inputs."""
    tracker = PerclosTracker(window_s=60.0)
    t, value = 0.0, 0.0
    for i in range(400):  # 20 s at 20 fps, closed 1 frame in 5
        value = tracker.update(i % 5 == 0, t)
        t += 0.05
    assert value == pytest.approx(0.2, abs=0.01)


# ============================================================ T3 uncalibrated closure is marked
def test_t3_drowsiness_before_calibration_says_it_was_judged_without_calibration(tmp_path):
    module: Any = DriverAttentionModule(shipped_config(
        tmp_path, calibration_duration_s=10.0, calibration_max_pose_spread_deg=20,
        calibration_max_duration_s=30))
    frame = _textured_frame()
    fired = []
    t = 0.0
    for lms, seconds in ((landmarks(lid=OPEN), 3.0), (landmarks(lid=SHUT), 4.0)):
        module._face_mesh.process = lambda rgb, lm=lms: _FakeFaceMeshResult(lm)
        end = t + seconds
        while t < end:
            module.set_frame_time(t)
            fired += [e for e in module.process(frame) if e.type == "drowsiness"]
            t += 1.0 / 30.0
    assert not module._calibrator.is_calibrated
    assert module.perception_signals()["eye_state"].closed is None
    assert len(fired) == 1, "an uncalibrated closure still alerts (the mandatory rule is unchanged)"
    assert "not calibrated" in fired[0].contributing_signals


def test_t3_a_calibrated_drowsiness_carries_no_such_mark(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(1.0, face()), Seg(3.0, face(lid=SHUT))])
    assert r.count("drowsiness") == 1
    assert "not calibrated" not in r.fired("drowsiness").detail


# ============================================================ T4 shadow: a gap is unknown, not "no"
def test_t4_the_shadow_abstains_on_a_frame_after_a_gap_instead_of_saying_no():
    engine = BehaviorEngine()
    t, _ = engine_run(engine, 0.0, 1.0)
    for i in range(8):  # eyes shut on every frame, one frame per second
        engine.update(engine_frame(t + 1.0 * (i + 1), eye="closed"))
    verdict = engine.verdicts[DROWSINESS]
    assert verdict.state == sig.ABSTAIN, verdict
    assert "frame gap" in verdict.reason


def test_t4_the_shadow_is_unchanged_on_a_steady_stream():
    engine = BehaviorEngine()
    t, _ = engine_run(engine, 0.0, 1.0)
    _, started = engine_run(engine, t, 2.5, eye="closed")
    assert DROWSINESS in started
    assert engine.verdicts[DROWSINESS].state == sig.SUPPORTED
    awake = BehaviorEngine()
    engine_run(awake, 0.0, 2.0, eye="open")
    assert awake.verdicts[DROWSINESS].state == sig.NOT_SUPPORTED, "no frame gap: a real 'no'"

