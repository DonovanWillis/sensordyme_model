"""M10.1 Priority 1: frame-rate-dependent eye evidence.

Before M10.1 (docs/TEMPORAL_REASONING.md O1, O2):

* the iris veto compared the RAW per-frame iris-openness with the 5-frame EAR
  median, which runs 2 frames late, so the last 2 frames of every closure were
  vetoed: 0.07 s at 30 fps, 0.4 s at 5 fps. The same 0.7 s long blinks read
  PERCLOS 0.23 at 30 fps and 0.14 at 5 fps, and fatigue_warning never fired at
  5 fps;
* a lost face fed the closure tracker "open", so a closure reset after 2
  face-lost FRAMES: 0.07 s at 30 fps, 0.4 s at 5 fps.

Now the iris median runs over the same frames as the EAR median, and a lost face
holds the closure (crediting nothing) for eye_closure_face_loss_hold_s (0.4 s)
at every rate. Below eye_timing_min_fps the module reports capability
`eye_timing` impaired, so drowsiness / fatigue_warning are reported unavailable
(monitoring DEGRADED) instead of implied.

None of the thresholds moved (drowsy_seconds 2.0, perclos_threshold 0.15,
perclos_closure_ratio 0.4, eye_closure_release_frames 2, signal_median_window 5).
The tests below also pin that normal and sub-threshold blinking still raises
nothing at any rate: the fix restores measured closure time, it does not add
sensitivity.

SYNTHETIC SCENARIOS ONLY (tests/m10_temporal_harness.py).
"""

from __future__ import annotations

import pytest
from m10_temporal_harness import SHUT, Seg, face, run

import modules.driver_attention as da
from core import monitoring_state as ms
from core.eye_signal import AlignedOptionalSmoother, SignalSmoother
from modules.base import GracedConditionTracker

RATES = (30.0, 15.0, 10.0, 5.0)
JITTERED = ((30.0, 0.0), (15.0, 0.0), (10.0, 0.0), (5.0, 0.0), (10.0, 0.3), (5.0, 0.3))


def _blinks(seconds: float, closed: float, period: float) -> list[Seg]:
    out: list[Seg] = []
    for _ in range(round(seconds / period)):
        out += [Seg(closed, face(lid=SHUT)), Seg(period - closed, face())]
    return out


def _perclos_trace(tmp_path, monkeypatch, segments, fps, jitter=0.0):
    seen: list[float] = []
    original = da.PerclosTracker.update

    def spy(self, closed, now):
        value = original(self, closed, now)
        seen.append(value)
        return value

    monkeypatch.setattr(da.PerclosTracker, "update", spy)
    result = run(tmp_path, monkeypatch, segments, fps=fps, jitter=jitter, seed=17)
    return result, seen


# --------------------------------------------------------------------------- PERCLOS (O2)
@pytest.mark.parametrize(("fps", "jitter"), JITTERED)
def test_long_blinks_read_the_same_perclos_and_fire_at_every_rate(tmp_path, monkeypatch, fps, jitter):
    """0.7 s closures every 2 s for 30 s after 10 s open: 10.5 s closed of 40 s."""
    result, perclos = _perclos_trace(tmp_path, monkeypatch, [Seg(10.0, face()), *_blinks(30.0, 0.7, 2.0)],
                                     fps, jitter)
    # 30 fps reads 0.250 (truth 0.2625 less the median's onset lag). Before
    # M10.1: 0.226 / 0.214 / 0.178 / 0.142 at 30 / 15 / 10 / 5 fps.
    assert perclos[-1] == pytest.approx(0.25, abs=0.04), perclos[-1]
    assert result.count("fatigue_warning") == 1


@pytest.mark.parametrize(("fps", "jitter"), JITTERED)
def test_normal_blinking_raises_nothing_at_any_rate(tmp_path, monkeypatch, fps, jitter):
    """0.25 s blinks every 3 s for 60 s (8 % closed): no fatigue, no drowsiness."""
    result = run(tmp_path, monkeypatch, [Seg(5.0, face()), *_blinks(60.0, 0.25, 3.0)], fps=fps,
                 jitter=jitter, seed=17)
    assert not {"fatigue_warning", "drowsiness"} & set(result.types()), result.types()


@pytest.mark.parametrize(("fps", "jitter"), JITTERED)
def test_sub_threshold_long_blinks_raise_nothing_at_any_rate(tmp_path, monkeypatch, fps, jitter):
    """0.4 s closures every 4 s for 60 s: 10 % closed, under perclos_threshold 0.15."""
    result, perclos = _perclos_trace(tmp_path, monkeypatch, [Seg(5.0, face()), *_blinks(60.0, 0.4, 4.0)],
                                     fps, jitter)
    assert max(perclos) < 0.15
    assert not {"fatigue_warning", "drowsiness"} & set(result.types()), result.types()


# --------------------------------------------------------------------------- sustained closure
@pytest.mark.parametrize(("fps", "jitter"), JITTERED)
def test_a_real_closure_fires_drowsiness_at_every_supported_rate(tmp_path, monkeypatch, fps, jitter):
    result = run(tmp_path, monkeypatch, [Seg(3.0, face()), Seg(3.0, face(lid=SHUT)), Seg(2.0, face())],
                 fps=fps, jitter=jitter, seed=17)
    latency = result.latency("drowsiness", 1)
    assert latency is not None, result.types()
    # drowsy_seconds 2.0 + the median's 2-frame onset lag + one frame of sampling.
    assert 2.0 <= latency <= 2.0 + 3.0 * (1 + jitter) / fps + 0.02, latency


@pytest.mark.parametrize("fps", RATES)
def test_closure_with_the_head_pose_unreadable_still_fires(tmp_path, monkeypatch, fps):
    """Partial observation: eyes measurable, gaze not (solvePnP fails)."""
    result = run(tmp_path, monkeypatch, [Seg(3.0, face()), Seg(3.0, face(lid=SHUT), pose_fail=True),
                                         Seg(2.0, face())], fps=fps)
    assert result.count("drowsiness") == 1


# --------------------------------------------------------------------------- face loss (O1)
@pytest.mark.parametrize("fps", RATES)
def test_a_brief_face_loss_holds_the_closure_at_every_rate(tmp_path, monkeypatch, fps):
    """1.5 s shut, 0.3 s no face, 1.0 s shut: 1.5 + 0.5 s observed fires 0.5 s in.
    Before M10.1 this fired at 5 fps only."""
    result = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(1.5, face(lid=SHUT)), Seg(0.3, None),
                                         Seg(1.0, face(lid=SHUT)), Seg(2.0, face())], fps=fps)
    latency = result.latency("drowsiness", 3)
    assert latency is not None and 0.5 - 1e-6 <= latency <= 0.5 + 4.0 / fps + 0.02, latency


@pytest.mark.parametrize("fps", RATES)
def test_a_long_face_loss_resets_the_closure_at_every_rate(tmp_path, monkeypatch, fps):
    """1.5 s shut, 1.0 s no face, 1.0 s shut: the loss outlasts the 0.4 s hold, so
    the second closure starts again (1.0 s < drowsy_seconds): nothing fires."""
    result = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(1.5, face(lid=SHUT)), Seg(1.0, None),
                                         Seg(1.0, face(lid=SHUT)), Seg(2.0, face())], fps=fps)
    assert "drowsiness" not in result.types()


@pytest.mark.parametrize("fps", RATES)
def test_face_loss_time_is_never_credited_as_closure(tmp_path, monkeypatch, fps):
    """1.2 s shut, 0.35 s no face, 0.5 s shut: 1.7 s observed (the loss would make 2.05)."""
    result = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(1.2, face(lid=SHUT)), Seg(0.35, None),
                                         Seg(0.5, face(lid=SHUT)), Seg(2.0, face())], fps=fps)
    assert "drowsiness" not in result.types()


def test_hold_unit_semantics():
    # count_true_time_only, release after 2 open frames, as the module builds it.
    def tracker():
        return GracedConditionTracker(2.0, 0.4, release_frames=2, count_true_time_only=True)

    for step in (1 / 30, 0.2):
        t = tracker()
        now = 0.0
        while now < 1.0 - 1e-9:
            t.update(True, now)
            now += step
        held_from = now
        while now < held_from + 0.39:  # hold frames: neither open nor closed
            t.hold(now, 0.4)
            now += step
        credited = t.elapsed(now)
        assert credited == pytest.approx(1.0, abs=step + 1e-6), (step, credited)
        assert t.elapsed(now) > 0.0  # held, not reset
        t2 = tracker()
        t2.update(True, 0.0)
        t2.update(True, 0.5)
        t2.hold(0.6, 0.4)
        t2.hold(0.95, 0.4)  # 0.45 s since the last measured frame (0.5): reset
        assert t2.elapsed(1.0) == 0.0


def test_hold_does_not_count_as_open_frames():
    t = GracedConditionTracker(2.0, 0.4, release_frames=2, count_true_time_only=True)
    for i in range(10):
        t.update(True, i * 0.1)
    for i in range(3):
        t.hold(1.0 + i * 0.1, 0.4)  # 3 face-lost frames: more than release_frames, still held
    t.update(True, 1.3)
    assert t.elapsed(1.3) > 0.8


# --------------------------------------------------------------------------- the aligned smoother
def test_aligned_smoother_runs_in_step_with_the_signal_it_vetoes():
    ear, iris = SignalSmoother(5), AlignedOptionalSmoother(5)
    # A 4-frame closure: raw both drop on frames 2-5. Smoothed, both must read
    # closed on the SAME frames, or the veto cuts the closure short.
    raw = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    ear_closed = [ear.update(v) < 0.5 for v in raw]
    iris_closed = [(x := iris.update(v)) is not None and x < 0.5 for v in raw]
    assert ear_closed == iris_closed
    assert sum(ear_closed) == 4


def test_aligned_smoother_ignores_unmeasurable_frames_and_says_so():
    s = AlignedOptionalSmoother(5)
    assert s.update(None) is None
    assert s.update(0.8) == 0.8
    assert s.update(None) == 0.8
    assert s.update(0.2) == pytest.approx(0.5)
    s.reset()
    assert s.update(None) is None


# --------------------------------------------------------------------------- capability (degraded, not claimed)
def _eye_timing_after(tmp_path, monkeypatch, fps: float, seconds: float = 3.0, jitter: float = 0.0):
    result = run(tmp_path, monkeypatch, [Seg(seconds, face())], fps=fps, jitter=jitter, seed=17)
    return result.module.capability_status()["eye_timing"], result.module


@pytest.mark.parametrize(("fps", "jitter"), JITTERED)
def test_eye_timing_is_available_at_supported_rates(tmp_path, monkeypatch, fps, jitter):
    cap, module = _eye_timing_after(tmp_path, monkeypatch, fps, jitter=jitter)
    assert cap.available, cap
    assert module.processing_fps() == pytest.approx(fps, rel=0.35)


@pytest.mark.parametrize("fps", [4.0, 3.0, 2.0, 1.0])
def test_eye_timing_is_impaired_below_the_verified_rate(tmp_path, monkeypatch, fps):
    cap, module = _eye_timing_after(tmp_path, monkeypatch, fps, seconds=8.0)
    assert cap.impaired, cap
    assert "eye-closure timing" in (cap.reason or "") and "verified from 4.5/s" in (cap.reason or "")
    # What the device then reports: those two events unavailable, monitoring DEGRADED, not READY.
    reqs = module.event_requirements()
    caps = {**module.capability_status(), "camera": ms.Capability(), "inference": ms.Capability(),
            "audible_alert": ms.Capability()}
    caps["attention"] = ms.Capability()
    evaluation = ms.evaluate(stopped=None, capabilities=caps,
                             policies=[ms.Policy("driver", "preset", "Driver safety", reqs)],
                             role_required={"camera", "inference", "face"})
    assert evaluation.state is ms.MonitoringState.DEGRADED
    health = evaluation.policies[0]
    assert set(health.unavailable_events) == {"drowsiness", "fatigue_warning"}
    assert any("eye_timing" in r for r in health.reasons)


def test_eye_timing_is_starting_until_the_rate_is_measured(tmp_path, monkeypatch):
    # Two frames: one interval, fewer than the 5 the rate needs.
    result = run(tmp_path, monkeypatch, [Seg(0.05, face())], fps=30.0)
    module = result.module
    module._frame_intervals.clear()
    module._frame_intervals.append(1 / 30)
    assert module.processing_fps() is None
    assert module.capability_status()["eye_timing"].state == ms.STARTING


def test_eye_timing_is_an_analysis_capability_everywhere():
    """Withdrawn with the camera / inference and paused while stopped, on the
    device, the backend and the app alike."""
    import re
    from pathlib import Path

    import cloud.app
    import services.runtime

    backend, device = cloud.app._ANALYSIS_CAPABILITIES, services.runtime._ANALYSIS_CAPABILITIES

    swift = (Path(__file__).resolve().parent.parent / "ios/Sources/DeviceStatus.swift").read_text()
    match = re.search(r"static let analysisCapabilities: Set<String> = \[([^\]]*)\]", swift)
    assert match is not None
    app = {s.strip().strip('"') for s in match.group(1).split(",")}
    assert "eye_timing" in device and device == backend == app
