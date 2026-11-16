"""M10: alert episodes over time -- one event per real episode, repetition and
escalation per core/alerts.py, no duplicates or flapping from noisy signals.

SYNTHETIC SCENARIOS ONLY. The production DriverAttentionModule plays a timeline
(tests/m10_temporal_harness.py); its events and per-frame live conditions are
then handed, frame by frame, to the alert policy built exactly as main.py builds
it from the shipped config.yaml `alerts:` block (sinks replaced by a recorder),
the same calls services/runtime.py makes: notify() for alertable events, then
observe() with the module's tracked / active conditions.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from m10_temporal_harness import SHUT, Result, Seg, face, run

from core.events import Event
from main import ALERT_EVENT_TYPES_PI, build_alert_manager

REPO = Path(__file__).resolve().parent.parent


class _Recorder:
    name, kind, audible = "recorder", "buzzer", True

    def __init__(self) -> None:
        self.played: list[tuple[float, str, str]] = []
        self.now = 0.0

    def readiness(self) -> tuple[bool, str]:
        return True, ""

    def play(self, pattern, event, category) -> None:
        self.played.append((self.now, event.type, pattern.name))

    def emit(self, event, level) -> None:  # pragma: no cover - play() is used
        raise AssertionError("pattern sink expected")


def alert_replay(result: Result):
    """Feed `result` through the shipped alert policy; returns (recorder, policy)."""
    alerts_cfg = yaml.safe_load((REPO / "config.yaml").read_text())["alerts"]
    recorder = _Recorder()
    policy = build_alert_manager(dict(alerts_cfg, sinks=["console"]), is_pi=False,
                                 sink_overrides={"console": lambda: recorder})
    tracked = result.module.alert_episode_types()
    by_time: dict[float, list] = {}
    for e in result.events:
        by_time.setdefault(e.t, []).append(Event.now(e.type, e.duration, e.confidence, "driver_attention",
                                                     e.detail))
    for t, active in result.active:
        recorder.now = t
        policy.process_frame(t, by_time.get(t, []), alertable=ALERT_EVENT_TYPES_PI, active=active,
                             tracked=tracked)
    policy.end_all(result.active[-1][0] + 1.0)
    return recorder, policy


def episodes(policy, event_type: str) -> list[dict]:
    return [e for e in policy.recent_episodes() if e["type"] == event_type]


def test_a_25s_distraction_is_one_episode_that_repeats_then_escalates(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(25.0, face(yaw=45)), Seg(3.0, face())])
    assert r.count("distraction") == 1
    recorder, policy = alert_replay(r)
    eps = episodes(policy, "distraction")
    assert len(eps) == 1 and eps[0]["escalated"]
    patterns = [p for _t, typ, p in recorder.played if typ == "distraction"]
    # warning: "double" at onset, repeat every 5 s, "urgent" from 10 s into the episode
    assert patterns[:2] == ["double", "double"] and set(patterns[2:]) == {"urgent"}, patterns
    assert len(patterns) == 5, patterns  # fired at ~3 s, active ~22 s: onset + 4 repeats
    ended = r.seg_starts[2]
    assert all(t <= ended + 0.5 for t, typ, _p in recorder.played if typ == "distraction")


def test_a_closure_with_landmark_glitches_is_one_critical_episode(tmp_path, monkeypatch):
    glitchy = [s for _ in range(24) for s in (Seg(4 / 30, face(lid=SHUT)), Seg(1 / 30, face()))] * 5
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), *glitchy, Seg(3.0, face())])
    assert r.count("drowsiness") == 1
    _recorder, policy = alert_replay(r)
    eps = episodes(policy, "drowsiness")
    assert len(eps) == 1 and eps[0]["category"] == "critical"
    assert eps[0]["actuations"] >= 4, "a ~17 s live closure repeats every 4 s"


def test_brief_flicks_back_to_the_road_do_not_split_the_alert(tmp_path, monkeypatch):
    away = [s for _ in range(8) for s in (Seg(2.0, face(yaw=45)), Seg(0.2, face()))]
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), *away, Seg(3.0, face())], fps=10.0)
    assert r.count("distraction") == 1
    _recorder, policy = alert_replay(r)
    eps = episodes(policy, "distraction")
    assert len(eps) == 1 and eps[0]["escalated"]


def test_a_real_recovery_ends_the_episode_and_the_next_one_starts_fresh(tmp_path, monkeypatch):
    """Two distractions with 2 s back on the road between them: two events, two
    episodes, the second one starting at the onset pattern (escalation does not
    carry across a real recovery -- recorded as an open item, not changed here)."""
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(14.0, face(yaw=45)), Seg(2.0, face()),
                                    Seg(5.0, face(yaw=45)), Seg(2.0, face())])
    assert r.count("distraction") == 2
    recorder, policy = alert_replay(r)
    eps = episodes(policy, "distraction")
    assert len(eps) == 2 and eps[0]["escalated"] and not eps[1]["escalated"]
    second = [p for t, typ, p in recorder.played if typ == "distraction" and t >= r.seg_starts[3]]
    assert second[0] == "double"


def test_a_camera_stall_inside_a_closure_keeps_one_episode(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(6.0, face(lid=SHUT)), Seg(3.0, stall=True),
                                    Seg(6.0, face(lid=SHUT)), Seg(2.0, face())])
    assert r.count("drowsiness") == 1
    recorder, policy = alert_replay(r)
    eps = episodes(policy, "drowsiness")
    assert len(eps) == 1
    stall_start, stall_end = r.seg_starts[2], r.seg_starts[3]
    assert not [t for t, typ, _p in recorder.played if typ == "drowsiness" and stall_start < t < stall_end]


def test_face_lost_is_one_attention_lost_episode_ending_when_the_face_returns(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(15.0, None, person=False),
                                    Seg(3.0, face(), person=True)])
    assert r.count("attention_lost") == 1
    _recorder, policy = alert_replay(r)
    eps = episodes(policy, "attention_lost")
    assert len(eps) == 1 and eps[0]["category"] == "critical"
    assert eps[0]["ended_at"] < r.seg_starts[2] + 1.0, "ends within the release grace of the face returning"
