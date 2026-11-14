"""M10.1 Priority 2: repeated distraction and the escalation reset (open item O4).

The shipped episode policy (core/alerts.py, M0.1) ends an episode when its
condition clears, and a new episode starts at the onset pattern. A driver who
keeps relapsing in bursts shorter than `escalate_after_s` (10 s for "warning")
therefore never hears `urgent`, however long it goes on.

`test_relapsing_distraction_reaches_urgent_under_the_shipped_policy` reproduces
that as a FAILING regression (strict xfail): it fails on the shipped config
because the fix -- `alerts.relapse_escalation` -- is off until the owner approves
it (docs/ESCALATION_POLICY.md). Enabling it in config.yaml turns the xfail into
an unexpected pass, which fails the suite on purpose so this file gets updated
with the decision.

SYNTHETIC SCENARIOS ONLY: the production DriverAttentionModule plays a timeline
(tests/m10_temporal_harness.py) and its events / live conditions go through the
alert policy exactly as main.py builds it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
import yaml
from m10_temporal_harness import Result, Seg, face, run

from core.alerts import DEFAULT_CATEGORIES, AlertCategory, AlertManager, RelapseEscalation
from core.config_schema import validate
from core.events import Event
from main import ALERT_EVENT_TYPES_PI, build_alert_manager, relapse_escalation

REPO = Path(__file__).resolve().parent.parent
SHIPPED_ALERTS = yaml.safe_load((REPO / "config.yaml").read_text())["alerts"]
ENABLED = dict(SHIPPED_ALERTS["relapse_escalation"], enabled=True)


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


def replay(result: Result, relapse: Optional[dict] = None):
    """`result` through the policy main.py builds from config.yaml's alerts block
    (with `relapse_escalation` replaced when given); returns (recorder, policy)."""
    cfg = dict(SHIPPED_ALERTS, sinks=["console"])
    if relapse is not None:
        cfg["relapse_escalation"] = relapse
    recorder = _Recorder()
    policy = build_alert_manager(cfg, is_pi=False, sink_overrides={"console": lambda: recorder})
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


def relapsing(cycles: int = 6, away: float = 6.0, back: float = 2.0, yaw: float = 45.0) -> list[Seg]:
    """A driver who keeps looking away for `away` s with `back` s on the road between."""
    out = [Seg(2.0, face())]
    for _ in range(cycles):
        out += [Seg(away, face(yaw=yaw)), Seg(back, face())]
    return out


def distraction_patterns(recorder: _Recorder) -> list[str]:
    return [p for _t, typ, p in recorder.played if typ == "distraction"]


# --------------------------------------------------------------------------- reproduction
@pytest.fixture(scope="module")
def relapse_run(tmp_path_factory) -> Result:
    mp = pytest.MonkeyPatch()
    try:
        return run(tmp_path_factory.mktemp("relapse"), mp, relapsing())
    finally:
        mp.undo()


def test_relapsing_driver_produces_one_episode_per_look_away(relapse_run):
    # Separate episodes are right: each look-away is detected, alerted and ended.
    assert relapse_run.count("distraction") == 6
    _recorder, policy = replay(relapse_run)
    eps = [e for e in policy.recent_episodes() if e["type"] == "distraction"]
    assert len(eps) == 6
    # Every episode alerts for ~3.4 s (fires ~3 s into a 6 s look, ends on the
    # 0.4 s release): under escalate_after_s (10 s) and under repeat_s (5 s).
    assert all(2.5 < e["ended_at"] - e["started_at"] < 4.5 for e in eps)


def test_shipped_policy_restarts_every_relapse_at_the_onset_pattern(relapse_run):
    """The defect, pinned as current behaviour: ~20 s of alerted distraction in
    48 s, and only the onset `double` is ever heard."""
    recorder, policy = replay(relapse_run)
    assert distraction_patterns(recorder) == ["double"] * 6
    assert not any(e["escalated"] for e in policy.recent_episodes() if e["type"] == "distraction")


@pytest.mark.xfail(strict=True, reason="O4 reproduced: relapse escalation awaits owner approval "
                                       "(alerts.relapse_escalation.enabled: false; docs/ESCALATION_POLICY.md)")
def test_relapsing_distraction_reaches_urgent_under_the_shipped_policy(relapse_run):
    recorder, _policy = replay(relapse_run)
    assert "urgent" in distraction_patterns(recorder)


# --------------------------------------------------------------------------- the proposed policy
def test_enabled_policy_escalates_once_the_run_reaches_the_warning_bar(relapse_run):
    recorder, policy = replay(relapse_run, ENABLED)
    patterns = distraction_patterns(recorder)
    # 3 episodes x ~3.4 s alerted = ~10.2 s >= escalate_after_s 10: the 4th starts urgent.
    assert patterns == ["double", "double", "double", "urgent", "urgent", "urgent"], patterns
    eps = [e for e in policy.recent_episodes() if e["type"] == "distraction"]
    assert [e["escalated"] for e in eps] == [False, False, False, True, True, True]
    assert eps[3]["carried_s"] >= 10.0


def test_enabled_policy_changes_patterns_only_never_timing_or_count(relapse_run):
    shipped, shipped_policy = replay(relapse_run)
    enabled, enabled_policy = replay(relapse_run, ENABLED)
    assert [(t, typ) for t, typ, _p in shipped.played] == [(t, typ) for t, typ, _p in enabled.played]
    strip = ("escalated", "carried_s", "episode_id")
    assert ([{k: v for k, v in e.items() if k not in strip} for e in shipped_policy.recent_episodes()]
            == [{k: v for k, v in e.items() if k not in strip} for e in enabled_policy.recent_episodes()])


def test_genuine_recovery_ends_the_run(tmp_path, monkeypatch):
    segs = [*relapsing(cycles=3), Seg(65.0, face()), *relapsing(cycles=1)[1:]]
    r = run(tmp_path, monkeypatch, segs)
    recorder, _policy = replay(r, ENABLED)
    # 3 relapses (~10.2 s alerted), then 65 s attentive (> recovery_s 60): fresh start.
    assert distraction_patterns(recorder) == ["double", "double", "double", "double"]


def test_a_continuous_episode_escalates_exactly_as_before(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(25.0, face(yaw=45)), Seg(3.0, face())])
    shipped, _ = replay(r)
    enabled, _ = replay(r, ENABLED)
    assert distraction_patterns(shipped) == distraction_patterns(enabled) == [
        "double", "double", "urgent", "urgent", "urgent"]


# --------------------------------------------------------------------------- policy unit tests
WARNING = AlertCategory("warning", "double", "urgent", repeat_s=5.0, escalate_after_s=10.0, max_repeats=24)
CRITICAL = AlertCategory("critical", "critical", "critical", repeat_s=4.0, escalate_after_s=0.0, max_repeats=30)


def _policy(relapse: Optional[RelapseEscalation]):
    recorder = _Recorder()
    policy = AlertManager(
        sinks=[recorder], categories={**DEFAULT_CATEGORIES, "warning": WARNING, "critical": CRITICAL},
        type_categories={"distraction": "warning", "eyes_down": "warning", "phone_use": "warning",
                         "drowsiness": "critical", "eating": "warning"},
        relapse=relapse,
    )
    return recorder, policy


RELAPSE = RelapseEscalation(enabled=True, types=frozenset({"distraction", "eyes_down", "phone_use"}),
                            recovery_s=60.0)
TRACKED = ("distraction", "eyes_down", "phone_use", "drowsiness", "eating")


def _episode(policy, recorder, event_type: str, start: float, length: float) -> None:
    """One tracked episode alerted from `start` for `length` s, observed every 0.1 s."""
    recorder.now = start
    policy.notify(Event.now(event_type, 3.0, None, "driver_attention"), now=start)
    t = start
    while t < start + length:
        t = round(t + 0.1, 6)
        recorder.now = t
        policy.observe(t, active=[event_type] if t < start + length else [], tracked=TRACKED)


def test_unit_off_by_default_and_when_disabled():
    for relapse in (None, RelapseEscalation(enabled=False, types=RELAPSE.types)):
        recorder, policy = _policy(relapse)
        for i in range(5):
            _episode(policy, recorder, "distraction", i * 6.0, 4.0)
        assert [p for _t, _e, p in recorder.played] == ["double"] * 5


def test_unit_types_share_one_run_and_concurrent_time_counts_once():
    recorder, policy = _policy(RELAPSE)
    # eyes_down and phone_use alerting over the SAME 4 s: 4 s of run, not 8.
    recorder.now = 0.0
    policy.notify(Event.now("eyes_down", 2.5, None, "m"), now=0.0)
    policy.notify(Event.now("phone_use", 2.5, None, "m"), now=0.0)
    t = 0.0
    while t < 4.0:
        t = round(t + 0.1, 6)
        policy.observe(t, active=["eyes_down", "phone_use"] if t < 4.0 else [], tracked=TRACKED)
    _episode(policy, recorder, "distraction", 6.0, 5.0)  # run 4 -> 9 s
    _episode(policy, recorder, "eyes_down", 13.0, 2.0)   # carried 9 s: still under 10
    _episode(policy, recorder, "phone_use", 17.0, 2.0)   # carried 11 s: escalated onset
    onsets = [(round(t, 1), e, p) for t, e, p in recorder.played]
    assert onsets[-2] == (13.0, "eyes_down", "double")
    assert onsets[-1] == (17.0, "phone_use", "urgent")
    assert policy.recent_episodes()[-1]["carried_s"] == pytest.approx(11.0, abs=0.15)


def test_unit_an_episode_escalates_mid_way_on_carried_time():
    recorder, policy = _policy(RELAPSE)
    _episode(policy, recorder, "distraction", 0.0, 4.0)
    _episode(policy, recorder, "distraction", 6.0, 4.0)   # run 8 s
    _episode(policy, recorder, "distraction", 12.0, 6.0)  # repeat at 17 s: 5 + 8 >= 10
    third = [p for t, _e, p in recorder.played if t >= 12.0]
    assert third == ["double", "urgent"]


def test_unit_recovery_boundary():
    for gap, expected in ((59.0, "urgent"), (61.0, "double")):
        recorder, policy = _policy(RELAPSE)
        for i in range(3):
            _episode(policy, recorder, "distraction", i * 6.0, 4.0)  # run 12 s, last end 16 s
        _episode(policy, recorder, "distraction", 16.0 + gap, 2.0)
        assert recorder.played[-1][2] == expected, gap


def test_unit_unlisted_and_critical_types_are_untouched():
    recorder, policy = _policy(RELAPSE)
    for i in range(4):
        _episode(policy, recorder, "distraction", i * 6.0, 4.0)
    _episode(policy, recorder, "eating", 30.0, 2.0)       # warning category, not a listed type
    _episode(policy, recorder, "drowsiness", 34.0, 2.0)   # critical: one pattern, nothing to escalate
    assert [p for _t, e, p in recorder.played if e == "eating"] == ["double"]
    assert [p for _t, e, p in recorder.played if e == "drowsiness"] == ["critical"]


# --------------------------------------------------------------------------- configuration
def test_config_ships_it_off_and_validates():
    assert SHIPPED_ALERTS["relapse_escalation"]["enabled"] is False
    assert relapse_escalation(SHIPPED_ALERTS) is None
    built = relapse_escalation({"relapse_escalation": ENABLED})
    assert built is not None and built.enabled and built.recovery_s == 60.0
    assert built.types == frozenset({"distraction", "eyes_down", "eyes_up", "phone_use"})


@pytest.mark.parametrize("block", [{"enabled": "yes"}, {"enabled": 1}, "on"])
def test_only_the_boolean_true_enables_it(block):
    assert relapse_escalation({"relapse_escalation": block}) is None


@pytest.mark.parametrize(("block", "fragment"), [
    ({"enabled": "yes"}, "enabled"),
    ({"enabled": True, "types": "distraction"}, "types"),
    ({"enabled": True, "recovery_s": 1}, "recovery_s"),
    ({"enabled": True, "recovery_s": 9999}, "recovery_s"),
    ("on", "mapping"),
])
def test_schema_rejects_bad_blocks(block, fragment):
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    config["alerts"]["relapse_escalation"] = block
    errors, _warnings = validate(config)
    assert any("relapse_escalation" in e and fragment in e for e in errors), errors


def test_recovery_bounds_are_enforced_by_the_policy_type():
    with pytest.raises(ValueError, match="recovery_s"):
        RelapseEscalation(enabled=True, types=frozenset({"distraction"}), recovery_s=2.0)
