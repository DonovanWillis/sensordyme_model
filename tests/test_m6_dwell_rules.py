"""M6: per-subject dwell-in-zone on the device (core/rules.py `dwell_s`).

"Alert me if someone stays inside this marked area for longer than 10 seconds"
is a claim about ONE person's continuous visit. Every test drives the engine
with an explicit clock (`now=`), at 10 frames per second, so the timing is
deterministic to the frame.
"""

from __future__ import annotations

from datetime import UTC, datetime

from core.detections import Detection, DetectionFrame
from core.monitoring_config import MonitoringConfig
from core.rules import RuleEngine, validate_rule_set
from core.zones import Zone

W, H = 1000, 1000
FPS = 10
#: The marked area: the left half of the frame.
AREA = Zone("marked area", ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)))
#: Feet points at x=0.25 (inside) and x=0.75 (outside).
INSIDE = (200.0, 400.0, 300.0, 800.0)
INSIDE_OTHER = (350.0, 400.0, 450.0, 800.0)
OUTSIDE = (700.0, 400.0, 800.0, 800.0)

DWELL_RULE = {
    "target": "person", "predicate": "lingering", "zone": "marked area", "dwell_s": 10,
    "persist_s": 10, "description": "Someone stays inside the marked area for 10 s",
    "severity": "warning", "cooldown_s": 60,
}


def _engine(rules=None, zones=(AREA,)):
    return RuleEngine(rules if rules is not None else [DWELL_RULE], zones=list(zones),
                      clock=lambda: datetime(2026, 11, 11, 12, 0, tzinfo=UTC))


def _frame(*boxes):
    return DetectionFrame(W, H, tuple(Detection("person", 0.9, box) for box in boxes))


def _run(engine, timeline, until_s):
    """Feed frames from t=0 to `until_s`; `timeline(t)` gives the boxes at t.
    Returns [(t, alert)] for every confirmed alert."""
    alerts = []
    for i in range(int(until_s * FPS) + 1):
        t = i / FPS
        for alert in engine.observe_frame(_frame(*timeline(t)), now=t):
            alerts.append((t, alert))
    return alerts


def test_leaving_at_9s_never_alerts():
    engine = _engine()
    alerts = _run(engine, lambda t: [INSIDE] if t < 9.0 else [OUTSIDE], until_s=30)
    assert alerts == []


def test_staying_10s_alerts_exactly_once():
    engine = _engine()
    alerts = _run(engine, lambda t: [INSIDE], until_s=30)
    assert len(alerts) == 1, alerts
    t, alert = alerts[0]
    # Confirmed on the frame after the 10 s mark (the engine's one-frame
    # confirmation), never before it.
    assert 10.0 <= t <= 10.0 + 1.5 / FPS
    assert alert.zone == "marked area" and alert.detection == "person"
    # One episode: still active, not re-alerted while the person stays.
    assert engine.active_alerts() == [alert.rule_id]


def test_staying_exactly_until_10s_then_leaving_does_not_alert_early():
    engine = _engine()
    alerts = _run(engine, lambda t: [INSIDE] if t < 9.95 else [], until_s=20)
    assert alerts == []


def test_a_succession_of_people_is_not_one_dwell():
    """Person A inside 0-9 s, person B (elsewhere in the area) 5-14 s: the area
    is occupied for 14 s, but nobody stayed 10 s."""
    def timeline(t):
        boxes = []
        if t < 9.0:
            boxes.append(INSIDE)
        if 5.0 <= t < 14.0:
            boxes.append(INSIDE_OTHER)
        return boxes

    assert _run(_engine(), timeline, until_s=30) == []
    # ...and B staying on to 15 s is a real dwell, alerted once.
    alerts = _run(_engine(), lambda t: ([INSIDE] if t < 9.0 else []) + ([INSIDE_OTHER] if t >= 5.0 else []),
                  until_s=30)
    assert len(alerts) == 1 and 15.0 <= alerts[0][0] <= 15.2


def test_brief_occlusion_does_not_restart_the_clock():
    # Missing for 0.5 s at 6 s (behind a chair) is the same visit.
    alerts = _run(_engine(), lambda t: [] if 6.0 <= t < 6.5 else [INSIDE], until_s=20)
    assert len(alerts) == 1 and alerts[0][0] <= 10.2


def test_leaving_for_longer_than_the_grace_restarts_the_visit():
    alerts = _run(_engine(), lambda t: [OUTSIDE] if 6.0 <= t < 8.0 else [INSIDE], until_s=30)
    # The second visit starts at 8 s, so the dwell completes at 18 s.
    assert len(alerts) == 1 and 18.0 <= alerts[0][0] <= 18.2


def test_an_undrawn_area_waits_and_never_fires_frame_wide():
    engine = _engine(zones=())
    assert engine.pending_zones() == ["marked area"]
    assert _run(engine, lambda t: [INSIDE, OUTSIDE], until_s=30) == []
    # Drawn later: the clock starts from when it can observe, not from boot.
    engine.set_zones([AREA])
    alerts = []
    for i in range(15 * FPS):
        t = 30 + i / FPS
        alerts += engine.observe_frame(_frame(INSIDE), now=t)
    assert len(alerts) == 1


def test_a_dwell_without_an_area_is_refused_not_run_frame_wide():
    bad = {**DWELL_RULE, "zone": ""}
    assert validate_rule_set([bad], label_ok=lambda label: True)
    assert _engine(rules=[bad]).rules == []
    unreadable = {**DWELL_RULE, "dwell_s": "ten"}
    assert validate_rule_set([unreadable], label_ok=lambda label: True)
    assert _engine(rules=[unreadable]).rules == []
    assert validate_rule_set([{**DWELL_RULE, "predicate": "absent"}], label_ok=lambda label: True)
    assert validate_rule_set([DWELL_RULE], label_ok=lambda label: True) == []


def test_dwell_rules_coexist_with_other_rules():
    presence = {"target": "dog", "predicate": "present", "description": "dog", "persist_s": 1.0}
    engine = _engine(rules=[DWELL_RULE, presence])
    alerts = []
    for i in range(12 * FPS):
        t = i / FPS
        frame = DetectionFrame(W, H, (Detection("person", 0.9, INSIDE), Detection("dog", 0.9, OUTSIDE)))
        alerts += [a.detection for a in engine.observe_frame(frame, now=t)]
    assert sorted(alerts) == ["dog", "person"]


def test_a_saved_dwell_rule_survives_a_restart(tmp_path):
    path = tmp_path / "monitoring.json"
    config = MonitoringConfig(path)
    config.save([DWELL_RULE], instruction="stay inside", version="v1")
    restored = MonitoringConfig(path)
    assert restored.load()
    engine = _engine(rules=restored.rules)
    assert engine.rules[0].dwell_s == 10.0
    assert len(_run(engine, lambda t: [INSIDE], until_s=12)) == 1
