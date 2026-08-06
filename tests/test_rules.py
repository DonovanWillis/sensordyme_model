"""Tests for AI-configured rule enforcement (core/rules.py).

These cover the behaviours the product promises when a user types a rule in
plain language: it debounces, it respects the hours it was given, it only fires
inside the zone it named, and it stays inert rather than firing frame-wide when
the zone hasn't been drawn yet.
"""

from datetime import datetime

from core.rules import Alert, ConfiguredRule, RuleEngine, _in_window, _parse_schedule
from core.zones import Zone


def rule(**overrides):
    base = {
        "detection": "person_present",
        "description": "Alert if someone enters the room",
        "zone": "",
        "persist_s": 2.0,
        "min_count": 1,
        "confidence": 0.5,
        "severity": "warning",
        "cooldown_s": 30.0,
        "schedule": "",
        "notify": "push",
    }
    base.update(overrides)
    return base


def drive(engine, detection, seconds, *, start=1000.0, step=0.5, **kwargs):
    """Feed a detection as present for `seconds`; return every alert emitted."""
    alerts = []
    ticks = int(seconds / step) + 1
    for i in range(ticks):
        alerts += engine.observe(detection, True, now=start + i * step, **kwargs)
    return alerts


# --------------------------------------------------------------------- schedule parsing
def test_parse_schedule_handles_windows_and_junk():
    assert _parse_schedule("23:00-06:00") == (1380, 360)
    assert _parse_schedule("09:30-17:00") == (570, 1020)
    assert _parse_schedule("") is None
    assert _parse_schedule("garbage") is None
    assert _parse_schedule("25:00-06:00") is None      # out of range -> always on
    assert _parse_schedule("06:00-06:00") is None      # zero-length -> always on


def test_in_window_wraps_past_midnight():
    overnight = (1380, 360)  # 23:00-06:00
    assert _in_window(overnight, 23 * 60 + 30)   # 23:30
    assert _in_window(overnight, 2 * 60)         # 02:00
    assert not _in_window(overnight, 12 * 60)    # midday
    daytime = (570, 1020)  # 09:30-17:00
    assert _in_window(daytime, 12 * 60)
    assert not _in_window(daytime, 2 * 60)
    assert _in_window(None, 3 * 60)              # no schedule is always active


# --------------------------------------------------------------------- confirmation
def test_single_frame_never_alerts():
    engine = RuleEngine([rule(persist_s=2.0)])
    assert engine.observe("person_present", True, now=1000.0) == []


def test_sustained_detection_confirms_once():
    engine = RuleEngine([rule(persist_s=2.0)])
    alerts = drive(engine, "person_present", 5.0)
    assert len(alerts) == 1, "a held condition should confirm exactly once"
    assert isinstance(alerts[0], Alert)
    assert alerts[0].description == "Alert if someone enters the room"
    assert alerts[0].severity == "warning"


def test_unknown_detection_is_ignored():
    engine = RuleEngine([rule(detection="person_present")])
    assert drive(engine, "something_else", 5.0) == []


def test_silent_rules_track_but_do_not_alert():
    engine = RuleEngine([rule(notify="silent")])
    assert drive(engine, "person_present", 5.0) == []
    assert engine.active_alerts(), "a silent rule should still be tracked, just not surfaced"


def test_min_count_requires_enough_subjects():
    engine = RuleEngine([rule(min_count=3, description="More than three people")])
    assert drive(engine, "person_present", 5.0, count=2) == []
    assert len(drive(engine, "person_present", 5.0, count=3, start=5000.0)) == 1


def test_low_confidence_never_confirms():
    engine = RuleEngine([rule(confidence=0.9)])
    assert drive(engine, "person_present", 6.0, confidence=0.2) == []


def test_zero_persistence_is_floored():
    """A bad generation must not produce a rule that alerts on one frame."""
    engine = RuleEngine([rule(persist_s=0.0)])
    assert engine.observe("person_present", True, now=1000.0) == []


# --------------------------------------------------------------------- schedules
def test_rule_is_inert_outside_its_hours():
    engine = RuleEngine([rule(schedule="23:00-05:00")], clock=lambda: datetime(2026, 8, 6, 14, 0))
    assert drive(engine, "person_present", 6.0) == []


def test_rule_fires_inside_its_hours():
    engine = RuleEngine([rule(schedule="23:00-05:00")], clock=lambda: datetime(2026, 8, 6, 23, 30))
    assert len(drive(engine, "person_present", 6.0)) == 1


def test_rule_does_not_bank_progress_before_its_window_opens():
    """Observing at 22:59 must not let the rule fire the instant 23:00 arrives."""
    clock = {"now": datetime(2026, 8, 6, 22, 59)}
    engine = RuleEngine([rule(schedule="23:00-05:00", persist_s=2.0)], clock=lambda: clock["now"])
    drive(engine, "person_present", 10.0)              # all before the window
    clock["now"] = datetime(2026, 8, 6, 23, 0)
    assert engine.observe("person_present", True, now=2000.0) == [], "must restart inside the window"


# --------------------------------------------------------------------- zones
POOL = Zone(name="pool", polygon=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))
INSIDE_POOL = [(600.0, 600.0, 700.0, 900.0)]    # feet at (0.65, 0.9)
OUTSIDE_POOL = [(10.0, 10.0, 110.0, 200.0)]     # feet at (0.06, 0.2)
FRAME = (1000, 1000)


def test_zone_rule_fires_only_inside_the_zone():
    engine = RuleEngine([rule(zone="pool")], zones=[POOL])
    assert drive(engine, "person_present", 6.0, boxes=OUTSIDE_POOL, frame_size=FRAME) == []
    alerts = drive(engine, "person_present", 6.0, start=5000.0, boxes=INSIDE_POOL, frame_size=FRAME)
    assert len(alerts) == 1
    assert alerts[0].zone == "pool"


def test_undrawn_zone_makes_the_rule_inert_not_frame_wide():
    engine = RuleEngine([rule(zone="pool")], zones=[])  # user hasn't drawn it yet
    assert drive(engine, "person_present", 6.0, boxes=INSIDE_POOL, frame_size=FRAME) == []
    assert engine.pending_zones() == ["pool"]


def test_pending_zones_clears_once_drawn():
    engine = RuleEngine([rule(zone="pool")], zones=[])
    assert engine.pending_zones() == ["pool"]
    engine.set_zones([POOL])
    assert engine.pending_zones() == []


# --------------------------------------------------------------------- rule set handling
def test_malformed_rules_are_skipped_not_fatal():
    engine = RuleEngine([rule(), {"description": "no detection id"}, "not a dict", None])
    assert len(engine.rules) == 1


def test_replace_rules_drops_in_flight_progress():
    engine = RuleEngine([rule(persist_s=2.0)])
    engine.observe("person_present", True, now=1000.0)
    engine.replace_rules([rule(persist_s=2.0)])  # new proposal from the app
    assert engine.observe("person_present", True, now=1000.5) == [], "old progress must not carry over"


def test_severity_and_notify_are_clamped():
    parsed = ConfiguredRule.from_dict(0, rule(severity="apocalyptic", notify="carrier_pigeon"))
    assert parsed.severity == "warning"
    assert parsed.notify == "push"


def test_rule_without_detection_is_dropped():
    assert ConfiguredRule.from_dict(0, {"description": "vague wish"}) is None
