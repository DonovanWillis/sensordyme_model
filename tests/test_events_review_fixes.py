"""Regression tests for the events / rules / alerts / config review.

Each section names the defect it pins. Every test here failed against the code
as it was before the fix; they are grouped by file rather than by severity.
"""

from __future__ import annotations

import csv
import errno
import json
import os
import sys
import threading
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from core.alerts import AlertLevel, AlertManager
from core.config_schema import validate
from core.dataset import CaptureStore
from core.detections import Detection, DetectionFrame
from core.event_engine import CONFIRMED, EventEngine, RuleSpec
from core.events import Event
from core.logger import EventLogger, repair
from core.monitoring_config import load_or_empty
from core.reports import generate_weekly_report, last_days, load_events, summarize
from core.rules import Alert, RuleEngine
from core.telemetry import TelemetryLogger
from core.zones import Zone
from scripts.migrate_config import LIST_MIGRATIONS, migrate

REPO = Path(__file__).resolve().parent.parent

W = H = 1000
POOL = Zone(name="pool", polygon=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))
IN_POOL = (600.0, 600.0, 700.0, 900.0)      # feet at (0.65, 0.9)
OUT_OF_POOL = (10.0, 10.0, 110.0, 200.0)    # feet at (0.06, 0.2)
BASE_RULE = {"persist_s": 2.0, "confidence": 0.5, "cooldown_s": 30.0,
             "notify": "push", "severity": "warning"}


def frame(*boxes, label="person", confidence=0.9) -> DetectionFrame:
    return DetectionFrame(W, H, tuple(Detection(label, confidence, b) for b in boxes))


def drive(engine: RuleEngine, make_frame, seconds=10.0, start=1000.0, step=0.5) -> list[Alert]:
    alerts: list[Alert] = []
    for i in range(int(seconds / step) + 1):
        f = make_frame(i) if callable(make_frame) else make_frame
        alerts += engine.observe_frame(f, now=start + i * step)
    return alerts


def an_event(kind="fall") -> Event:
    return Event.now(kind, 1.0, 0.9, "test")


class _FailingFile:
    """A file whose writes reach the OS and fail there, like a full card."""

    def __init__(self, real, err=errno.ENOSPC):
        self._real = real
        self._err = err

    def write(self, text):
        return self._real.write(text)

    def flush(self):
        raise OSError(self._err, os.strerror(self._err))

    def fileno(self):
        return self._real.fileno()

    def close(self):
        self._real.close()


# =========================================================== F1: logging never raises
def test_event_log_write_failure_is_logged_not_raised(tmp_path):
    """A full card used to raise out of log() -- which runs before the alert
    for the same event is dispatched -- and then kill the inference loop."""
    logger = EventLogger(tmp_path / "events.csv")
    logger._file = _FailingFile(logger._file)
    logger._writer = csv.writer(logger._file)
    for _ in range(3):
        logger.log(an_event())  # must not raise
    logger.close()  # must not raise either


def test_event_log_on_an_unwritable_directory_does_not_stop_boot(tmp_path):
    directory = tmp_path / "ro"
    directory.mkdir()
    directory.chmod(0o500)
    try:
        if os.access(directory, os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        logger = EventLogger(directory / "events.csv")  # must not raise
        logger.log(an_event())                          # nor this
        logger.close()
    finally:
        directory.chmod(0o700)


def test_telemetry_write_failure_returns_false_instead_of_raising(tmp_path):
    logger = TelemetryLogger(tmp_path / "telemetry.csv", interval_s=0.0)
    path = tmp_path / "telemetry.csv"
    file, _writer, header = logger._files[path]
    failing = _FailingFile(file, errno.EROFS)
    logger._files[path] = (failing, csv.writer(failing), header)
    assert logger.log_if_due(0.0, "t", "driver_attention", {"ear": 0.3}) is False
    logger.close()


def test_telemetry_is_rotated_once_it_passes_its_size_limit(tmp_path):
    """Nothing ever trimmed this stream, and at ~15-26 MB a day it is what
    fills a long-running device's card."""
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=0.0, max_bytes=400)
    for i in range(50):
        assert logger.log_if_due(float(i), f"t{i}", "driver_attention", {"ear": 0.3})
    logger.close()

    rotated = tmp_path / "telemetry.1.csv"
    assert rotated.exists()
    assert path.stat().st_size < 400 + 200          # bounded, not 50 rows' worth
    assert rotated.stat().st_size < 400 + 200
    for p in (path, rotated):
        with p.open() as f:
            assert next(csv.reader(f))[0] == "timestamp"  # each generation keeps its header


def test_sibling_telemetry_is_rotated_too_and_keeps_its_own_header(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=0.0, max_bytes=300)
    for i in range(40):
        logger.log_if_due(float(i), f"t{i}", "baby_monitoring", {"a": i, "b": 0.5})
    logger.close()
    rotated = tmp_path / "telemetry_baby_monitoring.1.csv"
    live = tmp_path / "telemetry_baby_monitoring.csv"
    assert rotated.exists()
    with live.open() as f:
        assert next(csv.reader(f)) == ["timestamp", "module", "a", "b"]


# =========================================================== F16: empty files get a header
def test_an_all_damaged_event_log_is_restarted_with_its_header(tmp_path):
    path = tmp_path / "events.csv"
    path.write_bytes(b"\x00" * 64)        # power cut: the only line never finished
    with EventLogger(path) as logger:
        logger.log(an_event())
    rows = list(csv.reader(path.open(newline="")))
    assert rows[0][0] == "timestamp"
    assert rows[1][1] == "fall"


def test_an_empty_event_log_gets_a_header(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("")
    with EventLogger(path) as logger:
        logger.log(an_event())
    assert path.read_text().startswith("timestamp,")


def test_an_empty_telemetry_sibling_gets_a_header(tmp_path):
    (tmp_path / "telemetry_baby_monitoring.csv").write_text("")
    logger = TelemetryLogger(tmp_path / "telemetry.csv", interval_s=0.0)
    logger.log_if_due(0.0, "ts", "baby_monitoring", {"a": 1, "b": 2.0})
    logger.close()
    rows = list(csv.reader((tmp_path / "telemetry_baby_monitoring.csv").open()))
    assert rows[0] == ["timestamp", "module", "a", "b"]


def test_repair_still_reports_what_it_removed(tmp_path):
    path = tmp_path / "events.csv"
    path.write_bytes(b"h\n\x00\x00\n")
    assert repair(path) == 1


# =========================================================== F5: alert keys and levels
def _rule_event(detection="person") -> Event:
    return Event.now(detection, 2.0, 0.9, "occupancy")


def test_an_info_rule_no_longer_silences_a_critical_rule_on_the_same_object():
    manager = AlertManager(sinks=[], cooldown_s=15.0)
    assert manager.notify(_rule_event(), now=100.0, key="rule0:present:person",
                          level=AlertLevel.SOFT) is AlertLevel.SOFT
    assert manager.notify(_rule_event(), now=105.0, key="rule1:in_zone:person",
                          level=AlertLevel.CONTINUOUS) is AlertLevel.CONTINUOUS


def test_duplicates_are_merged_per_key_but_a_new_episode_alerts():
    """M0.1 (D1): previously "the cooldown still applies per key" (SUPPRESSED at
    105 s). A rule's re-confirmation is gated by its own EventEngine cooldown_s;
    the alert layer only merges a duplicate inside the open episode."""
    manager = AlertManager(sinks=[], cooldown_s=15.0)
    manager.notify(_rule_event(), now=100.0, key="rule1", level=AlertLevel.CONTINUOUS)
    assert manager.notify(_rule_event(), now=101.0, key="rule1",
                          level=AlertLevel.CONTINUOUS) is AlertLevel.SUPPRESSED
    assert manager.notify(_rule_event(), now=105.0, key="rule1",
                          level=AlertLevel.CONTINUOUS) is AlertLevel.CONTINUOUS


def test_a_level_override_beats_fixed_levels_and_escalation():
    manager = AlertManager(sinks=[], cooldown_s=1.0, escalation_window_s=60.0,
                           fixed_levels={"person": AlertLevel.SOFT})
    assert manager.notify(_rule_event(), now=0.0, level=AlertLevel.CONTINUOUS) is AlertLevel.CONTINUOUS
    # info rule repeated within the escalation window stays soft when told to
    assert manager.notify(_rule_event("cup"), now=10.0, level=AlertLevel.SOFT) is AlertLevel.SOFT
    assert manager.notify(_rule_event("cup"), now=20.0, level=AlertLevel.SOFT) is AlertLevel.SOFT


def test_the_default_call_signature_is_unchanged():
    manager = AlertManager(sinks=[], cooldown_s=10.0, escalation_window_s=60.0)
    assert manager.notify(_rule_event(), now=0.0) is AlertLevel.SOFT
    # M0.1: a duplicate inside the open episode is merged (was: 5 s -> SUPPRESSED).
    assert manager.notify(_rule_event(), now=1.0) is AlertLevel.SUPPRESSED
    assert manager.notify(_rule_event(), now=20.0) is AlertLevel.CONTINUOUS


def test_a_failing_sink_neither_raises_nor_silences_the_others():
    heard = []

    class Broken:
        def emit(self, event, level):
            raise OSError("stdout gone")

    class Good:
        def emit(self, event, level):
            heard.append(level)

    manager = AlertManager(sinks=[Broken(), Good()])
    assert manager.notify(an_event(), now=0.0) is AlertLevel.SOFT
    assert heard == [AlertLevel.SOFT]


# =========================================================== F6: buzzer overlap
class _FakeBuzzer:
    def __init__(self, pin):
        self.is_on = False
        self.lock = threading.Lock()

    def on(self):
        self.is_on = True

    def off(self):
        self.is_on = False


def test_a_short_buzz_does_not_cut_a_longer_alarm_short(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpiozero", types.SimpleNamespace(Buzzer=_FakeBuzzer))
    from core.alerts import GPIOBuzzerSink

    sink = GPIOBuzzerSink(17)
    sink.emit(an_event("elevated_risk"), AlertLevel.SOFT)        # 0.15s
    time.sleep(0.05)
    sink.emit(an_event("fall"), AlertLevel.CONTINUOUS)           # 0.6s
    time.sleep(0.3)   # past the soft buzz's end, well inside the alarm's
    assert sink._buzzer.is_on, "the soft buzz switched the loud alarm off"
    time.sleep(0.6)
    assert not sink._buzzer.is_on, "the buzzer must still go off at the end"


# =========================================================== F7: counting rules
def test_a_count_rule_keeps_its_hours():
    at_3am = lambda: datetime(2026, 10, 15, 3, 0)  # noqa: E731, DTZ001  # naive = device local
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "count",
                          "schedule": "09:00-17:00", "description": "shop hours"}], clock=at_3am)
    assert drive(engine, frame(OUT_OF_POOL, OUT_OF_POOL)) == []


def test_a_count_seen_before_the_window_opens_must_settle_again_inside_it():
    clock_time = [datetime(2026, 10, 15, 8, 59)]  # noqa: DTZ001  # naive = device local
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "count",
                          "schedule": "09:00-17:00", "description": "x"}],
                        clock=lambda: clock_time[0])
    f = frame(OUT_OF_POOL, OUT_OF_POOL)
    for i in range(10):
        assert engine.observe_frame(f, now=1000.0 + i) == []
    clock_time[0] = datetime(2026, 10, 15, 9, 0)  # noqa: DTZ001
    assert engine.observe_frame(f, now=1010.0) == [], "reported the instant the window opened"
    assert engine.observe_frame(f, now=1011.0) == []
    assert [a.count for a in engine.observe_frame(f, now=1012.5)] == [2]


def test_a_count_rule_counts_inside_its_zone_only():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "count", "zone": "pool",
                          "description": "people in the pool"}], zones=[POOL])
    alerts = drive(engine, frame(IN_POOL, OUT_OF_POOL, OUT_OF_POOL))
    assert [a.count for a in alerts] == [1]


def test_a_count_rule_whose_zone_is_not_drawn_is_inert():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "count", "zone": "pool",
                          "description": "people in the pool"}])
    assert drive(engine, frame(IN_POOL, IN_POOL)) == []


def test_a_count_rule_with_min_count_reports_only_at_or_above_it_and_the_all_clear():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "count", "min_count": 4,
                          "persist_s": 1.0, "description": "more than three people"}])
    sequence = [1, 2, 5, 6, 3, 2]
    alerts = []
    t = 0.0
    for n in sequence:
        for _ in range(4):
            alerts += engine.observe_frame(frame(*([OUT_OF_POOL] * n)), now=t)
            t += 0.5
    assert [(a.count, a.previous_count) for a in alerts] == [(5, 2), (6, 5), (3, 6)]


# =========================================================== F8: absence in a zone
def _absent_in_pool(zones=(POOL,)):
    return RuleEngine([{**BASE_RULE, "target": "person", "predicate": "absent", "zone": "pool",
                        "description": "nobody at the pool"}], zones=list(zones))


def test_absent_in_a_zone_fires_when_the_zone_is_empty():
    assert len(drive(_absent_in_pool(), frame())) == 1


def test_absent_in_a_zone_fires_when_the_only_person_is_elsewhere():
    assert len(drive(_absent_in_pool(), frame(OUT_OF_POOL))) == 1


def test_absent_in_a_zone_does_not_fire_while_someone_is_in_it():
    assert drive(_absent_in_pool(), frame(IN_POOL)) == []


def test_absent_in_an_undrawn_zone_is_inert():
    assert drive(_absent_in_pool(zones=()), frame()) == []


def test_absent_in_a_zone_cannot_judge_a_sighting_without_a_location():
    boxless = DetectionFrame(W, H, (Detection("person", 0.9, None),))
    assert drive(_absent_in_pool(), boxless) == []


# =========================================================== F9: min_count inside a zone
def test_min_count_in_a_zone_counts_only_the_people_in_the_zone():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "present", "zone": "pool",
                          "min_count": 3, "description": "3+ people in the pool"}], zones=[POOL])
    assert drive(engine, frame(IN_POOL, OUT_OF_POOL, OUT_OF_POOL)) == []
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "present", "zone": "pool",
                          "min_count": 3, "description": "3+ people in the pool"}], zones=[POOL])
    assert len(drive(engine, frame(IN_POOL, IN_POOL, IN_POOL))) == 1


def test_a_plain_zone_rule_still_fires_with_one_person_in_the_zone():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "in_zone", "zone": "pool",
                          "description": "someone in the pool"}], zones=[POOL])
    assert len(drive(engine, frame(IN_POOL, OUT_OF_POOL))) == 1


# =========================================================== F10: stillness
def _still_rule(persist_s=5.0):
    return RuleEngine([{**BASE_RULE, "target": "person", "predicate": "still",
                        "persist_s": persist_s, "description": "not moving"}])


def test_a_person_walking_the_whole_time_is_not_still():
    def walking(i):
        x = 10.0 + 40.0 * i            # 80 px/s with a ~210 px diagonal
        return frame((x, 10.0, x + 100.0, 200.0))
    assert drive(_still_rule(), walking, seconds=15.0) == []


def test_a_person_who_stays_put_is_still():
    def jitter(i):
        dx = 3.0 if i % 2 else -3.0    # detector jitter, a few pixels
        return frame((300.0 + dx, 300.0, 400.0 + dx, 600.0))
    assert len(drive(_still_rule(), jitter, seconds=15.0)) == 1


@pytest.mark.parametrize("step", [0.04, 0.5, 1.0])
def test_stillness_does_not_depend_on_frame_rate(step):
    still = drive(_still_rule(3.0), frame((300.0, 300.0, 400.0, 600.0)), seconds=10.0, step=step)
    assert len(still) == 1

    def drifting(i):
        x = 10.0 + 60.0 * (i * step)   # 60 px/s regardless of how often it is sampled
        return frame((x, 300.0, x + 100.0, 600.0))
    assert drive(_still_rule(3.0), drifting, seconds=10.0, step=step) == []


def test_movement_restarts_the_still_clock():
    def stop_go(i):
        t = i * 0.5
        x = 300.0 if (t % 6.0) < 4.0 else 300.0 + 120.0 * ((t % 6.0) - 4.0)
        return frame((x, 300.0, x + 100.0, 600.0))
    # never still for 5 continuous seconds
    assert drive(_still_rule(5.0), stop_go, seconds=30.0) == []


# =========================================================== F13: rule timezone
def _night_rule(**extra):
    return {**BASE_RULE, "target": "person", "predicate": "present",
            "schedule": "23:00-06:00", "description": "someone at night", **extra}


def test_a_rule_reads_its_hours_in_its_own_timezone():
    utc_7am = lambda: datetime(2026, 10, 15, 7, 0, tzinfo=UTC)  # noqa: E731  # 00:00 in Los Angeles
    in_la = RuleEngine([_night_rule(timezone="America/Los_Angeles")], clock=utc_7am)
    assert len(drive(in_la, frame(OUT_OF_POOL))) == 1
    in_utc = RuleEngine([_night_rule(timezone="UTC")], clock=utc_7am)
    assert drive(in_utc, frame(OUT_OF_POOL)) == []


def test_the_engine_default_timezone_applies_to_rules_without_one():
    utc_7am = lambda: datetime(2026, 10, 15, 7, 0, tzinfo=UTC)  # noqa: E731
    engine = RuleEngine([_night_rule()], clock=utc_7am, timezone="America/Los_Angeles")
    assert len(drive(engine, frame(OUT_OF_POOL))) == 1
    engine.set_timezone("Europe/London")       # 08:00 there
    engine.replace_rules([_night_rule()])
    assert drive(engine, frame(OUT_OF_POOL), start=5000.0) == []


def test_an_unknown_timezone_falls_back_to_device_time():
    at_midnight = lambda: datetime(2026, 10, 15, 0, 0)  # noqa: E731, DTZ001  # naive = device local
    engine = RuleEngine([_night_rule(timezone="Mars/Olympus_Mons")], clock=at_midnight)
    assert len(drive(engine, frame(OUT_OF_POOL))) == 1


# =========================================================== F14: relations with oneself
def test_one_person_is_not_two_people_close_together():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "near",
                          "secondary": "person", "description": "two people close together"}])
    assert drive(engine, frame((10.0, 10.0, 110.0, 300.0))) == []


def test_two_overlapping_people_are_close_together():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "near",
                          "secondary": "person", "description": "two people close together"}])
    assert len(drive(engine, frame((10.0, 10.0, 110.0, 300.0), (90.0, 10.0, 190.0, 300.0)))) == 1


def test_different_labels_still_relate_as_before():
    engine = RuleEngine([{**BASE_RULE, "target": "dog", "predicate": "near", "secondary": "couch",
                          "description": "dog on couch"}])
    f = DetectionFrame(W, H, (Detection("dog", 0.9, (100, 100, 200, 200)),
                              Detection("couch", 0.9, (150, 150, 400, 300))))
    assert len(drive(engine, f)) == 1


# =========================================================== F19: silent rules and the red screen
def test_a_silent_rule_is_tracked_but_does_not_raise_the_alert_state():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "present",
                          "notify": "silent", "description": "bookkeeping"}])
    assert drive(engine, frame(OUT_OF_POOL)) == []
    assert engine.active_alerts(), "still tracked"
    assert engine.is_alert_active() is False


def test_a_notifying_rule_still_raises_the_alert_state():
    engine = RuleEngine([{**BASE_RULE, "target": "person", "predicate": "present",
                          "description": "someone"}])
    drive(engine, frame(OUT_OF_POOL))
    assert engine.is_alert_active() is True


# =========================================================== F20: restarted tracks
def test_a_track_restarted_after_a_gap_can_confirm_like_a_new_one():
    spec = RuleSpec(detection="x", min_confidence=0.6, persist_s=1.0, min_frames=3)

    def frames_to_confirm(engine, start):
        for i in range(200):
            if any(t.state == CONFIRMED for t in engine.update("x", True, 0.6, now=start + i * 0.1)):
                return i + 1
        return None

    fresh = EventEngine({"x": spec})
    restarted = EventEngine({"x": spec})
    restarted.update("x", True, 0.6, now=0.0)          # seen once, then a 1.5s gap
    assert frames_to_confirm(restarted, 1.5) == frames_to_confirm(fresh, 0.0)


# =========================================================== F12: training capture manifest
def _store(directory):
    return CaptureStore({"enabled": True, "dir": str(directory), "min_interval_s": 0.0,
                         "max_samples": 500})


def _write(path, frame):
    Path(path).write_bytes(b"jpg")


def test_a_torn_manifest_line_hides_nothing_after_it(tmp_path):
    first = _store(tmp_path)
    for i in range(3):
        first.capture(None, event_type="fall", now=1000.0 + i, writer=_write)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + '{"id": "torn", "sess')  # power cut mid-append

    second = _store(tmp_path)
    assert second.count == 3
    second.capture(None, event_type="fall", now=2000.0, writer=_write)

    third = _store(tmp_path)
    assert third.count == 4, "a sample appended after the torn line was lost"
    for line in manifest.read_text().splitlines():
        if line.startswith('{"id": "torn"'):
            assert line == '{"id": "torn", "sess', "the new record was glued onto the fragment"


def test_purge_deletes_frames_the_manifest_lost_track_of(tmp_path):
    store = _store(tmp_path)
    for i in range(2):
        store.capture(None, event_type="fall", now=1000.0 + i, writer=_write)
    (tmp_path / "manifest.jsonl").write_text("")               # manifest lost entirely
    unrelated = tmp_path / "holiday.jpg"
    unrelated.write_bytes(b"not ours")

    fresh = _store(tmp_path)
    assert fresh.purge() == 2
    assert [p.name for p in tmp_path.glob("*.jpg")] == ["holiday.jpg"]


def test_a_manifest_record_with_an_unknown_field_still_loads(tmp_path):
    store = _store(tmp_path)
    sample = store.capture(None, event_type="fall", now=1.0, writer=_write)
    manifest = tmp_path / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    record["added_by_a_newer_version"] = True
    manifest.write_text(json.dumps(record) + "\n")
    assert [s.id for s in _store(tmp_path).samples()] == [sample.id]


# =========================================================== F15: reports
_HEADER = "timestamp,type,duration_s,confidence,module,contributing_signals\r\n"


def test_the_weekly_report_covers_the_last_seven_days_only(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(_HEADER
                    + "2025-01-01T03:00:00+00:00,fall,1.000,0.9,fall_detection,\r\n"
                    + "2026-10-14T03:30:00+00:00,fall,1.000,0.9,fall_detection,\r\n")
    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    assert [r["timestamp"][:4] for r in last_days(load_events(path), now=now)] == ["2026"]


def test_report_buckets_are_in_the_requested_local_zone():
    from zoneinfo import ZoneInfo

    rows = [{"timestamp": "2026-10-14T03:30:00+00:00", "type": "fall"}]
    stats = summarize(rows, tz=ZoneInfo("America/New_York"))
    assert dict(stats["per_hour"]) == {23: 1}
    assert dict(stats["per_day"]) == {"2026-10-13": 1}


def test_a_torn_row_does_not_abort_the_report(tmp_path):
    path = tmp_path / "events.csv"
    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    path.write_text(_HEADER + "2026-10-14T03:30:00+00:00,fall,1.000,0.9,fall_detection,\r\n"
                    + "2026-10-14T0")
    report = generate_weekly_report(path, tmp_path / "out", now=now, tz=UTC)
    assert report.exists()


def test_an_empty_log_still_produces_a_report(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(_HEADER)
    assert generate_weekly_report(path, tmp_path / "out").exists()


# =========================================================== F17: monitoring.json shapes
@pytest.mark.parametrize("content", [
    "null", "[]", '"rules"',
    '{"rules": [{"target": "dog"}], "updated_at": "yesterday", "overrides_mode": true}',
])
def test_a_wrongly_shaped_monitoring_file_never_crashes_boot(tmp_path, content):
    path = tmp_path / "monitoring.json"
    path.write_text(content)
    load_or_empty(path)  # must not raise


def test_only_rule_objects_are_restored_and_only_a_real_true_overrides(tmp_path):
    path = tmp_path / "monitoring.json"
    path.write_text('{"rules": {"target": "dog"}, "overrides_mode": "false"}')
    config = load_or_empty(path)
    assert config.rules == []
    assert config.active is False

    path.write_text('{"rules": [{"target": "dog"}, "junk", 3], "overrides_mode": "false"}')
    config = load_or_empty(path)
    assert config.rules == [{"target": "dog"}]
    assert config.overrides_mode is False


def test_a_well_formed_monitoring_file_still_restores(tmp_path):
    path = tmp_path / "monitoring.json"
    path.write_text(json.dumps({"rules": [{"target": "dog"}], "overrides_mode": True,
                                "updated_at": 12.5, "instruction": "watch the dog"}))
    config = load_or_empty(path)
    assert config.active and config.updated_at == 12.5 and config.instruction == "watch the dog"


# =========================================================== F18: alert config validation
def _repo_config():
    return yaml.safe_load((REPO / "config.yaml").read_text())


def test_the_repo_config_still_validates_cleanly():
    assert validate(_repo_config()) == ([], [])


@pytest.mark.parametrize(("key", "value"), [
    ("loud_types", "fall"),
    ("soft_types", "fatigue_warning"),
    ("sinks", "gpio_buzzer"),
    ("sinks", [17]),
    ("escalation_window_s", "a minute"),
    ("gpio_buzzer_pin", "17"),
])
def test_scalar_or_mistyped_alert_settings_are_errors(key, value):
    config = _repo_config()
    config["alerts"][key] = value
    errors, _ = validate(config)
    assert any(f"alerts.{key}" in e for e in errors), errors


def test_mac_max_alerts_are_checked_too():
    config = _repo_config()
    config["mac_max"]["alerts"]["loud_types"] = "drowsiness"
    errors, _ = validate(config)
    assert any("mac_max.alerts.loud_types" in e for e in errors)


def test_an_unknown_sink_is_a_warning():
    config = _repo_config()
    config["alerts"]["sinks"] = ["console", "siren"]
    errors, warnings = validate(config)
    assert not errors
    assert any("siren" in w for w in warnings)


# =========================================================== F2/F4: config migration
_SHAPES = {
    "one line": "alerts:\n  loud_types: [drowsiness, phone_use]\n  soft_types: [fatigue_warning]\n",
    "one line after a scalar": "alerts:\n  cooldown_s: 15\n  loud_types: [drowsiness] # tone\n",
    "one line, empty": "alerts:\n  loud_types: []\n",
    "one line, trailing comma": "alerts:\n  loud_types: [a, b,]\n",
    "bracket alone, no comma": "alerts:\n  loud_types: [\n    a,\n    b\n  ]\n  cooldown_s: 15\n",
    "bracket alone, comma": "alerts:\n  loud_types: [\n    a,\n    b,\n  ]\n",
}


@pytest.mark.parametrize("name", sorted(_SHAPES))
def test_list_migration_leaves_a_parseable_config_for_every_list_shape(tmp_path, name):
    path = tmp_path / "config.yaml"
    path.write_text(_SHAPES[name])
    original = yaml.safe_load(_SHAPES[name])["alerts"]
    migrate(path)
    parsed = yaml.safe_load(path.read_text())["alerts"]
    members = [m for dotted, m, _ in LIST_MIGRATIONS if dotted == "alerts.loud_types"]
    assert all(m in parsed["loud_types"] for m in members)
    assert all(item in parsed["loud_types"] for item in original["loud_types"])
    for key, value in original.items():
        if key != "loud_types":
            assert parsed[key] == value, f"{key} changed"
    assert migrate(path) == []  # converges


def test_distraction_and_eyes_up_reach_deployed_devices():
    members = {m for dotted, m, _ in LIST_MIGRATIONS if dotted == "alerts.loud_types"}
    assert {"distraction", "eyes_up"} <= members


# =========================================================== F21: taxonomy honesty
def test_the_taxonomy_matches_what_the_device_does():
    from core import taxonomy

    for mode in ("elderly_monitoring", "security_monitoring"):
        schedule = taxonomy.find(mode, "schedule")
        assert schedule.support != taxonomy.PLANNED, "rules.py enforces schedules today"
    presence = taxonomy.find("elderly_monitoring", "presence")
    assert "YOLO" not in presence.how, "fall_detection runs MediaPipe Pose, no object detector"
    posture = taxonomy.find("elderly_monitoring", "posture")
    assert "not implemented" not in posture.caveat


def test_the_generated_capability_doc_is_current(tmp_path):
    import runpy

    script = REPO / "scripts" / "generate_detections_doc.py"
    source = script.read_text().replace(
        'out = Path(__file__).resolve().parent.parent / "docs" / "DETECTIONS.md"',
        f"out = Path({str(tmp_path / 'DETECTIONS.md')!r})",
    )
    shim = tmp_path / "gen.py"
    shim.write_text(source)
    runpy.run_path(str(shim), run_name="__main__")
    assert (tmp_path / "DETECTIONS.md").read_text() == (REPO / "docs" / "DETECTIONS.md").read_text()


def test_stillness_is_described_to_the_interpreter():
    from core.capabilities import default_registry

    sheet = default_registry().describe_for_llm()
    assert "'still' means" in sheet
