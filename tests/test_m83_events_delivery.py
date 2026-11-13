"""M8.3 events: identity, alert-delivery evidence and the durable backlog, device side.

What the device knows about one event, hop by hop (docs/EVENT_PIPELINE.md):

  * the Event's own `event_id` IS the uplink's `event_key`, so the alert
    policy's record, the spool line and the stored row share one identity;
  * the alert policy records what it did with every event (alerted / merged /
    suppressed) and what each in-cabin output REPORTED -- "sounded" only when a
    real output said its whole pattern rendered;
  * an event merged into an alert episode whose event is already queued is not
    queued again (one episode, one stored alert, one notification);
  * a late output report goes up under the same event_key;
  * the spool never loses an event to a torn append or a compaction while a
    batch is in flight, and a restore never reorders the backlog.

Deterministic: injected clocks, scripted transports, sinks that report on demand.
Nothing here verifies physical sound; that needs the Pi (docs/MONDAY_RUNBOOK.md).
"""

from __future__ import annotations

import json
import time
import types

import pytest
import requests

from core.alerts import (
    PATTERNS as _PATTERNS,
)
from core.alerts import (
    AlertLevel,
    AlertManager,
    ConsoleSink,
    GPIOBuzzerSink,
    UnavailableSink,
    _PatternPlayer,
)
from core.events import Event
from core.uplink import CloudUplink


# ---------------------------------------------------------------------- outputs
class ReportingBuzzer:
    """An in-cabin output that reports each actuation, now or when released."""

    name, kind, audible = "test_buzzer", "buzzer", True
    reports_completion = True

    def __init__(self, *, hold: bool = False, fail: str | None = None, simulated: bool = False) -> None:
        self.hold, self.fail, self.simulated = hold, fail, simulated
        self.held: list = []
        self.played: list[str] = []

    def play(self, pattern, event, category, report=None) -> None:
        self.played.append(event.type)
        outcome = ("failed", self.fail) if self.fail else ("played", None)
        if self.hold:
            self.held.append((report, outcome))
        elif report is not None:
            report(*outcome)

    def release(self) -> None:
        held, self.held = self.held, []
        for report, outcome in held:
            report(*outcome)

    def emit(self, event, level) -> None:  # pragma: no cover - play() is used
        raise AssertionError("play() is the interface")


class LegacyBuzzer:
    """An audible output from before M8.3: plays, cannot say whether it rendered."""

    name, kind, audible = "legacy_buzzer", "buzzer", True

    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, pattern, event, category) -> None:
        self.played.append(event.type)

    def emit(self, event, level) -> None:  # pragma: no cover
        raise AssertionError


def _manager(*sinks) -> AlertManager:
    return AlertManager(list(sinks), fixed_levels={"drowsiness": AlertLevel.CONTINUOUS, "fall": AlertLevel.CONTINUOUS},
                        type_categories={"drowsiness": "critical", "fall": "critical"})


def _d(manager: AlertManager, event: Event):
    """The policy's record of what it did with `event` (it must have one)."""
    delivery = manager.delivery(event)
    assert delivery is not None, "notify() records a delivery for every event it is handed"
    return delivery


def _event(kind: str = "drowsiness") -> Event:
    return Event.now(kind, 2.0, 0.9, "driver_attention")


# ---------------------------------------------------------------------- identity
def test_every_event_has_its_own_stable_id_outside_equality():
    a, b = _event(), _event()
    assert a.event_id and b.event_id and a.event_id != b.event_id
    assert len(a.event_id) == 32 and int(a.event_id, 16) >= 0
    clone = Event(a.timestamp, a.type, a.duration, a.confidence, a.module_name, a.contributing_signals)
    assert clone == a, "the id is identity, not content: equality is unchanged"
    assert "event_id" not in repr(a)


def test_the_uplink_event_key_is_the_events_own_id():
    """Before M8.3 the key was a fresh uuid minted in push_event: the alert
    policy, the spool and the stored row could not be tied to one occurrence."""
    uplink = _uplink()
    event = _event()
    uplink.push_event(event, "loud")
    assert uplink._queue[-1]["event_key"] == event.event_id


# ---------------------------------------------------------------------- delivery evidence
def test_an_output_that_reports_played_is_the_only_way_to_sounded():
    buzzer = ReportingBuzzer(hold=True)
    manager = _manager(buzzer)
    event = _event()
    manager.notify(event, now=0.0)
    delivery = _d(manager, event)
    assert delivery is not None and delivery.decision == "alerted"
    assert delivery.output() == "pending", "commanded is not sounded"
    buzzer.release()
    summary = delivery.summary()
    assert summary["output"] == "sounded"
    assert summary["outputs"][0]["played"] == 1 and summary["outputs"][0]["name"] == "test_buzzer"
    assert summary["episode"] and summary["category"] == "critical"


def test_a_failed_output_is_failed_never_sounded():
    manager = _manager(ReportingBuzzer(fail="GPIO17: device or resource busy"))
    event = _event()
    manager.notify(event, now=0.0)
    summary = _d(manager, event).summary()
    assert summary["output"] == "failed"
    assert summary["outputs"][0]["error"] == "GPIO17: device or resource busy"


def test_a_sink_that_raises_is_failed():
    class Raising(LegacyBuzzer):
        def play(self, pattern, event, category) -> None:
            raise OSError("pin claimed by another process")

    manager = _manager(Raising())
    event = _event()
    manager.notify(event, now=0.0)
    assert _d(manager, event).output() == "failed"


def test_an_output_that_cannot_report_is_unconfirmed_never_sounded():
    manager = _manager(LegacyBuzzer())
    event = _event()
    manager.notify(event, now=0.0)
    assert _d(manager, event).output() == "unconfirmed"


def test_the_console_is_not_an_alert_output():
    manager = _manager(ConsoleSink())
    event = _event()
    manager.notify(event, now=0.0)
    delivery = _d(manager, event)
    assert delivery.decision == "alerted" and delivery.output() == "none"
    assert delivery.summary()["outputs"] == []


def test_an_unavailable_output_reports_skipped_with_its_reason():
    manager = _manager(UnavailableSink("gpio_buzzer", "buzzer", True, "gpiozero not installed"))
    event = _event()
    manager.notify(event, now=0.0)
    summary = _d(manager, event).summary()
    assert summary["output"] == "failed"
    assert summary["outputs"][0]["skipped"] == 1
    assert "gpiozero not installed" in summary["outputs"][0]["error"]


def test_a_simulated_output_never_reads_as_sounded():
    manager = _manager(ReportingBuzzer(simulated=True))
    event = _event()
    manager.notify(event, now=0.0)
    summary = _d(manager, event).summary()
    assert summary["output"] == "simulated"
    assert summary["outputs"][0]["simulated"] is True


def test_a_repeat_that_plays_after_a_failed_onset_counts():
    buzzer = ReportingBuzzer(fail="busy")
    manager = _manager(buzzer)
    event = _event()
    manager.notify(event, now=0.0)
    assert _d(manager, event).output() == "failed"
    buzzer.fail = None
    manager.observe(4.0, active={"drowsiness"}, tracked={"drowsiness"})  # critical repeats every 4 s
    assert _d(manager, event).output() == "sounded"


def test_a_merged_and_a_suppressed_event_actuate_nothing_and_say_so():
    buzzer = ReportingBuzzer()
    manager = AlertManager([buzzer], cooldown_s=15.0, fixed_levels={"fatigue_warning": AlertLevel.SOFT},
                           type_categories={"fatigue_warning": "notice"})
    first, again = _event("fall"), _event("fall")
    manager.notify(first, now=0.0)
    assert manager.notify(again, now=0.5) is AlertLevel.SUPPRESSED
    merged = _d(manager, again)
    assert merged.decision == "merged" and merged.episode == _d(manager, first).episode
    assert merged.output() == "none"
    manager.notify(_event("fatigue_warning"), now=0.0)
    notice = _event("fatigue_warning")
    manager.notify(notice, now=5.0)
    assert _d(manager, notice).decision == "suppressed" and _d(manager, notice).episode is None


def test_episode_refs_differ_across_boots():
    one, two = _manager(ReportingBuzzer()), _manager(ReportingBuzzer())
    a, b = _event(), _event()
    one.notify(a, now=0.0)
    two.notify(b, now=0.0)
    assert _d(one, a).episode.endswith("-1") and _d(two, b).episode.endswith("-1")
    assert _d(one, a).episode != _d(two, b).episode, "episode 1 of another boot is another episode"


def test_the_pattern_player_reports_played_failed_and_displaced():
    results: list[tuple[str, str | None]] = []
    ok = _PatternPlayer(lambda on_s: None)
    ok.play(_PATTERNS["chirp"], "notice", lambda r, e: results.append((r, e)))
    broken = _PatternPlayer(lambda on_s: "write failed")
    broken.play(_PATTERNS["chirp"], "notice", lambda r, e: results.append((r, e)))
    deadline = time.monotonic() + 5.0
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sorted(results, key=str) == [("failed", "write failed"), ("played", None)]

    # Displacement is decided synchronously, before any rendering.
    gate = types.SimpleNamespace(calls=0)

    def render(on_s):
        gate.calls += 1

    player = _PatternPlayer(render)
    player._thread = types.SimpleNamespace(is_alive=lambda: True)  # a worker "busy": nothing renders
    dropped: list[str] = []
    player.play(_PATTERNS["chirp"], "critical", lambda r, e: dropped.append("critical:" + r))
    player.play(_PATTERNS["chirp"], "critical", lambda r, e: dropped.append("critical2:" + r))
    player.play(_PATTERNS["chirp"], "notice", lambda r, e: dropped.append("notice:" + r))
    assert dropped == ["notice:skipped"], "a less urgent pattern is skipped, and says so"


def test_a_gpio_write_error_is_returned_not_swallowed():
    sink = GPIOBuzzerSink.__new__(GPIOBuzzerSink)
    sink._lock, sink._off_at, sink._last_error, sink._pin = __import__("threading").Lock(), 0.0, None, 17

    class DeadPin:
        def on(self):
            raise OSError("GPIO17 not exported")

        def off(self):
            pass

    sink._buzzer = DeadPin()
    assert sink._buzz(0.0) == "GPIO17 not exported"
    sink._buzzer = types.SimpleNamespace(on=lambda: None, off=lambda: None)
    assert sink._buzz(0.0) is None


# ---------------------------------------------------------------------- uplink
class _Backend:
    """Stores by event_key as cloud/app.py does; can be down or lose responses."""

    def __init__(self) -> None:
        self.down = False
        self.posts: list[dict] = []
        self.on_post = None

    def post(self, url, json=None, **_kwargs):
        if self.on_post is not None:
            self.on_post()
        if self.down:
            raise requests.ConnectionError("SIMULATED: backend unreachable")
        self.posts.append(json)
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=dict)


def _uplink(spool=None, backend=None, queue_max=500) -> CloudUplink:
    config = {"base_url": "https://backend.example", "device_key": "dk_" + "x" * 32, "poll_interval_s": 2.0,
              "queue_max": queue_max, "command_auth": "off"}
    if spool is not None:
        config["spool_path"] = str(spool)
    uplink = CloudUplink(config)
    uplink._requests = backend or _Backend()
    return uplink


def _sent(backend) -> list[dict]:
    return [e for body in backend.posts for e in body["events"]]


def test_an_event_merged_into_a_queued_episode_is_not_queued_again():
    """Before M8.3 every alertable event was queued, merged or not: a detector
    re-reporting an open episode (after a reconnect, a module rebuild) became a
    second stored alert and a second notification for one warning."""
    buzzer = ReportingBuzzer()
    manager = _manager(buzzer)
    backend = _Backend()
    uplink = _uplink(backend=backend)
    uplink.delivery_lookup = manager.delivery
    first, again = _event("fall"), _event("fall")
    for event, now in ((first, 0.0), (again, 0.5)):
        manager.notify(event, now=now)
        uplink.push_event(event, "loud")
    uplink.sync_once()
    sent = _sent(backend)
    assert [e["event_key"] for e in sent] == [first.event_id]
    assert sent[0]["alert"]["decision"] == "alerted" and sent[0]["alert"]["output"] == "sounded"
    assert uplink.merged_events == 1
    assert buzzer.played == ["fall"], "and the policy actuated once"


def test_an_event_merged_into_an_episode_with_no_event_of_its_own_is_its_record():
    """The runtime re-opens a live episode after a recovery with a synthetic
    event it never uploads (services/runtime.py _reopen_live_episodes). If the
    detector then reports it, that report is the episode's only record: it is
    sent, carrying what the episode actually actuated, never dropped."""
    buzzer = ReportingBuzzer()
    manager = _manager(buzzer)
    backend = _Backend()
    uplink = _uplink(backend=backend)
    uplink.delivery_lookup = manager.delivery
    reopened = Event.now("drowsiness", 0.0, None, "driver_attention", "condition still active")
    manager.notify(reopened, now=0.0)  # not pushed, as in the runtime
    report = _event()
    manager.notify(report, now=1.0)
    uplink.push_event(report, "loud")
    uplink.sync_once()
    sent = _sent(backend)
    assert [e["event_key"] for e in sent] == [report.event_id]
    assert sent[0]["alert"]["output"] == "sounded" and sent[0]["alert"]["episode"] == \
        _d(manager, reopened).episode
    assert uplink.merged_events == 0


def test_a_late_output_report_goes_up_under_the_same_key():
    buzzer = ReportingBuzzer(hold=True)
    manager = _manager(buzzer)
    backend = _Backend()
    uplink = _uplink(backend=backend)
    uplink.delivery_lookup = manager.delivery
    event = _event()
    manager.notify(event, now=0.0)
    uplink.push_event(event, "loud")
    uplink.sync_once()
    assert _sent(backend)[0]["alert"]["output"] == "pending", "sent at once, not held for the buzzer"
    buzzer.release()  # the pattern finished rendering
    uplink.sync_once()
    sent = _sent(backend)
    assert [e["event_key"] for e in sent] == [event.event_id, event.event_id]
    assert sent[1]["alert"]["output"] == "sounded" and sent[1]["alert"]["rev"] > sent[0]["alert"]["rev"]
    assert {k: v for k, v in sent[1].items() if k not in ("alert", "age_s")} == \
        {k: v for k, v in sent[0].items() if k not in ("alert", "age_s")}, "the same event, a newer report"


def test_a_report_that_lands_while_still_queued_updates_the_queued_item(tmp_path):
    buzzer = ReportingBuzzer(hold=True)
    manager = _manager(buzzer)
    backend = _Backend()
    backend.down = True
    spool = tmp_path / "uplink_spool.jsonl"
    uplink = _uplink(spool, backend)
    uplink.delivery_lookup = manager.delivery
    event = _event()
    manager.notify(event, now=0.0)
    uplink.push_event(event, "loud")
    buzzer.release()
    assert len(uplink._queue) == 1 and uplink._queue[0]["alert"]["output"] == "sounded"
    # The restored backlog after a power cut has the newer report too, once.
    restored = _uplink(spool, backend)
    assert [e["alert"]["output"] for e in restored._queue] == ["sounded"]


def test_age_is_sent_for_this_boot_only(tmp_path, monkeypatch):
    clock = types.SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    backend = _Backend()
    backend.down = True
    spool = tmp_path / "uplink_spool.jsonl"
    uplink = _uplink(spool, backend)
    uplink.push_event(_event(), "loud")
    clock.now = 160.0
    backend.down = False
    uplink.sync_once()
    assert _sent(backend)[0]["age_s"] == pytest.approx(60.0), "an hour late is an hour old, on the device's own clock"
    assert not any(k.startswith("_") for k in _sent(backend)[0]), "internal keys never go on the wire"

    backend.down = True
    uplink.push_event(_event(), "loud")
    uplink.sync_once()
    restarted = _uplink(spool, backend)  # a new boot: the monotonic origin changed
    backend.down = False
    restarted.sync_once()
    assert "age_s" not in _sent(backend)[-1], "a restored event has no age to give"


# ---------------------------------------------------------------------- spool
def test_a_torn_append_does_not_cost_the_next_event(tmp_path, monkeypatch):
    """Before M8.3: an append that failed part-way (disk full) left half a line,
    and the NEXT event was appended onto it -- unreadable on restore. One torn
    write cost two events, silently."""
    import core.uplink as uplink_module

    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    backend.down = True
    uplink = _uplink(spool, backend)
    uplink.push_event(_event("a"), "loud")

    real_open = open
    tear = types.SimpleNamespace(armed=True)

    class TornHandle:
        def __init__(self, handle):
            self._handle = handle

        def write(self, text):
            if tear.armed:
                tear.armed = False
                self._handle.write(text[: len(text) // 2])
                self._handle.flush()
                raise OSError(28, "No space left on device")
            return self._handle.write(text)

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

    def torn_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return TornHandle(handle) if "a" in mode else handle

    monkeypatch.setattr(uplink_module, "open", torn_open, raising=False)
    uplink.push_event(_event("b"), "loud")  # torn on disk, kept in memory
    uplink.push_event(_event("c"), "loud")
    monkeypatch.undo()
    restored = _uplink(spool, backend)  # power cut, power on
    assert [e["type"] for e in restored._queue] == ["a", "b", "c"]


def test_a_rewrite_while_a_batch_is_in_flight_keeps_it_on_disk(tmp_path):
    """The backlog on disk is the queue AND the batch being sent. A rewrite
    during the POST (compaction, a late alert report) used to write the queue
    alone; the POST then failed and a power cut lost the batch."""
    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    uplink = _uplink(spool, backend, queue_max=3)
    uplink.push_event(_event("a"), "loud")
    uplink.push_event(_event("b"), "loud")

    def during_post():
        backend.down = True
        with uplink._lock:
            uplink._spool_rewrite()  # what a compaction or a dirty-spool rewrite does

    backend.on_post = during_post
    uplink.sync_once()  # the POST fails
    lines = [json.loads(line)["type"] for line in spool.read_text().splitlines()]
    assert lines == ["a", "b"]
    restored = _uplink(spool, _Backend())
    assert [e["type"] for e in restored._queue] == ["a", "b"]


def test_a_restore_keeps_first_position_and_latest_content(tmp_path):
    spool = tmp_path / "uplink_spool.jsonl"
    a = {"event_key": "k1", "type": "a", "alert": {"output": "pending", "rev": 1}}
    b = {"event_key": "k2", "type": "b"}
    a2 = {**a, "alert": {"output": "sounded", "rev": 2}}
    spool.write_text("".join(json.dumps(x) + "\n" for x in (a, b, a2)))
    restored = _uplink(spool, _Backend())
    assert [(e["type"], (e.get("alert") or {}).get("output")) for e in restored._queue] == \
        [("a", "sounded"), ("b", None)], "the backlog keeps the order events happened in"
