"""M12: the device's local event journal behind GET /api/events (core/local_events.py).

One identity per occurrence (the backend's event_key), honest sync state,
bounded on the SD card, no image data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from core import local_auth as la
from core.dashboard import DashboardServer
from core.events import Event
from core.local_events import LocalEventJournal
from sim.local_client import LocalPhone


def _event(kind="distraction", **kw) -> Event:
    return Event(datetime.now(UTC), kind, kw.pop("duration", 2.5), kw.pop("confidence", 0.81), "driver_attention",
                 kw.pop("signals", "gaze_off"))


class _Delivery:
    def __init__(self):
        self.rev = 1

    def summary(self):
        return {"decision": "alerted", "rev": self.rev, "outputs": {"buzzer": "sounded" if self.rev > 1 else "pending"}}


def test_rows_carry_the_events_own_id_and_no_image_fields(tmp_path):
    journal = LocalEventJournal(tmp_path / "j.jsonl")
    event = _event()
    journal.record(event, "loud")
    page = journal.page()
    row = page["events"][0]
    assert row["event_key"] == event.event_id and row["type"] == "distraction" and row["severity"] == "loud"
    assert row["sync"] == "local_only" and row["age_s"] is not None and row["simulated"] is False
    allowed = {"seq", "event_key", "type", "timestamp", "severity", "confidence", "signals", "duration",
               "experimental", "simulated", "alert", "age_s", "sync"}
    assert set(row) == allowed
    text = (tmp_path / "j.jsonl").read_text()
    for word in ("jpeg", "jpg", "frame", "image", "snapshot", "base64"):
        assert word not in text.lower()


def test_sync_state_follows_the_uplink_backlog(tmp_path):
    journal = LocalEventJournal(tmp_path / "j.jsonl")
    a, b = _event(), _event()
    journal.record(a, "loud")
    journal.record(b, "loud")
    rows = {r["event_key"]: r["sync"] for r in journal.page(pending_keys={b.event_id}, uplink=True)["events"]}
    assert rows == {a.event_id: "sent", b.event_id: "pending"}


def test_experimental_observations_are_always_log(tmp_path):
    journal = LocalEventJournal(tmp_path / "j.jsonl")
    journal.record(_event("high_risk"), "loud")
    row = journal.page()["events"][0]
    assert row["experimental"] is True and row["severity"] == "log"


def test_a_later_alert_report_reaches_the_served_row(tmp_path):
    journal = LocalEventJournal(tmp_path / "j.jsonl")
    delivery = _Delivery()
    journal.record(_event(), "loud", delivery=delivery)
    assert journal.page()["events"][0]["alert"]["outputs"]["buzzer"] == "pending"
    delivery.rev = 2
    assert journal.page()["events"][0]["alert"]["outputs"]["buzzer"] == "sounded"


def test_after_cursor_returns_only_new_rows_oldest_first(tmp_path):
    journal = LocalEventJournal(tmp_path / "j.jsonl")
    events = [_event() for _ in range(5)]
    for e in events:
        journal.record(e, "soft")
    page = journal.page(after=2)
    assert [r["seq"] for r in page["events"]] == [3, 4, 5]
    assert page["latest_seq"] == 5
    assert [r["seq"] for r in journal.page(limit=2)["events"]] == [5, 4]


def test_the_journal_survives_a_restart_a_torn_line_and_keeps_its_sequence(tmp_path):
    path = tmp_path / "j.jsonl"
    first = LocalEventJournal(path)
    kept = _event()
    first.record(kept, "loud")
    with path.open("a") as handle:
        handle.write('{"seq": 2, "event_ke')  # a power cut mid-write
    second = LocalEventJournal(path)
    page = second.page()
    assert [r["event_key"] for r in page["events"]] == [kept.event_id]
    assert page["events"][0]["age_s"] is None  # another boot: no monotonic age
    second.record(_event(), "soft")
    assert second.page()["latest_seq"] == 2


def test_the_journal_is_bounded(tmp_path):
    path = tmp_path / "j.jsonl"
    journal = LocalEventJournal(path, max_events=10, max_bytes=10_000)
    for _ in range(100):
        journal.record(_event(signals="x" * 200), "soft")
    page = journal.page(limit=500)
    assert page["retention"]["kept"] <= 10 and page["latest_seq"] == 100
    assert len(path.read_text().splitlines()) <= 13
    assert path.stat().st_size <= 10_000


def test_a_journal_that_cannot_write_never_costs_the_event(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    journal = LocalEventJournal(blocker / "j.jsonl")
    assert journal.record(_event(), "loud") is not None  # still served from memory
    assert journal.fault is not None


@pytest.fixture
def device(tmp_path):
    authority = la.LocalAuthority("SD-1A2B3C4D", la.DeviceIdentity.load_or_create(tmp_path / "id.pem"),
                                  la.ClientStore(tmp_path / "c.json"))
    journal = LocalEventJournal(tmp_path / "events.jsonl")
    server = DashboardServer(host="0.0.0.0", port=0, token=None, local_authority=authority,  # noqa: S104 - LAN bind under test
                             event_journal=journal)
    url = server.start()
    yield server, url, authority
    server.close()


def test_get_api_events_needs_a_session_and_returns_the_dashboards_events(device):
    server, url, authority = device
    phone = LocalPhone(url)
    assert phone.call("GET", "/api/events").status == 401
    phone.trust(authority.enroll(la.b64u(phone.key.public_key), "p", "ble"))
    phone.open_session()
    event = _event()
    server.push_event(event, "loud")
    reply = phone.get("/api/events?limit=10")
    assert reply.status == 200 and reply.body["schema"] == "sd-local-events/1"
    assert reply.body["events"][0]["event_key"] == event.event_id
    assert phone.get(f"/api/events?after={reply.body['latest_seq']}").body["events"] == []
    assert phone.get("/api/events?after=nope").status == 400


def test_sync_is_unknown_when_the_backlog_cannot_be_read(device):
    server, _, _ = device
    server.pending_event_keys = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    server.push_event(_event(), "loud")
    page, status = server.events_response({})
    assert status == 200 and page["events"][0]["sync"] == "unknown"
    assert json.dumps(page)  # serialisable
