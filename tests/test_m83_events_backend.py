"""M8.3 events at the backend: one stored row per event and per episode, honest
times, alert-delivery evidence kept exactly as the device reported it.

Real cloud/app.py behind a TestClient on a throwaway database; bodies built by
the real CloudUplink where the device's shape matters. The end-to-end version
on the virtual device is tests/test_m83_events_sim.py.
"""

from __future__ import annotations

import importlib
import sys
import time
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

DEVICE = {"X-Device-Key": "dk_events_m83"}


def _load_app(db_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(db_path))
    monkeypatch.setenv("SENSORDYME_MDNS", "0")
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    app_module.gate.reset()
    return app_module


class _Stack:
    def __init__(self, tmp_path, monkeypatch) -> None:
        self.db_path = tmp_path / "events.db"
        self.monkeypatch = monkeypatch
        self.notified: list[str] = []
        self.considered: list[str] = []
        self.app = _load_app(self.db_path, monkeypatch)
        self.client = TestClient(self.app.app)
        with self.app.db() as conn:
            conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                         ("Events unit", "EV-M8-3X", DEVICE["X-Device-Key"], time.time()))
        token = self.client.post("/v1/signup", json={"email": "events@example.com",
                                                     "password": "supersecret1"}).json()["token"]
        self.auth = {"Authorization": f"Bearer {token}"}
        self.device_id = self.client.post("/v1/devices/claim", json={"claim_code": "EV-M8-3X"},
                                          headers=self.auth).json()["device_id"]
        self._spy()

    def _spy(self) -> None:
        """Count every notification the backend sends (its _notify task), and
        every NEW row it considers notifying for (the cooldown gate is asked
        once per newly stored event -- a re-sent one never reaches it)."""
        self.monkeypatch.setattr(self.app, "_notify", lambda owner, title, *a, **k: self.notified.append(title))
        real = self.app.gate.evaluate

        def evaluate(user_id, severity, source="module"):
            self.considered.append(severity)
            return real(user_id, severity, source=source)

        self.monkeypatch.setattr(self.app.gate, "evaluate", evaluate)

    def restart(self) -> None:
        """A backend process restart on the same database."""
        self.app = _load_app(self.db_path, self.monkeypatch)
        self.client = TestClient(self.app.app)
        self._spy()

    def ingest(self, *events, **extra):
        response = self.client.post("/v1/ingest", json={"events": list(events), **extra}, headers=DEVICE)
        assert response.status_code == 200, response.text
        return response

    def events(self, **params) -> list[dict]:
        response = self.client.get(f"/v1/devices/{self.device_id}/events", params=params, headers=self.auth)
        assert response.status_code == 200
        return response.json()["events"]


@pytest.fixture
def stack(tmp_path, monkeypatch):
    return _Stack(tmp_path, monkeypatch)


def _alert(output="sounded", rev=2, episode="0123456789ab-1", decision="alerted"):
    return {"decision": decision, "episode": episode, "category": "critical", "output": output, "rev": rev,
            "outputs": [{"name": "gpio_buzzer", "kind": "buzzer", "audible": True,
                         "played": 1 if output == "sounded" else 0, "failed": 0, "skipped": 0,
                         "pending": 1 if output == "pending" else 0, "unconfirmed": 0}]}


def _event(key="k1", **fields):
    return {"event_key": key, "timestamp": datetime.now(UTC).isoformat(), "type": "drowsiness",
            "confidence": 0.9, "signals": "eyes closed", "severity": "loud", "source": "module",
            "duration": 2.0, **fields}


# ---------------------------------------------------------------------- the row the app reads
def test_an_event_row_carries_identity_both_clocks_evidence_and_provenance(stack):
    """Before M8.3 the app got the device's timestamp and nothing else: no
    arrival time, no key, nothing about whether the device actually warned."""
    stack.ingest(_event(age_s=4.0, alert=_alert()))
    row = stack.events()[0]
    assert row["event_key"] == "k1"
    assert isinstance(row["received_at"], float)
    assert row["time_basis"] == "device_monotonic"
    assert row["occurred_at"] == pytest.approx(row["received_at"] - 4.0)
    assert row["alert"]["output"] == "sounded" and row["alert"]["outputs"][0]["name"] == "gpio_buzzer"
    assert row["simulated"] is False


def test_the_backend_never_derives_sounded(stack):
    """An event that reports nothing about its outputs has no alert evidence:
    a stored event is a detection, never proof that anyone was warned."""
    stack.ingest(_event("plain"))
    stack.ingest(_event("forged", alert={"output": "sounded", "decision": "made-up"}))
    stack.ingest(_event("odd", alert={"output": "loud", "decision": "alerted"}))
    rows = {r["event_key"]: r for r in stack.events()}
    assert rows["plain"]["alert"] is None
    assert rows["forged"]["alert"] is None, "an unknown decision is not evidence"
    assert rows["odd"]["alert"] is None, "an unknown output is not evidence"


def test_an_alert_report_is_bounded_to_known_fields(stack):
    report = _alert()
    report["outputs"][0].update(error="x" * 500, frame="base64-jpeg...", played=True)
    report["extra"] = {"image": "..."}
    stack.ingest(_event(alert=report))
    stored = stack.events()[0]["alert"]
    assert set(stored) == {"decision", "episode", "category", "output", "rev", "outputs"}
    assert "frame" not in stored["outputs"][0] and len(stored["outputs"][0]["error"]) == 160
    assert stored["outputs"][0]["played"] == 0, "a bool is not a count"


# ---------------------------------------------------------------------- duplicates and late reports
def test_a_late_alert_report_updates_the_row_never_a_second_row_or_notification(stack):
    stack.ingest(_event(alert=_alert("pending", rev=1)))
    stack.ingest(_event(alert=_alert("sounded", rev=2)))
    stack.ingest(_event(alert=_alert("pending", rev=1)))  # an older report, arriving late
    rows = stack.events()
    assert len(rows) == 1
    assert rows[0]["alert"]["output"] == "sounded" and rows[0]["alert"]["rev"] == 2
    assert len(stack.notified) == 1 and len(stack.considered) == 1


def test_a_second_event_for_one_episode_is_not_a_second_alert(stack):
    """Before M8.3: an episode re-reported under a new key (a detector firing
    again after a reconnect) was stored again and put to the notifier again."""
    stack.ingest(_event("onset", signals="onset", alert=_alert(episode="aaaaaaaaaaaa-7")))
    stack.ingest(_event("again", signals="again", alert=_alert(episode="aaaaaaaaaaaa-7", decision="merged")))
    stack.ingest(_event("next-boot", signals="next-boot", alert=_alert(episode="bbbbbbbbbbbb-7")))
    stored = [r["signals"] for r in stack.events()]
    assert stored == ["next-boot", "onset"], "one row per episode; episode 7 of another boot is another alert"
    assert stack.considered == ["loud", "loud"], "the re-reported episode never reached the notifier"


def test_a_replayed_batch_creates_nothing(stack):
    body = [_event("r1", alert=_alert(episode="cccccccccccc-1")), _event("r2", type="phone_use")]
    stack.ingest(*body)
    before = stack.events()
    stack.ingest(*body)
    stack.ingest(*body)
    assert stack.events() == before
    assert len(stack.considered) == 2 and len(stack.notified) == 1


def test_a_backend_restart_between_storing_and_answering_is_harmless(stack):
    """The device never heard the answer, so it sends the batch again to a
    restarted backend: still one row, still one notification."""
    batch = [_event("b1", alert=_alert(episode="dddddddddddd-1")), _event("b2", type="phone_use")]
    stack.ingest(*batch)
    stack.restart()
    stack.ingest(*batch)
    assert sorted(r["event_key"] for r in stack.events()) == ["b1", "b2"]
    assert len(stack.notified) == 1, "notified on first arrival only, not again by the restarted process"


# ---------------------------------------------------------------------- clocks
def test_a_wrong_device_clock_never_moves_the_event(stack):
    """A Pi with no RTC boots in 1970 (or at its last shutdown) until NTP."""
    stack.ingest(_event("epoch", timestamp="1970-01-01T00:02:11+00:00", age_s=1.5))
    row = stack.events()[0]
    assert row["timestamp"] == "1970-01-01T00:02:11+00:00", "the device's claim is kept verbatim"
    assert row["time_basis"] == "device_monotonic"
    assert row["occurred_at"] == pytest.approx(row["received_at"] - 1.5)


def test_without_an_age_an_implausible_device_clock_falls_back_to_arrival(stack):
    future = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    stack.ingest(_event("future", timestamp=future), _event("garbage", timestamp="not a time"))
    rows = {r["event_key"]: r for r in stack.events()}
    for key in ("future", "garbage"):
        assert rows[key]["time_basis"] == "received"
        assert rows[key]["occurred_at"] == rows[key]["received_at"]


def test_without_an_age_a_plausible_device_clock_dates_a_delayed_event(stack):
    """An event restored from the spool after a restart arrives with no age; it
    happened an hour ago, and the timeline must say so, not 'just now'."""
    an_hour_ago = datetime.now(UTC) - timedelta(hours=1)
    stack.ingest(_event("late", timestamp=an_hour_ago.isoformat()))
    row = stack.events()[0]
    assert row["time_basis"] == "device_clock"
    assert row["occurred_at"] == pytest.approx(an_hour_ago.timestamp(), abs=1e-3)
    assert row["received_at"] - row["occurred_at"] == pytest.approx(3600, abs=5)


def test_explain_uses_the_backend_clock_when_the_device_clock_was_wrong(stack):
    stack.ingest(_event("epoch", timestamp="1970-01-01T00:02:11+00:00", age_s=0.0))
    event_id = stack.events()[0]["id"]
    explained = stack.client.get(f"/v1/devices/{stack.device_id}/events/{event_id}/explain",
                                 headers=stack.auth).json()
    expected = datetime.fromtimestamp(stack.events()[0]["occurred_at"]).astimezone().strftime("%-I:%M %p")
    assert explained["when"] == expected


# ---------------------------------------------------------------------- real vs simulated
def test_simulated_is_recorded_from_the_event_its_report_or_its_model(stack):
    simulated_model = {"module": "driver_attention",
                       "detector": {"name": "scripted", "sha256": None, "backend": "simulated"}}
    stack.ingest(_event("flagged", simulated=True))
    stack.ingest(_event("by-model", model=simulated_model))
    stack.ingest(_event("by-report"), health={"simulated": True})
    stack.ingest(_event("real", model={"module": "driver_attention",
                                       "detector": {"name": "x.hef", "sha256": None, "backend": "hailo"}}))
    rows = {r["event_key"]: r["simulated"] for r in stack.events()}
    assert rows == {"flagged": True, "by-model": True, "by-report": True, "real": False}


def test_a_device_cannot_mark_real_history_simulated_with_a_non_bool(stack):
    stack.ingest(_event("str", simulated="yes"))
    assert stack.events()[0]["simulated"] is False


# ---------------------------------------------------------------------- paging
def test_pages_by_id_never_overlap_or_skip_while_events_arrive(stack):
    for i in range(5):
        stack.ingest(_event(f"p{i}", type=f"t{i}"))
    first = stack.events(limit=2)
    stack.ingest(_event("p5", type="t5"))  # arrives between the two page loads
    second = stack.events(limit=2, before_id=first[-1]["id"])
    third = stack.events(limit=10, before_id=second[-1]["id"])
    keys = [r["event_key"] for r in first + second + third]
    assert keys == ["p4", "p3", "p2", "p1", "p0"]
    assert stack.events(limit=1)[0]["event_key"] == "p5"


def test_a_simulated_alert_is_labelled_on_the_lock_screen(stack):
    stack.ingest(_event("sim", simulated=True))
    stack.app.gate.reset()
    stack.ingest(_event("real", type="fall"))
    assert len(stack.notified) == 2
    assert stack.notified[0].endswith("(simulated)") and not stack.notified[1].endswith("(simulated)")
