"""M8 lifecycle: the undelivered-event backlog survives a restart and a power cut.

Before M8 the uplink's queue lived only in memory: an alert raised while the
backend was unreachable was lost if the device restarted (or lost power) before
the network came back -- exactly the drive that ends with the ignition off. The
backlog is now also a JSON-lines spool beside the event log, appended (fsync'd)
as each event is queued and rewritten atomically as events are delivered. These
drive the real CloudUplink with a scripted transport; the end-to-end version on
the virtual device is tests/test_m8_lifecycle_e2e.py.
"""

from __future__ import annotations

import json
import types

import requests

from core.uplink import CloudUplink


class _Backend:
    """A transport: down while `down`, else stores events by event_key (as cloud/app.py does)."""

    def __init__(self) -> None:
        self.down = False
        self.lose_response = False
        self.stored: dict[str, str] = {}
        self.posts = 0

    def post(self, url, json=None, **_kwargs):
        self.posts += 1
        if self.down:
            raise requests.ConnectionError("SIMULATED: backend unreachable")
        for event in json["events"]:
            self.stored.setdefault(event["event_key"], event["type"])
        if self.lose_response:
            raise requests.ConnectionError("SIMULATED: response lost after the backend stored the batch")
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                     json=lambda: {"command": None})


def _event(kind: str):
    return types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "2026-11-16T08:00:00"),
                                 type=kind, confidence=0.9, contributing_signals={"ear": 0.1}, duration=2.0)


def _boot(spool, backend, queue_max=500) -> CloudUplink:
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_" + "x" * 32,
                          "poll_interval_s": 2.0, "spool_path": str(spool), "queue_max": queue_max,
                          "command_auth": "off"})
    uplink._requests = backend
    return uplink


def test_events_queued_offline_survive_a_power_cut_and_are_delivered_once(tmp_path):
    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    backend.down = True
    first = _boot(spool, backend)
    first.push_event(_event("drowsiness"), "loud")
    first.push_event(_event("phone_use"), "loud")
    first.sync_once()  # offline: kept
    assert first.backlog()["queued"] == 2
    # Power cut: `first` is never closed or flushed. Only the disk survives.
    del first
    backend.down = False
    second = _boot(spool, backend)
    assert second.backlog() == {"queued": 2, "capacity": 500, "dropped": 0, "restored": 2, "persistent": True}
    second.sync_once()
    assert sorted(backend.stored.values()) == ["drowsiness", "phone_use"]
    assert not spool.exists(), "a delivered backlog leaves nothing on disk"
    third = _boot(spool, backend)
    third.sync_once()
    assert len(backend.stored) == 2, "nothing is re-sent after delivery"


def test_a_power_cut_after_the_backend_stored_a_batch_replays_it_harmlessly(tmp_path):
    """The response was lost, so the device still holds the batch; the backend dedupes by event_key."""
    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    backend.lose_response = True
    first = _boot(spool, backend)
    first.push_event(_event("drowsiness"), "loud")
    first.sync_once()
    assert len(backend.stored) == 1 and spool.exists()
    del first  # power cut before the device learned it was delivered
    backend.lose_response = False
    second = _boot(spool, backend)
    second.sync_once()
    assert list(backend.stored.values()) == ["drowsiness"], "one alert, one stored row"
    assert not spool.exists()


def test_a_torn_or_damaged_spool_never_stops_the_uplink(tmp_path):
    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    backend.down = True
    first = _boot(spool, backend)
    for kind in ("a", "b", "c"):
        first.push_event(_event(kind), "loud")
    lines = spool.read_text().splitlines()
    # Garbage in the middle, and the last append torn by the power cut.
    spool.write_text(lines[0] + "\n\x00\xffnot json\n[1, 2]\n" + lines[1] + "\n" + lines[2][: len(lines[2]) // 2])
    second = _boot(spool, backend)
    assert [e["type"] for e in second._queue] == ["a", "b"]
    assert second.backlog()["restored"] == 2
    assert [json.loads(line)["type"] for line in spool.read_text().splitlines()] == ["a", "b"], \
        "the spool is compacted to what was restored"


def test_the_spool_is_bounded_like_the_queue_and_keeps_the_newest(tmp_path):
    spool = tmp_path / "uplink_spool.jsonl"
    backend = _Backend()
    backend.down = True
    uplink = _boot(spool, backend, queue_max=5)
    for i in range(23):
        uplink.push_event(_event(f"e{i}"), "loud")
    assert uplink.backlog()["queued"] == 5 and uplink.dropped_events == 18
    assert len(spool.read_text().splitlines()) <= 2 * 5, "compacted, not growing for the whole outage"
    restored = _boot(spool, backend, queue_max=5)
    assert [e["type"] for e in restored._queue] == [f"e{i}" for i in range(18, 23)]


def test_without_a_spool_path_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_x", "command_auth": "off"})
    uplink._requests = _Backend()
    uplink._requests.down = True
    uplink.push_event(_event("drowsiness"), "loud")
    uplink.sync_once()
    assert list(tmp_path.iterdir()) == []
    assert uplink.backlog()["persistent"] is False
