"""M8.4 storage integrity on the digital twin (D3).

A full card is tested on a REAL filesystem: sim.faults.small_volume mounts a
few-MB HFS+ image (macOS hdiutil), so ENOSPC comes from the kernel and the
device's own buffered writers behave exactly as they do on a full SD card --
nothing is mocked. Elsewhere those tests skip (no root to mount on Linux);
the read-only and permission faults run everywhere (tests/test_m84_twin_faults.py).

Also: spool format migration (an older firmware's undelivered backlog), and the
backend's 90-day retention boundary. Bounded losses are asserted as bounds and
printed, so the evidence summary carries the numbers.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sim import scenario as sc
from sim.faults import EventLedger, can_make_volumes, compare, small_volume
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment
from sim.uplink import SimUplink

pytest.importorskip("fastapi")

C = CALIBRATION_S
needs_volume = pytest.mark.skipif(not can_make_volumes(), reason="a real small filesystem needs macOS hdiutil")


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, *rest) for t, scene, *rest in segments)))


@pytest.fixture
def ledger():
    SimUplink.LEDGER = EventLedger()
    yield SimUplink.LEDGER
    SimUplink.LEDGER = None


def _csv_rows(path: Path) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(path.read_text(errors="replace"))))
    return [r for r in rows if r]


# ---------------------------------------------------------------------- a full card, end to end
@needs_volume
def test_a_full_card_costs_bounded_local_records_never_monitoring_alerts_or_delivery(tmp_path, ledger):
    """The device's data directory is on a 3 MB volume that fills up mid-run.
    Monitoring, alerting, stop/resume and delivery to the app carry on; health
    says the card is full; nothing on disk is left half-written; once space is
    back the device recovers by itself, and a reboot comes up clean."""
    pytest.importorskip("mediapipe")
    script = _scenario("full_card", (C + 10, ASLEEP), (C + 15, ATTENTIVE),
                       (C + 30, ATTENTIVE, Faults(network_down=True)), (C + 35, ASLEEP, Faults(network_down=True)),
                       (C + 40, ATTENTIVE, Faults(network_down=True)), (C + 50, ATTENTIVE),
                       (C + 80, ASLEEP), (C + 85, ATTENTIVE))
    with small_volume(3072, root=tmp_path) as volume, SimStack(tmp_path / "stack") as stack:
        (volume.path / "device").mkdir()
        os.symlink(volume.path / "device", stack.root / "device")
        monitor = InvariantMonitor(stack)
        device = stack.boot(script)
        stack.claim()
        device.advance(C + 2)
        rows_before = len(_csv_rows(stack.state_dir / "events.csv"))
        left = volume.fill(leave=0)
        assert left < 64 * 1024, f"the volume did not fill ({left} bytes free)"

        device.advance(30.0)                       # an alert on a full card, delivered live
        assert stack.event_types().count("drowsiness") == 1
        storage = stack.row()["health"]["storage"]
        assert storage["state"] == "failed" and "full" in storage["detail"], storage
        device.advance(25.0)                       # an alert raised offline, on a full card
        assert stack.protection("stop", confirm=True).status_code == 200
        device.advance(6.0)
        assert stack.status()["monitoring_state"] == "stopped"
        assert stack.protection("resume").status_code == 200
        device.advance(10.0)
        assert stack.event_types().count("drowsiness") == 2, "the offline alert was delivered from memory"

        volume.free()
        device.advance(30.0)                       # space is back: a third alert, and health recovers
        assert stack.event_types().count("drowsiness") == 3
        storage = stack.row()["health"]["storage"]
        # A 3 MB volume is "nearly full" (< 50 MB) by the device's rule, never "failed" any more.
        assert storage["state"] == "warning" and "nearly full" in storage["detail"], storage
        assert "spool_fault" not in stack.row()["health"]["uplink_backlog"]
        assert stack.status()["monitoring_state"] == "ready"

        device.advance(25.0)  # past state_model.RESUME_SETTLE_S since the resume (see the M8.4
        # recovery suite's xfail: a restart within it reads "resuming")
        stack.restart(script)
        stack.device.advance(5.0)
        assert stack.status()["monitoring_state"] == "ready"
        rows = _csv_rows(stack.state_dir / "events.csv")
        torn = [r for r in rows if len(r) != 6]
        assert torn == [], f"torn rows in events.csv: {torn[:3]}"
        leftovers = sorted(p.name for p in stack.state_dir.iterdir() if p.name.endswith(".tmp"))
        integrity = compare(ledger.entries, stack.events(limit=1000))
        assert integrity["lost"] == [] and integrity["duplicate_keys"] == [], integrity
        monitor.assert_clean()
        print(f"full card: events.csv rows {rows_before} -> {len(rows)} for {integrity['generated']} events "
              f"generated (+ monitoring-state rows); stray tmp files {leftovers}; backend lost 0")


# ---------------------------------------------------------------------- the writers, one by one
@needs_volume
def test_every_config_and_state_writer_keeps_the_old_file_whole_on_a_full_card(tmp_path, monkeypatch):
    from core import config_lkg, device_state
    from core.monitoring_config import MonitoringConfig

    with small_volume(2048, root=tmp_path) as volume:
        d = volume.path
        config = d / "config.yaml"
        config.write_text("role: driver_safety\nmodule: driver_attention\n" + "# pad\n" * 200)
        assert config_lkg.save_last_good(config)
        lkg = config_lkg.last_good_path(config)
        lkg_before = lkg.read_bytes()
        monkeypatch.setenv("SENSORDYME_DEVICE_STATE_PATH", str(d / "device_state.json"))
        assert device_state.update(mode="driver_safety", zones=[])
        state_before = (d / "device_state.json").read_bytes()
        rules = MonitoringConfig(d / "monitoring.json")
        rules.save([{"target": "cell phone", "predicate": "present"}], version="v1")
        rules_before = (d / "monitoring.json").read_bytes()

        volume.fill(leave=0)
        big = b"x" * 600_000
        assert config_lkg.atomic_write_bytes(lkg, big) is False
        assert lkg.read_bytes() == lkg_before, "a failed atomic write left the old LKG whole"
        assert device_state.update(zones=[{"name": "z" * 200_000}]) is False
        assert (d / "device_state.json").read_bytes() == state_before
        rules.save([{"target": "cell phone", "predicate": "present", "pad": "p" * 600_000}], version="v2")
        assert (d / "monitoring.json").read_bytes() == rules_before
        stray = sorted(p.name for p in d.iterdir() if ".tmp" in p.name)
        # device_state.update leaves its fixed-name .tmp behind (one file, replaced next time);
        # atomic_write_bytes and MonitoringConfig clean up after themselves.
        assert all(name == "device_state.json.tmp" for name in stray), stray

        volume.free()
        assert config_lkg.atomic_write_bytes(lkg, b"role: driver_safety\n")
        assert device_state.update(zones=[])
        rules.save([], version="v3")
        assert json.loads((d / "monitoring.json").read_text())["version"] == "v3"


@needs_volume
def test_event_log_rows_written_while_full_are_a_bounded_loss_and_never_torn(tmp_path):
    """events.csv on a full card: the writer keeps what fits in its 8 KiB buffer
    and writes it once space returns; beyond that, rows are dropped (logged,
    rate limited) -- the documented bound. No row is ever half-written."""
    from core.events import Event
    from core.logger import EventLogger

    def event(i: int) -> Event:
        return Event(type=f"drowsiness_{i}", timestamp=datetime.now(UTC), duration=1.0, confidence=0.9,
                     module_name="driver_attention", contributing_signals="eyes closed for 2.0 s")

    with small_volume(2048, root=tmp_path) as volume:
        log = EventLogger(volume.path / "events.csv")
        for i in range(3):
            log.log(event(i))
        volume.fill(leave=0)
        during = 400
        for i in range(3, 3 + during):
            log.log(event(i))                     # never raises
        volume.free()
        for i in range(3 + during, 6 + during):
            log.log(event(i))
        log.close()
        rows = _csv_rows(volume.path / "events.csv")
        assert all(len(r) == 6 for r in rows), [r for r in rows if len(r) != 6][:3]
        types = [r[1] for r in rows[1:]]
        assert types[:3] == ["drowsiness_0", "drowsiness_1", "drowsiness_2"]
        assert types[-3:] == [f"drowsiness_{i}" for i in range(3 + during, 6 + during)], "rows after recovery"
        kept_during = len(types) - 6
        assert 0 < kept_during < during, kept_during
        print(f"event log on a full card: {kept_during} of {during} rows written while full were kept "
              f"(buffered), {during - kept_during} lost; 0 torn")


# ---------------------------------------------------------------------- spool format migration
def test_a_spool_left_by_older_firmware_is_restored_and_delivered_once(tmp_path, ledger):
    """An M8 spool (before M8.3's alert report, boot id, monotonic stamp and
    simulated flag), plus a line from a NEWER firmware with a field this one does
    not know, plus a torn last line: all whole lines are delivered, once."""
    pytest.importorskip("mediapipe")
    old = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    legacy = [
        {"event_key": "m8legacy0001", "model": None, "timestamp": old, "type": "drowsiness", "confidence": 0.9,
         "signals": "eyes closed", "duration": 2.0, "severity": "loud", "source": "module"},
        {"event_key": "m8legacy0002", "timestamp": old, "type": "phone_use", "confidence": None,
         "signals": "", "duration": 3.0, "severity": "soft", "source": "module"},
        {"event_key": "future000003", "timestamp": old, "type": "distraction", "confidence": 0.7, "signals": "",
         "duration": 1.0, "severity": "soft", "source": "module", "_boot": "x", "_mono": 12.0,
         "a_field_from_the_future": {"v": 9}},
    ]
    with SimStack(tmp_path) as stack:
        stack.state_dir.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(item) + "\n" for item in legacy) + '{"event_key": "torn", "ty'
        (stack.state_dir / "uplink_spool.jsonl").write_text(body)
        device = stack.boot(sc.get("attentive"))
        stack.claim()
        device.advance(6.0)
        stored = {e["event_key"]: e for e in stack.events()}
        for item in legacy:
            assert item["event_key"] in stored, (item["event_key"], sorted(stored))
            assert stored[item["event_key"]]["type"] == item["type"]
        assert "torn" not in stored
        assert device.runtime.uplink.restored_events == 3
        device.advance(6.0)
        assert sum(1 for e in stack.events() if e["event_key"].startswith(("m8legacy", "future"))) == 3
        assert not (stack.state_dir / "uplink_spool.jsonl").exists(), "delivered: the spool is emptied"


# ---------------------------------------------------------------------- retention boundaries
def test_retention_prunes_by_arrival_at_exactly_ninety_days(tmp_path):
    with SimStack(tmp_path) as stack:
        stack.claim()
        app = stack.backend.app
        key = {"X-Device-Key": stack.factory["device_key"]}
        long_ago = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        events = [{"event_key": f"r{i}", "timestamp": long_ago if i == 3 else datetime.now(UTC).isoformat(),
                   "type": "drowsiness", "confidence": 0.9, "signals": "", "severity": "loud",
                   "source": "module", "duration": 1.0} for i in range(4)]
        assert stack.api.post("/v1/ingest", json={"events": events}, headers=key).status_code == 200
        now = time.time()
        cutoff = now - app.EVENT_RETENTION_DAYS * 86400
        with app.db() as conn:
            conn.execute("UPDATE events SET received_at = ? WHERE event_key = 'r0'", (cutoff - 1,))
            conn.execute("UPDATE events SET received_at = ? WHERE event_key = 'r1'", (cutoff,))
            conn.execute("UPDATE events SET received_at = ? WHERE event_key = 'r2'", (cutoff + 1,))
        assert app.EVENT_RETENTION_DAYS == 90.0
        assert app._prune_events(now) == 1
        kept = sorted(e["event_key"] for e in stack.events())
        # r0 (one second past 90 days) goes; r1 at exactly the cut-off stays; r3 HAPPENED
        # 100 days ago but arrived now (a device back from a long outage): kept, by arrival.
        assert kept == ["r1", "r2", "r3"], kept
        # Bounded and documented: a key whose row was pruned is not remembered, so a
        # device re-sending it more than 90 days later would store it again.
        assert stack.api.post("/v1/ingest", json={"events": [events[0]]}, headers=key).status_code == 200
        assert sorted(e["event_key"] for e in stack.events()) == ["r0", "r1", "r2", "r3"]


def test_the_spool_capacity_boundary_keeps_the_newest_and_counts_the_rest(tmp_path):
    from core.uplink import CloudUplink

    spool = tmp_path / "uplink_spool.jsonl"
    lines = [{"event_key": f"k{i:04d}", "type": "drowsiness", "timestamp": "2026-11-13T00:00:00+00:00",
              "severity": "loud", "source": "module"} for i in range(505)]
    spool.write_text("".join(json.dumps(item) + "\n" for item in lines))
    uplink = CloudUplink({"base_url": "http://127.0.0.1:9", "device_key": "dk_x", "spool_path": str(spool),
                          "max_queue": 500})
    backlog = uplink.backlog()
    assert backlog["queued"] == 500 and backlog["dropped"] == 5 and backlog["persistent"] is True
    assert [i["event_key"] for i in uplink._queue][:1] == ["k0005"], "the oldest five are the ones dropped"
    assert len(spool.read_text().splitlines()) == 500
