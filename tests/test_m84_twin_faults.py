"""M8.4 failure injection on the digital twin (docs/FAILURE_MATRIX.md, rows 21+).

The gaps M8.3 left: a partially working link (lossy, duplicating, reordering,
slow), malformed accelerator tensors through the device runtime, a backend
restart in the middle of an outage, an alert's acknowledgement lost again and
again across a restart, and a data directory that turns read-only. Each runs
the real runtime, uplink and backend on one SimClock with the truth
InvariantMonitor attached, and compares what the device generated (its
EventLedger, by event_key) with what the backend stored. Deterministic: seeded
faults, no sleeps.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sim import scenario as sc
from sim.faults import EventLedger, LinkFaults, compare, read_only, running_as_root
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import (
    ASLEEP,
    ATTENTIVE,
    CALIBRATION_S,
    LOOKING_AWAY,
    PHONE_IN_LAP,
    Faults,
    Scenario,
    Segment,
)
from sim.uplink import SimUplink

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
REPO = Path(__file__).resolve().parent.parent
DOWN = Faults(network_down=True)


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, *rest) for t, scene, *rest in segments)))


#: Three drowsiness episodes, a phone and a distraction over ~2.5 minutes.
BUSY = _scenario("busy", (C + 3, ASLEEP), (C + 8, ATTENTIVE), (C + 30, PHONE_IN_LAP), (C + 36, ATTENTIVE),
                 (C + 60, LOOKING_AWAY), (C + 65, ATTENTIVE), (C + 90, ASLEEP), (C + 95, ATTENTIVE),
                 (C + 130, ASLEEP), (C + 135, ATTENTIVE))


@pytest.fixture
def ledger():
    SimUplink.LEDGER = EventLedger()
    yield SimUplink.LEDGER
    SimUplink.LEDGER = None


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


@pytest.fixture
def monitor(stack):
    return InvariantMonitor(stack)


def _integrity(stack, ledger) -> dict:
    return compare(ledger.entries, stack.events(limit=1000))


def _assert_every_event_once(stack, ledger) -> dict:
    result = _integrity(stack, ledger)
    assert result["generated"] > 0, "the scenario generated nothing: the test proves nothing"
    assert result["lost"] == [], result
    assert result["unexpected"] == [], result
    assert result["duplicate_keys"] == [] and result["duplicate_type_timestamp"] == [], result
    return result


# ---------------------------------------------------------------------- partial network loss
LINKS: dict[str, dict[str, Any]] = {
    # 30 % of requests never arrive, 20 % arrive but the answer is lost.
    "lossy": {"loss": 0.3, "answer_loss": 0.2},
    # A retransmitting proxy: every request reaches the backend twice.
    "duplicating": {"duplicate": 1.0},
    # 30 % of requests are held past the device's timeout and delivered 8 s later,
    # after newer ones: duplicate AND out of order.
    "reordering": {"hold": 0.3, "hold_s": 8.0},
    # Everything at once.
    "mixed": {"loss": 0.15, "answer_loss": 0.15, "duplicate": 0.1, "hold": 0.15, "hold_s": 6.0},
}


@pytest.mark.parametrize("kind", sorted(LINKS))
def test_a_partially_working_link_delivers_every_event_once_and_commands_once(stack, monitor, ledger, kind):
    link = LinkFaults(stack.transport, seed=8_4, clock=stack.clock, **LINKS[kind])
    device = stack.boot(BUSY, transport=link)
    stack.claim()
    device.advance(C + 100)
    assert stack.protection("stop", confirm=True).status_code == 200
    device.advance(15.0)
    assert stack.protection("resume").status_code == 200
    device.advance(40.0)
    # The link heals: whatever it still holds arrives (late), then one clean cycle.
    link.enabled = False
    link.release_held(force=True)
    device.advance(10.0)

    result = _assert_every_event_once(stack, ledger)
    assert stack.event_types().count("drowsiness") == 3, stack.event_types()
    assert monitor.stops_applied == 1 and monitor.resumes_applied == 1, monitor.protection_effects
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["stale"] is False, status
    monitor.assert_clean()
    stats = link.stats
    exercised = {"lossy": stats.lost_requests and stats.lost_answers, "duplicating": stats.duplicated,
                 "reordering": stats.delivered_late, "mixed": stats.lost_requests and stats.delivered_late}[kind]
    assert exercised, f"the {kind} fault never happened: {stats}"
    print(f"{kind}: {result['generated']} events, {stats.sent} requests, lost={stats.lost_requests} "
          f"answers_lost={stats.lost_answers} dup={stats.duplicated} late={stats.delivered_late}")


def test_late_deliveries_never_reorder_history_or_overwrite_newer_health(stack, monitor, ledger):
    """Held batches reach the backend after newer ones: each event keeps the time
    it happened (occurred_at follows the device's own order), and an old health
    sample arriving late never replaces the newer one (state model edges hold)."""
    link = LinkFaults(stack.transport, seed=3, clock=stack.clock, hold=0.5, hold_s=10.0)
    device = stack.boot(BUSY, transport=link)
    stack.claim()
    device.advance(C + 150)
    link.enabled = False
    link.release_held(force=True)
    device.advance(6.0)
    _assert_every_event_once(stack, ledger)
    assert link.stats.delivered_late >= 5, link.stats
    rows = stack.events(limit=1000)
    by_device_order = sorted(rows, key=lambda r: r["timestamp"])
    occurred = [r["occurred_at"] for r in by_device_order]
    assert occurred == sorted(occurred), "a late delivery moved an event in time"
    monitor.assert_clean()


# ---------------------------------------------------------------------- alert acknowledgement loss
class _AckLoss:
    """The backend stores each event batch; its answer is lost `times` times."""

    def __init__(self, inner, times: int) -> None:
        self._inner, self.remaining, self.stored_batches = inner, times, 0

    def post(self, url, **kwargs):
        response = self._inner.post(url, **kwargs)
        if self.remaining > 0 and (kwargs.get("json") or {}).get("events"):
            self.remaining -= 1
            self.stored_batches += 1
            import requests

            raise requests.ConnectionError("SIMULATED: the ingest acknowledgement was lost")
        return response


def test_an_alert_whose_acknowledgement_is_lost_repeatedly_and_across_a_power_cut_is_stored_once(
        stack, monitor, ledger, monkeypatch):
    app = stack.backend.app
    notified: list[str] = []
    real = app.gate.evaluate

    def evaluate(user_id, severity, source="module"):
        decision = real(user_id, severity, source=source)
        notified.append(severity)
        return decision

    monkeypatch.setattr(app.gate, "evaluate", evaluate)
    script = _scenario("ack_loss", (C + 3, ASLEEP), (C + 8, ATTENTIVE))
    link = _AckLoss(stack.transport, times=4)
    device = stack.boot(script, transport=link)
    stack.claim()
    device.advance(C + 10)
    assert link.stored_batches >= 2, "the acknowledgement was not lost yet"
    # Still unacknowledged on the device; the power goes.
    assert any(e["type"] == "drowsiness" for e in device.runtime.uplink._queue) or link.remaining == 0
    origin = device.scenario_clock.origin
    rebooted = stack.cold_restart(script, transport=link)
    rebooted.scenario_clock.origin = origin  # the world goes on: the eyes do not close again
    rebooted.advance(20.0)
    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1, stack.event_types()
    assert notified.count("loud") == 1, f"notified {notified.count('loud')} times for one alert"
    _assert_every_event_once(stack, ledger)
    monitor.assert_clean()


# ---------------------------------------------------------------------- backend restart mid-outage
def test_a_backend_restart_during_an_outage_with_a_pending_stop_converges_once(stack, monitor, ledger):
    """The device is offline with an alert queued; the owner asks it to stop; the
    backend restarts; the link comes back. One alert row, the stop applied once."""
    script = _scenario("restart_mid_outage", (C + 2, ATTENTIVE, DOWN), (C + 4, ASLEEP, DOWN),
                       (C + 9, ATTENTIVE, DOWN), (C + 25, ATTENTIVE))
    device = stack.boot(script)
    stack.claim()
    device.advance(C + 12)
    assert stack.protection("stop", confirm=True).status_code == 200
    stack.restart_backend()
    device.advance(18.0)
    status = stack.status()
    assert status["monitoring_state"] == "stopped" and status["protection"] == "stopped", status
    assert (status.get("protection_command") or {}).get("outcome") == "confirmed", status.get("protection_command")
    assert [e["type"] for e in stack.events()].count("drowsiness") == 1
    assert monitor.stops_applied == 1
    _assert_every_event_once(stack, ledger)
    monitor.assert_clean()


# ---------------------------------------------------------------------- invalid accelerator tensors
def _fake_hailo():
    sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))
    import fake_hailo

    return fake_hailo


CLS16, BOX16 = "damoyolo_tinynasL20_T/conv66", "damoyolo_tinynasL20_T/conv65"


def _malformed(fake_hailo, fault: str):
    """A chip answering with one defect. The box-head defect sits at the anchor of
    a real candidate (a planted phone): the decoder checks box logits only where
    it decodes a box (core/accel.py, "no full pass"), so a NaN at an anchor with
    no candidate cannot change the answer and is, by design, not inspected."""
    phone = (67, 0.9, 300, 280, 364, 408, 16)
    row, col = round(344 / 16 - 0.5), round(332 / 16 - 0.5)  # the planted phone's anchor

    def backend(batch):
        fake_hailo.zero_backend(batch)  # the input is validated as the real stream would
        heads = fake_hailo.planted_heads([phone])
        if fault == "nan_scores":
            heads[CLS16][0, 3, 3, 5] = np.nan
        elif fault == "inf_box":
            heads[BOX16][0, row, col, 0] = np.inf
        elif fault == "wrong_grid":
            heads[CLS16] = np.zeros((1, 20, 20, 81), np.float32)
        elif fault == "missing_head":
            heads.pop(CLS16)
        return heads
    return backend


@pytest.mark.parametrize("fault", ["nan_scores", "inf_box", "wrong_grid", "missing_head"])
def test_malformed_accelerator_tensors_degrade_objects_and_never_crash_the_device(stack, monitor, monkeypatch, fault):
    """The device's real HailoYOLO validation (core/accel.py, M8.3) inside the real
    runtime: a malformed NPU result raises, the module treats it as a detector
    failure -- objects impaired, face monitoring and alerting kept -- and a sane
    result brings it back. Never a crash, never "no objects" passed off as an answer."""
    from core import accel
    from core.accel import HailoYOLO

    fake_hailo = _fake_hailo()
    monkeypatch.setattr(accel, "record_inference", lambda: None)
    fake_hailo.install(_malformed(fake_hailo, fault))
    try:
        model = HailoYOLO("fake.hef", sha256="0" * 64)
        device = stack.boot(_scenario("npu", (C + 12, ASLEEP), (C + 17, ATTENTIVE)))
        stack.claim()
        device.advance(C + 2)
        assert stack.status()["monitoring_state"] == "ready"
        scripted = device.sim.detector.predict
        calls = {"npu": 0}
        mode = {"bad": True}

        def predict(frame, **kwargs):
            if mode["bad"]:
                calls["npu"] += 1
                return model.predict(frame, **kwargs)  # raises: the validation under test
            return scripted(frame, **kwargs)

        monkeypatch.setattr(device.sim.detector, "predict", predict)
        device.advance(18.0)  # eyes close at C+12 while the NPU answers garbage
        assert calls["npu"] > 0, "the malformed tensors never reached the decoder"
        status = stack.status()
        assert status["monitoring_state"] == "degraded", status
        assert status["protection"] == "protecting"
        impaired = {i["name"]: i for i in status["capabilities"]["impaired"]}
        assert "objects" in impaired and "objects" not in status["capabilities"]["available"], status["capabilities"]
        assert "face" in status["capabilities"]["available"]
        assert "drowsiness" in stack.event_types(), "face monitoring must keep alerting while objects are impaired"
        assert device.runtime._monitoring_state is not None

        mode["bad"] = False
        device.advance(75.0)  # past the module's retry backoff
        assert stack.status()["monitoring_state"] == "ready", stack.status()
        monitor.assert_clean()
    finally:
        fake_hailo.uninstall()


# ---------------------------------------------------------------------- permission changes
@pytest.mark.skipif(running_as_root(), reason="root writes through a read-only mode")
def test_a_data_directory_that_turns_read_only_costs_the_local_log_never_monitoring_or_alerts(
        stack, monitor, ledger):
    """The kernel remounts the card read-only (or a deploy breaks permissions):
    events.csv, the spool and saved state cannot be written. The device keeps
    monitoring, alerting and delivering, a stop still works, and health says the
    disk is the problem instead of reporting everything fine."""
    script = _scenario("read_only", (C + 10, ASLEEP), (C + 15, ATTENTIVE))
    device = stack.boot(script)
    stack.claim()
    device.advance(C + 2)
    with read_only(stack.state_dir):
        device.advance(20.0)
        assert stack.protection("stop", confirm=True).status_code == 200
        device.advance(6.0)
        status = stack.status()
        assert status["monitoring_state"] == "stopped", status
        assert "drowsiness" in stack.event_types(), "the alert reached the app with the disk read-only"
        health = stack.row()["health"]
        storage = health.get("storage") or {}
        assert storage.get("state") == "failed", health.get("storage")
        assert health["uplink_backlog"]["persistent"] is False or health["uplink_backlog"]["queued"] == 0
        # (events.csv was open before the chmod, so its rows still land; a kernel
        # read-only remount fails those too -- tests/test_event_log_durability.py.)
    assert stack.protection("resume").status_code == 200
    device.advance(30.0)
    assert stack.status()["monitoring_state"] == "ready"
    assert (stack.row()["health"].get("storage") or {}).get("state") == "ok"
    _assert_every_event_once(stack, ledger)
    monitor.assert_clean()


@pytest.mark.skipif(running_as_root(), reason="root writes through a read-only mode")
def test_a_reboot_onto_a_read_only_data_directory_comes_up_protecting(stack, monitor):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C + 2)
    stack.power_off()
    with read_only(stack.state_dir):
        device = stack.boot(sc.get("attentive"))
        device.advance(10.0)
        status = stack.status()
        assert status["monitoring_state"] == "ready", status
        assert (stack.row()["health"].get("storage") or {}).get("state") == "failed"
    monitor.assert_clean()
