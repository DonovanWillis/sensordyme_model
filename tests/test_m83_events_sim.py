"""M8.3 events end to end on the virtual device: detection -> alert episode ->
alert output -> spool/uplink -> /v1/ingest -> GET events (what the app reads).

The real runtime, alert policy, CloudUplink and cloud/app.py on one SimClock;
only the camera, FaceMesh, the detector and the buzzer are simulated (the
buzzer is sim/runtime.py's SimBuzzerSink: it reports each actuation, marked
simulated, and can be told to fail or to report late). No sleeps.

SIMULATED evidence: plumbing, identities and timing. A real buzzer being
audible is a Pi-only check (docs/EVENT_PIPELINE.md, "hardware-only").
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.events import Event
from sim.harness import SimStack
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
DOWN = Faults(network_down=True)


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0, ATTENTIVE),
                                 *(Segment(float(t), scene, faults) for t, scene, faults in segments)))


class _Gate:
    """Records every newly stored event the backend considers notifying for."""

    def __init__(self, app, monkeypatch) -> None:
        app.gate.reset()
        self.considered: list[str] = []
        real = app.gate.evaluate

        def evaluate(user_id, severity, source="module"):
            self.considered.append(severity)
            return real(user_id, severity, source=source)

        monkeypatch.setattr(app.gate, "evaluate", evaluate)


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path / "stack") as s:
        s.claim()
        yield s


def _drowsy(stack) -> list[dict]:
    return [e for e in stack.events() if e["type"] == "drowsiness"]


def _evidence(name: str, payload: dict) -> None:
    out = os.environ.get("SD_M83_EVIDENCE_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"{name}.json").write_text(json.dumps({"simulated": True, **payload}, indent=1, default=str))


# ---------------------------------------------------------------------- evidence of delivery
def test_an_alert_raised_offline_arrives_once_with_its_output_evidence_and_its_true_time(stack, monkeypatch):
    """Eyes close while the backend is unreachable. After reconnection the app
    gets ONE drowsiness alert, saying what the device's buzzer reported (here a
    simulated one -- never "sounded"), placed when it happened, not when it arrived."""
    gate = _Gate(stack.backend.app, monkeypatch)
    device = stack.boot(_scenario("offline", (C + 1, ATTENTIVE, DOWN), (C + 3, ASLEEP, DOWN),
                                  (C + 8, ATTENTIVE, DOWN), (C + 25, ATTENTIVE, Faults())))
    device.advance(C + 20)
    assert _drowsy(stack) == [], "nothing reaches the backend during the outage"
    queued = [e for e in device.runtime.uplink._queue if e["type"] == "drowsiness"]
    assert len(queued) == 1
    detected_wall = stack.clock.wall() - (stack.clock.monotonic() - queued[0]["_mono"])
    device.advance(10.0)
    rows = _drowsy(stack)
    _evidence("m83_offline_alert", {"rows": rows, "buzzer": device.buzzer.played})
    assert len(rows) == 1
    row = rows[0]
    assert row["alert"]["decision"] == "alerted" and row["alert"]["episode"]
    assert row["alert"]["output"] == "simulated", "a simulated buzzer is never reported as sounded"
    assert row["alert"]["outputs"][0]["name"] == "sim_buzzer" and row["alert"]["outputs"][0]["played"] >= 1
    assert row["simulated"] is True
    assert row["time_basis"] == "device_monotonic"
    assert row["occurred_at"] == pytest.approx(detected_wall, abs=0.01)
    assert row["received_at"] - row["occurred_at"] > 10.0, "delivered late, and the app can tell"
    assert gate.considered.count("loud") == sum(1 for e in stack.events() if e["severity"] == "loud")


def test_a_failing_buzzer_is_reported_failed_and_never_as_sounded(stack):
    device = stack.boot(_scenario("buzzer_fails", (C + 3, ASLEEP, Faults()), (C + 8, ATTENTIVE, Faults())))
    device.advance(C)
    device.buzzer.fail = "GPIO17: [Errno 5] Input/output error"
    device.advance(12.0)
    row = _drowsy(stack)[0]
    _evidence("m83_buzzer_failed", {"row": row})
    assert row["alert"]["output"] == "failed"
    assert row["alert"]["outputs"][0]["failed"] >= 1 and row["alert"]["outputs"][0]["played"] == 0
    assert "Input/output error" in row["alert"]["outputs"][0]["error"]


def test_a_late_output_report_updates_the_stored_alert_without_a_second_row_or_notification(stack, monkeypatch):
    gate = _Gate(stack.backend.app, monkeypatch)
    device = stack.boot(_scenario("late_report", (C + 3, ASLEEP, Faults()), (C + 8, ATTENTIVE, Faults())))
    device.advance(C)
    device.buzzer.hold = True  # the pattern is still rendering
    device.advance(8.0)
    first = _drowsy(stack)
    assert len(first) == 1 and first[0]["alert"]["output"] == "pending"
    device.buzzer.release()
    device.advance(4.0)
    rows = _drowsy(stack)
    _evidence("m83_late_report", {"before": first, "after": rows})
    assert len(rows) == 1 and rows[0]["id"] == first[0]["id"]
    assert rows[0]["alert"]["output"] == "simulated" and rows[0]["alert"]["rev"] > first[0]["alert"]["rev"]
    assert gate.considered.count("loud") == sum(1 for e in stack.events() if e["severity"] == "loud"), \
        "the report update never reached the notifier"


# ---------------------------------------------------------------------- one episode, one alert
def test_an_open_episode_re_reported_is_one_stored_alert(stack, monkeypatch):
    """The detector reports an episode that is still open again (as after a
    reconnect or a rebuilt tracker), through the production dispatch path.
    Before M8.3 that was a second stored alert, put to the notifier again
    (held back only if it fell inside the 20 s cooldown)."""
    gate = _Gate(stack.backend.app, monkeypatch)
    device = stack.boot(_scenario("re_report", (C + 3, ASLEEP, Faults()), (C + 20, ATTENTIVE, Faults())))
    device.advance(C + 8)
    runtime = device.runtime
    assert "drowsiness" in runtime.alert_manager.open_keys("module")
    again = Event.now("drowsiness", 2.0, 0.9, runtime.module_name, "re-reported")
    runtime._dispatch_module_events(runtime.module, [again], now=stack.clock.monotonic())
    device.advance(4.0)
    rows = _drowsy(stack)
    _evidence("m83_re_report", {"rows": rows, "merged": getattr(runtime.uplink, "merged_events", None)})
    assert len(rows) == 1
    assert runtime.uplink.merged_events == 1
    assert gate.considered.count("loud") == sum(1 for e in stack.events() if e["severity"] == "loud")
    assert runtime.alert_manager.delivery(again).decision == "merged", "the policy itself is unchanged"


def test_a_camera_reconnect_mid_episode_resounds_locally_but_stores_one_alert(stack):
    device = stack.boot(_scenario("camera_reconnect", (C + 3, ASLEEP, Faults()),
                                  (C + 6, ASLEEP, Faults(camera="frozen")),
                                  (C + 30, ASLEEP, Faults()), (C + 40, ATTENTIVE, Faults())))
    device.advance(C + 45)
    rows = _drowsy(stack)
    _evidence("m83_camera_reconnect", {"rows": rows, "transitions": device.sim.transitions})
    assert "failed" in device.sim.transitions, "the freeze was long enough to fail monitoring"
    assert len(rows) == 1


# ---------------------------------------------------------------------- restarts and clocks
def test_a_power_cut_mid_outage_delivers_the_alert_once_after_power_on(stack, monkeypatch):
    gate = _Gate(stack.backend.app, monkeypatch)
    outage = _scenario("cut", (C + 1, ATTENTIVE, DOWN), (C + 3, ASLEEP, DOWN), (C + 8, ATTENTIVE, DOWN))
    device = stack.boot(outage)
    device.advance(C + 12)
    spool = device.state_dir / "uplink_spool.jsonl"
    assert spool.exists() and any(json.loads(line)["type"] == "drowsiness" for line in spool.read_text().splitlines())
    device = stack.cold_restart(_scenario("after", (C + 1, ATTENTIVE, Faults())))
    device.advance(6.0)
    rows = _drowsy(stack)
    _evidence("m83_power_cut", {"rows": rows})
    assert len(rows) == 1
    assert rows[0]["alert"]["output"] == "simulated", "the buzzer report recorded before the cut survives it"
    assert rows[0]["time_basis"] == "device_clock", "no age across a reboot: the device clock, judged plausible"
    assert gate.considered.count("loud") == sum(1 for e in stack.events() if e["severity"] == "loud")


def test_a_device_wall_clock_three_days_wrong_does_not_move_the_alert(stack, monkeypatch):
    import core.events as events_module

    real = datetime

    class WrongClock(real):
        @classmethod
        def now(cls, tz=None):
            return real.now(tz) - timedelta(days=3)

    monkeypatch.setattr(events_module, "datetime", WrongClock)
    device = stack.boot(_scenario("clock", (C + 3, ASLEEP, Faults()), (C + 8, ATTENTIVE, Faults())))
    device.advance(C + 10)
    row = _drowsy(stack)[0]
    _evidence("m83_wrong_clock", {"row": row})
    device_time = real.fromisoformat(row["timestamp"]).timestamp()
    assert row["received_at"] - device_time > 2.9 * 86400, "the device's claim is kept, verbatim"
    assert row["time_basis"] == "device_monotonic"
    assert 0.0 <= row["received_at"] - row["occurred_at"] < 3.0, "placed by the backend's clock"
