"""M8.1 health freshness end to end: the virtual device against the real backend.

Every scenario asserts through GET /v1/devices (what the app reads). A
"frozen" sample is produced the way the field produces it: the health sampling
thread stops handing samples to the uplink (wedged in nmcli, say) while the
uplink keeps polling and re-sending the last one. Set SENSORDYME_M81_EVIDENCE
to a directory to write each scenario's state-transition trace there as JSON.
"""

from __future__ import annotations

import copy
import json
import os
import types
from pathlib import Path

import pytest

from core import health_stamp as hs
from sim import scenario as sc
from sim.backend import InProcessBackend
from sim.harness import SimStack
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment
from sim.uplink import TestClientTransport

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
EVIDENCE = os.environ.get("SENSORDYME_M81_EVIDENCE")


def _timeline(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, faults)
                                                  for t, scene, faults in segments)))


# ---------------------------------------------------------------------- harness
class _Recorder:
    """A transport that keeps a copy of every ingest body the device sent."""

    def __init__(self, inner) -> None:
        self.inner, self.bodies = inner, []

    def post(self, url, **kwargs):
        if kwargs.get("json") is not None:
            self.bodies.append(copy.deepcopy(kwargs["json"]))
        return self.inner.post(url, **kwargs)


class _Trace:
    def __init__(self, stack: SimStack, name: str) -> None:
        self.stack, self.name, self.steps = stack, name, []

    def mark(self, label: str, device=None) -> dict:
        row = self.stack.row()
        status = row["status"]
        fresh = status["health_freshness"]
        held = None
        if device is not None and device.runtime is not None and device.runtime.uplink is not None:
            stamp = (device.runtime.uplink._health or {}).get("stamp") or {}
            held = {"boot_id": stamp.get("boot_id"), "generation": stamp.get("generation"),
                    "revision": stamp.get("revision")}
        step = {
            "label": label,
            "sim_t": round(self.stack.clock.monotonic(), 2),
            "backend_now": round(self.stack.clock.wall(), 2),
            "app": {"online": row["online"], "stale": status["stale"], "link_stale": status["link_stale"],
                    "health_stale": status["health_stale"], "monitoring_state": status["monitoring_state"],
                    "protection": status["protection"], "reasons": status["reasons"],
                    "health_withheld": row["health"] is None},
            "freshness": {k: fresh[k] for k in ("state", "age_s", "boot_id", "generation", "repeats",
                                                "observed_at", "received_at", "device_generated_at")}
            | {"last_ignored": (fresh["last_ignored"] or {}).get("reason")},
            "device_held_sample": held,
        }
        self.steps.append(step)
        return step

    def save(self) -> None:
        if EVIDENCE:
            path = Path(EVIDENCE)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{self.name}.json").write_text(json.dumps(self.steps, indent=2))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


@pytest.fixture
def trace(request, stack):
    t = _Trace(stack, request.node.name)
    yield t
    t.save()


def _freeze(device) -> None:
    """The sampling thread stops handing samples over; the uplink keeps polling."""
    device.runtime.uplink.set_health = lambda report: None


def _thaw(device) -> None:
    del device.runtime.uplink.set_health


def _ready(step: dict) -> bool:
    app = step["app"]
    return (app["monitoring_state"] == "ready" and app["protection"] == "protecting"
            and not app["stale"] and step["freshness"]["state"] == "fresh")


def _unknown_on_live_link(step: dict) -> bool:
    app = step["app"]
    return (app["online"] is True and app["link_stale"] is False and app["health_stale"] is True
            and app["stale"] is True and app["monitoring_state"] == "unknown" and app["protection"] == "unknown"
            and app["health_withheld"] is True)


# ---------------------------------------------------------------------- scenarios
def test_frozen_snapshot_with_successful_polling_goes_unknown_then_recovers(stack, trace):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C + 2)
    assert _ready(trace.mark("baseline: live and fresh", device))

    _freeze(device)
    frozen_gen = trace.mark("health thread wedged", device)["freshness"]["generation"]
    syncs = device.runtime.uplink.syncs
    device.advance(20.0)
    step = trace.mark("20 s frozen: still inside the 30 s threshold", device)
    assert step["freshness"]["state"] == "fresh" and step["freshness"]["generation"] == frozen_gen
    device.advance(15.0)
    step = trace.mark("35 s frozen: polling, evidence stale", device)
    assert device.runtime.uplink.syncs - syncs >= 15, "the device kept polling throughout"
    assert _unknown_on_live_link(step), step
    assert step["freshness"]["generation"] == frozen_gen and step["freshness"]["repeats"] >= 15
    assert step["app"]["reasons"][0].startswith("the device is connected, but its monitoring status")
    # The device itself never stopped monitoring (M0.2 is untouched): only the evidence froze.
    assert str(device.runtime._monitoring_state) == "ready"

    _thaw(device)
    device.advance(3.0)
    step = trace.mark("health thread back: a new generation", device)
    assert _ready(step) and step["freshness"]["generation"] > frozen_gen


def test_network_outage_then_recovery_after_new_health(stack, trace):
    device = stack.boot(_timeline("outage", (C + 5, ATTENTIVE, Faults(network_down=True)), (C + 65, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C + 4)
    assert _ready(trace.mark("before the outage", device))
    device.advance(40.0)
    step = trace.mark("40 s into the outage", device)
    assert step["app"]["link_stale"] is True and step["app"]["online"] is False
    assert step["app"]["protection"] == "unknown" and step["freshness"]["state"] == "stale"
    assert step["app"]["reasons"][0].startswith("no report from the device")
    device.advance(24.0)
    step = trace.mark("network back: the newest sample arrives", device)
    assert _ready(step), step
    assert step["freshness"]["age_s"] < 8, "the delivered sample was taken during the outage, seconds ago"


def test_delayed_observation_a_never_seen_but_old_sample_is_stale_on_arrival(stack, trace):
    device = stack.boot(_timeline("outage", (C + 5, ATTENTIVE, Faults(network_down=True)), (C + 65, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C + 4)
    seen = trace.mark("before the outage", device)["freshness"]["generation"]
    device.advance(2.5)  # one failed sync during the outage holds a sample the backend never saw
    _freeze(device)
    held = trace.mark("outage: sample held, sampling wedged", device)["device_held_sample"]["generation"]
    assert held > seen
    device.advance(60.0)
    step = trace.mark("network back: the held sample is delivered 60 s late", device)
    assert step["freshness"]["generation"] == held, "a new generation to the backend..."
    assert step["freshness"]["age_s"] >= 55, "...whose own age (device monotonic) says it is old"
    assert step["freshness"]["received_at"] >= step["backend_now"] - 3, "although it was just received"
    assert _unknown_on_live_link(step), step


def test_reordered_reports_never_overwrite_newer_health(stack, trace):
    recorder = _Recorder(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=recorder)
    stack.claim()
    device.advance(C + 6)
    newest = trace.mark("in order", device)
    assert _ready(newest)
    old = next(b for b in recorder.bodies if (b.get("health") or {}).get("stamp", {}).get("generation", 0)
               < newest["freshness"]["generation"] - 1)
    old_gen = old["health"]["stamp"]["generation"]
    old["health"]["monitoring"] = {**old["health"]["monitoring"], "state": "failed"}  # visibly different
    resp = stack.api.post("/v1/ingest", json=old, headers={"X-Device-Key": stack.factory["device_key"]})
    assert resp.status_code == 200, "an out-of-order report is never a refused poll"
    step = trace.mark(f"generation {old_gen} delivered late", device)
    assert step["freshness"]["generation"] == newest["freshness"]["generation"]
    assert step["freshness"]["last_ignored"] == "out_of_order"
    assert step["app"]["monitoring_state"] == "ready", "the older (failed) sample did not overwrite"


def test_device_reboot_restarts_the_generation_and_old_boot_reports_are_ignored(stack, trace):
    recorder = _Recorder(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=recorder)
    stack.claim()
    device.advance(C + 4)
    before = trace.mark("first boot", device)
    assert _ready(before) and before["freshness"]["generation"] > 3
    pre_reboot = recorder.bodies[-1]

    device = stack.restart(sc.get("attentive"), transport=recorder)
    device.advance(3.0)
    after = trace.mark("rebooted: new boot id, generation restarts", device)
    assert after["freshness"]["boot_id"] != before["freshness"]["boot_id"]
    assert after["freshness"]["generation"] < before["freshness"]["generation"], "restart accepted explicitly"
    assert after["freshness"]["state"] == "fresh" and after["app"]["health_stale"] is False
    assert after["freshness"]["last_ignored"] is None, "a reboot is not out of order"

    resp = stack.api.post("/v1/ingest", json=pre_reboot, headers={"X-Device-Key": stack.factory["device_key"]})
    assert resp.status_code == 200
    late = trace.mark("a pre-reboot report delivered late", device)
    assert late["freshness"]["boot_id"] == after["freshness"]["boot_id"]
    assert late["freshness"]["last_ignored"] == "retired_boot"


def test_device_wall_clock_jumps_change_nothing(stack, trace):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C + 2)
    assert _ready(trace.mark("clock right", device))
    stamper = device.runtime._health_stamper
    for label, offset in (("clock jumps 2 h forward", 7200.0), ("clock jumps to 1970 (no RTC)", -stack.clock.wall()),
                          ("clock steps back 5 min (NTP)", -300.0)):
        stamper.wall = (lambda o: lambda: stack.clock.wall() + o)(offset)
        device.advance(6.0)
        step = trace.mark(label, device)
        assert _ready(step), step
        assert step["freshness"]["age_s"] < 8
        assert step["freshness"]["device_generated_at"] == pytest.approx(stack.clock.wall() + offset, abs=6)
    _freeze(device)
    stamper.wall = lambda: stack.clock.wall() + 10 * 86400  # a "future" stamp cannot keep it fresh
    device.advance(35.0)
    assert _unknown_on_live_link(trace.mark("frozen with a future wall clock", device))


def test_signed_device_clock_jumps_compose_with_the_signed_clock_answer(stack, trace, monkeypatch):
    """command_auth: required. The device's wall clock jumps; requests are refused
    for clock_skew until it adopts the backend's signed time; freshness is never
    judged by the device clock in either direction."""
    import time as real_time

    from core import uplink as uplink_mod
    from core.uplink import CloudUplink

    skew = {"s": 0.0}

    def device_wall() -> float:
        return stack.clock.wall() + skew["s"]

    monkeypatch.setattr(uplink_mod, "time", types.SimpleNamespace(
        **{**{k: getattr(real_time, k) for k in dir(real_time) if not k.startswith("_")},
           "time": device_wall, "monotonic": lambda: real_time.monotonic()}))  # noqa: PLW0108 - late lookup: SimClock replaces it
    stack.claim()
    device = CloudUplink({"base_url": stack.backend.base_url, "device_key": stack.factory["device_key"],
                          "command_auth": "required", "poll_interval_s": 0})
    device._requests = stack.transport
    stamper = hs.HealthStamper(wall=device_wall)

    def sample() -> None:
        device.set_health(stamper.stamp({"camera": {"state": "ok"},
                                         "monitoring": {"state": "ready", "reasons": []},
                                         "monitoring_active": True,
                                         "protection": {"state": "protecting", "reasons": []}}))

    def poll(seconds: float) -> None:
        for _ in range(int(seconds / 2)):
            stack.clock.advance(2.0)
            device.sync_once()

    sample()
    device.sync_once()
    step = trace.mark("signed, clocks agree")
    assert step["freshness"]["state"] == "fresh" and step["app"]["protection"] == "protecting"

    for label, jump in (("device clock +2 h", 7200.0), ("device clock -4 h", -14400.0)):
        skew["s"] += jump
        stack.clock.advance(2.0)
        sample()
        device.sync_once()
        assert not device.connected, "refused for clock_skew"
        refused = trace.mark(f"{label}: refused, backend keeps the previous sample")
        assert refused["freshness"]["generation"] == stamper.generation - 1
        device.sync_once()  # the signed clock answer was adopted for request timestamps only
        assert device.connected
        step = trace.mark(f"{label}: signed with the backend's time, new sample fresh")
        assert step["freshness"]["generation"] == stamper.generation and step["freshness"]["state"] == "fresh"
        assert step["freshness"]["age_s"] < 5
        assert step["freshness"]["device_generated_at"] == pytest.approx(device_wall(), abs=5)
    poll(34.0)
    step = trace.mark("no new sample for 34 s, still polling (signed)")
    assert _unknown_on_live_link(step), step


def test_backend_restart_keeps_freshness_state(stack, trace):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C + 2)
    _freeze(device)
    device.advance(35.0)
    before = trace.mark("frozen and stale, before the backend restart", device)
    assert _unknown_on_live_link(before)

    stack.backend.close()
    stack.backend = InProcessBackend(stack.root / "backend", stack.clock)
    stack.transport = TestClientTransport(stack.backend.client)
    device.network._inner = stack.transport
    device.advance(4.0)
    after = trace.mark("backend restarted: same database", device)
    assert _unknown_on_live_link(after), "a restart must not make a frozen sample fresh"
    assert after["freshness"]["observed_at"] == before["freshness"]["observed_at"]
    assert after["freshness"]["repeats"] > before["freshness"]["repeats"]

    _thaw(device)
    device.advance(3.0)
    assert _ready(trace.mark("health thread back", device))


def test_alerts_are_delivered_while_health_is_stale(stack, trace):
    device = stack.boot(_timeline("asleep_late", (C + 40, ASLEEP, Faults()), (C + 46, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C + 2)
    assert _ready(trace.mark("baseline", device))
    _freeze(device)
    device.advance(36.0)
    assert _unknown_on_live_link(trace.mark("evidence stale before the driver falls asleep", device))
    device.advance(12.0)
    step = trace.mark("driver asleep while health is stale", device)
    assert _unknown_on_live_link(step)
    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1 and drowsy[0]["severity"] == "loud", stack.event_types()
    assert any(p[0] == "drowsiness" for p in device.buzzer.played), "the local alert sounded"
    assert str(device.runtime._monitoring_state) == "ready", "the device's own safety state is unaffected"
