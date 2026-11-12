"""M8 lifecycle, end to end on the virtual device: POWER -> boot -> monitoring -> backend
-> app, through faults, restarts and reconnection.

The device is the real runtime (main.build_runtime, services/runtime.py, the real
uplink) against the real backend (cloud/app.py), on one SimClock. Every app-side
assertion goes through GET /v1/devices, as the iPhone app does. Each scenario
records its state transitions -- the device's monitoring state, its uplink and
backlog, and what the app is shown -- and, with M8_EVIDENCE_DIR set, writes them
as JSON evidence.

Simulated, end to end: no camera sensor, no Hailo, no Wi-Fi radio, no systemd.
What only the Pi can prove is listed in docs (and in the M8 report).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
DOWN = Faults(network_down=True)


UP = Faults()


def _timeline(name: str, *segments: tuple, first: Faults = UP) -> Scenario:
    return Scenario(name, name, (Segment(0.0, ATTENTIVE, first),
                                 *(Segment(float(t), scene, faults) for t, scene, faults in segments)))


class Timeline:
    """What the device and the app say, step by step (the scenario's evidence)."""

    def __init__(self, stack: SimStack, name: str) -> None:
        self.stack = stack
        self.name = name
        self.rows: list[dict] = []

    def mark(self, label: str) -> dict:
        device = self.stack.device
        runtime = device.runtime if device is not None else None
        entry: dict = {"t": round(self.stack.clock.monotonic(), 2), "step": label}
        if runtime is not None:
            uplink = runtime.uplink
            entry["device"] = {
                "monitoring_state": runtime._monitoring_state,
                "protection": runtime.protection_state()["state"],
                "module": runtime.module_name,
                "preset": runtime.active_preset,
                "uplink_connected": uplink.connected if uplink is not None else None,
                "backlog": uplink.backlog() if uplink is not None else None,
                "config": runtime._config_report()["source"],
                "buzzer_actuations": len(device.buzzer.played),
            }
        else:
            entry["device"] = "powered off"
        if self.stack.device_id is not None:
            row = self.stack.row()
            status = row["status"]
            entry["app"] = {"online": row["online"], "stale": status["stale"],
                            "monitoring_state": status["monitoring_state"], "protection": status["protection"],
                            "monitoring_active": status["monitoring_active"],
                            "reasons": status.get("reasons", [])[:3],
                            "events": [e["type"] for e in self.stack.events()]}
        self.rows.append(entry)
        return entry

    def save(self) -> None:
        out = os.environ.get("M8_EVIDENCE_DIR")
        if out:
            path = Path(out) / f"{self.name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"scenario": self.name, "simulated": True, "timeline": self.rows},
                                       indent=2, default=str))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


@pytest.fixture
def monitor(stack):
    return InvariantMonitor(stack)


def _app(entry: dict) -> dict:
    return entry["app"]


# ---------------------------------------------------------------------- first boot
def test_first_boot_with_no_state_comes_up_protecting(stack, monitor):
    timeline = Timeline(stack, "first_boot")
    assert not stack.state_dir.exists()
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    first = timeline.mark("powered on, claimed")
    assert first["device"]["monitoring_state"] in (None, "starting")
    device.advance(1.0)
    assert _app(timeline.mark("1 s"))["monitoring_state"] == "starting"
    device.advance(C)
    ready = timeline.mark("after calibration")
    assert _app(ready)["monitoring_state"] == "ready" and _app(ready)["protection"] == "protecting"
    assert (stack.state_dir / "data" / "config.last_good.yaml").exists()
    assert not (stack.state_dir / "uplink_spool.jsonl").exists(), "nothing undelivered"
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- detector degradation
def test_detector_degradation_then_recovery(stack, monitor):
    timeline = Timeline(stack, "detector_degradation")
    device = stack.boot(sc.get("detector_failure"))
    stack.claim()
    device.advance(C)
    timeline.mark("ready, detector about to fail")
    device.advance(10.0)
    degraded = timeline.mark("detector failing")
    assert _app(degraded)["monitoring_state"] == "degraded"
    assert _app(degraded)["monitoring_active"] is True, "face monitoring continues"
    assert _app(degraded)["protection"] == "protecting", "driver_safety does not require objects"
    device.advance(80.0)
    recovered = timeline.mark("detector back, module retry backoff elapsed")
    assert _app(recovered)["monitoring_state"] == "ready"
    states = [r["device"]["monitoring_state"] for r in timeline.rows]
    assert states == ["ready", "degraded", "ready"], states
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- lost backend connection
def test_lost_backend_local_alerts_continue_queue_bounded_and_flush_once(stack, monitor):
    script = _timeline("backend_lost", (C + 5, ATTENTIVE, DOWN), (C + 15, ASLEEP, DOWN),
                       (C + 20, ATTENTIVE, DOWN), (C + 50, ATTENTIVE, Faults()))
    timeline = Timeline(stack, "lost_backend_connection")
    device = stack.boot(script)
    stack.claim()
    device.advance(C + 5)
    timeline.mark("online, ready")
    device.advance(20.0)
    offline = timeline.mark("backend unreachable 20 s, driver fell asleep at 10 s")
    assert offline["device"]["uplink_connected"] is False
    assert offline["device"]["buzzer_actuations"] > 0, "the local alert sounded with no cloud"
    assert offline["device"]["monitoring_state"] == "ready", "monitoring does not depend on the backend"
    assert offline["device"]["backlog"]["queued"] >= 1
    assert _app(offline)["stale"] is False or _app(offline)["monitoring_state"] == "unknown"
    device.advance(20.0)
    stale = timeline.mark("40 s offline")
    assert _app(stale)["stale"] is True and _app(stale)["monitoring_state"] == "unknown"
    assert "drowsiness" not in _app(stale)["events"]
    device.advance(10.0)
    back = timeline.mark("backend reachable again")
    assert _app(back)["stale"] is False and _app(back)["monitoring_state"] == "ready"
    assert _app(back)["events"].count("drowsiness") == 1, "delivered, once"
    assert back["device"]["backlog"]["queued"] == 0
    assert not (stack.state_dir / "uplink_spool.jsonl").exists()
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- device restart
def test_power_cut_during_an_outage_keeps_the_queued_alert(stack, monitor):
    """Before M8 the backlog was memory only: an alert raised offline was lost if the
    device lost power (ignition off) before the network came back."""
    offline = _timeline("asleep_offline", (C + 5, ATTENTIVE, DOWN), (C + 10, ASLEEP, DOWN),
                        (C + 15, ATTENTIVE, DOWN), first=Faults())
    timeline = Timeline(stack, "device_restart_power_cut_during_outage")
    device = stack.boot(offline)
    stack.claim()
    device.advance(C + 20)
    before = timeline.mark("asleep while offline: alerted locally, queued")
    assert before["device"]["backlog"]["queued"] >= 1
    assert (stack.state_dir / "uplink_spool.jsonl").exists(), "queued on disk as it was raised"

    stack.power_cut()
    timeline.mark("power cut: no shutdown code ran")
    device = stack.boot(_timeline("still_offline", (20.0, ATTENTIVE, Faults()), first=DOWN))
    restored = timeline.mark("powered on, still offline")
    assert restored["device"]["backlog"]["restored"] >= 1
    device.advance(30.0)
    after = timeline.mark("network back")
    assert _app(after)["stale"] is False
    assert _app(after)["events"].count("drowsiness") == 1, "the alert raised before the power cut, once"
    assert not (stack.state_dir / "uplink_spool.jsonl").exists()
    timeline.save()
    monitor.assert_clean()


def test_restart_and_reconnection_keep_identity_mode_rules_and_nodes(stack, monitor):
    from test_sim_e2e import _activate_phone_rule

    timeline = Timeline(stack, "device_restart_and_reconnection")
    device = stack.boot(sc.get("attentive"))
    device_id = stack.claim()
    device.advance(C)
    hardware_id = stack.row()["health"]["device_id"]
    assert stack.set_preset("driver_safety:long_haul").status_code == 200
    device.advance(4.0)
    _activate_phone_rule(stack)
    device.advance(4.0)
    node = stack.api.post(f"/v1/devices/{device_id}/nodes", json={"node": "people_counter", "enabled": True},
                          headers=stack.headers)
    assert node.status_code == 200, node.text
    device.advance(4.0)
    configured = timeline.mark("configured: preset, rule, node")
    assert "people_counter" in device.runtime._companions

    for kind in ("application restart", "cold restart (power cut)"):
        if kind.startswith("application"):
            device = stack.restart(sc.get("attentive"))
        else:
            stack.power_cut()
            device = stack.boot(sc.get("attentive"))
        booted = timeline.mark(f"{kind}: booted, before any backend contact")
        assert device.identity.hardware_id == hardware_id
        assert booted["device"]["preset"] == "driver_safety:long_haul"
        assert [r.detection for r in device.runtime.rule_engine.rules] == ["cell phone"]
        assert "people_counter" in device.runtime._companions, f"{kind}: the node came back from disk"
        device.advance(6.0)
        reconnected = timeline.mark(f"{kind}: reconnected")
        row = stack.row()
        assert row["id"] == device_id and row["health"]["device_id"] == hardware_id, "same device, still paired"
        assert _app(reconnected)["monitoring_state"] == "ready"
        assert row["active_preset"] == "driver_safety:long_haul"
    assert configured["app"]["online"]
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- camera absent at boot
def test_camera_absent_at_boot_fails_truthfully_then_recovers(stack, monitor):
    script = Scenario("camera_absent_at_boot", "no frames until 25 s",
                      (Segment(0.0, ATTENTIVE, Faults(camera="frozen")), Segment(25.0, ATTENTIVE, Faults())))
    timeline = Timeline(stack, "camera_absent_at_boot")
    device = stack.boot(script)
    stack.claim()
    device.advance(5.0)
    assert _app(timeline.mark("no frame yet (grace)"))["monitoring_state"] == "starting"
    device.advance(10.0)
    failed = timeline.mark("no frame 15 s after boot")
    assert _app(failed)["monitoring_state"] == "failed" and _app(failed)["protection"] == "not_protecting"
    assert any("camera" in r for r in _app(failed)["reasons"]), _app(failed)["reasons"]
    device.advance(10.0 + C + 2)
    recovered = timeline.mark("camera producing frames, calibrated")
    assert _app(recovered)["monitoring_state"] == "ready"
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- recovery: model + config
def test_recovery_from_a_boot_time_model_failure_and_a_damaged_config(stack, monitor):
    timeline = Timeline(stack, "recovery_model_and_config")
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    timeline.mark("healthy boot recorded as last known good")
    device = stack.restart(sc.get("model_failure"), config_damage="invalid_yaml")
    device.advance(10.0)
    failed = timeline.mark("restart: config.yaml broken, model fails to load")
    assert failed["device"]["config"] == "last_known_good"
    assert _app(failed)["monitoring_state"] == "failed"
    device.sim.model_failure = False
    device.advance(30.0 + C)
    recovered = timeline.mark("model rebuilt in-process; still on the last known good config")
    assert _app(recovered)["monitoring_state"] == "ready" and _app(recovered)["protection"] == "protecting"
    assert stack.row()["health"]["config"]["source"] == "last_known_good"
    timeline.save()
    monitor.assert_clean()


# ---------------------------------------------------------------------- signed uplink in-process
def test_a_required_command_auth_device_round_trips_through_the_in_process_backend(stack, monkeypatch):
    """sim.uplink.TestClientTransport dropped `data=`, the raw body a `required`-mode device
    signs (X-SD-Auth), so such a device could not run in the simulator at all."""
    from sim.device import VirtualDevice

    plain = VirtualDevice.config

    def required(self):
        config = plain(self)
        config["cloud"] = {**config["cloud"], "command_auth": "required"}
        return config

    monkeypatch.setattr(VirtualDevice, "config", required)
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    assert device.runtime.uplink.command_auth_mode == "required"
    device.advance(C)
    status = stack.status()
    assert status["stale"] is False and status["monitoring_state"] == "ready", status
    assert device.network.sent > 0 and device.runtime.uplink.connected
    # A signed command back the other way: stop, verified and applied once.
    assert stack.protection("stop", confirm=True).status_code == 200
    device.advance(4.0)
    assert stack.status()["protection"] == "stopped"
    assert device.runtime.uplink.command_auth_rejections == 0
