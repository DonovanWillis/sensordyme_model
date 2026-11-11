"""End-to-end: the virtual SensorDyme against the real backend, on one SimClock.

The device is a real DeviceRuntime built by main.build_runtime (sim/), with
scripted perception in place of camera + models. The backend is cloud/app.py
behind a TestClient. Every assertion goes through what the app uses: GET
/v1/devices (the typed `status` block), GET /v1/devices/{id}/events, and the
codes on refused requests. Time is the SimClock's: a 30 s outage takes
milliseconds, and every run is the same run.
"""

from __future__ import annotations

import json

import pytest

from sim import scenario as sc
from sim.harness import SimStack
from sim.perception import Scene
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _timeline(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, faults)
                                                  for t, scene, faults in segments)))


ASLEEP_ONCE = _timeline("asleep_once", (C + 3, ASLEEP, Faults()), (C + 8, ATTENTIVE, Faults()))
PHONE_VISIBLE = Scene(phone=True, label="phone on the dash")


# ---------------------------------------------------------------------- (1) first boot
def test_first_boot_unclaimed_then_claimed_reaches_ready(stack):
    device = stack.boot(sc.get("attentive"))
    device.advance(4.0)
    assert stack.devices() == [], "an unclaimed device must not appear on any account"

    stack.claim()
    device.advance(C)
    status = stack.status()
    assert status["monitoring_state"] == "ready", status
    assert status["monitoring_active"] is True
    assert status["protection"] == "protecting"
    assert status["stale"] is False
    assert status["role"] == "driver_safety"
    assert set(status["capabilities"]["available"]) >= {"camera", "inference", "face", "objects", "audible_alert"}

    # STARTING came first: READY is earned by analysed frames, not assumed.
    assert device.sim.transitions[:2] == ["starting", "ready"], device.sim.transitions

    # Marked simulated end to end.
    row = stack.row()
    assert row["health"]["simulated"] is True
    assert row["health"]["device_id"].startswith("SD-SIM-")
    assert row["monitoring"]["simulated"] is True
    actuators = row["health"]["monitoring"]["actuators"]
    assert [a["name"] for a in actuators if a["audible"]] == ["sim_buzzer"]
    assert "SIMULATED" in actuators[0]["detail"]
    # The backend stores only SD-XXXXXXXX ids, so a virtual device has none there.
    assert row["hardware_id"] is None


# ---------------------------------------------------------------------- (2) drowsiness
class _LoseResponses:
    """The request reaches the backend; the answer never reaches the device."""

    def __init__(self, inner, times: int) -> None:
        self._inner, self.remaining = inner, times

    def post(self, url, **kwargs):
        response = self._inner.post(url, **kwargs)
        if self.remaining > 0 and kwargs.get("json", {}).get("events"):
            self.remaining -= 1
            import requests

            raise requests.ConnectionError("SIMULATED: response lost after the backend stored the batch")
        return response


def test_driver_asleep_is_one_drowsiness_episode_visible_once(stack):
    lossy = _LoseResponses(stack.transport, times=1)
    device = stack.boot(ASLEEP_ONCE, transport=lossy)
    stack.claim()
    device.advance(C + 15)

    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1, stack.event_types()
    assert drowsy[0]["severity"] == "loud"
    assert drowsy[0]["model"]["detector"]["backend"] == "simulated"
    assert lossy.remaining == 0, "the lost-response retry path was not exercised"
    # The local alert: one critical episode on the (simulated) buzzer.
    assert next(p for p in device.buzzer.played if p[0] == "drowsiness")[1] == "critical"
    assert stack.status()["monitoring_state"] == "ready"


@pytest.mark.parametrize(("name", "expected"), [
    ("phone", {"phone_use", "eyes_down"}),
    ("distracted", {"distraction"}),
    ("absent", {"attention_lost"}),
])
def test_scripted_scenes_drive_the_real_module_to_its_own_events(stack, name, expected):
    device = stack.boot(sc.get(name))
    stack.claim()
    device.advance(C + 16)
    types = set(stack.event_types())
    assert expected <= types, types
    assert "drowsiness" not in types, "eyes were open throughout"


# ---------------------------------------------------------------------- (3) presets
def test_preset_switch_is_applied_and_a_role_forbidden_preset_is_refused_with_a_code(stack):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)

    response = stack.set_preset("driver_safety:strict")
    assert response.status_code == 200, response.text
    device.advance(4.0)
    row = stack.row()
    assert row["active_preset"] == "driver_safety:strict"
    assert row["desired_preset"] is None, "the command never converged"
    assert device.runtime.active_preset == "driver_safety:strict"
    assert row["status"]["monitoring_state"] == "ready"

    refused = stack.set_preset("security_monitoring")
    assert refused.status_code == 409
    assert refused.json()["code"] == "role_forbids_preset"
    device.advance(4.0)
    assert stack.row()["active_preset"] == "driver_safety:strict"


def test_a_forbidden_preset_queued_before_the_device_reported_its_role_is_dropped_with_a_code(stack):
    stack.claim()  # paired, never powered on: the backend does not know its role yet
    assert stack.set_preset("security_monitoring").status_code == 200

    device = stack.boot(sc.get("attentive"))
    device.advance(C)
    status = stack.status()
    assert status["preset_rejection"]["code"] == "role_forbids_preset"
    assert status["preset_rejection"]["preset"] == "security_monitoring"
    assert device.runtime.module_name == "driver_attention"
    assert stack.row()["desired_preset"] is None


# ---------------------------------------------------------------------- (4) stop / resume
def test_explicit_stop_and_resume_through_the_backend(stack):
    script = _timeline("stop_resume", (C + 10, ASLEEP, Faults()), (C + 15, ATTENTIVE, Faults()),
                       (C + 25, ASLEEP, Faults()), (C + 31, ATTENTIVE, Faults()))
    device = stack.boot(script)
    stack.claim()
    device.advance(C)

    unconfirmed = stack.protection("stop")
    assert unconfirmed.status_code == 400 and unconfirmed.json()["code"] == "confirmation_required"

    stopped = stack.protection("stop", confirm=True)
    assert stopped.status_code == 200
    command_id = stopped.json()["command_id"]
    device.advance(5.0)
    status = stack.status()
    assert status["protection"] == "stopped"
    assert status["monitoring_state"] == "stopped"
    assert status["monitoring_active"] is False
    assert stack.row()["health"]["protection"]["last_command_id"] == command_id

    device.advance(10.0)  # asleep while stopped: nothing is analysed, nothing alerts
    assert "drowsiness" not in stack.event_types()

    resumed = stack.protection("resume")
    assert resumed.status_code == 200
    device.advance(6.0)
    status = stack.status()
    assert status["protection"] == "protecting"
    assert status["monitoring_state"] == "ready"
    assert stack.row()["health"]["protection"]["last_command_id"] == resumed.json()["command_id"]

    device.advance(12.0)  # asleep again, now protected
    assert stack.event_types().count("drowsiness") == 1


# ---------------------------------------------------------------------- (5) camera freeze
def test_camera_freeze_fails_with_reasons_and_recovers_to_ready(stack):
    device = stack.boot(sc.get("camera_freeze"))
    stack.claim()
    device.advance(C + 5)
    assert stack.status()["monitoring_state"] == "ready"

    device.advance(6.0)  # frozen since C+5
    status = stack.status()
    assert status["monitoring_state"] == "failed"
    assert status["protection"] == "not_protecting"
    assert status["monitoring_active"] is False
    assert any("camera" in r for r in status["reasons"]), status["reasons"]
    assert any(c["name"] == "camera" for c in status["capabilities"]["impaired"])
    assert "monitoring_impaired" in stack.event_types(), "FAILED must be signalled, not only reported"

    device.advance(20.0)  # the sensor is back at C+25
    status = stack.status()
    assert status["monitoring_state"] == "ready", status
    assert status["protection"] == "protecting"
    assert device.sim.transitions[-3:] == ["ready", "failed", "ready"]


def test_frame_drops_and_stalls_are_told_apart(stack):
    device = stack.boot(_timeline("drops_then_stall", (C, ATTENTIVE, Faults(camera="dropping")),
                                  (C + 20, ATTENTIVE, Faults(camera="stalled")),
                                  (C + 40, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C + 15)
    assert stack.status()["monitoring_state"] == "ready", "half the frames is still a working camera"
    device.advance(20.0)
    assert "failed" in device.sim.transitions[device.sim.transitions.index("ready"):], \
        "multi-second gaps must not read as a working camera"
    device.advance(10.0)
    assert stack.status()["monitoring_state"] == "ready"


# ---------------------------------------------------------------------- detector faults
def test_detector_failure_degrades_then_recovers(stack):
    device = stack.boot(sc.get("detector_failure"))
    stack.claim()
    device.advance(C + 10)
    status = stack.status()
    assert status["monitoring_state"] == "degraded", status
    assert status["monitoring_active"] is True, "face-based monitoring continues without the detector"
    assert any(c["name"] == "objects" for c in status["capabilities"]["impaired"])
    assert any("SIMULATED detector fault" in r for r in status["reasons"]), status["reasons"]

    device.advance(80.0)  # failing until C+45, then the module's own retry backoff
    assert stack.status()["monitoring_state"] == "ready"


def test_a_model_that_cannot_be_loaded_boots_failed_and_reachable(stack):
    device = stack.boot(sc.get("model_failure"))
    stack.claim()
    device.advance(10.0)
    status = stack.status()
    assert status["stale"] is False, "a FAILED device must stay reachable"
    assert status["monitoring_state"] == "failed"
    assert status["protection"] == "not_protecting"
    # The cause is on the capability; the top-level reason names what it costs.
    module = next(c for c in status["capabilities"]["impaired"] if c["name"] == "module")
    assert "SIMULATED model failure" in module["reason"]
    assert any("face" in r for r in status["reasons"]), status["reasons"]


def test_no_audible_output_on_a_driver_device_is_failed(stack):
    device = stack.boot(sc.get("no_buzzer"))
    stack.claim()
    device.advance(C)
    status = stack.status()
    assert status["monitoring_state"] == "failed"
    assert any("audible_alert" in r for r in status["reasons"]), status["reasons"]


# ---------------------------------------------------------------------- (6) outage
def test_backend_outage_goes_stale_device_keeps_monitoring_and_reconnects(stack):
    script = _timeline("outage", (C + 5, ATTENTIVE, Faults(network_down=True)),
                       (C + 15, Scene(eyes_closed=True, label="asleep offline"),
                        Faults(network_down=True)),
                       (C + 20, ATTENTIVE, Faults(network_down=True)),
                       (C + 60, ATTENTIVE, Faults()))
    device = stack.boot(script)
    stack.claim()
    device.advance(C + 5)
    assert stack.status()["stale"] is False

    device.advance(10.0)
    assert stack.status()["stale"] is False, "10 s of silence is not yet stale"
    device.advance(25.0)  # 35 s without a report
    row = stack.row()
    assert row["online"] is False
    assert row["health"] is None, "a stale health report must be withheld"
    status = row["status"]
    assert status["stale"] is True
    assert status["monitoring_state"] == "unknown"
    assert status["protection"] == "unknown"
    assert status["monitoring_active"] is None
    # Offline, the device kept watching and alerted locally.
    assert any(p[0] == "drowsiness" for p in device.buzzer.played)
    assert device.runtime.uplink.connected is False
    assert "drowsiness" not in stack.event_types()

    device.advance(25.0)  # back online at C+60
    status = stack.status()
    assert status["stale"] is False and status["monitoring_state"] == "ready"
    assert stack.event_types().count("drowsiness") == 1, "the queued alert arrives, once"
    assert device.network.refused > 0


# ---------------------------------------------------------------------- (7) restart
def test_restart_keeps_identity_pairing_mode_and_rules(stack):
    device = stack.boot(sc.get("attentive"))
    device_id = stack.claim()
    device.advance(C)
    hardware_id = stack.row()["health"]["device_id"]
    assert stack.set_preset("driver_safety:long_haul").status_code == 200
    device.advance(4.0)  # the mode converges before the next request replaces desired_preset
    assert device.runtime.active_preset == "driver_safety:long_haul"
    _activate_phone_rule(stack)
    device.advance(6.0)
    assert stack.status()["rules"]["active"]

    device = stack.restart(sc.get("attentive"))  # power cycle; no key given: it is read back
    assert device.identity.hardware_id == hardware_id
    assert device.runtime.active_preset == "driver_safety:long_haul", "the owner's mode survives a restart"
    assert [r.detection for r in device.runtime.rule_engine.rules] == ["cell phone"]
    device.advance(6.0)
    row = stack.row()
    assert row["id"] == device_id and row["health"]["device_id"] == hardware_id
    assert row["active_preset"] == "driver_safety:long_haul"
    assert row["status"]["monitoring_state"] == "ready", row["status"]
    assert row["status"]["rules"]["active"]
    assert (stack.state_dir / "calibration.json").exists(), "calibration persisted, not redone"


# ---------------------------------------------------------------------- (8) custom rules
PHONE_RULE = {"target": "cell phone", "predicate": "present", "severity": "warning", "persist_s": 1.0,
              "notify": "push", "description": "phone visible"}


def _activate_phone_rule(stack) -> str:
    preview = stack.ai_configure("tell me when a phone is visible")
    assert preview.status_code == 200, preview.text
    draft = preview.json()["draft_version"]
    activated = stack.set_preset("", draft_version=draft, rules=[PHONE_RULE])
    assert activated.status_code == 200, activated.text
    assert activated.json()["rules_active"] == 1
    return activated.json()["rules_version"]


def test_custom_rule_draft_activate_runs_and_an_unsupported_request_is_refused(stack):
    script = _timeline("phone_on_dash", (C + 10, PHONE_VISIBLE, Faults()), (C + 16, ATTENTIVE, Faults()))
    device = stack.boot(script)
    stack.claim()
    device.advance(C)

    # A stale draft is refused with its own code, not as "any 409".
    stale = stack.set_preset("", draft_version="0" * 16, rules=[PHONE_RULE])
    assert stale.status_code == 409 and stale.json()["code"] == "draft_superseded"

    _activate_phone_rule(stack)
    device.advance(6.0)
    status = stack.status()
    assert status["rules"]["active"] and status["rules"]["rejected"] is None
    assert status["mode"] == "standard_plus_custom", "custom rules ADD to driver safety, never replace it"
    assert status["monitoring_state"] == "ready"
    assert device.runtime.module_name == "driver_attention"

    device.advance(10.0)  # the phone is on the dash from C+10
    custom = [e for e in stack.events() if e["source"] == "custom_rule"]
    assert [e["type"] for e in custom] == ["cell phone"]

    # Unsupported, asked in words: nothing is proposed to run, and it says why.
    refused = stack.ai_configure("count the birds that land in my garden")
    assert refused.status_code == 200
    body = refused.json()
    assert body["rules"] == [] and body["unsupported"], body

    # Unsupported, as a rule the backend passes but THIS device cannot run:
    # refused as a whole by the device, the previous rule keeps running.
    preview = stack.ai_configure("tell me when a dog is in the car")
    dog = stack.set_preset("", draft_version=preview.json()["draft_version"],
                           rules=[{"target": "dog", "predicate": "present"}])
    assert dog.status_code == 200, dog.text
    device.advance(6.0)
    status = stack.status()
    assert status["rules"]["rejected"] is not None, status["rules"]
    assert any("dog" in r for r in status["rules"]["rejected"]["reasons"])
    assert status["last_rejection"]["kind"] == "rules"
    assert [r.detection for r in device.runtime.rule_engine.rules] == ["cell phone"]


def test_the_status_block_never_claims_more_than_the_device_reported(stack):
    """No always-green: every 'available' capability is one the device itself reported."""
    device = stack.boot(sc.get("detector_failure"))
    stack.claim()
    device.advance(C + 10)
    row = stack.row()
    reported = row["health"]["monitoring"]["capabilities"]
    for name in row["status"]["capabilities"]["available"]:
        assert reported[name]["state"] == "available", (name, reported[name])
    assert json.dumps(row).count("SD-SIM-") >= 1
