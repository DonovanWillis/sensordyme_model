"""M8.5 journey gap-closers, app side (docs/ACCEPTANCE.md journey matrix).

The M8.5 acceptance audit mapped the owner's 30 journeys onto the existing
suites and found four proven only in part end to end. Each test here is a
SimStack run -- the real backend (cloud/app.py) and the real device runtime on
one SimClock, with scripted camera/models: SIMULATED, never a hardware result.

* Journey 11 (supported distraction event): phone use was proven end to end by
  event TYPE only (tests/test_sim_e2e.py); here the stored phone_use row is a
  loud alert with its output evidence, like the drowsiness journey.
* Journey 14 (app notification handling): the gate and send_push were tested
  backend-only; here a loud alert raised ON THE DEVICE reaches the account's
  configured push channel exactly once, labelled "(simulated)". The HTTP leg
  to the push provider itself is tests/test_phase1_cloud_notify.py; delivery
  INTO the iOS app stays uncovered (there is no APNs channel -- the app polls).
* Journey 18 (camera privacy while stopped): the per-viewer frame policy is
  tests/test_m84_privsec_camera_device.py (standalone dashboard) and the owner
  command path tests/test_m84_privsec_camera_backend.py; here the missing link:
  a stop issued through the BACKEND closes the running device's remote /stream
  gate, the app is told, and resume reopens it.
* Journey 26 (configuration restoration): the LKG path was proven on an
  unconfigured device (tests/test_m8_lifecycle_boot.py) and preset/zones/rules
  each separately across clean restarts; here all three come back together
  after a reboot onto a damaged config.yaml.
"""

from __future__ import annotations

import pytest

from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
PHONE_RULE = {"target": "cell phone", "predicate": "present", "severity": "warning", "persist_s": 1.0,
              "notify": "push", "description": "phone visible"}


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene) for t, scene in segments)))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


# ---------------------------------------------------------------- journey 11: phone use, end to end
def test_phone_use_reaches_the_app_loud_with_its_output_evidence(stack):
    """The M8.5-supported distraction (a handled phone, eyes off the road) is not
    just an event type: it is a loud, simulated-labelled alert whose in-cabin
    output evidence reaches history, exactly like the drowsiness journey."""
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("phone"))
    stack.claim()
    device.advance(C + 16)

    phone = [e for e in stack.events() if e["type"] == "phone_use"]
    assert len(phone) == 1, stack.event_types()
    event = phone[0]
    assert event["severity"] == "loud" and event["simulated"] is True
    assert event["alert"]["decision"] == "alerted" and event["alert"]["output"] == "simulated"
    output = event["alert"]["outputs"][0]
    assert output["name"] == "sim_buzzer" and output["played"] >= 1 and output["failed"] == 0
    assert ("phone_use", "warning") in [(p[0], p[1]) for p in device.buzzer.played]
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


# ---------------------------------------------------------------- journey 14: device -> push channel
def test_a_loud_alert_from_the_device_is_pushed_once_and_labelled_simulated(stack, monkeypatch):
    """Device -> real ingest -> notification gate -> send_push, in one chain.

    send_push is recorded here (its real HTTP delivery to the provider is
    tests/test_phase1_cloud_notify.py); what this pins is that an alert raised
    on the device reaches the account's configured topic exactly once and that
    a simulated device can never pass for a real one on a lock screen."""
    pushes: list[tuple[str, str]] = []
    monkeypatch.setattr(stack.backend.app, "send_push",
                        lambda user_row, title, body: pushes.append((user_row["ntfy_topic"], title)))
    # cloud/notify.py's gate is one per process and its cooldown runs on the
    # real clock, so an earlier test's notification would silence this one.
    stack.backend.app.gate.reset()
    assert stack.api.post("/v1/account/ntfy", json={"ntfy_topic": "m85-journey-topic"},
                          headers=stack.headers).status_code == 200

    monitor = InvariantMonitor(stack)
    device = stack.boot(_scenario("one_episode", (C + 20, ASLEEP), (C + 25, ATTENTIVE)))
    stack.claim()
    device.advance(C + 30)

    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1 and drowsy[0]["severity"] == "loud"
    assert len(pushes) == 1, pushes
    topic, title = pushes[0]
    assert topic == "m85-journey-topic"
    assert title.endswith("(simulated)"), title

    device.advance(25.0)  # attentive again, past the notify cooldown: still one push
    assert len(pushes) == 1 and len([e for e in stack.events() if e["type"] == "drowsiness"]) == 1
    monitor.assert_clean()


# ---------------------------------------------------------------- journey 18: stop closes the stream
def test_a_backend_stop_closes_the_remote_stream_gate_and_resume_reopens_it(stack):
    """A stop issued in the app (through the backend, not the dashboard) is what
    closes the device's remote /stream gate, and the app is told the reason.
    What each kind of viewer is then sent frame by frame is proven in
    tests/test_m84_privsec_camera_device.py against the same gate."""
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"), dashboard_port=0)
    stack.claim()
    device.advance(C)
    dash = device.runtime.dashboard
    assert dash.stream_access(local=False) == "monitoring"
    camera = stack.status()["camera_mode"]
    assert camera["supported"] is True and camera["remote_stream"] == "monitoring"

    stop = stack.protection("stop", confirm=True)
    assert stop.status_code == 200
    device.advance(5.0)
    status = stack.status()
    assert status["monitoring_state"] == "stopped"
    assert status["protection_command"]["outcome"] == "confirmed"
    assert dash.stream_access(local=False) == "off_while_stopped", \
        "from the confirmed stop on, a remote viewer is sent nothing"
    assert dash.stream_access(local=True) == "local_maintenance"
    camera = status["camera_mode"]
    assert camera["remote_stream"] == "off_while_stopped" and camera["camera_only"] is False

    stack.protection("resume")
    device.advance(8.0)
    assert stack.status()["monitoring_state"] == "ready"
    assert dash.stream_access(local=False) == "monitoring"
    assert stack.status()["camera_mode"]["remote_stream"] == "monitoring"
    monitor.assert_clean()


# ---------------------------------------------------------------- journey 26: everything restored
def test_preset_zones_and_rules_all_return_after_a_reboot_onto_the_last_known_good_config(stack):
    """tests/test_m8_lifecycle_boot.py proves the LKG path on an unconfigured
    device; the restart tests prove preset/zones/rules each on a clean reboot.
    This is the owner's actual journey: a fully configured device reboots onto
    a damaged config.yaml and everything is still there, in the app and on the
    device, while the fault is reported."""
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"))
    device_id = stack.claim()
    device.advance(C)

    zones = [{"name": "dash", "kind": "restricted", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}]
    saved = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": zones}, headers=stack.headers)
    assert saved.status_code == 200 and saved.json()["rejected"] == 0
    assert stack.set_preset("driver_safety:strict").status_code == 200
    device.advance(4.0)
    draft = stack.ai_configure("tell me when a phone is visible").json()
    activated = stack.set_preset("", draft_version=draft["draft_version"], rules=[PHONE_RULE])
    assert activated.status_code == 200 and activated.json()["rules_active"] == 1, activated.text
    device.advance(6.0)
    row = stack.row()
    assert row["active_preset"] == "driver_safety:strict" and row["status"]["rules"]["active"]
    assert [z.name for z in device.runtime._effective_zones(device.runtime.module_name)] == ["dash"]

    device = stack.restart(sc.get("attentive"), config_damage="truncated")
    device.advance(C + 4)
    row = stack.row()
    assert row["id"] == device_id and row["online"], "same key from the saved copy: still paired"
    reported = row["health"]["config"]
    assert reported["source"] == "last_known_good" and reported["fault"], reported
    status = row["status"]
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting"
    assert row["active_preset"] == "driver_safety:strict"
    assert status["rules"]["active"] and status["rules"]["rejected"] is None, status["rules"]
    assert [z.name for z in device.runtime._effective_zones(device.runtime.module_name)] == ["dash"], \
        "the owner's drawn area is back on the device after the LKG boot"
    monitor.assert_clean()
