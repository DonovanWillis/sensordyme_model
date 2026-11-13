"""M8.4 full user journeys on the digital twin, at the backend/device level.

The iOS stream covers the UI; this proves the same journeys through the API
the app calls, against the real backend and the real device runtime (scripted
camera and models), with the truth InvariantMonitor attached so no transition
anywhere in a journey may contradict itself. Each step asserts on real API
responses and on what the device actually did. Journeys already proven
elsewhere are mapped, not repeated (docs/DIGITAL_TWIN.md, "Journeys").

With SD_M84_EVIDENCE_DIR set, each journey writes a small JSON trace of its
steps (simulated: true) for the acceptance evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core import state_model
from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.perception import Scene
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
PHONE_ON_DASH = Scene(phone=True, label="phone on the dash")
PHONE_RULE = {"target": "cell phone", "predicate": "present", "severity": "warning", "persist_s": 1.0,
              "notify": "push", "description": "phone visible"}


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, *rest) for t, scene, *rest in segments)))


class Trace:
    """The steps of one journey, as evidence."""

    def __init__(self, name: str, stack: SimStack) -> None:
        self.name, self.stack, self.steps = name, stack, []

    def step(self, what: str, *, app_view: bool = True, **observed) -> None:
        user = None
        if app_view and self.stack.device_id is not None:
            rows = [d for d in self.stack.devices() if d["id"] == self.stack.device_id]
            user = state_model.derive(rows[0], self.stack.clock.wall()).state if rows else None
        self.steps.append({"step": what, "user_state": user, **observed})

    def write(self) -> None:
        out = os.environ.get("SD_M84_EVIDENCE_DIR")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / f"journey_{self.name}.json").write_text(
                json.dumps({"simulated": True, "journey": self.name, "steps": self.steps}, indent=1, default=str))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _signup(stack, email: str) -> dict:
    response = stack.api.post("/v1/signup", json={"email": email, "password": "another-strong-pass-1"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


# ---------------------------------------------------------------------- the whole product, in order
def test_the_whole_product_journey_from_first_boot_to_sign_out(stack):
    """Account -> discovery -> claim -> camera setup -> mode -> monitoring
    confirmed -> alert with output evidence -> history -> explain -> Ask Jupiter
    review + activation (confirmed only by the device) -> stop -> resume ->
    disconnect -> reconnect -> reboot -> sign out."""
    monitor = InvariantMonitor(stack)
    trace = Trace("whole_product", stack)
    script = _scenario("journey", (C + 20, ASLEEP), (C + 25, ATTENTIVE),
                       (C + 60, PHONE_ON_DASH), (C + 66, ATTENTIVE))
    device = stack.boot(script)

    # -- account (SimStack signed up the demo account) and discovery: nothing until claimed
    me = stack.api.get("/v1/devices", headers=stack.headers)
    assert me.status_code == 200 and me.json()["devices"] == []
    device.advance(4.0)
    assert stack.devices() == [], "a device reporting in but unclaimed appears on no account"
    catalogue = stack.api.get("/v1/presets", headers=stack.headers)
    assert catalogue.status_code == 200 and catalogue.json()
    trace.step("signed in; device powered on, not yet on the account")

    # -- claim (a mistyped code first)
    wrong = stack.api.post("/v1/devices/claim", json={"claim_code": "ZZZZ-ZZZZ-ZZZZ"}, headers=stack.headers)
    assert wrong.status_code == 404, wrong.text
    stack.claim()
    row = stack.row()
    assert row["health"]["device_id"].startswith("SD-SIM-") and row["health"]["simulated"] is True
    assert row["monitoring"]["simulated"] is True
    trace.step("claimed", hardware_id=row["hardware_id"])

    # -- camera setup: the owner draws an area; the DEVICE applies it
    zones = [{"name": "dash", "kind": "restricted", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]}]
    saved = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": zones}, headers=stack.headers)
    assert saved.status_code == 200 and saved.json()["rejected"] == 0
    device.advance(C)
    assert [z.name for z in device.runtime._effective_zones(device.runtime.module_name)] == ["dash"]
    trace.step("area drawn and applied on the device")

    # -- mode activation, confirmed by the device's report (not by the 200)
    chosen = stack.set_preset("driver_safety:strict")
    assert chosen.status_code == 200, chosen.text
    assert stack.row()["active_preset"] != "driver_safety:strict", "a mode is active only once the device reports it"
    device.advance(4.0)
    row = stack.row()
    assert row["active_preset"] == "driver_safety:strict" and row["desired_preset"] is None
    assert device.runtime.active_preset == "driver_safety:strict"

    # -- monitoring confirmation
    status = stack.status()
    assert (status["monitoring_state"], status["protection"], status["monitoring_active"]) == \
        ("ready", "protecting", True)
    assert state_model.derive(stack.row(), stack.clock.wall()).protecting is True
    trace.step("mode active, monitoring confirmed", state=status["monitoring_state"])

    # -- an alert, with what the in-cabin output did (eyes close at C+20 for 5 s)
    device.advance(20.0)
    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1
    alert = drowsy[0]
    assert alert["severity"] == "loud" and alert["simulated"] is True
    assert alert["alert"]["decision"] == "alerted" and alert["alert"]["output"] == "simulated"
    assert alert["alert"]["outputs"][0]["name"] == "sim_buzzer"
    assert any(p[0] == "drowsiness" for p in device.buzzer.played)
    trace.step("drowsiness alert", output=alert["alert"]["output"], event_key=alert["event_key"])

    # -- history and event detail / explain
    page = stack.api.get(f"/v1/devices/{stack.device_id}/events", params={"limit": 2}, headers=stack.headers)
    assert page.status_code == 200 and len(page.json()["events"]) <= 2
    newest_first = page.json()["events"]
    older = stack.api.get(f"/v1/devices/{stack.device_id}/events",
                          params={"limit": 50, "before_id": newest_first[-1]["id"]}, headers=stack.headers)
    ids = [e["id"] for e in newest_first] + [e["id"] for e in older.json()["events"]]
    assert ids == sorted(ids, reverse=True) and len(ids) == len(set(ids)) == len(stack.events())
    explained = stack.api.get(f"/v1/devices/{stack.device_id}/events/{alert['id']}/explain", headers=stack.headers)
    assert explained.status_code == 200, explained.text
    assert json.dumps(explained.json()), "an explanation came back"
    missing = stack.api.get(f"/v1/devices/{stack.device_id}/events/999999/explain", headers=stack.headers)
    assert missing.status_code == 404
    trace.step("history paged and an alert explained")

    # -- Ask Jupiter: review a draft, activate it; "active" only once the DEVICE runs it
    preview = stack.ai_configure("tell me when a phone is visible")
    assert preview.status_code == 200, preview.text
    draft = preview.json()
    assert draft["draft_version"] and not draft.get("unsupported"), draft
    activated = stack.set_preset("", draft_version=draft["draft_version"], rules=[PHONE_RULE])
    assert activated.status_code == 200 and activated.json()["rules_active"] == 1, activated.text
    assert not stack.status()["rules"]["active"], "activation is not confirmed before the device runs it"
    device.advance(6.0)
    status = stack.status()
    assert status["rules"]["active"] and status["rules"]["rejected"] is None, status["rules"]
    assert status["mode"] == "standard_plus_custom" and device.runtime.module_name == "driver_attention"
    device.advance(35.0)  # the phone is on the dash from C+60
    assert [e["type"] for e in stack.events() if e["source"] == "custom_rule"] == ["cell phone"]
    trace.step("Jupiter rule reviewed, activated, confirmed by the device, fired")

    # -- stop (confirmed by the device's echo), then resume
    stop = stack.protection("stop", confirm=True)
    assert stop.status_code == 200
    assert (stack.status().get("protection_command") or {}).get("outcome") != "confirmed"
    device.advance(5.0)
    status = stack.status()
    assert status["monitoring_state"] == "stopped" and status["protection"] == "stopped"
    assert status["protection_command"]["outcome"] == "confirmed"
    assert status["protection_command"]["id"] == stop.json()["command_id"]
    assert not set(status["capabilities"]["available"]) & {"face", "objects", "attention"}
    trace.step("stopped (device echoed the command)")
    resume = stack.protection("resume")
    device.advance(8.0)
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting"
    assert status["protection_command"]["id"] == resume.json()["command_id"]
    trace.step("resumed")

    # -- disconnect, then reconnect: never "all clear" from old evidence
    device.state.fault_overrides["network_down"] = True
    device.advance(35.0)
    row = stack.row()
    assert row["online"] is False and row["status"]["monitoring_state"] == "unknown" and row["health"] is None
    assert state_model.derive(row, stack.clock.wall()).state == state_model.OFFLINE
    trace.step("disconnected: offline, nothing claimed")
    device.state.fault_overrides.pop("network_down")
    device.advance(4.0)
    assert stack.row()["online"] is True and stack.status()["monitoring_state"] == "ready"
    trace.step("reconnected")

    # -- reboot: same unit, same pairing, same mode, same rule
    hardware_id = stack.row()["health"]["device_id"]
    device.advance(60.0)  # past the resume's settle window (see the M8.4 recovery suite's xfail)
    origin = device.scenario_clock.origin
    device = stack.restart(script)
    device.scenario_clock.origin = origin
    device.advance(6.0)
    row = stack.row()
    assert row["health"]["device_id"] == hardware_id and row["active_preset"] == "driver_safety:strict"
    assert row["status"]["rules"]["active"] and row["status"]["monitoring_state"] == "ready"
    trace.step("rebooted: same identity, mode and rule")

    # -- sign out: this token is dead, the device keeps protecting
    monitor.assert_clean()  # every app-side check so far (they need the token)
    monitor.check_backend = False
    out = stack.api.post("/v1/logout", headers=stack.headers)
    assert out.status_code == 200
    assert stack.api.get("/v1/devices", headers=stack.headers).status_code == 401
    device.advance(4.0)
    assert device.runtime.protection_state()["state"] == "protecting"
    trace.step("signed out: token refused; device still protecting", app_view=False)
    monitor.assert_clean(events=False)
    trace.write()


# ---------------------------------------------------------------------- ownership
def test_another_account_can_neither_see_claim_nor_command_this_device(stack):
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    other = _signup(stack, "someone-else@example.com")
    did = stack.device_id
    assert stack.api.get("/v1/devices", headers=other).json()["devices"] == []
    attempts = {
        "claim": stack.api.post("/v1/devices/claim", json={"claim_code": stack.factory["claim_code"]}, headers=other),
        "events": stack.api.get(f"/v1/devices/{did}/events", headers=other),
        "stop": stack.api.post(f"/v1/devices/{did}/protection", json={"action": "stop", "confirm": True}, headers=other),
        "preset": stack.api.post(f"/v1/devices/{did}/preset", json={"preset": "driver_safety:strict"}, headers=other),
        "zones": stack.api.post(f"/v1/devices/{did}/zones", json={"zones": []}, headers=other),
        "rename": stack.api.post(f"/v1/devices/{did}/name", json={"name": "mine now"}, headers=other),
        "ai": stack.api.post("/v1/ai/configure", json={"device_id": did, "prompt": "watch the phone"}, headers=other),
    }
    for what, response in attempts.items():
        assert response.status_code in (403, 404, 409), (what, response.status_code, response.text)
    unauthenticated = stack.api.get(f"/v1/devices/{did}/events")
    assert unauthenticated.status_code == 401
    device.advance(6.0)
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting", "nothing changed"
    assert stack.row()["name"] != "mine now"
    monitor.assert_clean()


# ---------------------------------------------------------------------- invalid and stale state
def test_invalid_requests_and_a_frozen_device_are_told_apart_from_a_healthy_one(stack):
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    bad = {
        "unconfirmed_stop": stack.protection("stop"),
        "unknown_action": stack.api.post(f"/v1/devices/{stack.device_id}/protection",
                                         json={"action": "explode", "confirm": True}, headers=stack.headers),
        "forbidden_preset": stack.set_preset("security_monitoring"),
        "stale_draft": stack.set_preset("", draft_version="0" * 16, rules=[PHONE_RULE]),
        "no_such_device": stack.api.get("/v1/devices/987654/events", headers=stack.headers),
    }
    assert bad["unconfirmed_stop"].status_code == 400 and bad["unconfirmed_stop"].json()["code"] == "confirmation_required"
    assert bad["unknown_action"].status_code == 422
    assert bad["forbidden_preset"].status_code == 409 and bad["forbidden_preset"].json()["code"] == "role_forbids_preset"
    assert bad["stale_draft"].status_code == 409 and bad["stale_draft"].json()["code"] == "draft_superseded"
    assert bad["no_such_device"].status_code == 404
    device.advance(4.0)
    assert stack.status()["monitoring_state"] == "ready"

    # A device whose health thread froze while its link still polls: stale, never "ready".
    runtime = device.runtime
    frozen = runtime._collect_health("uplink")
    runtime._collect_health = lambda caller="loop": frozen
    device.advance(40.0)
    row = stack.row()
    assert row["online"] is True and row["status"]["stale"] is True
    assert state_model.derive(row, stack.clock.wall()).state == state_model.STALE
    assert row["status"]["capabilities"] is None and row["health"] is None
    del runtime._collect_health
    device.advance(4.0)
    assert stack.status()["monitoring_state"] == "ready"
    monitor.assert_clean()


# ---------------------------------------------------------------------- reconnect keeps the backlog
def test_an_alert_raised_while_disconnected_reaches_history_once_with_its_true_time(stack):
    monitor = InvariantMonitor(stack)
    down = Faults(network_down=True)
    device = stack.boot(_scenario("offline_alert", (C + 2, ATTENTIVE, down), (C + 5, ASLEEP, down),
                                  (C + 10, ATTENTIVE, down), (C + 50, ATTENTIVE)))
    stack.claim()
    device.advance(C + 40)
    assert state_model.derive(stack.row(), stack.clock.wall()).state == state_model.OFFLINE
    assert [e["type"] for e in stack.events()].count("drowsiness") == 0
    device.advance(15.0)
    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 1
    assert drowsy[0]["received_at"] - drowsy[0]["occurred_at"] > 35.0, "placed when it happened, not when it arrived"
    monitor.assert_clean()
