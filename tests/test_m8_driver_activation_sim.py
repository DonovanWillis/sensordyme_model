"""Mission 8, Checkpoint C: Distracted Driving (the `driver_safety` preset) end to end.

One journey, asserted step by step through the real code on the virtual device
(sim/: the real DeviceRuntime, driver_attention module, CloudUplink and
cloud/app.py on one SimClock; only camera, FaceMesh, Hands and the detector are
scripted):

  1. app      POST /v1/devices/{id}/preset           (bearer auth, ownership)
  2. backend  stores desired_preset; /v1/ingest answers with `set_preset`
  3. uplink   hands the command to the runtime
  4. device   DeviceRuntime._switch_preset_now builds driver_attention
  5. device   reports active_preset, monitoring state and status.perception
  6. backend  converges (desired_preset cleared) and serves it to the app
  7. events   driver alerts reach GET /v1/devices/{id}/events, one per episode

Each step's actual API/state is captured into a trace (JSON). The trace is
written to tmp_path, and also to $SD_M8_EVIDENCE_DIR when that is set.

SIMULATED evidence: plumbing and state transitions, never accuracy on real
footage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sim.harness import SimStack
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, LOOKING_AWAY, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
GENERAL = {"role": "general", "module": "shared_perception"}


# ---------------------------------------------------------------------- harness
class _Recorder:
    """A transport that records every uplink exchange: what the device reported
    (active_preset) and which command the backend answered with."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.exchanges: list[dict] = []

    def post(self, url, **kwargs):
        response = self._inner.post(url, **kwargs)
        body = kwargs.get("json") or {}
        try:
            answer = response.json()
        except ValueError:
            answer = {}
        if url.endswith("/v1/ingest"):
            self.exchanges.append({
                "reported_active_preset": body.get("active_preset"),
                "events": [e.get("type") for e in body.get("events") or []],
                "status": response.status_code,
                "command": (answer or {}).get("command"),
            })
        return response

    def commands(self, key: str) -> list:
        return [x["command"][key] for x in self.exchanges if x["command"] and key in x["command"]]


class _Trace:
    def __init__(self, name: str, tmp_path: Path) -> None:
        self.name, self.steps, self._tmp = name, [], tmp_path

    def step(self, label: str, **facts) -> None:
        self.steps.append({"step": label, **facts})

    def dump(self) -> Path:
        text = json.dumps({"journey": self.name, "simulated": True, "steps": self.steps}, indent=1, default=str)
        path = self._tmp / f"{self.name}.json"
        path.write_text(text)
        out = os.environ.get("SD_M8_EVIDENCE_DIR")
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / f"{self.name}.json").write_text(text)
        return path


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0, ATTENTIVE), *(Segment(float(t), scene) for t, scene in segments)))


def _snapshot(stack) -> dict:
    row = stack.row()
    status = row["status"]
    perception = status.get("perception") or {}
    return {
        "active_preset": row["active_preset"], "desired_preset": row["desired_preset"],
        "monitoring_state": status["monitoring_state"], "monitoring_active": status["monitoring_active"],
        "protection": status["protection"], "reasons": status["reasons"],
        "capabilities": status["capabilities"],
        "nodes": [{k: n.get(k) for k in ("id", "kind", "state", "validation", "experimental",
                                         "missing_capabilities")} for n in perception.get("nodes", [])],
        "components": {k: v["state"] for k, v in (perception.get("components") or {}).items()},
        "policies": [{k: p.get(k) for k in ("id", "operational", "unavailable_events")}
                     for p in status.get("policies") or []],
        "preset_rejection": status.get("preset_rejection"), "last_rejection": status.get("last_rejection"),
    }


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path / "stack") as s:
        yield s


def _boot(stack, scenario, **kwargs):
    recorder = _Recorder(stack.transport)
    stack.claim()
    device = stack.boot(scenario, transport=recorder, **kwargs)
    return device, recorder


# ---------------------------------------------------------------------- journeys
def test_distracted_driving_activation_reaches_the_device_and_alerts_reach_the_app(stack, tmp_path):
    """A general-purpose unit running Occupancy is switched to Distracted Driving
    from the app: the full module swap, every hop asserted."""
    trace = _Trace("m8_distracted_driving_activation", tmp_path)
    # The driver looks away 5 s at scenario t=40 (after activation + calibration).
    device, recorder = _boot(stack, _scenario("activation", (40.0, LOOKING_AWAY), (45.0, ATTENTIVE)),
                             config_overrides=GENERAL)
    device.advance(4.0)
    before = _snapshot(stack)
    trace.step("0 before", **before)
    assert before["active_preset"] == "occupancy" and device.runtime.module_name == "shared_perception"
    assert [n["id"] for n in before["nodes"] if n["kind"] == "preset"] == ["occupancy"]

    # 1. app -> backend, with auth: no token, someone else's token, then the owner.
    api, device_id = stack.api, stack.device_id
    anonymous = api.post(f"/v1/devices/{device_id}/preset", json={"preset": "driver_safety"})
    stranger = stack.backend.signup("stranger@example.com", "stranger-password-1")
    foreign = api.post(f"/v1/devices/{device_id}/preset", json={"preset": "driver_safety"}, headers=stranger)
    response = stack.set_preset("driver_safety")
    trace.step("1 app POST /preset", anonymous=anonymous.status_code, foreign=foreign.status_code,
               owner=response.status_code, body=response.json())
    assert anonymous.status_code == 401
    assert foreign.status_code == 404 and foreign.json()["code"] == "device_not_found"
    assert response.status_code == 200 and response.json()["desired_preset"] == "driver_safety"

    # 2. backend stores the desire; the device has not acted yet.
    pending = stack.row()
    trace.step("2 backend desired_preset", desired_preset=pending["desired_preset"],
               active_preset=pending["active_preset"])
    assert pending["desired_preset"] == "driver_safety" and pending["active_preset"] == "occupancy"

    # 3-4. next uplink cycle: ingest answers set_preset; the runtime switches.
    device.advance(2.0)
    delivered = recorder.commands("set_preset")
    trace.step("3 ingest command", set_preset=delivered)
    assert delivered and delivered[0] == "driver_safety"
    runtime = device.runtime
    trace.step("4 runtime switched", active_preset=runtime.active_preset, module=runtime.module_name,
               alert_types=sorted(runtime.alert_event_types or []))
    assert runtime.active_preset == "driver_safety" and runtime.module_name == "driver_attention"
    assert {"drowsiness", "distraction", "phone_use", "attention_lost"} <= set(runtime.alert_event_types or [])

    # 5-6. the device reports it; the backend converges and serves it.
    device.advance(4.0)
    after = _snapshot(stack)
    reported = [x["reported_active_preset"] for x in recorder.exchanges]
    trace.step("5-6 device-reported state", reported_active_preset=reported[-3:], **after)
    assert reported[-1] == "driver_safety"
    assert after["active_preset"] == "driver_safety" and after["desired_preset"] is None
    assert after["preset_rejection"] is None and after["last_rejection"] is None
    preset_nodes = [n for n in after["nodes"] if n["kind"] == "preset"]
    assert [n["id"] for n in preset_nodes] == ["driver_safety"]
    assert preset_nodes[0]["validation"] == "synthetic_tests" and preset_nodes[0]["experimental"] is False
    assert after["components"]["face_landmarks"] == "ok" and after["components"]["object_detector"] == "ok"
    assert after["components"]["pose"] in ("idle", "unavailable"), "driver safety runs no pose model"
    assert any(p["id"] == "preset:driver_safety" for p in after["policies"])
    assert after["monitoring_state"] in ("starting", "ready"), after

    # The driver's forward gaze is calibrated before gaze alerts can be judged.
    device.advance(C)
    calibrated = _snapshot(stack)
    trace.step("5b calibrated", **calibrated)
    assert calibrated["monitoring_state"] == "ready" and calibrated["protection"] == "protecting"
    assert {"face", "objects", "audible_alert"} <= set(calibrated["capabilities"]["available"])

    # 7. a sustained look away: one distraction episode, in the app's history.
    device.advance(40.0 + 8.0 - device.state.clock.now())
    events = stack.events()
    trace.step("7 events", events=[{k: e[k] for k in ("type", "severity", "source", "duration")} for e in events],
               buzzer=list(device.buzzer.played))
    distraction = [e for e in events if e["type"] == "distraction"]
    assert len(distraction) == 1, [e["type"] for e in events]
    assert distraction[0]["severity"] == "loud" and distraction[0]["source"] == "module"
    assert distraction[0]["model"]["module"] == "driver_attention"
    assert distraction[0]["model"]["detector"]["backend"] == "simulated"
    assert any(p[0] == "distraction" for p in device.buzzer.played), "the cabin alert sounded (simulated)"
    trace.dump()


def test_driver_card_settings_apply_on_a_driver_unit_and_a_loosened_floor_is_refused(stack, tmp_path):
    """On a driver_safety unit the card activates the same preset with owner
    settings; the device applies them (2.0 s look-away) and the role floor
    refuses a looser one at the backend, before anything is sent."""
    trace = _Trace("m8_distracted_driving_settings", tmp_path)
    device, recorder = _boot(stack, _scenario("settings", (C + 10, LOOKING_AWAY), (C + 13, ATTENTIVE)))
    device.advance(C)
    looser = stack.set_preset("driver_safety", settings={"distraction_seconds": 4.0})
    trace.step("refused looser", status=looser.status_code, body=looser.json())
    assert looser.status_code == 409 and looser.json()["code"] == "role_forbids_preset"

    response = stack.set_preset("driver_safety", settings={"distraction_seconds": 2.0})
    assert response.status_code == 200
    wanted = response.json()["desired_preset"]
    device.advance(4.0)
    after = _snapshot(stack)
    trace.step("applied", wanted=wanted, commands=recorder.commands("set_preset"), **after)
    assert recorder.commands("set_preset") == [wanted]
    assert after["active_preset"] == wanted and after["desired_preset"] is None
    assert after["monitoring_state"] == "ready"

    device.advance(14.0)  # 3 s look away at C+10: fires at the owner's 2.0 s, not the default 3.0 s
    distraction = [e for e in stack.events() if e["type"] == "distraction"]
    trace.step("event", distraction=distraction)
    assert len(distraction) == 1 and distraction[0]["duration"] == pytest.approx(2.0, abs=0.15)
    trace.dump()


def test_an_alert_episode_is_one_stored_event_repeating_locally_then_a_new_episode_is_new(stack, tmp_path):
    """Eyes closed 8 s: one drowsiness row, the critical pattern repeating while
    it lasts (bounded), ended when the eyes open; a later closure is a second
    episode and a second row."""
    trace = _Trace("m8_driver_alert_episodes", tmp_path)
    script = _scenario("episodes", (C + 3, ASLEEP), (C + 11, ATTENTIVE), (C + 25, ASLEEP), (C + 29, ATTENTIVE))
    device, _ = _boot(stack, script)
    device.advance(C + 11.5)
    first = [e for e in stack.events() if e["type"] == "drowsiness"]
    played_first = [p for p in device.buzzer.played if p[0] == "drowsiness"]
    open_keys = sorted(device.runtime.alert_manager.open_keys("module"))
    trace.step("first episode", rows=len(first), played=played_first, open_after_eyes_open=open_keys)
    assert len(first) == 1 and first[0]["severity"] == "loud"
    assert played_first and all(p[1] == "critical" for p in played_first)
    assert 2 <= len(played_first) <= 10, "repeats while the eyes stay closed, bounded"

    device.advance(2.0)
    assert "drowsiness" not in device.runtime.alert_manager.open_keys("module"), "ended with the condition"
    played_between = len([p for p in device.buzzer.played if p[0] == "drowsiness"])
    device.advance(10.0)
    assert len([p for p in device.buzzer.played if p[0] == "drowsiness"]) == played_between, "silent when open"

    device.advance(6.0)  # second closure at C+25
    rows = [e for e in stack.events() if e["type"] == "drowsiness"]
    trace.step("second episode", rows=len(rows), types=stack.event_types())
    assert len(rows) == 2, "a new closure is a new episode, never merged into the old one"
    assert stack.status()["monitoring_state"] == "ready"
    trace.dump()


def test_event_history_is_ordered_complete_limited_and_survives_a_restart(stack, tmp_path):
    trace = _Trace("m8_driver_event_history", tmp_path)
    script = _scenario("history", (C + 3, ASLEEP), (C + 8, ATTENTIVE), (C + 12, LOOKING_AWAY), (C + 18, ATTENTIVE))
    device, _ = _boot(stack, script)
    device.advance(C + 20)
    events = stack.events()  # oldest first
    types = [e["type"] for e in events]
    trace.step("history", events=[{k: e[k] for k in ("id", "type", "severity", "source", "duration")}
                                  for e in events])
    assert types.index("drowsiness") < types.index("distraction"), "in the order they happened"
    ids = [e["id"] for e in events]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    for event in events:
        assert set(event) >= {"id", "timestamp", "type", "confidence", "signals", "severity", "source",
                              "duration", "model"}
    for event in (e for e in events if e["source"] == "module"):
        assert event["model"]["module"] == "driver_attention"
        assert event["model"]["detector"]["backend"] == "simulated", "marked simulated, never a real model"
    newest = stack.api.get(f"/v1/devices/{stack.device_id}/events", params={"limit": 1},
                           headers=stack.headers).json()["events"]
    assert [e["id"] for e in newest] == [ids[-1]], "limit returns the newest first"
    assert stack.api.get(f"/v1/devices/{stack.device_id}/events").status_code == 401

    device = stack.restart(script)
    device.advance(4.0)
    again = stack.events()
    trace.step("after restart", ids=[e["id"] for e in again], active_preset=stack.row()["active_preset"])
    assert [e["id"] for e in again][:len(ids)] == ids, "history kept, nothing duplicated by the restart"
    assert stack.row()["active_preset"] == "driver_safety"
    trace.dump()
