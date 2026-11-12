"""M8 end-to-end on the virtual device: the three scenarios the nodes stream owns.

  1. a supported custom prompt, activated: words -> typed policy -> draft ->
     Activate -> device -> rule on the shared tracker -> alert -> restart -> back
  2. simultaneous compatible nodes: a preset, its node, an owner's companion and
     custom rules on ONE detector run per frame, through every mode change
  3. unsupported modes refused, with a code, at the backend and on the device,
     and the device's refusal reaching the app's status

Everything goes through the HTTP API the app uses; the device is reached only
through its uplink. Each test captures the API responses and state transitions
it walks through (set SD_M8_EVIDENCE_DIR to keep them). Simulated evidence.
"""

from __future__ import annotations

import pytest
from test_m8_nodes_modes import (
    GENERAL,
    boot,
    counters,
    draw,
    enable_node,
    people,
    perception,
    record,
    restart,
    room,
    walk,
)
from test_sim_reliability import _CommandTap

from sim.harness import SimStack
from sim.scenario import ATTENTIVE, CALIBRATION_S, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

DOOR = {"name": "front door", "polygon": [[0.0, 0.5], [0.4, 0.5], [0.4, 1.0], [0.0, 1.0]], "kind": "restricted"}
PROMPT = "alert me if someone stays in the front door area for more than 5 seconds"


def _status(stack) -> dict:
    status = stack.status()
    return {"mode": status["mode"], "monitoring_state": status["monitoring_state"],
            "rules": status["rules"], "last_rejection": status["last_rejection"],
            "preset_rejection": status["preset_rejection"],
            "nodes": [(n["id"], n["state"]) for n in (status["perception"] or {}).get("nodes", [])]}


def _activate_prompt(stack, prompt: str = PROMPT) -> tuple[dict, dict]:
    preview = stack.ai_configure(prompt)
    assert preview.status_code == 200, preview.text
    body = preview.json()
    activated = stack.set_preset("", draft_version=body["draft_version"], rules=body["rules"])
    assert activated.status_code == 200, activated.text
    return body, activated.json()


# ====================================================================== 1. custom prompt
def test_e2e_a_supported_custom_prompt_runs_on_the_shared_tracker_and_survives_a_restart(stack):
    transcript: dict = {}
    script = room("front_door", (20, people(0.2)), *walk(40, 0.2, 0.8), (60, people(0.8)))
    device, tap = boot(stack, script)
    draw(stack, device, DOOR)
    transcript["before"] = _status(stack)
    assert transcript["before"]["mode"] == "standard"
    assert ("people_counter", "active") in transcript["before"]["nodes"]

    preview, activated = _activate_prompt(stack)
    (policy,) = preview["policies"]
    assert preview["source"] == "policy_parser" and preview["unsupported"] == []
    assert policy["condition"] == "lingering" and policy["temporal"] == {"dwell_s": 5.0}
    assert policy["zone"] == {"name": "front door", "drawn": True} and policy["status"] == "ready"
    assert policy["required_capabilities"] == ["camera", "inference", "objects"]
    assert activated["rules_active"] == 1
    transcript["preview"] = {"source": preview["source"], "policies": preview["policies"]}
    transcript["activate"] = activated

    device.advance(4.0)
    runtime = device.runtime
    transcript["activated"] = _status(stack)
    assert transcript["activated"]["mode"] == "ai_custom" and transcript["activated"]["monitoring_state"] == "ready"
    assert transcript["activated"]["rules"]["active"] and transcript["activated"]["rules"]["rejected"] is None
    nodes = dict(transcript["activated"]["nodes"])
    assert nodes == {"occupancy": "inactive", "rules": "active", "people_counter": "inactive"}, \
        "custom rules replace the preset AND its node; the app is told so"
    assert {d.consumer for d in runtime.perception.demands()} == {"rules"}

    device.advance(30.0)  # the person has been at the door since t=20
    alerts = [e for e in stack.events() if e["source"] == "custom_rule"]
    assert [e["type"] for e in alerts] == ["person"], "one visit, one alert"
    (dwell,) = runtime.rule_engine._dwell.values()
    assert all(visit.track_id is not None for visit in dwell._tracks), \
        "the rule's dwell clock follows the shared tracker's id"
    assert not [e for e in stack.events() if e["source"] == "node:people_counter"], "the replaced node is silent"
    transcript["alerts"] = alerts

    device = restart(stack, script, tap)
    device.advance(4.0)
    transcript["after_restart"] = _status(stack)
    assert transcript["after_restart"]["mode"] == "ai_custom"
    assert [r.dwell_s for r in device.runtime.rule_engine.rules] == [5.0], "the policy survived the restart"

    assert stack.set_preset("occupancy").status_code == 200
    device.advance(4.0)
    transcript["back_to_preset"] = _status(stack)
    assert transcript["back_to_preset"]["mode"] == "standard"
    assert ("people_counter", "active") in transcript["back_to_preset"]["nodes"]
    assert "people_counter" in {d.consumer for d in device.runtime.perception.demands()}, "its demand is back"
    record("e2e_custom_prompt_activation", transcript)


# ====================================================================== 2. concurrency
def test_e2e_simultaneous_compatible_nodes_share_one_detector_run_per_frame(stack):
    transcript: dict = {"phases": []}
    script = room("site", (5, people(0.2, 0.8)), *walk(40, 0.8, 0.3), (60, people(0.2)))
    device, _tap = boot(stack, script)
    assert stack.set_preset("workplace_safety").status_code == 200
    draw(stack, device, DOOR)
    response = enable_node(stack, "people_counter", settings={"settle_s": 1.0})
    assert response.status_code == 200 and response.json()["pending"] is True
    transcript["enable_node"] = response.json()
    device.advance(4.0)
    listed = stack.api.get(f"/v1/devices/{stack.device_id}/nodes", headers=stack.headers).json()
    transcript["nodes_api"] = listed

    def phase(name: str, seconds: float) -> dict:
        calls0, frames0 = counters(device)
        updates0 = perception(device)["tracking"]["updates"]
        device.advance(seconds)
        calls, frames = device.sim.detector.calls - calls0, device.runtime.perception.frames - frames0
        report = perception(device)
        result = {"phase": name, "frames": frames, "detector_runs": calls,
                  "tracker_updates": report["tracking"]["updates"] - updates0,
                  "consumers": report["components"]["object_detector"]["consumers"],
                  "nodes": [(n["id"], n["state"]) for n in report["nodes"]], "status": _status(stack)}
        transcript["phases"].append(result)
        assert 0 < calls <= frames, result
        assert result["tracker_updates"] == calls, "one association per detector run, whoever consumes it"
        return result

    first = phase("preset + preset node + owner companion", 30.0)
    assert set(first["consumers"]) >= {"shared_perception", "zone_watch", "people_counter"}
    assert dict(first["nodes"]) == {"workplace_safety": "active", "zone_watch": "active", "people_counter": "active"}
    backend = stack.status()["perception"]  # what the app reads
    counter = next(n for n in backend["nodes"] if n["id"] == "people_counter")
    assert counter["enabled_by"] == "owner" and counter["metrics"]["count"] == 2 and counter["metrics"]["tracked"]
    assert backend["tracking"]["updates"] > 0 and backend["tracking"]["max_tracks"] == 64

    _activate_prompt(stack)
    device.advance(3.0)
    second = phase("custom rules + owner companion (preset and its node replaced)", 30.0)
    assert set(second["consumers"]) >= {"rules", "people_counter"} and "zone_watch" not in second["consumers"]
    assert dict(second["nodes"]) == {"workplace_safety": "inactive", "rules": "active",
                                     "people_counter": "active", "zone_watch": "inactive"}
    sources = {e["source"] for e in stack.events()}
    assert {"node:zone_watch", "node:people_counter", "custom_rule"} <= sources
    record("e2e_simultaneous_nodes", transcript)


def test_e2e_driver_safety_keeps_its_module_while_companions_and_rules_join(tmp_path):
    """The driver_safety role: everything here is additive (role floor)."""
    script = Scenario("cabin", "cabin", (Segment(0.0, ATTENTIVE),))
    with SimStack(tmp_path) as stack:
        tap = _CommandTap(stack.transport)
        stack.claim()
        device = stack.boot(script, transport=tap)
        device.advance(CALIBRATION_S)
        for node_id in ("people_counter", "zone_watch"):
            assert enable_node(stack, node_id).status_code == 200
        device.advance(4.0)
        calls0, frames0 = counters(device)
        device.advance(20.0)
        calls, frames = device.sim.detector.calls - calls0, device.runtime.perception.frames - frames0
        assert calls == pytest.approx(frames / 3, abs=2), "the driver stride answers both companions"
        nodes = {n["id"]: n["state"] for n in stack.status()["perception"]["nodes"]}
        assert nodes == {"driver_safety": "active", "people_counter": "active", "zone_watch": "active"}
        assert device.runtime.module_name == "driver_attention"


# ====================================================================== 3. refusals
def test_e2e_unsupported_modes_are_refused_with_a_code_and_the_device_refusal_reaches_the_app(tmp_path):
    transcript: dict = {}
    # --- a general device: backend refusals, then device refusals of commands a stale backend sends
    with SimStack(tmp_path / "general") as stack:
        device, tap = boot(stack, room("room", (2, people(0.3))))
        device.advance(3.0)
        api = {
            "unknown_mode": stack.set_preset("pool_safety"),
            "bad_setting": stack.set_preset("security_monitoring@linger_s=99999"),
            "unknown_node": enable_node(stack, "forklift_detector"),
        }
        transcript["backend"] = {k: (r.status_code, r.json()) for k, r in api.items()}
        assert api["unknown_mode"].status_code == 400 and api["unknown_mode"].json()["code"] == "unknown_preset"
        assert api["bad_setting"].status_code == 400 and api["bad_setting"].json()["code"] == "invalid_setting"
        assert api["unknown_node"].status_code == 400 and api["unknown_node"].json()["code"] == "unknown_node"
        assert device.runtime.active_preset == "occupancy", "nothing reached the device"

        tap.inject.append({"set_preset": "pool_safety"})
        device.advance(4.0)
        refusal = stack.status()["last_rejection"]
        transcript["device_unknown_preset"] = refusal
        assert refusal["kind"] == "preset" and refusal["subject"] == "pool_safety"
        assert refusal["code"] == "unknown_preset" and refusal["reasons"]
        assert device.runtime.active_preset == "occupancy", "the device kept its mode"

        tap.inject.append({"set_nodes": {"forklift_detector": {"enabled": True}}})
        device.advance(4.0)
        refusal = stack.status()["last_rejection"]
        transcript["device_unknown_node"] = refusal
        assert refusal["kind"] == "nodes" and refusal["code"] == "unknown_node"
        assert "forklift_detector" in refusal["reasons"][0]

    # --- a driver_safety device: the role refuses room modes at both ends
    with SimStack(tmp_path / "driver") as stack:
        tap = _CommandTap(stack.transport)
        stack.claim()
        device = stack.boot(Scenario("cabin", "cabin", (Segment(0.0, ATTENTIVE),)), transport=tap)
        device.advance(6.0)
        response = stack.set_preset("occupancy")
        transcript["driver_backend"] = (response.status_code, response.json())
        assert response.status_code == 409 and response.json()["code"] == "role_forbids_preset"

        tap.inject.append({"set_preset": "baby_monitoring"})  # what a stale backend would still send
        device.advance(4.0)
        refusal = stack.status()["last_rejection"]
        transcript["driver_device_refusal"] = refusal
        assert refusal["kind"] == "preset" and refusal["subject"] == "baby_monitoring"
        assert refusal["code"] == "role_forbids_preset", "a coded refusal, not only a sentence"
        assert device.runtime.module_name == "driver_attention" and device.runtime.active_preset == "driver_safety"
    assert GENERAL["role"] == "general"
    record("e2e_unsupported_mode_rejection", transcript)


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s
