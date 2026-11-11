"""Mission 7 backend (docs/M7_IMPLEMENTATION_MAP.md): device-truth status, refusals the
device makes, companion nodes, availability that says "unknown", node tiers, and Jupiter
checked against the TARGET device -- plus the two guarantees that LLM output is only ever
data and that the per-frame path never reaches a language model."""

# ruff: noqa: F811 - `backend` is a pytest fixture imported from test_app_status_contract;
# each test naming it as a parameter is how pytest injects it, not a redefinition.

from __future__ import annotations

import ast
import builtins
import json
import time
from pathlib import Path

import pytest
from test_app_status_contract import (  # noqa: F401 - `backend` is a fixture (see above)
    DRIVER,
    GENERAL,
    _health,
    _ingest,
    _monitoring_report,
    _protection,
    _pushed,
    _status,
    backend,
)

from core import command_auth as sd_auth
from core import monitoring_state as ms

REPO = Path(__file__).resolve().parent.parent


def _perception(components=None, nodes=None) -> dict:
    """A `monitoring.perception` block as the wire contract defines it."""
    return {
        "schema": "sd-perception/1",
        "simulated": True,
        "components": components if components is not None else {
            "object_detector": {"state": "ok", "model": {"name": "damo_t.hef", "sha256": "e6" * 32},
                                "last_run_age_s": 0.1, "runs": 120, "errors": 0, "consumers": ["driver_safety"]},
            "face_landmarks": {"state": "ok", "runs": 360},
            "hand_landmarks": {"state": "idle"},
            "pose": {"state": "unavailable", "reason": "no pose model on this device"},
        },
        "nodes": nodes if nodes is not None else [
            {"id": "driver_safety", "kind": "preset", "state": "active", "validation": "synthetic_tests",
             "required_capabilities": ["face", "objects"], "missing_capabilities": [], "experimental": False},
        ],
    }


def _report(client, *, role=DRIVER, preset="driver_safety", perception=None, last_rejection=None,
            policies=(), actuators=(), last_command_id=None, with_perception=True):
    report = _monitoring_report(ms.MonitoringState.READY, role=role)
    report["policies"] = list(policies)
    report["actuators"] = list(actuators)
    if with_perception:
        report["perception"] = perception if perception is not None else _perception()
    protection = _protection(role=role)
    protection["last_command_id"] = last_command_id
    return _ingest(client, health=_health(report, protection),
                   monitoring=_pushed(role, last_rejection=last_rejection), preset=preset)


def _make_stale(app_module):
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET last_seen = ?", (time.time() - 3600,))


# ---------------------------------------------------------------------- status forwarding
def test_status_forwards_perception_policies_actuators_and_the_last_command_id(backend):
    client, headers, _ = backend
    perception = _perception()
    perception["components"]["face_landmarks"]["state"] = "smiling"  # not in the vocabulary
    _report(client, perception=perception, last_command_id="cmd-42",
            policies=[{"id": "driver_safety", "kind": "preset", "name": "Driver Safety", "can_evaluate": True,
                       "operational": True, "unavailable_events": [], "reasons": []}],
            actuators=[{"name": "buzzer", "kind": "gpio", "audible": True, "ready": True, "detail": "ok",
                        "commands_ok": 3, "commands_failed": 0, "physically_verified": False}])
    status = _status(client, headers)
    comps = status["perception"]["components"]
    assert comps["object_detector"]["state"] == "ok"
    assert comps["object_detector"]["model"]["name"] == "damo_t.hef"
    assert comps["object_detector"]["consumers"] == ["driver_safety"]
    assert comps["face_landmarks"]["state"] == "unknown", "an unknown state is never read as ok"
    assert comps["pose"] == {**comps["pose"], "state": "unavailable", "reason": "no pose model on this device"}
    assert status["perception"]["nodes"][0]["id"] == "driver_safety"
    assert status["perception"]["simulated"] is True
    assert status["last_command_id"] == "cmd-42"
    assert status["policies"][0]["can_evaluate"] is True
    assert status["actuators"][0]["physically_verified"] is False
    json.dumps(status)


def test_perception_is_null_when_stale_or_not_reported_never_filled_in(backend):
    client, headers, app_module = backend
    _report(client, with_perception=False)
    assert _status(client, headers)["perception"] is None, "older firmware: not reported"
    _report(client)
    assert _status(client, headers)["perception"] is not None
    _make_stale(app_module)
    status = _status(client, headers)
    assert status["perception"] is None and status["policies"] is None and status["last_command_id"] is None


# ---------------------------------------------------------------------- device-side refusals
def test_a_preset_the_device_refuses_stops_being_resent_and_says_why(backend):
    client, headers, _ = backend
    _report(client, role=GENERAL, preset="security_monitoring")
    assert client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring"}, headers=headers).status_code == 200
    assert _report(client, role=GENERAL, preset="security_monitoring") == {"set_preset": "baby_monitoring"}

    refusal = {"kind": "preset", "subject": "baby_monitoring", "reasons": ["module failed to build: no pose model"],
               "code": "module_unavailable", "at": time.time()}
    command = _report(client, role=GENERAL, preset="security_monitoring", last_rejection=refusal)
    assert not (command or {}).get("set_preset"), "a refused mode is not re-sent"
    status = _status(client, headers)
    assert status["preset_rejection"]["preset"] == "baby_monitoring"
    assert status["preset_rejection"]["code"] == "module_unavailable"
    assert "no pose model" in status["preset_rejection"]["reason"]
    assert status["last_rejection"]["code"] == "module_unavailable"
    devices = client.get("/v1/devices", headers=headers).json()["devices"]
    assert devices[0]["desired_preset"] is None
    # Still refused on the next poll: nothing comes back.
    assert not (_report(client, role=GENERAL, preset="security_monitoring", last_rejection=refusal) or {}).get(
        "set_preset")


def test_an_old_refusal_does_not_cancel_a_new_request_for_the_same_mode(backend):
    client, headers, _ = backend
    refusal = {"kind": "preset", "subject": "baby_monitoring", "reasons": ["no pose model"], "at": 1000.0}
    _report(client, role=GENERAL, preset="security_monitoring", last_rejection=refusal)
    # The owner asks again (say, after installing the model); the device still
    # reports the OLD refusal until it answers this request.
    client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring"}, headers=headers)
    command = _report(client, role=GENERAL, preset="security_monitoring", last_rejection=refusal)
    assert command == {"set_preset": "baby_monitoring"}
    assert _status(client, headers)["preset_rejection"] is None
    # A NEW refusal (different time) is this request's answer.
    newer = {**refusal, "at": 2000.0, "reason": "still no pose model"}
    assert not (_report(client, role=GENERAL, preset="security_monitoring", last_rejection=newer) or {}).get(
        "set_preset")
    assert _status(client, headers)["preset_rejection"]["code"] == "device_refused"


def test_a_refusal_of_another_mode_does_not_cancel_this_one(backend):
    client, headers, _ = backend
    _report(client, role=GENERAL, preset="security_monitoring")
    client.post("/v1/devices/1/preset", json={"preset": "elderly_monitoring"}, headers=headers)
    other = {"kind": "preset", "subject": "baby_monitoring", "reasons": ["no"], "at": 5.0}
    assert _report(client, role=GENERAL, preset="security_monitoring",
                   last_rejection=other) == {"set_preset": "elderly_monitoring"}


# ---------------------------------------------------------------------- companion nodes
def test_a_node_is_delivered_until_the_device_reports_it_then_cleared(backend):
    client, headers, _ = backend
    _report(client)
    resp = client.post("/v1/devices/1/nodes", json={"node": "people_counter", "enabled": True}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ok": True, "node": "people_counter", "enabled": True, "pending": True}
    command = _report(client)
    assert command == {"set_nodes": {"people_counter": {"enabled": True, "settings": {}}}}
    # Re-sent while the device has not reported it.
    assert _report(client)["set_nodes"]["people_counter"]["enabled"] is True

    running = _perception(nodes=[
        {"id": "driver_safety", "kind": "preset", "state": "active"},
        {"id": "people_counter", "kind": "companion", "state": "active", "validation": "synthetic_tests",
         "experimental": True},
    ])
    assert not (_report(client, perception=running) or {}).get("set_nodes"), "acknowledged: cleared"
    listing = client.get("/v1/devices/1/nodes", headers=headers).json()
    counter = next(n for n in listing["nodes"] if n["id"] == "people_counter")
    assert counter["desired"] is None and counter["reported"]["state"] == "active"
    assert counter["validation"] == "synthetic_tests" and counter["experimental"] is True
    # Asking again for what already runs is a no-op: nothing pending, nothing sent.
    again = client.post("/v1/devices/1/nodes", json={"node": "people_counter", "enabled": True}, headers=headers)
    assert again.json()["pending"] is False
    assert not (_report(client, perception=running) or {}).get("set_nodes")


def test_turning_a_node_off_converges_on_inactive_or_absent(backend):
    client, headers, _ = backend
    running = _perception(nodes=[{"id": "people_counter", "kind": "companion", "state": "active"}])
    _report(client, perception=running)
    client.post("/v1/devices/1/nodes", json={"node": "people_counter", "enabled": False}, headers=headers)
    assert _report(client, perception=running) == {"set_nodes": {"people_counter": {"enabled": False, "settings": {}}}}
    stopped = _perception(nodes=[{"id": "people_counter", "kind": "companion", "state": "inactive"}])
    assert not (_report(client, perception=stopped) or {}).get("set_nodes")


def test_node_requests_are_validated(backend):
    client, headers, _ = backend
    _report(client)
    unknown = client.post("/v1/devices/1/nodes", json={"node": "mind_reader", "enabled": True}, headers=headers)
    assert unknown.status_code == 400 and unknown.json()["code"] == "unknown_node"
    bad = client.post("/v1/devices/1/nodes", json={"node": "people_counter", "enabled": True,
                                                    "settings": {"Bad-Key": 1}}, headers=headers)
    assert bad.status_code == 400 and bad.json()["code"] == "invalid_setting"


def test_firmware_without_perception_is_refused_rather_than_asked_for_ever(backend):
    client, headers, _ = backend
    _report(client, with_perception=False)
    resp = client.post("/v1/devices/1/nodes", json={"node": "people_counter", "enabled": True}, headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "firmware_unsupported"


def test_a_node_the_device_refuses_is_recorded_and_not_resent(backend):
    client, headers, _ = backend
    _report(client)
    client.post("/v1/devices/1/nodes", json={"node": "zone_watch", "enabled": True}, headers=headers)
    assert "set_nodes" in _report(client)
    refusal = {"kind": "nodes", "subject": "zone_watch", "reasons": ["objects capability is unavailable"],
               "code": "capability_missing", "at": time.time()}
    assert not (_report(client, last_rejection=refusal) or {}).get("set_nodes")
    status = _status(client, headers)
    assert status["nodes_rejection"]["node"] == "zone_watch"
    assert status["nodes_rejection"]["code"] == "capability_missing"
    # A new request for that node clears the old refusal.
    client.post("/v1/devices/1/nodes", json={"node": "zone_watch", "enabled": True}, headers=headers)
    assert _status(client, headers)["nodes_rejection"] is None


def test_a_set_nodes_command_is_signed_like_every_other():
    key = sd_auth.DeviceKey("dk_" + "x" * 32)
    command = {"set_nodes": {"people_counter": {"enabled": True, "settings": {}}}}
    envelope = sd_auth.sign_command(key, command, device="SD-0A1B2C3D", request_nonce="a" * 32)
    assert envelope["types"] == ["set_nodes"]
    sd_auth.verify_command(sd_auth.Keyring(key), command, envelope, devices=["SD-0A1B2C3D"],
                           request_nonce="a" * 32)
    with pytest.raises(sd_auth.AuthError):
        sd_auth.verify_command(sd_auth.Keyring(key), {"set_nodes": {"people_counter": {"enabled": False}}}, envelope,
                               devices=["SD-0A1B2C3D"], request_nonce="a" * 32)


# ---------------------------------------------------------------------- availability truth
def test_a_capability_the_device_does_not_report_is_unknown_not_available(backend):
    client, headers, _ = backend
    no_face = _perception(components={"object_detector": {"state": "ok"}})
    _report(client, role=GENERAL, preset="security_monitoring", perception=no_face)
    presets = {p["id"]: p for p in client.get("/v1/devices/1/presets", headers=headers).json()["presets"]}
    driver = presets["driver_safety"]
    face = next(c for c in driver["capabilities"] if c["name"] == "face")
    objects = next(c for c in driver["capabilities"] if c["name"] == "objects")
    # _monitoring_report reports face available through the module's capability map;
    # with perception saying nothing about face, the module's report is what we have.
    assert objects["state"] == "available" and objects["source"] == "perception"
    assert face["source"] == "monitoring"
    assert driver["available"] is True


def test_unreported_capabilities_make_readiness_unknown(backend):
    client, headers, _ = backend
    report = _monitoring_report(ms.MonitoringState.READY, role=GENERAL)
    report["capabilities"] = {"camera": {"state": "available", "reason": None}}
    report["perception"] = _perception(components={})
    _ingest(client, health=_health(report, _protection(role=GENERAL)), monitoring=_pushed(GENERAL),
            preset="security_monitoring")
    presets = {p["id"]: p for p in client.get("/v1/devices/1/presets", headers=headers).json()["presets"]}
    driver = presets["driver_safety"]
    assert driver["available"] is True, "not refused: the device may well run it"
    assert driver["ready"] is None, "but not claimed ready either: face was never reported"
    assert next(c for c in driver["capabilities"] if c["name"] == "face")["state"] == "unknown"


def test_a_component_the_device_does_not_have_makes_the_preset_unsupported(backend):
    client, headers, _ = backend
    no_face = _perception(components={"object_detector": {"state": "ok"},
                                      "face_landmarks": {"state": "unavailable", "reason": "no face model"}})
    _report(client, role=GENERAL, preset="security_monitoring", perception=no_face)
    driver = next(p for p in client.get("/v1/devices/1/presets", headers=headers).json()["presets"]
                  if p["id"] == "driver_safety")
    assert driver["available"] is False and driver["code"] == "capability_unsupported"
    assert "no face model" in driver["reason"] and driver["ready"] is False


# ---------------------------------------------------------------------- node tiers
def test_capability_tiers_serves_nodes():
    pytest.importorskip("fastapi")
    from cloud import app as app_module
    from core.model_registry import load_registry

    registry = load_registry()
    claimed = {c.id: c.claimed_tier for c in registry.capabilities}
    some_cap = sorted(claimed)[0]
    document = {"monitoring_nodes": {
        "strong_claim": {"claimed_tier": "production_permitted", "product_capabilities": [some_cap]},
        "unbacked": {"tier": "video_evidence", "depends_on": ["no_such_capability"]},
        "modest": {"tier": "exists_in_code"},
    }}
    nodes = {n["id"]: n for n in app_module.registry_node_tiers(registry, document)}
    assert nodes["strong_claim"]["tier"] == claimed[some_cap], "capped by its component"
    assert nodes["strong_claim"]["capped_by"] == [some_cap]
    assert nodes["strong_claim"]["production_validated"] is False
    assert nodes["unbacked"]["tier"] == "exists_in_code"
    assert nodes["modest"]["tier"] == "exists_in_code" and nodes["modest"]["capped_by"] == []
    assert app_module.registry_node_tiers(registry, {}) == []


def test_the_tiers_endpoint_includes_a_nodes_list(backend):
    client, headers, _ = backend
    body = client.get("/v1/capability-tiers", headers=headers).json()
    assert isinstance(body["nodes"], list)
    assert all(not n["production_validated"] for n in body["nodes"])


# ---------------------------------------------------------------------- Jupiter on the target device
def _proposal(target, predicate="present", **extra):
    rule = {"target": target, "predicate": predicate, "description": f"{predicate} {target}", **extra}
    return {"preset": "", "variant": "", "explanation": "", "rules": [rule], "settings": {}, "unsupported": []}


def test_jupiter_refuses_a_pose_rule_on_a_device_without_pose(backend, monkeypatch):
    client, headers, app_module = backend
    _report(client, role=GENERAL, preset="security_monitoring")  # pose: unavailable
    seen = {}

    def fake_llm(prompt, reg=None):
        seen["reg"] = reg
        return _proposal("on_floor")

    monkeypatch.setattr(app_module, "_ai_configure_llm", fake_llm)
    result = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "tell me if someone falls"},
                         headers=headers).json()
    assert result["rules"] == []
    assert "this device reports no pose capability" in json.dumps(result["unsupported"])
    assert "no pose model on this device" in json.dumps(result["unsupported"])
    assert seen["reg"] is not None, "the model was briefed with the device's capabilities"


def test_jupiter_accepts_a_pose_rule_when_the_device_has_pose_whatever_the_server_has(backend, monkeypatch):
    client, headers, app_module = backend
    with_pose = _perception()
    with_pose["components"]["pose"] = {"state": "ok"}
    _report(client, role=GENERAL, preset="elderly_monitoring", perception=with_pose)
    monkeypatch.setattr(app_module, "_ai_configure_llm", lambda prompt, reg=None: _proposal("on_floor"))
    result = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "tell me if someone falls"},
                         headers=headers).json()
    assert [r["target"] for r in result["rules"]] == ["on_floor"], result["unsupported"]


def test_jupiter_accepts_a_person_in_zone_rule_on_a_device_with_objects(backend):
    client, headers, _ = backend
    _report(client, role=GENERAL, preset="security_monitoring")
    client.put("/v1/devices/1/zones", json={"zones": [
        {"name": "marked area", "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]], "kind": "restricted"}]},
        headers=headers)
    result = client.post("/v1/ai/configure", json={
        "device_id": 1, "prompt": "Alert me if someone stays in the marked area for 10 seconds"},
        headers=headers).json()
    assert [(r["target"], r["predicate"]) for r in result["rules"]] in (
        [("person", "in_zone")], [("person", "lingering")]), result
    assert result["unsupported"] == []


def test_jupiter_refuses_object_rules_on_a_device_with_no_detector(backend, monkeypatch):
    client, headers, app_module = backend
    blind = _perception(components={"object_detector": {"state": "unavailable", "reason": "no .hef installed"},
                                    "face_landmarks": {"state": "ok"}})
    _report(client, perception=blind)
    monkeypatch.setattr(app_module, "_ai_configure_llm", lambda prompt, reg=None: _proposal("cell phone"))
    result = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "phones"}, headers=headers).json()
    assert result["rules"] == [] and "no .hef installed" in json.dumps(result["unsupported"])


def test_an_old_device_keeps_todays_validation(backend, monkeypatch):
    client, headers, app_module = backend
    _report(client, role=GENERAL, preset="security_monitoring", with_perception=False)
    calls = []

    def legacy_llm(prompt):  # the pre-M7 signature: an old device must not change the call
        calls.append(prompt)
        return _proposal("person")

    monkeypatch.setattr(app_module, "_ai_configure_llm", legacy_llm)
    result = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "people"}, headers=headers).json()
    assert calls == ["people"] and [r["target"] for r in result["rules"]] == ["person"]


# ---------------------------------------------------------------------- LLM output is data, never code
def test_model_output_is_never_executed(backend, monkeypatch, tmp_path):
    """Whatever the model returns -- here, Python and shell in every field -- is validated
    as data and refused; nothing it says is evaluated, executed or compiled."""
    client, headers, app_module = backend
    _report(client, role=GENERAL, preset="security_monitoring")
    marker = tmp_path / "pwned"
    payload = f"__import__('os').system('touch {marker}')"
    hostile = _proposal(payload, predicate=payload, secondary=payload, zone=payload, description=payload,
                        persist_s=payload, schedule=payload)
    hostile["explanation"] = payload
    hostile["preset"] = payload
    monkeypatch.setattr(app_module, "_ai_configure_llm", lambda prompt, reg=None: hostile)

    executed = []
    for name in ("eval", "exec", "compile"):
        real = getattr(builtins, name)

        def spy(source, *args, _real=real, **kwargs):
            if isinstance(source, (str, bytes)) and "__import__('os')" in str(source):
                executed.append(source)
                raise AssertionError("model output reached " + _real.__name__)
            return _real(source, *args, **kwargs)

        monkeypatch.setattr(builtins, name, spy)
    resp = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "anything"}, headers=headers)
    assert resp.status_code == 200, resp.text
    assert executed == [] and not marker.exists()
    assert resp.json()["rules"] == [], "an unknown target/condition never becomes a runnable rule"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def _module_path(name: str):
    candidate = REPO.joinpath(*name.split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path
    return None


def test_the_per_frame_path_never_reaches_a_language_model():
    """Everything a device runs per frame -- the runtime loop, rules, uplink, the M7
    perception engine and nodes when present, and every preset's module -- is closed
    under its static imports without touching jupiter/, cloud/, an LLM SDK, or the VLM
    judge. Imports inside functions count (ast.walk sees them). The VLM judge is used
    only by the comparison-only driver_attention_max module, which no preset runs."""
    import yaml

    presets = yaml.safe_load((REPO / "presets.yaml").read_text())["presets"]
    preset_modules = {f"modules.{p['module']}" for p in presets.values() if p.get("module")}
    assert "modules.driver_attention_max" not in preset_modules
    roots = {"services.runtime", "core.rules", "core.uplink", "core.perception", "core.nodes", *preset_modules}
    forbidden = ("jupiter", "cloud", "anthropic", "openai", "core.vlm_judge")
    # The Mac-only comparison module (profile mac_max, never a device profile): not
    # followed, and proven below to be imported only on that branch.
    mac_only = {"modules.driver_attention_max"}
    seen, stack = set(), [r for r in roots if _module_path(r)]
    while stack:
        name = stack.pop()
        if name in seen or name in mac_only:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is None:
            continue
        for imported in _imports(path):
            assert not any(imported == f or imported.startswith(f + ".") for f in forbidden), (name, imported)
            for prefix in (imported, imported.rpartition(".")[0]):
                if prefix and _module_path(prefix) and prefix not in seen:
                    stack.append(prefix)
    assert "services.runtime" in seen and len(seen) > 10


def test_the_vlm_judge_module_is_only_imported_on_the_mac_max_profile():
    """main.py reaches modules.driver_attention_max (and through it the VLM judge) only
    inside `if profile == "mac_max":`, the Mac comparison profile; a device runs `pi`."""
    tree = ast.parse((REPO / "main.py").read_text())
    guarded, total = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "mac_max" in ast.unparse(node.test):
            guarded += sum(1 for n in ast.walk(node) if isinstance(n, ast.ImportFrom)
                           and n.module == "modules.driver_attention_max")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modules.driver_attention_max":
            total += 1
    assert total >= 1 and guarded == total
