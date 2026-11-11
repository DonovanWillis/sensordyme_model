"""M6 backend contract: preset library, per-device availability, settings,
custom-rule status and toggling, and prompt -> policy through /v1/ai/configure.

docs/PRESET_LIBRARY.md is the field-by-field contract these pin. Device reports
are built from the real core/monitoring_state.py MonitoringReport, the shape
services/runtime.py sends.
"""

from __future__ import annotations

import importlib
import json
import sys
import time

import pytest

from core import monitoring_state as ms
from core import roles
from core.presets import DEVICE_POLICY_FEATURES
from core.rules import RuleEngine, validate_rule_set

DRIVER = roles.ROLES["driver_safety"]
GENERAL = roles.ROLES["general"]
DEVICE = {"X-Device-Key": "dk_test"}
DWELL_PROMPT = "Alert me if someone stays inside this marked area for longer than 10 seconds"
ZONE = {"name": "front door", "kind": "restricted", "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Unit", "AA-BB-CC", "dk_test", time.time()))
    token = client.post("/v1/signup", json={"email": "m6@example.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return client, headers, app_module


# ---------------------------------------------------------------------- device side
def report(client, *, role=GENERAL, preset="security_monitoring", mode="standard", features=DEVICE_POLICY_FEATURES,
           custom_rules=(), rules_version="", active=False, impaired=None, rejection=None, extra_caps=()):
    """One ingest carrying a real monitoring report. Returns the command."""
    caps = {name: ms.Capability() for name in ("camera", "inference", "objects", "face", "pose", *extra_caps)}
    caps["alert_output"] = ms.Capability()
    caps["audible_alert"] = ms.Capability()
    for name, reason in (impaired or {}).items():
        caps[name] = ms.Capability(ms.IMPAIRED, reason)
    monitoring_report = ms.MonitoringReport(
        state=ms.MonitoringState.READY, reasons=(), role=role.name, active_preset=preset, module="x", mode=mode,
        custom_rules=tuple(custom_rules), capabilities=caps, policies=(), actuators=(), last_frame_age_s=0.1,
        last_inference_age_s=0.1, model={}, versions={},
    ).to_dict()
    pushed = {"instruction": "", "rule_count": len(custom_rules), "version": rules_version, "active": active,
              "overrides_mode": active, "mode": mode, "role": role.describe(), "last_rejection": rejection,
              "protection": {"state": "protecting", "role": role.name, "reasons": []}, "state": "ready"}
    if features is not None:
        pushed["features"] = list(features)
    body = {"events": [], "active_preset": preset, "rules_version": rules_version,
            "health": {"monitoring": monitoring_report}, "monitoring": pushed}
    resp = client.post("/v1/ingest", json=body, headers=DEVICE)
    assert resp.status_code == 200, resp.text
    return resp.json()["command"]


def device_presets(client, headers) -> dict:
    resp = client.get("/v1/devices/1/presets", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def by_id(body: dict) -> dict:
    return {p["id"]: p for p in body["presets"]}


def configure(client, headers, prompt=DWELL_PROMPT) -> dict:
    resp = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": prompt}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def activate(client, headers, draft_version=None, **extra):
    body = {"preset": "", **extra}
    if draft_version:
        body["draft_version"] = draft_version
    resp = client.post("/v1/devices/1/preset", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------- GET /v1/presets
def test_the_catalogue_endpoint(backend):
    client, headers, _ = backend
    assert client.get("/v1/presets").status_code == 401
    body = client.get("/v1/presets", headers=headers).json()
    assert body["catalog_version"]
    presets = {p["id"]: p for p in body["presets"]}
    assert set(presets) >= {"driver_safety", "baby_monitoring", "security_monitoring", "elderly_monitoring",
                            "occupancy", "workplace_safety"}
    driver = presets["driver_safety"]
    assert {v["id"] for v in driver["variants"]} == {"standard", "strict", "teen_driver", "long_haul"}
    assert driver["validation"]["status"] == "unevaluated"
    assert presets["security_monitoring"]["validation"]["status"] == "benchmarked_offline"
    assert presets["security_monitoring"]["validation"]["evidence"]
    # M7: node-backed (module shared_perception), runnable, honest about PPE.
    assert presets["workplace_safety"]["availability"] == "available"
    assert presets["workplace_safety"]["nodes"] == ["zone_watch"]
    keys = {s["key"] for s in driver["settings"]}
    assert {"drowsy_seconds", "phone_use_seconds", "enable_hands"} <= keys


# ---------------------------------------------------------------------- availability
def test_a_device_that_never_reported_is_stale(backend):
    client, headers, _ = backend
    body = device_presets(client, headers)
    assert body["stale"] is True and body["active_preset"] is None
    presets = by_id(body)
    assert presets["driver_safety"] == {**presets["driver_safety"], "available": False, "code": "device_stale"}
    assert presets["occupancy"]["code"] == "device_stale" and presets["occupancy"]["reason"]


def test_role_forbids_and_active_preset_with_its_settings(backend):
    client, headers, _ = backend
    report(client, role=DRIVER, preset="driver_safety:strict@phone_use_seconds=0.8")
    body = device_presets(client, headers)
    presets = by_id(body)
    assert body["stale"] is False
    assert body["active_preset"] == "driver_safety" and body["active_variant"] == "strict"
    assert presets["driver_safety"]["available"] and presets["driver_safety"]["active"]
    assert presets["driver_safety"]["settings"]["drowsy_seconds"] == 1.5      # the variant
    assert presets["driver_safety"]["settings"]["phone_use_seconds"] == 0.8   # the owner's setting
    for other in ("baby_monitoring", "security_monitoring", "elderly_monitoring"):
        assert presets[other]["code"] == "role_forbids_preset" and not presets[other]["available"]
    assert presets["workplace_safety"]["code"] == "role_forbids_preset"
    assert presets["driver_safety"]["validation_status"] == "unevaluated"


def test_impaired_and_unsupported_capabilities(backend):
    client, headers, _ = backend
    report(client, impaired={"pose": "pose landmarks not produced for 10 s",
                             "audible_alert": "no audible alert output is configured"})
    presets = by_id(device_presets(client, headers))
    assert presets["elderly_monitoring"]["code"] == "capability_impaired"
    assert "pose" in presets["elderly_monitoring"]["reason"]
    assert presets["driver_safety"]["code"] == "capability_unsupported"
    # Pose is only optional for security: it stays selectable.
    assert presets["security_monitoring"]["available"] and presets["security_monitoring"]["active"]


def test_ai_custom_runs_no_preset(backend):
    client, headers, _ = backend
    report(client, mode="ai_custom", active=True)
    assert device_presets(client, headers)["active_preset"] is None


# ---------------------------------------------------------------------- POST .../preset settings
def test_settings_are_validated_naming_the_key(backend):
    client, headers, _ = backend
    report(client)
    for settings, key in [({"linger_s": 1}, "linger_s"), ({"bogus": 1}, "bogus"), ({"linger_s": "30"}, "linger_s")]:
        resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring", "settings": settings},
                           headers=headers)
        assert resp.status_code == 400 and resp.json()["code"] == "invalid_setting"
        assert key in resp.json()["detail"]


def test_settings_are_delivered_as_the_name_the_device_applies_and_converge(backend):
    client, headers, _ = backend
    report(client)
    resp = client.post("/v1/devices/1/preset",
                       json={"preset": "security_monitoring:away", "settings": {"linger_s": 60, "crowd_threshold": 6}},
                       headers=headers)
    assert resp.status_code == 200, resp.text
    name = "security_monitoring:away@crowd_threshold=6,linger_s=60"
    assert resp.json()["desired_preset"] == name
    assert report(client)["set_preset"] == name
    # The device reports the same mode (spelled its own way): converged, nothing re-sent.
    assert report(client, preset="security_monitoring:away@linger_s=60.0,crowd_threshold=6") is None
    presets = by_id(device_presets(client, headers))
    assert presets["security_monitoring"]["settings"]["linger_s"] == 60.0


def test_settings_need_firmware_that_understands_them(backend):
    client, headers, _ = backend
    report(client, features=None)
    resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring", "settings": {"linger_s": 60}},
                       headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "firmware_unsupported"
    # Without settings the same firmware switches modes exactly as before.
    assert client.post("/v1/devices/1/preset", json={"preset": "security_monitoring"},
                       headers=headers).status_code == 200


def test_role_floors_win_over_settings(backend):
    client, headers, _ = backend
    report(client, role=DRIVER, preset="driver_safety")
    resp = client.post("/v1/devices/1/preset", json={"preset": "driver_safety", "settings": {"drowsy_seconds": 2.5}},
                       headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "role_forbids_preset"
    assert "floor" in resp.json()["detail"]
    ok = client.post("/v1/devices/1/preset", json={"preset": "driver_safety", "settings": {"drowsy_seconds": 1.5}},
                     headers=headers)
    assert ok.status_code == 200


def test_incompatible_combinations_are_refused_with_a_code(backend):
    client, headers, _ = backend
    report(client)
    # Settings with no mode to apply them to.
    resp = client.post("/v1/devices/1/preset", json={"preset": "", "settings": {"linger_s": 60}}, headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "incompatible_policy"
    # Rules kept beside a mode on a device where rules REPLACE the mode.
    resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring", "keep_rules": True},
                       headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "incompatible_policy"
    # A mode this software does not have is not selectable at all.
    resp = client.post("/v1/devices/1/preset", json={"preset": "teleportation"}, headers=headers)
    assert resp.status_code == 400 and resp.json()["code"] == "unknown_preset"


def test_an_additive_device_keeps_its_rules_through_a_settings_change(backend):
    """One preset + N custom rules + the role floor: changing the driver preset's
    settings must not silently drop the rules running beside it."""
    client, headers, _ = backend
    client.post("/v1/devices/1/zones", json={"zones": [ZONE]}, headers=headers)
    report(client, role=DRIVER, preset="driver_safety")
    draft = configure(client, headers, "alert me if someone stays in the front door for 10 seconds")
    first = activate(client, headers, draft["draft_version"])
    command = report(client, role=DRIVER, preset="driver_safety")
    assert command["set_rules"] and command["rules_version"] == first["rules_version"]
    report(client, role=DRIVER, preset="driver_safety", mode="standard_plus_custom", active=True,
           rules_version=first["rules_version"], custom_rules=["rule0:lingering:person"])

    resp = client.post("/v1/devices/1/preset", json={"preset": "driver_safety", "settings": {"drowsy_seconds": 1.5}},
                       headers=headers)
    assert resp.status_code == 200 and resp.json()["rules_active"] == 1
    command = report(client, role=DRIVER, preset="driver_safety", mode="standard_plus_custom", active=True,
                     rules_version=first["rules_version"], custom_rules=["rule0:lingering:person"])
    # The switch and the rules travel together: the device re-applies the rules after switching.
    assert command["set_preset"] == "driver_safety@drowsy_seconds=1.5"
    assert command["set_rules"][0]["dwell_s"] == 10.0
    kept_version = command["rules_version"]
    # Converged in standard_plus_custom: no mode or rules are re-sent (re-sending
    # the mode would make the device exit the rules beside it).
    command = report(client, role=DRIVER, preset="driver_safety@drowsy_seconds=1.5", mode="standard_plus_custom",
                     active=True, rules_version=kept_version, custom_rules=["rule0:lingering:person"])
    assert not {"set_preset", "set_rules"} & set(command or {})


# ---------------------------------------------------------------------- prompt -> policy
def test_dwell_prompt_to_policy_to_device(backend):
    client, headers, _ = backend
    client.post("/v1/devices/1/zones", json={"zones": [ZONE]}, headers=headers)
    report(client)
    draft = configure(client, headers)
    assert draft["source"] == "policy_parser" and draft["preset"] == ""
    (policy,) = draft["policies"]
    assert policy["zone"] == {"name": "front door", "drawn": True} and policy["status"] == "ready"
    assert policy["temporal"] == {"dwell_s": 10.0} and policy["rule_id"] == draft["rules"][0]["id"]
    assert policy["explanation"]["does_not_monitor"]
    # Nothing runs until the owner confirms.
    assert "set_rules" not in (report(client) or {})
    activated = activate(client, headers, draft["draft_version"])
    command = report(client)
    (rule,) = command["set_rules"]
    assert rule["dwell_s"] == 10.0 and rule["zone"] == "front door"
    # What the device receives, it accepts and can run.
    assert validate_rule_set(command["set_rules"], label_ok=lambda label: True) == []
    assert RuleEngine(command["set_rules"]).rules[0].dwell_s == 10.0
    assert command["rules_version"] == activated["rules_version"]


def test_an_undrawn_area_is_waiting(backend):
    client, headers, _ = backend
    report(client)
    draft = configure(client, headers)
    assert draft["policies"][0]["status"] == "needs_zone"
    activated = activate(client, headers, draft["draft_version"])
    report(client, mode="ai_custom", active=True, rules_version=activated["rules_version"],
           custom_rules=["rule0:lingering:person"])
    rules = client.get("/v1/devices/1/rules", headers=headers).json()
    assert rules["rules"][0]["status"] == "waiting" and "marked area" in rules["rules"][0]["reason"]


@pytest.mark.parametrize("prompt", [
    "alert me if a stranger stays at the door for 10 seconds",
    "tell me if the baby is crying",
    "tell me if my husband looks angry",
    "notify me if someone looks suspicious",
])
def test_unsupported_prompts_fail_clearly_and_set_up_nothing(backend, prompt, monkeypatch):
    client, headers, app_module = backend
    # Even a model that would happily produce "a person is present" is never asked.
    monkeypatch.setattr(app_module, "_ai_configure_llm", lambda p: pytest.fail("the model must not be asked"))
    draft = configure(client, headers, prompt)
    assert draft["rules"] == [] and draft["policies"] == []
    assert draft["unsupported"] and draft["unsupported"][0]["support"] == "unsupported"
    assert draft["source"] == "policy_guard"


def test_model_output_is_validated_against_the_typed_schema(backend, monkeypatch):
    client, headers, app_module = backend
    rule = {"target": "person", "predicate": "lingering", "secondary": "", "description": "dwell anywhere",
            "zone": "", "persist_s": 10, "min_count": 1, "confidence": 0.6, "severity": "warning",
            "cooldown_s": 60, "schedule": "", "notify": "push", "dwell_s": 10}
    monkeypatch.setattr(app_module, "_ai_configure_llm", lambda p: {
        "preset": "", "variant": "", "explanation": "x", "settings": {}, "unsupported": [], "source": "jupiter",
        "rules": [rule, {**rule, "zone": "porch", "description": "dwell on the porch"}]})
    draft = configure(client, headers, "watch the porch")
    assert [r["zone"] for r in draft["rules"]] == ["porch"]
    assert any("area" in u["reason"] for u in draft["unsupported"])


def test_multiple_policies_coexist_and_round_trip(backend):
    client, headers, _ = backend
    client.post("/v1/devices/1/zones", json={"zones": [ZONE]}, headers=headers)
    report(client)
    draft = configure(client, headers, "alert me if someone stays in the front door for 10 seconds. "
                                       "Tell me if a dog enters the garden")
    assert len(draft["policies"]) == 2
    activated = activate(client, headers, draft["draft_version"])
    assert activated["rules_active"] == 2
    listed = client.get("/v1/devices/1/rules", headers=headers).json()
    assert [r["id"] for r in listed["rules"]] == [r["id"] for r in draft["rules"]]
    assert listed["rules"][0]["dwell_s"] == 10.0 and listed["rules"][0]["zone"] == "front door"


# ---------------------------------------------------------------------- GET/PATCH rules
def _two_rules(client, headers):
    client.post("/v1/devices/1/zones", json={"zones": [ZONE]}, headers=headers)
    report(client)
    draft = configure(client, headers, "alert me if someone stays in the front door for 10 seconds. "
                                       "Tell me if a dog enters the garden")
    activated = activate(client, headers, draft["draft_version"])
    report(client)  # delivery
    return draft, activated


def test_rule_status_comes_from_the_device(backend):
    client, headers, _ = backend
    _draft, activated = _two_rules(client, headers)
    version = activated["rules_version"]
    # Delivered but not yet confirmed.
    listed = client.get("/v1/devices/1/rules", headers=headers).json()
    assert listed["rules_version"] == version and listed["additive"] is False
    assert [r["status"] for r in listed["rules"]] == ["unknown", "waiting"]
    # The device reports rule0 running.
    report(client, mode="ai_custom", active=True, rules_version=version,
           custom_rules=["rule0:lingering:person", "rule1:in_zone:dog"])
    listed = client.get("/v1/devices/1/rules", headers=headers).json()
    first, second = listed["rules"]
    assert first["status"] == "running" and first["reason"] is None
    assert first["targets"] == ["person"] and first["required_capabilities"] == ["camera", "inference", "objects"]
    assert second["status"] == "waiting"   # the garden is not drawn


def test_a_rejected_set_is_reported_rejected(backend):
    client, headers, _ = backend
    _draft, activated = _two_rules(client, headers)
    report(client, rules_version="old", rejection={"kind": "rules", "subject": activated["rules_version"],
                                                     "reasons": ["rule 1: this device cannot detect 'person'"],
                                                     "at": time.time()})
    first = client.get("/v1/devices/1/rules", headers=headers).json()["rules"][0]
    assert first["status"] == "rejected" and "cannot detect" in first["reason"]


def test_a_stale_device_reports_unknown(backend):
    client, headers, app_module = backend
    _two_rules(client, headers)
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET last_seen = ? WHERE id = 1", (time.time() - 3600,))
    first = client.get("/v1/devices/1/rules", headers=headers).json()["rules"][0]
    assert first["status"] == "unknown"


def test_disabling_a_rule_stops_delivering_it(backend):
    client, headers, _ = backend
    draft, activated = _two_rules(client, headers)
    rule_id = draft["rules"][1]["id"]
    resp = client.patch(f"/v1/devices/1/rules/{rule_id}", json={"enabled": False}, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rule"]["enabled"] is False and body["rule"]["status"] == "disabled"
    assert body["rules_version"] != activated["rules_version"] and body["rules_active"] == 1
    command = report(client, rules_version=activated["rules_version"])
    assert [r["target"] for r in command["set_rules"]] == ["person"]
    assert command["rules_version"] == body["rules_version"]
    # Re-enabled: delivered again, under a new version.
    again = client.patch(f"/v1/devices/1/rules/{rule_id}", json={"enabled": True}, headers=headers).json()
    command = report(client, rules_version=body["rules_version"])
    assert [r["target"] for r in command["set_rules"]] == ["person", "dog"]
    assert command["rules_version"] == again["rules_version"]
    # Persisted: a fresh read sees the same ids and states.
    listed = client.get("/v1/devices/1/rules", headers=headers).json()
    assert [r["enabled"] for r in listed["rules"]] == [True, True]


def test_unknown_rule_is_404_with_a_code(backend):
    client, headers, _ = backend
    resp = client.patch("/v1/devices/1/rules/nope", json={"enabled": False}, headers=headers)
    assert resp.status_code == 404 and resp.json()["code"] == "rule_not_found"
    resp = client.patch("/v1/devices/9/rules/nope", json={"enabled": False}, headers=headers)
    assert resp.status_code == 404 and resp.json()["code"] == "device_not_found"


def test_legacy_rows_without_ids_get_stable_ids(backend):
    client, headers, app_module = backend
    legacy = {"preset": "", "rules": [{"target": "person", "predicate": "present", "description": "anyone"}]}
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET ai_rules = ?, custom_monitoring_wanted = 1 WHERE id = 1",
                     (json.dumps(legacy),))
    first = client.get("/v1/devices/1/rules", headers=headers).json()["rules"][0]["id"]
    assert first == client.get("/v1/devices/1/rules", headers=headers).json()["rules"][0]["id"]
    assert client.patch(f"/v1/devices/1/rules/{first}", json={"enabled": False},
                        headers=headers).status_code == 200


def test_an_additive_device_can_still_drop_its_rules_explicitly(backend):
    client, headers, _ = backend
    client.post("/v1/devices/1/zones", json={"zones": [ZONE]}, headers=headers)
    report(client, role=DRIVER, preset="driver_safety")
    draft = configure(client, headers, "alert me if someone stays in the front door for 10 seconds")
    first = activate(client, headers, draft["draft_version"])
    resp = client.post("/v1/devices/1/preset", json={"preset": "driver_safety", "keep_rules": False},
                       headers=headers)
    assert resp.json()["rules_active"] == 0
    # The device still runs them beside the mode: the mode is re-sent so it exits them.
    command = report(client, role=DRIVER, preset="driver_safety", mode="standard_plus_custom", active=True,
                     rules_version=first["rules_version"], custom_rules=["rule0:lingering:person"])
    assert command["set_preset"] == "driver_safety" and "set_rules" not in command
    rules = client.get("/v1/devices/1/rules", headers=headers).json()["rules"]
    assert rules[0]["status"] == "disabled" and rules[0]["enabled"] is True
