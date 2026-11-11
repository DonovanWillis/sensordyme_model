"""The app status contract (docs/APP_STATUS_CONTRACT.md): GET /v1/devices `status`.

The app showed a green "All clear" while the device was FAILED or STOPPED: it
read none of the device's monitoring-truth fields. These pin what the backend
now computes from the device's own reports, and the machine-readable refusal
codes the app branches on instead of "any 409 means the proposal changed".
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import time
from pathlib import Path

import pytest

from core import monitoring_state as ms
from core import roles

DRIVER = roles.ROLES["driver_safety"]
GENERAL = roles.ROLES["general"]
CONTRACT_DOC = Path(__file__).resolve().parent.parent / "docs" / "APP_STATUS_CONTRACT.md"


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Car", "AA-BB-CC", "dk_test", time.time()))
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return client, headers, app_module


# ---------------------------------------------------------------------- device reports
def _monitoring_report(state: ms.MonitoringState, *, role=DRIVER, reasons=(), custom_rules=(),
                       mode="standard", impaired=None) -> dict:
    """A real MonitoringReport.to_dict(), the exact shape services/runtime.py sends."""
    caps = {name: ms.Capability() for name in ("camera", "inference", "face", "objects")}
    for name, reason in (impaired or {}).items():
        caps[name] = ms.Capability(ms.IMPAIRED, reason)
    return ms.MonitoringReport(
        state=state, reasons=tuple(reasons), role=role.name, active_preset="driver_safety",
        module="driver_attention", mode=mode, custom_rules=tuple(custom_rules), capabilities=caps,
        policies=(), actuators=(), last_frame_age_s=0.1, last_inference_age_s=0.1,
        model={}, versions={},
    ).to_dict()


def _protection(state="protecting", reasons=(), role=DRIVER) -> dict:
    return {"state": state, "role": role.name, "reasons": list(reasons), "degraded": [],
            "monitoring_state": "ready", "last_command_id": None}


def _health(report: dict, protection: dict) -> dict:
    return {"camera": {"state": "ok", "detail": "capturing", "metrics": {}},
            "inference": {"state": "ok", "detail": "running", "metrics": {}},
            "monitoring": report, "monitoring_active": report["monitoring"], "protection": protection}


def _pushed(role=DRIVER, *, as_dict=True, state="ready", last_rejection=None, mode="standard",
            active=False, protection=None) -> dict:
    """What _publish_monitoring sends: role as describe() (or, older, a bare name)."""
    return {"instruction": "", "rule_count": 0, "active": active, "mode": mode,
            "role": role.describe() if as_dict else role.name,
            "protection": protection or _protection(role=role), "last_rejection": last_rejection,
            "state": state}


def _ingest(client, *, health=None, monitoring=None, preset="driver_safety", rules_version=None):
    body = {"events": [], "active_preset": preset, "health": health, "monitoring": monitoring}
    if rules_version is not None:
        body["rules_version"] = rules_version
    resp = client.post("/v1/ingest", json=body, headers={"X-Device-Key": "dk_test"})
    assert resp.status_code == 200, resp.text
    return resp.json()["command"]


def _status(client, headers) -> dict:
    devices = client.get("/v1/devices", headers=headers).json()["devices"]
    assert len(devices) == 1
    return devices[0]["status"]


def _report_ready(client, role=DRIVER, **kw):
    report = _monitoring_report(ms.MonitoringState.READY, role=role, **kw)
    _ingest(client, health=_health(report, _protection(role=role)), monitoring=_pushed(role))


# ---------------------------------------------------------------------- state and liveness
def test_a_live_ready_driver_device_reports_everything_the_app_needs(backend):
    client, headers, _ = backend
    _report_ready(client, impaired={"objects": "detector retrying"})
    status = _status(client, headers)
    assert status["monitoring_state"] == "ready"
    assert status["monitoring_active"] is True
    assert status["protection"] == "protecting"
    assert status["stale"] is False and isinstance(status["reported_at"], float)
    assert status["role"] == "driver_safety" and status["role_title"] == "Driver safety"
    assert status["allowed_presets"] == ["driver_safety"]
    assert status["mode"] == "standard"
    assert "face" in status["capabilities"]["available"]
    assert status["capabilities"]["impaired"] == [{"name": "objects", "reason": "detector retrying"}]
    assert status["rules"] == {"active": [], "additive": True, "rejected": None}
    assert status["last_rejection"] is None and status["preset_rejection"] is None


@pytest.mark.parametrize(("state", "protection", "reason"), [
    (ms.MonitoringState.FAILED, "not_protecting", "required capability unavailable — face: no landmarks"),
    (ms.MonitoringState.STOPPED, "stopped", "monitoring was stopped explicitly"),
    (ms.MonitoringState.DEGRADED, "protecting", "phone_use: needs objects"),
    (ms.MonitoringState.STARTING, "not_protecting", "camera: starting"),
])
def test_failed_stopped_degraded_and_starting_reach_the_app_with_reasons(backend, state, protection, reason):
    client, headers, _ = backend
    report = _monitoring_report(state, reasons=[reason])
    _ingest(client, health=_health(report, _protection(protection, [reason])), monitoring=_pushed())
    status = _status(client, headers)
    assert status["monitoring_state"] == state.value
    assert status["monitoring_active"] is (state in (ms.MonitoringState.READY, ms.MonitoringState.DEGRADED))
    assert status["protection"] == protection
    assert status["reasons"] == [reason]
    assert status["protection_reasons"] == [reason]


def test_a_stale_report_is_unknown_never_the_last_ready(backend):
    client, headers, app_module = backend
    _report_ready(client)
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET last_seen = ? WHERE id = 1", (time.time() - 31,))
    status = _status(client, headers)
    assert status["stale"] is True
    assert status["monitoring_state"] == "unknown"
    assert status["monitoring_active"] is None
    assert status["protection"] == "unknown"
    assert status["capabilities"] is None and status["rules"]["active"] is None
    assert "30 s" in status["reasons"][0]
    # Configuration facts are not liveness claims: the role still constrains modes.
    assert status["role"] == "driver_safety" and status["allowed_presets"] == ["driver_safety"]


def test_a_device_that_never_reported_is_unknown(backend):
    client, headers, _ = backend
    status = _status(client, headers)
    assert (status["monitoring_state"], status["protection"], status["monitoring_active"]) == (
        "unknown", "unknown", None)
    assert status["stale"] is True and status["reported_at"] is None
    assert status["role"] is None and status["allowed_presets"] is None
    assert status["reasons"] == ["the device has not reported yet"]


def test_firmware_without_a_monitoring_state_is_unknown_not_ready(backend):
    client, headers, _ = backend
    old_health = {"camera": {"state": "ok", "detail": "capturing", "metrics": {}}, "monitoring_active": True}
    _ingest(client, health=old_health, monitoring={"active": False})
    status = _status(client, headers)
    assert status["stale"] is False
    assert status["monitoring_state"] == "unknown" and status["monitoring_active"] is None
    assert status["protection"] == "unknown"


def test_a_live_device_that_sends_no_health_is_unknown_but_not_stale(backend):
    client, headers, _ = backend
    _ingest(client, health=None, monitoring=None)
    status = _status(client, headers)
    assert status["stale"] is False
    assert status["monitoring_state"] == "unknown" and status["monitoring_active"] is None
    assert "does not report" in status["reasons"][0]


# ---------------------------------------------------------------------- role shapes
def test_role_resolves_from_the_bare_name_as_well_as_the_dict(backend):
    client, headers, _ = backend
    report = _monitoring_report(ms.MonitoringState.READY)
    _ingest(client, health=_health(report, _protection()), monitoring=_pushed(as_dict=False))
    status = _status(client, headers)
    assert status["role"] == "driver_safety" and status["allowed_presets"] == ["driver_safety"]
    # ...and the backend enforces it from that shape too (it used to read only the dict).
    resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring"}, headers=headers)
    assert resp.status_code == 409 and resp.json()["code"] == "role_forbids_preset"


def test_role_resolves_from_health_alone(backend):
    client, headers, _ = backend
    report = _monitoring_report(ms.MonitoringState.READY)
    _ingest(client, health=_health(report, _protection()), monitoring=None)
    assert _status(client, headers)["role"] == "driver_safety"


def test_a_general_device_allows_every_preset(backend):
    client, headers, _ = backend
    _report_ready(client, role=GENERAL)
    status = _status(client, headers)
    assert status["role"] == "general" and status["allowed_presets"] is None
    assert status["rules"]["additive"] is False


# ---------------------------------------------------------------------- refusal codes
def test_every_refusal_the_app_distinguishes_carries_a_code_and_a_string_detail(backend):
    client, headers, _ = backend
    _ingest(client, monitoring={"active": False})  # pre-M0.1 firmware: no protection block
    cases = [
        (client.post("/v1/devices/1/protection", json={"action": "stop", "confirm": True}, headers=headers),
         409, "firmware_unsupported"),
        (client.post("/v1/devices/1/protection", json={"action": "stop"}, headers=headers),
         400, "confirmation_required"),
        (client.post("/v1/devices/1/preset", json={"preset": "no_such_mode"}, headers=headers),
         400, "unknown_preset"),
        (client.post("/v1/devices/99/preset", json={"preset": "driver_safety"}, headers=headers),
         404, "device_not_found"),
        (client.post("/v1/devices/1/preset", json={"preset": "", "draft_version": "stale"}, headers=headers),
         409, "draft_superseded"),
    ]
    _report_ready(client)
    cases.append((client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring"}, headers=headers),
                  409, "role_forbids_preset"))
    for resp, status_code, code in cases:
        assert resp.status_code == status_code, (code, resp.text)
        body = resp.json()
        assert body["code"] == code
        assert isinstance(body["detail"], str) and body["detail"]


def test_role_refusal_is_told_apart_from_a_superseded_draft(backend):
    """Both are 409; the role check runs first. The app re-runs Jupiter only on draft_superseded."""
    client, headers, _ = backend
    _report_ready(client)
    resp = client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring", "draft_version": "x"},
                       headers=headers)
    assert (resp.status_code, resp.json()["code"]) == (409, "role_forbids_preset")
    assert "driver_safety" in resp.json()["detail"]


def test_other_conflicts_carry_codes_too(backend):
    client, _headers, _ = backend
    again = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"})
    assert (again.status_code, again.json()["code"]) == (409, "account_exists")
    other = client.post("/v1/signup", json={"email": "o@t.com", "password": "password1"}).json()["token"]
    taken = client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"},
                        headers={"Authorization": f"Bearer {other}"})
    assert (taken.status_code, taken.json()["code"]) == (409, "device_already_claimed")


# ---------------------------------------------------------------------- rules
_RULE = {"target": "person", "predicate": "present", "description": "Alert when a person appears"}


def test_rules_the_device_refused_are_reported_rejected_not_active(backend):
    client, headers, _ = backend
    _report_ready(client)
    activated = client.post("/v1/devices/1/preset", json={"preset": "", "rules": [_RULE]}, headers=headers).json()
    version = activated["rules_version"]
    assert activated["rules_active"] == 1 and version  # the backend's validation passed
    command = _ingest(client, monitoring=_pushed(), health=_health(
        _monitoring_report(ms.MonitoringState.READY), _protection()))
    assert command["rules_version"] == version
    # The device refuses the set (all or nothing) and keeps running nothing custom.
    refusal = {"kind": "rules", "subject": version, "reasons": ["rule 0: target 'person' not detected"],
               "at": time.time()}
    report = _monitoring_report(ms.MonitoringState.READY, custom_rules=())
    _ingest(client, health=_health(report, _protection()), monitoring=_pushed(last_rejection=refusal),
            rules_version=version)
    status = _status(client, headers)
    assert status["rules"]["active"] == []
    assert status["rules"]["rejected"]["version"] == version
    assert status["rules"]["rejected"]["reasons"] == refusal["reasons"]
    assert status["last_rejection"]["kind"] == "rules"


def test_an_old_rules_refusal_does_not_mark_a_later_accepted_set_rejected(backend):
    client, headers, _ = backend
    _report_ready(client)
    first = client.post("/v1/devices/1/preset", json={"preset": "", "rules": [_RULE]}, headers=headers).json()
    refusal = {"kind": "rules", "subject": first["rules_version"], "reasons": ["no"], "at": time.time()}
    second = client.post("/v1/devices/1/preset", json={"preset": "", "rules": [_RULE]}, headers=headers).json()
    assert second["rules_version"] != first["rules_version"]
    running = _monitoring_report(ms.MonitoringState.READY, custom_rules=["rule0:present:person"],
                                 mode="standard_plus_custom")
    _ingest(client, health=_health(running, _protection()),
            monitoring=_pushed(last_rejection=refusal, mode="standard_plus_custom", active=True),
            rules_version=second["rules_version"])
    status = _status(client, headers)
    assert status["rules"]["rejected"] is None
    assert status["rules"]["active"] == ["rule0:present:person"]
    assert status["mode"] == "standard_plus_custom"


def test_rules_sent_with_a_mode_on_an_additive_device_are_not_said_to_replace_it(backend):
    client, headers, _ = backend
    _report_ready(client)
    resp = client.post("/v1/devices/1/preset", json={"preset": "driver_safety", "rules": [_RULE]},
                       headers=headers).json()
    reason = resp["unsupported"][0]["reason"]
    assert "replaces" not in reason and "alongside its monitoring mode" in reason


def test_rules_sent_with_a_mode_on_a_general_device_keep_the_replacing_explanation(backend):
    client, headers, _ = backend
    _report_ready(client, role=GENERAL)
    resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring", "rules": [_RULE]},
                       headers=headers).json()
    assert "replaces the mode's own detection" in resp["unsupported"][0]["reason"]


# ---------------------------------------------------------------------- dropped presets
def test_a_forbidden_preset_dropped_at_ingest_is_recorded_for_the_app(backend):
    client, headers, app_module = backend
    _report_ready(client)
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET desired_preset = 'baby_monitoring' WHERE id = 1")
    assert "set_preset" not in (_ingest(client, monitoring=_pushed()) or {})
    status = _status(client, headers)
    rejection = status["preset_rejection"]
    assert rejection["preset"] == "baby_monitoring" and rejection["code"] == "role_forbids_preset"
    assert "driver_safety" in rejection["reason"]
    # The next accepted request clears it.
    assert client.post("/v1/devices/1/preset", json={"preset": "driver_safety:strict"},
                       headers=headers).status_code == 200
    assert _status(client, headers)["preset_rejection"] is None


def test_the_drop_uses_the_role_from_the_report_that_carries_it(backend):
    """The first report with a role must already stop a forbidden stored preset."""
    client, headers, app_module = backend
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET desired_preset = 'baby_monitoring' WHERE id = 1")
    report = _monitoring_report(ms.MonitoringState.READY)
    command = _ingest(client, health=_health(report, _protection()), monitoring=_pushed())
    assert "set_preset" not in (command or {})
    assert _status(client, headers)["preset_rejection"]["preset"] == "baby_monitoring"


# ---------------------------------------------------------------------- the contract document
def test_every_status_field_is_documented(backend):
    client, headers, _ = backend
    _report_ready(client)
    status = _status(client, headers)
    doc = CONTRACT_DOC.read_text()
    documented = set(re.findall(r"^\| `([a-z_.]+)`", doc, flags=re.MULTILINE))
    fields = set(status) | {f"rules.{k}" for k in status["rules"]} | {
        f"capabilities.{k}" for k in status["capabilities"]}
    assert fields <= documented, sorted(fields - documented)
    codes = set(re.findall(r"^\| `([a-z_]+)` \| \d{3}", doc, flags=re.MULTILINE))
    assert {"role_forbids_preset", "draft_superseded", "firmware_unsupported"} <= codes
    json.dumps(status)  # plain JSON all the way down


def test_the_app_decodes_every_status_field(backend):
    """ios/Sources/DeviceStatus.swift and _device_status are two halves of one
    contract; a field the app does not decode is a field it cannot show."""
    client, headers, _ = backend
    _report_ready(client)
    status = _status(client, headers)
    swift = (Path(__file__).resolve().parent.parent / "ios" / "Sources" / "DeviceStatus.swift").read_text()
    keys = re.search(r"enum CodingKeys: String, CodingKey \{(.*?)\n    \}", swift, flags=re.DOTALL).group(1)
    for field in status:
        assert f'"{field}"' in keys or re.search(rf"\bcase\b[^\n]*\b{field}\b", keys), field
