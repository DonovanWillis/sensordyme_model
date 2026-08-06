"""End-to-end test of the cloud backend: the full customer journey over the real API."""

import importlib
import sys

import pytest

pytest.importorskip("fastapi")  # backend deps are dev/cloud-only, not installed on the Pi
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    return app_module, TestClient(app_module.app)


def test_health_probe_is_unauthenticated(backend):
    _, client = backend
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["service"] == "sensordyme-cloud"


def test_full_customer_journey(backend):
    app_module, client = backend

    # 0) factory provisions a device (what cloud/manage.py create-device does)
    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Car unit", "AB-CD-EF", "dk_testkey", _time.time()),
        )

    # 1) customer signs up in the app
    r = client.post("/v1/signup", json={"email": "driver@example.com", "password": "supersecret1"})
    assert r.status_code == 200
    auth = {"Authorization": f"Bearer {r.json()['token']}"}

    # duplicate signup rejected; bad login rejected; good login works
    assert client.post("/v1/signup", json={"email": "driver@example.com", "password": "supersecret1"}).status_code == 409
    assert client.post("/v1/login", json={"email": "driver@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/v1/login", json={"email": "driver@example.com", "password": "supersecret1"}).status_code == 200

    # 2) customer pairs the device with its claim code
    r = client.post("/v1/devices/claim", json={"claim_code": "AB-CD-EF"}, headers=auth)
    assert r.status_code == 200
    device_id = r.json()["device_id"]

    # 3) device comes online and pushes an alert (what core/uplink.py sends)
    device_headers = {"X-Device-Key": "dk_testkey"}
    r = client.post(
        "/v1/ingest",
        json={
            "active_preset": "driver_distraction",
            "events": [
                {
                    "timestamp": "2026-07-16T10:00:00+00:00",
                    "type": "phone_use_while_distracted",
                    "confidence": 0.95,
                    "signals": "phone + hand_overlap + gaze_off",
                    "severity": "loud",
                }
            ],
        },
        headers=device_headers,
    )
    assert r.status_code == 200
    assert r.json()["stored"] == 1 and r.json()["command"] is None
    assert client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": "bogus"}).status_code == 401

    # 4) app sees the device online with its alert history
    r = client.get("/v1/devices", headers=auth)
    device = r.json()["devices"][0]
    assert device["online"] is True and device["active_preset"] == "driver_distraction"
    r = client.get(f"/v1/devices/{device_id}/events", headers=auth)
    assert r.json()["events"][0]["type"] == "phone_use_while_distracted"

    # 5) customer switches mode from the app -> device receives the command on next sync
    r = client.post(f"/v1/devices/{device_id}/preset", json={"preset": "elderly_monitoring"}, headers=auth)
    assert r.status_code == 200
    r = client.post("/v1/ingest", json={"events": [], "active_preset": "driver_distraction"}, headers=device_headers)
    assert r.json()["command"] == {"set_preset": "elderly_monitoring"}

    # 6) device applies it and reports back -> command clears
    r = client.post("/v1/ingest", json={"events": [], "active_preset": "elderly_monitoring"}, headers=device_headers)
    assert r.json()["command"] is None

    # 6b) AI configuration: plain language -> a preset + rules (keyword fallback path,
    # since no GEMINI_API_KEY is set in the test environment)
    r = client.post(
        "/v1/ai/configure",
        json={"device_id": device_id, "prompt": "watch my baby and alert me if her face gets covered"},
        headers=auth,
    )
    assert r.status_code == 200
    proposal = r.json()
    assert proposal["preset"] == "baby_monitoring"
    assert proposal["explanation"] and isinstance(proposal["rules"], list)
    r = client.post(
        "/v1/ai/configure",
        json={"device_id": device_id, "prompt": "alert me if grandpa falls"},
        headers=auth,
    )
    assert r.json()["preset"] == "elderly_monitoring"

    # 6c) connected phone number: loud alerts also go out as SMS (log fallback w/o Twilio)
    assert client.post("/v1/account/phone", json={"phone": "not-a-number"}, headers=auth).status_code == 400
    assert client.post("/v1/account/phone", json={"phone": "+15551234567"}, headers=auth).status_code == 200
    r = client.post(
        "/v1/ingest",
        json={"events": [{"timestamp": "2026-07-16T10:05:00+00:00", "type": "drowsiness",
                          "severity": "loud", "signals": "sustained closure"}]},
        headers=device_headers,
    )
    assert r.status_code == 200  # sms path exercised via log fallback, never breaks ingestion

    # 6d) rename the device from the app
    assert client.post(f"/v1/devices/{device_id}/name", json={"name": "Dad's Car"}, headers=auth).status_code == 200
    assert client.get("/v1/devices", headers=auth).json()["devices"][0]["name"] == "Dad's Car"

    # 6e) order more devices from the app
    assert client.post("/v1/orders", json={"quantity": 0, "address": "x"}, headers=auth).status_code == 400
    r = client.post("/v1/orders", json={"quantity": 2, "address": "1 Main Street, Springfield, USA"}, headers=auth)
    assert r.status_code == 200 and r.json()["status"] == "placed"
    orders = client.get("/v1/orders", headers=auth).json()["orders"]
    assert len(orders) == 1 and orders[0]["quantity"] == 2

    # 6f) preset with a variant flows through the command channel untouched.
    # The fall rule configured in 6b also rides down on this poll: the device has
    # not acknowledged a rules_version yet, and rules the user configured are
    # useless until the device that enforces them actually receives them.
    client.post(f"/v1/devices/{device_id}/preset", json={"preset": "driver_distraction:teen_driver"}, headers=auth)
    r = client.post("/v1/ingest", json={"events": [], "active_preset": "elderly_monitoring"}, headers=device_headers)
    command = r.json()["command"]
    assert command["set_preset"] == "driver_distraction:teen_driver"
    assert [rule["detection"] for rule in command["set_rules"]] == ["fall"]

    # Once acknowledged, only the preset command remains outstanding.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "elderly_monitoring", "rules_version": command["rules_version"]},
        headers=device_headers,
    )
    assert r.json()["command"] == {"set_preset": "driver_distraction:teen_driver"}

    # 7) another user cannot see or control this device
    r2 = client.post("/v1/signup", json={"email": "stranger@example.com", "password": "supersecret2"})
    stranger = {"Authorization": f"Bearer {r2.json()['token']}"}
    assert client.get(f"/v1/devices/{device_id}/events", headers=stranger).status_code == 404
    assert client.post(f"/v1/devices/{device_id}/preset", json={"preset": "x"}, headers=stranger).status_code == 404
    assert client.post("/v1/devices/claim", json={"claim_code": "AB-CD-EF"}, headers=stranger).status_code == 409
