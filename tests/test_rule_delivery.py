"""Rules configured in the app must reach the device that enforces them.

Before this path existed, /v1/ai/configure validated a proposal and stored it on
the device row, and nothing ever read it back out — so a custom rule changed the
text in the app and nothing about what the camera watched for. These tests cover
the delivery contract: send on change, stay quiet otherwise.
"""

import importlib
import json
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

DEVICE_HEADERS = {"X-Device-Key": "dk_test"}


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Test unit", "AA-BB-CC", "dk_test", _time.time()),
        )
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return app_module, client, headers


def store_rules(app_module, rules):
    with app_module.db() as conn:
        conn.execute(
            "UPDATE devices SET ai_rules = ? WHERE id = 1",
            (json.dumps({"preset": "security_monitoring", "rules": rules}),),
        )


def poll(client, rules_version=""):
    """One device heartbeat; returns the command block (possibly None)."""
    resp = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "security_monitoring", "rules_version": rules_version},
        headers=DEVICE_HEADERS,
    )
    assert resp.status_code == 200
    return resp.json()["command"]


RULE = {
    "detection": "person_present", "description": "Alert if someone enters",
    "zone": "", "persist_s": 2.0, "min_count": 1, "confidence": 0.5,
    "severity": "warning", "cooldown_s": 30.0, "schedule": "", "notify": "push",
}


def test_device_with_no_rules_gets_no_rule_command(backend):
    _, client, _ = backend
    assert poll(client) is None


def test_stored_rules_are_delivered_to_the_device(backend):
    app_module, client, _ = backend
    store_rules(app_module, [RULE])

    command = poll(client)
    assert command is not None, "a configured rule must reach the device"
    assert command["set_rules"] == [RULE]
    assert command["rules_version"]


def test_rules_are_not_redelivered_once_acknowledged(backend):
    """Re-sending every poll would restart in-flight rule tracking each time."""
    app_module, client, _ = backend
    store_rules(app_module, [RULE])

    version = poll(client)["rules_version"]
    assert poll(client, rules_version=version) is None


def test_changed_rules_are_delivered_again(backend):
    app_module, client, _ = backend
    store_rules(app_module, [RULE])
    version = poll(client)["rules_version"]

    edited = dict(RULE, persist_s=10.0)
    store_rules(app_module, [edited])

    command = poll(client, rules_version=version)
    assert command is not None
    assert command["set_rules"] == [edited]
    assert command["rules_version"] != version


def test_cleared_rules_are_delivered_as_an_empty_set(backend):
    """Deleting every rule must actively tell the device to stop enforcing."""
    app_module, client, _ = backend
    store_rules(app_module, [RULE])
    version = poll(client)["rules_version"]

    store_rules(app_module, [])
    command = poll(client, rules_version=version)
    assert command is not None
    assert command["set_rules"] == []


def test_malformed_stored_rules_do_not_break_ingest(backend):
    """A device must keep reporting events even if its stored proposal is junk."""
    app_module, client, _ = backend
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET ai_rules = ? WHERE id = 1", ("not json at all",))

    resp = client.post(
        "/v1/ingest",
        json={"events": [{"type": "person", "severity": "log"}], "active_preset": "security_monitoring"},
        headers=DEVICE_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["stored"] == 1


def test_rule_delivery_coexists_with_a_preset_command(backend):
    """Both commands ride the same response; neither may clobber the other."""
    app_module, client, headers = backend
    store_rules(app_module, [RULE])
    client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring"}, headers=headers)

    command = poll(client)
    assert command["set_preset"] == "baby_monitoring"
    assert command["set_rules"] == [RULE]
