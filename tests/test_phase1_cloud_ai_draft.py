"""Previewing an AI configuration must not change what the camera is doing.

/v1/ai/configure is the PREVIEW. AIConfigureView calls it from `configure()`,
the moment the person stops typing — before the proposal has been rendered,
long before they read it, and long before Activate.

It used to write the validated proposal straight into `devices.ai_rules`, which
is the column the ingest handler delivers to the device from. On a device
already in AI Custom the delivery gate is

    custom_monitoring_wanted and version and version != reported_version

and `custom_monitoring_wanted` is written ONLY by set_preset, so a preview
cannot clear it, while `version` is a content hash of ai_rules, which a preview
flips by definition. So typing a new idea into the box and waiting two seconds
was enough to ship, commit and persist a configuration nobody had accepted —
and hot-swap the detection module on a live camera to match it.

The fix separates the two states: a preview writes `ai_rules_draft`, and
set_preset (the app's Activate, and the only act of acceptance in the flow)
promotes the draft into `ai_rules`.
"""

import importlib
import json
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

DEVICE_HEADERS = {"X-Device-Key": "dk_test"}


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    # Pin the AI provider so this suite never depends on whether a local
    # model server happens to be running.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Test unit", "AA-BB-CC", "dk_test", time.time()),
        )
    token = client.post(
        "/v1/signup", json={"email": "draft@example.com", "password": "password1"}
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return app_module, client, headers


def poll(client, rules_version=""):
    """One device heartbeat; returns the command block (possibly None)."""
    resp = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "elderly_monitoring", "rules_version": rules_version},
        headers=DEVICE_HEADERS,
    )
    assert resp.status_code == 200
    return resp.json()["command"]


def stored(app_module, column):
    with app_module.db() as conn:
        return conn.execute(f"SELECT {column} FROM devices WHERE id = 1").fetchone()[column]


def activate(client, headers, preset=""):
    resp = client.post("/v1/devices/1/preset", json={"preset": preset}, headers=headers)
    assert resp.status_code == 200
    return resp


def test_a_preview_does_not_touch_the_running_configuration(backend):
    """THE bug. A device already running AI Custom, converged. The user opens
    Configure-with-AI and types something else. Nothing has been activated —
    so the device must be told nothing, and the stored running rules must be
    byte-for-byte what they were."""
    app_module, client, headers = backend

    # In AI Custom, with a proposal the user actually activated, delivered
    # and acknowledged.
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    activate(client, headers, preset="")
    command = poll(client)
    assert command is not None and "set_rules" in command, "the activated rules must reach the device"
    live_version = command["rules_version"]
    live_rules = stored(app_module, "ai_rules")
    assert poll(client, rules_version=live_version) is None, "converged"

    # The user opens Configure-with-AI again and types a different idea. The
    # preview comes back. They have not tapped anything.
    preview = client.post(
        "/v1/ai/configure",
        json={"device_id": 1, "prompt": "tell me if there is an intruder at the door"},
        headers=headers,
    )
    assert preview.status_code == 200
    assert preview.json()["preset"] == "security_monitoring"  # a genuinely different config

    # The device's next poll must say nothing at all.
    assert poll(client, rules_version=live_version) is None, (
        "previewing a proposal delivered it to the device without anyone activating it"
    )
    # ...and the running configuration is untouched, not merely undelivered.
    assert stored(app_module, "ai_rules") == live_rules


def test_a_preview_is_kept_as_a_draft_so_activate_can_still_apply_it(backend):
    """The preview is not discarded — it is exactly what Activate promotes.
    Without this the fix above would just break Configure-with-AI."""
    app_module, client, headers = backend

    proposal = client.post(
        "/v1/ai/configure", json={"device_id": 1, "prompt": "watch grandpa for falls"},
        headers=headers,
    ).json()

    assert stored(app_module, "ai_rules") is None, "a preview must not be the live configuration"
    draft = stored(app_module, "ai_rules_draft")
    assert draft is not None
    assert json.loads(draft)["preset"] == proposal["preset"] == "elderly_monitoring"


def test_the_app_flow_still_works_end_to_end(backend):
    """Type -> preview -> Activate -> the device receives and runs the rules.
    The whole point of the endpoint, unchanged."""
    app_module, client, headers = backend

    proposal = client.post(
        "/v1/ai/configure", json={"device_id": 1, "prompt": "watch grandpa for falls"},
        headers=headers,
    ).json()
    assert poll(client) is None, "nothing is delivered before Activate"

    activate(client, headers, preset="")

    command = poll(client)
    assert command is not None
    assert command["set_rules"] == proposal["rules"]
    assert command["rules_instruction"] == proposal["explanation"]
    # Promoted, and the draft consumed rather than left lying around.
    assert json.loads(stored(app_module, "ai_rules"))["rules"] == proposal["rules"]
    assert stored(app_module, "ai_rules_draft") is None


def test_an_abandoned_preview_never_reaches_the_device_on_its_own(backend):
    """A proposal someone looked at and walked away from changes nothing.

    Switching to a named preset afterwards must carry only the preset switch
    — the pre-existing contract, covered by test_cloud_backend.py's journey —
    and the running configuration must still be the one that was activated.
    """
    app_module, client, headers = backend

    # An activated, running, acknowledged configuration.
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    activate(client, headers, preset="")
    live_version = poll(client)["rules_version"]
    live_rules = stored(app_module, "ai_rules")

    # A preview the user abandons, then a plain mode switch from the picker.
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch for an intruder"}, headers=headers)
    command = poll(client, rules_version=live_version)
    assert command is None, "an abandoned preview was delivered without being activated"

    activate(client, headers, preset="baby_monitoring")
    command = poll(client, rules_version=live_version)
    assert command == {"set_preset": "baby_monitoring"}, (
        "an abandoned preview rode along with an unrelated preset switch"
    )
    assert stored(app_module, "ai_rules") == live_rules


def test_activating_twice_does_not_re_promote_a_consumed_draft(backend):
    """The draft is consumed by promotion, so `ai_rules` stays the record of
    what is running rather than something a stray second tap can revive."""
    app_module, client, headers = backend

    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    activate(client, headers, preset="")
    live_rules = stored(app_module, "ai_rules")

    activate(client, headers, preset="")
    assert stored(app_module, "ai_rules") == live_rules
    assert stored(app_module, "ai_rules_draft") is None


def test_explain_reads_the_activated_configuration(backend):
    """/explain grounds an alert in the rule it came from. That rule must be
    one the device was actually running — reading a preview would explain an
    event with a configuration that never existed on the device."""
    app_module, client, headers = backend

    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    activate(client, headers, preset="")

    # An event named after the activated rule's own target, which is how
    # jupiter/explain.py's parse_rule_for_event ties an alert back to the
    # rule it came from.
    activated_rule = json.loads(stored(app_module, "ai_rules"))["rules"][0]
    client.post(
        "/v1/ingest",
        json={"events": [{"type": activated_rule["target"], "severity": "loud",
                          "timestamp": "2026-08-29T10:00:00", "source": "custom_rule",
                          "confidence": 0.9, "signals": "person on the floor"}],
              "active_preset": "elderly_monitoring"},
        headers=DEVICE_HEADERS,
    )
    with app_module.db() as conn:
        event_id = conn.execute("SELECT id FROM events ORDER BY id DESC LIMIT 1").fetchone()["id"]

    resp = client.get(f"/v1/devices/1/events/{event_id}/explain", headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # `rule` is populated, i.e. parse_rule_for_event found the ACTIVATED
    # proposal through `ai_rules` -- exactly as it did before the draft
    # column existed. Empty would mean /explain had lost sight of the
    # configuration the alert came from.
    assert body["rule"], body

    # And a proposal that was only ever PREVIEWED is invisible to /explain,
    # which is right: no event can come from a rule the device never ran.
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch for an intruder"}, headers=headers)
    assert client.get(
        f"/v1/devices/1/events/{event_id}/explain", headers=headers
    ).json()["rule"] == body["rule"]


def test_a_preexisting_database_gains_the_draft_column(tmp_path, monkeypatch):
    """The migration path: a database created before ai_rules_draft existed
    must gain it on the next start, not error and not stay broken."""
    db_path = tmp_path / "old.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE devices (id INTEGER PRIMARY KEY, name TEXT, claim_code TEXT,"
        " device_key TEXT, owner_id INTEGER, active_preset TEXT, desired_preset TEXT,"
        " last_seen REAL, created_at REAL NOT NULL);"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("SENSORDYME_DB", str(db_path))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    importlib.import_module("cloud.app")

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(devices)")}
    conn.close()
    assert "ai_rules_draft" in columns
    assert "ai_rules" in columns
