"""End-to-end test of the cloud backend: the full customer journey over the real API."""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")  # backend deps are dev/cloud-only, not installed on the Pi
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    return app_module, TestClient(app_module.app)


def test_health_probe_is_unauthenticated(backend):
    _, client = backend
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["service"] == "sensordyme-cloud"


def test_logout_revokes_only_that_token(backend):
    """Signing out on one device must invalidate that session immediately and
    everywhere -- not just clear the local copy -- and must not touch a
    DIFFERENT session the same account is using elsewhere."""
    _, client = backend
    client.post("/v1/signup", json={"email": "two-devices@example.com", "password": "supersecret1"})

    # Two independent logins -> two independent tokens for the same account
    # (e.g. phone + tablet).
    token_a = client.post(
        "/v1/login", json={"email": "two-devices@example.com", "password": "supersecret1"}
    ).json()["token"]
    token_b = client.post(
        "/v1/login", json={"email": "two-devices@example.com", "password": "supersecret1"}
    ).json()["token"]
    auth_a = {"Authorization": f"Bearer {token_a}"}
    auth_b = {"Authorization": f"Bearer {token_b}"}

    assert client.get("/v1/devices", headers=auth_a).status_code == 200
    assert client.get("/v1/devices", headers=auth_b).status_code == 200

    assert client.post("/v1/logout", headers=auth_a).status_code == 200

    # Device A's session is dead immediately...
    resp = client.get("/v1/devices", headers=auth_a)
    assert resp.status_code == 401
    # ...but device B, a completely different session for the same account,
    # is untouched.
    assert client.get("/v1/devices", headers=auth_b).status_code == 200


def test_logging_out_twice_does_not_error(backend):
    """The same token revoked a second time (a retried request, a double
    tap) must not 500 -- INSERT OR REPLACE, not a unique-constraint crash."""
    _, client = backend
    client.post("/v1/signup", json={"email": "double-logout@example.com", "password": "supersecret1"})
    token = client.post(
        "/v1/login", json={"email": "double-logout@example.com", "password": "supersecret1"}
    ).json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert client.post("/v1/logout", headers=auth).status_code == 200
    # Second call is unauthenticated now (the token it would revoke is
    # already revoked) -- 401, not a crash.
    assert client.post("/v1/logout", headers=auth).status_code == 401


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

    # 6f) preset with a variant flows through the command channel, normalised.
    # A superseded name is stored as the mode it now refers to: the device will
    # apply and report the CURRENT name, so keeping the old one would mean the
    # convergence check never matches and the command re-sends on every poll.
    #
    # The fall rule configured in 6b must NOT ride along here. It was only
    # ever proposed via /v1/ai/configure, never applied — and this switch is
    # to a named preset, not AI Custom. Delivering it anyway was an actual bug
    # found during live-hardware verification: ai_rules is never cleared by a
    # normal preset switch (reopening Configure-with-AI later must still show
    # it), so a version-mismatch-only check can't tell "never applied" apart
    # from "applied, then the user picked a different preset" — either way it
    # kept redelivering the rules and the device kept reapplying them, forcing
    # the device back into full custom monitoring against the preset the user
    # just explicitly chose. custom_monitoring_wanted is what now distinguishes
    # them: it is only true right after an empty-preset ("AI Custom") request.
    client.post(f"/v1/devices/{device_id}/preset", json={"preset": "driver_distraction:teen_driver"}, headers=auth)
    r = client.post("/v1/ingest", json={"events": [], "active_preset": "elderly_monitoring"}, headers=device_headers)
    command = r.json()["command"]
    assert command == {"set_preset": "driver_safety:teen_driver"}, (
        "an unapplied AI proposal must never ride along on an unrelated preset switch"
    )

    # Once acknowledged, nothing remains outstanding.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety:teen_driver"},
        headers=device_headers,
    )
    assert r.json()["command"] is None

    # Explicitly applying AI Custom (preset "") is what actually delivers the
    # configured rules — this is what AIConfigureView's "Apply" does for a
    # proposal with no named preset.
    assert client.post(f"/v1/devices/{device_id}/preset", json={"preset": ""}, headers=auth).status_code == 200
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety:teen_driver"},
        headers=device_headers,
    )
    command = r.json()["command"]
    # (target, predicate), which is the shape core/rules.py executes. This
    # used to read `rule["detection"] == "fall"` -- an EVENT TYPE, which no
    # detector emits as a label, so the rule this journey delivered was one
    # the device could never fire. See tests/test_phase1_cloud_ai_fallback.py.
    assert [(rule["target"], rule["predicate"]) for rule in command["set_rules"]] == [
        ("on_floor", "present")
    ]

    # 6g) and it CONVERGES: once the device reports the mode as active, the
    # command is cleared rather than re-sent for ever. Verified against the
    # superseded name too, because rows written before the rename still hold it
    # and a device reporting "driver_safety" must satisfy them.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety:teen_driver",
              "rules_version": command["rules_version"]},
        headers=device_headers,
    )
    # An empty/absent command is the success case: nothing left outstanding.
    assert "set_preset" not in (r.json().get("command") or {}), \
        "the preset command re-sent after the device already reported that mode"

    # 7) another user cannot see or control this device
    r2 = client.post("/v1/signup", json={"email": "stranger@example.com", "password": "supersecret2"})
    stranger = {"Authorization": f"Bearer {r2.json()['token']}"}
    assert client.get(f"/v1/devices/{device_id}/events", headers=stranger).status_code == 404
    assert client.post(f"/v1/devices/{device_id}/preset", json={"preset": "x"}, headers=stranger).status_code == 404
    assert client.post("/v1/devices/claim", json={"claim_code": "AB-CD-EF"}, headers=stranger).status_code == 409


def _claim_a_device(app_module, client):
    """Provision, sign up and claim one device — the setup every mode test needs."""
    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Test unit", "ZZ-99-XX", "dk_modetest", _time.time()),
        )
    r = client.post("/v1/signup", json={"email": "mode-test@example.com", "password": "supersecret1"})
    auth = {"Authorization": f"Bearer {r.json()['token']}"}
    device_id = client.post("/v1/devices/claim", json={"claim_code": "ZZ-99-XX"}, headers=auth).json()["device_id"]
    return auth, {"X-Device-Key": "dk_modetest"}, device_id


def test_preset_command_survives_matching_name_while_custom_monitoring_active(backend):
    """The bug this reproduces: AI Custom Monitoring layers on top of a preset
    without changing active_preset, so "go back to the preset you were already
    on" is a request whose desired name equals the device's current
    active_preset from the very first poll. String-equality convergence reads
    that as "already done" and never delivers the command, so a user who turns
    on custom monitoring while already on Driver Safety can never tap Driver
    Safety again to leave it — the exact symptom reported: the mode row looks
    unresponsive.

    The device's `monitoring.active` flag is what distinguishes the two cases
    convergence cannot tell apart from active_preset alone, and it rides on
    every ingest call in `body.monitoring`.
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    # Device starts on driver_safety, then the user turns on custom monitoring —
    # active_preset never changes; only monitoring.active flips on.
    client.post("/v1/ingest", json={"events": [], "active_preset": "driver_safety"}, headers=device_headers)
    client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety",
              "monitoring": {"active": True, "instruction": "watch for phones"}},
        headers=device_headers,
    )

    # User taps "Driver Safety" again to leave custom monitoring — the desired
    # name is identical to what the device already reports.
    assert client.post(
        f"/v1/devices/{device_id}/preset", json={"preset": "driver_safety"}, headers=auth
    ).status_code == 200

    # The device's next poll still shows the OLD state (it hasn't acted on the
    # command yet) — active_preset unchanged, monitoring still active. The
    # command must still be delivered: nothing about this state means "done".
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety",
              "monitoring": {"active": True, "instruction": "watch for phones"}},
        headers=device_headers,
    )
    assert r.json()["command"] == {"set_preset": "driver_safety"}, (
        "the command was dropped because the name already matched active_preset, "
        "even though the device still reports custom monitoring active"
    )

    # Once the device actually exits custom monitoring and reports it, THEN it
    # converges — not a moment before.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety",
              "monitoring": {"active": False}},
        headers=device_headers,
    )
    assert "set_preset" not in (r.json().get("command") or {})


def test_preset_command_to_a_different_mode_still_works_during_custom_monitoring(backend):
    """The ordinary case — switching to a genuinely different preset while
    custom monitoring is active — must keep working exactly as before."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety",
              "monitoring": {"active": True, "instruction": "watch for phones"}},
        headers=device_headers,
    )
    client.post(f"/v1/devices/{device_id}/preset", json={"preset": "baby_monitoring"}, headers=auth)
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety",
              "monitoring": {"active": True, "instruction": "watch for phones"}},
        headers=device_headers,
    )
    assert r.json()["command"] == {"set_preset": "baby_monitoring"}


def test_leaving_ai_custom_does_not_resurrect_itself_next_poll(backend):
    """The bug found during live hardware verification of the AI-Custom fix:
    ai_rules is never cleared by switching to a normal preset (reopening
    Configure-with-AI later must still show what was last configured), so a
    naive rules_version comparison alone cannot tell "not delivered yet" apart
    from "the user picked a different preset a while ago" — it kept resending
    the same set_rules command forever, and the device kept reapplying it,
    pulling itself back into custom monitoring within one poll of every single
    switch away. Reproduced directly against the live device before this test
    existed: `monitoring.active` flipped back to true within one poll cycle of
    confirming it had gone false. custom_monitoring_wanted is the fix — it
    tracks intent, separately from the ai_rules content itself.
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    # Configure AI Custom rules (a prompt matching no preset keyword -> "").
    r = client.post(
        "/v1/ai/configure",
        json={"device_id": device_id, "prompt": "tell me if someone is in the kitchen after dark"},
        headers=auth,
    )
    assert r.json()["preset"] == ""

    # Apply it — this is what AIConfigureView's "Apply" actually does.
    assert client.post(
        f"/v1/devices/{device_id}/preset", json={"preset": ""}, headers=auth
    ).status_code == 200

    # Device picks up the rules and reports custom monitoring active.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety"},
        headers=device_headers,
    )
    rules_version = r.json()["command"]["rules_version"]
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety", "rules_version": rules_version,
              "monitoring": {"active": True, "instruction": "kitchen watch"}},
        headers=device_headers,
    )
    assert r.json().get("command") is None, "converged: rules applied, nothing left to deliver"

    # User switches to a normal preset.
    assert client.post(
        f"/v1/devices/{device_id}/preset", json={"preset": "driver_safety"}, headers=auth
    ).status_code == 200

    # Device exits custom monitoring and reports it — the already-fixed half.
    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety", "rules_version": rules_version,
              "monitoring": {"active": False}},
        headers=device_headers,
    )
    assert r.json().get("command") is None, "leaving custom monitoring must not itself trigger a command"

    # The bug: several more polls after that, still reporting the OLD
    # rules_version (the device has no reason to have changed it — nobody
    # told it to). None of them may resend set_rules — that would silently
    # re-arm custom monitoring the user just turned off.
    for _ in range(5):
        r = client.post(
            "/v1/ingest",
            json={"events": [], "active_preset": "driver_safety", "rules_version": rules_version,
                  "monitoring": {"active": False}},
            headers=device_headers,
        )
        command = r.json().get("command") or {}
        assert "set_rules" not in command, (
            "ai_rules was resent after the user left AI Custom — this is exactly what pulled "
            "the device back into custom monitoring within one poll interval on real hardware"
        )


def test_reapplying_ai_custom_after_leaving_it_still_works(backend):
    """The other half: custom_monitoring_wanted must not get stuck off. Turning
    AI Custom back on after leaving it must still deliver the rules."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    client.post(
        "/v1/ai/configure",
        json={"device_id": device_id, "prompt": "tell me if someone is in the kitchen after dark"},
        headers=auth,
    )
    client.post(f"/v1/devices/{device_id}/preset", json={"preset": ""}, headers=auth)
    client.post(f"/v1/devices/{device_id}/preset", json={"preset": "driver_safety"}, headers=auth)

    # Back to AI Custom.
    assert client.post(
        f"/v1/devices/{device_id}/preset", json={"preset": ""}, headers=auth
    ).status_code == 200

    r = client.post(
        "/v1/ingest",
        json={"events": [], "active_preset": "driver_safety"},
        headers=device_headers,
    )
    assert "set_rules" in (r.json().get("command") or {}), (
        "re-selecting AI Custom after having left it must still deliver the configured rules"
    )


def test_ai_configure_response_carries_interpretation_confidence(monkeypatch, tmp_path):
    """jupiter/confidence.py's score must actually reach the app through
    /v1/ai/configure, nested under `jupiter` alongside provider/model — this
    is the API contract ios/Sources/BackendAPI.swift's AIJupiterMeta decodes.

    Uses the `scripted` provider rather than the `backend` fixture's pinned
    "unavailable" (which never calls _ai_configure_llm at all, so the
    `jupiter` key would never appear) and rather than the real Ollama model
    (keeping this test fast and offline, unlike the live checks already run
    manually against qwen2.5:7b-instruct)."""
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "scripted")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    auth, _, device_id = _claim_a_device(app_module, client)
    r = client.post(
        "/v1/ai/configure", json={"device_id": device_id, "prompt": "alert me if a person appears"},
        headers=auth,
    )
    assert r.status_code == 200
    jupiter_meta = r.json()["jupiter"]
    assert jupiter_meta["provider"] == "scripted"
    ic = jupiter_meta["interpretation_confidence"]
    assert isinstance(ic["score"], (int, float))
    assert 0.0 <= ic["score"] <= 1.0
    assert isinstance(ic["basis"], list)


def test_a_notification_never_fires_for_an_event_that_was_not_stored(backend, monkeypatch):
    """The invariant: the notification gate must never be asked to evaluate an
    event that was not actually written to the database — "an alert was sent
    without being recorded" is the complaint this guards against.

    This used to be enforced by capping both the INSERT loop and the
    notification-evaluation loop at the same 200-event slice of a batch. That
    cap was itself a separate bug (see test_a_large_backlog_batch_is_not_
    silently_truncated in test_cloud_backend.py): a device flushing a backlog
    of more than 200 events after an outage had everything past the 200th
    silently discarded while `stored` claimed the full count arrived. Now
    there is no cap — the invariant instead holds because the notification
    loop iterates `newly_inserted_events` (rows that actually landed),
    never `body.events` (what was merely in the request). Verified two ways:
    the row count actually stored, and — more directly — that the
    notification gate is called exactly once per stored row, by
    monkeypatching cloud.notify.gate.evaluate to record every call.
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    evaluated_severities = []
    real_evaluate = app_module.gate.evaluate

    def spying_evaluate(user_id, severity, source):
        evaluated_severities.append(severity)
        return real_evaluate(user_id, severity, source=source)

    monkeypatch.setattr(app_module.gate, "evaluate", spying_evaluate)

    # 250 loud events in one batch — over the OLD 200 cap, and every one
    # severe enough that, if evaluated, it would be a candidate for a push.
    events = [
        {"timestamp": f"2026-08-08T10:00:{i:02d}+00:00", "type": "test_event",
         "severity": "loud", "signals": f"event-{i}"}
        for i in range(250)
    ]
    r = client.post("/v1/ingest", json={"events": events}, headers=device_headers)
    assert r.status_code == 200

    stored = client.get(f"/v1/devices/{device_id}/events", params={"limit": 1000}, headers=auth).json()["events"]
    assert len(stored) == 250, "every event in the batch must now be stored -- no silent cap"
    assert len(evaluated_severities) == 250, (
        f"the notification gate evaluated {len(evaluated_severities)} events but "
        f"{len(stored)} were stored — every stored event must be evaluated exactly once, "
        f"and nothing must be evaluated that wasn't stored"
    )


def test_resending_the_same_event_key_does_not_create_a_duplicate(backend):
    """The bug: core/uplink.py treats a lost HTTP response as a failed send and
    re-queues the whole batch for the next poll — but the server may have
    already inserted those events before the response was lost. Without an
    idempotency key, the resend becomes a second row: a duplicate alert in
    Recent Activity, and (per the notification-gate test above) a second
    notification for something that already fired once.

    Simulates exactly that: the same event, with the same event_key, arrives
    in two separate /v1/ingest POSTs (a real retry spans two HTTP requests,
    not two entries in one batch)."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    event = {
        "timestamp": "2026-08-08T12:00:00+00:00", "type": "fall",
        "severity": "loud", "signals": "rapid drop", "event_key": "retry-test-key-abc123",
    }
    r1 = client.post("/v1/ingest", json={"events": [event]}, headers=device_headers)
    assert r1.status_code == 200
    r2 = client.post("/v1/ingest", json={"events": [event]}, headers=device_headers)  # the retry
    assert r2.status_code == 200

    stored = client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]
    assert len(stored) == 1, f"expected exactly one stored row for one event_key, got {len(stored)}"


def test_two_genuinely_different_events_with_no_key_both_still_store(backend):
    """The other half of the invariant: firmware that predates event_key (or
    any event missing one) must not be deduplicated against unrelated events —
    NULL must never collide with NULL in the partial unique index."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    events = [
        {"timestamp": "2026-08-08T12:00:00+00:00", "type": "fall", "severity": "loud"},
        {"timestamp": "2026-08-08T12:00:01+00:00", "type": "fall", "severity": "loud"},
    ]
    r = client.post("/v1/ingest", json={"events": events}, headers=device_headers)
    assert r.status_code == 200

    stored = client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]
    assert len(stored) == 2, "two distinct keyless events must not be deduplicated against each other"


def test_the_worked_example_end_to_end_through_the_real_ingest_endpoint(backend, monkeypatch):
    """The exact sequence the product spec gives, driven through the real API
    rather than the gate directly: 10:00:00 alert A notifies; :05 B suppressed
    (still stored); :10 C suppressed (still stored); :15 D suppressed (still
    stored); :20+ E notifies again. A retry of A (same event_key) notifies
    zero additional times regardless of timing — that suppression is identity
    based (event_key -> OR IGNORE -> never reaches the notification loop at
    all) and stacks with, rather than substitutes for, the 20s cooldown.
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    pushed = []
    monkeypatch.setattr(app_module, "send_push", lambda user, title, body: pushed.append((title, body)))
    monkeypatch.setattr(app_module, "send_sms", lambda user, body: None)

    import cloud.notify as notify_module

    # The `gate` singleton is module-level and outlives any one test's
    # `backend` fixture (the fixture reloads cloud.app but not cloud.notify),
    # so a real-wall-clock timestamp left over from an earlier test could
    # make this test's injected, much-smaller `now` look like it's already
    # "inside" a cooldown that never actually happened for this account.
    notify_module.gate.reset()
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(notify_module.time, "time", lambda: clock["t"])

    def post(event_key, signals, ts="2026-08-08T12:00:00+00:00"):
        r = client.post(
            "/v1/ingest",
            json={"events": [{
                "timestamp": ts, "type": "fall", "severity": "loud",
                "signals": signals, "event_key": event_key,
            }]},
            headers=device_headers,
        )
        assert r.status_code == 200

    post("alert-a", "alert A")                              # 10:00:00
    assert len(pushed) == 1, "alert A must notify"

    post("alert-a", "alert A")                              # a retry, same instant
    assert len(pushed) == 1, "a retry of the SAME event must not notify a second time"

    clock["t"] += 5
    post("alert-b", "alert B")                              # 10:00:05
    assert len(pushed) == 1, "alert B, inside the 20s cooldown, must be suppressed"

    clock["t"] += 5
    post("alert-c", "alert C")                              # 10:00:10
    assert len(pushed) == 1, "alert C, still inside the cooldown, must also be suppressed"

    clock["t"] += 5
    post("alert-d", "alert D")                              # 10:00:15
    assert len(pushed) == 1, "alert D, still inside the cooldown, must also be suppressed"

    clock["t"] += 5
    post("alert-e", "alert E")                              # 10:00:20
    assert len(pushed) == 2, "alert E, at the cooldown boundary, must notify again"

    stored = client.get(f"/v1/devices/{device_id}/events", params={"limit": 100}, headers=auth).json()["events"]
    assert len(stored) == 5, (
        "every qualifying alert (A-E) must be stored and visible in Recent Activity, "
        "whether or not the cooldown suppressed its notification — the retry of A is the "
        "only one of the six posts that does not add a new row"
    )


def test_a_notification_failure_does_not_affect_event_storage(backend, monkeypatch):
    """Recent Activity must stay independent of the notification pipeline: if
    sending a push blows up, the event is already committed and must still be
    there and still show up in the timeline."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    def boom(user, title, body):
        raise RuntimeError("simulated push provider outage")

    monkeypatch.setattr(app_module, "send_push", boom)
    monkeypatch.setattr(app_module, "send_sms", lambda user, body: None)

    r = client.post(
        "/v1/ingest",
        json={"events": [{
            "timestamp": "2026-08-08T12:00:00+00:00", "type": "fall",
            "severity": "loud", "signals": "rapid drop", "event_key": "storage-independence-key",
        }]},
        headers=device_headers,
    )
    assert r.status_code == 200, "a notification-delivery failure must not fail the ingest request"

    stored = client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]
    assert len(stored) == 1, "the event must still be stored even though notifying about it failed"


def test_send_with_retry_recovers_from_a_transient_failure(backend, monkeypatch):
    """The bug this exists to fix: a blip at the exact moment a genuinely new
    critical alert lands used to lose that notification permanently, with
    zero visibility. A transient failure must not need a second real-world
    event to arrive before the user hears about the first one."""
    app_module, _ = backend
    monkeypatch.setattr(app_module.time, "sleep", lambda s: None)  # don't actually wait in tests

    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("simulated transient provider error")

    app_module._send_with_retry(flaky)
    assert len(attempts) == 3, "must keep retrying through transient failures, not give up on the first"


def test_send_with_retry_gives_up_and_raises_after_exhausting_attempts(backend, monkeypatch):
    """A GENUINE outage (not a blip) must still surface as a real failure to
    the caller -- retrying forever, or swallowing the final failure, would
    just move "silently lost" from the first attempt to the last one."""
    app_module, _ = backend
    monkeypatch.setattr(app_module.time, "sleep", lambda s: None)

    attempts = []

    def always_fails():
        attempts.append(1)
        raise RuntimeError("simulated persistent provider outage")

    with pytest.raises(RuntimeError, match="persistent provider outage"):
        app_module._send_with_retry(always_fails)
    assert len(attempts) == app_module.NOTIFY_SEND_ATTEMPTS


def test_send_push_now_detects_a_bad_http_status_instead_of_treating_it_as_success(backend, monkeypatch):
    """Before this pass, send_push never called raise_for_status() -- an ntfy
    500 or a rejected topic looked EXACTLY like success everywhere in the
    system: no exception, no log line, no retry. This is the regression test
    for that specific gap, independent of the retry mechanism itself."""
    import requests as real_requests

    app_module, _ = backend
    monkeypatch.setattr(app_module.time, "sleep", lambda s: None)

    class _FailingResponse:
        def raise_for_status(self):
            raise real_requests.HTTPError("500 Server Error")

    monkeypatch.setattr(real_requests, "post", lambda *a, **kw: _FailingResponse())

    with pytest.raises(real_requests.HTTPError):
        app_module.send_push({"ntfy_topic": "some-topic"}, "title", "body")


def test_notify_logs_each_channel_independently_after_retries_are_exhausted(backend, monkeypatch, caplog):
    """A push failure must not cancel the SMS attempt, and either failure
    must be visible in the logs -- not silently absorbed the way both were
    before this pass."""
    import logging as _logging

    import cloud.notify as notify_module

    app_module, client = backend
    _, device_headers, _ = _claim_a_device(app_module, client)
    # _claim_a_device always uses the same account, and the 20s per-account
    # cooldown gate is process-wide state that outlives any one test -- a
    # test that ran moments ago in the same process can otherwise suppress
    # this one's notification before it's ever attempted, independent of
    # anything this test is actually checking.
    notify_module.gate.reset()
    monkeypatch.setattr(app_module.time, "sleep", lambda s: None)

    push_calls, sms_calls = [], []

    def failing_push(user, title, body):
        push_calls.append(1)
        raise RuntimeError("push provider down")

    def failing_sms(user, body):
        sms_calls.append(1)
        raise RuntimeError("sms provider down")

    monkeypatch.setattr(app_module, "send_push", failing_push)
    monkeypatch.setattr(app_module, "send_sms", failing_sms)

    with caplog.at_level(_logging.WARNING, logger="uvicorn.error"):
        client.post(
            "/v1/ingest",
            json={"events": [{
                "timestamp": "2026-08-08T13:00:00+00:00", "type": "fall",
                "severity": "loud", "signals": "rapid drop", "event_key": "both-channels-fail",
            }]},
            headers=device_headers,
        )

    assert push_calls and sms_calls, "both channels must actually be attempted"
    messages = [r.message for r in caplog.records]
    assert any("push delivery failed" in m for m in messages)
    assert any("sms delivery failed" in m for m in messages)


def test_one_malformed_event_does_not_sink_the_whole_batch(backend, caplog):
    """Discovered during the production QA pass, reproduced against the live
    code before any fix existed: `events: list[dict]` is unvalidated by
    Pydantic beyond "is a dict" — a malformed `confidence` (a string instead
    of a number) reached `float(...)` uncaught, raised inside the `with db()`
    block BEFORE conn.commit() ran, and took the whole transaction down with
    it — losing every good event in the same batch, not just the bad one.

    Worse than data loss on its own: the resulting 500 is exactly what
    core/uplink.py's retry logic treats as a failed send, so it re-queues the
    identical batch — and the same malformed event 500s again, forever,
    permanently blocking every event queued behind it. This test is the
    single highest-value regression test added this pass: a bad payload from
    a buggy module or a corrupted queue entry must degrade to "one event
    lost, logged" — never "the ingest pipeline wedges."
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    events = [
        {"timestamp": "t1", "type": "fall", "severity": "loud", "confidence": 0.9},
        {"timestamp": "t2", "type": "fall", "severity": "loud", "confidence": "not-a-number-at-all"},
        {"timestamp": "t3", "type": "fall", "severity": "loud", "confidence": 0.8},
    ]
    r = client.post("/v1/ingest", json={"events": events}, headers=device_headers)
    assert r.status_code == 200, "a malformed event in the batch must not 500 the whole request"

    stored = client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]
    stored_timestamps = {s["timestamp"] for s in stored}
    assert stored_timestamps == {"t1", "t3"}, (
        f"the two well-formed events must still be stored even though a third was malformed, "
        f"got: {stored_timestamps}"
    )

    # And the queue does not deadlock: retrying (what the device's uplink
    # actually does after a failed send) must not 500 a second time either.
    r2 = client.post("/v1/ingest", json={"events": events}, headers=device_headers)
    assert r2.status_code == 200, "a retry of the same malformed batch must not 500 either"


def test_a_malformed_event_is_logged_not_silently_dropped(backend, caplog):
    """Silently losing one event is an acceptable last resort; silently losing
    it with no trace anywhere is not — an operator needs to be able to answer
    "why is this device missing an event" for real."""
    import logging as _logging

    app_module, client = backend
    _, device_headers, _ = _claim_a_device(app_module, client)

    with caplog.at_level(_logging.WARNING, logger="uvicorn.error"):
        client.post(
            "/v1/ingest",
            json={"events": [{"timestamp": "t1", "type": "fall", "confidence": "garbage"}]},
            headers=device_headers,
        )
    assert any("malformed" in record.message for record in caplog.records)


def test_a_large_backlog_batch_is_not_silently_truncated(backend):
    """The bug: ingest() only processed `body.events[:200]` while reporting
    `stored` as the FULL request size regardless -- and core/uplink.py clears
    its entire local queue (up to `queue_max`, 500) the moment it gets ANY 200
    response, never re-queuing what was dropped. A device reconnecting after
    an outage long enough to queue more than 200 events would silently lose
    every one past the 200th, with both sides believing the upload succeeded.
    This sends more than the old cap to prove nothing is thrown away, and that
    `stored` reflects what is actually in the database, not the request size.
    """
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    batch_size = 250  # over the old 200-event cap
    events = [
        {"timestamp": f"t{i}", "type": "fall", "severity": "loud", "confidence": 0.9,
         "event_key": f"key-{i}"}
        for i in range(batch_size)
    ]
    r = client.post("/v1/ingest", json={"events": events}, headers=device_headers)
    assert r.status_code == 200
    assert r.json()["stored"] == batch_size, (
        "every event in a backlog batch over the old 200-event cap must be stored, "
        "and `stored` must reflect that -- not silently truncate"
    )

    stored = client.get(f"/v1/devices/{device_id}/events?limit=1000", headers=auth).json()["events"]
    assert len({s["timestamp"] for s in stored}) == batch_size, (
        "every event must actually be a row in the database, not just claimed as stored"
    )


def test_stored_count_reflects_actual_inserts_not_request_size(backend):
    """`stored` must count what landed in the database. A retried event_key
    (already stored, correctly deduplicated) and a malformed event (correctly
    skipped) are both real, intentional non-inserts -- neither should be
    reported as "stored", even though both were present in the request."""
    app_module, client = backend
    _, device_headers, _ = _claim_a_device(app_module, client)

    first = client.post(
        "/v1/ingest",
        json={"events": [{"timestamp": "t1", "type": "fall", "event_key": "dup-key"}]},
        headers=device_headers,
    )
    assert first.json()["stored"] == 1

    retry_plus_bad = client.post(
        "/v1/ingest",
        json={"events": [
            {"timestamp": "t1", "type": "fall", "event_key": "dup-key"},  # already stored
            {"timestamp": "t2", "type": "fall", "confidence": "not-a-number"},  # malformed
            {"timestamp": "t3", "type": "fall", "event_key": "fresh-key"},  # genuinely new
        ]},
        headers=device_headers,
    )
    assert retry_plus_bad.json()["stored"] == 1, (
        "of 3 events in the request, only 1 is a genuinely new row -- `stored` must say 1, not 3"
    )


def test_repeated_wrong_logins_are_rate_limited(backend):
    """Nothing else stands between /v1/login and an online password-guessing
    script but this. 10 attempts/5min per IP (cloud/ratelimit.py); the 11th
    within the window must be refused before it even checks the password,
    not just fail auth the normal way."""
    _, client = backend
    client.post("/v1/signup", json={"email": "target@example.com", "password": "correct-horse-1"})

    for _ in range(10):
        r = client.post("/v1/login", json={"email": "target@example.com", "password": "wrong"})
        assert r.status_code == 401

    r = client.post("/v1/login", json={"email": "target@example.com", "password": "wrong"})
    assert r.status_code == 429, "the 11th attempt in the window must be throttled, not evaluated"

    # And the real password doesn't get through the throttle either -- this
    # is a rate limit on the endpoint, not a smarter "let good guesses past".
    r = client.post("/v1/login", json={"email": "target@example.com", "password": "correct-horse-1"})
    assert r.status_code == 429


def test_rate_limit_is_per_ip_not_global(backend):
    """A limiter keyed globally instead of per-caller would let one abusive
    client lock out every legitimate one sharing the process -- worse than
    the bug it exists to prevent. Verified directly against the class, since
    TestClient itself always presents as one fixed peer address."""
    from cloud.ratelimit import RateLimiter

    limiter = RateLimiter(max_attempts=3, window_s=60.0)
    for _ in range(3):
        assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    # A different caller has its own, untouched budget.
    assert limiter.allow("5.6.7.8")


def test_rate_limit_window_expires(backend):
    from cloud.ratelimit import RateLimiter

    t = {"now": 0.0}
    limiter = RateLimiter(max_attempts=2, window_s=10.0)
    assert limiter.allow("1.2.3.4", now=t["now"])
    assert limiter.allow("1.2.3.4", now=t["now"])
    assert not limiter.allow("1.2.3.4", now=t["now"])

    t["now"] = 11.0  # past the window
    assert limiter.allow("1.2.3.4", now=t["now"]), "an expired attempt must age out of the window"


# ------------------------------------------------------------------ evidence-based alert text
def test_readable_title_maps_known_types_and_falls_back_for_unknown(backend):
    app_module, _ = backend
    assert app_module._readable_title("fall") == "🔴 Potential Fall Detected"
    assert app_module._readable_title("something_new") == "SensorDyme: something_new"


def test_notification_detail_prefers_signals_over_duration(backend):
    """A module's own signals text (e.g. occupancy's "zone=pool confirmed
    after 12.4s") is already better evidence than the generic duration
    fallback and must win when both are present."""
    app_module, _ = backend
    detail = app_module._notification_detail(
        {"type": "fall", "signals": "zone=pool confirmed after 12.4s", "duration": 34.0}
    )
    assert detail == "zone=pool confirmed after 12.4s"


def test_notification_detail_falls_back_to_duration_when_signals_empty(backend):
    """The worked example from the product spec: fall_detection's "fall"
    event carries no signals text, but Event.duration (how long the person
    was down) is real, already-computed evidence."""
    app_module, _ = backend
    detail = app_module._notification_detail({"type": "fall", "signals": "", "duration": 34.0})
    assert detail == "Person transitioned rapidly and remained down for 34 seconds"


def test_notification_detail_uses_generic_template_for_an_unlisted_type_with_duration(backend):
    app_module, _ = backend
    detail = app_module._notification_detail({"type": "something_new", "signals": "", "duration": 12.0})
    assert detail == "SensorDyme: something_new — sustained for 12 seconds"


def test_notification_detail_is_bare_alert_when_neither_signals_nor_duration_available(backend):
    """No evidence to show beats inventing some -- the old, honest fallback."""
    app_module, _ = backend
    assert app_module._notification_detail({"type": "fall", "signals": ""}) == "alert"
    assert app_module._notification_detail({"type": "fall", "signals": "", "duration": 0.0}) == "alert"


def test_a_fall_event_with_no_signals_produces_evidence_based_notification_text(backend, monkeypatch):
    """End to end through the real ingest endpoint: a fall event with the
    same shape core/uplink.py now actually sends (signals empty, duration
    real) must produce a push notification with a real title and evidence,
    not "SensorDyme: fall" / "alert"."""
    app_module, client = backend
    auth, device_headers, device_id = _claim_a_device(app_module, client)

    pushed = []
    monkeypatch.setattr(app_module, "send_push", lambda user, title, body: pushed.append((title, body)))
    monkeypatch.setattr(app_module, "send_sms", lambda user, body: None)

    import cloud.notify as notify_module

    notify_module.gate.reset()  # process-wide singleton; see the worked-example test above

    r = client.post(
        "/v1/ingest",
        json={"events": [{
            "timestamp": "2026-08-08T12:00:00+00:00", "type": "fall", "severity": "loud",
            "signals": "", "duration": 34.0, "event_key": "fall-1",
        }]},
        headers=device_headers,
    )
    assert r.status_code == 200
    assert len(pushed) == 1
    title, body = pushed[0]
    assert title == "🔴 Potential Fall Detected"
    assert body == "Person transitioned rapidly and remained down for 34 seconds"

    stored = client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]
    assert stored[0]["duration"] == 34.0, "duration must reach the timeline, not just the notification"


# --------------------------------------------------- stored state as untrusted input
class TestMalformedStoredStateDoesNotBreakTheDeviceList:
    """`GET /v1/devices` must survive an unreadable `monitoring` column.

    Every JSON-column read in cloud/app.py is wrapped except one, and the one was
    on the endpoint the app polls continuously. A single unreadable value — a
    legacy row, a partially-written record, a hand-edited database — made the
    device list a hard 500 for that account, permanently, with no route back
    except editing the database. The app's whole home screen is that endpoint.

    The value comes out of our own table rather than off the wire, which is
    exactly why it read as trustworthy and was not guarded.
    """

    def _claimed_device(self, client):
        client.post("/v1/signup", json={"email": "stored@example.com", "password": "supersecret1"})
        token = client.post(
            "/v1/login", json={"email": "stored@example.com", "password": "supersecret1"}
        ).json()["token"]
        return token

    def test_unreadable_monitoring_reports_none_rather_than_500(self, backend):
        app_module, client = backend
        token = self._claimed_device(client)
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at, monitoring)"
                " VALUES ('cam', 'C1', 'dk_1', 1, 0, ?)",
                ("{this is not json",),
            )

        resp = client.get("/v1/devices", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["devices"][0]["monitoring"] is None

    def test_monitoring_that_is_valid_json_but_not_an_object_is_also_none(self, backend):
        """`json.loads` succeeds on `"[]"` and `"7"`. The app expects an object,
        so passing either through would move the failure into the client."""
        app_module, client = backend
        token = self._claimed_device(client)
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at, monitoring)"
                " VALUES ('cam', 'C2', 'dk_2', 1, 0, ?)",
                ("[1, 2, 3]",),
            )

        resp = client.get("/v1/devices", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200
        assert resp.json()["devices"][0]["monitoring"] is None

    def test_a_good_monitoring_config_still_comes_through(self, backend):
        """The guard must not quietly swallow valid config."""
        app_module, client = backend
        token = self._claimed_device(client)
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at, monitoring)"
                " VALUES ('cam', 'C3', 'dk_3', 1, 0, ?)",
                ('{"mode": "baby_monitoring"}',),
            )

        resp = client.get("/v1/devices", headers={"Authorization": f"Bearer {token}"})

        assert resp.json()["devices"][0]["monitoring"] == {"mode": "baby_monitoring"}

    def test_one_bad_row_does_not_hide_the_other_devices(self, backend):
        """The failure mode that mattered: one unreadable row took the whole
        account's device list with it."""
        app_module, client = backend
        token = self._claimed_device(client)
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at, monitoring)"
                " VALUES ('broken', 'C4', 'dk_4', 1, 0, ?)", ("{nope",),
            )
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at, monitoring)"
                " VALUES ('fine', 'C5', 'dk_5', 1, 0, ?)", ('{"mode": "security_monitoring"}',),
            )

        resp = client.get("/v1/devices", headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 200
        names = {d["name"]: d["monitoring"] for d in resp.json()["devices"]}
        assert names["broken"] is None
        assert names["fine"] == {"mode": "security_monitoring"}


class TestWriteContentionWaitsRatherThanFailing:
    """A blocked writer must wait, not 500.

    WAL serialises writes, and `/v1/ingest` writes on every device poll. If a
    write gives up, the uncaught `sqlite3.OperationalError: database is locked`
    becomes a 500 — and core/uplink.py treats any non-200 as a failed send and
    re-queues the identical batch, so the endpoint most likely to meet contention
    is the one whose client retries hardest.
    """

    def test_a_busy_timeout_is_configured_on_every_connection(self, backend):
        app_module, _ = backend
        with app_module.db() as conn:
            busy_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_ms > 0, "a zero busy timeout fails immediately on any contention"
        assert busy_ms == int(app_module.BUSY_TIMEOUT_S * 1000)

    def test_the_timeout_is_long_enough_to_outlast_a_device_poll(self, backend):
        """The device polls every ~2s; a budget shorter than that turns routine
        overlap into a failed batch."""
        app_module, _ = backend
        assert app_module.BUSY_TIMEOUT_S >= 5.0

    def test_a_write_held_open_elsewhere_does_not_raise_immediately(self, backend):
        """With a busy timeout the second writer blocks and then succeeds once the
        first commits, instead of raising at once."""
        import threading

        app_module, _ = backend
        holder_has_lock = threading.Event()
        release_holder = threading.Event()

        def hold_the_write_lock():
            conn = app_module._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO devices (name, claim_code, device_key, created_at)"
                    " VALUES ('holder', 'H1', 'dk_h1', 0)"
                )
                holder_has_lock.set()
                release_holder.wait(10)
                conn.commit()
            finally:
                conn.close()

        holder = threading.Thread(target=hold_the_write_lock)
        holder.start()
        try:
            assert holder_has_lock.wait(5), "could not acquire the first write lock"
            # Let the holder go shortly; the point is that this write waits for it
            # rather than failing the instant it finds the database busy.
            threading.Timer(0.3, release_holder.set).start()
            with app_module.db() as conn:
                conn.execute(
                    "INSERT INTO devices (name, claim_code, device_key, created_at)"
                    " VALUES ('waiter', 'W1', 'dk_w1', 0)"
                )
        finally:
            release_holder.set()
            holder.join(10)

        with app_module.db() as conn:
            names = {r["name"] for r in conn.execute("SELECT name FROM devices")}
        assert {"holder", "waiter"} <= names


class TestAIConfigureWritesOnlyToADeviceYouOwn:
    """The draft write must be scoped to the owner, not just to the device id.

    `/v1/ai/configure` checks ownership, then makes a model call that takes
    seconds, then writes. Those were three separate steps with the ownership fact
    only established before the slow one, and the write said `WHERE id = ?`. A
    device re-claimed during the model call therefore received a draft from its
    previous owner. It is the only write in cloud/app.py that was not gated in the
    same statement or transaction as its ownership check.
    """

    def _signup(self, client, email):
        client.post("/v1/signup", json={"email": email, "password": "supersecret1"})
        return client.post(
            "/v1/login", json={"email": email, "password": "supersecret1"}
        ).json()["token"]

    def test_a_device_owned_by_someone_else_is_not_written_to(self, backend):
        app_module, client = backend
        mine = self._signup(client, "owner-a@example.com")
        theirs = self._signup(client, "owner-b@example.com")
        del theirs
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('theirs', 'X1', 'dk_x1', 2, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code = 'X1'").fetchone()["id"]

        resp = client.post(
            "/v1/ai/configure",
            json={"device_id": device_id, "prompt": "tell me when someone falls"},
            headers={"Authorization": f"Bearer {mine}"},
        )

        assert resp.status_code == 404
        with app_module.db() as conn:
            draft = conn.execute(
                "SELECT ai_rules_draft FROM devices WHERE id = ?", (device_id,)
            ).fetchone()["ai_rules_draft"]
        assert draft is None, "a draft was written to a device the caller does not own"

    def test_losing_the_device_during_the_model_call_does_not_write(self, backend, monkeypatch):
        """The race itself: ownership is real when checked and gone by the write."""
        app_module, client = backend
        mine = self._signup(client, "racer@example.com")
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('mine', 'X2', 'dk_x2', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code = 'X2'").fetchone()["id"]

        real_fallback = app_module._ai_fallback

        def reclaim_midway(prompt):
            # Stands in for the multi-second model call: the device changes hands
            # while this is running, exactly as a concurrent claim would do.
            with app_module.db() as conn:
                conn.execute("UPDATE devices SET owner_id = 99 WHERE id = ?", (device_id,))
            return real_fallback(prompt)

        monkeypatch.setattr(app_module, "_ai_fallback", reclaim_midway)

        resp = client.post(
            "/v1/ai/configure",
            json={"device_id": device_id, "prompt": "tell me when someone falls"},
            headers={"Authorization": f"Bearer {mine}"},
        )

        assert resp.status_code == 404
        with app_module.db() as conn:
            draft = conn.execute(
                "SELECT ai_rules_draft FROM devices WHERE id = ?", (device_id,)
            ).fetchone()["ai_rules_draft"]
        assert draft is None, "the draft landed on a device the caller had lost"

    def test_the_normal_path_still_writes_a_draft(self, backend):
        """The scoping must not break the feature it protects."""
        app_module, client = backend
        mine = self._signup(client, "normal@example.com")
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('mine', 'X3', 'dk_x3', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code = 'X3'").fetchone()["id"]

        resp = client.post(
            "/v1/ai/configure",
            json={"device_id": device_id, "prompt": "tell me when someone falls"},
            headers={"Authorization": f"Bearer {mine}"},
        )

        assert resp.status_code == 200, resp.text
        with app_module.db() as conn:
            draft = conn.execute(
                "SELECT ai_rules_draft FROM devices WHERE id = ?", (device_id,)
            ).fetchone()["ai_rules_draft"]
        assert draft is not None and "rules" in draft


class TestRequestSizeIsBounded:
    """Every one of these fields had a floor and no ceiling.

    "at least 8 characters", "must not be empty", "valid E.164" — the cheap
    direction was checked and the expensive one was not. A password is 200_000
    PBKDF2 iterations on a threadpool worker; a prompt is language-model tokens on
    an authenticated endpoint with no rate limit of its own.

    The limits are set far above any legitimate value, so the tests below assert
    both halves: the cap rejects the absurd, and a realistic request still works.
    """

    def _token(self, client, email="caps@example.com"):
        client.post("/v1/signup", json={"email": email, "password": "supersecret1"})
        return client.post(
            "/v1/login", json={"email": email, "password": "supersecret1"}
        ).json()["token"]

    # ------------------------------------------------------------------ passwords
    def test_an_absurd_password_is_rejected_before_hashing(self, backend):
        app_module, client = backend
        resp = client.post("/v1/signup", json={
            "email": "huge@example.com",
            "password": "x" * (app_module.MAX_PASSWORD_LEN + 1),
        })
        assert resp.status_code == 422

    def test_a_long_but_realistic_passphrase_still_works(self, backend):
        """A cap that rejects a real password-manager passphrase would be a bug
        of its own."""
        _, client = backend
        passphrase = "correct horse battery staple " * 4  # 116 chars
        assert client.post("/v1/signup", json={
            "email": "phrase@example.com", "password": passphrase,
        }).status_code == 200
        assert client.post("/v1/login", json={
            "email": "phrase@example.com", "password": passphrase,
        }).status_code == 200

    def test_the_short_password_rule_is_unchanged(self, backend):
        """The floor must still be a 400 with its own message, not a 422."""
        _, client = backend
        resp = client.post("/v1/signup", json={"email": "s@example.com", "password": "short"})
        assert resp.status_code == 400
        assert "8+" in resp.json()["detail"]

    def test_an_absurd_email_is_rejected(self, backend):
        app_module, client = backend
        resp = client.post("/v1/signup", json={
            "email": "a" * (app_module.MAX_EMAIL_LEN + 1) + "@example.com",
            "password": "supersecret1",
        })
        assert resp.status_code == 422

    # -------------------------------------------------------------------- prompts
    def test_an_absurd_ai_prompt_is_rejected_before_the_model_is_called(self, backend):
        app_module, client = backend
        token = self._token(client, "prompt@example.com")
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('cam', 'P1', 'dk_p1', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code='P1'").fetchone()["id"]

        resp = client.post("/v1/ai/configure", json={
            "device_id": device_id, "prompt": "x" * (app_module.MAX_AI_PROMPT_LEN + 1),
        }, headers={"Authorization": f"Bearer {token}"})

        assert resp.status_code == 422

    def test_a_wordy_but_reasonable_prompt_still_works(self, backend):
        app_module, client = backend
        token = self._token(client, "wordy@example.com")
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('cam', 'P2', 'dk_p2', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code='P2'").fetchone()["id"]

        prompt = "Tell me when someone falls over in the living room. " * 20  # ~1040 chars
        resp = client.post("/v1/ai/configure", json={"device_id": device_id, "prompt": prompt},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text

    # --------------------------------------------------------------------- ingest
    def test_an_absurd_event_batch_is_rejected(self, backend):
        app_module, client = backend
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, created_at)"
                " VALUES ('cam', 'I1', 'dk_i1', 0)"
            )
        events = [{"type": "fall", "timestamp": "2026-01-01T00:00:00"}] * (app_module.MAX_INGEST_EVENTS + 1)
        resp = client.post("/v1/ingest", json={"events": events},
                           headers={"X-Device-Key": "dk_i1"})
        assert resp.status_code == 422

    def test_a_completely_full_device_queue_is_still_accepted(self, backend):
        """The cap must sit ABOVE what a real device can send. core/uplink.py
        re-queues any non-200 and retries the identical batch, so a cap a real
        device could reach would be a permanent retry loop rather than load
        shedding — the device's own queue is bounded at 500."""
        app_module, client = backend
        assert app_module.MAX_INGEST_EVENTS > 500
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, created_at)"
                " VALUES ('cam', 'I2', 'dk_i2', 0)"
            )
        events = [
            {"type": "fall", "timestamp": f"2026-01-01T00:00:{i % 60:02d}", "event_key": f"k{i}"}
            for i in range(500)
        ]
        resp = client.post("/v1/ingest", json={"events": events},
                           headers={"X-Device-Key": "dk_i2"})
        assert resp.status_code == 200, resp.text

    def test_an_absurd_zone_list_is_rejected(self, backend):
        app_module, client = backend
        token = self._token(client, "zones@example.com")
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('cam', 'Z1', 'dk_z1', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code='Z1'").fetchone()["id"]
        zones = [{"name": f"z{i}", "points": [[0, 0], [1, 0], [1, 1]]} for i in range(app_module.MAX_ZONES + 1)]
        resp = client.post(f"/v1/devices/{device_id}/zones", json={"zones": zones},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 422


class TestAIProposalMatchesWhatTheAppCanDecode:
    """`/v1/ai/configure`'s response must carry every field iOS declares
    non-optional.

    `AIProposal` in ios/Sources/BackendAPI.swift is a `Codable` with seven
    non-optional properties. A missing one is not a degraded response — it is a
    `DecodingError`, so the whole Configure-with-AI screen fails with a generic
    message no matter how good the proposal was. Nothing else checks this: the
    Python tests assert on the dict, the Swift tests never see the wire format,
    and `ios/` is excluded from ruff and pyright.

    Read out of the Swift source rather than restated here, so adding a
    non-optional field on the iOS side fails this test instead of failing on a
    phone.
    """

    IOS_BACKEND_API = REPO / "ios" / "Sources" / "BackendAPI.swift"

    def _swift_required_fields(self) -> set[str]:
        import re

        source = self.IOS_BACKEND_API.read_text()
        match = re.search(r"struct AIProposal[^{]*\{(.*?)\n\}", source, re.DOTALL)
        assert match, "AIProposal is no longer declared as a struct in BackendAPI.swift"
        body = match.group(1)
        required: set[str] = set()
        for raw in body.splitlines():
            line = raw.split("//")[0].strip()
            # A brace means a computed property (`var isCustom: Bool { ... }`).
            # Those are derived on the client and never appear on the wire, so
            # requiring them would make this test demand keys the API must not
            # send — which is exactly what it did on the first run.
            if "{" in line:
                continue
            # `let x: T` / `var x: T` — non-optional, no default, so decoding
            # requires the key. `var x: T?` and `= default` are both fine to omit.
            m = re.match(r"(?:let|var)\s+(\w+)\s*:\s*([^=]+)$", line)
            if not m:
                continue
            name, swift_type = m.group(1), m.group(2).strip()
            if swift_type.endswith("?"):
                continue
            required.add(name)
        assert required, "parsed no required fields — has the struct's formatting changed?"
        return required

    def _proposal(self, backend, prompt: str) -> dict:
        app_module, client = backend
        client.post("/v1/signup", json={"email": "shape@example.com", "password": "supersecret1"})
        token = client.post(
            "/v1/login", json={"email": "shape@example.com", "password": "supersecret1"}
        ).json()["token"]
        with app_module.db() as conn:
            conn.execute(
                "INSERT INTO devices (name, claim_code, device_key, owner_id, created_at)"
                " VALUES ('cam', 'S1', 'dk_s1', 1, 0)"
            )
            device_id = conn.execute("SELECT id FROM devices WHERE claim_code='S1'").fetchone()["id"]
        resp = client.post("/v1/ai/configure", json={"device_id": device_id, "prompt": prompt},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        return resp.json()

    @pytest.mark.parametrize("prompt", [
        "tell me when someone falls",          # matches a preset
        "asdfghjkl nonsense zzz",              # nothing interpretable
        "count the giraffes in the room",      # interpretable but unsupported
    ])
    def test_every_field_the_app_requires_is_present(self, backend, prompt):
        """All three outcomes, because they take different code paths and only
        one of them was ever exercised by hand."""
        body = self._proposal(backend, prompt)
        missing = self._swift_required_fields() - set(body.keys())
        assert not missing, (
            f"/v1/ai/configure omitted {sorted(missing)}, which AIProposal declares "
            f"non-optional — the app would fail to decode the whole response"
        )

    def test_the_fields_have_the_types_the_app_expects(self, backend):
        """A String where the app wants a String. `null` decodes no better than a
        missing key for a non-optional property."""
        body = self._proposal(backend, "tell me when someone falls")
        for key in ("preset", "variant", "explanation", "source"):
            assert isinstance(body[key], str), f"{key} is {type(body[key]).__name__}, not a string"
        assert isinstance(body["rules"], list)
        assert isinstance(body["settings"], dict)
        assert isinstance(body["unsupported"], list)

    def test_the_only_source_value_the_app_branches_on_is_still_emitted(self, backend):
        """AIConfigureView shows its "no model configured" state on
        `source == "unavailable"` and nothing else, so that exact string is a
        contract rather than a label."""
        body = self._proposal(backend, "asdfghjkl nonsense zzz")
        assert body["source"] == "unavailable"


def test_interactive_api_docs_are_off_unless_explicitly_enabled(backend, tmp_path, monkeypatch):
    """/docs, /redoc and /openapi.json must not be served by default.

    FastAPI serves all three automatically, so the backend was publishing a
    complete, browsable map of its own attack surface -- every path, every
    parameter, every schema -- to anybody who could reach the port, including a
    "Try it out" button. That is a reconnaissance gift, not a feature, on a box
    whose whole job is to sit on a home LAN.

    They stay available for development behind an explicit opt-in, because a
    debugging aid that cannot be turned on gets replaced by something worse.
    """
    _, client = backend
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, f"{path} is being served"

    monkeypatch.setenv("SENSORDYME_API_DOCS", "1")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "docs-on.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    opted_in = importlib.import_module("cloud.app")
    try:
        with TestClient(opted_in.app) as dev_client:
            assert dev_client.get("/openapi.json").status_code == 200
            assert dev_client.get("/docs").status_code == 200
    finally:
        for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
            sys.modules.pop(mod, None)


def test_the_database_is_not_readable_by_other_accounts(backend):
    """The DB holds password hashes and JWT revocations; 0600 or it is pointless.

    The JWT secret one file over was already written with 0600 via a careful
    open(..., 0o600) dance; the database it protects was left at whatever the
    process umask happened to be, which on a shared or multi-user box means
    world-readable password hashes. sqlite creates the file on first connect, so
    this is asserted after the app module has initialised it -- and the -wal and
    -shm sidecars matter just as much, since in WAL mode recent rows live there.
    """
    app_module, _ = backend
    db = Path(app_module.DB_PATH)
    assert db.exists(), "the backend should have created the database on import"
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if not path.exists():
            continue  # -wal/-shm only exist once WAL has been written to
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600, f"{path.name} is {oct(mode)}, expected 0o600"
