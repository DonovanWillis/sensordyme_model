"""Regression tests for the cloud-backend review (findings F1-F16).

Each test names the finding it pins. Every one of them failed against the
backend as it was before the fix; the repro scripts they came from are summarised
in each docstring.
"""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import jwt
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
DEVICE = {"X-Device-Key": "dk_test"}
SQUARE = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SENSORDYME_JWT_SECRET_FILE", str(tmp_path / ".jwt_secret"))
    monkeypatch.setenv("SENSORDYME_BACKEND_ID_FILE", str(tmp_path / ".backend_id"))
    monkeypatch.setenv("SENSORDYME_MDNS", "0")
    monkeypatch.delenv("SENSORDYME_JWT_SECRET", raising=False)
    monkeypatch.delenv("SENSORDYME_CLIENT_IP_HEADER", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    app_module.gate.reset()
    client = TestClient(app_module.app, raise_server_exceptions=False)
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Unit", "AA-BB-CC", "dk_test", time.time()),
        )
    auth = _signup(client, "owner@example.com")
    assert client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=auth).status_code == 200
    return app_module, client, auth


def _signup(client, email, password="password1"):
    r = client.post("/v1/signup", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _poll(client, **fields):
    r = client.post("/v1/ingest", json={"events": [], **fields}, headers=DEVICE)
    assert r.status_code == 200, r.text
    return r.json()["command"]


def _rule(target="person", **overrides):
    rule = {"target": target, "predicate": "present", "secondary": "",
            "description": f"{target} visible", "zone": "", "persist_s": 2.0,
            "min_count": 1, "confidence": 0.6, "severity": "warning",
            "cooldown_s": 60.0, "schedule": "", "notify": "push"}
    rule.update(overrides)
    return rule


def _custom_llm(prompt):
    target = "dog" if "dog" in prompt else "person"
    return {"preset": "", "variant": "", "explanation": f"watch for {target}",
            "settings": {}, "unsupported": [], "source": "jupiter", "rules": [_rule(target)]}


def _event(**overrides):
    event = {"event_key": uuid.uuid4().hex, "timestamp": "2026-10-15T00:00:00",
             "type": "fall", "confidence": 0.9, "signals": "", "severity": "loud", "duration": 3}
    event.update(overrides)
    return event


# ======================================================================= F1
def test_f1_a_token_for_a_row_that_is_now_someone_else_is_refused(backend):
    """Tokens named only a row id. Replacing the database under the same key
    (a Fly import over a deployment someone had signed in to) made a live token
    authenticate as whoever held that row in the new file -- reproduced: the
    tester's pre-import token listed and commanded "Bob's nursery cam"."""
    app_module, client, auth = backend
    assert client.get("/v1/devices", headers=auth).status_code == 200
    with app_module.db() as conn:  # row 1 now belongs to a different account
        conn.execute("UPDATE users SET email = 'bob@example.com' WHERE id = 1")
    r = client.get("/v1/devices", headers=auth)
    assert r.status_code == 401
    assert client.post("/v1/devices/1/preset", json={"preset": "driver_safety"}, headers=auth).status_code == 401


def test_f1_a_token_minted_without_the_email_claim_is_refused_once(backend):
    """Deliberate one-time cost: sessions issued before tokens carried the
    account's email must sign in again."""
    app_module, client, _ = backend
    legacy = jwt.encode({"uid": 1, "exp": int(time.time()) + 600, "jti": "legacy"},
                        app_module.JWT_SECRET, algorithm="HS256")
    assert client.get("/v1/devices", headers={"Authorization": f"Bearer {legacy}"}).status_code == 401


def test_f1_tokens_carry_the_account_email(backend):
    _, client, auth = backend
    token = auth["Authorization"].split()[1]
    assert jwt.decode(token, options={"verify_signature": False})["email"] == "owner@example.com"
    login = client.post("/v1/login", json={"email": "Owner@Example.com", "password": "password1"})
    assert login.status_code == 200
    assert client.get("/v1/devices", headers={"Authorization": f"Bearer {login.json()['token']}"}).status_code == 200


def _seed_import(path, emails):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, email TEXT);"
        "CREATE TABLE IF NOT EXISTS devices(id INTEGER PRIMARY KEY, owner_id INTEGER);"
        "CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY);"
    )
    conn.executemany("INSERT INTO users(email) VALUES (?)", [(e,) for e in emails])
    conn.commit()
    return conn


def _run_import(source, target, secret, *extra):
    env = {**os.environ, "SENSORDYME_JWT_SECRET_FILE": str(secret)}
    env.pop("SENSORDYME_JWT_SECRET", None)
    return subprocess.run(  # noqa: S603 - runs this repo's own script with sys.executable
        [sys.executable, str(REPO / "scripts" / "fly_import_db.py"), str(source), "--target", str(target), *extra],
        capture_output=True, text=True, env=env, check=False,
    )


def test_f1_import_rotates_the_signing_key_unless_told_not_to(tmp_path):
    source, target, secret = tmp_path / "import.db", tmp_path / "sensordyme.db", tmp_path / ".jwt_secret"
    _seed_import(source, ["bob@x"]).close()
    _seed_import(target, ["tester@x"]).close()
    secret.write_text("old-key")

    result = _run_import(source, target, secret)
    assert result.returncode == 0, result.stderr
    assert secret.read_text() != "old-key", "sessions minted before the import must end"
    assert oct(secret.stat().st_mode & 0o777) == "0o600"

    rotated = secret.read_text()
    result = _run_import(source, target, secret, "--keep-sessions")
    assert result.returncode == 0, result.stderr
    assert secret.read_text() == rotated


# ======================================================================= F2
def test_f2_configure_returns_the_draft_version(backend):
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    with app_module.db() as conn:
        draft = conn.execute("SELECT ai_rules_draft FROM devices WHERE id = 1").fetchone()[0]
    assert proposal["draft_version"] == app_module._rules_version(draft)
    assert "draft_version" not in json.loads(draft), "the version describes the draft, it is not part of it"


def test_f2_activate_runs_the_rules_as_reviewed_not_as_first_proposed(backend):
    """Reproduced before the fix: the user set a rule to Silent with a 30 s
    confirm; the device ran it as push with 2 s."""
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    edited = dict(proposal["rules"][0], notify="silent", persist_s=30.0, confidence=0.85)

    r = client.post("/v1/devices/1/preset", headers=auth, json={
        "preset": "", "draft_version": proposal["draft_version"], "rules": [edited]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rules_active"] == 1 and body["unsupported"] == []

    command = _poll(client)
    assert command["rules_version"] == body["rules_version"]
    sent = command["set_rules"][0]
    assert (sent["notify"], sent["persist_s"], sent["confidence"]) == ("silent", 30.0, 0.85)
    assert command["rules_instruction"] == proposal["explanation"]


def test_f2_a_newer_preview_cannot_be_activated_in_place_of_the_one_reviewed(backend):
    """Reproduced before the fix: the user approved "person"; a debounced
    preview for an edited prompt landed first; the device ran "dog"."""
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    shown = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a dog"}, headers=auth)

    r = client.post("/v1/devices/1/preset", headers=auth, json={
        "preset": "", "draft_version": shown["draft_version"], "rules": shown["rules"]})
    assert r.status_code == 409
    with app_module.db() as conn:
        row = conn.execute("SELECT ai_rules, custom_monitoring_wanted FROM devices WHERE id = 1").fetchone()
    assert row["ai_rules"] is None and row["custom_monitoring_wanted"] == 0, "a refused Activate changes nothing"
    assert _poll(client) is None


def test_f2_a_retried_activate_of_the_running_proposal_is_not_refused(backend):
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    body = {"preset": "", "draft_version": proposal["draft_version"], "rules": proposal["rules"]}
    assert client.post("/v1/devices/1/preset", headers=auth, json=body).status_code == 200
    assert client.post("/v1/devices/1/preset", headers=auth, json=body).status_code == 200
    assert client.post("/v1/devices/1/preset", headers=auth,
                       json={**body, "draft_version": "0" * 16}).status_code == 409
    assert client.post("/v1/devices/1/preset", headers=auth,
                       json={**body, "draft_version": "é" * 16}).status_code == 409, "not a 500"


def test_f2_edited_rules_are_revalidated_not_trusted(backend):
    """The app's rules are input: garbage numbers must not 500, an impossible
    target must not reach the device, unknown fields must not be stored."""
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    rules = [
        dict(proposal["rules"][0], persist_s="abc", confidence="__INF__", cooldown_s=None, min_count="x",
             injected="x" * 10),
        _rule("quantum_entanglement"),
    ]
    # Raw JSON text: 1e999 parses to infinity, which a JSON encoder will not write.
    raw = json.dumps({"preset": "", "draft_version": proposal["draft_version"], "rules": rules})
    r = client.post("/v1/devices/1/preset", content=raw.replace('"__INF__"', "1e999"),
                    headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["rules_active"] == 1
    assert [u["request"] for u in r.json()["unsupported"]] == ["quantum_entanglement visible"]
    sent = _poll(client)["set_rules"]
    assert len(sent) == 1 and "injected" not in sent[0]
    assert (sent[0]["persist_s"], sent[0]["confidence"], sent[0]["cooldown_s"], sent[0]["min_count"]) == (
        2.0, 0.6, 60.0, 1)


def test_f2_an_app_that_sends_only_the_preset_behaves_as_before(backend):
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    r = client.post("/v1/devices/1/preset", headers=auth, json={"preset": ""})
    assert r.status_code == 200 and r.json()["rules_active"] == 1
    assert _poll(client)["set_rules"] == proposal["rules"]


@pytest.mark.parametrize(("timezone", "kept"), [
    ("America/New_York", True), ("America/Argentina/Buenos_Aires", True), ("Etc/GMT+5", True),
    ("../../etc/passwd", False), ("Europe/London; rm", False), ("x" * 65, False), (42, False),
])
def test_f2_a_rule_timezone_passes_through_only_when_it_looks_like_one(backend, timezone, kept):
    """The device's RuleEngine reads an optional IANA `timezone` per rule (the
    owner's zone for `schedule`); the app sends TimeZone.current.identifier."""
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth).json()
    rule = dict(proposal["rules"][0], schedule="23:00-06:00", timezone=timezone)
    r = client.post("/v1/devices/1/preset", headers=auth, json={
        "preset": "", "draft_version": proposal["draft_version"], "rules": [rule]})
    assert r.status_code == 200, r.text
    sent = _poll(client)["set_rules"][0]
    assert sent.get("timezone") == (timezone if kept else None)


# ======================================================================= F3
def test_f3_rules_sent_with_a_named_mode_are_reported_not_silently_accepted(backend):
    """The device can only run delivered rules as AI Custom, which replaces the
    mode's own module; a named mode never delivers them. Before: the rule sat in
    a draft for ever while the app said "Active"."""
    _, client, auth = backend
    r = client.post("/v1/devices/1/preset", headers=auth,
                    json={"preset": "elderly_monitoring", "rules": [_rule("on_floor")]})
    assert r.status_code == 200
    body = r.json()
    assert body["rules_active"] == 0 and body["rules_version"] == ""
    assert len(body["unsupported"]) == 1 and "alongside a monitoring mode" in body["unsupported"][0]["reason"]
    assert _poll(client, active_preset="driver_safety") == {"set_preset": "elderly_monitoring"}


def test_f3_a_mode_proposal_never_promises_rules(backend):
    """Activating a fallback proposal the way the app does (its own preset)
    must deliver everything the proposal claimed -- which is now the mode."""
    app_module, client, auth = backend
    proposal = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "let me know if grandma falls"},
                           headers=auth).json()
    assert proposal["preset"] == "elderly_monitoring" and proposal["source"] == "fallback"
    assert proposal["rules"] == [] and "added on top" not in proposal["explanation"].lower()
    client.post("/v1/devices/1/preset", headers=auth, json={"preset": proposal["preset"]})
    assert _poll(client, active_preset="driver_safety") == {"set_preset": "elderly_monitoring"}

    # Whatever the source, validation moves a mode proposal's rules out.
    validated = app_module._validate_proposal(
        {"preset": "security_monitoring", "variant": "", "explanation": "", "settings": {},
         "unsupported": [], "rules": [_rule("person")]})
    assert validated["rules"] == [] and len(validated["unsupported"]) == 1


# ======================================================================= F4
def test_f4_deleting_the_last_zone_reaches_the_device(backend):
    _, client, auth = backend
    client.post("/v1/devices/1/zones", json={"zones": [{"name": "pool", "polygon": SQUARE}]}, headers=auth)
    version = _poll(client)["zones_version"]
    assert _poll(client, zones_version=version) is None

    client.post("/v1/devices/1/zones", json={"zones": []}, headers=auth)
    command = _poll(client, zones_version=version)
    assert command is not None and command["set_zones"] == [], "the device must be told to clear the zone"
    assert _poll(client, zones_version=command["zones_version"]) is None, "and it converges"


def test_f4_a_device_that_never_had_zones_is_told_nothing(backend):
    _, client, _ = backend
    assert _poll(client) is None


# ======================================================================= F5
def test_f5_a_storage_failure_is_a_503_not_a_200_that_loses_events(backend):
    """Reproduced before the fix with SQLITE_FULL: 200, `stored: 3`, 1 row on
    disk -- and the device clears its queue on any 200."""
    app_module, client, auth = backend
    real_connect = app_module._connect

    def full_disk():
        conn = real_connect()
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count={pages}")
        return conn

    events = [_event(signals="x" * 3000) for _ in range(5)]
    app_module._connect = full_disk
    try:
        r = client.post("/v1/ingest", json={"events": events}, headers=DEVICE)
    finally:
        app_module._connect = real_connect
    assert r.status_code == 503, r.text
    assert client.get("/v1/devices/1/events", headers=auth).json()["events"] == [], "nothing half-stored"

    # The device's retry of the identical batch then lands whole.
    r = client.post("/v1/ingest", json={"events": events}, headers=DEVICE)
    assert r.status_code == 200 and r.json()["stored"] == 5


def test_f5_a_malformed_event_is_still_skipped_alone(backend):
    _, client, _auth = backend
    r = client.post("/v1/ingest", json={"events": [_event(confidence="high"), _event()]}, headers=DEVICE)
    assert r.status_code == 200 and r.json()["stored"] == 1


def test_an_event_with_a_huge_integer_is_skipped_not_a_500(backend):
    """float(10**400) raises OverflowError, which the narrowed per-event except
    did not catch: the batch 500'd, the device re-sent it every poll, and every
    newer alert queued behind it."""
    _, client, _ = backend
    huge = json.dumps(_event()).replace('"confidence": 0.9', '"confidence": 1' + "0" * 400)
    after = json.dumps(_event(signals="after"))
    body = f'{{"events": [{huge}, {after}]}}'
    r = client.post("/v1/ingest", content=body,
                    headers={**DEVICE, "Content-Type": "application/json"})
    assert r.status_code == 200 and r.json()["stored"] == 1, r.text


def test_f5_retention_deletes_by_age_and_by_per_device_cap(backend, monkeypatch):
    app_module, client, auth = backend
    now = time.time()
    with app_module.db() as conn:
        for i in range(30):
            received = now - 200 * 86400 if i < 10 else now - i
            conn.execute(
                "INSERT INTO events (device_id, timestamp, type, received_at) VALUES (1, ?, ?, ?)",
                ("t", f"e{i}", received),
            )
    monkeypatch.setattr(app_module, "EVENT_RETENTION_DAYS", 90.0)
    monkeypatch.setattr(app_module, "EVENT_MAX_PER_DEVICE", 5)
    monkeypatch.setattr(app_module, "_PRUNE_BATCH", 3)  # exercise the batching
    assert app_module._prune_events(now) == 25
    kept = [e["type"] for e in client.get("/v1/devices/1/events", headers=auth).json()["events"]]
    assert kept == ["e29", "e28", "e27", "e26", "e25"], "the newest are kept"


def test_f5_ingest_triggers_retention_at_most_once_per_interval(backend, monkeypatch):
    app_module, client, _ = backend
    runs = []
    monkeypatch.setattr(app_module, "_prune_events", lambda now=None: runs.append(now) or 0)
    monkeypatch.setattr(app_module, "_next_prune_at", 0.0)
    _poll(client)
    _poll(client)
    assert len(runs) == 1


# ======================================================================= F7
def test_f7_one_ipv6_host_gets_one_bucket_not_one_per_address(backend, monkeypatch):
    """Reproduced before the fix: 40 guesses from 40 addresses in one /64, all
    evaluated, none throttled."""
    _, client, _ = backend
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    codes = [
        client.post("/v1/login", json={"email": "victim@example.com", "password": f"guess{i}"},
                    headers={"Fly-Client-IP": f"2001:db8:1:2::{i:x}"}).status_code
        for i in range(1, 13)
    ]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]


def test_f7_failed_logins_are_limited_per_account_across_addresses(backend, monkeypatch):
    _, client, _ = backend
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    codes = [
        client.post("/v1/login", json={"email": "owner@example.com", "password": f"wrong{i}"},
                    headers={"Fly-Client-IP": f"198.51.100.{i}"}).status_code
        for i in range(1, 23)
    ]
    assert codes[:20] == [401] * 20 and codes[20:] == [429, 429]
    # Unknown emails are throttled the same way, so a 429 reveals nothing.
    codes = [
        client.post("/v1/login", json={"email": "nobody@example.com", "password": f"w{i}"},
                    headers={"Fly-Client-IP": f"203.0.113.{i}"}).status_code
        for i in range(1, 23)
    ]
    assert codes[20:] == [429, 429]


def test_f7_a_successful_login_does_not_use_up_the_account_limit(backend, monkeypatch):
    _, client, _ = backend
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    for i in range(25):
        r = client.post("/v1/login", json={"email": "owner@example.com", "password": "password1"},
                        headers={"Fly-Client-IP": f"192.0.2.{i + 1}"})
        assert r.status_code == 200


def test_f7_limiter_forgets_expired_callers_and_is_bounded():
    """Before: every caller's key was kept for ever (~860 bytes each)."""
    from cloud.ratelimit import RateLimiter

    limiter = RateLimiter(max_attempts=10, window_s=300.0, max_keys=1000)
    for i in range(5000):
        limiter.allow(f"10.0.{i // 256}.{i % 256}", now=0.0)
    assert len(limiter) == 1000, "the table is capped"
    limiter.allow("late-caller", now=10_000.0)  # long after every window
    assert len(limiter) == 1, "expired callers are forgotten"


def test_f7_rate_limit_key_normalisation(backend):
    app_module, _, _ = backend
    key = app_module._rate_limit_key
    assert key("2001:db8:1:2::1") == key("2001:db8:1:2:ffff::9") == "2001:db8:1:2::/64"
    assert key("2001:db8:1:3::1") != key("2001:db8:1:2::1")
    assert key("::ffff:198.51.100.7") == "198.51.100.7"
    assert key("198.51.100.7") == "198.51.100.7"
    assert key("testclient") == "testclient"


# ======================================================================= F8
def test_f8_unencodable_text_from_a_device_cannot_break_the_device_list(backend):
    """Reproduced before the fix: GET /v1/devices answered 400 for the whole
    account after one report with a lone surrogate."""
    _, client, auth = backend
    raw = (b'{"events": [{"type": "fall", "severity": "log", "signals": "x \\udcff"}],'
           b' "active_preset": "driver_safety\\ud800",'
           b' "health": {"network": {"state": "ok", "detail": "ssid \\udcff", "\\ud800k": 1}},'
           b' "monitoring": {"active": false, "instruction": "x \\ud800"}}')
    r = client.post("/v1/ingest", content=raw, headers={**DEVICE, "Content-Type": "application/json"})
    assert r.status_code == 200, r.text
    assert r.json()["stored"] == 1, "the event is kept, with the bad character replaced"
    r = client.get("/v1/devices", headers=auth)
    assert r.status_code == 200, r.text
    assert client.get("/v1/devices/1/events", headers=auth).status_code == 200


# ======================================================================= F9
def test_f9_the_backup_includes_commits_still_in_the_wal(tmp_path):
    """Reproduced before the fix: an account created moments before the import
    was in neither the backup nor anywhere else."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed_import(source, ["bob@x"]).close()
    live = _seed_import(target, ["early@x"])  # the running backend's connection
    live.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    live.execute("INSERT INTO users(email) VALUES ('moments-ago@x')")
    live.commit()
    assert os.path.getsize(f"{target}-wal") > 0, "premise: the commit is in the WAL"

    result = _run_import(source, target, tmp_path / ".jwt_secret")
    assert result.returncode == 0, result.stderr
    backup = next(tmp_path.glob("sensordyme.db.replaced-*"))
    emails = [r[0] for r in sqlite3.connect(backup).execute("SELECT email FROM users")]
    assert "moments-ago@x" in emails
    live.close()
    assert [r[0] for r in sqlite3.connect(target).execute("SELECT email FROM users")] == ["bob@x"]


# ======================================================================= F10
def test_f10_reactivating_unchanged_rules_is_delivered_again(backend):
    """A device that left AI Custom keeps reporting the old rules_version
    (core/uplink.py never resets it), so re-activating the same rules matched
    and nothing was sent."""
    app_module, client, auth = backend
    app_module._ai_configure_llm = _custom_llm
    client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth)
    client.post("/v1/devices/1/preset", json={"preset": ""}, headers=auth)
    version = _poll(client, active_preset="driver_safety")["rules_version"]
    assert _poll(client, active_preset="driver_safety", rules_version=version,
                 monitoring={"active": True}) is None

    client.post("/v1/devices/1/preset", json={"preset": "baby_monitoring"}, headers=auth)
    _poll(client, active_preset="baby_monitoring", rules_version=version, monitoring={"active": False})

    r = client.post("/v1/devices/1/preset", json={"preset": ""}, headers=auth)
    command = _poll(client, active_preset="baby_monitoring", rules_version=version,
                    monitoring={"active": False})
    assert command is not None and "set_rules" in command
    assert command["rules_version"] == r.json()["rules_version"] != version


def test_f10_existing_rule_sets_keep_their_fingerprint(backend):
    """Rows from before the activation counter must not all be re-sent once on deploy."""
    app_module, _, _ = backend
    assert app_module._rules_version('{"rules": []}') == app_module._rules_version('{"rules": []}', 0)


# ======================================================================= F11
def test_f11_a_point_sent_as_an_object_is_rejected_not_a_500(backend):
    _, client, auth = backend
    r = client.post("/v1/devices/1/zones", headers=auth,
                    json={"zones": [{"name": "x", "polygon": [{"x": 0.1, "y": 0.1}] * 3}]})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "zones": [], "rejected": 1}


# ======================================================================= F12
@pytest.mark.parametrize(("event", "channels"), [
    ({"severity": "loud"}, (True, True)),
    ({"severity": "soft", "source": "custom_rule"}, (True, False)),
    # "push" is the default rule choice (the app offers only Push/Silent), so
    # a critical rule set to it must still text an owner with sms_on_loud on.
    ({"severity": "loud", "notify": "push", "source": "custom_rule"}, (True, True)),
    ({"severity": "soft", "notify": "push", "source": "custom_rule"}, (True, False)),
    ({"severity": "soft", "notify": "sms"}, (True, True)),
    ({"severity": "loud", "notify": "silent"}, (False, False)),
])
def test_f12_channel_choice(backend, event, channels):
    app_module, _, _ = backend
    assert app_module._notify_channels(event) == channels


def test_f12_a_soft_custom_rule_alert_is_pushed_but_not_texted(backend, monkeypatch):
    app_module, client, auth = backend
    client.post("/v1/account/phone", json={"phone": "+15551234567"}, headers=auth)
    sent = []
    monkeypatch.setattr(app_module, "send_push", lambda *a: sent.append("push"))
    monkeypatch.setattr(app_module, "send_sms", lambda *a: sent.append("sms"))
    client.post("/v1/ingest", headers=DEVICE,
                json={"events": [_event(type="person", severity="soft", source="custom_rule")]})
    assert sent == ["push"]

    app_module.gate.reset()
    sent.clear()
    client.post("/v1/ingest", headers=DEVICE, json={"events": [_event(severity="loud")]})
    assert sent == ["push", "sms"]


def test_f12_a_silent_rule_does_not_use_up_the_cooldown(backend, monkeypatch):
    app_module, client, _ = backend
    sent = []
    monkeypatch.setattr(app_module, "send_push", lambda owner, title, detail: sent.append(detail))
    monkeypatch.setattr(app_module, "send_sms", lambda *a: None)
    client.post("/v1/ingest", headers=DEVICE, json={"events": [
        _event(severity="loud", notify="silent", signals="the silent one"),
        _event(severity="loud", signals="the one that should buzz"),
    ]})
    assert sent == ["the one that should buzz"]


# ======================================================================= F13
def test_f13_model_calls_are_capped_across_accounts(backend):
    _app_module, client, auth = backend
    from cloud import ratelimit

    held = 0
    while ratelimit.AI_GLOBAL_CONCURRENCY.acquire(ratelimit.AI_GLOBAL_KEY):
        held += 1
    try:
        assert held == 8
        r = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"}, headers=auth)
        assert r.status_code == 429
    finally:
        for _ in range(held):
            ratelimit.AI_GLOBAL_CONCURRENCY.release(ratelimit.AI_GLOBAL_KEY)
    assert client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "a person"},
                       headers=auth).status_code == 200
    # The per-account slot was handed back even though the global one refused.
    assert ratelimit.AI_CONCURRENCY.acquire("1") and ratelimit.AI_CONCURRENCY.acquire("1")
    ratelimit.AI_CONCURRENCY.release("1")
    ratelimit.AI_CONCURRENCY.release("1")


# ======================================================================= F6 / F14 (mDNS)
class _FakeZeroconf:
    def __init__(self):
        self.registered = []

    def register_service(self, info):
        self.registered.append(info)

    def unregister_service(self, info):
        pass

    def close(self):
        pass


@pytest.fixture()
def advertiser_world(monkeypatch, tmp_path):
    import zeroconf as _zeroconf

    from cloud import mdns

    built = []
    monkeypatch.setattr(_zeroconf, "Zeroconf", lambda *a, **k: built.append(_FakeZeroconf()) or built[-1])
    monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(tmp_path / ".backend_id"))
    (tmp_path / ".backend_id").write_text("bk_0123456789ab\n")
    state = {"lan": "10.0.0.5", "interfaces": {"10.0.0.5": "en0"}, "owners": []}
    monkeypatch.setattr(mdns, "lan_ip", lambda: state["lan"])
    monkeypatch.setattr(mdns, "_ipv4_interfaces", lambda: dict(state["interfaces"]))
    monkeypatch.setattr(mdns, "hostname_owners", lambda zc, name, timeout_ms=0: list(state["owners"]))
    return mdns, state, built


def test_f6_the_shared_name_is_claimed_when_nobody_else_answers(advertiser_world):
    mdns, _state, built = advertiser_world
    advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
    try:
        assert advertiser.start()
        assert built[-1].registered[-1].server == "sensordyme-backend.local."
    finally:
        advertiser.stop()


def test_f6_a_second_backend_takes_its_own_name_instead_of_colliding(advertiser_world):
    """zeroconf never probes the `server` host name, so two backends both
    answered sensordyme-backend.local and a device could reach the wrong one."""
    mdns, state, built = advertiser_world
    state["owners"] = ["10.0.0.9"]  # another machine already answers for it
    advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
    try:
        assert advertiser.start()
        assert built[-1].registered[-1].server == "sensordyme-backend-01234567.local."
        # Chosen once: a later move keeps the name clients already learned.
        state["owners"] = []
        state["lan"], state["interfaces"] = "10.0.1.7", {"10.0.1.7": "en0"}
        assert advertiser.refresh()
        assert built[-1].registered[-1].server == "sensordyme-backend-01234567.local."
    finally:
        advertiser.stop()


def test_f6_an_answer_from_this_machine_is_not_a_conflict(advertiser_world):
    mdns, state, built = advertiser_world
    state["owners"] = ["10.0.0.5"]  # e.g. a sleep proxy answering for us
    advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
    try:
        assert advertiser.start()
        assert built[-1].registered[-1].server == "sensordyme-backend.local."
    finally:
        advertiser.stop()


def test_f14_every_lan_address_is_announced_and_tunnels_are_not(advertiser_world):
    mdns, state, built = advertiser_world
    # Full-tunnel VPN: the default route is the tunnel; the LAN is en0 + en7.
    state["lan"] = "100.64.0.3"
    state["interfaces"] = {"100.64.0.3": "utun4", "192.168.1.20": "en0", "10.9.0.4": "en7",
                           "127.0.0.1": "lo0", "169.254.3.3": "en5"}
    assert mdns.lan_addresses() == ["192.168.1.20", "10.9.0.4"]
    advertiser = mdns.Advertiser(8787, refresh_interval_s=0)
    try:
        assert advertiser.start()
        import socket

        announced = [socket.inet_ntoa(a) for a in built[-1].registered[-1].addresses]
        assert announced == ["192.168.1.20", "10.9.0.4"]
    finally:
        advertiser.stop()


def test_f14_the_routed_lan_address_comes_first(advertiser_world):
    mdns, state, _ = advertiser_world
    state["lan"] = "10.9.0.4"
    state["interfaces"] = {"192.168.1.20": "en0", "10.9.0.4": "en7"}
    assert mdns.lan_addresses() == ["10.9.0.4", "192.168.1.20"]
    state["lan"] = None
    assert mdns.lan_addresses() == [], "no default route still means not on a network"


# ======================================================================= F15
def test_f15_the_diagnostic_checks_this_backends_record_not_the_first_one(monkeypatch, tmp_path):
    """With two backends visible, it used to check whichever answered first."""
    import socket

    import zeroconf as _zeroconf

    from cloud import mdns
    from scripts import diagnose_backend_mac as diag

    class Info:
        def __init__(self, backend_id, address):
            self.properties = {b"id": backend_id.encode()}
            self.addresses = [socket.inet_aton(address)]
            self.port = 8787

    records = {
        "Other._sensordyme-api._tcp.local.": Info("bk_other", "10.0.0.5"),
        "Mine._sensordyme-api._tcp.local.": Info("bk_mine", "10.0.0.77"),
    }

    class FakeZC:
        def get_service_info(self, type_, name, timeout=0):
            return records[name]

        def close(self):
            pass

    def browser(zc, type_, listener):
        for name in records:
            listener.add_service(zc, type_, name)

    monkeypatch.setattr(_zeroconf, "Zeroconf", FakeZC)
    monkeypatch.setattr(_zeroconf, "ServiceBrowser", browser)
    monkeypatch.setattr(mdns, "lan_ip", lambda: "10.0.0.5")
    id_file = tmp_path / ".backend_id"
    id_file.write_text("bk_mine\n")
    monkeypatch.setattr(mdns, "BACKEND_ID_FILE", str(id_file))

    ok, detail = diag.mdns_advertising(timeout=0)
    # OUR record says 10.0.0.77 while this machine is 10.0.0.5: stale. The
    # other backend's matching address must not make it look current.
    assert not ok and "STALE" in detail

    id_file.write_text("bk_nobody\n")
    ok, detail = diag.mdns_advertising(timeout=0)
    assert not ok and "is not advertising" in detail

    id_file.unlink()
    assert mdns.read_backend_id() is None
    assert not id_file.exists(), "the diagnostic must never mint an identity"
