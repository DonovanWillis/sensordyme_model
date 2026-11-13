"""M8.3 security: the route-authorization matrix of the backend (SECURITY_REVIEW_M6 §6).

Enumerates every route FastAPI serves, so a route added later cannot slip past
it: each one must be classified here (public, device key, account, or scoped
to one device), every non-public route must depend on an authenticator, and
every device-scoped route is driven as a SECOND account against the first
account's device -- it must answer 404 `device_not_found`, the same answer as
for a device that does not exist, and change nothing.

The bodies are valid, so a refusal is the ownership check and not a 422 from
validation (a 422 would pass a weaker "not 200" test while proving nothing).
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import types

import pytest

pytest.importorskip("fastapi")
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

KEY_A = "dk_" + "a1" * 24
KEY_B = "dk_" + "b2" * 24

#: Routes anyone may call.
PUBLIC = {("GET", "/v1/health"), ("POST", "/v1/signup"), ("POST", "/v1/login")}
#: Routes a device calls with its own key (sd-auth-v1 or the bearer key).
DEVICE_KEY = {("POST", "/v1/ingest")}
#: Account routes that name no device: they act on the caller's own rows only.
ACCOUNT = {
    ("POST", "/v1/logout"), ("POST", "/v1/account/ntfy"), ("POST", "/v1/account/phone"),
    ("POST", "/v1/orders"), ("GET", "/v1/orders"), ("GET", "/v1/devices"),
    ("POST", "/v1/devices/claim"), ("GET", "/v1/presets"), ("GET", "/v1/capability-tiers"),
}
#: Routes that name a device in the body rather than the path.
DEVICE_IN_BODY = {("POST", "/v1/ai/configure")}


def _routes(app) -> set[tuple[str, str]]:
    return {(method, r.path) for r in app.app.routes if isinstance(r, APIRoute) for method in (r.methods or ())}


def _device_scoped(app) -> set[tuple[str, str]]:
    return {(m, p) for m, p in _routes(app) if "{device_id}" in p} | DEVICE_IN_BODY


def _boot(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_JUPITER_PROVIDER", "off")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SENSORDYME_JWT_SECRET_FILE", str(tmp_path / ".jwt_secret"))
    monkeypatch.setenv("SENSORDYME_BACKEND_ID_FILE", str(tmp_path / ".backend_id"))
    monkeypatch.setenv("SENSORDYME_MDNS", "0")
    monkeypatch.setenv("SENSORDYME_DEVICE_AUTH", "permissive")
    monkeypatch.delenv("SENSORDYME_JWT_SECRET", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit", "cloud.manage"):
        sys.modules.pop(mod, None)
    app = importlib.import_module("cloud.app")
    app.gate.reset()
    return app, TestClient(app.app, raise_server_exceptions=False)


@pytest.fixture()
def two_accounts(tmp_path, monkeypatch):
    app, client = _boot(tmp_path, monkeypatch)
    with app.db() as conn:
        for name, code, key in (("Alice cam", "AAAA-AAAA-AAAA", KEY_A), ("Bob cam", "BBBB-BBBB-BBBB", KEY_B)):
            conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                         (name, code, key, time.time()))
    auth = {}
    for who, code in (("alice", "AAAA-AAAA-AAAA"), ("bob", "BBBB-BBBB-BBBB")):
        token = client.post("/v1/signup", json={"email": f"{who}@example.com", "password": "password1"}).json()["token"]
        auth[who] = {"Authorization": f"Bearer {token}"}
        assert client.post("/v1/devices/claim", json={"claim_code": code}, headers=auth[who]).status_code == 200
    with app.db() as conn:
        ids = {r["name"]: r["id"] for r in conn.execute("SELECT id, name FROM devices")}
        rules = json.dumps({"rules": [{"id": "r1", "target": "person", "enabled": True}]})
        conn.execute("UPDATE devices SET ai_rules = ? WHERE id = ?", (rules, ids["Alice cam"]))
        event = conn.execute(
            "INSERT INTO events (device_id, timestamp, type, confidence, signals, severity, received_at) "
            "VALUES (?, ?, 'fall', 0.9, 'private', 'critical', ?)",
            (ids["Alice cam"], "2026-11-13T00:00:00", time.time()),
        ).lastrowid
    return types.SimpleNamespace(app=app, client=client, auth=auth, alice=ids["Alice cam"],
                                 bob=ids["Bob cam"], event=event)


def _snapshot(app) -> list:
    with app.db() as conn:
        devices = [tuple(r) for r in conn.execute("SELECT * FROM devices ORDER BY id")]
        events = [tuple(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
    return [devices, events]


def _request(s, method: str, path: str, device_id: int, headers: dict):
    """One call of `method path` naming `device_id`, with a body that validates."""
    node = sorted(s.app.node_catalog())[0]
    url = path.replace("{device_id}", str(device_id)).replace("{event_id}", str(s.event)).replace("{rule_id}", "r1")
    bodies = {
        ("POST", "/v1/devices/{device_id}/name"): {"name": "taken over"},
        ("POST", "/v1/devices/{device_id}/protection"): {"action": "stop", "confirm": True},
        ("POST", "/v1/devices/{device_id}/preset"): {"preset": "fall_detection"},
        ("POST", "/v1/devices/{device_id}/zones"): {"zones": [{"name": "door", "polygon": [[0, 0], [1, 0], [1, 1]]}]},
        ("POST", "/v1/devices/{device_id}/nodes"): {"node": node, "enabled": True},
        ("PATCH", "/v1/devices/{device_id}/rules/{rule_id}"): {"enabled": False},
        ("POST", "/v1/ai/configure"): {"device_id": device_id, "prompt": "tell me when someone falls"},
    }
    return s.client.request(method, url, json=bodies.get((method, path)), headers=headers)


# ------------------------------------------------------------------ the inventory
def test_every_route_is_classified(two_accounts):
    """A new route fails here until someone decides who may call it."""
    routes = _routes(two_accounts.app)
    classified = PUBLIC | DEVICE_KEY | ACCOUNT | _device_scoped(two_accounts.app)
    assert routes - classified == set(), (
        f"unclassified routes {sorted(routes - classified)}: add each to PUBLIC, DEVICE_KEY, ACCOUNT or "
        f"DEVICE_IN_BODY in this file, and give a device-scoped one an ownership check (_owned_device)")
    assert classified - routes == set(), f"stale entries {sorted(classified - routes)}"
    assert not any(p.startswith(("/docs", "/redoc", "/openapi")) for _, p in routes), "API docs are off by default"


def _depends_on(dependant, names: set[str]) -> bool:
    return any(getattr(d.call, "__name__", "") in names or _depends_on(d, names) for d in dependant.dependencies)


def test_every_non_public_route_depends_on_an_authenticator(two_accounts):
    for route in two_accounts.app.app.routes:
        if not isinstance(route, APIRoute):
            continue
        keys = {(m, route.path) for m in (route.methods or ())}
        if keys <= PUBLIC:
            continue
        wanted = {"current_device"} if keys <= DEVICE_KEY else {"current_token"}
        assert _depends_on(route.dependant, wanted), f"{sorted(keys)} does not depend on {wanted}"


# ------------------------------------------------------------------ the matrix
def test_no_device_scoped_route_answers_another_account(two_accounts):
    s = two_accounts
    scoped = sorted(_device_scoped(s.app))
    assert len(scoped) >= 13, scoped  # the inventory, not an empty loop
    before = _snapshot(s.app)
    matrix = {}
    for method, path in scoped:
        cross = _request(s, method, path, s.alice, s.auth["bob"])
        missing = _request(s, method, path, 987654, s.auth["bob"])
        anonymous = _request(s, method, path, s.alice, {})
        matrix[f"{method} {path}"] = (cross.status_code, missing.status_code, anonymous.status_code)
        assert cross.status_code == 404, f"{method} {path} as another account: {cross.status_code} {cross.text[:200]}"
        # Indistinguishable from a device that does not exist: no ownership oracle.
        assert missing.status_code == 404 and cross.json()["detail"] == missing.json()["detail"], (method, path)
        assert anonymous.status_code == 401, (method, path, anonymous.status_code)
        assert "Alice" not in cross.text and "fall" not in cross.text
    assert _snapshot(s.app) == before, "another account's request changed the device or its events"
    # Positive control: the owner reads every scoped GET (the refusals above are
    # ownership, not a route that refuses everyone).
    for method, path in scoped:
        if method == "GET":
            assert _request(s, method, path, s.alice, s.auth["alice"]).status_code == 200, path
    print(json.dumps(matrix, indent=1))


def test_an_event_is_not_reachable_through_the_callers_own_device(two_accounts):
    """explain names a device AND an event: Bob's own device id with Alice's event id."""
    s = two_accounts
    answer = s.client.get(f"/v1/devices/{s.bob}/events/{s.event}/explain", headers=s.auth["bob"])
    assert answer.status_code == 404
    assert s.client.get(f"/v1/devices/{s.bob}/events", headers=s.auth["bob"]).json()["events"] == []


def test_account_routes_only_show_the_callers_own_rows(two_accounts):
    s = two_accounts
    listed = s.client.get("/v1/devices", headers=s.auth["bob"]).json()["devices"]
    assert [d["id"] for d in listed] == [s.bob]
    assert all("claim_code" not in d and "device_key" not in d for d in listed)
    s.client.post("/v1/orders", json={"quantity": 1, "address": "Alice's house"}, headers=s.auth["alice"])
    assert s.client.get("/v1/orders", headers=s.auth["bob"]).json()["orders"] == []
    # Ownership cannot move by claim: Alice's code, presented by Bob.
    taken = s.client.post("/v1/devices/claim", json={"claim_code": "AAAA-AAAA-AAAA"}, headers=s.auth["bob"])
    assert taken.status_code == 409 and taken.json()["code"] == "device_already_claimed"
    assert "Alice" not in taken.text
    with s.app.db() as conn:
        owner = conn.execute("SELECT owner_id FROM devices WHERE id = ?", (s.alice,)).fetchone()[0]
        alice_uid = conn.execute("SELECT id FROM users WHERE email = 'alice@example.com'").fetchone()[0]
    assert owner == alice_uid


def test_a_device_key_cannot_reach_account_routes(two_accounts):
    s = two_accounts
    for headers in ({"X-Device-Key": KEY_A}, {"Authorization": f"Bearer {KEY_A}"}):
        assert s.client.get("/v1/devices", headers=headers).status_code == 401
        assert s.client.get(f"/v1/devices/{s.alice}/events", headers=headers).status_code == 401


# ------------------------------------------------------------------ what a client may keep
def test_every_v1_answer_is_marked_no_store(two_accounts):
    """The app's URLSession.shared kept GET answers in its on-disk URLCache when
    no Cache-Control came back (reproduced on Foundation: cached without the
    header, not cached with no-store), where they outlived sign-out. Every /v1
    answer -- success, refusal, device poll -- now says no-store."""
    s = two_accounts
    answers = [
        s.client.get("/v1/health"),
        s.client.post("/v1/login", json={"email": "alice@example.com", "password": "password1"}),
        s.client.get("/v1/devices", headers=s.auth["alice"]),
        s.client.get(f"/v1/devices/{s.alice}/events", headers=s.auth["alice"]),
        s.client.get(f"/v1/devices/{s.alice}/events", headers=s.auth["bob"]),
        s.client.get("/v1/devices"),
        s.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY_A}),
    ]
    assert [r.status_code for r in answers] == [200, 200, 200, 200, 404, 401, 200]
    for answer in answers:
        assert answer.headers.get_list("cache-control") == ["no-store"], answer.request.url
