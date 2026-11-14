"""M8 security: the backend side of device auth, claim and credentials at rest.

Each test reproduces a defect found in the M8 review against the real
cloud/app.py (docs/SECURITY_REVIEW_M6.md §5):

* request nonces lived in process memory, so a captured signed request was
  accepted again after a backend restart;
* a revoked key was reported as "unknown" and the app saw only "offline";
* signed requests could not be required per device;
* claim-code guessing was limited per address only;
* an existing JWT secret file kept whatever mode it had.
"""

from __future__ import annotations

import importlib
import os
import stat
import sys
import time
import types

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core import command_auth as a

KEY = "dk_" + "66" * 24
OTHER = "dk_" + "88" * 24
BODY = b'{"events": []}'


def _boot(tmp_path, monkeypatch, mode="permissive"):
    """Import cloud.app fresh on `tmp_path` -- a backend (re)start."""
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SENSORDYME_JWT_SECRET_FILE", str(tmp_path / ".jwt_secret"))
    monkeypatch.setenv("SENSORDYME_BACKEND_ID_FILE", str(tmp_path / ".backend_id"))
    monkeypatch.setenv("SENSORDYME_MDNS", "0")
    monkeypatch.setenv("SENSORDYME_DEVICE_AUTH", mode)
    monkeypatch.delenv("SENSORDYME_JWT_SECRET", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit", "cloud.manage"):
        sys.modules.pop(mod, None)
    app = importlib.import_module("cloud.app")
    app.gate.reset()
    return app, TestClient(app.app, raise_server_exceptions=False)


def _backend(tmp_path, monkeypatch, mode="permissive"):
    app, client = _boot(tmp_path, monkeypatch, mode)
    with app.db() as conn:
        for name, code, key in (("Unit", "AA-BB-CC", KEY), ("Other", "DD-EE-FF", OTHER)):
            conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                         (name, code, key, time.time()))
    token = client.post("/v1/signup", json={"email": "owner@example.com", "password": "password1"}).json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=auth).status_code == 200
    with app.db() as conn:
        device_id = conn.execute("SELECT id FROM devices WHERE claim_code = 'AA-BB-CC'").fetchone()[0]
    return types.SimpleNamespace(app=app, client=client, auth=auth, device_id=device_id)


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    return _backend(tmp_path, monkeypatch)


def _signed_headers(key=KEY, body=BODY, ts=None, nonce=None):
    header = a.sign_request(a.DeviceKey(key), "POST", "/v1/ingest", body, nonce=nonce or a.new_nonce(),
                            issued_at=int(time.time()) if ts is None else ts)
    return {"Content-Type": "application/json", "X-SD-Auth": header}


def _ingest(client, headers, body=BODY):
    return client.post("/v1/ingest", content=body, headers=headers)


def _row(backend):
    return next(d for d in backend.client.get("/v1/devices", headers=backend.auth).json()["devices"]
                if d["id"] == backend.device_id)


# ------------------------------------------------------------------ replay across a backend restart
@pytest.mark.parametrize("mode", ["permissive", "required"])
def test_a_captured_request_is_refused_after_a_backend_restart(tmp_path, monkeypatch, mode):
    backend = _backend(tmp_path, monkeypatch, mode)
    headers = _signed_headers()
    assert _ingest(backend.client, headers).status_code == 200
    _, restarted = _boot(tmp_path, monkeypatch, mode)  # same database, new process state
    again = _ingest(restarted, headers)
    assert again.status_code == 401 and again.json()["code"] == "device_auth_replayed_nonce"
    assert _ingest(restarted, _signed_headers()).status_code == 200, "fresh requests still work"


def test_stored_nonces_are_pruned_once_they_cannot_pass_the_clock_check(backend):
    store = backend.app._REQUEST_NONCES
    t0 = 1_900_000_000.0
    assert store.add("a" * 32, t0) and not store.add("a" * 32, t0 + 1)
    assert store.add("b" * 32, t0 + 2 * a.CLOCK_SKEW_S + 60)  # prune pass runs here
    with backend.app.db() as conn:
        kept = {r[0] for r in conn.execute("SELECT nonce FROM auth_nonces")}
    assert "a" * 32 not in kept and "b" * 32 in kept


def test_the_stored_nonces_are_bounded(backend):
    store = backend.app._StoredNonces(max_entries=3)
    t0 = 1_900_000_000.0
    for i in range(5):
        store._next_prune = 0.0  # force the prune pass on every add
        assert store.add(f"{i:032x}", t0 + i)
    with backend.app.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_nonces").fetchone()[0] <= 4


# ------------------------------------------------------------------ revocation, reported truthfully
def test_a_revoked_previous_key_is_refused_as_revoked_and_the_app_is_told(backend):
    import cloud.manage as manage

    new_key = manage.rotate_key(backend.device_id)
    assert _ingest(backend.client, _signed_headers(key=KEY)).status_code == 200, "previous key during rotation"
    manage.revoke_previous_key(backend.device_id)

    signed = _ingest(backend.client, _signed_headers(key=KEY))
    assert signed.status_code == 401 and signed.json()["code"] == "device_auth_revoked_key"
    bearer = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    assert bearer.status_code == 401 and bearer.json()["code"] == "device_auth_revoked_key"
    credentials = _row(backend)["credentials"]
    assert credentials["state"] == "refused" and credentials["refusal"]["reason"] == "revoked_key"

    assert _ingest(backend.client, _signed_headers(key=new_key)).status_code == 200
    assert _row(backend)["credentials"]["state"] == "accepted", "an accepted report supersedes the refusal"


def test_a_forged_tag_under_a_revoked_kid_is_not_reported_against_the_device(backend):
    import cloud.manage as manage

    manage.rotate_key(backend.device_id)
    manage.revoke_previous_key(backend.device_id)
    forged = _signed_headers(key=KEY)
    forged["X-SD-Auth"] = forged["X-SD-Auth"][:-64] + "0" * 64  # anyone can name a public kid
    r = _ingest(backend.client, forged)
    assert r.status_code == 401 and r.json()["code"] == "device_auth_unknown_key"
    assert _row(backend)["credentials"]["refusal"] is None


def test_a_revoked_device_cannot_ingest_or_receive_commands_until_reissued(backend):
    import cloud.manage as manage

    r = backend.client.post(f"/v1/devices/{backend.device_id}/protection",
                            json={"action": "stop", "confirm": True}, headers=backend.auth)
    assert r.status_code == 200
    manage.revoke_device(backend.device_id)
    with_event = b'{"events": [{"type": "x", "timestamp": "t"}]}'
    for headers in (_signed_headers(body=with_event), {"X-Device-Key": KEY, "X-SD-Nonce": a.new_nonce()}):
        r = backend.client.post("/v1/ingest", content=with_event,
                                headers={"Content-Type": "application/json", **headers})
        assert r.status_code == 401 and r.json()["code"] == "device_auth_revoked_device"
        assert "command" not in r.json()
    with backend.app.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    row = _row(backend)
    assert row["credentials"]["state"] == "revoked"
    assert row["status"]["monitoring_state"] == "unknown"
    # The refusal LEADS the reasons: the app shows only the first one, and
    # "no report from the device in the last 30 s" hid the actionable cause
    # behind it indefinitely (M8.5 reconnect audit).
    assert "revoked" in row["status"]["reasons"][0]

    new_key = manage.rotate_key(backend.device_id)
    assert _ingest(backend.client, _signed_headers(key=KEY)).json()["code"] == "device_auth_revoked_key"
    r = _ingest(backend.client, _signed_headers(key=new_key))
    assert r.status_code == 200 and r.json()["command"]["set_protection"]["action"] == "stop"
    assert _row(backend)["credentials"]["state"] == "accepted"


# ------------------------------------------------------------------ per-device enforcement
def test_signed_requests_can_be_required_for_one_device(backend):
    import cloud.manage as manage

    assert backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY}).status_code == 200
    assert _row(backend)["credentials"]["auth"] == "bearer"
    with pytest.raises(SystemExit, match="not been seen signing"):
        manage.set_auth_mode(backend.device_id, True)
    assert _ingest(backend.client, _signed_headers()).status_code == 200
    assert _row(backend)["credentials"]["auth"] == "signed"
    manage.set_auth_mode(backend.device_id, True)

    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    assert r.status_code == 401 and r.json()["code"] == "device_auth_signature_required"
    assert _row(backend)["credentials"]["signature_required"] is True
    other = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": OTHER})
    assert other.status_code == 200, "other devices keep the backend's mode"
    assert _ingest(backend.client, _signed_headers()).status_code == 200

    manage.set_auth_mode(backend.device_id, False)
    assert backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY}).status_code == 200


def test_a_device_cannot_loosen_a_required_backend(tmp_path, monkeypatch):
    backend = _backend(tmp_path, monkeypatch, mode="required")
    import cloud.manage as manage

    manage.set_auth_mode(backend.device_id, False)
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    assert r.status_code == 401 and r.json()["code"] == "device_auth_signature_required"


# ------------------------------------------------------------------ one key, one device
def test_a_device_key_cannot_act_as_another_device(backend):
    r = backend.client.post(f"/v1/devices/{backend.device_id}/protection",
                            json={"action": "stop", "confirm": True}, headers=backend.auth)
    assert r.status_code == 200
    nonce = a.new_nonce()
    other = backend.client.post("/v1/ingest", json={"events": [], "health": {"device_id": "SD-0A1B2C3D"}},
                                headers={"X-Device-Key": OTHER, "X-SD-Nonce": nonce})
    assert other.status_code == 200 and other.json()["command"] is None, "B never receives A's command"
    # B's key cannot sign as A: the tag is checked with the key A's kid names.
    forged = a.sign_request(a.DeviceKey(OTHER), "POST", "/v1/ingest", BODY, nonce=a.new_nonce())
    forged = forged.replace(f"kid={a.key_id(OTHER)}", f"kid={a.key_id(KEY)}")
    r = _ingest(backend.client, {"Content-Type": "application/json", "X-SD-Auth": forged})
    assert r.status_code == 401 and r.json()["code"] == "device_auth_bad_signature"
    # And an envelope made for B does not verify on A.
    mine = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY, "X-SD-Nonce": nonce})
    envelope = mine.json()["command_auth"]
    with pytest.raises(a.AuthError) as refused:
        a.verify_command(a.Keyring(a.DeviceKey(OTHER)), mine.json()["command"], envelope,
                         devices={"SD-0A1B2C3D"}, request_nonce=nonce)
    assert refused.value.reason == "unknown_key"


# ------------------------------------------------------------------ claim
def test_claim_guessing_is_limited_per_account_across_addresses(backend, monkeypatch):
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    token = backend.client.post("/v1/signup", json={"email": "guesser@example.com", "password": "password1"},
                                headers={"Fly-Client-IP": "198.51.100.200"}).json()["token"]
    guesser = {"Authorization": f"Bearer {token}"}
    statuses = []
    for i in range(25):  # a new address every time: the per-IP limit never trips
        r = backend.client.post("/v1/devices/claim", json={"claim_code": f"00-00-{i:02X}"},
                                headers={**guesser, "Fly-Client-IP": f"203.0.113.{i}"})
        statuses.append(r.status_code)
    assert statuses[:20] == [404] * 20 and set(statuses[20:]) == {429}
    # Even the right code is refused for the rest of the window: the limit is
    # on the account, not on wrong answers only.
    r = backend.client.post("/v1/devices/claim", json={"claim_code": "DD-EE-FF"},
                            headers={**guesser, "Fly-Client-IP": "203.0.113.99"})
    assert r.status_code == 429


def test_a_successful_claim_is_not_a_failure_and_another_account_is_refused(backend):
    from cloud.ratelimit import CLAIM_FAILURE_LIMITER

    for _ in range(5):
        assert backend.client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"},
                                   headers=backend.auth).status_code == 200
    assert len(CLAIM_FAILURE_LIMITER) == 0
    token = backend.client.post("/v1/signup", json={"email": "b@example.com", "password": "password1"}).json()["token"]
    other = {"Authorization": f"Bearer {token}"}
    r = backend.client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=other)
    assert r.status_code == 409 and r.json()["code"] == "device_already_claimed"
    assert [d["id"] for d in backend.client.get("/v1/devices", headers=backend.auth).json()["devices"]] \
        == [backend.device_id]
    assert backend.client.get("/v1/devices", headers=other).json()["devices"] == []


# ------------------------------------------------------------------ backend secrets at rest
@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX modes")
def test_an_existing_jwt_secret_file_is_made_owner_only(tmp_path, monkeypatch):
    secret = tmp_path / ".jwt_secret"
    secret.write_text("s" * 64)
    os.chmod(secret, 0o644)
    app, _ = _boot(tmp_path, monkeypatch)
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert app.JWT_SECRET == "s" * 64, "the existing secret is kept, so sessions survive"
    for path in (tmp_path / "test.db",):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_auth_refusals_never_log_a_key_or_tag(backend, caplog):
    import cloud.manage as manage

    caplog.set_level("DEBUG")
    manage.rotate_key(backend.device_id)
    manage.revoke_previous_key(backend.device_id)
    headers = _signed_headers(key=KEY)
    _ingest(backend.client, headers)
    backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    _ingest(backend.client, _signed_headers(key=OTHER, ts=1))
    for secret in (KEY, headers["X-SD-Auth"].split("sig=")[1], a.DeviceKey(OTHER)._command.hex()):
        assert secret not in caplog.text
    assert "revoked_key" in caplog.text
