"""sd-auth-v1 end to end: the real CloudUplink against the real cloud/app.py.

Compatibility (a legacy request gets the same response as before), each
device/backend mode pairing, replay / tamper / clock refusals, rotation and
revocation through cloud/manage.py, and that no key or tag is logged.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import types

import pytest

pytest.importorskip("fastapi")
import requests
from fastapi.testclient import TestClient

from core import command_auth as a
from core.uplink import CloudUplink

KEY = "dk_" + "66" * 24
HEALTH = {"device_id": "SD-0A1B2C3D", "state": "ok"}


class _Requests:
    """requests.post over a TestClient, keeping what the device sent."""

    def __init__(self, client):
        self.client, self.sent = client, []

    def post(self, url, json=None, data=None, headers=None, **_kwargs):
        path = url.split("://", 1)[-1].split("/", 1)[1]
        self.sent.append({"json": json, "data": data, "headers": dict(headers or {})})
        if data is not None:
            resp = self.client.post("/" + path, content=data, headers=headers)
        else:
            resp = self.client.post("/" + path, json=json, headers=headers)

        def raise_for_status():
            if resp.status_code >= 400:
                raise requests.HTTPError(str(resp.status_code), response=resp)
        return types.SimpleNamespace(status_code=resp.status_code, json=resp.json,
                                     raise_for_status=raise_for_status, headers=resp.headers)


def _backend(tmp_path, monkeypatch, mode="permissive"):
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
    client = TestClient(app.app, raise_server_exceptions=False)
    with app.db() as conn:  # inserted the way older tools do: no auth_kid (lazy backfill)
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Unit", "AA-BB-CC", KEY, time.time()))
    r = client.post("/v1/signup", json={"email": "owner@example.com", "password": "password1"})
    auth = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=auth).status_code == 200
    with app.db() as conn:
        device_id = conn.execute("SELECT id FROM devices").fetchone()[0]
    return types.SimpleNamespace(app=app, client=client, auth=auth, device_id=device_id)


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    return _backend(tmp_path, monkeypatch)


def _device(backend, **cloud):
    uplink = CloudUplink({"base_url": "http://localhost:8787", "device_key": KEY, "poll_interval_s": 0, **cloud})
    uplink._requests = _Requests(backend.client)
    uplink.set_health(HEALTH)
    return uplink


def _request_stop(backend):
    r = backend.client.post(f"/v1/devices/{backend.device_id}/protection", json={"action": "stop", "confirm": True},
                            headers=backend.auth)
    assert r.status_code == 200, r.text
    return r.json()["command_id"]


# ------------------------------------------------------------------ compatibility
def test_a_legacy_request_gets_the_legacy_response(backend):
    _request_stop(backend)
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    assert r.status_code == 200 and set(r.json()) == {"ok", "stored", "command"}
    assert r.json()["command"]["set_protection"]["action"] == "stop"


@pytest.mark.parametrize("mode", ["off", "permissive", "required"])
def test_every_device_mode_works_against_the_default_backend(backend, mode):
    command_id = _request_stop(backend)
    device = _device(backend, command_auth=mode)
    device.sync_once()
    assert device.connected
    assert device.pop_pending_protection() == {"action": "stop", "confirm": True, "id": command_id}
    assert device.command_auth_rejections == 0
    sent = device._requests.sent[-1]["headers"]
    assert ("X-Device-Key" in sent) == (mode != "required")


def test_the_envelope_is_addressed_to_the_reported_hardware_id(backend):
    _request_stop(backend)
    device = _device(backend)
    seen = {}
    original = device._authenticated_command

    def spy(data, command, nonce, health):
        seen.update(data)
        return original(data, command, nonce, health)
    device._authenticated_command = spy
    device.sync_once()
    assert seen["command_auth"]["dev"] == "SD-0A1B2C3D" and seen["command_auth"]["kid"] == a.key_id(KEY)
    device.set_health(None)
    _request_stop(backend)
    device.sync_once()
    assert seen["command_auth"]["dev"] == f"kid:{a.key_id(KEY)}"


# ------------------------------------------------------------------ backend refusals
def _signed_post(backend, body=b'{"events": []}', ts=None, nonce=None, key=KEY):
    header = a.sign_request(a.DeviceKey(key), "POST", "/v1/ingest", body, nonce=nonce or a.new_nonce(),
                            issued_at=int(time.time()) if ts is None else ts)
    return backend.client.post("/v1/ingest", content=body,
                               headers={"Content-Type": "application/json", "X-SD-Auth": header}), header


def test_a_replayed_request_is_refused(backend):
    first, header = _signed_post(backend)
    assert first.status_code == 200
    again = backend.client.post("/v1/ingest", content=b'{"events": []}',
                                headers={"Content-Type": "application/json", "X-SD-Auth": header})
    assert again.status_code == 401 and again.json()["code"] == "device_auth_replayed_nonce"


def test_a_tampered_body_is_refused(backend):
    _, header = _signed_post(backend)
    r = backend.client.post("/v1/ingest", content=b'{"events": [], "active_preset": "x"}',
                            headers={"Content-Type": "application/json", "X-SD-Auth": header.replace(
                                header.split("nonce=")[1][:32], a.new_nonce())})
    assert r.status_code == 401 and r.json()["code"] == "device_auth_bad_signature"


@pytest.mark.parametrize("header", ["sd-auth-v1 garbage", "sd-auth-v2 kid=0123456789abcdef,ts=1,nonce="
                                    + "0" * 32 + ",sig=" + "0" * 64])
def test_a_malformed_or_future_header_is_refused(backend, header):
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-SD-Auth": header})
    assert r.status_code == 401 and r.json()["code"].startswith("device_auth_")


def test_an_unknown_key_is_refused(backend):
    r, _ = _signed_post(backend, key="dk_" + "77" * 24)
    assert r.status_code == 401 and r.json()["code"] == "device_auth_unknown_key"


def test_clock_skew_is_refused_with_the_server_time_and_the_device_recovers(backend):
    r, _ = _signed_post(backend, ts=int(time.time()) - 3600)
    assert r.status_code == 401 and r.json()["code"] == "clock_skew"
    assert abs(r.json()["server_time"] - time.time()) < 5
    device = _device(backend, command_auth="required")
    device._clock_offset_s = -7200.0  # a Pi that booted with an old clock
    device.sync_once()
    assert not device.connected and abs(device._clock_offset_s) < 5
    device.sync_once()
    assert device.connected


def test_a_required_backend_refuses_the_bearer_key(tmp_path, monkeypatch):
    backend = _backend(tmp_path, monkeypatch, mode="required")
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
    assert r.status_code == 401 and r.json()["code"] == "device_auth_signature_required"
    _request_stop(backend)
    device = _device(backend, command_auth="required")
    device.sync_once()
    assert device.connected and device.pop_pending_protection()["action"] == "stop"


def test_an_off_backend_never_signs(tmp_path, monkeypatch):
    backend = _backend(tmp_path, monkeypatch, mode="off")
    _request_stop(backend)
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY, "X-SD-Nonce": "a" * 32})
    assert "command_auth" not in r.json()
    assert _signed_post(backend)[0].status_code == 401


def test_a_typo_in_the_backend_mode_stops_the_server(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="SENSORDYME_DEVICE_AUTH"):
        _backend(tmp_path, monkeypatch, mode="requried")


# ------------------------------------------------------------------ rotation
def test_rotation_then_revocation(backend, capsys):
    import cloud.manage as manage

    new_key = manage.rotate_key(backend.device_id)
    assert new_key not in capsys.readouterr().err
    with pytest.raises(SystemExit, match="already rotating"):
        manage.rotate_key(backend.device_id)

    old_device = _device(backend, command_auth="required")  # not yet updated
    new_device = _device(backend, command_auth="required", device_key=new_key, previous_device_key=KEY)
    for device in (old_device, new_device):
        _request_stop(backend)
        device.sync_once()
        assert device.connected and device.pop_pending_protection()["action"] == "stop"

    manage.revoke_previous_key(backend.device_id)
    old_device.sync_once()
    assert not old_device.connected
    _request_stop(backend)
    new_device.sync_once()
    assert new_device.connected and new_device.pop_pending_protection()["action"] == "stop"


# ------------------------------------------------------------------ leaks
def test_no_key_or_tag_reaches_the_backend_log(backend, caplog, capsys):
    caplog.set_level("DEBUG")
    device = _device(backend, command_auth="required")
    _request_stop(backend)
    device.sync_once()
    _, header = _signed_post(backend)
    backend.client.post("/v1/ingest", content=b"{}", headers={"Content-Type": "application/json", "X-SD-Auth": header})
    _signed_post(backend, ts=1)
    _signed_post(backend, key="dk_" + "77" * 24)
    logged = caplog.text + capsys.readouterr().out
    key = a.DeviceKey(KEY)
    tags = [a.parse_request_header(s["headers"]["X-SD-Auth"]).sig for s in device._requests.sent]
    for secret in [KEY, key._request.hex(), key._command.hex(), header.split("sig=")[1], *tags]:
        assert secret not in logged
    assert "device auth refused" in caplog.text, "refusals are logged, by reason"


def test_the_signed_response_carries_no_key_material(backend):
    _request_stop(backend)
    nonce = a.new_nonce()
    r = backend.client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY, "X-SD-Nonce": nonce})
    text = json.dumps(r.json())
    key = a.DeviceKey(KEY)
    assert r.json()["command_auth"]["rn"] == nonce
    for secret in (KEY, key._request.hex(), key._command.hex()):
        assert secret not in text
