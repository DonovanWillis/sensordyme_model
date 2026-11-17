"""M12 sd-local-v1, the account path (docs/LOCAL_ACCESS.md §4.2).

The owner's signed-in phone registers its PUBLIC key with the backend; the
backend relays it to the device as a SIGNED `set_local_clients` command until
the device reports trusting it; a revocation is relayed until the device stops
reporting it. The backend never holds anything that would let it act locally.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core import command_auth as a
from core import local_auth as la
from core.uplink import CloudUplink
from sim.local_client import PhoneKey

KEY_A = "dk_" + "a1" * 24
KEY_B = "dk_" + "b2" * 24
HW = "SD-0A1B2C3D"


@pytest.fixture()
def backend(tmp_path, monkeypatch):
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
    client = TestClient(app.app, raise_server_exceptions=False)
    with app.db() as conn:
        for name, code, key in (("Alice cam", "AAAA-AAAA-AAAA", KEY_A), ("Bob cam", "BBBB-BBBB-BBBB", KEY_B)):
            conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                         (name, code, key, time.time()))
    auth, ids = {}, {}
    for who, code in (("alice", "AAAA-AAAA-AAAA"), ("bob", "BBBB-BBBB-BBBB")):
        token = client.post("/v1/signup", json={"email": f"{who}@example.com", "password": "password1"}).json()["token"]
        auth[who] = {"Authorization": f"Bearer {token}"}
        ids[who] = client.post("/v1/devices/claim", json={"claim_code": code}, headers=auth[who]).json()["device_id"]

    class B:
        pass

    b = B()
    b.app, b.client, b.auth, b.ids, b.tmp = app, client, auth, ids, tmp_path
    return b


def _device_authority(tmp_path):
    return la.LocalAuthority(HW, la.DeviceIdentity.load_or_create(tmp_path / "id.pem"),
                             la.ClientStore(tmp_path / "clients.json"))


def _ingest(b, local_access: dict | None, nonce: str | None = None) -> dict:
    health = {"device_id": HW}
    if local_access is not None:
        health["local_access"] = local_access
    headers = {"X-Device-Key": KEY_A}
    if nonce is not None:
        headers["X-SD-Nonce"] = nonce
    reply = b.client.post("/v1/ingest", json={"events": [], "health": health,
                                              "monitoring": {"protection": {"state": "protecting"}}}, headers=headers)
    assert reply.status_code == 200, reply.text
    return reply.json()


def _register(b, key: PhoneKey, who: str = "alice", device: str = "alice"):
    return b.client.post(f"/v1/devices/{b.ids[device]}/local-clients",
                         json={"public_key": la.b64u(key.public_key), "name": "Alice's iPhone"}, headers=b.auth[who])


def test_only_the_owner_can_register_a_phone_and_only_a_real_point(backend):
    b = backend
    key = PhoneKey()
    assert _register(b, key, who="bob", device="alice").status_code == 404
    bad = b.client.post(f"/v1/devices/{b.ids['alice']}/local-clients", json={"public_key": "AAAA"},
                        headers=b.auth["alice"])
    assert bad.status_code == 400
    ok = _register(b, key)
    assert ok.status_code == 200 and ok.json()["client_id"] == key.client_id and ok.json()["on_device"] is False


def test_a_registered_phone_is_relayed_signed_until_the_device_trusts_it(backend):
    b = backend
    authority = _device_authority(b.tmp)
    key = PhoneKey()
    _register(b, key)
    nonce = a.new_nonce()
    reply = _ingest(b, authority.report(), nonce=nonce)
    change = reply["command"]["set_local_clients"]
    assert change == {"add": [{"public_key": la.b64u(key.public_key), "name": "Alice's iPhone"}], "revoke": []}
    a.verify_command(a.Keyring(a.DeviceKey(KEY_A)), reply["command"], reply["command_auth"], devices={HW},
                     request_nonce=nonce)
    assert "set_local_clients" in reply["command_auth"]["types"]
    authority.apply_remote(change)
    reply = _ingest(b, authority.report())
    assert "set_local_clients" not in (reply["command"] or {})
    listed = b.client.get(f"/v1/devices/{b.ids['alice']}/local-clients", headers=b.auth["alice"]).json()
    assert listed["clients"][0]["on_device"] is True
    # The app gets the device's PUBLIC key with the device row, even once health is stale.
    devices = b.client.get("/v1/devices", headers=b.auth["alice"]).json()["devices"]
    assert devices[0]["local_identity"]["device_public_key"] == la.b64u(authority.identity.public_key)


def test_a_revocation_is_relayed_until_the_device_stops_reporting_the_phone(backend):
    b = backend
    authority = _device_authority(b.tmp)
    key = PhoneKey()
    _register(b, key)
    authority.apply_remote(_ingest(b, authority.report())["command"]["set_local_clients"])
    assert key.client_id in authority.report()["clients"]
    assert b.client.delete(f"/v1/devices/{b.ids['alice']}/local-clients/{key.client_id}",
                           headers=b.auth["alice"]).status_code == 200
    change = _ingest(b, authority.report())["command"]["set_local_clients"]
    assert change == {"add": [], "revoke": [key.client_id]}
    authority.apply_remote(change)
    assert key.client_id not in authority.report()["clients"]
    assert "set_local_clients" not in (_ingest(b, authority.report())["command"] or {})


def test_never_sent_to_firmware_without_local_access(backend):
    b = backend
    _register(b, PhoneKey())
    assert "set_local_clients" not in (_ingest(b, None)["command"] or {})


def test_the_device_refuses_an_unsigned_add_but_honours_an_unsigned_revoke(tmp_path):
    authority = _device_authority(tmp_path)
    owner = PhoneKey()
    authority.enroll(la.b64u(owner.public_key), "owner", "ble")
    uplink = CloudUplink({"base_url": "http://127.0.0.1:9", "device_key": KEY_A})
    uplink.local_authority = authority
    stranger = PhoneKey()
    uplink._command_verified = False
    uplink._accept_local_clients({"add": [{"public_key": la.b64u(stranger.public_key), "name": "x"}],
                                  "revoke": []})
    assert stranger.client_id not in authority.report()["clients"]
    assert uplink._last_rejection == "local_clients_require_signed_command"
    uplink._command_verified = True
    uplink._accept_local_clients({"add": [{"public_key": la.b64u(stranger.public_key), "name": "x"}]})
    assert stranger.client_id in authority.report()["clients"]
    uplink._command_verified = False
    uplink._accept_local_clients({"revoke": [stranger.client_id]})
    assert stranger.client_id not in authority.report()["clients"]
    assert owner.client_id in authority.report()["clients"]


def test_pending_event_keys_are_what_the_backend_has_not_acknowledged():
    from datetime import UTC, datetime

    from core.events import Event

    uplink = CloudUplink({"base_url": "http://127.0.0.1:9", "device_key": KEY_A})
    event = Event(datetime.now(UTC), "distraction", 2.0, 0.8, "driver_attention")
    uplink.push_event(event, "loud")
    assert uplink.pending_event_keys() == {event.event_id}


def test_a_new_owner_does_not_inherit_the_previous_owners_phones(backend):
    b = backend
    authority = _device_authority(b.tmp)
    key = PhoneKey()
    _register(b, key)
    authority.apply_remote(_ingest(b, authority.report())["command"]["set_local_clients"])
    with b.app.db() as conn:  # the camera changes hands: unowned again, then claimed by someone else
        conn.execute("UPDATE devices SET owner_id = NULL WHERE id = ?", (b.ids["alice"],))
    token = b.client.post("/v1/signup", json={"email": "carol@example.com", "password": "password1"}).json()["token"]
    carol = {"Authorization": f"Bearer {token}"}
    assert b.client.post("/v1/devices/claim", json={"claim_code": "AAAA-AAAA-AAAA"}, headers=carol).status_code == 200
    change = _ingest(b, authority.report())["command"]["set_local_clients"]
    assert key.client_id in change["revoke"]
