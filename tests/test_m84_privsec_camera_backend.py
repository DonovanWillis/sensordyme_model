"""M8.4 owner Decision C, backend side: the camera-only setting belongs to the owner.

POST /v1/devices/{id}/camera-mode is the only way to turn camera-only on:
owner-scoped (another account gets the same 404 as a missing device),
confirmation required to turn it ON, refused with `firmware_unsupported` for a
device whose software cannot honour it, delivered as a signed
`set_camera_mode` command until the device REPORTS the same value, and shown
to the app as what the device reports (status.camera_mode). A database from
before M8.4 reads as OFF, and a newly claimed device never inherits camera
access someone else turned on.

The end-to-end tests run the real runtime + uplink + backend (sim/).
"""

from __future__ import annotations

import importlib
import json
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core import command_auth as a

KEY_A = "dk_" + "a1" * 24
KEY_B = "dk_" + "b2" * 24
HW = "SD-0A1B2C3D"


def _camera_block(camera_only: bool, remote: str = "monitoring", command_id=None) -> dict:
    return {"schema": 1, "camera_only": camera_only, "remote_stream": remote, "local_while_stopped": True,
            "since": None, "last_command_id": command_id, "persisted": True}


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
    b.app, b.client, b.auth, b.ids = app, client, auth, ids
    return b


def _ingest(b, camera: dict | None, key: str = KEY_A, nonce: str | None = None) -> dict:
    monitoring = {"protection": {"state": "protecting"}}
    if camera is not None:
        monitoring["camera"] = camera
    headers = {"X-Device-Key": key}
    if nonce is not None:
        headers["X-SD-Nonce"] = nonce
    reply = b.client.post("/v1/ingest", json={"events": [], "health": {"device_id": HW}, "monitoring": monitoring},
                          headers=headers)
    assert reply.status_code == 200, reply.text
    return reply.json()


def _status(b, who: str = "alice") -> dict:
    devices = b.client.get("/v1/devices", headers=b.auth[who]).json()["devices"]
    return devices[0]["status"]


def _set(b, who: str, device: int, camera_only: bool, confirm: bool = True):
    return b.client.post(f"/v1/devices/{device}/camera-mode", json={"camera_only": camera_only, "confirm": confirm},
                         headers=b.auth[who])


# ------------------------------------------------------------------ authorization
def test_only_the_owner_can_change_it(backend):
    b = backend
    _ingest(b, _camera_block(False))
    other = _set(b, "bob", b.ids["alice"], True)
    missing = _set(b, "bob", 987654, True)
    assert other.status_code == 404 and other.json()["detail"] == missing.json()["detail"]
    anonymous = b.client.post(f"/v1/devices/{b.ids['alice']}/camera-mode", json={"camera_only": True, "confirm": True})
    assert anonymous.status_code == 401
    with b.app.db() as conn:
        stored = json.loads(conn.execute("SELECT camera_mode FROM devices WHERE id = ?",
                                         (b.ids["alice"],)).fetchone()[0])
    assert stored["camera_only"] is False  # the claim-time OFF, untouched by Bob
    assert "set_camera_mode" not in (_ingest(b, _camera_block(False))["command"] or {})
    # The device key is not an owner credential.
    as_device = b.client.post(f"/v1/devices/{b.ids['alice']}/camera-mode",
                              json={"camera_only": True, "confirm": True}, headers={"X-Device-Key": KEY_A})
    assert as_device.status_code == 401


def test_turning_it_on_must_be_confirmed(backend):
    b = backend
    _ingest(b, _camera_block(False))
    refused = _set(b, "alice", b.ids["alice"], True, confirm=False)
    assert refused.status_code == 400 and refused.json()["code"] == "confirmation_required"
    assert _set(b, "alice", b.ids["alice"], False, confirm=False).status_code == 200  # off needs no confirmation


def test_firmware_that_cannot_honour_it_is_refused_explicitly(backend):
    b = backend
    _ingest(b, None)  # an M8.3 device: monitoring without a camera block
    refused = _set(b, "alice", b.ids["alice"], True)
    assert refused.status_code == 409 and refused.json()["code"] == "firmware_unsupported"
    status = _status(b)["camera_mode"]
    assert status["supported"] is False and status["camera_only"] is None and status["remote_stream"] is None


# ------------------------------------------------------------------ delivery + reported state
def test_delivered_signed_until_the_device_reports_it_and_the_app_sees_the_device_truth(backend):
    b = backend
    _ingest(b, _camera_block(False))
    assert _status(b)["camera_mode"] == {
        "supported": True, "camera_only": False, "remote_stream": "monitoring", "desired_camera_only": False,
        "pending": False, "command_id": _status(b)["camera_mode"]["command_id"], "stale": False}
    command_id = _set(b, "alice", b.ids["alice"], True).json()["command_id"]
    # Not applied until the device says so.
    status = _status(b)["camera_mode"]
    assert status["camera_only"] is False and status["desired_camera_only"] is True and status["pending"] is True

    nonce = a.new_nonce()
    reply = _ingest(b, _camera_block(False), nonce=nonce)
    assert reply["command"]["set_camera_mode"] == {"camera_only": True, "id": command_id}
    envelope = reply["command_auth"]
    a.verify_command(a.Keyring(a.DeviceKey(KEY_A)), reply["command"], envelope, devices={HW}, request_nonce=nonce)
    assert "set_camera_mode" in envelope["types"]

    reply = _ingest(b, _camera_block(True, "camera_only", command_id))
    assert "set_camera_mode" not in (reply["command"] or {})
    status = _status(b)["camera_mode"]
    assert status["camera_only"] is True and status["pending"] is False and status["remote_stream"] == "camera_only"


def test_a_device_that_lost_its_setting_is_brought_back_to_the_owner_choice(backend):
    b = backend
    _ingest(b, _camera_block(False))
    _set(b, "alice", b.ids["alice"], False)
    # The device reports ON (e.g. a state file restored from elsewhere): the owner said off.
    reply = _ingest(b, _camera_block(True, "camera_only"))
    assert reply["command"]["set_camera_mode"]["camera_only"] is False
    assert _status(b)["camera_mode"]["pending"] is True and _status(b)["camera_mode"]["camera_only"] is True


def test_never_sent_to_firmware_that_does_not_report_a_camera_block(backend):
    b = backend
    _ingest(b, _camera_block(False))
    _set(b, "alice", b.ids["alice"], True)
    assert "set_camera_mode" not in (_ingest(b, None)["command"] or {})


def test_a_pre_m84_database_reads_as_off(backend):
    b = backend
    with b.app.db() as conn:
        conn.execute("UPDATE devices SET camera_mode = NULL")
    _ingest(b, _camera_block(False))
    status = _status(b)["camera_mode"]
    assert status["desired_camera_only"] is False and status["pending"] is False
    assert "set_camera_mode" not in (_ingest(b, _camera_block(False))["command"] or {})


def test_a_new_owner_never_inherits_camera_only(backend):
    b = backend
    with b.app.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Used cam", "CCCC-CCCC-CCCC", "dk_" + "c3" * 24, time.time()))
    claimed = b.client.post("/v1/devices/claim", json={"claim_code": "CCCC-CCCC-CCCC"}, headers=b.auth["bob"])
    device = claimed.json()["device_id"]
    with b.app.db() as conn:
        stored = json.loads(conn.execute("SELECT camera_mode FROM devices WHERE id = ?", (device,)).fetchone()[0])
    assert stored["camera_only"] is False and stored["id"]
    # A device still holding ON from a previous life is told OFF on its first report.
    reply = _ingest(b, _camera_block(True, "camera_only"), key="dk_" + "c3" * 24)
    assert reply["command"]["set_camera_mode"]["camera_only"] is False


def test_stale_reports_do_not_claim_what_the_stream_does_now(backend, monkeypatch):
    b = backend
    _ingest(b, _camera_block(False, "off_while_stopped"))
    with b.app.db() as conn:
        conn.execute("UPDATE devices SET last_seen = ?", (time.time() - 3600,))
    status = _status(b)["camera_mode"]
    assert status["stale"] is True and status["remote_stream"] is None and status["camera_only"] is False


# ------------------------------------------------------------------ end to end (real runtime + uplink)
def test_end_to_end_the_owner_turns_camera_only_on_and_it_survives_a_restart(tmp_path):
    from sim.harness import SimStack
    from sim.scenario import CALIBRATION_S, Scenario, Segment

    scenario = Scenario("attentive", "attentive", (Segment(0.0),))
    with SimStack(tmp_path) as stack:
        device = stack.boot(scenario)
        stack.claim()
        device.advance(CALIBRATION_S + 4.0)
        camera = stack.status()["camera_mode"]
        assert camera["supported"] is True and camera["camera_only"] is False  # the default

        reply = stack.api.post(f"/v1/devices/{stack.device_id}/camera-mode",
                               json={"camera_only": True, "confirm": True}, headers=stack.headers)
        assert reply.status_code == 200, reply.text
        device.advance(6.0)
        camera = stack.status()["camera_mode"]
        assert camera["camera_only"] is True and camera["pending"] is False

        device = stack.restart(scenario)
        device.advance(6.0)
        assert stack.status()["camera_mode"]["camera_only"] is True, "the owner's choice did not survive a restart"

        stack.api.post(f"/v1/devices/{stack.device_id}/camera-mode", json={"camera_only": False},
                       headers=stack.headers)
        device.advance(6.0)
        assert stack.status()["camera_mode"]["camera_only"] is False
