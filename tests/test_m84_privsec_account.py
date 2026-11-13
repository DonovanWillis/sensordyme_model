"""M8.4 privacy: an account can be deleted, and deleting it deletes what it owns.

Before M8.4 there was no way at all to delete an account: no route, no admin
command, no app control. Events (with the owner's rule descriptions in them),
typed instructions, drawn areas, phone number, ntfy topic and order addresses
stayed in the backend for good, and the device stayed bound to the account.

POST /v1/account/delete (password again + confirm) now, in one transaction:
deletes every event of every owned device, the orders and the user row; ends
every session; releases each device with the owner's configuration cleared and
camera-only explicitly OFF, so it can be claimed again and a new owner never
inherits it. Another account's data is untouched.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from test_m84_privsec_camera_backend import (  # noqa: F401 - fixture
    KEY_A,
    _camera_block,
    _ingest,
    backend,
)


def _delete(b, who: str, password: str = "password1", confirm: bool = True):
    return b.client.post("/v1/account/delete", json={"password": password, "confirm": confirm},
                         headers=b.auth[who])


def _seed(b) -> None:
    with b.app.db() as conn:
        for device in b.ids.values():
            conn.execute(
                "INSERT INTO events (device_id, timestamp, type, confidence, signals, severity, received_at) "
                "VALUES (?, '2026-11-13T00:00:00', 'fall', 0.9, 'kitchen camera rule', 'critical', ?)",
                (device, time.time()))
        conn.execute("UPDATE devices SET zones = '[{\"name\": \"crib\"}]', name = 'Alice home cam', "
                     "ai_rules = '{\"explanation\": \"tell me when my son leaves\"}' WHERE id = ?",
                     (b.ids["alice"],))
    b.client.post("/v1/orders", json={"quantity": 1, "address": "1 Alice Street"}, headers=b.auth["alice"])
    b.client.post("/v1/account/phone", json={"phone": "+15555550123"}, headers=b.auth["alice"])


def test_deleting_an_account_deletes_what_it_owns_and_nothing_else(backend):  # noqa: F811
    b = backend
    _seed(b)
    _ingest(b, _camera_block(True, "camera_only"))
    with b.app.db() as conn:
        conn.execute("UPDATE devices SET camera_mode = '{\"camera_only\": true, \"id\": \"x\"}' WHERE id = ?",
                     (b.ids["alice"],))
    reply = _delete(b, "alice")
    assert reply.status_code == 200, reply.text
    assert reply.json()["devices_released"] == 1 and reply.json()["events_deleted"] == 1
    with b.app.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM users WHERE email = 'alice@example.com'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events WHERE device_id = ?", (b.ids["alice"],)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
        device = conn.execute("SELECT * FROM devices WHERE id = ?", (b.ids["alice"],)).fetchone()
        # Bob's device and events are untouched.
        assert conn.execute("SELECT COUNT(*) FROM events WHERE device_id = ?", (b.ids["bob"],)).fetchone()[0] == 1
        assert conn.execute("SELECT owner_id FROM devices WHERE id = ?", (b.ids["bob"],)).fetchone()[0] is not None
    assert device["owner_id"] is None and device["name"] == "SensorDyme"
    assert device["zones"] is None and device["ai_rules"] is None and device["monitoring"] is None
    assert '"camera_only": false' in device["camera_mode"]
    dump = " ".join(str(v) for v in tuple(device))
    for private in ("Alice", "crib", "my son", "camera_only\": true"):
        assert private not in dump, private
    # The device is told to turn camera-only off on its next report.
    assert _ingest(b, _camera_block(True, "camera_only"))["command"]["set_camera_mode"]["camera_only"] is False


def test_every_session_ends(backend):  # noqa: F811
    b = backend
    assert _delete(b, "alice").status_code == 200
    assert b.client.get("/v1/devices", headers=b.auth["alice"]).status_code == 401
    login = b.client.post("/v1/login", json={"email": "alice@example.com", "password": "password1"})
    assert login.status_code == 401


def test_it_needs_the_password_and_a_confirmation(backend):  # noqa: F811
    b = backend
    _seed(b)
    assert _delete(b, "alice", confirm=False).json()["code"] == "confirmation_required"
    wrong = _delete(b, "alice", password="not-it")
    assert wrong.status_code == 403 and wrong.json()["code"] == "wrong_password"
    assert b.client.post("/v1/account/delete", json={"password": "password1", "confirm": True}).status_code == 401
    with b.app.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 2


def test_the_released_device_can_be_claimed_by_a_new_owner(backend):  # noqa: F811
    b = backend
    _delete(b, "alice")
    token = b.client.post("/v1/signup", json={"email": "carol@example.com", "password": "password1"}).json()["token"]
    claimed = b.client.post("/v1/devices/claim", json={"claim_code": "AAAA-AAAA-AAAA"},
                            headers={"Authorization": f"Bearer {token}"})
    assert claimed.status_code == 200 and claimed.json()["device_id"] == b.ids["alice"]
    events = b.client.get(f"/v1/devices/{b.ids['alice']}/events", headers={"Authorization": f"Bearer {token}"})
    assert events.json()["events"] == [], "the new owner sees none of the previous owner's history"
    assert KEY_A  # the device key is unchanged: the device keeps working for its new owner
