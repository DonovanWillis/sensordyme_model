"""M12 end to end on the virtual device: the phone reaches the camera with the backend gone.

The real runtime (SimRuntime over services/runtime.py) + the real backend
(cloud/app.py) + a Python phone that speaks sd-local-v1 exactly as the iOS app
does (sim/local_client.py):

1. The owner, signed in, registers the phone's PUBLIC key with the account.
2. The backend relays it to the device as a SIGNED command; the device
   reports trusting it; the app gets the device's public key from the account.
3. The network to the backend goes DOWN.
4. On the LAN alone, the phone opens a device-signed session, reads state,
   stops monitoring with a signed request (confirmed from /api/state, not
   assumed), resumes it, and reads the local event history -- the same
   event_key the backend stores, marked "pending" while it cannot be sent.
5. The network returns: the backend converges on what the device did.

Simulated: no Wi-Fi radio, no iPhone, no Pi. See docs/LOCAL_ACCESS.md §10.
"""

from __future__ import annotations

import pytest

from core import local_auth as la
from sim import scenario as sc
from sim.harness import SimStack
from sim.local_client import LocalPhone
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _scenario(down_at: float, up_at: float) -> Scenario:
    return Scenario("m12-local", "backend lost, LAN kept", (
        Segment(0.0, ATTENTIVE, Faults()),
        Segment(down_at, ATTENTIVE, Faults(network_down=True)),
        Segment(down_at + 4.0, ASLEEP, Faults(network_down=True)),
        Segment(down_at + 14.0, ATTENTIVE, Faults(network_down=True)),
        Segment(up_at, ATTENTIVE, Faults()),
    ))


def test_the_phone_runs_the_camera_over_the_lan_with_the_backend_unreachable(stack):
    down_at = C + 20.0
    up_at = down_at + 60.0
    device = stack.boot(_scenario(down_at, up_at), dashboard_port=0)
    stack.claim()
    t0 = stack.clock.monotonic()

    def advance_to(t: float) -> None:  # seconds since boot
        device.advance(max(0.0, t - (stack.clock.monotonic() - t0)))

    device.advance(C + 5.0)
    url = device.runtime.dashboard.url
    phone = LocalPhone(url, name="Owner iPhone")

    # Not yet trusted: the LAN alone opens nothing.
    assert phone.open_session().status == 403

    # 1-2. Registered through the account, relayed signed, reported by the device.
    reply = stack.api.post(f"/v1/devices/{stack.device_id}/local-clients",
                           json={"public_key": la.b64u(phone.key.public_key), "name": "Owner iPhone"},
                           headers=stack.headers)
    assert reply.status_code == 200, reply.text
    device.advance(6.0)
    assert phone.key.client_id in device.runtime.local_authority.report()["clients"]
    identity = stack.row()["local_identity"]
    assert identity["device_public_key"] == la.b64u(device.runtime.local_authority.identity.public_key)
    phone.trust({"device_public_key": identity["device_public_key"], "device_id": device.runtime._local_device_id()})
    listed = stack.api.get(f"/v1/devices/{stack.device_id}/local-clients", headers=stack.headers).json()
    assert listed["clients"][0]["on_device"] is True

    # 3. The backend becomes unreachable.
    advance_to(down_at + 1.0)
    assert device.runtime.uplink.connected is False

    # 4. LAN only: a device-signed session, live state, a signed stop confirmed by the device.
    assert phone.open_session().status == 200
    state = phone.get("/api/state")
    assert state.status == 200
    assert phone.signed("POST", "/api/protection", {"action": "stop", "confirm": True}).status == 202
    device.advance(2.0)
    protection = phone.get("/api/state").body["policy"]["protection"]
    assert protection["state"] == "stopped" and protection["by"] == "local_app"
    assert phone.signed("POST", "/api/protection", {"action": "resume"}).status == 202
    device.advance(C + 2.0)
    assert phone.get("/api/state").body["policy"]["protection"]["state"] != "stopped"

    # The drowsy segment raised an alert while offline: it is in the LOCAL history,
    # waiting to be backed up.
    advance_to(down_at + 18.0)
    local = phone.get("/api/events?limit=50").body
    alerts = [e for e in local["events"] if e["type"] == "drowsiness"]
    assert alerts, [e["type"] for e in local["events"]]
    assert alerts[0]["sync"] == "pending" and alerts[0]["simulated"] is True
    key = alerts[0]["event_key"]

    # 5. The network returns: the same event, once, under the same key; the backend
    # converges on what the device did locally.
    advance_to(up_at + 8.0)
    stored = [e for e in stack.events() if e.get("event_key") == key]
    assert len(stored) == 1
    after = {e["event_key"]: e["sync"] for e in phone.get("/api/events?limit=50").body["events"]}
    assert after[key] == "sent"
    assert stack.status()["monitoring_state"] != "stopped"


def test_a_revoked_phone_is_cut_off_through_the_account(stack):
    device = stack.boot(sc.get("attentive"), dashboard_port=0)
    stack.claim()
    device.advance(C)
    phone = LocalPhone(device.runtime.dashboard.url)
    stack.api.post(f"/v1/devices/{stack.device_id}/local-clients",
                   json={"public_key": la.b64u(phone.key.public_key)}, headers=stack.headers)
    device.advance(6.0)
    identity = stack.row()["local_identity"]
    phone.trust({"device_public_key": identity["device_public_key"], "device_id": device.runtime._local_device_id()})
    assert phone.open_session().status == 200
    stack.api.delete(f"/v1/devices/{stack.device_id}/local-clients/{phone.key.client_id}", headers=stack.headers)
    device.advance(6.0)
    after = phone.get("/api/state")
    assert after.status == 401 and after.body["code"] == "session_invalid"
    assert phone.open_session().body["code"] == "unknown_client"
