"""M8.5 journey gap-closers, security side (docs/ACCEPTANCE.md journey matrix).

* Journey 27 (account revocation): tests/test_m84_privsec_account.py proves the
  deletion itself backend-only (released device, sessions ended, data gone),
  and tests/test_m8_security_backend.py proves key/device revocation. The gap:
  nothing drove a deleted account's token at a RUNNING device's command
  endpoints. Here the token can neither see nor command the device after the
  deletion, and the device keeps protecting.
* Journey 28 (invalid command rejection): bad signatures, replays and malformed
  commands are covered (tests/test_command_auth*.py, tests/test_m8_security_*);
  the gap was a VALIDLY SIGNED command of an unknown type. The device's command
  dispatch is an allowlist, so a correct signature buys obedience to nothing
  the firmware does not know.
* Journey 7 (device registration): the factory step every test uses is
  sim.backend.provision(), a replica of `cloud/manage.py create-device`; the
  real create_device() itself was untested. Here it provisions a row the stack's
  backend claims by the printed code and authenticates by the printed key.
"""

from __future__ import annotations

import re
import sys

import pytest
from test_command_auth_uplink import _signed, _uplink

from sim import scenario as sc
from sim.backend import DEMO_PASSWORD
from sim.harness import SimStack
from sim.scenario import CALIBRATION_S

pytest.importorskip("fastapi")

C = CALIBRATION_S


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


# ------------------------------------------------------------- journey 27: a deleted account cannot act
def test_a_deleted_accounts_token_can_neither_see_nor_command_and_the_device_keeps_protecting(stack):
    pytest.importorskip("mediapipe")
    device = stack.boot(sc.get("attentive"))
    did = stack.claim()
    device.advance(C)
    assert stack.status()["monitoring_state"] == "ready"

    gone = stack.api.post("/v1/account/delete", json={"password": DEMO_PASSWORD, "confirm": True},
                          headers=stack.headers)
    assert gone.status_code == 200 and gone.json()["devices_released"] == 1, gone.text

    attempts = {
        "devices": stack.api.get("/v1/devices", headers=stack.headers),
        "events": stack.api.get(f"/v1/devices/{did}/events", headers=stack.headers),
        "stop": stack.api.post(f"/v1/devices/{did}/protection",
                               json={"action": "stop", "confirm": True}, headers=stack.headers),
        "preset": stack.api.post(f"/v1/devices/{did}/preset",
                                 json={"preset": "driver_safety:strict"}, headers=stack.headers),
        "zones": stack.api.post(f"/v1/devices/{did}/zones", json={"zones": []}, headers=stack.headers),
        "ai": stack.api.post("/v1/ai/configure", json={"device_id": did, "prompt": "watch the phone"},
                             headers=stack.headers),
    }
    for what, response in attempts.items():
        assert response.status_code == 401, (what, response.status_code, response.text)

    device.advance(6.0)
    assert device.runtime.protection_state()["state"] == "protecting", \
        "revoking the account never turns protection off on the device"
    assert device.runtime._stopped is None


# ------------------------------------------------------------- journey 28: signed but unknown = ignored
@pytest.mark.parametrize("mode", ["permissive", "required"])
def test_a_validly_signed_command_of_an_unknown_type_is_obeyed_by_nothing(mode):
    """The dispatch in core/uplink.py is an allowlist: a correct signature on a
    command type this firmware does not know (`reboot` here) is verified, then
    ignored -- it is not a rejection (nothing was forged) and not an effect."""
    unknown = {"reboot": {"id": "r1", "confirm": True}}
    uplink, backend = _uplink(_signed(unknown), command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_preset() is None and uplink.pop_pending_protection() is None
    assert uplink.pop_pending_nodes() is None and uplink.pop_pending_zones() is None
    assert uplink.pop_pending_rules() is None and uplink.pop_pending_camera_mode() is None
    assert uplink.command_auth_rejections == 0, "verified-but-unknown is ignored, not refused as forged"
    assert uplink.connected and len(backend.requests) == 1


# ------------------------------------------------------------- journey 7: the real factory step
def test_manage_create_device_provisions_a_unit_the_backend_claims_and_authenticates(stack, capsys):
    """cloud/manage.py create-device against the stack's live backend database:
    the printed claim code pairs the unit and the printed key ingests; a wrong
    key is refused. (sim.backend.provision() mirrors this SQL for every other
    test; this pins the real command they mirror.)"""
    sys.modules.pop("cloud.manage", None)  # bind to the stack's cloud.app and its database
    import cloud.manage as manage

    manage.create_device("Factory unit (SIMULATED)")
    out = capsys.readouterr().out
    key_match = re.search(r"\b(dk_[0-9a-f]{48})\b", out)
    code_match = re.search(r"\b([A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4})\b", out)
    assert key_match and code_match, f"create-device must print the key and the claim code: {out!r}"
    key, code = key_match.group(1), code_match.group(1)

    assert stack.api.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": "bogus"}).status_code == 401
    assert stack.api.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": key}).status_code == 200
    claimed = stack.api.post("/v1/devices/claim", json={"claim_code": code}, headers=stack.headers)
    assert claimed.status_code == 200, claimed.text
    with stack.backend.app.db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE id = ?", (claimed.json()["device_id"],)).fetchone()
    from core.command_auth import key_id
    assert row["auth_kid"] == key_id(key), "the factory records the key id sd-auth-v1 recognises"
