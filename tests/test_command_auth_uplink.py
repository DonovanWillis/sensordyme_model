"""The device side of sd-auth-v1 (core/uplink.py): what each cloud.command_auth mode
sends, which commands it applies, and that nothing secret reaches the log."""

from __future__ import annotations

import json
import types

import pytest
import requests

from core import command_auth as a
from core.uplink import CloudUplink

KEY = "dk_" + "33" * 24
OLD = "dk_" + "44" * 24
STOP = {"set_protection": {"action": "stop", "confirm": True, "id": "p1"}}
HEALTH = {"device_id": "SD-0A1B2C3D", "state": "ok"}


class _Backend:
    """A fake v1 backend: records each request; answers with `reply(nonce, kid)`."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def post(self, url, json=None, data=None, headers=None, **_kwargs):
        headers = dict(headers or {})
        self.requests.append({"url": url, "json": json, "data": data, "headers": headers})
        if "X-SD-Auth" in headers:
            auth = a.parse_request_header(headers["X-SD-Auth"])
            nonce, kid = auth.nonce, auth.kid
        else:
            nonce, kid = headers.get("X-SD-Nonce"), None
        body = self.reply(nonce, kid)
        if isinstance(body, Exception):
            raise body
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: body)


def _uplink(reply, **cloud):
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": KEY, "poll_interval_s": 0, **cloud})
    backend = _Backend(reply)
    uplink._requests = types.SimpleNamespace(post=backend.post)
    uplink.set_health(HEALTH)
    return uplink, backend


def _signed(command, key=KEY, device="SD-0A1B2C3D", rn_override=None):
    def reply(nonce, _kid):
        envelope = a.sign_command(a.DeviceKey(key), command, device=device, request_nonce=rn_override or nonce)
        return {"command": command, "command_auth": envelope}
    return reply


def _unsigned(command):
    return lambda _nonce, _kid: {"command": command}


# ------------------------------------------------------------------ modes on the wire
def test_off_is_the_legacy_request_byte_for_byte():
    uplink, backend = _uplink(_unsigned(STOP), command_auth="off")
    uplink.sync_once()
    sent = backend.requests[0]
    assert sent["headers"] == {"X-Device-Key": KEY} and sent["data"] is None and sent["json"]["health"] == HEALTH
    assert uplink.pop_pending_protection()["action"] == "stop"


def test_permissive_is_the_default_and_adds_only_a_nonce():
    uplink, backend = _uplink(_unsigned(STOP))
    assert uplink.command_auth_mode == "permissive"
    uplink.sync_once()
    headers = backend.requests[0]["headers"]
    assert headers["X-Device-Key"] == KEY and set(headers) == {"X-Device-Key", "X-SD-Nonce"}
    assert len(headers["X-SD-Nonce"]) == 32


def test_required_never_sends_the_key_and_signs_the_exact_bytes():
    uplink, backend = _uplink(_signed(STOP), command_auth="required")
    uplink.sync_once()
    sent = backend.requests[0]
    assert "X-Device-Key" not in sent["headers"] and sent["json"] is None
    assert KEY not in json.dumps(sent["headers"]) and KEY.encode() not in sent["data"]
    auth = a.parse_request_header(sent["headers"]["X-SD-Auth"])
    ring = a.Keyring(a.DeviceKey(KEY))
    a.verify_request(ring, auth, "POST", "/v1/ingest", sent["data"], nonces=a.NonceCache(), now=auth.issued_at)
    assert json.loads(sent["data"])["health"] == HEALTH
    assert uplink.pop_pending_protection()["action"] == "stop"


def test_a_fresh_nonce_per_request():
    uplink, backend = _uplink(_unsigned({}))
    for _ in range(3):
        uplink.sync_once()
    assert len({r["headers"]["X-SD-Nonce"] for r in backend.requests}) == 3


# ------------------------------------------------------------------ what is applied
@pytest.mark.parametrize("mode", ["permissive", "required"])
def test_a_valid_signed_command_is_applied(mode):
    uplink, _ = _uplink(_signed({"set_preset": "driver_safety"}), command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_preset() == "driver_safety" and uplink.command_auth_rejections == 0


def test_the_kid_address_works_before_the_hardware_id_is_known():
    uplink, _ = _uplink(_signed(STOP, device=f"kid:{a.key_id(KEY)}"), command_auth="required")
    uplink.set_health(None)
    uplink.sync_once()
    assert uplink.pop_pending_protection()["action"] == "stop"


@pytest.mark.parametrize("mode", ["permissive", "required"])
@pytest.mark.parametrize(("reply", "reason"), [
    (_signed(STOP, key="dk_" + "55" * 24), "unknown_key"),
    (_signed(STOP, device="SD-FFFFFFFF"), "wrong_device"),
    (_signed(STOP, rn_override="00" * 16), "stale_response"),
    (lambda n, k: {"command": {"set_protection": {"action": "resume", "id": "p1"}},
                   "command_auth": _signed(STOP)(n, k)["command_auth"]}, "body_mismatch"),
    (lambda n, k: {"command": {**STOP, "set_preset": "x"}, "command_auth": _signed(STOP)(n, k)["command_auth"]},
     "type_mismatch"),
    (lambda n, k: {"command": STOP, "command_auth": {"v": "sd-auth-v9"}}, "unsupported_version"),
    (lambda n, k: {"command": STOP, "command_auth": "forged"}, "malformed"),
])
def test_an_invalid_signed_command_is_never_applied(mode, reply, reason, capsys):
    uplink, _ = _uplink(reply, command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_protection() is None and uplink.pop_pending_preset() is None
    assert uplink.command_auth_rejections == 1
    assert f"({reason})" in capsys.readouterr().out
    assert uplink.connected, "a refused command is not a connectivity failure"


def test_required_refuses_an_unsigned_command():
    uplink, _ = _uplink(_unsigned(STOP), command_auth="required")
    uplink.sync_once()
    assert uplink.pop_pending_protection() is None and uplink.command_auth_rejections == 1


def test_permissive_still_obeys_a_backend_that_does_not_sign_and_says_so_once(capsys):
    uplink, _ = _uplink(_unsigned({"set_preset": "a"}))
    uplink.sync_once()
    uplink.sync_once()
    assert uplink.pop_pending_preset() == "a"
    assert capsys.readouterr().out.count("does not sign") == 1


def test_a_replayed_envelope_is_refused_even_with_its_own_request_nonce():
    captured = {}

    def reply(nonce, kid):
        if "env" not in captured:
            captured["env"] = _signed(STOP)(nonce, kid)
        return captured["env"]
    uplink, _ = _uplink(reply, command_auth="required")
    uplink.sync_once()
    assert uplink.pop_pending_protection() is not None
    uplink.sync_once()  # the same response replayed: answers the previous request
    assert uplink.pop_pending_protection() is None and uplink.command_auth_rejections == 1


def test_a_duplicate_command_id_is_refused():
    def reply(nonce, kid):
        return {"command": STOP, "command_auth": a.sign_command(a.DeviceKey(KEY), STOP, device="SD-0A1B2C3D",
                                                               request_nonce=nonce, command_id="ab" * 12)}
    uplink, _ = _uplink(reply, command_auth="required")
    uplink.sync_once()
    uplink.sync_once()
    assert uplink.command_auth_rejections == 1


def test_rotation_accepts_the_previous_key_until_revoked():
    uplink, _ = _uplink(_signed(STOP, key=OLD), command_auth="required", previous_device_key=OLD)
    uplink.sync_once()
    assert uplink.pop_pending_protection() is not None
    revoked, _ = _uplink(_signed(STOP, key=OLD), command_auth="required", previous_device_key=OLD,
                         revoked_key_ids=[a.key_id(OLD)])
    revoked.sync_once()
    assert revoked.pop_pending_protection() is None and revoked.command_auth_rejections == 1


def test_an_unknown_mode_fails_closed(capsys):
    uplink, _ = _uplink(_unsigned(STOP), command_auth="Permisive")
    assert uplink.command_auth_mode == "required"
    uplink.sync_once()
    assert uplink.pop_pending_protection() is None


def test_required_with_an_unusable_key_sends_nothing():
    uplink, backend = _uplink(_signed(STOP), command_auth="required", device_key="dk_short")
    uplink.sync_once()
    assert backend.requests == [] and not uplink.connected


# ------------------------------------------------------------------ clock
def test_a_clock_skew_refusal_teaches_the_request_clock_only():
    def skew(nonce, _kid):  # the backend signs its time, bound to the refused request (M8)
        answer = {"code": "clock_skew", "server_time": 4_000_000_000,
                  "clock_auth": a.sign_clock(a.DeviceKey(KEY), request_nonce=nonce, server_time=4_000_000_000)}
        return requests.HTTPError("401", response=types.SimpleNamespace(status_code=401, json=lambda: answer))
    uplink, backend = _uplink(skew, command_auth="required")
    uplink.sync_once()
    assert uplink._clock_offset_s > 1e9
    backend.reply = _signed(STOP)
    uplink.sync_once()
    assert a.parse_request_header(backend.requests[-1]["headers"]["X-SD-Auth"]).issued_at >= 4_000_000_000
    assert uplink.pop_pending_protection() is not None


def test_an_unreadable_401_leaves_the_clock_alone():
    response = types.SimpleNamespace(status_code=401, json=lambda: {"code": "device_auth_bad_signature"})
    uplink, _ = _uplink(lambda n, k: requests.HTTPError("401", response=response), command_auth="required")
    uplink.sync_once()
    assert uplink._clock_offset_s == 0.0


# ------------------------------------------------------------------ leaks
@pytest.mark.parametrize("mode", ["off", "permissive", "required"])
def test_no_key_or_tag_reaches_the_log(mode, capsys, caplog):
    key = a.DeviceKey(KEY)
    replies = [_signed(STOP), _signed(STOP, key=OLD), _signed(STOP, rn_override="00" * 16), _unsigned(STOP),
               lambda n, k: requests.ConnectionError("down"),
               lambda n, k: requests.HTTPError("401", response=types.SimpleNamespace(
                   status_code=401, json=lambda: {"code": "clock_skew", "server_time": 1}))]
    uplink, backend = _uplink(replies[0], command_auth=mode, previous_device_key=OLD)
    sigs = []
    for reply in replies:
        backend.reply = reply
        uplink.sync_once()
    for request in backend.requests:
        if "X-SD-Auth" in request["headers"]:
            sigs.append(a.parse_request_header(request["headers"]["X-SD-Auth"]).sig)
    logged = capsys.readouterr().out + caplog.text
    for secret in [KEY, OLD, key._request.hex(), key._command.hex(), *sigs]:
        assert secret not in logged
    assert KEY not in repr(uplink._keyring) and OLD not in repr(uplink._keyring)
