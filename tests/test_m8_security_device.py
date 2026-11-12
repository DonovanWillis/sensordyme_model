"""M8 security: the device side of sd-auth-v1 (core/uplink.py, core/command_auth.py).

* A captured signed command replayed after a device restart is refused. The
  device keeps no replay state on disk, and needs none: an envelope is bound to
  the nonce of the request it answers, and a restarted device never sends that
  nonce again. This pins that property.
* A `required` device adopted the time in any clock_skew answer, unsigned, so an
  on-path attacker could make it stamp its signed requests in the future and
  deliver them later (M8 fix: `clock_auth`).
* Compatibility with backends that predate signing.
"""

from __future__ import annotations

import types

import pytest
import requests

from core import command_auth as a
from core.uplink import CloudUplink

KEY = "dk_" + "33" * 24
ATTACKER = "dk_" + "99" * 24
STOP = {"set_protection": {"action": "stop", "confirm": True, "id": "p1"}}
HEALTH = {"device_id": "SD-0A1B2C3D", "state": "ok"}


class _Backend:
    """Answers each request with `reply(nonce)`; keeps every request and answer."""

    def __init__(self, reply):
        self.reply, self.requests, self.answers = reply, [], []

    def post(self, url, json=None, data=None, headers=None, **_kwargs):
        headers = dict(headers or {})
        self.requests.append({"json": json, "data": data, "headers": headers})
        if "X-SD-Auth" in headers:
            nonce = a.parse_request_header(headers["X-SD-Auth"]).nonce
        else:
            nonce = headers.get("X-SD-Nonce")
        body = self.reply(nonce)
        self.answers.append(body)
        if isinstance(body, Exception):
            raise body
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: body)


def _device(reply, **cloud):
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": KEY, "poll_interval_s": 0, **cloud})
    backend = _Backend(reply)
    uplink._requests = types.SimpleNamespace(post=backend.post)
    uplink.set_health(HEALTH)
    return uplink, backend


def _signs(command, key=KEY):
    return lambda nonce: {"command": command, "command_auth": a.sign_command(
        a.DeviceKey(key), command, device="SD-0A1B2C3D", request_nonce=nonce)}


def _skew(answer):
    """A clock_skew refusal with this body."""
    response = types.SimpleNamespace(status_code=401, json=lambda: answer)
    return requests.HTTPError("401", response=response)


# ------------------------------------------------------------------ replay across a device restart
@pytest.mark.parametrize("mode", ["permissive", "required"])
def test_a_captured_signed_command_is_refused_after_a_device_restart(mode, capsys):
    first, backend = _device(_signs(STOP), command_auth=mode)
    first.sync_once()
    assert first.pop_pending_protection()["action"] == "stop"
    captured = backend.answers[-1]

    restarted, _ = _device(lambda _nonce: captured, command_auth=mode)  # same config, fresh process
    restarted.sync_once()
    assert restarted.pop_pending_protection() is None
    assert restarted.command_auth_rejections == 1 and "(stale_response)" in capsys.readouterr().out
    assert restarted.connected, "a refused command is not a connectivity failure"


def test_every_request_after_a_restart_carries_a_new_nonce():
    nonces = []
    for _ in range(2):  # two boots
        device, backend = _device(lambda _n: {"command": None}, command_auth="required")
        device.sync_once()
        device.sync_once()
        nonces += [a.parse_request_header(r["headers"]["X-SD-Auth"]).nonce for r in backend.requests]
    assert len(set(nonces)) == 4


# ------------------------------------------------------------------ clock answers
def test_an_unsigned_clock_answer_is_not_adopted(capsys):
    device, backend = _device(lambda _n: _skew({"code": "clock_skew", "server_time": 4_000_000_000}),
                              command_auth="required")
    device.sync_once()
    device.sync_once()
    assert device._clock_offset_s == 0.0
    out = capsys.readouterr().out
    assert out.count("ignored the backend's clock_skew time (unsigned_clock)") == 1
    backend.reply = _signs(STOP)
    device.sync_once()
    stamped = a.parse_request_header(backend.requests[-1]["headers"]["X-SD-Auth"]).issued_at
    assert stamped < 3_000_000_000, "requests keep the device clock"


@pytest.mark.parametrize(("forge", "reason"), [
    (lambda nonce: a.sign_clock(a.DeviceKey(ATTACKER), request_nonce=nonce, server_time=4_000_000_000), "unknown_key"),
    (lambda nonce: {**a.sign_clock(a.DeviceKey(KEY), request_nonce=nonce, server_time=3_000_000_000),
                    "t": 4_000_000_000}, "bad_signature"),
    (lambda _nonce: a.sign_clock(a.DeviceKey(KEY), request_nonce="0" * 32, server_time=4_000_000_000),
     "stale_response"),
    (lambda _nonce: {"v": "sd-auth-v1", "kid": "x"}, "malformed"),
])
def test_a_forged_or_replayed_clock_answer_is_not_adopted(forge, reason, capsys):
    device, _ = _device(lambda nonce: _skew({"code": "clock_skew", "server_time": 4_000_000_000,
                                             "clock_auth": forge(nonce)}), command_auth="required")
    device.sync_once()
    assert device._clock_offset_s == 0.0
    assert f"({reason})" in capsys.readouterr().out


def test_a_signed_clock_answer_is_adopted_for_request_timestamps_only():
    device, backend = _device(lambda nonce: _skew({
        "code": "clock_skew", "server_time": 4_000_000_000,
        "clock_auth": a.sign_clock(a.DeviceKey(KEY), request_nonce=nonce, server_time=4_000_000_000)}),
        command_auth="required")
    device.sync_once()
    assert device._clock_offset_s > 1e9
    backend.reply = _signs(STOP)
    device.sync_once()
    assert a.parse_request_header(backend.requests[-1]["headers"]["X-SD-Auth"]).issued_at >= 4_000_000_000
    assert device.pop_pending_protection()["action"] == "stop"


def test_a_previous_key_signed_clock_answer_is_accepted_during_rotation():
    old = "dk_" + "44" * 24
    device, _ = _device(lambda nonce: _skew({
        "code": "clock_skew", "clock_auth": a.sign_clock(a.DeviceKey(old), request_nonce=nonce, server_time=4_000_000_000)}),
        command_auth="required", previous_device_key=old)
    device.sync_once()
    assert device._clock_offset_s > 1e9


@pytest.mark.parametrize("mode", ["off", "permissive"])
def test_modes_that_send_the_bearer_key_never_learn_a_clock(mode):
    device, _ = _device(lambda nonce: _skew({
        "code": "clock_skew", "clock_auth": a.sign_clock(a.DeviceKey(KEY), request_nonce=nonce or "0" * 32,
                                                         server_time=4_000_000_000)}), command_auth=mode)
    device.sync_once()
    assert device._clock_offset_s == 0.0


# ------------------------------------------------------------------ an older backend
def _pre_signing_backend(command):
    """A backend from before M6.5: bearer key only, ignores X-SD-Nonce, never signs."""
    def reply(_nonce):
        return {"ok": True, "stored": 0, "command": command}
    return reply


@pytest.mark.parametrize("mode", ["off", "permissive"])
def test_a_new_device_still_obeys_a_backend_that_predates_signing(mode):
    device, backend = _device(_pre_signing_backend(STOP), command_auth=mode)
    device.sync_once()
    assert device.pop_pending_protection()["action"] == "stop"
    assert backend.requests[0]["headers"]["X-Device-Key"] == KEY


def test_a_required_device_never_obeys_an_unsigned_backend():
    device, _ = _device(_pre_signing_backend(STOP), command_auth="required")
    device.sync_once()
    assert device.pop_pending_protection() is None and device.command_auth_rejections == 1
