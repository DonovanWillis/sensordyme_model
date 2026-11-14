"""M8 security end to end: the real CloudUplink against the real cloud/app.py.

Two scenarios, each recorded as a list of transitions (what went on the wire,
what the backend answered, and the device, backend and app state after it):

* credential mismatch -- a wrong device key, a wrong signing key on either side,
  and a revoked key: refused, reported, and nothing changes;
* command replay -- a captured signed command delivered again, in the same
  process and after a device restart, and a captured signed request delivered
  again, in the same process and after a backend restart: all refused.

The scenarios only use a `stack` (an app-side client, a device transport, a
backend restart and a database query), so the same code drives an in-process
backend here and a live uvicorn backend for the M8 evidence run.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import time
import types
from collections.abc import Callable
from typing import Any, Optional

import pytest

from core import command_auth as a
from core.uplink import CloudUplink

HEALTH = {"device_id": "SD-0A1B2C3D", "state": "ok"}
WRONG_KEY = "dk_" + "13" * 24
ATTACKER_KEY = "dk_" + "99" * 24
EVENT = {"type": "phone_use", "timestamp": "2026-11-11T10:00:00Z", "confidence": 0.9, "event_key": "e-1"}


# ------------------------------------------------------------------ the on-path tap
def _answer(status: int, body: Any):
    import requests

    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"HTTP {status}", response=types.SimpleNamespace(status_code=status,
                                                                                       json=lambda: body))
    return types.SimpleNamespace(status_code=status, json=lambda: body, raise_for_status=raise_for_status,
                                 headers={})


class Tap:
    """Between the device and the backend: records every exchange and can
    replace the request or the answer (an on-path attacker's capabilities)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.exchanges: list[dict] = []
        self.rewrite_answer: Optional[Callable[[dict], dict]] = None
        self.replace_answer: Optional[dict] = None

    def post(self, url, **kwargs):
        response = self.inner.post(url, **kwargs)
        status = int(response.status_code)
        try:
            body = response.json()
        except ValueError:
            body = None
        if self.replace_answer is not None:
            status, body = 200, self.replace_answer
        elif self.rewrite_answer is not None and status == 200:
            body = self.rewrite_answer(body)
        self.exchanges.append({"headers": dict(kwargs.get("headers") or {}), "data": kwargs.get("data"),
                               "json": kwargs.get("json"), "status": status, "answer": body})
        return _answer(status, body)


# ------------------------------------------------------------------ recording
def _wire(exchange: dict) -> dict:
    """What crossed the wire, without secrets: auth kind, kid, status, code."""
    headers = exchange["headers"]
    if "X-SD-Auth" in headers:
        kind, kid = "signed", a.parse_request_header(headers["X-SD-Auth"]).kid
    else:
        kind, kid = "bearer", None
        if headers.get("X-Device-Key"):
            try:
                kid = a.key_id(headers["X-Device-Key"])
            except ValueError:
                kid = None
    answer = exchange["answer"] if isinstance(exchange["answer"], dict) else {}
    command = answer.get("command") or {}
    return {"auth": kind, "kid": kid, "status": exchange["status"], "code": answer.get("code"),
            "command_types": sorted(command) if isinstance(command, dict) else None,
            "signed_command": "command_auth" in answer}


def snapshot(stack, device: CloudUplink, device_id: int) -> dict:
    row = next((d for d in stack.app_get("/v1/devices")["devices"] if d["id"] == device_id), None)
    events = stack.query("SELECT COUNT(*) FROM events WHERE device_id = ?", (device_id,))[0][0]
    pending = device._pending_protection
    return {
        "device": {"connected": device.connected, "auth_rejections": device.command_auth_rejections,
                   "pending_protection": (pending or {}).get("action"), "queued_events": len(device._queue),
                   "clock_offset_s": round(device._clock_offset_s)},
        "backend": {"events_stored": events, "last_seen": row["last_seen"] if row else None},
        "app": {"online": row["online"] if row else None, "credentials": row["credentials"] if row else None,
                "monitoring_state": row["status"]["monitoring_state"] if row else None,
                "reasons": row["status"]["reasons"] if row else None},
    }


class Recorder:
    def __init__(self, stack, device_id: int) -> None:
        self.stack, self.device_id, self.steps = stack, device_id, []

    def step(self, name: str, device: CloudUplink, tap: Tap, *, expect: str) -> dict:
        before = len(tap.exchanges)
        device.sync_once()
        record = {"step": name, "expect": expect,
                  "wire": [_wire(x) for x in tap.exchanges[before:]],
                  "after": snapshot(self.stack, device, self.device_id)}
        self.steps.append(record)
        return record


def make_device(stack, tap: Tap, *, key: str, mode: str, **cloud) -> CloudUplink:
    device = CloudUplink({"base_url": stack.base_url, "device_key": key, "command_auth": mode,
                          "poll_interval_s": 0, **cloud})
    device._requests = tap
    device.set_health(HEALTH)
    return device


# ------------------------------------------------------------------ scenario 1
def credential_mismatch(stack) -> list[dict]:
    """Wrong device key, wrong signing key (device side and backend side),
    revoked key: refused, reported, no state change."""
    rec = Recorder(stack, stack.device_id)
    tap = Tap(stack.transport)

    good = make_device(stack, tap, key=stack.device_key, mode="required")
    rec.step("baseline: the right key, signed", good, tap, expect="accepted")
    baseline = rec.steps[-1]["after"]

    # 1a. a device configured with a key the backend never issued (bearer, permissive)
    wrong = make_device(stack, tap, key=WRONG_KEY, mode="permissive")
    wrong._queue.append(dict(EVENT))  # an alert waiting to go up
    rec.step("1a wrong device key (bearer)", wrong, tap, expect="401, nothing stored, event kept on device")

    # 1b. the same wrong key, signing (required): its kid names no device
    wrong_signed = make_device(stack, tap, key=WRONG_KEY, mode="required")
    rec.step("1b wrong device key (signed)", wrong_signed, tap, expect="401 device_auth_unknown_key")

    # 1c. a request signed with the wrong secret under the device's own (public) kid
    forged_tap = Tap(stack.transport)
    forger = make_device(stack, forged_tap, key=ATTACKER_KEY, mode="required")
    real_kid, fake_kid = a.key_id(stack.device_key), a.key_id(ATTACKER_KEY)
    original_sign = a.sign_request

    def sign_as_victim(key, *args, **kwargs):
        return original_sign(key, *args, **kwargs).replace(f"kid={fake_kid}", f"kid={real_kid}")
    a.sign_request = sign_as_victim
    try:
        rec.step("1c wrong signing key under the device's kid", forger, forged_tap,
                 expect="401 device_auth_bad_signature; not reported against the device")
    finally:
        a.sign_request = original_sign

    # 1d. backend side: the command arrives signed with a key the device does not hold
    stack.app_post(f"/v1/devices/{stack.device_id}/protection", {"action": "stop", "confirm": True})

    def resign(answer: dict) -> dict:
        if not isinstance(answer, dict) or "command_auth" not in answer:
            return answer
        env = answer["command_auth"]
        forged = a.sign_command(a.DeviceKey(ATTACKER_KEY), answer["command"], device=env["dev"],
                                request_nonce=env["rn"])
        return {**answer, "command_auth": forged}
    tap.rewrite_answer = resign
    rec.step("1d command signed with the wrong key (on-path re-sign)", good, tap,
             expect="device refuses (unknown_key); stop not applied")
    tap.rewrite_answer = None

    # 1e. revoked key
    stack.manage("rotate_key")
    stack.manage("revoke_previous_key")
    rec.step("1e revoked key (signed)", good, tap, expect="401 device_auth_revoked_key; app: credentials refused")
    stale = make_device(stack, tap, key=stack.previous_key, mode="permissive")
    rec.step("1f revoked key (bearer)", stale, tap, expect="401 device_auth_revoked_key")
    rec.baseline = baseline  # type: ignore[attr-defined]
    return rec.steps


# ------------------------------------------------------------------ scenario 2
def command_replay(stack) -> list[dict]:
    """A captured signed command and a captured signed request, replayed."""
    rec = Recorder(stack, stack.device_id)
    tap = Tap(stack.transport)
    device = make_device(stack, tap, key=stack.device_key, mode="required")

    stack.app_post(f"/v1/devices/{stack.device_id}/protection", {"action": "stop", "confirm": True})
    rec.step("2a stop delivered, signed, applied", device, tap, expect="stop applied")
    captured_answer = tap.exchanges[-1]["answer"]
    captured_request = tap.exchanges[-1]
    device._pending_protection = None  # the main loop took it

    tap.replace_answer = captured_answer
    rec.step("2b captured stop replayed (same process)", device, tap, expect="refused stale_response")

    restarted = make_device(stack, tap, key=stack.device_key, mode="required")
    rec.step("2c captured stop replayed after a device restart", restarted, tap, expect="refused stale_response")
    tap.replace_answer = None

    replay = stack.raw_post(captured_request)
    rec.steps.append({"step": "2d captured signed request replayed to the backend", "expect": "401 replayed_nonce",
                      "wire": [{"status": replay[0], "code": (replay[1] or {}).get("code")}],
                      "after": snapshot(stack, restarted, stack.device_id)})
    stack.restart_backend()
    replay = stack.raw_post(captured_request)
    rec.steps.append({"step": "2e the same request replayed after a backend restart",
                      "expect": "401 replayed_nonce",
                      "wire": [{"status": replay[0], "code": (replay[1] or {}).get("code")}],
                      "after": snapshot(stack, restarted, stack.device_id)})
    rec.step("2f the restarted device carries on", restarted, tap, expect="accepted")
    return rec.steps


# ------------------------------------------------------------------ in-process stack
class _ClientTransport:
    def __init__(self, stack) -> None:
        self.stack = stack

    def post(self, url, json=None, data=None, headers=None, **_kwargs):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[1]
        client = self.stack.client
        if data is not None:
            response = client.post(path, content=data, headers=headers)
        else:
            response = client.post(path, json=json, headers=headers)
        return response


class InProcessStack:
    base_url = "http://backend.test"

    def __init__(self, tmp_path, monkeypatch) -> None:
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self._boot()
        self.device_key = "dk_" + "66" * 24
        with self.app.db() as conn:
            conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                         ("Unit", "AA-BB-CC", self.device_key, time.time()))
        token = self.client.post("/v1/signup", json={"email": "o@example.com", "password": "password1"}).json()["token"]
        self.auth = {"Authorization": f"Bearer {token}"}
        self.device_id = self.client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"},
                                          headers=self.auth).json()["device_id"]
        self.transport = _ClientTransport(self)

    def _boot(self) -> None:
        from fastapi.testclient import TestClient

        mp = self.monkeypatch
        mp.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
        mp.setenv("SENSORDYME_DB", str(self.tmp_path / "test.db"))
        mp.setenv("SENSORDYME_JWT_SECRET_FILE", str(self.tmp_path / ".jwt_secret"))
        mp.setenv("SENSORDYME_BACKEND_ID_FILE", str(self.tmp_path / ".backend_id"))
        mp.setenv("SENSORDYME_MDNS", "0")
        mp.setenv("SENSORDYME_DEVICE_AUTH", "permissive")
        mp.delenv("SENSORDYME_JWT_SECRET", raising=False)
        for mod in ("cloud.app", "cloud", "cloud.ratelimit", "cloud.manage"):
            sys.modules.pop(mod, None)
        self.app = importlib.import_module("cloud.app")
        self.app.gate.reset()
        self.client = TestClient(self.app.app, raise_server_exceptions=False)

    def restart_backend(self) -> None:
        self._boot()

    def app_get(self, path: str) -> dict:
        return self.client.get(path, headers=self.auth).json()

    def app_post(self, path: str, body: dict) -> dict:
        response = self.client.post(path, json=body, headers=self.auth)
        assert response.status_code == 200, response.text
        return response.json()

    def raw_post(self, exchange: dict) -> tuple[int, Optional[dict]]:
        response = self.client.post("/v1/ingest", content=exchange["data"], headers=exchange["headers"])
        return response.status_code, response.json()

    def query(self, sql: str, params=()) -> list:
        conn = sqlite3.connect(self.tmp_path / "test.db")
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def manage(self, action: str) -> None:
        import cloud.manage as manage

        result = getattr(manage, action)(self.device_id)
        if action == "rotate_key":
            self.previous_key, self.device_key = self.device_key, result


@pytest.fixture()
def stack(tmp_path, monkeypatch, capsys):
    return InProcessStack(tmp_path, monkeypatch)


def _by_name(steps: list[dict]) -> dict[str, dict]:
    return {s["step"].split(" ", 1)[0]: s for s in steps}


# ------------------------------------------------------------------ assertions
def test_credential_mismatch_is_refused_reported_and_changes_nothing(stack, capsys):
    steps = _by_name(credential_mismatch(stack))
    base = steps["baseline:"]["after"]
    assert base["device"]["connected"] and base["app"]["credentials"]["state"] == "accepted"

    s = steps["1a"]
    assert s["wire"][0]["status"] == 401 and s["after"]["device"]["connected"] is False
    assert s["after"]["device"]["queued_events"] == 1, "the device keeps its event"
    assert s["after"]["backend"] == base["backend"], "nothing stored, last_seen unchanged"
    assert s["after"]["app"]["credentials"]["refusal"] is None, "an unknown key is not attributed to a device"

    assert steps["1b"]["wire"][0]["code"] == "device_auth_unknown_key"
    assert steps["1c"]["wire"][0]["code"] == "device_auth_bad_signature"
    assert steps["1c"]["after"]["app"]["credentials"]["refusal"] is None, "forgeable: never paints the device red"
    for name in ("1b", "1c"):
        assert steps[name]["after"]["backend"] == base["backend"]

    s = steps["1d"]
    assert s["wire"][0]["status"] == 200 and s["wire"][0]["signed_command"]
    assert s["after"]["device"]["pending_protection"] is None and s["after"]["device"]["auth_rejections"] == 1

    for name in ("1e", "1f"):
        s = steps[name]
        assert s["wire"][0]["status"] == 401 and s["wire"][0]["code"] == "device_auth_revoked_key"
        assert s["after"]["app"]["credentials"]["state"] == "refused"
        assert s["after"]["app"]["credentials"]["refusal"]["reason"] == "revoked_key"
    assert steps["1f"]["after"]["backend"]["events_stored"] == 0
    out = capsys.readouterr().out
    assert "refused a remote command (unknown_key)" in out
    # (rotate-key prints the NEW key to the operator; the device never prints one)
    assert stack.previous_key not in out and WRONG_KEY not in out and ATTACKER_KEY not in out


def test_command_replay_is_refused_across_device_and_backend_restarts(stack, capsys):
    steps = _by_name(command_replay(stack))
    assert steps["2a"]["after"]["device"]["pending_protection"] == "stop"
    for name in ("2b", "2c"):
        s = steps[name]
        assert s["after"]["device"]["pending_protection"] is None, name
        assert s["after"]["device"]["auth_rejections"] == 1, name
    for name in ("2d", "2e"):
        assert steps[name]["wire"][0] == {"status": 401, "code": "device_auth_replayed_nonce"}, name
    assert steps["2f"]["after"]["device"]["connected"]
    assert capsys.readouterr().out.count("(stale_response)") == 2


def test_the_scenarios_record_json_safe_transitions(stack):
    json.dumps(command_replay(stack))
