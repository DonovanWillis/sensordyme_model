"""M8.3 security: what reaches a log (SECURITY_REVIEW_M6 §6).

Reproduced defects, each failing on 66cf6b6:

* a failed ntfy push logged `%r` of the requests exception, whose text is the
  request URL -- and the ntfy topic in it is the account's alert credential;
  a failed Twilio send logged the account SID the same way;
* a backend without Twilio configured printed every account's phone number;
* `subprocess.TimeoutExpired` quotes nmcli's argv, which carries the Wi-Fi
  passphrase; the rollback path logged it (the join paths already caught the
  timeout and are pinned here).

Plus a sweep of representative backend paths (signup, login, claim, ingest,
refused device auth) asserting no password, email, claim code, device key or
session token is logged.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
import requests
from fastapi.testclient import TestClient

from core import command_auth as a
from core import network
from core.logging_setup import describe_failure, mask_phone

TOPIC = "alerts-for-alice-q7Zx9"
SID = "AC" + "5e" * 16
KEY = "dk_" + "c3" * 24
CLAIM = "CCCC-DDDD-EEEE"
EMAIL = "carol@example.com"
PASSWORD = "correct-horse-battery"
PSK = "Wifi-Passphrase-77"


def _boot(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
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
    with app.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Unit", CLAIM, KEY, time.time()))
    return app, TestClient(app.app, raise_server_exceptions=False)


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    return _boot(tmp_path, monkeypatch)


def _failing_post(url):
    """A requests.post that fails the way requests does: the URL in the message."""
    def post(*_args, **_kwargs):
        response = requests.Response()
        response.status_code = 404
        response.url = url
        raise requests.HTTPError(f"404 Client Error: Not Found for url: {url}", response=response)
    return post


# ------------------------------------------------------------------ outbound-notification failures
def test_a_failed_push_never_logs_the_ntfy_topic(backend, monkeypatch, caplog):
    app, _ = backend
    monkeypatch.setattr(app, "NOTIFY_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(requests, "post", _failing_post(f"https://ntfy.sh/{TOPIC}"))
    with caplog.at_level(logging.DEBUG):
        app._notify({"ntfy_topic": TOPIC, "phone": None, "sms_on_loud": 0}, "Fall", "detail", "Unit",
                    push=True, sms=False)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "push delivery failed" in logged and "HTTPError (HTTP 404)" in logged
    assert TOPIC not in logged


def test_a_failed_sms_never_logs_the_twilio_account(backend, monkeypatch, caplog):
    app, _ = backend
    monkeypatch.setattr(app, "NOTIFY_RETRY_BACKOFF_S", 0)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", SID)
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tw-" + "9" * 30)
    monkeypatch.setenv("TWILIO_FROM", "+15550001111")
    monkeypatch.setattr(requests, "post", _failing_post(
        f"https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json"))
    with caplog.at_level(logging.DEBUG):
        app._notify({"ntfy_topic": None, "phone": "+15557654321", "sms_on_loud": 1}, "Fall", "d", "Unit",
                    push=False, sms=True)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "sms delivery failed" in logged
    assert SID not in logged and "9" * 30 not in logged and "7654321" not in logged


def test_the_log_only_sms_masks_the_phone_number(backend, monkeypatch, capsys):
    app, _ = backend
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"):
        monkeypatch.delenv(name, raising=False)
    app.send_sms({"phone": "+1 (555) 765-4321", "sms_on_loud": 1}, "Fall detected")
    out = capsys.readouterr().out
    assert "Fall detected" in out and "***21" in out
    assert "765-4321" not in out and "7654321" not in out


def test_describe_failure_and_mask_phone():
    assert describe_failure(requests.ConnectionError(f"Max retries exceeded with url: /{TOPIC}")) == "ConnectionError"
    assert describe_failure(ValueError("x")) == "ValueError"
    assert mask_phone("+44 7700 900123") == "***23" and mask_phone("") == "***" and mask_phone(None) == "***"


# ------------------------------------------------------------------ the Wi-Fi passphrase and nmcli argv
class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture()
def wifi_log():
    handler = _Records()
    logger = logging.getLogger("sensordyme.wifi")
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


def _stalled_nmcli(monkeypatch):
    monkeypatch.setattr(network, "nmcli_available", lambda: True)

    def run(cmd, **kwargs):
        if any(arg in ("password", "wifi-sec.psk", "up") for arg in cmd):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(network.subprocess, "run", run)


def test_a_stalled_rollback_never_logs_the_restored_passphrase(monkeypatch, wifi_log):
    _stalled_nmcli(monkeypatch)
    network.WifiManager()._roll_back("Home", created=False, previous="Home", restore_psk=PSK)
    assert any("could not restore the previous network" in line for line in wifi_log)
    assert all(PSK not in line for line in wifi_log), wifi_log
    assert any("<redacted>" in line for line in wifi_log)


@pytest.mark.parametrize("method", ["connect", "provision"])
def test_a_stalled_join_never_returns_or_logs_the_passphrase(monkeypatch, wifi_log, method):
    _stalled_nmcli(monkeypatch)
    ok, message = getattr(network.WifiManager(), method)("Office", PSK)
    assert not ok and PSK not in message
    assert all(PSK not in line for line in wifi_log), wifi_log


def test_the_redaction_keeps_the_rest_of_the_command():
    assert network._redacted(["device", "wifi", "connect", "Cafe", "password", PSK]) == [
        "device", "wifi", "connect", "Cafe", "password", "<redacted>"]
    assert network._redacted(["connection", "modify", "id", "Cafe", "wifi-sec.psk", PSK])[-1] == "<redacted>"


# ------------------------------------------------------------------ a sweep of backend paths
def test_representative_backend_paths_log_no_credentials(backend, caplog, capsys):
    _, client = backend
    with caplog.at_level(logging.DEBUG):
        token = client.post("/v1/signup", json={"email": EMAIL, "password": PASSWORD}).json()["token"]
        client.post("/v1/login", json={"email": EMAIL, "password": PASSWORD + "x"})
        client.post("/v1/login", json={"email": EMAIL, "password": PASSWORD})
        auth = {"Authorization": f"Bearer {token}"}
        client.post("/v1/devices/claim", json={"claim_code": "ZZZZ-ZZZZ-ZZZZ"}, headers=auth)
        client.post("/v1/devices/claim", json={"claim_code": CLAIM}, headers=auth)
        client.post("/v1/account/ntfy", json={"ntfy_topic": TOPIC}, headers=auth)
        client.post("/v1/account/phone", json={"phone": "+15557654321"}, headers=auth)
        client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY})
        client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": KEY[:-2] + "00"})
        body = b'{"events": []}'
        stale = a.sign_request(a.DeviceKey(KEY), "POST", "/v1/ingest", body, nonce=a.new_nonce(),
                               issued_at=int(time.time()) - 3600)
        client.post("/v1/ingest", content=body, headers={"Content-Type": "application/json", "X-SD-Auth": stale})
        client.post("/v1/logout", headers=auth)
        client.get("/v1/devices", headers=auth)  # refused: signed out
    printed = capsys.readouterr()
    logged = "\n".join(r.getMessage() for r in caplog.records) + printed.out + printed.err
    assert "device auth refused" in logged, "the sweep reached the refusal path"
    for secret in (PASSWORD, EMAIL, CLAIM, "ZZZZ-ZZZZ-ZZZZ", KEY, KEY[3:], token, TOPIC, "7654321"):
        assert secret not in logged, f"a credential reached the log: {secret[:4]}…"
