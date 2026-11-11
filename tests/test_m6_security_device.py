"""M6 security at the device layer: dashboard auth, Wi-Fi provisioning input, logs.

Every assertion here is against the real server / parser, not the UI:
  * on a LAN bind every /api and /stream path refuses an unauthenticated
    request; a token is compared in constant time;
  * the ?token= query fallback is honoured on GET /stream ONLY (the MJPEG <img>),
    never on /api or on a POST -- a URL in a history or a link must not be able to
    change the preset, stop monitoring, or move the device to another network;
  * Wi-Fi credentials from BLE/the dashboard are bounded and free of control
    characters before they reach nmcli argv or the journal;
  * the passphrase is never logged, and an SSID cannot forge journal lines.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

import pytest

from core import network
from core.dashboard import DashboardServer
from services import ble_gatt

TOKEN = "0123456789abcdef0123456789abcdef"


def _call(url, method="GET", headers=None, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})  # noqa: S310
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as resp:  # noqa: S310 - local test server
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.fixture
def lan_dashboard():
    server = DashboardServer(host="0.0.0.0", port=0, token=TOKEN,  # noqa: S104 - the LAN bind under test
                             presets={"driver_safety": {"module": "driver_attention"}})
    url = server.start()
    yield server, url
    server.close()


PROTECTED_GETS = ["/api/state", "/api/health", "/api/history", "/api/presets", "/stream"]


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_every_api_and_stream_path_refuses_without_the_token(lan_dashboard, path):
    _, url = lan_dashboard
    assert _call(url + path) == 401
    assert _call(url + path, headers={"X-Device-Token": "wrong"}) == 401


@pytest.mark.parametrize("path", PROTECTED_GETS)
def test_the_header_token_opens_every_path(lan_dashboard, path):
    _, url = lan_dashboard
    assert _call(url + path, headers={"X-Device-Token": TOKEN}) == 200


@pytest.mark.parametrize("path", ["/api/state", "/api/health", "/api/history"])
def test_a_query_token_no_longer_opens_the_api(lan_dashboard, path):
    _, url = lan_dashboard
    assert _call(f"{url}{path}?token={TOKEN}") == 401


def test_a_query_token_still_opens_the_video_stream(lan_dashboard):
    _, url = lan_dashboard
    assert _call(f"{url}/stream?token={TOKEN}") == 200


@pytest.mark.parametrize(("path", "body"), [
    ("/api/preset", {"preset": "driver_safety"}),
    ("/api/protection", {"action": "stop", "confirm": True}),
    ("/setup/wifi", {"ssid": "attacker", "password": "password123"}),
])
def test_a_query_token_cannot_authorize_a_write(lan_dashboard, path, body):
    server, url = lan_dashboard
    applied = []
    server.apply_wifi = lambda ssid, password: applied.append(ssid) or (True, "ok")
    assert _call(f"{url}{path}?token={TOKEN}", "POST", body=body) == 401
    assert server._pending_preset is None and server._pending_protection is None and not applied


def test_writes_with_the_header_token_are_accepted(lan_dashboard):
    server, url = lan_dashboard
    headers = {"X-Device-Token": TOKEN}
    assert _call(url + "/api/preset", "POST", headers, {"preset": "driver_safety"}) == 202
    assert server._pending_preset == "driver_safety"


def test_no_token_on_a_lan_bind_fails_closed():
    server = DashboardServer(host="0.0.0.0", port=0, token=None)  # noqa: S104 - the LAN bind under test
    url = server.start()
    try:
        for path in PROTECTED_GETS:
            assert _call(url + path) == 401, path
        assert _call(url + "/api/protection", "POST", body={"action": "stop", "confirm": True}) == 401
    finally:
        server.close()


def test_the_repo_config_ships_no_shared_token_and_the_compare_is_constant_time():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent
    assert yaml.safe_load((repo / "config.yaml").read_text())["dashboard"]["token"] is None
    assert "hmac.compare_digest" in (repo / "core" / "dashboard.py").read_text()


# ----------------------------------------------------------------- Wi-Fi credentials
@pytest.mark.parametrize(("ssid", "password", "problem"), [
    ("Home", "correct horse", None),
    ("Home", "", None),                         # open network
    ("Home", None, None),                       # open network (BLE psk: null)
    ("Café ☕", "pässwörd123", None),           # UTF-8 SSIDs and passphrases are real
    ("x" * 32, "p" * 63, None),
    ("", "pw", "network name required"),
    (None, "pw", "network name required"),
    ("x" * 33, "pw", "longer than 32 bytes"),
    ("é" * 17, "pw", "longer than 32 bytes"),   # 34 bytes
    ("Home\nINFO forged line", "pw", "control characters"),
    ("Home\x00", "pw", "control characters"),
    ("Home", "pw\x00", "control characters"),
    ("Home", "a" * 129, "longer than 128 bytes"),
    ("Home", 12345678, "must be text"),
])
def test_credential_problem(ssid, password, problem):
    found = network.credential_problem(ssid, password)
    if problem is None:
        assert found is None
    else:
        assert found and problem in found


def test_bad_credentials_never_reach_nmcli(monkeypatch):
    monkeypatch.setattr(network, "nmcli_available", lambda: True)

    def no_nmcli(*_args, **_kwargs):
        raise AssertionError("nmcli was invoked with invalid credentials")

    monkeypatch.setattr(network, "_run", no_nmcli)
    wifi = network.WifiManager()
    for method in (wifi.provision, wifi.connect):
        ok, message = method("Home\nforged", "secretpass")
        assert not ok and "control characters" in message
        ok, message = method("x" * 40, "secretpass")
        assert not ok and "32 bytes" in message


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def ble_log():
    handler = _Records()
    logger = logging.getLogger("sensordyme.ble")
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


def test_a_credential_write_never_logs_the_passphrase_or_a_raw_newline(ble_log):
    received = []
    payload = json.dumps({"ssid": "Home\nERROR fake line", "psk": "hunter2hunter2"}).encode()
    assert ble_gatt.dispatch_credentials(payload, lambda s, p: received.append((s, p)))
    assert received == [("Home\nERROR fake line", "hunter2hunter2")], "validation happens in provision()"
    joined = "\n".join(ble_log)
    assert "hunter2" not in joined
    assert all("\n" not in line for line in ble_log), "an SSID forged a journal line"


def test_a_malformed_credential_write_is_ignored(ble_log):
    received = []
    for bad in (b"not json", b"[1,2]", json.dumps({"ssid": None}).encode(),
                json.dumps({"ssid": "x", "psk": 5}).encode()):
        assert not ble_gatt.dispatch_credentials(bad, lambda s, p: received.append(s))
    assert received == []


def test_the_join_reports_the_refusal_to_the_app(monkeypatch):
    """The refusal must come back over BLE as the provisioning error, not vanish."""
    from services.provisioning import ProvisioningService

    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    service = ProvisioningService(network.WifiManager(), {"method": "ble"})
    pushed = []
    monkeypatch.setattr(service, "_push_status", lambda status, force=False: pushed.append(status))
    monkeypatch.setattr(service, "_describe", lambda online=None: {})
    service._join_network("x" * 40, "secretpass", attempt=1)
    assert service._provision["state"] == "error"
    assert "32 bytes" in service._provision["error"]
