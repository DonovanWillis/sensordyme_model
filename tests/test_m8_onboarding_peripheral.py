"""The virtual Bluetooth peripheral (sim/ble_peripheral.py), over its localhost link.

It runs the device's real provisioning service, proof-of-possession state
machine and GATT server logic; these tests drive it the way the phone does --
scan, connect, read, subscribe, write -- and check what the device says and
does. Nothing here is a claim about a physical Bluetooth link.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from services import ble_pop as pop
from services.ble_gatt import (
    CHAR_AUTH_UUID,
    CHAR_DEVICE_INFO_UUID,
    CHAR_NETWORKS_UUID,
    CHAR_STATUS_UUID,
    CHAR_WIFI_CREDS_UUID,
    SERVICE_UUID,
)
from services.ble_pop import Limits
from sim.ble_central import Central, LinkError, PopClient
from sim.ble_peripheral import DeviceSpec, SimBLEDevice, SimNetwork, ensure_loopback, redact

REPO = Path(__file__).resolve().parent.parent
HOME = SimNetwork("Home", "correct-horse-battery", 80)
CAFE = SimNetwork("Cafe", None, 60)


@pytest.fixture
def boot(tmp_path):
    started: list[SimBLEDevice] = []

    def _boot(**overrides) -> tuple[SimBLEDevice, int]:
        settings = {"networks": (HOME, CAFE), "join_delay_s": 0.1, "outcome_linger_s": 0.5, **overrides}
        spec = DeviceSpec(state_dir=tmp_path / f"dev{len(started)}", **settings)
        device = SimBLEDevice(spec)
        port = device.start(port=0)
        started.append(device)
        assert device.wait_advertising(), "the window opens while the device is offline"
        return device, port

    yield _boot
    for device in started:
        device.stop()


def _connect(port: int) -> Central:
    central = Central("127.0.0.1", port)
    central.connect()
    central.subscribe(CHAR_STATUS_UUID)
    if CHAR_AUTH_UUID in central.characteristics:
        central.subscribe(CHAR_AUTH_UUID)
    return central


def _code(device: SimBLEDevice) -> str:
    return pop.load_setup_code(str(device.setup_code_path))


def _outcome(central: Central, state: str) -> dict:
    return central.wait_status(lambda s: (s.get("provision") or {}).get("state") == state)


def test_it_advertises_the_real_gatt_table_and_says_how_it_takes_credentials(boot):
    device, port = boot()
    central = Central("127.0.0.1", port)
    adv = central.scan()
    assert adv["advertising"] and adv["services"] == [SERVICE_UUID] and adv["name"] == device.name
    central.connect()
    assert central.characteristics[CHAR_AUTH_UUID] == ["read", "write", "notify"]
    info = central.read_json(CHAR_DEVICE_INFO_UUID)
    assert info["id"] == device.device_id and info["auth"] == "pop"
    assert info["hostname"].startswith("sensordyme-sim-"), "not this Mac's hostname"
    assert central.read_json(CHAR_STATUS_UUID)["state"] == "provisioning"

    _legacy, legacy_port = boot(auth="legacy")
    plain = Central("127.0.0.1", legacy_port)
    plain.connect()
    assert CHAR_AUTH_UUID not in plain.characteristics
    assert plain.read_json(CHAR_DEVICE_INFO_UUID)["auth"] == "legacy"


def test_first_time_setup_proves_the_code_then_joins_and_closes_the_window(boot):
    device, port = boot()
    central = _connect(port)
    client = PopClient(central, _code(device), device.device_id)
    assert client.prove()["op"] == "proven"
    assert client.send_credentials("Home", "correct-horse-battery") == {"op": "accepted"}
    joined = _outcome(central, "ok")
    assert joined["provision"]["attempt"] == 1 and joined["state"] == "online"
    assert device.wifi.joins == [("Home", True)]
    deadline = time.monotonic() + 5
    while central.disconnected is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert central.disconnected == "the device closed its Bluetooth window"
    assert not device.peripheral.advertising

    transcript = json.dumps(device.transcript.records)
    for secret in (_code(device), pop.format_setup_code(_code(device)), "correct-horse-battery"):
        assert secret not in transcript
    kinds = [(r["dir"], r.get("value", {}).get("op")) for r in device.transcript.records
             if r.get("char") == "auth"]
    assert kinds == [("app->device", "hello"), ("device->app", "challenge"), ("app->device", "proof"),
                     ("device->app", "proven"), ("app->device", "creds"), ("device->app", "accepted")]


def test_a_refused_passphrase_keeps_the_window_open_and_a_fresh_handshake_retries(boot):
    device, port = boot()
    central = _connect(port)
    client = PopClient(central, _code(device), device.device_id)
    assert client.prove()["op"] == "proven"
    client.send_credentials("Home", "wrong-password")
    refused = _outcome(central, "error")
    assert "not accepted by 'Home'" in refused["provision"]["error"]
    assert device.peripheral.advertising
    assert client.prove()["op"] == "proven", "after a failed join a new handshake is accepted"
    client.send_credentials("Home", "correct-horse-battery")
    assert _outcome(central, "ok")["provision"]["attempt"] == 2


def test_plain_credentials_are_refused_under_pop_and_accepted_in_legacy(boot):
    device, port = boot()
    central = _connect(port)
    assert central.write(CHAR_WIFI_CREDS_UUID, json.dumps({"ssid": "Home", "psk": "correct-horse-battery"}).encode()) is None
    time.sleep(0.3)
    assert device.wifi.joins == [], "the plain write was acknowledged and ignored"

    _legacy, legacy_port = boot(auth="legacy")
    plain = _connect(legacy_port)
    plain.write(CHAR_WIFI_CREDS_UUID, json.dumps({"ssid": "Cafe", "psk": None}).encode())
    assert _outcome(plain, "ok")["provision"]["ssid"] == "Cafe"


def test_lockout_reports_the_wait_survives_reconnecting_and_lifts(boot):
    device, port = boot(limits=Limits(max_failures=3, lockout_base_s=1.0))
    central = _connect(port)
    wrong = PopClient(central, "ZZZZ-ZZZZ-ZZZZ", device.device_id)
    replies = [wrong.prove() for _ in range(3)]
    assert [r["error"] for r in replies] == ["wrong_code"] * 3
    # The device rounds its remaining wait up to whole seconds (int(left) + 1).
    assert "retry_after_s" not in replies[0] and replies[2]["retry_after_s"] in (1, 2)
    central.close()

    again = _connect(port)  # a new link: the lockout is the device's, not the connection's
    right = PopClient(again, _code(device), device.device_id)
    locked = right.prove()
    assert locked["error"] == "locked" and 1 <= locked["retry_after_s"] <= 2
    time.sleep(locked["retry_after_s"])
    assert right.prove()["op"] == "proven"


def test_timeouts_spend_the_challenge_and_the_proof(boot):
    device, port = boot(limits=Limits(challenge_ttl_s=0.3, proven_ttl_s=0.3))
    central = _connect(port)
    client = PopClient(central, _code(device), device.device_id)
    challenge = client.hello()
    time.sleep(0.5)
    import base64

    keys = pop.session_keys(client.code_key, challenge["id"], client.na, base64.b64decode(challenge["nd"]))
    late = client.exchange({"op": "proof", "tag": pop.b64e(pop.app_proof(keys))})
    assert late["error"] == "unexpected"
    assert client.prove()["op"] == "proven"
    time.sleep(0.5)
    assert client.send_credentials("Home", "correct-horse-battery")["error"] == "unexpected"
    assert device.wifi.joins == []


def test_long_values_cross_several_att_packets_and_are_reassembled(boot):
    many = tuple(SimNetwork(f"Network number {i:02d} with a long name", "x" * 8, 90 - i) for i in range(20))
    _device, port = boot(auth="legacy", networks=(*many, HOME), mtu=23)
    central = _connect(port)
    central.read(CHAR_NETWORKS_UUID)  # the first read starts the scan
    time.sleep(0.3)
    raw = central.read(CHAR_NETWORKS_UUID)
    listed = json.loads(raw)
    assert 0 < len(raw) <= 480 and len(listed) < 21, "trimmed by bytes, never cut mid-JSON"
    ssid = "Home"
    psk = "correct-horse-battery"
    payload = json.dumps({"ssid": ssid, "psk": psk, "pad": "p" * 60}).encode()
    assert len(payload) > 23
    central.write(CHAR_WIFI_CREDS_UUID, payload)
    assert _outcome(central, "ok")["provision"]["ssid"] == ssid
    assert central.write(CHAR_WIFI_CREDS_UUID, b"{" + b" " * 600 + b"}") == "invalid attribute value length"


def test_an_installed_device_is_reachable_in_its_reconfigure_window_and_moves(boot):
    device, port = boot(online_ssid="Cafe")
    central = _connect(port)
    status = central.wait_status(lambda s: s.get("reconfigurable") is True)
    assert status["state"] == "online" and status["ssid"] == "Cafe" and status["provision"] is None
    client = PopClient(central, _code(device), device.device_id)
    assert client.prove()["op"] == "proven"
    client.send_credentials("Home", "correct-horse-battery")
    moved = _outcome(central, "ok")
    assert moved["ssid"] == "Home" and moved["provision"]["attempt"] == 1


def test_the_link_is_loopback_only():
    assert ensure_loopback("127.0.0.1") == "127.0.0.1"
    assert ensure_loopback("localhost") == "127.0.0.1"
    for host in ("0.0.0.0", "192.168.1.20", "example.com"):  # noqa: S104 - asserting the refusal
        with pytest.raises(SystemExit):
            ensure_loopback(host)


def test_redaction_keeps_structure_and_drops_key_material():
    tag = pop.b64e(b"\x01" * 32)
    assert redact("auth", json.dumps({"op": "proof", "tag": tag}).encode()) == {"op": "proof", "tag": "<32 bytes>"}
    assert redact("wifi_creds", b'{"ssid":"Home","psk":"hunter22"}') == {"ssid": "Home", "psk": "<redacted>"}
    assert redact("auth", b"\xff\x00") == "<2 bytes, not JSON>"


def test_a_closed_window_refuses_connections(boot):
    device, port = boot()
    device.service.stop(timeout=2.0)
    central = Central("127.0.0.1", port)
    assert central.scan()["advertising"] is False
    with pytest.raises(LinkError, match="not advertising"):
        central.connect()


def test_the_command_line_prints_where_to_connect_and_never_the_code(tmp_path):
    process = subprocess.Popen(  # noqa: S603 - our own module, fixed argv
        [sys.executable, "-m", "sim.ble_peripheral", "--port", "0", "--state-dir", str(tmp_path)],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        lines = []
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if line.startswith("SIMBLE READY"):
                break
        port = int(lines[-1].split("port=")[1].split()[0])
        output = "".join(lines)
        assert f"-orynSimulatedBLE 127.0.0.1:{port}" in output
        code = pop.load_setup_code(str(tmp_path / "ble_setup_code"))
        assert code not in output and pop.format_setup_code(code) not in output
        assert Central("127.0.0.1", port).scan()["advertising"]
    finally:
        process.terminate()
        process.wait(timeout=10)
