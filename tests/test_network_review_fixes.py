"""Regression tests for the networking / provisioning / uplink / dashboard review.

One section per finding (F1-F17), plus three follow-ups from the runtime review
and the uplink's `notify` field. Nothing here touches the real network or
Bluetooth: nmcli is a fake runner, D-Bus/GLib are stubs that mirror dbus-python's
registration rules, and HTTP goes to servers on 127.0.0.1.
"""

from __future__ import annotations

import importlib.util
import json
import os
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

import core.network as network
import services.ble_gatt as ble_gatt
import services.provisioning as provisioning_module
from core.network import WifiManager, _split_terse
from core.uplink import CloudUplink
from services.ble_gatt import (
    _GATT_MAX_ATTRIBUTE_BYTES,
    BLEProvisioningServer,
    BLEStartError,
    _encode_status,
    _WriteAssembler,
    parse_command,
    parse_credentials,
)
from services.provisioning import ProvisioningService

REPO = Path(__file__).resolve().parent.parent


def _wait_for(predicate, timeout=5.0, step=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


# ============================================================================ D-Bus / GLib stub
class _StubDBus:
    """Just enough dbus-python + GLib for BLEProvisioningServer._serve().

    Mirrors the rules that matter (dbus-python 1.3.2): dbus.SystemBus() returns
    ONE shared connection unless private=True; an exported object stays
    registered until remove_from_connection(); registering a taken path raises
    KeyError (dbus_bindings/conn-methods.c). GLib.MainLoop.quit() before run()
    is lost, as in GLib.
    """

    def __init__(self):
        self.registry: dict = {}
        self.connections: list = []
        self.adapter = True
        self.find_gate: threading.Event | None = None
        self.subprocess_calls: list = []
        stub = self

        class Conn:
            def __init__(self, private):
                self.private = private
                self.closed = False
                stub.connections.append(self)

            def get_object(self, *_a):
                return types.SimpleNamespace()

            def close(self):
                self.closed = True

        self._shared = None

        def SystemBus(private=False):
            if private:
                return Conn(True)
            if stub._shared is None:
                stub._shared = Conn(False)
            return stub._shared

        class Object:
            def __init__(self, conn, path):
                key = (id(conn), path)
                if key in stub.registry:
                    raise KeyError(
                        f"Can't register the object-path handler for '{path}': there is already a handler"
                    )
                stub.registry[key] = self
                self._stub_key = key

            def remove_from_connection(self, connection=None, path=None):
                stub.registry.pop(self._stub_key, None)

        def decorator(*_a, **_k):
            return lambda f: f

        class Interface:
            def __init__(self, _obj, iface):
                self.iface = iface

            def GetManagedObjects(self):
                if stub.find_gate is not None:
                    stub.find_gate.wait(5)
                if not stub.adapter:
                    return {}
                return {"/org/bluez/hci0": {"org.bluez.GattManager1": {}}}

            def Set(self, *_a):
                pass

            def RegisterApplication(self, _path, _opts, reply_handler, error_handler):
                reply_handler()

            def RegisterAdvertisement(self, _path, _opts, reply_handler, error_handler):
                reply_handler()

            def UnregisterAdvertisement(self, _path):
                pass

            def UnregisterApplication(self, _path):
                pass

        class MainLoop:
            def __init__(self):
                self._event = None
                self.ran = False

            def run(self):
                self.ran = True
                self._event = threading.Event()  # an earlier quit() is lost, as in GLib
                self._event.wait()

            def quit(self):
                if self._event is not None:
                    self._event.set()

        def timeout_add(ms, fn):
            def tick():
                while True:
                    time.sleep(ms / 1000.0)
                    if not fn():
                        return

            threading.Thread(target=tick, daemon=True).start()
            return 1

        dbus = types.ModuleType("dbus")
        dbus.SystemBus = SystemBus
        dbus.Interface = Interface
        dbus.ObjectPath = str
        dbus.Array = lambda v, signature=None: list(v)
        dbus.Boolean = bool
        dbus.String = str
        service = types.ModuleType("dbus.service")
        service.Object = Object
        service.method = decorator
        service.signal = decorator
        dbus.service = service
        mainloop = types.ModuleType("dbus.mainloop")
        glib = types.ModuleType("dbus.mainloop.glib")
        glib.DBusGMainLoop = lambda set_as_default=False: None
        mainloop.glib = glib
        dbus.mainloop = mainloop
        gi = types.ModuleType("gi")
        repo = types.ModuleType("gi.repository")
        repo.GLib = types.SimpleNamespace(MainLoop=MainLoop, timeout_add=timeout_add)
        gi.repository = repo
        self.modules = {
            "dbus": dbus, "dbus.service": service, "dbus.mainloop": mainloop,
            "dbus.mainloop.glib": glib, "gi": gi, "gi.repository": repo,
        }

    def chrc(self, index):
        path = f"/com/sensordyme/provisioning/service0/char{index}"
        return next(obj for (_c, p), obj in self.registry.items() if p == path)


@pytest.fixture()
def stub_dbus(monkeypatch):
    stub = _StubDBus()
    for name, module in stub.modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def fake_run(command, **_kwargs):
        stub.subprocess_calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)  # never a real `sudo btmgmt`
    return stub


def _server(**kwargs):
    kwargs.setdefault("on_credentials", lambda ssid, psk: None)
    return BLEProvisioningServer("SensorDyme-TEST", {"id": "SD-00000000"}, **kwargs)


# ============================================================================ F1
def test_f1_the_bluetooth_window_reopens_in_the_same_process(stub_dbus):
    """The second window used to die with KeyError on the shared connection while
    the service reported it open -- so Wi-Fi lost after the boot window closed
    could not be fixed over Bluetooth until a reboot."""
    first = _server()
    first.start()
    assert first.is_alive()
    first.stop()
    assert not stub_dbus.registry, "exported objects were left registered after the window closed"
    assert all(c.private and c.closed for c in stub_dbus.connections)

    second = _server()
    second.start()
    try:
        assert second.is_alive(), f"second window died: {second.failure}"
        assert second.failure is None
    finally:
        second.stop()


def test_f1_a_server_that_cannot_come_up_raises_instead_of_reporting_open(stub_dbus):
    stub_dbus.adapter = False
    server = _server()
    with pytest.raises(BLEStartError, match="no Bluetooth adapter"):
        server.start()
    assert not server.is_alive()


class _FakeWifi:
    def __init__(self, online=False):
        self.online = online

    def is_online(self):
        return self.online

    def current_ssid(self):
        return "HomeNet" if self.online else None

    def ip_address(self):
        return "192.168.1.50" if self.online else None

    def watch(self, on_change, stop):
        self.on_change = on_change

    def scan(self):
        return []

    def provision(self, ssid, psk):
        self.online = True
        return True, f"joined '{ssid}'"


class _FakeBLE:
    """Mirrors BLEProvisioningServer's public surface. Records pushes while running."""

    instances: ClassVar[list] = []
    start_error: ClassVar[Exception | None] = None

    def __init__(self, device_name, device_info, on_credentials, scan_networks=None, on_command=None):
        self.device_info = device_info
        self.on_credentials = on_credentials
        self.statuses: list = []
        self.running = False
        self.alive = True
        self.failure = None
        self.paused_calls: list = []
        self.refresh_calls: list = []
        _FakeBLE.instances.append(self)

    def start(self):
        if _FakeBLE.start_error is not None:
            raise _FakeBLE.start_error
        self.running = True

    def stop(self):
        self.running = False

    def is_alive(self):
        return self.running and self.alive

    def set_status(self, status):
        if self.running:
            self.statuses.append(json.loads(json.dumps(status)))

    def refresh_networks(self, force=False):
        self.refresh_calls.append(force)

    def set_scans_paused(self, paused):
        self.paused_calls.append(paused)


@pytest.fixture()
def fake_ble(monkeypatch):
    monkeypatch.setattr(provisioning_module, "nmcli_available", lambda: True)
    monkeypatch.setattr(ble_gatt, "BLEProvisioningServer", _FakeBLE)
    _FakeBLE.instances = []
    _FakeBLE.start_error = None
    yield _FakeBLE
    _FakeBLE.start_error = None


def _service(wifi, **config):
    base = {"method": "ble", "grace_s": 0, "poll_s": 0.02, "reconfigure_window_s": 0.0,
            "outcome_linger_s": 0.05}
    base.update(config)
    return ProvisioningService(wifi, base)


class _Running:
    def __init__(self, service):
        self.service = service
        self.stop = threading.Event()
        self.thread = threading.Thread(target=service.run, args=(self.stop,), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        wake = getattr(self.service, "_wake", None)
        if wake is not None:
            wake.set()
        self.thread.join(timeout=3)


def test_f1_a_dead_ble_server_is_reported_and_reopened(fake_ble, monkeypatch):
    monkeypatch.setattr(ProvisioningService, "_BLE_RETRY_S", 0.2)
    service = _service(_FakeWifi(online=False))
    with _Running(service):
        assert _wait_for(lambda: fake_ble.instances)
        fake_ble.instances[0].alive = False          # bluetoothd went away under it
        fake_ble.instances[0].failure = "bluetoothd restarted"
        assert _wait_for(lambda: service._ble is None or service._ble is not fake_ble.instances[0])
        healthy, detail = service.status()
        if service._ble is None:
            assert healthy is False and "bluetoothd restarted" in detail
        assert _wait_for(lambda: len(fake_ble.instances) >= 2 and fake_ble.instances[-1].running), \
            "the window was never reopened"
        assert service.status()[0] is True


def test_f1_a_window_that_fails_to_start_is_retried_not_reported_open(fake_ble, monkeypatch):
    monkeypatch.setattr(ProvisioningService, "_BLE_RETRY_S", 0.2)
    fake_ble.start_error = BLEStartError("no Bluetooth adapter found")
    service = _service(_FakeWifi(online=False))
    with _Running(service):
        assert _wait_for(lambda: service._ble_failure is not None)
        healthy, detail = service.status()
        assert healthy is False and "no Bluetooth adapter" in detail
        fake_ble.start_error = None
        assert _wait_for(lambda: service._ble is not None and service._ble.running, timeout=3)
        assert service.status()[0] is True


# ============================================================================ F2
class _RealisticWifi(_FakeWifi):
    """Online the moment `connection up` succeeds; provision() returns later
    (settle poll + _prefer), in the order WifiManager.provision really does."""

    def provision(self, ssid, psk):
        time.sleep(0.1)
        self.online = True
        self.on_change()          # nmcli monitor wakes the run loop
        time.sleep(0.4)           # _settled_on + _prefer
        return True, f"joined '{ssid}'"


def test_f2_a_successful_join_past_the_window_still_reports_ok(fake_ble):
    """It used to tear Bluetooth down on "connected" -- before the worker had
    pushed "ok" -- so the app saw the link drop mid-join and showed a failure."""
    service = _service(_RealisticWifi(), poll_s=10, reconfigure_window_s=0.0, outcome_linger_s=0.6)
    with _Running(service):
        assert _wait_for(lambda: fake_ble.instances)
        ble = fake_ble.instances[0]
        ble.on_credentials("HomeNet", "secret")
        service._connect_thread.join(timeout=5)
        provisions = [(s.get("provision") or {}).get("state") for s in ble.statuses]
        assert "ok" in provisions, f"the outcome never reached the app: {provisions}"
        assert ble.running, "Bluetooth closed before the app could read the outcome"
        assert _wait_for(lambda: not ble.running, timeout=4), "window never closed after the linger"


def test_f2_the_window_is_not_torn_down_while_a_join_is_in_flight(fake_ble):
    release = threading.Event()

    class SlowJoin(_FakeWifi):
        def provision(self, ssid, psk):
            self.online = True
            self.on_change()
            release.wait(5)
            return True, "joined"

    service = _service(SlowJoin(), reconfigure_window_s=0.0)
    with _Running(service):
        assert _wait_for(lambda: fake_ble.instances)
        ble = fake_ble.instances[0]
        ble.on_credentials("HomeNet", "secret")
        time.sleep(0.3)  # many loop iterations, all online and past the window
        assert ble.running, "torn down mid-join"
        release.set()
        service._connect_thread.join(timeout=5)


# ============================================================================ F9
def test_f9_every_write_gets_an_attempt_number_echoed_in_its_outcome(fake_ble):
    results = iter([(False, "That password was not accepted by 'HomeNet'."), (True, "joined")])
    go = threading.Event()

    class Wifi(_FakeWifi):
        def provision(self, ssid, psk):
            go.wait(5)
            return next(results)

    service = _service(Wifi(online=False))
    ble = _FakeBLE("x", {}, lambda *a: None)
    ble.running = True
    service._ble = ble

    for expected in (1, 2):
        go.clear()
        service._on_credentials("HomeNet", "pw")
        # Set and pushed before the handler returns, i.e. before the write response.
        assert ble.statuses[-1]["provision"] == {"state": "connecting", "ssid": "HomeNet", "attempt": expected}
        go.set()
        service._connect_thread.join(timeout=3)
        outcome = ble.statuses[-1]["provision"]
        assert outcome["attempt"] == expected
        assert outcome["state"] in ("ok", "error")
    assert ble.statuses[-1]["provision"]["state"] == "ok"


# ============================================================================ F15
def test_f15_the_credential_handler_runs_no_nmcli_on_the_ble_thread(fake_ble):
    calls = []
    release = threading.Event()

    class Recording(_FakeWifi):
        def is_online(self):
            calls.append(threading.get_ident())
            return False

        def current_ssid(self):
            calls.append(threading.get_ident())

        def ip_address(self):
            calls.append(threading.get_ident())

        def provision(self, ssid, psk):
            release.wait(5)
            return True, "joined"

    service = _service(Recording())
    ble = _FakeBLE("x", {}, lambda *a: None)
    ble.running = True
    service._ble = ble
    service._on_credentials("HomeNet", "pw")
    try:
        assert threading.get_ident() not in calls, "nmcli ran inside the BLE write handler"
        assert ble.statuses[-1]["provision"]["state"] == "connecting"
    finally:
        release.set()
        service._connect_thread.join(timeout=3)


def test_f15_an_unchanged_description_is_not_re_sent_every_tick(fake_ble):
    service = _service(_FakeWifi(online=True), poll_s=0.01, always_available=True)
    with _Running(service):
        assert _wait_for(lambda: fake_ble.instances and fake_ble.instances[0].statuses)
        time.sleep(0.3)  # ~30 loop iterations
    assert len(fake_ble.instances[0].statuses) == 1, len(fake_ble.instances[0].statuses)


def test_f15_an_online_device_holding_its_window_is_not_polled_every_second(fake_ble):
    counts = []

    class Counting(_FakeWifi):
        def is_online(self):
            counts.append(1)
            return True

    service = _service(Counting(online=True), poll_s=10, reconfigure_window_s=30)
    with _Running(service):
        time.sleep(1.4)  # the old 1s ceiling applied here: a second iteration at t=1.0
    assert len(counts) == 1, f"{len(counts)} connectivity checks in 1.4s"


# ============================================================================ extra: lifecycle/status
def test_stop_keeps_the_handle_of_a_thread_that_did_not_stop(fake_ble):
    """Dropping it let start() launch a second thread beside a wedged one."""
    release = threading.Event()
    service = _service(_FakeWifi())
    service.run = lambda stop: release.wait(5)  # ignores stop: wedged in a BLE call
    service.start()
    first = service._thread
    service.stop(timeout=0.1)
    assert service._thread is first and first.is_alive()
    service.start()
    assert service._thread is first, "a second provisioning thread was started"
    release.set()
    first.join(timeout=3)


@pytest.mark.parametrize("method", ["off", "hotspot"])
def test_a_deliberately_dormant_method_reports_healthy(monkeypatch, method):
    monkeypatch.setattr(provisioning_module, "nmcli_available", lambda: True)
    service = ProvisioningService(_FakeWifi(), {"method": method})
    service.start()
    service._thread.join(timeout=3)
    healthy, detail = service.status()
    assert healthy is True, detail
    assert method in detail


# ============================================================================ F5
def test_f5_an_implicit_refresh_does_not_rescan_a_fresh_list():
    scans = []

    def scan():
        scans.append(1)
        return [{"ssid": "HomeNet", "signal": 80, "secured": True}]

    server = _server(scan_networks=scan)
    server.refresh_networks()
    assert _wait_for(lambda: not server._scanning and server._scanned_at is not None)
    server.refresh_networks()             # what every networks read does
    time.sleep(0.05)
    assert len(scans) == 1
    server.refresh_networks(force=True)   # the explicit "rescan" command
    assert _wait_for(lambda: len(scans) == 2)


def test_f5_nothing_scans_while_a_join_holds_the_radio():
    scans = []
    server = _server(scan_networks=lambda: scans.append(1) or [])
    server.set_scans_paused(True)
    server.refresh_networks(force=True)
    time.sleep(0.05)
    assert scans == []


def test_f5_the_device_and_app_no_longer_ping_pong_scans():
    """Model of the iOS app: any status with scanning=false -> read networks
    (offset 0). It used to start the next scan every time."""
    scans = []

    def scan():
        scans.append(time.monotonic())
        time.sleep(0.02)
        return [{"ssid": "HomeNet", "signal": 80, "secured": True}]

    server = _server(scan_networks=scan)

    class AppReadingOnEveryStatus:
        def push_value(self, data):
            if not json.loads(data).get("scanning"):
                threading.Thread(target=server.refresh_networks, daemon=True).start()

    server._status_chrc = AppReadingOnEveryStatus()
    server.refresh_networks()
    time.sleep(1.0)
    assert len(scans) == 1, f"{len(scans)} scans in 1s with nobody asking"


def test_f5_provisioning_pauses_scans_for_the_length_of_a_join(fake_ble):
    service = _service(_FakeWifi())
    ble = _FakeBLE("x", {}, lambda *a: None)
    ble.running = True
    service._ble = ble
    service._on_credentials("HomeNet", "pw")
    service._connect_thread.join(timeout=3)
    assert ble.paused_calls == [True, False]


def test_f5_the_rescan_command_forces_a_scan(fake_ble):
    service = _service(_FakeWifi())
    ble = _FakeBLE("x", {}, lambda *a: None)
    service._ble = ble
    service._on_command({"action": "rescan"})
    service._on_command(["not", "a", "dict"])  # ignored, not raised
    assert ble.refresh_calls == [True]


# ============================================================================ F10
def _status(provision, ssid="Brightspeed_0112F425"):
    return {"state": "online", "ssid": ssid, "ip": "192.168.1.37", "device_id": "SD-A1B2C3D4",
            "reconfigurable": False, "provision": provision, "scanning": False}


def test_f10_a_typical_outcome_keeps_every_field_and_is_compact():
    status = _status({"state": "ok", "ssid": "Brightspeed_0112F425", "ip": "192.168.1.37",
                      "message": "joined 'Brightspeed_0112F425'", "attempt": 3})
    encoded = _encode_status(status)
    assert json.loads(encoded) == status
    assert len(encoded) < len(json.dumps(status).encode())
    assert b", " not in encoded


def test_f10_a_status_never_exceeds_the_gatt_attribute_and_keeps_what_the_app_reads():
    ssid = "\U0001F3E0" * 8  # a 32-byte SSID, the maximum
    long_error = "Error: " + "x" * 900
    status = _status({"state": "error", "ssid": ssid, "error": long_error,
                      "message": "m" * 300, "attempt": 7}, ssid=ssid)
    encoded = _encode_status(status)
    assert len(encoded) <= _GATT_MAX_ATTRIBUTE_BYTES
    parsed = json.loads(encoded)
    assert parsed["device_id"] == "SD-A1B2C3D4"
    assert parsed["provision"]["state"] == "error"
    assert parsed["provision"]["attempt"] == 7
    assert parsed["provision"]["error"].startswith("Error: ")


def test_f10_later_read_chunks_come_from_the_same_document(stub_dbus):
    """A status change between two Read Blob requests spliced two documents."""
    server = _server()
    server.start()
    try:
        status_chrc = stub_dbus.chrc(2)
        server.set_status({"state": "provisioning", "provision": {"state": "connecting", "attempt": 1}})
        first = bytes(status_chrc.ReadValue({"offset": 0}))
        server.set_status({"state": "online", "provision": {"state": "ok", "ip": "10.0.0.9", "attempt": 1}})
        rest = bytes(status_chrc.ReadValue({"offset": 10}))
        assert rest == first[10:]
    finally:
        server.stop()


# ============================================================================ F11
def test_f11_a_stop_during_setup_does_not_leave_the_server_running(stub_dbus, monkeypatch):
    """stop() used to find no main loop yet, quit nothing, and the thread went on
    to serve (and advertise) for the life of the process."""
    monkeypatch.setattr(BLEProvisioningServer, "_START_TIMEOUT_S", 0.05)
    stub_dbus.find_gate = threading.Event()   # setup blocks in GetManagedObjects
    server = _server()
    server.start()                             # returns after the short wait
    thread = server._thread
    threading.Timer(0.2, stub_dbus.find_gate.set).start()
    server.stop()
    thread.join(timeout=3)
    assert not thread.is_alive(), "the GATT thread kept running after stop()"
    assert not stub_dbus.registry
    assert not any("add-adv" in c for c in stub_dbus.subprocess_calls)


# ============================================================================ F12
def test_f12_a_long_write_is_reassembled_from_its_pieces():
    payload = json.dumps({"ssid": "Cafe: Free", "psk": "p" * 60}).encode()
    assembler = _WriteAssembler()
    assert assembler.feed(payload[:20], 0) is None
    assert assembler.feed(payload[20:45], 20) is None
    assert assembler.feed(payload[45:], 45) == payload


def test_f12_out_of_order_and_oversized_writes_are_dropped():
    assembler = _WriteAssembler()
    assert assembler.feed(b'{"ssid":', 0) is None
    assert assembler.feed(b'"x"}', 3) is None          # gap: offset does not follow
    assert assembler.feed(b'{"ssid":"x"}', 0) == b'{"ssid":"x"}'
    assert _WriteAssembler(limit=8).feed(b'{"ssid":"toolong"}', 0) is None


def test_f12_credential_and_command_payloads_are_validated():
    assert parse_credentials(b'{"ssid":"Home","psk":"pw"}') == ("Home", "pw")
    assert parse_credentials(b'{"ssid":"Open","psk":null}') == ("Open", "")
    assert parse_credentials(b'{"ssid":"Open"}') == ("Open", "")
    for bad in (b'[]', b'"x"', b'{"ssid":null}', b'{"ssid":5}', b'{"ssid":"a","psk":1}', b"\xff", b"{"):
        assert parse_credentials(bad) is None, bad
    assert parse_command(b'{"action":"rescan"}') == {"action": "rescan"}
    assert parse_command(b'["rescan"]') is None


def test_f12_the_gatt_characteristic_dispatches_a_chunked_credential_write(stub_dbus):
    got = []
    server = _server(on_credentials=lambda ssid, psk: got.append((ssid, psk)))
    server.start()
    try:
        creds = stub_dbus.chrc(1)
        data = json.dumps({"ssid": "HomeNet", "psk": "hunter22"}).encode()
        creds.WriteValue(list(data[:10]), {})
        creds.WriteValue(list(data[10:]), {"offset": 10})
        creds.WriteValue(list(b'{"ssid": null}'), {})
        assert got == [("HomeNet", "hunter22")]
    finally:
        server.stop()


# ============================================================================ F4 / F8 / F13 / extra 3
class _FakeNM:
    """nmcli stand-in that ESCAPES terse output exactly as nmcli does."""

    def __init__(self, *, active="Home", profiles=("Home",), scan="", psk_stdout=""):
        self.active = active
        self.profiles = list(profiles)
        self.scan = scan
        self.psk_stdout = psk_stdout
        self.calls: list = []

    @staticmethod
    def esc(value):
        return value.replace("\\", "\\\\").replace(":", "\\:")

    def run(self, args, timeout=10):
        self.calls.append(list(args))
        ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(returncode=0, stderr="", stdout="".join(
                f"{self.esc(n)}:802-11-wireless\n" for n in self.profiles))
        if args[:3] == ["-t", "-f", "NAME,TYPE,DEVICE"]:
            return types.SimpleNamespace(returncode=0, stderr="",
                                         stdout=f"{self.esc(self.active)}:802-11-wireless:wlan0\n")
        if args[:3] == ["-t", "-f", "STATE"]:
            return types.SimpleNamespace(returncode=0, stderr="", stdout="connected\n")
        if args[:3] == ["-t", "-f", "ACTIVE,SSID"]:
            return types.SimpleNamespace(returncode=0, stderr="",
                                         stdout=f"no:Other\nyes:{self.esc(self.active)}\n")
        if args[:3] == ["-t", "-f", "IP4.ADDRESS"]:
            return types.SimpleNamespace(returncode=0, stderr="", stdout="IP4.ADDRESS[1]:192.168.1.9/24\n")
        if args[:3] == ["-t", "-f", "SSID,SECURITY"]:
            return types.SimpleNamespace(returncode=0, stderr="", stdout=self.scan)
        if "-g" in args and "802-11-wireless-security.psk" in args:
            return types.SimpleNamespace(returncode=0, stderr="", stdout=self.psk_stdout)
        if args[:2] == ["connection", "add"]:
            self.profiles.append(args[args.index("con-name") + 1])
            return ok
        if args[:2] == ["connection", "up"]:
            self.active = args[-1]
            return ok
        if args[:2] == ["connection", "delete"]:
            if args[-1] in self.profiles:
                self.profiles.remove(args[-1])
            return ok
        return ok


@pytest.fixture()
def nm(monkeypatch):
    fake = _FakeNM()
    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", fake.run)
    monkeypatch.setattr(network, "_SETTLE_TIMEOUT_S", 0.3)
    return fake


@pytest.mark.parametrize(("raw", "fields"), [
    ("yes:Cafe\\: Free", ["yes", "Cafe: Free"]),
    ("a\\\\b:WPA2", ["a\\b", "WPA2"]),
    ("ends\\\\:80:WPA2", ["ends\\", "80", "WPA2"]),
    (":70:WPA2", ["", "70", "WPA2"]),
    ("plain", ["plain"]),
])
def test_f4_terse_fields_are_split_and_unescaped(raw, fields):
    assert _split_terse(raw) == fields


def test_f4_queries_return_names_as_the_app_knows_them(nm):
    nm.active = "Cafe: Free"
    nm.profiles = ["Cafe: Free", "Back\\slash"]
    wifi = WifiManager()
    assert wifi.current_ssid() == "Cafe: Free"
    assert wifi.active_profile() == "Cafe: Free"
    assert wifi.saved_networks() == ["Cafe: Free", "Back\\slash"]
    assert wifi.saved_wifi_names() == ["Cafe: Free", "Back\\slash"]


def test_current_ssid_never_triggers_a_radio_scan(nm):
    """Called every few seconds for health; a bare `device wifi` list rescans
    whenever the AP list is older than 30s."""
    WifiManager().current_ssid()
    query = next(c for c in nm.calls if "ACTIVE,SSID" in c)
    assert query[-2:] == ["--rescan", "no"]


def test_f4_a_network_with_a_colon_in_its_name_can_be_provisioned(nm):
    """The join worked; current_ssid() said "Cafe\\: Free"; provision() waited out
    its settle time, reported failure and rolled the working join back."""
    ok, message = WifiManager().provision("Cafe: Free", "pw123456")
    assert ok, message
    assert nm.active == "Cafe: Free"
    assert "Cafe: Free" in nm.profiles


def test_f8_a_restored_passphrase_is_byte_for_byte_the_stored_one(nm):
    nm.psk_stdout = " pass:word\\ \n"   # what `-e no` prints for " pass:word\ "
    assert WifiManager._stored_psk("Home") == " pass:word\\ "
    query = next(c for c in nm.calls if "802-11-wireless-security.psk" in c)
    assert query[:2] == ["-e", "no"], "terse output escapes ':' and '\\' unless -e no"


@pytest.mark.parametrize(("security", "expected"), [
    ("WPA3", "sae"), ("WPA2 WPA3", "wpa-psk"), ("WPA2", "wpa-psk"), ("WPA1 WPA2", "wpa-psk"),
])
def test_f13_key_management_follows_the_network(nm, security, expected):
    nm.scan = f"Other:WPA2\nNewNet:{security}\n"
    WifiManager().provision("NewNet", "password1")
    add = next(c for c in nm.calls if c[:2] == ["connection", "add"])
    assert add[add.index("wifi-sec.key-mgmt") + 1] == expected
    assert "802-11-wireless.hidden" not in add


def test_f13_a_typed_network_that_is_not_in_the_scan_is_added_as_hidden(nm):
    nm.scan = "Other:WPA2\n"
    WifiManager().provision("Secret", "password1")
    add = next(c for c in nm.calls if c[:2] == ["connection", "add"])
    assert add[add.index("802-11-wireless.hidden") + 1] == "yes"


def test_f13_connection_names_always_go_through_the_id_selector(nm):
    nm.active = "Home"
    WifiManager().provision("uuid", "password1")   # a legal SSID that is also an nmcli keyword
    for call in nm.calls:
        if call[:2] in (["connection", "up"], ["connection", "delete"], ["connection", "modify"]):
            assert call[2] == "id", call


# ============================================================================ F7
def test_f7_a_stalled_retry_does_not_forget_a_network_that_worked(monkeypatch):
    calls = []

    def run(args, timeout=10):
        calls.append(args)
        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(returncode=0, stdout="HomeNet:802-11-wireless\n", stderr="")
        raise subprocess.TimeoutExpired(cmd="nmcli", timeout=45)

    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", run)
    ok, _ = WifiManager().connect("HomeNet", "typo")
    assert ok is False
    assert not any(c[:2] == ["connection", "delete"] for c in calls)


def test_f7_setup_wifi_uses_the_validate_first_path(monkeypatch):
    from core.dashboard import DashboardServer

    used = []
    monkeypatch.setattr(WifiManager, "provision", lambda self, s, p: used.append("provision") or (True, "ok"))
    monkeypatch.setattr(WifiManager, "connect", lambda self, s, p: used.append("connect") or (True, "ok"))
    DashboardServer(host="127.0.0.1", port=0).apply_wifi("HomeNet", "pw")
    assert used == ["provision"]


def test_f7_the_setup_page_carries_the_token_and_the_endpoint_accepts_it():
    from core import dashboard
    from core.dashboard import DashboardServer

    script = dashboard._SETUP_PAGE.split("<script>", 1)[1]
    assert 'URLSearchParams(location.search).get("token")' in script
    assert "X-Device-Token" in script

    server = DashboardServer(host="0.0.0.0", port=0, token="s3cret")  # noqa: S104
    url = server.start().replace("localhost", "127.0.0.1")
    try:
        body = json.dumps({"ssid": "HomeNet", "password": "pw"}).encode()
        request = urllib.request.Request(url + "/setup/wifi", data=body, headers={  # noqa: S310
            "Content-Type": "application/json", "X-Device-Token": "s3cret"})
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - fixed http:// URL
            assert response.status == 200
            assert json.loads(response.read())["ok"] is True
    finally:
        server.close()


# ============================================================================ F6
_HOTSPOT = REPO / "scripts" / "provisioning_hotspot.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("state", "online"), [
    ("connected", True),
    ("connected (site only)", True),
    ("connected (local only)", True),
    ("connecting", False),
    ("disconnected", False),
])
def test_f6_the_hotspot_script_recognises_a_connected_device(state, online):
    """Full connectivity prints as plain "connected"; the old pattern wanted
    "connected (global" and started a hotspot over a working connection."""
    script = f'nmcli() {{ echo "{state}"; }}; source "{_HOTSPOT}"; nm_online && echo ONLINE || echo OFFLINE'
    out = subprocess.run(  # noqa: S603 - fixed test command
        ["bash", "-c", script], capture_output=True, text=True, timeout=10, check=False,  # noqa: S607
    )
    assert out.stdout.strip() == ("ONLINE" if online else "OFFLINE"), out.stderr


# ============================================================================ F3 + notify
class _Backend:
    def __init__(self, mode):
        self.mode = mode
        self.seen: list = []
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, status, location=None, body=b""):
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                backend.seen.append((self.command, self.path, self.headers.get("X-Device-Key")))
                if backend.mode == "redirect":
                    self._reply(301, f"http://localhost:{backend.port}/elsewhere/v1/ingest")
                else:
                    self._reply(int(backend.mode))

            def do_GET(self):
                backend.seen.append((self.command, self.path, self.headers.get("X-Device-Key")))
                self._reply(405)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _uplink_once(base_url, events=3):
    uplink = CloudUplink({"base_url": base_url, "device_key": "dk_secret", "poll_interval_s": 0})
    for i in range(events):
        uplink.push_event(types.SimpleNamespace(
            timestamp=types.SimpleNamespace(isoformat=lambda: "2026-10-15T00:00:00"),
            type=f"fall_{i}", confidence=0.9, contributing_signals="", duration=1.0), "loud")
    post = uplink._requests.post

    def once(*args, **kwargs):
        try:
            return post(*args, **kwargs)
        finally:
            uplink._stop.set()

    uplink._requests = types.SimpleNamespace(post=once)
    uplink._run()
    return uplink


def test_f3_a_redirect_is_not_followed_and_drops_nothing(capsys):
    backend = _Backend("redirect")
    try:
        uplink = _uplink_once(f"http://127.0.0.1:{backend.port}")
    finally:
        backend.close()
    assert backend.seen == [("POST", "/v1/ingest", "dk_secret")], "the device key followed the redirect"
    assert len(uplink._queue) == 3, "events were dropped for a redirect"
    out = capsys.readouterr().out
    assert "redirect" in out.lower() and "base_url" in out


@pytest.mark.parametrize("status", [404, 405, 410, 401, 429, 500, 503])
def test_f3_answers_that_say_nothing_about_the_batch_keep_it(status):
    backend = _Backend(str(status))
    try:
        uplink = _uplink_once(f"http://127.0.0.1:{backend.port}")
    finally:
        backend.close()
    assert len(uplink._queue) == 3, f"HTTP {status} dropped the batch"


@pytest.mark.parametrize(("status", "events"), [(400, 3), (422, 3), (413, 1)])
def test_f3_a_body_the_backend_will_always_refuse_is_still_dropped(status, events):
    backend = _Backend(str(status))
    try:
        uplink = _uplink_once(f"http://127.0.0.1:{backend.port}", events=events)
    finally:
        backend.close()
    assert len(uplink._queue) == 0


def test_notify_is_sent_only_when_a_rule_chose_one():
    uplink = CloudUplink({"base_url": "http://unused.invalid", "device_key": "k"})
    event = types.SimpleNamespace(
        timestamp=types.SimpleNamespace(isoformat=lambda: "2026-10-15T00:00:00"),
        type="e", confidence=0.5, contributing_signals="", duration=0.0)
    uplink.push_event(event, "soft")
    uplink.push_event(event, "soft", source="custom_rule", notify="sms")
    first, second = list(uplink._queue)
    assert "notify" not in first
    assert second["notify"] == "sms"


# ============================================================================ F14
def test_f14_sudoers_lists_exactly_the_commands_the_code_runs():
    rules = [line for line in (REPO / "deploy" / "sudoers" / "sensordyme-reboot").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert all(line.startswith("__SENSORDYME_USER__ ") for line in rules), "user must be templated"
    commands = {line.split("NOPASSWD: ", 1)[1] for line in rules}
    instance = BLEProvisioningServer._BTMGMT_INSTANCE
    expected = {
        "/usr/bin/systemctl reboot",
        "/usr/bin/btmgmt connectable on",
        f"/usr/bin/btmgmt rm-adv {instance}",
        f"/usr/bin/btmgmt add-adv -u {ble_gatt.SERVICE_UUID} -c -g -n {instance}",
        "/usr/bin/btmgmt advinfo",
    }
    assert commands == expected
    assert "/usr/bin/btmgmt" not in commands, "btmgmt with ANY arguments is back"


def test_f14_the_argv_ble_gatt_builds_matches_the_sudoers_lines(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    server = _server()
    server._advertise_via_btmgmt()
    server._stop_btmgmt_advertising()
    server.is_advertising()
    rules = (REPO / "deploy" / "sudoers" / "sensordyme-reboot").read_text()
    for cmd in calls:
        assert cmd[:2] == ["sudo", "-n"]
        assert f"NOPASSWD: /usr/bin/{' '.join(cmd[2:])}\n" in rules, cmd


def test_f14_setup_validates_the_rendered_rule_before_installing_it():
    setup = (REPO / "setup_pi.sh").read_text()
    assert "visudo -cf" in setup
    assert setup.index("visudo -cf") < setup.index("/etc/sudoers.d/sensordyme-reboot\n")
    assert "__SENSORDYME_USER__" in setup and "SERVICE_USER" in setup
    unit = (REPO / "sensordyme.service").read_text()
    assert "User=__SENSORDYME_USER__" in unit and "User=pi" not in unit
    assert "s#__SENSORDYME_USER__#${SERVICE_USER}#g" in setup


@pytest.mark.skipif(shutil.which("visudo") is None, reason="needs visudo")
def test_f14_the_rendered_rule_parses(tmp_path):
    rendered = (REPO / "deploy" / "sudoers" / "sensordyme-reboot").read_text().replace(
        "__SENSORDYME_USER__", "sensordyme")
    path = tmp_path / "sensordyme-reboot"
    path.write_text(rendered)
    result = subprocess.run(  # noqa: S603 - fixed test command
        ["visudo", "-cf", str(path)], capture_output=True, text=True, timeout=10, check=False,  # noqa: S607
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ============================================================================ F16
def test_f16_connections_beyond_the_cap_are_refused_not_threaded(monkeypatch):
    from core import dashboard
    from core.dashboard import DashboardServer

    monkeypatch.setattr(dashboard, "_MAX_CONNECTIONS", 2)
    server = DashboardServer(host="127.0.0.1", port=0)
    server.start()
    idle = [socket.create_connection(("127.0.0.1", server._port), timeout=5) for _ in range(2)]
    try:
        time.sleep(0.2)  # both are now parked handler threads waiting for a request line
        extra = socket.create_connection(("127.0.0.1", server._port), timeout=5)
        extra.settimeout(3)
        assert extra.recv(1) == b"", "a third connection was served beyond the cap"
        extra.close()
        for s in idle:
            s.close()
        assert _wait_for(lambda: _get_status(server) == 200, timeout=5), "slots were never released"
    finally:
        for s in idle:
            s.close()
        server.close()


def _get_status(server):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server._port}/api/state", timeout=2) as r:
            return r.status
    except (urllib.error.URLError, OSError):
        return None


def test_f16_the_peer_check_works_on_descriptors_above_1023():
    from core.dashboard import _peer_closed

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 2100:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
        except (ValueError, OSError):
            pytest.skip("cannot raise the descriptor limit")
    a, b = socket.socketpair()
    high = os.dup2(a.fileno(), 2048)
    sock = socket.socket(fileno=high)
    try:
        assert _peer_closed(sock) is False, "a live peer was reported gone (select() fd limit)"
        b.close()
        assert _peer_closed(sock) is True
    finally:
        sock.close()
        a.close()


# ============================================================================ F17
def _load_client(monkeypatch):
    bleak = types.ModuleType("bleak")
    bleak.BleakClient = bleak.BleakScanner = object
    monkeypatch.setitem(sys.modules, "bleak", bleak)
    spec = importlib.util.spec_from_file_location(
        "ble_provision_client_under_test", REPO / "scripts" / "ble_provision_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f17_the_dev_client_ignores_the_previous_attempts_outcome(monkeypatch):
    client = _load_client(monkeypatch)
    stale = {"provision": {"state": "error", "error": "old typo", "attempt": 4}}
    connecting = {"provision": {"state": "connecting", "attempt": 5}}
    done = {"provision": {"state": "ok", "ip": "10.0.0.2", "attempt": 5}}
    assert client.pick_outcome([stale, connecting], baseline_attempt=4) is None
    assert client.pick_outcome([stale, connecting, stale, done], baseline_attempt=4) == done["provision"]
    # Firmware without attempt numbers: only what follows this write's "connecting".
    old_stale = {"provision": {"state": "error", "error": "old"}}
    old_connecting = {"provision": {"state": "connecting"}}
    old_done = {"provision": {"state": "ok"}}
    assert client.pick_outcome([old_stale]) is None
    assert client.pick_outcome([old_stale, old_connecting, old_done]) == {"state": "ok"}


def test_f9_a_status_described_before_a_write_cannot_republish_the_old_outcome(fake_ble):
    """The run loop describes, then a write replaces the outcome, then the loop
    pushes: that push used to carry the previous attempt's answer."""
    service = _service(_FakeWifi())
    ble = _FakeBLE("x", {}, lambda *a: None)
    ble.running = True
    service._ble = ble
    service._provision = {"state": "error", "ssid": "HomeNet", "error": "old", "attempt": 1}
    service._attempt = 1
    described_earlier = service._describe_cached()
    release = threading.Event()
    service._wifi.provision = lambda s, p: (release.wait(5), (True, "joined"))[1]
    service._on_credentials("HomeNet", "pw")
    service._push_status(described_earlier)
    try:
        assert ble.statuses[-1]["provision"]["attempt"] == 2
        assert ble.statuses[-1]["provision"]["state"] == "connecting"
    finally:
        release.set()
        service._connect_thread.join(timeout=3)
