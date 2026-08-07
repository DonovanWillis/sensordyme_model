"""Tests for the Wi-Fi/BLE provisioning state machine (fake wifi + fake BLE)."""

import threading
import time

import services.provisioning as provisioning_module
from services.provisioning import ProvisioningService


class FakeWifi:
    def __init__(self, online=False):
        self.online = online
        self.connect_calls = []

    def is_online(self):
        return self.online

    def current_ssid(self):
        return "TestNet" if self.online else None

    def ip_address(self):
        return "192.168.1.50" if self.online else None

    def watch(self, on_change, stop):
        # Mirrors WifiManager.watch. The real one wakes the loop on any
        # NetworkManager event; here the loop just falls back to its interval.
        self.watched = True
        return None

    def connect(self, ssid, psk):
        self.connect_calls.append((ssid, psk))
        self.online = True  # credentials "work"
        return True, f"joined '{ssid}'"


class FakeBLEServer:
    instances = []

    def __init__(self, device_name, device_info, on_credentials, on_command=None,
                 scan_networks=None):
        # Mirrors the real BLEProvisioningServer signature. A double that
        # drifts from it fails at construction and every assertion after it
        # reports something unrelated.
        self.device_name = device_name
        self.scan_networks = scan_networks
        self.on_credentials = on_credentials
        self.statuses = []
        self.running = False
        FakeBLEServer.instances.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def set_status(self, status):
        self.statuses.append(status)

    def refresh_networks(self):
        self.refreshed = getattr(self, "refreshed", 0) + 1


def _make_service(monkeypatch, wifi, grace_s=0.05, poll_s=0.02, reconfigure_window_s=0.0):
    # Pretend we're on a Pi and swap the real GATT server for the fake.
    monkeypatch.setattr(provisioning_module, "nmcli_available", lambda: True)
    import services.ble_gatt as ble_gatt

    monkeypatch.setattr(ble_gatt, "BLEProvisioningServer", FakeBLEServer)
    FakeBLEServer.instances = []
    return ProvisioningService(wifi, {
        "method": "ble",
        "grace_s": grace_s,
        "poll_s": poll_s,
        "reconfigure_window_s": reconfigure_window_s,
    })


def _run_briefly(service, seconds):
    stop = threading.Event()
    thread = threading.Thread(target=service.run, args=(stop,))
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join(timeout=2)


def test_a_settled_online_device_does_not_advertise(monkeypatch):
    """Past the reconfigure window, a working device is silent on Bluetooth.

    Advertising forever would leave a provisioning surface open on every
    SensorDyme in the world for anyone within radio range.
    """
    service = _make_service(monkeypatch, FakeWifi(online=True), reconfigure_window_s=0.0)
    _run_briefly(service, 0.2)
    assert FakeBLEServer.instances == []
    assert service.state == "online"


def test_a_freshly_booted_online_device_can_still_be_moved(monkeypatch):
    """The point of the reconfigure window.

    Setup used to be reachable only while the device was offline, so pointing a
    working device at a different network meant first breaking its Wi-Fi. Taking
    a product apart to reconfigure it is not something a consumer should do, and
    "move it to the new house" is a normal thing to want.
    """
    service = _make_service(monkeypatch, FakeWifi(online=True), reconfigure_window_s=30)
    _run_briefly(service, 0.2)
    assert len(FakeBLEServer.instances) == 1
    assert FakeBLEServer.instances[0].running
    assert service.state == "reconfigurable"


def test_the_window_reports_the_device_id_so_the_app_can_match_it(monkeypatch):
    """Found over Bluetooth and later over Wi-Fi, it must be one device."""
    service = _make_service(monkeypatch, FakeWifi(online=True), reconfigure_window_s=30)
    _run_briefly(service, 0.2)
    statuses = FakeBLEServer.instances[0].statuses
    assert statuses, "the device never described itself"
    assert statuses[-1]["device_id"].startswith("SD-")


def test_a_successful_setup_closes_the_window_early(monkeypatch):
    """Once someone has deliberately pointed the device at a network, the
    window has served its purpose; holding it open is exposure for nothing."""
    wifi = FakeWifi(online=False)
    service = _make_service(monkeypatch, wifi, reconfigure_window_s=30)
    stop = threading.Event()
    thread = threading.Thread(target=service.run, args=(stop,))
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not FakeBLEServer.instances:
            time.sleep(0.01)
        ble = FakeBLEServer.instances[0]
        ble.on_credentials("HomeNet", "secret")

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and ble.running:
            time.sleep(0.01)
        assert not ble.running, "window stayed open after a successful setup"
    finally:
        stop.set()
        thread.join(timeout=2)


def test_offline_boot_opens_ble_after_grace(monkeypatch):
    service = _make_service(monkeypatch, FakeWifi(online=False))
    _run_briefly(service, 0.3)
    assert len(FakeBLEServer.instances) == 1
    assert FakeBLEServer.instances[0].running
    assert service.state == "provisioning"
    service.stop()  # teardown closes the window
    assert not FakeBLEServer.instances[0].running


def test_credentials_connect_save_and_close_the_window(monkeypatch):
    wifi = FakeWifi(online=False)
    service = _make_service(monkeypatch, wifi)
    stop = threading.Event()
    thread = threading.Thread(target=service.run, args=(stop,))
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not FakeBLEServer.instances:
            time.sleep(0.01)
        ble = FakeBLEServer.instances[0]

        ble.on_credentials("HomeNet", "secret")  # the app writes credentials
        assert wifi.connect_calls == [("HomeNet", "secret")]
        assert any(s.get("state") == "online" for s in ble.statuses)  # success notified

        deadline = time.monotonic() + 2  # watcher loop notices online and closes BLE
        while time.monotonic() < deadline and ble.running:
            time.sleep(0.01)
        assert not ble.running
        assert service.state == "online"
    finally:
        stop.set()
        thread.join(timeout=2)


def test_dormant_off_pi():
    service = ProvisioningService(FakeWifi(), {"method": "ble"})  # real nmcli_available: Mac
    service.run(threading.Event())  # returns immediately
    assert service.state == "unavailable"


def test_method_off_is_dormant(monkeypatch):
    monkeypatch.setattr(provisioning_module, "nmcli_available", lambda: True)
    service = ProvisioningService(FakeWifi(), {"method": "off"})
    service.run(threading.Event())
    assert service.state == "off"


def test_joining_wifi_does_not_block_the_ble_event_loop(monkeypatch):
    """The credential write runs on the GLib main loop thread. Joining a network
    takes up to 45 seconds, and while that blocked the loop no status could
    reach the phone — the app showed "Sending…" and silence for the whole
    attempt, which reads as a broken device rather than a slow one."""
    import threading
    import time as _time

    from core.network import WifiManager
    from services.provisioning import ProvisioningService

    started = threading.Event()
    release = threading.Event()

    class SlowWifi(WifiManager):
        def is_online(self):
            return False

        def connect(self, ssid, password):
            started.set()
            release.wait(timeout=5)
            return True, "joined"

    service = ProvisioningService(SlowWifi(), {"method": "ble"})
    service._ble = FakeBLEServer("x", {}, lambda *a: None)

    began = _time.monotonic()
    service._on_credentials("HomeNet", "secret")
    elapsed = _time.monotonic() - began

    # The handler must return immediately rather than waiting for the join.
    assert elapsed < 0.5, f"credential handler blocked for {elapsed:.1f}s"
    assert started.wait(timeout=2), "the join never started"

    # And "connecting" must already be visible to the phone.
    assert service._ble.statuses[-1].get("state") == "connecting"

    release.set()


def test_a_second_credential_write_while_joining_is_ignored(monkeypatch):
    """Two nmcli invocations racing for the same radio is worse than a dropped
    duplicate write."""
    import threading

    from core.network import WifiManager
    from services.provisioning import ProvisioningService

    calls = []
    release = threading.Event()

    class SlowWifi(WifiManager):
        def is_online(self):
            return False

        def connect(self, ssid, password):
            calls.append(ssid)
            release.wait(timeout=5)
            return True, "joined"

    service = ProvisioningService(SlowWifi(), {"method": "ble"})
    service._ble = FakeBLEServer("x", {}, lambda *a: None)

    service._on_credentials("First", "a")
    service._on_credentials("Second", "b")
    release.set()
    if service._connect_thread:
        service._connect_thread.join(timeout=5)

    assert calls == ["First"]


def test_the_service_subscribes_to_network_changes(monkeypatch):
    """Polling alone put 8.1 seconds (measured on a Pi 5) between losing Wi-Fi
    and being visible on Bluetooth, nearly all of it asleep in the interval.
    The device should be told, not discover it on the next tick."""
    wifi = FakeWifi(online=True)
    service = _make_service(monkeypatch, wifi, reconfigure_window_s=0)
    _run_briefly(service, 0.15)
    assert getattr(wifi, "watched", False), "never subscribed to NetworkManager events"


def test_stopping_wakes_the_loop_rather_than_waiting_out_the_interval(monkeypatch):
    """With a 10s production interval, a shutdown that waits it out looks like a
    hung service to systemd and eventually gets killed."""
    service = _make_service(monkeypatch, FakeWifi(online=True), poll_s=30, reconfigure_window_s=0)
    service.start()
    began = time.monotonic()
    service.stop(timeout=5)
    assert time.monotonic() - began < 3, "stop() waited out the poll interval"
