"""Scanning for Wi-Fi must never block the Bluetooth event loop.

Every BLE read, write, and notification is served by one GLib main-loop thread.
A scan takes up to twenty seconds of radio time. Doing that work inside a read
handler does not merely make the network list slow — it makes the entire
connection unresponsive for the duration, which is what made "Choose a Network"
appear to hang and made setup look broken.

These tests hold the boundary: reads answer from cache, scans happen elsewhere.
"""

import threading
import time

from services.ble_gatt import BLEProvisioningServer


def make_server(scan_networks):
    return BLEProvisioningServer(
        device_name="SensorDyme-TEST",
        device_info={"name": "SensorDyme-TEST"},
        on_credentials=lambda ssid, psk: None,
        scan_networks=scan_networks,
    )


def test_the_server_accepts_a_scan_callback():
    """provisioning.py passes scan_networks=; a signature that omits it raises
    TypeError inside _open_ble_window and kills the provisioning thread, leaving
    no Bluetooth window at all."""
    server = make_server(list)
    assert server._scan_networks is not None


def test_a_slow_scan_does_not_delay_the_cached_answer():
    """The regression. A read during a slow scan must return at once."""
    release = threading.Event()

    def slow_scan():
        release.wait(timeout=5)
        return [{"ssid": "HomeNet", "signal": 80, "secured": True}]

    server = make_server(slow_scan)
    server.refresh_networks()

    started = time.monotonic()
    with server._scan_lock:
        cached = list(server._networks)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, "reading the cache waited on the scan"
    assert cached == []  # honestly empty rather than blocking for a full list
    release.set()


def test_a_scan_populates_the_cache():
    server = make_server(lambda: [{"ssid": "HomeNet", "signal": 80, "secured": True}])
    server.refresh_networks()
    for _ in range(50):
        if not server._scanning:
            break
        time.sleep(0.02)
    assert [n["ssid"] for n in server._networks] == ["HomeNet"]
    assert server._scanning is False


def test_concurrent_refreshes_collapse_into_one_scan():
    """Several nmcli invocations against one radio slow each other down."""
    calls = []
    release = threading.Event()

    def counting_scan():
        calls.append(1)
        release.wait(timeout=5)
        return []

    server = make_server(counting_scan)
    for _ in range(5):
        server.refresh_networks()
    time.sleep(0.1)
    release.set()
    assert len(calls) == 1


def test_a_failed_rescan_keeps_the_networks_already_found():
    """An emptied list reads as "there are none", which is a different and wrong
    answer from "that attempt did not work"."""
    results = [[{"ssid": "HomeNet", "signal": 80, "secured": True}], []]

    def flaky_scan():
        return results.pop(0)

    server = make_server(flaky_scan)
    for _ in range(2):
        server.refresh_networks()
        for _ in range(50):
            if not server._scanning:
                break
            time.sleep(0.02)
    assert [n["ssid"] for n in server._networks] == ["HomeNet"]


def test_a_raising_scan_does_not_kill_the_worker():
    def explode():
        raise OSError("wifi radio busy")

    server = make_server(explode)
    server.refresh_networks()
    for _ in range(50):
        if not server._scanning:
            break
        time.sleep(0.02)
    assert server._scanning is False
    assert server._networks == []
