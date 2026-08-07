"""The device ID is what keeps one physical SensorDyme from becoming two.

A device is discovered over Bluetooth and over Wi-Fi, at different IPs, under
different advertised names, across reboots and network changes. Identity has to
survive all of it, or the app shows duplicates.
"""

import core.identity as identity
from core.identity import device_id, display_name, is_device_id


def test_the_id_is_well_formed():
    assert is_device_id(device_id())


def test_the_id_does_not_change_between_calls():
    assert device_id() == device_id()


def test_the_id_survives_a_hostname_or_ip_change(monkeypatch):
    """Neither is part of the derivation; this pins that they never become part
    of it, since both change every time the device moves network."""
    monkeypatch.setattr(identity, "_cpu_serial", lambda: "100000001a2b3c4d")
    first = device_id()
    monkeypatch.setattr("socket.gethostname", lambda: "somewhere-else")
    assert device_id() == first


def test_two_boards_get_different_ids(monkeypatch):
    monkeypatch.setattr(identity, "_cpu_serial", lambda: "100000001a2b3c4d")
    first = device_id()
    monkeypatch.setattr(identity, "_cpu_serial", lambda: "10000000deadbeef")
    assert device_id() != first


def test_the_serial_is_not_broadcast_in_the_clear(monkeypatch):
    """The ID goes out over mDNS to every device on the network. A raw CPU
    serial there is a permanent tracking token for the household."""
    monkeypatch.setattr(identity, "_cpu_serial", lambda: "100000001a2b3c4d")
    assert "1a2b3c4d" not in device_id().lower()


def test_a_board_reporting_a_zero_serial_falls_through_to_the_mac(monkeypatch):
    """Some boards report all zeroes rather than omitting the field, which would
    otherwise give every one of them the same identity."""
    monkeypatch.setattr(identity, "_cpu_serial", lambda: None)
    monkeypatch.setattr(identity, "_mac_address", lambda: "b8:27:eb:11:22:33")
    monkeypatch.setattr(identity, "_machine_id", lambda: None)
    assert is_device_id(device_id())


def test_an_unidentifiable_machine_still_gets_one_stable_id(monkeypatch):
    """A dev Mac has none of these. Inventing a fresh ID per call would look
    like a new device appearing on every scan."""
    monkeypatch.setattr(identity, "_cpu_serial", lambda: None)
    monkeypatch.setattr(identity, "_mac_address", lambda: None)
    monkeypatch.setattr(identity, "_machine_id", lambda: None)
    assert device_id() == device_id()


def test_the_display_name_is_derived_from_the_id():
    assert display_name("SD-A1B2C3D4") == "SensorDyme-A1B2"


def test_a_name_is_not_an_id():
    assert not is_device_id("SensorDyme-TEST")
    assert not is_device_id("SensorDyme")
