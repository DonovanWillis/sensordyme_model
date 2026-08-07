"""Moving a working device to a different network.

The dangerous moment in this product: someone carries a device into a new
building and re-points it over Bluetooth. If the new password is wrong and the
old connection has already been given up, the device is on nothing, possibly out
of Bluetooth range, with no way back. The working network is not released until
the new one is known to work.
"""

import types

import core.network as network
from core.network import WifiManager


class FakeNM:
    """A NetworkManager stand-in that tracks profiles and the active connection."""

    def __init__(self, *, active="Hotspot", saved=("Hotspot",), joinable=()):
        self.profiles = list(saved)
        self.active = active
        self.joinable = set(joinable)
        self.calls = []

    def run(self, args, timeout=10):
        self.calls.append(args)
        ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")

        if args[:3] == ["-t", "-f", "NAME,TYPE"]:
            return types.SimpleNamespace(
                returncode=0, stderr="",
                stdout="".join(f"{n}:802-11-wireless\n" for n in self.profiles))
        if args[:2] == ["-t", "-f"] and "NAME,TYPE,DEVICE" in args[2]:
            return types.SimpleNamespace(
                returncode=0, stderr="",
                stdout=f"{self.active}:802-11-wireless:wlan0\n" if self.active else "")
        if args[:3] == ["-t", "-f", "STATE"]:
            return types.SimpleNamespace(
                returncode=0, stderr="",
                stdout="connected\n" if self.active else "disconnected\n")
        if args[:2] == ["-t", "-f"] and "ACTIVE,SSID" in args[2]:
            return types.SimpleNamespace(
                returncode=0, stderr="", stdout=f"yes:{self.active}\n" if self.active else "")
        if args[:2] == ["connection", "add"]:
            self.profiles.append(args[args.index("con-name") + 1])
            return ok
        if args[:2] == ["connection", "delete"]:
            if args[2] in self.profiles:
                self.profiles.remove(args[2])
            return ok
        if args[:2] == ["connection", "up"]:
            target = args[2]
            if target in self.joinable:
                self.active = target
                return ok
            return types.SimpleNamespace(
                returncode=4, stdout="",
                stderr="Error: Connection activation failed: (7) Secrets were required.")
        return ok


def install(monkeypatch, nm):
    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", nm.run)
    monkeypatch.setattr(WifiManager, "ip_address", lambda self: "192.168.1.37" if nm.active else None)
    return WifiManager()


def test_the_profile_is_saved_before_the_working_network_is_given_up(monkeypatch):
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"HomeWiFi", "Hotspot"})
    wifi = install(monkeypatch, nm)
    assert wifi.provision("HomeWiFi", "correct")[0] is True

    add = next(i for i, c in enumerate(nm.calls) if c[:2] == ["connection", "add"])
    up = next(i for i, c in enumerate(nm.calls) if c[:2] == ["connection", "up"] and c[2] == "HomeWiFi")
    assert add < up, "the device switched away before the new network was saved"


def test_a_wrong_password_leaves_the_device_on_its_original_network(monkeypatch):
    """The whole point. A mistyped password must cost a few seconds, not the
    device — it may be somewhere with no other way to reach it."""
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"Hotspot"})
    wifi = install(monkeypatch, nm)

    ok, message = wifi.provision("HomeWiFi", "wrong")

    assert ok is False
    assert "password" in message.lower()
    assert nm.active == "Hotspot", "device was left off its working network"


def test_a_failed_switch_does_not_leave_the_bad_profile_saved(monkeypatch):
    """Left behind, NetworkManager retries a network that rejected it on every
    future boot, and it can outrank the one that works."""
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"Hotspot"})
    wifi = install(monkeypatch, nm)
    wifi.provision("HomeWiFi", "wrong")
    assert "HomeWiFi" not in nm.profiles


def test_a_network_that_already_worked_is_kept_when_a_retry_fails(monkeypatch):
    """Re-entering a password for a known network must not forget it."""
    nm = FakeNM(active="Hotspot", saved=["Hotspot", "HomeWiFi"], joinable={"Hotspot"})
    wifi = install(monkeypatch, nm)
    wifi.provision("HomeWiFi", "wrong")
    assert "HomeWiFi" in nm.profiles


def test_the_old_network_stays_saved_as_a_fallback_after_a_successful_move(monkeypatch):
    """Requirement: the hotspot remains available if the new network later fails."""
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"HomeWiFi", "Hotspot"})
    wifi = install(monkeypatch, nm)
    assert wifi.provision("HomeWiFi", "correct")[0] is True
    assert "Hotspot" in nm.profiles, "the fallback network was deleted"


def test_the_new_network_becomes_preferred(monkeypatch):
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"HomeWiFi", "Hotspot"})
    wifi = install(monkeypatch, nm)
    wifi.provision("HomeWiFi", "correct")
    modify = [c for c in nm.calls if c[:2] == ["connection", "modify"]
              and "connection.autoconnect-priority" in c]
    assert modify, "the newly configured network was not made preferred"


def test_activation_returning_zero_is_not_treated_as_connected(monkeypatch):
    """`nmcli connection up` returning 0 means activation began, not that the
    network works. Trusting it reports a device online that never got an
    address."""
    nm = FakeNM(active="Hotspot", saved=["Hotspot"], joinable={"HomeWiFi", "Hotspot"})
    wifi = install(monkeypatch, nm)
    monkeypatch.setattr(WifiManager, "ip_address", lambda self: None)  # never gets an address
    monkeypatch.setattr(WifiManager, "_settled_on", lambda self, ssid: False)

    ok, _ = wifi.provision("HomeWiFi", "correct")
    assert ok is False
    assert nm.active == "Hotspot"
