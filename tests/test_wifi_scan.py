"""Scanning for Wi-Fi networks on the device.

The device scans, not the phone. iOS cannot enumerate networks without an
entitlement Apple effectively does not grant — and even if it could, the phone's
list is the wrong one. The Pi is the machine that has to associate, and offering
it a network it cannot see produces a setup that fails with no explanation.
"""

import core.network as network
from core.network import WifiManager


def fake_nmcli(monkeypatch, stdout: str):
    class Result:
        pass

    result = Result()
    result.stdout = stdout
    result.returncode = 0
    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", lambda *args, **kwargs: result)


def test_no_nmcli_returns_nothing_rather_than_failing(monkeypatch):
    """A dev Mac has no NetworkManager; provisioning never runs there anyway."""
    monkeypatch.setattr(network, "nmcli_available", lambda: False)
    assert WifiManager().scan() == []


def test_networks_are_parsed_strongest_first(monkeypatch):
    fake_nmcli(monkeypatch, "Office:55:WPA3\nHomeNet:82:WPA2\nGuest:61:WPA2\n")
    networks = WifiManager().scan()
    assert [n["ssid"] for n in networks] == ["HomeNet", "Guest", "Office"]
    assert networks[0]["signal"] == 82


def test_duplicate_access_points_collapse_to_the_strongest(monkeypatch):
    """A mesh reports one row per access point; three copies of one network is
    not a list, it is noise."""
    fake_nmcli(monkeypatch, "HomeNet:41:WPA2\nHomeNet:88:WPA2\nHomeNet:63:WPA2\n")
    networks = WifiManager().scan()
    assert len(networks) == 1
    assert networks[0]["signal"] == 88


def test_hidden_networks_are_omitted(monkeypatch):
    """No SSID to show. A blank row is worse than no row; the name can still be
    typed by hand."""
    fake_nmcli(monkeypatch, ":70:WPA2\nHomeNet:82:WPA2\n")
    assert [n["ssid"] for n in WifiManager().scan()] == ["HomeNet"]


def test_an_ssid_containing_a_colon_survives_parsing(monkeypatch):
    """nmcli escapes colons inside fields; splitting naively renames the network."""
    fake_nmcli(monkeypatch, r"Cafe\: Free:45:" + "\n")
    networks = WifiManager().scan()
    assert networks[0]["ssid"] == "Cafe: Free"


def test_open_networks_are_marked_unsecured(monkeypatch):
    fake_nmcli(monkeypatch, "OpenWifi:60:--\nLocked:60:WPA2\n")
    by_name = {n["ssid"]: n for n in WifiManager().scan()}
    assert by_name["OpenWifi"]["secured"] is False
    assert by_name["Locked"]["secured"] is True


def test_a_malformed_signal_does_not_drop_the_network(monkeypatch):
    fake_nmcli(monkeypatch, "HomeNet:notanumber:WPA2\n")
    networks = WifiManager().scan()
    assert len(networks) == 1
    assert networks[0]["signal"] == 0


def test_short_lines_are_skipped(monkeypatch):
    fake_nmcli(monkeypatch, "garbage\nHomeNet:82:WPA2\n")
    assert [n["ssid"] for n in WifiManager().scan()] == ["HomeNet"]


def test_the_list_is_capped(monkeypatch):
    fake_nmcli(monkeypatch, "".join(f"Net{i}:{i}:WPA2\n" for i in range(50)))
    assert len(WifiManager().scan(limit=5)) == 5


def test_a_failed_scan_never_breaks_provisioning(monkeypatch):
    """Losing the list is a worse setup screen; raising is a dead device."""
    monkeypatch.setattr(network, "nmcli_available", lambda: True)

    def explode(*args, **kwargs):
        raise OSError("wifi radio busy")

    monkeypatch.setattr(network, "_run", explode)
    assert WifiManager().scan() == []
