"""Tests for the WifiManager nmcli wrapper (parsing via monkeypatched _run)."""

from types import SimpleNamespace

import core.network as network_module
from core.network import WifiManager


def _fake_run(stdout="", returncode=0, stderr=""):
    def runner(args, timeout=10):
        return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)

    return runner


def test_dry_run_without_nmcli():
    # On the dev Mac nmcli doesn't exist: connect dry-runs, online is assumed.
    wifi = WifiManager()
    ok, message = wifi.connect("HomeNet", "pw")
    assert ok and "dry-run" in message
    assert wifi.is_online() is True
    healthy, detail = wifi.status()
    assert healthy and "dev machine" in detail


def test_connect_requires_ssid():
    ok, message = WifiManager().connect("", "pw")
    assert not ok and "required" in message


def test_parsing_with_fake_nmcli(monkeypatch):
    monkeypatch.setattr(network_module, "nmcli_available", lambda: True)

    monkeypatch.setattr(network_module, "_run", _fake_run("connected (site only)\n"))
    assert WifiManager().is_online() is True

    monkeypatch.setattr(network_module, "_run", _fake_run("disconnected\n"))
    assert WifiManager().is_online() is False

    monkeypatch.setattr(network_module, "_run", _fake_run("no:Other\nyes:HomeNet\n"))
    assert WifiManager().current_ssid() == "HomeNet"

    monkeypatch.setattr(
        network_module, "_run", _fake_run("IP4.ADDRESS[1]:192.168.1.50/24\n")
    )
    assert WifiManager().ip_address() == "192.168.1.50"

    monkeypatch.setattr(
        network_module, "_run",
        _fake_run("HomeNet:802-11-wireless\nWired 1:802-3-ethernet\n"),
    )
    assert WifiManager().saved_wifi_names() == ["HomeNet"]


def test_connect_failure_surfaces_stderr(monkeypatch):
    monkeypatch.setattr(network_module, "nmcli_available", lambda: True)
    monkeypatch.setattr(
        network_module, "_run", _fake_run(returncode=4, stderr="Error: no network with SSID")
    )
    ok, message = WifiManager().connect("Nope", "pw")
    assert not ok and "SSID" in message
