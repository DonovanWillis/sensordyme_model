"""Whether the device is visible over Bluetooth, and whether it knows.

If advertising fails, a consumer at a new location has no way in at all: the
app cannot see the device, and there is no network to reach it on. So the one
thing this code must never do is be wrong about whether it worked.
"""

import subprocess
import types

from services.ble_gatt import BLEProvisioningServer


def make_server():
    return BLEProvisioningServer(
        device_name="SensorDyme-3D24",
        device_info={},
        on_credentials=lambda ssid, psk: None,
    )


def fake_run(monkeypatch, *, returncode=0, stdout="", stderr=""):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_a_silent_success_is_still_a_success(monkeypatch):
    """The regression.

    btmgmt prints "Instance added: 1" only when stdout is a terminal. Under
    systemd it is a pipe and prints nothing, so matching on that text reported
    every successful advertisement as a failure.
    """
    fake_run(monkeypatch, returncode=0, stdout="")
    server = make_server()
    server._advertise_via_btmgmt()
    assert server._btmgmt_active is True


def test_a_real_failure_is_still_a_failure(monkeypatch):
    fake_run(monkeypatch, returncode=1, stderr="Invalid Parameters")
    server = make_server()
    server._advertise_via_btmgmt()
    assert server._btmgmt_active is False


def test_teardown_removes_the_instance_even_if_we_think_we_never_added_one(monkeypatch):
    """A leaked instance is exactly the case where the flag was wrong, and a
    leftover instance blocks the next boot's advertisement."""
    calls = fake_run(monkeypatch, returncode=0)
    server = make_server()
    server._btmgmt_active = False
    server._stop_btmgmt_advertising()
    assert any("rm-adv" in c for c in calls[0]), "teardown skipped the removal"


def test_advertising_state_is_observed_not_remembered(monkeypatch):
    """Read from the instance list. On the Pi 5's CYW43455 the `advertising`
    flag in `btmgmt info` stays clear while instance-based advertising runs, so
    reading that flag reports a perfectly visible device as invisible."""
    fake_run(monkeypatch, returncode=0,
             stdout="Max instances: 5\nInstances list with 1 item\n")
    assert make_server().is_advertising() is True


def test_no_instances_means_not_advertising(monkeypatch):
    fake_run(monkeypatch, returncode=0,
             stdout="Max instances: 5\nInstances list with 0 items\n")
    server = make_server()
    server._btmgmt_active = True  # what we believe
    assert server.is_advertising() is False  # what is true


def test_a_broken_btmgmt_reports_not_advertising_rather_than_raising(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("btmgmt missing")

    monkeypatch.setattr(subprocess, "run", explode)
    assert make_server().is_advertising() is False
