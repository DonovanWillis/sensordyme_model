"""4D: the prepared least-privilege unit and its grants (static; nothing is installed).

The unit is hardware-pending: these tests prove what can be proven off-device --
that it keeps the stock unit's recovery semantics, grants nothing beyond the
access matrix in docs/LEAST_PRIVILEGE.md, and that the code paths it switches
(core/privilege.py) behave. `systemd-analyze verify` runs where systemd exists.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from core import privilege

REPO = Path(__file__).resolve().parent.parent
HARDENED = "deploy/systemd/sensordyme.service.hardened"
DROP_IN = "deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf"


def _unit(path: str) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for raw in (REPO / path).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "[")) and "=" in line:
            key, value = line.split("=", 1)
            keys.setdefault(key, []).append(value)
    return keys


# ----------------------------------------------------------------- same recovery semantics
@pytest.mark.parametrize("key", [
    "StartLimitIntervalSec", "StartLimitBurst", "StartLimitAction", "OnFailure", "Type", "NotifyAccess",
    "WatchdogSec", "Restart", "RestartSec", "RestartPreventExitStatus", "TimeoutStartSec", "After", "Wants",
])
def test_recovery_semantics_match_the_stock_unit(key):
    assert _unit(HARDENED).get(key) == _unit("sensordyme.service").get(key), key


def test_preflight_stays_advisory_and_runs_the_read_only_copy():
    unit = _unit(HARDENED)
    assert unit["ExecStartPre"] == ["-/opt/sensordyme/.venv/bin/python /opt/sensordyme/scripts/preflight.py"]
    assert unit["ExecStart"] == ["/opt/sensordyme/.venv/bin/python /opt/sensordyme/main.py --no-display"]
    assert unit["WorkingDirectory"] == ["/opt/sensordyme"]
    assert "__SENSORDYME" not in (REPO / HARDENED).read_text(), "no template tokens: fixed layout"


# ----------------------------------------------------------------- least privilege
def test_a_dedicated_unprivileged_user_with_only_the_groups_the_matrix_needs():
    unit = _unit(HARDENED)
    assert unit["User"] == ["sensordyme"] and unit["Group"] == ["sensordyme"]
    assert set(unit["SupplementaryGroups"][0].split()) == {"video", "gpio", "audio", "netdev", "bluetooth", "hailo"}
    for absent in ("sudo", "adm", "root", "disk", "dialout", "i2c", "spi", "plugdev"):
        assert absent not in unit["SupplementaryGroups"][0].split()


def test_no_privilege_can_be_gained():
    unit = _unit(HARDENED)
    for key, value in {"NoNewPrivileges": "yes", "CapabilityBoundingSet": "", "AmbientCapabilities": "",
                       "RestrictSUIDSGID": "yes", "ProtectSystem": "strict", "ProtectHome": "yes",
                       "PrivateTmp": "yes", "DevicePolicy": "closed", "ProtectKernelModules": "yes",
                       "ProtectKernelTunables": "yes", "RestrictNamespaces": "yes", "LockPersonality": "yes",
                       "SystemCallArchitectures": "native"}.items():
        assert unit.get(key) == [value], key
    assert "PrivateUsers" not in unit, "a user namespace would unmap the device groups"
    assert unit["Environment"].count("SENSORDYME_PRIVILEGE=direct") == 1


def test_the_only_writable_places_are_state_cache_and_logs():
    unit = _unit(HARDENED)
    assert unit["StateDirectory"] == ["sensordyme"] and unit["CacheDirectory"] == ["sensordyme"]
    assert unit["LogsDirectory"] == ["sensordyme"]
    assert unit["BindPaths"] == ["/var/lib/sensordyme:/opt/sensordyme/data"]
    assert "ReadWritePaths" not in unit
    env = dict(e.split("=", 1) for e in unit["Environment"])
    writable = ("/var/lib/sensordyme", "/var/cache/sensordyme", "/var/log/sensordyme")
    for name in ("HOME", "MPLCONFIGDIR", "XDG_CACHE_HOME", "HAILORT_LOGGER_PATH"):
        assert env[name].startswith(writable), name


def test_every_default_write_path_in_device_code_lands_in_data():
    """The bind mount covers data/ only, so a default write path elsewhere would fail read-only."""
    pattern = re.compile(r"""["'](data/[^"']*)["']""")
    found = set()
    for root in ("core", "services", "modules"):
        for file in (REPO / root).rglob("*.py"):
            found.update(pattern.findall(file.read_text()))
    assert found, "the scan must see the known data/ paths"
    assert {"data/events.csv", "data/device_state.json", "data/.reboot_state.json"} <= found
    preflight = (REPO / "scripts/preflight.py").read_text()
    assert 'Path("data/preflight.json")' in preflight


def test_socket_families_and_devices_are_the_matrix_and_nothing_else():
    unit = _unit(HARDENED)
    assert unit["RestrictAddressFamilies"] == ["AF_UNIX AF_INET AF_INET6 AF_NETLINK"]
    allowed = {v.split()[0] for v in unit["DeviceAllow"]}
    assert allowed == {"char-video4linux", "char-media", "char-dma_heap", "/dev/vcio", "/dev/hailo0",
                       "char-gpiochip", "char-rpi-gpiomem", "char-alsa"}
    assert all(v.endswith(" rw") for v in unit["DeviceAllow"])
    for broad in ("block-", "/dev/mem", "/dev/kmem", "char-tty", "/dev/i2c", "/dev/spidev"):
        assert not any(broad in v for v in unit["DeviceAllow"]), broad


def test_the_btmgmt_drop_in_adds_only_what_btmgmt_needs():
    drop = _unit(DROP_IN)
    assert drop == {"CapabilityBoundingSet": ["CAP_NET_ADMIN"], "AmbientCapabilities": ["CAP_NET_ADMIN"],
                    "RestrictAddressFamilies": ["AF_BLUETOOTH"]}
    assert "[Install]" not in (REPO / DROP_IN).read_text()


def test_the_reboot_grant_is_reboot_only_for_the_service_user():
    rule = (REPO / "deploy/polkit/20-sensordyme-reboot.rules").read_text()
    actions = set(re.findall(r'"(org\.freedesktop\.[^"]+)"', rule))
    assert actions == {"org.freedesktop.login1.reboot", "org.freedesktop.login1.reboot-multiple-sessions",
                       "org.freedesktop.login1.reboot-ignore-inhibit"}
    assert 'subject.user === "sensordyme"' in rule


def test_the_hailo_udev_rule_narrows_to_the_group():
    rule = [ln for ln in (REPO / "deploy/udev/60-sensordyme-hailo.rules").read_text().splitlines()
            if ln and not ln.startswith("#")]
    assert rule == ['KERNEL=="hailo[0-9]*", GROUP="hailo", MODE="0660"']


def test_the_stock_unit_and_sudoers_are_untouched_by_the_hardened_variant():
    stock = (REPO / "sensordyme.service").read_text()
    assert "User=__SENSORDYME_USER__" in stock and "NoNewPrivileges" not in stock
    assert "systemctl reboot" in (REPO / "deploy/sudoers/sensordyme-reboot").read_text()


def test_the_plan_script_prints_and_executes_nothing(tmp_path):
    marker = tmp_path / "ran"
    fake = tmp_path / "bin"
    fake.mkdir()
    for tool in ("sudo", "systemctl", "useradd", "rsync", "install", "udevadm"):
        (fake / tool).write_text(f"#!/bin/sh\necho {tool} >> {marker}\n")
        (fake / tool).chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    bash = shutil.which("bash") or "/bin/bash"
    out = subprocess.run([bash, str(REPO / "scripts/release/least_privilege_plan.sh"), "--from-dir", "/srv/x"],  # noqa: S603
                         capture_output=True, text=True, env=env, check=True).stdout
    assert not marker.exists(), "the plan must not run anything"
    assert "ROLLBACK" in out and "/srv/x/data/" in out and "systemd-analyze verify" in out


@pytest.mark.skipif(not shutil.which("systemd-analyze"), reason="systemd-analyze absent (macOS)")
def test_systemd_analyze_verifies_the_unit(tmp_path):
    """Rendered into a scratch tree whose ExecStart exists, so verify judges the unit, not the host."""
    app = tmp_path / "opt"
    (app / ".venv/bin").mkdir(parents=True)
    (app / ".venv/bin/python").symlink_to(shutil.which("python3") or "/usr/bin/python3")
    (app / "scripts").mkdir()
    (app / "scripts/preflight.py").write_text("")
    (app / "main.py").write_text("")
    text = (REPO / HARDENED).read_text().replace("/opt/sensordyme", str(app))
    units = tmp_path / "units"
    units.mkdir()
    (units / "sensordyme.service").write_text(text)
    shutil.copy(REPO / "deploy/systemd/sensordyme-recover.service", units / "sensordyme-recover.service")
    analyze = shutil.which("systemd-analyze") or "systemd-analyze"
    result = subprocess.run([analyze, "verify", "--man=no", str(units / "sensordyme.service")],  # noqa: S603
                            capture_output=True, text=True, check=False,
                            env={**os.environ, "SYSTEMD_UNIT_PATH": f"{units}:"})
    problems = [ln for ln in result.stderr.splitlines()
                if "sensordyme.service" in ln and "recover" not in ln and "/usr/local/lib/sensordyme" not in ln]
    assert result.returncode == 0 or not problems, result.stderr


# ----------------------------------------------------------------- core.privilege
@pytest.mark.parametrize(("value", "expected"), [
    (None, ["sudo", "-n", "btmgmt", "advinfo"]), ("sudo", ["sudo", "-n", "btmgmt", "advinfo"]),
    ("direct", ["btmgmt", "advinfo"]), (" DIRECT ", ["btmgmt", "advinfo"]),
    ("root", ["sudo", "-n", "btmgmt", "advinfo"]),  # unknown: the stock path, which fails closed under NNP
])
def test_privileged_argv(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(privilege.ENV, raising=False)
    else:
        monkeypatch.setenv(privilege.ENV, value)
    assert privilege.privileged(["btmgmt", "advinfo"]) == expected


def test_direct_mode_reaches_both_call_sites(monkeypatch):
    from services.ble_gatt import BLEProvisioningServer
    from services.runtime import DeviceRuntime

    calls = []
    fake = lambda cmd, **_k: calls.append(cmd) or types.SimpleNamespace(returncode=0, stdout="", stderr="")  # noqa: E731
    monkeypatch.setenv(privilege.ENV, "direct")
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr("services.runtime.subprocess.run", fake)
    server = BLEProvisioningServer("SensorDyme-TEST", {"id": "SD-00000000"}, on_credentials=lambda s, p: None)
    server._advertise_via_btmgmt()
    server._stop_btmgmt_advertising()
    server.is_advertising()
    assert DeviceRuntime.__new__(DeviceRuntime)._issue_reboot() is True
    assert calls[-1] == ["systemctl", "reboot"]
    assert all(cmd[0] == "btmgmt" for cmd in calls[:-1]) and len(calls) == 6
    assert server._btmgmt_active is False  # stop ran last
