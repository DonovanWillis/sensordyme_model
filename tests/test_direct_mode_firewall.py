"""The firewall on the direct-mode access point (G7): the ruleset text and the dispatcher.

No nft runs here. The ruleset is pinned by its exact text (nft itself is not on
the development machines), and the dispatcher script runs under /bin/sh with
fake `nft`, `nmcli` and `logger` programs that only record their arguments.
"""

from __future__ import annotations

import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from core import ap_firewall
from core.network import DIRECT_PROFILE

REPO = Path(__file__).resolve().parent.parent
DISPATCHER = REPO / "deploy" / "networkmanager" / "90-sensordyme-direct"

EXPECTED_WLAN0 = """#!/usr/sbin/nft -f
# SensorDyme direct mode: what a phone on the device's own access point may reach.
# Rendered by core/ap_firewall.py. Loaded by the NetworkManager dispatcher script
# while the "sensordyme-direct" profile is up; removed when it goes down.
# Allowed on wlan0: DHCP, DNS, mDNS and the dashboard (TCP 8899).
# Everything else is dropped, SSH included. Nothing is forwarded to or from it.

table inet sensordyme_direct
delete table inet sensordyme_direct

table inet sensordyme_direct {
\tchain input {
\t\ttype filter hook input priority -10; policy accept;
\t\tiifname != "wlan0" accept
\t\tct state established,related accept
\t\tct state invalid drop
\t\tudp sport 68 udp dport 67 accept comment "DHCP"
\t\tudp dport 53 accept comment "DNS"
\t\ttcp dport 53 accept comment "DNS"
\t\tudp dport 5353 accept comment "mDNS"
\t\ttcp dport 8899 accept comment "dashboard"
\t\tcounter drop comment "everything else, SSH included"
\t}

\tchain forward {
\t\ttype filter hook forward priority -10; policy accept;
\t\tiifname "wlan0" counter drop comment "no routing from the access point"
\t\toifname "wlan0" counter drop comment "no routing to the access point"
\t}
}
"""


def test_the_ruleset_is_exactly_the_reviewed_text():
    assert ap_firewall.render_ruleset() == EXPECTED_WLAN0


def test_only_dhcp_dns_mdns_and_the_dashboard_are_accepted_on_the_ap():
    rules = [line.strip() for line in ap_firewall.render_ruleset().splitlines()]
    accepted_ports = sorted(
        (r.split()[0], r.split()[r.split().index("dport") + 1]) for r in rules if "dport" in r and "accept" in r
    )
    assert accepted_ports == [("tcp", "53"), ("tcp", "8899"), ("udp", "53"), ("udp", "5353"), ("udp", "67")]
    assert not any("22" in r.split() for r in rules), "no SSH"
    assert "counter drop comment \"everything else, SSH included\"" in rules


def test_the_dashboard_port_and_interface_follow_the_config():
    text = ap_firewall.render_ruleset("wlan1", 9000)
    assert 'iifname != "wlan1" accept' in text
    assert "tcp dport 9000 accept" in text
    assert "8899" not in text and "wlan0" not in text


@pytest.mark.parametrize(("interface", "port"), [
    ('wlan0"; flush ruleset; "', 8899), ("", 8899), ("a" * 16, 8899), ("wlan0", 0), ("wlan0", 70000),
    ("wlan0", True), ("wlan0", "8899"),
])
def test_nothing_can_be_injected_into_the_ruleset(interface, port):
    with pytest.raises(ValueError, match=r"interface name|TCP port"):
        ap_firewall.render_ruleset(interface, port)


# ============================================================================ the dispatcher script
def test_the_dispatcher_names_match_the_python_side():
    text = DISPATCHER.read_text()
    assert f"PROFILE={DIRECT_PROFILE}\n" in text
    assert f"TABLE={ap_firewall.TABLE}\n" in text
    assert f"RULESET={ap_firewall.RULESET_PATH}\n" in text
    assert f'MARKER="$RUN_DIR/{os.path.basename(ap_firewall.MARKER_PATH)}"' in text
    assert f"RUN_DIR={os.path.dirname(ap_firewall.MARKER_PATH)}\n" in text
    assert os.access(DISPATCHER, os.X_OK)


def test_the_dispatcher_is_posix_sh():
    result = subprocess.run(["/bin/sh", "-n", str(DISPATCHER)], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
    assert DISPATCHER.read_text().startswith("#!/bin/sh\n")


@pytest.fixture()
def dispatcher(tmp_path):
    """The real script with its PATH and file locations pointed into tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    nft_status = tmp_path / "nft_status"
    nft_status.write_text("0")
    for name in ("nft", "nmcli", "logger"):
        tool = bin_dir / name
        extra = f'exit "$(cat {nft_status})"' if name == "nft" else "exit 0"
        tool.write_text(f'#!/bin/sh\necho "{name} $*" >> {log}\n{extra}\n')
        tool.chmod(0o755)
    run_dir = tmp_path / "run"
    ruleset = tmp_path / "direct_mode.nft"
    ruleset.write_text(ap_firewall.render_ruleset())
    text = DISPATCHER.read_text()
    for old, new in (("PATH=/usr/sbin:/usr/bin:/sbin:/bin", f"PATH={bin_dir}:/usr/bin:/bin"),
                     (f"RULESET={ap_firewall.RULESET_PATH}", f"RULESET={ruleset}"),
                     (f"RUN_DIR={os.path.dirname(ap_firewall.MARKER_PATH)}", f"RUN_DIR={run_dir}")):
        assert old in text
        text = text.replace(old, new)
    script = tmp_path / "90-sensordyme-direct"
    script.write_text(text)
    script.chmod(0o755)

    def run(action: str, connection: str = DIRECT_PROFILE, nft_ok: bool = True):
        nft_status.write_text("0" if nft_ok else "1")
        env = {"CONNECTION_ID": connection, "PATH": "/usr/bin:/bin"}
        result = subprocess.run(["/bin/sh", str(script), "wlan0", action],  # noqa: S603
                                capture_output=True, text=True, env=env, timeout=10, check=False)
        return result

    def calls() -> list[str]:
        return log.read_text().splitlines() if log.exists() else []

    import types
    return types.SimpleNamespace(run=run, calls=calls, marker=run_dir / "firewall-active", ruleset=ruleset)


def test_other_connections_are_ignored(dispatcher):
    for action in ("pre-up", "up", "down"):
        assert dispatcher.run(action, connection="Home").returncode == 0
    assert dispatcher.calls() == []
    assert not dispatcher.marker.exists()


@pytest.mark.parametrize("action", ["pre-up", "up"])
def test_bringing_the_ap_up_loads_the_ruleset_then_writes_the_marker(dispatcher, action):
    result = dispatcher.run(action)
    assert result.returncode == 0, result.stderr
    assert dispatcher.calls()[0] == f"nft -f {dispatcher.ruleset}"
    assert dispatcher.marker.exists()
    assert stat.S_IMODE(dispatcher.marker.stat().st_mode) == 0o644
    assert not any(c.startswith("nmcli") for c in dispatcher.calls())


def test_a_ruleset_that_does_not_load_takes_the_ap_down(dispatcher):
    dispatcher.run("pre-up")
    assert dispatcher.marker.exists()
    result = dispatcher.run("up", nft_ok=False)
    assert result.returncode == 1
    assert not dispatcher.marker.exists(), "the manager must not see a firewall that is not there"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and f"nmcli connection down id {DIRECT_PROFILE}" not in dispatcher.calls():
        time.sleep(0.05)
    assert f"nmcli connection down id {DIRECT_PROFILE}" in dispatcher.calls()


@pytest.mark.parametrize("action", ["pre-down", "down"])
def test_taking_the_ap_down_removes_the_marker_and_the_table(dispatcher, action):
    dispatcher.run("up")
    result = dispatcher.run(action)
    assert result.returncode == 0
    assert not dispatcher.marker.exists()
    assert f"nft delete table inet {ap_firewall.TABLE}" in dispatcher.calls()
