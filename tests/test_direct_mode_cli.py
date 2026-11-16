"""The direct-mode command-line tools: the factory passphrase tool and the operator CLI.

No real nmcli or nft: the operator CLI runs in-process against the fake
NetworkManager, and `install` writes under a scratch root.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from direct_mode_fakes import FakeNM

import core.network as network
from core import ap_firewall
from core import direct_mode as dm

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli(*args, cwd=None):
    return subprocess.run([PY, *args], capture_output=True, text=True, timeout=60, check=False, cwd=cwd)  # noqa: S603


# ============================================================================ scripts/direct_mode_psk.py
def test_generate_writes_once_0600_and_prints_the_label(tmp_path):
    out = tmp_path / "direct_mode_psk"
    result = _cli(str(REPO / "scripts" / "direct_mode_psk.py"), "generate", "--out", str(out),
                  "--device-id", "SD-1A2B3C4D")
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    psk = dm.load_psk(str(out))
    assert result.stdout.splitlines() == [
        "SensorDyme SD-1A2B3C4D",
        "Direct Wi-Fi: SensorDyme-1A2B",
        f"Wi-Fi password: {psk}",
        f"Wi-Fi QR: WIFI:T:WPA;S:SensorDyme-1A2B;P:{psk};;",
    ]

    again = _cli(str(REPO / "scripts" / "direct_mode_psk.py"), "generate", "--out", str(out),
                 "--device-id", "SD-1A2B3C4D")
    assert again.returncode == 1 and "already exists" in again.stderr
    assert dm.load_psk(str(out)) == psk, "the label's passphrase was not replaced"

    label = _cli(str(REPO / "scripts" / "direct_mode_psk.py"), "label", "--psk-file", str(out),
                 "--device-id", "SD-1A2B3C4D")
    assert label.returncode == 0 and label.stdout == result.stdout


def test_label_refuses_a_loose_file_without_printing_it(tmp_path):
    out = tmp_path / "psk"
    psk = dm.generate_psk()
    dm.write_psk(str(out), psk)
    out.chmod(0o644)
    result = _cli(str(REPO / "scripts" / "direct_mode_psk.py"), "label", "--psk-file", str(out),
                  "--device-id", "SD-1A2B3C4D")
    assert result.returncode == 1
    assert "0600" in result.stderr
    assert psk not in result.stdout + result.stderr


# ============================================================================ scripts/direct_mode.py
def test_ruleset_prints_the_rendered_ruleset():
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "ruleset")
    assert result.returncode == 0
    assert result.stdout == ap_firewall.render_ruleset("wlan0", 8899)


def test_status_with_the_shipped_config_is_unavailable_and_says_why():
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "status")
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert set(status) == {"mode", "ssid", "since", "last_error", "fallback_enabled"}
    assert status["mode"] == "unavailable"
    assert "disabled" in status["last_error"]


def test_enter_with_the_shipped_config_refuses(tmp_path):
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "enter")
    assert result.returncode == 1
    assert "refused: direct mode is disabled" in result.stderr


@pytest.fixture()
def enabled_config(tmp_path):
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    psk_path = tmp_path / "psk"
    psk = dm.generate_psk()
    dm.write_psk(str(psk_path), psk)
    config["network"]["direct_mode"].update(enabled=True, psk_path=str(psk_path),
                                            state_path=str(tmp_path / "state.json"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, psk


def test_dry_run_prints_the_plan_and_changes_nothing(enabled_config, monkeypatch, capsys):
    path, psk = enabled_config
    nm = FakeNM()
    nm.add_client("Home", active=True)
    monkeypatch.setattr(network, "nmcli_available", lambda: True)
    monkeypatch.setattr(network, "_run", nm.run)
    monkeypatch.chdir(REPO)  # main() chdirs there; monkeypatch restores the cwd
    cli = _load("direct_mode")

    assert cli.main(["--config", str(path), "enter", "--dry-run"]) == 0
    enter_plan = capsys.readouterr().out
    assert cli.main(["--config", str(path), "leave", "--dry-run"]) == 0
    leave_plan = capsys.readouterr().out

    assert nm.mutating() == [], "a dry run changes nothing"
    assert all(c[:2] == ["-t", "-f"] for c in nm.calls), "only read-only queries"
    assert psk not in enter_plan + leave_plan
    assert "wifi-sec.psk '<redacted>'" in enter_plan
    assert "nmcli connection up id sensordyme-direct" in enter_plan
    assert enter_plan.rstrip().endswith("nmcli connection up id Home"), "the rollback names the real network"
    assert "nmcli connection down id sensordyme-direct" in leave_plan
    assert "nft delete table inet sensordyme_direct" in leave_plan


def test_install_dry_run_lists_every_step():
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "install", "--dry-run")
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert lines[0].endswith(ap_firewall.RULESET_PATH)
    assert lines[1].endswith(ap_firewall.DISPATCHER_PATH)
    assert lines[2].endswith(ap_firewall.PRE_UP_LINK)
    assert lines[3].startswith(f"nft -c -f {ap_firewall.RULESET_PATH}")


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="checks the non-root refusal")
def test_install_needs_root():
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "install")
    assert result.returncode == 1 and "root" in result.stderr


def test_install_writes_the_ruleset_the_dispatcher_and_its_pre_up_link(tmp_path):
    result = _cli(str(REPO / "scripts" / "direct_mode.py"), "install", "--root", str(tmp_path))
    assert result.returncode == 0, result.stderr
    ruleset = tmp_path / ap_firewall.RULESET_PATH.lstrip("/")
    dispatcher = tmp_path / ap_firewall.DISPATCHER_PATH.lstrip("/")
    link = tmp_path / ap_firewall.PRE_UP_LINK.lstrip("/")
    assert ruleset.read_text() == ap_firewall.render_ruleset()
    assert stat.S_IMODE(ruleset.stat().st_mode) == 0o644
    assert dispatcher.read_bytes() == (REPO / "deploy" / "networkmanager" / "90-sensordyme-direct").read_bytes()
    assert stat.S_IMODE(dispatcher.stat().st_mode) == 0o755
    assert link.is_symlink() and os.readlink(link) == "../90-sensordyme-direct"
    assert link.resolve() == dispatcher.resolve()
    # Idempotent: a second install replaces, never duplicates or fails.
    assert _cli(str(REPO / "scripts" / "direct_mode.py"), "install", "--root", str(tmp_path)).returncode == 0
