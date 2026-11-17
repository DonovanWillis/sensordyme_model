"""setup_pi.sh installs the direct-mode firewall only when config.yaml enables direct mode.

The block between its markers runs under bash with a fake `sudo` that records
what it was asked to run. Nothing is installed anywhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash") or "/bin/bash"


def _run_block(tmp_path: Path, config_text: str):
    text = (REPO / "setup_pi.sh").read_text()
    block = text[text.index("# >>> direct mode firewall"): text.index("# <<< direct mode firewall")]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "sudo.log"
    (bindir / "sudo").write_text(f'#!/bin/sh\necho "$*" >> {log}\n')
    (bindir / "sudo").chmod(0o755)
    work = tmp_path / "device"
    work.mkdir()
    (work / "config.yaml").write_text(config_text)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "PYTHON": sys.executable,
           "SERVICE_USER": "sensordyme"}
    proc = subprocess.run([BASH, "-c", "set -euo pipefail\nSERVICE_USER=sensordyme\n" + block],  # noqa: S603
                          cwd=work, env=env, capture_output=True, text=True, check=False)
    return proc, (log.read_text().splitlines() if log.exists() else [])


@pytest.mark.parametrize("config_text", [
    "network:\n  provisioning:\n    method: ble\n",
    "network:\n  direct_mode:\n    enabled: false\n",
    "network:\n  direct_mode:\n    enabled: yes-please\n",
    "",
])
def test_nothing_is_installed_unless_direct_mode_is_enabled(tmp_path, config_text):
    proc, sudo_calls = _run_block(tmp_path, config_text)
    assert proc.returncode == 0, proc.stderr
    assert sudo_calls == []
    assert proc.stdout == ""


def test_the_shipped_config_installs_nothing(tmp_path):
    proc, sudo_calls = _run_block(tmp_path, (REPO / "config.yaml").read_text())
    assert proc.returncode == 0, proc.stderr
    assert sudo_calls == []


def test_an_enabled_config_installs_the_firewall_and_names_the_passphrase_step(tmp_path):
    proc, sudo_calls = _run_block(tmp_path, "network:\n  direct_mode:\n    enabled: true\n")
    assert proc.returncode == 0, proc.stderr
    assert sudo_calls == [f"{sys.executable} scripts/direct_mode.py install"]
    assert "scripts/direct_mode_psk.py generate" in proc.stdout
    assert "as sensordyme" in proc.stdout
