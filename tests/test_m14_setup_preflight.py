"""M14: setup_pi.sh stops BEFORE changing anything on a device it cannot install on.

The device lock is cp311/aarch64 wheels for Raspberry Pi OS Bookworm. A Pi that
arrives with Raspberry Pi OS Trixie (Debian 13, Python 3.13), a 32-bit
userland, or a package install with no config.yaml must be refused by the
read-only platform preflight: no sudo, no apt, no venv. These tests run the
REAL setup_pi.sh with fake `sudo`, `apt-get`, `python3` and `uname` on PATH
that only record what they were asked to do.

Also: the transfer package's wheels, copied to <install>.wheels as INSTALL.md
says, are what the hashed install uses (offline, still hash-checked).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_m85_deploy_lock import _run_setup_block

REPO = Path(__file__).resolve().parent.parent

BOOKWORM = 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nVERSION_CODENAME=bookworm\nID=debian\n'
TRIXIE = 'PRETTY_NAME="Debian GNU/Linux 13 (trixie)"\nVERSION_CODENAME=trixie\nID=debian\n'

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _fakes(bin_dir: Path, log: Path) -> None:
    bin_dir.mkdir(parents=True)
    scripts = {
        # Records every privileged call; never runs anything.
        "sudo": '#!/bin/sh\necho "sudo $*" >> "$CALL_LOG"\nexit 0\n',
        "apt-get": '#!/bin/sh\necho "apt-get $*" >> "$CALL_LOG"\nexit 0\n',
        "uname": '#!/bin/sh\nif [ "$1" = "-m" ]; then echo "$FAKE_MACHINE"; else echo Linux; fi\n',
        # `python3 -c <probe>` answers "major.minor bits"; anything else is recorded.
        "python3": ('#!/bin/sh\nif [ "$1" = "-c" ]; then echo "$FAKE_PY"; exit 0; fi\n'
                    'echo "python3 $*" >> "$CALL_LOG"\nexit 0\n'),
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)


def _run_setup(tmp_path: Path, *, os_release: str, machine: str = "aarch64", py: str = "3.11 64",
               config: bool = True, pi: bool = True, allow: bool = False):
    if os.geteuid() == 0:
        pytest.skip("setup_pi.sh refuses to run as root")
    work = tmp_path / "sensordyme"
    work.mkdir(parents=True)
    shutil.copy(REPO / "setup_pi.sh", work / "setup_pi.sh")
    if config:
        (work / "config.yaml").write_text("module: driver_attention\n")
    (tmp_path / "os-release").write_text(os_release)
    (tmp_path / "model").write_text("Raspberry Pi 5 Model B Rev 1.0\x00" if pi else "QEMU Virtual Machine\x00")
    log = tmp_path / "calls.log"
    _fakes(tmp_path / "bin", log)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "CALL_LOG": str(log),
           "FAKE_MACHINE": machine, "FAKE_PY": py, "SENSORDYME_OS_RELEASE": str(tmp_path / "os-release"),
           "SENSORDYME_DT_MODEL": str(tmp_path / "model"), "SENSORDYME_USER": subprocess.check_output(
               ["id", "-un"], text=True).strip()}  # noqa: S607
    env.pop("SENSORDYME_ALLOW_UNHASHED", None)
    env.pop("SENSORDYME_CONFIG", None)
    if allow:
        env["SENSORDYME_ALLOW_UNHASHED"] = "1"
    proc = subprocess.run(["bash", str(work / "setup_pi.sh")], cwd=tmp_path, env=env,  # noqa: S603, S607
                          capture_output=True, text=True, check=False, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls, work


def _assert_nothing_changed(proc, calls, work):
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Nothing was installed or changed" in proc.stderr
    assert "HG-OS" in proc.stderr, "the refusal names the hardware gate in the operator guide"
    assert calls == [], f"the refusal must come before any sudo/apt/venv call: {calls}"
    assert not (work / ".venv").exists()
    assert "Installing apt packages" not in proc.stdout


def test_trixie_is_refused_before_anything_changes(tmp_path):
    proc, calls, work = _run_setup(tmp_path, os_release=TRIXIE, py="3.13 64")
    _assert_nothing_changed(proc, calls, work)
    assert "Debian GNU/Linux 13 (trixie)" in proc.stderr and "Bookworm (Debian 12) only" in proc.stderr
    assert "python3 is '3.13 64'" in proc.stderr
    assert "Do NOT reflash" in proc.stderr


def test_a_32_bit_userland_on_the_64_bit_kernel_is_refused(tmp_path):
    proc, calls, work = _run_setup(tmp_path, os_release=BOOKWORM, machine="aarch64", py="3.11 32")
    _assert_nothing_changed(proc, calls, work)
    assert "python3 is '3.11 32'" in proc.stderr and "64-bit" in proc.stderr


def test_a_32_bit_kernel_is_refused(tmp_path):
    proc, calls, work = _run_setup(tmp_path, os_release=BOOKWORM, machine="armv7l")
    _assert_nothing_changed(proc, calls, work)
    assert "CPU architecture is 'armv7l'" in proc.stderr


def test_a_package_install_without_config_yaml_is_refused(tmp_path):
    """The bundle never carries config.yaml; without one the service exits 78 at every start."""
    proc, calls, work = _run_setup(tmp_path, os_release=BOOKWORM, config=False)
    _assert_nothing_changed(proc, calls, work)
    assert "config/config.yaml.example" in proc.stderr


def test_a_non_pi_dev_box_is_refused_unless_explicitly_allowed(tmp_path):
    proc, calls, work = _run_setup(tmp_path / "a", os_release=TRIXIE, py="3.13 64", pi=False)
    _assert_nothing_changed(proc, calls, work)
    # Allowed: the OS checks are skipped with a warning and the script proceeds
    # (into the fakes; the dependency block then fails on the fake venv, which is fine).
    proc, calls, _ = _run_setup(tmp_path / "b", os_release=TRIXIE, py="3.13 64", pi=False, allow=True)
    assert "OS/Python checks skipped (development box only)" in proc.stderr
    assert any(c.startswith("sudo apt-get") for c in calls)


def test_the_allow_flag_never_skips_the_checks_on_a_pi(tmp_path):
    proc, calls, work = _run_setup(tmp_path, os_release=TRIXIE, py="3.13 64", allow=True)
    _assert_nothing_changed(proc, calls, work)


def test_bookworm_aarch64_python_311_passes_the_preflight(tmp_path):
    proc, calls, _ = _run_setup(tmp_path, os_release=BOOKWORM)
    assert "==> Platform: Debian GNU/Linux 12 (bookworm), aarch64, python3 3.11" in proc.stdout
    assert "Nothing was installed or changed" not in proc.stderr
    assert calls and calls[0] == "sudo apt-get update", "the first change comes after the preflight"


# ---------------------------------------------------------------------- offline wheels
def test_the_package_wheels_beside_the_install_are_used_hash_checked(tmp_path):
    wheels = tmp_path / "pi.wheels"  # _run_setup_block installs into tmp_path/pi
    wheels.mkdir()
    (wheels / "pyyaml-6.0.2-cp311-cp311-manylinux_2_17_aarch64.whl").write_bytes(b"")
    proc, calls = _run_setup_block(tmp_path, hashed_ok=True, pi=True)
    assert proc.returncode == 0, proc.stderr
    assert calls == [("-m pip install --require-hashes --only-binary=:all: --disable-pip-version-check "
                      f"--no-input -r requirements-device.lock --no-index --find-links {wheels}")]


def test_an_empty_wheel_directory_falls_back_to_the_index_still_hashed(tmp_path):
    (tmp_path / "pi.wheels").mkdir()
    proc, calls = _run_setup_block(tmp_path, hashed_ok=True, pi=True)
    assert proc.returncode == 0
    assert calls == [("-m pip install --require-hashes --only-binary=:all: --disable-pip-version-check "
                      "--no-input -r requirements-device.lock")]


# ---------------------------------------------------------------------- the generated INSTALL.md
def test_install_md_fresh_path_carries_config_refuses_an_existing_install_and_uses_the_wheels():
    from scripts.release import build_transfer_package as btp

    made = {"release": "abcdef123456", "commit": "abcdef123456" + "0" * 28,
            "manifest": "/x/sensordyme-abcdef123456.manifest.json",
            "tarball": "/x/sensordyme-abcdef123456.tar.gz", "manifest_sha256": "f" * 64}
    text = btp.install_md(made, {"model_sha256": "e" * 64}, {"count": 40, "bytes": 3e8},
                          {"path": "damoyolo_tinynasL20_T.hef", "sha256": "e" * 64, "bytes": 1})
    fresh = text.split("## A. Fresh device", 1)[1].split("## B.", 1)[0]
    block = fresh.split("```bash", 1)[1].split("```", 1)[0].strip()
    # One chained command: it stops at once when ~/sensordyme already exists.
    assert block.startswith("mkdir ~/sensordyme \\") and "mkdir -p ~/sensordyme " not in block
    assert all(line.rstrip().endswith("\\") for line in block.splitlines()[:-1])
    assert "cp -n config/config.yaml.example ~/sensordyme/config.yaml && chmod 600" in block
    # The wheel directory is the one setup_pi.sh looks for: <install dir>.wheels.
    assert "~/sensordyme.wheels/" in block and "$(pwd).wheels" in (REPO / "setup_pi.sh").read_text()
    assert block.endswith("cd ~/sensordyme && bash setup_pi.sh")
    assert "Trixie" in text and "4.20.0" in text and "before changing anything" in text
    upgrade = text.split("## B.", 1)[1]
    assert "RB-E1" in upgrade and "adopt" in upgrade and "local_identity.pem" in upgrade
