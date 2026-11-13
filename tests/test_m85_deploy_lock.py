"""M8.5 (M84-4): every device install path installs the hashed lock, wheels only.

setup_pi.sh (its install block, run with a fake venv python that records argv),
the staged install and the legacy rollback (release_guard.py), and the release
gate that refuses a ref whose install paths are not hashed. Plus the lock itself:
complete for the device import set (markers evaluated for the Pi), every entry
hashed, gpiozero/lgpio folded in, nothing that would have to be built.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.release import check_device_deps as deps
from scripts.release import make_release_manifest as rm
from scripts.release import release_guard as rg

REPO = Path(__file__).resolve().parent.parent
BASE = "87d91b3"  # M8.4 candidate code: the unhashed installs this mission removes
HASHED = ["--require-hashes", "--only-binary=:all:"]


# ---------------------------------------------------------------------- setup_pi.sh
def _install_block() -> str:
    text = (REPO / "setup_pi.sh").read_text()
    block = text.split("# >>> device dependency install", 1)[1].split("# <<< device dependency install", 1)[0]
    return block.split("\n", 1)[1]  # drop the rest of the marker line


def _run_setup_block(tmp_path, *, hashed_ok: bool, pi: bool, allow: bool = False):
    work = tmp_path / "pi"
    (work / ".venv" / "bin").mkdir(parents=True)
    log = work / "pip.argv"
    fake = work / ".venv" / "bin" / "python"
    fake.write_text('#!/bin/sh\necho "$*" >> "$PIP_LOG"\n'
                    'case "$*" in *--require-hashes*) exit "$HASHED_EXIT";; esac\nexit 0\n')
    fake.chmod(0o755)
    model = work / "model"
    model.write_text("Raspberry Pi 5 Model B Rev 1.0\x00" if pi else "QEMU Virtual Machine\x00")
    script = work / "block.sh"
    script.write_text("set -euo pipefail\n" + _install_block() + "\necho BLOCK_DONE\n")
    env = {**os.environ, "PIP_LOG": str(log), "HASHED_EXIT": "0" if hashed_ok else "1",
           "SENSORDYME_DT_MODEL": str(model)}
    env.pop("SENSORDYME_ALLOW_UNHASHED", None)
    if allow:
        env["SENSORDYME_ALLOW_UNHASHED"] = "1"
    proc = subprocess.run(["bash", str(script)], cwd=work, env=env, capture_output=True, text=True,  # noqa: S603, S607
                          check=False)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def test_setup_installs_the_hashed_lock_wheels_only(tmp_path):
    proc, calls = _run_setup_block(tmp_path, hashed_ok=True, pi=True)
    assert proc.returncode == 0 and "BLOCK_DONE" in proc.stdout
    assert calls == [("-m pip install --require-hashes --only-binary=:all: --disable-pip-version-check "
                      "--no-input -r requirements-device.lock")]


@pytest.mark.parametrize("allow", [False, True])
def test_on_a_pi_a_failed_hashed_install_never_falls_back(tmp_path, allow):
    proc, calls = _run_setup_block(tmp_path, hashed_ok=False, pi=True, allow=allow)
    assert proc.returncode == 1 and "NOT falling back" in proc.stderr
    assert len(calls) == 1 and "--require-hashes" in calls[0]


def test_a_dev_box_falls_back_only_when_told_to(tmp_path):
    proc, calls = _run_setup_block(tmp_path, hashed_ok=False, pi=False)
    assert proc.returncode == 1 and len(calls) == 1
    proc, calls = _run_setup_block(tmp_path / "again", hashed_ok=False, pi=False, allow=True)
    assert proc.returncode == 0 and "WITHOUT hash checking" in proc.stderr
    assert calls[-1] == "-m pip install -r requirements-device.txt"


def test_setup_has_no_other_unhashed_install():
    text = (REPO / "setup_pi.sh").read_text()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    pip_lines = [ln for ln in code if re.search(r"\bpip\b.*\binstall\b|DEVICE_PIP", ln)]
    assert not any("--upgrade pip" in ln or "gpiozero==" in ln for ln in code)
    unhashed = [ln for ln in pip_lines if "requirements-device.txt" in ln]
    assert len(unhashed) == 1 and rm.UNHASHED_FALLBACK_MARK in unhashed[0]


def test_the_setup_flags_are_the_staged_install_flags():
    """One definition of THE device install: setup_pi.sh and release_guard.py agree."""
    text = (REPO / "setup_pi.sh").read_text()
    flags = re.search(r"DEVICE_PIP=\(\.venv/bin/python -m pip install (.*?) -r requirements-device\.lock\)", text)
    assert flags and tuple(flags.group(1).split()) == rg.PIP_FLAGS


# ---------------------------------------------------------------------- the guard's installs
def test_the_staged_install_argv_and_the_offline_wheelhouse_option():
    argv = rg.pip_install_argv(Path("/r/.venv/bin/python"), Path("/r/requirements-device.lock"))
    assert argv[:4] == ["/r/.venv/bin/python", "-m", "pip", "install"] and all(f in argv for f in HASHED)
    assert argv[-2:] == ["-r", "/r/requirements-device.lock"] and "--no-index" not in argv
    offline = rg.pip_install_argv(Path("p"), Path("l"), "/home/u/wheels")
    assert offline[-3:] == ["--no-index", "--find-links", "/home/u/wheels"]


def test_the_legacy_rollback_installs_the_previous_lock_or_nothing(tmp_path):
    from test_m8_lifecycle_release import (
        ACTIVE,
        CRASH_LOOP,
        STARTING,
        _device,
        _FakeSystemd,
        _guard,
    )

    root = _device(tmp_path)
    (root / "requirements-device.lock").write_text("pyyaml==6.0.2 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    systemd = _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]])
    guard = _guard(root, systemd)
    assert guard.snapshot() == 0
    assert guard.rollback(reason="test") == 0
    pips = [c for c in systemd.commands if "pip" in " ".join(c)]
    assert len(pips) == 1 and all(f in pips[0] for f in HASHED) and pips[0][-1].endswith("requirements-device.lock")
    assert not any("requirements-device.txt" in " ".join(c) for c in systemd.commands)

    root2 = _device(tmp_path / "nolock")
    systemd2 = _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]])
    guard2 = _guard(root2, systemd2)
    assert guard2.snapshot() == 0 and guard2.rollback(reason="test") == 0
    assert not any("pip" in " ".join(c) for c in systemd2.commands), "no lock: nothing unhashed instead"


# ---------------------------------------------------------------------- the release gate
def _files(ref: str | None) -> dict[str, bytes]:
    out = {}
    for path in rm.INSTALL_PATH_FILES:
        if ref is None:
            out[path] = (REPO / path).read_bytes() if (REPO / path).exists() else b""
        else:
            shown = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"],  # noqa: S603, S607
                                   capture_output=True, check=False)
            if shown.returncode == 0:
                out[path] = shown.stdout
    return out


def test_every_install_path_in_this_tree_is_hashed():
    paths = rm.install_paths(_files(None))
    assert paths["ok"], paths


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_the_gate_fails_the_unhashed_m84_install_paths():
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", BASE], check=False).returncode:  # noqa: S603, S607
        pytest.skip(f"{BASE} not in this clone")
    paths = rm.install_paths(_files(BASE))
    assert not paths["ok"]
    joined = " ".join(paths["violations"])
    assert "setup_pi.sh" in joined and "Makefile" in joined and "--upgrade pip" in joined
    assert "scripts/release/release_guard.py" in paths["missing_hashed_install"]


@pytest.mark.parametrize(("line", "ok"), [
    ("pip install -r requirements-device.txt\n", False),
    ("ssh pi 'cd x && .venv/bin/pip install -q -r requirements-device.txt'\n", False),
    ("pip install gpiozero==2.0.1\n", False),
    ("pip install --upgrade pip\n", False),
    ("pip install -r requirements-device.txt  # unhashed-fallback: refuses on a device\n", True),
    ("# pip install -r requirements-device.txt (a comment)\n", True),
])
def test_the_gate_judges_each_install_line(line, ok):
    setup = "pip install --require-hashes --only-binary=:all: -r requirements-device.lock\n" + line
    assert rm.install_paths({"setup_pi.sh": setup.encode()})["ok"] is ok


def test_a_manifest_records_install_paths_and_the_gate_reads_them():
    gate = {"install_paths": {"files": ["setup_pi.sh"], "violations": ["setup_pi.sh:1: pip install -r x"],
                              "missing_hashed_install": [], "ok": False}}
    manifest = {"model_pins": {"consistent": True, "config_yolo_hef": "m", "config_yolo_hef_sha256": "a"},
                "decoder": {"yolo_decoder": "legacy"}, **gate}
    results = {g["gate"]: g["result"] for g in rm.release_gates(manifest)}
    assert results["every device install path uses pip --require-hashes with the lock (M84-4)"] == "FAIL"
    del manifest["install_paths"]
    assert {g["gate"]: g["result"] for g in rm.release_gates(manifest)}["device install paths"] == "INFO"


# ---------------------------------------------------------------------- the lock
def test_the_lock_is_complete_for_what_the_device_imports():
    result = deps.check()
    assert result["ok"], result["lock"]["problems"]
    assert result["lock"]["ok"] and result["lock"]["packages"] == len(deps.lock_entries(deps.DEVICE_LOCK.read_text()))


def test_gpiozero_and_lgpio_are_locked_and_simpleaudio_is_not_installed_on_the_pi():
    entries = deps.lock_entries((REPO / "requirements-device.lock").read_text())
    assert entries["gpiozero"]["version"] == "2.0.1" and entries["lgpio"]["version"] == "0.2.2.0"
    assert "colorzero" in entries, "gpiozero's dependency is locked too (pip --require-hashes needs it)"
    assert "simpleaudio" not in entries, "no Linux wheel: it was the one package built at install time"
    # the cp311 aarch64 lgpio wheel, and the aarch64 mediapipe wheel whose .tflite files the registry pins
    assert "a2f71fb95b149d8ac82c7c6bae70f054f6dc42a006ad35c90c7d8e54921fbcf4" in entries["lgpio"]["hashes"]
    assert "1061d499d341e7f11dd7f18d5bee747e862db8ef47665ff777323282225d4315" in entries["mediapipe"]["hashes"]
    assert all(e["hashes"] for e in entries.values())
    markers = deps.requirement_markers()
    assert not deps.marker_applies(markers["simpleaudio"]) and deps.marker_applies(markers["gpiozero"])
    assert deps.marker_applies(markers["simpleaudio"], {**deps.DEVICE_ENV, "sys_platform": "darwin"})


def test_the_lock_check_catches_each_kind_of_gap(tmp_path):
    req = tmp_path / "requirements-device.txt"
    lock = tmp_path / "requirements-device.lock"
    h = "    --hash=sha256:" + "0" * 64
    req.write_text('pyyaml==6.0.2\nsimpleaudio==1.0.4 ; sys_platform != "linux"\n'
                   'gpiozero==2.0.1 ; platform_machine == "aarch64"\n')
    lock.write_text(f"pyyaml==6.0.2 \\\n{h}\ngpiozero==2.0.1 \\\n{h}\n")
    assert deps.check_lock(req, lock)["ok"]
    lock.write_text(f"pyyaml==6.0.2 \\\n{h}\n")
    assert "gpiozero==2.0.1 is a device pin but is not in the lock" in deps.check_lock(req, lock)["problems"]
    lock.write_text(f"pyyaml==6.0.2 \\\n{h}\ngpiozero==2.0.1 \\\n{h}\nsimpleaudio==1.0.4 \\\n{h}\n")
    assert any("keeps it off the device" in p for p in deps.check_lock(req, lock)["problems"])
    lock.write_text(f"pyyaml==6.0.1 \\\n{h}\ngpiozero==2.0.1\n")
    problems = deps.check_lock(req, lock)["problems"]
    assert any("lock 6.0.1 != pin 6.0.2" in p for p in problems) and any("no --hash" in p for p in problems)
    lock.write_text(f"pyyaml==6.0.2 \\\n{h}\ngpiozero==2.0.1 \\\n{h}\n")
    assert any("requests" in p for p in deps.check_lock(req, lock, {"requests": "requests"})["problems"])
    req.write_text('pyyaml==6.0.2 ; python_full_version >= "3.11"\n')
    assert any("unsupported requirement marker" in p for p in deps.check_lock(req, lock)["problems"])


def test_the_manifest_pins_are_what_the_device_installs():
    req = rm.requirement_pins((REPO / "requirements-device.txt").read_text())
    assert "simpleaudio" in req["not_on_device"] and "simpleaudio" not in req["pins"]
    assert req["pins"]["gpiozero"] == "2.0.1"
    everything = {p: (REPO / p).read_bytes() for p in ("requirements-device.txt", "requirements-device.lock")}
    lock = rm.lock_status(everything)
    assert lock is not None and lock["consistent"], lock


def test_release_verify_fails_a_venv_that_did_not_come_from_the_lock():
    manifest = {"deploy_set": {"files": {}}, "model_pins": {"config_yolo_hef_sha256": "a"},
                "requirements": {"r": {"pins": {"pyyaml": "6.0.2"}}}, "commit": "c" * 40,
                "dependency_lock": {"locked": {"pyyaml": "6.0.2", "idna": "3.20", "setuptools": "84.2.1"}}}
    snap = {"files": {}, "model": {"sha256": "a", "pin": "a"}, "pip_freeze": "PyYAML==6.0.2\nidna==3.7\n"}
    checks = {c["check"]: c for c in rm.verify(manifest, snap)["checks"]}
    assert checks["installed requirements"]["result"] == "PASS"
    assert checks["installed packages = the hashed lock"]["result"] == "FAIL"
    assert "idna" in checks["installed packages = the hashed lock"]["detail"]
    assert "setuptools" not in checks["installed packages = the hashed lock"]["detail"], "pip freeze hides it"
    snap["pip_freeze"] = "PyYAML==6.0.2\nidna==3.20\n"
    checks = {c["check"]: c for c in rm.verify(manifest, snap)["checks"]}
    assert checks["installed packages = the hashed lock"]["result"] == "PASS"


def test_a_staged_releases_own_records_are_not_code_drift():
    manifest = {"deploy_set": {"files": {"main.py": {"sha256": "1", "sha256_lf": "1"}}},
                "model_pins": {"config_yolo_hef_sha256": "a"}, "requirements": {}, "commit": "c" * 40}
    snap = {"files": {"main.py": {"sha256": "1", "sha256_lf": "1"}, "STAGED.json": {"sha256": "2", "sha256_lf": "2"},
                      "ARTIFACTS.json": {"sha256": "3", "sha256_lf": "3"}},
            "model": {"sha256": "a", "pin": "a"}, "pip_freeze": ""}
    checks = {c["check"]: c for c in rm.verify(manifest, snap)["checks"]}
    assert checks["deployed files"]["result"] == "PASS"
    assert "STAGED.json" in checks["device state files (informational)"]["detail"]
    snap["files"]["core/STAGED.json"] = {"sha256": "4", "sha256_lf": "4"}
    assert {c["check"]: c for c in rm.verify(manifest, snap)["checks"]}["deployed files"]["result"] == "FAIL"
