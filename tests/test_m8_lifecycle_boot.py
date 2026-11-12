"""M8 lifecycle: startup on a damaged config, a model that fails to build, a missing
accelerator, and the install-time model check.

Each test names the defect it pins. The virtual-device tests drive the real
runtime and backend (sim/, one SimClock); the rest are unit tests where the
virtual device cannot model the edge (bash provisioning, the Hailo probe).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import main
from sim import scenario as sc
from sim.device import CONFIG_DAMAGE
from sim.harness import SimStack
from sim.scenario import CALIBRATION_S

REPO = Path(__file__).resolve().parent.parent
C = CALIBRATION_S
BASH = shutil.which("bash") or "/bin/bash"


@pytest.fixture
def stack(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("mediapipe")
    with SimStack(tmp_path) as s:
        yield s


def _events_csv(stack) -> str:
    path = stack.state_dir / "events.csv"
    return path.read_text() if path.exists() else ""


# ---------------------------------------------------------------------- config corruption
@pytest.mark.parametrize("damage", sorted(CONFIG_DAMAGE))
def test_a_damaged_config_boots_on_the_last_known_good_and_reports_it(stack, damage):
    """Before M8: exit 78 and the unit stopped -- no monitoring, no buzzer, no health, no app
    link -- until someone with SSH fixed config.yaml. A truncated file that still parsed
    booted on whatever survived the cut, with no fault at all."""
    device = stack.boot(sc.get("attentive"))
    device_id = stack.claim()
    device.advance(C)
    lkg = stack.state_dir / "data" / "config.last_good.yaml"
    config = stack.state_dir / "config.yaml"
    assert lkg.read_bytes() == config.read_bytes(), "the config that reached READY is the last known good"
    assert oct(lkg.stat().st_mode & 0o777) == "0o600", "it holds the device key"
    good = lkg.read_bytes()

    reboots: list[int] = []
    device = stack.restart(sc.get("attentive"), config_damage=damage)
    device.runtime._issue_reboot = lambda: reboots.append(1) or False
    damaged = config.read_bytes()
    assert damaged != good
    device.advance(C + 4)

    row = stack.row()
    assert row["id"] == device_id and row["online"], "same key from the saved copy: still paired"
    status = row["status"]
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting", status
    reported = row["health"]["config"]
    assert reported["source"] == "last_known_good" and reported["fault"], reported
    assert reported["running_from"] == str(lkg)
    assert "config_fault" in _events_csv(stack)
    assert config.read_bytes() == damaged, "the broken file is left for a person, never rewritten"
    assert lkg.read_bytes() == good, "running FROM the fallback never makes the broken file 'good'"
    assert reboots == []

    device = stack.restart(sc.get("attentive"))  # config.yaml fixed
    device.advance(4.0)
    assert stack.row()["health"]["config"] == {"source": "config", "path": str(config), "fault": None}


def test_a_damaged_config_on_first_boot_stops_once_without_a_loop(tmp_path):
    """No last-known-good yet: the M0.2 behaviour stands (exit 78, RestartPreventExitStatus)."""
    config = tmp_path / "config.yaml"
    config.write_text("module: [unclosed\n")
    for _attempt in range(3):
        with pytest.raises(SystemExit) as exited:
            main.build_runtime(["--config", str(config), "--no-display"])
        assert exited.value.code == main.CONFIG_ERROR_EXIT
    assert not (tmp_path / "data").exists(), "nothing is recorded from a config that never ran"


def test_a_last_known_good_that_is_itself_broken_is_not_used(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("module: [unclosed\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.last_good.yaml").write_text("modules: 7\nmodule: x\n")
    with pytest.raises(SystemExit) as exited:
        main.build_runtime(["--config", str(config), "--no-display"])
    assert exited.value.code == main.CONFIG_ERROR_EXIT


def test_a_deliberate_trailing_deletion_is_not_mistaken_for_a_torn_write(tmp_path):
    from core import config_lkg

    config = tmp_path / "config.yaml"
    full = (REPO / "config.yaml").read_text()
    config.write_text(full)
    assert config_lkg.save_last_good(config)
    config.write_text(full[: full.index("\nmac_max:") + 1])  # whole lines removed by a person
    assert config_lkg.torn_write(config) is None
    config.write_text(full[: full.index("\nmac_max:") - 7])  # cut mid-line by a power loss
    assert "truncated" in (config_lkg.torn_write(config) or "")


def test_migrate_config_never_leaves_a_half_written_config(tmp_path, monkeypatch):
    """`make deploy` runs scripts/migrate_config.py on the device's config.yaml. It used
    `write_text` (truncate, then write): a power loss mid-deploy left an unparseable config."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import migrate_config
    finally:
        sys.path.remove(str(REPO / "scripts"))
    config = tmp_path / "config.yaml"
    original = "module: driver_attention\ncloud:\n  enabled: true\nnetwork:\n  provisioning:\n    method: ble\n"
    config.write_text(original)
    os.chmod(config, 0o600)

    def power_loss(_fd):
        raise OSError("SIMULATED: power lost during the write")

    monkeypatch.setattr(os, "fsync", power_loss)
    with pytest.raises(OSError, match="power lost"):
        migrate_config.migrate(config)
    assert config.read_text() == original, "the old file, whole"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")], "no temp file stranded"
    monkeypatch.undo()
    assert migrate_config.migrate(config), "the migration itself still applies"
    yaml.safe_load(config.read_text())
    assert oct(config.stat().st_mode & 0o777) == "0o600", "the device key's file mode is kept"


# ---------------------------------------------------------------------- saved state
def test_damaged_saved_state_is_reported_and_kept_aside_not_silently_forgotten(stack):
    """Before M8 a damaged device_state.json / monitoring.json read as empty with a log
    line only: the owner's mode, nodes and rules vanished, the app was never told, and the
    next save overwrote the evidence."""
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    assert stack.set_preset("driver_safety:strict").status_code == 200
    device.advance(4.0)
    assert stack.row()["health"]["saved_state"] == {"faults": []}
    state = stack.state_dir / "device_state.json"
    damaged = state.read_bytes()[:20]
    state.write_bytes(damaged)

    device = stack.restart(sc.get("attentive"))
    device.advance(C + 4)
    row = stack.row()
    faults = row["health"]["saved_state"]["faults"]
    assert len(faults) == 1 and faults[0].startswith("device_state (device_state.json) was damaged"), faults
    assert (stack.state_dir / "device_state.json.corrupt").read_bytes() == damaged
    assert "saved_state_fault" in _events_csv(stack)
    status = row["status"]
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting", \
        "the safe fallback: the role's configured mode"
    assert device.runtime.active_preset != "driver_safety:strict"


# ---------------------------------------------------------------------- model initialisation
def test_a_model_that_fails_at_boot_is_rebuilt_without_a_restart(stack):
    """Before M8: an UnavailableModule stayed for the life of the process, so a transient
    load failure kept the device FAILED (impaired buzzer every 15 s) until a person restarted it."""
    device = stack.boot(sc.get("model_failure"))
    stack.claim()
    device.advance(20.0)
    assert stack.status()["monitoring_state"] == "failed"
    assert device.runtime._module_rebuild_attempts == 0, "the first retry waits 30 s"
    device.advance(15.0)  # first retry at 30 s: still failing
    assert device.runtime._module_rebuild_attempts == 1
    assert stack.status()["monitoring_state"] == "failed"

    device.sim.model_failure = False  # e.g. the model file became readable
    device.advance(60.0 + C)  # second retry at 60 s, then calibration
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting", status
    assert device.runtime.module_name == "driver_attention"
    assert "module_recovered" in _events_csv(stack)
    assert not any(c["name"] == "module" for c in status["capabilities"]["impaired"])


def test_a_model_that_keeps_failing_is_retried_with_bounded_backoff(stack):
    device = stack.boot(sc.get("model_failure"))
    device.advance(600.0)
    # Retries at 30, 60, 120, 240 and 480 s (the wait doubles from 30 s, cap 600 s): not every frame.
    assert device.runtime._module_rebuild_attempts == 5
    assert device.runtime.module.reason.startswith("SIMULATED model failure")
    assert device.runtime._monitoring_state == "failed"


# ---------------------------------------------------------------------- missing accelerator
def test_no_hailo_and_no_cpu_backend_is_reported_as_no_detector_not_as_cpu(tmp_path, monkeypatch):
    """Before M8 the accelerator read 'OK — running on CPU' on a device with no HAT and no
    CPU backend (ultralytics is not installed on a device), while nothing detected objects."""
    pytest.importorskip("mediapipe")
    from core import accel, detectors

    config = yaml.safe_load((REPO / "config.yaml").read_text())
    config["logging"] = {"csv_path": str(tmp_path / "events.csv"),
                         "telemetry_csv_path": str(tmp_path / "telemetry.csv")}
    config["cloud"] = {"enabled": False}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # the device image: no CPU detector
    monkeypatch.setattr(detectors, "_model_cache", {})
    monkeypatch.setattr(accel, "detect", lambda: accel.AcceleratorInfo(False, False, "none",
                                                                       "no Hailo device (running on CPU)"))
    runtime = main.build_runtime(["--config", str(path), "--no-display"])
    try:
        runtime._verify_detection_backend()
        assert runtime._detection_degraded, "no detector at all"
        healthy, detail = runtime._accelerator_health()
        assert healthy is False and detail.startswith("object detection unavailable"), detail
        component = runtime._collect_component_health("test")["accelerator"]
        assert component["state"] != "ok" and "running on CPU" not in component["detail"], component
        assert component["metrics"]["detector_available"] is False
        # Face-based driver monitoring still runs, so the device is analysing (DEGRADED, not off).
        assert runtime._module_runs_without_detector()
    finally:
        runtime.module.close()


# ---------------------------------------------------------------------- install-time model check
PINNED = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
RETIRED = "4c71eab80976a2c9f1b134877890abce02871f6d4d92f53ac5a3f8bf2f24add2"


def _provision(tmp_path: Path, *, present: bytes | None, config_pin: str = PINNED, download: bytes = b"PINNED"):
    """Run setup_pi.sh's model block with fake sha256sum/curl (content decides the 'hash')."""
    text = (REPO / "setup_pi.sh").read_text()
    block = text[text.index("# >>> model provisioning"): text.index("# <<< model provisioning")]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "sha256sum").write_text(
        "#!/bin/sh\ncase \"$(cat \"$1\")\" in\n"
        f"  PINNED) echo \"{PINNED}  $1\";;\n  RETIRED) echo \"{RETIRED}  $1\";;\n"
        "  *) echo \"0000000000000000000000000000000000000000000000000000000000000000  $1\";;\nesac\n")
    (bindir / "curl").write_text(
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do if [ \"$1\" = -o ]; then out=\"$2\"; fi; shift; done\n"
        f"printf '%s' '{download.decode()}' > \"$out\"\n")
    for tool in bindir.iterdir():
        tool.chmod(0o755)
    work = tmp_path / "device"
    (work / "models").mkdir(parents=True)
    (work / "config.yaml").write_text(f"accelerator:\n  yolo_hef_sha256: {config_pin}\n")
    if present is not None:
        (work / "models" / "damoyolo_tinynasL20_T.hef").write_bytes(present)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "PYTHON": sys.executable}
    proc = subprocess.run([BASH, "-c", "set -euo pipefail\n" + block], cwd=work, env=env,  # noqa: S603
                          capture_output=True, text=True, check=False)
    return proc, work / "models"


def test_setup_replaces_the_retired_model_the_runtime_refuses(tmp_path):
    """Historical defect (audit F1): setup_pi.sh KEPT the retired build, so a re-run 'succeeded'
    and the service then refused the .hef at load and booted with no detector."""
    proc, models = _provision(tmp_path, present=b"RETIRED")
    assert proc.returncode == 0, proc.stderr
    assert (models / "damoyolo_tinynasL20_T.hef").read_bytes() == b"PINNED"
    assert (models / "_archive_unreproducible" / f"damoyolo_tinynasL20_T.hef.{RETIRED}").read_bytes() == b"RETIRED"


def test_setup_refuses_when_the_config_pins_a_different_model(tmp_path):
    proc, models = _provision(tmp_path, present=None, config_pin="ab" * 32)
    assert proc.returncode == 1 and "runtime would refuse" in proc.stderr
    assert not (models / "damoyolo_tinynasL20_T.hef").exists(), "nothing installed"


@pytest.mark.parametrize(("present", "ok"), [(b"PINNED", True), (None, True), (b"SOMETHING ELSE", False)])
def test_setup_installs_only_the_pinned_model(tmp_path, present, ok):
    proc, models = _provision(tmp_path, present=present)
    assert (proc.returncode == 0) is ok, proc.stderr
    if ok:
        assert (models / "damoyolo_tinynasL20_T.hef").read_bytes() == b"PINNED"


def test_setup_and_the_runtime_accept_exactly_one_model():
    text = (REPO / "setup_pi.sh").read_text()
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert f'MODEL_SHA256="{config["accelerator"]["yolo_hef_sha256"]}"' in text
    block = text[text.index("# >>> model provisioning"): text.index("# <<< model provisioning")]
    keep_branches = [line for line in block.splitlines() if "Keeping it" in line]
    assert keep_branches == [], "no branch may keep a model the runtime refuses"
