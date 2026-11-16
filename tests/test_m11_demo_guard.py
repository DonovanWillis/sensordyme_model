"""M11: the mode-cycling demo rehearsal refuses on a role device before writing anything.

scripts/demo_cycle.sh rewrites `module:` in config.yaml. A driver_safety unit may only run
driver_attention (core/roles.py), so cycling other modes would only produce refused configs and
restarts on a safety device. The guard runs before the first write.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _rig(tmp_path: Path, role_line: str) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "demo_cycle.sh", tmp_path / "scripts" / "demo_cycle.sh")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"{role_line}module: driver_attention\ndashboard: {{port: 8899}}\n")
    return cfg


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("modes", [None, "driver_attention baby_monitoring", "occupancy"])
def test_role_device_is_refused_before_config_is_touched(tmp_path, modes):
    cfg = _rig(tmp_path, "role: driver_safety\n")
    before = hashlib.sha256(cfg.read_bytes()).hexdigest()
    env = {"PATH": "/usr/bin:/bin", **({"MODES": modes} if modes else {})}
    bash = shutil.which("bash") or "bash"
    proc = subprocess.run([bash, str(tmp_path / "scripts" / "demo_cycle.sh")], capture_output=True, text=True,  # noqa: S603 - the repo script, copied
                          env=env, timeout=30, check=False)
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr and "demo-check" in proc.stderr
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == before


def test_the_shipped_config_is_a_role_device():
    """The default the friend installs is driver_safety: the guard is what applies to it."""
    assert "role: driver_safety" in (REPO / "config.yaml").read_text()
