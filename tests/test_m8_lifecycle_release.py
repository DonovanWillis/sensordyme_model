"""M8 lifecycle: a release that does not come up is rolled back to the previous one.

`make deploy` used to rsync over the running release and stop after the restart:
a release that passed the deploy preflight but could not start (an ImportError,
a crash before READY, a crash loop after it) stayed on the device. These run
scripts/release/release_guard.py against a temp checkout and a fake systemd --
never a real unit, never this Mac's systemd.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.release import release_guard as rg

REPO = Path(__file__).resolve().parent.parent


class _FakeSystemd:
    """`systemctl show` answers from a script per restart; records every command run."""

    def __init__(self, per_start: list[list[dict]]) -> None:
        self.per_start = per_start  # one probe script per (re)start, the last answer repeats
        self.start = 0
        self.index = 0
        self.commands: list[list[str]] = []
        self.now = 0.0
        #: Exit status of the config check (the new release's loader on config.yaml).
        self.config_ok = True

    def run(self, argv: list[str]) -> int:
        self.commands.append(list(argv))
        if argv[:3] == ["sudo", "systemctl", "restart"]:
            self.start += 1
            self.index = 0
        if len(argv) == 3 and argv[1] == "-c" and "config_lkg" in argv[2]:
            return 0 if self.config_ok else 1
        return 0

    def probe(self) -> dict:
        script = self.per_start[min(self.start, len(self.per_start) - 1)]
        answer = script[min(self.index, len(script) - 1)]
        self.index += 1
        return answer

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def clock(self) -> float:
        return self.now


ACTIVE = {"ActiveState": "active", "SubState": "running", "NRestarts": "0"}
STARTING = {"ActiveState": "activating", "SubState": "start", "NRestarts": "0"}
CRASH_LOOP = [STARTING, {"ActiveState": "activating", "SubState": "auto-restart", "NRestarts": "1"}]
FAILED = {"ActiveState": "failed", "SubState": "failed", "NRestarts": "5", "ExecMainStatus": "1"}


def _checkout(root: Path, version: str) -> None:
    (root / "core").mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(f"VERSION = {version!r}\n")
    (root / "core" / "x.py").write_text(f"# {version}\n")
    (root / "requirements-device.txt").write_text("pyyaml\n")


def _device(tmp_path: Path) -> Path:
    root = tmp_path / "sensordyme"
    _checkout(root, "v1")
    # Device state a deploy must never touch, in either direction.
    (root / "data").mkdir()
    (root / "data" / "device_state.json").write_text('{"preset": "driver_safety"}')
    (root / "data" / "events.csv").write_text("t,type\n")
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / "m.hef").write_bytes(b"HEF")
    config = root / "config.yaml"
    config.write_text("module: driver_attention\ncloud:\n  device_key: dk_device_own_key\n")
    os.chmod(config, 0o600)
    return root


def _deploy_v2(root: Path) -> None:
    """What `make deploy` does between snapshot and restart: new code, a migrated config."""
    _checkout(root, "v2")
    (root / "core" / "new_module.py").write_text("import does_not_exist\n")
    with open(root / "config.yaml", "a") as handle:
        handle.write("new_key_old_code_refuses: true\n")


def _guard(root: Path, systemd: _FakeSystemd) -> rg.Guard:
    out: list[str] = []
    guard = rg.Guard(root, run=systemd.run, probe=systemd.probe, clock=systemd.clock, sleep=systemd.sleep,
                     user="sdpi", out=out.append)
    guard.log = out  # type: ignore[attr-defined]
    return guard


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "release" not in p.relative_to(root).parts}


def test_a_release_that_crash_loops_is_rolled_back_byte_for_byte(tmp_path):
    root = _device(tmp_path)
    before = _tree(root)
    systemd = _FakeSystemd([[ACTIVE], CRASH_LOOP, [STARTING, ACTIVE]])
    guard = _guard(root, systemd)
    assert guard.snapshot() == 0
    _deploy_v2(root)
    systemd.run(["sudo", "systemctl", "restart", "sensordyme.service"])  # the deploy's restart

    assert guard.confirm_or_rollback(timeout_s=120, settle_s=10) == 1, "the deploy must fail loudly"
    assert _tree(root) == before, "code, config.yaml and every device file exactly as before the deploy"
    assert not (root / "core" / "new_module.py").exists()
    assert oct((root / "config.yaml").stat().st_mode & 0o777) == "0o600"
    restarts = [c for c in systemd.commands if c[:3] == ["sudo", "systemctl", "restart"]]
    assert len(restarts) == 2, "the previous release is restarted after the restore"
    install = systemd.commands.index(["sudo", "bash", "scripts/release/install_units.sh", "--user", "sdpi"])
    last_restart = len(systemd.commands) - 1 - systemd.commands[::-1].index(restarts[-1])
    assert install < last_restart, "its units are re-installed before the restart"
    record = json.loads((root / "data" / "release" / "rolled_back.json").read_text())
    assert record["previous_healthy"] is True and "restarted on its own" in record["reason"]
    assert not (root / "data" / "release" / "pending.json").exists()


def test_a_release_that_never_reaches_ready_is_rolled_back(tmp_path):
    root = _device(tmp_path)
    systemd = _FakeSystemd([[ACTIVE], [STARTING], [ACTIVE]])  # Type=notify: never READY -> activating
    guard = _guard(root, systemd)
    guard.snapshot()
    _deploy_v2(root)
    systemd.run(["sudo", "systemctl", "restart", "sensordyme.service"])
    assert guard.confirm_or_rollback(timeout_s=60, settle_s=10) == 1
    assert (root / "main.py").read_text() == "VERSION = 'v1'\n"


def test_a_unit_that_fails_outright_is_rolled_back_at_once(tmp_path):
    root = _device(tmp_path)
    systemd = _FakeSystemd([[ACTIVE], [FAILED], [ACTIVE]])
    guard = _guard(root, systemd)
    guard.snapshot()
    _deploy_v2(root)
    systemd.run(["sudo", "systemctl", "restart", "sensordyme.service"])
    assert guard.confirm_or_rollback(timeout_s=600, settle_s=10) == 1
    assert systemd.now <= 10 + 2, "a failed unit is not waited out: only the rollback's settle elapsed"


def test_a_healthy_release_is_confirmed_and_kept(tmp_path):
    root = _device(tmp_path)
    systemd = _FakeSystemd([[ACTIVE], [STARTING, STARTING, ACTIVE]])
    guard = _guard(root, systemd)
    guard.snapshot()
    _deploy_v2(root)
    systemd.run(["sudo", "systemctl", "restart", "sensordyme.service"])
    assert guard.confirm_or_rollback(timeout_s=120, settle_s=10) == 0
    assert (root / "main.py").read_text() == "VERSION = 'v2'\n"
    assert (root / "data" / "release" / "confirmed.json").exists()
    assert guard.rollback() == 0 and (root / "main.py").read_text() == "VERSION = 'v2'\n", \
        "after confirmation there is nothing pending to roll back"


def test_a_release_that_only_runs_on_the_fallback_config_is_rolled_back(tmp_path):
    """The runtime survives a broken config.yaml on its last known good copy (M8) and goes
    active, so 'active' alone would confirm a release whose migration broke the config."""
    root = _device(tmp_path)
    systemd = _FakeSystemd([[ACTIVE], [ACTIVE], [ACTIVE]])
    guard = _guard(root, systemd)
    guard.snapshot()
    before = (root / "config.yaml").read_bytes()
    _deploy_v2(root)
    systemd.config_ok = False
    systemd.run(["sudo", "systemctl", "restart", "sensordyme.service"])
    assert guard.confirm_or_rollback(timeout_s=60, settle_s=10) == 1
    assert (root / "config.yaml").read_bytes() == before, "the pre-deploy config.yaml is back"
    record = json.loads((root / "data" / "release" / "rolled_back.json").read_text())
    assert "does not load" in record["reason"]


def test_a_second_snapshot_over_an_unconfirmed_deploy_is_refused(tmp_path):
    root = _device(tmp_path)
    guard = _guard(root, _FakeSystemd([[ACTIVE]]))
    assert guard.snapshot() == 0
    _deploy_v2(root)
    assert guard.snapshot() == 2, "snapshotting the half-deployed tree would make it the rollback target"
    assert "VERSION = 'v1'" in (root / "data" / "release" / "previous" / "main.py").read_text()


def test_the_snapshot_never_holds_device_state_or_secrets(tmp_path):
    root = _device(tmp_path)
    (root / "cloud").mkdir()
    (root / "cloud" / ".jwt_secret").write_text("secret")
    guard = _guard(root, _FakeSystemd([[ACTIVE]]))
    guard.snapshot()
    held = {p.relative_to(root / "data" / "release" / "previous").as_posix()
            for p in (root / "data" / "release" / "previous").rglob("*") if p.is_file()}
    assert held == {"main.py", "core/x.py", "requirements-device.txt", "MANIFEST.json"}
    copy = root / "data" / "release" / "config.yaml.previous"
    assert oct(copy.stat().st_mode & 0o777) == "0o600", "the config copy holds the device key"


def test_the_guard_runs_from_stdin_as_make_deploy_streams_it(tmp_path):
    """`ssh pi '... python - snapshot --root .' < release_guard.py`: no copy needed on the device."""
    root = _device(tmp_path)
    script = (REPO / "scripts" / "release" / "release_guard.py").read_bytes()
    proc = subprocess.run([sys.executable, "-", "snapshot", "--root", str(root)], input=script,  # noqa: S603
                          capture_output=True, check=False, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr.decode()
    assert (root / "data" / "release" / "pending.json").exists()


def test_make_deploy_snapshots_first_and_rolls_back_on_any_failure():
    makefile = (REPO / "Makefile").read_text()
    recipe = makefile.split("\ndeploy: ", 1)[1].split("\n\n", 1)[0]
    snapshot = recipe.index("- snapshot --root .' < scripts/release/release_guard.py")
    assert snapshot < recipe.index("rsync -av"), "the running release is captured before anything changes"
    restart = recipe.index("systemctl restart sensordyme.service")
    assert restart < recipe.index("confirm-or-rollback"), "health is judged after the restart"
    chain = recipe[recipe.index("rsync -av"):]
    # Every step after the snapshot is one && chain whose failure runs the rollback.
    for step in ("pip install", "migrate_config.py", "install_units.sh", "systemctl restart", "confirm-or-rollback"):
        line = next(ln for ln in chain.splitlines() if step in ln)
        assert line.rstrip().endswith("&& \\") or "confirm-or-rollback" in line, line
    assert "|| { ssh pi@sensordyme.local" in chain and "release_guard.py rollback" in chain and "exit 1; }" in chain
