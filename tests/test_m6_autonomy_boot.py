"""M6 autonomy: boot and restart policy can never become an unbounded reboot loop.

`StartLimitAction=reboot` was bounded only within one boot (systemd's counter
resets on reboot), so any failure deterministic at startup -- an import error
after a bad deploy, a corrupt presets.yaml -- rebooted the Pi for ever (audit 05
F12/F13, row 18). The start limit now only fails the unit; OnFailure= runs a
root-owned guard that reboots on `start-limit-hit` alone, at most 3 times in 6 h,
and only after durably recording the reboot.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import start_limit_guard as guard
from services import runtime

REPO = Path(__file__).resolve().parent.parent
HOUR = 3600.0
BASH = shutil.which("bash") or "bash"


def _unit(path: str) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for raw in (REPO / path).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith(("#", "[")) and "=" in line:
            key, value = line.split("=", 1)
            keys.setdefault(key, []).append(value)
    return keys


# ----------------------------------------------------------------- decision logic
def test_only_a_start_limit_failure_can_reboot():
    for result in ("exit-code", "signal", "watchdog", "timeout", "success", "", "core-dump"):
        reboot, _, reason = guard.decide(result, [], 1000.0)
        assert not reboot, (result, reason)
    reboot, history, _ = guard.decide("start-limit-hit", [], 1000.0)
    assert reboot and history == [1000.0]


def test_the_reboot_budget_is_bounded_across_boots():
    now = 100 * HOUR
    history: list[float] = []
    reboots = 0
    for minute in range(0, 120, 2):  # a deterministic crash: start limit every ~2 min
        reboot, history, _ = guard.decide("start-limit-hit", history, now + minute * 60)
        reboots += reboot
    assert reboots == guard.DEFAULT_MAX_REBOOTS


def test_the_budget_refills_after_the_window():
    now = 100 * HOUR
    old = [now - 7 * HOUR, now - 6.5 * HOUR, now - 6.1 * HOUR]
    reboot, history, _ = guard.decide("start-limit-hit", old, now)
    assert reboot and history == [now], "entries older than the window are forgotten"


def test_a_clock_that_went_backwards_counts_against_the_budget():
    now = 100 * HOUR
    future = [now + HOUR] * guard.DEFAULT_MAX_REBOOTS  # recorded before the clock jumped back
    reboot, _, _ = guard.decide("start-limit-hit", future, now)
    assert not reboot


# ----------------------------------------------------------------- the script end to end
class _Systemctl:
    def __init__(self, result):
        self.result = result
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **_kwargs):
        self.calls.append(cmd)
        out = self.result if cmd[:2] == ["systemctl", "show"] else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out + "\n", stderr="")

    @property
    def rebooted(self):
        return ["systemctl", "reboot"] in self.calls


def test_main_reboots_after_recording(tmp_path):
    state = tmp_path / "state" / "reboots.json"
    run = _Systemctl("start-limit-hit")
    assert guard.main(["--state", str(state)], run=run, clock=lambda: 5000.0) == 0
    assert run.rebooted
    assert guard.load_history(state) == [5000.0]


def test_main_never_reboots_for_a_config_refusal(tmp_path):
    state = tmp_path / "reboots.json"
    run = _Systemctl("exit-code")  # exit 78 is RestartPreventExitStatus: Result=exit-code
    guard.main(["--state", str(state)], run=run, clock=lambda: 5000.0)
    assert not run.rebooted and not state.exists()


def test_main_stops_rebooting_when_the_budget_is_spent(tmp_path, capsys):
    state = tmp_path / "reboots.json"
    clock = iter(range(1000, 100000, 60))
    reboots = 0
    for _ in range(10):
        run = _Systemctl("start-limit-hit")
        guard.main(["--state", str(state)], run=run, clock=lambda: float(next(clock)))
        reboots += run.rebooted
    assert reboots == 3
    assert "leaving the service failed and the device up" in capsys.readouterr().out


def test_an_unrecordable_reboot_does_not_happen(tmp_path):
    state = tmp_path / "is_a_dir"
    state.mkdir()  # writing the state file onto a directory fails
    run = _Systemctl("start-limit-hit")
    assert guard.main(["--state", str(state)], run=run, clock=lambda: 5000.0) == 1
    assert not run.rebooted


def test_dry_run_changes_nothing(tmp_path):
    state = tmp_path / "reboots.json"
    run = _Systemctl("start-limit-hit")
    guard.main(["--state", str(state), "--dry-run"], run=run, clock=lambda: 5000.0)
    assert not run.rebooted and not state.exists()


def test_a_corrupt_state_file_reads_as_empty_but_is_rewritten(tmp_path):
    state = tmp_path / "reboots.json"
    state.write_text("{not json")
    run = _Systemctl("start-limit-hit")
    guard.main(["--state", str(state)], run=run, clock=lambda: 5000.0)
    assert run.rebooted and guard.load_history(state) == [5000.0]


# ----------------------------------------------------------------- the units
def test_the_service_unit_no_longer_reboots_on_its_own():
    unit = _unit("sensordyme.service")
    assert unit["StartLimitAction"] == ["none"]
    assert unit["OnFailure"] == ["sensordyme-recover.service"]
    assert unit["RestartPreventExitStatus"] == ["78"], "a bad config must stop, not retry"
    assert unit["Restart"] == ["always"] and unit["Type"] == ["notify"]
    assert all(cmd.startswith("-") for cmd in unit["ExecStartPre"]), "preflight stays advisory"


def test_the_service_never_waits_for_the_network():
    """The device must monitor with no Wi-Fi at all (first boot, a car)."""
    unit = _unit("sensordyme.service")
    for key in ("After", "Wants", "Requires", "BindsTo", "Requisite"):
        for value in unit.get(key, []):
            assert "network-online" not in value and "NetworkManager-wait-online" not in value, (key, value)
    assert "bluetooth.service" in " ".join(unit.get("Wants", [])), "BT wanted, never required"
    assert "bluetooth.service" not in " ".join(unit.get("Requires", []))


def test_the_start_timeout_covers_the_worst_case_camera_bring_up():
    """READY=1 is sent after camera verify + model warm-up; the watchdog only arms after READY."""
    unit = _unit("sensordyme.service")
    worst_camera = runtime._CAMERA_OPEN_RETRIES * (runtime._CAMERA_VERIFY_TIMEOUT_S + runtime._CAMERA_OPEN_BACKOFF_S)
    assert float(unit["TimeoutStartSec"][0]) >= worst_camera + 60, "no margin left for model load"
    assert float(unit["WatchdogSec"][0]) > 3 * 2.0, "watchdog must exceed several health check intervals"


def test_the_recovery_unit_runs_a_root_owned_copy_not_the_checkout():
    unit = _unit("deploy/systemd/sensordyme-recover.service")
    assert unit["User"] == ["root"] and unit["Type"] == ["oneshot"]
    exec_start = unit["ExecStart"][0]
    assert "/usr/local/lib/sensordyme/start_limit_guard.py" in exec_start
    assert "__SENSORDYME_DIR__" not in exec_start, "root must not execute a file the service user can edit"
    assert "--max 3" in exec_start and "--window 21600" in exec_start
    assert "[Install]" not in (REPO / "deploy/systemd/sensordyme-recover.service").read_text()


def test_setup_installs_the_guard_before_the_service():
    setup = (REPO / "setup_pi.sh").read_text()
    guard_at = setup.index("/usr/local/lib/sensordyme/start_limit_guard.py")
    recover_at = setup.index("/etc/systemd/system/sensordyme-recover.service")
    assert guard_at < setup.index("sudo systemctl daemon-reload")
    assert recover_at < setup.index("sudo systemctl daemon-reload")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_install_units_renders_without_placeholders(tmp_path):
    for script in ("scripts/release/install_units.sh", "setup_pi.sh"):
        assert subprocess.run([BASH, "-n", str(REPO / script)], check=False).returncode == 0, script  # noqa: S603
    out = tmp_path / "rendered"
    subprocess.run([BASH, str(REPO / "scripts/release/install_units.sh"), "--render-only", str(out),  # noqa: S603
                    "--user", "sdtest", "--app-dir", "/home/sdtest/sensordyme"], check=True,
                   capture_output=True)
    rendered = (out / "sensordyme.service").read_text()
    assert "__SENSORDYME" not in rendered
    assert "User=sdtest" in rendered and "/home/sdtest/sensordyme/main.py" in rendered
    assert (out / "start_limit_guard.py").read_bytes() == (REPO / "scripts/start_limit_guard.py").read_bytes()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_install_units_refuses_root(tmp_path):
    result = subprocess.run([BASH, str(REPO / "scripts/release/install_units.sh"), "--render-only",  # noqa: S603
                             str(tmp_path), "--user", "root"], capture_output=True, text=True, check=False)
    assert result.returncode == 1 and "root" in result.stderr
