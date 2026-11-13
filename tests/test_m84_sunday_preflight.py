"""M8.4 Sunday: scripts/hw_validation/sunday_preflight.py, the one hardware preflight entry point.

Pinned here, on a Mac with no hardware:

  * the dry run (--fake) executes every check end to end and reports
    SIMULATED / NOT RUN, never PASS, and exits 3, never 0;
  * a real run off the Pi reports every hardware check BLOCKED (never PASS),
    and a hardware PASS is impossible off the Pi whatever a check computed;
  * the judging logic PASSes and FAILs correctly on a Pi-shaped test double
    (canned command output: logic tests, not evidence);
  * nothing changes state without --with-stopped-stages AND a typed
    confirmation; a declined or non-interactive confirmation stops nothing;
    a stopped service is always started again;
  * config.yaml is never edited (hashed before and after), the evidence
    summary is written as JSON + Markdown, and the token never reaches it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures_m84_sunday"
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import sunday_preflight as sp  # noqa: E402

TOKEN = "sunday-secret-token-abcdef0123456789"


def _report(out: Path) -> dict:
    (path,) = sorted(out.glob("sunday_preflight_*.json"))
    return json.loads(path.read_text())


def _by_id(report: dict) -> dict[str, dict]:
    return {c["id"]: c for c in report["checks"]}


def _recorder(calls: list[list[str]]):
    """A stand-in for subprocess.call that records argv and runs nothing."""

    def call(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    return call


# ------------------------------------------------------------------ 1. the dry run
def test_fake_dry_run_is_simulated_never_pass_and_writes_both_summaries(tmp_path, capsys):
    out = tmp_path / "ev"
    code = sp.main(["--fake", "--out", str(out), "--install-dir", str(tmp_path / "no-install")])
    assert code == 3
    report = _report(out)
    assert report["verdict"] == "SIMULATED" and report["simulated"] is True and report["on_pi"] is False
    statuses = {c["status"] for c in report["checks"]}
    assert "PASS" not in statuses
    assert statuses <= {"SIMULATED", "NOT RUN"}, statuses
    for c in report["checks"]:
        if c["hardware"]:
            assert c["status"] in ("SIMULATED", "NOT RUN"), c
    ids = [c["id"] for c in report["checks"]]
    # the read-only order the brief asks for
    order = ["identity", "os", "packages", "pins", "hailo", "camera_listing", "service", "health_truth",
             "perception", "stream"]
    assert [i for i in ids if i in order] == order
    md = next(out.glob("sunday_preflight_*.md")).read_text()
    assert "DRY RUN" in md and "| 13 | stream |" in md
    printed = capsys.readouterr().out
    assert "NOT hardware evidence" in printed and "NEXT:" in printed


def test_fake_dry_run_never_stops_a_service_even_with_the_flag(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sp.Preflight, "stage", lambda self, name, *extra: (3, f"STAGE {name}: SIMULATED-PASS"))
    code = sp.main(["--fake", "--with-stopped-stages", "--out", str(tmp_path / "ev")], call=_recorder(calls),
                   confirm=lambda prompt: pytest.fail("a dry run must not ask"))
    assert code == 3 and calls == []
    checks = _by_id(_report(tmp_path / "ev"))
    assert {checks[f"stage_{n}"]["status"] for n in ("model", "capture", "parity", "latency")} == {"SIMULATED"}


# ------------------------------------------------------------------ 2. a real run off the Pi
class _NotAPi(sp.RealSystem):
    def read(self, path: str) -> Optional[str]:
        return None

    def run(self, cmd: list[str], timeout: float = 20.0) -> Optional[str]:
        return None


def test_off_the_pi_every_hardware_check_is_blocked_or_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sp.Preflight, "stage", lambda self, name, *extra: (2 if name == "parse-hef" else 0,
                                                                          f"STAGE {name}"))
    code = sp.main(["--out", str(tmp_path / "ev"), "--install-dir", str(tmp_path / "none")], system=_NotAPi(),
                   env={"SENSORDYME_TOKEN": TOKEN})
    assert code == 2
    report = _report(tmp_path / "ev")
    for c in report["checks"]:
        if c["hardware"]:
            assert c["status"] in ("BLOCKED", "NOT RUN"), c
    assert report["next_manual_step"].startswith("step 1 (identity)")


def test_a_hardware_pass_off_the_pi_is_downgraded_to_blocked(tmp_path):
    pf = sp.Preflight(sp.build_parser().parse_args(["--out", str(tmp_path)]), _NotAPi(), env={})
    check = pf.add(sp.Check("x", 1, "x", hardware=True), sp.PASS, "looked fine")
    assert check.status == sp.BLOCKED and "not a Raspberry Pi" in check.detail
    soft = pf.add(sp.Check("y", 5, "y", hardware=False), sp.PASS, "repo pins")
    assert soft.status == sp.PASS


# ------------------------------------------------------------------ 3. the judging logic on a Pi-shaped double
class _PiDouble(sp.FakeSystem):
    """Canned Pi answers with on_pi True: exercises PASS/FAIL logic. A logic test, not evidence."""

    fake = False

    def __init__(self, install_dir: Path, *, health: Optional[dict] = None, stopped: bool = False) -> None:
        super().__init__(install_dir)
        self._health = health
        self._stopped = stopped
        self.state = "active"

    def on_pi(self) -> bool:
        return True

    def exists(self, path: str) -> bool:
        return path == "/dev/hailo0" or Path(path).exists()

    def run(self, cmd: list[str], timeout: float = 20.0) -> Optional[str]:
        if cmd[:2] == ["systemctl", "show"] and cmd[-1] == "ActiveState" and len(cmd) == 5:
            return f"ActiveState={self.state}"
        if cmd[0].endswith("python") and cmd[1:] == ["--version"]:
            return "Python 3.11.2"
        return super().run(cmd, timeout)

    def get_json(self, url: str, headers: dict[str, str]):
        if url.endswith("/api/health"):
            if self._health is not None:
                return self._health
            health = self.health(stopped=self._stopped)
            health["device"]["simulated"] = False
            health["device"]["monitoring"]["perception"]["simulated"] = False
            return health
        return super().get_json(url, headers)


def _install(tmp_path: Path) -> Path:
    install = tmp_path / "sensordyme"
    if (install / "config.yaml").exists():
        return install
    (install / "models").mkdir(parents=True)
    hef = install / "models" / sp.HEF_NAME
    hef.write_bytes(b"test hef bytes")
    config = (REPO / "config.yaml").read_text()
    pin = hashlib.sha256(hef.read_bytes()).hexdigest()
    real_pin = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
    assert real_pin in config
    (install / "config.yaml").write_text(config.replace(real_pin, pin))
    return install


def _run_double(tmp_path, monkeypatch, system=None, extra=(), env=None, **kw) -> dict:
    install = _install(tmp_path)
    monkeypatch.setattr(sp.Preflight, "stage", lambda self, name, *e: (0, f"STAGE {name}: PASS"))
    system = system or _PiDouble(install)
    system.install_dir = install
    system.outputs["systemctl show"] = system.outputs["systemctl show"].replace(
        "WorkingDirectory=/home/pi/sensordyme", f"WorkingDirectory={install}").replace(
        f"WorkingDirectory={sp.hc.DEFAULT_INSTALL_DIR}", f"WorkingDirectory={install}")
    commit = sp.hc.git_commit(sp.REPO_ROOT) or "nocommit"
    sp.main(["--install-dir", str(install), "--out", str(tmp_path / "ev"), "--expect-commit",
             commit.removeprefix("file:")[:12], *extra], system=system,
            env={"SENSORDYME_TOKEN": TOKEN} if env is None else env, **kw)
    return _report(tmp_path / "ev")


def test_a_healthy_pi_double_passes_every_read_only_check(tmp_path, monkeypatch):
    report = _run_double(tmp_path, monkeypatch)
    checks = _by_id(report)
    read_only = ["identity", "power", "os", "packages", "software_revision", "pins", "model_artifact", "hailo",
                 "parse_hef", "camera_listing", "service", "auth", "health_truth", "perception", "stream",
                 "config_unchanged"]
    assert {i: checks[i]["status"] for i in read_only} == dict.fromkeys(read_only, "PASS")
    assert checks["privacy_stopped"]["status"] == "NOT RUN"  # monitoring runs
    assert checks["backend_truth"]["status"] == "NOT RUN"
    assert checks["stage_model"]["status"] == "NOT RUN"
    assert report["verdict"] == "PASS" and report["exit_code"] == 0
    assert "step 12" in report["next_manual_step"] or "step 22" in report["next_manual_step"] \
        or "step 8" in report["next_manual_step"]


def test_the_token_never_reaches_the_evidence(tmp_path, monkeypatch):
    _run_double(tmp_path, monkeypatch)
    for path in (tmp_path / "ev").glob("sunday_preflight_*"):
        assert TOKEN not in path.read_text()


def test_throttling_a_wrong_camera_and_a_contradiction_are_failures(tmp_path, monkeypatch):
    install = _install(tmp_path)
    health = json.loads((FIX / "device_health_ready.json").read_text())
    health["device"]["monitoring"]["state"] = "failed"  # protecting + active kept: torn
    system = _PiDouble(install, health=health)
    system.outputs["vcgencmd get_throttled"] = "throttled=0x50005"
    system.outputs["rpicam-hello"] = "0 : imx708 [4608x2592 10-bit RGGB] (/base/...)"
    checks = _by_id(_run_double(tmp_path, monkeypatch, system=system))
    assert checks["power"]["status"] == "FAIL"
    assert checks["camera_listing"]["status"] == "FAIL"
    assert checks["health_truth"]["status"] == "FAIL"
    assert "protecting_requires_monitoring" in checks["health_truth"]["detail"]
    # the recorded fixture is the virtual device: also flagged as not the hardware path
    assert "simulated" in checks["health_truth"]["detail"]


def test_no_camera_while_the_service_holds_it_is_blocked_not_failed(tmp_path, monkeypatch):
    install = _install(tmp_path)
    system = _PiDouble(install)
    system.outputs["rpicam-hello"] = "No cameras available!"
    checks = _by_id(_run_double(tmp_path, monkeypatch, system=system))
    assert checks["camera_listing"]["status"] == "BLOCKED"


def test_without_a_token_the_dashboard_checks_are_blocked(tmp_path, monkeypatch):
    checks = _by_id(_run_double(tmp_path, monkeypatch, env={}))
    assert {checks[i]["status"] for i in ("auth", "health_truth", "stream")} == {"BLOCKED"}


def test_a_stopped_device_runs_the_privacy_check_instead_of_the_stream(tmp_path, monkeypatch):
    install = _install(tmp_path)
    system = _PiDouble(install, stopped=True)
    system.stream = lambda base, token, seconds: {  # type: ignore[method-assign]
        "status": 200, "parts": [{"t": 0.5, "bytes": 9000, "jpeg": True, "headers": {
            "content-type": "image/jpeg", "x-frame-seq": "7", "x-frame-age-ms": "40", "x-frame-source": "camera",
            "x-stream-id": "a"}}]}
    checks = _by_id(_run_double(tmp_path, monkeypatch, system=system))
    assert checks["stream"]["status"] == "NOT RUN"
    assert checks["privacy_stopped"]["status"] == "FAIL"
    assert "Decision C" in checks["privacy_stopped"]["detail"]
    assert "TODO-INTEGRATOR" in checks["privacy_stopped"]["next"]


# ------------------------------------------------------------------ 4. state changes need a flag and a typed confirmation
def test_declined_confirmation_stops_nothing(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    report = _run_double(tmp_path, monkeypatch, extra=["--with-stopped-stages"], call=_recorder(calls),
                         confirm=lambda prompt: "no")
    assert calls == []
    assert {c["status"] for c in report["checks"] if c["id"].startswith("stage_") and c["id"] != "stage_thermal"} \
        == {"NOT RUN"}


def test_no_terminal_means_no_confirmation(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    _run_double(tmp_path, monkeypatch, extra=["--with-stopped-stages"], call=_recorder(calls))
    assert calls == []


def test_confirmed_stop_runs_the_stages_and_always_restarts_the_service(tmp_path, monkeypatch):
    install = _install(tmp_path)
    system = _PiDouble(install)
    calls: list[list[str]] = []

    def call(argv: list[str]) -> int:
        calls.append(argv)
        system.state = "inactive" if argv[2] == "stop" else "active"
        return 0

    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    report = _run_double(tmp_path, monkeypatch, system=system, extra=["--with-buzzer"], call=call,
                         confirm=lambda prompt: "STOP")
    assert calls == [["sudo", "systemctl", "stop", "sensordyme.service"],
                     ["sudo", "systemctl", "start", "sensordyme.service"]]
    checks = _by_id(report)
    assert {checks[f"stage_{n}"]["status"] for n in ("model", "capture", "parity", "latency", "buzzer")} == {"PASS"}
    assert checks["service_restored"]["status"] == "PASS"


def test_a_crashing_stage_still_restarts_the_service(tmp_path, monkeypatch):
    install = _install(tmp_path)
    system = _PiDouble(install)
    calls: list[list[str]] = []

    def call(argv: list[str]) -> int:
        calls.append(argv)
        system.state = "inactive" if argv[2] == "stop" else "active"
        return 0

    def boom(self, name, *extra):
        if name == "capture":
            raise RuntimeError("stage crashed")
        return 0, f"STAGE {name}: PASS"

    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    install_cfg = install / "config.yaml"
    commit = (sp.hc.git_commit(sp.REPO_ROOT) or "x").removeprefix("file:")[:12]
    monkeypatch.setattr(sp.Preflight, "stage", boom)
    sp.main(["--install-dir", str(install), "--out", str(tmp_path / "ev"), "--expect-commit", commit,
             "--with-stopped-stages"], system=system, env={"SENSORDYME_TOKEN": TOKEN}, call=call,
            confirm=lambda prompt: "STOP")
    assert calls[-1] == ["sudo", "systemctl", "start", "sensordyme.service"]
    checks = _by_id(_report(tmp_path / "ev"))
    assert checks["service_restored"]["status"] == "PASS"
    assert install_cfg.exists()


# ------------------------------------------------------------------ 5. config.yaml is never edited; output placement
def test_a_config_change_during_the_run_is_a_failure(tmp_path, monkeypatch):
    install = _install(tmp_path)
    original = sp.Preflight.perception_check

    def tamper(self):
        (install / "config.yaml").write_text((install / "config.yaml").read_text() + "\n# changed\n")
        original(self)

    monkeypatch.setattr(sp.Preflight, "perception_check", tamper)
    system = _PiDouble(install)
    monkeypatch.setattr(sp.Preflight, "stage", lambda self, name, *e: (0, f"STAGE {name}: PASS"))
    sp.main(["--install-dir", str(install), "--out", str(tmp_path / "ev")], system=system,
            env={"SENSORDYME_TOKEN": TOKEN})
    checks = _by_id(_report(tmp_path / "ev"))
    assert checks["config_unchanged"]["status"] == "FAIL"


def test_evidence_inside_the_install_tree_is_refused(tmp_path):
    install = tmp_path / "sensordyme"
    install.mkdir()
    assert sp.main(["--fake", "--install-dir", str(install), "--out", str(install / "ev")]) == 2
    assert not (install / "ev").exists()


def test_the_preflight_source_never_writes_config(tmp_path):
    """Static guard: no write to config.yaml anywhere in the orchestrator."""
    text = (REPO / "scripts" / "hw_validation" / "sunday_preflight.py").read_text()
    for needle in ("config_path().write", "safe_dump", "migrate_config", "config.yaml\").write"):
        assert needle not in text
    shutil.rmtree(tmp_path, ignore_errors=True)
