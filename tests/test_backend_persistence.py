"""The backend staying up, not just being reachable when it happens to be running.

Every other stage of connectivity (Wi-Fi join, DHCP, mDNS discovery, the app's
auto-relocate-on-failure) is covered elsewhere and was working the day this file
was added. What was not covered, and what actually broke: the backend on a
self-hosted Mac is one `uvicorn` process with nothing to restart it, so it can be
"not running" for reasons that have nothing to do with networking at all — and
that looks, from the device and the app, identical to a real network failure. See
scripts/install_backend_mac.sh and scripts/diagnose_backend_mac.py.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from cloud import mdns
from scripts import diagnose_backend_mac as diag

REPO = Path(__file__).resolve().parents[1]
PLIST_TEMPLATE = REPO / "deploy" / "launchd" / "com.sensordyme.backend.plist.template"
INSTALL_SCRIPT = REPO / "scripts" / "install_backend_mac.sh"

#: What the agent binds so the phone can reach it; named to keep the linter from
#: reading a test fixture as a real bind-to-the-world.
_ANY_INTERFACE = "0.0.0.0"  # noqa: S104


class TestLaunchdPlist:
    def _load(self) -> dict:
        filled = PLIST_TEMPLATE.read_text().replace("__REPO_DIR__", "/tmp/sensordyme-test")
        return plistlib.loads(filled.encode())

    def test_well_formed_once_the_placeholder_is_filled(self):
        self._load()  # must not raise

    def test_runs_uvicorn_against_the_right_app_and_port(self):
        args = self._load()["ProgramArguments"]
        assert args[0].endswith(".venv/bin/uvicorn")
        assert "cloud.app:app" in args
        assert str(mdns.DEFAULT_PORT) in args

    def test_advertises_the_same_port_it_actually_binds(self):
        """A launchd env that announces a port uvicorn isn't on sends the app
        somewhere definite and wrong — worse than not announcing at all."""
        plist = self._load()
        args = plist["ProgramArguments"]
        bound_port = args[args.index("--port") + 1]
        assert plist["EnvironmentVariables"]["SENSORDYME_PORT"] == bound_port

    def test_restarts_on_crash_but_not_forever_on_a_broken_config(self):
        plist = self._load()
        assert plist["KeepAlive"]["Crashed"] is True
        assert plist["ThrottleInterval"] > 0

    def test_starts_at_login(self):
        assert self._load()["RunAtLoad"] is True


class TestInstallScript:
    def test_exists_and_is_executable(self):
        assert INSTALL_SCRIPT.exists()
        assert INSTALL_SCRIPT.stat().st_mode & 0o111, "must be chmod +x to be runnable directly"

    def test_points_at_the_plist_template(self):
        assert "com.sensordyme.backend.plist.template" in INSTALL_SCRIPT.read_text()

    def test_is_safe_to_rerun(self):
        """Re-running after a `git pull` should replace the old agent, not fail
        because one is already loaded."""
        assert "bootout" in INSTALL_SCRIPT.read_text()


class TestDiagnoseBackend:
    def test_port_listening_false_when_nothing_is_there(self):
        # Port 1 is a privileged port nothing in this test suite binds.
        assert diag.port_listening(1) is False

    def test_port_listening_parses_lsof_as_one_flag_not_two(self):
        """lsof wants `-iTCP:8787` as a single argument; splitting it into
        `-iTCP` and `:8787` makes lsof treat the port as a bogus filename and
        report nothing, which reads as "backend not running" even when it is —
        the exact false negative this check exists to avoid."""
        with patch.object(diag.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
            diag.port_listening(8787)
        called_args = mock_run.call_args[0][0]
        assert any(a == "-iTCP:8787" for a in called_args)
        assert not any(a == "-iTCP" for a in called_args)

    def test_http_get_json_surfaces_connection_errors(self):
        with pytest.raises(OSError):
            diag.http_get_json("http://127.0.0.1:1/v1/health", timeout=0.5)

    def test_launchd_status_when_not_installed(self):
        with patch.object(diag.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
            ok, detail = diag.launchd_status()
        assert ok is False
        assert "install_backend_mac.sh" in detail

    def test_launchd_status_when_running(self):
        with patch.object(diag.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="state = running\n"
            )
            ok, detail = diag.launchd_status()
        assert ok is True

    def test_launchd_status_when_installed_but_stopped(self):
        with patch.object(diag.subprocess, "run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="state = not running\n"
            )
            ok, detail = diag.launchd_status()
        assert ok is False
        assert "not running" in detail

    def test_check_reports_and_returns_the_condition(self, capsys):
        assert diag.check("thing", True, "detail") is True
        assert diag.check("thing", False, "detail") is False
        out = capsys.readouterr().out
        assert "✓" in out and "✗" in out


class TestAgentActuallyStartsTheBackend:
    """`launchd_status` answers "is the job running", which is a different
    question from "is the job the backend" — and the two were reported as if they
    were the same.

    What happened: the installed plist was hand-edited to run `main.py`, the
    camera/detector entrypoint, instead of `uvicorn cloud.app:app`. The uvicorn
    process started before that edit was still alive, so the port was open,
    `/v1/health` answered, launchd said "running", and every check passed — while
    the agent was configured to start the wrong program at the next login. The
    class of bug is "the file on disk disagrees with the file in the repo", and
    only comparing them catches it.
    """

    def _write(self, tmp_path, monkeypatch, program_args, env=None):
        plist = tmp_path / "com.sensordyme.backend.plist"
        payload: dict = {"ProgramArguments": program_args}
        if env is not None:
            payload["EnvironmentVariables"] = env
        plist.write_bytes(plistlib.dumps(payload))
        monkeypatch.setattr(diag, "INSTALLED_PLIST", plist)
        return plist

    def test_missing_agent_is_reported_as_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(diag, "INSTALLED_PLIST", tmp_path / "absent.plist")
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is False
        assert "install_backend_mac.sh" in detail

    def test_an_agent_running_uvicorn_against_the_app_passes(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, [
            "/repo/.venv/bin/uvicorn", "cloud.app:app", "--host", _ANY_INTERFACE, "--port", "8787",
        ], env={"SENSORDYME_PORT": "8787"})
        ok, _ = diag.launchd_runs_the_backend()
        assert ok is True

    def test_an_agent_running_the_detector_is_caught(self, tmp_path, monkeypatch):
        """The exact drift that was live on the dev Mac."""
        self._write(tmp_path, monkeypatch, ["/repo/.venv/bin/python", "/repo/main.py"])
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is False
        assert "does NOT start the backend" in detail
        assert "main.py" in detail, "the detail must name what it actually runs"

    def test_uvicorn_against_the_wrong_app_is_caught(self, tmp_path, monkeypatch):
        self._write(tmp_path, monkeypatch, [
            "/repo/.venv/bin/uvicorn", "some.other:app", "--port", "8787",
        ])
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is False
        assert "not the backend app" in detail

    def test_a_port_it_binds_but_does_not_announce_is_caught(self, tmp_path, monkeypatch):
        """Announcing a port uvicorn is not on sends the app somewhere definite
        and wrong, which is worse than not announcing at all."""
        self._write(tmp_path, monkeypatch, [
            "/repo/.venv/bin/uvicorn", "cloud.app:app", "--port", "9999",
        ], env={"SENSORDYME_PORT": "8787"})
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is False
        assert "9999" in detail and "8787" in detail

    def test_a_corrupt_plist_is_a_finding_not_a_crash(self, tmp_path, monkeypatch):
        plist = tmp_path / "com.sensordyme.backend.plist"
        plist.write_bytes(b"this is not a plist")
        monkeypatch.setattr(diag, "INSTALLED_PLIST", plist)
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is False
        assert "could not read" in detail

    def test_the_shipped_template_passes_its_own_check(self, tmp_path, monkeypatch):
        """Whatever the check demands, the template this repo installs must satisfy
        — otherwise a correct install reports itself as broken."""
        filled = PLIST_TEMPLATE.read_text().replace("__REPO_DIR__", "/tmp/sensordyme-test")
        plist = tmp_path / "com.sensordyme.backend.plist"
        plist.write_bytes(plistlib.dumps(plistlib.loads(filled.encode())))
        monkeypatch.setattr(diag, "INSTALLED_PLIST", plist)
        ok, detail = diag.launchd_runs_the_backend()
        assert ok is True, detail


class TestInstallScriptVerifiesWhatItWrote:
    def _text(self) -> str:
        return INSTALL_SCRIPT.read_text()

    def test_it_checks_the_plist_it_wrote_starts_the_backend(self):
        assert "cloud.app:app" in self._text()

    def test_it_backs_up_an_existing_agent_before_replacing_it(self):
        """This script is the documented repair for a hand-edited plist, so the
        edit it replaces should be recoverable rather than simply gone."""
        assert "replaced-" in self._text()

    def test_it_refuses_when_another_process_holds_the_port(self):
        text = self._text()
        assert "lsof" in text and "8787" in text

    def test_it_offers_a_restart_command(self):
        """Restarting is how a moved backend re-announces itself, so the command
        for it belongs where someone will find it."""
        assert "kickstart" in self._text()
