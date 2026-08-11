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

import json
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
