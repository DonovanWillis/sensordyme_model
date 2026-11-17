"""scripts/sim/uitest_control.py: the demo login the UI-test controller signs with."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sim" / "uitest_control.py"
# Loaded by path: scripts/sim is not a package, and `sim` is the repo's simulator.
_spec = importlib.util.spec_from_file_location("uitest_control", _PATH)
assert _spec is not None and _spec.loader is not None
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)


def _stack(tmp_path: Path, replies: list[tuple[int, object]]):
    stack = uc.Stack(tmp_path, backend_port=1, python="python3")
    calls = []

    def api(method, path, body=None, auth=True):
        calls.append((method, path, auth))
        return replies.pop(0)

    stack.api = api
    return stack, calls


def test_a_successful_login_is_cached(tmp_path):
    stack, calls = _stack(tmp_path, [(200, {"token": "abc"})])
    assert stack.token() == "abc"
    assert stack.token() == "abc"
    assert calls == [("POST", "/v1/login", False)], "one login, then the cached token"


@pytest.mark.parametrize("reply", [
    (401, {"detail": "invalid email or password"}),  # was: KeyError('token')
    (200, {"token": None}),                           # was: returned None -> "Bearer None"
    (200, {"token": ""}),
    (502, None),                                      # was: TypeError (None is not subscriptable)
    (200, ["token"]),
])
def test_a_failed_login_says_so_and_caches_nothing(tmp_path, reply):
    stack, _ = _stack(tmp_path, [reply])
    with pytest.raises(RuntimeError, match=f"demo login failed: HTTP {reply[0]}"):
        stack.token()
    assert stack._token is None


def test_the_app_alert_journal_is_found_in_the_simulator_data_container(tmp_path, monkeypatch):
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return uc.subprocess.CompletedProcess(command, 0, stdout=f"{tmp_path}\n", stderr="")

    monkeypatch.setattr(uc.subprocess, "run", run)
    assert uc.app_alert_journal("SIM-1") == tmp_path / "Library/Caches/alert-decisions.jsonl"
    assert seen == [["xcrun", "simctl", "get_app_container", "SIM-1", "com.sensordyme.app", "data"]]


def test_no_app_container_is_an_error_not_an_empty_journal(monkeypatch):
    monkeypatch.setattr(uc.subprocess, "run", lambda command, **kwargs: uc.subprocess.CompletedProcess(
        command, 2, stdout="", stderr="No such app"))
    with pytest.raises(ValueError, match="data container was not found: No such app"):
        uc.app_alert_journal("booted")
