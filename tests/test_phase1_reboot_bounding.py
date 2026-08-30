"""The last-resort reboot must be bounded, and must fail CLOSED when it cannot remember.

Two independent unbounded-reboot paths existed before this suite:

  1. `reboot_min_interval_s` spaced reboots out but bounded nothing. A permanently
     dead camera satisfies "more than 30 minutes since the last one" forever, so the
     device power-cycled every 30 minutes indefinitely.
  2. The rate-limit marker was written inside `try: ... except OSError: pass`. On a
     full or read-only `data/` -- a real SD-card failure mode this repo documents
     elsewhere (core/logger.py) -- the write raised, the exception was swallowed, and
     the reboot proceeded with NO rate limit at all. Every subsequent boot then
     re-read "no reboots yet", so the limit could never engage.

Path 2 is the nastier one: the condition that disables the rate limit (a failing
disk) is *correlated* with the condition that triggers reboots in the first place,
so the two failures arrive together.

These tests drive the real `DeviceRuntime._last_resort` with `subprocess.run`
patched, so nothing reboots anything -- what is asserted is whether the reboot
WOULD have been issued.
"""

from __future__ import annotations

import json

import pytest

from services.runtime import DeviceRuntime


class _Recorder:
    """Stands in for subprocess.run, recording reboot attempts instead of making them."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args, **_kwargs):
        self.calls.append(list(args))
        return


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    """A DeviceRuntime with just enough state for _last_resort, rooted in tmp_path.

    __init__ is bypassed deliberately: building a real runtime opens a camera and
    starts threads, none of which _last_resort touches. This keeps the test about
    the reboot decision and nothing else.
    """
    device = DeviceRuntime.__new__(DeviceRuntime)
    device.config = {"health": {"allow_reboot": True, "reboot_min_interval_s": 1800,
                                "reboot_max_total": 3}}
    device._REBOOT_STATE_PATH = tmp_path / "data" / ".reboot_state.json"
    # nmcli_available() gates the whole path on "this is a real Pi, not a dev Mac".
    monkeypatch.setattr("services.runtime.nmcli_available", lambda: True)
    return device


@pytest.fixture()
def reboots(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("services.runtime.subprocess.run", recorder)
    return recorder


def test_a_reboot_is_issued_and_recorded(runtime, reboots):
    """Baseline: the mechanism still works. A bound that never fires is not a fix."""
    runtime._last_resort("camera")

    assert len(reboots.calls) == 1
    assert reboots.calls[0] == ["sudo", "-n", "systemctl", "reboot"]
    state = json.loads(runtime._REBOOT_STATE_PATH.read_text())
    assert state["count"] == 1
    assert state["last_at"] > 0


def test_total_reboots_are_bounded(runtime, reboots, monkeypatch):
    """The regression: a permanently dead component must not reboot forever.

    Each attempt is spaced past `reboot_min_interval_s`, so the interval limit is
    satisfied every time -- exactly the case where the old code rebooted endlessly.
    """
    clock = [1_000_000.0]
    monkeypatch.setattr("services.runtime.time.time", lambda: clock[0])

    for _ in range(10):
        runtime._last_resort("camera")
        clock[0] += 3600  # an hour later: always past the 1800s interval

    assert len(reboots.calls) == 3, (
        f"expected reboots to stop at reboot_max_total=3, got {len(reboots.calls)}"
    )
    assert json.loads(runtime._REBOOT_STATE_PATH.read_text())["count"] == 3


def test_the_interval_limit_still_applies(runtime, reboots, monkeypatch):
    """Back-to-back failures inside the window get one reboot, not several."""
    clock = [1_000_000.0]
    monkeypatch.setattr("services.runtime.time.time", lambda: clock[0])

    runtime._last_resort("camera")
    clock[0] += 60  # a minute later, well inside reboot_min_interval_s
    runtime._last_resort("camera")

    assert len(reboots.calls) == 1


def test_unwritable_state_blocks_the_reboot_instead_of_uncapping_it(runtime, reboots, monkeypatch):
    """The critical regression: a failing disk must STOP reboots, not unbound them.

    Before the fix the write error was swallowed by `except OSError: pass` and the
    reboot went ahead unrecorded -- so on a full SD card the device would power-cycle
    on every escalation, forever, with nothing on disk to stop it.
    """
    def _refuse(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("services.runtime.os.replace", _refuse)

    runtime._last_resort("camera")

    assert reboots.calls == [], (
        "a reboot was issued despite the rate-limit record failing to persist; "
        "the next boot would see no record and the bound would reset every time"
    )


def test_a_corrupt_state_file_does_not_crash_the_health_monitor(runtime, reboots):
    """Garbage on disk must degrade to 'no reboots yet', not raise inside the monitor."""
    runtime._REBOOT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    runtime._REBOOT_STATE_PATH.write_text("{not json at all")

    runtime._last_resort("camera")  # must not raise

    assert len(reboots.calls) == 1


def test_reboot_stays_disabled_when_config_says_so(runtime, reboots):
    """health.allow_reboot: false must remain absolute, bound or no bound."""
    runtime.config["health"]["allow_reboot"] = False

    runtime._last_resort("camera")

    assert reboots.calls == []
    assert not runtime._REBOOT_STATE_PATH.exists()


def test_max_total_zero_means_unbounded_for_operators_who_want_that(runtime, reboots, monkeypatch):
    """`reboot_max_total: 0` opts out of the total bound, keeping the interval limit.

    Documented escape hatch: a deployment that would rather keep power-cycling than
    sit degraded can say so explicitly, which is different from the old behaviour of
    being unbounded by accident.
    """
    runtime.config["health"]["reboot_max_total"] = 0
    clock = [1_000_000.0]
    monkeypatch.setattr("services.runtime.time.time", lambda: clock[0])

    for _ in range(6):
        runtime._last_resort("camera")
        clock[0] += 3600

    assert len(reboots.calls) == 6
