"""Regression tests for the runtime-level fixes from the full-repo review.

Each test names the failure it pins. They drive the real DeviceRuntime (built
without _build(), so no camera or threads) and main.py's boot helpers.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

import main
from core import device_state
from core.alerts import AlertManager
from core.events import Event
from core.logger import EventLogger
from modules.base import VisionModule, register_module
from services.runtime import DeviceRuntime


@register_module("coordinator_fix_zone_a")
class _ZoneModuleA(VisionModule):
    name = "coordinator_fix_zone_a"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.zone_calls: list[list] = []

    def process(self, frame: np.ndarray) -> list[Event]:
        return []

    def set_zones(self, zones: list) -> None:
        self.zone_calls.append(list(zones))


@register_module("coordinator_fix_zone_b")
class _ZoneModuleB(_ZoneModuleA):
    name = "coordinator_fix_zone_b"


_PRESETS = {
    "mode_a": {"title": "Mode A", "module": "coordinator_fix_zone_a"},
    "mode_b": {
        "title": "Mode B",
        "module": "coordinator_fix_zone_b",
        "variants": {"strict": {"title": "Strict", "overrides": {"sensitivity": 9}}},
    },
}

_APP_ZONES = [{"name": "crib", "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]],
               "kind": "safe"}]


class _FakeUplink:
    """The slice of CloudUplink that _apply_pending_commands reads."""

    def __init__(self) -> None:
        self.zones = None
        self.preset = None
        self.monitoring = None
        self.active_preset = None

    def pop_pending_preset(self):
        pending, self.preset = self.preset, None
        return pending

    def pop_pending_zones(self):
        pending, self.zones = self.zones, None
        return pending

    def pop_pending_rules(self):
        return None

    def set_monitoring(self, described):
        self.monitoring = described

    def set_active_preset(self, name):
        self.active_preset = name


def _runtime(tmp_path, *, config_module="coordinator_fix_zone_a") -> DeviceRuntime:
    config = {
        "module": config_module,
        "modules": {"coordinator_fix_zone_a": {}, "coordinator_fix_zone_b": {}},
    }
    runtime = DeviceRuntime(
        config=config,
        profile="pi",
        module=_ZoneModuleA({}),
        module_name="coordinator_fix_zone_a",
        alert_manager=AlertManager(sinks=[]),
        alert_cfg={},
        alert_event_types=set(),
        alert_types_for=lambda name, preset=None: set(),
        presets=_PRESETS,
        active_preset="mode_a",
        display=False,
        debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    return runtime


def _engine_zones(rt: DeviceRuntime) -> list[str]:
    return sorted(z.name for z in rt.rule_engine._zones.values())


@pytest.fixture
def runtime(tmp_path):
    rt = _runtime(tmp_path)
    try:
        yield rt
    finally:
        rt.logger.close()


# ------------------------------------------------------------------ app zones
def test_app_zones_reach_the_running_module_not_only_the_rule_engine(runtime):
    """A crib drawn in the app used to reach only AI-configured rules, so the
    module's own safe_zone_exit could never fire."""
    runtime.uplink = _FakeUplink()
    runtime.uplink.zones = _APP_ZONES

    runtime._apply_pending_commands()

    assert [z.name for z in runtime.module.zone_calls[-1]] == ["crib"]
    assert _engine_zones(runtime) == ["crib"]


def test_app_zones_follow_a_preset_switch(runtime):
    runtime.uplink = _FakeUplink()
    runtime.uplink.zones = _APP_ZONES
    runtime._apply_pending_commands()

    runtime._switch_preset("mode_b")

    assert isinstance(runtime.module, _ZoneModuleB)
    assert [z.name for z in runtime.module.zone_calls[-1]] == ["crib"]
    assert _engine_zones(runtime) == ["crib"]


def test_app_zones_survive_a_restart_without_the_network(runtime, tmp_path):
    """An offline baby monitor must not come back without its crib zone."""
    runtime.uplink = _FakeUplink()
    runtime.uplink.zones = _APP_ZONES
    runtime._apply_pending_commands()

    rebooted = _runtime(tmp_path)
    try:
        assert [z.name for z in rebooted.module.zone_calls[-1]] == ["crib"]
        assert _engine_zones(rebooted) == ["crib"]
    finally:
        rebooted.logger.close()


def test_deleting_every_zone_in_the_app_clears_them(runtime):
    runtime.uplink = _FakeUplink()
    runtime.uplink.zones = _APP_ZONES
    runtime._apply_pending_commands()
    runtime.uplink.zones = []
    runtime._apply_pending_commands()

    assert runtime.module.zone_calls[-1] == []
    assert _engine_zones(runtime) == []


def test_without_app_zones_modules_keep_their_config_zones(runtime):
    assert runtime.module.zone_calls == [], "nothing may override config.yaml zones unasked"


# ------------------------------------------------------------------ preset persistence
def test_a_preset_switch_is_restored_at_boot(runtime):
    runtime._switch_preset("mode_b:strict")

    restored = main.restored_preset(
        "coordinator_fix_zone_a", {"coordinator_fix_zone_b": {}}, _PRESETS
    )

    assert restored is not None
    name, preset, overrides = restored
    assert name == "mode_b:strict"
    assert preset["module"] == "coordinator_fix_zone_b"
    assert overrides == {"sensitivity": 9}


def test_editing_config_module_outranks_the_saved_preset(runtime):
    """scripts/demo_cycle.sh switches modes by editing config.yaml's module:."""
    runtime._switch_preset("mode_b")

    assert main.restored_preset(
        "some_other_module", {"coordinator_fix_zone_b": {}}, _PRESETS
    ) is None


def test_a_saved_preset_that_no_longer_exists_is_ignored():
    device_state.save_preset("retired_mode", "coordinator_fix_zone_a")
    assert main.restored_preset("coordinator_fix_zone_a", {}, _PRESETS) is None


def test_an_unreadable_state_file_reads_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("[1, 2")
    monkeypatch.setenv("SENSORDYME_DEVICE_STATE_PATH", str(path))
    assert device_state.load() == {}
    assert device_state.saved_zones() is None


# ------------------------------------------------------------------ alert before log
def test_a_rule_alert_sounds_even_when_the_event_log_cannot_be_written(runtime):
    order: list[str] = []

    class _Alerts:
        def notify(self, event, **_kwargs):
            order.append("notify")

    class _BrokenLog:
        def log(self, event):
            order.append("log")
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    runtime.alert_manager = _Alerts()
    runtime.logger = _BrokenLog()

    class _Alert:
        detection, duration_s, confidence = "person", 2.0, 0.9
        severity, notify, rule_id = "critical", "push", "r1"

        def message(self):
            return "person in the pool"

    with pytest.raises(OSError, match="No space"):
        runtime._emit_rule_alert(_Alert())
    assert order[0] == "notify"


# ------------------------------------------------------------------ inference health
def test_starting_is_bounded_so_a_wedged_loop_cannot_read_healthy(runtime, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("services.runtime.time.monotonic", lambda: clock[0])
    runtime._heartbeat = 0.0
    runtime._heartbeat_reset_at = 1000.0

    assert runtime._inference_check()[0] is True
    clock[0] += 31.0
    assert runtime._inference_check()[0] is False


def test_a_restart_that_did_not_happen_keeps_reporting_the_old_loop(runtime):
    class _Supervisor:
        def restart(self, name):
            raise RuntimeError("inference did not stop")

    runtime.supervisor = _Supervisor()
    runtime._heartbeat = 123.0

    with pytest.raises(RuntimeError):
        runtime._restart_inference()
    assert runtime._heartbeat == 123.0


# ------------------------------------------------------------------ last resort
class _Recorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, args, **_kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, self.returncode)


@pytest.fixture
def bare_runtime(tmp_path, monkeypatch):
    device = DeviceRuntime.__new__(DeviceRuntime)
    device.config = {"health": {"allow_reboot": True, "reboot_min_interval_s": 1800,
                                "reboot_max_total": 3}}
    device._REBOOT_STATE_PATH = tmp_path / "data" / ".reboot_state.json"
    monkeypatch.setattr("services.runtime.nmcli_available", lambda: True)
    return device


def test_with_a_watchdog_the_process_is_restarted_before_any_reboot(bare_runtime, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("services.runtime.subprocess.run", recorder)
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    monkeypatch.setenv("WATCHDOG_USEC", "45000000")
    clock = [1_000_000.0]
    monkeypatch.setattr("services.runtime.time.time", lambda: clock[0])

    bare_runtime._last_resort("camera")
    assert recorder.calls == [], "rebooted before the watchdog restarted the process"
    state = json.loads(bare_runtime._REBOOT_STATE_PATH.read_text())
    assert state["process_restarts"]["camera"] == clock[0]

    clock[0] += 120  # the new process: down again soon after the restart
    bare_runtime._last_resort("camera")
    assert len(recorder.calls) == 1


def test_a_reboot_command_that_fails_does_not_spend_the_bound(bare_runtime, monkeypatch):
    recorder = _Recorder(returncode=1)
    monkeypatch.setattr("services.runtime.subprocess.run", recorder)

    bare_runtime._last_resort("camera")

    assert len(recorder.calls) == 1
    state = json.loads(bare_runtime._REBOOT_STATE_PATH.read_text())
    assert state["count"] == 0


# ------------------------------------------------------------------ config errors
def test_an_invalid_config_exits_with_ex_config(tmp_path):
    """Exit 78 is in RestartPreventExitStatus: a bad config stops the service
    instead of reboot-looping the Pi."""
    bad = tmp_path / "config.yaml"
    bad.write_text("module: driver_attention\nalerts:\n  cooldown_s: -5\n")

    with pytest.raises(SystemExit) as exited:
        main.run(["--config", str(bad)])
    assert exited.value.code == main.CONFIG_ERROR_EXIT


# ------------------------------------------------------------------ degraded recovery
def test_a_degraded_device_recovers_when_its_detector_becomes_available(runtime, monkeypatch):
    """Boot verified the detector once; a transient failure used to leave the
    device refusing to analyse until someone restarted it."""
    runtime.config["modules"]["coordinator_fix_zone_a"] = {"yolo_weights": "yolo11n.pt"}
    attempts = []

    def _require_backend(weights, device):
        attempts.append(weights)
        if len(attempts) == 1:
            raise RuntimeError("Hailo init failed")
        return "hailo"

    monkeypatch.setattr("core.detectors.require_backend", _require_backend)
    monkeypatch.setattr("core.detectors.model_identity", lambda weights, device: {"name": weights})

    runtime._verify_detection_backend()
    assert runtime._detection_degraded == "Hailo init failed"

    runtime._retry_detection_backend()
    assert runtime._detection_degraded is None
    assert runtime.model_identity["detector"] == {"name": "yolo11n.pt"}
