"""How the runtime wires direct mode in: only when enabled, and then into provisioning (R1).

A disabled (default) config must build exactly what it built before: no
manager, no service, no health component, no hook in provisioning.
"""

from __future__ import annotations

import threading
import time

from test_runtime_smoke import CountingModule, FakeCamera, _config

import services.runtime as runtime_module
from core.alerts import AlertManager
from services.runtime import DeviceRuntime


def _run_runtime(tmp_path, monkeypatch, direct_mode):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    config = _config(tmp_path)
    if direct_mode is not None:
        config["network"]["direct_mode"] = direct_mode
    module = CountingModule({})
    runtime = DeviceRuntime(
        config=config, profile="pi", module=module, module_name="counting",
        alert_manager=AlertManager(sinks=[]), alert_cfg={}, alert_event_types={"drowsiness"},
        alert_types_for=lambda name, preset=None: {"drowsiness"}, presets={}, active_preset=None,
        display=False, debug=False,
    )
    thread = threading.Thread(target=runtime.run)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (runtime.supervisor is None or module.frames < 2):
        time.sleep(0.05)
    return runtime, thread


def _parts(runtime):
    """(provisioning, supervisor, health), all built by the time the runtime runs."""
    assert runtime.provisioning is not None and runtime.supervisor is not None and runtime.health is not None
    return runtime.provisioning, runtime.supervisor, runtime.health


def _stop(runtime, thread):
    runtime.shutdown.set()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_a_default_config_builds_no_direct_mode_at_all(tmp_path, monkeypatch):
    runtime, thread = _run_runtime(tmp_path, monkeypatch, None)
    try:
        provisioning, supervisor, health = _parts(runtime)
        assert runtime.direct_mode is None
        assert provisioning._direct_mode_active is None
        assert supervisor.get("direct_mode") is None
        assert "direct_mode" not in health.snapshot()["components"]
    finally:
        _stop(runtime, thread)


def test_an_explicitly_disabled_config_is_the_same_as_none(tmp_path, monkeypatch):
    runtime, thread = _run_runtime(tmp_path, monkeypatch, {"enabled": False, "fallback_enabled": True})
    try:
        assert runtime.direct_mode is None
        assert _parts(runtime)[0]._direct_mode_active is None
    finally:
        _stop(runtime, thread)


def test_an_enabled_config_hands_provisioning_the_direct_mode_check(tmp_path, monkeypatch):
    runtime, thread = _run_runtime(tmp_path, monkeypatch, {"enabled": True,
                                                           "state_path": str(tmp_path / "dm.json")})
    try:
        provisioning, supervisor, _ = _parts(runtime)
        assert runtime.direct_mode is not None
        assert provisioning._direct_mode_active == runtime.direct_mode.is_active
        assert supervisor.get("direct_mode") is not None
        # On this Mac there is no NetworkManager: reported, never pretended.
        assert runtime.direct_mode.status()["mode"] == "unavailable"
    finally:
        _stop(runtime, thread)
