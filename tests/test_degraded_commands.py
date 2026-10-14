"""A degraded device must still obey the app.

When the detector a module needs is missing, the inference loop idles in a
degraded state. It used to `continue` before reading pending commands, so the
one thing that could fix it -- switching to a mode that needs no detector -- was
never read. The app showed the switch as pending for ever.
"""

import threading
import time
import types

from core.frame_bus import FrameBus
from services.runtime import DeviceRuntime


class _Dashboard:
    stream_interval = 0.1

    def __init__(self, pending):
        self.pending = pending

    def pop_pending_preset(self):
        pending, self.pending = self.pending, None
        return pending


def _runtime(**overrides):
    runtime = object.__new__(DeviceRuntime)
    runtime.bus = FrameBus()
    runtime.shutdown = threading.Event()
    runtime._heartbeat = 0.0
    runtime.uplink = None
    runtime.active_preset = "driver_safety"
    runtime.presets = {"driver_safety": {}, "elderly_monitoring": {}}
    runtime.monitoring = types.SimpleNamespace(active=False)
    for key, value in overrides.items():
        setattr(runtime, key, value)
    return runtime


def _run_for(runtime, seconds):
    stop = threading.Event()
    thread = threading.Thread(target=runtime._inference_loop, args=(stop,), daemon=True)
    thread.start()
    time.sleep(seconds)
    stop.set()
    thread.join(2)


def test_a_degraded_device_can_still_be_switched_to_a_working_mode():
    switched: list[str] = []
    runtime = _runtime(_detection_degraded="no detector", dashboard=_Dashboard("elderly_monitoring"))
    runtime._switch_preset = switched.append
    _run_for(runtime, 0.3)
    assert switched == ["elderly_monitoring"]


def test_commands_are_read_while_the_camera_delivers_nothing():
    switched: list[str] = []
    runtime = _runtime(_detection_degraded=None, dashboard=_Dashboard("elderly_monitoring"))
    runtime._switch_preset = switched.append
    _run_for(runtime, 0.3)
    assert switched == ["elderly_monitoring"]


def test_a_slow_module_swap_is_not_reported_as_a_hung_loop(monkeypatch):
    """A swap loads models on the inference thread. Past the 10s heartbeat the
    check used to report the loop dead, restart it mid-swap, and spend the
    restart budget that leads to a last-resort reboot."""
    import services.runtime as runtime_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock["now"])
    runtime = _runtime(_detection_degraded=None, dashboard=None)
    runtime._heartbeat = clock["now"]

    with runtime._module_swap():
        clock["now"] += 30.0
        ok, detail = runtime._inference_check()
        assert ok, detail
        assert "switching" in detail
        clock["now"] += runtime_module._MAX_SWAP_S
        assert runtime._inference_check()[0] is False, "a swap that never ends is a hang"

    ok, _ = runtime._inference_check()
    assert ok, "the heartbeat is stamped when the swap completes"


def test_a_command_that_raises_does_not_stop_the_loop():
    calls: list[str] = []

    def explode(name):
        calls.append(name)
        raise RuntimeError("old_module.close() raised")

    runtime = _runtime(_detection_degraded="x", dashboard=_Dashboard("elderly_monitoring"))
    runtime._switch_preset = explode
    stop = threading.Event()
    thread = threading.Thread(target=runtime._inference_loop, args=(stop,), daemon=True)
    thread.start()
    time.sleep(0.3)
    try:
        assert calls == ["elderly_monitoring"]
        assert thread.is_alive(), "a failing command killed the inference loop"
    finally:
        stop.set()
        thread.join(2)
