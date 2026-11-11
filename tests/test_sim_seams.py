"""The injection seams sim/ uses, and proof that production behaviour is unchanged.

Each seam is a method or keyword argument whose DEFAULT is exactly the code it
replaced. These pin that default (and that sim/ is the only thing using the
alternative), alongside the existing suites that exercise the same paths
(test_uplink_*, test_runtime_*, test_m01_*, test_m02_*).
"""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

import main
import services.runtime as runtime_module
from core.alerts import ConsoleSink, UnavailableSink
from core.uplink import CloudUplink
from modules.base import VisionModule
from services.runtime import DeviceRuntime, _InferenceLoopState


# ---------------------------------------------------------------------- core/uplink.py
class _Requests:
    def __init__(self, status=200, command=None, stop=None):
        self.calls, self.status, self.command, self.stop = [], status, command or {}, stop

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.stop is not None:
            self.stop.set()
        response = types.SimpleNamespace(status_code=self.status, headers={})
        response.json = lambda: {"command": self.command}
        response.raise_for_status = lambda: None
        return response


def _uplink():
    return CloudUplink({"base_url": "http://backend.local:8787", "device_key": "dk_x", "poll_interval_s": 0.01})


def test_the_uplink_worker_is_a_loop_of_sync_once():
    uplink = _uplink()
    calls = []
    uplink.sync_once = lambda: calls.append(1) or (len(calls) >= 3 and uplink._stop.set()) or False
    uplink._run()
    assert len(calls) == 3


def test_sync_once_is_one_ingest_with_the_device_key_and_applies_the_command():
    uplink = _uplink()
    uplink._requests = _Requests(command={"set_preset": "driver_safety:strict"})
    assert uplink.sync_once() is False
    (url, kwargs), = uplink._requests.calls
    assert url == "http://backend.local:8787/v1/ingest"
    assert kwargs["headers"] == {"X-Device-Key": "dk_x"} and kwargs["allow_redirects"] is False
    assert uplink.connected is True
    assert uplink.pop_pending_preset() == "driver_safety:strict"


def test_sync_once_never_raises_and_reports_offline():
    uplink = _uplink()

    class Down:
        def post(self, *a, **k):
            raise OSError("connection refused")

    uplink._requests = Down()
    assert uplink.sync_once() is False
    assert uplink.connected is False


# ---------------------------------------------------------------------- services/runtime.py
class _Module(VisionModule):
    name = "seam_test"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        return []


def _runtime(**overrides):
    from core.alerts import AlertManager

    return DeviceRuntime(
        config={"module": "seam_test", "modules": {"seam_test": {}}}, profile="pi", module=_Module({}),
        module_name="seam_test", alert_manager=AlertManager(sinks=[]), alert_cfg={},
        alert_event_types=set(), alert_types_for=lambda n, p=None: set(), presets={}, active_preset=None,
        display=False, debug=False, **overrides,
    )


def test_build_module_defaults_to_the_registered_class(monkeypatch):
    built = []
    monkeypatch.setattr(runtime_module, "get_module_class",
                        lambda name: (lambda cfg: built.append((name, cfg)) or _Module(cfg)))
    module = _runtime()._build_module("seam_test", {"x": 1})
    assert isinstance(module, _Module) and built == [("seam_test", {"x": 1})]


def test_make_uplink_defaults_to_cloud_uplink():
    uplink = _runtime()._make_uplink({"base_url": "http://b", "device_key": "k"})
    assert type(uplink) is CloudUplink


def test_monitoring_path_defaults_to_the_module_default(monkeypatch):
    seen = []
    monkeypatch.setattr(runtime_module, "load_or_empty", lambda *a: seen.append(a) or _empty())
    assert DeviceRuntime.monitoring_path is None
    _runtime()
    assert seen == [()], "production must call load_or_empty() with no argument, as before"


def _empty():
    from core.monitoring_config import MonitoringConfig

    return MonitoringConfig("/nonexistent/monitoring.json")


def test_one_inference_pass_processes_one_fresh_frame_and_waits_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime = _runtime()
    runtime.logger = runtime_module._NullEventLog()
    runtime.telemetry = runtime_module._NullEventLog()
    loop = runtime._new_inference_loop_state()
    assert isinstance(loop, _InferenceLoopState) and loop.frame_timeout_s == 0.5 and loop.last_seq == 0
    loop.frame_timeout_s = 0.0
    stop = threading.Event()
    assert runtime._inference_pass(loop, stop) is True  # no frame: a heartbeat, nothing processed
    assert runtime.module.frames == 0 and runtime._heartbeat > 0
    runtime.bus.publish(np.zeros((8, 8, 3), dtype=np.uint8))
    assert runtime._inference_pass(loop, stop) is True
    assert runtime._inference_pass(loop, stop) is True  # same frame: not processed twice
    assert runtime.module.frames == 1 and loop.last_seq == 1


def test_the_inference_loop_is_a_loop_of_inference_passes(monkeypatch):
    runtime = _runtime()
    passes = []
    stop = threading.Event()

    def one(loop, s):
        passes.append(loop)
        if len(passes) == 4:
            stop.set()
        return True

    monkeypatch.setattr(runtime, "_inference_pass", one)
    runtime._consecutive_process_failures = 7
    runtime._inference_loop(stop)
    assert len(passes) == 4 and len({id(p) for p in passes}) == 1
    assert runtime._consecutive_process_failures == 0, "each loop generation starts a fresh failure budget"


def test_a_false_pass_ends_the_loop(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(runtime, "_inference_pass", lambda loop, stop: False)
    runtime._inference_loop(threading.Event())  # returns instead of spinning


# ---------------------------------------------------------------------- main.py
def test_build_alert_manager_without_overrides_is_unchanged():
    manager = main.build_alert_manager({"sinks": ["console", "gpio_buzzer", "nonsense"]}, False)
    kinds = [type(s) for s in manager._sinks]
    assert kinds == [ConsoleSink, UnavailableSink, UnavailableSink]
    assert manager._sinks[1].name == "gpio_buzzer"


def test_sink_overrides_replace_only_the_named_sink():
    marker = object()
    manager = main.build_alert_manager({"sinks": ["console", "gpio_buzzer"]}, False,
                                       sink_overrides={"gpio_buzzer": lambda: marker})
    assert type(manager._sinks[0]) is ConsoleSink and manager._sinks[1] is marker


def test_run_is_build_runtime_then_run(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(run=lambda: calls.append("run"))
    monkeypatch.setattr(main, "build_runtime", lambda argv=None, **kw: calls.append(("build", argv, kw)) or fake)
    main.run(["--config", "x.yaml"])
    assert calls == [("build", ["--config", "x.yaml"], {}), "run"]


@pytest.fixture
def repo_config(tmp_path, monkeypatch):
    import yaml

    monkeypatch.chdir(tmp_path)
    config = yaml.safe_load((main.Path(main.__file__).parent / "config.yaml").read_text())
    config["modules"]["driver_attention"]["calibration_path"] = str(tmp_path / "cal.json")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_build_runtime_defaults_build_a_device_runtime_without_starting_it(repo_config, monkeypatch):
    pytest.importorskip("mediapipe")
    runtime = main.build_runtime(["--config", str(repo_config), "--no-display"])
    try:
        assert type(runtime) is DeviceRuntime
        assert runtime.module_name == "driver_attention" and runtime.camera is None, "nothing started"
    finally:
        runtime.module.close()


def test_a_module_factory_failure_still_boots_failed(repo_config):
    def broken(module_cls, module_config):
        raise RuntimeError("no model")

    runtime = main.build_runtime(["--config", str(repo_config), "--no-display"], module_factory=broken)
    from modules.base import UnavailableModule

    assert isinstance(runtime.module, UnavailableModule) and "no model" in runtime.module.reason
