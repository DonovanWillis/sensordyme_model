"""End-to-end smoke test of DeviceRuntime with a fake camera and module.

Proves the whole wiring: capture thread -> FrameBus -> inference loop ->
events/alerts/dashboard, /api/health serving real monitor state, and a clean
shutdown that releases everything. No real camera, display, or network.
"""

import json
import threading
import time
import urllib.request

import numpy as np

import services.runtime as runtime_module
from core.alerts import AlertManager
from core.camera import CameraError
from core.events import Event
from modules.base import VisionModule
from services.runtime import DeviceRuntime


class FakeCamera:
    is_pi = False
    fps = 60.0

    def __init__(self, config):
        self.released = False

    def read(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def reopen(self):
        pass

    def release(self):
        self.released = True


class CountingModule(VisionModule):
    name = "counting"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0
        self.closed = False

    def process(self, frame):
        self.frames += 1
        if self.frames == 3:
            return [Event.now("drowsiness", 1.0, 0.9, self.name, "test")]
        return []

    def close(self):
        self.closed = True


def _config(tmp_path):
    return {
        "module": "counting",
        "camera": {"width": 64, "height": 48, "fps": 60},
        "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 0},
        "cloud": {"enabled": False},
        "accelerator": {"enabled": True, "yolo_hef": None},
        "network": {"provisioning": {"method": "off"}},
        "health": {"check_interval_s": 0.5, "allow_reboot": False},
        "logging": {
            "csv_path": str(tmp_path / "events.csv"),
            "telemetry_csv_path": str(tmp_path / "telemetry.csv"),
        },
        "modules": {"counting": {}},
    }


def test_runtime_processes_frames_serves_health_and_shuts_down(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = CountingModule({})
    runtime = DeviceRuntime(
        config=_config(tmp_path),
        profile="pi",
        module=module,
        module_name="counting",
        alert_manager=AlertManager(sinks=[]),
        alert_cfg={},
        alert_event_types={"drowsiness"},
        alert_types_for=lambda name, preset=None: {"drowsiness"},
        presets={},
        active_preset=None,
        display=False,
        debug=False,
    )

    thread = threading.Thread(target=runtime.run)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and module.frames < 5:
            time.sleep(0.05)
        assert module.frames >= 5  # capture -> bus -> inference is flowing

        url = runtime.dashboard.url
        with urllib.request.urlopen(url + "/api/health", timeout=5) as resp:
            health = json.loads(resp.read())
        assert health["state"] in ("ok", "degraded")  # degraded: wifi/accel details off-Pi
        assert health["components"]["camera"]["state"] == "ok"
        assert health["components"]["inference"]["state"] == "ok"
        assert "accelerator" in health["components"]

        with urllib.request.urlopen(url + "/api/state", timeout=5) as resp:
            state = json.loads(resp.read())
        assert state["total_events"] >= 1  # the drowsiness event reached the dashboard
    finally:
        runtime.shutdown.set()
        thread.join(timeout=10)

    assert not thread.is_alive()  # clean shutdown
    assert runtime.camera.released
    assert module.closed
    assert (tmp_path / "events.csv").exists()  # the event hit the CSV log too
