"""M8.3 through the running DeviceRuntime: video stays live while monitoring is not.

Before M8.3 the stream was fed ONLY by the inference loop's annotate path, so
an explicit stop (the loop returns before reading a frame) or a module whose
annotate() fails froze the stream on its last picture with the connection
open -- and a new viewer was handed that frozen picture as its first frame
(docs/PHASE7_COMPETITION_FREEZE.md: "the video is not proof of life").

Now the camera keeps streaming, unannotated and labelled X-Frame-Source:
camera, and every annotated part carries the age of the CAMERA capture it
was drawn from (the FrameBus publish time), not the encode time.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import services.runtime as runtime_module
from core.alerts import AlertManager
from core.dashboard import DashboardServer
from modules.base import VisionModule
from services.runtime import DeviceRuntime
from sim.stream_client import StreamClient

TOKEN = "m83-runtime-token"


class FakeCamera:
    is_pi = False
    fps = 30.0

    def __init__(self, config):
        self.n = 0

    def read(self, timeout_s=None):
        self.n += 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 128
        frame[0, 0, 0] = self.n % 256
        return frame

    def reopen(self):
        pass

    def release(self):
        pass


class PlainModule(VisionModule):
    name = "plain"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        return []


class SlowModule(PlainModule):
    """Takes 150 ms per frame: annotated frames are visibly older than they are fresh."""

    name = "slow"

    def process(self, frame):
        time.sleep(0.15)
        return super().process(frame)


class BadAnnotateModule(PlainModule):
    name = "bad_annotate"

    def annotate(self, frame):
        raise ValueError("synthetic annotate failure")


def _runtime(tmp_path, module, name):
    config = {
        "module": name,
        "camera": {"width": 64, "height": 48, "fps": 30},
        "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 0, "stream_fps": 20, "token": TOKEN},
        "cloud": {"enabled": False},
        "accelerator": {"enabled": True, "yolo_hef": None},
        "network": {"provisioning": {"method": "off"}},
        "health": {"check_interval_s": 0.5, "allow_reboot": False},
        "logging": {
            "csv_path": str(tmp_path / "events.csv"),
            "telemetry_csv_path": str(tmp_path / "telemetry.csv"),
        },
        "modules": {name: {}},
    }
    return DeviceRuntime(
        config=config, profile="pi", module=module, module_name=name,
        alert_manager=AlertManager(sinks=[]), alert_cfg={}, alert_event_types=set(),
        alert_types_for=lambda n, preset=None: set(), presets={}, active_preset=None,
        display=False, debug=False,
    )


class _Running:
    def __init__(self, runtime: DeviceRuntime, module: PlainModule) -> None:
        self.runtime, self.module = runtime, module
        self.thread = threading.Thread(target=runtime.run)

    def __enter__(self) -> "_Running":
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self.module.frames < 3:
            time.sleep(0.05)
        assert self.module.frames >= 3, "the runtime never started processing"
        return self

    def __exit__(self, *exc) -> None:
        self.runtime.shutdown.set()
        self.thread.join(timeout=10)


def test_video_stays_live_and_labelled_while_monitoring_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = PlainModule({})
    runtime = _runtime(tmp_path, module, "plain")
    with _Running(runtime, module):
        dashboard = runtime.dashboard
        assert dashboard is not None
        port = dashboard._port
        with StreamClient(port, TOKEN, heartbeat=True) as live:
            assert live.read_until(lambda c: len(c.jpeg_parts()) >= 8)
        sources = [p.headers["x-frame-source"] for p in live.jpeg_parts()]
        # The runtime starts drawing only once a viewer exists, so the very first
        # part may be the camera's (labelled so); from the first analysed frame
        # on, the analysed view is what is shown.
        first = sources.index("annotated")
        assert set(sources[first:]) == {"annotated"}, sources

        dashboard.request_protection({"action": "stop", "confirm": True})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and runtime.protection_state()["state"] != "stopped":
            time.sleep(0.05)
        assert runtime.protection_state()["state"] == "stopped"
        processed = module.frames
        time.sleep(DashboardServer.ANNOTATION_GAP_S + 0.2)  # the runtime's last push ages out

        with StreamClient(port, TOKEN, heartbeat=True) as stopped:
            assert stopped.read_until(lambda c: len(c.jpeg_parts()) >= 10, timeout=8), (
                "the stream froze when monitoring stopped: the camera is still running, "
                "the picture is not"
            )
        parts = stopped.jpeg_parts()
        assert {p.headers["x-frame-source"] for p in parts} == {"camera"}
        seqs = [p.int_header("X-Frame-Seq") for p in parts]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "the seq must advance"
        assert all(p.int_header("X-Frame-Age-Ms") < 1000 for p in parts)
        assert module.frames == processed, "nothing may be analysed while stopped"

        dashboard.request_protection({"action": "resume"})
        with StreamClient(port, TOKEN) as resumed:
            assert resumed.read_until(
                lambda c: any(p.headers.get("x-frame-source") == "annotated" for p in c.jpeg_parts()),
                timeout=8,
            ), "annotated frames did not come back after resume"


def test_a_module_that_cannot_annotate_still_shows_the_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = BadAnnotateModule({})
    runtime = _runtime(tmp_path, module, "bad_annotate")
    with _Running(runtime, module), StreamClient(_port(runtime), TOKEN) as client:
        assert client.read_until(lambda c: len(c.jpeg_parts()) >= 5, timeout=8)
    assert {p.headers["x-frame-source"] for p in client.jpeg_parts()} == {"camera"}


def test_annotated_frames_carry_the_capture_age_not_the_encode_age(tmp_path, monkeypatch):
    """The runtime passes the FrameBus capture time: a 150 ms inference shows as >= 150 ms."""
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = SlowModule({})
    runtime = _runtime(tmp_path, module, "slow")
    with _Running(runtime, module), StreamClient(_port(runtime), TOKEN) as client:
        assert client.read_until(lambda c: len(c.jpeg_parts()) >= 4, timeout=8)
    annotated = [p for p in client.jpeg_parts() if p.headers["x-frame-source"] == "annotated"]
    assert annotated
    assert min(p.int_header("X-Frame-Age-Ms") for p in annotated) >= 140


def _port(runtime: DeviceRuntime) -> int:
    assert runtime.dashboard is not None, "the dashboard did not start"
    return runtime.dashboard._port

