"""End-to-end cover for the live-feed path: camera -> bus -> inference -> /stream.

Nothing in the suite previously read /stream through a running DeviceRuntime, so
every defect between "the loop has a frame" and "a viewer receives a JPEG" was
invisible to the tests -- which is the whole of the live feed, and the one part of
the product a user looks at directly.

The specific failure pinned down here: annotate() was called UNGUARDED, unlike
process() right above it, and it only runs when someone is watching (see
`want_stream` in services/runtime.py). So a module that annotates badly took down
the entire inference loop at the exact moment the app opened the feed -- and only
then. The device looked healthy until someone watched it.
"""

import logging
import socket
import threading
import time

import numpy as np

import services.runtime as runtime_module
from core.alerts import AlertManager
from modules.base import VisionModule
from services.runtime import DeviceRuntime


class FakeCamera:
    is_pi = False
    fps = 30.0

    def __init__(self, config):
        self.released = False

    def read(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = 128  # not uniform black, so the JPEG is a real image
        return frame

    def reopen(self):
        pass

    def release(self):
        self.released = True


class PlainModule(VisionModule):
    name = "plain"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        return []


class BadAnnotateModule(PlainModule):
    """Annotates badly. Only ever called while a viewer is connected."""

    name = "bad_annotate"

    def __init__(self, config):
        super().__init__(config)
        self.annotate_calls = 0

    def annotate(self, frame):
        self.annotate_calls += 1
        raise ValueError("synthetic annotate failure")


def _config(tmp_path):
    return {
        "module": "plain",
        "camera": {"width": 64, "height": 48, "fps": 30},
        "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 0, "stream_fps": 20},
        "cloud": {"enabled": False},
        "accelerator": {"enabled": True, "yolo_hef": None},
        "network": {"provisioning": {"method": "off"}},
        "health": {"check_interval_s": 0.5, "allow_reboot": False},
        "logging": {
            "csv_path": str(tmp_path / "events.csv"),
            "telemetry_csv_path": str(tmp_path / "telemetry.csv"),
        },
        "modules": {"plain": {}, "bad_annotate": {}},
    }


def _runtime(tmp_path, module, module_name):
    return DeviceRuntime(
        config=_config(tmp_path),
        profile="pi",
        module=module,
        module_name=module_name,
        alert_manager=AlertManager(sinks=[]),
        alert_cfg={},
        alert_event_types=set(),
        alert_types_for=lambda name, preset=None: set(),
        presets={},
        active_preset=None,
        display=False,
        debug=False,
    )


def _read_stream(port: int, seconds: float) -> bytes:
    """Read /stream exactly as the app's MJPEG reader does."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    try:
        sock.sendall(b"GET /stream HTTP/1.1\r\nHost: device\r\n\r\n")
        sock.settimeout(1.0)
        buffer = b""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                break
            buffer += chunk
        return buffer
    finally:
        sock.close()


def test_a_viewer_receives_real_jpeg_frames_through_the_running_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = PlainModule({})
    runtime = _runtime(tmp_path, module, "plain")
    thread = threading.Thread(target=runtime.run)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and module.frames < 5:
            time.sleep(0.05)
        assert module.frames >= 5

        body = _read_stream(runtime.dashboard._port, 2.0)
        assert b"multipart/x-mixed-replace" in body
        assert body.count(b"\xff\xd8") >= 2, "no JPEG frames reached the viewer"
        assert body.count(b"\xff\xd8") == body.count(b"\xff\xd9"), "truncated JPEG frames"
    finally:
        runtime.shutdown.set()
        thread.join(timeout=10)


def test_a_module_that_fails_to_annotate_does_not_kill_the_inference_loop(tmp_path, monkeypatch):
    """annotate() runs ONLY while a viewer is connected, so an unguarded failure
    here meant opening the live feed was what broke detection."""
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)

    # The project's loggers do not propagate to the root logger, so pytest's
    # caplog never sees them; attach to the supervisor's own logger instead.
    crashes: list[str] = []

    class _CrashRecorder(logging.Handler):
        def emit(self, record):
            if "crashed" in record.getMessage():
                crashes.append(record.getMessage())

    recorder = _CrashRecorder()
    supervisor_log = logging.getLogger("sensordyme.supervisor")
    supervisor_log.addHandler(recorder)

    module = BadAnnotateModule({})
    runtime = _runtime(tmp_path, module, "bad_annotate")
    thread = threading.Thread(target=runtime.run)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and module.frames < 5:
            time.sleep(0.05)
        assert module.frames >= 5
        frames_before = module.frames

        _read_stream(runtime.dashboard._port, 1.5)  # opening the feed triggers annotate()
        assert module.annotate_calls > 0, "annotate() was never exercised"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and module.frames < frames_before + 10:
            time.sleep(0.05)
        assert module.frames >= frames_before + 10, (
            "the inference loop stopped processing frames after annotate() raised -- "
            "watching the camera must never take detection down"
        )
        assert thread.is_alive()

        # The supervisor WILL restart a crashed inference loop, which is why this
        # bug hid: frames resume and the loop looks fine. But a restart tears down
        # and rebuilds the module -- zeroing every timer, PERCLOS window and risk
        # EMA -- and burns the restart budget that escalates to a device reboot.
        # A failing annotate() must be contained like a failing process(), i.e.
        # cost one dropped frame, not a restart.
        assert not crashes, (
            f"opening the live feed crashed the inference loop {len(crashes)} time(s): "
            "annotate() is not contained the way process() is"
        )
    finally:
        supervisor_log.removeHandler(recorder)
        runtime.shutdown.set()
        thread.join(timeout=10)
