"""The Hailo model's lifecycle: one device, one held pipeline, verified loads.

`hailo_platform` exists only on a Pi with the AI HAT+, so this installs a fake
with the properties that matter: the chip is EXCLUSIVE (a second VDevice in the
process fails, as the real one does), pipeline/activation contexts are counted,
concurrent infer() calls are detected, and a device fault can be injected.
Everything above it — core.accel's cache, integrity check, pipeline handling
and recovery, and core.detectors on top — is the real code.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest

from core import accel, detectors

PHONE = 67


class _Chip:
    """What the fake hardware remembers across objects."""

    def __init__(self) -> None:
        self.open = False
        self.vdevices_opened = 0
        self.pipeline_enters = 0
        self.pipeline_exits = 0
        self.activations = 0
        self.deactivations = 0
        self.infers = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.fail_next_infer = False
        self.fail_vdevice = False


chip = _Chip()


class _Info:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class HEF:
    def __init__(self, path):
        self.path = path

    def get_input_vstream_infos(self):
        return [_Info("input", (640, 640, 3))]

    def get_output_vstream_infos(self):
        return [_Info("nms", (80, 5, 100))]  # one output: NMS on chip


class _Activation:
    def __enter__(self):
        chip.activations += 1
        return self

    def __exit__(self, *exc):
        chip.deactivations += 1
        return False


class _NetworkGroup:
    def create_params(self):
        return {}

    def activate(self, params):
        return _Activation()


class VDevice:
    def __init__(self):
        if chip.fail_vdevice:
            raise RuntimeError("HAILO_DEVICE_IN_USE")
        if chip.open:
            # The real chip is exclusive: this is the failure a second instance
            # in one process hits.
            raise RuntimeError("HAILO_OUT_OF_PHYSICAL_DEVICES")
        chip.open = True
        chip.vdevices_opened += 1

    def configure(self, hef, params):
        return [_NetworkGroup()]

    def release(self):
        chip.open = False


class InferVStreams:
    def __init__(self, network_group, input_params, output_params):
        pass

    def __enter__(self):
        chip.pipeline_enters += 1
        return self

    def __exit__(self, *exc):
        chip.pipeline_exits += 1
        return False

    def infer(self, inputs):
        chip.in_flight += 1
        chip.max_in_flight = max(chip.max_in_flight, chip.in_flight)
        try:
            if chip.fail_next_infer:
                chip.fail_next_infer = False
                raise RuntimeError("HAILO_STREAM_ABORTED_BY_USER")
            time.sleep(0.0005)  # long enough for overlapping callers to collide
            chip.infers += 1
            per_class = [np.zeros((0, 5), dtype=np.float32) for _ in range(80)]
            per_class[PHONE] = np.array([[0.2, 0.2, 0.6, 0.5, 0.9]], dtype=np.float32)
            return {"nms": [per_class]}
        finally:
            chip.in_flight -= 1


@pytest.fixture
def hailo(tmp_path, monkeypatch):
    """A fake HAT, a real-looking .hef on disk, and accel configured for it."""
    global chip
    chip = _Chip()
    fake = types.SimpleNamespace(
        HEF=HEF, VDevice=VDevice, InferVStreams=InferVStreams,
        ConfigureParams=types.SimpleNamespace(create_from_hef=lambda hef, interface=None: {}),
        HailoStreamInterface=types.SimpleNamespace(PCIe="pcie"),
        FormatType=types.SimpleNamespace(UINT8="uint8", FLOAT32="float32"),
        InputVStreamParams=types.SimpleNamespace(make=lambda ng, format_type=None: {}),
        OutputVStreamParams=types.SimpleNamespace(make=lambda ng, format_type=None: {}),
    )
    monkeypatch.setitem(sys.modules, "hailo_platform", fake)

    hef = tmp_path / "damoyolo_tinynasL20_T.hef"
    hef.write_bytes(b"compiled model bytes " * 1000)
    sha = accel.file_sha256(str(hef))

    detectors.release_all()

    def configure(pin=sha, path=hef):
        cfg = {"yolo_hef": str(path)}
        if pin is not None:
            cfg["yolo_hef_sha256"] = pin
        accel.configure(cfg, info=accel.AcceleratorInfo(True, True, "hailo", "fake HAT"))

    configure()
    yield types.SimpleNamespace(path=hef, sha=sha, configure=configure)
    detectors.release_all()
    accel.configure({}, info=accel.AcceleratorInfo(False, False, "none", "no HAT"))


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ------------------------------------------------------------------ one device
def test_two_weights_names_share_one_device(hailo):
    """Model A and model B are both cache keys for the one configured .hef. A
    second VDevice on the exclusive chip used to fail and fall back to a CPU
    backend the device does not have."""
    a = detectors.get_yolo_model("yolo11n.pt")
    b = detectors.get_yolo_model("yolo11s.pt")
    assert a is b
    assert chip.vdevices_opened == 1
    assert a.predict(FRAME)[0].boxes and b.predict(FRAME)[0].boxes, "both coexist and infer"


def test_a_released_model_reloads_on_the_same_chip(hailo):
    first = detectors.get_yolo_model("yolo11n.pt")
    detectors.release_all()
    assert chip.open is False, "release must free the chip"
    second = detectors.get_yolo_model("yolo11n.pt")
    assert second is not first
    assert chip.vdevices_opened == 2
    assert second.predict(FRAME)[0].boxes


# ------------------------------------------------------------------ held pipeline
def test_the_pipeline_is_set_up_once_not_per_frame(hailo):
    """Before: InferVStreams entered and the network group activated, then both
    exited, around EVERY frame. Now: once, held."""
    model = detectors.get_yolo_model("yolo11n.pt")
    for _ in range(100):
        model.predict(FRAME)
    assert chip.infers == 100
    assert chip.pipeline_enters == 1
    assert chip.activations == 1
    assert chip.pipeline_exits == 0, "nothing is torn down between frames"


def test_close_releases_the_pipeline_and_the_device_once(hailo):
    model = detectors.get_yolo_model("yolo11n.pt")
    model.predict(FRAME)
    model.close()
    model.close()  # idempotent
    assert (chip.pipeline_exits, chip.deactivations, chip.open) == (1, 1, False)
    with pytest.raises(RuntimeError, match="after close"):
        model.predict(FRAME)


def test_concurrent_inference_is_serialised(hailo):
    model = detectors.get_yolo_model("yolo11n.pt")
    errors = []

    def worker():
        try:
            for _ in range(25):
                assert model.predict(FRAME)[0].boxes
        except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert chip.infers == 200
    assert chip.max_in_flight == 1, "two threads drove the device at once"
    assert chip.pipeline_enters == 1


def test_a_device_fault_rebuilds_the_pipeline_on_the_next_frame(hailo):
    """The held streams may be dead after a fault. The fault still propagates
    (the runtime counts failures as before); the next frame sets up afresh
    instead of reusing a broken pipeline for ever."""
    model = detectors.get_yolo_model("yolo11n.pt")
    model.predict(FRAME)
    chip.fail_next_infer = True
    with pytest.raises(RuntimeError, match="ABORTED"):
        model.predict(FRAME)
    assert chip.pipeline_exits == 1, "the faulted pipeline was released"
    assert model.predict(FRAME)[0].boxes
    assert chip.pipeline_enters == 2


def test_hailo_init_failure_is_reported_not_fatal(hailo):
    chip.fail_vdevice = True
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    healthy, detail = accel.status()
    assert not healthy and "Hailo init failed" in detail


def test_a_device_that_becomes_available_again_is_usable_after_release(hailo):
    chip.fail_vdevice = True
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    chip.fail_vdevice = False
    hailo.configure()  # e.g. the service restarts after the HAT is reseated
    assert accel.maybe_hailo_yolo("yolo11n.pt") is not None


# ------------------------------------------------------------------ integrity
def test_a_pinned_model_loads_and_carries_its_hash(hailo):
    model = accel.maybe_hailo_yolo("yolo11n.pt")
    assert model is not None and model.sha256 == hailo.sha
    assert accel.status()[0] is True


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda p: p.write_bytes(p.read_bytes() + b"\x00"), id="modified"),
        pytest.param(lambda p: p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2]), id="truncated"),
        pytest.param(lambda p: p.write_bytes(b"a different model entirely"), id="replaced"),
    ],
)
def test_a_model_that_does_not_match_its_pin_is_refused(hailo, tamper):
    tamper(hailo.path)
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert chip.vdevices_opened == 0, "the refused file must never reach the chip"
    healthy, detail = accel.status()
    assert not healthy and "integrity check FAILED" in detail and hailo.sha in detail


def test_an_incorrect_pin_is_refused(hailo):
    hailo.configure(pin="0" * 64)
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert "integrity check FAILED" in accel.status()[1]


def test_a_missing_model_is_reported(hailo):
    hailo.configure(path=hailo.path.with_name("absent.hef"))
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert "missing file" in accel.status()[1]


def test_a_model_replaced_while_running_is_caught_at_the_next_load(hailo):
    """Verification is per load, never per frame: the running instance keeps
    the file it verified, and a reload re-verifies the file now on disk."""
    model = detectors.get_yolo_model("yolo11n.pt")
    hailo.path.write_bytes(b"swapped after boot")
    assert model.predict(FRAME)[0].boxes, "no per-frame hashing"
    detectors.release_all()
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert "integrity check FAILED" in accel.status()[1]


def test_an_unpinned_model_still_loads_and_is_still_identified(hailo):
    hailo.configure(pin=None)
    model = accel.maybe_hailo_yolo("yolo11n.pt")
    assert model is not None and model.sha256 == hailo.sha


def test_with_no_cpu_fallback_the_refusal_reason_reaches_the_person(hailo, monkeypatch):
    """On a device there is no ultralytics, so a refused model means no detector
    at all — and the error must say it was an integrity failure, not "install
    something"."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # import fails, as on a device
    hailo.path.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="integrity check FAILED"):
        detectors.get_yolo_model("yolo11n.pt")


# ------------------------------------------------------------------ identity
def test_the_identity_names_the_file_that_ran_not_the_cache_key(hailo):
    detectors.get_yolo_model("yolo11n.pt")
    assert detectors.model_identity("yolo11n.pt") == {
        "name": "damoyolo_tinynasL20_T.hef",
        "sha256": hailo.sha,
        "backend": "hailo",
    }
    assert detectors.model_identity("never-loaded.pt") is None


def test_boot_verification_records_the_identity_events_will_carry(hailo):
    """services/runtime.py stamps every uplinked event with this identity."""
    from services.runtime import DeviceRuntime

    runtime: Any = types.SimpleNamespace(
        config={"modules": {"driver_attention": {"yolo_weights": "yolo11n.pt"}}},
        module_name="driver_attention", module=types.SimpleNamespace(name="driver_attention"),
        _WEIGHTS_KEYS=DeviceRuntime._WEIGHTS_KEYS, _detection_degraded=None, _model_identity=None,
        _boot_module=None, _boot_module_config=None,
    )
    runtime.__dict__["_active_module_config"] = types.MethodType(DeviceRuntime._active_module_config, runtime)
    runtime.__dict__["_module_identity_name"] = types.MethodType(DeviceRuntime._module_identity_name, runtime)
    DeviceRuntime._verify_detection_backend(runtime)
    assert runtime._detection_degraded is None
    assert runtime._model_identity == {
        "module": "driver_attention",
        "detector": {"name": "damoyolo_tinynasL20_T.hef", "sha256": hailo.sha, "backend": "hailo"},
    }
    assert DeviceRuntime._model_version(runtime) == f"damoyolo_tinynasL20_T.hef@{hailo.sha[:12]}"


def test_closing_does_not_hang_behind_a_wedged_inference(hailo, monkeypatch):
    """A device that stops answering leaves an inference thread inside infer()
    holding the lock. Shutdown used to wait on it for ever (systemd SIGKILL);
    close() now gives up after a bounded wait and marks the model closed."""
    model = detectors.get_yolo_model("yolo11n.pt")
    model.predict(FRAME)  # pipeline held
    release = threading.Event()
    monkeypatch.setattr(InferVStreams, "infer", lambda self, inputs: release.wait() or {})
    worker = threading.Thread(target=lambda: model.predict(FRAME), daemon=True)
    worker.start()
    time.sleep(0.05)
    monkeypatch.setattr(type(model), "_CLOSE_TIMEOUT_S", 0.1)
    started = time.monotonic()
    model.close()
    assert time.monotonic() - started < 1.0, "close() waited on the wedged inference"
    with pytest.raises(RuntimeError, match="after close"):
        model._ensure_ready()
    release.set()
    worker.join(timeout=2)


def test_a_file_swapped_during_the_load_is_refused(hailo, monkeypatch):
    """The file is read twice — once to hash, once by HEF(). A swap between the
    two would pass a check-then-load; the re-hash after loading catches it."""
    original = HEF.__init__

    def swap_mid_load(self, path):
        original(self, path)
        hailo.path.write_bytes(b"swapped between the hash and the load")

    monkeypatch.setattr(HEF, "__init__", swap_mid_load)
    assert accel.maybe_hailo_yolo("yolo11n.pt") is None
    assert "changed while it was being loaded" in accel.status()[1]
    assert chip.open is False, "the instance built from the swapped file was released"
