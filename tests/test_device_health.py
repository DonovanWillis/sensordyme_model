"""Device health reporting: only claims what the device can actually observe.

The point of this pipeline is that "AI Online" in the app means the Hailo
completed an inference, and "Camera OK" means frames arrived — not that a row
exists in a database. These tests hold that line.
"""

import importlib
import json
import sys

import pytest

from core import accel, device_health
from core.device_health import FAILED, OK, UNKNOWN, WARNING


class FakeBus:
    def __init__(self, age=0.1, sequence=100):
        self._age = age
        self.sequence = sequence

    def age_s(self):
        return self._age


class FakeCapture:
    """Mirrors services.CaptureService: `is_running` is a PROPERTY, not a method.

    An earlier version of this double defined it as a method. The tests passed
    and the real device crashed with "'bool' object is not callable" on its
    first health sample — a fake that is easier to write than the real thing is
    a fake that hides the bug it exists to catch.
    """

    def __init__(self, running=True):
        self._running = running

    @property
    def is_running(self):
        return self._running


class FakeWifi:
    def __init__(self, ssid="HomeNetwork"):
        self._ssid = ssid

    def current_ssid(self):
        return self._ssid


@pytest.fixture(autouse=True)
def reset_accel_counters(monkeypatch):
    """Inference counters are module-level; isolate every test from the others."""
    monkeypatch.setattr(accel, "_inference_count", 0, raising=False)
    monkeypatch.setattr(accel, "_last_inference_monotonic", None, raising=False)
    # Reset the per-caller fps baselines. This was a single Optional[tuple]
    # until fps sampling was keyed per caller (the dashboard thread and the
    # inference loop were consuming each other's baseline and reporting 0 fps
    # on a healthy camera); it is a dict now. Same reset, accurate type.
    monkeypatch.setattr(device_health, "_last_fps_sample", {}, raising=False)


# --------------------------------------------------------------------------- camera
def test_camera_ok_when_frames_are_arriving():
    component = device_health._camera(FakeBus(age=0.1), FakeCapture())
    assert component.state == OK
    assert component.metrics["frame_age_s"] == 0.1


def test_camera_fails_when_no_frame_has_ever_arrived():
    assert device_health._camera(FakeBus(age=float("inf")), FakeCapture()).state == FAILED


def test_camera_fails_when_frames_go_stale():
    component = device_health._camera(FakeBus(age=30.0), FakeCapture())
    assert component.state == FAILED
    assert "30.0s" in component.detail


def test_camera_fails_when_capture_service_stopped():
    component = device_health._camera(FakeBus(age=0.1), FakeCapture(running=False))
    assert component.state == FAILED
    assert "stopped" in component.detail


def test_camera_unknown_without_a_bus():
    assert device_health._camera(None, None).state == UNKNOWN


def test_fps_is_measured_between_samples_not_taken_from_config():
    """First sample establishes a baseline and reports no rate."""
    bus = FakeBus(sequence=100)
    assert "fps" not in device_health._camera(bus, FakeCapture()).metrics
    bus.sequence = 160
    assert device_health._camera(bus, FakeCapture()).metrics["fps"] > 0


# --------------------------------------------------------------------------- inference
def test_inference_unknown_before_the_loop_reports():
    assert device_health._inference(None, "occupancy").state == UNKNOWN


def test_inference_ok_when_the_loop_is_ticking():
    component = device_health._inference(0.2, "occupancy")
    assert component.state == OK
    assert "occupancy" in component.detail


def test_inference_fails_when_the_loop_stalls():
    assert device_health._inference(60.0, "occupancy").state == FAILED


# --------------------------------------------------------------------------- accelerator
def test_accelerator_is_not_claimed_without_a_completed_inference(monkeypatch):
    """A seated HAT with a runtime is not proof that anything ran on it."""
    monkeypatch.setattr(accel, "status", lambda: (True, "Hailo in use for object detection (x.hef)"))
    component = device_health._accelerator()
    assert component.state != FAILED
    assert "Hailo-8 accelerator" not in component.detail
    assert "CPU" in component.detail


def test_accelerator_claimed_only_after_a_real_inference(monkeypatch):
    monkeypatch.setattr(accel, "status", lambda: (True, "Hailo in use"))
    accel.record_inference()
    component = device_health._accelerator()
    assert component.state == OK
    assert "Hailo-8" in component.detail
    assert component.metrics["inferences"] == 1


def test_accelerator_goes_stale_when_inference_stops(monkeypatch):
    monkeypatch.setattr(accel, "status", lambda: (True, "Hailo in use"))
    accel.record_inference()
    # Rewind the last-inference stamp well past the liveness window.
    monkeypatch.setattr(accel, "_last_inference_monotonic", -1000.0, raising=False)
    component = device_health._accelerator()
    assert component.state == WARNING
    assert "idle" in component.detail


# --------------------------------------------------------------------------- network
def test_network_reports_the_associated_ssid():
    component = device_health._network(FakeWifi("Cafe WiFi"))
    assert component.state == OK
    assert component.metrics["ssid"] == "Cafe WiFi"


def test_network_fails_when_not_associated():
    assert device_health._network(FakeWifi(None)).state == FAILED


def test_network_never_raises_into_the_caller():
    class Broken:
        def current_ssid(self):
            raise OSError("nmcli exploded")

    assert device_health._network(Broken()).state == UNKNOWN


# --------------------------------------------------------------------------- report
def test_collect_produces_every_component():
    report = device_health.collect(
        bus=FakeBus(), capture=FakeCapture(), wifi=FakeWifi(),
        heartbeat_age_s=0.1, module_name="occupancy", uplink_connected=True,
    )
    assert set(report) >= {"camera", "inference", "accelerator", "network", "system", "reported_at"}
    assert report["cloud_connected"] is True
    assert report["camera"]["state"] == OK


def test_collect_is_json_serialisable():
    """It travels to the backend as JSON on every sync."""
    report = device_health.collect(bus=FakeBus(), capture=FakeCapture(), wifi=FakeWifi())
    assert json.loads(json.dumps(report))["camera"]["state"] == OK


# --------------------------------------------------------------------------- backend
@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Test unit", "AA-BB-CC", "dk_test", _time.time()),
        )
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return app_module, client, headers


HEALTH = {"camera": {"state": "ok", "detail": "capturing", "metrics": {"fps": 12.0}}}


def test_reported_health_reaches_the_app(backend):
    _, client, headers = backend
    client.post("/v1/ingest", json={"events": [], "health": HEALTH}, headers={"X-Device-Key": "dk_test"})

    device = client.get("/v1/devices", headers=headers).json()["devices"][0]
    assert device["health"]["camera"]["state"] == "ok"
    assert device["health"]["camera"]["metrics"]["fps"] == 12.0


def test_health_is_withheld_once_the_device_goes_quiet(backend):
    """A stale report rendered as live is the failure this pipeline prevents."""
    app_module, client, headers = backend
    client.post("/v1/ingest", json={"events": [], "health": HEALTH}, headers={"X-Device-Key": "dk_test"})

    with app_module.db() as conn:  # device last spoke an hour ago
        conn.execute("UPDATE devices SET last_seen = last_seen - 3600 WHERE id = 1")

    device = client.get("/v1/devices", headers=headers).json()["devices"][0]
    assert device["online"] is False
    assert device["health"] is None


def test_a_sync_without_health_keeps_the_previous_report(backend):
    """An older device build that sends no health must not erase what we have."""
    _, client, headers = backend
    client.post("/v1/ingest", json={"events": [], "health": HEALTH}, headers={"X-Device-Key": "dk_test"})
    client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": "dk_test"})

    device = client.get("/v1/devices", headers=headers).json()["devices"][0]
    assert device["health"]["camera"]["state"] == "ok"


def test_devices_without_health_report_none(backend):
    _, client, headers = backend
    client.post("/v1/ingest", json={"events": []}, headers={"X-Device-Key": "dk_test"})
    assert client.get("/v1/devices", headers=headers).json()["devices"][0]["health"] is None
