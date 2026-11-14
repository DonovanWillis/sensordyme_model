"""M10 OBSERVE: always-on per-stage latency and the developer diagnostics endpoint.

Pinned here:
  * core/stage_timing.py is bounded (fixed rings, fixed stage set), its
    percentiles are right, and it is cheap (the overhead is measured);
  * every instrumented stage records from its real call site: HailoYOLO on the
    fake chip (preprocess / inference / decode), the perception engine
    (detector, tracker, face / hands), the GPIO buzzer (command -> pin driven),
    the capture thread and the running inference loop;
  * /api/health carries the timings tagged with the host, so a Mac number is
    labelled "desktop", never "pi";
  * /api/diagnostics: off by default (404 + reason), token-gated (401), refused
    without a configured token even on loopback (403), schema sd-diagnostics/1,
    and no secrets or pixels in the body.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import services.runtime as runtime_module
from core import diagnostics as dx
from core import perception as pc
from core import stage_timing as st
from core.alerts import PATTERNS, AlertManager, GPIOBuzzerSink
from core.config_schema import validate
from core.dashboard import DashboardServer
from modules.base import VisionModule
from services.runtime import DeviceRuntime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import fake_hailo  # noqa: E402

TOKEN = "m10-observe-token"


@pytest.fixture(autouse=True)
def _fresh_timings():
    st.TIMINGS.reset()
    yield
    st.TIMINGS.reset()


# ---------------------------------------------------------------------- the rings
def test_a_stage_keeps_only_its_window_and_reports_correct_percentiles():
    timings = st.StageTimings(window=100)
    for v in range(1, 1001):  # 1000 samples into a 100-sample ring
        timings.add("frame_age", float(v))
    summary = timings.snapshot()["stages"]["frame_age"]
    assert summary["n"] == 100 and summary["count"] == 1000, "the window is bounded, the lifetime count is not"
    assert summary["max_ms"] == 1000.0 and summary["last_ms"] == 1000.0
    assert summary["p50_ms"] == pytest.approx(950.5) and summary["p95_ms"] == pytest.approx(995.05)
    ring = timings._stats["frame_age"]._buf
    assert len(ring) == 100, "the ring is never resized"


def test_unknown_stages_and_bad_values_are_refused_not_added():
    timings = st.StageTimings(window=8)
    assert timings.add("not_a_stage", 1.0) is False
    assert "not_a_stage" not in timings.snapshot()["stages"], "a typo must not grow a new ring"
    for bad in (float("nan"), -1.0, float("inf")):
        assert timings.add("frame_age", bad) is False
    assert timings.snapshot()["stages"]["frame_age"] is None, "no sample yet reads None, never 0"


def test_processed_fps_comes_from_the_frame_interval():
    timings = st.StageTimings(window=16)
    for _ in range(16):
        timings.add("frame_interval", 40.0)
    assert timings.snapshot()["processed_fps"] == 25.0


def test_the_instrumentation_is_cheap():
    """The cost of one add(), measured here, is what docs/DIAGNOSTICS.md quotes.

    About a dozen add() calls per analysed frame: under 20 microseconds each
    keeps the total far under 1% of a 33 ms frame on any machine that runs this
    suite. The number printed is from THIS host (a Mac in CI), not a Pi."""
    timings = st.StageTimings()
    n = 50_000
    started = time.perf_counter()
    for i in range(n):
        timings.add("frame_age", float(i % 50))
    per_add_us = (time.perf_counter() - started) / n * 1e6
    started = time.perf_counter()
    for _ in range(200):
        timings.snapshot()
    snapshot_ms = (time.perf_counter() - started) / 200 * 1e3
    print(f"\nstage_timing on {st.platform_tag()['class']}: add() {per_add_us:.2f} us, snapshot() {snapshot_ms:.3f} ms")
    assert per_add_us < 20.0
    assert snapshot_ms < 20.0


def test_the_platform_tag_never_calls_a_desktop_a_pi(monkeypatch):
    tag = st.platform_tag()
    assert tag["class"] in ("pi", "desktop", "simulated")
    if not tag["is_raspberry_pi"]:
        assert tag["class"] == "desktop" and "NOT a Raspberry Pi" in tag["note"]
    assert st.platform_tag(simulated=True)["class"] == "simulated"
    fake_hailo.install()
    try:
        assert st.platform_tag()["class"] == "simulated", "the fake Hailo runtime makes every figure SIMULATED"
        assert st.platform_tag()["hailo_runtime"] == "simulated"
    finally:
        fake_hailo.uninstall()


def test_the_report_carries_memory_temperature_and_throttle():
    st.TIMINGS.add("frame_interval", 33.0)
    report = st.report(system_metrics={"temperature_c": 61.2, "throttle_now": [], "throttled": "0x0"})
    assert report["schema"] == st.SCHEMA
    assert report["soc_temperature_c"] == 61.2 and report["throttle_now"] == [] and report["throttled_raw"] == "0x0"
    assert report["rss_mb"] is None or report["rss_mb"] > 0
    assert report["rss_source"] in ("current (/proc/self/statm)", "peak (getrusage)", "unavailable")
    assert set(report["stages"]) == set(st.STAGES)


# ---------------------------------------------------------------------- call sites
def test_hailo_predict_records_preprocess_inference_and_decode(monkeypatch):
    from core import accel

    monkeypatch.setattr(accel, "record_inference", lambda: None)
    fake_hailo.install(fake_hailo.planted_backend([(67, 0.8, 200, 150, 260, 260, 32)]))
    try:
        model = accel.HailoYOLO("fake.hef", sha256="0" * 64)
        model.predict(np.zeros((480, 640, 3), np.uint8), conf=0.2)
        model.close()
    finally:
        fake_hailo.uninstall()
    for stage in ("hailo_preprocess", "hailo_inference", "decoder_cpu"):
        assert st.TIMINGS.count(stage) == 1, stage


class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = [cls], [conf], [np.asarray(xyxy, float)]


class _Detector:
    names: ClassVar[dict] = {0: "person", 67: "cell phone"}

    def predict(self, frame, verbose=False, classes=None, conf=0.25, **_kw):
        time.sleep(0.002)
        return [types.SimpleNamespace(boxes=[_Box(0, 0.9, (10, 10, 50, 90)), _Box(67, 0.6, (20, 50, 30, 60))])]


class _Graph:
    def process(self, image):
        lm = types.SimpleNamespace(landmark=[types.SimpleNamespace(x=0.5, y=0.5)] * 478)
        return types.SimpleNamespace(multi_face_landmarks=[lm], multi_hand_landmarks=[lm])


def test_the_perception_engine_records_models_and_tracker_and_sums_model_time():
    engine = pc.PerceptionEngine()
    det = engine.wrap_detector(_Detector())
    face = engine.wrap_graph(pc.FACE_LANDMARKS, _Graph())
    hands = engine.wrap_graph(pc.HAND_LANDMARKS, _Graph())
    frame = np.zeros((100, 120, 3), np.uint8)
    engine.begin_frame(frame, frame_id=1)
    det.predict(frame, classes=[0, 67], conf=0.2)
    det.predict(frame, classes=[0], conf=0.5)  # served from the same run: not a second sample
    face.process(frame.copy())
    hands.process(frame.copy())
    det.predict(frame[:50, :50], conf=0.2)  # a crop: passes through, still model time
    engine.end_frame()
    for stage in ("detector_total", "tracking", "face_landmarks", "hands"):
        assert st.TIMINGS.count(stage) == 1, stage
    assert engine.frame_model_ms >= 4.0, "both detector runs (full frame + crop) count as model time"
    engine.begin_frame(frame, frame_id=2)
    assert engine.frame_model_ms == 0.0, "reset per frame"


def test_the_gpio_buzzer_records_command_to_pin_driven():
    sink = GPIOBuzzerSink.__new__(GPIOBuzzerSink)
    sink._lock, sink._off_at, sink._last_error, sink._pin = threading.Lock(), 0.0, None, 17
    sink._buzzer = types.SimpleNamespace(on=lambda: None, off=lambda: None)
    done = threading.Event()
    pattern = PATTERNS["notice_chirp"] if "notice_chirp" in PATTERNS else next(iter(PATTERNS.values()))
    sink.play(pattern, None, types.SimpleNamespace(name="warning"), report=lambda *_: done.set())  # type: ignore[arg-type]
    assert done.wait(10)
    assert st.TIMINGS.count("alert_output") == 1, "one sample per pattern (its first pulse), not per pulse"
    summary = st.TIMINGS.snapshot()["stages"]["alert_output"]
    assert summary["last_ms"] < 1000.0


# ---------------------------------------------------------------------- config
def test_the_switch_is_a_bool_and_off_in_the_repo():
    import yaml

    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert config["dashboard"]["developer_diagnostics"] is False, "diagnostics ship OFF"
    errors, _ = validate({**config, "dashboard": {**config["dashboard"], "developer_diagnostics": "yes"}})
    assert any("developer_diagnostics" in e for e in errors), "a truthy string must not switch it on"
    errors, warnings = validate({**config, "dashboard": {**config["dashboard"], "developer_diagnostics": True}})
    assert not any("developer_diagnostics" in e for e in errors)
    assert any("developer_diagnostics" in w and "token" in w for w in warnings)


# ---------------------------------------------------------------------- sanitizing
def test_sanitize_drops_pixels_and_secrets_at_any_depth():
    raw = {
        "ok": 1.23456789, "frame": np.zeros((4, 4, 3), np.uint8), "nested": {
            "dashboard_token": "abc", "wifi_ssid": "Home", "psk": "pw", "device_key": "k",
            "image": b"\xff\xd8", "embedding": np.ones(5), "score": np.float32(0.5),
            "deeper": {"Authorization": "Bearer x", "label": "cell phone"}},
        "many": list(range(100)), "nan": float("nan"),
    }
    out = dx.sanitize(raw)
    text = json.dumps(out)
    for leaked in ("abc", "Home", "pw", "Bearer", "dashboard_token", "ssid", "psk", "device_key"):
        assert leaked not in text, leaked
    assert "frame" not in out and "image" not in out["nested"]
    assert out["nested"]["embedding"] is None, "an array is never carried"
    assert out["nested"]["score"] == 0.5 and out["nested"]["deeper"] == {"label": "cell phone"}
    assert len(out["many"]) == 32 and out["nan"] is None and out["ok"] == 1.2346


# ---------------------------------------------------------------------- the dashboard gate
def _get(url, token=None):
    req = urllib.request.Request(url, headers={"X-Device-Token": token} if token else {})  # noqa: S310 - local
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - local test server
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


@pytest.mark.parametrize(("enabled", "token", "sent", "status"), [
    (False, TOKEN, TOKEN, 404),   # the switch is off: not found, with the reason
    (True, TOKEN, None, 401),     # no token sent: refused like every /api path
    (True, TOKEN, "wrong", 401),  # a wrong token: refused
    (True, None, None, 403),      # loopback, no token configured: still refused
    (True, TOKEN, TOKEN, 200),
])
def test_the_endpoint_is_off_by_default_and_never_served_without_the_token(enabled, token, sent, status):
    calls = []
    server = DashboardServer(host="127.0.0.1", port=0, token=token, diagnostics_enabled=enabled,
                             diagnostics_provider=lambda: calls.append(1) or {"schema": dx.SCHEMA})
    url = server.start()
    try:
        got, body = _get(url + "/api/diagnostics", sent)
    finally:
        server.close()
    assert got == status, body
    if status == 404:
        assert "developer_diagnostics" in body["error"] and body["enabled"] is False
    if status == 403:
        assert "dashboard.token" in body["error"]
    assert bool(calls) == (status == 200), "the provider runs only for an authorised, enabled request"


def test_a_failing_provider_is_a_fixed_sentence_not_a_traceback():
    def boom():
        raise RuntimeError("secret internal state")

    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN, diagnostics_enabled=True,
                             diagnostics_provider=boom)
    url = server.start()
    try:
        got, body = _get(url + "/api/diagnostics", TOKEN)
    finally:
        server.close()
    assert got == 500 and "secret" not in json.dumps(body)


# ---------------------------------------------------------------------- the running runtime
class FakeCamera:
    is_pi = False
    fps = 30.0

    def __init__(self, config):
        self.n = 0

    def read(self, timeout_s=None):
        self.n += 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[0, 0, 0] = self.n % 256
        return frame

    def reopen(self):
        pass

    def release(self):
        pass


class DebugModule(VisionModule):
    """Reports a debug state with a developing condition, an array and a secret-looking key."""

    name = "debuggy"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        time.sleep(0.001)
        return []

    def get_debug_state(self):
        return {
            "state": "DISTRACTED", "gaze": "Left",
            "distraction_state": "DEVELOPING", "distraction_duration_s": 1.4,
            "phone_use_state": "NONE", "phone_use_duration_s": 0.0,
            "phone_detected": True, "phone_confidence": 0.61, "phone_hold": "unknown",
            "hand_overlaps_phone": False,
            "engine": {"driver": {"method": "largest_face", "known": True},
                       "phone_evidence": {"state": "visible"}, "abstaining": {"phone_use": "no hand"}},
            "events_fired_this_frame": [],
            "raw_frame": np.zeros((48, 64, 3), np.uint8),
            "api_token": "should-never-leave",
        }


def _runtime(tmp_path, *, diagnostics: bool, token=TOKEN):
    config = {
        "module": "debuggy",
        "camera": {"width": 64, "height": 48, "fps": 30},
        "dashboard": {"enabled": True, "host": "127.0.0.1", "port": 0, "token": token,
                      "developer_diagnostics": diagnostics},
        "cloud": {"enabled": False},
        "accelerator": {"enabled": True, "yolo_hef": None, "yolo_hef_sha256": "ab" * 32},
        "network": {"provisioning": {"method": "off"}},
        "health": {"check_interval_s": 0.5, "allow_reboot": False},
        "logging": {"csv_path": str(tmp_path / "events.csv"), "telemetry_csv_path": str(tmp_path / "t.csv")},
        "modules": {"debuggy": {}},
    }
    module = DebugModule({})
    runtime = DeviceRuntime(
        config=config, profile="pi", module=module, module_name="debuggy",
        alert_manager=AlertManager(sinks=[]), alert_cfg={}, alert_event_types=set(),
        alert_types_for=lambda n, preset=None: set(), presets={}, active_preset=None,
        display=False, debug=False,
    )
    return runtime, module


class _Running:
    def __init__(self, runtime, module):
        self.runtime, self.module = runtime, module
        self.thread = threading.Thread(target=runtime.run)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and self.module.frames < 5:
            time.sleep(0.05)
        assert self.module.frames >= 5, "the runtime never started processing"
        return f"http://127.0.0.1:{self.runtime.dashboard._port}"

    def __exit__(self, *exc):
        self.runtime.shutdown.set()
        self.thread.join(timeout=10)


def test_health_carries_the_running_loops_stage_timings_tagged_with_the_host(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    runtime, module = _runtime(tmp_path, diagnostics=False)
    with _Running(runtime, module) as base:
        status, health = _get(base + "/api/health", TOKEN)
        got, body = _get(base + "/api/diagnostics", TOKEN)
    assert status == 200
    perf = health["performance"]
    assert perf["schema"] == st.SCHEMA and perf["platform"]["class"] in ("desktop", "pi", "simulated")
    for stage in ("camera_read", "frame_age", "behaviour_reasoning", "alert_dispatch", "frame_to_decision",
                  "frame_interval"):
        assert perf["stages"][stage] is not None, f"{stage} has no sample from the running loop"
    assert perf["stages"]["behaviour_reasoning"]["p50_ms"] >= 0.5, "process() time is the module's own reasoning"
    assert perf["processed_fps"] is not None and perf["processed_fps"] > 0
    assert health["diagnostics"] == {"enabled": False, "schema": dx.SCHEMA}
    assert got == 404 and body["enabled"] is False, "OFF by default: not found, with the reason"


def test_diagnostics_end_to_end_schema_contents_and_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    runtime, module = _runtime(tmp_path, diagnostics=True)
    with _Running(runtime, module) as base:
        got, _ = _get(base + "/api/diagnostics")
        assert got == 401
        _, health = _get(base + "/api/health", TOKEN)
        body = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # the first request switches the snapshot on
            got, body = _get(base + "/api/diagnostics", TOKEN)
            assert got == 200, body
            if body["frame"] is not None:
                break
            time.sleep(0.2)
    assert health["diagnostics"]["enabled"] is True
    assert body is not None and body["schema"] == "sd-diagnostics/1" and body["enabled"] is True
    frame = body["frame"]
    assert frame is not None and frame["module"] == "debuggy"
    assert frame["detections"]["items"] is None and frame["detections"]["state"], "no detector: said, not empty"
    assert frame["evidence"]["driver_association"] == {"method": "largest_face", "known": True}
    assert frame["evidence"]["hand_phone"]["phone_confidence"] == 0.61
    conditions = {c["condition"]: c for c in frame["open_conditions"]}
    assert conditions["distraction"]["state"] == "developing" and conditions["distraction"]["running_s"] == 1.4
    assert "phone_use" not in conditions, "NONE is not an open condition"
    assert body["versions"]["decoder"] == "legacy" and body["versions"]["detector_pin_sha256"] == "ab" * 32
    assert "mediapipe_face_mesh" in body["versions"]["mediapipe"]
    assert body["latency"]["schema"] == st.SCHEMA and body["platform"]["class"] in ("desktop", "pi", "simulated")
    assert isinstance(body["unknown_or_degraded"], list)
    assert "available" in body["capabilities"]
    text = json.dumps(body)
    for leaked in (TOKEN, "should-never-leave", "api_token", "raw_frame"):
        assert leaked not in text, leaked


def test_diagnostics_cost_nothing_on_the_loop_until_someone_asks(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    runtime, module = _runtime(tmp_path, diagnostics=True)
    calls = []
    real = module.get_debug_state
    module.get_debug_state = lambda: calls.append(1) or real()  # type: ignore[method-assign]
    with _Running(runtime, module):
        time.sleep(0.5)
    assert calls == [], "no request, no snapshot: the module's debug state is never built"
