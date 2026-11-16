"""M13 sd-overlay/1: the clean frame plus what the AI saw on it (core/overlay.py).

Pinned here: only real observations; no confidence numbers; "could not look"
(None) is never shown as "nothing there" ([]); a simulated source says so on
every frame; head pose is labelled head pose; a viewer that asks gets clean
frames with the analysis, and every other viewer gets exactly what it got
before.
"""

from __future__ import annotations

import json

import numpy as np
from test_m83_stream_runtime import TOKEN, FakeCamera, _Running, _runtime

import services.runtime as runtime_module
from core import overlay as ov
from core import perception as pc
from modules.base import VisionModule
from sim.stream_client import StreamClient


def _status(state="ok", run_ms=12.5):
    return pc.ComponentStatus(name=pc.OBJECT_DETECTOR, state=state, run_ms=run_ms)


def _obs(detections=(), face_visible=None, face=None, sim=False, components=None, hands=None):
    return pc.Observation(
        frame_id=7, captured_at=100.0, source="sim" if sim else "camera", simulated=sim, width=640, height=480,
        detections=detections, detections_age_s=0.0 if detections is not None else None,
        fresh_detections=detections is not None, detection_consumers=frozenset(), face_visible=face_visible,
        face=face, hands=hands,
        components=components or {pc.OBJECT_DETECTOR: _status(), pc.FACE_LANDMARKS: pc.ComponentStatus(
            name=pc.FACE_LANDMARKS, state="ok", run_ms=8.0)})


def _walk_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk_keys(v)


def test_boxes_are_normalised_tracked_and_carry_no_confidence():
    person = pc.ObjectDetection("person", 0.91, (64.0, 48.0, 320.0, 480.0), class_id=0, track_id=3)
    phone = pc.ObjectDetection("cell phone", 0.42, (200.0, 200.0, 240.0, 260.0), class_id=67, track_id=9)
    doc = ov.OverlayBuilder(clock=lambda: 100.05).build(
        _obs((person, phone)), {"driver": {"box": (64, 48, 320, 480), "method": "configured_seat",
                                           "identity": "driver", "people": 1},
                               "driver_phone_boxes": [(200, 200, 240, 260)]}, module="driver_attention")
    assert doc["schema"] == "sd-overlay/1" and doc["sim"] is False and doc["analysed"] is True
    labels = {d["label"]: d for d in doc["det"]}
    assert labels["person"] == {"label": "person", "box": [0.1, 0.1, 0.5, 1.0], "track": 3, "role": "driver"}
    assert labels["phone"]["role"] == "driver"
    assert "confidence" not in set(_walk_keys(doc))
    assert doc["perf"]["object_detector_ms"] == 12.5 and doc["perf"]["frame_age_ms"] == 50


def test_unavailable_detection_is_none_not_empty_and_is_listed():
    doc = ov.OverlayBuilder().build(_obs(None, components={
        pc.OBJECT_DETECTOR: _status("failed", None),
        pc.FACE_LANDMARKS: pc.ComponentStatus(name=pc.FACE_LANDMARKS, state="idle")}))
    assert doc["det"] is None
    assert doc["unavailable"] == [pc.OBJECT_DETECTOR]  # idle (not in use) is not "unavailable"


def test_face_not_looked_for_vs_not_visible_vs_measured():
    assert ov.OverlayBuilder().build(_obs(face_visible=None))["face"] is None
    assert ov.OverlayBuilder().build(_obs(face_visible=False))["face"] == {"visible": False}
    face = pc.FaceObservation(468, eye_state=pc.EyeState(ear=0.12, closed=True),
                              head_pose=pc.HeadPose(31.0, -4.0, True))
    doc = ov.OverlayBuilder().build(_obs(face_visible=True, face=face))
    assert doc["face"]["eyes_closed"] is True
    assert doc["face"]["head_pose"] == {"yaw_deg": 31.0, "pitch_deg": -4.0, "calibrated": True, "kind": "head_pose"}
    assert "gaze" not in json.dumps(doc)


def test_a_simulated_source_says_so_and_no_observation_says_not_analysed():
    assert ov.OverlayBuilder().build(_obs(sim=True))["sim"] is True
    doc = ov.OverlayBuilder().build(None)
    assert doc["analysed"] is False and doc["det"] is None


def test_processed_fps_is_measured_from_analysed_frames():
    builder = ov.OverlayBuilder()
    for i in range(30):
        builder.note_frame(i * 0.1)  # 10 fps
    assert ov.OverlayBuilder().build(_obs())["perf"]["fps"] is None
    assert 9.5 <= builder.build(_obs())["perf"]["fps"] <= 10.5


def test_the_header_round_trips_and_stays_small():
    face = pc.FaceObservation(468, eye_state=pc.EyeState(closed=False), head_pose=pc.HeadPose(1.0, 2.0, False))
    contours = {k: [(0.123, 0.456)] * n for k, n in (("oval", 36), ("eye_l", 16), ("eye_r", 16))}
    dets = tuple(pc.ObjectDetection("person", 0.9, (1, 2, 3, 4), track_id=i) for i in range(5))
    doc = ov.OverlayBuilder().build(_obs(dets, face_visible=True, face=face), {"face_contours": contours})
    header = ov.encode_header(doc)
    assert "\n" not in header and "=" not in header
    assert ov.decode_header(header) == json.loads(json.dumps(doc))
    assert len(header) < 4096


def test_an_uncalibrated_raw_pitch_is_shown_in_range_and_a_calibrated_one_unchanged():
    """The raw Euler fit puts an upright head near +-180 deg; the overlay shows ~0."""
    def pitch(p, calibrated, extras=False):
        face = pc.FaceObservation(468, eye_state=pc.EyeState(closed=False),
                                  head_pose=pc.HeadPose(3.0, p, calibrated))
        ex = {"head": {"yaw_deg": 3.0, "pitch_deg": p, "calibrated": calibrated, "direction": "Forward"}} if extras else {}
        return ov.OverlayBuilder().build(_obs(face_visible=True, face=face), ex)["face"]["head_pose"]["pitch_deg"]

    for via_module in (False, True):
        assert pitch(178.0, False, via_module) == -2.0
        assert pitch(-176.5, False, via_module) == 3.5
        assert pitch(-12.0, False, via_module) == -12.0
        assert pitch(-4.0, True, via_module) == -4.0


def test_the_last_event_carries_its_history_event_key():
    from core.dashboard import DashboardServer
    from core.events import Event

    server = DashboardServer(host="127.0.0.1", port=0, module_name="driver_attention")
    event = Event.now("distraction", 2.0, None, "driver_attention", "turned 40 deg left of forward + held 2.0 s")
    server.push_event(event, "loud")
    summary = server.last_event_summary()
    assert isinstance(event.event_id, str) and summary["event_key"] == event.event_id
    assert summary["signals"].startswith("turned 40 deg left")


# ------------------------------------------------------------------ through the running runtime
class OverlayModule(VisionModule):
    """Draws a white square when annotating: an overlay viewer must never see it."""

    name = "overlay_probe"

    def __init__(self, config):
        super().__init__(config)
        self.frames = 0

    def process(self, frame):
        self.frames += 1
        return []

    def annotate(self, frame):
        frame[:, :] = 255
        return frame

    def overlay_extras(self):
        return {"state": "FOCUSED", "behaviours": [{"type": "distraction", "phase": "developing", "for_s": 1.2}]}


def _decode_jpeg(body: bytes):
    import cv2

    return cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_an_overlay_viewer_gets_clean_frames_with_the_analysis_and_others_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = OverlayModule({})
    runtime = _runtime(tmp_path, module, "overlay_probe")
    port = None
    with _Running(runtime, module):
        port = runtime.dashboard._port
        with StreamClient(port, TOKEN, overlay=True) as live:
            assert live.read_until(lambda c: sum(1 for p in c.jpeg_parts() if "x-sd-overlay" in p.headers) >= 5,
                                   timeout=8)
        assert live.headers.get("x-stream-overlay") == "sd-overlay/1"
        analysed = [p for p in live.jpeg_parts() if "x-sd-overlay" in p.headers]
        for part in analysed:
            doc = ov.decode_header(part.headers["x-sd-overlay"])
            assert doc["schema"] == "sd-overlay/1" and doc["module"] == "overlay_probe"
            assert doc["behaviours"] == [{"type": "distraction", "phase": "developing", "for_s": 1.2}]
            assert doc["w"] == 64 and doc["h"] == 48
            image = _decode_jpeg(part.body)
            assert image.mean() < 200, "the overlay viewer was sent the device-drawn (annotated) picture"
        with StreamClient(port, TOKEN) as legacy:
            assert legacy.read_until(lambda c: any(p.headers.get("x-frame-source") == "annotated"
                                                   for p in c.jpeg_parts()), timeout=8)
        drawn = [p for p in legacy.jpeg_parts() if p.headers.get("x-frame-source") == "annotated"]
        assert all("x-sd-overlay" not in p.headers for p in legacy.jpeg_parts())
        assert _decode_jpeg(drawn[-1].body).mean() > 240, "an ordinary viewer still gets the device-drawn view"


# ------------------------------------------------------------------ the real driver module's half
def test_the_driver_modules_extras_are_what_it_measured(tmp_path, monkeypatch):
    from m10_temporal_harness import Seg, face, run

    turned = run(tmp_path, monkeypatch, [Seg(1.0, face()), Seg(4.0, face(yaw=40.0))])
    extras = turned.module.overlay_extras()
    assert extras["head"]["direction"] in ("Left", "Right") and extras["head"]["calibrated"] is True
    assert {"oval", "eye_l", "eye_r"} <= set(extras["face_contours"])
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in extras["face_contours"]["oval"])
    distraction = [b for b in extras["behaviours"] if b["type"] == "distraction"]
    assert distraction and distraction[0]["phase"] == "confirmed" and distraction[0]["for_s"] >= 3.0
    assert extras["driver"]["box"] is not None

    gone = run(tmp_path, monkeypatch, [Seg(1.0, face()), Seg(6.0, None)])
    extras = gone.module.overlay_extras()
    assert "face_contours" not in extras and "head" not in extras
    assert [b["type"] for b in extras["behaviours"]] in (["face_not_visible"], [])

    attentive = run(tmp_path, monkeypatch, [Seg(3.0, face())])
    assert attentive.module.overlay_extras()["behaviours"] == []


def test_the_demo_preflight_samples_the_live_ai_view(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "sensordyme_demo_preflight_m13", Path(__file__).resolve().parents[1] / "scripts" / "demo_preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = preflight
    spec.loader.exec_module(preflight)
    monkeypatch.setattr(runtime_module, "Camera", FakeCamera)
    module = OverlayModule({})
    runtime = _runtime(tmp_path, module, "overlay_probe")
    with _Running(runtime, module):
        sample = preflight._overlay_sample("127.0.0.1", runtime.dashboard._port, TOKEN, seconds=3.0)
        refused = preflight._overlay_sample("127.0.0.1", runtime.dashboard._port, "wrong", seconds=1.0)
    assert sample["analysed"] >= 3 and sample["last"]["schema"] == "sd-overlay/1"
    assert sample["last"]["perf"]["fps"] is not None
    assert refused["frames"] == 0 and refused["analysed"] == 0
