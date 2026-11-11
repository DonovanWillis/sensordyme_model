"""The shared perception engine (core/perception.py): once per frame, typed, never fabricated."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from core import perception as pc


class _Box:
    def __init__(self, cls: int, conf: float, xyxy) -> None:
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class _Result:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeDetector:
    names: ClassVar[dict] = {0: "person", 39: "bottle", 67: "cell phone"}
    model_name = "fake-detector"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail = False

    def predict(self, frame, verbose=False, classes=None, conf=0.25, **_kw):
        self.calls.append({"classes": classes, "conf": conf, "shape": frame.shape})
        if self.fail:
            raise RuntimeError("NPU stream aborted")
        boxes = [_Box(0, 0.9, (10, 10, 50, 90)), _Box(0, 0.3, (60, 10, 90, 90)),
                 _Box(67, 0.4, (20, 50, 30, 60)), _Box(39, 0.6, (70, 70, 80, 80))]
        keep = [b for b in boxes if (classes is None or b.cls[0] in classes) and b.conf[0] >= conf]
        return [_Result(keep)]


class FakeGraph:
    def __init__(self, faces: int = 1) -> None:
        self.calls = 0
        self.faces = faces
        self.fail = False

    def process(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("graph crashed")
        lm = type("LM", (), {"landmark": [type("P", (), {"x": 0.5, "y": 0.5})()] * 478})()
        return type("R", (), {"multi_face_landmarks": [lm] * self.faces or None,
                              "multi_hand_landmarks": [lm] if self.faces else None})()


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def engine(clock):
    return pc.PerceptionEngine(max_age_s=1.0, clock=clock)


FRAME = np.zeros((100, 120, 3), dtype=np.uint8)


def _frame(engine, i, frame=None):
    frame = FRAME.copy() if frame is None else frame
    engine.begin_frame(frame, frame_id=i, source="sim", simulated=True)
    return frame


def test_two_callers_one_inference_each_filtered_to_its_request(engine):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    assert engine.wrap_detector(raw) is det
    frame = _frame(engine, 1)
    with engine.consuming("driver_attention"):
        a = det.predict(frame, classes=[0, 67], conf=0.2)
    with engine.consuming("rules"):
        b = det.predict(frame, classes=[0], conf=0.5)
    assert len(raw.calls) == 1, "the second caller is answered from this frame's result"
    assert sorted(int(x.cls[0]) for x in a[0].boxes) == [0, 0, 67]
    assert [float(x.conf[0]) for x in b[0].boxes] == [0.9]
    obs = engine.end_frame()
    assert obs.fresh_detections and obs.detection_consumers == {"driver_attention", "rules"}
    assert {d.label for d in obs.detections} == {"person", "cell phone"}
    assert obs.components[pc.OBJECT_DETECTOR].state == pc.OK
    assert obs.components[pc.OBJECT_DETECTOR].consumers == ("driver_attention", "rules")


def test_learned_classes_cover_later_frames_in_one_run(engine):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    frame = _frame(engine, 1)
    det.predict(frame, classes=[67], conf=0.2)
    det.predict(frame, classes=[0], conf=0.5)  # widens the request: a second run, once
    engine.end_frame()
    for i in range(2, 6):
        frame = _frame(engine, i)
        det.predict(frame, classes=[67], conf=0.2)
        det.predict(frame, classes=[0], conf=0.5)
        engine.end_frame()
    assert len(raw.calls) == 2 + 4
    assert raw.calls[-1]["classes"] == [0, 67] and raw.calls[-1]["conf"] == 0.2


def test_a_crop_or_a_frame_outside_a_pass_goes_straight_through(engine):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    det.predict(FRAME, classes=[0])  # no frame in flight (warm-up)
    frame = _frame(engine, 1)
    det.predict(frame[0:50, 0:50], classes=[0])  # a crop
    det.predict(frame, classes=[0])
    det.predict(frame[0:50, 0:50], classes=[0])
    assert len(raw.calls) == 4 and det.runs == 1 and det.passthrough_runs == 3


def test_a_failure_is_raised_to_every_caller_once_and_clears_the_boxes(engine, clock):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    frame = _frame(engine, 1)
    det.predict(frame, classes=[0])
    assert engine.end_frame().detections
    raw.fail = True
    clock.t += 0.1
    frame = _frame(engine, 2)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="NPU"):
            det.predict(frame, classes=[0])
    assert len(raw.calls) == 2, "one failing attempt per frame, not one per caller"
    obs = engine.end_frame()
    assert obs.detections is None, "a frozen box from before the failure is not carried"
    status = obs.components[pc.OBJECT_DETECTOR]
    assert status.state == pc.FAILED and "NPU" in status.reason and status.errors == 1
    raw.fail = False
    clock.t += 0.1
    frame = _frame(engine, 3)
    det.predict(frame, classes=[0])
    assert engine.end_frame().components[pc.OBJECT_DETECTOR].state == pc.OK


def test_between_runs_detections_carry_until_stale_then_none(engine, clock):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    with engine.consuming("m"):
        det.predict(_frame(engine, 1), classes=[0])
    assert engine.end_frame().fresh_detections
    clock.t += 0.5
    _frame(engine, 2)
    obs = engine.end_frame()
    assert obs.detections is not None and not obs.fresh_detections and obs.detections_age_s == 0.5
    clock.t += 1.0
    _frame(engine, 3)
    obs = engine.end_frame()
    assert obs.detections is None and obs.count("person") is None
    assert obs.detection_frame() is None
    status = obs.components[pc.OBJECT_DETECTOR]
    assert status.state == pc.STALE and "no result for 1.5s" in status.reason


def test_graph_runs_once_per_frame_and_face_is_none_when_not_run(engine):
    graph = FakeGraph()
    face = engine.wrap_graph(pc.FACE_LANDMARKS, graph, model={"name": "fake-face"})
    _frame(engine, 1)
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    face.process(rgb)
    face.process(rgb.copy())
    signals = {"eye_state": pc.EyeState(ear=0.31, iris_openness=0.8, closed=False),
               "head_pose": pc.HeadPose(yaw_deg=3.0, pitch_deg=-1.0, calibrated=True), "gaze": None}
    obs = engine.end_frame(signals)
    assert graph.calls == 1
    assert obs.face_visible is True and obs.face.landmark_count == 478
    assert obs.face.eye_state.ear == 0.31 and obs.face.head_pose.calibrated and obs.face.gaze is None
    assert obs.components[pc.FACE_LANDMARKS].model == {"name": "fake-face"}
    _frame(engine, 2)
    obs = engine.end_frame(signals)
    assert obs.face_visible is None and obs.face is None, "not run this frame: unknown, not 'no face'"
    graph.faces = 0
    _frame(engine, 3)
    face.process(rgb)
    obs = engine.end_frame(signals)
    assert obs.face_visible is False and obs.face is None


def test_hands_are_boxes_from_landmarks(engine):
    hands = engine.wrap_graph(pc.HAND_LANDMARKS, FakeGraph())
    _frame(engine, 1)
    hands.process(np.zeros((100, 120, 3), dtype=np.uint8))
    obs = engine.end_frame()
    assert obs.hands == (pc.HandObservation(box=(60.0, 50.0, 60.0, 50.0), landmark_count=478),)


def test_demands_run_the_detector_only_when_due_and_not_already_run(engine, clock):
    raw = FakeDetector()
    det = engine.wrap_detector(raw)
    engine.demand(pc.Demand("occupancy", pc.OBJECT_DETECTOR, frozenset({"person"}), min_interval_s=0.5))
    frame = _frame(engine, 1)
    engine.run_due(frame)
    with engine.consuming("driver_attention"):
        det.predict(frame, classes=[0], conf=0.25)  # same frame: answered from the demand's run
    obs = engine.end_frame()
    assert len(raw.calls) == 1 and obs.detection_consumers == {"occupancy", "driver_attention"}
    clock.t += 0.2
    engine.run_due(_frame(engine, 2))
    engine.end_frame()
    assert len(raw.calls) == 1, "not due yet"
    clock.t += 0.4
    engine.run_due(_frame(engine, 3))
    assert engine.end_frame().fresh_detections and len(raw.calls) == 2
    engine.withdraw("occupancy")
    clock.t += 1.0
    engine.run_due(_frame(engine, 4))
    engine.end_frame()
    assert len(raw.calls) == 2, "a withdrawn demand runs nothing"


def test_demand_without_any_detector_loads_one_or_reports_unavailable(engine):
    engine.demand(pc.Demand("occupancy", pc.OBJECT_DETECTOR, frozenset({"person"})))

    def broken():
        raise RuntimeError("no backend")

    engine.set_detector_loader(broken)
    engine.run_due(_frame(engine, 1))
    status = engine.end_frame().components[pc.OBJECT_DETECTOR]
    assert status.state == pc.UNAVAILABLE and "no backend" in status.reason
    raw = FakeDetector()
    engine.set_detector_loader(lambda: raw)
    engine.set_unavailable(pc.OBJECT_DETECTOR, None)
    engine.run_due(_frame(engine, 2))
    assert engine.end_frame().components[pc.OBJECT_DETECTOR].state == pc.OK and len(raw.calls) == 1


def test_a_failing_demand_run_never_raises_into_the_pass(engine):
    raw = FakeDetector()
    raw.fail = True
    engine.wrap_detector(raw)
    engine.demand(pc.Demand("occupancy", pc.OBJECT_DETECTOR, frozenset({"person"})))
    engine.run_due(_frame(engine, 1))
    assert engine.end_frame().components[pc.OBJECT_DETECTOR].state == pc.FAILED


def test_factories_wrap_only_inside_an_activated_engine(engine):
    raw = FakeDetector()
    assert pc.shared_detector(raw) is raw
    with engine.activated():
        assert isinstance(pc.shared_detector(raw), pc.SharedDetector)
        assert isinstance(pc.shared_graph(pc.POSE, FakeGraph()), pc.SharedGraph)
    assert pc.current() is None
    assert pc.unwrap(engine.wrap_detector(raw)) is raw


def test_adopt_wraps_a_modules_full_frame_models():
    engine = pc.PerceptionEngine()
    module = type("M", (), {})()
    module._yolo_model = FakeDetector()
    module._face_mesh = FakeGraph()
    module._pose_verifier = FakeGraph()  # crops only: never adopted
    engine.adopt(module)
    assert isinstance(module._yolo_model, pc.SharedDetector)
    assert isinstance(module._face_mesh, pc.SharedGraph)
    assert not isinstance(module._pose_verifier, pc.SharedGraph)


def test_wire_format(engine):
    wire = engine.to_wire(simulated=True, nodes=[{"id": "driver_safety"}])
    assert wire["schema"] == "sd-perception/1" and wire["simulated"] is True
    assert set(wire["components"]) == set(pc.COMPONENTS)
    assert wire["components"]["pose"] == {"state": "idle", "model": None, "last_run_age_s": None, "runs": 0,
                                          "errors": 0, "reason": None, "consumers": [], "run_ms": None}
    assert wire["nodes"] == [{"id": "driver_safety"}]


def test_detection_frame_applies_per_label_floors(engine):
    det = engine.wrap_detector(FakeDetector())
    det.predict(_frame(engine, 1), classes=None, conf=0.2)
    obs = engine.end_frame()
    frame = obs.detection_frame({"person": 0.5, "*": 0.35})
    assert sorted(d.label for d in frame.detections) == ["bottle", "cell phone", "person"]
    assert frame.count("person") == 1
    assert obs.count("person", 0.0) == 2


def test_memory_is_bounded(engine):
    for _ in range(100):
        engine.wrap_detector(FakeDetector())
    assert len(engine._wrapped) <= 32
    det = engine.wrap_detector(FakeDetector())
    for i in range(200):
        with engine.consuming(f"c{i}"):
            det.predict(_frame(engine, i), classes=[0])
        engine.end_frame()
    assert len(engine._components[pc.OBJECT_DETECTOR].consumers) <= 32
