"""The shared perception engine: one camera frame, each model at most once, typed observations.

Before this, every VisionModule owned its own models. Weights were shared
(core/detectors.py caches them), results were not: a second consumer of person
detection — a custom rule, an occupancy count — would have re-run the detector
on the same frame, and every module built its own MediaPipe graphs. This module
sits at the call sites the modules already use and makes each model run at most
once per frame, whoever asks:

    SharedDetector   wraps the detector get_yolo_model() hands out (ultralytics
                     or core.accel.HailoYOLO). The first predict() on a frame
                     runs the model with the UNION of the classes callers have
                     asked for and the LOWEST confidence any of them asked for;
                     every later predict() on the same frame is answered from
                     that result, filtered to what that caller asked for. A
                     crop (core.detectors.predict_roi) is not the frame and
                     passes straight through.
    SharedGraph      the same for a MediaPipe graph (face landmarks, hands,
                     pose): one process() per frame, keyed on the frame.

Modules call `.predict` / `.process` exactly as before; nothing else changes for
them. Consumers that own no model (monitoring nodes, core/nodes.py) declare a
DEMAND instead — component, labels, how often — and the engine runs the
component on a frame only when a demand is due and nobody has run it yet.

Each frame ends in one typed `Observation`: what was detected, the face and
hands when their models ran, signals the module computed (eye state, head
pose, gaze) only when it actually computed them, and the status of every
component. Nothing is ever fabricated: a component that did not run, failed or
went stale reports None and says why, never an empty "nothing there". "None" and
"nothing seen" are different facts and the types keep them apart.

Component states (the wire contract in docs/M7_IMPLEMENTATION_MAP.md):

  ok           produced a result within max_age_s
  stale        someone is consuming it, but the last result is older than max_age_s
  failed       the last attempt raised; the error is the reason
  unavailable  it cannot run on this device (no backend, refused load)
  idle         nobody has needed it

Tracking (M8): the engine also owns the device's ONE object tracker
(core/tracking.py ObjectTracker). Every detector run's boxes are associated
once, right where the run is parsed, so each ObjectDetection carries a
persistent `track_id` and each Observation carries the live `tracks` (visible
and coasting, with smoothed confidence and confirmation). Nodes and custom
rules read those ids; none of them runs a tracker of its own. The tracker's
counters and mean update time are reported as `perception.tracking`.

Pure Python and the standard library (plus core.detections for the rule view
and core.tracking): no model is imported here, so the backend, tests and the
simulator can use it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from core.detections import Detection, DetectionFrame
from core.tracking import ObjectTracker, Track, box_iou

SCHEMA = "sd-perception/1"

OBJECT_DETECTOR = "object_detector"
FACE_LANDMARKS = "face_landmarks"
HAND_LANDMARKS = "hand_landmarks"
POSE = "pose"
COMPONENTS = (OBJECT_DETECTOR, FACE_LANDMARKS, HAND_LANDMARKS, POSE)

#: Which monitoring capability (core/monitoring_state.py) each component provides.
CAPABILITY_OF = {OBJECT_DETECTOR: "objects", FACE_LANDMARKS: "face", HAND_LANDMARKS: "hands", POSE: "pose"}
COMPONENT_OF = {cap: comp for comp, cap in CAPABILITY_OF.items()}

OK, STALE, FAILED, UNAVAILABLE, IDLE = "ok", "stale", "failed", "unavailable", "idle"
STATES = (OK, STALE, FAILED, UNAVAILABLE, IDLE)

DEFAULT_MAX_AGE_S = 1.0
#: A consumer not seen for this long is no longer counted as consuming.
_CONSUMER_TTL_S = 10.0
#: Bounds: per-component consumer memory and the number of wrapped models/graphs.
_MAX_CONSUMERS = 32
_MAX_WRAPPED = 32
#: Run-time EMA weight for each new sample.
_EMA = 0.2


# ---------------------------------------------------------------------- the observation contract
@dataclass(frozen=True)
class ObjectDetection:
    """One detector box. `label` is the detector's own class name ("cell phone")."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]  # pixel-space x1, y1, x2, y2
    class_id: Optional[int] = None
    track_id: Optional[int] = None  # None: no tracker assigned one

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(float(self.confidence), 4),
                "box": [round(float(v), 1) for v in self.box], "class_id": self.class_id,
                "track_id": self.track_id}


@dataclass(frozen=True)
class EyeState:
    """Eye openness as measured this frame. Each field None when not measured."""

    ear: Optional[float] = None  # eye aspect ratio (smoothed)
    iris_openness: Optional[float] = None
    closed: Optional[bool] = None  # None: no calibrated threshold to judge against yet


@dataclass(frozen=True)
class HeadPose:
    """Head angles in degrees. `calibrated`: relative to this driver's neutral pose;
    otherwise raw camera-frame Euler angles (display and telemetry only)."""

    yaw_deg: float
    pitch_deg: float
    calibrated: bool


@dataclass(frozen=True)
class Gaze:
    """Eye-in-head gaze offset in degrees, from the iris (None in FaceObservation when
    the eye was closed or the iris fit was not usable)."""

    yaw_offset_deg: float
    pitch_offset_deg: float


@dataclass(frozen=True)
class FaceObservation:
    """A face the landmark model found this frame, with what the module derived from it."""

    landmark_count: int
    eye_state: Optional[EyeState] = None
    head_pose: Optional[HeadPose] = None
    gaze: Optional[Gaze] = None


@dataclass(frozen=True)
class HandObservation:
    box: tuple[float, float, float, float]  # pixel-space, from the landmark extent
    landmark_count: int


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    state: str
    model: Optional[Mapping[str, Any]] = None
    last_run_age_s: Optional[float] = None  # since the last successful result
    runs: int = 0  # inferences on full frames (cache hits are not runs)
    errors: int = 0
    reason: Optional[str] = None
    consumers: tuple[str, ...] = ()
    run_ms: Optional[float] = None  # exponential moving average of one inference

    @property
    def capability(self) -> str:
        return CAPABILITY_OF[self.name]

    @property
    def usable(self) -> bool:
        return self.state == OK

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "model": dict(self.model) if self.model else None,
            "last_run_age_s": None if self.last_run_age_s is None else round(self.last_run_age_s, 2),
            "runs": self.runs,
            "errors": self.errors,
            "reason": self.reason,
            "consumers": list(self.consumers),
            "run_ms": None if self.run_ms is None else round(self.run_ms, 2),
        }


@dataclass(frozen=True)
class Observation:
    """Everything the shared models said about one frame.

    `detections` is None when object detection is unavailable, failed or stale
    for this frame — never an empty tuple standing in for "could not look".
    `fresh_detections` is True only when the detector ran on THIS frame; between
    runs the latest result is carried while it is younger than max_age_s.
    `face_visible` / `hands` are None when their models did not run this frame.
    """

    frame_id: int
    captured_at: float  # monotonic
    source: str  # "camera" | "sim" | "replay"
    simulated: bool
    width: int
    height: int
    detections: Optional[tuple[ObjectDetection, ...]]
    detections_age_s: Optional[float]
    fresh_detections: bool
    detection_consumers: frozenset[str]  # who received the detector's output on this frame
    face_visible: Optional[bool]
    face: Optional[FaceObservation]
    hands: Optional[tuple[HandObservation, ...]]
    components: Mapping[str, ComponentStatus]
    schema: str = SCHEMA
    #: The shared tracker's live tracks (core/tracking.py), visible and coasting;
    #: None whenever `detections` is None, or when no tracker ran (a replay).
    tracks: Optional[tuple[Track, ...]] = None

    def available(self, capability: str) -> bool:
        status = self.components.get(COMPONENT_OF.get(capability, ""))
        return status is not None and status.usable

    def count(self, label: str, min_confidence: float = 0.0) -> Optional[int]:
        """How many `label` boxes; None when detections are unavailable."""
        if self.detections is None:
            return None
        return sum(1 for d in self.detections if d.label == label and d.confidence >= min_confidence)

    def detection_frame(self, min_confidence: Union[float, Mapping[str, float]] = 0.0,
                        labels: Optional[Iterable[str]] = None) -> Optional[DetectionFrame]:
        """The rule engine's view (core/detections.py), or None when unavailable.

        `min_confidence` is one floor for every label or a per-label mapping
        (its "*" key, if present, is the default).
        """
        if self.detections is None:
            return None
        wanted = set(labels) if labels is not None else None

        def floor(label: str) -> float:
            if isinstance(min_confidence, Mapping):
                return float(min_confidence.get(label, min_confidence.get("*", 0.0)))
            return float(min_confidence)

        kept = tuple(
            Detection(label=d.label, confidence=d.confidence, box=d.box, track_id=d.track_id)
            for d in self.detections
            if (wanted is None or d.label in wanted) and d.confidence >= floor(d.label)
        )
        return DetectionFrame(width=self.width, height=self.height, detections=kept)

    def attach_tracks(self, frame: Optional[DetectionFrame], min_iou: float = 0.9) -> Optional[DetectionFrame]:
        """`frame` (a module's own view of THIS frame's detector run) with the
        shared tracker's ids attached: each box takes the id of the same-label
        detection it coincides with (IoU >= min_iou; a module may round its
        boxes). Boxes with no such twin keep `track_id` None."""
        if frame is None or not self.detections or not frame.detections:
            return frame
        tracked = [d for d in self.detections if d.track_id is not None]
        if not tracked:
            return frame
        out = []
        changed = False
        for d in frame.detections:
            best, best_iou = None, min_iou
            if d.box is not None and d.track_id is None:
                for t in tracked:
                    if t.label != d.label:
                        continue
                    overlap = box_iou(tuple(float(v) for v in d.box), t.box)  # type: ignore[arg-type]
                    if overlap >= best_iou:
                        best, best_iou = t, overlap
            if best is not None:
                out.append(dataclasses.replace(d, track_id=best.track_id))
                changed = True
            else:
                out.append(d)
        return dataclasses.replace(frame, detections=tuple(out)) if changed else frame


@dataclass(frozen=True)
class Demand:
    """A consumer that owns no model asking for a component's output."""

    consumer: str
    component: str
    labels: frozenset[str] = frozenset()  # detector class names; empty = whatever runs
    min_interval_s: float = 0.0  # 0 = every frame
    min_confidence: float = 0.25


# ---------------------------------------------------------------------- per-frame result views
class _BoxView:
    """The filtered results a SharedDetector returns: shaped like an ultralytics
    Result as far as the modules read it (an iterable `.boxes`)."""

    __slots__ = ("boxes",)

    def __init__(self, boxes: list) -> None:
        self.boxes = boxes


def _class_of(box: Any) -> int:
    return int(box.cls[0])


def _conf_of(box: Any) -> float:
    return float(box.conf[0])


def _xyxy_of(box: Any) -> tuple[float, float, float, float]:
    raw = box.xyxy[0]
    values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    x1, y1, x2, y2 = (float(v) for v in values[:4])
    return x1, y1, x2, y2


# ---------------------------------------------------------------------- component bookkeeping
@dataclass
class _Component:
    name: str
    runs: int = 0
    errors: int = 0
    last_ok_at: Optional[float] = None
    last_attempt_at: Optional[float] = None
    error: Optional[str] = None  # set by a failed attempt, cleared by a success
    unavailable: Optional[str] = None
    #: M8.5 (M84-5): refused for good (its model file failed the integrity check
    #: at boot). Unlike `unavailable`, a run never clears it.
    blocked: Optional[str] = None
    run_ms: Optional[float] = None
    model: Optional[dict] = None
    consumers: dict[str, float] = field(default_factory=dict)

    def seen(self, consumer: Optional[str], now: float) -> None:
        if not consumer:
            return
        self.consumers[consumer] = now
        if len(self.consumers) > _MAX_CONSUMERS:
            oldest = min(self.consumers, key=self.consumers.__getitem__)
            del self.consumers[oldest]

    def ok(self, now: float, elapsed_ms: float) -> None:
        self.runs += 1
        self.last_ok_at = now
        self.last_attempt_at = now
        self.error = None
        self.unavailable = None
        self.run_ms = elapsed_ms if self.run_ms is None else (1 - _EMA) * self.run_ms + _EMA * elapsed_ms

    def failed(self, now: float, exc: BaseException) -> None:
        self.errors += 1
        self.last_attempt_at = now
        self.error = str(exc) or type(exc).__name__

    def status(self, now: float, max_age_s: float) -> ComponentStatus:
        active = tuple(sorted(c for c, t in self.consumers.items() if now - t <= _CONSUMER_TTL_S))
        age = None if self.last_ok_at is None else max(0.0, now - self.last_ok_at)
        if self.blocked is not None:
            state, reason = UNAVAILABLE, self.blocked
        elif self.unavailable is not None:
            state, reason = UNAVAILABLE, self.unavailable
        elif self.error is not None:
            state, reason = FAILED, self.error
        elif age is not None and age <= max_age_s:
            state, reason = OK, None
        elif active:
            state = STALE
            reason = (f"no result for {age:.1f}s" if age is not None
                      else "not run yet")
        else:
            state, reason = IDLE, None if self.last_attempt_at is None else "not in use"
        return ComponentStatus(
            name=self.name, state=state, model=dict(self.model) if self.model else None,
            last_run_age_s=age, runs=self.runs, errors=self.errors, reason=reason,
            consumers=active, run_ms=self.run_ms,
        )


# ---------------------------------------------------------------------- shared model proxies
class SharedDetector:
    """A detector that runs at most once per frame, however many callers ask.

    Callers keep the `predict(frame, verbose=, classes=, conf=)` call surface.
    Everything else (`names`, `model_name`, `close`) reaches the wrapped model.
    """

    def __init__(self, engine: "PerceptionEngine", model: Any) -> None:
        self.wrapped = model
        self._engine = engine
        #: Classes callers have asked for (None once anyone asked for all).
        self._learned: Optional[set[int]] = set()
        self._learned_conf: Optional[float] = None
        self._key: Optional[int] = None
        self._results: Any = None
        self._run_classes: Optional[frozenset[int]] = None
        self._run_conf: float = 0.0
        self._error: Optional[BaseException] = None
        self.runs = 0  # inferences on full frames
        self.passthrough_runs = 0  # crops, warm-up frames, anything that is not the current frame

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def _learn(self, classes: Optional[frozenset[int]], conf: float) -> None:
        if classes is None:
            self._learned = None
        elif self._learned is not None:
            self._learned |= classes
        self._learned_conf = conf if self._learned_conf is None else min(self._learned_conf, conf)

    def _covers(self, classes: Optional[frozenset[int]], conf: float) -> bool:
        if conf < self._run_conf:
            return False
        if self._run_classes is None:
            return True
        return classes is not None and classes <= self._run_classes

    def _filtered(self, classes: Optional[frozenset[int]], conf: float) -> Any:
        if classes == self._run_classes and conf == self._run_conf:
            return self._results  # exactly what was run: the model's own result objects
        return [_BoxView([b for b in r.boxes
                          if (classes is None or _class_of(b) in classes) and _conf_of(b) >= conf])
                for r in self._results]

    def predict(self, frame: Any, verbose: bool = False, classes: Any = None, conf: float = 0.25,
                **kwargs: Any) -> Any:
        engine = self._engine
        key = engine.frame_key(frame)
        if key is None:
            self.passthrough_runs += 1
            return self.wrapped.predict(frame, verbose=verbose, classes=classes, conf=conf, **kwargs)
        wanted = None if classes is None else frozenset(int(c) for c in classes)
        conf = float(conf)
        self._learn(wanted, conf)
        engine.note_consumer(OBJECT_DETECTOR)
        if self._key == key:
            if self._error is not None:
                raise self._error  # one failure per frame, reported to every caller
            if self._covers(wanted, conf):
                engine.note_served()
                return self._filtered(wanted, conf)
        run_classes = None if self._learned is None else frozenset(self._learned)
        run_conf = float(self._learned_conf if self._learned_conf is not None else conf)
        started = time.perf_counter()
        try:
            results = self.wrapped.predict(
                frame, verbose=verbose,
                classes=None if run_classes is None else sorted(run_classes), conf=run_conf, **kwargs)
        except Exception as exc:
            self._key, self._error, self._results = key, exc, None
            engine.component_failed(OBJECT_DETECTOR, exc)
            raise
        self.runs += 1
        self._key, self._error, self._results = key, None, results
        self._run_classes, self._run_conf = run_classes, run_conf
        engine.detector_ran(self, results, (time.perf_counter() - started) * 1000.0)
        engine.note_served()
        return self._filtered(wanted, conf)


class SharedGraph:
    """A MediaPipe-style graph (`process(image)`) run at most once per frame.

    MediaPipe's tracking graphs are stateful: calling one twice on the same
    frame would also corrupt its tracking, so the cache is a correctness
    matter, not only a cost one.
    """

    def __init__(self, engine: "PerceptionEngine", component: str, graph: Any) -> None:
        self.wrapped = graph
        self.component = component
        self._engine = engine
        self._key: Optional[tuple] = None
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self.runs = 0
        self.passthrough_runs = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)

    def process(self, image: Any) -> Any:
        engine = self._engine
        key = engine.graph_key(image)
        if key is None:
            self.passthrough_runs += 1
            return self.wrapped.process(image)
        engine.note_consumer(self.component)
        if self._key == key:
            if self._error is not None:
                raise self._error
            return self._result
        started = time.perf_counter()
        try:
            result = self.wrapped.process(image)
        except Exception as exc:
            self._key, self._error, self._result = key, exc, None
            engine.component_failed(self.component, exc)
            raise
        self.runs += 1
        self._key, self._error, self._result = key, None, result
        engine.graph_ran(self.component, result, (time.perf_counter() - started) * 1000.0)
        return result

    def close(self) -> None:
        self._engine.forget(self)
        close = getattr(self.wrapped, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------- the engine
_ACTIVE = threading.local()


class PerceptionEngine:
    """Per-frame scheduling and caching of the shared perception components.

    One per DeviceRuntime. The runtime brackets each analysed frame with
    begin_frame()/end_frame(); in between, `activated()` makes get_yolo_model()
    and the module graph factories hand out shared proxies, and `run_due()`
    runs whatever a model-less consumer has asked for and nobody has run yet.
    """

    def __init__(self, *, max_age_s: float = DEFAULT_MAX_AGE_S,
                 clock: Optional[Callable[[], float]] = None,
                 tracking: Optional[Mapping[str, Any]] = None) -> None:
        self.max_age_s = float(max_age_s)
        self._clock = clock
        #: The device's one object tracker (core/tracking.py); `tracking` passes
        #: its tuning (iou_min, lost_grace_s, max_tracks, ...).
        self.tracker = ObjectTracker(**dict(tracking or {}))
        self._lock = threading.RLock()
        self._components = {name: _Component(name) for name in COMPONENTS}
        self._wrapped: dict[int, tuple[Any, Any]] = {}  # id(model or graph) -> (it, its proxy)
        self._detector: Optional[SharedDetector] = None
        self._detector_loader: Optional[Callable[[], Any]] = None
        self._demands: dict[tuple[str, str], Demand] = {}
        # The frame in flight.
        self._frame: Any = None
        self._frame_id: Optional[int] = None
        self._frame_meta: dict = {}
        self._consumer: Optional[str] = None
        self._served: set[str] = set()
        self._ran: set[str] = set()  # components that ran successfully on this frame
        self._face_result: Any = None
        self._hands_result: Any = None
        # The detector's last successful output, parsed, and when.
        self._detections: Optional[tuple[ObjectDetection, ...]] = None
        self._detections_at: Optional[float] = None
        self.frames = 0

    # ------------------------------------------------------------------ time
    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.monotonic()

    # ------------------------------------------------------------------ activation
    @contextlib.contextmanager
    def activated(self) -> Iterator["PerceptionEngine"]:
        """Make shared_detector()/shared_graph() wrap through this engine (this thread)."""
        previous = getattr(_ACTIVE, "engine", None)
        _ACTIVE.engine = self
        try:
            yield self
        finally:
            _ACTIVE.engine = previous

    @contextlib.contextmanager
    def consuming(self, consumer: str) -> Iterator[None]:
        """Attribute the model calls made inside to `consumer` (a module or node id)."""
        previous, self._consumer = self._consumer, consumer
        try:
            yield
        finally:
            self._consumer = previous

    # ------------------------------------------------------------------ wrapping
    def wrap_detector(self, model: Any) -> Any:
        if model is None or (isinstance(model, SharedDetector) and model._engine is self):
            return model
        model = unwrap(model)  # a proxy bound to another engine (a restarted runtime) is re-bound
        with self._lock:
            entry = self._wrapped.get(id(model))
            if entry is not None and entry[0] is model:
                proxy = entry[1]
            else:
                if len(self._wrapped) >= _MAX_WRAPPED:
                    self._wrapped.pop(next(iter(self._wrapped)))
                proxy = SharedDetector(self, model)
                self._wrapped[id(model)] = (model, proxy)
            self._detector = proxy
            return proxy

    def wrap_graph(self, component: str, graph: Any, model: Optional[dict] = None) -> Any:
        if graph is None or (isinstance(graph, SharedGraph) and graph._engine is self):
            return graph
        graph = unwrap(graph)
        if component not in self._components or component == OBJECT_DETECTOR:
            raise ValueError(f"not a graph component: {component!r}")
        with self._lock:
            entry = self._wrapped.get(id(graph))
            if entry is not None and entry[0] is graph:
                return entry[1]
            if len(self._wrapped) >= _MAX_WRAPPED:
                self._wrapped.pop(next(iter(self._wrapped)))
            proxy = SharedGraph(self, component, graph)
            self._wrapped[id(graph)] = (graph, proxy)
            if model is not None:
                self._components[component].model = dict(model)
            elif self._components[component].model is None:
                self._components[component].model = {"name": type(graph).__name__}
            return proxy

    def forget(self, proxy: Any) -> None:
        with self._lock:
            for key, (_, wrapped_proxy) in list(self._wrapped.items()):
                if wrapped_proxy is proxy:
                    del self._wrapped[key]

    #: Module attributes that hold a full-frame model, and what each one is.
    _DETECTOR_ATTRS = ("_yolo_model", "_model")
    _GRAPH_ATTRS = (("_face_mesh", FACE_LANDMARKS), ("_hands", HAND_LANDMARKS), ("_pose", POSE))

    def adopt(self, module: Any) -> Any:
        """Wrap the full-frame models a module already built (its constructor ran
        before any frame). Lazily-built ones are wrapped by the factories."""
        for attr in self._DETECTOR_ATTRS:
            model = getattr(module, attr, None)
            if model is not None and callable(getattr(model, "predict", None)) and hasattr(model, "names"):
                setattr(module, attr, self.wrap_detector(model))
        for attr, component in self._GRAPH_ATTRS:
            graph = getattr(module, attr, None)
            if graph is not None and callable(getattr(graph, "process", None)):
                setattr(module, attr, self.wrap_graph(component, graph))
        return module

    def set_detector_loader(self, loader: Optional[Callable[[], Any]]) -> None:
        """How to get a detector when a model-less consumer demands one and no
        module has loaded it (e.g. an occupancy node beside fall detection)."""
        self._detector_loader = loader

    def set_model(self, component: str, model: Optional[Mapping[str, Any]]) -> None:
        with self._lock:
            self._components[component].model = dict(model) if model else None

    def set_unavailable(self, component: str, reason: Optional[str]) -> None:
        """A component that cannot run on this device at all (None clears it)."""
        with self._lock:
            self._components[component].unavailable = reason

    def block(self, component: str, reason: str) -> None:
        """Refused for the life of the process (M8.5: its model file failed the boot
        integrity check, core/model_registry.py). Reported UNAVAILABLE with the reason
        whatever runs; core.model_registry also keeps the model itself from loading."""
        with self._lock:
            self._components[component].blocked = reason

    # ------------------------------------------------------------------ demands
    def demand(self, demand: Demand) -> None:
        with self._lock:
            self._demands[(demand.consumer, demand.component)] = demand

    def withdraw(self, consumer: str) -> None:
        with self._lock:
            for key in [k for k in self._demands if k[0] == consumer]:
                del self._demands[key]

    def demands(self) -> tuple[Demand, ...]:
        with self._lock:
            return tuple(self._demands.values())

    # ------------------------------------------------------------------ the frame
    def frame_key(self, frame: Any) -> Optional[int]:
        """This frame's id if `frame` IS the frame in flight (not a crop or a copy)."""
        if self._frame is not None and frame is self._frame:
            return self._frame_id
        return None

    def graph_key(self, image: Any) -> Optional[tuple]:
        """Graphs receive a colour-converted copy, so they are keyed on the frame
        in flight and the image size; a crop of another size passes through."""
        if self._frame is None:
            return None
        shape = getattr(image, "shape", None)
        if shape is None or tuple(shape[:2]) != tuple(self._frame.shape[:2]):
            return None
        return (self._frame_id, tuple(shape[:2]))

    def begin_frame(self, frame: Any, *, frame_id: int, captured_at: Optional[float] = None,
                    source: str = "camera", simulated: bool = False) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id = int(frame_id)
            shape = getattr(frame, "shape", (0, 0))
            self._frame_meta = {
                "captured_at": self._now() if captured_at is None else float(captured_at),
                "source": source, "simulated": bool(simulated),
                "height": int(shape[0]) if len(shape) > 0 else 0,
                "width": int(shape[1]) if len(shape) > 1 else 0,
            }
            self._served = set()
            self._ran = set()
            self._face_result = None
            self._hands_result = None
            self.frames += 1

    def note_consumer(self, component: str) -> None:
        with self._lock:
            self._components[component].seen(self._consumer, self._now())

    def note_served(self) -> None:
        if self._consumer:
            self._served.add(self._consumer)

    def component_failed(self, component: str, exc: BaseException) -> None:
        with self._lock:
            self._components[component].failed(self._now(), exc)
            if component == OBJECT_DETECTOR:
                # No carrying a frozen box past a failure: the last boxes are not
                # what the camera shows now, and nobody can tell for how long.
                self._detections = None
                self._detections_at = None

    def detector_ran(self, detector: SharedDetector, results: Any, elapsed_ms: float) -> None:
        names = getattr(detector.wrapped, "names", {}) or {}
        parsed = []
        for result in results or []:
            for box in getattr(result, "boxes", None) or []:
                class_id = _class_of(box)
                parsed.append(ObjectDetection(
                    label=str(names.get(class_id, class_id)), confidence=_conf_of(box),
                    box=_xyxy_of(box), class_id=class_id))
        with self._lock:
            now = self._now()
            self._components[OBJECT_DETECTOR].ok(now, elapsed_ms)
            # One association per detector run, for every consumer at once.
            ids = self.tracker.update([(d.label, d.box, d.confidence) for d in parsed], now)
            self._detections = tuple(dataclasses.replace(d, track_id=tid) for d, tid in zip(parsed, ids, strict=True))
            self._detections_at = now
            self._ran.add(OBJECT_DETECTOR)

    def graph_ran(self, component: str, result: Any, elapsed_ms: float) -> None:
        with self._lock:
            self._components[component].ok(self._now(), elapsed_ms)
            self._ran.add(component)
            if component == FACE_LANDMARKS:
                self._face_result = result
            elif component == HAND_LANDMARKS:
                self._hands_result = result

    def detector(self) -> Optional[SharedDetector]:
        """The shared detector, loading it through the loader if nobody has."""
        if self._detector is None and self._detector_loader is not None:
            try:
                model = self._detector_loader()
            except Exception as exc:  # noqa: BLE001 - reported as the component's state
                self.set_unavailable(OBJECT_DETECTOR, f"object detector could not be loaded: {exc}")
                return None
            if model is None:
                return None
            self.wrap_detector(model)
        return self._detector

    def run_due(self, frame: Any) -> None:
        """Run the detector for any demand that is due and not yet served this frame.

        A failure here is recorded (the component reports FAILED) and swallowed:
        a model-less consumer's demand must never stop the module's own pass.
        """
        demands = [d for d in self.demands() if d.component == OBJECT_DETECTOR]
        if not demands or OBJECT_DETECTOR in self._ran:
            return
        comp = self._components[OBJECT_DETECTOR]
        now = self._now()
        due = [d for d in demands
               if comp.last_attempt_at is None or now - comp.last_attempt_at >= d.min_interval_s]
        if not due:
            for d in demands:
                comp.seen(d.consumer, now)
            return
        detector = self.detector()
        if detector is None:
            return
        names = getattr(detector.wrapped, "names", {}) or {}
        ids_of = {str(n): int(i) for i, n in names.items()}
        requests = []
        for d in due:
            classes = None if not d.labels else sorted(ids_of[label] for label in d.labels if label in ids_of)
            if classes == []:
                continue  # this detector has none of the labels: nothing to run for it
            requests.append((d, classes))
            # Learned up front, so the first of these runs covers all of them.
            detector._learn(None if classes is None else frozenset(classes), float(d.min_confidence))
        for d, classes in requests:
            with self.consuming(d.consumer):
                try:
                    detector.predict(frame, verbose=False, classes=classes, conf=d.min_confidence)
                except Exception:  # noqa: BLE001 - recorded in component_failed
                    return

    def end_frame(self, signals: Optional[Mapping[str, Any]] = None) -> Observation:
        """Close the frame in flight and describe it. `signals` is what the module
        derived from the face this frame (VisionModule.perception_signals)."""
        with self._lock:
            now = self._now()
            meta = self._frame_meta or {"captured_at": now, "source": "camera", "simulated": False,
                                        "height": 0, "width": 0}
            fresh = OBJECT_DETECTOR in self._ran
            age = None if self._detections_at is None else max(0.0, now - self._detections_at)
            detections = self._detections if age is not None and age <= self.max_age_s else None
            face_visible, face = self._face(signals)
            hands = self._hands(meta) if HAND_LANDMARKS in self._ran else None
            observation = Observation(
                frame_id=self._frame_id if self._frame_id is not None else -1,
                captured_at=meta["captured_at"], source=meta["source"], simulated=meta["simulated"],
                width=meta["width"], height=meta["height"],
                detections=detections, detections_age_s=None if detections is None else age,
                fresh_detections=fresh and detections is not None,
                detection_consumers=frozenset(self._served) if fresh else frozenset(),
                face_visible=face_visible, face=face, hands=hands,
                components=self._status_unlocked(now),
                tracks=self.tracker.tracks() if detections is not None else None,
            )
            self._frame = None
            return observation

    def _face(self, signals: Optional[Mapping[str, Any]]) -> tuple[Optional[bool], Optional[FaceObservation]]:
        if FACE_LANDMARKS not in self._ran:
            return None, None
        faces = getattr(self._face_result, "multi_face_landmarks", None)
        if not faces:
            return False, None
        landmarks = getattr(faces[0], "landmark", None) or ()
        signals = signals or {}
        eye = signals.get("eye_state")
        pose = signals.get("head_pose")
        gaze = signals.get("gaze")
        return True, FaceObservation(
            landmark_count=len(landmarks),
            eye_state=eye if isinstance(eye, EyeState) else None,
            head_pose=pose if isinstance(pose, HeadPose) else None,
            gaze=gaze if isinstance(gaze, Gaze) else None,
        )

    def _hands(self, meta: dict) -> tuple[HandObservation, ...]:
        hands = getattr(self._hands_result, "multi_hand_landmarks", None) or []
        w, h = meta["width"], meta["height"]
        out = []
        for hand in hands:
            points = getattr(hand, "landmark", None) or []
            if not points:
                continue
            xs = [float(p.x) * w for p in points]
            ys = [float(p.y) * h for p in points]
            out.append(HandObservation(box=(min(xs), min(ys), max(xs), max(ys)), landmark_count=len(points)))
        return tuple(out)

    # ------------------------------------------------------------------ status
    def _status_unlocked(self, now: float) -> dict[str, ComponentStatus]:
        return {name: comp.status(now, self.max_age_s) for name, comp in self._components.items()}

    def status(self) -> dict[str, ComponentStatus]:
        with self._lock:
            return self._status_unlocked(self._now())

    def component(self, name: str) -> ComponentStatus:
        return self.status()[name]

    def consumed_by(self, consumer: str) -> set[str]:
        """Components `consumer` has used recently (by calls or demands)."""
        now = self._now()
        with self._lock:
            used = {name for name, comp in self._components.items()
                    if now - comp.consumers.get(consumer, -1e18) <= _CONSUMER_TTL_S}
            used |= {d.component for d in self._demands.values() if d.consumer == consumer}
            return used

    def to_wire(self, *, simulated: bool, nodes: Iterable[Mapping[str, Any]] = ()) -> dict:
        """`monitoring.perception` as the device reports it (sd-perception/1)."""
        return {
            "schema": SCHEMA,
            "simulated": bool(simulated),
            "components": {name: status.to_dict() for name, status in self.status().items()},
            "nodes": [dict(n) for n in nodes],
            "tracking": self.tracking_report(),
        }

    def tracking_report(self) -> dict:
        """The shared tracker's counters and cost (`perception.tracking`)."""
        with self._lock:
            return {"tracker": "iou_centroid", **self.tracker.stats()}


# ---------------------------------------------------------------------- factories for modules
def current() -> Optional[PerceptionEngine]:
    """The engine active on this thread (inside a runtime's analysed frame), if any."""
    return getattr(_ACTIVE, "engine", None)


def shared_detector(model: Any) -> Any:
    """`model`, wrapped by the active engine when there is one (core.detectors)."""
    engine = current()
    return engine.wrap_detector(model) if engine is not None else model


def shared_graph(component: str, graph: Any, model: Optional[dict] = None) -> Any:
    """A MediaPipe graph, wrapped by the active engine when there is one.

    Modules build graphs in their constructor, before any frame, so the runtime
    also adopts each module it builds (PerceptionEngine.adopt); this factory
    covers graphs built lazily inside a frame.
    """
    engine = current()
    return engine.wrap_graph(component, graph, model) if engine is not None else graph


def unwrap(model: Any) -> Any:
    """The model behind a shared proxy (or `model` itself)."""
    return getattr(model, "wrapped", model) if isinstance(model, (SharedDetector, SharedGraph)) else model
