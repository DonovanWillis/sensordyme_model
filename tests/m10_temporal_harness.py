"""Deterministic temporal scenario harness for the production DriverAttentionModule (M10).

SYNTHETIC SCENARIOS ONLY. A scenario is a list of named segments, each a
stretch of per-frame signals (a head pose / eyelid gap rendered as fake
FaceMesh landmarks, "face lost", or a camera STALL with no frames at all),
played at a chosen frame rate with optional deterministic timing jitter.
Frame capture times are injected through VisionModule.set_frame_time, the
seam the runtime and replays use, so no wall clock is read. Everything
between the seams -- smoothing, calibration, the evidence clock, the shipped
trackers, risk fusion, event emission -- is the real module code.

The configuration is the SHIPPED one (config.yaml modules.driver_attention)
with only these test seams changed: a 1 s calibration window (it is a lead-in,
not under test), a temporary calibration file, and the object detector / hand
model replaced by fakes returning chosen boxes. Passing shows the temporal
logic does what it is designed to on constructed inputs; it is not a measured
accuracy and says nothing about real cabins, faces or lenses.

Kept free of anything newer than base 4dfd761 so the same scenarios can be
run against that commit for the before/after table (docs/TEMPORAL_REASONING.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import yaml
from test_behavior_integration import DRIVER_BOX, _FakeHands, _textured_frame, landmarks
from test_driver_attention_scenarios import _FakeFaceMeshResult
from test_phase1_driver_fixes import _FakeBox, _FakeResult

import modules.driver_attention as driver_attention_module
from modules.driver_attention import DriverAttentionModule

REPO = Path(__file__).resolve().parent.parent

#: Eyelid half-gaps (px) for the fake landmarks: 6 = open (EAR 0.40), 4 = lids
#: lowered as by a downward look (EAR 0.27, iris still visible), 1 = shut
#: (iris-openness 0.17, the lid covers the iris).
OPEN, LOWERED, SHUT = 6.0, 4.0, 1.0


def shipped_config(tmp_path: Path, **overrides: Any) -> dict:
    """config.yaml's driver_attention section with only the test seams changed."""
    shipped = yaml.safe_load((REPO / "config.yaml").read_text())["modules"]["driver_attention"]
    config = dict(shipped)
    config.update({
        "calibration_path": str(tmp_path / "m10_calibration.json"),
        "calibration_duration_s": 1.0,
        "calibration_max_pose_spread_deg": 5.0,
        "calibration_max_duration_s": 2.0,
        "behavior_v2": False,
    })
    config.update(overrides)
    return config


class FakeDetector:
    """Object detector stand-in: a driver person box (optional) plus chosen phones."""

    names: ClassVar[dict[int, str]] = {0: "person", 67: "cell phone"}

    def __init__(self) -> None:
        self.person = True
        self.phones: list[tuple] = []

    def predict(self, frame, **kwargs):
        boxes = ([_FakeBox(0, DRIVER_BOX, 0.9)] if self.person else []) + [
            _FakeBox(67, b, 0.8) for b in self.phones]
        return [_FakeResult(boxes)]


@dataclass
class Seg:
    """One stretch of a scenario timeline.

    face: landmark kwargs (yaw, pitch, lid, iris_dy) or None for "no face found".
    stall: no frames at all for `seconds` (camera freeze / blocked inference).
    pose_fail: the face is found but the head-pose fit fails (solvePnP returns
    nothing): eyes measurable, gaze unknown.
    phones / person / hands: detector and hand-model output from this segment on
    (None leaves the previous value).
    """

    seconds: float
    face: Optional[dict] = field(default_factory=dict)
    stall: bool = False
    pose_fail: bool = False
    fps: Optional[float] = None  # this segment's frame rate (None: the run's)
    phones: Optional[list] = None
    person: Optional[bool] = None
    hands: Optional[list] = None
    name: str = ""


def face(**kw: Any) -> dict:
    return dict(kw)


@dataclass
class Fired:
    t: float  # frame time relative to the end of calibration
    type: str
    duration: float
    confidence: Optional[float]
    detail: str


@dataclass
class Result:
    events: list[Fired]
    module: Any
    seg_starts: list[float]
    frames: int
    #: (frame time, alert conditions active after that frame), every frame
    active: list[tuple[float, frozenset]] = field(default_factory=list)

    def active_during(self, event_type: str, start: float, end: float) -> list[bool]:
        """Whether `event_type`'s alert condition was live on each frame in [start, end)."""
        return [event_type in a for t, a in self.active if start <= t < end]

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def count(self, event_type: str) -> int:
        return self.types().count(event_type)

    def first(self, event_type: str) -> Optional[Fired]:
        return next((e for e in self.events if e.type == event_type), None)

    def fired(self, event_type: str) -> Fired:
        """The first `event_type`, which the scenario requires to have fired."""
        event = self.first(event_type)
        assert event is not None, f"{event_type} never fired; events: {self.types()}"
        return event

    def latency(self, event_type: str, seg_index: int) -> Optional[float]:
        """Seconds from the start of segment `seg_index` to the first `event_type`."""
        fired = self.first(event_type)
        return None if fired is None else round(fired.t - self.seg_starts[seg_index], 3)


def build(tmp_path: Path, monkeypatch, *, fps: float, detector: Optional[FakeDetector] = None,
          hands: Optional[_FakeHands] = None, **overrides: Any):
    """A calibrated module, its fakes, and the frame time calibration ended at."""
    detector = detector or FakeDetector()
    monkeypatch.setattr(driver_attention_module, "get_yolo_model", lambda weights: detector)
    module: Any = DriverAttentionModule(shipped_config(tmp_path, **overrides))
    module._hands = hands if hands is not None else _FakeHands()
    frame = _textured_frame()
    neutral = landmarks()
    module._face_mesh.process = lambda rgb, lm=neutral: _FakeFaceMeshResult(lm)
    t, step = 100.0, 1.0 / fps
    for _ in range(int(2.0 * fps) + 3):
        module.set_frame_time(t)
        module.process(frame)
        t += step
    assert module._calibrator.is_calibrated, "calibration lead-in did not complete"
    return module, frame, detector, t


def run(tmp_path: Path, monkeypatch, segments: list[Seg], *, fps: float = 30.0, jitter: float = 0.0,
        seed: int = 0, detector: Optional[FakeDetector] = None, hands: Optional[_FakeHands] = None,
        **overrides: Any) -> Result:
    """Play `segments` and collect every event fired.

    The segments lay out a CONTINUOUS timeline; frames sample it at their own
    capture times (every 1/fps s, jittered uniformly by +/- `jitter` of the
    interval, deterministically from `seed`; a segment's own `fps` overrides
    the rate inside it). A segment shorter than a frame interval may therefore
    get no frame at all, exactly as with a real camera. A stall segment has no
    frames: the next frame is captured when it ends.
    """
    hands = hands if hands is not None else _FakeHands()
    module, frame, detector, t0 = build(tmp_path, monkeypatch, fps=fps, detector=detector, hands=hands,
                                        **overrides)
    rng = random.Random(seed)
    real_pose_fit = driver_attention_module.estimate_head_rotation
    starts: list[float] = []
    acc = 0.0
    for seg in segments:
        starts.append(round(acc, 6))
        acc += seg.seconds
    ends = [*starts[1:], acc]
    events: list[Fired] = []
    active: list[tuple[float, frozenset]] = []
    frames = 0
    rel, index, entered = 0.0, 0, -1
    while rel < acc - 1e-9:
        while rel >= ends[index] - 1e-9:
            index += 1
        while entered < index:  # apply every segment's detector/hand changes in order
            entered += 1
            seg = segments[entered]
            if seg.phones is not None:
                detector.phones = list(seg.phones)
            if seg.person is not None:
                detector.person = seg.person
            if seg.hands is not None:
                hands.boxes = list(seg.hands)
        seg = segments[index]
        if seg.stall:
            rel = ends[index]
            continue
        monkeypatch.setattr(driver_attention_module, "estimate_head_rotation",
                            (lambda *a, **k: None) if seg.pose_fail else real_pose_fit)
        lms = None if seg.face is None else landmarks(**seg.face)
        module._face_mesh.process = lambda rgb, lm=lms: _FakeFaceMeshResult(lm)
        module.set_frame_time(t0 + rel)
        for e in module.process(frame):
            events.append(Fired(round(rel, 4), e.type, float(e.duration or 0.0), e.confidence,
                                e.contributing_signals or ""))
        active.append((round(rel, 4), frozenset(module.active_alert_conditions())))
        frames += 1
        step = 1.0 / (seg.fps or fps)
        if jitter:
            step *= 1.0 + rng.uniform(-jitter, jitter)
        rel += step
    return Result(events, module, starts, frames, active)


def blank_frame() -> np.ndarray:
    return _textured_frame()
