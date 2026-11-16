"""Deterministic replay: a recorded clip through a real VisionModule, on the clip's clock.

Every frame gets a timestamp from the clip itself (frame index / container fps),
not from the wall clock. That timestamp goes to the module the same way the
device's capture time does (``set_frame_time``, the M0.2 seam). For the length
of the replay, ``time.monotonic`` is also replaced by that clock, so cooldowns,
sustain timers and anything else the module times run on recorded time. The same
clip and the same configuration therefore produce the same events, however fast
or slow the host is.

``target_fps`` drops frames to the rate the device actually processes (a Pi does
not analyse every frame a 30 fps camera produces). ``resize`` matches the
device's capture size. Both are recorded with the result.

Per-frame processing time is measured with the real clock. It is the HOST's
cost (a desktop surrogate), never the Pi's.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Optional
from unittest import mock

import numpy as np

#: Replay clocks start here, not at 0: code that treats 0.0 as "never" stays correct.
CLOCK_BASE_S = 1000.0


@dataclass(frozen=True)
class PredictedEvent:
    t: float  # seconds from the clip's first frame
    type: str
    confidence: Optional[float]
    duration: float
    signals: str = ""


@dataclass
class ReplayResult:
    events: list[PredictedEvent]
    times: list[float]  # clip time of every PROCESSED frame
    active: list[frozenset[str]]  # module.active_alert_conditions() after each processed frame
    process_ms: list[float]  # host processing time per processed frame
    duration_s: float
    source_fps: float
    frames_read: int
    settings: dict = field(default_factory=dict)
    #: Alert types the module said it could NOT evaluate after each processed frame
    #: (a capability its event_requirements() names was not available: starting,
    #: impaired, calibrating). Empty sets when the module declares no requirements;
    #: ``settings["availability_declared"]`` says which.
    unavailable: list[frozenset[str]] = field(default_factory=list)

    def unavailable_spans(self) -> Optional[dict[str, list[tuple[float, float]]]]:
        """Per alert type, [start, end) clip-time spans it could not be evaluated; None if undeclared."""
        if not self.settings.get("availability_declared"):
            return None
        spans: dict[str, list[tuple[float, float]]] = {}
        for n, (t, kinds) in enumerate(zip(self.times, self.unavailable, strict=True)):
            end = self.times[n + 1] if n + 1 < len(self.times) else self.duration_s
            for kind in sorted(kinds):
                runs = spans.setdefault(kind, [])
                if runs and abs(runs[-1][1] - t) < 1e-9:
                    runs[-1] = (runs[-1][0], end)
                else:
                    runs.append((t, end))
        return spans


class ReplayClock:
    """A monotonic clock that only moves when the replay moves it."""

    def __init__(self, base: float = CLOCK_BASE_S) -> None:
        self.base = base
        self.now = base

    def __call__(self) -> float:
        return self.now

    def at(self, clip_t: float) -> float:
        self.now = self.base + clip_t
        return self.now

    @contextlib.contextmanager
    def installed(self) -> Iterator[ReplayClock]:
        with mock.patch.object(time, "monotonic", self):
            yield self


def read_frames(path: str, *, target_fps: Optional[float] = None, resize: Optional[tuple[int, int]] = None,
                fps_override: Optional[float] = None) -> tuple[float, Iterator[tuple[float, np.ndarray]]]:
    """(source fps, iterator of (clip_t, frame)), sub-sampled to ``target_fps``."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        cap.release()
        raise ValueError(f"{path} has no frame rate; pass fps_override")

    def frames() -> Iterator[tuple[float, np.ndarray]]:
        index, next_t = 0, 0.0
        step = 1.0 / target_fps if target_fps else 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                t = index / fps
                index += 1
                if step and t + 1e-9 < next_t:
                    continue
                next_t = (next_t + step) if step else t
                if resize is not None and (frame.shape[1], frame.shape[0]) != resize:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                yield t, frame
        finally:
            cap.release()

    return fps, frames()


def unavailable_alerts(module: object) -> Optional[frozenset[str]]:
    """Event types ``module`` cannot evaluate right now; None if it declares no requirements."""
    requirements = getattr(module, "event_requirements", None)
    status = getattr(module, "capability_status", None)
    if not callable(requirements) or not callable(status):
        return None
    needs = requirements()
    if not isinstance(needs, dict) or not needs:
        return None
    caps = status()
    if not isinstance(caps, dict):
        return None
    return frozenset(event for event, required in needs.items()
                     if any(name in caps and not caps[name].available for name in required))


def replay(module_factory: Callable[[], object], path: str, *, target_fps: Optional[float] = None,
           resize: Optional[tuple[int, int]] = (640, 480), fps_override: Optional[float] = None) -> ReplayResult:
    """Run one clip through a fresh module built by ``module_factory``."""
    clock = ReplayClock()
    fps, frames = read_frames(path, target_fps=target_fps, resize=resize, fps_override=fps_override)
    events: list[PredictedEvent] = []
    times: list[float] = []
    active: list[frozenset[str]] = []
    cost: list[float] = []
    unavailable: list[frozenset[str]] = []
    declared = False
    with clock.installed():
        module = module_factory()  # built on the replay clock: its start-up timestamps are clip time too
        try:
            for t, frame in frames:
                stamp = clock.at(t)
                module.set_frame_time(stamp)  # type: ignore[attr-defined]
                started = time.perf_counter()
                emitted = module.process(frame)  # type: ignore[attr-defined]
                cost.append((time.perf_counter() - started) * 1000.0)
                times.append(t)
                for e in emitted:
                    events.append(PredictedEvent(t, e.type, e.confidence, float(e.duration or 0.0),
                                                 getattr(e, "contributing_signals", "") or ""))
                conditions = getattr(module, "active_alert_conditions", None)
                active.append(frozenset(conditions() if conditions else ()))
                cannot = unavailable_alerts(module)
                declared = declared or cannot is not None
                unavailable.append(cannot or frozenset())
        finally:
            close = getattr(module, "close", None)
            if close:
                close()
    duration = (times[-1] + 1.0 / fps) if times else 0.0
    return ReplayResult(events, times, active, cost, duration, fps, len(times),
                        {"target_fps": target_fps, "resize": list(resize) if resize else None,
                         "fps_override": fps_override, "timebase": "frame_index/fps",
                         "availability_declared": declared}, unavailable)
