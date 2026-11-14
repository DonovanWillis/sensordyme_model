"""Per-frame cost of the M10 stage timing on THIS host (the printed class says which: a Mac is not a Pi).

A/B over the real shared-perception path: the same frame loop (detector proxy +
two graph proxies + the tracker) with core.stage_timing.TIMINGS.add replaced by a
no-op (B) and live (A), plus the runtime's own per-frame adds/perf_counter calls.
"""
import statistics
import sys
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import perception as pc
from core import stage_timing as st


class Box:
    def __init__(self, c, p, xyxy):
        self.cls, self.conf, self.xyxy = [c], [p], [np.asarray(xyxy, float)]


class Det:
    names: ClassVar[dict] = {0: "person", 67: "cell phone"}

    def predict(self, frame, verbose=False, classes=None, conf=0.25, **_):
        return [types.SimpleNamespace(boxes=[Box(0, 0.9, (10, 10, 50, 90)), Box(67, 0.6, (20, 50, 30, 60))])]


class Graph:
    def process(self, image):
        lm = types.SimpleNamespace(landmark=[types.SimpleNamespace(x=0.5, y=0.5)] * 21)
        return types.SimpleNamespace(multi_face_landmarks=[lm], multi_hand_landmarks=[lm])


def run(frames: int) -> float:
    engine = pc.PerceptionEngine()
    det = engine.wrap_detector(Det())
    face = engine.wrap_graph(pc.FACE_LANDMARKS, Graph())
    hands = engine.wrap_graph(pc.HAND_LANDMARKS, Graph())
    frame = np.zeros((48, 64, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(frames):
        picked = time.monotonic()
        st.TIMINGS.add("frame_age", 0.5)
        engine.begin_frame(frame, frame_id=i)
        started = time.perf_counter()
        det.predict(frame, conf=0.2)
        face.process(frame)
        hands.process(frame)
        ms = (time.perf_counter() - started) * 1000.0
        st.TIMINGS.add("behaviour_reasoning", max(0.0, ms - engine.frame_model_ms))
        engine.end_frame()
        st.TIMINGS.add("frame_interval", 33.0)
        d = time.perf_counter()
        st.TIMINGS.add_since("alert_dispatch", d)
        st.TIMINGS.add("frame_to_decision", (time.monotonic() - picked) * 1000.0)
    return (time.perf_counter() - t0) / frames * 1e6


real_add = st.TIMINGS.add
results = {"live": [], "noop": []}
for _ in range(7):
    st.TIMINGS.add = real_add  # type: ignore[method-assign]
    results["live"].append(run(5000))
    st.TIMINGS.add = lambda *a, **k: True  # type: ignore[method-assign]
    results["noop"].append(run(5000))
live, noop = statistics.median(results["live"]), statistics.median(results["noop"])
print(f"host: {st.platform_tag()['class']} ({st.platform_tag()['machine']})")
print(f"per frame, shared-perception path with stand-in models: live {live:.1f} us, adds no-op {noop:.1f} us, "
      f"instrumentation {live - noop:.1f} us/frame ({(live - noop) / 33333 * 100:.3f}% of a 33 ms frame)")
