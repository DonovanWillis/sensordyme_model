"""M8 shared tracking: one ObjectTracker in the perception engine, ids every consumer shares.

Unit level (core/tracking.py) and engine level (core/perception.py): identity
across frames, deterministic association, confirmation and confidence
hysteresis, bounded state, zone crossing and dwell, the rule engine's dwell
following the shared id, and the cost bound. Synthetic boxes only: these prove
behaviour and bounds, never accuracy on real footage.
"""

from __future__ import annotations

import contextlib
import time
from typing import ClassVar

import numpy as np
import pytest

from core import perception as pc
from core.detections import Detection, DetectionFrame
from core.rules import RuleEngine
from core.tracking import (
    SHARED_MAX_DETECTIONS,
    SHARED_MAX_TRACKS,
    ConfidenceGate,
    ObjectTracker,
    ZoneMembership,
    box_iou,
)
from core.zones import Zone

W, H = 640, 480
LEFT = Zone("left", ((0.0, 0.5), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0)), "restricted")


def person(x: float, conf: float = 0.9, label: str = "person"):
    """A standing person centred at x (0..1 of the width), feet at 0.95 H."""
    return (label, (x * W - 50, 0.30 * H, x * W + 50, 0.95 * H), conf)


# ---------------------------------------------------------------------- the tracker
def test_a_walking_person_keeps_one_id_and_a_second_person_gets_another():
    tracker = ObjectTracker()
    ids = [tracker.update([person(0.10 + 0.02 * i)], now=i * 0.1)[0] for i in range(30)]
    assert set(ids) == {1}, "a person walking across the room is one person"
    both = tracker.update([person(0.15), person(0.10 + 0.02 * 30)], now=3.0)
    assert both[1] == 1 and both[0] == 2, "the newcomer is new, the walker keeps their id"


def test_association_is_deterministic_and_class_scoped():
    def run():
        tracker = ObjectTracker()
        out = []
        for i in range(10):
            out.append(tracker.update([person(0.3), person(0.5), person(0.3, label="dog")], now=i * 0.2))
        return out, [t.to_dict() for t in tracker.tracks()]
    assert run() == run(), "same input, same ids, same state"
    ids, tracks = run()
    assert ids[-1] == [1, 2, 3] and {t["label"] for t in tracks} == {"person", "dog"}


def test_an_overlapping_match_beats_a_closer_centre_only_match():
    tracker = ObjectTracker()
    tracker.update([("person", (100, 100, 200, 400), 0.9)], now=0.0)
    # Two candidates: one overlaps the track heavily, one is a far smaller box
    # whose centre sits closer to the track's centre but barely overlaps.
    ids = tracker.update([("person", (110, 100, 210, 400), 0.9), ("person", (145, 240, 155, 260), 0.9)], now=0.1)
    assert ids == [1, 2]


def test_confirmation_needs_two_runs_and_never_flaps_back():
    tracker = ObjectTracker()
    tracker.update([person(0.5)], now=0.0)
    assert not tracker.tracks()[0].confirmed, "one detector run is a flicker, not a track"
    tracker.update([person(0.5)], now=0.1)
    assert tracker.tracks()[0].confirmed
    tracker.update([], now=0.2)
    (coasting,) = tracker.tracks()
    assert coasting.confirmed and not coasting.visible, "a miss is coasting, not unconfirming"


def test_occlusion_inside_the_grace_keeps_the_id_and_beyond_it_retires():
    tracker = ObjectTracker(lost_grace_s=2.0)
    tracker.update([person(0.5)], now=0.0)
    tracker.update([], now=1.9)
    assert tracker.update([person(0.52)], now=1.95) == [1], "behind the sofa for 1.9 s: same person"
    tracker.update([], now=4.5)
    assert len(tracker) == 0 and tracker.retired == 1
    assert tracker.update([person(0.5)], now=4.6) == [2], "gone 2.6 s: a new id, never reused"


def test_smoothed_confidence_and_gate_hysteresis_stop_flicker():
    tracker = ObjectTracker()
    gate = ConfidenceGate(0.5, margin=0.1)
    admitted = []
    for i, conf in enumerate([0.9, 0.9, 0.45, 0.55, 0.45, 0.48, 0.42, 0.44, 0.41, 0.40, 0.38, 0.30, 0.30, 0.30]):
        tracker.update([person(0.5, conf)], now=i * 0.1)
        admitted.append(bool(gate.admit(tracker.tracks())))
    raw_flips = sum(1 for a, b in zip([0.9, 0.9, 0.45, 0.55, 0.45], [0.9, 0.45, 0.55, 0.45, 0.48], strict=True)
                    if (a >= 0.5) != (b >= 0.5))
    assert raw_flips >= 3, "the raw confidence crosses the 0.5 floor over and over"
    first_out = admitted.index(False)
    assert all(admitted[:first_out]) and not any(admitted[first_out:]), \
        f"admitted once, dropped once, never re-admitted on noise: {admitted}"
    assert first_out >= 6, "held through the dips by the EMA and the exit band"


def test_state_is_bounded_and_eviction_is_deterministic():
    tracker = ObjectTracker(max_tracks=4, max_detections=6)
    crowd = [person(0.05 + 0.15 * i) for i in range(6)]
    ids = tracker.update(crowd, now=0.0)
    assert len(tracker) == 4 and tracker.evicted == 2, "never more than max_tracks live"
    assert ids == [1, 2, 3, 4, 5, 6] and [t.track_id for t in tracker.tracks()] == [3, 4, 5, 6]
    many = [person(0.01 * i, conf=0.5 + 0.004 * i) for i in range(10)]
    ids = tracker.update(many, now=0.1)
    assert ids.count(None) == 4 and tracker.dropped_detections == 4, "the weakest boxes past the cap are dropped"
    assert ids[:4] == [None] * 4, "the four lowest-confidence boxes are the ones dropped"


def test_worst_case_update_cost_is_bounded():
    """64 live tracks x 64 detections of one label: the documented worst case."""
    tracker = ObjectTracker()
    boxes = [("person", (i * 9.0, 10.0, i * 9.0 + 8.0, 60.0), 0.9) for i in range(SHARED_MAX_DETECTIONS)]
    tracker.update(boxes, now=0.0)
    assert len(tracker) == SHARED_MAX_TRACKS
    started = time.perf_counter()
    rounds = 20
    for r in range(1, rounds + 1):
        shifted = [(lbl, (b[0] + r, b[1], b[2] + r, b[3]), c) for lbl, b, c in boxes]
        tracker.update(shifted, now=r * 0.1)
    per_update_ms = (time.perf_counter() - started) * 1000.0 / rounds
    room = ObjectTracker()
    started = time.perf_counter()
    for r in range(200):
        room.update([person(0.1 + 0.15 * k + 0.001 * r) for k in range(5)], now=r * 0.1)
    room_ms = (time.perf_counter() - started) * 1000.0 / 200
    print(f"\n[M8 tracking cost] worst case 64 tracks x 64 boxes: {per_update_ms:.2f} ms/update; "
          f"typical room (5 people): {room_ms:.3f} ms/update (CPython on this host, not a Pi)")
    assert len(tracker) == SHARED_MAX_TRACKS and tracker.created == SHARED_MAX_TRACKS
    assert room.created == 5 and room_ms < per_update_ms
    assert per_update_ms < 250.0, "bounded: far below one frame even at the worst case on a slow host"


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------- zones and dwell
def test_zone_crossing_and_dwell_follow_the_track_with_hysteresis():
    tracker = ObjectTracker()
    zones = ZoneMembership(settle_s=1.0)
    transitions = []
    # In the left area for 5 s (with a one-run blip outside at 2 s), then walks out.
    for i in range(80):
        t = i * 0.1
        x = 0.2 if t < 5.0 else 0.8
        if abs(t - 2.0) < 1e-9:
            x = 0.5  # one detector run outside: hysteresis, not an exit
        tracker.update([person(x)], now=t)
        transitions += zones.update(tracker.tracks(), [LEFT], W, H, t)
        if abs(t - 4.0) < 1e-9:
            assert zones.dwell("left", t) == {1: pytest.approx(4.0)}, "dwell counts from the first sighting inside"
    kinds = [(z.kind, z.zone, z.track_id) for z in transitions]
    assert kinds == [("entered", "left", 1), ("exited", "left", 1)]
    assert transitions[1].dwell_s == pytest.approx(4.9, abs=0.11)


def test_a_retired_track_exits_its_area_and_state_stays_bounded():
    tracker = ObjectTracker(lost_grace_s=1.0)
    zones = ZoneMembership(settle_s=0.5)
    out = []
    for i in range(20):
        tracker.update([person(0.2)], now=i * 0.1)
        out += zones.update(tracker.tracks(), [LEFT], W, H, i * 0.1)
    for i in range(20, 40):
        tracker.update([], now=i * 0.1)
        out += zones.update(tracker.tracks(), [LEFT], W, H, i * 0.1)
    assert [z.kind for z in out] == ["entered", "exited"] and len(zones) == 0


def test_the_rule_engines_dwell_clock_follows_the_shared_track_id():
    """A small, fast mover inside the area, never back at a spot within the dwell
    grace (1 s): the dwell matcher alone (at most 1.5 box diagonals/s) starts a
    new clock at every spot; given the shared tracker's id, the rule engine keeps
    ONE clock for it -- the device's tracker decides who is who, not the rule."""
    rule = {"target": "person", "predicate": "lingering", "zone": "left", "dwell_s": 3.0,
            "severity": "warning", "notify": "push"}

    def run(with_ids: bool) -> list:
        engine = RuleEngine()
        engine.set_zones([LEFT])
        engine.replace_rules([rule])
        alerts = []
        spots = [(fx, fy) for fy in (300, 370, 440) for fx in (30, 90, 150, 210)]  # feet points in "left"
        for i in range(60):
            t = i * 0.1
            fx, fy = spots[i % len(spots)]  # a new spot every 0.1 s, each revisited after 1.2 s
            box = (fx - 10, fy - 60, fx + 10, fy)
            det = Detection("person", 0.9, box, track_id=7 if with_ids else None)
            alerts += engine.observe_frame(DetectionFrame(W, H, (det,)), now=t)
        return alerts
    assert run(with_ids=False) == [], "without identity the clock restarts on every jump"
    assert len(run(with_ids=True)) == 1, "with the shared id it is one 6 s visit"


# ---------------------------------------------------------------------- the engine owns the one tracker
class _Box:
    def __init__(self, cls, xyxy, conf):
        self.cls, self.xyxy, self.conf = [cls], [np.array(xyxy, dtype=float)], [conf]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _Detector:
    names: ClassVar[dict[int, str]] = {0: "person", 16: "dog"}

    def __init__(self):
        self.calls = 0
        self.boxes: list = []
        self.fail = False

    def predict(self, frame, **_kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("NPU stream aborted")
        return [_Result([_Box(c, b, s) for c, b, s in self.boxes])]


def _frame(engine, detector, fid, now, *consumers):
    frame = np.zeros((H, W, 3), np.uint8)
    engine.begin_frame(frame, frame_id=fid, captured_at=now, source="sim", simulated=True)
    with engine.activated():
        engine.run_due(frame)
        for consumer in consumers:
            with engine.consuming(consumer), contextlib.suppress(RuntimeError):
                pc.shared_detector(detector).predict(frame, classes=[0], conf=0.3)
    return engine.end_frame()


def test_every_consumer_of_a_frame_gets_the_same_track_ids_from_one_run():
    clock = [0.0]
    engine = pc.PerceptionEngine(clock=lambda: clock[0])
    detector = _Detector()
    engine.set_detector_loader(lambda: detector)
    engine.demand(pc.Demand("people_counter", pc.OBJECT_DETECTOR, frozenset({"person"})))
    engine.demand(pc.Demand("zone_watch", pc.OBJECT_DETECTOR, frozenset({"person"})))
    observations = []
    for i in range(10):
        clock[0] = i * 0.1
        detector.boxes = [(0, (100 + 5 * i, 100, 200 + 5 * i, 450), 0.9), (0, (400, 100, 500, 450), 0.8)]
        observations.append(_frame(engine, detector, i, clock[0], "shared_perception", "rules"))
    assert detector.calls == 10, "one inference per frame for four consumers"
    assert engine.tracker.updates == 10, "one association per detector run, not per consumer"
    last = observations[-1]
    assert [d.track_id for d in last.detections] == [1, 2]
    assert [(t.track_id, t.confirmed, t.visible) for t in last.tracks] == [(1, True, True), (2, True, True)]
    frame = last.detection_frame(0.5, labels=("person",))
    assert [d.track_id for d in frame.detections] == [1, 2], "the rule view carries the same ids"
    # A module's own view of the same run (rounded boxes) gets the ids attached.
    module_view = DetectionFrame(W, H, tuple(Detection("person", d.confidence, tuple(round(v) for v in d.box))
                                             for d in last.detections))
    assert [d.track_id for d in last.attach_tracks(module_view).detections] == [1, 2]
    report = engine.to_wire(simulated=True)["tracking"]
    assert report["live"] == 2 and report["confirmed"] == 2 and report["updates"] == 10
    assert report["update_ms"] is not None and report["max_tracks"] == SHARED_MAX_TRACKS


def test_a_failed_detector_carries_no_tracks_and_ids_survive_a_short_failure():
    clock = [0.0]
    engine = pc.PerceptionEngine(clock=lambda: clock[0])
    detector = _Detector()
    detector.boxes = [(0, (100, 100, 200, 450), 0.9)]
    for i in range(3):
        clock[0] = i * 0.1
        _frame(engine, detector, i, clock[0], "m")
    detector.fail = True
    clock[0] = 0.5
    failed = _frame(engine, detector, 3, clock[0], "m")
    assert failed.detections is None and failed.tracks is None, "no boxes, no tracks, past a failure"
    detector.fail = False
    clock[0] = 1.0
    back = _frame(engine, detector, 4, clock[0], "m")
    assert [d.track_id for d in back.detections or ()] == [1], "a 0.6 s outage keeps the identity"


def test_the_tracker_runs_only_when_the_detector_runs():
    clock = [0.0]
    engine = pc.PerceptionEngine(clock=lambda: clock[0])
    detector = _Detector()
    detector.boxes = [(0, (100, 100, 200, 450), 0.9)]
    engine.set_detector_loader(lambda: detector)
    engine.demand(pc.Demand("node", pc.OBJECT_DETECTOR, frozenset({"person"}), min_interval_s=0.5))
    for i in range(20):
        clock[0] = i * 0.05
        obs = _frame(engine, detector, i, clock[0])
        if not obs.fresh_detections:
            assert obs.tracks is not None and obs.detections is not None, "carried while younger than max_age_s"
    assert detector.calls == engine.tracker.updates == 2, "1 s at a 0.5 s demand: two runs, two updates"
