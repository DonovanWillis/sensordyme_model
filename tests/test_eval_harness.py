"""Mission 4 Checkpoint B: the real-video evaluation harness (evaluation/).

The clips here are synthetic (a white square appears on a dark background):
the point is to pin the harness's arithmetic, determinism and guards, not any
model's accuracy. Real-video numbers come only from manifests of real data.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.events import Event
from evaluation import annotations as ann
from evaluation import manifest as mf
from evaluation.metrics import EventCounts, cluster_bootstrap, match_events, wilson
from evaluation.replay import replay
from evaluation.run import EvaluationRefusedError, evaluate
from modules.base import SustainedConditionTracker, VisionModule

FPS = 10


def _clip(path: Path, bright: list[tuple[int, int]], frames: int = 60) -> Path:
    """A 64x48 MJPG clip at 10 fps; frames in each [start, end) range show a white square."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (64, 48))
    for n in range(frames):
        frame = np.zeros((48, 64, 3), np.uint8)
        if any(a <= n < b for a, b in bright):
            frame[8:40, 16:48] = 255
        writer.write(frame)
    writer.release()
    return path


class SquareModule(VisionModule):
    """Fires 'square' after the square has been visible 0.5 s, timed with time.monotonic()."""

    name = "square_stub"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.tracker = SustainedConditionTracker(config.get("sustain_s", 0.5))
        self.visible = False

    def process(self, frame):
        self.visible = frame.mean() > 40
        fired = self.tracker.update(self.visible, time.monotonic())  # must be the replay clock
        return [] if fired is None else [Event.now("square", fired, 0.9, self.name)]

    def annotate(self, frame):
        return frame

    def active_alert_conditions(self) -> set[str]:
        return {"square"} if self.visible else set()


def _annotation(**doc) -> ann.Annotation:
    parsed, problems = ann.parse({"item": "x", **doc})
    assert parsed is not None, problems
    return parsed


# -- metrics -----------------------------------------------------------------------------

def test_event_matching_counts_caught_missed_false_alarms_repeats_and_latency():
    truth = _annotation(events=[{"type": "drowsy", "start_s": 10, "end_s": 20},
                                {"type": "drowsy", "start_s": 100, "end_s": 110}])
    predicted = [(12.5, "drowsy"), (15.0, "drowsy"), (21.0, "drowsy"), (50.0, "drowsy"), (13.0, "phone")]
    c = match_events(predicted, truth, "drowsy", duration_s=3600, late_s=2.0)
    assert (c.gt, c.caught, c.false_alarms, c.repeats, c.predictions) == (2, 1, 1, 2, 4)
    assert c.latencies == [2.5]
    assert c.recall == 0.5 and c.precision == 0.5 and c.false_alarms_per_hour == pytest.approx(1.0)
    assert c.alerts_per_hour == pytest.approx(4.0)


def test_ignore_spans_are_neither_errors_nor_hours():
    truth = _annotation(events=[], ignore=[{"start_s": 0, "end_s": 1800}, {"start_s": 900, "end_s": 2700}])
    c = match_events([(100.0, "phone"), (3000.0, "phone")], truth, "phone", duration_s=3600)
    assert c.false_alarms == 1 and c.hours == pytest.approx(0.25)
    assert c.recall is None, "no ground truth: recall is undefined, not 0 or 1"


def test_wilson_and_subject_bootstrap_intervals():
    interval = wilson(8, 10)
    assert interval is not None
    lo, hi = interval
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert wilson(0, 0) is None
    per_subject = {"a": EventCounts(gt=10, caught=9, hours=1), "b": EventCounts(gt=10, caught=5, hours=1),
                   "c": EventCounts(gt=10, caught=7, hours=1)}
    first = cluster_bootstrap(per_subject, lambda c: c.recall, n=500, seed=3)
    assert first == cluster_bootstrap(per_subject, lambda c: c.recall, n=500, seed=3), "seeded: reproducible"
    assert first is not None and first[0] <= 0.7 <= first[1] and first[0] >= 0.5 and first[1] <= 0.9
    assert cluster_bootstrap({"a": per_subject["a"]}, lambda c: c.recall) is None, "one subject: no interval"


def test_annotation_problems_are_reported_not_guessed():
    parsed, problems = ann.parse({"item": "x", "events": [{"type": "drowsy", "start_s": 5, "end_s": 2},
                                                          {"start_s": 1, "end_s": 2},
                                                          {"type": "nap", "start_s": 1, "end_s": 2}]},
                                 known_types={"drowsy"})
    assert parsed is None
    assert any("start_s < end_s" in p for p in problems)
    assert any("no type" in p for p in problems)
    assert any("unknown event type 'nap'" in p for p in problems)


# -- manifests ---------------------------------------------------------------------------

def _manifest(items: list[mf.Item], **kw) -> mf.Manifest:
    licence = kw.pop("licence", {"class": "own_collection", "id": "SD consent v1"})
    consent = kw.pop("consent", {"protocol": "SD-CONSENT-1", "retention_until": "2099-01-01"})
    return mf.Manifest(name="t", version="1", licence=licence, items=items, consent=consent, **kw)


def _item(n: int, subject: str, split=None, session=None) -> mf.Item:
    return mf.Item(id=f"i{n}", path=f"v/{n}.avi", sha256=f"{n:064x}", bytes=1, subject=subject,
                   session=session or f"{subject}-s", split=split)


def test_splits_are_by_subject_deterministic_and_never_move_existing_items():
    items = [_item(n, f"s{n % 7}") for n in range(40)]
    items[0].split = "test"  # subject s0 is already held out
    a = _manifest(items)
    mf.assign_splits(a, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    assert items[0].split == "test"
    by_subject = {}
    for i in a.items:
        by_subject.setdefault(i.subject, set()).add(i.split)
    assert all(len(s) == 1 for s in by_subject.values()), "a subject never spans splits"
    assert by_subject["s0"] == {"test"}
    b = _manifest([_item(n, f"s{n % 7}") for n in range(40)])
    b.items[0].split = "test"
    mf.assign_splits(b, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    assert [i.split for i in a.items] == [i.split for i in b.items] and a.id == b.id
    assert mf.validate(a) == []


def test_leakage_duplicates_licence_and_retention_are_refused():
    leaky = _manifest([_item(1, "s1", "train"), _item(2, "s1", "test")])
    assert any("leakage: subject 's1'" in p for p in mf.validate(leaky))
    dup = _manifest([_item(1, "s1", "train"), _item(2, "s2", "val")])
    dup.items[1].sha256 = dup.items[0].sha256
    assert any("same file" in p for p in mf.validate(dup))
    expired = _manifest([_item(1, "s1", "val")], consent={"protocol": "p", "retention_until": "2026-01-01"})
    assert any("retention ended" in p for p in mf.validate(expired, today=dt.date(2026, 11, 11)))
    no_consent = _manifest([_item(1, "s1", "val")], consent={})
    assert any("consent.protocol" in p for p in mf.validate(no_consent))
    research = _manifest([_item(1, "s1", "val")], licence={"class": "research_only", "id": "x"})
    assert not research.usable_for("evaluation") and not research.usable_for("training")
    evaluation_only = _manifest([_item(1, "s1", "val")], licence={"class": "evaluation_only", "id": "x"})
    assert evaluation_only.usable_for("evaluation") and not evaluation_only.usable_for("training")


def test_files_that_changed_or_do_not_decode_are_caught(tmp_path):
    good = _clip(tmp_path / "good.avi", [(0, 5)], frames=10)
    (tmp_path / "bad.avi").write_bytes(b"not a video at all")
    m = mf.build(tmp_path, [{"id": "g", "path": "good.avi", "subject": "a", "session": "a1", "split": "val"},
                            {"id": "b", "path": "bad.avi", "subject": "b", "session": "b1", "split": "val"}],
                 name="t", version="1", licence={"class": "evaluation_only", "id": "x"})
    problems = mf.verify_files(m, tmp_path)
    assert problems == ["b: video does not decode"]
    with good.open("ab") as f:
        f.write(b"\0")
    assert any("changed since the manifest" in p for p in mf.verify_files(m, tmp_path))


def test_near_duplicates_across_splits_are_found(tmp_path):
    _clip(tmp_path / "a.avi", [(0, 60)])
    _clip(tmp_path / "b.avi", [(0, 60)], frames=50)  # a re-cut of the same footage
    _clip(tmp_path / "c.avi", [])
    rows = [{"id": n, "path": f"{n}.avi", "subject": n, "session": n, "split": s}
            for n, s in (("a", "train"), ("b", "test"), ("c", "test"))]
    m = mf.build(tmp_path, rows, name="t", version="1", licence={"class": "evaluation_only", "id": "x"})
    assert mf.near_duplicates(m, tmp_path) == ["near-duplicate across splits: a (train) ~ b (test)"]


# -- replay ------------------------------------------------------------------------------

def test_replay_is_deterministic_and_runs_on_the_clips_clock(tmp_path):
    clip = str(_clip(tmp_path / "c.avi", [(10, 30), (40, 45)]))
    wall_before = time.monotonic()
    first = replay(lambda: SquareModule({}), clip, resize=None)
    second = replay(lambda: SquareModule({}), clip, resize=None)
    assert [(e.t, e.type) for e in first.events] == [(1.5, "square")], "square at 1.0 s + 0.5 s sustain"
    assert first.events == second.events and first.times == second.times
    assert first.duration_s == pytest.approx(6.0) and first.frames_read == 60
    assert time.monotonic() >= wall_before, "the real clock is restored after the replay"
    half = replay(lambda: SquareModule({}), clip, resize=None, target_fps=5)
    assert half.frames_read == 30 and half.times[:3] == [0.0, 0.2, 0.4]
    assert [(e.t, e.type) for e in half.events] == [(1.6, "square")], "processing rate changes latency, honestly"


def test_a_slow_host_does_not_change_the_result(tmp_path):
    clip = str(_clip(tmp_path / "c.avi", [(10, 30)]))

    class Slow(SquareModule):
        def process(self, frame):
            time.sleep(0.002)
            return super().process(frame)

    assert [e.t for e in replay(lambda: Slow({}), clip, resize=None).events] == [1.5]


# -- end to end --------------------------------------------------------------------------

def _dataset(tmp_path: Path) -> mf.Manifest:
    rows = []
    # (subject, split, frames showing the square, of which annotated as real events)
    clips = [("s1", "val", [(10, 30), (50, 58)], [(10, 30)]),  # the second square is not a real event
             ("s2", "val", [(40, 50)], [(40, 50)]),
             ("s3", "test", [(5, 25)], [(5, 25)])]
    for n, (subject, split, bright, real) in enumerate(clips):
        _clip(tmp_path / f"{n}.avi", bright)
        (tmp_path / f"{n}.json").write_text(json.dumps({
            "item": f"c{n}", "events": [{"type": "square", "start_s": a / FPS, "end_s": b / FPS} for a, b in real],
            "states": {"square": [[b[0] / FPS, (b[1] - 1) / FPS] for b in bright]},
            "ignore": [{"start_s": 0.0, "end_s": 0.2, "reason": "warm-up"}]}))
        rows.append({"id": f"c{n}", "path": f"{n}.avi", "subject": subject, "session": f"{subject}-1",
                     "split": split, "annotations": f"{n}.json", "cond_lighting": "day" if n else "night"})
    return mf.build(tmp_path, rows, name="synthetic", version="1",
                    licence={"class": "own_collection", "id": "test"},
                    consent={"protocol": "SD-CONSENT-1", "retention_until": "2099-01-01"})


def test_evaluate_scores_events_frames_and_strata(tmp_path):
    m = _dataset(tmp_path)
    report = evaluate(m, tmp_path, "val", lambda: SquareModule({}), resize=None)
    square = report["events"]["square"]
    # Each real square fires 0.5 s after it appears; the unannotated one in clip 0 is a false alarm.
    assert (square["gt"], square["caught"], square["false_alarms"]) == (2, 2, 1)
    assert square["latency_p50_s"] == pytest.approx(0.5)
    assert square["false_alarms_per_hour"] == pytest.approx(1 / ((12.0 - 0.4) / 3600))
    assert set(square["strata"]) == {"lighting=day", "lighting=night"}
    frames = report["frame_states"]["square"]
    assert frames["fp"] == 0 and frames["fn"] <= 2 and frames["tp"] > 20
    assert report["subjects"] == ["s1", "s2"]


def test_the_test_split_is_held_out_and_never_tuned_on(tmp_path):
    m = _dataset(tmp_path)
    with pytest.raises(EvaluationRefusedError, match="held out"):
        evaluate(m, tmp_path, "test", lambda: SquareModule({}))
    with pytest.raises(EvaluationRefusedError, match="tune on val"):
        evaluate(m, tmp_path, "test", lambda: SquareModule({}), frozen_test=True, tuning=True)
    assert evaluate(m, tmp_path, "test", lambda: SquareModule({}), frozen_test=True, resize=None)["subjects"] == ["s3"]


def test_unlicensed_or_tampered_data_is_refused(tmp_path):
    m = _dataset(tmp_path)
    m.licence = {"class": "research_only", "id": "x"}
    with pytest.raises(EvaluationRefusedError, match="does not permit"):
        evaluate(m, tmp_path, "val", lambda: SquareModule({}))
    m = _dataset(tmp_path)
    (tmp_path / "0.avi").write_bytes(b"x")
    with pytest.raises(EvaluationRefusedError, match="changed since"):
        evaluate(m, tmp_path, "val", lambda: SquareModule({}))
