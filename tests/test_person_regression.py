"""The M7 detector analysis: PR-curve operating points, failure attribution and per-class merging."""

from __future__ import annotations

import json

import pytest

from research.detector_bench import oid_eval as oe
from research.pilot import person_regression as pr

A, B, C = (0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7), (0.0, 0.6, 0.2, 0.9)


def _truths():
    return [
        oe.ImageTruth("i1", "g1", [oe.GroundTruth(A), oe.GroundTruth(B, occluded=True)], True, False),
        oe.ImageTruth("i2", "g2", [oe.GroundTruth(A)], True, False),
        oe.ImageTruth("i3", "g3", [], True, True),  # verified person-free
    ]


def _result():
    preds = {"i1": [(0.9, A), (0.4, B)], "i2": [(0.8, A)], "i3": [(0.6, C)]}
    return oe.evaluate_class(preds, _truths())


def test_operating_points_follow_the_pr_curve():
    r = _result()  # order: 0.9 TP, 0.8 TP, 0.6 FP, 0.4 TP  (npos 3)
    assert pr.precision_at_recall(r, 0.6) == {"precision": 1.0, "threshold": 0.8}
    p = pr.precision_at_recall(r, 1.0)
    assert p["threshold"] == 0.4 and p["precision"] == pytest.approx(0.75)
    assert pr.recall_at_precision(r, 0.8) == {"recall": pytest.approx(2 / 3), "threshold": 0.8}
    assert pr.recall_at_precision(r, 0.7)["recall"] == pytest.approx(1.0)
    assert pr.precision_at_recall(oe.evaluate_class({}, _truths()), 0.5) == {"precision": None, "threshold": None}


def test_failure_attribution():
    r = _result()
    missed = pr._missed(r, 0.5)
    assert missed == {("i1", 1)}  # the occluded person found only at 0.4
    d = pr.describe_misses(r, missed, hands={"i1": [B]})
    assert d["count"] == 1 and d["occluded"] == 1.0 and d["overlaps_a_hand_box"] == 1.0
    fp = pr.describe_false_positives(r, 0.5, hands={"i3": [C]}, boxes={"i3": [(0.6, C)]})
    assert fp == {"count": 1, "localisation": 0.0, "duplicate": 0.0, "misplaced": 0.0, "background": 1.0,
                  "background_on_a_hand_box": 1.0, "on_negative_images": 1}


def test_duplicates_are_separated_from_misplaced_boxes():
    shifted = (0.1, 0.15, 0.3, 0.35)  # IoU 0.6 with A
    off = (0.2, 0.2, 0.4, 0.4)  # IoU ~0.14 with A
    r = oe.evaluate_class({"i2": [(0.9, A), (0.8, shifted), (0.7, off)]}, _truths()[1:2])
    fp = pr.describe_false_positives(r, 0.5, hands={}, boxes={})
    assert fp["count"] == 2 and fp["duplicate"] == 0.5 and fp["misplaced"] == 0.5 and fp["background"] == 0.0


def test_merge_takes_each_class_from_its_source(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = lambda label: [json.dumps({"id": "i", "label": l, "score": 0.5, "box": list(A)}) for l in label]  # noqa: E731, E741
    a.write_text("\n".join(rows(["person", "phone", "hand"])) + "\n")
    b.write_text("\n".join(rows(["person", "hand"])) + "\n")
    out = tmp_path / "m.jsonl"
    assert pr.merge([f"person,phone={a}", f"hand={b}"], out) == 3
    labels = [json.loads(x)["label"] for x in out.read_text().splitlines()]
    assert sorted(labels) == ["hand", "person", "phone"]
    assert pr.load_preds(out)["hand"] == {"i": [(0.5, A)]}
