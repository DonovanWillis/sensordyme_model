"""M10 research tooling that runs without models or data: research/m10/compare.py helpers.

The fleet jobs (research/m10/oryn_m10.sh) feed these real prediction files; here they get small
hand-built ground truth and predictions whose answers are known by counting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.detector_bench import oid_eval as oe
from research.m10 import compare as cmp


def _result(preds: dict, truths: list[oe.ImageTruth]) -> oe.ClassResult:
    return oe.evaluate_class(preds, truths)


def _truths():
    big = oe.GroundTruth((0.0, 0.0, 0.5, 0.5))          # area 0.25 -> large
    small = oe.GroundTruth((0.0, 0.0, 0.04, 0.04))      # area 0.0016 -> small
    return [
        oe.ImageTruth("a", "g1", [big], True, False),
        oe.ImageTruth("b", "g2", [small], True, False),
        oe.ImageTruth("n1", "g3", [], True, True),       # verified negative
        oe.ImageTruth("n2", "g4", [], True, True),
    ]


def test_parse_fuse_and_refuses_overlap():
    name, parts = cmp.parse_fuse("B=C_reference:person+A:phone,hand")
    assert name == "B" and parts == [("C_reference", {"person"}), ("A", {"phone", "hand"})]
    with pytest.raises(ValueError, match="two sources"):
        cmp.parse_fuse("B=C:person,phone+A:phone")
    with pytest.raises(ValueError, match="bad --fuse"):
        cmp.parse_fuse("B=C")


def test_fuse_takes_each_class_from_its_source_unchanged(tmp_path: Path):
    c, a = tmp_path / "c.jsonl", tmp_path / "a.jsonl"
    c.write_text("\n".join(json.dumps(d) for d in [
        {"id": "x", "label": "person", "score": 0.9, "box": [0, 0, 1, 1]},
        {"id": "x", "label": "phone", "score": 0.8, "box": [0, 0, 1, 1]}]) + "\n")
    a.write_text("\n".join(json.dumps(d) for d in [
        {"id": "x", "label": "person", "score": 0.1, "box": [0, 0, 1, 1]},
        {"id": "x", "label": "phone", "score": 0.7, "box": [0, 0, 0.5, 0.5]},
        {"id": "x", "label": "hand", "score": 0.6, "box": [0, 0, 0.2, 0.2]}]) + "\n")
    out = tmp_path / "b.jsonl"
    rep = cmp.fuse({"C": c, "A": a}, cmp.parse_fuse("B=C:person+A:phone,hand")[1], out)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [(r["label"], r["score"]) for r in rows] == [("person", 0.9), ("phone", 0.7), ("hand", 0.6)]
    assert rep["lines_from"] == {"C": 1, "A": 2}


def test_large_recall_and_false_image_rate_count_what_they_say():
    preds = {"a": [(0.9, (0.0, 0.0, 0.5, 0.5))], "b": [(0.95, (0.0, 0.0, 0.04, 0.04))],
             "n1": [(0.6, (0.1, 0.1, 0.2, 0.2))]}
    r = _result(preds, _truths())
    assert cmp.large_recall(r, 0.5) == {"recall": 1.0, "n": 1, "hit": 1}
    assert cmp.large_recall(r, 0.92) == {"recall": 0.0, "n": 1, "hit": 0}   # only the small box survives
    assert cmp.false_image_rate(r, 0.5) == {"rate": 0.5, "images": 2, "with_false": 1}
    assert cmp.false_image_rate(r, 0.7)["with_false"] == 0
    # resampled ids keep multiplicity
    assert cmp.false_image_rate(r, 0.5, ["n1", "n1", "n2"])["rate"] == pytest.approx(2 / 3)
    assert cmp.large_recall(r, 0.5, ["a", "a", "b"]) == {"recall": 1.0, "n": 2, "hit": 2}


def test_wilson_interval():
    assert cmp.wilson(0, 0) is None
    ci = cmp.wilson(1, 42)
    assert ci is not None
    lo, hi = ci
    assert 0.0 < lo < 1 / 42 < hi < 0.15


def test_paired_bootstrap_of_a_model_against_itself_is_zero():
    preds = {"a": [(0.9, (0.0, 0.0, 0.5, 0.5))], "b": [(0.95, (0.0, 0.0, 0.04, 0.04))],
             "n1": [(0.6, (0.1, 0.1, 0.2, 0.2))]}
    r = _result(preds, _truths())
    out = cmp.paired(r, r, "person", n=50)
    for key in ("ap50", "r_at_p080", "large_recall_at_p080"):
        assert out[key]["delta"] == 0 and out[key]["ci95"] == [0, 0]


def test_class_row_reports_the_precision_080_operating_point():
    preds = {"a": [(0.9, (0.0, 0.0, 0.5, 0.5))], "b": [(0.95, (0.0, 0.0, 0.04, 0.04))],
             "n1": [(0.6, (0.1, 0.1, 0.2, 0.2))]}
    row = cmp.class_row(_result(preds, _truths()), "phone", n_boot=20)
    assert row["r_at_p080"] == 1.0 and row["threshold_p080"] == 0.9
    assert row["false_phone_images_at_p080"]["with_false"] == 0 and row["npos"] == 2
