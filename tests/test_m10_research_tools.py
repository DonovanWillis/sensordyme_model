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
from research.m10 import export_contract as ec


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


# ------------------------------------------------------------------ research/m10/export_contract.py
def _heads(class_channels: int = 81, end_ops=("Mul", "Sigmoid")):
    outs = []
    for g in (80, 40, 20):
        outs.append({"name": f"box{g}", "shape": [1, 68, g, g], "op": end_ops[0]})
        outs.append({"name": f"cls{g}", "shape": [1, class_channels, g, g], "op": end_ops[1]})
    return outs


def _graph(**kw):
    g = {"input": {"name": "images", "shape": [1, 3, 640, 640], "elem_type": "FLOAT"},
         "ops_before_first_conv": [], "outputs": _heads(),
         "class_convs": {"s0": {"channels": 81, "live_rows": [0, 67, 80], "dead_row_bias": [-20.0]}}}
    g.update(kw)
    return g


def _status(findings, item):
    (hit,) = [f for f in findings if f["item"] == item]
    return hit["status"]


def test_the_product_side_is_read_from_the_shipped_source():
    p = ec.product_contract()
    assert p["camera"]["picamera2_format"] == "RGB888" and p["camera"]["pi_read_converts_colour"] is False
    assert p["preprocess"]["colour_conversion"] == "BGR->RGB"
    assert p["preprocess"]["letterbox"] == {"scale": 1.0, "resized": [640, 480], "left": 0, "top": 80,
                                            "right": 0, "bottom": 80}
    assert p["preprocess"]["pad_value"] == [114, 114, 114] and p["preprocess"]["interpolation"] == "cv2.INTER_LINEAR"
    assert p["decode"]["reg_max"] == 16 and p["decode"]["nms_iou"] == 0.5
    assert p["decode"]["decoder_default"] == "legacy" and p["decode"]["extra_classes_configured"] is None
    assert p["hef"]["sha256_pinned"] == ec.PRODUCTION_HEF_SHA256


def test_a_clean_heads_graph_matches_the_device_contract():
    f = ec.compare(ec.product_contract(), {"exp08_heads81": _graph()}, None, None)
    assert _status(f, "colour: camera -> host") == "match"
    assert _status(f, "normalisation: exp08_heads81") == "match"
    assert _status(f, "outputs: exp08_heads81") == "match"
    assert _status(f, "DFL reg_max: exp08_heads81") == "match"
    assert _status(f, "head end ops: exp08_heads81") == "match"
    # the default product decode drops the hand channel: recorded as a mismatch until the opt-in map is set
    assert _status(f, "class map: exp08_heads81") == "MISMATCH unless opt-in map"


@pytest.mark.parametrize(("graph", "item"), [
    (_graph(ops_before_first_conv=["Sub", "Div"]), "normalisation: g"),
    (_graph(input={"name": "x", "shape": [1, 640, 640, 3], "elem_type": "FLOAT"}), "input tensor: g"),
    (_graph(outputs=_heads(end_ops=("Mul", "Identity"))), "head end ops: g"),
    (_graph(outputs=[o for o in _heads() if o["name"] != "cls20"]), "outputs: g"),
])
def test_contract_breaks_are_reported_as_findings(graph, item):
    f = ec.compare(ec.product_contract(), {"g": graph}, None, None)
    assert _status(f, item) == "MISMATCH"


def test_a_wrong_dfl_width_is_a_finding():
    outs = [dict(o, shape=[1, 64, *o["shape"][2:]]) if o["name"].startswith("box") else o for o in _heads()]
    f = ec.compare(ec.product_contract(), {"g": _graph(outputs=outs)}, None, None)
    assert _status(f, "outputs: g") == "MISMATCH"  # 64 channels is not a 68-ch box head


def test_colour_sensitivity_result_is_reported():
    rt = {"colour": {"m": {"rgb_as_shipped": {"person": 0.7}, "bgr_swapped": {"person": 0.6}, "rgb_better": True}}}
    f = ec.compare(ec.product_contract(), {}, None, rt)
    assert _status(f, "colour: m trained order") == "match"
    rt["colour"]["m"]["rgb_better"] = False
    assert _status(ec.compare(ec.product_contract(), {}, None, rt), "colour: m trained order") == "MISMATCH"


def test_head_roles_from_shapes():
    roles = ec.head_roles(_heads())
    assert sorted((r["role"], r["stride"]) for r in roles) == sorted(
        [("box", 8), ("class", 8), ("box", 16), ("class", 16), ("box", 32), ("class", 32)])
