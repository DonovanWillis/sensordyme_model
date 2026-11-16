"""M11 Phase D: the product-threshold report (research/m11/product_thresholds.py) on a constructed truth set.

Pins that the product's raw gates are read per class, that the matched-to-shipping threshold keeps at least
the shipping model's precision at the product gate (inclusive, as the device), and that a candidate whose
scores run higher is reported at a different operating point under the SAME raw threshold.
"""

from __future__ import annotations

import pytest

from research.detector_bench import oid_eval as oe
from research.m11 import product_thresholds as pt

BOX = (0.1, 0.1, 0.6, 0.9)  # large: area 0.4


def _truths(n: int) -> list[oe.ImageTruth]:
    return [oe.ImageTruth(image_id=f"i{k}", group=f"g{k}", gts=[oe.GroundTruth(BOX)], verified=True, negative=False)
            for k in range(n)]


def _preds(true_scores: list[float], false_scores: list[float]) -> dict[str, list[tuple[float, tuple]]]:
    preds: dict[str, list[tuple[float, tuple]]] = {}
    for k, s in enumerate(true_scores):
        preds.setdefault(f"i{k}", []).append((s, BOX))
    for k, s in enumerate(false_scores):  # background boxes, far from the truth
        preds.setdefault(f"i{k}", []).append((s, (0.80, 0.80, 0.95, 0.95)))
    return preds


def test_product_thresholds_are_the_configured_gates():
    assert pt.PRODUCT_THRESHOLDS == {"person": 0.50, "phone": 0.20, "hand": 0.20}


def test_matched_threshold_keeps_shipping_precision_and_reports_the_shift():
    truths = _truths(10)
    # shipping (legacy-like): 6 of 10 people above 0.5, 1 false box above 0.5 -> P = 6/7 at the gate
    legacy = oe.evaluate_class(_preds([0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.4, 0.3, 0.2, 0.1], [0.51, 0.3]), truths)
    # candidate (reference-like): scores run higher -> the same 0.50 admits more false boxes
    reference = oe.evaluate_class(_preds([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.45, 0.3],
                                         [0.62, 0.58, 0.55]), truths)
    report = pt.build({"C_legacy": {"person": legacy}, "C_reference": {"person": reference}})
    base = report["models"]["C_legacy"]["person"]["at_product_threshold"]
    cand = report["models"]["C_reference"]["person"]
    assert base["precision"] == pytest.approx(6 / 7, abs=1e-4)
    assert cand["at_product_threshold"]["fp"] == 3          # same raw 0.50, more false boxes
    m = cand["matched_to_shipping_at_product"]
    assert m["detail"]["precision"] >= base["precision"] - 1e-9
    assert m["threshold"] > 0.50 and m["recall_gain"] > 0  # higher threshold, still more recall at equal precision
    assert "C_legacy" in pt.markdown(report) and "matched to shipping" in pt.markdown(report)
