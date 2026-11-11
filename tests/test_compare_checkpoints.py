"""The Checkpoint F collation: val only, one dataset manifest, run selection, high-score ECE."""

from __future__ import annotations

import json

import pytest

from research.pilot import compare_checkpoints as cc


def _cls(ap50):
    sweep = [{"threshold": t, "precision": 0.5 + t / 2, "recall": 0.9 - t, "negative_images_with_a_false_box": 0.5 - t / 2,
              "negative_images": 42, "fp_per_verified_image": 1.0} for t in (0.1, 0.2, 0.3, 0.5)]
    calib = {"ece": 0.1, "bins": [{"lo": 0.0, "hi": 0.1, "n": 100, "mean_score": 0.05, "precision": 0.0},
                                  {"lo": 0.3, "hi": 0.4, "n": 10, "mean_score": 0.35, "precision": 0.15},
                                  {"lo": 0.9, "hi": 1.0, "n": 30, "mean_score": 0.95, "precision": 0.95}]}
    return {"npos": 10, "ap50": ap50, "ap50_95": ap50 / 2, "sweep": sweep, "calibration": calib,
            "bootstrap": {"ap50_ci95": [ap50 - 0.1, ap50 + 0.1]}}


def _write(path, split="val", manifest="m1", runs=None):
    body = {"split": split, "manifest_id": manifest}
    body.update({"runs": runs} if runs else {"classes": {"phone": _cls(0.6), "hand": _cls(0.4)}})
    path.write_text(json.dumps(body))
    return path


def test_compare_collates_reports_and_selects_a_run(tmp_path):
    a = _write(tmp_path / "a.json", runs={"r1": {"classes": {"phone": _cls(0.8), "person": _cls(0.7)}, "infer_ms": {"p50": 3}}})
    b = _write(tmp_path / "b.json")
    result = cc.compare([f"pre={a}#r1", f"new={b}"])
    assert result["models"]["pre"]["classes"]["phone"]["ap50"] == 0.8
    assert "hand" not in result["models"]["pre"]["classes"] and "hand" in result["models"]["new"]["classes"]
    phone = result["models"]["new"]["classes"]["phone"]
    assert phone["best_precision_at_recall_ge_0.70"] == pytest.approx(0.6)  # recall 0.8 @0.1, 0.7 @0.2
    assert phone["@0.5"]["negative_images_with_a_false_box"] == pytest.approx(0.25)
    assert "—" in cc.markdown(result)


def test_high_score_ece_ignores_the_low_score_mass():
    assert cc.high_score_ece(_cls(0.5)["calibration"]) == pytest.approx((10 * 0.2 + 30 * 0.0) / 40)
    assert cc.high_score_ece({"bins": []}) is None


def test_the_test_split_and_mixed_manifests_are_refused(tmp_path):
    with pytest.raises(SystemExit, match="frozen"):
        cc.compare([f"x={_write(tmp_path / 't.json', split='test')}"])
    with pytest.raises(SystemExit, match="different dataset manifests"):
        cc.compare([f"a={_write(tmp_path / 'a.json')}", f"b={_write(tmp_path / 'b.json', manifest='m2')}"])
