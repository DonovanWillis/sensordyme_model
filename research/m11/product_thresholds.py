"""M11 Phase D: what the PRODUCT's raw thresholds mean for each decoder, and the precision-matched equivalent.

No training and no new inference: it re-scores the stored device-path predictions of M10 job
`20261114-034527-625fa0` (the shipping .hef's heads through core.accel's legacy and reference decoders, and
EXP-08's reference path) with the Open Images scorer, val split only.

Why: the product gates detections with raw scores (`driver_attention.person_confidence_threshold` 0.50 for
person; `yolo_confidence_threshold` 0.20 for every other class, phone included). The two decoders produce
different score distributions (legacy: anchor offset 0.5, argmax; reference: offset 0, multilabel), so the
same 0.50 is a DIFFERENT operating point under each. Switching the decoder with thresholds unchanged would
silently move precision and recall. This report gives, per model and class:

  * the operating point at the product threshold (precision, recall, false boxes per image, localisation vs
    background false positives, recall by size/occlusion/truncation, large-person recall, false-phone images);
  * the precision-0.80 operating point (M10's matched-precision comparison);
  * for each candidate, the threshold that MATCHES the shipping legacy model's precision at the product
    threshold, and the recall it buys there (the threshold to propose IF the candidate passes INT8 parity --
    float scores; INT8 must re-derive it on the Pi).

    python -m research.m11.product_thresholds --data $OI --out $OUT \\
        --preds C_legacy=.../zoo_legacy_val.jsonl --preds C_reference=.../zoo_reference_val.jsonl \\
        --preds A_exp08_reference=.../exp08_reference_val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from research.detector_bench import oid_eval as oe
from research.m8.analysis import file_sha, load_preds, load_split, r_at_p
from research.m10.compare import false_image_rate, large_recall, wilson

#: the product's raw-score gates (config.yaml modules.driver_attention), per class
PRODUCT_THRESHOLDS = {"person": 0.50, "phone": 0.20, "hand": 0.20}
TARGET_PRECISION = 0.80
BASELINE = "C_legacy"


def _round(x: Any, k: int = 4) -> Any:
    return round(x, k) if isinstance(x, float) else x


def point(result: oe.ClassResult, label: str, threshold: Optional[float]) -> Optional[dict]:
    if threshold is None:
        return None
    op = oe.operating_point(result, threshold)
    out = {k: _round(v) for k, v in op.items() if k != "recall_by_attribute"}
    out["recall_by_attribute"] = {k: {"recall": _round(v["recall"]), "n": v["n"]}
                                  for k, v in op["recall_by_attribute"].items()}
    if label == "person":
        out["large_person_recall"] = large_recall(result, threshold)
    if label == "phone":
        fr = false_image_rate(result, threshold)
        out["false_phone_images"] = {**fr, "wilson95": wilson(fr["with_false"], fr["images"])}
    return out


def matched_to(result: oe.ClassResult, precision: Optional[float]) -> dict:
    """The highest-recall threshold whose precision is at least `precision` (inclusive, as the device)."""
    if precision is None:
        return {"threshold": None, "recall": None}
    rp = r_at_p(result.scored, result.npos, precision)
    return {"target_precision": _round(precision), "threshold": _round(rp["threshold"]), "recall": _round(rp["recall"])}


def build(results: dict[str, dict[str, oe.ClassResult]]) -> dict:
    report: dict = {"product_thresholds": PRODUCT_THRESHOLDS, "baseline": BASELINE, "models": {}}
    base = results.get(BASELINE, {})
    for name, classes in results.items():
        report["models"][name] = {}
        for label, r in classes.items():
            prod = point(r, label, PRODUCT_THRESHOLDS.get(label))
            rp = r_at_p(r.scored, r.npos, TARGET_PRECISION)
            row = {"npos": r.npos, "ap50": _round(oe.average_precision(r.scored, r.npos)),
                   "at_product_threshold": prod,
                   "at_precision_080": {"threshold": _round(rp["threshold"]), "recall": _round(rp["recall"]),
                                        "detail": point(r, label, rp["threshold"])}}
            if name != BASELINE and label in base:
                base_prod = point(base[label], label, PRODUCT_THRESHOLDS.get(label))
                target = (base_prod or {}).get("precision")
                m = matched_to(r, target)
                row["matched_to_shipping_at_product"] = {
                    **m, "shipping_precision": _round(target),
                    "shipping_recall": (base_prod or {}).get("recall"),
                    "recall_gain": _round(m["recall"] - base_prod["recall"])
                    if m["recall"] is not None and base_prod and base_prod.get("recall") is not None else None,
                    "detail": point(r, label, m["threshold"]) if m["threshold"] is not None else None}
            report["models"][name][label] = row
    return report


def markdown(report: dict) -> str:
    lines = ["# Product thresholds vs matched precision (float, Open Images val; NOT INT8, NOT in-cabin)", "",
             f"Product gates: {report['product_thresholds']}. Baseline: `{report['baseline']}` (what ships).", "",
             ("| Model | Class | AP50 | @product thr: P / R | false box / img | FP loc / bg | large-person R or "
              "false-phone imgs | @P0.80: thr / R | matched to shipping@product: thr / R (gain) |"),
             "|---|---|---|---|---|---|---|---|---|"]
    for name, classes in report["models"].items():
        for label, row in classes.items():
            p = row["at_product_threshold"] or {}
            extra = ""
            if "large_person_recall" in p:
                extra = f"{_round(p['large_person_recall']['recall'], 3)}"
            elif "false_phone_images" in p:
                fp = p["false_phone_images"]
                extra = f"{fp['with_false']}/{fp['images']}"
            m = row.get("matched_to_shipping_at_product") or {}
            mt = (f"{m.get('threshold')} / {m.get('recall')} ({m.get('recall_gain'):+})"
                  if m.get("threshold") is not None and m.get("recall_gain") is not None else "—")
            lines.append(
                f"| {name} | {label} | {row['ap50']} | {p.get('precision')} / {p.get('recall')} | "
                f"{p.get('fp_per_verified_image')} | {p.get('fp_localisation')} / {p.get('fp_background')} | {extra} | "
                f"{row['at_precision_080']['threshold']} / {row['at_precision_080']['recall']} | {mt} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preds", action="append", required=True, help="name=path")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_id, _rows, truths, _groups, _hv = load_split(a.data, a.split)
    results: dict[str, dict[str, oe.ClassResult]] = {}
    sources: dict[str, str] = {}
    for spec in a.preds:
        name, _, path = spec.partition("=")
        preds = load_preds(Path(path), set(PRODUCT_THRESHOLDS))
        sources[name] = file_sha(Path(path))
        results[name] = {label: oe.evaluate_class(preds[label], truths[label])
                         for label in PRODUCT_THRESHOLDS if label in preds and label in truths}
    report = build(results)
    report.update({"manifest_id": manifest_id, "split": a.split, "preds_sha16": sources,
                   "scorer": "research/detector_bench/oid_eval.py (Open Images protocol)",
                   "evidence_level": "L1 float, Open Images val; not Hailo INT8, not in-cabin, not NoIR"})
    (a.out / "product_thresholds.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (a.out / "product_thresholds.md").write_text(markdown(report))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
