"""Print and tabulate (CSV) the B2 characterisation JSON written by research.m84.characterise.

    python -m research.m84.summarise_b2 --slices b2_slices_val.json --errors b2_errors_val.json --out-dir DIR

Writes b2_luma_quartiles.csv, b2_attributes.csv, b2_fp_anatomy.csv and b2_top_fps.csv (small files
for docs/research/M84_MODEL_AUDIT.md), and prints the same tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

QS = ("luma_lt40", "luma_lt60", "Q1_dark", "Q2", "Q3", "Q4_bright")


def _w(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n# {path.name}")
    print(" | ".join(header))
    for r in rows:
        print(" | ".join("" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in r))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slices", required=True)
    p.add_argument("--errors", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads(Path(a.slices).read_text())
    e = json.loads(Path(a.errors).read_text())
    rows = []
    for m, md in s["models"].items():
        for c, cd in md["classes"].items():
            o = cd["overall"]
            row = [m, c, o["ap50"], *(o["ci95"] or [None, None]), cd["threshold_at_p0.80"]]
            for q in QS:
                qd = cd["ap50_by_luma_quartile"].get(q, {"ap50": None, "ci95": None, "npos": None})
                rq = cd["recall_at_p0.80_by_luma_quartile"].get(q, {})
                row += [qd["ap50"], *(qd["ci95"] or [None, None]), qd["npos"], rq.get("recall")]
            rows.append(row)
    hdr = ["model", "class", "ap50", "ci_lo", "ci_hi", "thr_p0.80"]
    for q in QS:
        hdr += [f"{q}_ap50", f"{q}_lo", f"{q}_hi", f"{q}_npos", f"{q}_recall_p0.80"]
    _w(out / "b2_luma_quartiles.csv", hdr, rows)
    print("luma quartile edges (mean Rec.601 luma, 0-255):", s["luma_quartile_edges"])
    rows = []
    keys = ["all", "size_large", "size_medium", "size_small", "occluded", "truncated", "depiction"]
    for m, md in s["models"].items():
        for c, cd in md["classes"].items():
            ra = cd["recall_at_p0.80_by_attribute"]
            rows.append([m, c, *[ra.get(k, {}).get("recall") for k in keys], *[ra.get(k, {}).get("n") for k in keys]])
    _w(out / "b2_attributes.csv", ["model", "class", *[f"R@P0.8_{k}" for k in keys], *[f"n_{k}" for k in keys]], rows)
    rows, top = [], []
    cats = ["localisation", "duplicate", "cross_class_person", "cross_class_phone", "cross_class_hand",
            "verified_absent_image", "background"]
    for m, md in e["models"].items():
        for c, cd in md["classes"].items():
            rows.append([m, c, cd["threshold"], cd["fp_total"], *[cd["fp_categories"].get(k, 0) for k in cats],
                         "; ".join(f"{k} {v}" for k, v in list(cd["lookalike_counts"].items())[:6])])
            for fp in cd["top_false_positives"][:15]:
                top.append([m, c, fp["id"], fp["score"], fp["category"], fp["same_class_iou"], fp["lookalike"], fp["luma_mean"]])
    _w(out / "b2_fp_anatomy.csv", ["model", "class", "threshold", "fp_total", *cats, "lookalikes"], rows)
    with (out / "b2_top_fps.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "image_id", "score", "category", "same_class_iou", "lookalike", "luma_mean"])
        w.writerows(top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
