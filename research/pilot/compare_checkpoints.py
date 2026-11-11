"""Side-by-side val comparison of detector checkpoints from existing score reports (EXP-08, Checkpoint F).

    python research/pilot/compare_checkpoints.py \
        --report pretrained-corrected=runs/.../bench_val_cuda.json#m5.t.reference@native \
        --report 2-epoch=runs/.../score_exp08a_rerun_val.json \
        --report best-60ep=runs/.../score_exp08_full60_val.json \
        --out comparison

Reads reports written by research/detector_bench (``bench score`` files, or one run of a
``bench run`` file selected with ``#<run key>``); computes nothing that needs images. Val only:
a report for any other split is refused. Writes comparison.json and comparison.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

CLASSES = ("phone", "person", "hand")
THRESHOLDS = (0.2, 0.3, 0.5)
MIN_RECALL = 0.70  # the EXP-08 success criterion: phone precision at recall >= 0.70


def load(spec: str) -> tuple[str, dict]:
    name, _, ref = spec.partition("=")
    path, _, run = ref.partition("#")
    data = json.loads(Path(path).read_text())
    if data.get("split") != "val":
        raise SystemExit(f"{name}: split {data.get('split')!r} is not val; the test split stays frozen")
    report = data["runs"][run] if run else data
    return name, {"manifest_id": data.get("manifest_id"), "source": ref, "classes": report["classes"],
                  "infer_ms": report.get("infer_ms"), "images_per_s": report.get("images_per_s")}


def _at(sweep: list[dict], threshold: float) -> Optional[dict]:
    return next((p for p in sweep if abs(p["threshold"] - threshold) < 1e-9), None)


def high_score_ece(calibration: dict, floor: float = 0.3) -> Optional[float]:
    """ECE over detections scoring >= floor: the full ECE is dominated by the many low-score boxes."""
    rows = [b for b in calibration.get("bins", []) if b["lo"] >= floor - 1e-9]
    n = sum(b["n"] for b in rows)
    return sum(b["n"] * abs(b["mean_score"] - b["precision"]) for b in rows) / n if n else None


def summarise(cls: dict) -> dict[str, Any]:
    sweep = cls.get("sweep", [])
    at_recall = [p for p in sweep if p["recall"] >= MIN_RECALL]
    out: dict[str, Any] = {
        "npos": cls.get("npos"), "ap50": cls.get("ap50"), "ap50_95": cls.get("ap50_95"),
        "ap50_ci95": (cls.get("bootstrap") or {}).get("ap50_ci95"),
        "ece": (cls.get("calibration") or {}).get("ece"),
        "ece_score_ge_0.3": high_score_ece(cls.get("calibration") or {}),
        "best_precision_at_recall_ge_0.70": max((p["precision"] for p in at_recall), default=None),
    }
    for t in THRESHOLDS:
        p = _at(sweep, t)
        if p:
            out[f"@{t}"] = {k: p.get(k) for k in ("precision", "recall", "negative_images_with_a_false_box",
                                                  "negative_images", "fp_per_verified_image")}
    return out


def compare(specs: list[str]) -> dict:
    reports = dict(load(s) for s in specs)
    manifests = {r["manifest_id"] for r in reports.values()}
    if len(manifests) != 1:
        raise SystemExit(f"reports use different dataset manifests: {sorted(map(str, manifests))}")
    return {"manifest_id": manifests.pop(), "split": "val",
            "models": {name: {"source": r["source"], "infer_ms": r["infer_ms"], "images_per_s": r["images_per_s"],
                              "classes": {c: summarise(r["classes"][c]) for c in CLASSES if c in r["classes"]}}
                       for name, r in reports.items()}}


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(result: dict) -> str:
    names = list(result["models"])
    lines = [f"Val split, manifest `{result['manifest_id']}`. — = class not produced by that model.", ""]
    for c in CLASSES:
        lines += [f"### {c}", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        rows = [("AP50", lambda s: _f(s["ap50"])), ("AP50:95", lambda s: _f(s["ap50_95"])),
                ("AP50 95% CI", lambda s: "—" if not s["ap50_ci95"] else f"{s['ap50_ci95'][0]:.3f} to {s['ap50_ci95'][1]:.3f}"),
                ("best precision at recall ≥ 0.70", lambda s: _f(s["best_precision_at_recall_ge_0.70"])),
                ("ECE (all / score ≥ 0.3)", lambda s: f"{_f(s['ece'])} / {_f(s['ece_score_ge_0.3'])}")]
        for t in THRESHOLDS:
            rows.append((f"P / R @{t}", lambda s, t=t: "—" if f"@{t}" not in s else
                         f"{_f(s[f'@{t}']['precision'])} / {_f(s[f'@{t}']['recall'])}"))
            rows.append((f"negatives with a false box @{t}", lambda s, t=t: "—" if f"@{t}" not in s else
                         f"{_f(s[f'@{t}']['negative_images_with_a_false_box'], 2)} (of {s[f'@{t}']['negative_images']})"))
        for label, fn in rows:
            cells = [fn(result["models"][n]["classes"][c]) if c in result["models"][n]["classes"] else "—" for n in names]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", action="append", required=True, help="name=path[#run key]")
    p.add_argument("--out", default=".")
    a = p.parse_args(argv)
    result = compare(a.report)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=1))
    (out / "comparison.md").write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
