"""M10 Task 3(b): architectures A / B / C at equal operating conditions, val split, one table.

    python -m research.m10.compare --data $SDJOB_DATA/openimages/oi7-sd-v1 --out DIR \
        --preds C_legacy=... --preds C_reference=... --preds A_exp08_reference=... [...] \
        --fuse B=C_reference:person+A_exp08_reference:phone,hand [--fuse ...] [--n 1000]

Reads prediction files only (JSON lines {id, label, score, box normalised xyxy}). No pixels.

Per model and class (Open Images protocol, research/detector_bench/oid_eval.py, the scorer every
earlier mission used): AP50 with a photographer-group bootstrap CI, recall at precision 0.80 and the
threshold that reaches it (each model/class gets its OWN precision-0.80 threshold, so operating
conditions are equal in precision, not in raw score), recall of LARGE (>= 2.25% of the image, the
driver case) plain person boxes at that threshold, and for phone the share of verified phone-free
images with at least one false phone at that threshold.

Fusion (Arch B) is exact and stated in the report: the fused file is the union, per image, of the
named classes' detections from each source, unchanged (scores, boxes). No cross-model suppression:
the sources contribute disjoint classes and each was already class-wise NMS'd at IoU 0.50.

Paired photographer-group bootstrap (same resampled groups for both models, --n resamples) of A - B
for: AP50, R@P0.80 (threshold re-derived per resample, as research/m8/analysis.py paired), large-person
recall at each model's full-sample precision-0.80 threshold (fixed), and the false-phone-image rate at
each model's full-sample phone precision-0.80 threshold (fixed).

The frozen test split is refused.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.detector_bench import oid_eval as oe
from research.m8.analysis import file_sha, load_preds, load_split, r_at_p

CLASSES = ("person", "phone", "hand")
TARGET_PRECISION = 0.80


# ------------------------------------------------------------------------------- fusion
def parse_fuse(spec: str) -> tuple[str, list[tuple[str, set[str]]]]:
    """'B=C_reference:person+A:phone,hand' -> ('B', [('C_reference', {'person'}), ('A', {'phone','hand'})])."""
    name, sep, rhs = spec.partition("=")
    if not sep or not name:
        raise ValueError(f"bad --fuse {spec!r}")
    parts = []
    seen: set[str] = set()
    for chunk in rhs.split("+"):
        src, sep, labels = chunk.partition(":")
        wanted = {x.strip() for x in labels.split(",") if x.strip()}
        if not sep or not src or not wanted:
            raise ValueError(f"bad --fuse part {chunk!r}")
        if wanted & seen:
            raise ValueError(f"--fuse {name}: class(es) {sorted(wanted & seen)} taken from two sources")
        seen |= wanted
        parts.append((src, wanted))
    return name, parts


def fuse(sources: dict[str, Path], parts: list[tuple[str, set[str]]], out: Path) -> dict:
    n = {}
    with out.open("w") as f:
        for src, labels in parts:
            n[src] = 0
            for line in sources[src].read_text().splitlines():
                if line.strip() and json.loads(line)["label"] in labels:
                    f.write(line.rstrip("\n") + "\n")
                    n[src] += 1
    return {"lines_from": n, "definition": {src: sorted(labels) for src, labels in parts}}


# ------------------------------------------------------------------------------- metrics
def _matched_at(result: oe.ClassResult, threshold: float) -> set:
    return {(s.image_id, s.matched) for s in result.scored if s.score >= threshold and s.tp and s.matched is not None}


def _flagged_at(result: oe.ClassResult, threshold: float) -> set:
    return {s.image_id for s in result.scored if s.score >= threshold and s.image_id in result.negative_ids}


def large_recall(result: oe.ClassResult, threshold: Optional[float], ids: Optional[list[str]] = None,
                 matched: Optional[set] = None) -> dict:
    """Recall of plain LARGE ground-truth boxes (area >= 2.25% of the image, oid_eval MEDIUM) at a fixed
    score threshold, over `ids` (with multiplicity; all images if None)."""
    if threshold is None:
        return {"recall": None, "n": 0, "hit": 0}
    if matched is None:
        matched = _matched_at(result, threshold)
    hit = tot = 0
    for image_id in (ids if ids is not None else list(result.gt_index)):
        for k, g in enumerate(x for x in result.gt_index.get(image_id, []) if not x.group_of):
            if g.size == "large":
                tot += 1
                hit += int((image_id, k) in matched)
    return {"recall": hit / tot if tot else None, "n": tot, "hit": hit}


def false_image_rate(result: oe.ClassResult, threshold: Optional[float], ids: Optional[list[str]] = None,
                     flagged: Optional[set] = None) -> dict:
    """Share of verified-negative images with >= 1 detection at/above threshold (every such box is false),
    over `ids` with multiplicity (all verified-negative images if None)."""
    seq = sorted(result.negative_ids) if ids is None else [i for i in ids if i in result.negative_ids]
    if threshold is None or not seq:
        return {"rate": None, "images": len(seq), "with_false": 0}
    if flagged is None:
        flagged = _flagged_at(result, threshold)
    k = sum(1 for i in seq if i in flagged)
    return {"rate": k / len(seq), "images": len(seq), "with_false": k}


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list[float]]:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def class_row(result: oe.ClassResult, label: str, n_boot: int) -> dict[str, Any]:
    rp = r_at_p(result.scored, result.npos, TARGET_PRECISION)
    row: dict[str, Any] = {
        "npos": result.npos, "verified_images": result.images, "negative_images": result.negative_images,
        "ap50": oe.average_precision(result.scored, result.npos),
        "r_at_p080": rp["recall"], "threshold_p080": rp["threshold"],
    }
    boot = oe.group_bootstrap(result, rp["threshold"] if rp["threshold"] is not None else 0.5, n=n_boot)
    row["ap50_ci95"] = boot.get("ap50_ci95") if boot else None
    if label == "person":
        row["large_recall_at_p080"] = large_recall(result, rp["threshold"])
    if label == "phone":
        fr = false_image_rate(result, rp["threshold"])
        row["false_phone_images_at_p080"] = {**fr, "wilson95": wilson(fr["with_false"], fr["images"])}
    return row


# ------------------------------------------------------------------------------- paired bootstrap
def paired(ra: oe.ClassResult, rb: oe.ClassResult, label: str, n: int, seed: int = 0) -> dict:
    """Paired group bootstrap of A - B. AP50 and R@P0.80 recomputed per resample; the large-person
    recall and false-phone-image rate at each model's FIXED full-sample precision-0.80 threshold."""
    ta = r_at_p(ra.scored, ra.npos, TARGET_PRECISION)["threshold"]
    tb = r_at_p(rb.scored, rb.npos, TARGET_PRECISION)["threshold"]
    by_group: dict[str, list[str]] = {}
    for image_id, group in ra.groups.items():
        by_group.setdefault(group, []).append(image_id)
    groups = sorted(by_group)
    per_a: dict[str, list] = {}
    per_b: dict[str, list] = {}
    for s in ra.scored:
        per_a.setdefault(s.image_id, []).append(s)
    for s in rb.scored:
        per_b.setdefault(s.image_id, []).append(s)
    npos_image = {i: len(g) for i, g in ra.gt_index.items()}

    cache = {id(r): (_matched_at(r, t) if t is not None else set(), _flagged_at(r, t) if t is not None else set())
             for r, t in ((ra, ta), (rb, tb))}

    def metrics(r: oe.ClassResult, per: dict, thr, ids: list[str]) -> dict:
        scored = [s for i in ids for s in per.get(i, [])]
        npos = sum(npos_image.get(i, 0) for i in ids)
        out = {"ap50": oe.average_precision(scored, npos) if npos else None,
               "r_at_p080": r_at_p(scored, npos, TARGET_PRECISION)["recall"] if npos else None}
        matched, flagged = cache[id(r)]
        if label == "person":
            out["large_recall_at_p080"] = large_recall(r, thr, ids, matched)["recall"]
        if label == "phone":
            out["false_phone_image_rate_at_p080"] = false_image_rate(r, thr, ids, flagged)["rate"]
        return out

    all_ids = [i for g in groups for i in by_group[g]]
    pa, pb = metrics(ra, per_a, ta, all_ids), metrics(rb, per_b, tb, all_ids)
    rng = random.Random(seed)  # noqa: S311 - resampling, not security
    deltas: dict[str, list[float]] = {k: [] for k in pa}
    for _ in range(n):
        ids = [i for g in rng.choices(groups, k=len(groups)) for i in by_group[g]]
        ma, mb = metrics(ra, per_a, ta, ids), metrics(rb, per_b, tb, ids)
        for k in deltas:
            if ma.get(k) is not None and mb.get(k) is not None:
                deltas[k].append(ma[k] - mb[k])
    out: dict[str, Any] = {"groups": len(groups), "threshold_a": ta, "threshold_b": tb}
    for k, raw in deltas.items():
        xs = sorted(raw)
        if not xs or pa.get(k) is None or pb.get(k) is None:
            continue
        out[k] = {"a": pa[k], "b": pb[k], "delta": pa[k] - pb[k],
                  "ci95": [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]],
                  "p_delta_le_0": sum(x <= 0 for x in xs) / len(xs), "n": len(xs)}
    return out


# ------------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preds", action="append", required=True, help="name=path")
    p.add_argument("--fuse", action="append", default=[], help="NAME=src:cls[,cls]+src:cls...")
    p.add_argument("--pair", action="append", default=[], help="A_name:B_name:class (A - B)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--boot", type=int, default=300, help="resamples for the per-model AP50 CI")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    manifest_id, _rows, truths, _groups, _hv = load_split(a.data, a.split)
    sources = {}
    for spec in a.preds:
        name, _, path = spec.partition("=")
        sources[name] = Path(path)
    report: dict[str, Any] = {"manifest_id": manifest_id, "split": a.split, "target_precision": TARGET_PRECISION,
                              "scorer": "research/detector_bench/oid_eval.py (Open Images protocol)",
                              "fusions": {}, "models": {}, "paired": {}}
    for spec in a.fuse:
        name, parts = parse_fuse(spec)
        path = a.out / f"fused_{name}_{a.split}.jsonl"
        report["fusions"][name] = fuse(sources, parts, path)
        sources[name] = path
    results: dict[str, dict[str, oe.ClassResult]] = {}
    for name, path in sources.items():
        preds = load_preds(path, set(CLASSES))
        results[name] = {}
        report["models"][name] = {"preds": str(path), "preds_sha16": file_sha(path), "classes": {}}
        for label in CLASSES:
            if label in preds:
                r = oe.evaluate_class(preds[label], truths[label])
                results[name][label] = r
                report["models"][name]["classes"][label] = class_row(r, label, a.boot)
        print(name, {k: (round(v["ap50"], 3) if v["ap50"] is not None else None, round(v["r_at_p080"], 3))
                     for k, v in report["models"][name]["classes"].items()}, flush=True)
    for spec in a.pair:
        na, nb, label = spec.split(":")
        if label not in results.get(na, {}) or label not in results.get(nb, {}):
            report["paired"][spec] = {"error": "class missing in one model"}
            continue
        report["paired"][spec] = paired(results[na][label], results[nb][label], label, a.n)
        print(spec, {k: (round(v["delta"], 4), [round(x, 4) for x in v["ci95"]])
                     for k, v in report["paired"][spec].items() if isinstance(v, dict)}, flush=True)
    (a.out / f"m10_compare_{a.split}.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
