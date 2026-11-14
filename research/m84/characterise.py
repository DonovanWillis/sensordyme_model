"""M8.4 B2 detector characterisation on the held-out VAL split (never test), from stored predictions.

    python -m research.m84.characterise imgstats --data D --split val --out stats_val.csv
    python -m research.m84.characterise slices   --data D --split val --stats stats_val.csv \
        --preds name=path ... --out slices.json
    python -m research.m84.characterise errors   --data D --split val --stats stats_val.csv \
        --preds name=path ... --out errors.json [--top 40]
    python -m research.m84.characterise mine     --data D --split train --preds coco80=path --out hardneg_train.csv

imgstats  per-image luminance statistics from the pixels (Rec.601 luma on a <=320-px copy):
          mean, p5, p50, p95, std, fraction of pixels below 32/255. LOW LIGHT IS A PROXY:
          Open Images has no illumination label and no NIR images; a dark web photo is not a
          NoIR cabin frame at night. The statistics only rank images by how dark they are.
slices    per model and class: AP50 overall and per luminance quartile (group bootstrap 95%
          CI), and recall at the model's own precision-0.80 threshold by luminance quartile,
          box size (OI relative area), occluded, truncated, depiction; PR curve points and
          reliability bins (for figures).
errors    error anatomy at fixed operating thresholds (person 0.4, phone 0.4, hand 0.3, the
          M8 product points): false positives split into localisation (0.1 <= IoU < 0.5 with
          a same-class box), duplicate, cross-class (on a ground-truth box of another of the
          three classes), on a verified-absent image, and background; the 3x4 confusion
          matrix; the strongest overlapping other-label detection when the file carries all
          COCO labels (look-alikes); misses by attribute; and the top-N false positives.
mine      hard-negative candidates from the TRAIN split (for a future training run only):
          images where the pretrained COCO model fires phone >= 0.3 with no phone box there,
          with the look-alike COCO class it confused. Never mines val or test.
Image-level Open Images results. They are NOT in-cabin or distracted-driving accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from research.detector_bench import oid_eval as oe
from research.m8.analysis import file_sha, load_split, r_at_p

CLASSES = ("person", "phone", "hand")
OPS = {"person": 0.4, "phone": 0.4, "hand": 0.3}
STAT_KEYS = ("luma_mean", "luma_p5", "luma_p50", "luma_p95", "luma_std", "dark_frac")


# --------------------------------------------------------------------------- imgstats
def imgstats(data: Path, split: str, out: Path) -> int:
    from PIL import Image

    from evaluation import manifest as mf

    if split == "test":
        raise SystemExit("the test split is frozen")
    items = sorted(mf.load(data / "manifest.json").split(split), key=lambda i: i.id)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "width", "height", *STAT_KEYS])
        for n, item in enumerate(items):
            with Image.open(data / item.path) as im:
                width, height = im.size
                im.draft("RGB", (320, 320))
                small = im.convert("RGB")
                small.thumbnail((320, 320))
            rgb = np.asarray(small, dtype=np.float32)
            y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            p5, p50, p95 = np.percentile(y, [5, 50, 95])
            w.writerow([item.id, width, height, round(float(y.mean()), 2), round(float(p5), 2), round(float(p50), 2),
                        round(float(p95), 2), round(float(y.std()), 2), round(float((y < 32).mean()), 4)])
            if n % 500 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    return len(items)


def load_stats(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f:
        return {r["id"]: {k: float(r[k]) for k in STAT_KEYS} for r in csv.DictReader(f)}


def luma_quartiles(stats: dict[str, dict[str, float]], ids: list[str]) -> tuple[list[float], dict[str, str]]:
    vals = np.array([stats[i]["luma_mean"] for i in ids if i in stats])
    edges = [float(x) for x in np.percentile(vals, [25, 50, 75])]
    def q(i: str) -> str:
        v = stats[i]["luma_mean"]
        return "Q1_dark" if v < edges[0] else "Q2" if v < edges[1] else "Q3" if v < edges[2] else "Q4_bright"
    return edges, {i: q(i) for i in ids if i in stats}


# --------------------------------------------------------------------------- helpers
def _subset(result: oe.ClassResult, ids: set[str]) -> oe.ClassResult:
    sub = oe.ClassResult()
    sub.scored = [s for s in result.scored if s.image_id in ids]
    sub.gt_index = {i: g for i, g in result.gt_index.items() if i in ids}
    sub.npos = sum(len(g) for g in sub.gt_index.values())
    sub.images = len(sub.gt_index)
    sub.negative_ids = {i for i in result.negative_ids if i in ids}
    sub.negative_images = len(sub.negative_ids)
    sub.groups = {i: g for i, g in result.groups.items() if i in ids}
    return sub


def _ap_ci(result: oe.ClassResult, n: int = 200, seed: int = 0) -> dict[str, Any]:
    point = oe.average_precision(result.scored, result.npos)
    by_group: dict[str, list[str]] = {}
    for i, g in result.groups.items():
        by_group.setdefault(g, []).append(i)
    per_img: dict[str, list[oe.Scored]] = {}
    for s in result.scored:
        per_img.setdefault(s.image_id, []).append(s)
    npos_img = {i: len(g) for i, g in result.gt_index.items()}
    groups = sorted(by_group)
    rng = random.Random(seed)  # noqa: S311 - resampling
    xs = []
    for _ in range(n if groups else 0):
        ids = [i for g in rng.choices(groups, k=len(groups)) for i in by_group[g]]
        npos = sum(npos_img.get(i, 0) for i in ids)
        if npos:
            ap = oe.average_precision([s for i in ids for s in per_img.get(i, [])], npos)
            if ap is not None:
                xs.append(ap)
    xs.sort()
    ci = [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]] if xs else None
    return {"ap50": point, "ci95": ci, "npos": result.npos, "images": result.images}


def _recall_slices(result: oe.ClassResult, threshold: Optional[float], key_of) -> dict[str, dict]:
    if threshold is None:
        return {}
    matched = {(s.image_id, s.matched) for s in result.scored if s.score >= threshold and s.tp and s.matched is not None}
    buckets: dict[str, list[int]] = {}
    for image_id, gts in result.gt_index.items():
        for n, g in enumerate([g for g in gts if not g.group_of]):
            for key in key_of(image_id, g):
                buckets.setdefault(key, []).append(int((image_id, n) in matched))
    return {k: {"recall": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(buckets.items())}


def _pr_points(result: oe.ClassResult, k: int = 120) -> dict[str, list[float]]:
    order = sorted(result.scored, key=lambda s: -s.score)
    if not order or not result.npos:
        return {"recall": [], "precision": [], "score": []}
    tp = np.cumsum([s.tp for s in order], dtype=float)
    n = np.arange(1, len(order) + 1, dtype=float)
    rec, prec = tp / result.npos, tp / n
    idx = np.unique(np.linspace(0, len(order) - 1, min(k, len(order))).astype(int))
    return {"recall": [round(float(rec[i]), 4) for i in idx], "precision": [round(float(prec[i]), 4) for i in idx],
            "score": [round(order[i].score, 4) for i in idx]}


# --------------------------------------------------------------------------- slices
def slices(data: Path, split: str, stats_path: Path, specs: list[str]) -> dict:
    from research.m8.analysis import load_preds

    manifest_id, _rows, truths, _g, _h = load_split(data, split)
    stats = load_stats(stats_path)
    all_ids = sorted({t.image_id for t in truths["person"]})
    edges, quart = luma_quartiles(stats, all_ids)
    out: dict[str, Any] = {"manifest_id": manifest_id, "split": split, "luma_quartile_edges": edges,
                           "proxy_note": "luminance of web photos as a low-light PROXY; not NIR, not in-cabin",
                           "models": {}}
    for spec in specs:
        name, _, path = spec.partition("=")
        preds = load_preds(Path(path), set(CLASSES))
        m: dict[str, Any] = {"preds": path, "preds_sha16": file_sha(Path(path)), "classes": {}}
        for label in CLASSES:
            if label not in preds:
                continue
            res = oe.evaluate_class(preds[label], truths[label])
            thr = r_at_p(res.scored, res.npos, 0.8)["threshold"]
            per_q = {}
            for qname in ("Q1_dark", "Q2", "Q3", "Q4_bright"):
                per_q[qname] = _ap_ci(_subset(res, {i for i, q in quart.items() if q == qname}))
            for cut in (60, 40):  # absolute darkness (mean luma out of 255), not quartiles
                per_q[f"luma_lt{cut}"] = _ap_ci(_subset(res, {i for i in all_ids if stats.get(i, {}).get("luma_mean", 255) < cut}))
            m["classes"][label] = {
                "overall": _ap_ci(res),
                "ap50_by_luma_quartile": per_q,
                "threshold_at_p0.80": thr,
                "recall_at_p0.80_by_luma_quartile": _recall_slices(
                    res, thr, lambda i, g: [quart.get(i, "na"), *(f"luma_lt{c}" for c in (60, 40)
                                                                  if stats.get(i, {}).get("luma_mean", 255) < c)]),
                "recall_at_p0.80_by_attribute": _recall_slices(
                    res, thr, lambda i, g: [f"size_{g.size}", *(k for k in ("occluded", "truncated", "depiction")
                                                                   if getattr(g, k)), "all"]),
                "pr": _pr_points(res),
                "reliability": oe.calibration(res),
            }
        out["models"][name] = m
        print(name, {k: round(v["overall"]["ap50"] or 0, 3) for k, v in m["classes"].items()}, flush=True)
    return out


# --------------------------------------------------------------------------- errors
def _boxes_by_label(rows: list[dict]) -> dict[str, dict[str, list[tuple]]]:
    from research.detector_bench.bench import HAND, PERSON_TREE, PHONE

    out: dict[str, dict[str, list[tuple]]] = {c: {} for c in CLASSES}
    for r in rows:
        for b in r["boxes"]:
            lab = "phone" if b["mid"] == PHONE else "hand" if b["mid"] == HAND else "person" if b["mid"] in PERSON_TREE else None
            if lab:
                out[lab].setdefault(r["id"], []).append((b["xmin"], b["ymin"], b["xmax"], b["ymax"]))
    return out


def _best(box: tuple, others: list[tuple]) -> tuple[float, float]:
    if not others:
        return 0.0, 0.0
    b, arr = np.asarray(box, float), np.asarray(others, float).reshape(-1, 4)
    return float(oe._iou(b, arr).max()), float(oe._ioa(b, arr).max())


def _matched_with_boxes(preds: dict[str, list], truths: list[oe.ImageTruth]):
    """(Scored, box) pairs: oe.match_image one detection at a time would lose its state, so match each
    image once and re-attach boxes. match_image drops detections ignored inside a group-of box; they
    are identified by replaying its rule on the same score order."""
    for truth in truths:
        if not truth.verified:
            continue
        dets = sorted(preds.get(truth.image_id, []), key=lambda d: -d[0])
        scored = oe.match_image(dets, truth)
        if len(scored) == len(dets):
            yield from zip(scored, [b for _, b in dets], strict=True)
            continue
        # some detections were ignored (group-of): keep the scored ones aligned by score and IoU
        j = 0
        for sc in scored:
            while j < len(dets) and abs(dets[j][0] - sc.score) > 1e-12:
                j += 1
            yield sc, (dets[j][1] if j < len(dets) else None)
            j += 1


def errors(data: Path, split: str, stats_path: Optional[Path], specs: list[str], top: int) -> dict:
    from research.m8.analysis import load_preds

    manifest_id, rows, truths, _g, _h = load_split(data, split)
    gt_boxes = _boxes_by_label(rows)
    stats = load_stats(stats_path) if stats_path else {}
    verified = {lab: {t.image_id for t in truths[lab] if t.verified} for lab in CLASSES}
    out: dict[str, Any] = {"manifest_id": manifest_id, "split": split, "thresholds": OPS, "models": {}}
    for spec in specs:
        name, _, path = spec.partition("=")
        allp = load_preds(Path(path))
        alt_labels = sorted(set(allp) - set(CLASSES))
        m: dict[str, Any] = {"preds_sha16": file_sha(Path(path)), "has_all_coco_labels": bool(alt_labels), "classes": {}}
        for label in CLASSES:
            if label not in allp:
                continue
            thr = OPS[label]
            res = oe.evaluate_class(allp[label], truths[label])
            cats: dict[str, int] = {}
            confusion = dict.fromkeys(("tp", "same_class_mislocalised", *[f"on_{c}" for c in CLASSES if c != label],
                                       "background"), 0)
            lookalike: dict[str, int] = {}
            fps = []
            for s, box in _matched_with_boxes(allp[label], truths[label]):
                if s.score < thr:
                    continue
                if s.tp:
                    confusion["tp"] += 1
                    continue
                other = {c: _best(box, gt_boxes[c].get(s.image_id, [])) for c in CLASSES if c != label} if box else {}
                cross = [c for c, (iou, ioa) in other.items() if iou >= 0.5 or (ioa >= 0.7 and c != "person")]
                if s.best_iou >= 0.5:
                    cat = "duplicate"
                elif s.best_iou >= 0.1:
                    cat = "localisation"
                elif cross:
                    cat = f"cross_class_{cross[0]}"
                elif s.image_id in res.negative_ids:
                    cat = "verified_absent_image"
                else:
                    cat = "background"
                cats[cat] = cats.get(cat, 0) + 1
                confusion["same_class_mislocalised" if s.best_iou >= 0.1 else f"on_{cross[0]}" if cross
                          else "background"] += 1
                alt = None
                if alt_labels and box:
                    cands = [(lab2, sc, _best(box, [b2])[0]) for lab2 in alt_labels
                             for sc, b2 in allp[lab2].get(s.image_id, [])]
                    cands = [c for c in cands if c[2] >= 0.5]
                    if cands:
                        alt = max(cands, key=lambda c: c[1])[0]
                        lookalike[alt] = lookalike.get(alt, 0) + 1
                fps.append({"id": s.image_id, "score": round(s.score, 4), "category": cat, "box": box,
                            "same_class_iou": round(s.best_iou, 3), "lookalike": alt,
                            "luma_mean": stats.get(s.image_id, {}).get("luma_mean")})
            fps.sort(key=lambda d: -d["score"])
            misses = _recall_slices(res, thr, lambda i, g: [f"size_{g.size}", *(k for k in ("occluded", "truncated", "depiction")
                                                                              if getattr(g, k)), "all"])
            m["classes"][label] = {"threshold": thr, "fp_total": len(fps), "fp_categories": cats,
                                   "confusion_row": confusion, "lookalike_counts": dict(sorted(lookalike.items(), key=lambda kv: -kv[1])),
                                   "recall_by_attribute": misses, "verified_images": len(verified[label]),
                                   "top_false_positives": fps[:top]}
        out["models"][name] = m
        print(name, {k: (v["fp_total"], v["fp_categories"]) for k, v in m["classes"].items()}, flush=True)
    return out


# --------------------------------------------------------------------------- mine
def mine(data: Path, split: str, spec: str, out: Path, thr: float = 0.3) -> int:
    from research.m8.analysis import load_preds

    if split != "train":
        raise SystemExit("hard negatives are mined from train only")
    _mid, rows, truths, _g, _h = load_split(data, split)
    gt_boxes = _boxes_by_label(rows)
    phone_verified = {t.image_id for t in truths["phone"] if t.verified}
    allp = load_preds(Path(spec.partition("=")[2]))
    alts = sorted(set(allp) - set(CLASSES))
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "phone_score", "phone_verified", "on_phone_box_iou", "on_hand_ioa", "lookalike", "lookalike_score", "x1", "y1", "x2", "y2"])
        for img, dets in sorted(allp.get("phone", {}).items()):
            for s, b in dets:
                if s < thr:
                    continue
                piou, _ = _best(b, gt_boxes["phone"].get(img, []))
                if piou >= 0.3:
                    continue
                _, hioa = _best(b, gt_boxes["hand"].get(img, []))
                cands = [(lab, sc) for lab in alts for sc, b2 in allp[lab].get(img, []) if _best(b, [b2])[0] >= 0.5]
                la = max(cands, key=lambda c: c[1]) if cands else ("", 0.0)
                w.writerow([img, round(s, 4), int(img in phone_verified), round(piou, 3), round(hioa, 3), la[0], round(la[1], 4),
                            *[round(v, 4) for v in b]])
                n += 1
    return n


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("imgstats", "slices", "errors", "mine"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--split", default="val", choices=("train", "val"))
        s.add_argument("--out", required=True)
    for name in ("slices", "errors"):
        sub.choices[name].add_argument("--stats")
        sub.choices[name].add_argument("--preds", action="append", required=True)
    sub.choices["errors"].add_argument("--top", type=int, default=40)
    sub.choices["mine"].add_argument("--preds", required=True)
    a = p.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.cmd == "imgstats":
        print("images", imgstats(data, a.split, out))
    elif a.cmd == "slices":
        out.write_text(json.dumps(slices(data, a.split, Path(a.stats), a.preds), indent=1))
    elif a.cmd == "errors":
        out.write_text(json.dumps(errors(data, a.split, Path(a.stats) if a.stats else None, a.preds, a.top), indent=1))
    else:
        print("hard-negative candidates", mine(data, a.split, a.preds, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
