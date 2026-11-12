"""M8 CV research: operating points, calibration, paired comparisons and phone interaction.

Reads existing prediction files only (JSON lines {id, label, score, box normalized xyxy}, as
written by research/detector_bench, research/pilot/predict_val.py and research/m8/coco80_preds.py).
No pixels, no training, and the frozen test split is refused.

    python -m research.m8.analysis ops    --data D --split val --preds name=path ... --out DIR
    python -m research.m8.analysis paired --data D --split val --a name=path --b name=path --out DIR
    python -m research.m8.analysis phone  --data D --split val --preds name=path \
        [--hands name=path] [--persons name=path] --out DIR

ops     per model and class: AP50, precision/recall/false-box rates over a threshold grid,
        precision at recall 0.6/0.7/0.8 and recall at precision 0.5-0.9 with the thresholds
        that reach them, reliability bins and ECE, and ECE after a cross-fitted Platt map
        (logistic on logit(score), fitted on one half of the photographer groups and scored
        on the other, both ways) -- does one monotone map make thresholds portable?
paired  photographer-group paired bootstrap of A - B: AP50, precision at recall 0.70,
        recall at precision 0.80, false positives at recall 0.70.
phone   phone-present vs phone-held from geometry. Every phone detection is described by its
        overlap with hand boxes (predicted hands from ``--hands``; human hand boxes where
        "hand" was verified), its containment in a person box, and the strongest
        overlapping look-alike detection (remote, tv, laptop, ... when the file carries all
        COCO classes). Gates over those features are scored on precision, recall split by
        human "in hand" (phone box overlapping a human hand box), and false phones on
        verified phone-free images. Image-level proxy: NOT driving-event accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.detector_bench import oid_eval as oe

PHONE_MID, HAND_MID, PERSON_MID = "/m/050k8", "/m/0k65p", "/m/01g317"
GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))
RECALLS = (0.6, 0.7, 0.8)
PRECISIONS = (0.5, 0.6, 0.7, 0.8, 0.9)
LOOKALIKES = ("remote", "tv", "laptop", "mouse", "keyboard", "book", "clock", "cup", "bottle", "toothbrush",
              "scissors", "handbag", "wallet")


# ------------------------------------------------------------------------------- loading
def load_split(data: Path, split: str):
    """(manifest id, rows, truths per class, groups, hand-verified ids)."""
    if split == "test":
        raise SystemExit("the test split is frozen; this tool refuses it")
    from evaluation import manifest as mf
    from research.detector_bench import bench

    manifest = mf.load(data / "manifest.json")
    items = manifest.split(split)
    wanted = {i.id for i in items}
    rows = [r for r in (json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x)
            if r["id"] in wanted]
    groups = {i.id: i.subject for i in items}
    truths = bench._truths(rows, groups)
    hand_verified = {r["id"] for r in rows if HAND_MID in r["verified_positive"] or HAND_MID in r["verified_negative"]}
    return manifest.id, rows, truths, groups, hand_verified


def load_preds(path: Path, labels: Optional[set[str]] = None) -> dict[str, dict[str, list[tuple[float, tuple]]]]:
    out: dict[str, dict[str, list]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if labels is not None and d["label"] not in labels:
                continue
            out.setdefault(d["label"], {}).setdefault(d["id"], []).append((float(d["score"]), tuple(d["box"])))
    return out


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------- curves
def _curve(scored: list[oe.Scored], npos: int):
    order = sorted(scored, key=lambda s: -s.score)
    scores = np.array([s.score for s in order], dtype=float)
    tp = np.cumsum([s.tp for s in order], dtype=float)
    n = np.arange(1, len(order) + 1, dtype=float)
    return scores, tp / max(npos, 1), tp / n, n - tp


def p_at_r(scored, npos, target):
    scores, recall, precision, fp = _curve(scored, npos)
    reach = np.nonzero(recall >= target - 1e-12)[0]
    if not len(reach):
        return {"precision": None, "threshold": None, "fp": None}
    first = int(reach[0])
    return {"precision": float(precision[first:].max()), "threshold": float(scores[first]), "fp": int(fp[first])}


def r_at_p(scored, npos, target):
    scores, recall, precision, _ = _curve(scored, npos)
    ok = np.nonzero(precision >= target - 1e-12)[0]
    if not len(ok):
        return {"recall": 0.0, "threshold": None}
    best = int(ok[np.argmax(recall[ok])])
    return {"recall": float(recall[best]), "threshold": float(scores[best])}


# ------------------------------------------------------------------------------- calibration
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(scores: np.ndarray, hits: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Logistic regression hit ~ sigmoid(a * logit(score) + b), Newton's method, tiny ridge."""
    x = _logit(scores)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = a * x + b
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - hits) * x) + 1e-3 * a, np.sum(p - hits)])
        h = np.array([[np.sum(w * x * x) + 1e-3, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def ece(scores: np.ndarray, hits: np.ndarray, bins: int = 10) -> Optional[float]:
    if not len(scores):
        return None
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for k in range(bins):
        sel = (scores >= edges[k]) & ((scores < edges[k + 1]) if k < bins - 1 else (scores <= 1))
        if sel.any():
            total += sel.mean() * abs(scores[sel].mean() - hits[sel].mean())
    return float(total)


def crossfit_calibration(result: oe.ClassResult) -> dict:
    """Platt map fitted on half the photographer groups and applied to the other half (both ways)."""
    if not result.scored:
        return {}
    def half(image_id: str) -> int:
        return int(hashlib.sha256(result.groups[image_id].encode()).hexdigest(), 16) % 2

    scores = np.array([s.score for s in result.scored])
    hits = np.array([s.tp for s in result.scored], dtype=float)
    side = np.array([half(s.image_id) for s in result.scored])
    calibrated = np.empty_like(scores)
    fits = []
    for k in (0, 1):
        a, b = fit_platt(scores[side != k], hits[side != k])
        fits.append({"a": round(a, 4), "b": round(b, 4)})
        calibrated[side == k] = 1 / (1 + np.exp(-(a * _logit(scores[side == k]) + b)))
    a_all, b_all = fit_platt(scores, hits)
    # the raw score at which the full-data map predicts a hit rate of 0.5 / 0.7 / 0.8 / 0.9
    def raw_for(p: float) -> Optional[float]:
        if abs(a_all) < 1e-9:
            return None
        z = (math.log(p / (1 - p)) - b_all) / a_all
        return round(float(1 / (1 + math.exp(-z))), 4)

    return {"ece_raw": ece(scores, hits), "ece_crossfit_platt": ece(calibrated, hits), "fits_by_half": fits,
            "fit_all": {"a": round(a_all, 4), "b": round(b_all, 4)},
            "raw_score_for_hit_rate": {str(p): raw_for(p) for p in (0.5, 0.7, 0.8, 0.9)}}


# ------------------------------------------------------------------------------- ops
def class_ops(result: oe.ClassResult) -> dict[str, Any]:
    grid = []
    for t in GRID:
        op = oe.operating_point(result, t)
        grid.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in op.items()
                     if k in ("threshold", "tp", "fp", "precision", "recall", "fp_per_negative_image",
                              "negative_images_with_a_false_box", "mean_tp_iou")})
    return {
        "npos": result.npos, "images": result.images, "negative_images": result.negative_images,
        "ap50": oe.average_precision(result.scored, result.npos),
        "precision_at_recall": {str(r): p_at_r(result.scored, result.npos, r) for r in RECALLS},
        "recall_at_precision": {str(p): r_at_p(result.scored, result.npos, p) for p in PRECISIONS},
        "grid": grid,
        "calibration_bins": oe.calibration(result)["bins"],
        "calibration": crossfit_calibration(result),
    }


def cmd_ops(data: Path, split: str, specs: list[str]) -> dict:
    manifest_id, _rows, truths, _g, _h = load_split(data, split)
    out: dict[str, Any] = {"manifest_id": manifest_id, "split": split, "models": {}}
    for spec in specs:
        name, _, path = spec.partition("=")
        preds = load_preds(Path(path), {"person", "phone", "hand"})
        out["models"][name] = {"preds": path, "preds_sha16": file_sha(Path(path)), "classes": {}}
        for label in ("person", "phone", "hand"):
            if label in preds:
                out["models"][name]["classes"][label] = class_ops(oe.evaluate_class(preds[label], truths[label]))
    return out


# ------------------------------------------------------------------------------- paired
def _resample_metrics(per_image: dict[str, list[oe.Scored]], npos_image: dict[str, int], ids: list[str]) -> dict:
    scored: list[oe.Scored] = []
    npos = 0
    for i in ids:
        scored += per_image.get(i, [])
        npos += npos_image.get(i, 0)
    if not npos:
        return {}
    par = p_at_r(scored, npos, 0.7)
    return {"ap50": oe.average_precision(scored, npos), "p_at_r0.70": par["precision"], "fp_at_r0.70": par["fp"],
            "r_at_p0.80": r_at_p(scored, npos, 0.8)["recall"]}


def cmd_paired(data: Path, split: str, a: str, b: str, n: int, seed: int = 0) -> dict:
    manifest_id, _rows, truths, _g, _h = load_split(data, split)
    (na, _, pa), (nb, _, pb) = a.partition("="), b.partition("=")
    preds_a, preds_b = load_preds(Path(pa)), load_preds(Path(pb))
    out: dict[str, Any] = {"manifest_id": manifest_id, "split": split, "a": na, "b": nb, "resamples": n,
                           "a_sha16": file_sha(Path(pa)), "b_sha16": file_sha(Path(pb)), "classes": {}}
    for label in ("person", "phone", "hand"):
        if label not in preds_a or label not in preds_b:
            continue
        ra = oe.evaluate_class(preds_a[label], truths[label])
        rb = oe.evaluate_class(preds_b[label], truths[label])
        per_a: dict[str, list] = {}
        per_b: dict[str, list] = {}
        for s in ra.scored:
            per_a.setdefault(s.image_id, []).append(s)
        for s in rb.scored:
            per_b.setdefault(s.image_id, []).append(s)
        npos_image = {i: len(g) for i, g in ra.gt_index.items()}
        by_group: dict[str, list[str]] = {}
        for image_id, group in ra.groups.items():
            by_group.setdefault(group, []).append(image_id)
        groups = sorted(by_group)
        all_ids = [i for g in groups for i in by_group[g]]
        point_a = _resample_metrics(per_a, npos_image, all_ids)
        point_b = _resample_metrics(per_b, npos_image, all_ids)
        rng = random.Random(seed)  # noqa: S311 - resampling
        deltas: dict[str, list[float]] = {k: [] for k in point_a}
        for _ in range(n):
            ids = [i for g in rng.choices(groups, k=len(groups)) for i in by_group[g]]
            ma, mb = _resample_metrics(per_a, npos_image, ids), _resample_metrics(per_b, npos_image, ids)
            for k in deltas:
                if ma.get(k) is not None and mb.get(k) is not None:
                    deltas[k].append(ma[k] - mb[k])
        summary = {}
        for k, raw_xs in deltas.items():
            xs = sorted(raw_xs)
            if not xs:
                continue
            summary[k] = {"a": point_a.get(k), "b": point_b.get(k),
                          "delta": (point_a[k] - point_b[k]) if point_a.get(k) is not None and point_b.get(k) is not None else None,
                          "ci95": [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]],
                          "p_delta_le_0": sum(x <= 0 for x in xs) / len(xs), "n": len(xs)}
        out["classes"][label] = {"npos": ra.npos, "groups": len(groups), **summary}
    return out


# ------------------------------------------------------------------------------- phone interaction
def _inter(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a) -> float:
    return max(1e-12, (a[2] - a[0]) * (a[3] - a[1]))


def _iou(a, b) -> float:
    i = _inter(a, b)
    return i / max(_area(a) + _area(b) - i, 1e-12)


def _expand(a, k: float):
    w, h = a[2] - a[0], a[3] - a[1]
    return (a[0] - k * w, a[1] - k * h, a[2] + k * w, a[3] + k * h)


def features(box, hands: list[tuple], persons: list[tuple], alts: list[tuple[str, float, tuple]]) -> dict:
    """Geometry of one phone box against hands, persons and look-alike detections."""
    f: dict[str, Any] = {}
    f["hand_overlap"] = max((_inter(box, h) / _area(box) for h in hands), default=0.0)  # share of the phone on a hand
    f["hand_iou"] = max((_iou(box, h) for h in hands), default=0.0)
    side = math.sqrt(_area(box))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    def gap(h):  # centre distance from the phone to the nearest point of the hand box, in phone sides
        dx = max(h[0] - cx, 0.0, cx - h[2])
        dy = max(h[1] - cy, 0.0, cy - h[3])
        return math.hypot(dx, dy) / side
    f["hand_gap"] = min((gap(h) for h in hands), default=float("inf"))
    f["in_person"] = max((_inter(box, p) / _area(box) for p in persons), default=0.0)
    best_alt = max(((s, lab) for lab, s, b in alts if _iou(box, b) >= 0.3), default=(0.0, None))
    f["alt_score"], f["alt_label"] = best_alt
    return f


def _bootstrap_ci(values_by_group: dict[str, tuple[int, int, int, int]], n: int = 300, seed: int = 0) -> dict:
    """values: group -> (tp, kept, matched_gt, npos); CI for precision and recall."""
    groups = sorted(values_by_group)
    rng = random.Random(seed)  # noqa: S311
    ps, rs = [], []
    for _ in range(n):
        tp = kept = m = npos = 0
        for g in rng.choices(groups, k=len(groups)):
            a, b, c, d = values_by_group[g]
            tp, kept, m, npos = tp + a, kept + b, m + c, npos + d
        if kept:
            ps.append(tp / kept)
        if npos:
            rs.append(m / npos)
    def ci(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * (len(xs) - 1))], 4), round(xs[int(0.975 * (len(xs) - 1))], 4)] if xs else None
    return {"precision_ci95": ci(ps), "recall_ci95": ci(rs)}


def cmd_phone(data: Path, split: str, spec: str, hands_spec: Optional[str], persons_spec: Optional[str],
              hand_thr: float, person_thr: float) -> dict:
    manifest_id, rows, truths, groups, hand_verified = load_split(data, split)
    name, _, path = spec.partition("=")
    preds = load_preds(Path(path))
    phone_preds = preds.get("phone", {})
    hand_src = load_preds(Path(hands_spec.partition("=")[2]), {"hand"}).get("hand", {}) if hands_spec else {}
    person_src = (load_preds(Path(persons_spec.partition("=")[2]), {"person"}).get("person", {}) if persons_spec
                  else preds.get("person", {}))
    gt_hands = {r["id"]: [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in r["boxes"] if b["mid"] == HAND_MID]
                for r in rows}
    alts_by_image: dict[str, list] = {}
    for lab in LOOKALIKES:
        for image_id, dets in preds.get(lab, {}).items():
            alts_by_image.setdefault(image_id, []).extend((lab, s, b) for s, b in dets)
    phone_truth = {t.image_id: t for t in truths["phone"] if t.verified}
    # human "in hand": a plain phone box overlapping a human hand box (>= 10% of the phone), images with hand verified
    held: dict[tuple[str, int], Optional[bool]] = {}
    for image_id, t in phone_truth.items():
        for k, g in enumerate(x for x in t.gts if not x.group_of):
            if image_id in hand_verified:
                held[(image_id, k)] = any(_inter(g.box, h) / _area(g.box) >= 0.1 for h in gt_hands[image_id])
            else:
                held[(image_id, k)] = None
    report: dict[str, Any] = {
        "manifest_id": manifest_id, "split": split, "model": name, "preds_sha16": file_sha(Path(path)),
        "hands_from": hands_spec or "none (human hand boxes only)", "hand_threshold": hand_thr,
        "persons_from": persons_spec or f"{name} (same file)", "person_threshold": person_thr,
        "ground_truth": {
            "phone_verified_images": len(phone_truth),
            "phone_free_images": sum(t.negative for t in phone_truth.values()),
            "phones_plain": len(held),
            "phones_on_hand_verified_images": sum(v is not None for v in held.values()),
            "phones_in_human_hand": sum(bool(v) for v in held.values()),
            "has_lookalike_classes": bool(alts_by_image),
            "has_predicted_hands": bool(hand_src),
        },
    }

    def pred_hands(i: str) -> list[tuple]:
        return [b for s, b in hand_src.get(i, []) if s >= hand_thr]

    def pred_persons(i: str) -> list[tuple]:
        return [b for s, b in person_src.get(i, []) if s >= person_thr]

    # Feature table for every phone detection on a phone-verified image.
    det_rows = []
    for image_id in phone_truth:
        for s, b in phone_preds.get(image_id, []):
            fp = features(b, pred_hands(image_id), pred_persons(image_id), alts_by_image.get(image_id, []))
            fg = features(b, gt_hands[image_id] if image_id in hand_verified else [], [], [])
            det_rows.append({"id": image_id, "score": s, "box": b, **fp,
                             "gt_hand_overlap": fg["hand_overlap"] if image_id in hand_verified else None})

    gates: dict[str, Callable[[dict], bool]] = {"none": lambda r: True}
    if hand_src:
        gates["pred_hand_overlap"] = lambda r: r["hand_overlap"] >= 0.1
        for k in (0.25, 0.5, 1.0):
            gates[f"pred_hand_within_{k}"] = (lambda kk: lambda r: r["hand_gap"] <= kk)(k)
    gates["oracle_human_hand_overlap"] = lambda r: (r["gt_hand_overlap"] or 0.0) >= 0.1
    gates[f"in_person_box(>= {person_thr})"] = lambda r: r["in_person"] >= 0.5
    if alts_by_image:
        for ratio in (0.5, 1.0, 2.0):
            gates[f"no_lookalike_ratio_{ratio}"] = (lambda q: lambda r: r["alt_score"] < q * r["score"])(ratio)

    def score_gate(gate, thr: float) -> dict:
        kept = {}
        for r in det_rows:
            if r["score"] >= thr and gate(r):
                kept.setdefault(r["id"], []).append((r["score"], r["box"]))
        result = oe.evaluate_class(kept, list(phone_truth.values()))
        op = oe.operating_point(result, thr)
        matched = {(s.image_id, s.matched) for s in result.scored if s.tp and s.matched is not None}
        by = {"in_hand": [0, 0], "not_in_hand": [0, 0], "hand_unverified": [0, 0]}
        for key, v in held.items():
            slot = "hand_unverified" if v is None else "in_hand" if v else "not_in_hand"
            by[slot][0] += 1
            by[slot][1] += key in matched
        vg: dict[str, list[int]] = {}
        for s in result.scored:
            g = vg.setdefault(groups[s.image_id], [0, 0, 0, 0])
            g[0] += s.tp
            g[1] += 1
            g[2] += bool(s.tp and s.matched is not None)
        for image_id, t in phone_truth.items():
            vg.setdefault(groups[image_id], [0, 0, 0, 0])[3] += len(t.gts)
        return {"precision": op["precision"], "recall": op["recall"], "tp": op["tp"], "fp": op["fp"],
                "fp_on_phone_free_images": round((op["fp_per_negative_image"] or 0) * result.negative_images),
                "phone_free_images_with_a_false_phone": op["negative_images_with_a_false_box"],
                "recall_by_human_hand": {k: {"n": n, "recall": (m / n if n else None)} for k, (n, m) in by.items()},
                **_bootstrap_ci({g: tuple(v) for g, v in vg.items()})}

    report["gates"] = {f"t{thr}:{g}": score_gate(fn, thr) for thr in (0.2, 0.3, 0.4, 0.5) for g, fn in gates.items()}

    # Held-vs-present as a classifier: among detections that hit a phone on a hand-verified image,
    # does the predicted-hand geometry agree with the human "in hand" label?
    truth_hits = []
    for image_id, t in phone_truth.items():
        if image_id not in hand_verified:
            continue
        plain = [g for g in t.gts if not g.group_of]
        used = [False] * len(plain)
        for score, box in sorted(phone_preds.get(image_id, []), key=lambda d: -d[0]):
            if score < 0.3 or not plain:
                continue
            ious = [_iou(box, g.box) for g in plain]
            k = int(np.argmax(ious))
            if ious[k] >= 0.5 and not used[k]:
                used[k] = True
                f = features(box, pred_hands(image_id), [], [])
                truth_hits.append((bool(held[(image_id, k)]), f["hand_overlap"], f["hand_gap"]))
    if hand_src and truth_hits:
        cls = {}
        gt = [h for h, _, _ in truth_hits]
        for k in (0.0, 0.25, 0.5, 1.0):
            pred = [(gap <= k) if k else (ov >= 0.1) for _, ov, gap in truth_hits]
            tp = sum(p and g for p, g in zip(pred, gt, strict=True))
            cls["overlap" if not k else f"within_{k}"] = {
                "held_precision": tp / max(sum(pred), 1), "held_recall": tp / max(sum(gt), 1),
                "predicted_held": sum(pred), "human_held": sum(gt), "n": len(gt)}
        report["held_classifier_on_true_phones@0.3"] = cls

    # Distributions: TP vs FP phone detections at 0.3.
    def dist(vals):
        vals = [v for v in vals if v is not None and v != float("inf")]
        if not vals:
            return None
        q = np.quantile(vals, [0.25, 0.5, 0.75])
        return {"n": len(vals), "q25": round(float(q[0]), 3), "median": round(float(q[1]), 3),
                "q75": round(float(q[2]), 3)}
    res = oe.evaluate_class({i: [(r["score"], r["box"]) for r in det_rows if r["id"] == i] for i in phone_truth},
                            list(phone_truth.values()))
    tp_keys = {(s.image_id, round(s.score, 9)) for s in res.scored if s.tp}
    neg_ids = res.negative_ids
    groups_rows = {"tp": [], "fp_on_phone_free": [], "fp_other": []}
    for r in det_rows:
        if r["score"] < 0.3:
            continue
        key = "tp" if (r["id"], round(r["score"], 9)) in tp_keys else "fp_on_phone_free" if r["id"] in neg_ids else "fp_other"
        groups_rows[key].append(r)
    report["feature_distributions@0.3"] = {
        k: {"n": len(v),
            "pred_hand_overlap": dist([r["hand_overlap"] for r in v]) if hand_src else None,
            "share_with_pred_hand_overlap>=0.1": (sum(r["hand_overlap"] >= 0.1 for r in v) / len(v)) if v and hand_src else None,
            "in_person>=0.5": (sum(r["in_person"] >= 0.5 for r in v) / len(v)) if v else None,
            "alt_score": dist([r["alt_score"] for r in v]) if alts_by_image else None,
            "alt_label_counts": _counts([r["alt_label"] for r in v if r["alt_label"]]) if alts_by_image else None}
        for k, v in groups_rows.items()}
    return report


def _counts(xs: list) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("ops", "paired", "phone"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True)
        s.add_argument("--split", default="val", choices=("train", "val"))
        s.add_argument("--out", required=True)
        s.add_argument("--tag", default="")
    sub.choices["ops"].add_argument("--preds", action="append", required=True)
    sub.choices["paired"].add_argument("--a", required=True)
    sub.choices["paired"].add_argument("--b", required=True)
    sub.choices["paired"].add_argument("--n", type=int, default=500)
    ph = sub.choices["phone"]
    ph.add_argument("--preds", required=True)
    ph.add_argument("--hands")
    ph.add_argument("--persons")
    ph.add_argument("--hand-thr", type=float, default=0.3)
    ph.add_argument("--person-thr", type=float, default=0.5)
    args = p.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    if args.cmd == "ops":
        report = cmd_ops(data, args.split, args.preds)
    elif args.cmd == "paired":
        report = cmd_paired(data, args.split, args.a, args.b, args.n)
    else:
        report = cmd_phone(data, args.split, args.preds, args.hands, args.persons, args.hand_thr, args.person_thr)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    target = out / f"m8_{args.cmd}_{args.split}{tag}.json"
    target.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
