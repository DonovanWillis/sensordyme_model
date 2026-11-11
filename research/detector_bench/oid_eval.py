"""Box-level scoring under the Open Images protocol, plus operating-point metrics.

Scoring rules (documented, and the same for every model):
  * A class is scored on an image ONLY if a human verified that class on it
    (positive or negative, after hierarchy expansion). Elsewhere an unboxed
    object may simply be unannotated, so a detection there is neither right nor
    wrong: it is not counted.
  * Validation/test boxes are exhaustive for verified-positive classes.
  * A detection is a true positive if IoU >= t with an unmatched non-group box;
    a second detection on an already-matched box is a false positive.
  * Group-of boxes: the first detection whose area lies >= 50% inside one counts
    once as a true positive; later ones there are ignored (Open Images rule).
  * AP is all-point interpolated (VOC 2010+ / Open Images) at IoU 0.5; AP@[.5:.95]
    is also reported, for localisation quality only.
  * Size buckets are relative box area: small < 0.25% of the image (about a
    32x32 object in a 640-px frame), medium < 2.25%, large otherwise.

This measures BOX detection. It says nothing about whether a person is *using*
a phone: that is behaviour recognition, measured by the video harness.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SMALL, MEDIUM = 0.0025, 0.0225


@dataclass
class GroundTruth:
    box: tuple[float, float, float, float]  # normalized xyxy
    group_of: bool = False
    occluded: bool = False
    truncated: bool = False
    depiction: bool = False

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])

    @property
    def size(self) -> str:
        return "small" if self.area < SMALL else "medium" if self.area < MEDIUM else "large"


@dataclass
class ImageTruth:
    image_id: str
    group: str  # leakage/bootstrap unit (photographer or near-duplicate cluster)
    gts: list[GroundTruth]
    verified: bool  # the class was human-verified on this image (positive or negative)
    negative: bool  # verified absent


@dataclass
class Scored:
    image_id: str
    score: float
    tp: bool
    matched: Optional[int]  # index of the matched gt (non-group), else None
    best_iou: float = 0.0  # best IoU with any non-group gt: separates localisation errors from background


@dataclass
class ClassResult:
    scored: list[Scored] = field(default_factory=list)
    npos: int = 0
    gt_index: dict[str, list[GroundTruth]] = field(default_factory=dict)
    images: int = 0
    negative_images: int = 0
    negative_ids: set[str] = field(default_factory=set)
    groups: dict[str, str] = field(default_factory=dict)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` (4,) against many ``b`` (N,4)."""
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-12)


def _ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of ``a`` (the detection)."""
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1e-12)


def match_image(dets: list[tuple[float, tuple[float, float, float, float]]], truth: ImageTruth,
                iou: float = 0.5) -> list[Scored]:
    """Score one image's detections (score, box) for one class."""
    if not truth.verified:
        return []
    plain = [g for g in truth.gts if not g.group_of]
    groups = [g for g in truth.gts if g.group_of]
    plain_arr = np.array([g.box for g in plain]).reshape(-1, 4)
    group_arr = np.array([g.box for g in groups]).reshape(-1, 4)
    used_plain = np.zeros(len(plain), bool)
    used_group = np.zeros(len(groups), bool)
    out = []
    for score, box in sorted(dets, key=lambda d: -d[0]):
        b = np.asarray(box, dtype=float)
        best_iou = 0.0
        if len(plain):
            ious = _iou(b, plain_arr)
            best = int(ious.argmax())
            best_iou = float(ious[best])
            if ious[best] >= iou:
                if not used_plain[best]:
                    used_plain[best] = True
                    out.append(Scored(truth.image_id, score, True, best, best_iou))
                else:
                    out.append(Scored(truth.image_id, score, False, None, best_iou))
                continue
        if len(groups):
            ioas = _ioa(b, group_arr)
            best = int(ioas.argmax())
            if ioas[best] >= 0.5:
                if not used_group[best]:
                    used_group[best] = True
                    out.append(Scored(truth.image_id, score, True, None))
                continue  # further detections in a group box are ignored
        out.append(Scored(truth.image_id, score, False, None, best_iou))
    return out


def average_precision(scored: list[Scored], npos: int) -> Optional[float]:
    if npos == 0:
        return None
    if not scored:
        return 0.0
    order = sorted(scored, key=lambda s: -s.score)
    tp = np.cumsum([s.tp for s in order])
    fp = np.cumsum([not s.tp for s in order])
    recall = tp / npos
    precision = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_class(predictions: dict[str, list[tuple[float, tuple]]], truths: Iterable[ImageTruth],
                   iou: float = 0.5) -> ClassResult:
    result = ClassResult()
    for truth in truths:
        if not truth.verified:
            continue
        result.images += 1
        result.groups[truth.image_id] = truth.group
        if truth.negative:
            result.negative_images += 1
            result.negative_ids.add(truth.image_id)
        result.npos += len(truth.gts)
        result.gt_index[truth.image_id] = truth.gts
        result.scored += match_image(predictions.get(truth.image_id, []), truth, iou)
    return result


def operating_point(result: ClassResult, threshold: float) -> dict:
    kept = [s for s in result.scored if s.score >= threshold]
    tp = sum(s.tp for s in kept)
    fp = len(kept) - tp
    on_negative = [s for s in kept if s.image_id in result.negative_ids]
    false = [s for s in kept if not s.tp]
    phantom_images = len({s.image_id for s in on_negative})
    out = {
        "threshold": threshold, "tp": tp, "fp": fp,
        "precision": tp / len(kept) if kept else None,
        "recall": tp / result.npos if result.npos else None,
        "fp_per_verified_image": fp / result.images if result.images else None,
        "fp_per_negative_image": len(on_negative) / result.negative_images if result.negative_images else None,
        "negative_images_with_a_false_box": (phantom_images / result.negative_images
                                             if result.negative_images else None),
        "negative_images": result.negative_images,
        # a false box overlapping a real one (0.1 <= IoU < match) is a localisation/duplicate error
        "mean_tp_iou": (sum(s.best_iou for s in kept if s.tp and s.matched is not None)
                        / max(1, sum(1 for s in kept if s.tp and s.matched is not None))),
        "fp_localisation": sum(s.best_iou >= 0.1 for s in false),
        "fp_background": sum(s.best_iou < 0.1 for s in false),
    }
    # recall by ground-truth attribute (non-group boxes)
    matched = {(s.image_id, s.matched) for s in kept if s.tp and s.matched is not None}
    buckets: dict[str, list[int]] = {}
    for image_id, gts in result.gt_index.items():
        plain = [g for g in gts if not g.group_of]
        for n, g in enumerate(plain):
            hit = int((image_id, n) in matched)
            for key in (g.size, *(k for k in ("occluded", "truncated", "depiction") if getattr(g, k))):
                buckets.setdefault(key, []).append(hit)
    out["recall_by_attribute"] = {k: {"recall": sum(v) / len(v), "n": len(v)} for k, v in sorted(buckets.items())}
    return out


def calibration(result: ClassResult, bins: int = 10) -> dict:
    """Reliability of the score as a precision estimate (detections >= the score floor)."""
    if not result.scored:
        return {"ece": None, "bins": []}
    scores = np.array([s.score for s in result.scored])
    hits = np.array([s.tp for s in result.scored], dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in itertools.pairwise(edges):
        sel = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if sel.any():
            gap = abs(scores[sel].mean() - hits[sel].mean())
            ece += sel.mean() * gap
            rows.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": int(sel.sum()),
                         "mean_score": float(scores[sel].mean()), "precision": float(hits[sel].mean())})
    return {"ece": float(ece), "bins": rows}


def group_bootstrap(result: ClassResult, threshold: float, n: int = 300, seed: int = 0) -> dict:
    """95% percentile intervals for AP@0.5 and recall@threshold, resampling photographer groups."""
    by_group: dict[str, list[str]] = {}
    for image_id, group in result.groups.items():
        by_group.setdefault(group, []).append(image_id)
    groups = sorted(by_group)
    if len(groups) < 2:
        return {}
    per_image: dict[str, list[Scored]] = {}
    for s in result.scored:
        per_image.setdefault(s.image_id, []).append(s)
    npos_image = {i: len(g) for i, g in result.gt_index.items()}
    rng = random.Random(seed)  # noqa: S311 - resampling, not security
    aps, recalls = [], []
    for _ in range(n):
        scored, npos, tp_at = [], 0, 0
        for g in rng.choices(groups, k=len(groups)):
            for image_id in by_group[g]:
                rows = per_image.get(image_id, [])
                scored += rows
                npos += npos_image.get(image_id, 0)
                tp_at += sum(s.tp for s in rows if s.score >= threshold)
        if npos:
            aps.append(average_precision(scored, npos))
            recalls.append(tp_at / npos)

    def ci(xs: list) -> Optional[list[float]]:
        xs = sorted(x for x in xs if x is not None)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]] if xs else None

    return {"ap50_ci95": ci(aps), "recall_at_threshold_ci95": ci(recalls), "resamples": n, "groups": len(groups)}
