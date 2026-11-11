"""Why did the EXP-08 3-class model lose person precision? Val-only analysis of existing predictions (M7 G).

    python -m research.pilot.person_regression analyze --data <oi7-sd-v1> \
        --preds pretrained=<preds.jsonl> --preds full60=<preds.jsonl> --out <dir>
    python -m research.pilot.person_regression merge --out merged.jsonl \
        --take person,phone=<pretrained preds> --take hand=<3-class preds>

Prediction files are JSON lines {id, label, score, box (normalized xyxy)} as written by
research/pilot/predict_val.py and research/detector_bench (bench_preds). Scoring reuses the
Open Images matcher in research/detector_bench/oid_eval.py, so every number agrees with
`bench score`. Nothing here reads pixels, trains, or touches the test split.

`analyze` reports, per model and class:
  * AP50 and the full precision/recall curve at recall 0.60 / 0.70 / 0.80: the precision
    (interpolated, as AP uses it) and the score threshold that reaches it;
  * recall reachable at precision 0.50 / 0.60 / 0.70 / 0.80, and its threshold;
  * ECE (oid_eval.calibration);
  * for person: what the misses and false positives are (size, occlusion, truncation,
    depiction, a group box in the image, a hand box nearby), and, between the first two
    models, which ground-truth people one finds and the other misses at the same recall.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from research.detector_bench import oid_eval as oe

RECALLS = (0.6, 0.7, 0.8)
PRECISIONS = (0.5, 0.6, 0.7, 0.8)
CLASSES = ("person", "phone", "hand")


# ---------------------------------------------------------------------- curves
def curve(result: oe.ClassResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scores, recall, precision) after each detection, highest score first."""
    order = sorted(result.scored, key=lambda s: -s.score)
    scores = np.array([s.score for s in order], dtype=float)
    tp = np.cumsum([s.tp for s in order], dtype=float)
    n = np.arange(1, len(order) + 1, dtype=float)
    recall = tp / max(result.npos, 1)
    precision = tp / n
    return scores, recall, precision


def precision_at_recall(result: oe.ClassResult, target: float) -> dict[str, Optional[float]]:
    """Interpolated precision at ``target`` recall (max precision at any recall >= target) and the
    lowest score threshold that reaches ``target`` recall. None when the recall is never reached."""
    scores, recall, precision = curve(result)
    reach = np.nonzero(recall >= target - 1e-12)[0]
    if not len(reach):
        return {"precision": None, "threshold": None}
    first = int(reach[0])
    return {"precision": float(precision[first:].max()), "threshold": float(scores[first])}


def recall_at_precision(result: oe.ClassResult, target: float) -> dict[str, Optional[float]]:
    """Highest recall at which precision is still >= ``target``, and the threshold giving it."""
    scores, recall, precision = curve(result)
    ok = np.nonzero(precision >= target - 1e-12)[0]
    if not len(ok):
        return {"recall": 0.0, "threshold": None}
    best = int(ok[np.argmax(recall[ok])])
    return {"recall": float(recall[best]), "threshold": float(scores[best])}


# ---------------------------------------------------------------------- failure attribution
def _overlap(box: tuple, others: list[tuple], thr: float = 0.3) -> bool:
    if not others:
        return False
    b = np.asarray(box, dtype=float)
    arr = np.asarray(others, dtype=float).reshape(-1, 4)
    return bool((oe._iou(b, arr) >= thr).any() or (oe._ioa(b, arr) >= 0.5).any())


def _missed(result: oe.ClassResult, threshold: float) -> set[tuple[str, int]]:
    """(image, plain-gt index) pairs not matched by any detection scoring >= threshold."""
    hit = {(s.image_id, s.matched) for s in result.scored if s.tp and s.matched is not None and s.score >= threshold}
    out = set()
    for image_id, gts in result.gt_index.items():
        for n, _g in enumerate(g for g in gts if not g.group_of):
            if (image_id, n) not in hit:
                out.add((image_id, n))
    return out


def _plain(result: oe.ClassResult, image_id: str) -> list[oe.GroundTruth]:
    return [g for g in result.gt_index.get(image_id, []) if not g.group_of]


def describe_misses(result: oe.ClassResult, keys: set[tuple[str, int]], hands: dict[str, list[tuple]]) -> dict:
    rows = [(image_id, _plain(result, image_id)[n]) for image_id, n in keys]
    n = max(len(rows), 1)
    groups = {i for i, gts in result.gt_index.items() if any(g.group_of for g in gts)}
    return {
        "count": len(rows),
        "size": dict(Counter(g.size for _, g in rows)),
        "occluded": round(sum(g.occluded for _, g in rows) / n, 3),
        "truncated": round(sum(g.truncated for _, g in rows) / n, 3),
        "depiction": round(sum(g.depiction for _, g in rows) / n, 3),
        "in_image_with_group_box": round(sum(i in groups for i, _ in rows) / n, 3),
        "overlaps_a_hand_box": round(sum(_overlap(g.box, hands.get(i, [])) for i, g in rows) / n, 3),
    }


def describe_false_positives(result: oe.ClassResult, threshold: float, hands: dict[str, list[tuple]],
                             boxes: dict[str, list[tuple[float, tuple]]]) -> dict:
    """False positives at ``threshold``: localisation (IoU >= 0.1 with a person) vs background, and
    whether background ones sit on a hand box. Localisation splits into duplicates (IoU >= 0.5 with a
    person already matched: an NMS problem) and misplaced boxes (IoU 0.1-0.5: a regression problem)."""
    fps = [s for s in result.scored if not s.tp and s.score >= threshold]
    n = max(len(fps), 1)
    loc = [s for s in fps if s.best_iou >= 0.1]
    dup = [s for s in loc if s.best_iou >= 0.5]
    bg = [s for s in fps if s.best_iou < 0.1]
    # The matcher does not keep the box; recover it from the prediction list by score.
    def box_of(s: oe.Scored) -> Optional[tuple]:
        return next((b for sc, b in boxes.get(s.image_id, []) if abs(sc - s.score) < 1e-9), None)
    bg_on_hand = sum(1 for s in bg if (b := box_of(s)) is not None and _overlap(b, hands.get(s.image_id, [])))
    return {"count": len(fps), "localisation": round(len(loc) / n, 3), "duplicate": round(len(dup) / n, 3),
            "misplaced": round((len(loc) - len(dup)) / n, 3), "background": round(len(bg) / n, 3),
            "background_on_a_hand_box": round(bg_on_hand / max(len(bg), 1), 3),
            "on_negative_images": sum(s.image_id in result.negative_ids for s in fps)}


# ---------------------------------------------------------------------- loading
def load_preds(path: Path) -> dict[str, dict[str, list[tuple[float, tuple]]]]:
    preds: dict[str, dict[str, list]] = {c: {} for c in CLASSES}
    for line in path.read_text().splitlines():
        if line:
            d = json.loads(line)
            if d["label"] in preds:
                preds[d["label"]].setdefault(d["id"], []).append((float(d["score"]), tuple(d["box"])))
    return preds


def load_truths(data: Path):
    """Val ground truth (bench._truths) and every hand box per image, verified or not."""
    from evaluation import manifest as mf
    from research.detector_bench import bench

    manifest = mf.load(data / "manifest.json")
    items = manifest.split("val")
    wanted = {i.id for i in items}
    rows = [r for r in (json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x)
            if r["id"] in wanted]
    truths = bench._truths(rows, {i.id: i.subject for i in items})
    hands = {r["id"]: [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in r["boxes"] if b["mid"] == bench.HAND]
             for r in rows}
    return manifest.id, truths, hands


# ---------------------------------------------------------------------- commands
def class_report(result: oe.ClassResult) -> dict[str, Any]:
    return {
        "npos": result.npos,
        "ap50": oe.average_precision(result.scored, result.npos),
        "precision_at_recall": {str(r): precision_at_recall(result, r) for r in RECALLS},
        "recall_at_precision": {str(p): recall_at_precision(result, p) for p in PRECISIONS},
        "ece": oe.calibration(result)["ece"],
    }


def analyze(data: Path, specs: list[str]) -> dict:
    manifest_id, truths, hands = load_truths(data)
    models = {}
    results: dict[str, dict[str, oe.ClassResult]] = {}
    preds_by: dict[str, dict] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        preds = load_preds(Path(path))
        preds_by[name] = preds
        results[name] = {c: oe.evaluate_class(preds[c], truths[c]) for c in CLASSES if preds[c]}
        models[name] = {c: class_report(r) for c, r in results[name].items()}
    report: dict[str, Any] = {"manifest_id": manifest_id, "split": "val", "models": models, "person": {}}
    for name, res in results.items():
        if "person" not in res:
            continue
        thr = precision_at_recall(res["person"], 0.7)["threshold"]
        if thr is None:
            continue
        report["person"][name] = {
            "threshold_at_recall_0.70": thr,
            "misses": describe_misses(res["person"], _missed(res["person"], thr), hands),
            "false_positives": describe_false_positives(res["person"], thr, hands, preds_by[name]["person"]),
        }
    names = [n for n in results if "person" in results[n]]
    if len(names) >= 2:  # people the first model finds and the second misses, at recall 0.70 each
        a, b = names[0], names[1]
        ra, rb = results[a]["person"], results[b]["person"]
        ta, tb = (precision_at_recall(r, 0.7)["threshold"] for r in (ra, rb))
        if ta is not None and tb is not None:
            ma, mb = _missed(ra, ta), _missed(rb, tb)
            report["person"]["found_by_first_missed_by_second"] = {
                "first": a, "second": b, **describe_misses(rb, mb - ma, hands)}
            report["person"]["found_by_second_missed_by_first"] = {
                "first": a, "second": b, **describe_misses(ra, ma - mb, hands)}
    return report


def merge(takes: list[str], out: Path) -> int:
    """Write one prediction file taking each class from the named source (architecture B)."""
    n = 0
    with out.open("w") as f:
        for take in takes:
            labels, _, path = take.partition("=")
            wanted = set(labels.split(","))
            for line in Path(path).read_text().splitlines():
                if line and json.loads(line)["label"] in wanted:
                    f.write(line + "\n")
                    n += 1
    return n


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--data", required=True)
    a.add_argument("--preds", action="append", required=True, help="name=path")
    a.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--take", action="append", required=True, help="label[,label]=path")
    m.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "merge":
        print(f"merged {merge(args.take, Path(args.out))} predictions into {args.out}")
        return 0
    report = analyze(Path(args.data), args.preds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "person_regression.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
