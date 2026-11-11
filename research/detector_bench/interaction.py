"""Phone visibility vs phone-in-hand: does requiring a hand near the phone cut false phones?

    python -m research.detector_bench.interaction --data <oi7-sd-v1> --preds a.jsonl b.jsonl --out DIR

Image-level proxy only (still photos): it measures whether a hand-overlap gate
removes false phone boxes and which true phones it keeps. It does not measure
phone USE while driving, which needs video, gaze and time (evaluation/).

Hands come from MediaPipe Hands (the product's own hand model; static images,
up to 4 hands, box = landmark extent + 10%). An oracle gate using the human hand
boxes (only on images where "hand" was human-verified) shows the ceiling.
Phones count as "in hand" in the ground truth when a human phone box overlaps a
human hand box. The val split only; nothing is tuned on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import manifest as mf
from research.detector_bench import oid_eval as oe

PHONE, HAND = "/m/050k8", "/m/0k65p"
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)


def _overlap_frac(phone, hand) -> float:
    ix = max(0.0, min(phone[2], hand[2]) - max(phone[0], hand[0]))
    iy = max(0.0, min(phone[3], hand[3]) - max(phone[1], hand[1]))
    area = max((phone[2] - phone[0]) * (phone[3] - phone[1]), 1e-9)
    return ix * iy / area


def _near_hand(phone, hands, min_overlap: float = 0.1) -> bool:
    return any(_overlap_frac(phone, h) >= min_overlap for h in hands)


def mediapipe_hands(paths: dict[str, Path]) -> dict[str, list[tuple]]:
    import mediapipe as mp
    import numpy as np
    from PIL import Image

    out = {}
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5) as hands:
        for image_id, path in paths.items():
            image = np.asarray(Image.open(path).convert("RGB"))
            result = hands.process(image)
            boxes = []
            for lm in result.multi_hand_landmarks or []:
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                boxes.append((max(0.0, x1 - pad_x), max(0.0, y1 - pad_y), min(1.0, x2 + pad_x), min(1.0, y2 + pad_y)))
            out[image_id] = boxes
    return out


def analyse(rows: list[dict], groups: dict[str, str], preds: dict[str, list], mp_hands: dict[str, list]) -> dict:
    report: dict = {}
    truths, gt_hands, in_hand = [], {}, {}
    for row in rows:
        pos, neg = set(row["verified_positive"]), set(row["verified_negative"])
        phones = [b for b in row["boxes"] if b["mid"] == PHONE]
        verified = (PHONE in pos and bool(phones)) or (PHONE in neg and PHONE not in pos)
        if not verified:
            continue
        hands = [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in row["boxes"] if b["mid"] == HAND]
        if HAND in pos or HAND in neg:
            gt_hands[row["id"]] = hands
        gts = [oe.GroundTruth((b["xmin"], b["ymin"], b["xmax"], b["ymax"]), b["group_of"] == 1) for b in phones]
        in_hand[row["id"]] = [bool(hands) and _near_hand(g.box, hands) for g in gts if not g.group_of]
        truths.append(oe.ImageTruth(row["id"], groups[row["id"]], gts, True, PHONE in neg and not phones))
    plain = [x for v in in_hand.values() for x in v]
    report["ground_truth"] = {"verified_images": len(truths), "phone_boxes": len(plain),
                              "phones_overlapping_a_human_hand_box": sum(plain),
                              "images_with_hand_verified": len(gt_hands)}
    gates = {
        "none": lambda image_id, box: True,
        "mediapipe_hand": lambda image_id, box: _near_hand(box, mp_hands.get(image_id, [])),
        "oracle_human_hand": lambda image_id, box: (image_id in gt_hands and _near_hand(box, gt_hands[image_id])),
    }
    for t in THRESHOLDS:
        for name, gate in gates.items():
            kept = {i: [(s, b) for s, b in dets if s >= t and gate(i, b)] for i, dets in preds.items()}
            result = oe.evaluate_class(kept, truths)
            op = oe.operating_point(result, t)
            matched = {(s.image_id, s.matched) for s in result.scored if s.tp and s.matched is not None}
            hit_in, hit_out, n_in, n_out = 0, 0, 0, 0
            for image_id, flags in in_hand.items():
                for k, held in enumerate(flags):
                    hit = (image_id, k) in matched
                    if held:
                        n_in, hit_in = n_in + 1, hit_in + hit
                    else:
                        n_out, hit_out = n_out + 1, hit_out + hit
            report[f"t{t}:{name}"] = {
                "recall": op["recall"], "precision": op["precision"], "fp": op["fp"],
                "fp_on_phone_free_images": round(op["fp_per_negative_image"] * result.negative_images)
                if op["fp_per_negative_image"] is not None else None,
                "phone_free_images_with_a_false_phone": op["negative_images_with_a_false_box"],
                "recall_phone_in_human_hand": hit_in / n_in if n_in else None,
                "recall_phone_not_in_hand": hit_out / n_out if n_out else None,
                "n_in_hand": n_in, "n_not_in_hand": n_out,
            }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--data", required=True)
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = Path(args.data)
    manifest = mf.load(data / "manifest.json")
    items = {i.id: i for i in manifest.split(args.split)}
    rows = [r for r in (json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x)
            if r["id"] in items and (PHONE in r["verified_positive"] or PHONE in r["verified_negative"])]
    groups = {i: items[i].subject for i in items}
    mp_hands = mediapipe_hands({r["id"]: data / items[r["id"]].path for r in rows})
    found = sum(1 for v in mp_hands.values() if v)
    out = {"split": args.split, "manifest_id": manifest.id, "images": len(rows),
           "mediapipe_images_with_a_hand": found, "runs": {}}
    for path in args.preds:
        preds: dict[str, list] = {}
        for line in Path(path).read_text().splitlines():
            d = json.loads(line)
            if d["label"] == "phone" and d["id"] in mp_hands:
                preds.setdefault(d["id"], []).append((d["score"], tuple(d["box"])))
        out["runs"][Path(path).stem] = analyse(rows, groups, preds, mp_hands)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"interaction_{args.split}.json").write_text(json.dumps(out, indent=1))
    for run, rep in out["runs"].items():
        print(run, json.dumps(rep["ground_truth"]))
        for key, v in rep.items():
            if key.startswith("t"):
                print(f"  {key:26s} R {v['recall']} P {v['precision']} FPfree {v['fp_on_phone_free_images']} "
                      f"negimg {v['phone_free_images_with_a_false_phone']} R_inhand {v['recall_phone_in_human_hand']} "
                      f"R_notinhand {v['recall_phone_not_in_hand']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
