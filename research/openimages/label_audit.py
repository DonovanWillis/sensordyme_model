"""Audit label completeness in the Open Images subset before training on it.

    python -m research.openimages.label_audit --data <oi7-sd-v1> --out DIR

Open Images boxes are exhaustive only for classes a human VERIFIED on an image.
A class can be, per image:
  positive   verified present (boxes exhaustive, except group-of)
  negative   verified absent
  unknown    never verified: unboxed instances may exist
Training must treat only `negative` as a confirmed absence. This reports, per
split and class, how often each case occurs, plus contradictions (a box for a
verified-absent class; verified present but no box), suspicious geometry
(degenerate, whole-image, extreme aspect), and ambiguous hand boxes (group-of
hands, hands that are mostly a phone box). It changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import manifest as mf

CLASSES = {"person": ({"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}, "/m/01g317"),
           "phone": ({"/m/050k8"}, "/m/050k8"),
           "hand": ({"/m/0k65p"}, "/m/0k65p")}


def class_state(row: dict, mids: set[str], key: str) -> str:
    pos, neg = set(row["verified_positive"]), set(row["verified_negative"])
    if key in pos:
        return "positive"
    if key in neg:
        return "negative"
    return "unknown"


def _overlap_frac(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def audit(rows: list[dict], split_of: dict[str, str]) -> dict:
    out: dict = defaultdict(lambda: defaultdict(Counter))
    pattern: dict = defaultdict(Counter)
    for row in rows:
        split = split_of[row["id"]]
        states = {}
        for name, (mids, key) in CLASSES.items():
            boxes = [b for b in row["boxes"] if b["mid"] in mids]
            state = class_state(row, mids, key)
            states[name] = state
            c = out[split][name]
            c[f"images_{state}"] += 1
            c["boxes"] += len(boxes)
            if boxes and state == "negative":
                c["contradiction_box_on_verified_absent"] += 1
            if boxes and state == "unknown":
                c["boxes_on_unverified_image"] += 1  # boxed but not verified: present, possibly incomplete
            if state == "positive" and not boxes:
                c["contradiction_verified_present_no_box"] += 1
            for b in boxes:
                w, h = b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]
                if w <= 0 or h <= 0:
                    c["box_degenerate"] += 1
                elif w * h > 0.95:
                    c["box_whole_image"] += 1
                elif max(w / h, h / w) * (row["height"] / row["width"] if w > h else row["width"] / row["height"]) > 8:
                    c["box_extreme_aspect"] += 1
                if b["group_of"] == 1:
                    c["box_group_of"] += 1
                if b.get("depiction") == 1:
                    c["box_depiction"] += 1
        phones = [(b["xmin"], b["ymin"], b["xmax"], b["ymax"]) for b in row["boxes"] if b["mid"] == "/m/050k8"]
        for b in row["boxes"]:
            if b["mid"] == "/m/0k65p":
                hb = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                if any(_overlap_frac(hb, p) > 0.6 for p in phones):
                    out[split]["hand"]["hand_box_mostly_phone"] += 1
        pattern[split][" ".join(f"{k}:{v[:3]}" for k, v in sorted(states.items()))] += 1
    result = {split: {cls: dict(c) for cls, c in classes.items()} for split, classes in out.items()}
    for split, p in pattern.items():
        result[split]["state_patterns"] = dict(p.most_common())
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Label-completeness audit (read-only)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = Path(args.data)
    manifest = mf.load(data / "manifest.json")
    split_of = {i.id: i.split for i in manifest.items}
    rows = [json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x]
    result = {"manifest_id": manifest.id, **audit([r for r in rows if r["id"] in split_of], split_of)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "label_audit.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
