"""Export the manifest's splits to COCO-format JSON for detector fine-tuning.

    python -m research.openimages.to_coco --data $SDJOB_DATA/openimages/oi7-sd-v1 --out $SDJOB_DATA/coco3

Classes: 1 person, 2 phone, 3 hand. Group-of boxes become ``iscrowd: 1``
(ignored by COCO-style losses and evaluators). Open Images boxes are exhaustive
only for human-verified classes, so an image where a class was never verified
may contain unboxed instances. Training on it would teach "that is background".
``--require-verified`` keeps only images where every class is verified
(positive or negative); otherwise the count of such images is reported, so the
label-noise exposure is known. The ``test`` split is never exported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import manifest as mf

CLASSES = [(1, "person", {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}, "/m/01g317"),
           (2, "phone", {"/m/050k8"}, "/m/050k8"),
           (3, "hand", {"/m/0k65p"}, "/m/0k65p")]


def export(data: Path, out: Path, *, require_verified: bool) -> dict:
    manifest = mf.load(data / "manifest.json")
    split_of = {i.id: i.split for i in manifest.items}
    rows = [json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x]
    out.mkdir(parents=True, exist_ok=True)
    summary = {"manifest_id": manifest.id}
    for split in ("train", "val"):
        images, annotations, unverified = [], [], 0
        for row in rows:
            if split_of.get(row["id"]) != split:
                continue
            verified = set(row["verified_positive"]) | set(row["verified_negative"])
            if not all(key in verified for *_, key in CLASSES):
                unverified += 1
                if require_verified:
                    continue
            image_id = len(images) + 1
            w, h = row["width"], row["height"]
            images.append({"id": image_id, "file_name": f"images/{row['id']}.jpg", "width": w, "height": h,
                           "oi_id": row["id"]})
            for b in row["boxes"]:
                cat = next((c for c, _, mids, _ in CLASSES if b["mid"] in mids), None)
                if cat is None:
                    continue
                x, y = b["xmin"] * w, b["ymin"] * h
                bw, bh = (b["xmax"] - b["xmin"]) * w, (b["ymax"] - b["ymin"]) * h
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat,
                                    "bbox": [x, y, bw, bh], "area": bw * bh,
                                    "iscrowd": 1 if b["group_of"] == 1 else 0})
        doc = {"images": images, "annotations": annotations,
               "categories": [{"id": c, "name": n} for c, n, *_ in CLASSES]}
        (out / f"{split}.json").write_text(json.dumps(doc))
        per_class = {n: sum(a["category_id"] == c and not a["iscrowd"] for a in annotations) for c, n, *_ in CLASSES}
        summary[split] = {"images": len(images), "boxes": per_class,
                          "images_with_an_unverified_class": unverified, "require_verified": require_verified}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export manifest splits to COCO JSON (train, val only)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require-verified", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(export(Path(args.data), Path(args.out), require_verified=args.require_verified), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
