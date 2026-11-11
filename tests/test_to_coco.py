"""research/openimages/to_coco.py: COCO export with per-image unverified categories."""

from __future__ import annotations

import json

from evaluation import manifest as mf
from research.openimages.to_coco import export

PERSON, PHONE, HAND = "/m/01g317", "/m/050k8", "/m/0k65p"


def _box(mid, group_of=0):
    return {"mid": mid, "xmin": 0.1, "xmax": 0.3, "ymin": 0.2, "ymax": 0.5, "group_of": group_of}


def test_export_records_unverified_categories_and_crowd(tmp_path):
    rows = [
        {"id": "a", "width": 100, "height": 50, "boxes": [_box(PERSON), _box(HAND, 1)],
         "verified_positive": [PERSON, HAND], "verified_negative": []},
        {"id": "b", "width": 100, "height": 50, "boxes": [_box(PHONE)],
         "verified_positive": [PHONE], "verified_negative": [PERSON, HAND]},
    ]
    items = [mf.Item(id=r["id"], path=f"images/{r['id']}.jpg", sha256=f"{n:064x}", bytes=1, subject=r["id"],
                     session=r["id"], split="train") for n, r in enumerate(rows)]
    mf.save(mf.Manifest(name="t", version="1", licence={"class": "evaluation_only", "id": "x"}, items=items),
            tmp_path / "manifest.json")
    (tmp_path / "boxes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = export(tmp_path, tmp_path / "coco", require_verified=False)
    doc = json.loads((tmp_path / "coco" / "train.json").read_text())
    by_name = {im["oi_id"]: im for im in doc["images"]}
    assert by_name["a"]["sd_unverified_categories"] == [2]  # phone never verified on "a"
    assert by_name["b"]["sd_unverified_categories"] == []
    assert summary["train"]["images_with_an_unverified_class"] == 1
    crowd = [a for a in doc["annotations"] if a["category_id"] == 3]
    assert len(crowd) == 1 and crowd[0]["iscrowd"] == 1  # group-of hand -> ignored crowd box
    assert doc["annotations"][0]["bbox"] == [10.0, 10.0, 20.0, 15.0]
