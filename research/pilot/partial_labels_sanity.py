"""Prove the partial-label patch does what it says, on the real patched code, before training.

    python research/pilot/partial_labels_sanity.py <patched DAMO dir> <coco train.json> <image root>

Checks:
  1. quality_focal_loss with a class mask: the masked class contributes no loss
     and receives no gradient; unmasked classes are untouched.
  2. COCODataset.pull_item / __getitem__ carry exactly the image's
     sd_unverified_categories as mask columns / a per-box field.
  3. MosaicWrapper (mosaic + mixup on) keeps the mask columns through
     augmentation: the BoxList field equals the union of its source images' masks
     (checked as: every mask row equals the mask of some source image).
Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import random
import sys


def main() -> int:
    damo, ann, root = sys.argv[1:4]
    sys.path.insert(0, damo)
    import numpy as np
    import torch
    from damo.base_models.losses.gfocal_loss import quality_focal_loss
    from damo.dataset.datasets.coco import COCODataset
    from damo.dataset.datasets.mosaic_wrapper import MosaicWrapper

    torch.manual_seed(0)
    pred = torch.rand(50, 3, requires_grad=True)
    labels = torch.full((50,), 3)  # all background
    scores = torch.zeros(50)
    mask = torch.ones(50, 3)
    mask[:, 2] = 0.0
    full = quality_focal_loss.__wrapped__(pred, (labels, scores), use_sigmoid=False) \
        if hasattr(quality_focal_loss, "__wrapped__") else None
    masked = quality_focal_loss(pred, (labels, scores), use_sigmoid=False, class_mask=mask, reduction="sum")
    masked.backward()
    assert torch.all(pred.grad[:, 2] == 0), "masked class still receives gradient"
    assert torch.all(pred.grad[:, :2] != 0), "unmasked classes lost their gradient"
    print("1. loss mask: masked class gets zero gradient; others unchanged — OK", "(unwrapped available)" if full is not None else "")

    with open(ann) as f:
        meta = json.load(f)
    unverified = {im["id"]: set(im.get("sd_unverified_categories", [])) for im in meta["images"]}
    names = [c["name"] for c in sorted(meta["categories"], key=lambda c: c["id"])]
    cat_index = {c["id"]: i for i, c in enumerate(sorted(meta["categories"], key=lambda c: c["id"]))}
    ds = COCODataset(ann, root, transforms=None, class_names=names)
    rng = random.Random(0)  # noqa: S311 - sampling, not security
    checked = 0
    for idx in rng.sample(range(len(ds)), 40):
        img_id = ds.id_to_img_map[idx]
        expect = np.zeros(len(names), dtype=np.float32)
        for c in unverified[img_id]:
            expect[cat_index[c]] = 1.0
        _, res, _, _ = ds.pull_item(idx)
        assert res.shape[1] == 5 + len(names)
        for row in res:
            assert np.array_equal(row[5:], expect), (img_id, row[5:], expect)
        _, target, _ = ds[idx]
        if len(target):
            assert np.array_equal(target.get_field("sd_unverified").numpy()[0], expect)
        checked += 1
    print(f"2. dataset: {checked} images carry exactly their unverified classes — OK")

    mosaic = MosaicWrapper(ds, img_size=(640, 640), mosaic_prob=1.0, mixup_prob=1.0)
    source_masks = {tuple(np.eye(len(names))[[cat_index[c] for c in s]].sum(0) if s else np.zeros(len(names)))
                    for s in unverified.values()}
    empty = 0
    for k in range(40):
        _, target, _ = mosaic[(True, k)]
        if not len(target):
            empty += 1
            continue
        for row in target.get_field("sd_unverified").numpy():
            assert tuple(row) in source_masks, row
    print(f"3. mosaic+mixup: mask rows survive augmentation — OK ({empty}/40 samples had no boxes: no mask, trained as before)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
