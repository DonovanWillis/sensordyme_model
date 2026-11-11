"""Make DAMO-YOLO (pinned commit 319572e) train correctly on PARTIALLY labelled images.

    python research/pilot/partial_labels_patch.py <DAMO-YOLO source dir>

Open Images boxes are exhaustive only for classes a human verified on an image
(research/openimages/label_audit.py: phone is unverified on 92% of training
images, hand on 39%). Stock COCO training treats every unboxed location as
background for EVERY class, so it learns "an unlabelled hand is not a hand".

This patch threads a per-image class mask from the COCO export
(`sd_unverified_categories`, research/openimages/to_coco.py) to the
classification loss, which then ignores, per (anchor, class), the classes that
were never verified on that image. Positives and box regression are unchanged.

Carrier: the mask rides as extra per-box columns through pull_item / mosaic /
random_affine / mixup (which keep rows), and as a per-box BoxList field; the
head takes the union over an image's boxes. Limitation: an image that ends up
with no boxes after augmentation carries no mask and is trained as before
(counted and logged by the head).

Every edit is an exact-string replacement with an assertion; on any mismatch
the patch refuses rather than guessing. It edits a COPY of the source (the job
copies the pinned tree first) and is idempotent.
"""

# ruff: noqa: ISC004 - the edit table pairs (old, new) source fragments built from adjacent literals

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# sensordyme-partial-labels"

EDITS: dict[str, list[tuple[str, str]]] = {
    "damo/dataset/datasets/coco.py": [
        (
            "    def __getitem__(self, inp):\n        if type(inp) is tuple:",
            "    def _sd_unverified_row(self, idx):  " + MARK + "\n"
            "        info = self.coco.imgs[self.id_to_img_map[idx]]\n"
            "        row = np.zeros(len(self.contiguous_class2id), dtype=np.float32)\n"
            "        for c in info.get('sd_unverified_categories', []):\n"
            "            row[self.contiguous_class2id[self.ori_id2class[c]]] = 1.0\n"
            "        return row\n\n"
            "    def __getitem__(self, inp):\n        if type(inp) is tuple:",
        ),
        (
            "        classes = torch.tensor(classes)\n        target.add_field('labels', classes)\n",
            "        classes = torch.tensor(classes)\n        target.add_field('labels', classes)\n"
            "        _sd_row = self._sd_unverified_row(idx)  " + MARK + "\n"
            "        target.add_field('sd_unverified', torch.as_tensor(np.tile(_sd_row, (len(classes), 1)))"
            ".reshape(len(classes), len(_sd_row)))\n",
        ),
        (
            "        res = np.zeros((len(target.bbox), 5))\n        for idx in range(len(target.bbox)):\n"
            "            res[idx, 0:4] = target.bbox[idx]\n            res[idx, 4] = classes[idx]\n",
            "        _sd_row = self._sd_unverified_row(idx)  " + MARK + "\n"
            "        res = np.zeros((len(target.bbox), 5 + len(_sd_row)))\n        for k in range(len(target.bbox)):\n"
            "            res[k, 0:4] = target.bbox[k]\n            res[k, 4] = classes[k]\n            res[k, 5:] = _sd_row\n",
        ),
    ],
    "damo/dataset/datasets/mosaic_wrapper.py": [
        (
            "            classes = torch.tensor(classes)[valid_idx]\n            target.add_field('labels', classes.long())\n",
            "            classes = torch.tensor(classes)[valid_idx]\n            target.add_field('labels', classes.long())\n"
            "            _sd = np.asarray(labels, dtype=np.float32)  " + MARK + "\n"
            "            _sd = _sd[:, 5:] if _sd.ndim == 2 else np.zeros((0, 0), dtype=np.float32)\n"
            "            target.add_field('sd_unverified', torch.as_tensor(_sd)[valid_idx])\n",
        ),
        (
            "        cls_labels = cp_labels[:, 4:5].copy()",
            "        cls_labels = cp_labels[:, 4:].copy()  " + MARK + " keep the mask columns",
        ),
    ],
    "damo/base_models/losses/gfocal_loss.py": [
        (
            "def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):",
            "def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True, class_mask=None):  " + MARK,
        ),
        (
            "    loss = loss.sum(dim=1, keepdim=False)\n    return loss\n",
            "    if class_mask is not None:  " + MARK + " (N, C): 0 = class never verified on this image\n"
            "        loss = loss * class_mask\n"
            "    loss = loss.sum(dim=1, keepdim=False)\n    return loss\n",
        ),
        (
            "                weight=None,\n                avg_factor=None,\n                reduction_override=None):\n"
            "        \"\"\"Forward function.\n        Args:\n            pred (torch.Tensor): Predicted joint representation of\n"
            "                classification and quality (IoU) estimation with shape (N, C),",
            "                weight=None,\n                avg_factor=None,\n                reduction_override=None,\n"
            "                class_mask=None):  " + MARK + "\n"
            "        \"\"\"Forward function.\n        Args:\n            pred (torch.Tensor): Predicted joint representation of\n"
            "                classification and quality (IoU) estimation with shape (N, C),",
        ),
        (
            "            use_sigmoid=self.use_sigmoid,\n            reduction=reduction,\n            avg_factor=avg_factor)\n"
            "        return loss_cls\n",
            "            use_sigmoid=self.use_sigmoid,\n            reduction=reduction,\n            avg_factor=avg_factor,\n"
            "            class_mask=class_mask)  " + MARK + "\n        return loss_cls\n",
        ),
    ],
    "damo/base_models/heads/zero_head.py": [
        (
            "            for label in labels:\n                gt_bbox_list.append(label.bbox)\n"
            "                gt_cls_list.append((label.get_field('labels')).long())\n",
            "            sd_masks = []  " + MARK + "\n"
            "            for label in labels:\n                gt_bbox_list.append(label.bbox)\n"
            "                gt_cls_list.append((label.get_field('labels')).long())\n"
            "                if label.has_field('sd_unverified') and len(label) > 0:\n"
            "                    sd_masks.append(label.get_field('sd_unverified').float().max(dim=0)[0])\n"
            "                else:\n"
            "                    sd_masks.append(None)\n"
            "            self._sd_unverified = sd_masks\n",
        ),
        (
            "        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),\n"
            "                                 avg_factor=num_total_pos)\n",
            "        class_mask = None  " + MARK + "\n"
            "        sd_masks = getattr(self, '_sd_unverified', None)\n"
            "        if sd_masks is not None and any(m is not None for m in sd_masks):\n"
            "            n_img = len(sd_masks)\n"
            "            per_img = torch.stack([\n"
            "                (1.0 - m.to(cls_scores.device)) if m is not None\n"
            "                else cls_scores.new_ones(self.cls_out_channels) for m in sd_masks])\n"
            "            class_mask = per_img[:, None, :].expand(n_img, cls_scores.shape[0] // n_img,\n"
            "                                                    self.cls_out_channels).reshape(-1, self.cls_out_channels)\n"
            "            self._sd_stats = (sum(m is None for m in sd_masks), n_img)\n"
            "        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),\n"
            "                                 avg_factor=num_total_pos, class_mask=class_mask)\n",
        ),
    ],
}


def apply(root: Path) -> list[str]:
    done = []
    for rel, edits in EDITS.items():
        path = root / rel
        text = path.read_text()
        if MARK in text:
            done.append(f"{rel}: already patched")
            continue
        for old, new in edits:
            if text.count(old) != 1:
                raise SystemExit(f"{rel}: expected exactly one match for an edit, found {text.count(old)}: "
                                 f"{old[:70]!r}. Refusing to patch an unexpected source tree.")
            text = text.replace(old, new)
        path.write_text(text)
        done.append(f"{rel}: {len(edits)} edits")
    return done


if __name__ == "__main__":
    for line in apply(Path(sys.argv[1])):
        print(line)
