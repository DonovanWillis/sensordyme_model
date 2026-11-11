"""EXP-08a: run a DAMO-YOLO checkpoint over the manifest's val split, write predictions as JSON lines.

Uses DAMO-YOLO's own test-time preprocessing (its config: stretch to 640x640, RGB)
and its own postprocessing (multiclass NMS), so the numbers describe the trained
model, not a re-implementation. Scoring happens separately, under the Open Images
protocol (research.detector_bench.bench score). Never touches the test split.

    python research/pilot/predict_val.py --damo <DAMO-YOLO dir> --config configs/exp08a.py \
        --ckpt <ckpt.pth> --data <oi7-sd-v1> --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--damo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default="person,phone,hand")
    p.add_argument("--conf", type=float, default=0.05)
    args = p.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    os.chdir(args.damo)  # the config resolves its backbone structure file relative to the repo
    sys.path.insert(0, args.damo)
    from damo.base_models.core.ops import RepConv
    from damo.config.base import parse_config
    from damo.detectors.detector import build_local_model
    from damo.structures.image_list import ImageList
    from damo.utils.demo_utils import transform_img

    labels = args.labels.split(",")
    cfg = parse_config(args.config)
    cfg.model.head.nms_conf_thre = args.conf
    model = build_local_model(cfg, "cuda")
    ckpt = torch.load(args.ckpt, map_location="cuda", weights_only=True)  # no arbitrary unpickling
    model.load_state_dict(ckpt["model"], strict=True)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    model.eval()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation import manifest as mf

    manifest = mf.load(Path(args.data) / "manifest.json")
    items = sorted(manifest.split("val"), key=lambda i: i.id)
    started, n = time.perf_counter(), 0
    with open(args.out, "w") as out, torch.no_grad():
        for item in items:
            image = np.asarray(Image.open(Path(args.data) / item.path).convert("RGB"))
            h, w = image.shape[:2]
            tensor = transform_img(image, 0, **cfg.test.augment.transform, infer_size=[640, 640]).tensors
            batch = ImageList(tensor.to("cuda"), [tensor.shape[-2:]], [tensor.shape[-2:]])
            result = model(batch)[0].resize((w, h))
            boxes = result.bbox.cpu().numpy()
            scores = result.get_field("scores").cpu().numpy()
            classes = result.get_field("labels").cpu().numpy().astype(int)
            for (bx1, by1, bx2, by2), s, c in zip(boxes, scores, classes, strict=True):
                x1, x2 = max(0.0, bx1 / w), min(1.0, bx2 / w)
                y1, y2 = max(0.0, by1 / h), min(1.0, by2 / h)
                if x2 > x1 and y2 > y1 and 0 <= c < len(labels):
                    out.write(json.dumps({"id": item.id, "label": labels[c], "score": float(s),
                                          "box": [float(x1), float(y1), float(x2), float(y2)]}) + "\n")
            n += 1
    print(f"predicted {n} val images in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
