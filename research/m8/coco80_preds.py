"""M8 CV: all-80-class predictions of the shipping DAMO-YOLO-T float ONNX on Open Images val.

    python -u -m research.m8.coco80_preds --data <oi7-sd-v1> --models <dir> --out-preds <dir> \
        --provider dml --report <SDJOB_OUT>

Why: the earlier bench kept only person/phone. Phone look-alikes (remote, tv, laptop, ...)
and person/phone/hand geometry need every class the network emits. Both product decoders
(core/accel.py legacy and reference) are run on the same raw head tensors, at the bench's
score floor (0.05) and NMS IoU (0.5), so the person/phone rows must reproduce the Mission 5
numbers (person AP50 0.656 legacy / 0.713 reference; phone 0.728 / 0.795) up to backend noise.

Writes JSON lines {id, label, score, box (normalized xyxy)} with label = COCO name, except
"cell phone" -> "phone" so every existing scorer reads the file unchanged. Val split only;
the frozen test split is refused. No training; the model file is verified by sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.accel import COCO_NAMES
from evaluation import manifest as mf
from research.detector_bench import models as dm

#: The float source of the shipping .hef (bench.CHECKPOINTS), pinned to the sha256 recorded in
#: core/model_registry.yaml (model_sha256) and every Mission 4-7 bench report.
MODEL_URL = ("https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/"
             "damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip")
MODEL_MEMBER = "damoyolo_tinynasL20_T.onnx"
MODEL_SHA256 = "02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657"
FLOOR, NMS_IOU, MAX_DETS = 0.05, 0.5, 100


def _name(c: int) -> str:
    return "phone" if c == 67 else COCO_NAMES[c]


def ensure_model(models: Path) -> Path:
    target = models / f"{Path(MODEL_MEMBER).stem}.onnx"
    if not target.exists():
        models.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MODEL_URL, timeout=600) as r:
            body = r.read()
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            data = z.read(MODEL_MEMBER)
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise SystemExit("downloaded model does not match the pinned sha256; refusing")
        target.write_bytes(data)
    got = hashlib.sha256(target.read_bytes()).hexdigest()
    if got != MODEL_SHA256:
        raise SystemExit(f"{target}: sha256 {got} != pinned {MODEL_SHA256}")
    return target


def normalize(boxes, scores, classes, scale, left, top, w, h):
    out = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes, strict=True):
        nx1 = min(max((x1 - left) / scale, 0.0), w) / w
        ny1 = min(max((y1 - top) / scale, 0.0), h) / h
        nx2 = min(max((x2 - left) / scale, 0.0), w) / w
        ny2 = min(max((y2 - top) / scale, 0.0), h) / h
        if nx2 > nx1 and ny2 > ny1:
            out.append({"label": _name(int(c)), "score": float(s), "box": [float(v) for v in (nx1, ny1, nx2, ny2)]})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out-preds", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--provider", default="dml", choices=("dml", "cuda", "cpu"))
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    data, out_preds, report_dir = Path(args.data), Path(args.out_preds), Path(args.report)
    out_preds.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    model_path = ensure_model(Path(args.models))
    manifest = mf.load(data / "manifest.json")
    items = sorted(manifest.split(args.split), key=lambda i: i.id)[: args.limit]
    bad = mf.verify_files(manifest, data, items, decode=False)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the manifest: {bad[:3]}")
    net = dm.DamoYolo(str(model_path), args.provider, variant="x", decoder="device_legacy")
    providers = net.session.get_providers()
    print("providers", providers, flush=True)
    final = {d: out_preds / f"coco80_{d}_{args.split}.jsonl" for d in ("legacy", "reference")}
    files = {d: p.with_suffix(".part").open("w") for d, p in final.items()}  # renamed only when complete
    infer_ms, t0 = [], time.perf_counter()
    counts = dict.fromkeys(files, 0)
    for n, item in enumerate(items):
        if n and n % 200 == 0:
            print(f"  {n}/{len(items)} {time.perf_counter() - t0:.0f}s", flush=True)
            (report_dir / "coco80_progress.json").write_text(json.dumps({"done": n, "of": len(items)}))
        with Image.open(data / item.path) as im:
            image = im.convert("RGB")
        canvas, sx, _sy, left, top = dm.preprocess(image, net.size, "letterbox114")
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        ti = time.perf_counter()
        outputs = net.session.run(net.outputs, {net.input: tensor})
        infer_ms.append((time.perf_counter() - ti) * 1e3)
        heads = dm.split_heads(outputs, net.size)
        for decoder, f in files.items():
            boxes, scores, classes = dm.device_decode(heads, decoder, FLOOR, NMS_IOU, net.size)
            for det in normalize(boxes[:MAX_DETS], scores[:MAX_DETS], classes[:MAX_DETS], sx, left, top,
                                 *image.size):
                f.write(json.dumps({"id": item.id, **det}) + "\n")
                counts[decoder] += 1
    for d, f in files.items():
        f.close()
        final[d].with_suffix(".part").replace(final[d])
    shas = {d: hashlib.sha256((out_preds / f"coco80_{d}_{args.split}.jsonl").read_bytes()).hexdigest()
            for d in files}
    report = {"split": args.split, "manifest_id": manifest.id, "images": len(items), "model_sha256": MODEL_SHA256,
              "providers": providers, "floor": FLOOR, "nms_iou": NMS_IOU, "max_dets": MAX_DETS,
              "detections": counts, "preds_sha256": shas, "preds_dir": str(out_preds),
              "infer_ms_p50": dm.percentile(infer_ms, 0.5), "infer_ms_p90": dm.percentile(infer_ms, 0.9),
              "wall_s": round(time.perf_counter() - t0, 1)}
    (report_dir / "coco80_report.json").write_text(json.dumps(report, indent=1))
    (out_preds / f"coco80_{args.split}.complete.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
