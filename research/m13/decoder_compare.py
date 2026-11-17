"""M13: legacy vs reference decoder on IDENTICAL float head tensors (no Pi, no INT8).

RUNS ON A WORKER (oryn, CPU only), never on the Pi. Read-only: no training, val
split only (the frozen test split is never read), model files verified by sha256.

What runs is the device's own inference code, core/accel.py HailoYOLO, behind
scripts/hw_validation/fake_hailo.py: the float ONNX of the shipping
damoyolo_tinynasL20_T answers `pipeline.infer` on the exact uint8 tensor the
device code built (research/m83/inference_path_val.py, the same procedure). Each
frame's six head tensors are then decoded by BOTH product decoders
(`HailoYOLO._decode_raw` with decoder "legacy" and "reference") with the call the
driver_attention module makes on the device: its classes (config.yaml
yolo_classes + person) at its conf (0.20).

Reported (report.json; every figure is "float tensors, x86 worker CPU", NOT INT8,
NOT a Pi):

  agreement   per class group at the product gates (person >= 0.50, phone >= 0.20,
              the eating/drinking classes >= 0.20): counts under each decoder,
              same-class IoU-matched boxes (IoU >= 0.5), IoU distribution, score
              delta, the centre shift per stride (the anchor offset), and the
              images where only one decoder reports a phone / a person.
  timing      decode wall time per frame for each decoder on the same tensors,
              alternated A/B per frame (x86 CPU; the Pi is slower).
  tensors     the raw heads (float32 .npz) for the 6 known-good images and the
              first --save-n val images where either decoder reports a person or
              a phone, plus the per-frame letterbox geometry, so a Mac can time
              both decoders on identical real-shaped inputs
              (benchmarks/m13_decoder_latency.py).

    python -u -m research.m13.decoder_compare --data $SDJOB_DATA/openimages/oi7-sd-v1 \\
        --onnx $SDJOB_DATA/models/damoyolo_tinynasL20_T.onnx --heads-onnx $SDJOB_DATA/m101/zoo.heads.onnx \\
        --out $SDJOB_OUT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

import numpy as np  # noqa: E402

from research.m83 import inference_path_val as ipv  # noqa: E402

#: The device's detector call for driver_attention (config.yaml modules.driver_attention).
RUN_CONF = 0.20
PERSON, PHONE = 0, 67
#: Product gates (docs/research/M11_MODEL_DECISION.md; config.yaml person_confidence_threshold).
GATES = {"person": ({PERSON}, 0.50), "phone": ({PHONE}, 0.20)}


def product_classes() -> dict[str, set[int]]:
    """config.yaml's driver_attention yolo_classes as COCO ids, split into groups."""
    import yaml

    from core.accel import COCO_NAMES

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    names = cfg["modules"]["driver_attention"]["yolo_classes"]
    ids = {cid for cid, name in COCO_NAMES.items() if name in names}
    return {"all": ids | {PERSON}, "eating": ids - {PHONE}}


def to_detections(result, hd):
    boxes = list(result.boxes)
    if not boxes:
        return hd.Detections.empty()
    xyxy = np.array([[float(v) for v in b.xyxy[0]] for b in boxes], dtype=np.float64)
    scores = np.array([float(b.conf[0]) for b in boxes], dtype=np.float64)
    classes = np.array([int(b.cls[0]) for b in boxes], dtype=np.int64)
    # Stride is not reported by the product decoder; 0 groups every centre shift together.
    return hd.Detections(xyxy, scores, classes, np.zeros(len(boxes), dtype=np.int64))


def gate(d, hd, classes: set[int], conf: float):
    keep = [i for i in range(len(d)) if int(d.classes[i]) in classes and d.scores[i] >= conf]
    return d.take(keep) if keep else hd.Detections.empty()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--heads-onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--save-n", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tensors_dir = args.out / "tensors"
    tensors_dir.mkdir(exist_ok=True)

    import cv2
    import fake_hailo
    import hwv_decode as hd
    import onnxruntime as ort
    from PIL import Image

    from core import accel
    from core.accel import HailoYOLO
    from evaluation import manifest as mf

    got = ipv.sha256_file(args.onnx)
    if got != ipv.ONNX_SHA256:
        raise SystemExit(f"{args.onnx}: sha256 {got} != pinned {ipv.ONNX_SHA256}")
    sess = ipv.build_session(args.heads_onnx, args.threads)
    groups = product_classes()
    run_classes = sorted(groups["all"])
    provenance = {
        "job": os.environ.get("SDJOB_ID"), "onnx_sha256": got, "heads_onnx_sha256": ipv.sha256_file(args.heads_onnx),
        "providers": sess.get_providers(), "threads": args.threads,
        "versions": {"onnxruntime": ort.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
        "core_accel_sha256": ipv.sha256_file(ROOT / "core" / "accel.py"),
        "run_conf": RUN_CONF, "run_classes": run_classes, "gates": {k: [sorted(v[0]), v[1]] for k, v in GATES.items()},
        "label": "float tensors (x86 ONNX stand-in for the chip), worker CPU: NOT INT8, NOT a Pi",
    }
    print(json.dumps(provenance, indent=1), flush=True)

    captured: dict = {}
    base = fake_hailo.onnx_backend(sess)

    def backend(batch):
        heads = base(batch)
        captured["batch"], captured["heads"] = batch, heads
        return heads

    fake_hailo.install(backend)
    accel.record_inference = lambda: None
    model = HailoYOLO("damoyolo_tinynasL20_T.hef (simulated: float ONNX)", decoder="legacy")
    decode = model._decode_raw  # bound method; `model.decoder` selects the path

    def both(heads, left, top, w, h, order):
        out, ms = {}, {}
        for name in order:
            model.decoder = name
            t0 = time.perf_counter()
            (res,) = decode(heads, 1.0, left, top, w, h, run_classes, RUN_CONF)
            ms[name] = (time.perf_counter() - t0) * 1e3
            out[name] = to_detections(res, hd)
        model.decoder = "legacy"
        return out, ms

    scales_hd = hd.group_heads(dict(fake_hailo.OUTPUTS), 640)
    specs = {"legacy": hd.LEGACY, "reference": hd.DEVICE_REFERENCE}
    parity = {"frames": 0, "equal": 0, "max_abs_diff": 0.0}

    def offline(heads, geo):
        """The pinned offline twins of the two product decoders (hwv_decode), which also
        report each box's stride, so the centre shift can be read per FPN level."""
        g = {"mode": "letterbox", "scale": 1.0, "left": geo["left"], "top": geo["top"],
             "frame_w": geo["w"], "frame_h": geo["h"]}
        return {k: hd.unmap(hd.decode(heads, scales_hd, spec, RUN_CONF, classes=run_classes), g)
                for k, spec in specs.items()}

    def check_parity(product, off):
        for k in product:
            a, b = ipv.rows_of_dets(product[k]), ipv.rows_of_dets(off[k])
            ok, diff = ipv.parity(a, b)
            parity["frames"] += 1
            parity["equal"] += int(ok)
            if np.isfinite(diff):
                parity["max_abs_diff"] = max(parity["max_abs_diff"], diff)

    agree = {g: hd.Agreement(0.5) for g in (*GATES, "eating")}
    only = {g: {"legacy_only_images": 0, "reference_only_images": 0, "both_images": 0} for g in GATES}
    timing = {"legacy": [], "reference": []}
    counts = {"legacy": 0, "reference": 0}
    saved: list[dict] = []

    def account(dets, img_id):
        for g, (classes, conf) in GATES.items():
            a, b = gate(dets["legacy"], hd, classes, conf), gate(dets["reference"], hd, classes, conf)
            agree[g].add(a, b)
            if len(a) and len(b):
                only[g]["both_images"] += 1
            elif len(a):
                only[g]["legacy_only_images"] += 1
            elif len(b):
                only[g]["reference_only_images"] += 1
        agree["eating"].add(gate(dets["legacy"], hd, groups["eating"], RUN_CONF),
                            gate(dets["reference"], hd, groups["eating"], RUN_CONF))
        for k in counts:
            counts[k] += len(dets[k])

    def save(name, heads, geo, dets):
        np.savez_compressed(tensors_dir / f"{name}.npz", **{k.replace("/", "__"): v for k, v in heads.items()})
        saved.append({"name": name, "geometry": geo,
                      "legacy_dets": len(dets["legacy"]), "reference_dets": len(dets["reference"])})

    # Known-good frames (640x480, letterbox scale 1.0, top pad 80): always saved.
    kg = ROOT / "scripts" / "hw_validation" / "known_good"
    sources = json.loads((kg / "SOURCES.json").read_text())
    for row in sources["images"]:
        frame = cv2.imread(str(kg / row["file"]), cv2.IMREAD_COLOR)
        model.predict(frame, classes=run_classes, conf=RUN_CONF)
        heads = {k: np.asarray(v, dtype=np.float32) for k, v in captured["heads"].items()}
        h, w = frame.shape[:2]
        left, top = (640 - w) // 2, (640 - h) // 2
        dets, ms = both(heads, left, top, w, h, ("legacy", "reference"))
        save(Path(row["file"]).stem, heads, {"left": left, "top": top, "w": w, "h": h}, dets)

    manifest = mf.load(args.data / "manifest.json")
    items = sorted(manifest.split("val"), key=lambda i: i.id)[: args.limit]
    bad = mf.verify_files(manifest, args.data, items, decode=False)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the manifest: {bad[:3]}")
    t_start = time.perf_counter()
    for n, item in enumerate(items):
        if n and n % 200 == 0:
            print(f"  {n}/{len(items)} {time.perf_counter() - t_start:.0f}s", flush=True)
        with Image.open(args.data / item.path) as im:
            rgb = np.asarray(ipv.fit_640(im.convert("RGB")))
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        model.predict(bgr, classes=run_classes, conf=RUN_CONF)
        heads = captured["heads"]
        h, w = bgr.shape[:2]
        left, top = (640 - w) // 2, (640 - h) // 2
        order = ("legacy", "reference") if n % 2 == 0 else ("reference", "legacy")
        dets, ms = both(heads, left, top, w, h, order)
        for k in timing:
            timing[k].append(ms[k])
        off = offline(heads, {"left": left, "top": top, "w": w, "h": h})
        check_parity(dets, off)
        account(off, item.id)
        interesting = any(int(c) in (PERSON, PHONE) for d in dets.values() for c in d.classes)
        if interesting and len(saved) < 6 + args.save_n:
            save(f"val_{len(saved) - 5:02d}", {k: np.asarray(v, dtype=np.float32) for k, v in heads.items()},
                 {"left": left, "top": top, "w": w, "h": h}, dets)

    def stats(v):
        a = np.asarray(v)
        return {"n": int(a.size), "p50": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
                "p95": float(np.quantile(a, 0.95)), "mean": float(a.mean())}

    report = {
        "provenance": provenance, "manifest_id": manifest.id, "split": "val", "images": len(items),
        "wall_s": round(time.perf_counter() - t_start, 1),
        "detections_at_run_conf": counts,
        "product_vs_offline_parity": parity,
        "agreement": {g: a.summary() for g, a in agree.items()},
        "presence": only,
        "decode_ms_x86": {k: stats(v) for k, v in timing.items()},
        "tensors": {"format": "npz float32, keys = HEF output names with '/' -> '__'", "files": saved},
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("images", "detections_at_run_conf", "presence", "decode_ms_x86")},
                     indent=1))
    for g, a in report["agreement"].items():
        print(g, json.dumps({k: a[k] for k in ("n_a", "n_b", "matched", "match_rate_a", "match_rate_b")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
