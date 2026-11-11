"""Benchmark pretrained detectors on the Open Images subset (box-level; not behaviour).

    python -m research.detector_bench.bench fetch --models $SDJOB_DATA/models
    python -m research.detector_bench.bench run --data $SDJOB_DATA/openimages/oi7-sd-v1 \
        --models $SDJOB_DATA/models --split val --provider cuda --out $SDJOB_OUT

Same images, same class mapping, same score floor, same NMS IoU (0.50) and the
same Open Images scoring for every model. Operating points are the product's
shipped thresholds (phone 0.20, person 0.50) plus a sweep. ``--max-side``
re-runs at reduced source resolution: 640 is about what the device's 640x480
camera gives a detector; 320 approximates a far or small subject.

The held-out ``test`` split needs ``--frozen-test``: model and decode choices
are made on ``val``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])

from evaluation import manifest as mf
from research.detector_bench import models as dm
from research.detector_bench import oid_eval as oe

#: Every checkpoint used, with its source and licence (verified 2026-11-11).
CHECKPOINTS = {
    "damoyolo_tinynasL20_T": {
        "url": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/"
               "damoyolo_tinynasL20_T/pretrained/2022-12-19/damoyolo_tinynasL20_T.zip",
        "member": "damoyolo_tinynasL20_T.onnx",
        "licence": "Apache-2.0 (tinyvision/DAMO-YOLO; Hailo zoo cfg license_name). Trained on COCO train2017.",
        "note": "float source of the shipping .hef (zoo v2.15.0/v2.20.0 compile the same 2022-12-19 ONNX)",
    },
    "damoyolo_tinynasL25_S": {
        "url": "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ObjectDetection/Detection-COCO/yolo/"
               "damoyolo_tinynasL25_S/pretrained/2022-12-19/damoyolo_tinynasL25_S.zip",
        "member": "damoyolo_tinynasL25_S.onnx",
        "licence": "Apache-2.0 (tinyvision/DAMO-YOLO). COCO train2017.",
        "note": "in the Hailo-8L zoo (76.7 fps batch 1 published)",
    },
    "yolox_nano": {"url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx",
                   "licence": "Apache-2.0 (Megvii YOLOX). COCO train2017.", "size": 416,
                   "note": "not in the Hailo-8L zoo"},
    "yolox_tiny": {"url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
                   "licence": "Apache-2.0 (Megvii YOLOX). COCO train2017.", "size": 416,
                   "note": "in the Hailo-8L zoo (214 fps batch 1 published)"},
    "yolox_s": {"url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
                "licence": "Apache-2.0 (Megvii YOLOX). COCO train2017.", "size": 640,
                "note": "zoo ships a Hailo-retrained 'leaky' variant instead (weights' licence unclear; not used)"},
}

#: Runs: (name, checkpoint, constructor kwargs). The DAMO-T variants isolate each production decode choice.
RUNS = [
    ("damo_t.production", "damoyolo_tinynasL20_T", {"variant": "production"}),
    ("damo_t.offset0", "damoyolo_tinynasL20_T", {"variant": "offset0"}),
    ("damo_t.offset0_multilabel", "damoyolo_tinynasL20_T", {"variant": "offset0_multilabel"}),
    ("damo_s.offset0_multilabel", "damoyolo_tinynasL25_S", {"variant": "offset0_multilabel"}),
    ("yolox_nano", "yolox_nano", {}),
    ("yolox_tiny", "yolox_tiny", {}),
    ("yolox_s", "yolox_s", {}),
]
SHIPPED = {"phone": 0.20, "person": 0.50}
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
PHONE, PERSON = "/m/050k8", "/m/01g317"
PERSON_TREE = {PERSON, "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(models: Path) -> dict:
    models.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, spec in CHECKPOINTS.items():
        target = models / f"{name}.onnx"
        if not target.exists():
            with urllib.request.urlopen(spec["url"], timeout=600) as r:  # noqa: S310 - fixed https sources
                body = r.read()
            if spec["url"].endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    record[f"{name}.zip_members"] = z.namelist()
                    target.write_bytes(z.read(spec["member"]))
                (models / f"{name}.zip.sha256").write_text(hashlib.sha256(body).hexdigest())
            else:
                target.write_bytes(body)
        record[name] = {**spec, "sha256": _sha256(target), "bytes": target.stat().st_size}
        zip_hash = models / f"{name}.zip.sha256"
        if zip_hash.exists():
            record[name]["zip_sha256"] = zip_hash.read_text()
    (models / "checkpoints.json").write_text(json.dumps(record, indent=1))
    return record


class GpuMonitor(threading.Thread):
    """Samples nvidia-smi (when present) while a run executes."""

    SMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, float, float, float]] = []
        self._halt = threading.Event()
        self.available = os.path.exists(self.SMI)

    def run(self) -> None:
        while self.available and not self._halt.is_set():
            try:
                line = subprocess.run(  # noqa: S603 - fixed binary
                    [self.SMI, "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False,
                ).stdout.strip().splitlines()[0]
                vals = [float(v) if v.strip() not in ("[N/A]", "N/A") else float("nan") for v in line.split(",")]
                self.samples.append(tuple(vals))  # type: ignore[arg-type]
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._halt.wait(0.5)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=5)
        if not self.samples:
            return {"available": self.available}
        cols = list(zip(*self.samples, strict=True))

        def stats(xs):
            xs = [x for x in xs if x == x]
            return {"max": max(xs), "mean": round(sum(xs) / len(xs), 1)} if xs else None

        return {"available": True, "samples": len(self.samples), "memory_used_mb": stats(cols[0]),
                "utilization_pct": stats(cols[1]), "temperature_c": stats(cols[2]), "power_w": stats(cols[3])}


def _truths(rows: list[dict], groups: dict[str, str]) -> dict[str, list[oe.ImageTruth]]:
    out: dict[str, list[oe.ImageTruth]] = {"phone": [], "person": []}
    for row in rows:
        pos, neg = set(row["verified_positive"]), set(row["verified_negative"])
        for label, mids, key in (("phone", {PHONE}, PHONE), ("person", PERSON_TREE, PERSON)):
            gts = [oe.GroundTruth((b["xmin"], b["ymin"], b["xmax"], b["ymax"]), b["group_of"] == 1,
                                 b["occluded"] == 1, b["truncated"] == 1, b["depiction"] == 1)
                   for b in row["boxes"] if b["mid"] in mids]
            # Verified present but never boxed (an annotation gap): a detection there could be right,
            # so the image is not scored for this class rather than counting it as a false positive.
            verified = (key in pos and bool(gts)) or (key in neg and key not in pos)
            out[label].append(oe.ImageTruth(row["id"], groups[row["id"]], gts, verified, key in neg and not gts))
    return out


def _load_image(path: Path, max_side: Optional[int]):
    from PIL import Image

    with Image.open(path) as im:
        image = im.convert("RGB")
    if max_side and max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))),
                             Image.Resampling.BILINEAR)
    return image


def _build(run_kwargs: dict, ckpt: str, models: Path, provider: str):
    path = str(models / f"{ckpt}.onnx")
    if ckpt.startswith("damoyolo"):
        return dm.DamoYolo(path, provider, **run_kwargs)
    return dm.Yolox(path, provider, size=CHECKPOINTS[ckpt]["size"])


def run(data: Path, models: Path, split: str, provider: str, out: Path, *, max_sides: list[Optional[int]],
        runs: list[str], limit: Optional[int], frozen_test: bool) -> dict:
    if split == "test" and not frozen_test:
        raise SystemExit("the test split is held out: pass --frozen-test for the final measurement")
    manifest = mf.load(data / "manifest.json")
    problems = mf.validate(manifest)
    if problems:
        raise SystemExit(f"manifest problems: {problems[:5]}")
    items = sorted(manifest.split(split), key=lambda i: i.id)[:limit]
    wanted = {i.id for i in items}
    groups = {i.id: i.subject for i in items}
    provenance = {i.id: i.provenance for i in items}
    rows = [r for r in (json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x)
            if r["id"] in wanted]
    truths = _truths(rows, groups)
    checkpoints = json.loads((models / "checkpoints.json").read_text())
    # verify the image files we are about to score are the ones the manifest recorded
    bad = mf.verify_files(manifest, data, items, decode=False)
    if bad:
        raise SystemExit(f"{len(bad)} files differ from the manifest: {bad[:3]}")

    decode_ms = []
    report: dict[str, Any] = {
        "manifest_id": manifest.id, "split": split, "images": len(items), "provider": provider,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "checkpoints": {k: {f: v for f, v in c.items() if f in ("sha256", "licence", "url", "note")}
                        for k, c in checkpoints.items() if isinstance(c, dict)},
        "runs": {},
    }
    import onnxruntime as ort

    report["onnxruntime"] = ort.__version__
    for run_name, ckpt, kwargs in RUNS:
        if runs and run_name not in runs:
            continue
        for max_side in max_sides:
            key = f"{run_name}@{max_side or 'native'}"
            idle = GpuMonitor()  # what else is on the GPU right before this run (shared worker)
            idle.start()
            time.sleep(1.2)
            gpu_before = idle.stop()
            monitor = GpuMonitor()
            monitor.start()
            t_build = time.perf_counter()
            model = _build(kwargs, ckpt, models, provider)
            build_s = time.perf_counter() - t_build
            warm = _load_image(data / items[0].path, max_side)
            for _ in range(10):
                model(warm)
            model.timing = dm.Timing()
            preds: dict[str, dict[str, list]] = {"phone": {}, "person": {}}
            t0 = time.perf_counter()
            for item in items:
                td = time.perf_counter()
                image = _load_image(data / item.path, max_side)
                decode_ms.append((time.perf_counter() - td) * 1e3)
                for label, score, x1, y1, x2, y2 in model(image):
                    preds[label].setdefault(item.id, []).append((score, (x1, y1, x2, y2)))
            wall = time.perf_counter() - t0
            gpu = monitor.stop()
            entry: dict[str, Any] = {"checkpoint": ckpt, "max_side": max_side, "session_build_s": round(build_s, 2),
                                     "wall_s": round(wall, 1), "images_per_s": round(len(items) / wall, 1),
                                     "gpu": gpu, "gpu_before_run": gpu_before, "classes": {}}
            for name, attr in (("pre_ms", "pre_ms"), ("infer_ms", "infer_ms"), ("post_ms", "post_ms")):
                xs = getattr(model.timing, attr)
                entry[name] = {"p50": dm.percentile(xs, 0.5), "p90": dm.percentile(xs, 0.9)}
            for label in ("phone", "person"):
                result = oe.evaluate_class(preds[label], truths[label])
                ap50 = oe.average_precision(result.scored, result.npos)
                ap_range = [oe.average_precision(oe.evaluate_class(preds[label], truths[label], iou=t).scored,
                                                result.npos) for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                                                                       0.85, 0.9, 0.95)]
                c = {"npos": result.npos, "verified_images": result.images,
                     "negative_images": result.negative_images, "ap50": ap50,
                     "ap50_95": (sum(a for a in ap_range if a is not None) / len(ap_range)) if result.npos else None,
                     "shipped": oe.operating_point(result, SHIPPED[label]),
                     "sweep": [{k: v for k, v in oe.operating_point(result, t).items() if k != "recall_by_attribute"}
                               for t in SWEEP],
                     "calibration": oe.calibration(result),
                     "bootstrap": oe.group_bootstrap(result, SHIPPED[label]) if max_side in (None, 640) else None}
                if label == "phone" and max_side in (None, 640):
                    c["failures"] = _failures(result, SHIPPED[label], provenance)
                entry["classes"][label] = c
            report["runs"][key] = entry
            print(f"{key:40s} phone AP50 {_f(entry['classes']['phone']['ap50'])} "
                  f"R@.20 {_f(entry['classes']['phone']['shipped']['recall'])} "
                  f"P@.20 {_f(entry['classes']['phone']['shipped']['precision'])} | person AP50 "
                  f"{_f(entry['classes']['person']['ap50'])} | infer p50 {_f(entry['infer_ms']['p50'])} ms "
                  f"| gpu {gpu.get('memory_used_mb')}", flush=True)
            del model
    report["jpeg_decode_ms"] = {"p50": dm.percentile(decode_ms, 0.5), "p90": dm.percentile(decode_ms, 0.9)}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_{split}_{provider}.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _f(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def _failures(result: oe.ClassResult, threshold: float, provenance: dict) -> dict:
    """Worst false positives on verified phone-free images, and missed phones (ids + sources, no pixels)."""
    fps = sorted((s for s in result.scored if s.score >= threshold and s.image_id in result.negative_ids),
                 key=lambda s: -s.score)[:12]
    matched = {(s.image_id, s.matched) for s in result.scored if s.tp and s.score >= threshold}
    missed = []
    for image_id, gts in result.gt_index.items():
        for n, g in enumerate(x for x in gts if not x.group_of):
            if (image_id, n) not in matched:
                missed.append({"image": image_id, "size": g.size, "area": round(g.area, 5),
                               "occluded": g.occluded, "truncated": g.truncated, "depiction": g.depiction,
                               "source": provenance.get(image_id, {}).get("landing_url")})
    missed.sort(key=lambda m: -m["area"])
    return {"false_positives_on_phone_free_images": [
                {"image": s.image_id, "score": round(s.score, 3),
                 "source": provenance.get(s.image_id, {}).get("landing_url")} for s in fps],
            "missed_phones_largest_first": missed[:12], "missed_total": len(missed)}


def parity(data: Path, models: Path, provider: str, out: Path, n: int = 40) -> dict:
    """Raw-output agreement of ``provider`` against CPU on the same inputs (numerical parity)."""
    import numpy as np

    manifest = mf.load(data / "manifest.json")
    items = sorted(manifest.split("val"), key=lambda i: i.id)[:n]
    result = {}
    for ckpt in CHECKPOINTS:
        a = _build({"variant": "offset0"} if ckpt.startswith("damoyolo") else {}, ckpt, models, provider)
        b = _build({"variant": "offset0"} if ckpt.startswith("damoyolo") else {}, ckpt, models, "cpu")
        worst, agree, total = 0.0, 0, 0
        for item in items:
            image = _load_image(data / item.path, None)
            canvas, *_ = dm.letterbox(image, a.size, centered=a.family == "damo-yolo", pad=114)
            tensor = (canvas if a.family == "damo-yolo" else canvas[:, :, ::-1]).transpose(2, 0, 1)[None]
            tensor = tensor.astype(np.float32)
            names_a = [o.name for o in a.session.get_outputs()]
            ra = a.session.run(names_a, {a.input: tensor})
            rb = b.session.run(names_a, {b.input: tensor})
            for x, y in zip(ra, rb, strict=True):
                worst = max(worst, float(np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-9)))
            da = {(d[0], round(d[2], 2), round(d[3], 2)) for d in a(image, 0.2)}
            db = {(d[0], round(d[2], 2), round(d[3], 2)) for d in b(image, 0.2)}
            agree += len(da & db)
            total += max(len(da | db), 1)
        result[ckpt] = {"max_relative_output_diff": worst, "detection_agreement_at_0.2": agree / total,
                        "images": len(items)}
        print(ckpt, result[ckpt], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"parity_{provider}_vs_cpu.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Detector baselines on the Open Images subset")
    p.add_argument("cmd", choices=("fetch", "run", "parity"))
    p.add_argument("--data")
    p.add_argument("--models", required=True)
    p.add_argument("--split", default="val", choices=mf.SPLITS)
    p.add_argument("--frozen-test", action="store_true")
    p.add_argument("--provider", default="cuda", choices=("cuda", "dml", "cpu"))
    p.add_argument("--max-side", default="native,640,320", help="comma list; 'native' = no downscale")
    p.add_argument("--runs", default="", help="comma list of run names (default: all)")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", default=os.environ.get("SDJOB_OUT", "."))
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        print(json.dumps(fetch(Path(args.models)), indent=1))
        return 0
    if args.cmd == "parity":
        parity(Path(args.data), Path(args.models), args.provider, Path(args.out))
        return 0
    sides = [None if s == "native" else int(s) for s in args.max_side.split(",")]
    run(Path(args.data), Path(args.models), args.split, args.provider, Path(args.out), max_sides=sides,
        runs=[r for r in args.runs.split(",") if r], limit=args.limit, frozen_test=args.frozen_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
