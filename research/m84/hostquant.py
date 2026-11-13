"""M8.4 B2: HOST-SIMULATED INT8 quantisation sensitivity of the shipping DAMO-YOLO-T heads graph.

    python -u -m research.m84.hostquant --data <oi7-sd-v1> --models <dir> --work <dir> --out <dir> \
        [--calib 256] [--limit N] [--threads 8]

THIS IS NOT A HAILO RESULT. It quantises the float ONNX heads graph (the same six head outputs the
Hailo Model Zoo compiles, research/detector_bench/models.py HEAD_NODES) with ONNX Runtime's QDQ
static quantisation and runs it on the x86 CPU. The Hailo Dataflow Compiler uses its own
quantisation algorithms (and optional fine-tuning), its own calibration set and per-layer
precision choices; its INT8 accuracy can be better or worse than this. What this measures is how
sensitive these weights, heads and thresholds are to 8-bit rounding of weights, activations and
outputs, so the Pi-side INT8 parity check (docs/HAILO_HARDWARE_VALIDATION.md) knows where to look.

Variants (weights int8 symmetric, activations uint8 asymmetric, QDQ):
  float            the float heads graph (reference)
  float_o8         float graph, each OUTPUT tensor rounded to uint8 (per-tensor min/max from the
                   calibration set): isolates the effect of 8-bit outputs alone (the .hef returns
                   uint8 outputs that HailoRT dequantises)
  pc_minmax        per-channel weights, MinMax calibration on --calib train images
  pc_minmax_o8     the same plus uint8 outputs (the closest analogue of the .hef here)
  pt_minmax_o8     per-TENSOR weights, MinMax, uint8 outputs (a pessimistic bracket)
  pc_pct_o8        per-channel, Percentile 99.999 calibration on 32 train images, uint8 outputs
Each is decoded with the product's own decoders (core/accel.py legacy and reference) at the
0.05 scoring floor, NMS 0.5, and scored on VAL (never test). Calibration images come from TRAIN.
Head-level error vs float is also reported (class-score and DFL box-distance errors).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from evaluation import manifest as mf
from research.detector_bench import models as dm
from research.m8.coco80_preds import FLOOR, MAX_DETS, NMS_IOU, ensure_model, normalize

QUANT_VARIANTS = {  # name: (per_channel, method, calib images)
    "pc_minmax": (True, "minmax", None),
    "pt_minmax": (False, "minmax", None),
    "pc_pct": (True, "percentile", 32),
}
EVAL = ["float", "float_o8", "pc_minmax", "pc_minmax_o8", "pt_minmax_o8", "pc_pct_o8"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tensor(image) -> tuple[np.ndarray, float, int, int]:
    canvas, sx, _sy, left, top = dm.preprocess(image, 640, "letterbox114")
    return canvas.transpose(2, 0, 1)[None].astype(np.float32), sx, left, top


def _load(data: Path, item):
    from PIL import Image

    with Image.open(data / item.path) as im:
        return im.convert("RGB")


class Reader:
    """onnxruntime CalibrationDataReader over a list of train items."""

    def __init__(self, data: Path, items, input_name: str):
        self.data, self.items, self.input_name, self.i = data, list(items), input_name, 0

    def get_next(self):
        if self.i >= len(self.items):
            return None
        t, *_ = _tensor(_load(self.data, self.items[self.i]))
        self.i += 1
        return {self.input_name: t}

    def rewind(self):
        self.i = 0


def quantise(heads: Path, work: Path, data: Path, calib_items, input_name: str) -> dict[str, Path]:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process

        pre = work / "heads.pre.onnx"
        if not pre.exists():
            quant_pre_process(str(heads), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception as e:  # noqa: BLE001 - pre-processing is an optimisation, not required
        print("quant_pre_process skipped:", repr(e), flush=True)
        src = heads
    # The zoo graph is opset 11; per-channel DequantizeLinear needs the `axis` attribute (opset >= 13).
    import onnx
    from onnx import version_converter

    m = onnx.load(str(src))
    opset = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    if opset < 13:
        src13 = work / "heads.op13.onnx"
        if not src13.exists():
            onnx.save(version_converter.convert_version(m, 13), str(src13))
        print(f"converted opset {opset} -> 13 for quantisation", flush=True)
        src = src13
    out = {}
    for name, (per_channel, method, n) in QUANT_VARIANTS.items():
        target = work / f"heads.{name}.op13.c{len(calib_items[: n or len(calib_items)])}.onnx"
        out[name] = target
        if target.exists():
            continue
        t0 = time.perf_counter()
        items = calib_items[: n or len(calib_items)]
        # ORT's MinMax calibrator flushes every k batches and raises "No data is collected" when the
        # image count is an exact multiple of k (nothing left at the end), so pick k that leaves a rest.
        k = next(k for k in (8, 7, 9, 6, 10, 5) if len(items) % k)
        extra = {"CalibMaxIntermediateOutputs": k} if method == "minmax" else {"CalibPercentile": 99.999}
        quantize_static(str(src), str(target), Reader(data, items, input_name), quant_format=QuantFormat.QDQ,
                        per_channel=per_channel, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax if method == "minmax" else CalibrationMethod.Percentile,
                        extra_options=extra)
        print(f"quantised {name} on {len(items)} train images in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def _session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    assert s.get_providers()[0] == "CPUExecutionProvider"
    return s


def _o8(outputs: list[np.ndarray], ranges: list[tuple[float, float]]) -> list[np.ndarray]:
    res = []
    for x, (lo0, hi0) in zip(outputs, ranges, strict=True):
        lo, hi = min(lo0, 0.0), max(hi0, 0.0)  # zero must be representable (uint8 zero point)
        scale = (hi - lo) / 255.0 or 1.0
        zp = round(-lo / scale)
        q = np.clip(np.round(x / scale) + zp, 0, 255)
        res.append(((q - zp) * scale).astype(np.float32))
    return res


def _dfl_dist(box: np.ndarray) -> np.ndarray:
    d = box.reshape(-1, 4, 17)
    d = np.exp(d - d.max(axis=2, keepdims=True))
    d /= d.sum(axis=2, keepdims=True)
    return (d * np.arange(17, dtype=np.float32)).sum(axis=2)


def head_errors(ref: list, other: list) -> dict:
    """Class-score abs error at anchors the float model scores >= 0.05, and DFL distance error (in
    stride units x stride = input px) at anchors the float model scores >= 0.20, pooled over grids."""
    cls_err, box_err, flips = [], [], {0.2: 0, 0.4: 0}
    anchors = {0.2: 0, 0.4: 0}
    for (rb, rc, stride), (ob, oc, _) in zip(ref, other, strict=True):
        rc2, oc2 = rc.reshape(-1, rc.shape[-1])[:, :80], oc.reshape(-1, oc.shape[-1])[:, :80]
        sel = rc2 >= 0.05
        cls_err.append(np.abs(rc2[sel] - oc2[sel]))
        for t in flips:
            r, o = rc2 >= t, oc2 >= t
            flips[t] += int((r != o).sum())
            anchors[t] += int(r.sum())
        keep = rc2.max(axis=1) >= 0.2
        if keep.any():
            box_err.append(np.abs(_dfl_dist(rb.reshape(-1, 68)[keep]) - _dfl_dist(ob.reshape(-1, 68)[keep])).ravel() * stride)
    ce = np.concatenate(cls_err) if cls_err else np.zeros(0)
    be = np.concatenate(box_err) if box_err else np.zeros(0)
    return {"cls_abs": ce, "box_px": be, "flips": flips, "anchors": anchors}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--calib", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    data, work, out = Path(a.data), Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    model = ensure_model(Path(a.models))
    heads = Path(dm._with_outputs(str(model), list(dm.DamoYolo.HEAD_NODES)))
    manifest = mf.load(data / "manifest.json")
    train = sorted(manifest.split("train"), key=lambda i: i.id)
    random.Random(84).shuffle(train)  # noqa: S311 - a reproducible sample, not security
    calib = train[: a.calib]
    (out / "calibration_ids.txt").write_text("\n".join(i.id for i in calib) + "\n")
    val = sorted(manifest.split("val"), key=lambda i: i.id)[: a.limit]
    fsess = _session(heads, a.threads)
    input_name = fsess.get_inputs()[0].name
    names = [o.name for o in fsess.get_outputs()]
    # per-output float ranges on the calibration images, for the uint8-output simulation
    lo = [np.inf] * len(names)
    hi = [-np.inf] * len(names)
    for item in calib:
        for k, x in enumerate(fsess.run(names, {input_name: _tensor(_load(data, item))[0]})):
            lo[k], hi[k] = min(lo[k], float(x.min())), max(hi[k], float(x.max()))
    ranges = list(zip(lo, hi, strict=True))
    qpaths = quantise(heads, work, data, calib, input_name)
    sessions = {"float": fsess, **{n: _session(pth, a.threads) for n, pth in qpaths.items()}}
    for n, s in sessions.items():  # outputs are positional below: every graph must keep the float order
        assert [o.name for o in s.get_outputs()] == names, (n, [o.name for o in s.get_outputs()])
    report: dict = {"note": "HOST-SIMULATED quantisation (ONNX Runtime QDQ on x86 CPU). NOT a Hailo/DFC result.",
                    "model_sha256": _sha(model), "heads_sha256": _sha(heads),
                    "quant_sha256": {n: _sha(pth) for n, pth in qpaths.items()},
                    "calibration": {"split": "train", "images": len(calib), "seed": 84},
                    "output_ranges": dict(zip(names, ranges, strict=True)), "manifest_id": manifest.id,
                    "val_images": len(val), "decode_floor": FLOOR, "nms_iou": NMS_IOU}
    files = {(v, d): (work / f"preds_{v}_{d}_val.jsonl").open("w") for v in EVAL for d in ("legacy", "reference")}
    err: dict[str, dict] = {v: {"cls": [], "box": [], "flips": {0.2: 0, 0.4: 0}, "anchors": {0.2: 0, 0.4: 0}}
                            for v in EVAL if v != "float"}
    ms: dict[str, list[float]] = {n: [] for n in sessions}
    t0 = time.perf_counter()
    for n, item in enumerate(val):
        image = _load(data, item)
        tensor, sx, left, top = _tensor(image)
        raw = {}
        for name, s in sessions.items():
            ti = time.perf_counter()
            raw[name] = s.run(None, {input_name: tensor})
            ms[name].append((time.perf_counter() - ti) * 1e3)
        variants = {"float": raw["float"], "float_o8": _o8(raw["float"], ranges),
                    "pc_minmax": raw["pc_minmax"], "pc_minmax_o8": _o8(raw["pc_minmax"], ranges),
                    "pt_minmax_o8": _o8(raw["pt_minmax"], ranges), "pc_pct_o8": _o8(raw["pc_pct"], ranges)}
        heads_by = {v: dm.split_heads(o, 640) for v, o in variants.items()}
        for v, h in heads_by.items():
            for d in ("legacy", "reference"):
                boxes, scores, classes = dm.device_decode(h, d, FLOOR, NMS_IOU, 640)
                for det in normalize(boxes[:MAX_DETS], scores[:MAX_DETS], classes[:MAX_DETS], sx, left, top, *image.size):
                    files[(v, d)].write(json.dumps({"id": item.id, **det}) + "\n")
            if v != "float":
                e = head_errors(heads_by["float"], h)
                err[v]["cls"].append(e["cls_abs"])
                err[v]["box"].append(e["box_px"])
                for t in e["flips"]:
                    err[v]["flips"][t] += e["flips"][t]
                    err[v]["anchors"][t] += e["anchors"][t]
        if n % 200 == 0:
            print(f"  {n}/{len(val)} {time.perf_counter() - t0:.0f}s", flush=True)
            (out / "progress.json").write_text(json.dumps({"done": n, "of": len(val)}))
    for f in files.values():
        f.close()
    summ = {}
    for v, e in err.items():
        ce = np.concatenate(e["cls"]) if e["cls"] else np.zeros(1)
        be = np.concatenate(e["box"]) if e["box"] else np.zeros(1)
        summ[v] = {"cls_abs_mean": float(ce.mean()), "cls_abs_p99": float(np.percentile(ce, 99)),
                   "cls_abs_max": float(ce.max()),
                   "box_px_mean": float(be.mean()), "box_px_p99": float(np.percentile(be, 99)),
                   "threshold_flips": {str(t): {"flips": e["flips"][t], "float_anchors_above": e["anchors"][t]}
                                       for t in e["flips"]}}
    report["head_error_vs_float"] = summ
    report["infer_ms_p50"] = {k: float(np.median(v)) for k, v in ms.items() if v}
    report["preds"] = {f"{v}_{d}": str(work / f"preds_{v}_{d}_val.jsonl") for v, d in files}
    report["preds_sha16"] = {f"{v}_{d}": _sha(work / f"preds_{v}_{d}_val.jsonl")[:16] for v, d in files}
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / "hostquant_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("head_error_vs_float", "infer_ms_p50", "wall_s")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
