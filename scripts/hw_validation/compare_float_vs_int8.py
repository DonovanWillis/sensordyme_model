#!/usr/bin/env python3
"""Compare a Pi capture bundle (Hailo-8L, int8 .hef) against the float ONNX of the
same model on the IDENTICAL saved input tensors, and compare the decoders.

RUNS ON A WORKER (or a Mac), never on the Pi. Needs `onnx` and `onnxruntime`;
both are imported lazily so the pure functions here are testable without them.

Model: Hailo Model Zoo damoyolo_tinynasL20_T (the 2022-12-19 pretrained zip,
which contains the .onnx). Head end nodes, from the Model Zoo yaml:
    Mul_224, Sigmoid_244   stride 8   (box DFL logits, class sigmoid)
    Mul_264, Sigmoid_284   stride 16
    Mul_304, Sigmoid_324   stride 32
The graph is cut at those nodes (they are added as graph outputs), so the float
tensors are the same intermediate tensors the .hef emits.

Reports (report.json, summary.md in --out):
  1. tensor mapping HEF name/shape <-> ONNX node/tensor/shape, paired by grid
     and role (68 channels = box head, otherwise class head);
  2. per-tensor numeric parity (mean/max abs error, RMSE, cosine, Pearson,
     worst-sample cosine), value ranges and a sigmoid-vs-logit check, and
     per-channel error for the class head (in quantisation steps when the
     bundle carries the HEF quant info), with the padding channels (>= 80);
  3. decoded-detection agreement, int8 vs float, for every decoder variant
     (legacy / legacy_offset0 / legacy_multilabel / reference), per
     preprocessing mode: match rates, matched-IoU distribution, score deltas;
  4. decoder vs decoder on the SAME device tensors (legacy vs reference etc.):
     the centre shift per stride isolates the anchor offset;
  5. production-pipeline (letterbox + legacy) vs reference-pipeline
     (stretch + reference) per image, in frame pixels;
  6. on-device production decoder output vs hwv_decode's legacy (a parity
     check that the offline legacy decoder is the one that ran).

    python compare_float_vs_int8.py --bundle capture_2026... \\
        --onnx damoyolo_tinynasL20_T.zip --out report_2026...
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402
import hwv_decode as hd  # noqa: E402

END_NODES = ("Mul_224", "Sigmoid_244", "Mul_264", "Sigmoid_284", "Mul_304", "Sigmoid_324")

#: Classes the product acts on, reported individually in the per-channel table.
PRODUCT_CLASSES = {0: "person", 39: "bottle", 41: "cup", 67: "cell phone"}

#: Advisory only. No threshold here has been validated on this model; they are
#: starting points for a human reading the report, not a gate.
ADVISORY = {"head_cosine_min": 0.99, "decoded_match_rate_min": 0.90, "matched_iou_p50_min": 0.90}


# ------------------------------------------------------------------ pure helpers
def to_nhwc(array: np.ndarray) -> np.ndarray:
    """An ONNX head output -> (1, G, G, C).

    Accepts NCHW (the ONNX convention), NHWC, or a flattened (1, C, G*G) /
    (1, G*G, C). A cube (C == G) is assumed NCHW.
    """
    a = np.asarray(array)
    if a.ndim == 3:
        a = a[None] if a.shape[0] != 1 else a
    if a.ndim == 4:
        _, d1, d2, d3 = a.shape
        if d2 == d3:
            return a.transpose(0, 2, 3, 1)
        if d1 == d2:
            return a
        raise ValueError(f"cannot interpret head shape {a.shape} as NCHW or NHWC")
    if a.ndim == 3:
        _, x, y = a.shape
        gy, gx = math.isqrt(y), math.isqrt(x)
        if gy * gy == y and x < y:
            return a.reshape(1, x, gy, gy).transpose(0, 2, 3, 1)
        if gx * gx == x and y < x:
            return a.reshape(1, gx, gx, y)
    raise ValueError(f"cannot interpret head shape {np.asarray(array).shape}")


def map_heads(hef_scales: list[hd.Scale], onnx_shapes: dict[str, tuple[int, ...]]) -> list[dict]:
    """Pair every HEF head with the ONNX end node of the same grid and role."""
    by_key: dict[tuple[int, str], str] = {}
    for node, shape in onnx_shapes.items():
        _, g, _, c = shape
        key = (g, "box" if c == hd.BOX_CHANNELS else "cls")
        if key in by_key:
            raise ValueError(f"two ONNX heads for grid {g} {key[1]}: {by_key[key]}, {node}")
        by_key[key] = node
    rows = []
    for s in hef_scales:
        for role, hef_name in (("box", s.box), ("cls", s.cls)):
            node = by_key.get((s.grid, role))
            if node is None:
                raise ValueError(f"no ONNX head for grid {s.grid} {role} (HEF {hef_name}); "
                                 f"ONNX heads: {onnx_shapes}")
            rows.append({"stride": s.stride, "grid": s.grid, "role": role, "hef": hef_name,
                         "onnx_node": node, "onnx_shape_nhwc": list(onnx_shapes[node])})
    return rows


def normalise_input(batch_rgb_uint8: np.ndarray, mean, std, swap_rb: bool = False) -> np.ndarray:
    """(1,H,W,3) uint8 RGB -> (1,3,H,W) float32 for the ONNX model."""
    x = batch_rgb_uint8.astype(np.float32)
    if swap_rb:
        x = x[..., ::-1]
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def compare_to_production(production: list[dict], dets: hd.Detections) -> dict:
    """On-device production boxes vs offline legacy (already unmapped), same order."""
    if len(production) != len(dets):
        return {"same_count": False, "device": len(production), "offline": len(dets)}
    if not production:
        return {"same_count": True, "max_coord_diff": 0.0, "max_score_diff": 0.0,
                "classes_equal": True}
    dev = sorted(production, key=lambda d: (-d["score"], d["xyxy"][0]))
    off = sorted(dets.to_list(), key=lambda d: (-d["score"], d["xyxy"][0]))
    return {
        "same_count": True,
        "max_coord_diff": max(abs(a - b) for d, o in zip(dev, off, strict=True)
                              for a, b in zip(d["xyxy"], o["xyxy"], strict=True)),
        "max_score_diff": max(abs(d["score"] - o["score"]) for d, o in zip(dev, off, strict=True)),
        "classes_equal": all(d["cls"] == o["cls"] for d, o in zip(dev, off, strict=True)),
    }


# ------------------------------------------------------------------ ONNX (lazy)
def load_onnx_bytes(path: Path, member: Optional[str]) -> tuple[bytes, dict]:
    info: dict = {"path": str(path), "file_sha256": hc.sha256_file(path)}
    if path.suffix.lower() != ".zip":
        data = path.read_bytes()
        info["onnx_sha256"] = hc.sha256_bytes(data)
        return data, info
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        info["zip_members"] = names
        onnx_members = [n for n in names if n.lower().endswith(".onnx")]
        if member is None:
            if len(onnx_members) != 1:
                raise SystemExit(f"{path}: {len(onnx_members)} .onnx members {onnx_members}; "
                                 "pass --onnx-member")
            member = onnx_members[0]
        data = zf.read(member)
        info["onnx_member"] = member
        info["onnx_sha256"] = hc.sha256_bytes(data)
        hints = []
        for n in names:
            if n.lower().endswith((".yaml", ".yml", ".alls", ".json", ".txt")):
                text = zf.read(n).decode(errors="replace")
                hints += [f"{n}: {ln.strip()}" for ln in text.splitlines()
                          if "normaliz" in ln.lower() or "mean" in ln.lower() or "std" in ln.lower()]
        info["normalisation_hints"] = hints[:40]
    return data, info


def build_session(onnx_bytes: bytes, end_nodes):
    import onnx
    import onnxruntime as ort

    model = onnx.load_model_from_string(onnx_bytes)
    nodes = {n.name: n for n in model.graph.node}
    missing = [e for e in end_nodes if e not in nodes]
    if missing:
        heads = sorted(n for n in nodes if n.startswith(("Mul", "Sigmoid")))[-24:]
        raise SystemExit(f"end node(s) {missing} not in the ONNX graph. Last Mul/Sigmoid nodes: "
                         f"{heads}. Pass --end-nodes.")
    existing = {o.name for o in model.graph.output}
    tensors = {}
    for e in end_nodes:
        t = nodes[e].output[0]
        tensors[e] = t
        if t not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    meta = {"input_name": inp.name, "input_shape": [str(d) for d in inp.shape],
            "input_type": inp.type, "end_node_tensors": tensors,
            "onnxruntime": ort.__version__, "onnx": onnx.__version__}
    return sess, tensors, meta


# ------------------------------------------------------------------ main
def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hailo int8 vs float ONNX parity (worker).")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True, help=".onnx or the Model Zoo .zip")
    parser.add_argument("--onnx-member", default=None)
    parser.add_argument("--end-nodes", default=",".join(END_NODES))
    parser.add_argument("--mean", default="0,0,0", help="ONNX input mean (RGB, 0-255 scale)")
    parser.add_argument("--std", default="1,1,1", help="ONNX input std")
    parser.add_argument("--swap-rb", action="store_true", help="feed BGR to the ONNX (diagnostic)")
    parser.add_argument("--conf", type=float, default=None, help="default: the bundle's")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--reference-nms-iou", type=float, default=None)
    parser.add_argument("--reference-max-det", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = hc.load_bundle(args.bundle)
    if manifest.get("status") != "complete":
        print(f"WARNING: bundle status is {manifest.get('status')!r}", file=sys.stderr)
    conf = args.conf if args.conf is not None else float(manifest["decode"]["conf"])
    mean = [float(v) for v in args.mean.split(",")]
    std = [float(v) for v in args.std.split(",")]
    variants = dict(hd.VARIANTS)
    variants["reference"] = hd.with_overrides(hd.REFERENCE, nms_iou=args.reference_nms_iou,
                                              max_det=args.reference_max_det)
    in_h = int(manifest["hef"]["input"]["shape"][0])
    scales = hd.group_heads({o["name"]: tuple(o["shape"]) for o in manifest["hef"]["outputs"]}, in_h)
    quant = {o["name"]: (o.get("quant_info") or {}) for o in manifest["hef"]["outputs"]}

    onnx_bytes, onnx_info = load_onnx_bytes(args.onnx, args.onnx_member)
    end_nodes = [e.strip() for e in args.end_nodes.split(",") if e.strip()]
    sess, tensors, sess_meta = build_session(onnx_bytes, end_nodes)
    fetch = [tensors[e] for e in end_nodes]

    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    mapping: Optional[list[dict]] = None
    parity: dict[str, hc.ParityAccumulator] = {}
    padding_max = 0.0
    int8_vs_float = {(v, m): hd.Agreement(args.match_iou) for v in variants for m in manifest["modes"]}
    cross_decoder = {(a, b, m): hd.Agreement(args.match_iou)
                     for a, b in (("legacy", "reference"), ("legacy", "legacy_offset0"),
                                  ("legacy", "legacy_multilabel"))
                     for m in manifest["modes"]}
    pipelines = {"device": hd.Agreement(args.match_iou), "float": hd.Agreement(args.match_iou)}
    by_image: dict[str, dict] = {}
    production_checks = []

    for k, sample in enumerate(samples):
        inp, hef_outs = hc.load_sample(args.bundle, sample)
        raw = sess.run(fetch, {sess_meta["input_name"]: normalise_input(inp, mean, std, args.swap_rb)})
        onnx_outs = {e: to_nhwc(r) for e, r in zip(end_nodes, raw, strict=True)}
        if mapping is None:
            mapping = map_heads(scales, {e: a.shape for e, a in onnx_outs.items()})
        float_outs = {}
        for row in mapping:
            dev, ref = hef_outs[row["hef"]], onnx_outs[row["onnx_node"]]
            parity.setdefault(row["hef"], hc.ParityAccumulator()).add(dev, ref, sample["id"])
            float_outs[row["hef"]] = ref
            if row["role"] == "cls" and dev.shape[-1] > hd.NUM_CLASSES:
                padding_max = max(padding_max, float(np.abs(dev[..., hd.NUM_CLASSES:]).max()))
        mode, geom = sample["mode"], sample["geometry"]
        decoded = {}
        for name, spec in variants.items():
            dev_d = hd.decode(hef_outs, scales, spec, conf)
            flt_d = hd.decode(float_outs, scales, spec, conf)
            int8_vs_float[(name, mode)].add(dev_d, flt_d)
            decoded[name] = (dev_d, flt_d)
        for (a, b, m), acc in cross_decoder.items():
            if m == mode:
                acc.add(decoded[a][0], decoded[b][0])
        if mode == "letterbox" and sample.get("production_detections") is not None:
            check = compare_to_production(sample["production_detections"],
                                          hd.unmap(decoded["legacy"][0], geom))
            production_checks.append({"sample": sample["id"], **check})
        img = by_image.setdefault(sample["image_id"], {})
        if mode == "letterbox":
            img["prod"] = (hd.unmap(decoded["legacy"][0], geom), hd.unmap(decoded["legacy"][1], geom))
        elif mode == "stretch":
            img["ref"] = (hd.unmap(decoded["reference"][0], geom),
                          hd.unmap(decoded["reference"][1], geom))
        if "prod" in img and "ref" in img:
            pipelines["device"].add(img["prod"][0], img["ref"][0])
            pipelines["float"].add(img["prod"][1], img["ref"][1])
            by_image.pop(sample["image_id"])
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(samples)} samples", flush=True)

    tensor_rows = []
    for row in mapping or []:
        acc = parity[row["hef"]]
        summary = acc.summary()
        q = quant.get(row["hef"]) or {}
        scale = q.get("qp_scale") if isinstance(q.get("qp_scale"), float) else None
        entry = {**row, "hef_quant_info": q, **summary,
                 "range_verdict_device": hc.range_verdict(
                     summary["range_device"]["min"], summary["range_device"]["max"], row["role"]),
                 "range_verdict_float": hc.range_verdict(
                     summary["range_float"]["min"], summary["range_float"]["max"], row["role"])}
        if row["role"] == "cls":
            chans = acc.per_channel(scale)
            entry["per_channel"] = chans
            entry["worst_channels"] = sorted(chans, key=lambda r: -r["mean_abs_err"])[:5]
            entry["product_channels"] = [dict(chans[c], name=n) for c, n in PRODUCT_CLASSES.items()
                                         if c < len(chans)]
        tensor_rows.append(entry)

    report = {
        "bundle": {"dir": str(args.bundle), "created_utc": manifest.get("created_utc"),
                   "model_sha256": manifest["model"]["sha256"],
                   "config_sha256": manifest["config"]["sha256"],
                   "code": manifest.get("code"), "hailo": manifest.get("hailo"),
                   "samples_used": len(samples)},
        "onnx": {**onnx_info, **sess_meta, "mean": mean, "std": std, "swap_rb": args.swap_rb},
        "decode": {"conf": conf, "match_iou": args.match_iou,
                   "variants": {k: dataclasses.asdict(v) for k, v in variants.items()}},
        "tensor_mapping": mapping,
        "tensors": tensor_rows,
        "hef_class_padding_channels_max_abs": padding_max,
        "int8_vs_float": {f"{v}/{m}": a.summary() for (v, m), a in int8_vs_float.items()},
        "decoder_vs_decoder_on_device": {f"{a}_vs_{b}/{m}": acc.summary()
                                         for (a, b, m), acc in cross_decoder.items()},
        "production_vs_reference_pipeline": {k: a.summary() for k, a in pipelines.items()},
        "production_decoder_parity": {
            "samples": len(production_checks),
            "all_match": all(c.get("same_count") and c.get("classes_equal", True)
                             and c.get("max_coord_diff", 0) < 0.05 for c in production_checks),
            "details": production_checks[:50],
        },
        "advisory_thresholds": ADVISORY,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    hc.write_json(args.out / "report.json", report)
    md = render_summary(report)
    (args.out / "summary.md").write_text(md)
    print(md)
    return 0


def render_summary(report: dict) -> str:
    lines = ["# Hailo int8 vs float ONNX: summary", "",
             (f"bundle {report['bundle']['dir']} ({report['bundle']['samples_used']} samples), "
              f"model {report['bundle']['model_sha256'][:12]}..., "
              f"onnx {report['onnx']['onnx_sha256'][:12]}..."),
             (f"conf {report['decode']['conf']}, match IoU {report['decode']['match_iou']}, "
              f"ONNX mean {report['onnx']['mean']} std {report['onnx']['std']}"), "",
             "## Tensor parity (device - float)", "",
             "| stride | role | HEF | ONNX | mean abs | max abs | cosine | worst cos | dev range | float range |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for t in report["tensors"]:
        lines.append(
            f"| {t['stride']} | {t['role']} | {t['hef']} | {t['onnx_node']} | {_fmt(t['mean_abs_err'])} "
            f"| {_fmt(t['max_abs_err'])} | {_fmt(t['cosine'], 5)} | {_fmt(t['worst_sample_cosine'], 5)} "
            f"| {_fmt(t['range_device']['min'], 2)}..{_fmt(t['range_device']['max'], 2)} "
            f"| {_fmt(t['range_float']['min'], 2)}..{_fmt(t['range_float']['max'], 2)} |")
    lines += ["", f"HEF class padding channels max |value|: {_fmt(report['hef_class_padding_channels_max_abs'])}",
              "", "## Decoded agreement, int8 (A) vs float (B), same decoder", "",
              "| decoder/mode | n int8 | n float | match int8 | match float | IoU p50 | IoU p5 | mean abs score delta |",
              "|---|---|---|---|---|---|---|---|"]
    for key, s in report["int8_vs_float"].items():
        iou = s["matched_iou"] or {}
        sd = s["score_delta_b_minus_a"] or {}
        lines.append(f"| {key} | {s['n_a']} | {s['n_b']} | {_fmt(s['match_rate_a'], 3)} "
                     f"| {_fmt(s['match_rate_b'], 3)} | {_fmt(iou.get('p50'), 3)} | {_fmt(iou.get('p5'), 3)} "
                     f"| {_fmt(sd.get('mean_abs'))} |")
    lines += ["", "## Decoder vs decoder on device tensors (centre shift A-B, input px)", ""]
    for key, s in report["decoder_vs_decoder_on_device"].items():
        shifts = ", ".join(f"s{k}: dx {v['dx_mean']:+.2f} dy {v['dy_mean']:+.2f} (n={v['n']})"
                           for k, v in s["centre_shift_a_minus_b_by_stride"].items())
        lines.append(f"- {key}: A {s['n_a']} / B {s['n_b']} boxes, matched {s['matched']}; {shifts or '-'}")
    lines += ["", "## Production pipeline (letterbox+legacy) vs reference (stretch+reference), frame px", ""]
    for key, s in report["production_vs_reference_pipeline"].items():
        iou = s["matched_iou"] or {}
        lines.append(f"- {key}: {s['images']} images, prod {s['n_a']} / ref {s['n_b']} boxes, "
                     f"matched {s['matched']}, IoU p50 {_fmt(iou.get('p50'), 3)}")
    p = report["production_decoder_parity"]
    lines += ["", f"Production decoder on device == offline legacy: {p['all_match']} ({p['samples']} samples)",
              "", "Thresholds in report.json `advisory_thresholds` are starting points, not validated gates."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
