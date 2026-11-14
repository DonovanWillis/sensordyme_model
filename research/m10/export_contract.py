"""M10 Task 2: the export / deployment contract, checked and recorded, for the shipping zoo model and for
the EXP-08 device-layout export, against what core/accel.py and core/camera.py actually do.

    # static part only (any machine, no model, no onnx package):
    python -m research.m10.export_contract --out DIR
    # full (oryn, damo-pilot venv: onnx + onnxruntime + cv2 + PIL):
    python -m research.m10.export_contract --out DIR \
        --onnx zoo=$D/models/damoyolo_tinynasL20_T.onnx --onnx zoo_heads=$D/m84/export_smoke/zoo.heads.extracted.onnx \
        --onnx exp08_heads81=$D/m84/export_smoke/exp08.heads81.onnx --onnx exp08=$D/m84/export_smoke/exp08.onnx \
        --damo $D/src/exp08_full60-DAMO-YOLO-<commit> --damo-config configs/exp08_full60.py \
        --data $D/openimages/oi7-sd-v1 --colour-images 1585

What is checked (each item records the product's value, the model side's value, the evidence and a
verdict; a mismatch is a FINDING, never silently "fixed"):

  input normalisation   is there mean/std arithmetic between the graph input and the first Conv (i.e.
                        does the network expect raw 0..255)? the DAMO config's image_mean / image_std
  colour order          core/camera.py delivers BGR (picamera2 "RGB888" is B,G,R in memory);
                        core/accel.py converts BGR->RGB before the chip; what the network was trained on
                        (DAMO dataset code) and an empirical swap-R/B sensitivity on val images
  dimensions/letterbox  640x480 camera -> 640x640 (scale, padding, pad value, interpolation) vs the
                        DAMO test transform (stretch or keep-ratio)
  tensor layout         ONNX NCHW float32 vs the .hef's NHWC uint8 input and float32 NHWC outputs
  outputs               names, order, shapes, roles (68-ch DFL box / C-ch class), end-node op types
                        (Sigmoid = probabilities, Mul = scale * gfl_reg logits), strides
  class mapping         81 class channels; which are live (EXP-08: rows 0/67/80) and what channel 80 is
                        on the zoo model (its own background slot)
  DFL                   reg_max = box channels / 4 - 1 vs core/accel.py _DFL_BINS - 1
  NMS / thresholds      core/accel.py IoU and max-detections vs the DAMO config; product confidence
                        thresholds (config.yaml) vs each model's precision-0.80 thresholds (M8.4 §5.4)
  calibration spec      what the DFC needs to reproduce the device input
  artifact hashes       every file named above

Evidence level: graph/config facts are exact; runtime facts are L1 (float ONNX on x86). Nothing here is a
compiled .hef (L3) or a Pi (L4). Val split only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

PRODUCTION_HEF_SHA256 = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
NORMALISATION_OPS = {"Sub", "Div", "Mul", "Add", "BatchNormalization"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== static (product side)
def product_contract(root: Path = ROOT) -> dict[str, Any]:
    """What the device does, read from the product source and config (no model, no onnx package)."""
    import yaml

    from core import accel

    camera_src = (root / "core" / "camera.py").read_text()
    accel_src = (root / "core" / "accel.py").read_text()
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cam = cfg.get("camera", {})
    width, height = int(cam.get("width", 640)), int(cam.get("height", 480))
    scale, left, top = accel.letterbox_transform(width, height, 640, 640)
    new_w, new_h = round(width * scale), round(height * scale)
    pi_read = camera_src[camera_src.index("if self.is_pi:"):camera_src.index("ok, frame = self._cap.read()")]
    predict = accel_src[accel_src.index("    def predict("):accel_src.index("    # -------------------------------------------------------------- raw decode")]
    da = cfg.get("modules", {}).get("driver_attention", {})
    return {
        "camera": {
            "size": [width, height],
            "picamera2_format": "RGB888" if '"format": "RGB888"' in camera_src else None,
            "pi_read_converts_colour": "cvtColor" in pi_read,
            "delivers": "BGR uint8 HxWx3 (picamera2 RGB888 is B,G,R in memory; core/camera.py _read)",
        },
        "preprocess": {
            "letterbox": {"scale": scale, "resized": [new_w, new_h], "left": left, "top": top,
                          "right": 640 - new_w - left, "bottom": 640 - new_h - top},
            "interpolation": "cv2.INTER_LINEAR" if "cv2.INTER_LINEAR" in predict else None,
            "pad_value": [114, 114, 114] if "value=(114, 114, 114)" in predict else None,
            "colour_conversion": "BGR->RGB" if "cv2.COLOR_BGR2RGB" in predict else None,
            "input_dtype": "uint8" if "FormatType.UINT8" in accel_src else None,
            "input_layout": "NHWC (1, 640, 640, 3)",
            "normalisation_on_host": "none (raw 0..255 uint8 handed to the chip)",
        },
        "decode": {
            "output_format": "FLOAT32 NHWC (1, G, G, C)" if "FormatType.FLOAT32" in accel_src else None,
            "dfl_bins": accel.HailoYOLO._DFL_BINS, "reg_max": accel.HailoYOLO._DFL_BINS - 1,
            "head_pairing": "by grid size and channel count (68 = box, else class), never by name",
            "nms_iou": accel.HailoYOLO._DEFAULT_NMS_IOU, "nms": "class-wise greedy",
            "reference_max_detections": accel.HailoYOLO._REFERENCE_MAX_DETECTIONS,
            "anchor_offset": {"legacy": 0.5, "reference": 0.0},
            "class_selection": {"legacy": "argmax per anchor", "reference": "every (anchor, class) >= conf"},
            "class_channels_read": "0..79 (COCO); 80+ dropped unless accelerator.yolo_extra_classes maps them",
            "decoder_default": cfg.get("accelerator", {}).get("yolo_decoder", "legacy"),
            "extra_classes_configured": cfg.get("accelerator", {}).get("yolo_extra_classes"),
        },
        "thresholds": {
            "driver_attention.yolo_confidence_threshold": da.get("yolo_confidence_threshold"),
            "driver_attention.person_confidence_threshold": da.get("person_confidence_threshold"),
            "driver_attention.yolo_every_n_frames": da.get("yolo_every_n_frames"),
        },
        "hef": {"path": cfg.get("accelerator", {}).get("yolo_hef"),
                "sha256_pinned": cfg.get("accelerator", {}).get("yolo_hef_sha256")},
    }


# ===================================================================================== graph facts (needs onnx)
def graph_facts(path: Path) -> dict[str, Any]:
    """Plain-dict facts of an ONNX graph: input, the ops between input and first Conv, outputs, live class rows."""
    import onnx
    from onnx import numpy_helper, shape_inference

    m = shape_inference.infer_shapes(onnx.load(str(path)))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    inputs = [i for i in g.input if i.name not in inits]

    def dims(vi):
        return [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]

    shapes = {vi.name: dims(vi) for vi in [*g.value_info, *g.output, *g.input]}
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    # walk from the input to the first Conv(s), recording every op on the way
    pre_conv: list[str] = []
    frontier = [inputs[0].name] if inputs else []
    seen: set[str] = set()
    while frontier:
        t = frontier.pop()
        for n in consumers.get(t, []):
            if n.name in seen:
                continue
            seen.add(n.name)
            if n.op_type == "Conv":
                continue
            pre_conv.append(n.op_type)
            frontier.extend(n.output)
    outs = []
    for o in g.output:
        node = producer.get(o.name)
        outs.append({"name": o.name, "shape": shapes.get(o.name) or dims(o),
                     "producer": node.name if node is not None else None,
                     "op": node.op_type if node is not None else None})
    # live class rows: for each Sigmoid output whose producer is a Conv, rows with any non-zero weight
    live: dict[str, Any] = {}
    for n in g.node:
        if n.op_type != "Sigmoid":
            continue
        src = producer.get(n.input[0])
        if src is None or src.op_type != "Conv" or len(src.input) < 2 or src.input[1] not in inits:
            continue
        w = numpy_helper.to_array(inits[src.input[1]])
        b = numpy_helper.to_array(inits[src.input[2]]) if len(src.input) > 2 and src.input[2] in inits else None
        rows = [int(r) for r in range(w.shape[0]) if abs(w[r]).max() > 0]
        live[n.output[0]] = {"channels": int(w.shape[0]), "live_rows": rows if len(rows) < 20 else f"{len(rows)} rows",
                             "dead_row_bias": (sorted({round(float(b[r]), 3) for r in range(w.shape[0])
                                                       if r not in rows}) if b is not None else None)}
    opset = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=None)
    return {"path": str(path), "sha256": sha256_file(path), "opset": opset,
            "input": {"name": inputs[0].name if inputs else None, "shape": dims(inputs[0]) if inputs else None,
                      "elem_type": onnx.TensorProto.DataType.Name(inputs[0].type.tensor_type.elem_type)
                      if inputs else None},
            "ops_before_first_conv": pre_conv, "outputs": outs, "class_convs": live,
            "op_histogram": _hist(n.op_type for n in g.node)}


def _hist(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def head_roles(outputs: list[dict]) -> list[dict]:
    """Classify head outputs (NCHW (1, C, G, G)) into box / class roles with stride, from shapes only."""
    rows = []
    for o in outputs:
        s = o.get("shape") or []
        if len(s) != 4 or not all(isinstance(x, int) and x > 0 for x in s):
            rows.append({**o, "role": "unknown"})
            continue
        c, gh, gw = s[1], s[2], s[3]
        role = "box" if c == 68 else "class"
        rows.append({"name": o["name"], "op": o.get("op"), "role": role, "channels": c, "grid": gh,
                     "stride": 640 // gh if gh and 640 % gh == 0 else None, "square": gh == gw})
    return rows


# ===================================================================================== DAMO config (needs the tree)
def damo_facts(damo: Path, config: str) -> dict[str, Any]:
    """Training/test preprocessing and postprocess facts from the DAMO-YOLO tree and experiment config.

    Text-based on purpose (no torch import): the values are read where they are defined and quoted.
    """
    out: dict[str, Any] = {"tree": str(damo), "config": config}
    cfg_text = (damo / config).read_text() if (damo / config).exists() else ""
    base = damo / "damo" / "config" / "base.py"
    base_text = base.read_text() if base.exists() else ""
    keys = ("image_mean", "image_std", "image_max_range", "keep_ratio", "nms_conf_thre", "nms_iou_thre",
            "num_classes", "reg_max", "input_size", "size_divisibility", "flip_prob", "autoaug_dict")
    for k in keys:
        hits = []
        for where, text in (("config", cfg_text), ("base.py", base_text)):
            for m in re.finditer(rf"^\s*[\w.\[\]'\"]*{k}\w*\s*[=:]\s*(.+)$", text, re.M):
                hits.append(f"{where}: {m.group(0).strip()[:160]}")
        if hits:
            out[k] = hits[:4]
    # how images are decoded for training/eval (PIL RGB vs cv2 BGR)
    dataset_dir = damo / "damo" / "dataset"
    colour = []
    if dataset_dir.exists():
        for py in sorted(dataset_dir.rglob("*.py")):
            text = py.read_text(errors="replace")
            for pat in (r"convert\(['\"]RGB['\"]\)", r"cv2\.imread", r"COLOR_BGR2RGB", r"COLOR_RGB2BGR",
                        r"\[\s*:\s*,\s*:\s*,\s*::-1\s*\]", r"to_bgr255", r"Image\.open"):
                for m in re.finditer(pat, text):
                    line = text[:m.start()].count("\n") + 1
                    colour.append(f"{py.relative_to(damo)}:{line}: {text.splitlines()[line - 1].strip()[:140]}")
    out["image_decode_sites"] = colour[:30]
    return out


# ===================================================================================== the comparison
def compare(product: dict, graphs: dict[str, dict], damo: Optional[dict], runtime: Optional[dict]) -> list[dict]:
    """Findings: product vs model side. Pure dict logic (unit-tested without models)."""
    f: list[dict] = []

    def add(item, status, product_side, model_side, evidence):
        f.append({"item": item, "status": status, "product": product_side, "model": model_side, "evidence": evidence})

    pre = product["preprocess"]
    # colour
    cam_bgr = product["camera"]["picamera2_format"] == "RGB888" and not product["camera"]["pi_read_converts_colour"]
    add("colour: camera -> host", "match" if cam_bgr and pre["colour_conversion"] == "BGR->RGB" else "MISMATCH",
        f"camera delivers BGR; predict() converts {pre['colour_conversion']}", "network input RGB (zoo convention)",
        "core/camera.py _read (picamera2 RGB888 = B,G,R bytes, device-verified corr 0.998); core/accel.py predict")
    if runtime and "colour" in runtime:
        for name, r in runtime["colour"].items():
            ok = r.get("rgb_better")
            add(f"colour: {name} trained order", "match" if ok else "MISMATCH" if ok is False else "unverified",
                "RGB fed to the network", r, "swap-R/B sensitivity on val images (L1 float, device path)")
    # normalisation
    for name, g in graphs.items():
        norm = [op for op in g.get("ops_before_first_conv", []) if op in NORMALISATION_OPS]
        add(f"normalisation: {name}", "match" if not norm else "MISMATCH",
            pre["normalisation_on_host"], {"ops_before_first_conv": g.get("ops_before_first_conv")},
            "ONNX graph walk from the input to the first Conv")
    if damo:
        add("normalisation: DAMO training config", "recorded", pre["normalisation_on_host"],
            {k: damo.get(k) for k in ("image_mean", "image_std")}, "DAMO tree, quoted lines")
    # dims / layout
    for name, g in graphs.items():
        shp = (g.get("input") or {}).get("shape")
        ok = shp == [1, 3, 640, 640]
        add(f"input tensor: {name}", "match" if ok else "MISMATCH",
            f"{pre['input_layout']} {pre['input_dtype']} (the .hef's input vstream)",
            {"onnx_input": g.get("input")}, "ONNX is NCHW float32 by construction; the DFC converts the layout and "
            "quantises the input; uint8 0..255 == float 0..255 only if no normalisation is folded (row above)")
    if damo:
        add("resize: letterbox vs training", "MISMATCH (known, measured)", pre["letterbox"],
            {k: damo.get(k) for k in ("keep_ratio", "image_max_range", "input_size")},
            "DAMO test transform vs core/accel.py letterbox; effect measured in M5 (zoo) and M10 (EXP-08 stretch "
            "diagnostic)")
    # outputs / DFL / classes
    for name, g in graphs.items():
        heads = head_roles(g.get("outputs", []))
        if not heads or any(h["role"] == "unknown" for h in heads):
            add(f"outputs: {name}", "not a heads graph", "six heads", g.get("outputs"), "ONNX outputs")
            continue
        boxes = [h for h in heads if h["role"] == "box"]
        classes = [h for h in heads if h["role"] == "class"]
        strides = sorted({h["stride"] for h in heads})
        ok = len(boxes) == 3 and len(classes) == 3 and strides == [8, 16, 32] and all(h["square"] for h in heads)
        add(f"outputs: {name}", "match" if ok else "MISMATCH", "3 x (68-ch box + C-ch class) at strides 8/16/32",
            heads, "ONNX output shapes (NCHW); the .hef emits NHWC")
        reg_max = {h["channels"] // 4 - 1 for h in boxes}
        add(f"DFL reg_max: {name}", "match" if reg_max == {product["decode"]["reg_max"]} else "MISMATCH",
            product["decode"]["reg_max"], sorted(reg_max), "box channels / 4 - 1")
        ends = {(h["role"], h["op"]) for h in heads}
        add(f"head end ops: {name}", "match" if ends == {("box", "Mul"), ("class", "Sigmoid")} else "MISMATCH",
            "box = raw DFL logits (softmax on host), class = probabilities", sorted(ends),
            "ONNX producer op of each output")
        c = {h["channels"] for h in classes}
        live = list(g.get("class_convs", {}).values())
        extra = product["decode"]["extra_classes_configured"]
        if c == {81}:
            rows = live[0]["live_rows"] if live else None
            if isinstance(rows, list) and rows == [0, 67, 80]:
                add(f"class map: {name}", "MISMATCH unless opt-in map" if not extra else "match",
                    product["decode"]["class_channels_read"], {"live_rows": rows, "hand_channel": 80},
                    "class-conv weights; channel 80 (hand) is dropped by the default decoders; "
                    "accelerator.yolo_extra_classes {80: hand} reads it (M10)")
            else:
                add(f"class map: {name}", "match", product["decode"]["class_channels_read"],
                    {"live_rows": rows, "channel_80": "the network's own background slot (not a class)"},
                    "class-conv weights; fleet job 20261113-043857-f182d1 channel-80 max 0.0079")
    # NMS / thresholds
    if damo:
        add("NMS IoU", "deliberate difference", product["decode"]["nms_iou"], damo.get("nms_iou_thre"),
            "core/accel.py _DEFAULT_NMS_IOU (0.50 measured better for phone, docs/DECODER_CANDIDATE.md)")
    add("confidence thresholds", "MISMATCH for any non-shipping model",
        product["thresholds"], {"precision_0.80_thresholds_M84": {
            "zoo_legacy": {"person": 0.379, "phone": 0.609}, "zoo_reference": {"person": 0.332, "phone": 0.609},
            "exp08": {"person": 0.300, "phone": 0.409}}},
        "raw-score thresholds are model- and decoder-specific (M8.4 §5.4, float); INT8 shifts scores (§5.5)")
    return f


# ===================================================================================== runtime (needs ORT + data)
def colour_sensitivity(heads_onnx: Path, extra, data: Path, limit: int, threads: int) -> dict:
    """Person/phone/hand AP50 on val through the device path, RGB (as shipped) vs R/B swapped at the chip."""
    import fake_hailo
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from core import accel
    from core.accel import HailoYOLO
    from evaluation import manifest as mf
    from research.m8.analysis import load_split
    from research.m83.inference_path_val import fit_640

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(heads_onnx), o, providers=["CPUExecutionProvider"])
    manifest = mf.load(data / "manifest.json")
    items = sorted(manifest.split("val"), key=lambda i: i.id)[:limit]
    ids = {i.id for i in items}
    _mid, _rows, truths, _g, _h = load_split(data, "val")
    from research.detector_bench import oid_eval as oe

    labels = {0: "person", 67: "phone", 80: "hand"}
    result: dict[str, Any] = {"images": len(items)}
    for swap in (False, True):
        fake_hailo.install(fake_hailo.onnx_backend(sess, swap_rb=swap))
        accel.record_inference = lambda: None
        model = HailoYOLO("contract.hef (simulated)", decoder="reference", extra_classes=extra)
        preds: dict[str, dict[str, list]] = {v: {} for v in labels.values()}
        try:
            for item in items:
                with Image.open(data / item.path) as im:
                    fitted = fit_640(im.convert("RGB"))
                bgr = np.ascontiguousarray(np.asarray(fitted)[..., ::-1])
                h, w = bgr.shape[:2]
                (res,) = model.predict(bgr, conf=0.05)
                for b in res.boxes:
                    lab = labels.get(int(b.cls[0]))
                    if lab:
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                        preds[lab].setdefault(item.id, []).append((float(b.conf[0]), (x1 / w, y1 / h, x2 / w, y2 / h)))
        finally:
            model.close()
            fake_hailo.uninstall()
        key = "bgr_swapped" if swap else "rgb_as_shipped"
        result[key] = {}
        for lab in labels.values():
            t = [x for x in truths[lab] if x.image_id in ids]
            r = oe.evaluate_class(preds[lab], t)
            if r.npos:
                result[key][lab] = round(oe.average_precision(r.scored, r.npos) or 0.0, 4)
    a, b = result["rgb_as_shipped"], result["bgr_swapped"]
    result["rgb_better"] = all(a[k] >= b[k] for k in a if k in b) if a else None
    return result


def head_value_ranges(heads_onnx: Path, threads: int) -> dict:
    """Value ranges on the known-good frames (letterboxed as the device does): probabilities vs logits."""
    import cv2
    import fake_hailo
    import numpy as np
    import onnxruntime as ort

    from core.accel import letterbox_transform

    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(heads_onnx), o, providers=["CPUExecutionProvider"])
    kg = ROOT / "scripts" / "hw_validation" / "known_good"
    sources = json.loads((kg / "SOURCES.json").read_text())
    rng: dict[str, list[float]] = {"box_min": [], "box_max": [], "cls_min": [], "cls_max": [], "ch80_max": []}
    for row in sources["images"]:
        frame = cv2.imread(str(kg / row["file"]), cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        scale, left, top = letterbox_transform(w, h, 640, 640)
        nw, nh = round(w * scale), round(h * scale)
        r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lb = cv2.copyMakeBorder(r, top, 640 - nh - top, left, 640 - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]).astype(np.float32)
        for t in fake_hailo.nchw_heads_to_hef(sess.run(None, {sess.get_inputs()[0].name: x})).values():
            if t.shape[-1] == 68:
                rng["box_min"].append(float(t.min()))
                rng["box_max"].append(float(t.max()))
            else:
                rng["cls_min"].append(float(t[..., :80].min()))
                rng["cls_max"].append(float(t[..., :80].max()))
                rng["ch80_max"].append(float(t[..., 80].max()))
    return {"frames": len(sources["images"]), "box_logits": [min(rng["box_min"]), max(rng["box_max"])],
            "class_probabilities_ch0_79": [min(rng["cls_min"]), max(rng["cls_max"])],
            "channel_80_max": max(rng["ch80_max"])}


# ===================================================================================== main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--onnx", action="append", default=[], help="name=path (zoo, zoo_heads, exp08, exp08_heads81)")
    p.add_argument("--damo", type=Path)
    p.add_argument("--damo-config")
    p.add_argument("--data", type=Path)
    p.add_argument("--colour-images", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    product = product_contract()
    graphs = {}
    for spec in a.onnx:
        name, _, path = spec.partition("=")
        graphs[name] = graph_facts(Path(path))
        print(name, json.dumps({k: graphs[name][k] for k in ("sha256", "input", "ops_before_first_conv")}), flush=True)
    damo = damo_facts(a.damo, a.damo_config) if a.damo and a.damo_config else None
    runtime: dict[str, Any] = {}
    heads = {name: Path(path) for name, _, path in (s.partition("=") for s in a.onnx)}
    for name in ("zoo_heads", "exp08_heads81"):
        if name in heads:
            runtime.setdefault("value_ranges", {})[name] = head_value_ranges(heads[name], a.threads)
    if a.data and a.colour_images and "exp08_heads81" in heads:
        runtime["colour"] = {"exp08_heads81": colour_sensitivity(heads["exp08_heads81"], {80: "hand"}, a.data,
                                                                 a.colour_images, a.threads)}
        print("colour", json.dumps(runtime["colour"]), flush=True)
    findings = compare(product, graphs, damo, runtime or None)
    report = {"evidence_level": "graph/config facts exact; runtime L1 float ONNX on x86; no .hef (L3), no Pi (L4)",
              "production_hef_sha256_pinned": product["hef"]["sha256_pinned"],
              "production_hef_sha256_expected": PRODUCTION_HEF_SHA256,
              "product": product, "graphs": graphs, "damo": damo, "runtime": runtime, "findings": findings}
    (a.out / "export_contract.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    for x in findings:
        print(f"[{x['status']}] {x['item']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
