"""M8.4 B4: export a 3-class DAMO-YOLO-T checkpoint in the exact head layout the device decoder reads,
and make float parity fixtures for a future compiled .hef. Runs on oryn (damo-pilot venv), CPU only.

    python -u -m research.m84.device_export --damo <DAMO tree> --config configs/<exp>.py --ckpt best_ckpt.pth \
        --zoo <damoyolo_tinynasL20_T.onnx> --data <oi7-sd-v1> --work <dir on $SDJOB_DATA> --out <dir> [--limit N]

Levels (kept separate everywhere this is reported):
  1. float result                    <- THIS script (ONNX Runtime, x86 CPU, float32)
  2. host-simulated quantisation     <- research/m84/hostquant.py (ONNX Runtime QDQ; not Hailo)
  3. compiled-artifact validation    <- NOT DONE: needs the Hailo Dataflow Compiler (x86 Linux SDK,
                                        not on the fleet) to produce a .hef, then `hailortcli parse-hef`
  4. actual Hailo runtime            <- NOT DONE: needs the Pi 5 + Hailo-8L (scripts/hw_validation)

What it writes:
  * <name>.onnx            native export (RepConv fused, raw heads, zoo opset), as EXP-08 (export_compare)
  * <name>.heads3.onnx     cut at the six head end nodes (scale*gfl_reg "Mul", sigmoid(gfl_cls)), the
                           same cut the zoo .yaml compiles: box (1,68,G,G) + class (1,3,G,G), G = 80/40/20
  * <name>.heads81.onnx    the SAME graph with the three class-conv rows embedded in an 81-channel class
                           conv: person -> channel 0, phone -> 67 (COCO ids the product's class map
                           already uses), hand -> 80 (the slot every product decoder drops, so hands
                           stay invisible to today's device code instead of being mislabelled as a COCO
                           class). Every other row has weight 0 and bias -20 (sigmoid 2e-9). The output
                           layout then equals the shipping .hef's: six outputs, 68/81 channels at 80/40/20
                           (scripts/hw_validation/fake_hailo.py), and core/accel.py loads it unchanged.
  * parity checks: heads81 channels {0,67,80} == heads3 channels {0,1,2} bit for bit; PyTorch raw head
    tensors vs ONNX (max abs diff, TF32 off, CPU).
  * device-path accuracy on VAL: the product's HailoYOLO.predict (letterbox, legacy and reference
    decoders) on the float heads81 through fake_hailo, scored under the Open Images protocol.
    The model was trained with DAMO's stretch resize; the device letterboxes. This measures the
    difference instead of assuming it away.
  * fixtures: known_good (scripts/hw_validation/known_good, the same format as expected_float.json)
    and 48 OI val images with people/phones/hands (ids + sha256, head statistics, float detections)
    for the INT8-vs-float parity stage once a .hef exists.
Never touches a .hef, config.yaml, or a decoder default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

import numpy as np  # noqa: E402

ROWS = {0: 0, 1: 67, 2: 80}  # 3-class row -> 81-channel slot
FLOOR = 0.05


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_heads(onnx_path: Path) -> tuple[list[str], dict]:
    """The six head end nodes, by structure: a Sigmoid on a Conv output with 3 channels (class) and a
    Mul of a Conv output with 68 channels (scale * gfl_reg), at each of the three grids."""
    import onnx
    from onnx import shape_inference

    m = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    shapes = {}
    for vi in [*m.graph.value_info, *m.graph.output]:
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    producer = {o: n for n in m.graph.node for o in n.output}
    cls, box = {}, {}
    for n in m.graph.node:
        s = shapes.get(n.output[0])
        if not s or len(s) != 4:
            continue
        src = producer.get(n.input[0])
        if n.op_type == "Sigmoid" and s[1] == 3 and src is not None and src.op_type == "Conv":
            cls[s[2]] = (n.name, src.name)
        if n.op_type == "Mul" and s[1] == 68 and any(producer.get(i) is not None and producer[i].op_type == "Conv"
                                                    for i in n.input):
            box[s[2]] = n.name
    grids = sorted(set(cls) & set(box), reverse=True)
    if grids != [80, 40, 20]:
        raise SystemExit(f"head search found grids {grids} (cls {sorted(cls)}, box {sorted(box)}); expected 80/40/20")
    nodes = [x for g in grids for x in (box[g], cls[g][0])]
    return nodes, {"grids": grids, "cls_convs": {g: cls[g][1] for g in grids}, "nodes": nodes}


def embed81(heads3: Path, heads81: Path, cls_convs: dict) -> None:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(heads3))
    inits = {i.name: i for i in m.graph.initializer}
    by_name = {n.name: n for n in m.graph.node}
    for conv_name in cls_convs.values():
        node = by_name[conv_name]
        w_name, b_name = node.input[1], node.input[2]
        w, b = numpy_helper.to_array(inits[w_name]), numpy_helper.to_array(inits[b_name])
        assert w.shape[0] == 3 and b.shape == (3,), (conv_name, w.shape, b.shape)
        w81 = np.zeros((81, *w.shape[1:]), dtype=w.dtype)
        b81 = np.full((81,), -20.0, dtype=b.dtype)
        for r, c in ROWS.items():
            w81[c], b81[c] = w[r], b[r]
        inits[w_name].CopyFrom(numpy_helper.from_array(w81, w_name))
        inits[b_name].CopyFrom(numpy_helper.from_array(b81, b_name))
    del m.graph.value_info[:]  # stale 3-channel shapes downstream of the class convs
    for o in m.graph.output:
        o.type.tensor_type.ClearField("shape")
    onnx.checker.check_model(m)
    onnx.save(m, str(heads81))


def torch_heads(damo: Path, config: str, ckpt: Path, x: np.ndarray) -> list[np.ndarray]:
    """Raw per-level (box = scale*gfl_reg, cls = sigmoid(gfl_cls)) tensors from PyTorch, fused RepConv."""
    sys.path.insert(0, str(damo))
    cwd = os.getcwd()
    os.chdir(damo)
    import torch
    from damo.base_models.core.ops import RepConv
    from damo.config.base import parse_config
    from damo.detectors.detector import build_local_model

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = parse_config(config)
    model = build_local_model(cfg, torch.device("cpu"))
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    os.chdir(cwd)
    out = []
    with torch.no_grad():
        feats = model.neck(model.backbone(torch.from_numpy(x)))
        h = model.head
        for f, gcls, greg, scale in zip(feats, h.gfl_cls, h.gfl_reg, h.scales, strict=True):
            out += [scale(greg(f)).float().numpy(), gcls(f).sigmoid().numpy()]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--damo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--zoo", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="candidate")
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    work, out, data = Path(a.work).resolve(), Path(a.out).resolve(), Path(a.data).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    damo, ckpt, zoo = Path(a.damo).resolve(), Path(a.ckpt).resolve(), Path(a.zoo).resolve()

    import onnxruntime as ort

    from research.detector_bench import models as dm
    from research.pilot.export_compare import export, graph_summary, structural_diff

    rep: dict = {"levels": {"float": "this report", "host_simulated_quantisation": "research/m84/hostquant.py",
                            "compiled_artifact_validation": "NOT DONE (no Dataflow Compiler on the fleet)",
                            "hailo_runtime": "NOT DONE (needs the Pi 5 + Hailo-8L)"},
                 "checkpoint": {"path": str(ckpt), "sha256": _sha(ckpt)}, "zoo_sha256": _sha(zoo)}
    zoo_g = graph_summary(zoo)
    cwd = os.getcwd()
    native, extra = export(damo, a.config, ckpt, work, 640, zoo_g["opset"])
    os.chdir(cwd)
    final = work / f"{a.name}.onnx"
    native.replace(final)
    nodes, info = find_heads(final)
    heads3 = Path(dm._with_outputs(str(final), nodes))  # <name>.heads.onnx
    heads3 = heads3.replace(work / f"{a.name}.heads3.onnx")
    heads81 = work / f"{a.name}.heads81.onnx"
    embed81(heads3, heads81, info["cls_convs"])
    zoo_heads = Path(dm._with_outputs(str(zoo), list(dm.DamoYolo.HEAD_NODES)))
    rep["export"] = {"onnx_sha256": _sha(final), "heads3_sha256": _sha(heads3), "heads81_sha256": _sha(heads81),
                     "head_nodes": nodes, "grids": info["grids"], "model_info": extra.get("model_info"),
                     "opset": zoo_g["opset"],
                     "structure_vs_zoo_heads": structural_diff(graph_summary(zoo_heads), graph_summary(heads81))}

    def sess(path: Path):
        o = ort.SessionOptions()
        o.intra_op_num_threads = a.threads
        s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        assert s.get_providers()[0] == "CPUExecutionProvider"
        return s

    s3, s81 = sess(heads3), sess(heads81)
    rep["export"]["heads81_runtime_shapes"] = [list(t.shape) for t in s81.run(None, {s81.get_inputs()[0].name:
                                                                                    np.zeros((1, 3, 640, 640), np.float32)})]

    # ---- parity: heads81 vs heads3, and PyTorch vs ONNX, on the known-good frames (letterboxed by predict()).
    import cv2

    kg = ROOT / "scripts" / "hw_validation" / "known_good"
    sources = json.loads((kg / "SOURCES.json").read_text())
    from core.accel import letterbox_transform  # noqa: F401 - the product module must import here

    par = {"heads81_vs_heads3_max_abs": 0.0, "torch_vs_onnx_max_abs": {"box": 0.0, "cls": 0.0}}
    tensors = []
    for row in sources["images"]:
        frame = cv2.imread(str(kg / row["file"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas = np.full((640, 640, 3), 114, np.uint8)
        top, left = (640 - rgb.shape[0]) // 2, (640 - rgb.shape[1]) // 2
        canvas[top:top + rgb.shape[0], left:left + rgb.shape[1]] = rgb
        tensors.append(np.ascontiguousarray(canvas.transpose(2, 0, 1)[None]).astype(np.float32))
    for x in tensors:
        o3 = s3.run(None, {s3.get_inputs()[0].name: x})
        o81 = s81.run(None, {s81.get_inputs()[0].name: x})
        for t3, t81 in zip(o3, o81, strict=True):
            if t3.shape[1] == 3:
                d = max(float(np.abs(t81[:, ROWS[r]] - t3[:, r]).max()) for r in ROWS)
                par["heads81_vs_heads3_max_abs"] = max(par["heads81_vs_heads3_max_abs"], d)
                others = np.delete(t81, list(ROWS.values()), axis=1)
                par["heads81_other_channels_max"] = max(par.get("heads81_other_channels_max", 0.0), float(others.max()))
            else:
                par["heads81_vs_heads3_max_abs"] = max(par["heads81_vs_heads3_max_abs"], float(np.abs(t81 - t3).max()))
        th = torch_heads(damo, a.config, ckpt, x)
        for t_t, t_o in zip(th, o3, strict=True):
            role = "cls" if t_o.shape[1] == 3 else "box"
            par["torch_vs_onnx_max_abs"][role] = max(par["torch_vs_onnx_max_abs"][role], float(np.abs(t_t - t_o).max()))
    rep["parity"] = par
    print(json.dumps(rep["parity"]), flush=True)

    # ---- the product's device path on the float heads81 (fake_hailo), known-good fixtures and val accuracy.
    import fake_hailo
    from PIL import Image

    from core import accel
    from core.accel import HailoYOLO
    from evaluation import manifest as mf
    from research.detector_bench import bench
    from research.m83.inference_path_val import fit_640

    captured: dict = {}
    base = fake_hailo.onnx_backend(s81)

    def backend(batch):
        heads = base(batch)
        captured["batch"], captured["heads"] = batch, heads
        return heads

    fake_hailo.install(backend)
    accel.record_inference = lambda: None
    # one simulated chip per process (like the real one): one HailoYOLO, decoder switched per call
    m_leg = HailoYOLO(f"{a.name}.hef (simulated: float heads81 ONNX)", decoder="legacy")

    def fixture(frame: np.ndarray, model) -> dict:
        (res,) = model.predict(frame, conf=FLOOR)
        stats = {}
        for name, arr in captured["heads"].items():
            a64 = arr.astype(np.float64)
            e = {"shape": list(arr.shape), "min": float(a64.min()), "max": float(a64.max()),
                 "mean": float(a64.mean()), "std": float(a64.std())}
            if arr.shape[-1] == 81:
                flat = a64[0].reshape(-1, 81)
                e["max_by_slot"] = {"person(0)": round(float(flat[:, 0].max()), 5),
                                    "phone(67)": round(float(flat[:, 67].max()), 5),
                                    "hand(80)": round(float(flat[:, 80].max()), 5)}
            stats[name] = e
        dets = [{"cls": int(b.cls[0]), "name": model.names[int(b.cls[0])], "score": round(float(b.conf[0]), 5),
                 "xyxy": [round(float(v), 2) for v in b.xyxy[0]]} for b in res.boxes]
        return {"input_tensor_sha256": hashlib.sha256(captured["batch"].tobytes()).hexdigest(), "heads": stats,
                "detections": dets}

    kgf = {"format": "sensordyme-known-good-float/1", "model": a.name, "checkpoint_sha256": rep["checkpoint"]["sha256"],
           "heads81_onnx_sha256": rep["export"]["heads81_sha256"], "conf": FLOOR, "decoder": "legacy",
           "channel_map": {"person": 0, "phone": 67, "hand": 80}, "images": []}
    for row in sources["images"]:
        frame = cv2.imread(str(kg / row["file"]), cv2.IMREAD_COLOR)
        kgf["images"].append({"file": row["file"], "sha256": row["sha256"], **fixture(frame, m_leg)})
    (out / f"{a.name}_known_good_expected_float.json").write_text(json.dumps(kgf, indent=1) + "\n")

    manifest = mf.load(data / "manifest.json")
    items = sorted(manifest.split("val"), key=lambda i: i.id)[: a.limit]
    rows = {r["id"]: r for r in (json.loads(x) for x in (data / "boxes.jsonl").read_text().splitlines() if x)}
    oi_fix = {"format": "sensordyme-oi-val-float/1", "model": a.name, "conf": FLOOR, "decoder": "legacy",
              "note": "Open Images V7 val images (CC BY 2.0, per-file attribution in the manifest); fetch by id",
              "channel_map": {"person": 0, "phone": 67, "hand": 80}, "images": []}
    want_fix = 48
    files = {d: (work / f"{a.name}_device_{d}_val.jsonl").open("w") for d in ("legacy", "reference")}
    label_of = {0: "person", 67: "phone"}
    t0, ms = time.perf_counter(), []
    for n, item in enumerate(items):
        with Image.open(data / item.path) as im:
            image = im.convert("RGB")
        image = fit_640(image)  # the Mission 4/5 bench input, as M8.3 inference_path_val
        frame = np.ascontiguousarray(np.asarray(image)[..., ::-1])
        w, h = image.size
        for d in ("legacy", "reference"):
            m_leg.decoder = d
            ti = time.perf_counter()
            (res,) = m_leg.predict(frame, conf=FLOOR)
            ms.append((time.perf_counter() - ti) * 1e3)
            for b in res.boxes:
                c = int(b.cls[0])
                if c in label_of:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    files[d].write(json.dumps({"id": item.id, "label": label_of[c], "score": float(b.conf[0]),
                                               "box": [x1 / w, y1 / h, x2 / w, y2 / h]}) + "\n")
        r = rows.get(item.id, {})
        mids = {b["mid"] for b in r.get("boxes", [])}
        m_leg.decoder = "legacy"
        if len(oi_fix["images"]) < want_fix and ({"/m/050k8", "/m/0k65p"} & mids):
            oi_fix["images"].append({"id": item.id, "sha256": item.sha256, "fit_640": [w, h],
                                     **fixture(frame, m_leg)})
        if n % 400 == 0:
            print(f"  {n}/{len(items)} {time.perf_counter() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    (out / f"{a.name}_oi_val_expected_float.json").write_text(json.dumps(oi_fix, indent=1) + "\n")
    rep["device_path"] = {"images": len(items), "predict_ms_p50": float(np.median(ms)) if ms else None,
                          "note": "x86 float ONNX through core/accel.py HailoYOLO on fake_hailo; NOT a Hailo number",
                          "hand": "channel 80 is dropped by both product decoders: hand AP is not measurable here"}
    for d in ("legacy", "reference"):
        bench_args = ["score", "--data", str(data), "--models", str(zoo.parent), "--preds",
                      str(work / f"{a.name}_device_{d}_val.jsonl"), "--split", "val", "--name", f"{a.name}_device_{d}",
                      "--out", str(out)]
        try:
            bench.main(bench_args)
        except SystemExit as e:
            if e.code:
                print("bench exit", e.code)
    rep["fixtures"] = {"known_good": f"{a.name}_known_good_expected_float.json",
                       "oi_val": f"{a.name}_oi_val_expected_float.json", "oi_val_images": len(oi_fix["images"])}
    (out / f"{a.name}_device_export.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("export", "parity", "device_path")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
