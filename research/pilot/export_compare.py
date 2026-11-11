"""Export an EXP-08 checkpoint to ONNX and compare it with the shipping model's float source (Checkpoint F).

    python research/pilot/export_compare.py --damo <patched DAMO tree> --config configs/<exp>.py \
        --ckpt best_ckpt.pth --zoo <damoyolo_tinynasL20_T.onnx> --out <dir> [--providers cuda,cpu]

Run inside the damo-pilot venv on oryn (needs torch, onnx, onnxruntime). It:
  * exports the checkpoint the way the Hailo zoo ONNX was exported: RepConv fused, raw head
    outputs (no NMS / no post-processing), the zoo model's input size and opset;
  * compares the two graphs structurally: inputs, outputs, op histogram. The shipping .hef was
    compiled from the zoo ONNX, so an identical op set and output layout (only the class count
    differing) is the strongest compatibility evidence available without the Hailo Dataflow
    Compiler. It is NOT a compile: INT8 compilation and on-device accuracy remain unverified;
  * times both graphs with onnxruntime (same session options, fixed threads, warm-up) and
    records parameters and GFLOPs.
Writes export_compare.json. Never touches a .hef.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_summary(path: Path) -> dict:
    import onnx

    m = onnx.load(str(path))
    dims = lambda v: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]  # noqa: E731
    inits = {i.name for i in m.graph.initializer}
    return {
        "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
        "inputs": {i.name: dims(i) for i in m.graph.input if i.name not in inits},
        "outputs": {o.name: dims(o) for o in m.graph.output},
        "ops": dict(sorted(collections.Counter(n.op_type for n in m.graph.node).items())),
        "nodes": len(m.graph.node),
    }


def structural_diff(zoo: dict, new: dict) -> dict:
    ops = sorted(set(zoo["ops"]) | set(new["ops"]))
    return {
        "ops_only_in_zoo": [o for o in ops if o not in new["ops"]],
        "ops_only_in_new": [o for o in ops if o not in zoo["ops"]],
        "op_count_changes": {o: [zoo["ops"].get(o, 0), new["ops"].get(o, 0)] for o in ops
                             if zoo["ops"].get(o, 0) != new["ops"].get(o, 0)},
        "same_op_set": set(zoo["ops"]) == set(new["ops"]),
        "same_output_count": len(zoo["outputs"]) == len(new["outputs"]),
        "input_shapes": [list(zoo["inputs"].values()), list(new["inputs"].values())],
        "output_shapes": [list(zoo["outputs"].values()), list(new["outputs"].values())],
    }


def time_session(path: Path, provider: str, shape: list[int], runs: int, threads: int) -> dict:
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    want = {"cuda": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}[provider]
    sess = ort.InferenceSession(str(path), opts, providers=[want])
    if sess.get_providers()[0] != want:  # ORT falls back silently; refuse to time the wrong device
        raise SystemExit(f"{path.name}: asked for {want}, got {sess.get_providers()}")
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(max(10, runs // 10)):
        sess.run(None, {name: x})
    ms = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return {"provider": want, "runs": runs, "threads": threads, "p50_ms": statistics.median(ms),
            "p90_ms": ms[int(0.9 * (len(ms) - 1))], "mean_ms": statistics.fmean(ms)}


def export(damo: Path, config: str, ckpt: Path, out: Path, size: int, opset: int) -> tuple[Path, dict]:
    sys.path.insert(0, str(damo))
    os.chdir(damo)
    import torch
    from damo.base_models.core.ops import RepConv, SiLU
    from damo.config.base import parse_config
    from damo.detectors.detector import build_local_model
    from damo.utils.model_utils import get_model_info, replace_module
    from torch import nn

    cfg = parse_config(config)
    cfg.model.head.export_with_post = False  # raw head outputs, as the zoo graph the .hef came from
    model = build_local_model(cfg, torch.device("cpu"))
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()
    model.head.nms = False
    info = get_model_info(model, (size, size))
    path = out / (ckpt.stem + ".onnx")
    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(path), input_names=["images"], opset_version=opset, dynamo=False)
        except TypeError:  # an exporter without the dynamo switch
            torch.onnx.export(model, dummy, str(path), input_names=["images"], opset_version=opset)
    return path, {"model_info": str(info)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--damo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--zoo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--providers", default="cuda,cpu")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    if "cuda" in a.providers:
        import torch  # noqa: F401 - loads the CUDA libraries ORT's CUDA provider needs
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    zoo, ckpt = Path(a.zoo).resolve(), Path(a.ckpt).resolve()
    zoo_g = graph_summary(zoo)
    shape = next(iter(zoo_g["inputs"].values()))
    size = int(shape[-1])
    onnx_path, extra = export(Path(a.damo).resolve(), a.config, ckpt, out, size, zoo_g["opset"])
    new_g = graph_summary(onnx_path)
    report = {
        "checkpoint": {"path": str(ckpt), "sha256": _sha256(ckpt)},
        "export": {"path": str(onnx_path), "sha256": _sha256(onnx_path), **extra},
        "zoo": {"path": str(zoo), "sha256": _sha256(zoo)},
        "graphs": {"zoo": zoo_g, "new": new_g},
        "structure": structural_diff(zoo_g, new_g),
        "latency": {},
        "hailo": "structural comparison only; not compiled (no Hailo Dataflow Compiler); INT8 unverified",
    }
    shape = [1 if not isinstance(d, int) or d == 0 else d for d in shape]
    for prov in a.providers.split(","):
        runs = a.runs if prov == "cuda" else max(20, a.runs // 4)
        report["latency"][prov] = {"zoo": time_session(zoo, prov, shape, runs, a.threads),
                                   "new": time_session(onnx_path, prov, shape, runs, a.threads)}
    (out / "export_compare.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("structure", "latency")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
