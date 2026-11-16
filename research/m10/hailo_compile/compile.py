"""Reproducible Hailo-8L compile of a SensorDyme candidate detector. NOT EXECUTED: no Dataflow Compiler
exists on the fleet (M8.4 tooling probe, oryn job 20261113-090503-dc1833). Read README.md first.

Runs on an authorised x86_64 Linux machine with the Hailo Dataflow Compiler (hailo_sdk_client) installed
from the Hailo Developer Zone, at the versions pinned in README.md. Four subcommands:

    check      no SDK needed: verify the ONNX sha256 and its six head outputs, the calibration set
               (calib.py output), the model script and the output path; print the plan. Run it first.
    compile    parse (translate_onnx_model) -> optimise (model script + calibration) -> compile -> .hef,
               saving the .har after each stage and a manifest with every hash and tool version.
    parse-hef  check `hailortcli parse-hef <hef>` text against the device contract (one uint8
               640x640x3 input; six outputs 80/40/20 x 68/81; no on-chip NMS).
    emulate    L3 parity: the DFC's quantised emulator vs float ONNX on the fixture images, per head
               (cosine, max abs error) and after the PRODUCT decoder (core/accel.py, class map
               {80: hand}) as detection agreement.

It never writes over, and refuses any output path named like, the production .hef
(models/damoyolo_tinynasL20_T.hef, sha256 e69136d3...960e), and promotes nothing.

The DFC calls below follow Hailo's documented Python API (ClientRunner: translate_onnx_model,
load_model_script, optimize, compile, save_har, infer_context/InferenceContext). They were written
without an SDK to test against: check every call against the user guide of the installed version and
record any change in README.md before the first real run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))

PRODUCTION_HEF_SHA256 = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
PRODUCTION_HEF_NAME = "damoyolo_tinynasL20_T.hef"
#: The EXP-08 device-layout export (research/m84/device_export.py, oryn job 20261113-090349-6d22e9).
KNOWN_ONNX = {"a0327a25da034e942bb1f6529b7747cb5ddef8c821dc9545ac40ccf394ed7aa1": "exp08.heads81.onnx"}
HW_ARCH = "hailo8l"
INPUT_SHAPE = (640, 640, 3)
EXPECTED_HEADS = sorted([(80, 68), (80, 81), (40, 68), (40, 81), (20, 68), (20, 81)])
FORBIDDEN_SCRIPT = ("nms_postprocess",)  # the product decodes raw heads; an on-chip NMS build bakes in a threshold


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------- guards (no SDK)
def refuse_production_path(out: Path) -> None:
    """Fail before any work if the output could shadow or replace the production .hef."""
    if out.name == PRODUCTION_HEF_NAME:
        raise SystemExit(f"refusing to write {out}: that is the production .hef's name")
    if (ROOT / "models") in out.resolve().parents:
        raise SystemExit(f"refusing to write {out}: inside the product's models/ directory")
    if out.exists() and sha256_file(out) == PRODUCTION_HEF_SHA256:
        raise SystemExit(f"refusing to overwrite {out}: it is the production .hef")


def check_script(text: str) -> list[str]:
    """Problems in a DFC model script (.alls) for this product; empty = acceptable."""
    problems = []
    for word in FORBIDDEN_SCRIPT:
        if re.search(rf"^\s*[^#\n]*\b{word}\b", text, re.M):
            problems.append(f"model script uses {word}: the product decodes raw heads (core/accel.py); an "
                            "on-chip NMS build compiles its score threshold in")
    for m in re.finditer(r"normalization\w*\s*=\s*normalization\(\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]", text):
        mean = [float(x) for x in m.group(1).split(",") if x.strip()]
        std = [float(x) for x in m.group(2).split(",") if x.strip()]
        if mean != [0.0, 0.0, 0.0] or std != [1.0, 1.0, 1.0]:
            problems.append(f"model script normalisation mean {mean} std {std}: the device feeds raw 0..255 and "
                            "the network was trained on raw 0..255 (export_contract.py); a folded "
                            "normalisation would change what the network sees")
    return problems


def check_calib(calib_dir: Path, *, min_frames: int = 1024) -> dict[str, Any]:
    arr = np.load(calib_dir / "calib_uint8_nhwc.npy", mmap_mode="r")
    manifest = json.loads((calib_dir / "calib_manifest.json").read_text())
    problems = []
    if arr.dtype != np.uint8 or arr.ndim != 4 or tuple(arr.shape[1:]) != INPUT_SHAPE:
        problems.append(f"calibration array is {arr.dtype} {arr.shape}, expected uint8 (N, 640, 640, 3)")
    if arr.shape[0] < min_frames:
        problems.append(f"{arr.shape[0]} calibration frames < {min_frames}")
    if not manifest.get("release_grade"):
        problems.append("calibration manifest is not release grade (built with --allow-small)")
    if hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest() != manifest.get("array_sha256"):
        problems.append("calibration array does not match its manifest sha256")
    return {"frames": int(arr.shape[0]), "array_sha256": manifest.get("array_sha256"),
            "mean_luma": manifest.get("mean_luma"), "problems": problems}


def check_onnx(onnx_path: Path) -> dict[str, Any]:
    """sha256, and (if the onnx package is present) that the outputs are the six device heads."""
    digest = sha256_file(onnx_path)
    out: dict[str, Any] = {"path": str(onnx_path), "sha256": digest, "known_as": KNOWN_ONNX.get(digest), "problems": []}
    try:
        import onnx
    except ImportError:
        out["outputs"] = "not checked (onnx package missing)"
        return out
    m = onnx.load(str(onnx_path), load_external_data=False)
    outs = []
    for o in m.graph.output:
        d = [x.dim_value for x in o.type.tensor_type.shape.dim]
        outs.append({"name": o.name, "shape": d})
    heads = sorted((d["shape"][2], d["shape"][1]) for d in outs if len(d["shape"]) == 4)
    if heads != EXPECTED_HEADS:
        out["problems"].append(f"ONNX outputs {outs} are not the six device heads {EXPECTED_HEADS} (grid, channels)")
    out["outputs"] = outs
    out["input"] = [{"name": i.name, "shape": [x.dim_value for x in i.type.tensor_type.shape.dim]}
                    for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    return out


def parse_hef_text(text: str) -> dict[str, Any]:
    """Check `hailortcli parse-hef` output against the device contract. Heuristic: it looks for HxWxC
    triples on input/output lines, because the exact print format varies between HailoRT versions."""
    inputs, outputs = [], []
    for line in text.splitlines():
        low = line.lower()
        triples = [tuple(int(v) for v in t) for t in re.findall(r"(\d+)x(\d+)x(\d+)", line)]
        if not triples:
            continue
        if "input" in low:
            inputs.extend(triples)
        elif "output" in low:
            outputs.extend(triples)
    problems = []
    if inputs != [(640, 640, 3)]:
        problems.append(f"inputs {inputs}, expected one 640x640x3")
    if "uint8" not in text.lower():
        problems.append("no UINT8 format printed for the input")
    heads = sorted((h, c) for h, w, c in outputs)
    if heads != EXPECTED_HEADS:
        problems.append(f"outputs {outputs}, expected six heads {EXPECTED_HEADS} (grid, channels)")
    if re.search(r"\bnms\b", text, re.I):
        problems.append("parse-hef mentions NMS: an on-chip NMS build is not what the product decodes")
    return {"inputs": inputs, "outputs": outputs, "problems": problems, "ok": not problems}


# ------------------------------------------------------------------------------- the DFC (SDK needed)
def run_compile(a) -> int:
    from hailo_sdk_client import (  # pyright: ignore[reportMissingImports]  (x86_64 Linux SDK only)
        ClientRunner,
    )

    refuse_production_path(a.out_hef)
    onnx_info = check_onnx(a.onnx)
    calib = check_calib(a.calib)
    script = a.model_script.read_text() if a.model_script else ""
    problems = onnx_info["problems"] + calib["problems"] + check_script(script)
    if problems and not a.dry_ok:
        raise SystemExit("refusing to compile:\n  " + "\n  ".join(problems))
    work = a.out_hef.parent
    work.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"started": dt.datetime.now(dt.UTC).isoformat(), "hw_arch": HW_ARCH,
                                "onnx": onnx_info, "calibration": calib, "model_script": script,
                                "model_script_sha256": hashlib.sha256(script.encode()).hexdigest(),
                                "versions": tool_versions(), "stages": {}}
    runner = ClientRunner(hw_arch=HW_ARCH)
    end_nodes = [o["name"] for o in onnx_info["outputs"]] if isinstance(onnx_info.get("outputs"), list) else None
    runner.translate_onnx_model(str(a.onnx), a.net_name, end_node_names=end_nodes,
                                net_input_shapes={onnx_info["input"][0]["name"]: [1, 3, 640, 640]})
    runner.save_har(str(work / f"{a.net_name}.parsed.har"))
    manifest["stages"]["parse"] = {"har_sha256": sha256_file(work / f"{a.net_name}.parsed.har")}
    if script:
        runner.load_model_script(script)
    data = np.load(a.calib / "calib_uint8_nhwc.npy").astype(np.float32)  # raw 0..255, as the device feeds
    runner.optimize(data)
    runner.save_har(str(work / f"{a.net_name}.optimized.har"))
    manifest["stages"]["optimize"] = {"har_sha256": sha256_file(work / f"{a.net_name}.optimized.har"),
                                      "calib_frames": int(data.shape[0])}
    hef = runner.compile()
    a.out_hef.write_bytes(hef)
    digest = sha256_file(a.out_hef)
    if digest == PRODUCTION_HEF_SHA256:  # impossible in practice; checked because the cost of being wrong is high
        raise SystemExit("compiled .hef hashes to the production .hef; refusing")
    manifest["stages"]["compile"] = {"hef": str(a.out_hef), "hef_sha256": digest, "bytes": a.out_hef.stat().st_size}
    manifest["finished"] = dt.datetime.now(dt.UTC).isoformat()
    (work / f"{a.net_name}.compile_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["stages"], indent=1))
    return 0


def tool_versions() -> dict[str, Any]:
    import platform
    from importlib import metadata

    out: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine(),
                           "system": platform.platform()}
    for pkg in ("hailo-dataflow-compiler", "hailo_sdk_client", "hailo-model-zoo", "hailort", "onnx", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def run_emulate(a) -> int:
    """L3: DFC quantised emulator vs float ONNX on fixture images, per head and after the product decoder."""
    import cv2
    import hwv_common as hc
    import onnxruntime as ort
    from hailo_sdk_client import (  # pyright: ignore[reportMissingImports]
        ClientRunner,
        InferenceContext,
    )

    from core.accel import HailoYOLO

    runner = ClientRunner(har=str(a.har))
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    images = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    batch = np.stack([hc.letterbox_bgr_to_rgb(cv2.imread(str(p), cv2.IMREAD_COLOR), 640, 640)[0] for p in images])
    float_out = [sess.run(None, {sess.get_inputs()[0].name:
                                 np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)}) for x in batch]
    with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
        q_out = runner.infer(ctx, batch.astype(np.float32))
    report: dict[str, Any] = {"images": [p.name for p in images], "har_sha256": sha256_file(a.har),
                              "onnx_sha256": sha256_file(a.onnx), "per_image": []}
    decoder = HailoYOLO.__new__(HailoYOLO)
    decoder._set_extra_classes({80: "hand"}, None)
    decoder._nms_iou = 0.5
    for i, f_heads in enumerate(float_out):
        f_nhwc = [np.ascontiguousarray(t.transpose(0, 2, 3, 1)) for t in f_heads]
        q_list = q_out if isinstance(q_out, list) else [q_out]
        q_nhwc = [np.asarray(t[i:i + 1]) for t in q_list]
        pairs = []
        for f in f_nhwc:
            match = [q for q in q_nhwc if q.shape == f.shape]
            if len(match) != 1:
                raise SystemExit(f"cannot pair emulator output for float head {f.shape}: {[q.shape for q in q_nhwc]}")
            q = match[0]
            cos = float(np.dot(f.ravel(), q.ravel()) / (np.linalg.norm(f) * np.linalg.norm(q) + 1e-12))
            pairs.append({"shape": list(f.shape), "cosine": cos, "max_abs": float(np.abs(f - q).max())})
        dets = {}
        for kind, heads in (("float", f_nhwc), ("int8_emulated", [next(q for q in q_nhwc if q.shape == f.shape)
                                                                 for f in f_nhwc])):
            raw, scales = {}, []
            for j, t in enumerate(heads):
                raw[f"h{j}"] = t
            for g in (80, 40, 20):
                box = next(k for k, t in raw.items() if t.shape[1] == g and t.shape[3] == 68)
                cls = next(k for k, t in raw.items() if t.shape[1] == g and t.shape[3] != 68)
                scales.append({"box": box, "cls": cls, "grid": g, "stride": 640 // g, "num_classes": raw[cls].shape[3]})
            decoder._scales = scales
            for d in ("legacy", "reference"):
                decoder.decoder = d
                (res,) = decoder._decode_raw(raw, 1.0, 0, 0, 640, 640, None, a.conf)
                dets[f"{kind}_{d}"] = [[int(b.cls[0]), round(float(b.conf[0]), 4),
                                        *(round(float(v), 1) for v in b.xyxy[0])] for b in res.boxes]
        report["per_image"].append({"heads": pairs, "detections": dets})
    a.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.report}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--onnx", type=Path, required=True)
    c.add_argument("--calib", type=Path, required=True)
    c.add_argument("--model-script", type=Path)
    c.add_argument("--out-hef", type=Path, required=True)
    k = sub.add_parser("compile")
    k.add_argument("--onnx", type=Path, required=True)
    k.add_argument("--calib", type=Path, required=True)
    k.add_argument("--model-script", type=Path)
    k.add_argument("--out-hef", type=Path, required=True)
    k.add_argument("--net-name", default="sensordyme_exp08_heads81")
    k.add_argument("--dry-ok", action="store_true", help="compile despite check problems (never for a release)")
    h = sub.add_parser("parse-hef")
    h.add_argument("text", type=Path, help="saved output of `hailortcli parse-hef <hef>`")
    e = sub.add_parser("emulate")
    e.add_argument("--har", type=Path, required=True, help="the .optimized.har from `compile`")
    e.add_argument("--onnx", type=Path, required=True)
    e.add_argument("--images", type=Path, required=True)
    e.add_argument("--conf", type=float, default=0.05)
    e.add_argument("--report", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "check":
        refuse_production_path(a.out_hef)
        result = {"onnx": check_onnx(a.onnx), "calibration": check_calib(a.calib),
                  "model_script_problems": check_script(a.model_script.read_text()) if a.model_script else [],
                  "out_hef": str(a.out_hef)}
        print(json.dumps(result, indent=1, default=str))
        bad = result["onnx"]["problems"] + result["calibration"]["problems"] + result["model_script_problems"]
        return 1 if bad else 0
    if a.cmd == "compile":
        return run_compile(a)
    if a.cmd == "parse-hef":
        r = parse_hef_text(a.text.read_text())
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    return run_emulate(a)


if __name__ == "__main__":
    raise SystemExit(main())
