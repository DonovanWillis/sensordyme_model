"""M10 Task 3(c): DESKTOP latency of each candidate network's head graph on oryn. NOT a Pi, NOT Hailo.

    python -u -m research.m10.latency --onnx zoo=PATH --onnx exp08=PATH --out $SDJOB_OUT \
        [--providers cpu1,cpu4,cuda] [--runs 200]

Batch 1, 640x640, float32 NCHW, the same head graphs the device path runs (research/m10/candidates_val.py).
Per network and provider: warm-up 20 runs, then --runs timed runs; p50/p90/mean in ms. "pair" times the
two networks back to back on one input (Arch B's two-network frame cost on THIS machine). The CUDA
provider runs with ONNX Runtime defaults. `nvidia-smi` utilisation is sampled before and after so
contention by another GPU user is visible.

These are x86 / RTX 4060 numbers. A Hailo-8L runs a compiled INT8 .hef on a different architecture; the
only Hailo-8L number for this network family in the repo is the zoo's published 124 fps (x86 host) and the
Pi measurement of 22.07 ms mean for the shipping .hef (docs/PHASE5_DEMO_READINESS.md section 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def smi() -> str | None:
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not os.path.exists(exe):
        return None
    try:
        return subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,temperature.gpu",  # noqa: S603
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p90": round(float(np.percentile(a, 90)), 3),
            "mean": round(float(a.mean()), 3), "n": int(a.size)}


def make_session(path: Path, provider: str):
    import onnxruntime as ort

    o = ort.SessionOptions()
    if provider.startswith("cpu"):
        o.intra_op_num_threads = int(provider[3:] or 1)
        o.inter_op_num_threads = 1
        s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        want = "CPUExecutionProvider"
    else:
        s = ort.InferenceSession(str(path), o, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        want = "CUDAExecutionProvider"
    if s.get_providers()[0] != want:
        raise SystemExit(f"{path} on {provider}: got providers {s.get_providers()} (silent fallback refused)")
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", action="append", required=True, help="name=path")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--providers", default="cpu1,cpu4,cuda")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    providers = [x.strip() for x in a.providers.split(",") if x.strip()]
    if "cuda" in providers:
        import torch  # noqa: F401 - loads cuDNN so ORT's CUDA EP can start (fleet quirk)
    import onnxruntime as ort

    nets = {}
    for spec in a.onnx:
        name, _, path = spec.partition("=")
        nets[name] = Path(path)
    rng = np.random.default_rng(10)
    x = rng.integers(0, 256, (1, 3, 640, 640)).astype(np.float32)
    report: dict = {"note": "oryn x86 CPU / RTX 4060 desktop latency; NOT a Raspberry Pi, NOT Hailo-8L",
                    "job": os.environ.get("SDJOB_ID"), "onnxruntime": ort.__version__,
                    "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
                    "networks": {n: {"path": str(pth), "sha256": sha256_file(pth)} for n, pth in nets.items()},
                    "nvidia_smi_before": smi(), "results": {}}
    for prov in providers:
        sessions = {n: make_session(pth, prov) for n, pth in nets.items()}
        res: dict = {}
        for n, s in sessions.items():
            feed = {s.get_inputs()[0].name: x}
            for _ in range(a.warmup):
                s.run(None, feed)
            ms = []
            for _ in range(a.runs):
                t0 = time.perf_counter()
                s.run(None, feed)
                ms.append((time.perf_counter() - t0) * 1e3)
            res[n] = stats(ms)
        if len(sessions) == 2:
            (n1, s1), (n2, s2) = sessions.items()
            f1, f2 = {s1.get_inputs()[0].name: x}, {s2.get_inputs()[0].name: x}
            ms = []
            for _ in range(a.runs):
                t0 = time.perf_counter()
                s1.run(None, f1)
                s2.run(None, f2)
                ms.append((time.perf_counter() - t0) * 1e3)
            res[f"pair({n1}+{n2})"] = stats(ms)
        report["results"][prov] = res
        print(prov, json.dumps(res), flush=True)
    report["nvidia_smi_after"] = smi()
    (a.out / "m10_latency.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
