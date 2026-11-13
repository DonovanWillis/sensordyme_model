"""M8.3: per-stage HOST CPU time of the driver module on recorded frames (not a Pi number).

    # Mac (known-good images, detector on the fake chip with planted objects)
    python -m research.m83.host_baseline --images scripts/hw_validation/known_good --frames 300 --out DIR
    # fleet x86 (Open Images val person photos at 640x480; detector heads from the float ONNX)
    python -m research.m83.host_baseline --data $OI --heads-onnx $M/damoyolo_tinynasL20_T.heads.onnx \
        --frames 400 --out $SDJOB_OUT

Builds the REAL DriverAttentionModule from config.yaml and gives it the REAL
core.accel.HailoYOLO (via core.detectors, with the integrity check) on a stand-in
chip (scripts/hw_validation/fake_hailo.py). Per frame it records, with
stage_timing's instrumentation: facemesh, hands, detector_total (= preprocess +
hailo_infer + decode), module_logic (process() minus the timed sub-stages) and
process_total.

What the numbers are, and are not:
  * facemesh, hands, preprocess, decode, module_logic: this host's CPU running
    the code the Pi runs. A Pi 5 Cortex-A76 is slower; the SHARES and the
    per-stage ordering are what transfer, not the milliseconds.
  * hailo_infer: the stand-in's time (planted: ~0; ONNX: x86 float inference).
    It is NEVER a Hailo latency. Measure that on the Pi (hwv_stage.py latency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "hw_validation"))


def frames_from_images(directory: Path, n: int):
    import cv2

    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    frames = [cv2.resize(cv2.imread(str(p)), (640, 480), interpolation=cv2.INTER_AREA) for p in paths]
    return [frames[i % len(frames)] for i in range(n)], [p.name for p in paths]


def frames_from_oi(data: Path, n: int):
    import cv2

    from evaluation import manifest as mf

    manifest = mf.load(data / "manifest.json")
    items = [i for i in sorted(manifest.split("val"), key=lambda i: i.id)
             if "person" in str(i.conditions.get("pools", ""))][:n]
    frames = []
    for item in items:
        img = cv2.imread(str(data / item.path))
        if img is not None:
            frames.append(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    return frames, f"oi7 val person pool, manifest {manifest.id}, {len(frames)} images"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path)
    src.add_argument("--data", type=Path)
    p.add_argument("--heads-onnx", type=Path, default=None)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    for name in ("images", "data", "heads_onnx", "out"):  # before the chdir below
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=True)

    import fake_hailo
    import numpy as np
    import stage_timing as st
    import yaml

    from core import accel, detectors
    from modules import get_module_class

    if args.heads_onnx:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = args.threads, 1
        backend = fake_hailo.onnx_backend(ort.InferenceSession(str(args.heads_onnx), opts,
                                                              providers=["CPUExecutionProvider"]))
        detector_source = f"float ONNX heads on x86 CPU ({args.threads} threads)"
    else:
        backend = fake_hailo.planted_backend([(67, 0.7, 300, 280, 364, 408, 16), (0, 0.9, 120, 100, 520, 620, 32)])
        detector_source = "planted heads (infer ~0 ms)"
    fake_hailo.install(backend)

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    tmp = Path(tempfile.mkdtemp(prefix="m83-baseline-"))
    hef = tmp / "damoyolo_tinynasL20_T.hef"
    hef.write_bytes(b"SIMULATED HEF\n" * 64)
    acc = dict(config["accelerator"], yolo_hef=str(hef), yolo_hef_sha256=accel.file_sha256(str(hef)))
    accel.configure(acc, info=accel.AcceleratorInfo(True, True, "hailo", "SIMULATED (fake_hailo)"))
    module_cfg = {**config["modules"]["driver_attention"], "recalibrate": False}
    os.chdir(tmp)  # the module may write calibration/data files: keep them out of the repo
    module = get_module_class("driver_attention")(module_cfg)

    if args.images:
        frames, source = frames_from_images(args.images, args.frames)
        source = f"{args.images.name}: {source}"
    else:
        frames, source = frames_from_oi(args.data, args.frames)
    recorder = st.StageRecorder()
    found: dict = {}
    for f in frames[:15]:  # warm-up: model graphs, first allocations
        module.set_frame_time(time.monotonic())
        module.process(f)
    for f in frames:
        found = st.instrument_module(module, recorder)
        module.set_frame_time(time.monotonic())
        t0 = time.perf_counter()
        module.process(f)
        recorder.add("process_total", (time.perf_counter() - t0) * 1000.0)
        recorder.end_frame()
    summary = recorder.summary()
    shares = {}
    procs = [fr["process_total"] for fr in recorder.frames]
    total = sum(procs)
    for stage in ("facemesh", "hands", "detector_total", "module_logic"):
        shares[stage] = round(sum(fr.get(stage, 0.0) for fr in recorder.frames) / total, 4) if total else None
    result = {
        "label": f"HOST CPU baseline on {platform.node()} ({platform.machine()}, {platform.platform()}); NOT a Pi "
                 "and NOT Hailo",
        "job": os.environ.get("SDJOB_ID"), "source": source, "detector": detector_source,
        "frames": len(frames), "instrumented": found, "summary": summary, "time_shares": shares,
        "yolo_every_n_frames": module_cfg.get("yolo_every_n_frames"),
        "config_sha256": hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest(),
        "core_accel_sha256": hashlib.sha256((ROOT / "core" / "accel.py").read_bytes()).hexdigest(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    with __import__("contextlib").suppress(Exception):
        import cv2
        import mediapipe

        result["versions"].update(opencv=cv2.__version__, mediapipe=mediapipe.__version__)
    (args.out / "host_baseline.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("label", "source", "detector", "time_shares")}, indent=1))
    for stage in ("process_total", "facemesh", "hands", "detector_total", "preprocess", "hailo_infer", "decode",
                  "module_logic"):
        s = summary.get(stage)
        if s:
            print(f"  {stage:<15} p50 {s['p50']:7.2f}  p95 {s['p95']:7.2f}  ms  (frames {s['frames_with_stage']})")
    close = getattr(module, "close", None)
    if callable(close):
        close()
    detectors.release_all()
    fake_hailo.uninstall()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
