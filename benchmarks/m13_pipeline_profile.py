"""M13: per-stage cost of the device's REAL inference loop on THIS host (a Mac is NOT a Pi).

What runs is services/runtime.py DeviceRuntime, built by main.build_runtime from
the repo's config.yaml exactly as the virtual device builds it (sim/device.py,
real models, no scripted perception), and driven one production pass at a time
(`_inference_pass`, the loop body, unchanged). Only the camera is replaced: a
fixed, looped 640x480 frame sequence is published on the runtime's own FrameBus,
either from a capture thread at camera rate (`--paced FPS`, production-like:
capture overlaps inference, frames are skipped when inference is slower) or one
frame per pass (`--paced 0`: every frame analysed, the pure per-frame cost).

Frames: a public-domain portrait (matplotlib's sample `grace_hopper.jpg`, US Navy
photo) pasted on a grey 640x480 canvas with a small per-frame sub-pixel drift, so
FaceMesh tracks a real face and the detector sees a real person. Not committed;
read from the installed matplotlib. Without it, `--frames-from synthetic` draws a
cartoon (FaceMesh then finds no face: a different, cheaper path).

Detector backends (the label travels into the result):
  ultralytics     what a dev Mac runs: yolo11n.pt on CPU torch (the device runs a
                  Hailo-8L instead -- these numbers say nothing about it).
  hailo-replay    the device's own core/accel.py HailoYOLO.predict (letterbox,
                  BGR->RGB, head pairing, decode, NMS, frame mapping) behind
                  scripts/hw_validation/fake_hailo.py, whose "chip" answers with
                  recorded float head tensors (research/m13/decoder_compare.py)
                  in ~0 ms. SIMULATED inference; the CPU work around it is real.

Per pass the benchmark times (wrapping instance attributes in this process only;
no repo file changes) the runtime's own stages from core/stage_timing.py plus:
capture pick (FrameBus wait, paced mode: idle time), the perception engine's
begin/end of frame, custom rules, monitoring nodes, telemetry, dashboard status,
annotate, the annotated JPEG encode, the sd-overlay/1 build + header encode and
its JPEG encode, and the whole pass. CPU% and RSS come from psutil.

    python benchmarks/m13_pipeline_profile.py --frames 600 --viewers annotated --out run.json
    python benchmarks/m13_pipeline_profile.py --repo /path/to/other/checkout ...   # A/B a revision
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent


def _repo_root(argv: list[str]) -> Path:
    for i, a in enumerate(argv):
        if a == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if a.startswith("--repo="):
            return Path(a.split("=", 1)[1]).resolve()
    return HERE.parent


REPO = _repo_root(sys.argv)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

#: Stages this harness adds on top of core/stage_timing.STAGES.
HARNESS_STAGES = (
    "pass_total", "capture_pick", "perceive_total", "engine_begin_frame", "engine_end_frame",
    "alert_dispatch_call", "rules", "nodes", "telemetry", "push_status", "annotate", "jpeg_encode_annotated",
    "overlay_build", "overlay_header_encode", "overlay_jpeg_encode",
)


# ---------------------------------------------------------------------- frames
def portrait_frames(n: int = 90, width: int = 640, height: int = 480) -> tuple[list[np.ndarray], str]:
    try:
        import matplotlib.cbook as cbook

        path = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 - optional input
        src = None
    if src is None:
        raise SystemExit("matplotlib's grace_hopper.jpg is not available; use --frames-from synthetic")
    scale = (height * 0.9) / src.shape[0]
    face = cv2.resize(src, (round(src.shape[1] * scale), round(src.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    frames = []
    for i in range(n):
        canvas = np.full((height, width, 3), 90, np.uint8)
        dx = 4.0 * np.sin(2 * np.pi * i / n)
        dy = 2.0 * np.cos(2 * np.pi * i / n)
        m = np.float32([[1, 0, (width - face.shape[1]) / 2 + dx], [0, 1, (height - face.shape[0]) / 2 + dy]])
        cv2.warpAffine(face, m, (width, height), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
        frames.append(canvas)
    return frames, f"matplotlib sample grace_hopper.jpg (public domain), {n} frames, drift +-4/2 px"


def synthetic_frames(n: int = 90, width: int = 640, height: int = 480) -> tuple[list[np.ndarray], str]:
    frames = []
    for i in range(n):
        f = np.full((height, width, 3), 58, np.uint8)
        cx = int(width / 2 + 8 * np.sin(2 * np.pi * i / n))
        cv2.circle(f, (cx, int(height * 0.4)), 86, (150, 175, 205), -1)
        cv2.rectangle(f, (192, 288), (448, 479), (90, 90, 110), -1)
        frames.append(f)
    return frames, f"synthetic cartoon, {n} frames"


# ---------------------------------------------------------------------- hailo replay
def install_hailo_replay(tensors_dir: Path, weights: str) -> dict:
    """Seed core.detectors' cache with the device's HailoYOLO on the fake chip, replaying heads."""
    import fake_hailo

    from core import accel, detectors

    files = sorted(tensors_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz head tensors in {tensors_dir}")
    heads = []
    for f in files:
        with np.load(f) as z:
            heads.append({k.replace("__", "/"): np.ascontiguousarray(z[k]) for k in z.files})
    state = {"i": 0}

    def backend(batch):
        out = heads[state["i"] % len(heads)]
        state["i"] += 1
        return out

    fake_hailo.install(backend)
    model = accel.HailoYOLO("damoyolo_tinynasL20_T.hef (SIMULATED: replayed float heads)", decoder="legacy")
    detectors._model_cache[(weights, None)] = model
    return {"tensor_sets": len(files), "decoder": model.decoder, "dir": str(tensors_dir)}


# ---------------------------------------------------------------------- timing harness
class Recorder:
    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {name: [] for name in HARNESS_STAGES}
        self.frame: dict[str, float] = {}
        self.active = False

    def add(self, stage: str, ms: float) -> None:
        if self.active:
            self.frame[stage] = self.frame.get(stage, 0.0) + ms

    def end_frame(self, processed: bool) -> None:
        if self.active and processed:
            for k, v in self.frame.items():
                self.samples.setdefault(k, []).append(v)
        self.frame = {}

    def wrap(self, owner: Any, attr: str, stage: str) -> None:
        original = getattr(owner, attr)

        @functools.wraps(original)
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                self.add(stage, (time.perf_counter() - t0) * 1000.0)

        setattr(owner, attr, timed)


class StageSink:
    """Replaces core.stage_timing.TIMINGS.add for the run: every runtime sample, per frame."""

    def __init__(self, rec: Recorder, original: Callable) -> None:
        self.rec, self.original = rec, original

    def __call__(self, stage: str, ms: float) -> bool:
        self.rec.add(stage, float(ms))
        return self.original(stage, ms)


def stats(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    s = sorted(values)

    def q(p):
        pos = p * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return {"n": len(s), "p50": round(q(0.5), 4), "p90": round(q(0.9), 4), "mean": round(statistics.fmean(s), 4),
            "max": round(s[-1], 4)}


def host_conditions() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    out = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
           "cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
           "ncpu": os.cpu_count(), "loadavg_before": os.getloadavg() if hasattr(os, "getloadavg") else None}
    procs = sh(["pgrep", "-fl", "Simulator|xcodebuild|XCTest"]) or ""
    out["ios_simulator_or_xcodebuild_running"] = bool(procs.strip())
    try:
        out["cv2"], out["numpy"] = cv2.__version__, np.__version__
        import mediapipe

        out["mediapipe"] = mediapipe.__version__
    except Exception:  # noqa: BLE001
        out["mediapipe"] = None
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    out["repo"] = str(REPO)
    return out


# ---------------------------------------------------------------------- build
def build_runtime(state_dir: Path, weights: str, *, dashboard: bool, module: Optional[str]):
    from sim import scenario as sc
    from sim.device import VirtualDevice

    class _Device(VirtualDevice):
        def config(self) -> dict:
            cfg = super().config()
            mods = cfg.get("modules", {})
            for mod_cfg in mods.values():
                for key in ("yolo_weights", "model_weights", "phone_weights"):
                    if mod_cfg.get(key) == "yolo11n.pt":
                        mod_cfg[key] = weights
            # No console sink: printing alerts is not pipeline cost.
            cfg["alerts"]["sinks"] = ["gpio_buzzer"]
            if module:
                cfg["module"] = module
            return cfg

    device = _Device(state_dir=state_dir, backend_url="http://127.0.0.1:9", scenario=sc.get("attentive"),
                     video="(benchmark frames: the harness publishes on the bus)",
                     device_key="m13-benchmark-not-a-real-key",
                     dashboard_port=0 if dashboard else None, log_level="warning")
    runtime = device.build()
    runtime._build()
    return device, runtime


def instrument(runtime, rec: Recorder) -> None:
    from core import stage_timing

    stage_timing.TIMINGS.add = StageSink(rec, stage_timing.TIMINGS.add)  # type: ignore[method-assign]
    rec.wrap(runtime.bus, "wait_for_fresh_stamped", "capture_pick")
    rec.wrap(runtime, "_perceive", "perceive_total")
    rec.wrap(runtime.perception, "begin_frame", "engine_begin_frame")
    rec.wrap(runtime.perception, "end_frame", "engine_end_frame")
    rec.wrap(runtime, "_dispatch_module_events", "alert_dispatch_call")
    rec.wrap(runtime, "_evaluate_custom_rules", "rules")
    rec.wrap(runtime, "_run_nodes", "nodes")
    rec.wrap(runtime.telemetry, "log_if_due", "telemetry")
    rec.wrap(runtime.module, "get_telemetry_snapshot", "telemetry")
    rec.wrap(runtime.module, "annotate", "annotate")
    rec.wrap(runtime.module, "annotate_debug", "annotate")
    if runtime.dashboard is not None:
        rec.wrap(runtime.dashboard, "push_status", "push_status")
        rec.wrap(runtime.dashboard, "push_frame", "jpeg_encode_annotated")
        if hasattr(runtime.dashboard, "push_overlay_frame"):
            rec.wrap(runtime.dashboard, "push_overlay_frame", "overlay_jpeg_encode")
    overlay = getattr(runtime, "_overlay", None)
    if overlay is not None and hasattr(overlay, "build"):
        rec.wrap(overlay, "build", "overlay_build")
    import services.runtime as rt_mod

    if hasattr(rt_mod, "overlay_mod"):
        rec.wrap(rt_mod.overlay_mod, "encode_header", "overlay_header_encode")


# ---------------------------------------------------------------------- run
def run_once(args, frames: list[np.ndarray], runtime, rec: Recorder) -> dict:
    import psutil

    from core import stage_timing

    proc = psutil.Process()
    loop = runtime._new_inference_loop_state()
    stop = threading.Event()
    published = {"n": 0}
    pub_stop = threading.Event()
    period = 1.0 / args.paced if args.paced > 0 else 0.0

    def publisher():
        i = 0
        next_at = time.perf_counter()
        while not pub_stop.is_set():
            runtime.bus.publish(frames[i % len(frames)].copy())  # a camera hands out a fresh array per frame
            published["n"] += 1
            i += 1
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                pub_stop.wait(delay)
            else:
                next_at = time.perf_counter()

    # warm-up (models, graphs, first stride ticks) -- untimed
    rec.active = False
    for i in range(args.warmup):
        runtime.bus.publish(frames[i % len(frames)].copy())
        runtime._inference_pass(loop, stop)
    stage_timing.TIMINGS.reset()
    rec.active = True
    rss_peak = 0
    if period:
        loop.frame_timeout_s = 1.0
        thread = threading.Thread(target=publisher, daemon=True)
        thread.start()
    cpu0, wall0 = proc.cpu_times(), time.perf_counter()
    processed = 0
    i = 0
    while processed < args.frames:
        if not period:
            runtime.bus.publish(frames[i % len(frames)].copy())
            i += 1
        before = runtime._last_processed_at
        t0 = time.perf_counter()
        runtime._inference_pass(loop, stop)
        rec.add("pass_total", (time.perf_counter() - t0) * 1000.0)
        did = runtime._last_processed_at != before
        rec.end_frame(did)
        processed += int(did)
        if processed % 50 == 0:
            rss_peak = max(rss_peak, proc.memory_info().rss)
    wall = time.perf_counter() - wall0
    cpu1 = proc.cpu_times()
    pub_stop.set()
    rss_peak = max(rss_peak, proc.memory_info().rss)
    cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    snap = stage_timing.TIMINGS.snapshot()
    stream = runtime.dashboard.stream_stats() if runtime.dashboard is not None else None
    return {
        "processed_frames": processed, "wall_s": round(wall, 3), "processed_fps": round(processed / wall, 2),
        "published_frames": published["n"] if period else processed,
        "cpu_percent_of_one_core": round(100.0 * cpu_s / wall, 1), "rss_mb_peak": round(rss_peak / 2**20, 1),
        "runtime_processed_fps": snap["processed_fps"],
        "stream": None if stream is None else {k: stream[k] for k in ("latest_seq", "camera_fallback_encodes")},
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=str(HERE.parent), help="checkout whose code runs (A/B a revision)")
    p.add_argument("--frames", type=int, default=600, help="processed frames per repeat")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--paced", type=float, default=30.0, help="camera fps on a capture thread; 0 = one frame per pass")
    p.add_argument("--viewers", choices=("none", "annotated", "overlay", "both"), default="none")
    p.add_argument("--detector", choices=("ultralytics", "hailo-replay"), default="ultralytics")
    p.add_argument("--tensors", type=Path, help="hailo-replay: directory of research/m13 .npz head tensors")
    p.add_argument("--weights", default=None, help="yolo11n.pt path (default: the repo root's, else the main checkout's)")
    p.add_argument("--module", default=None, help="config module override (default: config.yaml's)")
    p.add_argument("--frames-from", choices=("portrait", "synthetic"), default="portrait")
    p.add_argument("--cprofile", type=Path, help="also write a cProfile .pstats of the timed passes")
    p.add_argument("--out", type=Path)
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    weights = args.weights
    if weights is None:
        for cand in (REPO / "yolo11n.pt", HERE.parent / "yolo11n.pt", Path.home() / "sensordyme/sensordyme_model/yolo11n.pt"):
            if cand.exists():
                weights = str(cand)
                break
    if weights is None or not Path(weights).exists():
        raise SystemExit("yolo11n.pt not found; pass --weights (no download is attempted)")

    frames, frames_desc = (portrait_frames if args.frames_from == "portrait" else synthetic_frames)()
    replay = None
    if args.detector == "hailo-replay":
        if args.tensors is None:
            raise SystemExit("--detector hailo-replay needs --tensors")
        replay = install_hailo_replay(args.tensors, weights)

    import logging

    from core import stage_timing

    with tempfile.TemporaryDirectory(prefix="m13prof-") as tmp:
        device, runtime = build_runtime(Path(tmp), weights, dashboard=args.viewers != "none", module=args.module)
        logging.getLogger().setLevel(logging.ERROR)
        if runtime.dashboard is not None:
            if args.viewers in ("annotated", "both"):
                runtime.dashboard.viewer_joined(overlay=False)
            if args.viewers in ("overlay", "both"):
                runtime.dashboard.viewer_joined(overlay=True)
        rec = Recorder()
        instrument(runtime, rec)
        profiler = None
        if args.cprofile:
            import cProfile

            profiler = cProfile.Profile()
        repeats = []
        try:
            for _ in range(args.repeats):
                rec.samples = {name: [] for name in HARNESS_STAGES}
                if profiler is not None:
                    profiler.enable()
                totals = run_once(args, frames, runtime, rec)
                if profiler is not None:
                    profiler.disable()
                repeats.append({"totals": totals, "stages_ms": {k: stats(v) for k, v in rec.samples.items() if v}})
        finally:
            with contextlib.suppress(Exception):
                device.stop()
        if profiler is not None:
            profiler.dump_stats(str(args.cprofile))
        detector = getattr(runtime, "_model_identity", None)

    result = {
        "schema": "sd-m13-pipeline-profile/1",
        "label": args.label,
        "host_class": stage_timing.platform_tag(simulated=args.detector == "hailo-replay")["class"],
        "note": "DESKTOP measurement (this host): NOT Raspberry Pi 5 / Hailo-8L performance",
        "conditions": host_conditions(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "frames": frames_desc,
        "detector": {"backend": args.detector, "identity": detector, "replay": replay,
                     "label": ("ultralytics yolo11n.pt on CPU torch (dev Mac backend; the device uses Hailo)"
                               if args.detector == "ultralytics"
                               else "device HailoYOLO.predict on a SIMULATED chip (replayed float heads)")},
        "repeats": repeats,
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    summary_keys = ("pass_total", "detector_total", "hailo_preprocess", "decoder_cpu", "face_landmarks", "hands",
                    "tracking", "behaviour_reasoning", "engine_end_frame", "alert_dispatch", "rules", "nodes",
                    "telemetry", "push_status", "annotate", "jpeg_encode_annotated", "overlay_build",
                    "overlay_header_encode", "overlay_jpeg_encode", "frame_age", "frame_to_decision", "capture_pick")
    for r in repeats:
        print(json.dumps(r["totals"]))
        for k in summary_keys:
            s = r["stages_ms"].get(k)
            if s:
                print(f"  {k:24s} p50 {s['p50']:8.3f}  p90 {s['p90']:8.3f}  n {s['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
