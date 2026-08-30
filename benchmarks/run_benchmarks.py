"""Benchmark harness: measures per-module latency/FPS (and Pi thermal) over a fixed duration.

For each module, runs it against either a video file or the live camera, records
per-frame processing latency, and (on a Pi) samples `vcgencmd measure_temp` once a
second. Results are written to `benchmarks/results/{module}_{timestamp}.json`, with
a combined `summary_{timestamp}.csv` across all modules benchmarked in that run.

Modules are constructed straight from config.yaml's `modules.<name>` block, so
driver_attention is benchmarked under its real production settings — including
its YOLO11-nano + MediaPipe Hands stride (`yolo_every_n_frames`), not a
Face-Mesh-only fast path. Each result JSON embeds that exact config under
`module_config` so the numbers are reproducible.

`--profile mac_max` (additive; default remains `pi` and is unaffected) benchmarks
the Mac-only high-accuracy profile instead (YOLO11-x/RT-DETR-l on MPS, every-frame
Hands, gaze fusion) — this is what produces the "what does 10x compute buy" pi-vs-
mac_max comparison row: run the same test video through both profiles and diff the
FPS/latency numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from core.camera import Camera
from modules import available_modules, get_module_class

RESULTS_DIR = Path(__file__).parent / "results"


def read_cpu_temp_c() -> Optional[float]:
    """Read the Pi's CPU temperature via vcgencmd; None if unavailable (e.g. on Mac)."""
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1, check=True
        )
        return float(out.stdout.strip().split("=")[1].split("'")[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _frame_source(video_path: Optional[str], camera_config: dict) -> Iterator[np.ndarray]:
    """Yield frames from a video file if given, else fall back to the live camera."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    else:
        camera = Camera(camera_config)
        try:
            yield from camera.frames()
        finally:
            camera.release()


def benchmark_module(
    module_name: str, config: dict, duration_s: float, video_path: Optional[str], profile: str = "pi"
) -> dict:
    """Run `module_name` for `duration_s` seconds, recording per-frame latency and CPU temp.

    `profile="pi"` (the default) is byte-for-byte the original behavior: it reads
    `config["modules"][module_name]` and the module's own registry name, exactly as
    before this parameter existed. `profile="mac_max"` reads `config["mac_max"][module_name]`
    and looks up `f"{module_name}_max"` in the registry instead.
    """
    if profile == "mac_max":
        module_cls = get_module_class(f"{module_name}_max")
        mac_max_config = config["mac_max"]
        module_config = {**mac_max_config[module_name], "device": mac_max_config.get("device", "mps")}
    else:
        module_cls = get_module_class(module_name)
        module_config = config["modules"][module_name]
    module = module_cls(module_config)

    latencies_ms: list[float] = []
    temps_c: list[tuple[float, float]] = []

    start = time.monotonic()
    last_temp_sample = 0.0

    for frame in _frame_source(video_path, config.get("camera", {})):
        elapsed = time.monotonic() - start
        if elapsed >= duration_s:
            break

        t0 = time.perf_counter()
        module.process(frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        if elapsed - last_temp_sample >= 1.0:
            temp = read_cpu_temp_c()
            if temp is not None:
                temps_c.append((elapsed, temp))
            last_temp_sample = elapsed

    n = len(latencies_ms)
    total_s = time.monotonic() - start
    sorted_lat = sorted(latencies_ms)

    def percentile(p: float) -> float:
        if not sorted_lat:
            return 0.0
        idx = min(len(sorted_lat) - 1, int(len(sorted_lat) * p))
        return sorted_lat[idx]

    return {
        "module": module_name,
        "profile": profile,
        "frame_count": n,
        "duration_s": total_s,
        "fps": n / total_s if total_s > 0 else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
        "cpu_temp_c": temps_c,
        "module_config": module_config,
    }


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint: benchmark one or more modules and write JSON + summary CSV."""
    parser = argparse.ArgumentParser(description="SensorDyme module benchmarks")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--modules", nargs="*", default=None, help="Modules to benchmark (default: all registered pi modules)"
    )
    parser.add_argument("--minutes", type=float, default=1.0, help="Duration per module, in minutes")
    parser.add_argument("--video", default=None, help="Path to a video file (default: live camera)")
    parser.add_argument(
        "--profile",
        choices=["pi", "mac_max"],
        default="pi",
        help="pi (default, unchanged) or mac_max (additive, Mac-only high-accuracy profile)",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.profile == "mac_max":
        from modules.driver_attention_max import require_mps

        require_mps()
        # mac_max currently only has a driver_attention variant (see modules/driver_attention_max.py)
        module_names = args.modules or ["driver_attention"]
    else:
        module_names = args.modules or available_modules()
    duration_s = args.minutes * 60.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    summary_rows = []

    # Filenames are byte-identical to before this flag existed when profile=="pi" (the
    # default); mac_max gets its own prefix so the two profiles' results never collide
    # and are easy to diff side-by-side for the pi-vs-mac_max comparison.
    file_prefix = "" if args.profile == "pi" else f"{args.profile}_"

    for module_name in module_names:
        print(f"Benchmarking '{module_name}' ({args.profile} profile) for {args.minutes:.1f} min...")
        result = benchmark_module(module_name, config, duration_s, args.video, profile=args.profile)

        out_path = RESULTS_DIR / f"{file_prefix}{module_name}_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> {out_path}  (fps={result['fps']:.1f}, p95={result['latency_ms']['p95']:.1f}ms)")

        summary_rows.append(
            {
                "module": module_name,
                "profile": args.profile,
                "fps": f"{result['fps']:.2f}",
                "mean_ms": f"{result['latency_ms']['mean']:.2f}",
                "p50_ms": f"{result['latency_ms']['p50']:.2f}",
                "p95_ms": f"{result['latency_ms']['p95']:.2f}",
                "p99_ms": f"{result['latency_ms']['p99']:.2f}",
                "frame_count": result["frame_count"],
            }
        )

    if summary_rows:
        summary_path = RESULTS_DIR / f"{file_prefix}summary_{timestamp}.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
