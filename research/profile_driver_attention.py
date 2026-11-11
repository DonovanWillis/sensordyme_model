"""Where does driver_attention spend its per-frame time? (desktop CPU profile, not the Pi)

    python -m research.profile_driver_attention --data <oi7-sd-v1> --frames 300 --out DIR

Feeds real photographs (Open Images val person images, resized to the device's
640x480) through the REAL DriverAttentionModule built from config.yaml, under
cProfile, and splits the time into: MediaPipe FaceMesh, MediaPipe Hands, the
object detector (desktop surrogate: whatever get_yolo_model returns off-Pi), and
everything else (SensorDyme's own Python: trackers, signals, fusion, events).
The absolute numbers are this host's; the SHARE of SensorDyme Python is the
useful part, because that code runs unchanged on the Pi CPU.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import cv2
    import yaml

    from core import accel
    from evaluation import manifest as mf
    from main import get_module_class

    config = yaml.safe_load(Path("config.yaml").read_text())
    accel.configure(config.get("accelerator", {}))
    module = get_module_class("driver_attention")({**config["modules"]["driver_attention"], "recalibrate": False})
    manifest = mf.load(Path(args.data) / "manifest.json")
    items = [i for i in sorted(manifest.split("val"), key=lambda i: i.id) if "person" in i.conditions.get("pools", "")]
    frames = []
    for item in items[: args.frames]:
        img = cv2.imread(str(Path(args.data) / item.path))
        if img is not None:
            frames.append(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    for f in frames[:10]:  # warm-up (model init, first-call allocations)
        module.set_frame_time(time.monotonic())
        module.process(f)

    prof = cProfile.Profile()
    per_frame = []
    prof.enable()
    for f in frames:
        t0 = time.perf_counter()
        module.set_frame_time(time.monotonic())
        module.process(f)
        per_frame.append((time.perf_counter() - t0) * 1e3)
    prof.disable()

    stats = pstats.Stats(prof)
    groups = {
        "mediapipe_process_cum_s": sum(v[3] for k, v in stats.stats.items()  # type: ignore[attr-defined]
                                       if "mediapipe" in k[0] and k[2] == "process"),
        "detector_predict_cum_s": max((v[3] for k, v in stats.stats.items()  # type: ignore[attr-defined]
                                       if k[2] == "predict" and ("ultralytics" in k[0] or "accel.py" in k[0])),
                                      default=0.0),
        "cv2_cvtColor_calls": sum(v[0] for k, v in stats.stats.items() if "cvtColor" in k[2]),  # type: ignore[attr-defined]
    }
    own_all = [(v[2], f"{Path(k[0]).name}:{k[1]}:{k[2]}") for k, v in stats.stats.items()  # type: ignore[attr-defined]
               if "/src/modules/" in k[0] or "/src/core/" in k[0] or k[0].startswith(("modules/", "core/"))]
    own = sorted(own_all, reverse=True)[:15]
    per_frame.sort()
    result = {
        "frames": len(frames), "host_note": "desktop CPU profile; not Pi timings",
        "process_ms_p50": per_frame[len(per_frame) // 2], "process_ms_p90": per_frame[int(0.9 * (len(per_frame) - 1))],
        "profile_total_s": stats.total_tt, **groups,
        "sensordyme_python_self_time_top": [{"self_s": round(t, 4), "func": f} for t, f in own],
        "sensordyme_python_self_time_total_s": round(sum(t for t, _ in own_all), 4),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "profile_driver_attention.json").write_text(json.dumps(result, indent=1))
    stats.sort_stats("cumulative").print_stats(25)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
