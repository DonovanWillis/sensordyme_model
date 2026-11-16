"""M13: per-frame cost of core/eye_signal.AdaptiveBaseline at the production rate (THIS host, not a Pi).

driver_attention keeps two of them (EAR and iris openness, ear_baseline_window_s
= 60 s) and calls observe() + reference() on each, every frame. At 30 fps the
window holds 1,800 samples. This replays a 60 s warm window and then times
`--frames` frames of (observe + reference) x 2 with the checkout under --repo,
so a revision can be A/B'd on the same machine:

    python benchmarks/m13_baseline_cost.py --repo /path/to/base --out base.json
    python benchmarks/m13_baseline_cost.py --out m13.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root(argv: list[str]) -> Path:
    for i, a in enumerate(argv):
        if a == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
    return HERE.parent


REPO = _repo_root(sys.argv)
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=str(HERE.parent))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    import random

    from core.eye_signal import AdaptiveBaseline

    rng = random.Random(13)  # noqa: S311 - a reproducible signal, not a secret
    ear, iris = AdaptiveBaseline(window_s=60.0), AdaptiveBaseline(window_s=60.0)
    dt = 1.0 / args.fps
    now = 1000.0
    for _ in range(int(60 * args.fps)):  # a full window, as after the first minute of a drive
        now += dt
        ear.observe(rng.gauss(0.28, 0.03), now)
        iris.observe(rng.gauss(0.45, 0.05), now)
    per_frame = []
    for _ in range(args.frames):
        now += dt
        a, b = rng.gauss(0.28, 0.03), rng.gauss(0.45, 0.05)
        t0 = time.perf_counter()
        ear.observe(a, now)
        ear.reference(0.30)
        iris.observe(b, now)
        iris.reference(None)
        per_frame.append((time.perf_counter() - t0) * 1e3)
    s = sorted(per_frame)
    out = {"schema": "sd-m13-baseline-cost/1", "label": "desktop host, NOT a Raspberry Pi",
           "platform": platform.platform(), "python": platform.python_version(), "repo": str(REPO),
           "window_samples": int(60 * args.fps), "frames": args.frames,
           "per_frame_ms": {"p50": round(statistics.median(s), 4), "p90": round(s[int(0.9 * (len(s) - 1))], 4),
                            "mean": round(statistics.fmean(s), 4), "max": round(s[-1], 4)}}
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["per_frame_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
