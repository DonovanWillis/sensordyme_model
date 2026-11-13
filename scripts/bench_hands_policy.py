#!/usr/bin/env python3
"""Host cost of MediaPipe Hands under each hands_policy (M8.4 B5), on scripted cabins.

    python scripts/bench_hands_policy.py --out <file.json> [--repeats 1]

For each scenario below, runs the REAL DriverAttentionModule with a REAL
MediaPipe Hands graph (FaceMesh and the detector stay scripted) under
``hands_policy: every_tick`` (the pre-M8.4 behaviour) and ``phone_visible``
(the M8.4 default), and records the module's mean per-frame process() time and
how many times Hands ran. It then re-runs every sim.scenario.CABIN_SCENARIOS
entry with scripted hands under both policies and checks the shipped events are
identical (behavioural coverage).

HOST numbers (this machine's CPU), labelled as such: never a Pi 5 number. The
frames are synthetic textures, so Hands finds no hand and only its palm
detector runs -- the cost when a hand IS tracked differs (M83_INFERENCE_PATH.md
measured p50 32 ms on detector frames of real person photos).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.perception import run_cabin
from sim.scenario import CABIN_SCENARIOS

#: The device's measured frame rate (modules/driver_attention.py RISK_EMA_REFERENCE_DT_S).
FPS = 30.0
COST_SCENARIOS = ("sustained_look_away", "mirror_checks", "phone_mounted_unused", "held_phone_eyes_off_road")


def _real_hands():
    import mediapipe as mp

    return mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cost: dict = {}
    for name in COST_SCENARIOS:
        row = {}
        for policy in ("every_tick", "phone_visible"):
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_cabin(CABIN_SCENARIOS[name], tmp, fps=FPS, hands_factory=_real_hands, hands_policy=policy)
            row[policy] = {"process_ms_per_frame": trace["process_ms_per_frame"], "hands_calls": trace["hands_calls"],
                           "detector_calls": trace["detector_calls"], "frames": trace["frames"]}
        cost[name] = row
    coverage: dict = {}
    identical = True
    for name, scenario in sorted(CABIN_SCENARIOS.items()):
        events = {}
        for policy in ("every_tick", "phone_visible"):
            with tempfile.TemporaryDirectory() as tmp:
                trace = run_cabin(scenario, tmp, hands_policy=policy)
            events[policy] = [(e["t"], e["type"]) for e in trace["events"]
                              if e["type"] not in ("erratic_hand_movement",)]
        same = events["every_tick"] == events["phone_visible"]
        identical &= same
        coverage[name] = {"identical_events": same, "every_tick": events["every_tick"],
                          "phone_visible": events["phone_visible"]}
    result = {
        "host": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version()},
        "label": "HOST CPU, synthetic frames, real MediaPipe Hands; not a Pi 5 number",
        "fps": FPS,
        "cost": cost, "coverage_identical_on_all_scenarios": identical, "coverage": coverage,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps({"cost": cost, "coverage_identical_on_all_scenarios": identical}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
