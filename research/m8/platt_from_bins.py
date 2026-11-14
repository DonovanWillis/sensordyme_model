"""Platt map (hit ~ sigmoid(a * logit(score) + b)) fitted to the reliability bins of a bench score file.

    python -m research.m8.platt_from_bins score_<name>_val.json [...]

For score files whose per-detection predictions are not at hand (EXP-08 predictions live on oryn),
this fits the map to the 10 count-weighted bins that `bench score` stores. Checked against the
per-detection fit of research/m8/analysis.py on the pretrained reference-decoder person class
(job 20261111-235044-b2261f): raw score for a 0.8 hit rate 0.566 (bins) vs 0.565 (per detection).
Prints, per class, the fit and the raw score at which a detection is right 50/70/80/90% of the time.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def fit(bins: list[dict]) -> tuple[float, float]:
    s = np.clip(np.array([b["mean_score"] for b in bins]), 1e-4, 1 - 1e-4)
    y = np.array([b["precision"] for b in bins])
    n = np.array([b["n"] for b in bins], dtype=float)
    x = np.log(s / (1 - s))
    a, b = 1.0, 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-(a * x + b)))
        w = n * p * (1 - p)
        g = np.array([np.sum(n * (p - y) * x), np.sum(n * (p - y))])
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def raw_for(a: float, b: float, q: float) -> float:
    return float(1 / (1 + np.exp(-((np.log(q / (1 - q)) - b) / a))))


def main(paths: list[str]) -> int:
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        for label, c in d["classes"].items():
            bins = (c.get("calibration") or {}).get("bins")
            if not bins:
                continue
            a, b = fit(bins)
            print(json.dumps({"file": path, "class": label, "ece": c["calibration"]["ece"], "a": round(a, 3),
                              "b": round(b, 3), "raw_score_for_hit_rate": {q: round(raw_for(a, b, q), 3)
                                                                          for q in (0.5, 0.7, 0.8, 0.9)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
