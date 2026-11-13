#!/usr/bin/env python3
"""Behavioural evaluation harness over the scripted cabin scenarios (M8.4, A5/A6).

    python scripts/eval_behavior.py --out <file.json> [--scenario NAME ...] [--hands-policy POLICY]

Runs every sim.scenario.CABIN_SCENARIOS entry through the REAL
DriverAttentionModule (sim.perception.run_cabin) and writes machine-readable
JSON: per scenario, the shipped alerts, the experimental observations, the risk
score, the engine's shadow decisions and abstentions, hand-model calls; and
aggregated event-level precision / recall (Wilson 95% CIs), false alerts per
hour, detection delay, abstention and missing-evidence rates.

EVIDENCE CLASS: synthetic. The labels are the scripts themselves. These numbers
are regression evidence that the logic behaves as designed on constructed
inputs. They are NOT real-world accuracy and must never be quoted as such.
There is no human-labelled or model-generated in-cabin set today.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.behavior.evaluation import (
    ALERT_TYPES,
    aggregate,
    engine_rates,
    score_scenario,
)
from core.behavior.experimental import EXPERIMENTAL_ALERT_TYPES
from sim.perception import run_cabin
from sim.scenario import CABIN_SCENARIOS

SHADOW_CLAIMS = ("drowsiness", "distraction", "eyes_down", "phone_use", "attention_lost")


def evaluate(names: list[str], **overrides) -> dict:
    rows, shadow_rows, durations, per_scenario, reports = [], [], [], {}, []
    for name in names:
        scenario = CABIN_SCENARIOS[name]
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_cabin(scenario, tmp, **overrides)
        expected, allowed = scenario.label.expected, scenario.label.allowed
        scored = score_scenario(trace["events"], expected, allowed)
        shadow_expected = [e for e in expected if e[0] in SHADOW_CLAIMS]
        shadow_scored = score_scenario(trace["shadow_events"], shadow_expected, allowed, alert_types=SHADOW_CLAIMS)
        rows.append(scored)
        shadow_rows.append(shadow_scored)
        durations.append(scenario.duration)
        reports.append(trace["engine"])
        experimental = [e for e in trace["events"] if e["type"] in EXPERIMENTAL_ALERT_TYPES]
        engine = trace["engine"] or {}
        per_scenario[name] = {
            "kind": scenario.kind, "description": scenario.description,
            "expected": [list(e) for e in expected], "allowed": list(allowed),
            "alerts": [f"{e['type']}@{e['t']}" for e in trace["events"] if e["type"] in ALERT_TYPES],
            "experimental_observations": [f"{e['type']}@{e['t']}" for e in experimental],
            "risk_peak": trace["risk_peak"], "risk_mean": trace["risk_mean"],
            "alert_active_s": trace["active_s"],
            "production": scored, "shadow": shadow_scored if engine else None,
            "shadow_events": [f"{e['type']}@{e['t']}" for e in trace["shadow_events"]],
            "abstaining_at_end": engine.get("abstaining"), "phone_evidence_at_end": engine.get("phone_evidence"),
            "driver_at_end": engine.get("driver"), "comparison": engine.get("comparison"),
            "frames": trace["frames"], "hands_calls": trace["hands_calls"], "detector_calls": trace["detector_calls"],
            "process_ms_per_frame": trace["process_ms_per_frame"],
        }
    return {
        "evidence_class": "synthetic",
        "disclaimer": ("Scripted scenarios through the real module with model stand-ins. Regression evidence "
                       "only; NOT real-world accuracy. No human-labelled or model-generated in-cabin data exists."),
        "evidence_classes": {"synthetic": len(names), "human_labelled": 0, "model_generated": 0},
        "production": aggregate(rows, durations),
        "shadow": aggregate(shadow_rows, durations) if any(reports) else None,
        "engine": engine_rates(reports),
        "experimental_observations": sum(len(s["experimental_observations"]) for s in per_scenario.values()),
        "hands_calls": sum(s["hands_calls"] for s in per_scenario.values()),
        "detector_calls": sum(s["detector_calls"] for s in per_scenario.values()),
        "scenarios": per_scenario,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--hands-policy", default=None, help="override modules.driver_attention.hands_policy")
    args = parser.parse_args()
    names = args.scenario or sorted(CABIN_SCENARIOS)
    overrides = {"hands_policy": args.hands_policy} if args.hands_policy else {}
    result = evaluate(names, **overrides)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    summary = {k: result[k] for k in ("production", "shadow", "experimental_observations", "hands_calls")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
