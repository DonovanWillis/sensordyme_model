#!/usr/bin/env python3
"""Rehearse the 10 flagship distracted-driving scenarios -- SIMULATED.

    python scripts/demo/flagship_rehearsal.py            # the table
    python scripts/demo/flagship_rehearsal.py --json     # the same, as JSON
    python scripts/demo/flagship_rehearsal.py --only 3,6 # some rows (9 and 10 run together)

Scenarios 1-8: the real DriverAttentionModule at the shipped thresholds on
scripted cabin inputs (sim.perception.run_cabin). Scenarios 9-10: the virtual
device and the real backend through a camera freeze and recovery. The same
runs and checks as tests/test_m13_flagship_scenarios.py (sim/flagship.py).

Every line is SIMULATED: no camera sensor, no Pi, no Hailo, no driver. A FAIL
tagged "known gap" is a requirement the product does not meet yet, recorded
in sim/flagship.py KNOWN_GAPS and docs/DEMO_SCENARIOS.md -- not tuned away.
Exit status: 0 when every FAIL is a known gap, 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

STAMP = "SIMULATED"


def _quiet(fn, *args, **kwargs):
    """Run `fn` with the runtime's own console chatter kept off the table."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def collect(only: set[int], fps: float) -> list[dict]:
    import logging

    from sim import flagship

    logging.disable(logging.WARNING)
    rows = [_quiet(lambda c=case: flagship.describe_case(flagship.run_case(c, fps=fps)))
            for case in flagship.CASES if not only or case.number in only]
    if not only or only & {9, 10}:
        device_rows = _quiet(lambda: flagship.describe_device(flagship.run_device()))
        rows += [r for r in device_rows if not only or r["number"] in only]
    return rows


def _latency(row: dict) -> str:
    return "-" if row.get("latency_s") is None else f"{row['latency_s']:.2f}s"


def table(rows: list[dict]) -> str:
    from sim.flagship import KNOWN_GAPS

    out = [f"{STAMP} -- M13 flagship distracted-driving scenarios (stationary; never real driving)", ""]
    header = f"{'#':>2}  {'scenario':<26} {'harness':<28} {'latency':>8}  verdict"
    out += [header, "-" * len(header)]
    for row in rows:
        out.append(f"{row['number']:>2}  {row['scenario']:<26} {row['harness']:<28} {_latency(row):>8}  "
                   f"{row['verdict']}")
        wrap = textwrap.TextWrapper(width=118, initial_indent=" " * 4, subsequent_indent=" " * 18)
        for label, value in (("expected", row["expected"]), ("events", row["events"]),
                             ("phases", row["phases"]), ("attribution", row["attribution"]),
                             ("evidence", row["evidence"]),
                             ("why", "; ".join(row["why"]) or "-")):
            out.append(wrap.fill(f"{label:<13} {value}"))
        if row.get("log_only_events"):
            out.append(wrap.fill(f"{'log only':<13} {', '.join(row['log_only_events'])} (recorded, never alerted)"))
        if row.get("latency_detail"):
            detail = ", ".join(f"{k} {v}" for k, v in row["latency_detail"].items())
            out.append(wrap.fill(f"{'latencies':<13} {detail}"))
        for problem in row["problems"]:
            tag = f" [known gap {problem['gap']}]" if problem["gap"] else " [UNEXPECTED]"
            out.append(wrap.fill(f"{'not met':<13} {problem['text']}{tag}"))
        out.append(f"    {STAMP} ({row['harness'].split('(')[-1].rstrip(')')})")
    gaps = sorted({p["gap"] for r in rows for p in r["problems"] if p["gap"]})
    if gaps:
        out += ["", f"{STAMP} -- known gaps (requirements not met yet; not tuned away):"]
        for gap_id in gaps:
            gap = KNOWN_GAPS[gap_id]
            out.append(textwrap.fill(f"{gap.id} {gap.title} -- {gap.classification}. Fix: {gap.fix}", width=118,
                                     initial_indent="  ", subsequent_indent="     "))
    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    known = sum(1 for r in rows if r["verdict"].startswith("FAIL (known gap"))
    out += ["", (f"{STAMP} -- {passed} PASS, {known} FAIL on known gaps only, "
                 f"{len(rows) - passed - known} FAIL unexpected, of {len(rows)}")]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    parser.add_argument("--only", default="", help="comma-separated scenario numbers (1-10)")
    parser.add_argument("--fps", type=float, default=None, help="frame rate for scenarios 1-8 (default 10)")
    args = parser.parse_args(argv)
    from sim.flagship import FPS, KNOWN_GAPS, as_json_safe

    only = {int(n) for n in args.only.split(",") if n.strip()}
    rows = collect(only, args.fps or FPS)
    if args.json:
        gaps = sorted({p["gap"] for r in rows for p in r["problems"] if p["gap"]})
        print(json.dumps(as_json_safe({
            "simulated": True, "label": STAMP, "rows": rows,
            "known_gaps": [vars(KNOWN_GAPS[g]) for g in gaps],
        }), indent=2, default=str))
    else:
        print(table(rows))
    unexpected = [r for r in rows if r["verdict"] == "FAIL"]
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
