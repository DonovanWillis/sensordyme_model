#!/usr/bin/env python3
"""Rehearse the Jupiter "second wow" demo end to end -- SIMULATED.

    python scripts/demo/jupiter_rehearsal.py           # each step, as it happened
    python scripts/demo/jupiter_rehearsal.py --json
    python scripts/demo/jupiter_rehearsal.py --prompt "Alert me if someone stays in this area for 10 seconds"

"Alert me when a person remains in this area for 10 seconds." on a virtual
general-purpose device with the real backend: draw the area, the instruction,
the typed policy, Activate, the device's rule engine on its shared detector and
tracker, scripted people coming and going, and the alert in the backend's
history and the device's own local history. sim/jupiter_demo.py does the work;
tests/test_m13_jupiter_demo.py pins it. No language model is used (any model
call fails the run), no Pi, no camera: every line is SIMULATED.
Exit status: 0 when every step held, 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("GLOG_minloglevel", "2")

STAMP = "SIMULATED"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prompt", default=None, help="another phrasing to try (default: the demo's)")
    args = parser.parse_args(argv)
    from sim import jupiter_demo as demo

    logging.disable(logging.WARNING)
    with contextlib.redirect_stdout(io.StringIO()):
        result = demo.run(prompt=args.prompt or demo.PROMPT)
    if args.json:
        print(json.dumps({
            "simulated": True, "label": STAMP, "prompt": args.prompt or demo.PROMPT, "ok": result.ok,
            "latency_s": result.latency_s,
            "steps": [vars(s) for s in result.steps], "policies": result.preview.get("policies"),
            "backend_events": result.backend_events, "local_events": result.local_events,
        }, indent=2, default=str))
        return 0 if result.ok else 1
    print(f"{STAMP} -- Jupiter demo: {args.prompt or demo.PROMPT!r}")
    print(f"{STAMP} -- virtual general-purpose device + real backend; scripted people; no language model\n")
    for step in result.steps:
        mark = "PASS" if step.ok else "FAIL"
        print(f"[{STAMP}] step {step.n} {mark}  {step.title}")
        print(textwrap.fill(f"call   {step.call}", width=118, initial_indent="    ", subsequent_indent="           "))
        print(textwrap.fill(f"result {step.result}", width=118, initial_indent="    ", subsequent_indent="           "))
        for key, value in step.detail.items():
            text = "; ".join(map(str, value)) if isinstance(value, list) else str(value)
            print(textwrap.fill(f"{key}: {text}", width=118, initial_indent="      ", subsequent_indent="        "))
    print(f"\n{STAMP} -- alert {result.latency_s} s after the person entered the area "
          f"(device clock; dwell {demo.DWELL_S:g} s); overall {'PASS' if result.ok else 'FAIL'}")
    if result.backend_events:
        print(f"{STAMP} -- known gap J1: the alert's duration field reads {result.backend_events[0]['duration']} s "
              f"(the rule engine's confirmation step), not the time the person stayed")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
