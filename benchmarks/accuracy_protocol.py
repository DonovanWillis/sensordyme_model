"""Scripted accuracy protocol: turns "how accurate is it?" into a number.

Two subcommands:

  run    Walk you through a timed scenario script (act out each instruction in front
         of the camera while `make dev-max` runs in another terminal). Records the
         UTC start/end of every phase to benchmarks/results/protocol_<ts>.json.

  grade  Score data/events.csv against a recorded protocol: per-phase PASS/MISS
         (did any expected event type fire inside the phase window?), false
         positives (alertable events firing where nothing was expected), and an
         overall recall / false-positive summary. Results are printed and written
         to a JSON file next to the protocol.

Typical session:

    # terminal 1
    make dev-max
    # terminal 2 (once calibration is done)
    python benchmarks/accuracy_protocol.py run
    # afterwards
    python benchmarks/accuracy_protocol.py grade --protocol benchmarks/results/protocol_<ts>.json

The expected event types below are the mac_max profile's. A phase PASSes if at
least one of its expected types fires within [start, end + grace]; grace covers
sustained-condition trackers and the VLM confirmation debounce, which both fire
a few seconds after the behavior begins.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import ALERT_EVENT_TYPES_MAC_MAX  # the universe of alertable (beeping) types

LEAD_IN_S = 6.0  # "get ready" gap between phases
GRACE_S = 8.0  # events may fire this long after a phase ends (sustain trackers + VLM confirm)


@dataclass
class Phase:
    name: str
    seconds: float
    instruction: str
    expected: set = field(default_factory=set)  # PASS if any of these fires in-window
    tolerated: set = field(default_factory=set)  # never counted as false positives here


# Timings assume the mac_max defaults (drowsiness 1.5s, eyes-off-road 2s, eyes-down 1.5s,
# phone_use_while_distracted 1.5s, VLM confirm 2 x 4s). Each phase repeats its behavior so
# a single mistimed attempt doesn't fail the phase.
PHASES = [
    Phase(
        "attentive_baseline",
        20,
        "Look straight ahead as if driving. Blink normally. Do nothing else.",
        expected=set(),
    ),
    Phase(
        "sustained_eye_closure",
        16,
        "Close your eyes fully for about 3 seconds, open, then repeat once more.",
        expected={"drowsiness", "microsleep_warning", "vlm_confirmed_drowsiness"},
    ),
    Phase(
        "look_away_sustained",
        16,
        "Turn your head and eyes far to one side (like staring out the window) for ~4s, twice.",
        expected={"eyes_off_road", "excessive_glancing", "vlm_confirmed_distraction"},
    ),
    Phase(
        "eyes_down",
        14,
        "Look down at your lap for about 3 seconds, twice.",
        expected={"eyes_down"},
        tolerated={"eyes_off_road", "excessive_glancing"},
    ),
    Phase(
        "phone_in_hand",
        16,
        "Pick up your phone, hold it up in your hand, and look at it.",
        expected={"phone_in_hand", "phone_use_while_distracted", "vlm_confirmed_phone_use"},
        tolerated={"eyes_down", "eyes_off_road", "excessive_glancing", "vlm_confirmed_distraction"},
    ),
    Phase(
        "mounted_phone_control",
        16,
        "Prop the phone up in view (don't hold it), keep your eyes straight ahead.",
        expected=set(),  # false-positive control: a visible-but-unused phone must NOT alert
    ),
    Phase(
        "drinking",
        16,
        "Take repeated sips from a cup or bottle (bring it fully to your mouth each time).",
        expected={"eating_drinking", "vlm_confirmed_eating"},
        tolerated={"phone_in_hand", "eyes_down"},
    ),
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _countdown(seconds: float, label: str) -> None:
    remaining = int(seconds)
    while remaining > 0:
        print(f"\r  {label}: {remaining:3d}s ", end="", flush=True)
        time.sleep(1)
        remaining -= 1
    print("\r" + " " * 40 + "\r", end="", flush=True)


def run(out_dir: Path) -> Path:
    print("Accuracy protocol — make sure `make dev-max` is running and calibrated in another terminal.")
    print(f"{len(PHASES)} phases, ~{int(sum(p.seconds for p in PHASES) + LEAD_IN_S * len(PHASES))}s total. Ctrl+C aborts.\n")
    records = []
    for i, phase in enumerate(PHASES, 1):
        print(f"[{i}/{len(PHASES)}] NEXT: {phase.instruction}")
        _countdown(LEAD_IN_S, "starting in")
        start = _utcnow()
        print(f"  >>> GO: {phase.instruction}")
        _countdown(phase.seconds, "keep going")
        end = _utcnow()
        print(f"  done ({phase.name})\n")
        records.append(
            {
                "name": phase.name,
                "instruction": phase.instruction,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "expected": sorted(phase.expected),
                "tolerated": sorted(phase.tolerated),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"protocol_{_utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({"created_at": _utcnow().isoformat(), "grace_s": GRACE_S, "phases": records}, indent=2))
    print(f"Protocol recorded: {out_path}")
    print(f"Grade it with:\n  python benchmarks/accuracy_protocol.py grade --protocol {out_path}")
    return out_path


def _load_events(events_path: Path, t_min: datetime, t_max: datetime) -> list[dict]:
    events = []
    with events_path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["timestamp"])
            except (KeyError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if t_min <= ts <= t_max:
                events.append({"ts": ts, "type": row["type"], "signals": row.get("contributing_signals", "")})
    return events


def grade(protocol_path: Path, events_path: Path) -> int:
    protocol = json.loads(protocol_path.read_text())
    grace_s = float(protocol.get("grace_s", GRACE_S))
    phases = protocol["phases"]
    for p in phases:
        p["start_dt"] = datetime.fromisoformat(p["start"])
        p["end_dt"] = datetime.fromisoformat(p["end"])

    t_min = phases[0]["start_dt"]
    t_max = max(p["end_dt"] for p in phases)
    from datetime import timedelta

    events = _load_events(events_path, t_min - timedelta(seconds=2), t_max + timedelta(seconds=grace_s))
    alertable = [e for e in events if e["type"] in ALERT_EVENT_TYPES_MAC_MAX]

    results, false_positives = [], []
    for p in phases:
        window_end = p["end_dt"] + timedelta(seconds=grace_s)
        in_window = [e for e in alertable if p["start_dt"] <= e["ts"] <= window_end]
        hits = sorted({e["type"] for e in in_window} & set(p["expected"]))
        unexpected = [e for e in in_window if e["type"] not in set(p["expected"]) | set(p["tolerated"])]
        # An event inside two overlapping windows (grace spillover) is only a FP if it's
        # unexpected in BOTH — checked globally below via first-match attribution.
        results.append({"phase": p, "hits": hits, "unexpected": unexpected, "in_window": in_window})

    # Global false-positive pass: an alertable event is a FP only if NO phase whose
    # extended window contains it lists its type as expected or tolerated.
    for e in alertable:
        ok = False
        for p in phases:
            if p["start_dt"] <= e["ts"] <= p["end_dt"] + timedelta(seconds=grace_s):
                if e["type"] in set(p["expected"]) | set(p["tolerated"]):
                    ok = True
                    break
        if not ok:
            false_positives.append(e)

    print(f"\n=== Accuracy report ({protocol_path.name}) ===\n")
    expected_phases = [r for r in results if r["phase"]["expected"]]
    passed = 0
    for r in results:
        p = r["phase"]
        if p["expected"]:
            status = "PASS" if r["hits"] else "MISS"
            passed += bool(r["hits"])
            detail = ", ".join(r["hits"]) if r["hits"] else f"expected one of: {', '.join(p['expected'])}"
        else:
            n_bad = len([e for e in false_positives if p["start_dt"] <= e["ts"] <= p["end_dt"] + timedelta(seconds=grace_s)])
            status = "CLEAN" if n_bad == 0 else f"{n_bad} FALSE ALARM(S)"
            detail = "no alerts expected"
        print(f"  {p['name']:24s} {status:18s} {detail}")

    recall = passed / len(expected_phases) if expected_phases else 1.0
    print(f"\n  Recall: {passed}/{len(expected_phases)} expected-behavior phases detected ({recall:.0%})")
    print(f"  False positives: {len(false_positives)} alertable event(s) fired where nothing was expected")
    for e in false_positives:
        print(f"    - {e['ts'].isoformat()}  {e['type']}  ({e['signals']})")

    report_path = protocol_path.with_name(protocol_path.stem + "_report.json")
    report_path.write_text(
        json.dumps(
            {
                "protocol": protocol_path.name,
                "recall": recall,
                "phases": [
                    {"name": r["phase"]["name"], "pass": bool(r["hits"]) if r["phase"]["expected"] else None, "hits": r["hits"]}
                    for r in results
                ],
                "false_positives": [{"ts": e["ts"].isoformat(), "type": e["type"]} for e in false_positives],
            },
            indent=2,
        )
    )
    print(f"\n  Report written: {report_path}")
    return 0 if passed == len(expected_phases) and not false_positives else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="walk through the timed scenario script and record phase timestamps")
    run_p.add_argument("--out", default="benchmarks/results", help="directory for the protocol JSON")
    grade_p = sub.add_parser("grade", help="score data/events.csv against a recorded protocol")
    grade_p.add_argument("--protocol", required=True, help="protocol JSON written by `run`")
    grade_p.add_argument("--events", default="data/events.csv", help="events CSV to grade")
    args = parser.parse_args()

    if args.command == "run":
        run(Path(args.out))
        return 0
    return grade(Path(args.protocol), Path(args.events))


if __name__ == "__main__":
    sys.exit(main())
