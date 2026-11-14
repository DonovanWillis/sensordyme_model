#!/usr/bin/env python3
"""Cued stationary in-cabin validation session: no video, event-level scores (M10.1).

The participant performs a timed script of cues IN A PARKED VEHICLE while the
device runs its normal service; this tool prints each cue, records when it ran,
polls the device's own health for which alerts it could evaluate, and at the end
reads the device's event log. Nothing else is captured: NO frames, NO video, no
audio. The ground truth is the cue schedule, minus the cues the operator voids
(not performed as cued), and it is scored by evaluation/event_metrics.py against
the scenario matrix -- recall and latency per alert, false alerts (and per hour),
alerts on legitimate behaviour (mirror glances), abstentions (the device said it
could not evaluate), all behind the matrix's power check, so one session prints
INSUFFICIENT DATA, never a claim.

SAFETY (evaluation/schemas/incabin_scenarios_v1.yaml `safety`, in full): parked,
parking brake on; consenting adults only, no children in the vehicle; drowsiness
is ACTED; an operator outside the driver's seat runs it; anyone may stop at any
time (Ctrl-C keeps everything recorded so far). Never while driving; never on a
public road.

    V=~/hwval/sensordyme_model/scripts/hw_validation
    python3 $V/cabin_session.py plan --preset core --out ~/hwval/cabin/S01_day     # print the operator sheet
    python3 $V/cabin_session.py run --plan ~/hwval/cabin/S01_day \\
        --subject S01 --session S01-2026-11-16-a --vehicle V01 --seat driver \\
        --lighting day --eyewear none --occupancy driver_only --mount dash_centre
    python3 $V/cabin_session.py score --session ~/hwval/cabin/S01_day [--void 7,12]
    python3 $V/cabin_session.py run --plan DIR ... --fake   # dry run, no device, instant; SIMULATED

`run` talks to the service on this device (default http://127.0.0.1:8899): run it
ON the Pi so the cue clock and the event log share one clock. With a dashboard
token configured, export it as SENSORDYME_TOKEN first; it is sent as the
X-Device-Token header and never printed or written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

CUES = REPO / "evaluation" / "schemas" / "cued_session_v1.yaml"
SESSION_FORMAT = "sensordyme.incabin.cued_session_run/1"
DEFAULT_URL = "http://127.0.0.1:8899"


# ------------------------------------------------------------------ plan
def load_cues(path: Path = CUES) -> dict:
    return yaml.safe_load(path.read_text())


def build_plan(block_ids: list[str], cues: Optional[dict] = None) -> list[dict]:
    """The flat, numbered step list for these blocks (repeats expanded)."""
    cues = cues or load_cues()
    steps: list[dict] = []
    for bid in block_ids:
        block = cues["blocks"].get(bid)
        if block is None:
            raise SystemExit(f"unknown block {bid!r} (known: {', '.join(cues['blocks'])})")
        for rep in range(int(block.get("repeat", 1))):
            for step in block["steps"]:
                steps.append({"n": len(steps) + 1, "block": bid, "cell": block["cell"], "rep": rep + 1,
                              "say": step["say"], "s": float(step["s"]), "event": step.get("event"),
                              "attrs": dict(step.get("attrs") or {}), "ignore": block.get("ignore"),
                              "occupancy": block.get("occupancy")})
    return steps


def operator_sheet(steps: list[dict], blocks: list[str], cues: dict) -> str:
    total = sum(s["s"] for s in steps)
    lines = [f"# Cued session: {', '.join(blocks)}", "",
             (f"{len(steps)} cues, {total / 60:.1f} min. Parked, parking brake on, consenting adults only, "
              "acted drowsiness only. Anyone may stop at any time."), "",
             "| # | at | for | cue | ground truth |", "|---|---|---|---|---|"]
    t = 0.0
    for s in steps:
        truth = s["event"] or ("(not scored: " + s["ignore"] + ")" if s["ignore"] else "attentive")
        lines.append(f"| {s['n']} | {t:6.1f} s | {s['s']:g} s | {s['say']} | {truth} |")
        t += s["s"]
    lines += ["", "After the session: list the numbers of any cue NOT performed as cued, for `score --void`."]
    for bid in blocks:
        block = cues["blocks"][bid]
        lines.append(f"- **{bid}** ({block['cell']}): {block['title']}")
    return "\n".join(lines) + "\n"


def cmd_plan(args) -> int:
    cues = load_cues()
    blocks = cues["presets"][args.preset] if args.preset else [b.strip() for b in args.blocks.split(",") if b.strip()]
    steps = build_plan(blocks, cues)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps({"blocks": blocks, "steps": steps,
                                                    "cues_version": cues["version"]}, indent=1) + "\n")
    sheet = operator_sheet(steps, blocks, cues)
    (args.out / "operator_sheet.md").write_text(sheet)
    print(sheet)
    print(f"plan written to {args.out / 'plan.json'}")
    return 0


# ------------------------------------------------------------------ device access
class Device:
    """The running service's own API, read-only: /api/health and /api/history."""

    def __init__(self, url: str, token: Optional[str]) -> None:
        if not url.startswith(("http://", "https://")):
            raise SystemExit(f"--url must be http(s)://, got {url!r}")
        self.url, self._token = url.rstrip("/"), token

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(self.url + path)  # noqa: S310 - http(s) checked in __init__
        if self._token:
            req.add_header("X-Device-Token", self._token)
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310 - the operator's own device URL
            return json.loads(resp.read().decode())

    def health(self) -> dict:
        return self._get("/api/health")

    def events(self) -> list[dict]:
        return self._get("/api/history?limit=2000").get("events", [])


def unavailable_alerts(health: dict, alerts: frozenset) -> set[str]:
    """Alert types the device said it could not evaluate right now."""
    mon = health.get("monitoring") if isinstance(health.get("monitoring"), dict) else health
    state = mon.get("state")
    if state not in ("ready", "degraded"):
        return set(alerts)  # starting / failed / stopped: nothing is evaluated
    out: set[str] = set()
    for policy in mon.get("policies") or []:
        out |= set(policy.get("unavailable_events") or ()) & set(alerts)
    return out


@dataclass
class FakeDevice:
    """A stand-in for --fake: raises the alert each positive cue calls for, after its
    configured sustain time plus a fixed lag, and nothing else. SIMULATED."""

    sustain: dict[str, float]
    lag_s: float = 0.4

    def alert_for(self, step: dict) -> Optional[str]:
        kind, attrs = step["event"], step["attrs"]
        if kind == "eyes_off_road":
            return "eyes_down" if attrs.get("target") in ("lap", "mounted_phone") else "distraction"
        if kind == "long_closure":
            return "drowsiness"
        if kind == "phone_use":
            return "phone_use"
        if kind == "driver_absent":
            return "attention_lost"
        return None

    def events_for(self, step: dict, start: float) -> list[tuple[float, str]]:
        alert = self.alert_for(step)
        if alert is None:
            return []
        at = self.sustain.get(alert, 2.0) + self.lag_s
        return [(start + at, alert)] if at <= step["s"] + 2.0 else []


# ------------------------------------------------------------------ run
def _now_wall() -> float:
    return time.time()


def cmd_run(args) -> int:
    from evaluation import scenarios

    plan = json.loads((args.plan / "plan.json").read_text())
    steps = plan["steps"]
    matrix = scenarios.load()
    module_cfg = (yaml.safe_load((REPO / "config.yaml").read_text())["modules"] or {}).get(matrix.module, {})
    sustain = matrix.sustain_seconds(module_cfg)
    meta = {k: getattr(args, k) for k in ("subject", "session", "vehicle", "seat", "lighting", "eyewear",
                                          "occupancy", "mount", "notes")}
    device: Optional[Device] = None
    fake = FakeDevice(sustain) if args.fake else None
    health0: dict = {}
    if not args.fake:
        device = Device(args.url, os.environ.get("SENSORDYME_TOKEN") or None)
        try:
            health0 = device.health()
        except OSError as exc:
            print(f"BLOCKED: the service does not answer at {args.url} ({exc}); start it, then rerun", file=sys.stderr)
            return 2
        mon = health0.get("monitoring") or {}
        print(f"device monitoring: {mon.get('state')} {mon.get('reasons') or ''}")
        if mon.get("state") not in ("ready", "degraded"):
            print("BLOCKED: the device is not monitoring; nothing could be measured", file=sys.stderr)
            return 2
    print(f"{'SIMULATED (--fake): no device, instant clock' if args.fake else 'LIVE'}: {len(steps)} cues. "
          "Ctrl-C stops and keeps what was recorded.")
    t0 = _now_wall()
    virtual = 0.0
    record: list[dict] = []
    unavailable: dict[str, list[list[float]]] = {}
    open_spans: dict[str, float] = {}
    fake_events: list[tuple[float, str]] = []
    stopped_early = False
    try:
        for step in steps:
            start = virtual if args.fake else _now_wall() - t0
            print(f"\a[{step['n']:3d}] {step['say']}  ({step['s']:g} s)", flush=True)
            if args.fake:
                assert fake is not None
                fake_events += fake.events_for(step, start)
                virtual += step["s"]
                end = virtual
            else:
                assert device is not None
                deadline = time.monotonic() + step["s"]
                while (left := deadline - time.monotonic()) > 0:
                    try:
                        bad = unavailable_alerts(device.health(), matrix.alerts)
                    except OSError:
                        bad = set(matrix.alerts)  # the service did not answer: nothing was evaluated
                    t = _now_wall() - t0
                    for a in matrix.alerts:
                        if a in bad and a not in open_spans:
                            open_spans[a] = t
                        elif a not in bad and a in open_spans:
                            unavailable.setdefault(a, []).append([open_spans.pop(a), t])
                    time.sleep(min(1.0, max(0.0, left)))
                end = _now_wall() - t0
            record.append({**step, "start_s": round(start, 3), "end_s": round(end, 3)})
    except KeyboardInterrupt:
        stopped_early = True
        print("\nstopped by the operator; keeping what was recorded")
    end_s = virtual if args.fake else _now_wall() - t0
    for a, since in open_spans.items():
        unavailable.setdefault(a, []).append([since, end_s])
    if args.fake:
        predictions = [[round(t, 3), a] for t, a in fake_events if t <= end_s]
        raw_events: list[dict] = []
    else:
        assert device is not None
        raw_events = device.events()
        predictions = []
        for e in raw_events:
            try:
                t = datetime.fromisoformat(e["timestamp"]).timestamp() - t0
            except (KeyError, ValueError):
                continue
            if 0.0 <= t <= end_s:
                predictions.append([round(t, 3), e["type"]])
    session = {
        "format": SESSION_FORMAT, "simulated": bool(args.fake), "stopped_early": stopped_early,
        "started_utc": datetime.fromtimestamp(t0, UTC).isoformat(), "duration_s": round(end_s, 3),
        "meta": meta, "plan_blocks": plan["blocks"], "cues_version": plan.get("cues_version"),
        "matrix": {"version": matrix.version, "sha256": matrix.sha256},
        "device": {"url": None if args.fake else args.url,
                   "versions": (health0.get("monitoring") or {}).get("versions"),
                   "model": (health0.get("monitoring") or {}).get("model")},
        "steps": record, "predictions": sorted(predictions),
        "unavailable": dict(unavailable),
        "note": "no frames, video or audio were recorded; ground truth = the cue schedule",
    }
    out = args.plan / ("session_SIMULATED.json" if args.fake else "session.json")
    out.write_text(json.dumps(session, indent=1) + "\n")
    print(f"session written to {out}: {len(record)} cues, {len(predictions)} device events")
    return 0


# ------------------------------------------------------------------ score
SESSION_AXES = ("lighting", "eyewear", "occupancy", "mount")


def block_clips(session: dict, void: set[int], matrix=None) -> list[dict]:
    """The session as annotation v2 documents, ONE CLIP PER BLOCK RUN.

    The matrix attaches tags to a clip, and one session spans cells whose tags
    differ (a baseline is phone: none, a handheld block phone: handheld_hand), so
    each block is its own clip: the session's own axes (lighting, eyewear,
    occupancy, mount), plus whatever else its cell requires. Times are relative to
    the block's start; `offset_s` maps them back to the session clock. A block
    whose cell needs a session axis the session does not have is refused, with why.
    Calibration blocks are not scored at all."""
    from evaluation import scenarios

    matrix = matrix or scenarios.load()
    meta = session["meta"]
    runs: list[list[dict]] = []
    for step in session["steps"]:
        if runs and runs[-1][0]["block"] == step["block"] and step["n"] == runs[-1][-1]["n"] + 1:
            runs[-1].append(step)
        else:
            runs.append([step])
    clips = []
    for k, run in enumerate(runs):
        if run[0].get("ignore"):
            continue
        cell = matrix.cells[run[0]["cell"]]
        tags = {axis: meta[axis] for axis in SESSION_AXES}
        for axis, allowed in cell.tags.items():
            if axis in SESSION_AXES:
                if tags[axis] not in allowed:
                    raise SystemExit(f"block {run[0]['block']} counts toward {cell.id}, which needs {axis} in "
                                     f"{list(allowed)}; this session is {axis}={tags[axis]}")
            else:
                tags[axis] = allowed[0]
        offset = run[0]["start_s"]
        duration = max(run[-1]["end_s"] - offset, 0.01)
        events, uncertain = [], []
        for s in run:
            span = {"start_s": round(s["start_s"] - offset, 3),
                    "end_s": round(max(s["end_s"], s["start_s"] + 0.01) - offset, 3)}
            if s["n"] in void:
                uncertain.append({**span, "reason": "other"})
            elif s.get("event"):
                events.append({"type": s["event"], **span, "attrs": {**s["attrs"], "cue": s["n"]}})
        clips.append({"offset_s": offset, "duration_s": duration, "doc": {
            "schema": "sensordyme.incabin.annotation", "version": 2,
            "item": f"{meta['session']}#{k + 1:02d}-{run[0]['block']}", "annotator": "cue-schedule",
            "clip": {"subject": meta["subject"], "session": meta["session"], "vehicle": meta["vehicle"],
                     "recorded_on": session["started_utc"][:10], "seat": meta["seat"],
                     "duration_s": duration, "mount": meta["mount"]},
            "scenarios": [cell.id], "tags": tags, "events": events, "ignore": [], "uncertain": uncertain}})
    return clips


def score_session(session: dict, void: set[int]) -> dict:
    from evaluation import annotations_v2 as av2
    from evaluation import event_metrics as em
    from evaluation import scenarios
    from evaluation.annotations import Span

    matrix = scenarios.load()
    module_cfg = (yaml.safe_load((REPO / "config.yaml").read_text())["modules"] or {}).get(matrix.module, {})
    sustain = matrix.sustain_seconds(module_cfg)
    total = em.Scores()
    for clip in block_clips(session, void, matrix):
        annotation, problems = av2.parse(clip["doc"], matrix=matrix)
        if annotation is None:
            raise SystemExit(f"{clip['doc']['item']} is not a valid annotation: " + "; ".join(problems))
        lo, d = clip["offset_s"], clip["duration_s"]
        unavailable = {a: [Span(max(0.0, float(x) - lo), min(d, float(y) - lo)) for x, y in spans
                           if float(y) > lo and float(x) < lo + d]
                       for a, spans in (session.get("unavailable") or {}).items()}
        predictions = [(float(t) - lo, str(a)) for t, a in session["predictions"] if lo <= float(t) <= lo + d]
        total = total.add(em.score_clip(em.ClipInput(
            item=clip["doc"]["item"], subject=session["meta"]["subject"], annotation=annotation, duration_s=d,
            predictions=predictions, unavailable=unavailable), matrix, sustain))
    scores = total
    summary = em.summarise({session["meta"]["subject"]: scores}, matrix=matrix, minimums=matrix.overall,
                           include_diagnostics=True, bootstrap_n=200)
    per_alert = {}
    for a, t in sorted(scores.alerts.items()):
        per_alert[a] = {"raised": t.predictions, "first_catches": t.true_first, "repeats": t.repeats,
                        "false_alerts": t.false_alerts, "false_on_legitimate_behaviour": t.false_on_negative,
                        "boundary": t.boundary, "exposure_min": round(t.exposure_s / 60, 2),
                        "false_per_hour": round(t.false_alerts / t.hours, 2) if t.hours > 0 else None,
                        "unavailable_s": None if t.unavailable_s is None else round(t.unavailable_s, 1)}
    per_event = {}
    for e, d in sorted(scores.detection.items()):
        lat = sorted(round(x, 2) for x in d.latencies)
        per_event[e] = {"cued": d.events, "caught": d.caught, "abstained": d.abstained,
                        "silent_misses": d.events - d.caught - d.abstained,
                        "latency_from_cue_s": {"min": lat[0], "median": lat[len(lat) // 2], "max": lat[-1]}
                        if lat else None,
                        "latency_minus_sustain_s": sorted(round(x, 2) for x in d.excess)}
    negatives = {k: {"cued": v.events, "alerted": v.alerted} for k, v in sorted(scores.negatives.items())}
    return {"simulated": session.get("simulated", False), "session": session["meta"]["session"],
            "voided_cues": sorted(void), "per_event": per_event, "negatives": negatives, "alerts": per_alert,
            "unscored_device_events": scores.unscored_types, "power_checked_summary": summary,
            "caveats": ["latency is from the CUE and includes the participant's reaction time",
                        "one session never meets the matrix minimums: diagnostics, not claims",
                        "ground truth is the cue schedule (minus voided cues), not an observed annotation",
                        "alerts during calibration blocks and between blocks are not scored"]}


def cmd_score(args) -> int:
    path = args.session / "session.json"
    if not path.exists():
        path = args.session / "session_SIMULATED.json"
    session = json.loads(path.read_text())
    void = {int(x) for x in (args.void or "").split(",") if x.strip()}
    result = score_session(session, void)
    out = args.session / ("score_SIMULATED.json" if result["simulated"] else "score.json")
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    tag = "SIMULATED -- NOT EVIDENCE. " if result["simulated"] else ""
    print(f"{tag}session {result['session']}: diagnostics only (one session never meets the power check)")
    for e, d in result["per_event"].items():
        lat = d["latency_from_cue_s"]
        print(f"  {e:15s} caught {d['caught']}/{d['cued']}  abstained {d['abstained']}  silent misses "
              f"{d['silent_misses']}  latency from cue {lat}")
    for e, d in result["negatives"].items():
        print(f"  {e:20s} (must not alert) alerted {d['alerted']}/{d['cued']}")
    for a, d in result["alerts"].items():
        print(f"  alert {a:15s} raised {d['raised']}  false {d['false_alerts']} "
              f"({d['false_per_hour']}/h over {d['exposure_min']} min)  unavailable {d['unavailable_s']} s")
    print(f"written to {out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="build the cue list and print the operator sheet")
    group = pl.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=sorted(load_cues()["presets"]))
    group.add_argument("--blocks", help="comma list of block ids")
    pl.add_argument("--out", type=Path, required=True)
    rn = sub.add_parser("run", help="run the cues against the device (parked vehicle only)")
    rn.add_argument("--plan", type=Path, required=True, help="the plan directory")
    for key in ("subject", "session", "vehicle"):
        rn.add_argument(f"--{key}", required=True)
    rn.add_argument("--seat", default="driver")
    rn.add_argument("--lighting", required=True, choices=["day", "dusk", "night_ir", "direct_sun_glare"])
    rn.add_argument("--eyewear", required=True, choices=["none", "clear", "sunglasses_ir_pass", "sunglasses_ir_block"])
    rn.add_argument("--occupancy", required=True)
    rn.add_argument("--mount", required=True, choices=["dash_centre", "mirror_base", "a_pillar", "column"])
    rn.add_argument("--notes", default="", help="conditions to record, e.g. 'mount nudged ~5 deg yaw, no recal'")
    rn.add_argument("--url", default=DEFAULT_URL)
    rn.add_argument("--fake", action="store_true", help="dry run against a simulated device; SIMULATED")
    sc = sub.add_parser("score", help="score a recorded session (diagnostics, behind the power check)")
    sc.add_argument("--session", type=Path, required=True)
    sc.add_argument("--void", default="", help="cue numbers NOT performed as cued, e.g. 7,12")
    args = p.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
