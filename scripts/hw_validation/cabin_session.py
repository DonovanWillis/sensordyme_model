#!/usr/bin/env python3
"""Cued stationary in-cabin validation session: no video, event-level scores (M10.1, M11).

The participant performs a timed script of cues IN A PARKED VEHICLE while the
device runs its normal service; this tool prints each cue, records when it ran,
polls the device's own health for which alerts it could evaluate, and at the end
reads the device's event log. Nothing else is captured: NO frames, NO video, no
audio. Scoring goes through evaluation/event_metrics.py against the scenario
matrix -- recall and latency per alert, false alerts (and per hour), alerts on
legitimate behaviour (mirror glances), abstentions (the device said it could not
evaluate), all behind the matrix's power check.

TWO EVIDENCE LEVELS (M11). A cue is a REQUEST, not proof that the participant
performed it, so the cue schedule is never accuracy ground truth:

  * `score` alone -> evidence_level "diagnostic_cue_schedule": what the device did
    while cues were read out. Useful to find defects; NEVER an accuracy figure.
  * `score --labels FILE` -> evidence_level "verified_labels": the behaviour an
    INDEPENDENT labeller observed (format sensordyme.incabin.verified_labels/1:
    a live observer with `observe`, or a blinded annotator reviewing a consented,
    separately recorded video), on a clock synchronised with the device's, with
    consent recorded by reference. Cues then only cut the session into blocks and
    are cross-checked against the labels (cues not performed, behaviour that was
    not cued). Accuracy may be CLAIMED only from verified labels AND a passing
    power check across many sessions; one session is diagnostics, never a study.

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
    # independent observer, in a SECOND ssh session on the Pi (same clock as the device):
    python3 $V/cabin_session.py observe --session ~/hwval/cabin/S01_day --annotator O1 \\
        --method live_observer --clock same_host --consent-ref <consent record id>
    python3 $V/cabin_session.py score --session ~/hwval/cabin/S01_day --labels ~/hwval/cabin/S01_day/labels_O1.json

`run` talks to the service on this device (default http://127.0.0.1:8899): run it
ON the Pi so the cue clock and the event log share one clock. With a dashboard
token configured, export it as SENSORDYME_TOKEN first; it is sent as the
X-Device-Token header and never printed or written.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
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
LABELS_FORMAT = "sensordyme.incabin.verified_labels/1"
DEFAULT_URL = "http://127.0.0.1:8899"
#: how a label set may be produced. A cue schedule is not one of them.
LABEL_METHODS = ("live_observer", "video_review")
CLOCK_METHODS = ("same_host", "ntp", "video_clock_overlay")
#: labels whose clock is less certain than this cannot support latency or event matching.
MAX_CLOCK_UNCERTAINTY_S = 0.5
#: observer keys -> annotation v2 event types (the cued_session_v1 vocabulary); `u` marks "cannot see".
OBSERVER_KEYS = {"c": "long_closure", "o": "eyes_off_road", "p": "phone_use", "a": "driver_absent",
                 "l": "glance_mirror_left", "r": "glance_mirror_right", "b": "glance_mirror_rear",
                 "k": "glance_cluster", "f": "passenger_phone_use", "u": "UNCERTAIN"}
EVIDENCE_CUE = "diagnostic_cue_schedule"
EVIDENCE_VERIFIED = "verified_labels"


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


# ------------------------------------------------------------------ verified labels (M11)
def _known_event_types() -> set[str]:
    from evaluation import annotations_v2 as av2

    return set(av2.vocabulary()["events"])


def _uncertain_reasons() -> set[str]:
    from evaluation import annotations_v2 as av2

    return set(av2.vocabulary()["uncertain_reasons"])


def load_labels(doc: dict, session: dict) -> tuple[Optional[dict], list[str]]:
    """Validate an INDEPENDENT label set and put it on the session clock. Returns (labels, problems).

    Refused (labels None) unless: the format is right; a named labeller who is not the
    cue schedule and not the operator; a real labelling method (live observer, or video
    review); a synchronised clock whose uncertainty is at most MAX_CLOCK_UNCERTAINTY_S;
    a consent record reference; known event types with valid UTC spans."""
    problems: list[str] = []
    if doc.get("format") != LABELS_FORMAT:
        problems.append(f"format must be {LABELS_FORMAT!r}")
    prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
    annotator = str(prov.get("annotator") or "").strip()
    if not annotator or annotator.lower().replace("_", "-") in ("cue-schedule", "operator"):
        problems.append("provenance.annotator must be the independent labeller's pseudonym (not the cue schedule)")
    if prov.get("method") not in LABEL_METHODS:
        problems.append(f"provenance.method must be one of {list(LABEL_METHODS)}; a cue schedule is not a label")
    if prov.get("independent_of_operator") is not True:
        problems.append("provenance.independent_of_operator must be true: whoever reads the cues cannot label them")
    clock = prov.get("clock") if isinstance(prov.get("clock"), dict) else {}
    if clock.get("method") not in CLOCK_METHODS:
        problems.append(f"provenance.clock.method must be one of {list(CLOCK_METHODS)}")
    if clock.get("synchronized") is not True:
        problems.append("provenance.clock.synchronized must be true: labels and device events need one clock")
    unc = clock.get("uncertainty_s")
    if not isinstance(unc, (int, float)) or isinstance(unc, bool) or not 0 <= unc <= MAX_CLOCK_UNCERTAINTY_S:
        problems.append(f"provenance.clock.uncertainty_s must be a number in [0, {MAX_CLOCK_UNCERTAINTY_S}] s")
    offset = clock.get("offset_s", 0.0)
    if not isinstance(offset, (int, float)) or isinstance(offset, bool):
        problems.append("provenance.clock.offset_s (labeller clock minus device clock) must be a number")
        offset = 0.0
    if not str(prov.get("consent_ref") or "").strip():
        problems.append("provenance.consent_ref is required: the consent record's id (never a name)")
    if doc.get("session") is not None and doc.get("session") != session["meta"]["session"]:
        problems.append(f"labels are for session {doc.get('session')!r}; this session is {session['meta']['session']!r}")
    t0 = datetime.fromisoformat(session["started_utc"]).timestamp()
    known, reasons = _known_event_types(), _uncertain_reasons()

    def span(item: dict, where: str) -> Optional[tuple[float, float]]:
        try:
            a = datetime.fromisoformat(item["start_utc"]).timestamp() - t0 - float(offset)
            b = datetime.fromisoformat(item["end_utc"]).timestamp() - t0 - float(offset)
        except (KeyError, TypeError, ValueError):
            problems.append(f"{where} needs start_utc and end_utc (ISO 8601 with a UTC offset)")
            return None
        if b <= a:
            problems.append(f"{where} ends before it starts")
            return None
        return round(a, 3), round(b, 3)

    events: list[dict] = []
    for i, lab in enumerate(doc.get("labels") or []):
        if lab.get("type") not in known:
            problems.append(f"labels[{i}].type {lab.get('type')!r} is not an annotation v2 event type")
            continue
        sp_ = span(lab, f"labels[{i}]")
        if sp_ is not None:
            events.append({"type": lab["type"], "start_s": sp_[0], "end_s": sp_[1],
                           "attrs": dict(lab.get("attrs") or {})})
    uncertain: list[dict] = []
    for i, u in enumerate(doc.get("uncertain") or []):
        sp_ = span(u, f"uncertain[{i}]")
        if sp_ is not None:
            reason = u.get("reason") if u.get("reason") in reasons else "other"
            uncertain.append({"start_s": sp_[0], "end_s": sp_[1], "reason": reason,
                              **({"types": list(u["types"])} if u.get("types") else {})})
    if not events and not uncertain and not problems:
        problems.append("the label set is empty: an observer who saw nothing must still say so per behaviour")
    if problems:
        return None, problems
    return {"annotator": annotator, "provenance": prov, "events": events, "uncertain": uncertain}, []


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def _matching_cue(event: dict, steps: list[dict], slack_s: float = 2.0) -> Optional[dict]:
    """The cue this labelled behaviour answers: same type, overlapping the cue (+ reaction slack)."""
    return next((s for s in steps if s.get("event") == event["type"] and not s.get("ignore")
                 and _overlaps(event["start_s"], event["end_s"], s["start_s"] - 1.0, s["end_s"] + slack_s)), None)


def _labelled_in_window(labels: dict, run: list[dict], offset: float, duration: float) -> tuple[list, list]:
    lo, hi = offset, offset + duration
    events: list[dict] = []
    for e in labels["events"]:
        a, b = max(e["start_s"], lo), min(e["end_s"], hi)
        if b <= a:
            continue
        cue = _matching_cue(e, run)
        attrs = {**(cue["attrs"] if cue else {}), **e["attrs"], **({"cue": cue["n"]} if cue else {})}
        events.append({"type": e["type"], "start_s": round(a - lo, 3), "end_s": round(b - lo, 3), "attrs": attrs})
    uncertain = [{"start_s": round(max(u["start_s"], lo) - lo, 3), "end_s": round(min(u["end_s"], hi) - lo, 3),
                  "reason": u["reason"], **({"types": u["types"]} if u.get("types") else {})}
                 for u in labels["uncertain"] if min(u["end_s"], hi) > max(u["start_s"], lo)]
    return events, uncertain


def cue_label_agreement(session: dict, labels: dict) -> dict:
    """Did the participant do what was cued? Cued behaviours the labeller did not see, and
    behaviour the labeller saw that nobody cued: both are why a cue is not ground truth."""
    cued = [s for s in session["steps"] if s.get("event") and not s.get("ignore")]
    performed = [s["n"] for s in cued if any(
        e["type"] == s["event"] and _overlaps(e["start_s"], e["end_s"], s["start_s"] - 1.0, s["end_s"] + 2.0)
        for e in labels["events"])]
    unscripted = [e for e in labels["events"] if _matching_cue(e, session["steps"]) is None]
    by_type: dict[str, int] = {}
    for e in unscripted:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1
    return {"cued_behaviours": len(cued), "performed_as_cued": len(performed),
            "not_performed_cues": [s["n"] for s in cued if s["n"] not in performed],
            "unscripted_labelled_events": len(unscripted), "unscripted_by_type": by_type}


def _withhold_claims(node: Any, why: str) -> Any:
    """Every power-checked `claim` in a summary -> None, with the reason (cue-schedule labels)."""
    if isinstance(node, dict):
        out = {k: _withhold_claims(v, why) for k, v in node.items()}
        if "claim" in out and out["claim"] is not None:
            out["claim"], out["claim_withheld"] = None, why
        return out
    if isinstance(node, list):
        return [_withhold_claims(v, why) for v in node]
    return node


def _power_ok(summary: dict) -> bool:
    gated = [m for section in ("detection", "alerts", "negatives") for m in (summary.get(section) or {}).values()]
    return bool(gated) and all(m.get("status") == "ok" for m in gated)


# ------------------------------------------------------------------ score
SESSION_AXES = ("lighting", "eyewear", "occupancy", "mount")


def block_clips(session: dict, void: set[int], matrix=None, labels: Optional[dict] = None) -> list[dict]:
    """The session as annotation v2 documents, ONE CLIP PER BLOCK RUN.

    Without `labels` the events are the CUES (diagnostic only). With `labels` (the
    output of `load_labels`, already on the session clock) the events and the
    uncertain spans are what the independent labeller OBSERVED inside each block's
    window; a labelled event that matches a cue of the same type inherits that
    cue's attributes (e.g. the gaze target), which the labeller confirms by
    labelling it, and anything the labeller saw that was not cued is kept as is.

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
        if labels is None:
            for s in run:
                span = {"start_s": round(s["start_s"] - offset, 3),
                        "end_s": round(max(s["end_s"], s["start_s"] + 0.01) - offset, 3)}
                if s["n"] in void:
                    uncertain.append({**span, "reason": "other"})
                elif s.get("event"):
                    events.append({"type": s["event"], **span, "attrs": {**s["attrs"], "cue": s["n"]}})
        else:
            events, uncertain = _labelled_in_window(labels, run, offset, duration)
        clips.append({"offset_s": offset, "duration_s": duration, "doc": {
            "schema": "sensordyme.incabin.annotation", "version": 2,
            "item": f"{meta['session']}#{k + 1:02d}-{run[0]['block']}",
            "annotator": labels["annotator"] if labels is not None else "cue-schedule",
            "clip": {"subject": meta["subject"], "session": meta["session"], "vehicle": meta["vehicle"],
                     "recorded_on": session["started_utc"][:10], "seat": meta["seat"],
                     "duration_s": duration, "mount": meta["mount"]},
            "scenarios": [cell.id], "tags": tags, "events": events, "ignore": [], "uncertain": uncertain}})
    return clips


def score_session(session: dict, void: set[int], labels: Optional[dict] = None) -> dict:
    from evaluation import annotations_v2 as av2
    from evaluation import event_metrics as em
    from evaluation import scenarios
    from evaluation.annotations import Span

    if labels is not None and void:
        raise SystemExit("--void is for cue-schedule diagnostics; verified labels already record what happened")
    matrix = scenarios.load()
    module_cfg = (yaml.safe_load((REPO / "config.yaml").read_text())["modules"] or {}).get(matrix.module, {})
    sustain = matrix.sustain_seconds(module_cfg)
    total = em.Scores()
    for clip in block_clips(session, void, matrix, labels=labels):
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
    simulated = bool(session.get("simulated", False))
    level = "SIMULATED" if simulated else EVIDENCE_VERIFIED if labels is not None else EVIDENCE_CUE
    caveats = ["one session never meets the matrix minimums: diagnostics, not claims",
               "alerts during calibration blocks and between blocks are not scored"]
    agreement = None
    provenance = None
    if labels is None:
        why = ("cue-schedule labels: a cue is a request, not proof the behaviour happened; accuracy needs "
               "independently verified labels (score --labels)")
        summary = _withhold_claims(summary, why)
        caveats = [why, "latency is from the CUE and includes the participant's reaction time", *caveats]
    else:
        agreement = cue_label_agreement(session, labels)
        prov = labels["provenance"]
        provenance = {"annotator": labels["annotator"], "method": prov.get("method"),
                      "independent_of_operator": prov.get("independent_of_operator"),
                      "blinded_to_device_output": bool(prov.get("blinded_to_device_output")),
                      "clock": prov.get("clock"), "consent_ref": prov.get("consent_ref"),
                      "video": prov.get("video")}
        caveats = ["latency is from the LABELLED onset (the observer's timing, within the stated clock uncertainty)",
                   *caveats]
        if not provenance["blinded_to_device_output"]:
            caveats.append("the labeller could perceive the device's alerts (not blinded): onset and boundary "
                           "judgements may be biased toward the device; double-annotate for claims")
    claim_ok = (level == EVIDENCE_VERIFIED) and _power_ok(summary)
    for d in per_event.values():
        d["labelled" if labels is not None else "cued"] = d.pop("cued")
    return {"simulated": simulated, "session": session["meta"]["session"], "evidence_level": level,
            "accuracy_claim_permitted": claim_ok,
            "accuracy_claim_rule": ("verified independent labels AND every metric past the scenario matrix's "
                                    "power check (subjects, clips, events, hours) across sessions"),
            "label_provenance": provenance, "cue_label_agreement": agreement,
            "voided_cues": sorted(void), "per_event": per_event, "negatives": negatives, "alerts": per_alert,
            "unscored_device_events": scores.unscored_types, "power_checked_summary": summary,
            "caveats": caveats}


def cmd_score(args) -> int:
    path = args.session / "session.json"
    if not path.exists():
        path = args.session / "session_SIMULATED.json"
    session = json.loads(path.read_text())
    void = {int(x) for x in (args.void or "").split(",") if x.strip()}
    labels = None
    if args.labels is not None:
        labels, problems = load_labels(json.loads(args.labels.read_text()), session)
        if labels is None:
            print("BLOCKED: these labels cannot be used as verified ground truth:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 2
    result = score_session(session, void, labels)
    name = "score_SIMULATED.json" if result["simulated"] else \
        f"score_verified_{labels['annotator']}.json" if labels is not None else "score_cue_diagnostic.json"
    out = args.session / name
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    tag = "SIMULATED -- NOT EVIDENCE. " if result["simulated"] else ""
    print(f"{tag}session {result['session']}: evidence level {result['evidence_level']}; "
          f"accuracy claim permitted: {result['accuracy_claim_permitted']}")
    if labels is None:
        print("  CUE-SCHEDULE DIAGNOSTICS: a cue is not proof the behaviour happened; no accuracy figure here")
    else:
        ag = result["cue_label_agreement"]
        print(f"  labeller {labels['annotator']}: {ag['performed_as_cued']}/{ag['cued_behaviours']} cues performed; "
              f"not performed {ag['not_performed_cues']}; unscripted {ag['unscripted_by_type']}")
    truth_key = "labelled" if labels is not None else "cued"
    for e, d in result["per_event"].items():
        lat = d["latency_from_cue_s"]
        print(f"  {e:15s} caught {d['caught']}/{d[truth_key]}  abstained {d['abstained']}  silent misses "
              f"{d['silent_misses']}  latency {lat}")
    for e, d in result["negatives"].items():
        print(f"  {e:20s} (must not alert) alerted {d['alerted']}/{d['cued']}")
    for a, d in result["alerts"].items():
        print(f"  alert {a:15s} raised {d['raised']}  false {d['false_alerts']} "
              f"({d['false_per_hour']}/h over {d['exposure_min']} min)  unavailable {d['unavailable_s']} s")
    print(f"written to {out}")
    return 0


# ------------------------------------------------------------------ observe (independent live labeller)
def clock_status(method: str, uncertainty_s: Optional[float] = None) -> dict:
    """What the labeller's clock is, honestly. `same_host`: the observer's terminal runs ON the device
    (a second SSH session), so labels and device events share one clock by construction. `ntp`: another
    machine, synchronised by NTP as reported by timedatectl (read only)."""
    if method == "same_host":
        return {"method": "same_host", "synchronized": True, "offset_s": 0.0,
                "uncertainty_s": 0.15 if uncertainty_s is None else uncertainty_s,
                "note": "observer terminal on the device itself; uncertainty = key-press timing, not clock skew"}
    text = None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        text = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], capture_output=True,
                              text=True, timeout=5, check=False).stdout
    return {"method": "ntp", "synchronized": (text or "").strip() == "yes", "offset_s": 0.0,
            "uncertainty_s": 0.25 if uncertainty_s is None else uncertainty_s,
            "note": "timedatectl NTPSynchronized on the labeller's machine; the device must be NTP-synced too"}


def _terminal_keys():
    """Single key presses from the terminal (cbreak mode, restored on exit)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return
            yield ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def observe(args, keys=None, now=_now_wall) -> int:
    """Toggle-key labeller: first press of a key starts that behaviour, second press ends it."""
    if keys is None:
        if not sys.stdin.isatty():
            print("BLOCKED: observe needs a terminal (single key presses)", file=sys.stderr)
            return 2
        keys = _terminal_keys()
    clock = clock_status(args.clock, args.clock_uncertainty_s)
    if not clock["synchronized"]:
        print("WARNING: this machine's clock is not NTP-synchronised; these labels will be refused at scoring. "
              "Run the observer in a second ssh session ON the device (--clock same_host) instead.", file=sys.stderr)
    session_id = args.session_id
    sess_file = args.session / "session.json"
    if session_id is None and sess_file.exists():
        session_id = json.loads(sess_file.read_text())["meta"]["session"]
    print("Independent observer. Press a key when a behaviour STARTS and the same key when it ENDS:")
    for k, v in OBSERVER_KEYS.items():
        print(f"   {k}  {'cannot see (uncertain)' if v == 'UNCERTAIN' else v}")
    print("   q  finish.  Label what you SEE, not what was cued.", flush=True)
    opened: dict[str, float] = {}
    labels: list[dict] = []
    uncertain: list[dict] = []

    def iso(t: float) -> str:
        return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")

    def close(kind: str, t: float, at_quit: bool = False) -> None:
        start = opened.pop(kind)
        extra = {"closed_at_quit": True} if at_quit else {}
        if kind == "UNCERTAIN":
            uncertain.append({"start_utc": iso(start), "end_utc": iso(t), "reason": "face_not_visible", **extra})
        else:
            labels.append({"type": kind, "start_utc": iso(start), "end_utc": iso(t), "attrs": extra})
        print(f"  [-] {kind} ended ({t - start:.1f} s)", flush=True)

    for key in keys:
        t = now()
        if key == "q":
            break
        kind = OBSERVER_KEYS.get(key)
        if kind is None:
            continue
        if kind in opened:
            close(kind, t)
        else:
            opened[kind] = t
            print(f"  [+] {kind} started", flush=True)
    t_end = now()
    for kind in list(opened):
        close(kind, t_end, at_quit=True)
    doc = {"format": LABELS_FORMAT, "session": session_id,
           "provenance": {"annotator": args.annotator, "method": args.method,
                          "independent_of_operator": bool(args.independent),
                          "blinded_to_device_output": bool(args.blinded), "clock": clock,
                          "consent_ref": args.consent_ref, "video": {"recorded_by_this_tool": False},
                          "tool": "cabin_session.py observe"},
           "labels": sorted(labels, key=lambda x: x["start_utc"]),
           "uncertain": sorted(uncertain, key=lambda x: x["start_utc"])}
    args.session.mkdir(parents=True, exist_ok=True)
    out = args.session / f"labels_{args.annotator}.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(labels)} behaviours, {len(uncertain)} uncertain spans -> {out}")
    if not args.independent:
        print("NOTE: --independent was not given; scoring refuses labels from the person reading the cues.")
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
    sc.add_argument("--void", default="", help="cue-schedule diagnostics only: cue numbers NOT performed, e.g. 7,12")
    sc.add_argument("--labels", type=Path, default=None,
                    help=f"independent labels ({LABELS_FORMAT}); without them the score is diagnostics only")
    ob = sub.add_parser("observe", help="independent observer: label what actually happens (key presses)")
    ob.add_argument("--session", type=Path, required=True, help="the session directory (labels_<annotator>.json)")
    ob.add_argument("--session-id", default=None, help="default: read from session.json when it exists")
    ob.add_argument("--annotator", required=True, help="the observer's pseudonym (never a name)")
    ob.add_argument("--method", choices=["live_observer"], default="live_observer")
    ob.add_argument("--clock", choices=["same_host", "ntp"], required=True)
    ob.add_argument("--clock-uncertainty-s", type=float, default=None)
    ob.add_argument("--consent-ref", required=True, help="the consent record's id (never a name)")
    ob.add_argument("--independent", action="store_true",
                    help="I am NOT the person reading the cues (required for the labels to be accepted)")
    ob.add_argument("--blinded", action="store_true",
                    help="I cannot see or hear the device's alerts (rare live; normal for muted video review)")
    args = p.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "score": cmd_score, "observe": observe}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
