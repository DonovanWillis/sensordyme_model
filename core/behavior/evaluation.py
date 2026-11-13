"""Behavioural evaluation metrics over labelled event traces (M8.4, A6).

Event-level scoring of one decision source (the shipped alerts, or the engine's
shadow verdicts) against labelled windows:

  * a TRUE window ``(type, onset_s, end_s)`` is a true positive when an event of
    that type starts inside ``[onset_s, end_s + grace_s]`` (detection delay =
    event time - onset), else a false negative;
  * every other alert-tier event is a FALSE POSITIVE unless its type is listed
    as ``allowed`` for that scenario (acceptable, not required);
  * precision and recall carry Wilson 95% intervals; the false-alert rate is per
    hour of scripted time; abstention and missing-evidence rates come from the
    engine's own report.

EVIDENCE CLASSES are kept apart and never pooled: ``synthetic`` (scripted
scenarios, labels are the script itself), ``human_labelled`` (none exist for
in-cabin driver behaviour today) and ``model_generated`` (none). A number from
the synthetic class is regression evidence that the logic does what it was
designed to on constructed inputs -- NOT a real-world accuracy.

Pure: standard library only.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Optional

#: Event types that are ALERTS in the shipped product (validated tier).
ALERT_TYPES = frozenset({"drowsiness", "distraction", "eyes_down", "eyes_up", "phone_use", "attention_lost",
                         "fatigue_warning", "frequent_glancing"})
EVIDENCE_CLASSES = ("synthetic", "human_labelled", "model_generated")


def wilson(successes: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    """Wilson score interval for a proportion; None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def score_scenario(events: Iterable[dict], expected: Iterable[tuple[str, float, float]],
                   allowed: Iterable[str] = (), *, alert_types: Iterable[str] = ALERT_TYPES,
                   grace_s: float = 1.0) -> dict:
    """TP/FP/FN and detection delays for one scenario's events ({"t", "type"})."""
    alert_set = set(alert_types)
    allowed_set = set(allowed)
    pending = sorted((e for e in events if e["type"] in alert_set), key=lambda e: e["t"])
    tp, fn, delays, false_positives = 0, 0, [], []
    used: set[int] = set()
    for event_type, onset, end in expected:
        hit = next((i for i, e in enumerate(pending)
                    if i not in used and e["type"] == event_type and onset <= e["t"] <= end + grace_s), None)
        if hit is None:
            fn += 1
        else:
            used.add(hit)
            tp += 1
            delays.append(round(pending[hit]["t"] - onset, 2))
    expected_types = {e[0] for e in expected}
    for i, e in enumerate(pending):
        if i in used:
            continue
        if e["type"] in allowed_set:
            continue
        if e["type"] in expected_types and any(e["type"] == x[0] and x[1] <= e["t"] <= x[2] + grace_s
                                               for x in expected):
            continue  # a second onset inside a true window: a re-trigger, not a false alarm
        false_positives.append(e)
    return {"tp": tp, "fn": fn, "fp": len(false_positives), "delays_s": delays,
            "false_positives": [f"{e['type']}@{e['t']}" for e in false_positives]}


def aggregate(rows: list[dict], durations_s: list[float]) -> dict:
    """Totals with Wilson intervals over scored scenarios."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    delays = [d for r in rows for d in r["delays_s"]]
    hours = sum(durations_s) / 3600.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None, "precision_ci95": wilson(tp, tp + fp),
        "recall": round(tp / (tp + fn), 4) if tp + fn else None, "recall_ci95": wilson(tp, tp + fn),
        "false_alerts_per_hour": round(fp / hours, 2) if hours else None,
        "scripted_hours": round(hours, 4),
        "detection_delay_s": {
            "n": len(delays), "mean": round(statistics.fmean(delays), 3) if delays else None,
            "median": round(statistics.median(delays), 3) if delays else None,
            "max": max(delays) if delays else None,
        },
        "note": "CIs are Wilson 95%; n is small -- they are wide by construction" if tp + fp + fn else "",
    }


def engine_rates(reports: list[Optional[dict]]) -> dict:
    """Abstention rate per claim and missing-evidence rate per signal, frame-weighted."""
    claim_frames: dict[str, float] = {}
    claim_abstain: dict[str, float] = {}
    missing: dict[str, int] = {}
    frames = 0
    for report in reports:
        if not report:
            continue
        frames += int(report.get("frames") or 0)
        for name, count in (report.get("missing_frames") or {}).items():
            missing[name] = missing.get(name, 0) + int(count)
        for claim, comp in (report.get("comparison") or {}).items():
            rate = comp.get("abstain_rate")
            if rate is None:
                continue
            n = int(report.get("frames") or 0)
            claim_frames[claim] = claim_frames.get(claim, 0) + n
            claim_abstain[claim] = claim_abstain.get(claim, 0) + rate * n
    return {
        "frames": frames,
        "abstention_rate": {c: round(claim_abstain[c] / claim_frames[c], 4) for c in sorted(claim_frames)
                            if claim_frames[c]},
        "missing_evidence_rate": {s: round(n / frames, 4) for s, n in sorted(missing.items())} if frames else {},
    }


__all__ = ["ALERT_TYPES", "EVIDENCE_CLASSES", "aggregate", "engine_rates", "score_scenario", "wilson"]
