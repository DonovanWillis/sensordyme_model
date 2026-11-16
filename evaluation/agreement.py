"""Inter-annotator agreement for double-annotated clips (annotation schema v2).

Two annotators label the same clip independently. Per event type, their events
are paired greedily by temporal IoU (highest first, each event used once, IoU at
least ``event_iou_min``); unpaired events are disagreements. Reported per type:
counts, agreement F1 (2 x pairs / (n_a + n_b)), mean IoU of the pairs, and the
onset difference |start_a - start_b| (p50 / p90 / max). Per labelled state:
Cohen's kappa on a fixed time grid. A clip pair passes when every type and state
meets the thresholds in ``schemas/incabin_v2.json`` ``agreement``; a failing pair
goes to the adjudicator. Nothing is averaged away.

    python -m evaluation.agreement --a labels/A1 --b labels/A2 --out agreement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from evaluation import annotations_v2 as av2
from evaluation.annotations import Span


def iou(a: Span, b: Span) -> float:
    inter = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    return inter / union if union > 0 else 0.0


def _quantile(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def event_agreement(a: list[Span], b: list[Span], *, iou_min: float) -> dict:
    """Greedy max-IoU pairing of two annotators' spans of one event type."""
    pairs = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[float, float]] = []  # (iou, onset difference)
    for value, i, j in pairs:
        if value < iou_min or i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((value, abs(a[i].start_s - b[j].start_s)))
    n = len(a) + len(b)
    onsets = [d for _, d in matched]
    return {"n_a": len(a), "n_b": len(b), "matched": len(matched),
            "only_a": len(a) - len(matched), "only_b": len(b) - len(matched),
            "f1": (2 * len(matched) / n) if n else None,
            "mean_iou": (sum(v for v, _ in matched) / len(matched)) if matched else None,
            "onset_p50_s": _quantile(onsets, 0.5), "onset_p90_s": _quantile(onsets, 0.9),
            "onset_max_s": max(onsets) if onsets else None}


def cohen_kappa(a: list[Span], b: list[Span], duration_s: float, step_s: float,
                skip: Optional[list[Span]] = None) -> Optional[float]:
    """Cohen's kappa of two binary state labellings sampled every ``step_s`` seconds."""
    skip = skip or []
    n = both = only_a = only_b = 0
    k = 0
    while (t := k * step_s) < duration_s:
        k += 1
        if any(s.contains(t) for s in skip):
            continue
        x, y = any(s.contains(t) for s in a), any(s.contains(t) for s in b)
        n += 1
        both += x and y
        only_a += x and not y
        only_b += y and not x
    if n == 0:
        return None
    neither = n - both - only_a - only_b
    observed = (both + neither) / n
    pa, pb = (both + only_a) / n, (both + only_b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else None  # both constant: kappa undefined unless identical
    return (observed - expected) / (1 - expected)


def compare(a: av2.AnnotationV2, b: av2.AnnotationV2, *, thresholds: Optional[dict] = None,
            duration_s: Optional[float] = None) -> dict:
    """Agreement of two independent labellings of one clip."""
    t = thresholds or av2.vocabulary()["agreement"]
    problems = []
    if a.item != b.item:
        problems.append(f"different clips: {a.item!r} vs {b.item!r}")
    if a.annotator == b.annotator:
        problems.append(f"both labellings are by {a.annotator!r}: agreement needs two annotators")
    if av2.MIGRATED_ANNOTATOR in (a.annotator, b.annotator):
        problems.append("a migrated v1 labelling has no known annotator")
    duration = duration_s or a.clip.duration_s or b.clip.duration_s
    if duration is None:
        problems.append("clip duration unknown (clip.duration_s): state agreement cannot be sampled")
    if problems:
        return {"item": a.item, "refused": problems, "passes": False}
    assert duration is not None
    events = {}
    for kind in sorted({e.type for e in a.events} | {e.type for e in b.events}):
        events[kind] = event_agreement([e.span for e in a.events if e.type == kind],
                                       [e.span for e in b.events if e.type == kind],
                                       iou_min=float(t["event_iou_min"]))
    skip = list(a.ignore) + list(b.ignore)
    states = {}
    for name in sorted(set(a.states) | set(b.states)):
        states[name] = cohen_kappa(a.states.get(name, []), b.states.get(name, []), duration,
                                   float(t["state_step_s"]), skip)
    failures = []
    for kind, m in events.items():
        if m["f1"] is not None and m["f1"] < t["event_f1_min"]:
            failures.append(f"{kind}: agreement F1 {m['f1']:.2f} < {t['event_f1_min']}")
        if m["onset_p90_s"] is not None and m["onset_p90_s"] > t["onset_p90_max_s"]:
            failures.append(f"{kind}: onset p90 {m['onset_p90_s']:.2f}s > {t['onset_p90_max_s']}s")
    for name, kappa in states.items():
        if kappa is None or kappa < t["state_kappa_min"]:
            failures.append(f"state {name}: kappa {kappa} < {t['state_kappa_min']}")
    return {"item": a.item, "annotators": [a.annotator, b.annotator], "events": events,
            "states_kappa": states, "failures": failures, "passes": not failures,
            "thresholds": {k: t[k] for k in ("event_iou_min", "event_f1_min", "onset_p90_max_s",
                                             "state_kappa_min", "state_step_s")}}


def compare_dirs(dir_a: Path, dir_b: Path) -> dict:
    """Every clip labelled in both directories (files matched by name)."""
    out: dict = {"pairs": [], "unpaired": [], "problems": []}
    names_a = {p.name for p in dir_a.glob("*.json")}
    names_b = {p.name for p in dir_b.glob("*.json")}
    out["unpaired"] = sorted(names_a ^ names_b)
    for name in sorted(names_a & names_b):
        a, pa = av2.load_any(dir_a / name)
        b, pb = av2.load_any(dir_b / name)
        if a is None or b is None:
            out["problems"].append({"file": name, "a": pa, "b": pb})
            continue
        out["pairs"].append(compare(a, b))
    out["clips"] = len(out["pairs"])
    out["passing"] = sum(1 for p in out["pairs"] if p["passes"])
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inter-annotator agreement (annotation schema v2)")
    p.add_argument("--a", required=True, help="directory of annotator A's v2 JSON files")
    p.add_argument("--b", required=True, help="directory of annotator B's v2 JSON files (same file names)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = compare_dirs(Path(args.a), Path(args.b))
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if not result["clips"]:
        print("agreement: NO DATA (no clip labelled by both annotators)")
        return 1
    print(f"agreement: {result['passing']}/{result['clips']} clip pairs pass; "
          f"{len(result['unpaired'])} unpaired files; {len(result['problems'])} unreadable")
    return 0 if result["passing"] == result["clips"] and not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
