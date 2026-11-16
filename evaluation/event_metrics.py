"""Event-level scoring of module alerts against v2 ground truth, with a power check.

Builds on ``evaluation/metrics.py`` (``EventCounts``, ``wilson``,
``cluster_bootstrap``) and scores through the scenario matrix's rules, which map
ground-truth event types onto the product's alert types:

  * a POSITIVE event (e.g. long_closure) is CAUGHT by the first listed alert in
    [start - early_s, end + late_s]; its latency is that alert's time minus the
    event's onset, also reported minus the alert's configured sustain time. One
    alert may catch several concurrent positives (eyes off road AND phone use).
    A missed event is ABSTAINED when the module reported every alert that could
    catch it unavailable for at least ``abstain_fraction`` of the event's span
    (it declined: unknown / degraded / not ready), else a SILENT miss;
  * an alert matching an already-caught event is a REPEAT: neither true nor false;
  * an alert in a NEGATIVE event's window (a mirror glance) is a FALSE ALERT, and
    also an "alert on legitimate behaviour" for that negative type;
  * an alert in a BOUNDARY event's window (outside the validated scope) is
    tallied separately, neither true nor false;
  * any other alert is a FALSE ALERT;
  * alerts inside ignore spans or uncertain spans covering their type are not
    scored, and that time is not exposure.

Exposure hours are per alert type: clip time minus ignore and applicable
uncertain spans. The unknown rate is the share of exposure time the module said
an alert type could not be evaluated (from replay; NO DATA when not recorded).

The POWER CHECK (``power``) decides whether a number may be printed as a claim:
it needs the cell's minimum subjects, clips and events (or exposure hours).
Below it the status is ``insufficient_data``; with nothing at all, ``no_data``.
Claims are computed only for ``ok`` metrics; diagnostics only on request.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from evaluation.annotations import Span
from evaluation.annotations_v2 import AnnotationV2
from evaluation.metrics import EventCounts, cluster_bootstrap, wilson
from evaluation.scenarios import Matrix, Minimums

NO_DATA = "no_data"
INSUFFICIENT = "insufficient_data"
OK = "ok"


# -- interval arithmetic ---------------------------------------------------------------------

def union(spans: Iterable[Span], lo: float = 0.0, hi: float = float("inf")) -> list[Span]:
    """Disjoint, sorted spans covering the same time as ``spans``, clipped to [lo, hi]."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start_s):
        a, b = max(lo, s.start_s), min(hi, s.end_s)
        if b <= a:
            continue
        if out and a <= out[-1].end_s:
            if b > out[-1].end_s:
                out[-1] = Span(out[-1].start_s, b)
        else:
            out.append(Span(a, b))
    return out


def length(spans: Iterable[Span]) -> float:
    return sum(s.end_s - s.start_s for s in union(spans))


def overlap(a: Iterable[Span], b: Iterable[Span]) -> float:
    """Seconds covered by both span sets."""
    ua, ub = union(a), union(b)
    total, j = 0.0, 0
    for s in ua:
        while j < len(ub) and ub[j].end_s <= s.start_s:
            j += 1
        k = j
        while k < len(ub) and ub[k].start_s < s.end_s:
            total += max(0.0, min(s.end_s, ub[k].end_s) - max(s.start_s, ub[k].start_s))
            k += 1
    return total


def _inside(t: float, spans: Iterable[Span]) -> bool:
    return any(s.start_s <= t <= s.end_s for s in spans)


# -- tallies ---------------------------------------------------------------------------------

@dataclass
class DetectionTally:
    """One positive ground-truth type."""

    events: int = 0
    caught: int = 0
    abstained: int = 0  # missed while the module said it could not evaluate
    excluded: int = 0  # onset inside ignore/uncertain: not scored
    latencies: list[float] = field(default_factory=list)  # onset -> first catching alert
    excess: list[float] = field(default_factory=list)  # latency minus the catching alert's sustain time
    abstention_known: bool = True

    def add(self, o: DetectionTally) -> DetectionTally:
        return DetectionTally(self.events + o.events, self.caught + o.caught, self.abstained + o.abstained,
                              self.excluded + o.excluded, self.latencies + o.latencies, self.excess + o.excess,
                              self.abstention_known and o.abstention_known)

    @property
    def missed(self) -> int:
        return self.events - self.caught

    def counts(self) -> EventCounts:
        return EventCounts(gt=self.events, caught=self.caught, latencies=list(self.latencies))


@dataclass
class AlertTally:
    """One alert type the module emits."""

    predictions: int = 0  # scored (outside ignore/uncertain)
    true_first: int = 0  # caught at least one new positive
    repeats: int = 0
    false_alerts: int = 0
    false_on_negative: int = 0  # false alerts inside a negative (legitimate behaviour) window
    boundary: int = 0  # alerts during out-of-scope behaviour: neither true nor false
    not_scored: int = 0  # inside ignore/uncertain
    exposure_s: float = 0.0
    unavailable_s: Optional[float] = 0.0  # None: availability not recorded for some clip

    def add(self, o: AlertTally) -> AlertTally:
        unavailable = None if self.unavailable_s is None or o.unavailable_s is None \
            else self.unavailable_s + o.unavailable_s
        return AlertTally(self.predictions + o.predictions, self.true_first + o.true_first,
                          self.repeats + o.repeats, self.false_alerts + o.false_alerts,
                          self.false_on_negative + o.false_on_negative, self.boundary + o.boundary,
                          self.not_scored + o.not_scored, self.exposure_s + o.exposure_s, unavailable)

    @property
    def hours(self) -> float:
        return self.exposure_s / 3600

    def counts(self) -> EventCounts:
        return EventCounts(caught=self.true_first, false_alarms=self.false_alerts, repeats=self.repeats,
                           predictions=self.predictions, hours=self.hours)


@dataclass
class NegativeTally:
    """One negative (must-not-alert) ground-truth type."""

    events: int = 0
    alerted: int = 0  # events with at least one false alert in their window

    def add(self, o: NegativeTally) -> NegativeTally:
        return NegativeTally(self.events + o.events, self.alerted + o.alerted)


@dataclass
class Scores:
    detection: dict[str, DetectionTally] = field(default_factory=dict)
    alerts: dict[str, AlertTally] = field(default_factory=dict)
    negatives: dict[str, NegativeTally] = field(default_factory=dict)
    boundary_events: dict[str, int] = field(default_factory=dict)
    clips: int = 0
    seconds: float = 0.0
    uncertain_s: float = 0.0
    unscored_types: dict[str, int] = field(default_factory=dict)  # module events not in scoring.alerts

    def add(self, o: Scores) -> Scores:
        def keys(a: dict, b: dict) -> list[str]:
            return sorted(set(a) | set(b))

        def merge_int(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
            return {k: a.get(k, 0) + b.get(k, 0) for k in keys(a, b)}

        detection = {k: self.detection.get(k, DetectionTally()).add(o.detection.get(k, DetectionTally()))
                     for k in keys(self.detection, o.detection)}
        alerts = {k: self.alerts.get(k, AlertTally()).add(o.alerts.get(k, AlertTally()))
                  for k in keys(self.alerts, o.alerts)}
        negatives = {k: self.negatives.get(k, NegativeTally()).add(o.negatives.get(k, NegativeTally()))
                     for k in keys(self.negatives, o.negatives)}
        return Scores(detection, alerts, negatives,
                      merge_int(self.boundary_events, o.boundary_events), self.clips + o.clips,
                      self.seconds + o.seconds, self.uncertain_s + o.uncertain_s,
                      merge_int(self.unscored_types, o.unscored_types))


@dataclass
class ClipInput:
    item: str
    subject: str
    annotation: AnnotationV2
    duration_s: float
    predictions: list[tuple[float, str]]
    unavailable: Optional[dict[str, list[Span]]]  # alert type -> spans; None: not recorded


def score_clip(clip: ClipInput, matrix: Matrix, sustain_s: dict[str, float]) -> Scores:
    """Score one clip's module alerts against its v2 ground truth."""
    ann, d = clip.annotation, clip.duration_s
    rules = matrix.rules
    alert_types = sorted(matrix.alerts)
    # Ground-truth types each alert can answer for (for uncertain-span exclusion).
    answers = {a: {t for t, r in rules.items() if a in r.alerts} for a in alert_types}
    excluded = {a: union(ann.excluded_for(answers[a]), 0.0, d) for a in alert_types}

    scores = Scores(clips=1, seconds=d, uncertain_s=length(union((u.span for u in ann.uncertain), 0.0, d)))
    positives, negatives, boundaries = [], [], []
    for e in ann.events:
        rule = rules.get(e.type)
        if rule is None:
            continue
        window = rule.window(e.span.start_s, e.span.end_s)
        if rule.role == "positive":
            tally = scores.detection.setdefault(e.type, DetectionTally(abstention_known=clip.unavailable is not None))
            if _inside(e.span.start_s, ann.excluded_for({e.type})):
                tally.excluded += 1
                continue
            tally.events += 1
            positives.append([e, rule, window, None])  # [event, rule, window, caught_at]
        elif rule.role == "negative":
            scores.negatives.setdefault(e.type, NegativeTally()).events += 1
            negatives.append([e, rule, window, False])
        else:
            scores.boundary_events[e.type] = scores.boundary_events.get(e.type, 0) + 1
            boundaries.append((e, rule, window))

    for a in alert_types:
        tally = scores.alerts.setdefault(a, AlertTally())
        tally.exposure_s = d - length(excluded[a])
        if clip.unavailable is None:
            tally.unavailable_s = None
        else:
            spans = union(clip.unavailable.get(a, []), 0.0, d)
            tally.unavailable_s = length(spans) - overlap(spans, excluded[a])

    for t, a in sorted(clip.predictions):
        if a not in matrix.alerts:
            scores.unscored_types[a] = scores.unscored_types.get(a, 0) + 1
            continue
        tally = scores.alerts[a]
        if _inside(t, excluded[a]):
            tally.not_scored += 1
            continue
        tally.predictions += 1
        candidates = [p for p in positives if a in p[1].alerts and p[2][0] <= t <= p[2][1]]
        fresh = [p for p in candidates if p[3] is None]
        if fresh:
            tally.true_first += 1
            for p in fresh:
                p[3] = t
                det = scores.detection[p[0].type]
                det.caught += 1
                latency = max(0.0, t - p[0].span.start_s)
                det.latencies.append(latency)
                if a in sustain_s:
                    det.excess.append(latency - sustain_s[a])
            continue
        if candidates:
            tally.repeats += 1
            continue
        hit_negative = [n for n in negatives if a in n[1].alerts and n[2][0] <= t <= n[2][1]]
        if hit_negative:
            tally.false_alerts += 1
            tally.false_on_negative += 1
            for n in hit_negative:
                n[3] = True
            continue
        if any(a in r.alerts and w[0] <= t <= w[1] for _, r, w in boundaries):
            tally.boundary += 1
            continue
        tally.false_alerts += 1

    for n in negatives:
        if n[3]:
            scores.negatives[n[0].type].alerted += 1
    if clip.unavailable is not None:
        for e, rule, _, caught_at in positives:
            if caught_at is not None:
                continue
            span_len = e.span.end_s - e.span.start_s
            declined = all(
                overlap([e.span], clip.unavailable.get(a, [])) >= matrix.abstain_fraction * span_len
                for a in rule.alerts)
            if declined:
                scores.detection[e.type].abstained += 1
    return scores


# -- the power check and summaries -----------------------------------------------------------

def power(*, subjects: int, clips: int, minimums: Minimums, events: Optional[int] = None,
          negative_events: Optional[int] = None, hours: Optional[float] = None) -> tuple[str, list[str]]:
    """(status, reasons) for one metric: ok, insufficient_data or no_data."""
    sample = events if events is not None else negative_events if negative_events is not None else hours
    if clips == 0 or not sample:
        return NO_DATA, ["nothing recorded" if clips == 0 else "no events / exposure in the recorded clips"]
    short = []
    if subjects < minimums.subjects:
        short.append(f"{subjects}/{minimums.subjects} subjects")
    if clips < minimums.clips:
        short.append(f"{clips}/{minimums.clips} clips")
    if events is not None and events < minimums.events:
        short.append(f"{events}/{minimums.events} events")
    if negative_events is not None and negative_events < minimums.negative_events:
        short.append(f"{negative_events}/{minimums.negative_events} negative events")
    if hours is not None and hours < minimums.exposure_hours:
        short.append(f"{hours:.2f}/{minimums.exposure_hours} exposure hours")
    return (INSUFFICIENT, short) if short else (OK, [])


def _quantile(xs: list[float], q: float) -> Optional[float]:
    return EventCounts(latencies=list(xs)).latency(q)


def _dist(xs: list[float]) -> dict:
    return {"n": len(xs), "p50": _quantile(xs, 0.5), "p90": _quantile(xs, 0.9), "max": max(xs) if xs else None}


def _gate(status: str, reasons: list[str], n: dict, claim: dict, include_diagnostics: bool) -> dict:
    out = {"status": status, "reasons": reasons, "n": n, "claim": claim if status == OK else None}
    if include_diagnostics and status == INSUFFICIENT:
        out["diagnostic_not_a_claim"] = claim
    return out


def summarise(per_subject: dict[str, Scores], *, matrix: Matrix, minimums: Minimums,
              positives: Optional[Iterable[str]] = None, negatives: Optional[Iterable[str]] = None,
              include_diagnostics: bool = False, bootstrap_n: int = 2000, seed: int = 0) -> dict:
    """Every metric for a group of clips (a cell, or the whole split), each behind the power check."""
    total = Scores()
    for s in per_subject.values():
        total = total.add(s)
    subjects = sorted(per_subject)
    pos_types = sorted(set(positives) if positives is not None else set(matrix.types_with_role("positive")))
    neg_types = sorted(set(negatives) if negatives is not None else set(matrix.types_with_role("negative")))
    out: dict = {"subjects": len(subjects), "clips": total.clips, "hours": round(total.seconds / 3600, 4),
                 "uncertain_hours": round(total.uncertain_s / 3600, 4),
                 "unlabelable_fraction": (total.uncertain_s / total.seconds) if total.seconds else None,
                 "detection": {}, "alerts": {}, "negatives": {},
                 "boundary_events": total.boundary_events, "unscored_module_events": total.unscored_types}

    for kind in pos_types:
        tally = total.detection.get(kind, DetectionTally())
        with_events = {s: sc.detection[kind] for s, sc in per_subject.items()
                       if kind in sc.detection and sc.detection[kind].events}
        status, reasons = power(subjects=len(with_events), clips=total.clips, minimums=minimums,
                                events=tally.events)
        counts = tally.counts()
        claim = {
            "recall": counts.recall, "missed": tally.missed,
            "missed_abstained": tally.abstained if tally.abstention_known else None,
            "missed_silent": (tally.missed - tally.abstained) if tally.abstention_known else None,
            "recall_wilson95": wilson(tally.caught, tally.events),
            "recall_subject_bootstrap95": cluster_bootstrap(
                {s: t.counts() for s, t in with_events.items()}, lambda c: c.recall, n=bootstrap_n, seed=seed),
            "latency_s": _dist(tally.latencies),
            "latency_minus_sustain_s": _dist(tally.excess),
        }
        out["detection"][kind] = _gate(status, reasons, {"subjects": len(with_events), "events": tally.events,
                                                         "excluded": tally.excluded}, claim, include_diagnostics)

    for a in sorted(matrix.alerts):
        tally = total.alerts.get(a, AlertTally())
        exposed = {s: sc.alerts[a] for s, sc in per_subject.items() if a in sc.alerts and sc.alerts[a].exposure_s}
        status, reasons = power(subjects=len(exposed), clips=total.clips, minimums=minimums,
                                hours=tally.hours)
        counts = tally.counts()
        per_subject_rate = {s: (t.false_alerts / t.hours if t.hours else None) for s, t in sorted(exposed.items())}
        claim = {
            "false_alerts": tally.false_alerts, "false_alerts_per_hour": counts.false_alarms_per_hour,
            "false_alerts_per_hour_subject_bootstrap95": cluster_bootstrap(
                {s: t.counts() for s, t in exposed.items()}, lambda c: c.false_alarms_per_hour,
                n=bootstrap_n, seed=seed),
            "false_alerts_per_hour_by_subject": per_subject_rate,
            "false_alerts_on_legitimate_behaviour": tally.false_on_negative,
            "alerts_per_hour": counts.alerts_per_hour, "repeats": tally.repeats,
            "boundary_alerts": tally.boundary,
            "unknown_rate": (tally.unavailable_s / tally.exposure_s)
            if tally.unavailable_s is not None and tally.exposure_s else None,
        }
        precision_n = tally.true_first + tally.false_alerts
        p_status, _ = power(subjects=len(exposed), clips=total.clips, minimums=minimums,
                                    events=precision_n) if status == OK else (status, reasons)
        claim["precision"] = counts.precision if p_status == OK else None
        claim["precision_status"] = p_status
        if p_status == OK:
            claim["precision_wilson95"] = wilson(tally.true_first, precision_n)
            claim["precision_subject_bootstrap95"] = cluster_bootstrap(
                {s: t.counts() for s, t in exposed.items()}, lambda c: c.precision, n=bootstrap_n, seed=seed)
        out["alerts"][a] = _gate(status, reasons, {"subjects": len(exposed), "exposure_hours": round(tally.hours, 4),
                                                   "alerts_scored": tally.predictions,
                                                   "availability_recorded": tally.unavailable_s is not None},
                                 claim, include_diagnostics)

    for kind in neg_types:
        tally = total.negatives.get(kind, NegativeTally())
        with_events = sum(1 for sc in per_subject.values() if sc.negatives.get(kind, NegativeTally()).events)
        status, reasons = power(subjects=with_events, clips=total.clips, minimums=minimums,
                                negative_events=tally.events)
        claim = {"alerted": tally.alerted,
                 "alert_rate": tally.alerted / tally.events if tally.events else None,
                 "alert_rate_wilson95": wilson(tally.alerted, tally.events)}
        out["negatives"][kind] = _gate(status, reasons, {"subjects": with_events, "events": tally.events},
                                       claim, include_diagnostics)
    return out
