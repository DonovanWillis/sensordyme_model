"""Event- and frame-level scoring, with confidence intervals that respect who was filmed.

Event level (what a driver experiences):
  * a ground-truth event is CAUGHT if a prediction of the same type falls in
    [start - early_s, end + late_s]; detection latency = first such prediction - start
  * a prediction matching an already-caught event is a REPEAT (an episode reminder),
    not a false alarm, but it still counts in alerts per hour
  * any other prediction outside an ignore span is a FALSE ALARM
  * rates divide by evaluated hours (clip length minus ignore spans)

Frame level (whether a signal is right, e.g. eyes-closed): per-frame confusion
counts of a predicted binary state against labelled state spans.

Intervals: proportions get a Wilson interval; aggregate metrics get a cluster
bootstrap over SUBJECTS, because clips of one person are not independent and a
per-clip interval would be far too narrow.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Optional

from evaluation.annotations import Annotation, Span


@dataclass
class EventCounts:
    gt: int = 0  # ground-truth events
    caught: int = 0
    false_alarms: int = 0
    repeats: int = 0
    predictions: int = 0  # every prediction outside ignore spans
    hours: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, other: EventCounts) -> EventCounts:
        return EventCounts(
            self.gt + other.gt, self.caught + other.caught, self.false_alarms + other.false_alarms,
            self.repeats + other.repeats, self.predictions + other.predictions,
            self.hours + other.hours, self.latencies + other.latencies,
        )

    @property
    def recall(self) -> Optional[float]:
        return self.caught / self.gt if self.gt else None

    @property
    def precision(self) -> Optional[float]:
        """Fraction of distinct alerts that were real (repeats excluded from both sides)."""
        n = self.caught + self.false_alarms
        return self.caught / n if n else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    @property
    def false_alarms_per_hour(self) -> Optional[float]:
        return self.false_alarms / self.hours if self.hours else None

    @property
    def alerts_per_hour(self) -> Optional[float]:
        return self.predictions / self.hours if self.hours else None

    def latency(self, q: float) -> Optional[float]:
        if not self.latencies:
            return None
        xs = sorted(self.latencies)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    def to_dict(self) -> dict:
        return {
            "gt": self.gt, "caught": self.caught, "missed": self.gt - self.caught,
            "false_alarms": self.false_alarms, "repeats": self.repeats, "predictions": self.predictions,
            "hours": round(self.hours, 4), "recall": self.recall, "precision": self.precision, "f1": self.f1,
            "false_alarms_per_hour": self.false_alarms_per_hour, "alerts_per_hour": self.alerts_per_hour,
            "latency_p50_s": self.latency(0.5), "latency_p90_s": self.latency(0.9),
            "recall_wilson95": wilson(self.caught, self.gt),
        }


def match_events(predicted: Iterable[tuple[float, str]], annotation: Annotation, event_type: str,
                 duration_s: float, *, early_s: float = 0.0, late_s: float = 2.0) -> EventCounts:
    """Score one clip for one event type. ``predicted`` is (t_seconds, type)."""
    gts = sorted((e for e in annotation.events if e.type == event_type), key=lambda e: e.span.start_s)
    counts = EventCounts(gt=len(gts), hours=max(0.0, duration_s - annotation.ignored_seconds(duration_s)) / 3600)
    caught: set[int] = set()
    for t, kind in sorted(predicted):
        if kind != event_type or annotation.ignored(t):
            continue
        counts.predictions += 1
        hit = next((n for n, e in enumerate(gts)
                    if e.span.start_s - early_s <= t <= e.span.end_s + late_s), None)
        if hit is None:
            counts.false_alarms += 1
        elif hit in caught:
            counts.repeats += 1
        else:
            caught.add(hit)
            counts.latencies.append(max(0.0, t - gts[hit].span.start_s))
    counts.caught = len(caught)
    return counts


@dataclass
class FrameCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, o: FrameCounts) -> FrameCounts:
        return FrameCounts(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, self.tn + o.tn)

    def to_dict(self) -> dict:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f1 = 2 * p * r / (p + r) if p and r else None
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": p, "recall": r, "f1": f1,
                "precision_wilson95": wilson(self.tp, self.tp + self.fp),
                "recall_wilson95": wilson(self.tp, self.tp + self.fn)}


def frame_level(times: Sequence[float], predicted: Sequence[bool], truth: Sequence[Span],
                annotation: Annotation) -> FrameCounts:
    counts = FrameCounts()
    for t, pred in zip(times, predicted, strict=True):
        if annotation.ignored(t):
            continue
        real = any(s.contains(t) for s in truth)
        if pred and real:
            counts.tp += 1
        elif pred:
            counts.fp += 1
        elif real:
            counts.fn += 1
        else:
            counts.tn += 1
    return counts


def wilson(k: int, n: int, z: float = 1.96) -> Optional[tuple[float, float]]:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap(per_subject: dict[str, EventCounts], metric: Callable[[EventCounts], Optional[float]],
                      *, n: int = 2000, seed: int = 0, level: float = 0.95) -> Optional[tuple[float, float]]:
    """Percentile interval of ``metric`` resampling SUBJECTS with replacement."""
    subjects = sorted(per_subject)
    if len(subjects) < 2:
        return None  # one subject: no between-subject variability can be estimated
    rng = random.Random(seed)  # noqa: S311 - resampling, not security
    values = []
    for _ in range(n):
        total = EventCounts()
        for s in rng.choices(subjects, k=len(subjects)):
            total = total.add(per_subject[s])
        if (v := metric(total)) is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    lo = values[int((1 - level) / 2 * (len(values) - 1))]
    hi = values[int((1 + level) / 2 * (len(values) - 1))]
    return (lo, hi)
