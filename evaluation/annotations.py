"""Ground truth for one clip: what happened, when, and which spans do not count.

    {
      "item": "s01_day_001",
      "events":  [{"type": "drowsiness", "start_s": 41.0, "end_s": 47.5, "attrs": {}}],
      "states":  {"eyes_closed": [[41.0, 47.5]], "phone_in_hand": [[90.0, 101.2]]},
      "ignore":  [{"start_s": 0.0, "end_s": 20.0, "reason": "calibration"}]
    }

``events`` are behaviours an alert should catch (event-level scoring).
``states`` are per-frame binary labels (frame-level scoring of signals such as
eyes-closed or phone-in-hand). ``ignore`` spans are excluded from both: neither
a miss nor a false alarm, and not counted in the hours that rates divide by.
Times are seconds from the first frame of the clip, on the clip's own clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Span:
    start_s: float
    end_s: float

    def contains(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    @property
    def length(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class GroundTruthEvent:
    type: str
    span: Span
    attrs: dict = field(default_factory=dict, hash=False, compare=False)


@dataclass
class Annotation:
    item: str
    events: list[GroundTruthEvent]
    states: dict[str, list[Span]]
    ignore: list[Span]

    def ignored(self, t: float) -> bool:
        return any(s.contains(t) for s in self.ignore)

    def ignored_seconds(self, duration_s: float) -> float:
        """Seconds of [0, duration_s] covered by ignore spans (overlaps counted once)."""
        covered, end = 0.0, 0.0
        for s in sorted(self.ignore, key=lambda s: s.start_s):
            lo, hi = max(s.start_s, end, 0.0), min(s.end_s, duration_s)
            if hi > lo:
                covered += hi - lo
            end = max(end, s.end_s)
        return covered


def _span(raw, where: str, problems: list[str]) -> Optional[Span]:
    try:
        start, end = (raw["start_s"], raw["end_s"]) if isinstance(raw, dict) else raw
        span = Span(float(start), float(end))
    except (KeyError, TypeError, ValueError):
        problems.append(f"{where}: not a [start_s, end_s] span: {raw!r}")
        return None
    if not 0 <= span.start_s < span.end_s:
        problems.append(f"{where}: needs 0 <= start_s < end_s, got {span.start_s}..{span.end_s}")
        return None
    return span


def parse(doc: dict, *, known_types: Optional[set[str]] = None,
          duration_s: Optional[float] = None) -> tuple[Optional[Annotation], list[str]]:
    """Validate an annotation document. Returns (annotation or None, problems)."""
    problems: list[str] = []
    item = doc.get("item")
    if not item:
        problems.append("annotation has no item id")
    events = []
    for n, raw in enumerate(doc.get("events", [])):
        kind = raw.get("type") if isinstance(raw, dict) else None
        if not kind:
            problems.append(f"events[{n}]: no type")
            continue
        if known_types is not None and kind not in known_types:
            problems.append(f"events[{n}]: unknown event type {kind!r}")
        if (span := _span(raw, f"events[{n}]", problems)) is not None:
            events.append(GroundTruthEvent(kind, span, dict(raw.get("attrs") or {})))
    states = {}
    for name, spans in (doc.get("states") or {}).items():
        states[name] = [s for n, raw in enumerate(spans) if (s := _span(raw, f"states.{name}[{n}]", problems))]
    ignore = [s for n, raw in enumerate(doc.get("ignore", [])) if (s := _span(raw, f"ignore[{n}]", problems))]
    if duration_s is not None:
        for e in events:
            if e.span.start_s > duration_s:
                problems.append(f"{e.type} at {e.span.start_s}s starts after the clip ends ({duration_s:.1f}s)")
    if problems:
        return None, problems
    return Annotation(str(item), events, states, ignore), []


def load(path: str | Path, **kwargs) -> tuple[Optional[Annotation], list[str]]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, [f"{path}: {exc}"]
    return parse(doc, **kwargs)
