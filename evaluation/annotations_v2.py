"""Annotation schema v2: one clip's ground truth, plus who and what it is about.

v2 adds to v1 (``evaluation/annotations.py``), which it reuses for spans and events:

  * ``clip``: subject, session, vehicle, recorded_on (so vehicle-day is known),
    seat, other occupants (pseudonyms), duration;
  * ``scenarios``: the matrix cells the clip counts toward, and ``tags`` (its
    value on each matrix axis), checked against ``evaluation/scenarios.py``;
  * ``annotator``: who labelled it, so two independent labellings of the same
    clip can be compared (``evaluation/agreement.py``);
  * ``uncertain``: spans an annotator cannot label, for some event types or
    all. They are neither error nor evaluated time, and reported as their own
    rate rather than silently folded into ignore.

v1 documents still load: ``migrate_v1`` fills the v2 fields from the manifest
item (``load_any``). A v1 document whose item lacks vehicle or recording date is
refused, not guessed. The vocabulary is ``schemas/incabin_v2.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from evaluation.annotations import Annotation, GroundTruthEvent, Span, _span
from evaluation.annotations import parse as parse_v1

if TYPE_CHECKING:
    from evaluation.manifest import Item
    from evaluation.scenarios import Matrix

SCHEMA_NAME = "sensordyme.incabin.annotation"
VOCAB_PATH = Path(__file__).resolve().parent / "schemas" / "incabin_v2.json"
MIGRATED_ANNOTATOR = "unknown-v1"


def vocabulary(path: str | Path = VOCAB_PATH) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class ClipMeta:
    subject: str
    session: str
    vehicle: str
    recorded_on: str  # YYYY-MM-DD
    seat: str
    duration_s: Optional[float] = None
    occupants: tuple[str, ...] = ()  # everyone else visible in the clip (pseudonyms)
    mount: Optional[str] = None

    @property
    def vehicle_day(self) -> str:
        return f"{self.vehicle}@{self.recorded_on}"

    @property
    def people(self) -> tuple[str, ...]:
        return (self.subject, *self.occupants)


@dataclass(frozen=True)
class UncertainSpan:
    span: Span
    reason: str
    types: frozenset[str] = frozenset()  # empty: unlabelable for every type

    def covers(self, event_type: str) -> bool:
        return not self.types or event_type in self.types


@dataclass
class AnnotationV2:
    item: str
    annotator: str
    clip: ClipMeta
    scenarios: tuple[str, ...]
    tags: dict
    events: list[GroundTruthEvent]
    states: dict[str, list[Span]]
    ignore: list[Span]
    uncertain: list[UncertainSpan] = field(default_factory=list)
    migrated_from: Optional[int] = None

    def as_v1(self) -> Annotation:
        """The v1 view (uncertain spans dropped), for v1-era code paths."""
        return Annotation(self.item, list(self.events), dict(self.states), list(self.ignore))

    def excluded_for(self, event_types: set[str] | frozenset[str]) -> list[Span]:
        """Ignore spans plus every uncertain span covering any of ``event_types``."""
        return list(self.ignore) + [u.span for u in self.uncertain
                                    if not u.types or (u.types & set(event_types))]


def _date(text, where: str, problems: list[str]) -> Optional[str]:
    try:
        return dt.date.fromisoformat(str(text)).isoformat()
    except ValueError:
        problems.append(f"{where}: {text!r} is not YYYY-MM-DD")
        return None


def parse(doc: dict, *, vocab: Optional[dict] = None,
          matrix: Optional[Matrix] = None) -> tuple[Optional[AnnotationV2], list[str]]:
    """Validate a v2 document. Returns (annotation or None, every problem found)."""
    vocab = vocab if vocab is not None else vocabulary()
    problems: list[str] = []
    if doc.get("schema") != SCHEMA_NAME or doc.get("version") != 2:
        problems.append(f"not a v2 annotation (schema {doc.get('schema')!r}, version {doc.get('version')!r})")
    known_events = set(vocab.get("events", {}))
    known_states = set(vocab.get("states", {}))
    base, base_problems = parse_v1(doc, known_types=known_events)
    problems += base_problems
    annotator = str(doc.get("annotator") or "")
    if not annotator:
        problems.append("annotator is required (double annotation compares annotators)")

    raw_clip: dict = doc["clip"] if isinstance(doc.get("clip"), dict) else {}
    for key in vocab.get("document", {}).get("clip_required", ()):
        if not raw_clip.get(key):
            problems.append(f"clip.{key} is required")
    seat = str(raw_clip.get("seat") or "")
    if seat and seat not in vocab.get("seats", ()):
        problems.append(f"clip.seat {seat!r} is not one of {vocab.get('seats')}")
    recorded_on = _date(raw_clip.get("recorded_on"), "clip.recorded_on", problems) \
        if raw_clip.get("recorded_on") else None
    duration = raw_clip.get("duration_s")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or duration <= 0):
        problems.append(f"clip.duration_s must be a positive number, got {duration!r}")
        duration = None
    occupants = raw_clip.get("occupants") or []
    if not isinstance(occupants, list) or not all(isinstance(o, str) and o for o in occupants):
        problems.append("clip.occupants must be a list of pseudonyms")
        occupants = []
    if raw_clip.get("subject") in occupants:
        problems.append("clip.occupants must not repeat clip.subject")

    for name in (doc.get("states") or {}):
        if name not in known_states:
            problems.append(f"states.{name}: unknown state")
    ignore_reasons = set(vocab.get("ignore_reasons", ()))
    for n, raw in enumerate(doc.get("ignore") or []):
        if isinstance(raw, dict) and raw.get("reason") not in ignore_reasons:
            problems.append(f"ignore[{n}]: reason {raw.get('reason')!r} is not one of {sorted(ignore_reasons)}")

    uncertain = []
    reasons = set(vocab.get("uncertain_reasons", ()))
    for n, raw in enumerate(doc.get("uncertain") or []):
        if not isinstance(raw, dict):
            problems.append(f"uncertain[{n}]: not an object")
            continue
        span = _span(raw, f"uncertain[{n}]", problems)
        if raw.get("reason") not in reasons:
            problems.append(f"uncertain[{n}]: reason {raw.get('reason')!r} is not one of {sorted(reasons)}")
        types = frozenset(raw.get("types") or ())
        if bad := sorted(types - known_events):
            problems.append(f"uncertain[{n}]: unknown event types {bad}")
        if span is not None:
            uncertain.append(UncertainSpan(span, str(raw.get("reason")), types))

    scenarios = tuple(str(s) for s in (doc.get("scenarios") or ()))
    tags = dict(doc.get("tags") or {})
    if matrix is not None:
        problems += matrix.clip_tag_problems(scenarios, tags)
    elif not scenarios:
        problems.append("scenarios: a clip must name at least one matrix cell")

    if base is not None:
        for e in base.events:
            if duration is not None and e.span.end_s > float(duration) + 1e-6:
                problems.append(f"{e.type} ends at {e.span.end_s}s, after the clip ({duration}s)")
            for u in uncertain:
                if u.covers(e.type) and u.span.start_s <= e.span.start_s and e.span.end_s <= u.span.end_s:
                    problems.append(f"{e.type} at {e.span.start_s}s lies wholly inside an uncertain span "
                                    f"for its type: label it or mark it uncertain, not both")
    if problems or base is None:
        return None, problems
    clip = ClipMeta(subject=str(raw_clip["subject"]), session=str(raw_clip["session"]),
                    vehicle=str(raw_clip["vehicle"]), recorded_on=str(recorded_on), seat=seat,
                    duration_s=float(duration) if duration is not None else None,
                    occupants=tuple(occupants), mount=raw_clip.get("mount"))
    return AnnotationV2(item=base.item, annotator=annotator, clip=clip, scenarios=scenarios, tags=tags,
                        events=base.events, states=base.states, ignore=base.ignore, uncertain=uncertain,
                        migrated_from=doc.get("migrated_from")), []


def _split_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [v.strip() for v in str(value or "").split(";") if v.strip()]


def migrate_v1(doc: dict, item: Item, *, axes: Optional[dict] = None,
               duration_s: Optional[float] = None) -> tuple[Optional[dict], list[str]]:
    """A v1 document as v2, with clip metadata from the manifest item. Never guesses."""
    cond = item.conditions or {}
    problems = [f"v1 migration: item {item.id} has no conditions.{k} (needed for leakage checks)"
                for k in ("vehicle", "recorded_on", "seat") if not cond.get(k)]
    if doc.get("item") and doc.get("item") != item.id:
        problems.append(f"v1 migration: document is for {doc.get('item')!r}, manifest item is {item.id!r}")
    if problems:
        return None, problems
    tags = {k: v for k, v in cond.items() if axes is not None and k in axes}
    clip = {"subject": item.subject, "session": item.session, "vehicle": cond["vehicle"],
            "recorded_on": cond["recorded_on"], "seat": cond["seat"],
            "occupants": _split_list(cond.get("occupants"))}
    if duration_s is not None:
        clip["duration_s"] = duration_s
    out = {"schema": SCHEMA_NAME, "version": 2, "migrated_from": 1, "item": item.id,
           "annotator": MIGRATED_ANNOTATOR, "clip": clip, "scenarios": _split_list(cond.get("scenarios")),
           "tags": tags, "events": doc.get("events", []), "states": doc.get("states") or {},
           "ignore": doc.get("ignore", []), "uncertain": []}
    return out, []


def load_any(path: str | Path, item: Optional[Item] = None, *, vocab: Optional[dict] = None,
             matrix: Optional[Matrix] = None) -> tuple[Optional[AnnotationV2], list[str]]:
    """Load a v2 document, or a v1 document migrated with ``item``'s metadata."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return None, [f"{path}: {exc}"]
    if not isinstance(doc, dict):
        return None, [f"{path}: not a JSON object"]
    if doc.get("version", 1) == 1 and doc.get("schema") in (None, SCHEMA_NAME):
        if item is None:
            return None, [f"{path}: a v1 annotation needs its manifest item to migrate"]
        migrated, problems = migrate_v1(doc, item, axes=dict(matrix.axes) if matrix else None)
        if migrated is None:
            return None, problems
        doc = migrated
    return parse(doc, vocab=vocab, matrix=matrix)


def consistency_problems(annotation: AnnotationV2, item: Item) -> list[str]:
    """Where an annotation disagrees with its manifest item about who and what it is."""
    problems = []
    if annotation.item != item.id:
        problems.append(f"annotation is for {annotation.item!r}, manifest item is {item.id!r}")
    if annotation.clip.subject != item.subject:
        problems.append(f"clip.subject {annotation.clip.subject!r} != manifest subject {item.subject!r}")
    if annotation.clip.session != item.session:
        problems.append(f"clip.session {annotation.clip.session!r} != manifest session {item.session!r}")
    cond = item.conditions or {}
    for key, value in (("vehicle", annotation.clip.vehicle), ("recorded_on", annotation.clip.recorded_on)):
        if cond.get(key) and str(cond[key]) != value:
            problems.append(f"clip.{key} {value!r} != manifest {key} {cond[key]!r}")
    if set(_split_list(cond.get("occupants"))) != set(annotation.clip.occupants):
        problems.append(f"clip.occupants {sorted(annotation.clip.occupants)} != manifest occupants "
                        f"{sorted(_split_list(cond.get('occupants')))} (the split check reads the manifest)")
    return problems
