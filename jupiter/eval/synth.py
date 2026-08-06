"""Generate evaluation and training data from the capability registry.

The registry knows every (target, predicate) pair the device can execute, so
ground truth does not have to be written by hand — it can be enumerated, and the
natural language templated around it. That gives thousands of exactly-labelled
pairs for free, and they cannot drift from the device's real capabilities
because they are derived from the same source the validator uses.

Two limits worth stating plainly, because synthetic data flatters a model:

  * **Templated language is not user language.** Real requests are vaguer,
    longer, and more ambiguous than anything generated here. A model scoring
    well on this set has proved it can fill the schema, not that it understands
    people. The hand-written cases in `cases.py` exist for that, and real
    prompts should replace synthetic ones as they accumulate.
  * **Every generated example is answerable.** A model trained only on these
    learns that every request has an answer, which is the opposite of what
    SensorDyme needs. `generate()` therefore emits refusal cases too, built from
    labels the registry explicitly does NOT have.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator, Optional

from core.capabilities import registry

# Objects users ask about that the detector genuinely cannot see. Held here as
# refusal fixtures, not as a capability list — each is asserted absent from the
# registry when the dataset is built, so this cannot silently become wrong.
UNDETECTABLE = ("can", "package", "parcel", "smoke", "fire", "weapon", "syringe")

# Requests that are not answerable from a camera at all, at any capability level.
UNKNOWABLE = (
    "alert me if someone is thinking about stealing something",
    "tell me if someone is sad",
    "alert me if a person is lying to me",
    "notify me when someone feels unwell",
    "alert me if someone intends to break in",
)

_PRESENT_TEMPLATES = (
    "alert me if a {target} appears",
    "tell me when you see a {target}",
    "notify me if there is a {target}",
    "watch for a {target}",
    "let me know if a {target} shows up",
)
_ABSENT_TEMPLATES = (
    "tell me when the {target} is gone",
    "alert me when there is no {target}",
    "notify me once the {target} leaves",
)
_ZONE_TEMPLATES = (
    "alert me if a {target} enters the {zone}",
    "tell me when a {target} is in the {zone}",
    "notify me if a {target} goes into the {zone} area",
)
_NEAR_TEMPLATES = (
    "alert me if a {target} is near the {secondary}",
    "tell me when the {target} gets on the {secondary}",
    "notify me if a {target} is with the {secondary}",
)
_DURATION_SUFFIXES = (
    ("", None),
    (" for more than {n} seconds", "persist"),
    (" for at least {n} seconds", "persist"),
)
_SCHEDULE_SUFFIXES = (
    ("", None),
    (" between 11pm and 6am", "23:00-06:00"),
    (" after 10 at night", "22:00-06:00"),
)

ZONES = ("kitchen", "pool", "garage", "nursery", "driveway")


@dataclass
class Example:
    """One request and the rule it should produce (or the refusal it should get)."""

    request: str
    expect_supported: bool
    target: str = ""
    predicate: str = ""
    secondary: str = ""
    zone: str = ""
    persist_s: Optional[float] = None
    schedule: str = ""
    tags: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "request": self.request,
            "expect_supported": self.expect_supported,
            "target": self.target,
            "predicate": self.predicate,
            "secondary": self.secondary,
            "zone": self.zone,
            "persist_s": self.persist_s,
            "schedule": self.schedule,
            "tags": list(self.tags),
        }


def _pick(sequence, index):
    return sequence[index % len(sequence)]


def _article(word: str) -> str:
    """'a dog' / 'an apple'. Templated data that reads wrong teaches a model to
    expect language no human writes."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _phrase(template: str, **values) -> str:
    """Fill a template, fixing the article to suit the word that follows it."""
    text = template.format(**values)
    for word in values.values():
        if isinstance(word, str) and word:
            text = text.replace(f"a {word}", f"{_article(word)} {word}")
    return text


def generate(limit: int = 400, reg=None, seed: int = 0) -> list[Example]:
    """Build a deterministic dataset.

    Deterministic on purpose: an eval set that changes between runs cannot show
    whether a change helped. `seed` shifts template selection without making
    anything random.
    """
    reg = reg or registry()
    labels = sorted(reg.labels())
    predicates = reg.predicates()
    if not labels:
        return []

    # Refusals are built FIRST and reserved out of the budget. Appending them
    # last and truncating to `limit` silently produced sets with no refusals at
    # all, so a small run scored refusal_accuracy 1.0 against a provider that
    # refused nothing — the harness reporting a perfect score for the worst
    # possible behaviour. Refusal accuracy is the metric that matters most here;
    # it must never be measured on an empty pool.
    refusals = _refusal_examples(labels)
    reserved = min(len(refusals), max(1, int(limit * 0.2)))
    refusals = refusals[:reserved]
    limit = max(1, limit - len(refusals))

    examples: list[Example] = []

    # --- supported: presence, the commonest request shape ------------------
    for index, target in enumerate(labels):
        if len(examples) >= limit * 0.45:
            break
        template = _pick(_PRESENT_TEMPLATES, index + seed)
        suffix, kind = _pick(_DURATION_SUFFIXES, index + seed)
        seconds = 5 * (index % 6 + 1)
        request = _phrase(template, target=target) + suffix.format(n=seconds)
        examples.append(Example(
            request=request, expect_supported=True, target=target, predicate="present",
            persist_s=float(seconds) if kind == "persist" else None,
            tags=("supported", "present"),
        ))

    # --- supported: absence -------------------------------------------------
    if "absent" in predicates:
        for index, target in enumerate(labels[:20]):
            request = _phrase(_pick(_ABSENT_TEMPLATES, index + seed), target=target)
            examples.append(Example(
                request=request, expect_supported=True, target=target,
                predicate="absent", tags=("supported", "absent"),
            ))

    # --- supported: zones, with and without schedules -----------------------
    if "in_zone" in predicates:
        for index, (target, zone) in enumerate(itertools.product(labels[:12], ZONES)):
            if len(examples) >= limit * 0.8:
                break
            schedule_suffix, schedule = _pick(_SCHEDULE_SUFFIXES, index + seed)
            request = _phrase(_pick(_ZONE_TEMPLATES, index + seed), target=target, zone=zone)
            examples.append(Example(
                request=request + schedule_suffix, expect_supported=True, target=target,
                predicate="in_zone", zone=zone, schedule=schedule or "",
                tags=("supported", "zone") + (("schedule",) if schedule else ()),
            ))

    # --- supported: relations ----------------------------------------------
    if "near" in predicates:
        pairs = [(a, b) for a, b in itertools.product(labels[:10], labels[:10]) if a != b]
        for index, (target, secondary) in enumerate(pairs[:40]):
            request = _phrase(
                _pick(_NEAR_TEMPLATES, index + seed), target=target, secondary=secondary
            )
            examples.append(Example(
                request=request, expect_supported=True, target=target,
                predicate="near", secondary=secondary, tags=("supported", "relation"),
            ))

    return (examples[:limit] + refusals)


def _refusal_examples(labels) -> list[Example]:
    """Requests that must be declined, built from what the registry lacks."""
    examples: list[Example] = []
    # --- refusals: objects the detector cannot see --------------------------
    # Asserted, not assumed: if one of these ever becomes detectable the dataset
    # must stop treating it as a refusal, and a silent wrong label here would
    # train exactly the wrong behaviour.
    for target in UNDETECTABLE:
        if target in labels:
            continue
        for template in _PRESENT_TEMPLATES:
            examples.append(Example(
                request=_phrase(template, target=target), expect_supported=False,
                target=target, tags=("refusal", "undetectable"),
            ))

    # --- refusals: not knowable from a camera at all ------------------------
    for request in UNKNOWABLE:
        examples.append(Example(
            request=request, expect_supported=False, tags=("refusal", "unknowable"),
        ))
    return examples


def iter_jsonl(examples: list[Example]) -> Iterator[str]:
    import json

    for example in examples:
        yield json.dumps(example.as_dict())


def validate_dataset(examples: list[Example], reg=None) -> list[str]:
    """Check the generator against the registry it claims to derive from.

    Synthetic data is only as good as its labels, and a generator that drifts
    from the registry produces confidently wrong ground truth — worse than no
    dataset. Run this before trusting any score computed from it.
    """
    reg = reg or registry()
    problems: list[str] = []
    labels = reg.labels()

    for example in examples:
        if example.expect_supported:
            if example.target not in labels:
                problems.append(
                    f"labelled supported but '{example.target}' is not detectable: {example.request!r}"
                )
                continue
            verdict = reg.assess(example.target, example.predicate or "present")
            if not verdict.ok:
                problems.append(
                    f"labelled supported but the registry refuses it "
                    f"({verdict.support.value}): {example.request!r}"
                )
        else:
            if example.target and example.target in labels:
                problems.append(
                    f"labelled a refusal but '{example.target}' IS detectable: {example.request!r}"
                )
    return problems
