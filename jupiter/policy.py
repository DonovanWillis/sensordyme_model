"""Prompt -> typed monitoring policy. Deterministic; no code is generated or run.

A policy is the reviewed, typed form of one custom rule: what is watched
(`targets`), where (`zone`, always a user-drawn area), for how long
(`temporal`: dwell_s | sustain_s | count), what the device needs to do it
(`required_capabilities`), how loudly to say so, and — in plain language —
exactly what will and will NOT be monitored. The app shows it before the owner
activates anything (the existing draft -> Activate flow); the device runs the
compiled rule (`rule`, core/rules.py's shape) in its deterministic engine. No
language model is consulted per frame, ever.

Three entry points, one schema:

  refusal_for(text)        requests no interpreter may approximate (identity,
                           intent, emotion, sound, ...): refused up front with
                           core/taxonomy.py's reason, never downgraded to
                           "a person is present"
  parse(text, zones)       the deterministic parser for a bounded grammar of
                           area / duration requests ("alert me if someone stays
                           inside this marked area for longer than 10 seconds").
                           It claims a request only when it understood ALL of
                           it; anything else returns None and goes to the
                           language model (or the keyword fallback) as before
  policy_from_rule(rule)   the same typed schema applied to a rule from any
                           other source (a language model's proposal, the
                           app's reviewed copy). A rule that does not fit the
                           schema is refused with the reason
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from core.capabilities import Support, registry
from core.rules import DWELL_MAX_S, DWELL_MIN_S, DWELL_PREDICATES, KNOWN_PREDICATES, _zone_key
from core.taxonomy import REFUSALS

POLICY_SCHEMA_VERSION = 1

#: Perception kinds a label can come from (core/monitoring_state.py vocabulary).
_PERCEPTION = ("objects", "pose", "face", "hands")
#: What every policy needs, whatever it watches.
_BASE_CAPABILITIES = ("camera", "inference")

DEFAULT_SUSTAIN_S = 2.0
STATUS_READY = "ready"
STATUS_NEEDS_ZONE = "needs_zone"


class PolicyError(ValueError):
    """A rule that does not fit the typed policy schema."""


# ---------------------------------------------------------------------- the schema
@dataclass
class Policy:
    targets: list[str]
    condition: str
    zone: Optional[str]
    zone_drawn: bool
    temporal: dict
    required_capabilities: list[str]
    severity: str
    notify: str
    schedule: str
    rule: dict
    caveats: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return STATUS_NEEDS_ZONE if self.zone and not self.zone_drawn else STATUS_READY

    def explanation(self) -> dict:
        return explain(self)

    def as_dict(self) -> dict:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "targets": list(self.targets),
            "condition": self.condition,
            "zone": None if not self.zone else {"name": self.zone, "drawn": self.zone_drawn},
            "status": self.status,
            "temporal": dict(self.temporal),
            "required_capabilities": list(self.required_capabilities),
            "severity": self.severity,
            "notify": self.notify,
            "schedule": self.schedule or None,
            "explanation": self.explanation(),
        }


def _capabilities_for(labels: Iterable[str], reg) -> list[str]:
    kinds: set[str] = set()
    for label in labels:
        providers = [c for c in reg.providers_of_label(label) if c.available]
        kinds |= {c.kind for c in providers if c.kind in _PERCEPTION}
        if not providers:
            kinds.add("objects")
    return [*_BASE_CAPABILITIES, *sorted(kinds)]


def _drawn(zone: str, zones: Iterable[str]) -> bool:
    return bool(zone) and _zone_key(zone) in {_zone_key(z) for z in zones}


def _number(value, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default


def policy_from_rule(rule: dict, *, zones: Iterable[str] = (), reg=None) -> Policy:
    """The typed policy for one validated rule dict, or PolicyError.

    The same schema whatever produced the rule — the deterministic parser, a
    language model, or the app's edited copy — so a model cannot smuggle in a
    shape the parser would never produce (a dwell with no area, a condition
    the device does not evaluate, an object it cannot detect).
    """
    reg = reg or registry()
    if not isinstance(rule, dict):
        raise PolicyError("a rule must be an object")
    target = str(rule.get("target") or rule.get("detection") or "").strip().lower()
    condition = str(rule.get("predicate") or "present").strip().lower()
    secondary = str(rule.get("secondary") or "").strip().lower()
    zone = str(rule.get("zone") or "").strip()
    if not target:
        raise PolicyError("the rule names nothing to watch")
    if condition not in KNOWN_PREDICATES:
        raise PolicyError(f"'{condition}' is not a condition the device evaluates")
    for label in filter(None, (target, secondary)):
        verdict = reg.assess(label, "present")
        if not verdict.ok:
            raise PolicyError(verdict.reason)
    if condition == "in_zone" and not zone:
        raise PolicyError("'in_zone' needs an area the owner draws, and none was named")

    dwell = rule.get("dwell_s")
    if dwell not in (None, ""):
        dwell_s = _number(dwell, -1.0)
        if dwell_s <= 0:
            raise PolicyError("dwell_s must be a positive number of seconds")
        if not zone:
            raise PolicyError("a dwell needs an area to stay inside, and none was named")
        if condition not in DWELL_PREDICATES:
            raise PolicyError(f"a dwell cannot be combined with '{condition}'")
        temporal = {"dwell_s": max(DWELL_MIN_S, min(DWELL_MAX_S, dwell_s))}
    elif condition == "count":
        temporal = {"count": int(max(1, _number(rule.get("min_count"), 1))),
                    "sustain_s": _number(rule.get("persist_s"), DEFAULT_SUSTAIN_S)}
    else:
        temporal = {"sustain_s": _number(rule.get("persist_s"), DEFAULT_SUSTAIN_S)}

    targets = [target, *([secondary] if secondary else [])]
    caveats = [str(rule["caveat"])] if rule.get("caveat") else []
    return Policy(
        targets=targets, condition=condition, zone=zone or None, zone_drawn=_drawn(zone, zones),
        temporal=temporal, required_capabilities=_capabilities_for(targets, reg),
        severity=str(rule.get("severity") or "warning"), notify=str(rule.get("notify") or "push"),
        schedule=str(rule.get("schedule") or ""), rule=dict(rule), caveats=caveats,
    )


# ---------------------------------------------------------------------- explanation
def _seconds(value: float) -> str:
    value = float(value)
    if value >= 120 and value % 60 == 0:
        return f"{int(value // 60)} minutes"
    return f"{value:g} seconds" if value != 1 else "1 second"


def _noun(label: str) -> str:
    return "a person" if label == "person" else f"a {label}"


def explain(policy: Policy) -> dict:
    """Plain language: what will be monitored, and what will NOT."""
    target = policy.targets[0]
    who = _noun(target)
    where = f"inside the area '{policy.zone}'" if policy.zone else "anywhere in view"
    temporal = policy.temporal
    if "dwell_s" in temporal:
        summary = (f"Alert when {who} stays {where} for {_seconds(temporal['dwell_s'])} or longer "
                   f"without leaving.")
    elif policy.condition == "absent":
        summary = f"Alert when no {target} is {where} for {_seconds(temporal['sustain_s'])}."
    elif policy.condition == "count":
        summary = (f"Alert when at least {temporal['count']} of '{target}' are {where} "
                   f"(the count must hold for {_seconds(temporal['sustain_s'])}).")
    elif len(policy.targets) > 1:
        summary = (f"Alert when {who} is {policy.condition.replace('_', ' ')} {_noun(policy.targets[1])} "
                   f"{where}, held for {_seconds(temporal['sustain_s'])}.")
    else:
        summary = f"Alert when {who} is {where} for {_seconds(temporal['sustain_s'])}."
    summary += f" Severity: {policy.severity}; notification: {policy.notify}."

    monitors = [f"the object detector's '{label}' class" for label in policy.targets]
    if policy.zone:
        monitors.append(f"whether a subject's feet are inside the area you drew named '{policy.zone}'")
    if "dwell_s" in temporal:
        monitors.append("one subject's continuous time inside the area; a gap shorter than 1 second "
                        "(walking behind something) does not restart it, a longer one does")
    if policy.schedule:
        monitors.append(f"only between {policy.schedule}")
    monitors += policy.caveats

    not_monitored = []
    if target == "person":
        not_monitored += ["who the person is (no face recognition)",
                          "adults and children are not told apart"]
    not_monitored += ["why anyone is there (intent is never inferred)", "sound (no microphone)"]
    if policy.zone:
        not_monitored.append(f"anything outside '{policy.zone}'")
    if "dwell_s" in temporal:
        not_monitored.append("two people swapping places inside the area can be counted as one visit "
                             "(position matching, not identity)")
    not_monitored.append("anything the detector misses: heavily occluded, very far away, or in the dark "
                         "without infrared")
    status = None
    if policy.status == STATUS_NEEDS_ZONE:
        status = (f"Waiting: draw an area named '{policy.zone}' in the app. Until it exists this rule is "
                  f"inactive — it never fires anywhere else instead.")
    return {"summary": summary, "monitors": monitors, "does_not_monitor": not_monitored, "status": status}


# ---------------------------------------------------------------------- refusals
_REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("identity", re.compile(
        r"\b(recogni[sz]e[sd]?|recogni[sz]ing|recognition|identif(?:y|ies|ied|ying)|identity|who (?:is|it is)|"
        r"(?:un)?known (?:person|people|face|faces|visitor)|strangers?|familiar faces?|"
        r"(?:is|isn't|is not|it's|its) (?:me|my (?:husband|wife|son|daughter|mom|dad|mother|father|kid|child)))\b")),
    ("emotion", re.compile(
        r"\b(emotions?|emotional|angry|anger|sad|sadness|happy|upset|mood|scared|afraid|anxious|stressed|"
        r"frustrated|depressed|lonely)\b")),
    ("intent", re.compile(
        r"\b(intends?|intent|intention|suspicious(?:ly)?|up to no good|trying to steal|plans? to|planning to|"
        r"going to steal|wants? to steal|casing)\b")),
    ("audio", re.compile(
        r"\b(hears?|heard|hearing|sounds?|noises?|noisy|screams?|screaming|shouts?|shouting|yell(?:s|ing)?|"
        r"cry|cries|crying|bark(?:s|ing)?|voices?|talking|speaking|glass break(?:s|ing)?)\b")),
    ("intoxication", re.compile(r"\b(drunk|intoxicated|under the influence|high on)\b")),
    # Child-safety emergencies are medical claims too (M8): no model here has
    # been evaluated on infants, and baby monitoring's choking_hazard is only
    # "a small object near the mouth region", never choking detection.
    ("medical", re.compile(
        r"\b(seizures?|heart attack|stroke|not breathing|stops? breathing|breathing|diagnos\w*|unconscious|"
        r"chok(?:e|es|ed|ing)|suffocat\w*|smother\w*|sids|cot death|crib death|"
        r"face[- ]down|(?:rolls?|rolled|rolling|turns?|turned) (?:over )?(?:onto|on) (?:his|her|their|its|the) "
        r"(?:stomach|tummy|front|face)|fever|overheat\w*)\b")),
    ("repetition_window", re.compile(
        r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|several|multiple) times\b")),
    # M8.4: the driver module's experimental families -- the combined risk
    # score (high_risk, elevated_risk) and hand-motion tracking
    # (erratic_hand_movement) -- are recorded as observations, never offered
    # as alerts. No interpreter may turn a request for them back into an
    # alert, nor approximate one as "a person is present" or a raised hand.
    ("experimental_signal", re.compile(
        r"\b(high[- ]risk|elevated[- ]risk|risk (?:score|level|rating)s?|risky driving|"
        r"erratic (?:hand|hands|hand movements?|movements?|driving)|hand (?:movements?|motion)|"
        r"(?:hands?|arms?) (?:moving|flailing|waving around)|fidget\w*)\b")),
)

#: Refusals that are not core/taxonomy.py categories: (title, requires).
_EXTRA_REFUSALS = {
    "experimental_signal": (
        "An experimental driver observation (the combined risk score or erratic hand movement)",
        ("validation on real in-cabin footage first. These are recorded in Activity as experimental "
         "observations and never sound or notify; ask for an observable condition instead "
         "(someone in an area you draw, nobody in view)"),
    ),
}


def refusal_for(text: str) -> list[dict]:
    """`unsupported` entries for requests no interpreter may approximate.

    Matched on the request as typed. Over-refusing ("tell me if the dog is
    sad") costs one rephrase; under-refusing would turn "a stranger at the
    door" into "a person at the door", which looks like protection and is not.
    """
    lowered = " ".join(str(text or "").lower().split())
    entries = []
    for category, pattern in _REFUSAL_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        if category in _EXTRA_REFUSALS:
            title, requires = _EXTRA_REFUSALS[category]
            reason = f"{title} is not something SensorDyme can alert on ('{match.group(0)}')."
        else:
            detection = REFUSALS[category]
            title, requires = detection.title, detection.requires
            reason = f"{title} is not something SensorDyme can monitor ('{match.group(0)}')."
        entries.append({
            "request": str(text)[:500],
            "reason": reason,
            "requires": requires,
            "support": "unsupported",
            "alternatives": [],
            "category": category,
            "code": category,
        })
    return entries


# ---------------------------------------------------------------------- typed reasons
#: The closed set of `code`s every `unsupported` entry carries (M8.4), so the
#: app says WHY in one consistent headline instead of parsing prose. The
#: refusal categories above are codes too.
UNSUPPORTED_CODES = frozenset({
    *(category for category, _ in _REFUSAL_PATTERNS),
    "not_detectable",           # the detector does not recognise it (needs training data / a model / hardware)
    "not_supported",            # the device cannot evaluate it as asked
    "no_target",                # the request names nothing the camera could watch
    "needs_area",               # an area rule with no area
    "needs_second_object",      # a relation (near, holding) with one object
    "invalid_rule",             # a rule shape the typed policy schema refuses
    "not_beside_mode",          # a rule that came with a mode switch
    "interpreter_unavailable",  # nothing could interpret the request; nothing was guessed
})

_SUPPORT_CODES = {"needs_training_data": "not_detectable", "needs_model": "not_detectable",
                  "needs_hardware": "not_detectable"}


def unsupported_code(entry: dict) -> str:
    """The typed reason for one `unsupported` entry: its own `code` when it is
    a known one, else its refusal category, else what its `support` implies."""
    for key in ("code", "category"):
        value = str(entry.get(key) or "")
        if value in UNSUPPORTED_CODES:
            return value
    return _SUPPORT_CODES.get(str(entry.get("support") or ""), "not_supported")


def typed_unsupported(entries) -> list[dict]:
    """`entries` with a `code` from UNSUPPORTED_CODES on every one. Additive:
    nothing else in an entry changes, and a non-object entry is dropped."""
    return [{**entry, "code": unsupported_code(entry)} for entry in entries or [] if isinstance(entry, dict)]


# ---------------------------------------------------------------------- the parser
# M8.4 widened the phrasings below -- never the capabilities: every one still
# compiles to the same three conditions (in_zone, lingering, absent/count) on
# a label the registry serves. Lead-ins ("can you", "I want to know", "as soon
# as"), "there is/are", "shows up / is seen in", a duration without "for" or
# before the area, and "or longer" are read; anything else still goes on.
_TRIGGER = re.compile(
    r"^(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:(?:(?:alert|notify|tell|warn|ping|text|message|buzz)\s+me|let\s+me\s+know|i\s+want\s+to\s+know|"
    r"i\s+(?:want|need|would like)\s+(?:an?\s+)?(?:alert|notification)|"
    r"send\s+(?:me\s+)?an?\s+(?:alert|notification|push(?:\s+notification)?))\s+)?"
    r"(?:if|when|whenever|as\s+soon\s+as|any\s*time|every\s+time)\s+")
_PERSON = re.compile(
    r"^(?:someone|somebody|anyone|anybody|a person|any person|a man|a woman|a kid|a child|people|persons)\b")
_COUNT = re.compile(
    r"^(?:(?P<more>more than|over)\s+(?P<n1>\w+)|(?:at least)\s+(?P<n2>\w+)|(?P<n3>\w+)\s+or more)\s+"
    r"(?P<noun>[a-z]+)\b")
_ARTICLE = re.compile(r"^(?:a|an|any|the)\s+(?P<noun>[a-z]+(?: [a-z]+)?)\b")
_NOBODY = re.compile(r"^(?:no ?one|nobody|no person|no people)\b")
#: "there is / there are / there's" before a count, nobody or a thing.
_THERE = re.compile(r"^there(?:'s|\s+is|\s+are)\s+")
_AMOUNT = (r"(?P<n>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|"
           r"forty|forty-five|sixty|ninety)\s*(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)"
           r"(?:\s+or\s+(?:more|longer))?")
_DURATION = (r"(?:for\s+(?:(?:longer than|more than|over|at least)\s+)?|(?:longer than|more than|over|at least)\s+)"
             + _AMOUNT)
_PREP = r"(?:inside|in|into|within|at|on)"
_ZONE = r"(?P<zone>(?:the |this |that |my |our )?[a-z][a-z0-9 '_-]{0,40}?)"
_DWELL_VERB = re.compile(
    rf"^(?:stays?|remains?|lingers?|loiters?|waits?|hangs? around|is|are|spends? time)\s+{_PREP}\s+{_ZONE}\s+"
    rf"{_DURATION}$")
#: The duration before the area: "spends more than 5 minutes in the kitchen".
_DWELL_LEADING = re.compile(
    rf"^(?:stays?|remains?|lingers?|loiters?|waits?|hangs? around|spends?)\s+"
    rf"(?:(?:longer than|more than|over|at least)\s+)?{_AMOUNT}\s+{_PREP}\s+{_ZONE}$")
_ENTER_VERB = re.compile(
    rf"^(?:enters?\s+|(?:goes|walks|comes|steps|gets|go|walk|come|step|get)\s+(?:into|in)\s+|"
    rf"(?:shows?|turns?)\s+up\s+{_PREP}\s+|arrives?\s+(?:in|at)\s+|"
    rf"(?:is|are)\s+(?:seen|detected|spotted)\s+{_PREP}\s+|"
    rf"(?:is|are|appears?)\s+{_PREP}\s+){_ZONE}$")
#: After "there is a dog": just where.
_THERE_TAIL = re.compile(rf"^{_PREP}\s+{_ZONE}$")
_COUNT_TAIL = re.compile(rf"^(?:are\s+)?{_PREP}\s+{_ZONE}(?:\s+{_DURATION})?$")
_ABSENT_TAIL = re.compile(rf"^(?:is\s+)?{_PREP}\s+{_ZONE}(?:\s+{_DURATION})?$")
_EMPTY = re.compile(rf"^{_ZONE}\s+is empty(?:\s+{_DURATION})?$")

_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
          "forty-five": 45, "sixty": 60, "ninety": 90}
#: Words that mean "the area the owner marked", whichever it is.
_GENERIC_AREAS = frozenset({
    "area", "zone", "region", "marked area", "drawn area", "marked zone", "drawn zone", "selected area",
    "highlighted area", "marked region", "area i marked", "area i drew", "zone i drew",
})


def _count(word: Optional[str]) -> Optional[int]:
    if word is None:
        return None
    if word.isdigit():
        return int(word)
    return _WORDS.get(word)


def _seconds_of(match: re.Match) -> Optional[float]:
    n = match.group("n")
    if n is None:
        return None
    value = float(n) if re.match(r"^\d", n) else float(_WORDS.get(n, 0))
    unit = match.group("unit")
    if unit.startswith("m"):
        value *= 60
    elif unit.startswith("h"):
        value *= 3600
    return value if value > 0 else None


@dataclass
class ParseResult:
    policies: list[Policy]
    unsupported: list[dict]
    understood: str = ""

    def as_proposal(self) -> dict:
        """The proposal shape cloud/app.py validates (preset "" = custom rules)."""
        if self.policies:
            explanation = " ".join(p.explanation()["summary"] for p in self.policies)
        else:
            explanation = "Nothing was set up: " + "; ".join(u["reason"] for u in self.unsupported)
        return {
            "preset": "", "variant": "", "explanation": explanation,
            "rules": [dict(p.rule) for p in self.policies], "settings": {},
            "unsupported": list(self.unsupported), "source": "policy_parser",
        }


class _NotClaimedError(Exception):
    """The parser did not understand all of a clause; it claims nothing."""


def _clauses(text: str) -> list[str]:
    text = " ".join(text.lower().split()).strip(" .!?")
    parts = re.split(r"\s*[.;!?]\s+|,?\s+and\s+(?=(?:also\s+)?(?:alert|tell|notify|warn|let|if|when)\b)", text)
    return [p.strip(" ,.").removeprefix("also ").strip() for p in parts if p and p.strip(" ,.")]


def _label(noun: str, reg) -> tuple[Optional[str], Optional[dict]]:
    """(label, None) for a detectable noun, (None, unsupported) for one the
    registry declares but cannot serve, raise _NotClaimedError for an unknown word."""
    candidates = [noun, noun[:-1] if noun.endswith("s") else noun, noun[:-2] if noun.endswith("es") else noun]
    labels = reg.labels()
    for candidate in candidates:
        if candidate in labels:
            return candidate, None
    for candidate in candidates:
        declared = [c for c in reg.providers_of_label(candidate) if not c.available]
        if declared:
            verdict = reg.assess(candidate, "present")
            return None, {
                "request": noun, "reason": verdict.reason, "requires": verdict.requires,
                "support": verdict.support.value, "alternatives": list(verdict.alternatives),
                "code": "not_detectable",
            }
    raise _NotClaimedError(noun)


#: Words that mean the "zone" phrase carries a qualifier this grammar does not
#: read (a time, a condition): the request is not claimed rather than misread.
_ZONE_STOPWORDS = frozenset({
    "after", "before", "during", "between", "until", "while", "when", "if", "dark", "night", "tonight",
    "morning", "evening", "afternoon", "today", "weekend", "weekends", "and", "or", "but", "with",
    "without", "unless", "except", "for", "again",
})


def _zone(raw: str, zones: list[str], reg) -> tuple[str, Optional[str]]:
    """(zone name, note) for a zone phrase; _NotClaimedError when it names an object
    or carries a qualifier this grammar does not read."""
    phrase = re.sub(r"^(?:the|this|that|my|our)\s+", "", raw.strip())
    words = phrase.split()
    if not words or len(words) > 4 or _ZONE_STOPWORDS & set(words):
        raise _NotClaimedError(phrase)
    if phrase in _GENERIC_AREAS:
        if len(zones) == 1:
            return zones[0], None
        note = (f"{len(zones)} areas are drawn; this rule uses one named 'marked area' — rename the one "
                f"you mean, or say its name") if zones else None
        return "marked area", note
    for drawn in zones:
        if _zone_key(drawn) == _zone_key(phrase):
            return drawn, None
    name = re.sub(r"\s+(?:area|zone|region)$", "", phrase).strip()
    for drawn in zones:
        if _zone_key(drawn) == _zone_key(name):
            return drawn, None
    if name in reg.labels(available_only=False):
        # "on the couch" is about an object, not an area someone drew.
        raise _NotClaimedError(name)
    return name, None


def _rule(target: str, predicate: str, zone: str, description: str, *, sustain_s: float = DEFAULT_SUSTAIN_S,
          dwell_s: Optional[float] = None, min_count: int = 1) -> dict:
    rule = {
        "target": target, "predicate": predicate, "secondary": "", "description": description,
        "zone": zone, "persist_s": float(dwell_s if dwell_s is not None else sustain_s),
        "min_count": min_count, "confidence": 0.5, "severity": "warning", "cooldown_s": 60.0,
        "schedule": "", "notify": "push",
    }
    if dwell_s is not None:
        # persist_s = dwell_s too: firmware without per-subject dwell runs the
        # same rule as "the area occupied for dwell_s" rather than refusing it.
        rule["dwell_s"] = float(dwell_s)
    return rule


def _parse_clause(clause: str, zones: list[str], reg) -> tuple[Optional[dict], Optional[dict]]:
    """(rule, None) | (None, unsupported) for one clause; _NotClaimedError otherwise."""
    trigger = _TRIGGER.match(clause)
    if not trigger:
        raise _NotClaimedError(clause)
    body = clause[trigger.end():].strip()
    description = body[:1].upper() + body[1:]
    there = _THERE.match(body)
    if there:
        body = body[there.end():].strip()

    nobody = _NOBODY.match(body)
    if nobody:
        tail = _ABSENT_TAIL.match(body[nobody.end():].strip())
        if not tail:
            raise _NotClaimedError(clause)
        zone, _ = _zone(tail.group("zone"), zones, reg)
        sustain = _seconds_of(tail) if tail.group("n") else DEFAULT_SUSTAIN_S
        return _rule("person", "absent", zone, description, sustain_s=sustain or DEFAULT_SUSTAIN_S), None
    empty = _EMPTY.match(body)
    if empty:
        zone, _ = _zone(empty.group("zone"), zones, reg)
        sustain = _seconds_of(empty) if empty.group("n") else DEFAULT_SUSTAIN_S
        return _rule("person", "absent", zone, description, sustain_s=sustain or DEFAULT_SUSTAIN_S), None

    count = _COUNT.match(body)
    if count:
        n = _count(count.group("n1") or count.group("n2") or count.group("n3"))
        if n is None:
            raise _NotClaimedError(clause)
        threshold = n + 1 if count.group("more") else n
        noun = count.group("noun")
        target, unsupported = ("person", None) if noun in ("people", "persons") else _label(noun, reg)
        if unsupported:
            return None, unsupported
        tail = _COUNT_TAIL.match(body[count.end():].strip())
        if not tail:
            raise _NotClaimedError(clause)
        zone, _ = _zone(tail.group("zone"), zones, reg)
        sustain = (_seconds_of(tail) if tail.group("n") else None) or DEFAULT_SUSTAIN_S
        return _rule(target, "count", zone, description, sustain_s=sustain, min_count=threshold), None

    person = _PERSON.match(body)
    if person:
        target, rest = "person", body[person.end():].strip()
    else:
        article = _ARTICLE.match(body)
        if not article:
            raise _NotClaimedError(clause)
        noun = article.group("noun")
        # "a dog stays ..." -> the two-word capture swallowed the verb; retry with one word.
        first, _, second = noun.partition(" ")
        try:
            target, unsupported = _label(noun, reg)
            rest = body[article.end():].strip()
        except _NotClaimedError:
            target, unsupported = _label(first, reg)
            rest = (second + " " + body[article.end():]).strip()
        if unsupported:
            return None, unsupported

    if there:
        # "there is a dog in the kitchen": presence in the area, nothing else.
        bare = _THERE_TAIL.match(rest)
        if not bare:
            raise _NotClaimedError(clause)
        zone, _ = _zone(bare.group("zone"), zones, reg)
        return _rule(target, "in_zone", zone, description), None
    dwell = _DWELL_VERB.match(rest) or _DWELL_LEADING.match(rest)
    if dwell:
        zone, _ = _zone(dwell.group("zone"), zones, reg)
        seconds = _seconds_of(dwell)
        if seconds is None:
            raise _NotClaimedError(clause)
        return _rule(target, "lingering", zone, description, dwell_s=seconds), None
    enter = _ENTER_VERB.match(rest)
    if enter:
        zone, _ = _zone(enter.group("zone"), zones, reg)
        return _rule(target, "in_zone", zone, description), None
    raise _NotClaimedError(clause)


def parse(text: str, *, zones: Iterable[str] = (), reg=None) -> Optional[ParseResult]:
    """Typed policies for a request this grammar fully understands, else None.

    `zones` are the names of the areas drawn on the device; a rule naming one
    that is not drawn is still produced (status needs_zone) and runs once it is.
    Never returns a partial reading: one clause it cannot read and the whole
    request goes to the next interpreter instead.
    """
    reg = reg or registry()
    zones = [str(z) for z in zones if str(z).strip()]
    refusals = refusal_for(text)
    if refusals:
        return ParseResult(policies=[], unsupported=refusals)
    rules: list[dict] = []
    unsupported: list[dict] = []
    try:
        for clause in _clauses(text):
            rule, gap = _parse_clause(clause, zones, reg)
            if gap is not None:
                gap["request"] = clause
                unsupported.append(gap)
            elif rule is not None:
                rules.append(rule)
    except _NotClaimedError:
        return None
    if not rules and not unsupported:
        return None
    policies = []
    for rule in rules:
        try:
            policies.append(policy_from_rule(rule, zones=zones, reg=reg))
        except PolicyError as exc:  # the parser and the schema disagree: refuse, never guess
            unsupported.append({"request": rule["description"], "reason": str(exc), "requires": "",
                                "support": Support.UNSUPPORTED.value, "alternatives": [], "code": "invalid_rule"})
    return ParseResult(policies=policies, unsupported=unsupported)
