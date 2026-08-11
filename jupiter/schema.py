"""The rule schema Jupiter must produce, derived from the capability registry.

Nothing here hardcodes what SensorDyme can watch for. The vocabulary — every
label, every predicate — is read from `core.capabilities.registry()` at call
time, so registering a can-detection model tomorrow widens Jupiter's schema with
no edit to this file. A duplicated list would drift from the registry within a
release and start promising things the device cannot do, which is the exact
failure the capability layer exists to prevent.

The schema is the contract in three places at once:

  * the JSON Schema handed to providers that support structured output;
  * the GBNF grammar (jupiter/grammar.py) that makes malformed output
    impossible for llama.cpp-family runtimes; and
  * `RuleDraft.validate`, which re-checks everything locally, because a
    provider may honour neither.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.capabilities import Support, registry

# Bounds mirror cloud/app.py's `_validate_proposal` clamps. Jupiter proposes;
# the server still clamps. Stating them here lets the model land inside the
# range rather than have every value silently corrected afterwards.
PERSIST_MIN, PERSIST_MAX = 0.5, 3600.0
CONFIDENCE_MIN, CONFIDENCE_MAX = 0.2, 0.95
COOLDOWN_MIN, COOLDOWN_MAX = 10.0, 86400.0
COUNT_MIN, COUNT_MAX = 1, 100

#: What the model emits when nothing in the vocabulary fits the request.
NO_TARGET = "none"

SEVERITIES = ("info", "warning", "critical")
NOTIFY_KINDS = ("push", "sms", "silent")


@dataclass
class RuleDraft:
    """One rule as Jupiter proposes it, before the server validates it."""

    target: str
    predicate: str = "present"
    secondary: str = ""
    description: str = ""
    zone: str = ""
    persist_s: float = 3.0
    min_count: int = 1
    confidence: float = 0.6
    severity: str = "warning"
    cooldown_s: float = 60.0
    schedule: str = ""
    notify: str = "push"

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "predicate": self.predicate,
            "secondary": self.secondary,
            "description": self.description or f"{self.predicate} {self.target}".strip(),
            "zone": self.zone,
            "persist_s": self.persist_s,
            "min_count": self.min_count,
            "confidence": self.confidence,
            "severity": self.severity,
            "cooldown_s": self.cooldown_s,
            "schedule": self.schedule,
            "notify": self.notify,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RuleDraft":
        """Build from arbitrary provider output. Never raises.

        Model output is untrusted input. A wrong type here must produce a
        rejected rule, not a stack trace inside an API request.
        """
        def number(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def text(key: str) -> str:
            value = raw.get(key)
            return value.strip().lower() if isinstance(value, str) else ""

        try:
            min_count = int(raw.get("min_count", 1))
        except (TypeError, ValueError):
            min_count = 1

        description = raw.get("description")
        return cls(
            target=text("target"),
            predicate=text("predicate") or "present",
            secondary=text("secondary"),
            description=description.strip() if isinstance(description, str) else "",
            zone=text("zone"),
            persist_s=number("persist_s", 3.0),
            min_count=min_count,
            confidence=number("confidence", 0.6),
            severity=text("severity") or "warning",
            cooldown_s=number("cooldown_s", 60.0),
            schedule=raw.get("schedule", "") if isinstance(raw.get("schedule"), str) else "",
            notify=text("notify") or "push",
        )

    # ------------------------------------------------------------------ validation
    def validate(self, reg=None) -> tuple[bool, str, dict]:
        """Check against the live capability registry and clamp numeric fields.

        Returns (ok, reason, extra) where `extra` carries what the capability
        layer said, so a rejection can tell the user what would enable it rather
        than only that it failed.
        """
        reg = reg or registry()

        if not self.target:
            return False, "no target object was named", {}

        verdict = reg.assess(self.target, self.predicate)
        if not verdict.ok:
            return False, verdict.reason, {
                "support": verdict.support.value,
                "requires": verdict.requires,
                "alternatives": list(verdict.alternatives),
            }

        # These predicates are meaningless without their second half, and the
        # registry's per-label assessment above cannot see that — it only
        # checked "person" and "in_zone" independently, both of which pass on
        # their own. Left unchecked, a model that names the predicate but drops
        # the zone/secondary produces a rule that VALIDATES, looks active in
        # the app, and quietly means something much broader than asked: an
        # `in_zone` rule with no zone skips the zone check entirely in
        # core/rules.py and fires on a person ANYWHERE in frame — worse than
        # the near/holding case, which at least fails closed (never fires) per
        # RuleEngine._evaluate. Observed live: "alert if a vehicle enters the
        # driveway" produced target=person, predicate=in_zone, zone="".
        if self.predicate == "in_zone" and not self.zone:
            return False, "'in_zone' needs a zone to watch, and none was named", {
                "support": Support.UNSUPPORTED.value,
                "requires": "a zone the user draws in the app",
                "alternatives": [],
            }
        if self.predicate in ("near", "holding", "near_mouth") and not self.secondary:
            return False, f"'{self.predicate}' needs a second object, and none was named", {
                "support": Support.UNSUPPORTED.value,
                "requires": "a second object for the relation",
                "alternatives": [],
            }

        # A relational rule is only as good as its second object.
        if self.secondary:
            secondary_verdict = reg.assess(self.secondary, "present")
            if not secondary_verdict.ok:
                return False, secondary_verdict.reason, {
                    "support": secondary_verdict.support.value,
                    "requires": secondary_verdict.requires,
                    "alternatives": list(secondary_verdict.alternatives),
                }

        self.persist_s = _clamp(self.persist_s, PERSIST_MIN, PERSIST_MAX)
        self.confidence = _clamp(self.confidence, CONFIDENCE_MIN, CONFIDENCE_MAX)
        self.cooldown_s = _clamp(self.cooldown_s, COOLDOWN_MIN, COOLDOWN_MAX)
        self.min_count = int(_clamp(self.min_count, COUNT_MIN, COUNT_MAX))
        if self.severity not in SEVERITIES:
            self.severity = "warning"
        if self.notify not in NOTIFY_KINDS:
            self.notify = "push"
        if not _valid_schedule(self.schedule):
            self.schedule = ""

        extra = {}
        if verdict.support is Support.PARTIAL:
            extra["caveat"] = verdict.reason
        return True, "", extra


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _valid_schedule(text: str) -> bool:
    """'23:00-06:00' or empty. A malformed schedule is dropped rather than
    guessed at: a rule that silently watches the wrong hours is worse than one
    that watches all of them."""
    if not text:
        return True
    try:
        start, end = text.split("-", 1)
        for part in (start, end):
            hour, minute = (int(v) for v in part.strip().split(":", 1))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                return False
    except (ValueError, TypeError):
        return False
    return True


# --------------------------------------------------------------------- JSON Schema
def schema_for_registry(reg=None) -> dict:
    """JSON Schema for a Jupiter response, with the live vocabulary inlined.

    Labels and predicates become `enum`s so a provider with native structured
    output cannot emit an object the device has never heard of. The enum is
    generated, never written down: it is exactly what the registry reports today.
    """
    reg = reg or registry()
    # NO_TARGET is an escape hatch, and it is load-bearing. Enum-constraining
    # `target` to detectable labels removes the model's ability to say "that is
    # not on the list": asked to watch for a can it must emit SOME valid label,
    # and it picks an unrelated one. Measured on qwen2.5-1.5b: "alert me if a
    # can appears" produced target=person with the description "Cans appear in
    # the area of interest". Constrained decoding had turned an honest refusal
    # into a confident wrong answer — worse than the unconstrained failure,
    # which validation caught. A legal token for "cannot" fixes the cause.
    labels = sorted(reg.labels()) + [NO_TARGET]
    predicates = sorted(reg.predicates())

    rule = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": labels},
            "predicate": {"type": "string", "enum": predicates},
            "secondary": {"type": "string", "enum": ["", *labels]},
            "description": {"type": "string"},
            "zone": {"type": "string"},
            "persist_s": {"type": "number", "minimum": PERSIST_MIN, "maximum": PERSIST_MAX},
            "min_count": {"type": "integer", "minimum": COUNT_MIN, "maximum": COUNT_MAX},
            "confidence": {"type": "number", "minimum": CONFIDENCE_MIN, "maximum": CONFIDENCE_MAX},
            "severity": {"type": "string", "enum": list(SEVERITIES)},
            "cooldown_s": {"type": "number", "minimum": COOLDOWN_MIN, "maximum": COOLDOWN_MAX},
            "schedule": {"type": "string"},
            "notify": {"type": "string", "enum": list(NOTIFY_KINDS)},
        },
        "required": ["target", "predicate", "secondary", "description", "zone", "persist_s",
                     "min_count", "confidence", "severity", "cooldown_s", "schedule", "notify"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "understood": {
                "type": "string",
                "description": "One sentence restating the request in the user's own terms.",
            },
            "rules": {"type": "array", "items": rule},
            "unsupported": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "request": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["request", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["understood", "rules", "unsupported"],
        "additionalProperties": False,
    }


RULE_SCHEMA = schema_for_registry


@dataclass
class Interpretation:
    """Jupiter's full answer for one request, after local validation."""

    understood: str = ""
    rules: list[RuleDraft] = field(default_factory=list)
    unsupported: list[dict] = field(default_factory=list)
    raw: Optional[dict] = None

    def as_proposal(self, preset: str = "") -> dict:
        """The shape cloud/app.py's `_validate_proposal` already expects, so
        Jupiter plugs into the existing pipeline rather than beside it."""
        return {
            "preset": preset,
            "variant": "",
            "explanation": self.understood,
            "rules": [r.as_dict() for r in self.rules],
            "settings": {},
            "unsupported": self.unsupported,
        }


# Too generic to ever match a capability; kept out of the free-text scan below
# so "alert if you see a package" doesn't waste a lookup on "if"/"see"/"a".
_TARGET_GUESS_STOPWORDS = frozenset({
    "a", "an", "the", "my", "me", "someone", "something", "if", "when", "for",
    "of", "on", "in", "at", "to", "and", "or", "is", "are", "was", "were",
    "watch", "monitor", "alert", "notify", "detect", "notice", "check", "see",
    "there", "any", "please", "you",
})


def _target_guess_candidates(description: str) -> list[str]:
    """Candidate object names pulled from a rule's free-text description.

    Used only once the model has already said `target: "none"` -- the enum
    forces that whenever the real word isn't a detectable label (see
    schema_for_registry's NO_TARGET comment), which is exactly the case where
    the word the user actually said would otherwise be thrown away. It
    survives in `description`, which the prompt asks the model to fill with a
    short restatement ("monitor for a can") even on a refusal. Bigrams come
    first since some real labels are two words ("cell phone", "wine glass").
    """
    words = [w for w in re.findall(r"[a-z']+", description.lower()) if len(w) > 2]
    words = [w for w in words if w not in _TARGET_GUESS_STOPWORDS]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return bigrams + words


def _assess_no_target(description: str, predicate: str, reg) -> dict:
    """Best-effort alternatives/reason for a target:"none" rule.

    Without this, every "none" refusal carried the same generic text and an
    always-empty `alternatives`, even for requests the registry actually has a
    specific, useful answer for (e.g. "keys" -> objects.small_personal's real
    explanation of what an open-vocabulary detector would take). Scans
    candidate words through the SAME `reg.assess` a named target would get; it
    does not guess meaning, only re-runs the existing capability lookup
    (label match, category expansion, declared-but-unavailable capability, or
    nearest-spelling) against words the model already wrote down. A match
    against a real registered capability (`verdict.capabilities`) wins
    outright and stops the scan; a weaker spelling-only match is kept only
    until a stronger one turns up.
    """
    fallback = {
        "reason": "no object SensorDyme can detect matches this request",
        "requires": "a model trained on the thing you described",
        "support": Support.NEEDS_TRAINING_DATA.value,
        "alternatives": [],
    }
    best = None
    for candidate in _target_guess_candidates(description):
        verdict = reg.assess(candidate, predicate)
        if verdict.support is Support.UNSUPPORTED:
            continue
        if verdict.capabilities:
            best = verdict
            break
        if best is None and (verdict.alternatives or verdict.requires):
            best = verdict
    if best is None:
        return fallback
    return {
        "reason": best.reason,
        "requires": best.requires or fallback["requires"],
        "support": best.support.value,
        "alternatives": list(best.alternatives),
    }


def parse_response(payload: Any, reg=None) -> Interpretation:
    """Turn raw provider output into a validated Interpretation.

    Every rule the registry rejects moves to `unsupported` carrying the specific
    gap. This runs regardless of which provider produced the payload — the point
    is that Jupiter, Anthropic and a hand-written stub are all equally untrusted.
    """
    reg = reg or registry()
    if not isinstance(payload, dict):
        return Interpretation(understood="", rules=[], unsupported=[{
            "request": "(response)",
            "reason": "the model did not return an object",
        }])

    understood = payload.get("understood")
    result = Interpretation(
        understood=understood.strip() if isinstance(understood, str) else "",
        raw=payload,
    )

    for entry in payload.get("unsupported") or []:
        if isinstance(entry, dict) and entry.get("request"):
            result.unsupported.append({
                "request": str(entry.get("request", "")),
                "reason": str(entry.get("reason", "not supported")),
                "requires": str(entry.get("requires", "") or "additional model or training data"),
                "support": str(entry.get("support", "") or "unsupported"),
                "alternatives": list(entry.get("alternatives") or []),
            })

    for entry in payload.get("rules") or []:
        if not isinstance(entry, dict):
            continue
        draft = RuleDraft.from_dict(entry)
        if draft.target == NO_TARGET:
            guess = _assess_no_target(draft.description, draft.predicate, reg)
            result.unsupported.append({
                "request": draft.description or "(request)",
                **guess,
            })
            continue
        ok, reason, extra = draft.validate(reg)
        if ok:
            result.rules.append(draft)
            continue
        result.unsupported.append({
            "request": draft.description or f"{draft.predicate} {draft.target}".strip(),
            "reason": reason,
            "requires": extra.get("requires", "additional model or training data"),
            "support": extra.get("support", "unsupported"),
            "alternatives": extra.get("alternatives", []),
        })
    return result
