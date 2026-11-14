"""Why did I get this alert?

An alert that a person cannot interrogate is a notification they will eventually
mute. This turns one stored event into an explanation of *why the device
decided*, and it is built so the answer cannot be invented:

    event row  ->  facts (deterministic)  ->  sentence

The facts come from the database and the rule that fired. The language model is
optional and only ever paraphrases a fact list it was handed — it is never asked
what happened, because it does not know and would guess plausibly.

`explain()` therefore returns a usable explanation with no model at all. That is
the default path, and it is the one that runs when Jupiter is unreachable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Severity a user understands, derived from what the device recorded rather than
# assigned by a model. The mapping is deliberately conservative: "critical" is
# reserved for events the system genuinely confirms over time.
CRITICAL = "critical"
WARNING = "warning"
INFO = "info"

_SEVERITY_BY_STORED = {"loud": CRITICAL, "soft": WARNING, "log": INFO}

#: `events.source` for an alert from an AI-configured rule (services/runtime.py
#: `_emit_rule_alert` -> core/uplink.py push_event), as opposed to "module".
CUSTOM_RULE_SOURCE = "custom_rule"

# Event types whose consequence is severe enough to outrank the stored severity.
# Kept short and explicit; a device that shouts about everything gets ignored.
_ALWAYS_CRITICAL = frozenset({"fall", "on_floor", "safe_zone_exit", "zone_violation"})


@dataclass
class Fact:
    """One thing the device actually recorded, in the user's language."""

    label: str
    detail: str = ""

    def sentence(self) -> str:
        return f"{self.label}{f' — {self.detail}' if self.detail else ''}"


@dataclass
class Explanation:
    """A complete, grounded answer to "why did I get this?"."""

    headline: str
    severity: str
    when: str = ""
    facts: list[Fact] = field(default_factory=list)
    rule_summary: str = ""
    confidence: Optional[float] = None
    source: str = "facts"          # "facts" (deterministic) | provider name
    narrative: str = ""

    def as_dict(self) -> dict:
        return {
            "headline": self.headline,
            "severity": self.severity,
            "when": self.when,
            "facts": [{"label": f.label, "detail": f.detail} for f in self.facts],
            "rule": self.rule_summary,
            "confidence": self.confidence,
            "source": self.source,
            "narrative": self.narrative or self.plain_text(),
        }

    def plain_text(self) -> str:
        """The deterministic explanation. Always available, never invented."""
        parts = [self.headline]
        if self.facts:
            parts.append("SensorDyme saw: " + "; ".join(f.sentence() for f in self.facts) + ".")
        if self.rule_summary:
            parts.append(self.rule_summary)
        return " ".join(parts)


def humanise(event_type: str) -> str:
    """'phone_use_while_distracted' -> 'Phone use while distracted'."""
    return (event_type or "event").replace("_", " ").replace(":", " in ").capitalize()


def severity_for(event: dict) -> str:
    stored = str(event.get("severity") or "log").lower()
    base = _SEVERITY_BY_STORED.get(stored, INFO)
    event_type = str(event.get("type") or "").split(":")[0]
    if event_type in _ALWAYS_CRITICAL:
        return CRITICAL
    return base


def facts_from_event(event: dict, rule: Optional[dict] = None) -> list[Fact]:
    """Everything the device recorded, translated but not embellished.

    Only fields that are actually present produce a fact. An absent duration is
    simply not mentioned, rather than described as "brief" — the difference
    between reporting and narrating.
    """
    facts: list[Fact] = []

    signals = str(event.get("signals") or "").strip()
    if signals and str(event.get("source") or "") == CUSTOM_RULE_SOURCE:
        # A configured rule's signals are ONE sentence: the rule's own
        # description, or a count message (services/runtime.py sends
        # Alert.message()). Splitting it on commas turned "Bottle raised to the
        # mouth (an approximation of drinking, not a confirmed one)" into two
        # "observations", the second reading "Not a confirmed one)".
        facts.append(Fact(signals))
    elif signals:
        # Modules write these as "gaze_off + phone + hand_overlap".
        for signal in [s.strip() for s in signals.replace("+", ",").split(",") if s.strip()]:
            facts.append(Fact(humanise(signal)))

    # "Confidence", not "Detector confidence": it is the certainty that the
    # event's condition held, which for a measured rule (head angle, eye
    # closure) is not a detector's score at all — see core/confidence.py. Absent
    # when the device had no graded evidence, rather than shown as 0%.
    confidence = event.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 0:
        facts.append(Fact("Confidence", f"{round(float(confidence) * 100)}%"))

    duration = event.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        facts.append(Fact("Held for", f"{duration:.0f} seconds"))

    if rule:
        zone = rule.get("zone")
        if zone:
            facts.append(Fact("Area", str(zone)))

    model = _model_fact(event.get("model"))
    if model:
        facts.append(Fact("Model", model))
    # M8.4: an experimental observation (the driver module's risk score, hand
    # motion) is recorded, never alerted; the explanation says so rather than
    # reading like an alert. Stored as true or 1; absent on older rows.
    experimental = event.get("experimental")
    if experimental is True or (isinstance(experimental, int) and not isinstance(experimental, bool)
                                and experimental == 1):
        facts.append(Fact("Experimental observation", "not an alert"))
    return facts


def _model_fact(raw) -> str:
    """"damoyolo_tinynasL20_T.hef (sha256 e69136d3ebf3)" from a stored identity.

    Which exact file produced the event — the question an investigation starts
    with. Empty for events that predate model tracking.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return ""
    if not isinstance(raw, dict):
        return ""
    detector = raw.get("detector")
    if not isinstance(detector, dict) or not detector.get("name"):
        return ""
    sha = detector.get("sha256")
    return f"{detector['name']} (sha256 {sha[:12]})" if isinstance(sha, str) and sha else str(detector["name"])


def rule_summary(rule: Optional[dict]) -> str:
    """Why this crossed the threshold, in the rule's own configured numbers."""
    if not rule:
        return ""
    persist = rule.get("persist_s")
    parts = []
    if isinstance(persist, (int, float)) and persist > 0:
        parts.append(f"your rule alerts after {persist:.0f} seconds")
    schedule = rule.get("schedule")
    if schedule:
        parts.append(f"and only between {schedule.replace('-', ' and ')}")
    if not parts:
        return ""
    return "You were alerted because " + " ".join(parts) + "."


def explain(
    event: dict,
    rule: Optional[dict] = None,
    *,
    provider=None,
    reg=None,
) -> Explanation:
    """Explain one event. Works with no model; a model only rephrases the facts."""
    severity = severity_for(event)
    facts = facts_from_event(event, rule)
    confidence = event.get("confidence") if isinstance(event.get("confidence"), (int, float)) else None

    when = ""
    timestamp = event.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        try:
            when = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().strftime(
                "%-I:%M %p"
            )
        except (ValueError, TypeError):
            when = ""

    explanation = Explanation(
        headline=humanise(str(event.get("type") or "")),
        severity=severity,
        when=when,
        facts=facts,
        rule_summary=rule_summary(rule),
        confidence=round(float(confidence), 2) if confidence else None,
    )

    # Critical events are explained by the deterministic text, full stop.
    #
    # Measured, on this exact event: qwen2.5-1.5b was handed the facts "Rapid
    # drop", "Stillness" and "Detector confidence — 94%" and wrote "a rapid drop
    # in detector confidence" — merging two facts into a claim that is simply
    # untrue. On a fall alert. The plain sentence is correct and clear, and a
    # fluent wrong one is worse than a blunt right one when someone's parent is
    # on the floor.
    #
    # Rephrasing is still worth having for routine events, where readability
    # matters more than precision and the cost of a clumsy sentence is nil.
    #
    # With no recorded facts there is nothing to rephrase: the model was handed
    # "Observations:" and an empty list, and whatever it wrote back ("A stranger
    # tried the door handle") was shown as the explanation. The headline alone
    # is the honest answer then.
    if provider is None or severity == CRITICAL or not facts:
        return explanation

    # The model is handed the facts and asked only to join them into a sentence.
    # It is never told to explain the event, because it does not have the event —
    # that framing is the entire difference between grounded and invented.
    try:
        result = provider.complete(
            "You rewrite a list of observations into ONE plain sentence for a "
            "homeowner. Use ONLY the observations given. Add no detail, no cause, "
            "no speculation. No more than 30 words.",
            "Observations:\n" + "\n".join(f"- {f.sentence()}" for f in facts)
            + (f"\n- {explanation.rule_summary}" if explanation.rule_summary else ""),
        )
        narrative = (result.text or "").strip().strip('"')
        # A model that ignored the instruction and wrote an essay is discarded
        # rather than shown; the deterministic text is always good enough.
        if narrative and len(narrative) <= 300:
            explanation.narrative = narrative
            explanation.source = result.provider
    except Exception:  # noqa: BLE001, S110 - an explanation must never fail the request; deterministic text stands
        pass
    return explanation


def parse_rule_for_event(
    ai_rules_json: Optional[str], event_type: str, signals: Optional[str] = None
) -> Optional[dict]:
    """Find the stored rule an event came from, if exactly one can be named.

    An AI-rule event's type is only the rule's TARGET (services/runtime.py
    `_emit_rule_alert`), and one target often has several rules: "someone
    leaves" + "someone at the pool" are both `person`, the drinking
    composition is two `person` rules. Taking the first match explained a pool
    alert with the other rule's 23:00-06:00 schedule — a confident, invented
    reason. So: one candidate is the answer; several are narrowed by the
    event's `signals`, which for a configured rule is the rule's own
    description; still ambiguous means no rule, and the explanation makes no
    claim about thresholds rather than a wrong one.
    """
    try:
        rules = json.loads(ai_rules_json or "{}").get("rules") or []
    except (ValueError, TypeError, AttributeError):
        return None
    base = (event_type or "").split(":")[0]
    matches = [
        rule for rule in rules
        if isinstance(rule, dict) and (rule.get("target") == base or rule.get("detection") == base)
    ]
    if len(matches) == 1:
        return matches[0]
    said = str(signals or "").strip()
    if said:
        described = [r for r in matches if str(r.get("description") or "").strip() == said]
        if len(described) == 1:
            return described[0]
    return None
