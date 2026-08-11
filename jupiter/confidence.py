"""How confident Jupiter is that it correctly understood a configure request.

This is a DIFFERENT thing from two fields that already exist and already have
"confidence" in their name, and this module exists specifically so the three
never get merged into one number by accident:

  RuleDraft.confidence (jupiter/schema.py)
      An ALERT-TRIGGER THRESHOLD the rule itself carries: "only fire when the
      detector is at least this sure it saw the target." Chosen by the model as
      part of authoring the rule, shown to the user as "Sensitivity" in
      ios/Sources/AIConfigureView.swift — never as "confidence", because it
      isn't a measurement, it's a setting.

  Event.confidence / Explanation.confidence (core/events.py, jupiter/explain.py)
      DETECTION confidence: how sure the vision model was about one specific
      observation (a person, a fall, a phone). Already computed by the
      detector, already flows through to "Detector confidence — 94%" in an
      event's explanation. This module has nothing to do with vision.

  interpretation_confidence (here)
      How sure JUPITER is that it correctly turned a typed sentence into the
      rule(s) it proposed. This does not exist anywhere before this module —
      there was no signal for "the model probably misunderstood this" beyond
      a rule failing validation outright.

Not a number the model reports about itself. Self-reported LLM confidence is
well known to be poorly calibrated, and this repo's own eval harness exists
specifically to replace exactly that kind of trust with something measured
(see jupiter/eval/harness.py's docstring: "Nothing here estimates. Every
number comes from a call that actually happened"). This follows the same
rule: the score is a deterministic function of real, checkable properties of
the response — how much repair its JSON needed, how many of the fields the
model was supposed to fill in actually arrived well-formed, and whether the
request was only partially understood — never a value the model was asked to
invent.

Every point taken off is named. `ConfidenceReport.basis` is never empty when
the score is below 1.0, so a caller — and a user, several layers up in the
app — always has a REASON, not just a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Deductions. Fixed and documented rather than tuned, on purpose: a magic
# constant that quietly drifted would make this score exactly the kind of
# unaccountable number this module exists to avoid. Re-tune only against
# jupiter/eval/harness.py numbers, the same way model choice was — never by feel.
_JSON_NEEDED_REPAIR = 0.15
_UNCONSTRAINED_DECODING = 0.10
_FIELD_FIDELITY_WEIGHT = 0.30   # scaled by (1 - fidelity fraction)
_PARTIAL_UNDERSTANDING = 0.10

_FLOOR = 0.05  # something was still produced; never claim zero confidence in it

# What RuleDraft.from_dict() (jupiter/schema.py) expects for each required
# field, so fidelity can be checked against the RAW value the model sent —
# before RuleDraft quietly substitutes a default for anything missing or
# wrong-typed. Checking the raw value is the whole point: the default
# substitution is exactly the event this module needs to detect and explain,
# not paper over.
_STRING_FIELDS = ("target", "predicate", "secondary", "description", "zone", "severity", "notify")
_NUMBER_FIELDS = ("persist_s", "confidence", "cooldown_s")
_INT_FIELDS = ("min_count",)


@dataclass
class ConfidenceReport:
    """A score in [0, 1], and every reason it isn't 1.0."""

    score: float
    basis: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 2), "basis": list(self.basis)}


def _rule_field_fidelity(raw_rule: dict) -> tuple[float, list[str]]:
    """Fraction of this rule's required fields that arrived well-formed in the
    model's raw output, and which ones did not.

    A field counts as well-formed if it is present AND the right JSON type —
    not merely present, because a string field holding `null` or a number
    field holding a sentence is exactly the kind of near-miss that RuleDraft's
    defaulting exists to survive, and this function's job is to notice it
    rather than let the survival hide it.
    """
    if not isinstance(raw_rule, dict):
        return 0.0, ["the model's rule output wasn't a JSON object"]

    malformed: list[str] = []
    checks = (
        [(name, lambda v: isinstance(v, str)) for name in _STRING_FIELDS]
        + [(name, lambda v: isinstance(v, (int, float)) and not isinstance(v, bool))
           for name in _NUMBER_FIELDS]
        + [(name, lambda v: isinstance(v, int) and not isinstance(v, bool)) for name in _INT_FIELDS]
    )
    for name, is_valid in checks:
        if name not in raw_rule or not is_valid(raw_rule[name]):
            malformed.append(name)

    total = len(checks)
    fidelity = (total - len(malformed)) / total if total else 1.0
    return fidelity, malformed


def score(
    *,
    raw_rules: list,
    rules_accepted: int,
    fragments_unsupported: int,
    json_needed_repair: bool,
    constrained: bool,
) -> ConfidenceReport:
    """Score one interpretation. All inputs are things that actually happened
    during this request — see the module docstring for what each one is not.

    Args:
        raw_rules: the model's own rule objects, before RuleDraft defaulted
            anything — i.e. `payload.get("rules") or []`, exactly as the
            provider returned it.
        rules_accepted: how many rules survived validation
            (`len(interpretation.rules)`).
        fragments_unsupported: how many pieces of the request were moved to
            `unsupported` (`len(interpretation.unsupported)`).
        json_needed_repair: whether `extract_json()` had to fall back to
            fence-stripping or regex extraction rather than parsing the raw
            text directly.
        constrained: whether decoding was actually schema/grammar-constrained
            for this call (`ProviderResult.meta["constrained"]`).
    """
    value = 1.0
    basis: list[str] = []

    if json_needed_repair:
        value -= _JSON_NEEDED_REPAIR
        basis.append("the model's response needed extra parsing to find the JSON in it")

    if not constrained:
        value -= _UNCONSTRAINED_DECODING
        basis.append("the response wasn't schema-constrained, so its shape wasn't guaranteed")

    malformed_fields: set[str] = set()
    if raw_rules:
        fidelities = []
        for raw_rule in raw_rules:
            fidelity, malformed = _rule_field_fidelity(raw_rule)
            fidelities.append(fidelity)
            malformed_fields.update(malformed)
        mean_fidelity = sum(fidelities) / len(fidelities)
        if mean_fidelity < 1.0:
            value -= _FIELD_FIDELITY_WEIGHT * (1.0 - mean_fidelity)
            named = ", ".join(sorted(malformed_fields))
            basis.append(f"some fields arrived missing or the wrong type and were defaulted: {named}")

    if rules_accepted > 0 and fragments_unsupported > 0:
        value -= _PARTIAL_UNDERSTANDING
        basis.append("part of the request couldn't be matched to something the device can detect")

    return ConfidenceReport(score=max(_FLOOR, min(1.0, value)), basis=basis)
