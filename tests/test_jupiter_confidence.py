"""jupiter/confidence.py: interpretation confidence, and the three kinds of
"confidence" in this codebase that must never collapse into one number.

Unit tests here check the scoring function in isolation, with synthetic
inputs — every deduction path tested on its own, and combinations of them.
Integration tests at the bottom run the real interpret() pipeline (against
ScriptedProvider, deterministic, no network) to prove the wiring actually
reaches InterpretOutcome.meta the way cloud/app.py depends on.
"""

from __future__ import annotations

import json

import pytest

from jupiter.confidence import ConfidenceReport, _rule_field_fidelity, score
from jupiter.interpret import _extract_json_verbose, extract_json, interpret
from jupiter.providers import ScriptedProvider


class _ConstrainedScriptedProvider(ScriptedProvider):
    """ScriptedProvider reports no meta at all — accurate for what it is, a
    canned-response test double with no real decoding happening. The
    interpret() pipeline's actual providers (JupiterLocalProvider,
    AnthropicProvider) DO report `meta["constrained"]` truthfully when a
    schema was passed — verified live against qwen2.5:7b-instruct, which
    scored a clean response 1.0 for exactly this reason. This subclass
    stands in for that real behavior in a fast, offline test."""

    def complete(self, system, user, *, schema=None, grammar=None):
        result = super().complete(system, user, schema=schema, grammar=grammar)
        result.meta["constrained"] = True
        return result


def _rule(**overrides) -> dict:
    rule = {
        "target": "person", "predicate": "present", "secondary": "", "description": "someone",
        "zone": "", "persist_s": 3, "min_count": 1, "confidence": 0.6,
        "severity": "warning", "cooldown_s": 60, "schedule": "", "notify": "push",
    }
    rule.update(overrides)
    return rule


# ------------------------------------------------------------- field fidelity
class TestRuleFieldFidelity:
    def test_a_complete_well_typed_rule_is_perfect(self):
        fidelity, malformed = _rule_field_fidelity(_rule())
        assert fidelity == 1.0
        assert malformed == []

    def test_a_missing_field_is_caught(self):
        rule = _rule()
        del rule["confidence"]
        fidelity, malformed = _rule_field_fidelity(rule)
        assert fidelity < 1.0
        assert "confidence" in malformed

    def test_a_wrong_typed_field_is_caught_not_just_a_missing_one(self):
        """A field that's PRESENT but the wrong JSON type is exactly the case
        RuleDraft.from_dict() silently repairs — this must catch it too, not
        only outright absence."""
        fidelity, malformed = _rule_field_fidelity(_rule(confidence="pretty sure"))
        assert "confidence" in malformed

    def test_a_bool_does_not_count_as_a_number_or_int(self):
        """Python's bool is an int subclass; a rule field holding `true`
        instead of a number is still malformed, not an accidental pass."""
        fidelity, malformed = _rule_field_fidelity(_rule(min_count=True))
        assert "min_count" in malformed

    def test_non_dict_rule_scores_zero(self):
        fidelity, malformed = _rule_field_fidelity("not a rule")
        assert fidelity == 0.0
        assert malformed


# ------------------------------------------------------------------- scoring
class TestScore:
    def test_a_clean_single_rule_response_scores_perfectly(self):
        report = score(
            raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
            json_needed_repair=False, constrained=True,
        )
        assert report.score == 1.0
        assert report.basis == []

    def test_json_repair_is_penalized_and_explained(self):
        clean = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
                      json_needed_repair=False, constrained=True)
        repaired = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
                         json_needed_repair=True, constrained=True)
        assert repaired.score < clean.score
        assert any("parsing" in reason for reason in repaired.basis)

    def test_unconstrained_decoding_is_penalized_and_explained(self):
        constrained = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
                            json_needed_repair=False, constrained=True)
        unconstrained = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
                              json_needed_repair=False, constrained=False)
        assert unconstrained.score < constrained.score
        assert any("schema-constrained" in reason for reason in unconstrained.basis)

    def test_malformed_fields_are_penalized_proportionally_and_named(self):
        one_bad_field = score(
            raw_rules=[_rule(confidence="unsure")], rules_accepted=1, fragments_unsupported=0,
            json_needed_repair=False, constrained=True,
        )
        two_bad_fields = score(
            raw_rules=[_rule(confidence="unsure", cooldown_s=None)],
            rules_accepted=1, fragments_unsupported=0,
            json_needed_repair=False, constrained=True,
        )
        assert one_bad_field.score < 1.0
        assert two_bad_fields.score < one_bad_field.score
        assert any("confidence" in reason for reason in one_bad_field.basis)

    def test_partial_understanding_is_penalized_and_explained(self):
        report = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=1,
                       json_needed_repair=False, constrained=True)
        assert report.score < 1.0
        assert any("couldn't be matched" in reason for reason in report.basis)

    def test_a_clean_full_refusal_is_not_penalized(self):
        """The important non-obvious case: zero rules and some unsupported
        fragments is a confident, correct refusal — not a low-confidence
        result. Nothing here was malformed; the request just didn't match
        anything the device can detect. Conflating "declined cleanly" with
        "wasn't sure" would be exactly the wrong lesson for this score to
        teach — see jupiter/eval/harness.py's own refusal_accuracy metric,
        which treats a correct refusal as success, not failure."""
        report = score(raw_rules=[], rules_accepted=0, fragments_unsupported=2,
                       json_needed_repair=False, constrained=True)
        assert report.score == 1.0
        assert report.basis == []

    def test_deductions_stack(self):
        clean = score(raw_rules=[_rule()], rules_accepted=1, fragments_unsupported=0,
                      json_needed_repair=False, constrained=True)
        everything_wrong = score(
            raw_rules=[_rule(confidence="unsure")], rules_accepted=1, fragments_unsupported=1,
            json_needed_repair=True, constrained=False,
        )
        assert everything_wrong.score < clean.score
        assert len(everything_wrong.basis) >= 3

    def test_score_never_goes_below_the_floor(self):
        report = score(
            raw_rules=[_rule(target=None, predicate=None, confidence="no"), _rule(zone=1)],
            rules_accepted=1, fragments_unsupported=5,
            json_needed_repair=True, constrained=False,
        )
        assert report.score >= 0.05

    def test_score_never_exceeds_one(self):
        report = score(raw_rules=[], rules_accepted=0, fragments_unsupported=0,
                       json_needed_repair=False, constrained=True)
        assert report.score <= 1.0

    def test_multiple_rules_use_mean_fidelity(self):
        one_rule_bad = score(
            raw_rules=[_rule(confidence="unsure"), _rule(), _rule()],
            rules_accepted=3, fragments_unsupported=0,
            json_needed_repair=False, constrained=True,
        )
        all_rules_bad = score(
            raw_rules=[_rule(confidence="unsure")] * 3,
            rules_accepted=3, fragments_unsupported=0,
            json_needed_repair=False, constrained=True,
        )
        assert all_rules_bad.score < one_rule_bad.score < 1.0

    def test_as_dict_rounds_and_copies(self):
        report = ConfidenceReport(score=0.6666666, basis=["x"])
        d = report.as_dict()
        assert d["score"] == 0.67
        d["basis"].append("mutated")
        assert report.basis == ["x"]  # as_dict() doesn't leak a mutable reference


# --------------------------------------------------------- extract_json repair flag
class TestExtractJsonRepairFlag:
    def test_clean_json_needs_no_repair(self):
        payload, needed_repair = _extract_json_verbose('{"a": 1}')
        assert payload == {"a": 1}
        assert needed_repair is False

    def test_fenced_json_needed_repair(self):
        payload, needed_repair = _extract_json_verbose('```json\n{"a": 1}\n```')
        assert payload == {"a": 1}
        assert needed_repair is True

    def test_prose_wrapped_json_needed_repair(self):
        payload, needed_repair = _extract_json_verbose('Here you go: {"a": 1} hope that helps')
        assert payload == {"a": 1}
        assert needed_repair is True

    def test_unparseable_text_still_reports_repair_was_attempted(self):
        payload, needed_repair = _extract_json_verbose("no json at all")
        assert payload is None
        assert needed_repair is True

    def test_empty_text(self):
        assert _extract_json_verbose("") == (None, False)

    def test_public_extract_json_is_unchanged(self):
        """The pre-existing public function's contract must not regress —
        same tests that already existed for it, still passing."""
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert extract_json('Here you go: {"a": 1} hope that helps') == {"a": 1}
        assert extract_json("no json at all") is None
        assert extract_json("") is None


# ------------------------------------------------------------- integration
class TestInterpretSurfacesConfidence:
    def test_a_clean_response_reaches_meta_with_a_perfect_score(self):
        provider = _ConstrainedScriptedProvider(responses={
            "person": json.dumps({
                "understood": "watching for a person",
                "rules": [_rule()],
                "unsupported": [],
            })
        })
        outcome = interpret("alert me if a person appears", provider=provider)
        assert outcome.ok
        ic = outcome.meta["interpretation_confidence"]
        assert ic["score"] == 1.0
        assert ic["basis"] == []

    def test_a_malformed_field_lowers_the_score_the_caller_sees(self):
        provider = ScriptedProvider(responses={
            "person": json.dumps({
                "understood": "watching for a person",
                "rules": [_rule(confidence="pretty sure")],
                "unsupported": [],
            })
        })
        outcome = interpret("alert me if a person appears", provider=provider)
        ic = outcome.meta["interpretation_confidence"]
        assert ic["score"] < 1.0
        assert ic["basis"]

    def test_unparseable_response_has_no_interpretation_confidence(self):
        """No interpretation happened, so there is nothing to score — this
        must not appear as a fake 0.0 that looks like a measured low score."""
        provider = ScriptedProvider(config={"default": "not json at all, sorry"})
        outcome = interpret("alert me if a person appears", provider=provider)
        assert not outcome.ok
        assert "interpretation_confidence" not in outcome.meta

    def test_the_rule_generation_field_is_not_confused_with_interpretation_confidence(self):
        """RuleDraft.confidence (an alert-trigger threshold) and
        interpretation_confidence (how sure Jupiter is it understood the
        request) must both survive independently and disagree freely — a
        rule can carry a low alert threshold while Jupiter was fully
        confident about the interpretation, and vice versa."""
        provider = _ConstrainedScriptedProvider(responses={
            "person": json.dumps({
                "understood": "watching for a person",
                "rules": [_rule(confidence=0.2)],  # a very sensitive alert threshold
                "unsupported": [],
            })
        })
        outcome = interpret("alert me if a person appears", provider=provider)
        assert outcome.interpretation.rules[0].confidence == 0.2
        assert outcome.meta["interpretation_confidence"]["score"] == 1.0
