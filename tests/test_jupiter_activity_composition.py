"""Activity requests (sleeping, eating, drinking, ...) that have no predicate of
their own on the capability sheet, composed from primitives that ARE available
rather than refused outright.

Root cause this guards against: `action.recognition` (eating/drinking/sleeping/
falling/smoking as predicates) is correctly declared `available=False` — no
action-recognition model ships on the device — but `pose.body` (lying_down,
sitting, standing) and `spatial.proximity` (near, holding) already ARE available,
and the rule engine already does generic temporal confirmation via `persist_s`
(core/rules.py, core/event_engine.py). Nothing needed to be built for "alert if
I'm sleeping" to work as "lying down and still for 2 minutes" — Jupiter's prompt
just never taught the model to reach for that composition, so every one of these
requests fell straight into `unsupported`, even though the device already had
everything required. See jupiter/prompt.py's rule 6/7 and the two new few-shots.

These tests are deterministic (registry assessment, and a scripted provider) —
they do not require a running Ollama server. The prompt actually teaching a live
model to produce this composition is verified separately by running
jupiter/interpret.interpret() against a real model (see docs/JUPITER.md's
benchmarking section); a scripted response is what a unit suite can pin down.
"""

from __future__ import annotations

import json

import pytest

from core.capabilities import Support, default_registry
from jupiter.interpret import interpret
from jupiter.prompt import system_prompt
from jupiter.providers import ScriptedProvider
from jupiter.schema import RuleDraft


@pytest.fixture()
def reg():
    return default_registry()


# --------------------------------------------------------------- registry mechanics
class TestComposedCapabilitiesAreAlreadyAvailable:
    """The primitives an activity approximation needs are real, registered
    capabilities today — this is what makes composing them (rather than
    training a new model) the correct fix."""

    def test_lying_down_present_is_fully_supported(self, reg):
        verdict = reg.assess("lying_down", "present")
        assert verdict.support is Support.SUPPORTED

    def test_person_holding_is_partial_not_rejected(self, reg):
        verdict = reg.assess("person", "holding")
        assert verdict.support is Support.PARTIAL
        assert verdict.ok

    def test_literal_sleeping_predicate_is_still_honestly_unavailable(self, reg):
        """The composition is an approximation, not a claim that true action
        recognition exists — asking for the literal predicate must still say no."""
        assert "sleeping" not in reg.predicates()
        assert "eating" not in reg.predicates()
        assert "drinking" not in reg.predicates()

    def test_a_long_persist_s_rule_on_lying_down_validates(self, reg):
        draft = RuleDraft(target="lying_down", predicate="present", persist_s=120.0)
        ok, reason, extra = draft.validate(reg)
        assert ok, reason

    def test_holding_a_drink_object_validates_with_a_caveat(self, reg):
        """PARTIAL support must surface as a caveat, not silently become SUPPORTED —
        the user should see 'this is an approximation', per RuleDraft.validate."""
        draft = RuleDraft(target="person", predicate="holding", secondary="bottle")
        ok, _, extra = draft.validate(reg)
        assert ok
        assert extra.get("caveat")


# --------------------------------------------------------------- prompt content
class TestPromptTeachesComposition:
    """Regression guard on the actual instructions, since the mechanism above
    working is necessary but not sufficient — Jupiter also has to be told to use it."""

    def test_system_prompt_explains_posture_composition(self, reg):
        text = system_prompt(reg)
        assert "lying_down" in text
        assert "persist_s" in text

    def test_system_prompt_caps_category_enumeration(self, reg):
        """Regression test for a real failure hit while building this: teaching
        the model to compose from a category (e.g. food) with no cap made it
        enumerate every member as a separate rule (11 for `food`), which blew
        past the response token budget and came back as unparseable JSON. See
        the "AT MOST 2" instruction — dropping it reintroduces that failure."""
        text = system_prompt(reg)
        assert "AT MOST 2" in text or "at most 2" in text.lower()

    def test_system_prompt_warns_against_lexical_near_misses(self, reg):
        """Regression guard for a live-model failure: asked for 'keys', qwen2.5:7b
        proposed a rule for 'keyboard' — a real, valid label that is not what was
        asked for — while ALSO correctly flagging 'keys' unsupported. The rule
        engine would have watched for a keyboard nobody asked about. Validation
        cannot catch this (keyboard is a legitimate label); only the prompt can."""
        text = system_prompt(reg)
        assert "keyboard" in text.lower() or "sounds similar" in text.lower()


# --------------------------------------------------------------- pipeline, scripted
class TestInterpretComposesActivityRequests:
    def test_sleeping_request_produces_a_lying_down_rule(self, reg):
        provider = ScriptedProvider(responses={
            "sleeping": {
                "understood": "Approximating sleep as lying down and still for a while.",
                "rules": [{
                    "target": "lying_down", "predicate": "present", "secondary": "",
                    "description": "Lying down and still for over 2 minutes "
                                    "(an approximation of sleep, not a confirmed one)",
                    "zone": "", "persist_s": 120.0, "min_count": 1, "confidence": 0.6,
                    "severity": "info", "cooldown_s": 300.0, "schedule": "", "notify": "push",
                }],
                "unsupported": [],
            },
        })
        outcome = interpret("Alert if I am sleeping.", provider=provider, reg=reg, use_grammar=False)
        assert outcome.ok
        assert len(outcome.interpretation.rules) == 1
        rule = outcome.interpretation.rules[0]
        assert rule.target == "lying_down"
        assert rule.persist_s == 120.0
        assert not outcome.interpretation.unsupported

    def test_drinking_request_produces_holding_rules_not_a_refusal(self, reg):
        provider = ScriptedProvider(responses={
            "drinking": {
                "understood": "Approximating drinking as holding a cup or bottle.",
                "rules": [
                    {"target": "person", "predicate": "holding", "secondary": "bottle",
                     "description": "Holding a bottle", "zone": "", "persist_s": 2.0,
                     "min_count": 1, "confidence": 0.6, "severity": "info",
                     "cooldown_s": 60.0, "schedule": "", "notify": "push"},
                    {"target": "person", "predicate": "holding", "secondary": "cup",
                     "description": "Holding a cup", "zone": "", "persist_s": 2.0,
                     "min_count": 1, "confidence": 0.6, "severity": "info",
                     "cooldown_s": 60.0, "schedule": "", "notify": "push"},
                ],
                "unsupported": [],
            },
        })
        outcome = interpret("Alert if I am drinking.", provider=provider, reg=reg, use_grammar=False)
        assert outcome.ok
        assert {r.secondary for r in outcome.interpretation.rules} == {"bottle", "cup"}

    def test_genuinely_undetectable_target_still_refuses(self, reg):
        """The fix must not make Jupiter promise things it cannot do — a request
        with no available composition at all still has to land in `unsupported`."""
        provider = ScriptedProvider(responses={
            "keys": json.dumps({
                "understood": "I can't watch for that.",
                "rules": [],
                "unsupported": [{"request": "Alert if you see keys.",
                                  "reason": "'keys' is not an object the vision model recognises."}],
            }),
        })
        outcome = interpret("Alert if you see keys.", provider=provider, reg=reg, use_grammar=False)
        assert outcome.ok
        assert outcome.interpretation.rules == []
        assert outcome.interpretation.unsupported
