"""M6: prompt -> typed policy (jupiter/policy.py), with no model involved.

The deterministic parser is exact for the requests it reads and claims nothing
it did not read completely; unsupported requests fail with core/taxonomy.py's
reason and are never downgraded to plain presence; the same typed schema
judges a rule from any other interpreter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.rules import ConfiguredRule, validate_rule_set
from jupiter import policy as policy_module
from jupiter.policy import PolicyError, parse, policy_from_rule, refusal_for

DWELL_REQUEST = "Alert me if someone stays inside this marked area for longer than 10 seconds"


def test_dwell_in_a_drawn_area():
    result = parse(DWELL_REQUEST, zones=["front door"])
    assert result is not None and result.unsupported == []
    (policy,) = result.policies
    data = policy.as_dict()
    assert data["targets"] == ["person"]
    assert data["zone"] == {"name": "front door", "drawn": True}
    assert data["status"] == "ready"
    assert data["temporal"] == {"dwell_s": 10.0}
    assert data["required_capabilities"] == ["camera", "inference", "objects"]
    assert data["severity"] == "warning"
    explanation = data["explanation"]
    assert "10 seconds" in explanation["summary"] and "front door" in explanation["summary"]
    assert any("face recognition" in item for item in explanation["does_not_monitor"])
    assert any("intent" in item for item in explanation["does_not_monitor"])
    # The compiled rule is the device's shape, and the device accepts it.
    rule = policy.rule
    assert rule["predicate"] == "lingering" and rule["dwell_s"] == 10.0 and rule["persist_s"] == 10.0
    assert validate_rule_set([rule], label_ok=lambda label: label == "person") == []
    assert ConfiguredRule.from_dict(0, rule).dwell_s == 10.0


def test_an_undrawn_area_waits_for_one():
    (policy,) = parse(DWELL_REQUEST, zones=[]).policies
    data = policy.as_dict()
    assert data["status"] == "needs_zone" and data["zone"] == {"name": "marked area", "drawn": False}
    assert "draw an area" in data["explanation"]["status"]
    # Named areas are matched the way the device matches them (case, spacing).
    (named,) = parse("tell me if a person stays in the Front Door for 2 minutes", zones=["front_door"]).policies
    assert named.zone == "front_door" and named.zone_drawn and named.temporal == {"dwell_s": 120.0}


@pytest.mark.parametrize(("request_text", "category"), [
    ("alert me if a stranger stays at the door for 10 seconds", "identity"),
    ("tell me if someone I don't recognize enters the yard", "identity"),
    ("alert me when an unknown person enters the driveway", "identity"),
    ("notify me if someone looks suspicious near the car", "intent"),
    ("tell me if my son is sad", "emotion"),
    ("tell me if the baby is crying", "audio"),
    ("alert me if the dog barks", "audio"),
    ("alert me if grandpa stops breathing", "medical"),
    ("alert me if someone enters the yard 3 times", "repetition_window"),
])
def test_unsupported_requests_fail_with_the_taxonomy_reason(request_text, category):
    result = parse(request_text, zones=["door", "yard", "driveway"])
    assert result is not None and result.policies == []
    assert [u["category"] for u in result.unsupported] == [category]
    assert result.unsupported[0]["requires"]
    proposal = result.as_proposal()
    assert proposal["rules"] == [], "an unsupported request must never become a presence rule"
    assert refusal_for(request_text)


def test_an_object_the_device_cannot_detect_is_refused_not_substituted():
    result = parse("alert me if a package stays on the porch for 30 seconds", zones=["porch"])
    assert result.policies == []
    assert "package" in result.unsupported[0]["reason"]


def test_multiple_policies_in_one_request():
    result = parse(
        "Alert me if someone stays in the driveway for more than 2 minutes. "
        "Tell me if a dog enters the garden, and alert me if more than 3 people are in the lobby",
        zones=["Driveway", "lobby"],
    )
    kinds = [(p.targets[0], p.condition, p.zone, p.status, p.temporal) for p in result.policies]
    assert kinds == [
        ("person", "lingering", "Driveway", "ready", {"dwell_s": 120.0}),
        ("dog", "in_zone", "garden", "needs_zone", {"sustain_s": 2.0}),
        ("person", "count", "lobby", "ready", {"count": 4, "sustain_s": 2.0}),
    ]


@pytest.mark.parametrize("request_text", [
    "tell me if there is an intruder at the door",      # an unknown word: the model's job
    "tell me if a dog is on the couch",                  # an object relation, not an area
    "tell me if someone is in the kitchen after dark",   # a qualifier this grammar does not read
    "alert me if a person appears",                      # no area, no duration
    "watch my baby in the crib",                         # not a trigger sentence
])
def test_the_parser_claims_nothing_it_did_not_read_completely(request_text):
    assert parse(request_text) is None


def test_model_output_is_held_to_the_same_schema():
    good = {"target": "person", "predicate": "lingering", "zone": "porch", "dwell_s": 30}
    assert policy_from_rule(good, zones=["porch"]).temporal == {"dwell_s": 30.0}
    for bad, why in [
        ({"target": "person", "predicate": "lingering", "dwell_s": 10}, "area"),
        ({"target": "person", "predicate": "enters", "zone": "porch"}, "condition"),
        ({"target": "person", "predicate": "absent", "zone": "porch", "dwell_s": 10}, "dwell"),
        ({"target": "person", "predicate": "lingering", "zone": "porch", "dwell_s": "long"}, "dwell_s"),
        ({"target": "package", "predicate": "present"}, "package"),
        ({"target": "", "predicate": "present"}, "nothing"),
        ({"target": "person", "predicate": "in_zone"}, "area"),
    ]:
        with pytest.raises(PolicyError, match=why):
            policy_from_rule(bad)


def test_no_code_is_generated_or_evaluated():
    """The engine is data in, data out: nothing it produces is executable."""
    source = Path(policy_module.__file__).read_text()
    for forbidden in ("eval(", "exec(", "__import__", "importlib", "subprocess", "builtins"):
        assert forbidden not in source
