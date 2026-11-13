"""M8.4: Ask Jupiter's grammar, typed refusals and the experimental-signal guard.

The deterministic parser (jupiter/policy.py) reads more PHRASINGS than before,
never more CAPABILITIES: every new phrasing compiles to the same conditions on
a label the registry serves, and the device accepts the rule. Every refusal
the /v1/ai/configure preview returns carries a typed `code` from a closed set
the app renders. Requests for the driver module's experimental families
(the combined risk score, erratic hand movement) are refused up front: they
are observations, never alerts, and no interpreter may offer them as one.
"""

from __future__ import annotations

import importlib
import sys
import time

import pytest

from core.rules import ConfiguredRule, validate_rule_set
from jupiter.policy import (
    UNSUPPORTED_CODES,
    parse,
    refusal_for,
    typed_unsupported,
    unsupported_code,
)

ZONES = ["front door", "kitchen", "driveway", "garage"]


def _only(text: str):
    result = parse(text, zones=ZONES)
    assert result is not None, f"{text!r} was not claimed"
    assert result.unsupported == [], result.unsupported
    (policy,) = result.policies
    rule = policy.rule
    # The device's own engine accepts the compiled rule as written.
    assert validate_rule_set([rule], label_ok=lambda label: True) == []
    assert ConfiguredRule.from_dict(0, rule) is not None
    return policy


# ------------------------------------------------------------------ wider phrasings
@pytest.mark.parametrize(("text", "target", "predicate", "zone", "temporal"), [
    ("Can you alert me if someone stays in the kitchen longer than 10 seconds",
     "person", "lingering", "kitchen", {"dwell_s": 10.0}),
    ("I want to know when someone spends more than 5 minutes in the kitchen",
     "person", "lingering", "kitchen", {"dwell_s": 300.0}),
    ("alert me if someone stays in the kitchen for 30 seconds or longer",
     "person", "lingering", "kitchen", {"dwell_s": 30.0}),
    ("notify me as soon as someone shows up at the front door",
     "person", "in_zone", "front door", {"sustain_s": 2.0}),
    ("send me a push notification when someone arrives at the front door",
     "person", "in_zone", "front door", {"sustain_s": 2.0}),
    ("tell me when a person is seen in the driveway",
     "person", "in_zone", "driveway", {"sustain_s": 2.0}),
    ("let me know if there is a dog in the kitchen",
     "dog", "in_zone", "kitchen", {"sustain_s": 2.0}),
    ("alert me if there are more than 2 people in the garage",
     "person", "count", "garage", {"count": 3, "sustain_s": 2.0}),
    ("alert me if there is nobody in the kitchen for 10 minutes",
     "person", "absent", "kitchen", {"sustain_s": 600.0}),
    ("could you please warn me any time someone enters the garage",
     "person", "in_zone", "garage", {"sustain_s": 2.0}),
])
def test_new_phrasings_compile_to_the_same_supported_conditions(text, target, predicate, zone, temporal):
    policy = _only(text)
    assert policy.targets == [target]
    assert policy.condition == predicate
    assert policy.zone == zone and policy.zone_drawn
    assert policy.temporal == temporal


@pytest.mark.parametrize("text", [
    "alert me if someone stays in the kitchen after dark for 10 seconds",   # a time qualifier it does not read
    "alert me when someone is near the car",                                # a relation, not an area
    "alert me if someone sits in the kitchen for 10 seconds",               # posture, not dwell
    "alert me if there is a dog on the couch",                              # an object, not a drawn area
])
def test_what_the_grammar_does_not_read_is_not_claimed(text):
    """Not claimed means the request goes to the next interpreter, whose output
    is validated in turn -- never a partial or approximate reading here."""
    assert parse(text, zones=ZONES) is None


# ------------------------------------------------------------------ experimental signals
@pytest.mark.parametrize("text", [
    "alert me on high risk",
    "alert me if the risk score is high",
    "tell me when there is elevated risk",
    "warn me if the driver has erratic hand movement",
    "notify me about erratic hands while driving",
    "alert me if the driver's hand movement is unusual",
    "beep when the driver is fidgeting",
])
def test_experimental_driver_families_are_never_offered_as_alerts(text):
    entries = refusal_for(text)
    assert [e["code"] for e in entries] == ["experimental_signal"]
    assert "experimental" in entries[0]["reason"].lower()
    assert "never sound or notify" in entries[0]["requires"]
    result = parse(text, zones=ZONES)
    assert result is not None and result.policies == []
    assert result.as_proposal()["rules"] == []


def test_a_plain_area_request_is_not_mistaken_for_an_experimental_one():
    assert refusal_for("alert me if someone enters the driveway") == []
    assert refusal_for("tell me if someone is waving at the front door") == []


# ------------------------------------------------------------------ typed codes
def test_every_parser_refusal_carries_a_known_code():
    cases = {
        "alert me if a stranger stays at the door for 10 seconds": "identity",
        "tell me if the baby is crying": "audio",
        "alert me if someone enters the yard 3 times": "repetition_window",
        "alert me if a package stays on the porch for 30 seconds": "not_detectable",
    }
    for text, code in cases.items():
        result = parse(text, zones=["door", "yard", "porch"])
        assert result is not None
        assert [u["code"] for u in result.unsupported] == [code], text
        assert all(u["code"] in UNSUPPORTED_CODES for u in result.unsupported)


def test_typing_is_additive_and_derives_a_code_for_any_source():
    raw = [
        {"request": "a", "reason": "r", "requires": "", "support": "needs_training_data"},
        {"request": "b", "reason": "r", "requires": "", "support": "unsupported"},
        {"request": "c", "reason": "r", "requires": "", "category": "medical"},
        {"request": "d", "reason": "r", "requires": "", "code": "needs_area"},
        {"request": "e", "reason": "r", "requires": "", "code": "made_up"},
        "not an object",
    ]
    typed = typed_unsupported(raw)
    assert [t["code"] for t in typed] == ["not_detectable", "not_supported", "medical", "needs_area",
                                          "not_supported"]
    assert typed[0]["reason"] == "r" and typed[0]["support"] == "needs_training_data"
    assert unsupported_code({}) == "not_supported"


# ------------------------------------------------------------------ the endpoint
@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Unit", "AA-BB-CC", "dk_test", time.time()))
    token = client.post("/v1/signup", json={"email": "m84@example.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return client, headers, app_module


@pytest.mark.parametrize(("prompt", "code"), [
    ("alert me on high risk", "experimental_signal"),
    ("tell me if someone I don't recognize enters the yard", "identity"),
    ("alert me if a package stays on the porch for 30 seconds", "not_detectable"),
    ("alert me when the stock market drops", "interpreter_unavailable"),
])
def test_the_preview_returns_a_typed_reason_and_no_rule(backend, prompt, code):
    client, headers, _ = backend
    body = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": prompt}, headers=headers).json()
    assert body["rules"] == [] and body["preset"] == ""
    assert [u["code"] for u in body["unsupported"]] == [code]


def test_a_mode_suggestion_says_it_is_not_the_request_read(backend):
    client, headers, _ = backend
    body = client.post("/v1/ai/configure", json={"device_id": 1, "prompt": "let me know if grandma falls"},
                       headers=headers).json()
    assert body["preset"] == "elderly_monitoring" and body["source"] == "fallback"
    assert body["explanation"].startswith("No custom rule was set up")


@pytest.mark.parametrize("prompt", [
    "be careful near the outdoor waterfall",   # car, door, fall as parts of words
    "a car pulls into the driveway",           # a car is not a driver to watch
    "tell me if someone picks up my phone",    # nor is a phone
])
def test_the_keyword_fallback_matches_whole_words_only(backend, prompt):
    _, _, app_module = backend
    proposal = app_module._ai_fallback(prompt)
    assert proposal["preset"] == "" and proposal["source"] == "unavailable"
    assert proposal["unsupported"][0]["code"] == "interpreter_unavailable"


# ------------------------------------------------------------------ the app's suggestions
def test_every_suggestion_the_app_offers_is_read_exactly_and_the_refused_one_is_refused():
    """ios/Sources/JupiterViews.swift JupiterSuggestions: what the app says
    Jupiter can watch for must be what the grammar reads, with no model."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ios/Sources/JupiterViews.swift").read_text()
    block = source.split("enum JupiterSuggestions", 1)[1].split("\n}\n", 1)[0]
    supported, refused = block.split("static let refused", 1)
    texts = re.findall(r'text: "([^"]+)"', supported)
    assert len(texts) >= 5
    for text in texts:
        result = parse(text, zones=["marked area"])
        assert result is not None and result.unsupported == [] and len(result.policies) == 1, text
    (refused_text,) = re.findall(r'text: "([^"]+)"', refused)
    assert [e["code"] for e in refusal_for(refused_text)] == ["identity"]
