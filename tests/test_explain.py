"""Alert explanations: grounded in the stored event, never invented.

An alert a person cannot interrogate is one they will eventually mute. The
property under test is that the explanation is assembled from the database row
and the rule that fired — a model, when present, only rephrases that fact list
and can be removed without the feature breaking.
"""

import importlib
import json
import sys

import pytest

from jupiter.explain import (
    CRITICAL,
    INFO,
    WARNING,
    Explanation,
    explain,
    facts_from_event,
    humanise,
    parse_rule_for_event,
    rule_summary,
    severity_for,
)
from jupiter.providers import ProviderResult, ScriptedProvider

FALL = {
    "type": "fall", "severity": "loud", "confidence": 0.94,
    "signals": "rapid_drop + stillness", "duration": 8.0,
    "timestamp": "2026-08-06T14:41:05+00:00",
}


# --------------------------------------------------------------------- severity
def test_stored_severity_maps_to_user_language():
    assert severity_for({"type": "person", "severity": "loud"}) == CRITICAL
    assert severity_for({"type": "person", "severity": "soft"}) == WARNING
    assert severity_for({"type": "person", "severity": "log"}) == INFO


def test_falls_are_critical_regardless_of_stored_severity():
    """A fall logged quietly is still a fall."""
    assert severity_for({"type": "fall", "severity": "log"}) == CRITICAL
    assert severity_for({"type": "on_floor", "severity": "soft"}) == CRITICAL


def test_unknown_severity_degrades_to_informational():
    assert severity_for({"type": "person", "severity": "banana"}) == INFO
    assert severity_for({}) == INFO


# --------------------------------------------------------------------- facts
def test_facts_come_from_the_event_only():
    facts = facts_from_event(FALL)
    labels = [f.label for f in facts]
    assert "Rapid drop" in labels
    assert "Stillness" in labels
    assert any("confidence" in f.label.lower() for f in facts)


def test_absent_fields_produce_no_fact_rather_than_a_guess():
    """A missing duration must not become "briefly" — that is narration."""
    facts = facts_from_event({"type": "person"})
    assert facts == []


def test_zero_confidence_is_not_reported_as_a_fact():
    facts = facts_from_event({"type": "person", "confidence": 0})
    assert not any("confidence" in f.label.lower() for f in facts)


def test_zone_comes_from_the_rule_not_the_event():
    facts = facts_from_event({"type": "person"}, {"zone": "pool"})
    assert any(f.label == "Area" and f.detail == "pool" for f in facts)


# --------------------------------------------------------------------- rule summary
def test_rule_summary_quotes_the_configured_numbers():
    assert "10 seconds" in rule_summary({"persist_s": 10})


def test_rule_summary_includes_a_schedule_when_set():
    summary = rule_summary({"persist_s": 5, "schedule": "23:00-06:00"})
    assert "23:00" in summary and "06:00" in summary


def test_no_rule_produces_no_claim_about_one():
    assert rule_summary(None) == ""
    assert rule_summary({}) == ""


# --------------------------------------------------------------------- explanation
def test_explanation_works_with_no_model_at_all():
    """The deterministic path is the default, not a degraded fallback."""
    result = explain(FALL, {"persist_s": 5, "zone": "living room"})
    assert result.severity == CRITICAL
    assert result.headline == "Fall"
    assert result.facts
    assert result.source == "facts"
    text = result.plain_text()
    assert "Rapid drop" in text and "5 seconds" in text


def test_explanation_reports_the_time_it_happened():
    assert explain(FALL).when  # formatted from the stored timestamp


def test_a_malformed_timestamp_is_omitted_not_invented():
    assert explain({**FALL, "timestamp": "not a time"}).when == ""


def test_a_model_only_rephrases_and_is_labelled():
    """Uses a routine event: critical ones deliberately bypass the model."""
    routine = {"type": "person", "severity": "soft", "signals": "person_seen",
               "confidence": 0.8}
    provider = ScriptedProvider(config={"default": "Someone walked past."})
    result = explain(routine, {"persist_s": 5}, provider=provider)
    assert result.narrative == "Someone walked past."
    assert result.source == "scripted"
    # The facts are unchanged by the model — it never gets to add one.
    assert any(f.label == "Person seen" for f in result.facts)


def test_the_model_is_never_told_what_happened_only_the_facts():
    """The prompt must not contain the event type, or the model will narrate
    from its own idea of what a fall looks like."""
    routine = {"type": "person", "severity": "soft", "signals": "person_seen"}
    provider = ScriptedProvider(config={"default": "x"})
    explain(routine, provider=provider)
    system, user = provider.calls[0]
    assert "ONLY the observations" in system
    assert "Person seen" in user
    assert "person" not in system.lower().split("observations")[0]


def test_an_essay_from_the_model_is_discarded():
    """A model that ignores the instruction must not reach the user."""
    routine = {"type": "person", "severity": "soft", "signals": "person_seen"}
    provider = ScriptedProvider(config={"default": "word " * 200})
    result = explain(routine, provider=provider)
    assert result.narrative == ""
    assert result.source == "facts"


def test_a_failing_model_never_breaks_the_explanation():
    class Broken(ScriptedProvider):
        def complete(self, system, user, *, schema=None, grammar=None):
            raise RuntimeError("model died")

    routine = {"type": "person", "severity": "soft", "signals": "person_seen"}
    result = explain(routine, provider=Broken())
    assert result.source == "facts"
    assert result.plain_text()


def test_explanation_serialises_for_the_app():
    payload = explain(FALL, {"persist_s": 5}).as_dict()
    assert set(payload) >= {"headline", "severity", "facts", "rule", "narrative"}
    assert json.loads(json.dumps(payload))["severity"] == CRITICAL


# --------------------------------------------------------------------- rule lookup
def test_the_originating_rule_is_found_by_target():
    stored = json.dumps({"rules": [{"target": "fall", "persist_s": 30}]})
    assert parse_rule_for_event(stored, "fall")["persist_s"] == 30


def test_a_zoned_event_type_still_matches_its_rule():
    stored = json.dumps({"rules": [{"target": "zone_violation", "zone": "pool"}]})
    assert parse_rule_for_event(stored, "zone_violation:pool")["zone"] == "pool"


def test_missing_or_corrupt_rules_return_none():
    assert parse_rule_for_event(None, "fall") is None
    assert parse_rule_for_event("not json", "fall") is None


def test_humanise_turns_identifiers_into_english():
    assert humanise("phone_use_while_distracted") == "Phone use while distracted"
    assert humanise("") == "Event"


# --------------------------------------------------------------------- endpoint
@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at, ai_rules)"
            " VALUES (?, ?, ?, ?, ?)",
            ("Test unit", "AA-BB-CC", "dk_test", _time.time(),
             json.dumps({"rules": [{"target": "fall", "persist_s": 30, "zone": "living room"}]})),
        )
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    client.post("/v1/ingest", json={"events": [FALL]}, headers={"X-Device-Key": "dk_test"})
    return app_module, client, headers


def test_endpoint_explains_a_real_stored_event(backend):
    _, client, headers = backend
    event_id = client.get("/v1/devices/1/events", headers=headers).json()["events"]
    resp = client.get("/v1/devices/1/events/1/explain", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity"] == CRITICAL
    assert body["headline"] == "Fall"
    assert "30 seconds" in body["rule"]
    assert body["narrative"]


def test_endpoint_404s_for_an_event_that_does_not_exist(backend):
    _, client, headers = backend
    assert client.get("/v1/devices/1/events/9999/explain", headers=headers).status_code == 404


def test_another_user_cannot_explain_someone_elses_event(backend):
    _, client, _ = backend
    other = client.post("/v1/signup", json={"email": "x@x.com", "password": "password2"}).json()["token"]
    resp = client.get("/v1/devices/1/events/1/explain",
                      headers={"Authorization": f"Bearer {other}"})
    assert resp.status_code == 404


def test_critical_events_are_never_rephrased_by_a_model():
    """Measured: given "Rapid drop" and "Detector confidence — 94%", a 1.5B
    model wrote "a rapid drop in detector confidence" — two facts merged into an
    untruth, on a fall alert. A fluent wrong sentence is worse than a blunt
    right one when someone's parent is on the floor."""
    provider = ScriptedProvider(config={"default": "Something dramatic happened."})
    result = explain(FALL, {"persist_s": 5}, provider=provider)

    assert result.severity == CRITICAL
    assert result.source == "facts"
    assert result.narrative == ""
    assert provider.calls == [], "the model must not even be called"


def test_routine_events_may_still_be_rephrased():
    routine = {"type": "person", "severity": "log", "signals": "person_seen"}
    provider = ScriptedProvider(config={"default": "Someone walked through."})
    result = explain(routine, provider=provider)
    assert result.severity == INFO
    assert result.narrative == "Someone walked through."
