"""M8.4: "why did this happen" for an experimental observation says it was not an alert.

ALPHA's contract: an experimental observation (the driver module's combined
risk score, erratic hand movement) is stored with `experimental: true` and
`severity: "log"`. The explanation must not read like an alert, and an event
without the field explains exactly as before.
"""

from __future__ import annotations

from jupiter.explain import explain, facts_from_event, severity_for

EXPERIMENTAL = {"type": "high_risk", "severity": "log", "signals": "phone_visible + gaze_off",
                "confidence": 0.8, "experimental": True, "source": "module"}


def test_an_experimental_observation_is_explained_as_not_an_alert():
    facts = facts_from_event(EXPERIMENTAL)
    assert facts[-1].label == "Experimental observation" and facts[-1].detail == "not an alert"
    assert severity_for(EXPERIMENTAL) == "info"
    result = explain(EXPERIMENTAL, None, provider=None).as_dict()
    assert "not an alert" in result["narrative"]
    assert "You were alerted" not in result["narrative"]


def test_the_stored_integer_form_is_read_too():
    assert facts_from_event({**EXPERIMENTAL, "experimental": 1})[-1].label == "Experimental observation"


def test_without_the_field_nothing_changes():
    for value in (None, False, 0, "yes"):
        event = {**EXPERIMENTAL, "experimental": value}
        assert all(f.label != "Experimental observation" for f in facts_from_event(event))
