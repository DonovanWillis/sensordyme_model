"""M13 "second wow", end to end: an instruction becomes a running area + dwell rule. SIMULATED.

    "Alert me when a person remains in this area for 10 seconds."

The real backend (cloud/app.py) and the real device runtime on one SimClock
(sim/jupiter_demo.py), driven only through the calls the iPhone app makes:
zones, /v1/ai/configure, the review payload, Activate, then the event history
and the device's own GET /api/events (M12). The instruction is read by the
deterministic policy parser (jupiter/policy.py) with any language-model call
made to fail; the rule is data evaluated by the device's rule engine on the
shared detector and tracker. Scripted standing people, not real footage.
"""

from __future__ import annotations

import pytest

from jupiter import policy as jp
from sim import jupiter_demo as demo

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")


@pytest.fixture(scope="module")
def run():
    return demo.run()


def test_the_exact_instruction_is_read_by_the_deterministic_parser():
    parsed = jp.parse(demo.PROMPT, zones=[demo.AREA["name"]])
    assert parsed is not None and not parsed.unsupported
    (rule,) = [p.rule for p in parsed.policies]
    assert (rule["target"], rule["predicate"], rule["zone"], rule["dwell_s"]) == ("person", "lingering", "entrance", 10.0)
    # "this area" with nothing drawn is still the area the owner will draw, not the whole frame.
    undrawn = jp.parse(demo.PROMPT, zones=[])
    assert undrawn is not None and undrawn.policies[0].rule["zone"] == "marked area"


def test_every_step_of_the_demo_holds(run):
    failed = [(s.n, s.title, s.result) for s in run.steps if not s.ok]
    assert failed == []
    assert [s.n for s in run.steps] == list(range(9))


def test_the_proposal_is_a_typed_policy_and_nothing_else(run):
    preview = run.preview
    assert preview["source"] == "policy_parser" and preview["preset"] == "" and preview["unsupported"] == []
    (policy,) = preview["policies"]
    assert policy["condition"] == "lingering" and policy["temporal"] == {"dwell_s": 10.0}
    assert policy["zone"] == {"name": "entrance", "drawn": True} and policy["status"] == "ready"
    assert policy["required_capabilities"] == ["camera", "inference", "objects"]
    assert any("no face recognition" in line for line in policy["explanation"]["does_not_monitor"])
    (rule,) = preview["rules"]
    assert set(rule) <= {"target", "predicate", "secondary", "description", "zone", "persist_s", "min_count",
                         "confidence", "severity", "cooldown_s", "schedule", "notify", "dwell_s", "id"}
    assert all(isinstance(v, (str, int, float)) for v in rule.values()), "data only: no code reaches the device"


def test_nothing_fires_under_ten_seconds_or_outside_the_area_and_once_at_ten(run):
    assert run.steps[6].detail["silent_after"] == {"9 s inside": 0, "15 s outside": 0}
    assert run.latency_s is not None and demo.DWELL_S <= run.latency_s <= demo.DWELL_S + demo.DWELL_SLACK_S
    assert len(run.backend_events) == 1


def test_the_alert_is_the_same_occurrence_in_the_backend_and_on_the_device(run):
    (event,) = run.backend_events
    (local,) = run.local_events
    assert event["type"] == "person" and event["source"] == "custom_rule" and event["simulated"] is True
    assert event["signals"] == "A person remains in this area for 10 seconds", "the rule's own words say why"
    assert event["alert"]["decision"] == "alerted" and event["alert"]["output"] == "simulated"
    assert local["event_key"] == event["event_key"] and local["sync"] == "sent" and local["simulated"] is True


def test_an_identity_request_is_refused_and_the_running_rule_is_kept(run):
    assert run.refused["rules"] == [] and [u["code"] for u in run.refused["unsupported"]] == ["identity"]
    assert run.steps[8].ok


@pytest.mark.xfail(strict=True, reason="J1: a dwell alert's `duration` is the rule engine's confirmation step "
                                       "(~0.1 s), not the >= 10 s the person stayed (core/rules.py dwell branch)")
def test_known_gap_the_dwell_alerts_duration_is_the_time_in_the_area(run):
    (event,) = run.backend_events
    assert event["duration"] >= demo.DWELL_S, event["duration"]
