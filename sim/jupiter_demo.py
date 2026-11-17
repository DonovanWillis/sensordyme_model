"""M13 "second wow": a spoken-style instruction becomes a running area + dwell rule. SIMULATED.

    "Alert me when a person remains in this area for 10 seconds."

End to end on the virtual device (sim.harness.SimStack: the real backend
cloud/app.py, the real device runtime, its shared perception engine, tracker
and rule engine, on one SimClock), driven only through the calls the iPhone app
makes (ios/Sources/BackendAPI.swift):

  1. draw the area                POST /v1/devices/{id}/zones
  2. the instruction              POST /v1/ai/configure  -> a typed policy, a draft
  3. review                       the policy's condition, area, timing, capabilities, and what it does NOT watch
  4. activate                     POST /v1/devices/{id}/preset {preset: "", rules, draft_version}
  5. the device runs it           mode ai_custom, rule active, on the shared detector + tracker
  6. people come and go           9 s in the area (nothing), 15 s outside it (nothing), 15 s in it (one alert)
  7. the alert, both histories    the backend's event history and the device's own GET /api/events (M12)

The instruction is read by jupiter/policy.py's deterministic parser: no
language model is involved (the run proves it by making any model call fail),
and nothing it produces is code -- a rule is data the device's rule engine
evaluates, on a label the detector serves. The scene is scripted standing
people (sim/perception.py Scene.people); nothing here is accuracy on real
footage. tests/test_m13_jupiter_demo.py pins it; scripts/demo/jupiter_rehearsal.py prints it.
"""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROMPT = "Alert me when a person remains in this area for 10 seconds."
#: The area the owner draws (normalised; a person's feet at x=0.2 are inside, x=0.8 outside).
AREA = {"name": "entrance", "polygon": [[0.0, 0.5], [0.4, 0.5], [0.4, 1.0], [0.0, 1.0]], "kind": "restricted"}
INSIDE_X, OUTSIDE_X = 0.2, 0.8
GENERAL = {"role": "general", "module": "shared_perception"}
#: Device clock (seconds since boot) of each visit: (start, end, x).
SHORT_VISIT = (20.0, 29.0, INSIDE_X)  # 9 s inside: under the dwell, must stay silent
OUTSIDE_VISIT = (35.0, 50.0, OUTSIDE_X)  # 15 s in view but outside the area: silent
LONG_VISIT = (55.0, 70.0, INSIDE_X)  # 15 s inside: one alert at ~10 s
RUN_S = 80.0
DWELL_S = 10.0
#: The latency the expectation allows past the 10 s dwell: the shared detector
#: and tracker cadence plus the rule engine's next-frame confirmation.
DWELL_SLACK_S = 1.5
#: An instruction the device cannot honestly serve: refused, nothing set up.
REFUSED_PROMPT = "Alert me when a stranger remains in this area for 10 seconds."


@dataclass
class Step:
    n: int
    title: str
    call: str
    result: str
    ok: bool
    detail: dict = field(default_factory=dict)


@dataclass
class DemoResult:
    steps: list[Step]
    preview: dict
    activation: dict
    backend_events: list[dict]
    local_events: list[dict]
    fired_at: Optional[float]  # device clock when the device recorded the alert
    refused: dict

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def latency_s(self) -> Optional[float]:
        return None if self.fired_at is None else round(self.fired_at - LONG_VISIT[0], 2)


def scenario():
    from sim.perception import Scene
    from sim.scenario import Scenario, Segment

    empty = Scene(face=False, person=False, label="nobody in view")

    def person(x: float) -> Scene:
        return Scene(face=False, person=False, people=(x,), label="a person " + ("inside" if x < 0.4 else "outside"))

    segments = [Segment(0.0, empty)]
    for start, end, x in (SHORT_VISIT, OUTSIDE_VISIT, LONG_VISIT):
        segments += [Segment(start, person(x)), Segment(end, empty)]
    return Scenario("m13_jupiter_area_dwell", "Nobody; a person 9 s in the area; 15 s outside it; 15 s in it.",
                    tuple(segments))


def _custom(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("source") == "custom_rule"]


def run(root: Optional[str] = None, prompt: str = PROMPT) -> DemoResult:
    """The whole demo on a fresh backend + virtual device; every step's outcome."""
    from sim.harness import SimStack

    with tempfile.TemporaryDirectory() if root is None else nullcontext(root) as where, \
            SimStack(Path(where)) as stack:
        app = stack.backend.app

        def no_model(*_a, **_k):
            raise AssertionError("SIMULATED demo: the language model must not be asked")

        real_llm = app._ai_configure_llm
        app._ai_configure_llm = no_model  # this run proves the parser alone reads the instruction
        try:
            return _run(stack, app, prompt)
        finally:
            app._ai_configure_llm = real_llm


def _run(stack, app, prompt: str) -> DemoResult:
    from core import local_auth as la
    from sim.local_client import LocalPhone

    steps: list[Step] = []
    device = stack.boot(scenario(), config_overrides=GENERAL, dashboard_port=0)
    stack.claim()
    t0 = stack.clock.monotonic()

    def now() -> float:
        return round(stack.clock.monotonic() - t0, 2)

    device.advance(4.0)
    status = stack.status()
    steps.append(Step(0, "a general-purpose camera, paired and monitoring", "boot + POST /v1/devices/claim",
                      f"{status['monitoring_state']}, module {device.runtime.module_name}",
                      status["monitoring_state"] == "ready" and device.runtime.module_name == "shared_perception"))

    drawn = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": [AREA]}, headers=stack.headers)
    steps.append(Step(1, "draw the area", f"POST /v1/devices/{{id}}/zones {AREA['name']!r}",
                      f"HTTP {drawn.status_code}, rejected {drawn.json().get('rejected')}",
                      drawn.status_code == 200 and drawn.json().get("rejected") == 0))
    device.advance(2.5)

    response = stack.ai_configure(prompt)
    preview = response.json()
    rules = preview.get("rules") or []
    policies = preview.get("policies") or []
    policy = policies[0] if policies else {}
    steps.append(Step(2, "the instruction", f"POST /v1/ai/configure {prompt!r}",
                      f"HTTP {response.status_code}, source {preview.get('source')}, {len(rules)} rule(s), "
                      f"{len(preview.get('unsupported') or [])} unsupported",
                      response.status_code == 200 and preview.get("source") == "policy_parser"
                      and len(rules) == 1 and not preview.get("unsupported"),
                      {"explanation": preview.get("explanation"), "rule": rules[0] if rules else None}))

    caps = app._capabilities()  # the backend's own capability registry (the labels a rule may name)
    labels = caps.registry().labels() if caps is not None else set()
    rule = rules[0] if rules else {}
    reviewed = (policy.get("condition") == "lingering" and policy.get("temporal") == {"dwell_s": DWELL_S}
                and policy.get("zone") == {"name": AREA["name"], "drawn": True}
                and policy.get("status") == "ready" and policy.get("targets") == ["person"]
                and set(policy.get("required_capabilities") or ()) <= {"camera", "inference", "objects"}
                and rule.get("target") in labels and all(not callable(v) for v in rule.values()))
    explanation = policy.get("explanation") or {}
    steps.append(Step(3, "review the typed policy", "(the review screen reads `policies`)",
                      f"{policy.get('condition')} {policy.get('targets')} in {policy.get('zone')}, "
                      f"dwell {policy.get('temporal')}, status {policy.get('status')}", reviewed,
                      {"summary": explanation.get("summary"), "monitors": explanation.get("monitors"),
                       "does_not_monitor": explanation.get("does_not_monitor")}))

    # What the app sends on Activate: the reviewed rules (+ its time zone) and the draft it previewed.
    activation_rules = [{**r, "timezone": "UTC"} for r in rules]
    activated = stack.set_preset("", draft_version=preview.get("draft_version"), rules=activation_rules)
    activation = activated.json()
    steps.append(Step(4, "activate", "POST /v1/devices/{id}/preset {preset: '', rules, draft_version}",
                      f"HTTP {activated.status_code}, rules_active {activation.get('rules_active')}",
                      activated.status_code == 200 and activation.get("rules_active") == 1))

    phone = LocalPhone(device.runtime.dashboard.url, name="Owner iPhone")
    enrolled = stack.api.post(f"/v1/devices/{stack.device_id}/local-clients",
                              json={"public_key": la.b64u(phone.key.public_key), "name": "Owner iPhone"},
                              headers=stack.headers)
    device.advance(4.0)
    status = stack.status()
    engine_rules = device.runtime.rule_engine.rules
    running = (status["mode"] == "ai_custom" and status["rules"]["active"] and not status["rules"]["rejected"]
               and [(r.predicate, r.detection, r.zone, r.dwell_s) for r in engine_rules]
               == [("lingering", "person", AREA["name"], DWELL_S)])
    steps.append(Step(5, "the device runs it", "(device reports via the uplink; GET /v1/devices)",
                      f"mode {status['mode']}, rules {status['rules']['active']}, "
                      f"rule engine {[(r.predicate, r.detection, r.zone, r.dwell_s) for r in engine_rules]}",
                      running and enrolled.status_code == 200))

    fired_at: Optional[float] = None
    seen_before: dict[str, list[dict]] = {}
    while now() < RUN_S - 1e-6:
        device.advance(0.25)
        if fired_at is None and any(e["type"] == "person" for e in _local(device)):
            fired_at = now()
        for name, (_start, end, _x) in (("9 s inside", SHORT_VISIT), ("15 s outside", OUTSIDE_VISIT)):
            if name not in seen_before and now() >= end + 3.0:
                seen_before[name] = _custom(stack.events())
    quiet = all(not v for v in seen_before.values()) and len(seen_before) == 2
    long_ok = (fired_at is not None and LONG_VISIT[0] + DWELL_S <= fired_at <= LONG_VISIT[0] + DWELL_S + DWELL_SLACK_S)
    steps.append(Step(6, "people come and go (scripted, SIMULATED)",
                      "9 s inside, 15 s outside, then 15 s inside the area",
                      f"nothing for the 9 s visit or the visit outside; one alert "
                      f"{'at +%.2f s into the long visit' % (fired_at - LONG_VISIT[0]) if fired_at else 'NEVER'}",
                      quiet and long_ok, {"silent_after": {k: len(v) for k, v in seen_before.items()},
                                          "device_fired_at_s": fired_at}))

    backend = _custom(stack.events())
    identity = stack.row()["local_identity"]
    phone.trust({"device_public_key": identity["device_public_key"], "device_id": device.runtime._local_device_id()})
    session = phone.open_session()
    local_page = phone.get("/api/events?limit=50")
    local = [e for e in (local_page.body or {}).get("events", []) if e.get("type") == "person"]
    same = (len(backend) == 1 and len(local) == 1 and local[0]["event_key"] == backend[0]["event_key"]
            and backend[0]["signals"] == rule.get("description") and backend[0]["simulated"] is True
            and (backend[0].get("alert") or {}).get("decision") == "alerted" and local[0]["sync"] == "sent")
    steps.append(Step(7, "the alert, in both histories", "GET /v1/devices/{id}/events; device GET /api/events (sd-local-v1)",
                      f"backend {[(e['type'], e['signals'], e['severity']) for e in backend]}; "
                      f"device {[(e['type'], e['sync']) for e in local]} (session HTTP {session.status})",
                      same and session.status == 200))

    refused_response = stack.ai_configure(REFUSED_PROMPT)
    refused = refused_response.json()
    steps.append(Step(8, "an unsupported instruction is refused", f"POST /v1/ai/configure {REFUSED_PROMPT!r}",
                      f"{len(refused.get('rules') or [])} rule(s); refused: "
                      f"{[u.get('code') for u in refused.get('unsupported') or []]}",
                      refused_response.status_code == 200 and not refused.get("rules")
                      and bool(refused.get("unsupported")),
                      {"reason": [u.get("reason") for u in refused.get("unsupported") or []]}))
    # The refusal is a preview: the running rule is untouched.
    device.advance(4.0)
    still = [(r.predicate, r.detection) for r in device.runtime.rule_engine.rules] == [("lingering", "person")]
    steps[-1].ok = steps[-1].ok and still
    return DemoResult(steps, preview, activation, backend, local, fired_at, refused)


def _local(device) -> list[dict]:
    page, _ = device.runtime.dashboard.events_response({"limit": ["200"]})
    return list(page.get("events") or [])
