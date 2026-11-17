"""M14 stream D: a Jupiter rule BESIDE mandatory Driver Safety, end to end. SIMULATED.

    "Alert me when a cell phone stays in this area for 10 seconds."

The shipping unit is `role: driver_safety` (config.yaml). On that role, and only
that role, custom rules run beside the mandatory driver_attention module
(core/roles.py `additive_custom_rules`; services/runtime.py `_resolve_mode` ->
mode `standard_plus_custom`) and may only name labels that module's own
detector serves (`_rule_problems` -> DriverAttentionModule.available_labels():
config.yaml `yolo_classes` + person). On every other role (general,
child_monitoring, occupancy, property, workplace) custom rules REPLACE the
running mode (`ai_custom`). The M13 showcase ran on `general`.

Driven only through the calls the iPhone app makes, on the real backend
(cloud/app.py) and the real device runtime (sim.harness.SimStack, scripted
camera and models, one SimClock): draw the area -> /v1/ai/configure (the
deterministic parser, any language-model call made to fail) -> the typed policy
-> capability check against what this device reports -> Activate -> the device
acknowledges -> the rule runs on the driver module's own detector (no second
model or module) -> a scripted phone sits in the area 6 s (silent), then 20 s
(one alert at ~10 s) while the driver also closes their eyes, then later turns
away -> both histories (backend and the device's signed GET /api/events) ->
sd-overlay/1 `last_event` and its event_key.

The same timeline is also run WITHOUT the rule: every Driver Safety alert, its
device time, its delivery and the buzzer patterns must be identical, and the
detector must run exactly as often. Scripted inputs, not real footage: nothing
here is accuracy, latency or hardware evidence.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest
import yaml

from core import local_auth as la
from core import overlay as ov
from core.roles import config_violations, role_named
from jupiter import policy as jp
from sim.harness import SimStack
from sim.local_client import LocalPhone
from sim.perception import Scene
from sim.scenario import ATTENTIVE, CALIBRATION_S, LOOKING_AWAY, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

PROMPT = "Alert me when a cell phone stays in this area for 10 seconds."
#: The one area the owner draws (normalised). The scripted phone's box bottom-centre
#: is (0.49, 0.86): inside. The driver's own box bottom-centre (0.5, 0.98): outside.
AREA = {"name": "console", "kind": "restricted", "polygon": [[0.3, 0.55], [0.7, 0.55], [0.7, 0.95], [0.3, 0.95]]}
C = CALIBRATION_S
DWELL_S = 10.0
#: Past the dwell, as in M13: the detector stride + the rule engine's next-frame confirmation.
DWELL_SLACK_S = 1.5
SHORT_VISIT = (C + 10.0, C + 16.0)  # 6 s in the area: under the dwell, silent
LONG_VISIT = (C + 25.0, C + 45.0)  # 20 s in the area: one alert at ~10 s
EYES_CLOSED = (C + 33.0, C + 38.0)  # inside the long visit: drowsiness ~2 s later, beside the rule
TURNED_AWAY = (C + 55.0, C + 61.0)  # no phone: distraction ~3 s later
RUN_S = C + 75.0
#: Device seconds while the app's AI view is open (streaming costs wall time, not device time):
#: from just before the rule can fire to just after the latest it may.
OVERLAY_OPEN = (LONG_VISIT[0] + DWELL_S - 1.0, LONG_VISIT[0] + DWELL_S + DWELL_SLACK_S + 0.5)

PHONE = Scene(phone=True, label="phone in the area, eyes on the road")
PHONE_EYES_CLOSED = Scene(phone=True, eyes_closed=True, label="phone in the area, eyes closed")
DRIVER_TYPES = frozenset({"drowsiness", "attention_lost", "distraction", "eyes_down", "eyes_up", "phone_use"})


def scenario() -> Scenario:
    segments = [Segment(0.0, ATTENTIVE),
                Segment(SHORT_VISIT[0], PHONE), Segment(SHORT_VISIT[1], ATTENTIVE),
                Segment(LONG_VISIT[0], PHONE), Segment(EYES_CLOSED[0], PHONE_EYES_CLOSED),
                Segment(EYES_CLOSED[1], PHONE), Segment(LONG_VISIT[1], ATTENTIVE),
                Segment(TURNED_AWAY[0], LOOKING_AWAY), Segment(TURNED_AWAY[1], ATTENTIVE)]
    return Scenario("m14_jupiter_beside_driver_safety",
                    "A phone 6 s then 20 s in the drawn area (eyes closed for 5 s of it); later a 6 s head turn.",
                    tuple(segments))


@dataclass
class Journey:
    with_rule: bool
    role: str = ""
    preset: Optional[str] = None
    preview: dict = field(default_factory=dict)
    activation: dict = field(default_factory=dict)
    acknowledged_before_device: bool = False
    status_before: dict = field(default_factory=dict)
    status_running: dict = field(default_factory=dict)
    engine_rules: list = field(default_factory=list)
    module_labels: frozenset = frozenset()
    #: (device seconds since boot, type, severity, event_key, signals) as the device recorded them.
    timeline: list = field(default_factory=list)
    local_rows: list = field(default_factory=list)
    backend: list = field(default_factory=list)
    api_events: list = field(default_factory=list)
    explain: dict = field(default_factory=dict)
    overlay: Optional[dict] = None
    buzzer: list = field(default_factory=list)
    detector_calls: int = 0
    module_builds: int = 0
    modules_seen: set = field(default_factory=set)
    driver_settings: dict = field(default_factory=dict)
    final_preset: Optional[str] = None

    def driver_alerts(self) -> list[tuple]:
        return [(t, kind) for t, kind, severity, *_ in self.timeline
                if kind in DRIVER_TYPES and severity in ("loud", "soft")]

    def rule_alerts(self) -> list[tuple]:
        return [row for row in self.timeline if row[1] == "cell phone"]


def _journey(with_rule: bool) -> Journey:
    out = Journey(with_rule)
    with tempfile.TemporaryDirectory() as where, SimStack(Path(where)) as stack:
        app = stack.backend.app

        def no_model(*_a, **_k):
            raise AssertionError("SIMULATED journey: the language model must not be asked")

        app._ai_configure_llm = no_model  # the deterministic parser alone reads the prompt
        device = stack.boot(scenario(), dashboard_port=0)
        runtime = device.runtime
        assert runtime is not None and runtime.dashboard is not None
        stack.claim()
        t0 = stack.clock.monotonic()

        def now() -> float:
            return round(stack.clock.monotonic() - t0, 2)

        device.advance(4.0)
        out.role, out.preset = runtime.role.name, runtime.active_preset
        drawn = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": [AREA]}, headers=stack.headers)
        assert drawn.status_code == 200 and drawn.json()["rejected"] == 0, drawn.text
        phone = LocalPhone(runtime.dashboard.url, name="Owner iPhone")
        enrolled = stack.api.post(f"/v1/devices/{stack.device_id}/local-clients",
                                  json={"public_key": la.b64u(phone.key.public_key), "name": "Owner iPhone"},
                                  headers=stack.headers)
        assert enrolled.status_code == 200, enrolled.text
        device.advance(C + 1.0 - now())  # calibrated and protecting before anything is asked
        out.status_before = stack.status()

        if with_rule:
            response = stack.ai_configure(PROMPT)
            assert response.status_code == 200, response.text
            out.preview = response.json()
            rules = [{**r, "timezone": "UTC"} for r in out.preview.get("rules") or []]
            activated = stack.set_preset("", draft_version=out.preview.get("draft_version"), rules=rules)
            assert activated.status_code == 200, activated.text
            out.activation = activated.json()
            out.acknowledged_before_device = bool(stack.status()["rules"]["active"])
        device.advance(4.0)
        out.status_running = stack.status()
        out.engine_rules = [(r.predicate, r.detection, r.zone, r.dwell_s) for r in runtime.rule_engine.rules]
        out.module_labels = frozenset(runtime.module.available_labels() or ())

        seen: set[str] = set()
        viewing = False
        while now() < RUN_S - 1e-6:
            if not viewing and out.overlay is None and now() >= OVERLAY_OPEN[0]:
                # The app's AI view opens (both runs alike; it is presentation only).
                runtime.dashboard.viewer_joined(overlay=True)
                viewing = True
            elif viewing and now() >= OVERLAY_OPEN[1]:
                viewing = False
                # What the app's AI view receives with the next streamed frame (sd-overlay/1).
                frame = runtime.dashboard._current_frame(overlay=True)
                out.overlay = ov.decode_header(frame.overlay) if frame is not None and frame.overlay else {}
                runtime.dashboard.viewer_left(overlay=True)
            device.advance(0.25)
            out.modules_seen.add(runtime.module_name)
            page, _ = runtime.dashboard.events_response({"limit": ["200"]})
            for row in reversed(page.get("events") or []):
                if row["event_key"] not in seen:
                    seen.add(row["event_key"])
                    out.timeline.append((now(), row["type"], row["severity"], row["event_key"], row["signals"]))
                    out.local_rows.append(row)

        out.backend = stack.events()
        identity = stack.row()["local_identity"]
        phone.trust({"device_public_key": identity["device_public_key"], "device_id": runtime._local_device_id()})
        assert phone.open_session().status == 200
        local = phone.get("/api/events?limit=200")
        assert local.status == 200
        out.api_events = list((local.body or {}).get("events") or [])
        custom = [e for e in out.backend if e.get("source") == "custom_rule"]
        if custom:
            explained = stack.api.get(f"/v1/devices/{stack.device_id}/events/{custom[0]['id']}/explain",
                                      headers=stack.headers)
            assert explained.status_code == 200, explained.text
            out.explain = explained.json()
        out.buzzer = list(device.buzzer.played)
        assert device.sim is not None
        out.detector_calls = device.sim.detector.calls
        out.module_builds = len(device.sim.detectors)
        out.driver_settings = dict(runtime.config["modules"]["driver_attention"])
        out.final_preset = runtime.active_preset
    return out


@pytest.fixture(scope="module")
def with_rule() -> Journey:
    return _journey(True)


@pytest.fixture(scope="module")
def baseline() -> Journey:
    return _journey(False)


# ---------------------------------------------------------------------- the prompt
def test_the_prompt_is_read_by_the_deterministic_parser_into_one_typed_policy():
    parsed = jp.parse(PROMPT, zones=[AREA["name"]])
    assert parsed is not None and parsed.unsupported == []
    (policy,) = parsed.policies
    rule = policy.rule
    assert (rule["target"], rule["predicate"], rule["zone"], rule["dwell_s"]) == \
        ("cell phone", "lingering", "console", DWELL_S)
    assert policy.status == jp.STATUS_READY and policy.required_capabilities == ["camera", "inference", "objects"]
    # Named instead of "this area" (when more than one area is drawn), the same rule.
    named = jp.parse("Alert me when a cell phone stays in the console for 10 seconds.", zones=["console", "dash"])
    assert named is not None and [p.rule["zone"] for p in named.policies] == ["console"]
    # "a phone" alone is not a label the grammar reads: the demo must say "a cell phone".
    assert jp.parse("Alert me when a phone stays in this area for 10 seconds.", zones=["console"]) is None


# ---------------------------------------------------------------------- the journey
def test_the_preview_is_a_typed_policy_this_device_can_serve(with_rule):
    preview = with_rule.preview
    assert preview["source"] == "policy_parser" and preview["preset"] == "" and preview["unsupported"] == []
    (policy,) = preview["policies"]
    assert policy["targets"] == ["cell phone"] and policy["condition"] == "lingering"
    assert policy["temporal"] == {"dwell_s": DWELL_S} and policy["zone"] == {"name": "console", "drawn": True}
    assert policy["status"] == "ready" and policy["severity"] == "warning" and policy["notify"] == "push"
    (rule,) = preview["rules"]
    assert all(isinstance(v, (str, int, float)) for v in rule.values()), "data only: no code reaches the device"
    # Capability check on THIS device: what the policy needs is what the device reports available,
    # and the target is a label the mandatory driver module's own detector serves.
    available = set(with_rule.status_before["capabilities"]["available"])
    assert set(policy["required_capabilities"]) <= available, (policy["required_capabilities"], available)
    assert "cell phone" in with_rule.module_labels
    summary = policy["explanation"]["summary"]
    assert summary.startswith("Alert when a cell phone stays inside the area 'console' for 10 seconds or longer")


def test_the_device_acknowledges_and_runs_it_beside_driver_safety(with_rule):
    assert with_rule.activation["rules_active"] == 1 and with_rule.activation["unsupported"] == []
    assert not with_rule.acknowledged_before_device, "active only once the DEVICE reports it"
    rules = with_rule.status_running["rules"]
    assert rules["active"] == ["rule0:lingering:cell phone"] and rules["rejected"] is None and rules["additive"]
    assert with_rule.status_running["mode"] == "standard_plus_custom", "beside Driver Safety, not instead of it"
    assert with_rule.status_running["monitoring_state"] == "ready"
    assert with_rule.engine_rules == [("lingering", "cell phone", "console", DWELL_S)]


def test_the_device_is_driver_safety_and_stays_driver_safety(with_rule, baseline):
    for run in (with_rule, baseline):
        assert run.role == "driver_safety" and run.preset == run.final_preset
        assert run.modules_seen == {"driver_attention"}, "Driver Safety's module never stopped"
        for status in (run.status_before, run.status_running):
            assert (status["monitoring_state"], status["protection"]) == ("ready", "protecting"), status
    assert baseline.status_running["mode"] == "standard"
    # Not relaxed: every driver setting (thresholds included) is what config.yaml ships, in both runs.
    shipped = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    floors = {f.key for f in role_named("driver_safety").parameter_floors}
    for key in floors:
        assert with_rule.driver_settings[key] == baseline.driver_settings[key] == \
            shipped["modules"]["driver_attention"][key], key
    assert config_violations({**shipped, "modules": {"driver_attention": with_rule.driver_settings}}) == []


def test_it_uses_the_shared_inference_no_second_model_or_pipeline(with_rule, baseline):
    assert with_rule.module_builds == baseline.module_builds == 1, "no second module was built"
    assert with_rule.detector_calls == baseline.detector_calls > 0, "the rule added no detector run"


def test_one_rule_alert_at_ten_seconds_and_none_for_the_short_visit(with_rule, baseline):
    (alert,) = with_rule.rule_alerts()
    t, _kind, severity, _key, signals = alert
    assert LONG_VISIT[0] + DWELL_S <= t <= LONG_VISIT[0] + DWELL_S + DWELL_SLACK_S, t
    assert severity == "soft" and signals == "A cell phone stays in this area for 10 seconds"
    assert baseline.rule_alerts() == []


def test_driver_safety_alerts_are_unchanged_by_the_rule(with_rule, baseline):
    driver = with_rule.driver_alerts()
    assert [kind for _t, kind in driver] == ["drowsiness", "distraction"], driver
    assert driver == baseline.driver_alerts(), "same alerts at the same device times, with or without the rule"
    (drowsy_t, _), _ = driver
    (rule_t, *_), = with_rule.rule_alerts()
    assert abs(drowsy_t - rule_t) <= 1.0, "the overlap is exercised: drowsiness and the rule fire together"
    for run in (with_rule, baseline):
        for row in run.local_rows:
            if row["type"] in DRIVER_TYPES and row["severity"] in ("loud", "soft"):
                (output,) = row["alert"]["outputs"]
                assert row["alert"]["decision"] == "alerted" and output["played"] == 1 and output["skipped"] == 0
    assert [p for p in with_rule.buzzer if p[0] != "cell phone"] == baseline.buzzer
    assert ("drowsiness", "critical", "critical") in baseline.buzzer
    assert ("cell phone", "warning", "double") in with_rule.buzzer


def test_the_alert_is_the_same_occurrence_in_both_histories(with_rule):
    (custom,) = [e for e in with_rule.backend if e.get("source") == "custom_rule"]
    (_t, _kind, _severity, key, _signals), = with_rule.rule_alerts()
    assert custom["type"] == "cell phone" and custom["event_key"] == key
    assert custom["signals"] == "A cell phone stays in this area for 10 seconds", "the rule's own words say why"
    assert custom["simulated"] is True and custom["alert"]["decision"] == "alerted"
    assert custom["alert"]["output"] == "simulated" and custom["duration"] >= DWELL_S
    (local,) = [e for e in with_rule.api_events if e["type"] == "cell phone"]
    assert local["event_key"] == key and local["sync"] == "sent" and local["simulated"] is True
    driver_keys = {e["event_key"] for e in with_rule.backend if e["type"] in DRIVER_TYPES}
    assert driver_keys and driver_keys <= {e["event_key"] for e in with_rule.api_events}
    # The history's "why" for the custom-rule event (facts only: the twin has no model).
    explain = with_rule.explain
    assert explain["headline"] == "Cell phone" and explain["source"] == "facts"
    facts = {f["label"]: f["detail"] for f in explain["facts"]}
    assert "A cell phone stays in this area for 10 seconds" in facts and facts["Area"] == "console"
    assert explain["rule"] == "You were alerted because your rule alerts after 10 seconds."


def test_the_live_view_shows_the_rules_why_line_tied_to_its_row(with_rule):
    doc = with_rule.overlay
    assert doc and doc["schema"] == ov.SCHEMA and doc["sim"] is True and doc["module"] == "driver_attention"
    (_t, _kind, _severity, key, signals), = with_rule.rule_alerts()
    last = doc["last_event"]
    assert (last["type"], last["severity"], last["signals"], last["event_key"]) == \
        ("cell phone", "soft", signals, key), last
    assert any(d["label"] == "phone" for d in doc["det"] or []), "the phone is drawn in the AI view"


# ---------------------------------------------------------------------- which policies fit beside Driver Safety
#: (request, a rule to send instead of the preview's (None = the preview's own), expected outcome).
#: "runs": the device runs it beside driver_attention. "device_refuses": the backend
#: previews it as ready, the device refuses the set (its target is not a label the
#: driver module's detector serves) and keeps what it ran. "nothing_proposed": no rule
#: comes out of the preview (refused up front, or nothing could read it).
COMPATIBILITY = (
    # M13's showcase rule and the app's own suggestions (ios/Sources/JupiterViews.swift)
    ("Alert me if someone stays inside the marked area for longer than 10 seconds", None, "runs"),
    ("Tell me when someone enters the marked area", None, "runs"),
    ("Let me know when nobody is in the marked area for 5 minutes", None, "runs"),
    ("Alert me if more than 3 people are in the marked area", None, "runs"),
    ("Tell me if a dog enters the marked area", None, "device_refuses"),
    # the M14 demo rule, and another object the driver module's detector serves
    ("Alert me when a cell phone stays in the marked area for 10 seconds", None, "runs"),
    ("Tell me when a bottle enters the marked area", None, "runs"),
    # a pose composition a language model proposes for "alert me if I fall asleep at my desk"
    # (docs/JUPITER.md): served by the pose module, which Driver Safety does not run
    ("Alert me if I fall asleep at my desk", {"target": "resting", "predicate": "present", "persist_s": 60.0,
                                              "description": "resting for 60 s"}, "device_refuses"),
    # refused before any rule exists, on every role
    ("Tell me if someone I don't recognise enters the yard", None, "nothing_proposed"),
    ("Alert me about high risk driving", None, "nothing_proposed"),
    # M11's phrasing: not in the deterministic grammar; with no model configured nothing is guessed
    ("Alert me if a phone is visible for 5 seconds", None, "nothing_proposed"),
)


def test_which_existing_policies_run_beside_driver_safety(tmp_path):
    """One driver_safety virtual device; each policy activated in turn the way the app does.

    The twin's backend has no language model configured: a request the
    deterministic grammar does not read comes back `interpreter_unavailable`."""
    outcomes = {}
    with SimStack(tmp_path) as stack:
        device = stack.boot(Scenario("m14_compat", "An attentive driver.", (Segment(0.0, ATTENTIVE),)))
        runtime = device.runtime
        assert runtime is not None
        stack.claim()
        marked = {**AREA, "name": "marked area"}
        drawn = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": [marked]}, headers=stack.headers)
        assert drawn.status_code == 200
        device.advance(C + 1.0)
        assert runtime.role.name == "driver_safety"
        running: list = []
        for request, override, _expected in COMPATIBILITY:
            preview = stack.ai_configure(request).json()
            rules = [override] if override is not None else preview.get("rules") or []
            if not rules:
                outcomes[request] = "nothing_proposed"
                assert preview.get("unsupported"), preview
                continue
            if override is None:
                assert preview["source"] == "policy_parser", preview
                assert [p["status"] for p in preview["policies"]] == ["ready"], preview
            activated = stack.set_preset("", draft_version=preview["draft_version"], rules=rules)
            assert activated.status_code == 200 and activated.json()["rules_active"] == 1, activated.text
            device.advance(6.0)
            status = stack.status()
            assert status["mode"] == "standard_plus_custom" and runtime.module_name == "driver_attention"
            assert (status["monitoring_state"], status["protection"]) == ("ready", "protecting"), status
            engine = [(r.predicate, r.detection) for r in runtime.rule_engine.rules]
            if status["rules"]["rejected"] is None:
                outcomes[request] = "runs"
                running = engine
                assert engine == [(rules[0]["predicate"], rules[0]["target"])]
            else:
                outcomes[request] = "device_refuses"
                assert any(rules[0]["target"] in reason for reason in status["rules"]["rejected"]["reasons"])
                assert engine == running, "a refused set leaves the running rule in place"
            # Whatever happened, the role's floor never moved.
            assert runtime.module_name == "driver_attention" and runtime.role.name == "driver_safety"
    assert outcomes == {request: expected for request, _o, expected in COMPATIBILITY}
