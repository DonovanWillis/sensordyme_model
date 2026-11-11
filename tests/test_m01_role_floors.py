"""M0.1 — deployment roles, additive custom rules, fail-closed rules, explicit stop.

Acceptance criteria (architecture plan §17), all deterministic:

  3  On a driver_safety device a custom rule set ADDS alerts and the scripted
     drowsiness scenario alerts on the identical timeline with and without it.
     (Before: activating any custom rule swapped driver_attention out and
     discarded its events.)
  4  Fuzzing config, preset, rule, zone, dashboard, uplink and backend inputs
     finds no reachable state where a scripted 4 s eye closure goes unalerted
     while the device reports "protecting".
  5  An explicit, confirmed stop reports "stopped" on the device API, the uplink
     report and the backend/app model; resume restores "protecting".
  6  A rule set with an unknown predicate, an undetectable target or an
     undefined zone is rejected as a whole; the previous rules stay; the reason
     is reported. (Before: "enters" fired as plain presence after 0.5 s.)
  7  Roles without a floor are unaffected by driver restrictions.

Nothing here verifies that a buzzer is physically audible.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest
import yaml
from test_driver_attention_scenarios import _FakeFaceMeshResult, _make_module, make_fake_landmarks
from test_m01_alert_episodes import RecordingActuator

import modules.driver_attention as driver_attention_module
from core import roles
from core.alerts import AlertManager
from core.config_schema import validate
from core.events import Event
from core.logger import EventLogger
from core.monitoring_mode import MonitoringMode
from core.presets import load_presets
from core.rules import RuleEngine, validate_rule_set
from main import ALERT_EVENT_TYPES_PI, alert_types_for, build_alert_manager
from modules.base import VisionModule, register_module
from services.runtime import DeviceRuntime

REPO = Path(__file__).resolve().parent.parent


def _repo_config() -> dict:
    return yaml.safe_load((REPO / "config.yaml").read_text())


# ====================================================================== config floors
def test_the_shipped_config_declares_driver_safety_and_satisfies_it():
    config = _repo_config()
    assert config["role"] == "driver_safety"
    errors, _ = validate(config)
    assert errors == []


@pytest.mark.parametrize(("mutate", "expected"), [
    (lambda c: c.update(module="occupancy"), "requires `module:`"),
    (lambda c: c["alerts"].update(sinks=["console"]), "audible alert sink"),
    (lambda c: c["modules"]["driver_attention"].update(drowsy_seconds=3.0), "exceeds the safety floor"),
    (lambda c: c["modules"]["driver_attention"].update(face_lost_seconds=10), "exceeds the safety floor"),
    (lambda c: c["alerts"]["soft_types"].append("drowsiness"), "forbids `drowsiness` in alerts.soft_types"),
    (lambda c: c["alerts"].update(loud_types=[t for t in c["alerts"]["loud_types"] if t != "phone_use"]),
     "requires `phone_use`"),
    (lambda c: c["alerts"].update(critical_types=["drowsiness"]), "requires `attention_lost` in alerts.critical_types"),
    (lambda c: c["alerts"].update(categories={"critical": {"repeat_s": 0}}), "keep repeating"),
    (lambda c: c.update(role="no_such_role"), "`role:` must be one of"),
])
def test_a_config_that_weakens_the_driver_floor_is_an_error(mutate, expected):
    config = _repo_config()
    mutate(config)
    errors, _ = validate(config)
    assert any(expected in e for e in errors), errors


def test_a_stricter_threshold_is_allowed():
    config = _repo_config()
    config["modules"]["driver_attention"]["drowsy_seconds"] = 1.5
    assert validate(config)[0] == []


@pytest.mark.parametrize("role_name", ["general", "child_monitoring", "occupancy", "property", "workplace"])
def test_other_roles_carry_no_driver_restrictions(role_name):
    """Acceptance 7: no driver floor leaks into another role."""
    config = _repo_config()
    config.update(role=role_name, module="occupancy")
    config["alerts"].update(sinks=["console"], critical_types=[], loud_types=[], soft_types=[])
    config["modules"]["driver_attention"]["drowsy_seconds"] = 9.0
    assert validate(config)[0] == []


def test_a_config_without_a_role_is_general():
    config = _repo_config()
    del config["role"]
    config["module"] = "occupancy"
    assert validate(config)[0] == []
    assert roles.role_for_config(config).name == "general"


def test_mac_max_is_not_a_protected_profile():
    config = _repo_config()
    config["profile"] = "mac_max"
    errors, warnings = validate(config)
    assert errors == []
    assert any("not enforced on the mac_max developer profile" in w for w in warnings)


# ====================================================================== preset floors
PRESETS = load_presets()
DRIVER = roles.ROLES["driver_safety"]


@pytest.mark.parametrize("name", ["security_monitoring", "elderly_monitoring", "baby_monitoring",
                                  "security_monitoring:away"])
def test_a_driver_device_refuses_presets_that_stop_driver_monitoring(name):
    reason = roles.preset_violation(DRIVER, name, PRESETS)
    assert reason and "driver_safety" in reason


@pytest.mark.parametrize("name", ["driver_safety", "driver_safety:standard", "driver_safety:strict",
                                  "driver_safety:teen_driver", "driver_safety:long_haul"])
def test_every_shipped_driver_variant_respects_the_floor(name):
    assert roles.preset_violation(DRIVER, name, PRESETS) is None


def test_a_variant_that_loosens_a_threshold_is_refused():
    presets = copy.deepcopy(PRESETS)
    presets["driver_safety"]["variants"]["lax"] = {"overrides": {"drowsy_seconds": 4.0}}
    reason = roles.preset_violation(DRIVER, "driver_safety:lax", presets)
    assert reason and "exceeds the safety floor" in reason


def test_a_general_device_may_select_any_preset():
    for name in PRESETS:
        assert roles.preset_violation(roles.ROLES["general"], name, PRESETS) is None


# ====================================================================== real driver runtime
class ClockedActuator(RecordingActuator):
    """Records each actuation at the injected time."""

    def __init__(self, clock) -> None:
        super().__init__()
        self._clock = clock

    def play(self, pattern, event, category) -> None:
        self._now = self._clock["now"]
        super().play(pattern, event, category)


class FakeUplink:
    def __init__(self) -> None:
        self.rules = None
        self.zones = None
        self.preset = None
        self.protection = None
        self.monitoring_reports: list[dict] = []
        self.version = ""

    def pop_pending_rules(self):
        rules, self.rules = self.rules, None
        return rules

    def pop_pending_zones(self):
        zones, self.zones = self.zones, None
        return zones

    def pop_pending_preset(self):
        preset, self.preset = self.preset, None
        return preset

    def pop_pending_protection(self):
        request, self.protection = self.protection, None
        return request

    def rules_version(self):
        return self.version

    def pending_instruction(self):
        return ""

    def set_monitoring(self, described):
        self.monitoring_reports.append(described)

    def set_active_preset(self, name):
        pass

    def push_event(self, *args, **kwargs):
        pass


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    return t


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Per-test monitoring and device state. The session-wide monitoring file
    (tests/conftest.py) is shared, and these tests deliberately save invalid
    rule sets that must not leak into any other test."""
    import core.monitoring_config as monitoring_config
    import services.runtime as runtime_module

    path = tmp_path / "monitoring.json"
    monkeypatch.setattr(runtime_module, "load_or_empty", lambda: monitoring_config.load_or_empty(path))
    monkeypatch.setenv("SENSORDYME_DEVICE_STATE_PATH", str(tmp_path / "device_state.json"))
    monkeypatch.chdir(tmp_path)
    return path


def _driver_runtime(tmp_path, clock, monkeypatch, *, closed):
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio",
                        lambda *pts: 0.05 if closed["value"] else 0.40)
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    face = make_fake_landmarks()
    module._face_mesh.process = lambda rgb, lm=face: _FakeFaceMeshResult(lm)
    config = _repo_config()
    actuator = ClockedActuator(clock)
    manager = build_alert_manager({**config["alerts"], "sinks": []}, False)
    manager._sinks = [actuator]
    runtime = DeviceRuntime(
        config=config, profile="pi", module=module, module_name="driver_attention",
        alert_manager=manager, alert_cfg=config["alerts"], alert_event_types=ALERT_EVENT_TYPES_PI,
        alert_types_for=alert_types_for, presets=PRESETS, active_preset="driver_safety",
        display=False, debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    runtime.uplink = FakeUplink()
    return runtime, module, frame, actuator


def _drive(runtime, module, frame, clock, seconds, step=0.05):
    """The runtime's per-frame production path (dispatch + custom rules), injected time."""
    for _ in range(round(seconds / step)):
        runtime._apply_pending_commands()
        if runtime._stopped is None:
            events = module.process(frame)
            runtime._dispatch_module_events(module, events, clock["now"])
            runtime._evaluate_custom_rules(module, frame, clock["now"])
        clock["now"] = round(clock["now"] + step, 6)


def _drowsy_script(runtime, module, frame, clock, closed):
    _drive(runtime, module, frame, clock, 2.0)
    closed["value"] = True
    _drive(runtime, module, frame, clock, 12.0)       # sustained closure: repeats
    closed["value"] = False
    _drive(runtime, module, frame, clock, 3.0)
    closed["value"] = True
    _drive(runtime, module, frame, clock, 4.0)        # second episode within 15 s
    closed["value"] = False
    _drive(runtime, module, frame, clock, 1.0)


PHONE_RULE = {"target": "cell phone", "predicate": "present", "severity": "warning",
              "persist_s": 1.0, "notify": "push", "description": "phone visible"}


def test_custom_rules_add_to_driver_safety_and_keep_its_timeline_identical(
        tmp_path, clock, monkeypatch, isolated_state):
    """Acceptance 3."""
    timelines = []
    for with_rules in (False, True):
        clock["now"] = 0.0
        closed = {"value": False}
        runtime, module, frame, actuator = _driver_runtime(tmp_path / str(with_rules), clock, monkeypatch,
                                                           closed=closed)
        if with_rules:
            runtime.uplink.rules = [PHONE_RULE]
            runtime.uplink.version = "v1"
            runtime._apply_pending_commands()
            assert runtime.mode is MonitoringMode.ADDITIVE
            assert runtime.module_name == "driver_attention", "custom rules swapped driver safety out"
            assert runtime.protection_state()["state"] == "protecting"
            # A phone the (stubbed) detector sees, so the custom rule has something to fire on.
            module._last_overlay_boxes = [((100.0, 100.0, 160.0, 220.0), "cell phone")]
            module._last_overlay_confidences = [0.9]
        start = clock["now"]
        _drowsy_script(runtime, module, frame, clock, closed)
        drowsy = [(round(t - start, 2), c, p) for t, k, c, p in actuator.played if k == "drowsiness"]
        timelines.append(drowsy)
        if with_rules:
            assert any(k == "cell phone" for _, k, _, _ in actuator.played), "the custom rule never alerted"
        runtime.logger.close()
    assert timelines[0], "the scripted closure never alerted"
    assert timelines[0] == timelines[1]
    assert len({t for t, _, _ in timelines[0]}) >= 3  # repeated, and the second episode alerted


def test_before_m01_custom_rules_replaced_driver_safety_now_they_do_not(
        tmp_path, clock, monkeypatch, isolated_state):
    closed = {"value": False}
    runtime, _module, _frame, _actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.uplink.rules = [PHONE_RULE]
    runtime._apply_pending_commands()
    assert runtime.mode is not MonitoringMode.AI_CUSTOM
    report = runtime.uplink.monitoring_reports[-1]
    assert report["base_monitoring_disabled"] is False
    assert report["mode"] == MonitoringMode.ADDITIVE.value
    runtime.logger.close()


@pytest.mark.parametrize(("bad_rule", "reason"), [
    ({"target": "person", "predicate": "enters"}, "unsupported condition 'enters'"),
    ({"target": "dog", "predicate": "present"}, "cannot detect 'dog'"),
    ({"target": "person", "predicate": "present", "zone": "driveway"}, "area 'driveway' is not defined"),
    ({"target": "person", "predicate": "near"}, "needs a second object"),
])
def test_an_unsupported_rule_rejects_the_whole_set_and_keeps_the_previous_rules(
        tmp_path, clock, monkeypatch, isolated_state, bad_rule, reason):
    """Acceptance 6."""
    closed = {"value": False}
    runtime, _module, _frame, _ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.uplink.rules = [PHONE_RULE]
    runtime.uplink.version = "good"
    runtime._apply_pending_commands()
    assert [r.detection for r in runtime.rule_engine.rules] == ["cell phone"]

    runtime.uplink.rules = [PHONE_RULE, bad_rule]
    runtime.uplink.version = "bad"
    runtime._apply_pending_commands()
    assert [r.detection for r in runtime.rule_engine.rules] == ["cell phone"]  # previous kept
    rejection = runtime.uplink.monitoring_reports[-1]["last_rejection"]
    assert rejection["kind"] == "rules" and rejection["subject"] == "bad"
    assert any(reason in r for r in rejection["reasons"])
    runtime.logger.close()


def test_the_engine_itself_never_treats_an_unknown_predicate_as_presence():
    """Before M0.1 an 'enters' rule fired as soon as a person was seen."""
    from core.detections import Detection, DetectionFrame

    engine = RuleEngine([{"target": "person", "predicate": "enters", "persist_s": 0.5}])
    assert engine.rules == []
    frame = DetectionFrame(width=640, height=480,
                           detections=(Detection(label="person", confidence=0.9, box=(1, 1, 50, 50)),))
    fired = [a for t in range(30) for a in engine.observe_frame(frame, now=t * 0.1)]
    assert fired == []


def test_invalid_saved_rules_are_not_loaded_at_boot(tmp_path, clock, monkeypatch, isolated_state):
    from core.monitoring_config import load_or_empty

    load_or_empty(isolated_state).save([{"target": "person", "predicate": "enters"}], instruction="old")
    closed = {"value": False}
    runtime, *_ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    assert runtime.rule_engine.rules == []
    assert runtime._last_rejection["kind"] == "rules"
    assert runtime.mode is MonitoringMode.STANDARD
    runtime.logger.close()


def test_a_preset_command_that_would_stop_driver_monitoring_is_refused(
        tmp_path, clock, monkeypatch, isolated_state):
    closed = {"value": False}
    runtime, *_ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.uplink.preset = "security_monitoring"
    runtime._apply_pending_commands()
    assert runtime.module_name == "driver_attention"
    assert runtime.active_preset == "driver_safety"
    rejection = runtime.uplink.monitoring_reports[-1]["last_rejection"]
    assert rejection["kind"] == "preset" and "driver_safety" in rejection["reasons"][0]
    runtime.logger.close()


# ====================================================================== explicit stop
def test_stop_requires_confirmation(tmp_path, clock, monkeypatch, isolated_state):
    closed = {"value": False}
    runtime, *_ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.uplink.protection = {"action": "stop"}
    runtime._apply_pending_commands()
    assert runtime.protection_state()["state"] == "protecting"
    assert runtime._last_rejection["kind"] == "protection"
    runtime.logger.close()


def test_a_confirmed_stop_is_reported_everywhere_and_resume_restores(
        tmp_path, clock, monkeypatch, isolated_state):
    """Acceptance 5 (device API + uplink report; the backend/app half is below)."""
    closed = {"value": False}
    runtime, module, frame, actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    closed["value"] = True
    _drive(runtime, module, frame, clock, 3.0)
    assert actuator.played  # an episode is open and alerting
    runtime.uplink.protection = {"action": "stop", "confirm": True}
    t_cmd = clock["now"]
    _drive(runtime, module, frame, clock, 0.05)  # the very next loop iteration
    assert runtime.protection_state()["state"] == "stopped"
    assert runtime.uplink.monitoring_reports[-1]["protection"]["state"] == "stopped"
    assert clock["now"] - t_cmd <= 2.0
    played = len(actuator.played)
    _drive(runtime, module, frame, clock, 10.0)  # eyes still closed, but stopped
    assert len(actuator.played) == played, "a stopped device must not claim to alert"
    assert runtime.alert_manager.open_episodes() == []
    runtime.uplink.protection = {"action": "resume"}
    _drive(runtime, module, frame, clock, 0.05)
    assert runtime.protection_state()["state"] == "protecting"
    assert runtime.uplink.monitoring_reports[-1]["protection"]["state"] == "protecting"
    runtime.logger.close()


def test_stop_is_not_persisted_across_a_restart(tmp_path, clock, monkeypatch, isolated_state):
    closed = {"value": False}
    runtime, *_ = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime._apply_protection({"action": "stop", "confirm": True}, source="test")
    runtime.logger.close()
    again, *_ = _driver_runtime(tmp_path / "again", clock, monkeypatch, closed=closed)
    assert again.protection_state()["state"] == "protecting"
    again.logger.close()


# ====================================================================== other roles: unchanged
@register_module("m01_other_a")
class _OtherA(VisionModule):
    name = "m01_other_a"

    def process(self, frame):
        return []


@register_module("m01_other_b")
class _OtherB(VisionModule):
    name = "m01_other_b"

    def process(self, frame):
        return []


def test_a_general_device_keeps_free_preset_switching_and_replacing_custom_rules(tmp_path, isolated_state):
    """Acceptance 7, at runtime: no driver floor, presets switch freely."""
    presets = {"a": {"module": "m01_other_a"}, "b": {"module": "m01_other_b"}}
    runtime = DeviceRuntime(
        config={"modules": {"m01_other_a": {}, "m01_other_b": {}}}, profile="pi",
        module=_OtherA({}), module_name="m01_other_a", alert_manager=AlertManager(sinks=[]),
        alert_cfg={}, alert_event_types=set(), alert_types_for=lambda n, p=None: set(),
        presets=presets, active_preset="a", display=False, debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    assert runtime.role.name == "general"
    runtime._switch_preset("b")
    assert runtime.module_name == "m01_other_b"
    assert runtime.protection_state()["state"] == "protecting"
    assert runtime._resolve_mode("m01_other_b") is MonitoringMode.STANDARD
    runtime.monitoring.save([{"target": "person"}])
    assert runtime._resolve_mode("m01_other_b") is MonitoringMode.AI_CUSTOM  # unchanged behaviour
    runtime.logger.close()


# ====================================================================== backend + app model
@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import importlib
    import sys
    import time as _time

    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Car", "AA-BB-CC", "dk_test", _time.time()))
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return client, headers


def _ingest(client, monitoring):
    resp = client.post("/v1/ingest", json={"events": [], "active_preset": "driver_safety",
                                           "monitoring": monitoring}, headers={"X-Device-Key": "dk_test"})
    assert resp.status_code == 200
    return resp.json()["command"]


DRIVER_REPORT = {"active": False, "role": DRIVER.describe(),
                 "protection": {"state": "protecting", "role": "driver_safety", "reasons": []}}


def test_backend_refuses_a_preset_the_device_role_forbids(backend):
    client, headers = backend
    _ingest(client, DRIVER_REPORT)
    resp = client.post("/v1/devices/1/preset", json={"preset": "security_monitoring"}, headers=headers)
    assert resp.status_code == 409
    assert "driver_safety" in resp.json()["detail"]
    ok = client.post("/v1/devices/1/preset", json={"preset": "driver_safety:strict"}, headers=headers)
    assert ok.status_code == 200


def test_backend_stop_needs_confirmation_is_delivered_once_and_visible_to_the_app(backend):
    """Acceptance 5, backend/app half: the app model shows the device's own state."""
    client, headers = backend
    assert client.post("/v1/devices/1/protection", json={"action": "stop"}, headers=headers).status_code == 400
    assert client.post("/v1/devices/1/protection", json={"action": "stop", "confirm": True},
                       headers=headers).status_code == 200
    command = _ingest(client, DRIVER_REPORT)
    assert command["set_protection"] == {"action": "stop", "confirm": True}
    stopped = {**DRIVER_REPORT, "protection": {"state": "stopped", "role": "driver_safety", "reasons": []}}
    assert (_ingest(client, stopped) or {}).get("set_protection") is None  # converged: cleared
    device = client.get("/v1/devices", headers=headers).json()["devices"][0]
    assert device["monitoring"]["protection"]["state"] == "stopped"
    # A rebooted device reports protecting again and is NOT re-stopped by a stale command.
    assert (_ingest(client, DRIVER_REPORT) or {}).get("set_protection") is None


# ====================================================================== acceptance 4: floor fuzz
_CHANNELS = ("config", "preset", "rules", "zones", "dashboard_preset", "uplink_preset", "backend_preset",
             "protection_unconfirmed")


def _random_config(rng: random.Random) -> dict:
    config = _repo_config()
    choice = rng.randrange(10)
    if choice == 0:
        config["module"] = rng.choice(["occupancy", "fall_detection", "baby_monitoring", "driver_attention"])
    elif choice == 1:
        config["alerts"]["sinks"] = rng.sample(["console", "sound", "gpio_buzzer", "siren"], rng.randint(0, 3))
    elif choice == 2:
        key = rng.choice([f.key for f in DRIVER.parameter_floors])
        config["modules"]["driver_attention"][key] = round(rng.uniform(0.2, 12.0), 2)
    elif choice == 3:
        drop = rng.choice(sorted(DRIVER.mandatory_alert_types))
        config["alerts"]["loud_types"] = [t for t in config["alerts"]["loud_types"] if t != drop]
        if rng.random() < 0.5:
            config["alerts"]["critical_types"] = [t for t in config["alerts"]["critical_types"] if t != drop]
    elif choice == 4:
        config["alerts"]["soft_types"] = config["alerts"]["soft_types"] + [rng.choice(sorted(DRIVER.mandatory_alert_types))]
    elif choice == 5:
        config["alerts"]["categories"] = {rng.choice(["critical", "warning"]): {
            "repeat_s": rng.choice([0, 4.0, 6.0]), "max_repeats": rng.choice([0, 5, 30])}}
    elif choice == 6:
        config["alerts"]["critical_types"] = rng.sample(["drowsiness", "attention_lost", "eyes_down"], rng.randint(0, 3))
    elif choice == 7:
        config["profile"] = rng.choice(["pi", "mac_max"])
    elif choice == 8:
        config["role"] = rng.choice(["driver_safety", "driver_safety", "bogus"])
    return config


def _invariants_hold(config: dict) -> bool:
    """What a driver_safety pi config that passed validation must still guarantee."""
    if config.get("role") != "driver_safety" or config.get("profile", "pi") != "pi":
        return True
    alerts, module_cfg = config["alerts"], config["modules"]["driver_attention"]
    loud = set(alerts.get("loud_types", [])) | set(alerts.get("critical_types", []))
    return (
        config["module"] == "driver_attention"
        and bool(roles.AUDIBLE_SINKS & set(alerts.get("sinks", [])))
        and DRIVER.mandatory_alert_types <= loud
        and not (DRIVER.mandatory_alert_types & set(alerts.get("soft_types", [])))
        and DRIVER.critical_types <= set(alerts.get("critical_types", []))
        and all(f.violation(module_cfg[f.key]) is None for f in DRIVER.parameter_floors if f.key in module_cfg)
    )


def test_fuzz_no_accepted_config_weakens_the_driver_floor():
    """Acceptance 4 (config channel): 1,000 generated configs; every one the
    validator accepts still guarantees the floor."""
    rng = random.Random(20261110)
    accepted = rejected = 0
    for _ in range(1000):
        config = _random_config(rng)
        errors, _ = validate(config)
        if errors:
            rejected += 1
        else:
            accepted += 1
            assert _invariants_hold(config), config
    assert accepted > 50 and rejected > 300  # the generator explores both sides


def _random_rules(rng: random.Random) -> list[dict]:
    targets = ["person", "cell phone", "dog", "bottle", "sleeping", ""]
    predicates = ["present", "absent", "enters", "near", "count", "still", "teleport", "in_zone"]
    return [
        {"target": rng.choice(targets), "predicate": rng.choice(predicates),
         "secondary": rng.choice(["", "cell phone", "dog"]), "zone": rng.choice(["", "", "pool"]),
         "severity": rng.choice(["info", "warning", "critical"])}
        for _ in range(rng.randint(0, 3))
    ]


def test_fuzz_no_command_sequence_leaves_a_protecting_driver_device_unalerted(
        tmp_path, clock, monkeypatch, isolated_state):
    """Acceptance 4 (command channels): random sequences of preset, rule, zone,
    dashboard, uplink and protection commands on a real driver module; after each
    sequence a scripted 4 s eye closure must alert whenever the device reports
    'protecting'. 40 sequences x 25 commands = 1,000 commands."""
    rng = random.Random(4)
    preset_names = [*PRESETS, "driver_safety:strict", "driver_safety:teen_driver",
                    "security_monitoring:away", "", "nonexistent"]
    commands = 0
    for sequence in range(40):
        clock["now"] = 0.0
        closed = {"value": False}
        runtime, module, frame, actuator = _driver_runtime(tmp_path / f"s{sequence}", clock, monkeypatch,
                                                           closed=closed)
        for _ in range(25):
            commands += 1
            channel = rng.choice(_CHANNELS[1:])
            if channel in ("preset", "uplink_preset", "dashboard_preset", "backend_preset"):
                runtime.uplink.preset = rng.choice(preset_names)
            elif channel == "rules":
                runtime.uplink.rules = _random_rules(rng)
                runtime.uplink.version = f"v{commands}"
            elif channel == "zones":
                runtime.uplink.zones = rng.choice([[], [{"name": "pool", "polygon": [[0, 0], [1, 0], [1, 1]]}]])
            elif channel == "protection_unconfirmed":
                runtime.uplink.protection = {"action": "stop", "confirm": rng.choice([False, None, "yes"])}
            runtime._apply_pending_commands()
        before = len([p for p in actuator.played if p[1] == "drowsiness"])
        state = runtime.protection_state()["state"]
        closed["value"] = True
        _drive(runtime, module, frame, clock, 4.0)
        closed["value"] = False
        after = len([p for p in actuator.played if p[1] == "drowsiness"])
        if state == "protecting":
            assert after > before, f"sequence {sequence}: protecting but a 4 s closure did not alert"
        assert runtime.module_name == "driver_attention"
        runtime.logger.close()
    assert commands == 1000


def test_validate_rule_set_is_all_or_nothing():
    ok = lambda label: label in {"person", "cell phone"}  # noqa: E731
    assert validate_rule_set([{"target": "person"}], label_ok=ok, zone_names=[]) == []
    errors = validate_rule_set([{"target": "person"}, {"target": "person", "predicate": "x"}],
                               label_ok=ok, zone_names=[])
    assert len(errors) == 1 and errors[0].startswith("rule 2")


def test_rule_event_keys_carry_the_rule(tmp_path, clock, monkeypatch, isolated_state):
    """Custom-rule episodes are keyed per rule (an info rule never merges with a critical one)."""
    closed = {"value": False}
    runtime, module, frame, _actuator = _driver_runtime(tmp_path, clock, monkeypatch, closed=closed)
    runtime.uplink.rules = [PHONE_RULE]
    runtime._apply_pending_commands()
    module._last_overlay_boxes = [((100.0, 100.0, 160.0, 220.0), "cell phone")]
    module._last_overlay_confidences = [0.9]
    _drive(runtime, module, frame, clock, 3.0)
    keys = {e["key"] for e in runtime.alert_manager.recent_episodes() + runtime.alert_manager.open_episodes()}
    assert any(k.startswith("rule0:present:cell phone") for k in keys)
    assert isinstance(Event, type)
    runtime.logger.close()
