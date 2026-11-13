"""M8.4 owner decision A: experimental alert families are observations, never alerts.

high_risk, elevated_risk and erratic_hand_movement keep being computed and
logged, but by default: no in-cabin actuation (alert policy), severity "log"
on the device, the uplink and the backend, `experimental: true` on the wire and
in GET /v1/devices/{id}/events, and no notification. Only the explicit boolean
`alerts.experimental_alerts_audible: true` re-enables actuation; nothing else
-- config lists, presets, variants, migration, a reset, a reboot -- can.

Device, backend and presentation-contract invariants. Synthetic; no hardware.
"""

from __future__ import annotations

import copy
import types
from pathlib import Path

import pytest
import yaml

from core.alerts import AlertLevel, AlertManager
from core.behavior.experimental import (
    EXPERIMENTAL_ALERT_TYPES,
    OPT_IN_KEY,
    gate_alert_lists,
    opted_in,
)
from core.events import Event

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTAL = sorted(EXPERIMENTAL_ALERT_TYPES)
VALIDATED = ("drowsiness", "attention_lost", "distraction", "eyes_down", "eyes_up", "phone_use",
             "fatigue_warning", "frequent_glancing")


def _config() -> dict:
    return yaml.safe_load((REPO / "config.yaml").read_text())


class _Recorder:
    """An audible in-cabin output that records every command."""

    name, kind, audible = "gpio_buzzer", "buzzer", True

    def __init__(self) -> None:
        self.played: list[str] = []

    def play(self, pattern, event, category) -> None:
        self.played.append(event.type)


def _event(event_type: str) -> Event:
    return Event.now(event_type, 0.0, None, "driver_attention", "x")


#: The alerts block a unit configured before M8.4 still holds (migration never
#: removes list members): every experimental family listed audibly.
DEPLOYED_ALERTS = {
    "sinks": ["console", "sound", "gpio_buzzer"],
    "critical_types": ["drowsiness", "attention_lost"],
    "loud_types": ["drowsiness", "phone_use", "eyes_down", "eyes_up", "attention_lost", "high_risk", "distraction"],
    "soft_types": ["fatigue_warning", "frequent_glancing", "elevated_risk", "erratic_hand_movement"],
}


# ============================================================ alert policy (device actuators)
@pytest.mark.parametrize("event_type", EXPERIMENTAL)
def test_the_policy_never_actuates_an_experimental_family_by_default(event_type):
    sink = _Recorder()
    manager = AlertManager([sink], fixed_levels={event_type: AlertLevel.CONTINUOUS},
                           type_categories={event_type: "critical"})
    assert manager.notify(_event(event_type), now=0.0) is AlertLevel.SUPPRESSED
    manager.observe(10.0, active=[event_type], tracked=[event_type])
    assert sink.played == []
    assert manager.open_episodes() == []
    assert manager.permits(event_type) is False
    assert manager.delivery(_event(event_type)) is None


@pytest.mark.parametrize("event_type", VALIDATED)
def test_validated_alerts_still_actuate(event_type):
    sink = _Recorder()
    manager = AlertManager([sink], fixed_levels={event_type: AlertLevel.CONTINUOUS})
    assert manager.notify(_event(event_type), now=0.0) is AlertLevel.CONTINUOUS
    assert sink.played == [event_type]


def test_only_the_explicit_boolean_opt_in_makes_them_audible():
    for value in ("true", "yes", 1, "1", None, False):
        assert opted_in({OPT_IN_KEY: value}) is False
    assert opted_in({OPT_IN_KEY: True}) is True
    sink = _Recorder()
    manager = AlertManager([sink], experimental_audible=True)
    manager.notify(_event("high_risk"), now=0.0)
    assert sink.played == ["high_risk"]
    # A truthy non-bool passed by a caller is not an opt-in either.
    assert AlertManager([], experimental_audible=1).permits("high_risk") is False  # type: ignore[arg-type]


def test_a_deployed_config_still_listing_them_does_not_sound_them():
    from main import build_alert_manager

    sink = _Recorder()
    manager = build_alert_manager({**DEPLOYED_ALERTS, "sinks": []}, False)
    manager._sinks = [sink]
    for event_type in EXPERIMENTAL:
        manager.notify(_event(event_type), now=0.0)
    manager.notify(_event("drowsiness"), now=0.0)
    assert sink.played == ["drowsiness"]
    # ... and an explicit owner opt-in on the same config does.
    opted = build_alert_manager({**DEPLOYED_ALERTS, OPT_IN_KEY: True, "sinks": []}, False)
    opted._sinks = [sink]
    opted.notify(_event("high_risk"), now=0.0)
    assert sink.played[-1] == "high_risk"


def test_the_repo_config_and_a_reset_to_it_are_not_opted_in():
    alerts = _config()["alerts"]
    assert alerts[OPT_IN_KEY] is False
    for key in ("loud_types", "soft_types", "critical_types"):
        assert not set(alerts[key]) & EXPERIMENTAL_ALERT_TYPES
    assert gate_alert_lists(alerts)[1] == []


def test_gate_alert_lists_strips_them_without_mutating_the_config():
    original = copy.deepcopy(DEPLOYED_ALERTS)
    gated, removed = gate_alert_lists(DEPLOYED_ALERTS)
    assert original == DEPLOYED_ALERTS
    assert removed == EXPERIMENTAL
    assert "high_risk" not in gated["loud_types"] and "drowsiness" in gated["loud_types"]
    assert gate_alert_lists({**DEPLOYED_ALERTS, OPT_IN_KEY: True})[1] == []


# ============================================================ presets, variants, role
def test_no_preset_pattern_or_variant_makes_them_audible():
    from core.presets import load_presets, preset_overrides

    presets = load_presets()
    for name, preset in presets.items():
        for pattern in preset.get("alert_patterns") or []:
            assert not set(pattern.get("events") or []) & EXPERIMENTAL_ALERT_TYPES, (name, pattern)
        for variant in (preset.get("variants") or {}):
            _name, _preset, overrides = preset_overrides(f"{name}:{variant}", presets)
            assert OPT_IN_KEY not in overrides
            assert not {"loud_types", "soft_types", "critical_types"} & set(overrides)


def test_the_driver_role_no_longer_requires_high_risk_and_keeps_every_validated_floor():
    from core.roles import ROLES, config_violations

    driver = ROLES["driver_safety"]
    assert not driver.mandatory_alert_types & EXPERIMENTAL_ALERT_TYPES
    assert {"drowsiness", "attention_lost", "distraction", "eyes_down", "eyes_up",
            "phone_use"} <= driver.mandatory_alert_types
    assert driver.critical_types == frozenset({"drowsiness", "attention_lost"})
    config = _config()
    config["role"] = "driver_safety"
    assert config_violations(config) == []
    # A validated alert dropped from loud_types is still refused.
    config["alerts"]["loud_types"] = [t for t in config["alerts"]["loud_types"] if t != "phone_use"]
    assert any("phone_use" in e for e in config_violations(config))


# ============================================================ migration
def test_migration_writes_the_explicit_no_and_leaves_the_device_lists_alone(tmp_path):
    from scripts.migrate_config import migrate

    path = tmp_path / "config.yaml"
    path.write_text(
        "alerts:\n"
        "  sinks: [console, sound, gpio_buzzer]\n"
        "  critical_types: [drowsiness, attention_lost]\n"
        "  loud_types: [drowsiness, phone_use, eyes_down, eyes_up, attention_lost, high_risk, distraction]\n"
        "  soft_types: [fatigue_warning, frequent_glancing, elevated_risk, erratic_hand_movement]\n"
        "  gpio_buzzer_pin: 17\n"
        "cloud:\n"
        "  enabled: false\n"
    )
    migrate(path)
    alerts = yaml.safe_load(path.read_text())["alerts"]
    assert alerts[OPT_IN_KEY] is False
    assert "high_risk" in alerts["loud_types"]  # the device's own value, untouched...
    assert gate_alert_lists(alerts)[1] == EXPERIMENTAL  # ...and inert
    migrate(path)  # idempotent
    assert yaml.safe_load(path.read_text())["alerts"][OPT_IN_KEY] is False


# ============================================================ device runtime: severity + uplink
class _Uplink:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, str]] = []

    def push_event(self, event, severity, **kwargs) -> None:
        self.pushed.append((event.type, severity))


def _runtime(tmp_path, alert_cfg):
    from core.logger import EventLogger
    from main import ALERT_EVENT_TYPES_PI, build_alert_manager
    from services.runtime import DeviceRuntime

    sink = _Recorder()
    manager = build_alert_manager({**alert_cfg, "sinks": []}, False)
    manager._sinks = [sink]
    runtime = DeviceRuntime(
        config={"logging": {"csv_path": str(tmp_path / "e.csv"), "telemetry_csv_path": str(tmp_path / "t.csv")},
                "modules": {}},
        profile="pi", module=None, module_name="driver_attention", alert_manager=manager, alert_cfg=alert_cfg,
        alert_event_types=set(ALERT_EVENT_TYPES_PI), alert_types_for=lambda n, p=None: set(ALERT_EVENT_TYPES_PI),
        presets={}, active_preset=None, display=False, debug=False,
    )
    runtime.logger = EventLogger(str(tmp_path / "events.csv"))
    runtime.dashboard = None
    runtime.uplink = _Uplink()
    return runtime, sink


class _Module:
    name = "driver_attention"

    def __init__(self, active) -> None:
        self.active = set(active)

    def alert_episode_types(self):
        return frozenset({"high_risk", "erratic_hand_movement", "drowsiness"})

    def active_alert_conditions(self):
        return set(self.active)


def test_runtime_tags_them_log_uploads_them_as_observations_and_sounds_nothing(tmp_path):
    runtime, sink = _runtime(tmp_path, DEPLOYED_ALERTS)
    module = _Module({"high_risk", "erratic_hand_movement"})
    events = [_event(t) for t in EXPERIMENTAL] + [_event("drowsiness")]
    evidence_due = runtime._dispatch_module_events(module, events, 0.0)
    for t in (1.0, 5.0, 30.0):  # live conditions are never re-opened as episodes
        runtime._dispatch_module_events(module, [], t)
    assert sink.played == ["drowsiness"]
    pushed = dict(runtime.uplink.pushed)
    assert {pushed[t] for t in EXPERIMENTAL} == {"log"}
    assert pushed["drowsiness"] == "loud"
    assert evidence_due is True  # from the drowsiness, not the observations
    opened = runtime.alert_manager.open_episodes() + runtime.alert_manager.recent_episodes()
    assert {e["key"] for e in opened} == {"drowsiness"}


def test_the_uplink_payload_carries_experimental_true_and_severity_log(tmp_path):
    from core.uplink import CloudUplink

    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_x", "poll_interval_s": 0,
                          "spool_path": str(tmp_path / "spool.jsonl")})
    for event_type, severity in (("high_risk", "loud"), ("drowsiness", "loud")):
        event = types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "t"), type=event_type,
                                      confidence=None, contributing_signals="x", duration=0.0, event_id=None)
        uplink.push_event(event, severity)
    items = list(uplink._queue)
    assert items[0]["experimental"] is True and items[0]["severity"] == "log"
    assert "experimental" not in items[1] and items[1]["severity"] == "loud"


# ============================================================ backend: storage, row, notifications
def test_backend_stores_them_as_log_with_experimental_true_and_never_notifies(tmp_path, monkeypatch):
    from test_m83_events_backend import _Stack

    stack = _Stack(tmp_path, monkeypatch)
    base = {"timestamp": "2026-11-13T10:00:00+00:00", "confidence": None, "signals": "x", "duration": 0.0}
    stack.ingest(
        {**base, "event_key": "k-high", "type": "high_risk", "severity": "loud"},  # older firmware
        {**base, "event_key": "k-erratic", "type": "erratic_hand_movement", "severity": "log",
         "experimental": True},
        {**base, "event_key": "k-drowsy", "type": "drowsiness", "severity": "loud"},
    )
    rows = {r["type"]: r for r in stack.events()}
    for event_type in ("high_risk", "erratic_hand_movement"):
        assert rows[event_type]["experimental"] is True
        assert rows[event_type]["severity"] == "log"
    assert rows["drowsiness"]["experimental"] is False
    assert rows["drowsiness"]["severity"] == "loud"
    assert stack.considered == ["loud"]  # only the drowsiness reached the notification gate
    assert len(stack.notified) == 1
