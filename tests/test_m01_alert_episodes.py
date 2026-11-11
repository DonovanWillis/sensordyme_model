"""M0.1 — episode-aware alerting.

Reproduces and pins the fixes for two confirmed audit defects:

  D1  A new episode that started within `cooldown_s` of the previous alert of the
      same type produced NO alert for the whole episode: AlertManager dropped
      (rather than deferred) the alert, and every module tracker fires exactly
      once per episode, so nothing ever retried it.
  D2  The "loud"/"continuous" alert was a single 0.6 s pulse that never
      repeated or escalated while the driver stayed asleep or distracted.

Every test runs on an injected clock. Physical audibility is NOT verified here:
these tests assert what the alert policy asks the actuators to do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_driver_attention_scenarios import (
    _FakeFaceMeshResult,
    _make_module,
    make_fake_landmarks,
)

import modules.driver_attention as driver_attention_module
from core.alerts import (
    DEFAULT_CATEGORIES,
    MAX_PATTERN_S,
    PATTERNS,
    AlertLevel,
    AlertManager,
)
from core.events import Event
from main import ALERT_EVENT_TYPES_PI


class RecordingActuator:
    """An actuator that records what it was asked to play (never makes a sound)."""

    kind = "buzzer"
    audible = True

    def __init__(self) -> None:
        self.played: list[tuple[float, str, str, str]] = []  # (now, type, category, pattern)
        self.emitted: list[tuple[str, AlertLevel]] = []
        self._now = 0.0

    def play(self, pattern, event: Event, category) -> None:
        self.played.append((self._now, event.type, category.name, pattern.name))

    def emit(self, event: Event, level: AlertLevel) -> None:  # legacy interface
        self.emitted.append((event.type, level))


def _event(kind: str = "drowsiness") -> Event:
    return Event.now(kind, 2.0, 0.9, "driver_attention")


def _manager(actuator: RecordingActuator, **kwargs) -> AlertManager:
    return AlertManager(
        [actuator], cooldown_s=15.0, escalation_window_s=60.0,
        fixed_levels={"drowsiness": AlertLevel.CONTINUOUS, "eyes_down": AlertLevel.CONTINUOUS,
                      "fatigue_warning": AlertLevel.SOFT},
        type_categories={"drowsiness": "critical", "eyes_down": "warning", "fatigue_warning": "notice"},
        **kwargs,
    )


# ------------------------------------------------------------------ D1: no dropped episodes
def test_d1_a_new_episode_inside_the_old_cooldown_still_alerts():
    """Before M0.1: the second episode, 5 s after the first alert, returned
    SUPPRESSED and nothing ever sounded for it."""
    actuator = RecordingActuator()
    manager = _manager(actuator)
    assert manager.notify(_event(), now=100.0) is not AlertLevel.SUPPRESSED
    manager.observe(101.0, source="module", active=set(), tracked={"drowsiness"})  # episode 1 ends
    assert manager.notify(_event(), now=105.0) is not AlertLevel.SUPPRESSED
    assert len(actuator.played) == 2


@pytest.mark.parametrize("gap_s", [5.0, 10.0, 14.0])
def test_d1_reclosures_5_10_14_s_after_an_alert_each_alert(gap_s):
    actuator = RecordingActuator()
    manager = _manager(actuator)
    manager.notify(_event(), now=0.0)
    manager.observe(1.0, source="module", active=set(), tracked={"drowsiness"})
    manager.notify(_event(), now=gap_s)
    assert [p[1] for p in actuator.played] == ["drowsiness", "drowsiness"]


def test_a_duplicate_event_inside_an_open_episode_is_merged_not_replayed():
    actuator = RecordingActuator()
    manager = _manager(actuator)
    manager.notify(_event(), now=0.0)
    manager.observe(0.5, source="module", active={"drowsiness"}, tracked={"drowsiness"})
    assert manager.notify(_event(), now=0.6) is AlertLevel.SUPPRESSED
    assert len(actuator.played) == 1


def test_rolling_window_notices_keep_their_cooldown():
    """fatigue_warning / frequent_glancing are notices about a rolling window, not
    episodes of danger: they keep cooldown de-duplication (documented)."""
    actuator = RecordingActuator()
    manager = _manager(actuator)
    manager.notify(_event("fatigue_warning"), now=0.0)
    manager.observe(3.0, source="module", active=set(), tracked=set())
    assert manager.notify(_event("fatigue_warning"), now=5.0) is AlertLevel.SUPPRESSED


# ------------------------------------------------------------------ D2: repeat and escalate
def test_d2_a_sustained_episode_repeats_and_is_bounded():
    actuator = RecordingActuator()
    manager = _manager(actuator)
    critical = DEFAULT_CATEGORIES["critical"]
    manager.notify(_event(), now=0.0)
    t = 0.0
    while t < 60.0:
        t = round(t + 0.1, 6)
        manager.observe(t, source="module", active={"drowsiness"}, tracked={"drowsiness"})
    actuations = len(actuator.played)
    expected_min = math.floor((60.0 - 0.0) / critical.repeat_s) + 1
    assert actuations >= min(expected_min, critical.max_repeats + 1)
    assert actuations <= critical.max_repeats + 1  # bounded


def test_d2_warning_escalates_after_its_threshold():
    actuator = RecordingActuator()
    manager = _manager(actuator)
    warning = DEFAULT_CATEGORIES["warning"]
    manager.notify(_event("eyes_down"), now=0.0)
    t = 0.0
    while t < warning.escalate_after_s + 2 * warning.repeat_s:
        t = round(t + 0.1, 6)
        manager.observe(t, source="module", active={"eyes_down"}, tracked={"eyes_down"})
    patterns = [p[3] for p in actuator.played]
    assert patterns[0] == warning.initial_pattern
    assert warning.escalated_pattern in patterns[1:]
    assert patterns.index(warning.escalated_pattern) > 0


def test_episode_ends_when_the_condition_clears_and_stops_repeating():
    actuator = RecordingActuator()
    manager = _manager(actuator)
    manager.notify(_event(), now=0.0)
    manager.observe(1.0, source="module", active=set(), tracked={"drowsiness"})
    for t in range(2, 30):
        manager.observe(float(t), source="module", active=set(), tracked={"drowsiness"})
    assert len(actuator.played) == 1
    recent = manager.recent_episodes()
    assert recent[-1]["type"] == "drowsiness"
    assert recent[-1]["ended_at"] == 1.0
    assert recent[-1]["actuations"] == 1


def test_episode_lifecycle_is_explicit_start_active_end():
    actuator = RecordingActuator()
    manager = _manager(actuator)
    manager.notify(_event(), now=0.0)
    [episode] = manager.open_episodes()
    assert episode["state"] == "active" and episode["started_at"] == 0.0 and episode["episode_id"] >= 1
    manager.observe(0.5, source="module", active=set(), tracked={"drowsiness"})
    assert manager.open_episodes() == []
    assert manager.recent_episodes()[-1]["state"] == "ended"


# ------------------------------------------------------------------ patterns
def test_patterns_are_bounded():
    for pattern in PATTERNS.values():
        assert 0 < pattern.duration_s <= MAX_PATTERN_S
        assert 1 <= len(pattern.steps) <= 8
        assert all(on > 0 and off >= 0 for on, off in pattern.steps)


def test_categories_have_distinguishable_patterns():
    critical, warning, notice = (DEFAULT_CATEGORIES[n] for n in ("critical", "warning", "notice"))
    impaired = DEFAULT_CATEGORIES["impaired"]
    signatures = {PATTERNS[c.initial_pattern].steps for c in (critical, warning, notice, impaired)}
    assert len(signatures) == 4
    # distraction-class warnings never borrow the critical (sleep) pattern at onset
    assert warning.initial_pattern != critical.initial_pattern


def test_repeat_interval_always_leaves_room_for_the_pattern_to_finish():
    for category in DEFAULT_CATEGORIES.values():
        if category.repeat_s > 0:
            longest = max(PATTERNS[category.initial_pattern].duration_s,
                          PATTERNS[category.escalated_pattern].duration_s)
            assert category.repeat_s >= longest + 0.5


def test_a_sink_without_play_still_receives_the_legacy_level():
    class LegacySink:
        def __init__(self):
            self.levels = []

        def emit(self, event, level):
            self.levels.append(level)

    sink = LegacySink()
    manager = AlertManager([sink], type_categories={"drowsiness": "critical"},
                           fixed_levels={"drowsiness": AlertLevel.CONTINUOUS})
    manager.notify(_event(), now=0.0)
    assert sink.levels == [AlertLevel.CONTINUOUS]


# ------------------------------------------------------------------ end to end: real module
@pytest.fixture
def clock(monkeypatch):
    t = {"now": 0.0}
    monkeypatch.setattr(driver_attention_module.time, "monotonic", lambda: t["now"])
    return t


def _drive(module, frame, clock, manager, *, closed: dict, seconds: float, step: float = 0.05):
    """The runtime's per-frame alert path, on the real module, for `seconds`."""
    for _ in range(round(seconds / step)):
        events = module.process(frame)
        manager.process_frame(
            clock["now"], events, alertable=ALERT_EVENT_TYPES_PI,
            active=module.active_alert_conditions(), tracked=module.alert_episode_types(),
        )
        clock["now"] = round(clock["now"] + step, 6)


@pytest.mark.parametrize("gap_s", [5.0, 10.0, 14.0])
def test_real_module_reclosure_within_15_s_of_an_alert_alerts_again(tmp_path, clock, monkeypatch, gap_s):
    """Acceptance 1: eye closures starting 5/10/14 s after a drowsiness alert each alert.

    Before M0.1 the second closure produced 0 alerts (cooldown_s = 15 dropped it)."""
    closed = {"value": False}
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio",
                        lambda *pts: 0.05 if closed["value"] else 0.40)
    # Synthetic landmarks keep the iris open, so the iris veto (a real-camera
    # guard) is off here, exactly as tests/test_phase1_driver_fixes.py does.
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    open_face = make_fake_landmarks()
    module._face_mesh.process = lambda rgb, lm=open_face: _FakeFaceMeshResult(lm)
    actuator = RecordingActuator()
    manager = _manager(actuator)

    _drive(module, frame, clock, manager, closed=closed, seconds=2.0)          # settle, eyes open
    closed["value"] = True
    _drive(module, frame, clock, manager, closed=closed, seconds=2.5)          # first alert ~2.0 s in
    first = [p for p in actuator.played if p[1] == "drowsiness"]
    assert len(first) == 1
    alert_time = clock["now"]
    closed["value"] = False
    _drive(module, frame, clock, manager, closed=closed, seconds=gap_s - 2.5 + 0.0)  # eyes open
    closed["value"] = True
    _drive(module, frame, clock, manager, closed=closed, seconds=2.5)          # second closure
    second = [p for p in actuator.played if p[1] == "drowsiness"]
    assert len(second) >= 2, f"re-closure {gap_s}s after alert at {alert_time} did not alert"


def test_real_module_a_60_s_closure_repeats_and_escalates(tmp_path, clock, monkeypatch):
    """Acceptance 1b: a 60 s closure produces repeated actuations, bounded."""
    closed = {"value": False}
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio",
                        lambda *pts: 0.05 if closed["value"] else 0.40)
    # Synthetic landmarks keep the iris open, so the iris veto (a real-camera
    # guard) is off here, exactly as tests/test_phase1_driver_fixes.py does.
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    open_face = make_fake_landmarks()
    module._face_mesh.process = lambda rgb, lm=open_face: _FakeFaceMeshResult(lm)
    actuator = RecordingActuator()
    manager = _manager(actuator)
    closed["value"] = True
    _drive(module, frame, clock, manager, closed=closed, seconds=60.0)
    drowsy = [p for p in actuator.played if p[1] == "drowsiness"]
    critical = DEFAULT_CATEGORIES["critical"]
    expected = min(math.floor((60.0 - 2.0) / critical.repeat_s) + 1, critical.max_repeats + 1)
    assert len(drowsy) >= expected
    assert {p[2] for p in drowsy} == {"critical"}


def test_real_module_a_blink_shorter_than_the_threshold_never_alerts(tmp_path, clock, monkeypatch):
    closed = {"value": False}
    monkeypatch.setattr(driver_attention_module, "eye_aspect_ratio",
                        lambda *pts: 0.05 if closed["value"] else 0.40)
    # Synthetic landmarks keep the iris open, so the iris veto (a real-camera
    # guard) is off here, exactly as tests/test_phase1_driver_fixes.py does.
    module, frame = _make_module(tmp_path, clock, iris_veto=False)
    open_face = make_fake_landmarks()
    module._face_mesh.process = lambda rgb, lm=open_face: _FakeFaceMeshResult(lm)
    actuator = RecordingActuator()
    manager = _manager(actuator)
    for _ in range(10):
        closed["value"] = True
        _drive(module, frame, clock, manager, closed=closed, seconds=0.3)
        closed["value"] = False
        _drive(module, frame, clock, manager, closed=closed, seconds=1.5)
    # Blinks never raise a drowsiness episode. (Ten closures in 18 s do push
    # PERCLOS over its 0.15 bar, so the separate fatigue_warning NOTICE may fire:
    # that is a different, rolling-window signal and not under test here.)
    assert [p for p in actuator.played if p[1] == "drowsiness"] == []


def test_real_module_reports_its_episode_types_and_live_conditions(tmp_path, clock):
    module, _frame = _make_module(tmp_path, clock)
    tracked = module.alert_episode_types()
    assert {"drowsiness", "attention_lost", "distraction", "eyes_down", "phone_use"} <= tracked
    assert module.active_alert_conditions() == set()
    landmarks = make_fake_landmarks(pitch_deg=30.0)
    module._face_mesh.process = lambda rgb, lm=landmarks: _FakeFaceMeshResult(lm)
    for _ in range(80):
        module.process(np.zeros((480, 640, 3), dtype=np.uint8))
        clock["now"] += 0.05
    assert "eyes_down" in module.active_alert_conditions()
