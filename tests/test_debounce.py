"""Unit tests for sustained-condition debounce and alert escalation/cooldown logic."""

from core.alerts import AlertLevel, AlertManager
from core.events import Event
from modules.base import SustainedConditionTracker


def test_tracker_does_not_fire_before_sustain_duration():
    tracker = SustainedConditionTracker(sustain_s=1.5)
    assert tracker.update(True, now=0.0) is None
    assert tracker.update(True, now=1.0) is None


def test_tracker_fires_once_sustain_duration_elapses():
    tracker = SustainedConditionTracker(sustain_s=1.5)
    tracker.update(True, now=0.0)
    tracker.update(True, now=1.0)
    fired = tracker.update(True, now=1.6)
    assert fired == 1.6
    # Does not fire again while the condition remains continuously true.
    assert tracker.update(True, now=2.0) is None


def test_is_active_is_false_on_a_brief_blink_style_condition():
    """A live 'alert active' indicator driven by is_active must NOT fire on a blink --
    only the debounced state (not the raw per-frame condition) should light it up. This
    guards against the bug where the visual alert flashed on every blink/glance because
    it read the instantaneous condition instead of the tracker's debounced state."""
    tracker = SustainedConditionTracker(sustain_s=2.0)
    tracker.update(True, now=0.0)  # condition goes true (e.g. eyes close)
    assert tracker.is_active is False
    tracker.update(True, now=0.2)  # still within a 200ms blink
    assert tracker.is_active is False
    tracker.update(False, now=0.25)  # eyes reopen -- never sustained long enough
    assert tracker.is_active is False


def test_is_active_turns_on_only_once_sustained_and_off_once_condition_clears():
    tracker = SustainedConditionTracker(sustain_s=2.0)
    tracker.update(True, now=0.0)
    assert tracker.is_active is False
    tracker.update(True, now=1.0)
    assert tracker.is_active is False  # only 1.0s elapsed, below the 2.0s threshold
    tracker.update(True, now=2.1)
    assert tracker.is_active is True  # sustained threshold just crossed
    tracker.update(True, now=5.0)
    assert tracker.is_active is True  # stays active while the condition keeps holding
    tracker.update(False, now=5.1)
    assert tracker.is_active is False  # turns off the instant the condition clears


def test_tracker_rearms_after_condition_clears():
    tracker = SustainedConditionTracker(sustain_s=1.0)
    tracker.update(True, now=0.0)
    assert tracker.update(True, now=1.1) is not None  # fires
    tracker.update(False, now=1.2)  # condition clears -> rearm
    tracker.update(True, now=1.3)
    fired = tracker.update(True, now=2.4)
    assert fired is not None


class _RecordingSink:
    def __init__(self):
        self.calls = []

    def emit(self, event, level):
        self.calls.append((event.type, level))


def _make_event(event_type="drowsiness"):
    return Event.now(event_type, duration=2.0, confidence=0.9, module_name="driver_attention")


def test_alert_manager_first_alert_is_soft():
    sink = _RecordingSink()
    manager = AlertManager([sink], cooldown_s=10.0, escalation_window_s=60.0)
    level = manager.notify(_make_event(), now=0.0)
    assert level == AlertLevel.SOFT
    assert sink.calls == [("drowsiness", AlertLevel.SOFT)]


def test_alert_manager_suppresses_within_cooldown():
    sink = _RecordingSink()
    manager = AlertManager([sink], cooldown_s=10.0, escalation_window_s=60.0)
    manager.notify(_make_event(), now=0.0)
    level = manager.notify(_make_event(), now=5.0)
    assert level == AlertLevel.SUPPRESSED
    assert len(sink.calls) == 1


def test_alert_manager_escalates_to_continuous_within_window():
    sink = _RecordingSink()
    manager = AlertManager([sink], cooldown_s=10.0, escalation_window_s=60.0)
    manager.notify(_make_event(), now=0.0)
    level = manager.notify(_make_event(), now=20.0)  # past cooldown, inside escalation window
    assert level == AlertLevel.CONTINUOUS


def test_alert_manager_resets_to_soft_after_escalation_window():
    sink = _RecordingSink()
    manager = AlertManager([sink], cooldown_s=10.0, escalation_window_s=60.0)
    manager.notify(_make_event(), now=0.0)
    level = manager.notify(_make_event(), now=90.0)
    assert level == AlertLevel.SOFT


def test_alert_manager_tracks_event_types_independently():
    sink = _RecordingSink()
    manager = AlertManager([sink], cooldown_s=10.0, escalation_window_s=60.0)
    manager.notify(_make_event("drowsiness"), now=0.0)
    level = manager.notify(_make_event("distraction"), now=1.0)
    assert level == AlertLevel.SOFT


def test_fixed_loud_level_ignores_time_based_escalation():
    sink = _RecordingSink()
    manager = AlertManager(
        [sink], cooldown_s=10.0, escalation_window_s=60.0, fixed_levels={"phone_use": AlertLevel.CONTINUOUS}
    )
    # Even the very first phone_use alert is loud -- it doesn't need a repeat to escalate.
    level = manager.notify(_make_event("phone_use"), now=0.0)
    assert level == AlertLevel.CONTINUOUS


def test_fixed_soft_level_never_escalates_on_repeat():
    sink = _RecordingSink()
    manager = AlertManager(
        [sink], cooldown_s=10.0, escalation_window_s=60.0, fixed_levels={"fatigue_warning": AlertLevel.SOFT}
    )
    manager.notify(_make_event("fatigue_warning"), now=0.0)
    # Past the cooldown and within the window that would normally escalate a
    # non-fixed type to CONTINUOUS -- fatigue_warning stays SOFT regardless.
    level = manager.notify(_make_event("fatigue_warning"), now=20.0)
    assert level == AlertLevel.SOFT


def test_fixed_levels_still_respect_cooldown():
    sink = _RecordingSink()
    manager = AlertManager(
        [sink], cooldown_s=10.0, escalation_window_s=60.0, fixed_levels={"eyes_down": AlertLevel.CONTINUOUS}
    )
    manager.notify(_make_event("eyes_down"), now=0.0)
    level = manager.notify(_make_event("eyes_down"), now=5.0)
    assert level == AlertLevel.SUPPRESSED
