"""Tests for the event lifecycle engine (observed → developing → confirmed → resolved)."""

from core.event_engine import CONFIRMED, DEVELOPING, OBSERVED, RESOLVED, EventEngine, RuleSpec


def _engine(**overrides):
    spec = dict(detection="zone_violation:pool", severity="critical",
                min_confidence=0.5, persist_s=2.0, min_frames=3,
                occlusion_grace_s=1.0, resolve_after_s=5.0, cooldown_s=30.0)
    spec.update(overrides)
    return EventEngine({spec["detection"]: RuleSpec(**spec)}), spec["detection"]


def _states(transitions):
    return [t.state for t in transitions]


def test_single_frame_never_confirms():
    engine, det = _engine()
    transitions = engine.update(det, True, 0.99, now=0.0)
    assert _states(transitions) == [OBSERVED]  # first sighting is an observation, not an alert
    assert engine.state_of(det) == OBSERVED
    assert not engine.is_active(det)


def test_confirmation_requires_persistence_frames_and_confidence():
    engine, det = _engine()
    all_states = []
    for i in range(30):
        all_states += _states(engine.update(det, True, 0.9, now=i * 0.1))
    assert OBSERVED in all_states
    assert DEVELOPING in all_states     # half-way signal fired
    assert CONFIRMED in all_states      # 3s of presence > 2s persist_s
    assert engine.is_active(det)


def test_low_confidence_blocks_confirmation():
    engine, det = _engine(min_confidence=0.8)
    for i in range(40):
        transitions = engine.update(det, True, 0.4, now=i * 0.1)
        assert CONFIRMED not in _states(transitions)  # EMA of 0.4 never reaches 0.8
    assert engine.state_of(det) != CONFIRMED


def test_brief_occlusion_does_not_reset_progress():
    engine, det = _engine()
    for i in range(15):  # 1.5s of presence
        engine.update(det, True, 0.9, now=i * 0.1)
    engine.update(det, False, 0.0, now=1.6)  # 0.5s occlusion < 1s grace
    transitions = []
    for i in range(10):
        transitions += engine.update(det, True, 0.9, now=2.1 + i * 0.1)
    assert CONFIRMED in _states(transitions)  # progress survived the blink


def test_long_absence_before_confirmation_resets():
    engine, det = _engine()
    for i in range(15):
        engine.update(det, True, 0.9, now=i * 0.1)
    engine.update(det, False, 0.0, now=4.0)  # gone well past occlusion grace
    transitions = engine.update(det, True, 0.9, now=10.0)
    # A fresh appearance after a real absence starts the cycle over.
    assert engine.state_of(det) in (OBSERVED, DEVELOPING)
    assert CONFIRMED not in _states(transitions)


def test_confirmed_resolves_after_absence_and_cooldown_blocks_restart():
    engine, det = _engine()
    for i in range(40):
        engine.update(det, True, 0.9, now=i * 0.1)
    assert engine.is_active(det)

    transitions = []
    for i in range(3):
        transitions += engine.update(det, False, 0.0, now=10.0 + i * 3.0)
    assert RESOLVED in _states(transitions)
    assert not engine.is_active(det)

    # Immediately re-present: inside cooldown_s -> not even a new observation.
    assert engine.update(det, True, 0.9, now=17.0) == []
    # After the cooldown, the cycle can start again.
    assert _states(engine.update(det, True, 0.9, now=60.0)) == [OBSERVED]


def test_confirmed_event_survives_brief_occlusion():
    engine, det = _engine()
    for i in range(40):
        engine.update(det, True, 0.9, now=i * 0.1)
    engine.update(det, False, 0.0, now=5.0)  # brief absence < resolve_after_s
    assert engine.is_active(det)  # still an active event, not resolved


def test_unknown_detection_is_ignored():
    engine, _ = _engine()
    assert engine.update("nope", True, 1.0, now=0.0) == []
