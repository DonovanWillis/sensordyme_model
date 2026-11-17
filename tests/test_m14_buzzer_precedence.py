"""M14: on the one buzzer/speaker, a Jupiter (custom) rule's pattern never
displaces or plays ahead of a mandatory (driver-safety) pattern.

Found by the M14 Jupiter stream: a rule's "warning" had the same rank as a
driver "warning", and with two patterns waiting a new rule pattern dropped the
OLDER waiting one -- which could be the driver's.
"""

from __future__ import annotations

import types

from core.alerts import PATTERNS, AlertCategory, AlertLevel, AlertManager, _PatternPlayer
from core.events import Event


def _busy_player(rendered: list[str]) -> _PatternPlayer:
    player = _PatternPlayer(lambda on_s: None)
    player._thread = types.SimpleNamespace(is_alive=lambda: True)  # worker "busy": nothing renders
    return player


def _cat(name: str, custom: bool) -> AlertCategory:
    return AlertCategory(name=name, initial_pattern="chirp", escalated_pattern="chirp", custom_rule=custom)


def test_a_rule_pattern_never_displaces_a_waiting_driver_pattern_of_the_same_rank():
    outcome: list[str] = []
    player = _busy_player(outcome)
    player.play(PATTERNS["chirp"], _cat("warning", False), lambda r, e: outcome.append("driver1:" + r))
    player.play(PATTERNS["chirp"], _cat("warning", False), lambda r, e: outcome.append("driver2:" + r))
    player.play(PATTERNS["chirp"], _cat("warning", True), lambda r, e: outcome.append("rule:" + r))
    assert outcome == ["rule:skipped"]
    # Even a critical rule ranks below a mandatory pattern.
    player.play(PATTERNS["chirp"], _cat("critical", True), lambda r, e: outcome.append("rule-crit:" + r))
    assert outcome == ["rule:skipped", "rule-crit:skipped"]


def test_a_driver_pattern_displaces_a_waiting_rule_pattern_and_plays_first():
    outcome: list[str] = []
    player = _busy_player(outcome)
    player.play(PATTERNS["chirp"], _cat("warning", True), lambda r, e: outcome.append("rule:" + r))
    player.play(PATTERNS["chirp"], _cat("notice", False), lambda r, e: outcome.append("driver-notice:" + r))
    player.play(PATTERNS["chirp"], _cat("warning", False), lambda r, e: outcome.append("driver:" + r))
    assert outcome == ["rule:skipped"]
    best = max(player._pending, key=lambda p: p[0])
    assert best[0] == (1, 1), "the driver's warning plays first"


def test_string_categories_keep_their_old_ranking():
    outcome: list[str] = []
    player = _busy_player(outcome)
    player.play(PATTERNS["chirp"], "critical", lambda r, e: outcome.append("c1:" + r))
    player.play(PATTERNS["chirp"], "critical", lambda r, e: outcome.append("c2:" + r))
    player.play(PATTERNS["chirp"], "notice", lambda r, e: outcome.append("n:" + r))
    assert outcome == ["n:skipped"]


def test_a_category_object_without_the_flag_ranks_as_mandatory():
    # Found by the M14 final gate: tests/test_m10_observe.py hands the sink a
    # duck-typed category carrying only .name; the player must not crash and
    # must treat it as mandatory (never below a rule).
    outcome: list[str] = []
    player = _busy_player(outcome)
    player.play(PATTERNS["chirp"], _cat("warning", True), lambda r, e: outcome.append("rule:" + r))
    player.play(PATTERNS["chirp"], _cat("warning", True), lambda r, e: outcome.append("rule2:" + r))
    player.play(PATTERNS["chirp"], types.SimpleNamespace(name="warning"),  # type: ignore[arg-type]
                lambda r, e: outcome.append("bare:" + r))
    assert outcome == ["rule:skipped"]
    assert max(player._pending, key=lambda p: p[0])[0] == (1, 1)


class _RecordingSink:
    name = "recorder"

    def __init__(self) -> None:
        self.categories: list[AlertCategory] = []

    def emit(self, event, level) -> None:  # pragma: no cover - play() is used
        pass

    def play(self, pattern, event, category) -> None:
        self.categories.append(category)


def test_the_alert_manager_marks_rule_episodes_and_only_them():
    sink = _RecordingSink()
    manager = AlertManager([sink])
    rule_event = Event.now(type="cell phone", duration=10.0, confidence=1.0, module_name="driver_attention")
    driver_event = Event.now(type="distraction", duration=4.0, confidence=1.0, module_name="driver_attention")
    manager.notify(rule_event, key="rule-1", level=AlertLevel.SOFT, category="warning", source="rules", now=0.0)
    manager.notify(driver_event, now=0.0)
    assert [c.custom_rule for c in sink.categories] == [True, False]
    assert sink.categories[0].name == "warning", "the rule keeps its own category and pattern"
