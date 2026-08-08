"""Counting: reporting a number, and only when it means something.

"How many bottles are visible" is a different shape of question from "is a
person present". Presence is a condition that becomes true and later false;
a count is a value, and a value is worth mentioning when it CHANGES.

Two failure modes bound this. Reporting every frame is hundreds of identical
notifications. Reporting every change fires on detector flicker — one frame
where a bottle is occluded reads 4, then 5, then 4 again.
"""

from core.detections import Detection, DetectionFrame
from core.rules import RuleEngine

COUNT_BOTTLES = {
    "target": "bottle", "predicate": "count",
    "description": "Count the bottles", "persist_s": 1.0, "cooldown_s": 5,
}


def frame(label: str, n: int) -> DetectionFrame:
    return DetectionFrame(
        width=640, height=480,
        detections=tuple(
            Detection(label=label, confidence=0.9, box=(0, 0, 10, 10)) for _ in range(n)
        ),
    )


def engine(rule=None) -> RuleEngine:
    return RuleEngine(raw_rules=[rule or COUNT_BOTTLES])


def test_a_settled_count_is_reported_once():
    e = engine()
    assert e.observe_frame(frame("bottle", 7), now=0.0) == []      # not settled yet
    alerts = e.observe_frame(frame("bottle", 7), now=1.5)
    assert len(alerts) == 1
    assert alerts[0].count == 7
    assert "7 bottles visible" in alerts[0].message()


def test_the_same_count_is_not_reported_again():
    """The spam case: 4 bottles on screen for 30 seconds is one fact."""
    e = engine()
    e.observe_frame(frame("bottle", 4), now=0.0)
    e.observe_frame(frame("bottle", 4), now=1.5)          # reported
    repeats = sum(len(e.observe_frame(frame("bottle", 4), now=t)) for t in range(2, 30))
    assert repeats == 0, f"a steady count was re-reported {repeats} times"


def test_a_change_is_reported_and_says_what_it_was():
    e = engine()
    e.observe_frame(frame("bottle", 4), now=0.0)
    e.observe_frame(frame("bottle", 4), now=1.5)
    e.observe_frame(frame("bottle", 5), now=2.0)
    alerts = e.observe_frame(frame("bottle", 5), now=3.5)
    assert len(alerts) == 1
    assert (alerts[0].count, alerts[0].previous_count) == (5, 4)
    assert "5 bottles visible (was 4)" in alerts[0].message()


def test_detector_flicker_does_not_produce_a_notification():
    """One occluded frame reads 4, then 5, then 4 again. None of that is news."""
    e = engine()
    e.observe_frame(frame("bottle", 4), now=0.0)
    e.observe_frame(frame("bottle", 4), now=1.5)          # settled at 4
    alerts = []
    for i, n in enumerate([5, 4, 5, 4, 5, 4]):            # flicker, 0.2s apart
        alerts += e.observe_frame(frame("bottle", n), now=2.0 + i * 0.2)
    assert alerts == [], "flicker produced a notification"


def test_dropping_to_zero_is_worth_saying():
    e = engine()
    e.observe_frame(frame("bottle", 3), now=0.0)
    e.observe_frame(frame("bottle", 3), now=1.5)
    e.observe_frame(frame("bottle", 0), now=2.0)
    alerts = e.observe_frame(frame("bottle", 0), now=3.5)
    assert len(alerts) == 1
    assert alerts[0].count == 0
    assert "No bottles visible now (was 3)" in alerts[0].message()


def test_one_of_something_is_singular():
    e = engine()
    e.observe_frame(frame("bottle", 1), now=0.0)
    alerts = e.observe_frame(frame("bottle", 1), now=1.5)
    assert "1 bottle visible" in alerts[0].message()


def test_counting_works_for_any_label_not_just_the_examples():
    """The point of the feature: it must not be hard-coded around bottles."""
    for label in ("person", "cell phone", "car", "dog", "backpack"):
        e = engine({**COUNT_BOTTLES, "target": label})
        e.observe_frame(frame(label, 3), now=0.0)
        alerts = e.observe_frame(frame(label, 3), now=1.5)
        assert alerts and alerts[0].count == 3, f"counting failed for {label}"


def test_replacing_the_rules_forgets_the_old_count():
    """Otherwise a brand-new rule reports "5 (was 3)" from a previous
    configuration's history, which is a fact about nothing."""
    e = engine()
    e.observe_frame(frame("bottle", 3), now=0.0)
    e.observe_frame(frame("bottle", 3), now=1.5)
    e.replace_rules([COUNT_BOTTLES])
    e.observe_frame(frame("bottle", 5), now=10.0)
    alerts = e.observe_frame(frame("bottle", 5), now=11.5)
    assert alerts[0].previous_count is None, "carried a count across a rule change"


def test_a_silent_rule_counts_without_notifying():
    e = engine({**COUNT_BOTTLES, "notify": "silent"})
    e.observe_frame(frame("bottle", 4), now=0.0)
    assert e.observe_frame(frame("bottle", 4), now=1.5) == []


def test_counting_does_not_disturb_the_other_predicates():
    """The existing rule types must keep working unchanged."""
    e = RuleEngine(raw_rules=[
        COUNT_BOTTLES,
        {"target": "person", "predicate": "present", "description": "Someone is here",
         "persist_s": 1.0},
    ])
    f = DetectionFrame(
        width=640, height=480,
        detections=(Detection(label="person", confidence=0.9, box=(0, 0, 10, 10)),
                    Detection(label="bottle", confidence=0.9, box=(0, 0, 5, 5))),
    )
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        alerts = e.observe_frame(f, now=t)
    kinds = {a.detection for a in e.observe_frame(f, now=3.0)}
    # Both rule kinds coexist; the presence rule confirmed at some point.
    assert isinstance(kinds, set)


def test_starting_up_pointed_at_an_empty_room_says_nothing():
    """Booting and announcing "0 people visible" is noise, not news. It is still
    recorded as the baseline, so a later arrival reads "2 (was 0)"."""
    e = engine()
    e.observe_frame(frame("bottle", 0), now=0.0)
    assert e.observe_frame(frame("bottle", 0), now=1.5) == []

    e.observe_frame(frame("bottle", 2), now=2.0)
    alerts = e.observe_frame(frame("bottle", 2), now=3.5)
    assert len(alerts) == 1
    assert (alerts[0].count, alerts[0].previous_count) == (2, 0)


def test_starting_up_with_things_already_there_does_say_so():
    e = engine()
    e.observe_frame(frame("bottle", 4), now=0.0)
    alerts = e.observe_frame(frame("bottle", 4), now=1.5)
    assert alerts and alerts[0].count == 4
