"""Unit tests for mirror-check-tolerant distraction detection and repeated-glance detection."""

from modules.base import SustainedConditionTracker
from modules.driver_attention import GlanceCounter

YAW_THRESHOLD_DEG = 35.0
DISTRACTION_SECONDS = 3.0


def _fires(series: list[tuple[float, float]]) -> int:
    """series: list of (timestamp, yaw_deg). Returns how many distraction events fire."""
    tracker = SustainedConditionTracker(DISTRACTION_SECONDS)
    fired = 0
    for now, yaw in series:
        if tracker.update(abs(yaw) > YAW_THRESHOLD_DEG, now) is not None:
            fired += 1
    return fired


def test_mirror_check_one_second_produces_zero_events():
    series = [(0.0, 0.0), (0.2, 45.0), (1.2, 45.0), (1.3, 0.0), (2.0, 0.0)]
    assert _fires(series) == 0


def test_blindspot_check_1_5s_produces_zero_events():
    series = [(0.0, 0.0), (0.2, 60.0), (1.7, 60.0), (1.8, 0.0), (2.5, 0.0)]
    assert _fires(series) == 0


def test_sustained_phone_stare_4s_fires_one_event():
    series = [(0.0, 0.0), (0.2, 50.0), (4.2, 50.0), (4.3, 0.0)]
    assert _fires(series) == 1


def test_full_sequence_only_the_sustained_stare_fires():
    """Mirror check (1.0s), blindspot check (1.5s), then a sustained phone stare (4.0s).

    Glance forgiveness means the two brief glances reset the tracker cleanly with
    zero carried-over penalty -- only the third, sustained glance should fire.
    """
    series = [
        (0.0, 0.0),
        (0.2, 45.0), (1.2, 45.0), (1.3, 0.0),  # mirror check, 1.0s
        (3.0, 0.0),
        (3.2, 60.0), (4.7, 60.0), (4.8, 0.0),  # blindspot check, 1.5s
        (6.0, 0.0),
        (6.2, 50.0), (10.2, 50.0), (10.3, 0.0),  # sustained phone stare, 4.0s
    ]
    assert _fires(series) == 1


def test_glance_counter_fires_on_the_fourth_glance_within_window():
    counter = GlanceCounter(window_s=30.0, min_count=4)
    fired_at = []
    now = 0.0
    for i in range(4):
        fired = counter.update(True, now)  # glance starts
        if fired:
            fired_at.append(i)
        now += 0.5
        counter.update(False, now)  # glance ends
        now += 5.0
    assert fired_at == [3]


def test_glance_counter_does_not_fire_with_only_three_glances():
    counter = GlanceCounter(window_s=30.0, min_count=4)
    now = 0.0
    any_fired = False
    for _ in range(3):
        any_fired = counter.update(True, now) or any_fired
        now += 0.5
        counter.update(False, now)
        now += 5.0
    assert not any_fired


def test_glance_counter_forgets_glances_that_age_out_of_the_window():
    counter = GlanceCounter(window_s=30.0, min_count=4)
    now = 0.0
    for _ in range(3):
        counter.update(True, now)
        now += 0.5
        counter.update(False, now)
        now += 5.0
    now += 40.0  # big gap -- the first three glances are now outside the 30s window
    fired = counter.update(True, now)
    assert fired is False


def test_glance_counter_rearms_after_dropping_below_min_count():
    counter = GlanceCounter(window_s=5.0, min_count=2)
    now = 0.0
    assert counter.update(True, now) is False
    now += 0.2
    counter.update(False, now)
    now += 1.0
    assert counter.update(True, now) is True  # 2nd glance within the 5s window -> fires
    now += 0.2
    counter.update(False, now)
    now += 10.0  # both prior glances age out
    assert counter.update(True, now) is False
    now += 0.2
    counter.update(False, now)
    now += 1.0
    assert counter.update(True, now) is True  # fires again once count reaches 2 anew


def test_glance_counter_count_reflects_episodes_currently_in_window():
    counter = GlanceCounter(window_s=10.0, min_count=99)  # min_count unreachable -> just watch .count
    now = 0.0
    assert counter.count == 0
    counter.update(True, now)
    assert counter.count == 1
    now += 0.2
    counter.update(False, now)
    now += 1.0
    counter.update(True, now)
    assert counter.count == 2
    now += 0.2
    counter.update(False, now)
    now += 20.0  # both glances age out of the 10s window
    counter.update(False, now)
    assert counter.count == 0
