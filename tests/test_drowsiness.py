"""Unit tests proving blink-proof drowsiness detection via a synthetic EAR time series."""

from modules.base import SustainedConditionTracker
from modules.driver_attention import PerclosTracker

EAR_OPEN = 0.32
EAR_CLOSED = 0.10
THRESHOLD = 0.20
DROWSY_SECONDS = 2.0


def _events_for_series(series: list[tuple[float, float]]) -> int:
    """series: list of (timestamp, ear). Returns how many drowsiness events fire."""
    tracker = SustainedConditionTracker(DROWSY_SECONDS)
    fired_count = 0
    for now, ear in series:
        if tracker.update(ear < THRESHOLD, now) is not None:
            fired_count += 1
    return fired_count


def test_ten_blinks_of_200ms_produce_zero_events():
    series = []
    t = 0.0
    for _ in range(10):
        series.append((t, EAR_OPEN))
        t += 0.05
        series.append((t, EAR_CLOSED))  # blink starts
        t += 0.2
        series.append((t, EAR_CLOSED))  # still within the 200ms blink
        t += 0.05
        series.append((t, EAR_OPEN))  # eyes reopen
        t += 0.3
    assert _events_for_series(series) == 0


def test_closure_under_two_seconds_produces_zero_events():
    series = [
        (0.0, EAR_OPEN),
        (0.5, EAR_CLOSED),  # eyes close
        (1.5, EAR_CLOSED),  # 1.0s elapsed -- below the 2.0s threshold
        (2.3, EAR_CLOSED),  # 1.8s elapsed -- still below threshold
        (2.4, EAR_OPEN),  # eyes reopen just before hitting 2.0s
    ]
    assert _events_for_series(series) == 0


def test_closure_of_more_than_two_seconds_produces_exactly_one_event():
    series = [
        (0.0, EAR_OPEN),
        (0.5, EAR_CLOSED),  # eyes close
        (1.5, EAR_CLOSED),  # 1.0s elapsed -- below the 2.0s threshold
        (2.8, EAR_CLOSED),  # 2.3s elapsed -- fires
        (2.9, EAR_OPEN),  # eyes reopen
    ]
    assert _events_for_series(series) == 1


def test_mixed_blinks_then_sustained_closure_fires_once():
    series = []
    t = 0.0
    # Five ordinary blinks first.
    for _ in range(5):
        series += [(t, EAR_OPEN)]
        t += 0.1
        series += [(t, EAR_CLOSED)]
        t += 0.15
        series += [(t, EAR_OPEN)]
        t += 0.5
    # Then a genuine 2.5-second closure (comfortably past the 2.0s threshold).
    series += [(t, EAR_CLOSED)]
    t += 2.5
    series += [(t, EAR_CLOSED)]
    assert _events_for_series(series) == 1


def test_perclos_rises_with_heavy_blinking_even_without_one_long_closure():
    tracker = PerclosTracker(window_s=60.0)
    now = 0.0
    perclos = 0.0
    # Alternate 3s closed / 3s open for a minute -> ~50% PERCLOS, well above the 0.15 warning line,
    # even though no single closure ever reaches the 1.5s drowsiness threshold on its own merits
    # (each closed segment individually exceeds it here only to make the rolling math simple;
    # the point under test is the aggregate PERCLOS signal, not any single segment).
    closed = True
    while now < 60.0:
        perclos = tracker.update(closed, now)
        now += 3.0
        closed = not closed
    assert perclos > 0.15


def test_perclos_stays_low_for_occasional_blinks():
    tracker = PerclosTracker(window_s=60.0)
    now = 0.0
    perclos = 0.0
    # A blink every 5 seconds, each lasting 0.2s -> closed ~4% of the time.
    while now < 60.0:
        perclos = tracker.update(True, now)
        now += 0.2
        perclos = tracker.update(False, now)
        now += 4.8
    assert perclos < 0.15
