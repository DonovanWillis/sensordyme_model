"""Unit tests for core/behavior/ components (synthetic scenarios, deterministic time).

Every scenario here is SYNTHETIC: hand-built boxes, angles and timestamps.
Passing them shows the logic does what it says; it is not a measured accuracy.
"""

from __future__ import annotations

import pytest

from core.behavior.evidence import (
    UNKNOWN,
    Assessment,
    FrameGapGuard,
    KOfN,
    intersection_over_area,
)

STEP = 0.05



def test_unknown_must_carry_a_reason():
    with pytest.raises(ValueError, match="must say why"):
        Assessment(UNKNOWN)
    assert not Assessment(UNKNOWN, "why").known


def test_frame_gap_guard_credits_ordinary_gaps_and_nothing_across_a_stall():
    guard = FrameGapGuard(0.5)
    assert guard.step(0.0) == (0.0, False)
    assert guard.step(0.1) == pytest.approx((0.1, False))
    assert guard.step(2.1) == (0.0, True)  # a 2 s stall credits NOTHING (EvidenceClock would credit 0.5)
    assert guard.step(1.0) == (0.0, True)  # time going backwards is stale too
    assert guard.stale_gaps == 2


def test_k_of_n_needs_k_hits_in_the_window():
    f = KOfN(3, 5)
    assert [f.update(x) for x in (True, True, False, True, False, False, False)] == [
        False, False, False, True, True, False, False]


def test_intersection_over_area_is_the_phones_coverage_not_iou():
    phone = (100, 100, 120, 140)  # 800 px^2
    hand = (80, 90, 200, 220)  # much larger box, fully covering the phone
    assert intersection_over_area(phone, hand) == pytest.approx(1.0)
    # IoU of the same pair is 800 / 15600 ~= 0.05 -- below the shipped 0.2 IoU threshold.

