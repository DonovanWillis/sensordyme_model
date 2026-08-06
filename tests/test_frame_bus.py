"""Tests for the FrameBus latest-frame handoff."""

import threading
import time

import numpy as np

from core.frame_bus import FrameBus


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_latest_is_none_before_first_publish():
    bus = FrameBus()
    frame, seq = bus.latest()
    assert frame is None and seq == 0
    assert bus.age_s() == float("inf")


def test_publish_updates_latest_and_seq():
    bus = FrameBus()
    bus.publish(_frame(1))
    bus.publish(_frame(2))
    frame, seq = bus.latest()
    assert seq == 2
    assert frame[0, 0, 0] == 2  # consumers always see the newest, frame 1 was skipped
    assert bus.age_s() < 1.0


def test_wait_for_fresh_returns_immediately_when_new_frame_exists():
    bus = FrameBus()
    bus.publish(_frame(7))
    frame, seq = bus.wait_for_fresh(last_seq=0, timeout=0.01)
    assert seq == 1 and frame[0, 0, 0] == 7


def test_wait_for_fresh_times_out_without_new_frame():
    bus = FrameBus()
    bus.publish(_frame(1))
    start = time.monotonic()
    _, seq = bus.wait_for_fresh(last_seq=1, timeout=0.05)
    assert seq == 1  # unchanged -> caller knows it timed out
    assert time.monotonic() - start >= 0.04


def test_wait_for_fresh_wakes_on_cross_thread_publish():
    bus = FrameBus()
    result = {}

    def consumer():
        result["frame"], result["seq"] = bus.wait_for_fresh(last_seq=0, timeout=2.0)

    thread = threading.Thread(target=consumer)
    thread.start()
    time.sleep(0.05)
    bus.publish(_frame(9))
    thread.join(timeout=2)
    assert result["seq"] == 1 and result["frame"][0, 0, 0] == 9
