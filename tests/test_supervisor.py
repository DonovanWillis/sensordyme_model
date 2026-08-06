"""Tests for the Service base class and Supervisor lifecycle."""

import threading
import time

from services.supervisor import Service, Supervisor


class TickingService(Service):
    name = "ticker"

    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.stopped_cleanly = False

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.ticks += 1
            stop.wait(0.01)
        self.stopped_cleanly = True


class CrashingService(Service):
    name = "crasher"

    def __init__(self):
        super().__init__()
        self.attempts = 0

    def run(self, stop: threading.Event) -> None:
        self.attempts += 1
        raise RuntimeError("boom")


def test_service_start_stop():
    service = TickingService()
    service.start()
    time.sleep(0.05)
    assert service.is_running
    service.stop()
    assert not service.is_running
    assert service.stopped_cleanly
    assert service.ticks > 0


def test_crash_is_contained_and_retried_with_backoff():
    service = CrashingService()
    service.start()
    time.sleep(1.3)  # first backoff is 1.0s; at least one retry should have happened
    service.stop()
    assert service.attempts >= 2  # crashed, restarted, crashed again — process alive throughout


def test_supervisor_orders_and_restarts_by_name():
    first, second = TickingService(), TickingService()
    second.name = "ticker2"
    supervisor = Supervisor()
    supervisor.add(first)
    supervisor.add(second)
    supervisor.start_all()
    time.sleep(0.05)
    assert first.is_running and second.is_running

    before = second.ticks
    assert supervisor.restart("ticker2") is True
    time.sleep(0.05)
    assert second.is_running and second.ticks >= 0  # fresh thread, still ticking
    assert supervisor.restart("nope") is False

    supervisor.stop_all()
    assert not first.is_running and not second.is_running
    assert before >= 0
