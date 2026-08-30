"""A dead provisioning service must say so, not report its last intention forever.

`ProvisioningService` duck-types the `Service` interface but never inherited it, and
so was missing all three guarantees `services/capture.py` and `services/supervisor.py`
both make. The consequential one: `run()` was the RAW thread target, so any unhandled
exception killed the thread silently -- and `status()` reports `self.state`, a plain
string the dead thread left behind. The health monitor therefore went on being told
"healthy: state: waiting" indefinitely while Wi-Fi setup had stopped working.

That is precisely the failure mode `core/device_health.py`'s own docstring warns
against: "A monitoring product that guesses in its own favour is..." -- the rest of
this codebase is careful about it, and this one file was not.

The other two guarantees (no double-start, a fresh stop Event per generation) matter
because `stop()` clears `self._thread` even when the join TIMED OUT, and BLE calls
can block: without them a restart starts a second thread beside a wedged one, both
driving the same BlueZ objects.
"""

from __future__ import annotations

import threading

import pytest

from services.provisioning import ProvisioningService


@pytest.fixture()
def service():
    """A provisioning service that will not touch BLE or nmcli."""
    return ProvisioningService(wifi=None, config={"provisioning": {"method": "off"}})


def test_a_crashed_thread_reports_unhealthy(service, monkeypatch):
    """The regression: a crash used to leave status() reporting healthy forever."""
    def _explode(_stop):
        raise RuntimeError("BlueZ went away")

    monkeypatch.setattr(service, "run", _explode)
    service.state = "waiting"

    service.start()
    service._thread.join(timeout=5)

    healthy, detail = service.status()
    assert healthy is False, (
        f"a crashed provisioning thread still reported healthy: {detail!r}"
    )
    assert "crashed" in detail and "BlueZ went away" in detail


def test_a_crash_is_reported_to_the_health_monitor(service, monkeypatch):
    """Containment is not enough on its own -- the failure has to reach the monitor."""
    reported: list[tuple[str, str]] = []

    class _Health:
        def report_failure(self, name, detail):
            reported.append((name, detail))

    monkeypatch.setattr(service, "run", lambda _stop: (_ for _ in ()).throw(RuntimeError("boom")))

    service.start(health=_Health())
    service._thread.join(timeout=5)

    assert reported, "the crash was contained but never reported to the health monitor"
    assert reported[0][0] == service.name
    assert "boom" in reported[0][1]


def test_a_thread_that_exited_is_not_reported_as_healthy(service, monkeypatch):
    """Even a clean early return must not leave a stale 'healthy' state behind."""
    monkeypatch.setattr(service, "run", lambda _stop: None)
    service.state = "waiting"

    service.start()
    service._thread.join(timeout=5)

    healthy, detail = service.status()
    assert healthy is False
    assert "not running" in detail


def test_unavailable_is_a_legitimate_terminal_state(service, monkeypatch):
    """A dev Mac / hotspot install returns immediately BY DESIGN -- not a failure.

    Without this carve-out the liveness check would flag every non-BLE deployment,
    which is how a useful health signal gets switched off by whoever gets paged.
    """
    monkeypatch.setattr(service, "run", lambda _stop: None)
    service.state = "unavailable"

    service.start()
    service._thread.join(timeout=5)

    healthy, _detail = service.status()
    assert healthy is False, "unavailable is still surfaced, just not as a dead thread"


def test_start_refuses_to_run_two_threads(service):
    """stop() clears _thread even on a join timeout; a restart must not double up."""
    running = threading.Event()
    release = threading.Event()

    def _block(stop):
        running.set()
        release.wait(timeout=5)

    service.run = _block  # type: ignore[method-assign]

    service.start()
    assert running.wait(timeout=5)
    first = service._thread

    service.start()  # must be refused while the first is alive
    assert service._thread is first, "a second provisioning thread was started"

    release.set()
    first.join(timeout=5)


def test_each_generation_gets_a_fresh_stop_event(service):
    """An abandoned thread polls the OLD event; clearing a shared one revives it."""
    seen: list[threading.Event] = []

    def _capture(stop):
        seen.append(stop)

    service.run = _capture  # type: ignore[method-assign]

    service.start()
    service._thread.join(timeout=5)
    service.stop()

    service.start()
    service._thread.join(timeout=5)
    service.stop()

    assert len(seen) == 2
    assert seen[0] is not seen[1], (
        "both generations shared one stop Event -- clearing it tells an abandoned "
        "thread to resume rather than exit"
    )


def test_a_healthy_running_service_still_reports_healthy(service):
    """The bound must not make every live service look broken."""
    release = threading.Event()

    def _idle(stop):
        release.wait(timeout=5)

    service.run = _idle  # type: ignore[method-assign]
    service.state = "online"

    service.start()
    try:
        healthy, detail = service.status()
        assert healthy is True, f"a live service reported unhealthy: {detail!r}"
    finally:
        release.set()
        service._thread.join(timeout=5)
