"""Bounded stress: queues, leaks, retry storms, deadlocks, races, duplicates, silent failures.

Every test is capped in iterations AND wall time (threads are joined with a
timeout; a hang fails the test instead of the run). Numbers asserted are the
product's own bounds (backlog capacity, backoff cap); measured values are
printed (-s) for the report.
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import random
import threading
import time

import pytest

from core.events import Event
from core.uplink import MAX_QUEUE, CloudUplink
from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
WALL_LIMIT_S = 10.0


def _event(i: int = 0) -> Event:
    return Event.now("stress_event", 0.5, 0.9, "stress", f"n={i}")


def _fds() -> int:
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    return -1


class _Ok:
    status_code, headers, text = 200, {}, ""

    def __init__(self, body=None) -> None:
        self._body = body if body is not None else {"ok": True, "command": None}

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        return None


class _Recorder:
    """A backend stand-in that records every event key it receives (thread-safe)."""

    def __init__(self, lose_every: int = 0, fail: bool = False) -> None:
        self.keys: list[str] = []
        self.calls = 0
        self.lose_every, self.fail = lose_every, fail
        self._lock = threading.Lock()

    def post(self, url, json=None, **kwargs):
        import requests

        with self._lock:
            self.calls += 1
            if self.fail:
                raise requests.ConnectionError("SIMULATED: unreachable")
            self.keys.extend(e["event_key"] for e in (json or {}).get("events", []))
            if self.lose_every and self.calls % self.lose_every == 0:
                raise requests.ConnectionError("SIMULATED: response lost after storing")
        return _Ok()


def _uplink(transport, **cfg) -> CloudUplink:
    uplink = CloudUplink({"base_url": "http://127.0.0.1:1", "device_key": "dk_stress", **cfg})
    uplink._requests = transport
    return uplink


def _worker_thread() -> threading.Thread:
    return next(t for t in threading.enumerate() if t.name == "cloud-uplink")


# ---------------------------------------------------------------------- queue growth
def test_a_long_outage_backlog_stays_bounded_keeps_the_newest_and_drains_once(tmp_path):
    outage = Scenario("outage", "outage", (Segment(0.0), Segment(C, ATTENTIVE, Faults(network_down=True)),
                                           Segment(C + 60, ATTENTIVE, Faults())))
    with SimStack(tmp_path) as stack:
        device = stack.boot(outage)
        stack.claim()
        device.advance(C + 2)
        uplink = device.runtime.uplink
        capacity = uplink._queue.maxlen
        for i in range(3000):  # 6x the backlog capacity, while offline
            uplink.push_event(_event(i), "soft")
        device.advance(10.0)
        assert len(uplink._queue) <= capacity
        newest = uplink._queue[-1]["signals"]
        assert newest == "n=2999", "the newest alert must survive a full backlog"
        syncs_before = uplink.syncs
        device.advance(60.0)  # network back at C+60
        assert len(uplink._queue) == 0
        stored = [e for e in stack.events(limit=5000) if e["type"] == "stress_event"]
        assert len(stored) == capacity, (len(stored), capacity)
        assert len({e["timestamp"] for e in stored}) <= len(stored)
        print(f"backlog: capacity {capacity}, 3000 pushed offline, {len(stored)} stored once, "
              f"drained in {uplink.syncs - syncs_before} sync calls")


# ---------------------------------------------------------------------- retry storm + shutdown
def test_an_unreachable_backend_gets_a_bounded_number_of_attempts_and_the_worker_stops():
    transport = _Recorder(fail=True)
    uplink = _uplink(transport, poll_interval_s=0.02, max_backoff_s=0.2)
    uplink.start()
    worker = _worker_thread()
    time.sleep(2.0)
    uplink.close()
    worker.join(timeout=1.0)
    assert not worker.is_alive(), "the uplink worker did not stop within 1 s of close()"
    # Fixed cadence would be 100 attempts in 2 s; backoff halves the rate each failure up to the cap.
    assert 3 <= transport.calls <= 30, transport.calls
    print(f"retry storm: {transport.calls} attempts in 2.0 s (fixed cadence: 100); worker stopped")


# ---------------------------------------------------------------------- producer/consumer race
@pytest.mark.parametrize("lose_every", [0, 5])
def test_concurrent_alerts_while_syncing_are_delivered_all_and_once(lose_every):
    transport = _Recorder(lose_every=lose_every)
    uplink = _uplink(transport, poll_interval_s=0.002, max_backoff_s=0.01, queue_max=MAX_QUEUE)
    errors: list[BaseException] = []

    def produce(n: int) -> None:
        try:
            for i in range(100):
                uplink.push_event(_event(n * 1000 + i), "soft")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    uplink.start()
    worker = _worker_thread()
    producers = [threading.Thread(target=produce, args=(n,)) for n in range(8)]
    for t in producers:
        t.start()
    for t in producers:
        t.join(timeout=WALL_LIMIT_S)
    deadline = time.monotonic() + WALL_LIMIT_S
    while (uplink._queue or not uplink.connected) and time.monotonic() < deadline:
        time.sleep(0.01)
    uplink.close()
    worker.join(timeout=1.0)
    assert not errors and not worker.is_alive()
    signals = collections.Counter(transport.keys)
    assert len(signals) == 800, f"{800 - len(signals)} alerts lost"
    duplicates = sum(c - 1 for c in signals.values())
    if lose_every == 0:
        assert duplicates == 0
    print(f"race: 8x100 alerts while syncing, lose_every={lose_every}: 800 delivered, "
          f"{duplicates} resent after a lost response (the backend dedups by event_key)")


# ---------------------------------------------------------------------- runtime threads
class _CommandFeeder:
    """Answers every ingest with a fresh stop or resume, as a busy app would."""

    def __init__(self) -> None:
        self.issued: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._rand = random.Random(7)

    def post(self, url, json=None, **kwargs):
        with self._lock:
            action = self._rand.choice(["stop", "resume"])
            command_id = f"c{len(self.issued)}"
            self.issued.append((command_id, action))
        return _Ok({"ok": True, "command": {"set_protection": {"action": action, "confirm": True,
                                                                "id": command_id}}})


def test_inference_health_and_uplink_threads_run_together_without_deadlock_or_errors(tmp_path):
    """The production thread layout: inference (applies commands), health (monitoring check +
    health sample), uplink (sync) -- here all three at full speed for 1.5 s."""
    from sim.device import _NoWait

    errors: list[str] = []

    class _Errors(logging.Handler):
        def emit(self, record) -> None:
            if record.levelno >= logging.ERROR:
                errors.append(record.getMessage())

    handler = _Errors()
    logging.getLogger().addHandler(handler)
    try:
        with SimStack(tmp_path) as stack:
            device = stack.boot(sc.get("attentive"))
            stack.claim()
            device.advance(C)
            runtime, uplink = device.runtime, device.runtime.uplink
            feeder = _CommandFeeder()
            uplink._requests = feeder
            stop = threading.Event()
            counts = collections.Counter()
            loop = runtime._new_inference_loop_state()
            loop.frame_timeout_s = 0.0

            def guarded(name, body):
                def run():
                    try:
                        while not stop.is_set():
                            body()
                            counts[name] += 1
                    except BaseException as exc:  # noqa: BLE001 - reported below
                        errors.append(f"{name}: {exc!r}")
                return threading.Thread(target=run, name=f"stress-{name}")

            def infer():
                device.clock.advance(0.05)  # the sim's frame clock, as advance() pumps it
                frame = device.camera.grab()
                if frame is not None:
                    runtime.bus.publish(frame)
                runtime._inference_pass(loop, _NoWait())

            def health():
                runtime._monitoring_check()
                uplink.set_health(runtime._collect_health("uplink"))

            def sync():
                uplink.sync_once()
                uplink.push_event(_event(), "soft")
                time.sleep(0.001)  # yield: a spinning feeder would starve the other two of the GIL

            threads = [guarded("inference", infer), guarded("health", health), guarded("uplink", sync)]
            for t in threads:
                t.start()
            time.sleep(1.5)
            stop.set()
            for t in threads:
                t.join(timeout=WALL_LIMIT_S)
            hung = [t.name for t in threads if t.is_alive()]
            assert not hung, f"deadlock: {hung} still running {WALL_LIMIT_S} s after stop"
            assert not errors, errors[:5]
            # Quiesced: whatever was delivered last is what the device ends up in.
            runtime._apply_pending_commands()
            runtime._monitoring_check()
            delivered = dict(feeder.issued)
            last = runtime.protection_state()["last_command_id"]
            assert last in delivered
            assert (runtime._stopped is not None) == (delivered[last] == "stop")
            assert runtime.protection_state()["state"] == ("stopped" if delivered[last] == "stop" else "protecting")
            print(f"threads: {dict(counts)} iterations in 1.5 s, {len(feeder.issued)} commands, no errors")
    finally:
        logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------- leaks across restarts
def test_repeated_restarts_leak_no_threads_or_file_descriptors(tmp_path):
    with SimStack(tmp_path) as stack:
        monitor = InvariantMonitor(stack)
        device = stack.boot(sc.get("attentive"))
        stack.claim()
        device.advance(C)
        samples = []
        for _cycle in range(12):
            device = stack.restart(sc.get("attentive"))
            device.advance(3.0)
            gc.collect()
            samples.append((threading.active_count(), _fds(), len(gc.get_objects())))
        threads, fds, objects = zip(*samples, strict=True)
        assert max(threads) - min(threads) == 0, threads
        assert fds[-1] - fds[3] <= 2, fds
        per_cycle = (objects[-1] - objects[3]) / 8
        assert per_cycle < 5_000, f"{per_cycle:.0f} objects retained per restart"
        assert stack.status()["monitoring_state"] == "ready"
        monitor.assert_clean()
        print(f"restarts: threads {threads[-1]} (constant), fds {fds[3]}->{fds[-1]}, "
              f"objects {objects[3]}->{objects[-1]} ({per_cycle:.0f}/restart)")


# ---------------------------------------------------------------------- silent failures
class _Misbehaving:
    """A link that answers wrongly in one way while `mode` is set."""

    def __init__(self, inner) -> None:
        self._inner, self.mode = inner, None

    def post(self, url, **kwargs):
        import requests

        mode = self.mode
        if mode == "raise_unexpected":
            raise RuntimeError("SIMULATED: a bug in the HTTP stack")
        response = self._inner.post(url, **kwargs)  # the backend stores the batch
        if mode is None:
            return response
        if mode == "unreadable_200":
            return _Unreadable()
        if mode == "http_503":
            raise requests.HTTPError("HTTP 503", response=type("R", (), {"status_code": 503})())
        body = {"command_garbage": {"ok": True, "command": "garbage"},
                "command_list": {"ok": True, "command": ["set_preset", "x"]},
                "body_list": [1, 2]}[mode]
        return _Ok(body)


class _Unreadable(_Ok):
    def json(self):
        raise ValueError("SIMULATED: truncated response body")


@pytest.mark.parametrize("mode", ["unreadable_200", "http_503", "raise_unexpected", "command_garbage",
                                  "command_list", "body_list"])
def test_a_misbehaving_link_never_loses_or_doubles_an_alert(tmp_path, mode):
    script = Scenario("asleep_twice", "x", (Segment(0.0), Segment(C + 3, ASLEEP), Segment(C + 8, ATTENTIVE),
                                            Segment(C + 23, ASLEEP), Segment(C + 28, ATTENTIVE)))
    with SimStack(tmp_path) as stack:
        link = _Misbehaving(stack.transport)
        monitor = InvariantMonitor(stack)
        device = stack.boot(script, transport=link)
        stack.claim()
        device.advance(C)
        link.mode = mode
        device.advance(30.0)  # both episodes happen while the link misbehaves
        assert device.runtime.protection_state()["state"] == "protecting", "the device kept protecting"
        link.mode = None
        device.advance(10.0)
        drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
        assert len(drowsy) == 2, (mode, stack.event_types())
        assert device.runtime.uplink.connected is True
        monitor.assert_clean()
