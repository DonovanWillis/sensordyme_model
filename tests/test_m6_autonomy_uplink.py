"""M6 autonomy: the uplink reconnects with bounded, jittered backoff.

Before M6 the worker waited the fixed poll interval (2 s) after every failure,
for as long as an outage lasted -- an offline device retried DNS + TCP every two
seconds forever, and every device behind one backend outage reconnected in
lock-step. These tests drive the real worker loop with a scripted transport.
"""

import types
from itertools import pairwise

import requests

from core.uplink import BACKOFF_CAP_S, CloudUplink, backoff_delay


class _Waits:
    """Stands in for the worker's stop Event: records every wait, stops after `limit`."""

    def __init__(self, limit):
        self.limit = limit
        self.waits: list[float] = []

    def is_set(self):
        return len(self.waits) >= self.limit

    def wait(self, timeout=None):
        self.waits.append(timeout)
        return self.is_set()

    def set(self):
        self.limit = 0


class _Transport:
    """Answers each POST from a script: "fail" raises ConnectionError, else 200."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def post(self, url, json=None, **_kwargs):
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if step == "fail":
            raise requests.ConnectionError("network unreachable")
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                     json=lambda: {"command": None})


def _uplink(script, poll=2.0, cap=None):
    config = {"base_url": "https://backend.example", "device_key": "dk_secret", "poll_interval_s": poll}
    if cap is not None:
        config["max_backoff_s"] = cap
    uplink = CloudUplink(config)
    uplink._requests = _Transport(script)
    uplink._stop = _Waits(len(script))
    return uplink


def test_backoff_grows_and_is_bounded_by_the_cap():
    lo = [backoff_delay(n, 2.0, 30.0, rand=lambda: 0.0) for n in range(12)]
    hi = [backoff_delay(n, 2.0, 30.0, rand=lambda: 1.0) for n in range(12)]
    assert lo[0] == hi[0] == 2.0, "no failure: the normal poll interval"
    assert hi[1:5] == [4.0, 8.0, 16.0, 30.0]
    assert max(hi) == 30.0 and max(lo) == 15.0, "never beyond the cap; never below half of it"
    assert all(a <= b for a, b in pairwise(hi)), "monotone up to the cap"


def test_backoff_is_jittered_so_a_fleet_does_not_reconnect_in_lock_step():
    import random

    rng = random.Random(7)
    delays = {round(backoff_delay(5, 2.0, 30.0, rand=rng.random), 3) for _ in range(50)}
    assert len(delays) > 40
    assert all(15.0 <= d <= 30.0 for d in delays)


def test_a_zero_poll_interval_stays_zero():
    assert backoff_delay(9, 0.0) == 0.0


def test_the_worker_backs_off_while_offline_and_recovers_at_once():
    uplink = _uplink(["fail"] * 6 + ["ok", "ok"])
    uplink._run()
    waits = uplink._stop.waits
    assert len(waits) == 8
    offline, online = waits[:6], waits[6:]
    assert all(b >= a / 2 for a, b in pairwise(offline))
    assert offline[0] <= 4.0 and offline[-1] >= BACKOFF_CAP_S / 2, offline
    assert max(offline) <= BACKOFF_CAP_S
    assert online == [2.0, 2.0], "the first success returns to the normal poll interval"
    assert uplink.connected


def test_the_cap_is_configurable():
    uplink = _uplink(["fail"] * 10, cap=8.0)
    uplink._run()
    assert max(uplink._stop.waits) <= 8.0


def test_events_queued_offline_are_delivered_after_reconnection():
    sent = []

    class Recording(_Transport):
        def post(self, url, json=None, **kwargs):
            response = super().post(url, json=json, **kwargs)
            sent.extend(e["type"] for e in json["events"])
            return response

    uplink = _uplink(["fail", "fail", "ok"])
    uplink._requests = Recording(["fail", "fail", "ok"])
    event = types.SimpleNamespace(timestamp=types.SimpleNamespace(isoformat=lambda: "2026-11-16T00:00:00"),
                                  type="drowsiness", confidence=0.9, contributing_signals={}, duration=2.0)
    uplink.push_event(event, "loud")
    uplink._run()
    assert sent == ["drowsiness"], "kept through the outage, sent exactly once after it"
