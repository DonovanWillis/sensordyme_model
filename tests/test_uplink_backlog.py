"""What the device keeps when it cannot reach the backend.

The queue is bounded, so something has to be discarded during a long outage.
For a safety product the answer is not arbitrary: what is happening now matters
more than what happened an hour ago.
"""

import types

from core.uplink import CloudUplink


class FakeRequests:
    """A requests-alike that always fails, optionally with an HTTP response.

    Signals the worker to stop as soon as it has been called once, so a test
    exercises exactly one pass of the loop.
    """

    def __init__(self, status=None, stop=None, during_send=None):
        self.status = status
        self.stop = stop
        self.during_send = during_send
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.stop is not None:
            self.stop.set()
        # Events keep happening while a send is in flight. This is the window
        # the bug lived in: the batch has been taken and the queue emptied, so
        # anything recorded now sits in a queue that the retry will push into.
        if self.during_send is not None:
            self.during_send()
        if self.status is None:
            raise OSError("connection refused")
        response = types.SimpleNamespace(status_code=self.status)
        error = RuntimeError(f"{self.status} Client Error")
        error.response = response
        raise error


def make_uplink(queue_max=5, status=None):
    uplink = CloudUplink({
        "base_url": "http://backend.local:8787", "device_key": "dk_x",
        "poll_interval_s": 0.01, "queue_max": queue_max,
    })
    uplink._requests = FakeRequests(status, stop=uplink._stop)
    return uplink


def queue_event(uplink, name):
    """push_event takes an alert object; only the fields it reads matter here."""
    event = types.SimpleNamespace(
        timestamp=types.SimpleNamespace(isoformat=lambda: "2026-08-07T00:00:00"),
        type=name, confidence=1.0, contributing_signals={},
    )
    uplink.push_event(event, "loud")


def drain_once(uplink):
    """Run exactly one iteration of the worker body.

    The fake sets the stop event when the POST is attempted, so the loop runs
    its body once and then exits. Setting stop beforehand would skip the body
    entirely and make every assertion below vacuously true.
    """
    uplink._run()
    assert uplink._requests.calls == 1, "the worker body never ran"


def test_a_retry_does_not_evict_the_alerts_that_just_happened():
    """The regression.

    `appendleft` on a full deque evicts from the RIGHT — the newest end. So
    re-queuing a failed batch threw away the alerts that had just happened in
    order to keep ones that had already failed to send. For a safety product
    that is exactly backwards: during a long outage the device discarded what
    was happening now to preserve what happened an hour ago.
    """
    uplink = make_uplink(queue_max=3)
    for i in range(3):
        queue_event(uplink, f"old{i}")
    # Fill the queue with fresh events while the doomed send is in flight.
    uplink._requests.during_send = lambda: [queue_event(uplink, f"new{i}") for i in range(3)]

    drain_once(uplink)

    kept = [e["type"] for e in uplink._queue]
    assert all(k.startswith("new") for k in kept), \
        f"newer events were pushed out by a retry of older ones: {kept}"


def test_older_events_are_still_kept_when_there_is_room():
    """Dropping is a last resort, not the default: a short outage must not lose
    anything at all."""
    uplink = make_uplink(queue_max=10)
    for i in range(3):
        queue_event(uplink, f"old{i}")
    uplink._requests.during_send = lambda: queue_event(uplink, "new0")

    drain_once(uplink)

    kept = [e["type"] for e in uplink._queue]
    assert kept == ["old0", "old1", "old2", "new0"], kept


def test_a_rejected_device_key_is_not_reported_as_unreachable(capsys):
    """The backend answered — it refused. Calling that "cannot reach" sends
    someone to debug a network that is working perfectly."""
    uplink = make_uplink(status=401)
    drain_once(uplink)
    output = capsys.readouterr().out
    assert "cannot reach" not in output
    assert "401" in output and "key" in output.lower()


def test_an_unreachable_backend_still_says_so(capsys):
    uplink = make_uplink(status=None)
    drain_once(uplink)
    assert "cannot reach" in capsys.readouterr().out


def test_every_pushed_event_gets_a_key():
    """cloud/app.py deduplicates on (device_id, event_key). No key, no
    protection against a resend becoming a duplicate row."""
    uplink = make_uplink()
    queue_event(uplink, "fall")
    assert uplink._queue[-1]["event_key"], "push_event must generate a non-empty event_key"


def test_two_different_events_get_two_different_keys():
    uplink = make_uplink()
    queue_event(uplink, "fall")
    queue_event(uplink, "drowsiness")
    keys = [e["event_key"] for e in uplink._queue]
    assert keys[0] != keys[1]


def test_a_retried_event_keeps_the_same_key():
    """The property the whole fix depends on: a batch that fails to send and
    gets re-queued (core/uplink.py's `appendleft` recovery path) must carry
    the SAME event_key on the retry as the original attempt — a regenerated
    key on each send would defeat server-side dedup entirely, since the
    server would see two different keys for what is really one event."""
    uplink = make_uplink(queue_max=5)
    queue_event(uplink, "fall")
    key_before_send = uplink._queue[-1]["event_key"]

    drain_once(uplink)  # the send fails (FakeRequests always raises) and re-queues

    assert len(uplink._queue) == 1, "the failed event should have been re-queued, not dropped"
    assert uplink._queue[-1]["event_key"] == key_before_send, \
        "a retry must not regenerate the event_key — the server dedups on it"
