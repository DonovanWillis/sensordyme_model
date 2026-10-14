"""A batch the backend can never accept must not block every batch after it.

The worker re-queues a failed batch at the FRONT of the queue, which is right
for an outage and fatal for a batch that fails for a reason of its own: it is
retried forever, every newer alert queues behind it, and the heartbeat that
carries commands stops with it. Two ways that happened:

  * a non-finite float (NaN/inf) anywhere in the payload -- `requests` refuses
    to encode it before anything is sent, so the "send" fails identically on
    every poll and the log blames the network;
  * a 4xx answer the backend will give again for the same body (422 for a
    batch over its cap, 400 for a malformed one).
"""

import json
import math
import types

import requests

from core.uplink import MAX_QUEUE, CloudUplink


class RecordingRequests:
    """Encodes the body exactly as requests would, then answers in turn."""

    def __init__(self, stop, answers):
        self.stop = stop
        self.answers = list(answers)
        self.sent: list[dict] = []

    def post(self, url, json=None, **_kwargs):
        body = json
        try:
            _encode(body)
        except ValueError as exc:
            raise requests.exceptions.InvalidJSONError(exc) from exc
        status = self.answers.pop(0) if self.answers else 200
        if not self.answers:
            self.stop.set()
        if status != 200:
            error = requests.HTTPError(f"{status} Client Error")
            error.response = types.SimpleNamespace(status_code=status)
            raise error
        self.sent.append(body)
        return types.SimpleNamespace(raise_for_status=lambda: None, json=dict, status_code=200)


def _encode(body):
    return json.dumps(body, allow_nan=False)


def make(answers, queue_max=50):
    uplink = CloudUplink({
        "base_url": "http://backend.local:8787", "device_key": "dk_x",
        "poll_interval_s": 0, "queue_max": queue_max,
    })
    uplink._requests = RecordingRequests(uplink._stop, answers)
    return uplink


def push(uplink, name, confidence=0.9, duration=1.0):
    event = types.SimpleNamespace(
        timestamp=types.SimpleNamespace(isoformat=lambda: "2026-10-14T00:00:00"),
        type=name, confidence=confidence, contributing_signals={"x": 1.0}, duration=duration,
    )
    uplink.push_event(event, "loud")


def delivered(uplink):
    return [e["type"] for body in uplink._requests.sent for e in body["events"]]


def test_a_nan_confidence_is_sent_as_a_number_not_retried_forever():
    uplink = make([200])
    push(uplink, "nan_event", confidence=float("nan"), duration=float("inf"))
    push(uplink, "real_fall")
    uplink._run()
    assert delivered(uplink) == ["nan_event", "real_fall"]
    sent = uplink._requests.sent[0]["events"][0]
    assert math.isfinite(sent["confidence"]) and math.isfinite(sent["duration"])


def test_a_nan_in_health_or_signals_does_not_stop_the_heartbeat():
    uplink = make([200])
    uplink.set_health({"thermal": {"temp_c": float("nan")}})
    event = types.SimpleNamespace(
        timestamp=types.SimpleNamespace(isoformat=lambda: "2026-10-14T00:00:00"),
        type="e", confidence=0.5, contributing_signals={"ear": float("nan")}, duration=0.0,
    )
    uplink.push_event(event, "loud")
    uplink._run()
    assert delivered(uplink) == ["e"]
    assert uplink.connected


def test_a_batch_the_backend_rejects_is_dropped_not_retried_forever():
    uplink = make([422, 200])
    push(uplink, "poison")
    uplink._run()
    assert len(uplink._queue) == 0, "a 422 batch was re-queued to fail again"
    push(uplink, "later_alert")
    uplink._stop.clear()
    uplink._run()
    assert delivered(uplink) == ["later_alert"]


def test_auth_and_rate_limit_answers_are_still_retried():
    """401/403 can be fixed by re-keying, 429/408 by waiting: keep the events."""
    for status in (401, 403, 408, 429):
        uplink = make([status])
        push(uplink, "keep_me")
        uplink._run()
        assert [e["type"] for e in uplink._queue] == ["keep_me"], status


def test_queue_max_is_bounded_to_what_the_backend_accepts():
    """Above the backend's ingest cap every flush is a 422; 0 drops everything."""
    assert make([], queue_max=5000)._queue.maxlen == MAX_QUEUE
    assert make([], queue_max=0)._queue.maxlen == 1


def test_a_large_backlog_goes_up_in_slices_and_is_not_lost():
    """A 413 means "too big", not "wrong": the batch is retried smaller, never
    dropped. A 1000-event backlog used to go up whole and be discarded."""
    uplink = make([413, 200, 200, 200, 200], queue_max=1000)
    for i in range(300):
        push(uplink, f"e{i}")
    uplink._run()  # one backlog flush: the 413, then slices until the fake stops
    sent = delivered(uplink)
    remaining = [e["type"] for e in uplink._queue]
    assert sent + remaining == [f"e{i}" for i in range(300)], "events lost or reordered"
    assert len(sent) == 200, "the backlog is drained without waiting a poll per slice"
    assert max(len(body["events"]) for body in uplink._requests.sent) <= 50


def test_numpy_values_in_the_payload_are_sent_not_stuck():
    import numpy as np

    uplink = make([200])
    uplink.set_health({"temp_c": np.float32(51.5), "frames": np.int64(3), "hist": np.array([1.0, np.nan])})
    event = types.SimpleNamespace(
        timestamp=types.SimpleNamespace(isoformat=lambda: "2026-10-14T00:00:00"),
        type="e", confidence=np.float32(0.7), contributing_signals={"ear": np.float64(0.2)}, duration=0.0,
    )
    uplink.push_event(event, "loud")
    uplink._run()
    assert delivered(uplink) == ["e"]
    assert uplink._requests.sent[0]["health"] == {"temp_c": 51.5, "frames": 3, "hist": [1.0, None]}
