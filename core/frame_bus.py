"""FrameBus: the shared latest-frame slot between camera capture and its consumers.

The camera is opened exactly once (by services/capture.py) and its frames are
published here. Consumers — AI inference, the dashboard's live stream, a future
recording service — each read the *latest* frame at their own cadence instead of
forming a queue:

  * Inference always processes the freshest frame. If a frame takes longer than
    one camera interval to process, intermediate frames are simply skipped —
    latency stays bounded instead of growing a backlog.
  * The streaming encoder runs at its own (lower) fps without slowing capture
    or inference down.

Zero-copy by contract: `publish()` stores the array reference — no copy — and
consumers MUST NOT mutate a frame they got from the bus (annotate on a copy,
exactly as the modules already do). The capture side allocates a fresh array per
frame (both Picamera2's capture_array and cv2's read do), so a published frame
is never overwritten in place.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np


class FrameBus:
    """Single-slot, many-reader handoff for the newest camera frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0  # increments on every publish; readers use it to detect "new"
        self._published_at = 0.0  # monotonic time of the last publish
        #: Recent inter-frame intervals, PER CONSUMER, drained by interval_stats().
        #: One shared buffer was a destructive read: the inference loop samples
        #: every ~5s to feed the uplink, and the dashboard thread samples on every
        #: GET /api/health. Whichever read first got the data and the other got an
        #: empty dict -- no keys at all, so nothing downstream could tell "no
        #: stalls" from "not measured". A real 1.2s stall reached exactly one
        #: consumer. The metric added to prove the pipeline is not silently losing
        #: evidence was itself silently losing evidence.
        #:
        #: Bounded per consumer; 4096 at ~30 fps is over two minutes of history,
        #: far more than any sampling period needs.
        self._intervals: dict[str, deque[float]] = {}

    def publish(self, frame: np.ndarray) -> None:
        """Store `frame` as the newest and wake every waiting consumer."""
        with self._cond:
            previous_at = self._published_at
            self._frame = frame
            self._seq += 1
            self._published_at = time.monotonic()
            if previous_at:
                # Interval history, not just a rate. A MEAN fps cannot see the
                # failure that matters here: EventEngine discards an unconfirmed
                # track when two processed frames land more than occlusion_grace_s
                # (1.0s by default, and left at default by fall, every zone rule,
                # flailing and choking) apart. A pipeline averaging 30 fps with one
                # 1.2s stall loses in-progress evidence and reports 30 fps while
                # doing it. Bounded so a multi-day soak cannot grow this list.
                interval = self._published_at - previous_at
                for samples in self._intervals.values():
                    samples.append(interval)
            self._cond.notify_all()

    def interval_stats(self, caller: str = "default") -> dict:
        """Frame-interval distribution since THIS caller's last call, then reset.

        Percentiles over a window rather than an all-time mean, because the
        question is "did the pipeline ever stall", and an all-time aggregate
        buries exactly that.

        Keyed per caller so two consumers cannot consume each other's window --
        see the note on `_intervals`. A caller's first call registers it and
        returns an empty dict, the same way the fps baseline works: there is no
        honest window until there is a previous read to measure from.
        """
        with self._cond:
            buffer = self._intervals.get(caller)
            if buffer is None:
                self._intervals[caller] = deque(maxlen=4096)
                return {}
            samples = sorted(buffer)
            buffer.clear()
        if not samples:
            return {}

        def pct(p: float) -> float:
            return samples[min(len(samples) - 1, int(p * len(samples)))]

        return {
            "frames": len(samples),
            "p50_ms": round(1000 * pct(0.50), 1),
            "p95_ms": round(1000 * pct(0.95), 1),
            "p99_ms": round(1000 * pct(0.99), 1),
            "max_ms": round(1000 * samples[-1], 1),
            # The three thresholds that mean something to this product: a quarter
            # second is a visibly dropped frame, half a second is a stall, and one
            # second is the point at which in-progress evidence is discarded.
            "stalls_250ms": sum(1 for s in samples if s > 0.25),
            "stalls_500ms": sum(1 for s in samples if s > 0.50),
            "evidence_losing_stalls_1s": sum(1 for s in samples if s > 1.0),
        }

    def latest(self) -> tuple[Optional[np.ndarray], int]:
        """The newest frame and its sequence number, without waiting. (None, 0) before first publish."""
        with self._cond:
            return self._frame, self._seq

    @property
    def sequence(self) -> int:
        """Total frames published. Lets health measure real throughput between
        samples without holding a frame reference just to read the counter."""
        with self._cond:
            return self._seq

    def wait_for_fresh(self, last_seq: int, timeout: float = 1.0) -> tuple[Optional[np.ndarray], int]:
        """Block until a frame newer than `last_seq` arrives (or timeout).

        Returns the newest (frame, seq) either way; the caller compares seq to
        `last_seq` to tell a fresh frame from a timeout.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq, timeout=timeout)
            return self._frame, self._seq

    def age_s(self) -> float:
        """Seconds since the last publish (inf before the first frame).

        This is the health monitor's staleness signal: a rising age with the
        capture service "running" means the camera stopped producing frames.
        """
        with self._cond:
            if self._seq == 0:
                return float("inf")
            return time.monotonic() - self._published_at
