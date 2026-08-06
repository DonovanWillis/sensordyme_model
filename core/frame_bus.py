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
from typing import Optional

import numpy as np


class FrameBus:
    """Single-slot, many-reader handoff for the newest camera frame."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0  # increments on every publish; readers use it to detect "new"
        self._published_at = 0.0  # monotonic time of the last publish

    def publish(self, frame: np.ndarray) -> None:
        """Store `frame` as the newest and wake every waiting consumer."""
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._published_at = time.monotonic()
            self._cond.notify_all()

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
