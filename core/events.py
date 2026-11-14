"""The Event data model shared by every vision module, the logger, and alerts."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional


def new_event_id() -> str:
    """A fresh event identity: 32 lowercase hex characters (uuid4)."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Event:
    """A single detected occurrence emitted by a :class:`modules.base.VisionModule`.

    Attributes:
        timestamp: UTC time the event was recorded.
        type: Short event type, e.g. "drowsiness", "distraction", "fall", "occupancy_change".
        duration: Seconds the triggering condition was sustained before the event fired.
        confidence: Certainty in [0.0, 1.0] that the condition this event type
            names really occurred, or None when there is no graded evidence. Never
            a raw measurement or a threshold — see core/confidence.py, which
            defines the meaning and the only three ways a value is produced.
        module_name: Name of the module that emitted the event.
        contributing_signals: Optional free-text summary of what fed this event (e.g.
            "gaze_off + phone + hand_overlap"), for CSV analyzability. Empty string
            (the default) for modules that don't populate it.
        event_id: This occurrence's identity, minted once when the Event is built
            (M8.3, docs/EVENT_PIPELINE.md). The alert policy records what it did
            with the event under this id, and the uplink sends it as `event_key`,
            the key the backend stores the event under exactly once. Not part of
            equality or repr: two Events with the same fields still compare equal.
    """

    timestamp: datetime
    type: str
    duration: float
    confidence: Optional[float]
    module_name: str
    contributing_signals: str = ""
    event_id: str = field(default_factory=new_event_id, compare=False, repr=False)

    @staticmethod
    def now(
        type: str,  # noqa: A002 - mirrors the Event.type field; renaming changes the keyword API
        duration: float, confidence: Optional[float], module_name: str, contributing_signals: str = ""
    ) -> "Event":
        """Build an Event stamped with the current UTC time."""
        return Event(
            timestamp=datetime.now(UTC),
            type=type,
            duration=duration,
            confidence=confidence,
            module_name=module_name,
            contributing_signals=contributing_signals,
        )
