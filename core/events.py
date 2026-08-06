"""The Event data model shared by every vision module, the logger, and alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Event:
    """A single detected occurrence emitted by a :class:`modules.base.VisionModule`.

    Attributes:
        timestamp: UTC time the event was recorded.
        type: Short event type, e.g. "drowsiness", "distraction", "fall", "occupancy_change".
        duration: Seconds the triggering condition was sustained before the event fired.
        confidence: Detector confidence in [0.0, 1.0].
        module_name: Name of the module that emitted the event.
        contributing_signals: Optional free-text summary of what fed this event (e.g.
            "gaze_off + phone + hand_overlap"), for CSV analyzability. Empty string
            (the default) for modules that don't populate it.
    """

    timestamp: datetime
    type: str
    duration: float
    confidence: float
    module_name: str
    contributing_signals: str = ""

    @staticmethod
    def now(
        type: str, duration: float, confidence: float, module_name: str, contributing_signals: str = ""
    ) -> "Event":
        """Build an Event stamped with the current UTC time."""
        return Event(
            timestamp=datetime.now(timezone.utc),
            type=type,
            duration=duration,
            confidence=confidence,
            module_name=module_name,
            contributing_signals=contributing_signals,
        )
