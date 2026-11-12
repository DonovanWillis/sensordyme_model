"""Per-frame detections: the raw observation view that AI-configured rules need.

`Event` is the wrong input for custom rules. By the time a module emits one it
has already been debounced, and it carries no boxes and no counts — so a rule
like "someone in the pool area" or "more than three people" cannot be evaluated
from Events at all, and anything that could would be debounced twice.

This is the parallel, *undebounced* view of a single frame:

    module.process(frame)      -> list[Event]           (already-confirmed occurrences)
    module.get_detections()    -> DetectionFrame        (what was in THIS frame)

Modules opt in by overriding `get_detections`, the same way `get_telemetry_snapshot`
is optional. A module that doesn't override it simply can't back zone or count
rules, and `RuleEngine` treats that as "no observation" rather than guessing.

Boxes are pixel-space `(x1, y1, x2, y2)` in the frame's own coordinates, with the
frame size carried alongside, so zone membership (normalized 0..1 polygons) can be
resolved without every module having to know about zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Detection:
    """One thing seen in one frame."""

    label: str                                             # "person", "phone", "smoke", ...
    confidence: float = 1.0
    box: Optional[tuple[float, float, float, float]] = None  # pixel-space x1, y1, x2, y2
    #: The shared tracker's id for this subject (core/tracking.py ObjectTracker),
    #: None when no tracker followed it. Lets per-subject rules (dwell) follow
    #: the device's one track instead of re-matching boxes themselves.
    track_id: Optional[int] = None


@dataclass(frozen=True)
class DetectionFrame:
    """Everything one module saw in one frame, with the geometry to interpret it."""

    width: int
    height: int
    detections: tuple[Detection, ...] = field(default_factory=tuple)

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def labels(self) -> set[str]:
        return {d.label for d in self.detections}

    def of(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]

    def count(self, label: str) -> int:
        return sum(1 for d in self.detections if d.label == label)

    def boxes(self, label: str) -> list[tuple[float, float, float, float]]:
        return [d.box for d in self.detections if d.label == label and d.box is not None]

    def mean_confidence(self, label: str) -> float:
        """Average confidence for a label; 0.0 when absent.

        Mean rather than max: one high-confidence flicker among several weak
        detections should not present itself as a confident observation.
        """
        scores = [d.confidence for d in self.detections if d.label == label]
        return sum(scores) / len(scores) if scores else 0.0
