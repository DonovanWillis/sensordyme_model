"""Occupancy module: counts people per frame with YOLO11-nano, emits occupancy_change events."""

from __future__ import annotations

import cv2
import numpy as np

from core.detectors import get_yolo_model
from core.events import Event
from modules.base import VisionModule, register_module


@register_module("occupancy")
class OccupancyModule(VisionModule):
    """Detects people with YOLO11-nano and emits an event whenever the confirmed count changes."""

    name = "occupancy"

    def __init__(self, config: dict) -> None:
        """Load YOLO11-nano weights and read thresholds from config."""
        super().__init__(config)
        self._confidence_threshold = float(config["confidence_threshold"])
        self._confirm_frames = int(config["confirm_frames"])
        weights = config.get("model_weights", "yolo11n.pt")
        self._model = get_yolo_model(weights)

        self._last_emitted_count = -1
        self._pending_count: int | None = None
        self._pending_streak = 0
        self._last_boxes: list[tuple[int, int, int, int]] = []

    def process(self, frame: np.ndarray) -> list[Event]:
        """Detect people in `frame`; emit occupancy_change once a new count is confirmed."""
        events: list[Event] = []
        results = self._model.predict(frame, verbose=False, classes=[0], conf=self._confidence_threshold)

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                boxes.append(tuple(int(v) for v in xyxy))
                confidences.append(float(box.conf[0]))

        self._last_boxes = boxes
        count = len(boxes)

        if count == self._pending_count:
            self._pending_streak += 1
        else:
            self._pending_count = count
            self._pending_streak = 1

        if self._pending_streak >= self._confirm_frames and count != self._last_emitted_count:
            confidence = sum(confidences) / len(confidences) if confidences else 1.0
            events.append(Event.now("occupancy_change", 0.0, confidence, self.name))
            self._last_emitted_count = count

        return events

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw a box per detected person and the current count."""
        for (x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame, f"People: {len(self._last_boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
        )
        return frame
