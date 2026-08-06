"""Occupancy module: person counting plus zone security events.

Counting uses YOLO person detection with confirm_frames debounce (as before).
Optional user-drawn zones (config `zones:`) add security events through the
shared event lifecycle (core/event_engine.py):
  * zone_violation — a person's feet inside a `restricted` zone, sustained;
  * person_lingering — still inside the same zone past `linger_s`.
Both resolve when the zone empties, and brief occlusions don't reset progress.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from core.detectors import get_yolo_model
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.zones import load_zones
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

        # Optional security zones (feet-point membership; see core/zones.py).
        self._zones = load_zones(config.get("zones", {}))
        self._engine = EventEngine({})
        linger_s = float(config.get("linger_s", 30.0))
        zone_confirm_s = float(config.get("zone_confirm_s", 2.0))
        for zone in self._zones:
            if zone.kind in ("restricted", "boundary"):
                self._engine.add_rule(RuleSpec(
                    detection=f"zone_violation:{zone.name}", severity="critical",
                    persist_s=zone_confirm_s, min_frames=3, resolve_after_s=5.0,
                ))
                self._engine.add_rule(RuleSpec(
                    detection=f"person_lingering:{zone.name}", severity="warning",
                    persist_s=linger_s, min_frames=10, resolve_after_s=5.0,
                ))

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

        # Zone security events (only when zones are configured).
        if self._zones:
            now = time.monotonic()
            h, w = frame.shape[:2]
            transitions = []
            for zone in self._zones:
                if zone.kind not in ("restricted", "boundary"):
                    continue
                inside = any(zone.contains_box(b, w, h) for b in boxes)
                transitions += self._engine.update(f"zone_violation:{zone.name}", inside, 0.85, now)
                transitions += self._engine.update(f"person_lingering:{zone.name}", inside, 0.85, now)
            for tr in transitions:
                base, _, zone_name = tr.detection.partition(":")
                if tr.state == CONFIRMED:
                    events.append(Event.now(
                        base, tr.duration_s, tr.confidence, self.name,
                        f"zone={zone_name} confirmed after {tr.duration_s:.1f}s",
                    ))
                elif tr.state == RESOLVED:
                    events.append(Event.now(f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                                            f"zone={zone_name}"))

        return events

    def is_alert_active(self) -> bool:
        """True while any zone violation is confirmed and unresolved."""
        return bool(self._engine.active_detections())

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones, a box per detected person, and the current count."""
        h, w = frame.shape[:2]
        for zone in self._zones:
            pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 220), thickness=2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        for (x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame, f"People: {len(self._last_boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2
        )
        return frame
