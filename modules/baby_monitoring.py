"""Baby monitoring module: presence, safe/restricted zones, and stillness — honestly.

What this module actually detects (and what it refuses to fake) is documented in
core/taxonomy.py under `baby_monitoring`. The short version:

  SUPPORTED: child present/absent, leaving a configured safe zone (crib/play
  area), entering configured restricted zones (stairs/pool/stove — the USER
  draws these; a camera cannot know what a pool is), unusual stillness,
  person-count changes, activity bursts (logged observations).

  DELIBERATELY NOT CLAIMED: face-covered / face-down detection (adult face
  models are not reliable enough on infants for a safety-critical claim),
  sleeping vs awake, crying (no microphone). Faking those would be worse than
  omitting them.

All temporal reasoning goes through core/event_engine.py: nothing here alerts
on a single frame, brief occlusions don't reset progress, and every alert has
an observed → developing → confirmed lifecycle with a resolution.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from core.detectors import get_yolo_model
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.zones import Zone, load_zones
from modules.base import VisionModule, register_module

_PERSON_CLASS = 0  # COCO


@register_module("baby_monitoring")
class BabyMonitoringModule(VisionModule):
    """Child-safety monitoring: zones + presence + stillness with staged confirmation."""

    name = "baby_monitoring"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._confidence_threshold = float(config.get("confidence_threshold", 0.4))
        self._detect_every_n = max(1, int(config.get("detect_every_n_frames", 2)))
        self._still_max_motion = float(config.get("still_max_motion", 0.015))
        self._expect_presence = bool(config.get("expect_presence", False))
        self._weights = config.get("model_weights", "yolo11n.pt")
        self._model = None  # lazy: loaded on first frame (keeps tests light)

        self._zones: list[Zone] = load_zones(config.get("zones", {}))

        # --- event lifecycle rules --------------------------------------------
        self._engine = EventEngine({})
        still_s = float(config.get("still_warning_s", 120.0))
        absence_s = float(config.get("absence_warning_s", 30.0))
        zone_s = float(config.get("zone_confirm_s", 2.0))
        for zone in self._zones:
            if zone.kind == "restricted":
                self._engine.add_rule(RuleSpec(
                    detection=f"zone_violation:{zone.name}", severity="critical",
                    persist_s=zone_s, min_frames=3, resolve_after_s=5.0,
                ))
        if any(z.kind == "safe" for z in self._zones):
            self._engine.add_rule(RuleSpec(
                detection="safe_zone_exit", severity="critical",
                persist_s=zone_s, min_frames=3, resolve_after_s=5.0,
            ))
        self._engine.add_rule(RuleSpec(
            detection="unusual_stillness", severity="warning",
            persist_s=still_s, min_frames=10, occlusion_grace_s=3.0, resolve_after_s=10.0,
        ))
        if self._expect_presence:
            self._engine.add_rule(RuleSpec(
                detection="person_absent", severity="warning",
                persist_s=absence_s, min_frames=5, resolve_after_s=5.0,
            ))
        self._engine.add_rule(RuleSpec(
            detection="activity_burst", severity="info",
            persist_s=0.5, min_frames=2, resolve_after_s=3.0, cooldown_s=10.0,
        ))

        # --- per-frame state ---------------------------------------------------
        self._frame_index = 0
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._last_count = -1
        self._prev_gray: Optional[np.ndarray] = None
        self._last_motion = 0.0
        self._status = "watching"

    # ------------------------------------------------------------------ perception
    def _detect_people(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """YOLO person boxes (overridable in tests)."""
        if self._model is None:
            self._model = get_yolo_model(self._weights)
        results = self._model.predict(
            frame, verbose=False, classes=[_PERSON_CLASS], conf=self._confidence_threshold
        )
        boxes = []
        for result in results:
            for box in result.boxes:
                boxes.append(tuple(int(v) for v in box.xyxy[0].tolist()))
        return boxes

    def _motion_energy(self, frame: np.ndarray) -> float:
        """Mean absolute inter-frame difference on a downsampled grayscale (0..1)."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        motion = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray
        return motion

    # ------------------------------------------------------------------ reasoning
    def process(self, frame: np.ndarray) -> list[Event]:
        now = time.monotonic()
        self._frame_index += 1
        if self._frame_index % self._detect_every_n == 0 or self._last_count < 0:
            self._last_boxes = self._detect_people(frame)
        boxes = self._last_boxes
        self._last_motion = self._motion_energy(frame)
        h, w = frame.shape[:2]
        return self._reason(boxes, self._last_motion, w, h, now)

    def _reason(self, boxes: list, motion: float, w: int, h: int, now: float) -> list[Event]:
        """Pure decision layer (unit-testable without camera or model)."""
        events: list[Event] = []
        present = len(boxes) > 0

        # Person count changes (log-only context, mirrors security mode).
        if len(boxes) != self._last_count and self._last_count >= 0:
            events.append(Event.now(
                "person_count_change", 0.0, 0.9, self.name, f"count {self._last_count} -> {len(boxes)}"
            ))
        self._last_count = len(boxes)

        transitions = []
        # Restricted zones: someone's feet inside the zone.
        for zone in self._zones:
            if zone.kind != "restricted":
                continue
            inside = any(zone.contains_box(b, w, h) for b in boxes)
            transitions += self._engine.update(f"zone_violation:{zone.name}", inside, 0.9, now)

        # Safe zones: subject visible but outside every safe zone.
        safe_zones = [z for z in self._zones if z.kind == "safe"]
        if safe_zones:
            outside = present and not any(
                z.contains_box(b, w, h) for z in safe_zones for b in boxes
            )
            transitions += self._engine.update("safe_zone_exit", outside, 0.85, now)

        # Stillness: subject visible with very little motion.
        transitions += self._engine.update(
            "unusual_stillness", present and motion < self._still_max_motion, 0.8, now
        )
        # Activity bursts: large motion while visible (logged, never an alarm).
        transitions += self._engine.update("activity_burst", present and motion > 6 * self._still_max_motion, 0.7, now)

        if self._expect_presence:
            transitions += self._engine.update("person_absent", not present, 0.9, now)

        for tr in transitions:
            base = tr.detection.split(":")[0]
            zone_name = tr.detection.partition(":")[2]
            if tr.state == CONFIRMED:
                events.append(Event.now(
                    base, tr.duration_s, tr.confidence, self.name,
                    (f"zone={zone_name} " if zone_name else "") + f"confirmed after {tr.duration_s:.1f}s",
                ))
            elif tr.state == RESOLVED:
                events.append(Event.now(
                    f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                    f"zone={zone_name}" if zone_name else "",
                ))
            # observed/developing transitions are engine-internal breadcrumbs; the
            # dashboard timeline gets only meaningful events.

        active = self._engine.active_detections()
        self._status = ("ALERT: " + ", ".join(a.split(":")[-1] for a in active)) if active else (
            "watching" if present else "no one in view"
        )
        return events

    # ------------------------------------------------------------------ display
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        for zone in self._zones:
            pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], dtype=np.int32)
            color = (0, 0, 220) if zone.kind == "restricted" else (0, 200, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for (x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(frame, self._status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        return frame

    def is_alert_active(self) -> bool:
        return bool(self._engine.active_detections())

    def get_telemetry_snapshot(self) -> Optional[dict]:
        return {
            "person_count": max(self._last_count, 0),
            "motion_energy": round(self._last_motion, 4),
            "active_alerts": len(self._engine.active_detections()),
        }
