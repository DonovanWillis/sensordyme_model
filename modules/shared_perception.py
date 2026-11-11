"""shared_perception: the module a node-backed mode runs on.

Modes such as Occupancy and Workplace Safety have no model of their own: their
monitoring nodes (core/nodes.py) read the shared detector's output from each
frame's Observation (core/perception.py). Something still has to own the frame
loop's module slot, give custom rules a per-frame view, and keep the detector
running at a steady cadence; this is that module, and nothing more. It raises
no events of its own — every alert comes from a node or a rule.

It asks the shared detector (core/detectors.get_yolo_model, which the runtime's
perception engine wraps) for person plus whatever the active rules need, every
`detect_every_n_frames` frames. A node that asks for the detector on the same
frame is answered from the same inference.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from core.detections import Detection, DetectionFrame
from core.detectors import get_yolo_model
from core.events import Event
from core.monitoring_state import IMPAIRED, STARTING, Capability
from modules.base import VisionModule, register_module

_RETRY_INITIAL_S = 1.0
_RETRY_MAX_S = 30.0


@register_module("shared_perception")
class SharedPerceptionModule(VisionModule):
    """Runs the shared detector on a cadence; the monitoring nodes do the rest."""

    name = "shared_perception"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._weights = str(config.get("yolo_weights") or "")
        self._every = max(1, int(config.get("detect_every_n_frames", 2)))
        self._confidence = float(config.get("confidence_threshold", 0.35))
        self._requested: set[str] = {"person"}
        self._yolo_model = None
        self._class_ids: dict[str, int] = {}
        self._frame_index = 0
        self._boxes: list[tuple[str, float, tuple[float, float, float, float]]] = []
        self._size = (0, 0)
        self._error: Optional[str] = None
        self._failures = 0
        self._retry_at: Optional[float] = None

    def set_requested_labels(self, labels: set[str]) -> None:
        self._requested = {"person", *(str(label) for label in labels or ())}

    def _ensure_detector(self) -> None:
        if self._yolo_model is None:
            if not self._weights:
                raise RuntimeError("modules.shared_perception.yolo_weights is not set")
            self._yolo_model = get_yolo_model(self._weights)
        names = getattr(self._yolo_model, "names", {}) or {}
        self._class_ids = {str(n): int(i) for i, n in names.items()}

    def process(self, frame: np.ndarray) -> list[Event]:
        h, w = frame.shape[:2]
        self._size = (w, h)
        self._frame_index += 1
        if self._frame_index % self._every:
            return []
        wall = time.monotonic()
        if self._retry_at is not None and wall < self._retry_at:
            return []
        try:
            self._ensure_detector()
            classes = sorted(self._class_ids[label] for label in self._requested if label in self._class_ids)
            results = self._yolo_model.predict(frame, verbose=False, classes=classes or None,
                                               conf=self._confidence)
        except Exception as exc:  # noqa: BLE001 - reported as IMPAIRED and retried with backoff
            self._failures += 1
            self._error = str(exc) or type(exc).__name__
            self._retry_at = wall + min(_RETRY_INITIAL_S * 2 ** (self._failures - 1), _RETRY_MAX_S)
            self._boxes = []  # never carry a box past a failure
            return []
        self._error, self._failures, self._retry_at = None, 0, None
        names = getattr(self._yolo_model, "names", {}) or {}
        boxes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                xyxy = box.xyxy[0]
                values = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
                boxes.append((str(names.get(class_id, class_id)), float(box.conf[0]),
                              tuple(float(v) for v in values[:4])))
        self._boxes = boxes
        return []

    def get_detections(self) -> DetectionFrame:
        width, height = self._size
        return DetectionFrame(width=width, height=height, detections=tuple(
            Detection(label=label, confidence=conf, box=box) for label, conf, box in self._boxes))

    def available_labels(self) -> Optional[frozenset]:
        return frozenset(self._class_ids) if self._class_ids else None

    def capability_status(self) -> dict:
        if self._error is not None:
            return {"objects": Capability(IMPAIRED, f"object detection failed {self._failures}x ({self._error})")}
        if self._yolo_model is None:
            return {"objects": Capability(STARTING, "object detector loads on the first detection frame")}
        return {"objects": Capability()}

    def event_requirements(self) -> dict[str, frozenset[str]]:
        return {}

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        for label, conf, (x1, y1, x2, y2) in self._boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        return frame
