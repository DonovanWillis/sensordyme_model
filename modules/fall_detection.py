"""Fall detection module: MediaPipe Pose hip/shoulder tracking for rapid-drop-then-stillness."""

from __future__ import annotations

import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core import posture
from core.detections import Detection, DetectionFrame
from core.events import Event
from modules.base import VisionModule, register_module

HIP_LANDMARKS = [23, 24]
SHOULDER_LANDMARKS = [11, 12]


@register_module("fall_detection")
class FallDetectionModule(VisionModule):
    """Fires a fall event on a rapid vertical drop of hip/shoulder height followed by stillness."""

    name = "fall_detection"

    def __init__(self, config: dict) -> None:
        """Read thresholds from config and initialize MediaPipe Pose."""
        super().__init__(config)
        self._drop_velocity_threshold = float(config["drop_velocity_threshold"])
        self._stillness_duration_s = float(config["stillness_duration_s"])
        self._stillness_velocity_threshold = float(config["stillness_velocity_threshold"])

        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self._prev_y: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._fall_pending_since: Optional[float] = None
        self._still_since: Optional[float] = None
        self._fired_for_current_fall = False
        self._last_pose_proto = None
        self._alert_active = False
        self._last_posture: Optional[posture.Posture] = None
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._last_hip_y: Optional[float] = None

    def process(self, frame: np.ndarray) -> list[Event]:
        """Run pose estimation on `frame`, updating drop/stillness state and returning events."""
        now = time.monotonic()
        events: list[Event] = []
        h, _w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            self._last_pose_proto = None
            self._last_posture = None
            self._last_hip_y = None
            self._prev_y = None
            self._prev_t = None
            return events

        self._last_pose_proto = results.pose_landmarks
        landmarks = results.pose_landmarks.landmark

        # Posture from the landmarks this module was already computing, so
        # custom rules ("someone on the floor") cost no extra inference.
        frame_h, frame_w = frame.shape[:2]
        self._last_frame_size = (frame_w, frame_h)
        self._last_posture = posture.classify(landmarks, aspect=frame_w / max(frame_h, 1))
        self._last_hip_y = posture.hip_height(landmarks)
        indices = HIP_LANDMARKS + SHOULDER_LANDMARKS
        avg_y = sum(landmarks[i].y for i in indices) / len(indices) * h

        velocity = 0.0
        if self._prev_y is not None and self._prev_t is not None and now > self._prev_t:
            velocity = (avg_y - self._prev_y) / (now - self._prev_t)

        if velocity > self._drop_velocity_threshold * h:
            self._fall_pending_since = now
            self._still_since = None

        if self._fall_pending_since is not None:
            if abs(velocity) < self._stillness_velocity_threshold * h:
                if self._still_since is None:
                    self._still_since = now
                elif now - self._still_since >= self._stillness_duration_s and not self._fired_for_current_fall:
                    events.append(Event.now("fall", now - self._fall_pending_since, 0.8, self.name))
                    self._fired_for_current_fall = True
                    self._alert_active = True
            else:
                self._still_since = None

        if self._fired_for_current_fall and velocity > self._stillness_velocity_threshold * h * 3:
            self._fall_pending_since = None
            self._fired_for_current_fall = False
            self._alert_active = False

        self._prev_y = avg_y
        self._prev_t = now
        return events

    def get_detections(self) -> DetectionFrame:
        """The tracked person plus their posture, as labels custom rules can use.

        Posture is emitted as its own label ("lying_down", "on_floor") alongside
        "person", so a rule can ask for a posture the same way it asks for an
        object — the rule engine needs no special case, and a future posture
        model can replace this one without the rule layer noticing.

        No bounding box: MediaPipe Pose gives landmarks, not a detection box, and
        inventing one from landmark extents would let zone rules silently key off
        a rectangle that means something different from every other box here.
        """
        width, height = self._last_frame_size
        detections: list[Detection] = []
        if self._last_posture is not None:
            detections.append(Detection(label="person", confidence=0.9))
            if self._last_posture.label != posture.UNKNOWN:
                detections.append(
                    Detection(label=self._last_posture.label, confidence=self._last_posture.confidence)
                )
            if posture.is_on_floor(self._last_posture, self._last_hip_y):
                detections.append(Detection(label="on_floor", confidence=self._last_posture.confidence))
        return DetectionFrame(width=width, height=height, detections=tuple(detections))

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pose skeleton on `frame`."""
        if self._last_pose_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self._last_pose_proto, mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame

    def is_alert_active(self) -> bool:
        """True from the moment a fall is confirmed until the person moves again."""
        return self._alert_active
