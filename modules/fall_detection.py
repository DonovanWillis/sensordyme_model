"""Fall detection module: MediaPipe Pose hip/shoulder tracking for rapid-drop-then-stillness.

Also the elderly-mode zone and stillness detector, on the same landmarks: a
restricted/boundary zone ("zone_violation", same event name occupancy.py uses,
so the existing alert-severity wiring needs no change) tested against the
ankle midpoint rather than a fabricated bounding box (see get_detections'
docstring on why this module deliberately never invents one), and
"unusual_stillness" from how little the hip/shoulder centroid has moved --
reusing the same landmarks the fall detector already reads, no extra model.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core import posture
from core.activity import ActivityRecognizer
from core.detections import Detection, DetectionFrame
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.zones import load_zones
from modules.base import VisionModule, register_module

HIP_LANDMARKS = [23, 24]
SHOULDER_LANDMARKS = [11, 12]
ANKLE_LANDMARKS = [27, 28]
_MIN_ANKLE_VISIBILITY = 0.5


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
        self._activity = ActivityRecognizer()
        self._last_activity = None

        # Activity events go through the shared lifecycle, which is what turns
        # "running" into ONE alert with a start and an end rather than one per
        # frame for as long as someone keeps running.
        self._activity_engine = EventEngine({})
        self._activity_engine.add_rule(RuleSpec(
            detection="running", severity="warning",
            persist_s=float(config.get("running_confirm_s", 1.0)),
            min_frames=5, resolve_after_s=2.0, cooldown_s=15.0,
        ))
        self._activity_engine.add_rule(RuleSpec(
            detection="jumping", severity="info",
            persist_s=float(config.get("jump_confirm_s", 0.2)),
            min_frames=2, resolve_after_s=1.5, cooldown_s=5.0,
        ))
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._last_hip_y: Optional[float] = None

        # Elderly-mode zones (restricted/boundary only -- "safe" zones are
        # baby_monitoring's concern) and standalone stillness, both on the
        # landmarks already being read. Separate engine from _activity_engine
        # so a fall-adjacent bug in one can't cross-contaminate the other's
        # per-detection state.
        self._zones = load_zones(config.get("zones", {}))
        self._watched_zones = [z for z in self._zones if z.kind in ("restricted", "boundary")]
        self._zone_engine = EventEngine({})
        zone_confirm_s = float(config.get("zone_confirm_s", 2.0))
        for zone in self._watched_zones:
            self._zone_engine.add_rule(RuleSpec(
                detection=f"zone_violation:{zone.name}", severity="critical",
                persist_s=zone_confirm_s, min_frames=3, resolve_after_s=5.0,
            ))
        self._still_max_motion = float(config.get("still_max_motion", 0.01))
        self._zone_engine.add_rule(RuleSpec(
            detection="unusual_stillness", severity="warning",
            persist_s=float(config.get("unusual_stillness_s", 600.0)),
            min_frames=10, occlusion_grace_s=3.0, resolve_after_s=10.0,
        ))
        self._prev_centroid: Optional[tuple[float, float]] = None

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
            # The subject left: motion history describes someone who is no
            # longer there and would misclassify whoever arrives next.
            self._activity.reset()
            self._last_activity = None
            self._last_hip_y = None
            self._prev_y = None
            self._prev_t = None
            self._prev_centroid = None
            # Gone, not merely still, and gone, not merely outside the zone --
            # both are real absence, so both get an explicit False rather than
            # being left to freeze mid-confirmation.
            transitions = []
            for zone in self._watched_zones:
                transitions += self._zone_engine.update(f"zone_violation:{zone.name}", False, 0.85, now)
            transitions += self._zone_engine.update("unusual_stillness", False, 0.8, now)
            events += self._zone_transition_events(transitions)
            return events

        self._last_pose_proto = results.pose_landmarks
        landmarks = results.pose_landmarks.landmark

        # Posture from the landmarks this module was already computing, so
        # custom rules ("someone on the floor") cost no extra inference.
        frame_h, frame_w = frame.shape[:2]
        self._last_frame_size = (frame_w, frame_h)
        aspect = frame_w / max(frame_h, 1)
        self._last_posture = posture.classify(landmarks, aspect=aspect)
        self._last_hip_y = posture.hip_height(landmarks)

        # Activity from the same landmarks — no extra inference.
        self._last_activity = self._activity.update(landmarks, now=now, aspect=aspect)
        events += self._activity_events(now)
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

        zone_transitions = []
        # Ankle midpoint, not a fabricated box: zones.py's own point-in-polygon
        # primitive, on the one point "is this person in this area" actually
        # means for someone who might be lying down as easily as standing.
        # Occluded/invisible ankles (legs behind furniture, off the bottom of
        # frame) skip zone evaluation for this frame rather than guessing --
        # "can't see the feet" is not evidence the zone was left.
        if self._watched_zones:
            ankle_vis = [landmarks[i].visibility for i in ANKLE_LANDMARKS]
            if all(v >= _MIN_ANKLE_VISIBILITY for v in ankle_vis):
                foot_x = sum(landmarks[i].x for i in ANKLE_LANDMARKS) / len(ANKLE_LANDMARKS)
                foot_y = sum(landmarks[i].y for i in ANKLE_LANDMARKS) / len(ANKLE_LANDMARKS)
                for zone in self._watched_zones:
                    inside = zone.contains(foot_x, foot_y)
                    zone_transitions += self._zone_engine.update(
                        f"zone_violation:{zone.name}", inside, 0.85, now
                    )

        # Stillness from how little the hip/shoulder centroid has moved, in
        # normalized landmark coordinates (so the threshold means the same
        # thing at any capture resolution) -- independent of the vertical-only
        # velocity above, which exists to catch a FAST drop, not a slow lack
        # of movement over minutes.
        centroid = (
            sum(landmarks[i].x for i in indices) / len(indices),
            sum(landmarks[i].y for i in indices) / len(indices),
        )
        motion = 0.0
        if self._prev_centroid is not None:
            motion = math.hypot(centroid[0] - self._prev_centroid[0], centroid[1] - self._prev_centroid[1])
        self._prev_centroid = centroid
        zone_transitions += self._zone_engine.update(
            "unusual_stillness", motion < self._still_max_motion, 0.8, now
        )
        events += self._zone_transition_events(zone_transitions)

        return events

    def _zone_transition_events(self, transitions: list) -> list[Event]:
        """CONFIRMED/RESOLVED transitions -> Events, for both the zone and
        stillness engine and the person-gone early-return, so the translation
        (and the event-type naming) is written once."""
        events: list[Event] = []
        for tr in transitions:
            base, _, zone_name = tr.detection.partition(":")
            detail = f"zone={zone_name} confirmed after {tr.duration_s:.1f}s" if zone_name else ""
            if tr.state == CONFIRMED:
                events.append(Event.now(base, tr.duration_s, tr.confidence, self.name, detail))
            elif tr.state == RESOLVED:
                events.append(Event.now(
                    f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                    f"zone={zone_name}" if zone_name else "",
                ))
        return events

    def _activity_events(self, now: float) -> list[Event]:
        """Turn the activity state into lifecycle-managed events.

        Every frame feeds the engine, but only CONFIRMED and RESOLVED
        transitions become events — which is what makes thirty seconds of
        running one alert with a start and an end, not thirty alerts.
        """
        state = self._last_activity
        if state is None or not state.ready:
            return []

        events: list[Event] = []
        evidence = ", ".join(f"{k}={v}" for k, v in state.signals.items() if k != "reason")

        for attribute in (state.running, state.jumping):
            transitions = self._activity_engine.update(
                attribute.name, attribute.present, attribute.confidence or 0.0, now
            )
            for transition in transitions:
                if transition.state == CONFIRMED:
                    events.append(Event.now(
                        transition.detection, transition.duration_s,
                        transition.confidence, self.name, evidence,
                    ))
                elif transition.state == RESOLVED:
                    events.append(Event.now(
                        f"{transition.detection}_ended", transition.duration_s,
                        transition.confidence, self.name, evidence,
                    ))
        return events

    def get_activity(self) -> Optional[dict]:
        """Every attribute believed about the subject right now, independently.

        Multi-attribute by design: a person can be running and upright and not
        jumping all at once, and collapsing that to one label throws away the
        combinations that matter.
        """
        return self._last_activity.as_dict() if self._last_activity else None

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

        # Activities are labels too, so "alert me if someone runs in the hallway"
        # is expressible with the rule engine exactly as any object rule is.
        if self._last_activity is not None and self._last_activity.ready:
            for attribute in self._last_activity.active():
                if attribute.name not in {d.label for d in detections}:
                    detections.append(Detection(label=attribute.name, confidence=attribute.confidence))
        return DetectionFrame(width=width, height=height, detections=tuple(detections))

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pose skeleton on `frame`."""
        if self._last_pose_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self._last_pose_proto, mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame

    def is_alert_active(self) -> bool:
        """True while a fall, a zone violation, or unusual stillness is confirmed."""
        return self._alert_active or bool(self._zone_engine.active_detections())
