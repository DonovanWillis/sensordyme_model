"""BehaviorReasoner: the components composed, one call per video frame.

The module integration (modules/driver_attention.py, behind ``behavior_v2``)
only translates its per-frame measurements into a ``BehaviorFrame`` and the
returned ``BehaviorEvent``s into Events. Everything that decides lives here
and in the components, so all of it is testable without MediaPipe or a
detector.

Monitoring uncertainty: ``uncertainty()`` names every component currently
``unknown`` with its reason. A frame with missing inputs is reported, never
silently scored as safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.behavior.config import BehaviorConfig
from core.behavior.evidence import Assessment, Box, unknown
from core.behavior.eye_state import DrowsinessEvidence, EyeFrame, classify_eye_state
from core.behavior.glance import GlanceAnalyzer
from core.behavior.phone import DETECTOR_OK, IN_USE, PhoneDetection, PhoneFrame, PhoneInteraction

DROWSINESS = "drowsiness"
PHONE_USE = "phone_use"
PHONE_DETECTED = "phone_detected"


@dataclass(frozen=True)
class BehaviorFrame:
    t: float  # frame capture time (NOT evidence time: gaps must be visible here)
    face_present: bool
    face_box: Optional[Box] = None
    gaze: Optional[tuple[float, float]] = None  # calibrated fused (yaw, pitch)
    gaze_unknown_reason: str = "gaze unknown"
    eye: EyeFrame = field(default_factory=lambda: EyeFrame(face_present=False))
    detector: str = DETECTOR_OK
    detector_reason: str = ""
    detection: Optional[PhoneDetection] = None
    driver_region: Optional[Box] = None  # pixels


@dataclass(frozen=True)
class BehaviorEvent:
    type: str
    duration_s: float
    detail: str


class BehaviorReasoner:
    def __init__(self, config: Optional[BehaviorConfig] = None) -> None:
        self.config = config or BehaviorConfig()
        c = self.config
        self.phone = PhoneInteraction(c.phone)
        self.glance = GlanceAnalyzer(c.glance)
        self.drowsiness = DrowsinessEvidence(
            sustain_s=c.drowsy_seconds, release_frames=c.eye_release_frames,
            coast_s=c.face_loss_coast_s, max_gap_s=c.max_frame_gap_s,
        )
        self.eye_state: Assessment = unknown("no frame yet")
        self.eyes_occluded = 0

    def update(self, frame: BehaviorFrame) -> list[BehaviorEvent]:
        events: list[BehaviorEvent] = []

        self.eye_state = classify_eye_state(frame.eye, self.config.eye)
        self.eyes_occluded = sum(bool(e.occluded) for e in frame.eye.eyes)
        fired = self.drowsiness.update(frame.t, self.eye_state)
        if fired is not None:
            events.append(BehaviorEvent(DROWSINESS, fired, f"eye closure {fired:.1f}s ({self.eye_state.reason})"))

        gaze_reason = frame.gaze_unknown_reason if frame.face_present else "face not detected"
        for g in self.glance.update(frame.t, frame.gaze, gaze_reason):
            events.append(BehaviorEvent(g.type, g.duration_s, g.detail))

        phone = self.phone.update(PhoneFrame(
            t=frame.t, detector=frame.detector, detector_reason=frame.detector_reason,
            detection=frame.detection, face_box=frame.face_box if frame.face_present else None,
            gaze=frame.gaze, driver_region=frame.driver_region,
        ))
        if phone.use_started:
            events.append(BehaviorEvent(PHONE_USE, phone.toward_s, phone.assessment.reason))
        if phone.presence_started and phone.assessment.state != IN_USE:
            events.append(BehaviorEvent(PHONE_DETECTED, 0.0, phone.assessment.reason or phone.assessment.state))
        return events

    def assessments(self) -> dict[str, Assessment]:
        return {
            "eye_state": self.eye_state,
            "drowsiness": self.drowsiness.assessment,
            "glance": self.glance.assessment,
            "phone": self.phone.assessment,
        }

    def uncertainty(self) -> dict[str, str]:
        """Every component that cannot currently make a claim, with why."""
        return {name: a.reason for name, a in self.assessments().items() if not a.known}

    def snapshot(self) -> dict:
        """Flat per-frame fields for telemetry/debug (prefixed, CSV-friendly)."""
        return {
            "behavior_eye_state": self.eye_state.state,
            "behavior_eyes_occluded": self.eyes_occluded,
            "behavior_drowsy_state": self.drowsiness.assessment.state,
            "behavior_drowsy_closed_s": round(self.drowsiness.closed_s, 2),
            "behavior_gaze_zone": self.glance.assessment.state,
            "behavior_glance_s": round(self.glance.current_glance_s, 2),
            "behavior_offroad_budget_s": round(self.glance.budget_used_s, 2),
            "behavior_glance_count": self.glance.repeat_count,
            "behavior_phone_state": self.phone.assessment.state,
            "behavior_phone_toward_s": round(self.phone.toward_s, 2),
            "behavior_uncertain": ";".join(sorted(self.uncertainty())),
        }
