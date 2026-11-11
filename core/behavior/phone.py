"""Phone interaction: from "a phone box exists" to "the driver is using it".

The shipped rule (modules/driver_attention.py, ``phone_use_condition =
phone_visible and eyes_off_road``) treats ANY phone box ANYWHERE in the frame
plus ANY off-road gaze as phone use: a dash-mounted GPS plus a mirror check,
or a passenger's phone plus a glance at the radio, both qualify. No hand is
required and the gaze is never related to where the phone is.

This state machine asks the questions that rule skips, in order, and stops at
the first one it cannot answer -- reporting ``unknown`` with the reason rather
than guessing:

  1. Is the detector producing fresh results?        no  -> unknown
  2. Is a phone persistently present (k of n ticks)? no  -> none
  3. Can the phone be related to the driver (face)?  no  -> unknown
  4. Is it within the driver's reach region?         no  -> visible_only
  5. Is it held (hand covers the PHONE's area) or near the face?
        neither, and the box has been static        -> visible_only (mounted)
        neither, hands unavailable / box moving      -> unknown
  6. Is the gaze directed toward it, sustained?      yes -> in_use
                                                     no  -> held / near_face

Why step 5 degrades to ``unknown`` rather than ``visible_only``: offline, a
hand-overlap gate cut false phones on phone-free images from 56 to 6 when hand
boxes came from human labels, but MediaPipe Hands found a hand in only 8 of
152 web photos. "No hand detected" is therefore weak evidence of "no hand",
so a moving phone with no detected hand is reported as undecidable, not as
safe. A phone that has sat still for ``static_s`` with no hand is the one
case the data supports calling mounted.

All synthetic-scenario tested only; every parameter needs in-cabin validation
(docs/BEHAVIOUR_REASONING.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.behavior.evidence import (
    Assessment,
    Box,
    FrameGapGuard,
    KOfN,
    box_center,
    box_contains_point,
    intersection_over_area,
    unknown,
)

NONE = "none"
VISIBLE_ONLY = "visible_only"
HELD = "held"
NEAR_FACE = "near_face"
IN_USE = "in_use"
PHONE_STATES = (NONE, VISIBLE_ONLY, HELD, NEAR_FACE, IN_USE)

DETECTOR_OK = "ok"
DETECTOR_NOT_STARTED = "not_started"
DETECTOR_FAILED = "failed"


@dataclass(frozen=True)
class PhoneConfig:
    presence_k: int = 3  # phone seen on >= k of the last n detector ticks
    presence_n: int = 5
    hold_k: int = 1  # hand overlap seen on >= hold_k of the last hold_n ticks
    hold_n: int = 3
    hold_overlap: float = 0.3  # fraction of the PHONE box covered by a hand box
    near_face_overlap: float = 0.2  # fraction of the phone box inside the expanded face box
    face_margin_face_widths: float = 0.5  # face box grown by this many face widths per side
    reach_face_widths: float = 2.5  # driver reach region: +- this many face widths around the face
    reach_above_face_widths: float = 0.5  # ... from this far above the face top down to the frame bottom
    gaze_min_deg: float = 10.0  # gaze must deviate at least this much to point anywhere
    gaze_tolerance_deg: float = 45.0  # angle between gaze and face->phone direction
    in_line_face_widths: float = 0.3  # phone centre this close to the face centre: in the line of sight
    # (an ear sits ~0.5 face widths off-centre, so a phone held to the ear is NOT in line)
    use_sustain_s: float = 2.0  # held/near_face AND gaze toward phone, credited seconds
    release_s: float = 0.4  # gaze away this long resets the sustain; in_use ends after this
    static_s: float = 2.0  # phone box still this long with no hand -> treated as mounted
    static_tolerance: float = 0.10  # "still": centre moved less than this x the box diagonal
    max_detection_age_s: float = 1.0  # detector result older than this -> unknown
    max_gap_s: float = 0.5  # frame gap above this credits no time


@dataclass(frozen=True)
class PhoneDetection:
    """One fresh detector result (a tick). Boxes are pixels (x1, y1, x2, y2)."""

    t: float
    phone_boxes: tuple[Box, ...] = ()
    hand_boxes: tuple[Box, ...] = ()
    hands_available: bool = True  # the hand model ran on this tick
    phone_score: Optional[float] = None


@dataclass(frozen=True)
class PhoneFrame:
    """Everything the state machine needs for one video frame."""

    t: float
    detector: str = DETECTOR_OK
    detector_reason: str = ""
    detection: Optional[PhoneDetection] = None  # only on a frame where the detector produced a result
    face_box: Optional[Box] = None
    gaze: Optional[tuple[float, float]] = None  # calibrated (yaw, pitch) deg; +yaw frame-right, -pitch down
    driver_region: Optional[Box] = None  # optional fixed region (pixels) the driver's phone must be in


@dataclass(frozen=True)
class PhoneUpdate:
    assessment: Assessment
    use_started: bool = False  # rising edge of an in_use episode -> "phone_use"
    presence_started: bool = False  # rising edge of persistent presence -> "phone_detected"
    toward_s: float = 0.0


def expand_box(box: Box, margin: float) -> Box:
    return (box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin)


def gaze_toward(
    gaze: tuple[float, float], phone: Box, face: Box, config: PhoneConfig
) -> bool:
    """Is the calibrated gaze directed at the phone, as seen in the image plane?

    The gaze (yaw, pitch) maps to the image-plane direction (yaw, -pitch):
    +yaw turns the nose toward the frame's right and -pitch is down, i.e. +y
    in the image. The phone's direction is its centre relative to the face
    centre. Only the DIRECTION is compared -- without camera-to-cabin geometry
    the angle-to-distance mapping is unknown, so magnitude is not claimed.
    """
    face_w = max(face[2] - face[0], 1e-6)
    fcx, fcy = box_center(face)
    pcx, pcy = box_center(phone)
    dx, dy = (pcx - fcx) / face_w, (pcy - fcy) / face_w
    gx, gy = gaze[0], -gaze[1]
    g_mag = math.hypot(gx, gy)
    if math.hypot(dx, dy) <= config.in_line_face_widths:
        # The phone is in front of the face: looking "at the road" through it
        # is looking at it.
        return g_mag < config.gaze_min_deg
    if g_mag < config.gaze_min_deg:
        return False
    cos = (gx * dx + gy * dy) / (g_mag * math.hypot(dx, dy))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) <= config.gaze_tolerance_deg


class PhoneInteraction:
    """none / visible_only / held / near_face / in_use, or unknown with a reason."""

    def __init__(self, config: Optional[PhoneConfig] = None) -> None:
        self.config = config or PhoneConfig()
        c = self.config
        self._gap = FrameGapGuard(c.max_gap_s)
        self._presence = KOfN(c.presence_k, c.presence_n)
        self._hold = KOfN(c.hold_k, c.hold_n)
        self._boxes: tuple[Box, ...] = ()
        self._hands_available = False
        self._last_detection_t: Optional[float] = None
        self._static_anchor: Optional[tuple[float, float]] = None
        self._static_since: Optional[float] = None
        self._toward_s = 0.0
        self._away_since: Optional[float] = None
        self._in_use_episode = False
        self._last_in_use_t: Optional[float] = None
        self._was_present = False
        self.assessment: Assessment = unknown("no frame yet")
        self.phone_score: Optional[float] = None

    # ------------------------------------------------------------- helpers
    def _reset_detection_state(self) -> None:
        self._presence.reset()
        self._hold.reset()
        self._boxes = ()
        self._static_anchor = None
        self._static_since = None
        self._was_present = False

    def _reset_sustain(self) -> None:
        self._toward_s = 0.0
        self._away_since = None

    def _primary_box(self, face_box: Optional[Box]) -> Optional[Box]:
        if not self._boxes:
            return None
        if face_box is None:
            return max(self._boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        fc = box_center(face_box)
        return min(self._boxes, key=lambda b: math.dist(box_center(b), fc))

    def _ingest(self, det: PhoneDetection, face_box: Optional[Box]) -> None:
        self._last_detection_t = det.t
        self._presence.update(bool(det.phone_boxes))
        if det.phone_boxes:
            self._boxes = tuple(det.phone_boxes)
            self.phone_score = det.phone_score
        elif not self._presence.present:
            self._boxes = ()
        self._hands_available = det.hands_available
        primary = self._primary_box(face_box)
        if det.hands_available and primary is not None and det.phone_boxes:
            self._hold.update(any(
                intersection_over_area(primary, hand) >= self.config.hold_overlap for hand in det.hand_boxes
            ))
        elif not det.hands_available:
            self._hold.reset()
        # Static tracking, anchored where stillness began (slow drift accumulates).
        if primary is not None and det.phone_boxes:
            centre = box_center(primary)
            diag = math.hypot(primary[2] - primary[0], primary[3] - primary[1])
            if (self._static_anchor is None
                    or math.dist(centre, self._static_anchor) > self.config.static_tolerance * diag):
                self._static_anchor = centre
                self._static_since = det.t

    def _in_reach(self, phone: Box, face: Box, region: Optional[Box]) -> bool:
        c = self.config
        fw = face[2] - face[0]
        fcx = (face[0] + face[2]) / 2.0
        reach = (fcx - c.reach_face_widths * fw, face[1] - c.reach_above_face_widths * fw,
                 fcx + c.reach_face_widths * fw, math.inf)
        centre = box_center(phone)
        if not box_contains_point(reach, centre):
            return False
        return region is None or box_contains_point(region, centre)

    # ------------------------------------------------------------- update
    def update(self, frame: PhoneFrame) -> PhoneUpdate:
        c = self.config
        dt, _stale = self._gap.step(frame.t)

        if frame.detector != DETECTOR_OK:
            self._reset_detection_state()
            self._reset_sustain()
            reason = frame.detector_reason or (
                "object detector not started" if frame.detector == DETECTOR_NOT_STARTED
                else "object detector failed")
            return self._finish(frame.t, unknown(reason))

        if frame.detection is not None:
            self._ingest(frame.detection, frame.face_box)

        if self._last_detection_t is None:
            self._reset_sustain()
            return self._finish(frame.t, unknown("no object-detector result yet"))
        age = frame.t - self._last_detection_t
        if age > c.max_detection_age_s:
            self._reset_sustain()
            return self._finish(frame.t, unknown(f"object-detector result stale ({age:.1f}s old)"))

        if not self._presence.present:
            self._reset_sustain()
            return self._finish(frame.t, Assessment(NONE))
        presence_started = not self._was_present
        self._was_present = True

        face = frame.face_box
        phone = self._primary_box(face)
        if phone is None:  # present by persistence but no box remembered: cannot place it
            self._reset_sustain()
            return self._finish(frame.t, unknown("phone persistent but not localised"), presence_started)
        if face is None:
            self._reset_sustain()
            if frame.driver_region is not None and not box_contains_point(frame.driver_region, box_center(phone)):
                return self._finish(frame.t, Assessment(VISIBLE_ONLY, "outside driver region"), presence_started)
            return self._finish(frame.t, unknown("face not visible: cannot relate phone to driver"),
                                presence_started)
        if not self._in_reach(phone, face, frame.driver_region):
            self._reset_sustain()
            return self._finish(frame.t, Assessment(VISIBLE_ONLY, "outside driver reach region"),
                                presence_started)

        fw = face[2] - face[0]
        near_face = intersection_over_area(
            phone, expand_box(face, c.face_margin_face_widths * fw)) >= c.near_face_overlap
        held = self._hands_available and self._hold.present
        if not (held or near_face):
            self._reset_sustain()
            static = self._static_since is not None and frame.t - self._static_since >= c.static_s
            if static:
                return self._finish(frame.t, Assessment(VISIBLE_ONLY, "static, no hand (mounted?)"),
                                    presence_started)
            if not self._hands_available:
                return self._finish(frame.t, unknown("hand model unavailable: cannot tell held from mounted"),
                                    presence_started)
            return self._finish(frame.t, unknown(
                "phone moving but no hand detected (hand-detector recall is low)"), presence_started)

        relation = HELD if held else NEAR_FACE
        if frame.gaze is None:
            # Holding is known; whether it is being looked at is not. No accrual.
            self._away_since = None
            return self._finish(frame.t, Assessment(relation, "gaze unknown"), presence_started)
        if gaze_toward(frame.gaze, phone, face, c):
            self._toward_s += dt
            self._away_since = None
        else:
            if self._away_since is None:
                self._away_since = frame.t
            if frame.t - self._away_since > c.release_s:
                self._toward_s = 0.0
        if self._toward_s >= c.use_sustain_s:
            use_started = not self._in_use_episode
            self._in_use_episode = True
            self._last_in_use_t = frame.t
            detail = "hand-held" if held else "near face"
            return self._finish(frame.t, Assessment(IN_USE, f"{detail}, gaze toward phone"),
                                presence_started, use_started)
        return self._finish(frame.t, Assessment(relation), presence_started)

    def _finish(self, t: float, assessment: Assessment, presence_started: bool = False,
                use_started: bool = False) -> PhoneUpdate:
        if assessment.state != IN_USE and self._in_use_episode and (
                self._last_in_use_t is None or t - self._last_in_use_t > self.config.release_s):
            self._in_use_episode = False
        if assessment.state == NONE:
            self._was_present = False
        self.assessment = assessment
        return PhoneUpdate(assessment, use_started=use_started, presence_started=presence_started,
                           toward_s=self._toward_s)

    @property
    def in_use_active(self) -> bool:
        """True through an in_use episode, including its release grace."""
        return self._in_use_episode

    @property
    def toward_s(self) -> float:
        return self._toward_s
