"""Owner decision B (M8.4): distinct phone evidence states; a visible phone is not risk.

The shipped risk fusion credited ``phone_risk = 0.3`` for any phone box inside
the driver's person box -- a phone in a dash mount, a phone lying on the seat,
the passenger's phone when the passenger is the larger box -- with the eyes on
the road, ``0.5`` with the face lost, and ``0.6`` for a hand overlap whatever
the eyes were doing. So a merely visible phone raised the score on its own.

Here the evidence is classified into ordered, explicit levels, and only the
top three -- a phone HELD by the driver, combined with attention evidence --
may contribute to risk:

  insufficient_evidence        the detector failed / is stale / not started, or a
                               phone cannot be related to anyone (no person box)
  no_phone                     the detector ran and saw none
  phone_visible                seen, but not inside the driver's region
  phone_mounted                driver's region, still for phone_static_s, hands seen
                               elsewhere, none on it (PhoneHoldEvidence NOT_HELD)
  phone_near_driver            driver's region; holding unknown (no hand found, hand
                               model not run, or the phone moves without a hand)
  phone_held                   a hand box overlaps the phone (recent detector tick)
  possible_interaction         held AND the gaze is off the road toward the phone
                               (or pitched down at it when the direction cannot be judged)
  active_distraction_evidence  held/interaction AND eyes off the road for at least
                               phone_use_seconds of credited time

``phone_risk_for`` maps a level plus attention evidence to the fused risk term:
0 for every level below ``phone_held``; ``phone_held`` only counts with the eyes
off the road this frame. "No hand found" never lowers a level below
``phone_near_driver`` -- it is weak evidence (MediaPipe Hands misses most
gripping hands, docs/research/M8_1_CV_FOLLOWUP.md section 3) -- and never raises
one either.

M8.5 (production phone_use; docs/BEHAVIOR_ENGINE.md "M8.5 phone handling vs
off-road attention"): this module is also the single source of truth for the
PRODUCTION phone_use gate. ``supports_phone_use`` says which levels may make a
sustained eyes-off-road episode ``phone_use``: ``phone_held`` and above, or
``phone_near_driver`` WITH a handling cue. A phone whose holding is unobserved
(hold "unknown": no hand found, hand model off, or the not-held latch not yet
formed) is judged by ``handling_cue``: has its box moved within the last
``STILL_S`` (``PhonePlacement``); is it at face height beside the face; or is it
below the chin, inside the face's horizontal span, with the gaze pitched down
(the lap)? Without a cue the phone is treated as not handled (mount-like): the
look is still judged by eyes_down / distraction, never by phone_use. The cues
are image-geometry and timing heuristics: UNVALIDATED, synthetic-scenario
tested only.

Synthetic-scenario tested only; the levels' boundaries need in-cabin data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.behavior.evidence import Box, box_center

INSUFFICIENT = "insufficient_evidence"
NO_PHONE = "no_phone"
VISIBLE = "phone_visible"
MOUNTED = "phone_mounted"
NEAR_DRIVER = "phone_near_driver"
HELD = "phone_held"
INTERACTION = "possible_interaction"
ACTIVE = "active_distraction_evidence"

#: Ordered by how much the evidence says about the DRIVER using a phone.
#: INSUFFICIENT is not on the scale: it is the absence of an answer.
LEVELS: tuple[str, ...] = (NO_PHONE, VISIBLE, MOUNTED, NEAR_DRIVER, HELD, INTERACTION, ACTIVE)
RISK_LEVELS = frozenset({HELD, INTERACTION, ACTIVE})

#: Hold states as modules/driver_attention.PhoneHoldEvidence reports them.
HOLD_NONE, HOLD_HELD, HOLD_NOT_HELD, HOLD_UNKNOWN = "none", "held", "not_held", "unknown"

#: M8.5 handling cues for a phone in the driver's region whose holding is not
#: observed. Each is positive evidence the phone may be in someone's hand; none
#: of them is a hand. UNVALIDATED (synthetic scenarios only).
CUE_MOVING = "phone_moving"  # its box moved > STILL_TOLERANCE x diagonal within the last STILL_S
CUE_AT_FACE = "phone_at_face"  # centre at face height, at most AT_FACE_MARGIN_FW face widths beside the face
CUE_LAP = "phone_in_lap_zone"  # centre below the chin, inside the face's x-span, gaze pitched down
HANDLING_CUES: tuple[str, ...] = (CUE_MOVING, CUE_AT_FACE, CUE_LAP)
_CUE_TEXT = {
    CUE_MOVING: "the phone moved within the last 2 s",
    CUE_AT_FACE: "the phone is at the face",
    CUE_LAP: "the phone is in the lap zone below the face with the gaze down",
}

#: Constants of the M8.5 rule, all reused rather than new: STILL_S and
#: STILL_TOLERANCE are PhoneHoldEvidence's static_s / static_tolerance (config
#: phone_static_s 2.0 s; 10% of the phone box diagonal), so "moving" is exactly
#: "not still" in the not-held latch's sense; AT_FACE_MARGIN_FW is
#: core/behavior/phone.PhoneConfig.face_margin_face_widths.
STILL_S = 2.0
STILL_TOLERANCE = 0.10
AT_FACE_MARGIN_FW = 0.5


class PhonePlacement:
    """Has the driver's phone moved recently? Fed on object-detector ticks only.

    The anchor is where the phone was first seen or last came to rest. A tick
    whose phone centre is more than ``tolerance`` x the box diagonal from the
    anchor is a MOVE (the anchor follows it); the first sighting is not a move.
    A phone that disappears is remembered for ``memory_s`` of evidence time, so
    detector flicker neither invents a move (it reappears in place) nor hides
    one (it reappears elsewhere). ``moving(t)`` is True for ``still_s`` after the
    last move. Deterministic and bounded: one anchor and two timestamps.
    """

    def __init__(self, still_s: float = STILL_S, tolerance: float = STILL_TOLERANCE,
                 memory_s: Optional[float] = None) -> None:
        self.still_s = float(still_s)
        self.tolerance = float(tolerance)
        self.memory_s = float(still_s if memory_s is None else memory_s)
        self._anchor: Optional[tuple[float, float]] = None
        self._last_seen: Optional[float] = None
        self._moved_at: Optional[float] = None
        self.moves = 0

    def reset(self) -> None:
        self._anchor = None
        self._last_seen = None
        self._moved_at = None

    def update(self, t: float, phone: Optional[Box]) -> None:
        if phone is None:
            if self._last_seen is not None and t - self._last_seen > self.memory_s:
                self.reset()
            return
        centre = box_center(phone)
        diagonal = math.hypot(phone[2] - phone[0], phone[3] - phone[1])
        if self._anchor is None:
            self._anchor = centre
        elif math.dist(centre, self._anchor) > self.tolerance * max(diagonal, 1.0):
            self._anchor = centre
            self._moved_at = t
            self.moves += 1
        self._last_seen = t

    def moving(self, t: float) -> bool:
        return self._moved_at is not None and t - self._moved_at < self.still_s


def handling_cue(phone: Box, face: Optional[Box], *, moving: bool, looking_down: bool,
                 at_face_margin_fw: float = AT_FACE_MARGIN_FW) -> Optional[str]:
    """The first handling cue the evidence shows for the driver's phone, or None.

    ``face`` is the measured face box (pixels; None when no face is known, which
    leaves only the motion cue). ``looking_down`` is the eyes_down condition
    (calibrated pitch below pitch_threshold_deg) on this frame.
    """
    if moving:
        return CUE_MOVING
    if face is None:
        return None
    fx1, fy1, fx2, fy2 = face
    fw = max(fx2 - fx1, 1e-6)
    pcx, pcy = box_center(phone)
    if fy1 <= pcy <= fy2 and fx1 - at_face_margin_fw * fw <= pcx <= fx2 + at_face_margin_fw * fw:
        return CUE_AT_FACE
    if looking_down and pcy > fy2 and fx1 <= pcx <= fx2:
        return CUE_LAP
    return None


@dataclass(frozen=True)
class PhoneEvidenceInput:
    """What the module knows about phones on one frame."""

    detector_ok: bool  # the detector produced a fresh result (not failed / stale / not started)
    detector_reason: str = ""
    any_phone: bool = False  # any phone box this tick
    driver_known: bool = True  # a driver region exists to relate a phone to (person box / seat region)
    driver_phone: bool = False  # a phone inside the driver's region
    hold: str = HOLD_UNKNOWN  # PhoneHoldEvidence.state for the driver's phone
    #: The gaze is off the road this frame: True / False, or None when not measurable.
    eyes_off_road: Optional[bool] = None
    #: The gaze direction points at the phone (core/behavior/phone.gaze_toward), or None.
    gaze_toward_phone: Optional[bool] = None
    #: Credited seconds the eyes have been continuously off the road while the
    #: phone was held (the module's own phone-use clock), and the bar for "sustained".
    held_off_road_s: float = 0.0
    sustain_s: float = 2.0
    #: M8.5: a handling cue (HANDLING_CUES) for a phone whose holding is not observed, or None.
    handling_cue: Optional[str] = None


@dataclass(frozen=True)
class PhoneEvidence:
    level: str
    reason: str
    #: M8.5: set only on phone_near_driver -- the cue that makes it possible
    #: handling (eligible for production phone_use) -- else None.
    handling_cue: Optional[str] = None

    @property
    def rank(self) -> int:
        """Position on LEVELS; -1 for INSUFFICIENT (no answer, not 'low')."""
        return LEVELS.index(self.level) if self.level in LEVELS else -1

    def to_dict(self) -> dict:
        out = {"level": self.level, "reason": self.reason}
        if self.handling_cue is not None:
            out["handling_cue"] = self.handling_cue
        return out


def classify_handling(hold: str, cue: Optional[str] = None) -> PhoneEvidence:
    """The level of a phone inside the driver's region from its hold state and
    handling cue, before attention is considered (classify_phone refines a held
    phone with the gaze). Shared by the shadow engine and production (M8.5)."""
    if hold == HOLD_NOT_HELD:
        return PhoneEvidence(MOUNTED, "still, hands seen elsewhere, none on it")
    if hold == HOLD_HELD:
        return PhoneEvidence(HELD, "held")
    reason = "in the driver's region; holding not established"
    if cue is not None:
        reason += f"; possible handling: {_CUE_TEXT.get(cue, cue)} (unvalidated)"
        return PhoneEvidence(NEAR_DRIVER, reason, cue)
    return PhoneEvidence(NEAR_DRIVER, reason)


def supports_phone_use(evidence: Optional[PhoneEvidence]) -> bool:
    """M8.5 production gate: may a sustained eyes-off-road episode be phone_use?

    Held (or above), or near the driver with a handling cue. Mounted, visible,
    near the driver without a cue, no phone, insufficient evidence: no -- the
    episode is judged by eyes_down / distraction / eyes_up alone."""
    if evidence is None:
        return False
    if evidence.level in RISK_LEVELS:
        return True
    return evidence.level == NEAR_DRIVER and evidence.handling_cue is not None


def classify_phone(e: PhoneEvidenceInput) -> PhoneEvidence:
    if not e.detector_ok:
        return PhoneEvidence(INSUFFICIENT, e.detector_reason or "object detector has no fresh result")
    if not e.any_phone:
        return PhoneEvidence(NO_PHONE, "")
    if not e.driver_known:
        return PhoneEvidence(INSUFFICIENT, "phone seen but no driver region to relate it to")
    if not e.driver_phone:
        return PhoneEvidence(VISIBLE, "outside the driver's region")
    if e.hold != HOLD_HELD:
        return classify_handling(e.hold, e.handling_cue)
    # Held from here on.
    if e.eyes_off_road and e.held_off_road_s >= e.sustain_s:
        return PhoneEvidence(ACTIVE, f"held, eyes off road {e.held_off_road_s:.1f}s")
    if e.eyes_off_road and e.gaze_toward_phone is not False:
        # Toward the phone, or off road with the direction not judgeable.
        why = "gaze toward the phone" if e.gaze_toward_phone else "eyes off road, direction not judged"
        return PhoneEvidence(INTERACTION, f"held, {why}")
    if e.eyes_off_road is None:
        return PhoneEvidence(HELD, "held; gaze not measurable")
    return PhoneEvidence(HELD, "held" + ("" if not e.eyes_off_road else ", gaze away from the phone"))


def phone_risk_for(evidence: PhoneEvidence, eyes_off_road: Optional[bool]) -> float:
    """The risk-fusion phone term. Visible / mounted / near-driver / insufficient: 0."""
    if evidence.level == ACTIVE:
        return 1.0
    if evidence.level == INTERACTION:
        return 0.6
    if evidence.level == HELD and eyes_off_road:
        return 0.6
    return 0.0


__all__ = [
    "ACTIVE",
    "AT_FACE_MARGIN_FW",
    "CUE_AT_FACE",
    "CUE_LAP",
    "CUE_MOVING",
    "HANDLING_CUES",
    "HELD",
    "INSUFFICIENT",
    "INTERACTION",
    "LEVELS",
    "MOUNTED",
    "NEAR_DRIVER",
    "NO_PHONE",
    "RISK_LEVELS",
    "STILL_S",
    "STILL_TOLERANCE",
    "VISIBLE",
    "PhoneEvidence",
    "PhoneEvidenceInput",
    "PhonePlacement",
    "classify_handling",
    "classify_phone",
    "handling_cue",
    "phone_risk_for",
    "supports_phone_use",
]
