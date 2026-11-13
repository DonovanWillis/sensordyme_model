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

Synthetic-scenario tested only; the levels' boundaries need in-cabin data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class PhoneEvidence:
    level: str
    reason: str

    @property
    def rank(self) -> int:
        """Position on LEVELS; -1 for INSUFFICIENT (no answer, not 'low')."""
        return LEVELS.index(self.level) if self.level in LEVELS else -1

    def to_dict(self) -> dict:
        return {"level": self.level, "reason": self.reason}


def classify_phone(e: PhoneEvidenceInput) -> PhoneEvidence:
    if not e.detector_ok:
        return PhoneEvidence(INSUFFICIENT, e.detector_reason or "object detector has no fresh result")
    if not e.any_phone:
        return PhoneEvidence(NO_PHONE, "")
    if not e.driver_known:
        return PhoneEvidence(INSUFFICIENT, "phone seen but no driver region to relate it to")
    if not e.driver_phone:
        return PhoneEvidence(VISIBLE, "outside the driver's region")
    if e.hold == HOLD_NOT_HELD:
        return PhoneEvidence(MOUNTED, "still, hands seen elsewhere, none on it")
    if e.hold != HOLD_HELD:
        return PhoneEvidence(NEAR_DRIVER, "in the driver's region; holding not established")
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
    "HELD",
    "INSUFFICIENT",
    "INTERACTION",
    "LEVELS",
    "MOUNTED",
    "NEAR_DRIVER",
    "NO_PHONE",
    "RISK_LEVELS",
    "VISIBLE",
    "PhoneEvidence",
    "PhoneEvidenceInput",
    "classify_phone",
    "phone_risk_for",
]
