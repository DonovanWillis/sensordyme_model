"""Driver association: which person in the frame is the monitored driver (M8.4, A4).

What the shipped module does: every driver-relative rule (phones_for_driver,
the eating/drinking face gate, face-loss corroboration) takes the LARGEST
person box as the driver. That is an assumption, not a measurement: a
passenger leaning toward a wide-angle lens, or a rear passenger on a
short-focal mount, can be the larger box. Nothing in the product has been
evaluated on which person is the driver.

What this module adds, and only claims what the evidence supports:

  configured_seat_region   ``behavior.driver_region`` (0..1 frame coords) is set:
                           the driver is the person whose box centre lies in it.
                           A fixed region an installer drew; still unevaluated.
  face_anchored_person     the person box containing the centre of the face the
                           face model is measuring -- every face signal (eyes,
                           gaze) is already about THAT person, so relating the
                           phone to the same person is at least self-consistent.
  assumed_largest_person   no face to anchor to (or it lies in no box): the
                           shipped proxy, stated as an assumption.
  ambiguous                the face centre lies in more than one person box and
                           they disagree on which is largest: reported, not guessed.
  no_person_detected       no person box (detector ran and saw none, or did not run).

There is no first-boot SEAT calibration in this codebase (AttentionCalibrator
records EAR and neutral gaze, not a seat region), so "calibrated seat region"
is only the installer-configured one. ``agrees_with_largest`` reports when the
face-anchored driver differs from the largest box: the case where the shipped
proxy would relate a phone to the wrong person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.behavior.evidence import Box, box_area, box_center, box_contains_point

CONFIGURED_SEAT = "configured_seat_region"
FACE_ANCHORED = "face_anchored_person"
ASSUMED_LARGEST = "assumed_largest_person"
AMBIGUOUS = "ambiguous"
NO_PERSON = "no_person_detected"


@dataclass(frozen=True)
class DriverAssociation:
    method: str
    driver_box: Optional[Box]
    people: int
    agrees_with_largest: Optional[bool] = None  # None when there is no largest to compare
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.driver_box is not None

    def to_dict(self) -> dict:
        return {
            "driver_identity": self.method, "people": self.people,
            "agrees_with_largest_person": self.agrees_with_largest,
            "reason": self.reason, "accuracy": "unevaluated",
        }


def largest(boxes: list[Box]) -> Optional[Box]:
    return max(boxes, key=box_area) if boxes else None


def associate_driver(person_boxes: list[Box], face_box: Optional[Box] = None,
                     seat_region: Optional[Box] = None) -> DriverAssociation:
    """Pick the driver's person box from this tick's person boxes (pixels)."""
    people = len(person_boxes)
    if not person_boxes:
        return DriverAssociation(NO_PERSON, None, 0, None, "no person box")
    big = largest(person_boxes)
    if seat_region is not None:
        seated = [b for b in person_boxes if box_contains_point(seat_region, box_center(b))]
        if seated:
            chosen = largest(seated)
            return DriverAssociation(CONFIGURED_SEAT, chosen, people, chosen == big,
                                     "person centred in the configured driver region")
    if face_box is not None:
        centre = box_center(face_box)
        holding = [b for b in person_boxes if box_contains_point(b, centre)]
        if len(holding) == 1:
            return DriverAssociation(FACE_ANCHORED, holding[0], people, holding[0] == big,
                                     "person box containing the measured face")
        if len(holding) > 1:
            chosen = min(holding, key=box_area)  # the tightest box around the face
            if big in holding and chosen != big:
                return DriverAssociation(AMBIGUOUS, None, people, None,
                                         "measured face lies in several person boxes")
            return DriverAssociation(FACE_ANCHORED, chosen, people, chosen == big,
                                     "tightest person box containing the measured face")
    return DriverAssociation(ASSUMED_LARGEST, big, people, True,
                             "largest person box (assumption: no face to anchor to)")


__all__ = [
    "AMBIGUOUS",
    "ASSUMED_LARGEST",
    "CONFIGURED_SEAT",
    "FACE_ANCHORED",
    "NO_PERSON",
    "DriverAssociation",
    "associate_driver",
]
