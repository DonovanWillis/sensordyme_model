"""Driver association: which person in the frame is the monitored driver (M8.4 A4, M10).

What the shipped module did until M8.5: every driver-relative rule
(phones_for_driver, the eating/drinking face gate, face-loss corroboration)
took the LARGEST person box as the driver. That is an assumption, not a
measurement: a passenger leaning toward a wide-angle lens, or a rear passenger
on a short-focal mount, can be the larger box. Nothing in the product has been
evaluated on which person is the driver.

Precedence (M10), each only claiming what the evidence supports:

  configured_seat_region   ``behavior.driver_region`` (0..1 frame coords) is set:
                           the driver is the person whose box centre lies in it.
                           A fixed region an installer drew; still unevaluated.
  calibrated_seat_region   no configured region, but the installation calibration
                           derived one (core/behavior/driver_identity.py): same rule.
  face_anchored_person     the person box containing the centre of the face the
                           face model is measuring -- every face signal (eyes,
                           gaze) is already about THAT person, so relating the
                           phone to the same person is at least self-consistent.
  assumed_largest_person   no face to anchor to (or it lies in no box) and exactly
                           ONE person in frame: the only candidate.
  unresolved               several people, no driver region and no face to anchor
                           to: nobody is picked (M10; the largest box used to be).
  ambiguous                the face lies in several person boxes that disagree, or
                           several people sit in the driver region and the face
                           cannot tell them apart: reported, not guessed.
  driver_not_in_region     a driver region is known and nobody is in it (and the
                           driver's face is not on anyone): the driver is not seen.
  no_person_detected       no person box (detector ran and saw none, or did not run).

``identity`` says how far the answer can be trusted, for telemetry:
``identified`` (one candidate, or the region / face settles it), ``uncertain``
(several people and nothing that settles it, or a fallback), ``not_visible``
(a region is known and the driver is not in it), ``unknown`` (no person boxes).
A phone or food item is related to the driver only through ``driver_box``,
which is None for every ambiguous / unresolved / not-visible answer: no person
picked by guesswork lends the driver their phone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.behavior.evidence import Box, box_area, box_center, box_contains_point

CONFIGURED_SEAT = "configured_seat_region"
CALIBRATED_SEAT = "calibrated_seat_region"
FACE_ANCHORED = "face_anchored_person"
ASSUMED_LARGEST = "assumed_largest_person"
UNRESOLVED = "unresolved"
AMBIGUOUS = "ambiguous"
NOT_IN_REGION = "driver_not_in_region"
NO_PERSON = "no_person_detected"

IDENTIFIED = "identified"
UNCERTAIN = "uncertain"
NOT_VISIBLE = "not_visible"
UNKNOWN = "unknown"

#: A face is "on" a person box only if its centre lies in the box's upper part
#: (a seated person's head): a rear passenger's box that merely reaches down
#: behind the driver's head does not claim the driver's face.
FACE_MAX_REL_Y = 0.6
#: Two person boxes whose horizontal fit to the face differs by less than this
#: (in box widths) cannot be told apart by the face.
FACE_FIT_MARGIN = 0.1


@dataclass(frozen=True)
class DriverAssociation:
    method: str
    driver_box: Optional[Box]
    people: int
    agrees_with_largest: Optional[bool] = None  # None when there is no largest to compare
    reason: str = ""
    identity: str = UNKNOWN

    @property
    def known(self) -> bool:
        return self.driver_box is not None

    def to_dict(self) -> dict:
        return {
            "driver_identity": self.method, "identity": self.identity, "people": self.people,
            "agrees_with_largest_person": self.agrees_with_largest,
            "reason": self.reason, "accuracy": "unevaluated",
        }


def largest(boxes: list[Box]) -> Optional[Box]:
    return max(boxes, key=box_area) if boxes else None


def face_fit(box: Box, face_centre: tuple[float, float]) -> Optional[float]:
    """How well a person box fits a face: horizontal offset of the face from the
    box centre in box widths (lower is better), or None when the face centre is
    not in the box's head part."""
    if not box_contains_point(box, face_centre):
        return None
    height = max(box[3] - box[1], 1e-6)
    if (face_centre[1] - box[1]) / height > FACE_MAX_REL_Y:
        return None
    return abs(box_center(box)[0] - face_centre[0]) / max(box[2] - box[0], 1e-6)


def _seated_choice(seated: list[Box], face_box: Optional[Box]) -> tuple[Optional[Box], str]:
    """One of several people centred in the driver region, settled by the face, or None."""
    if face_box is None:
        return None, "several people in the driver region and no driver face to tell them apart"
    centre = box_center(face_box)
    fits = sorted((fit, i) for i, b in enumerate(seated) if (fit := face_fit(b, centre)) is not None)
    if not fits:
        return None, "several people in the driver region and the driver's face is on none of them"
    if len(fits) > 1 and fits[1][0] - fits[0][0] < FACE_FIT_MARGIN:
        return None, "several people in the driver region fit the driver's face equally"
    return seated[fits[0][1]], "person in the driver region whose head is at the driver's face"


def associate_driver(person_boxes: list[Box], face_box: Optional[Box] = None,
                     seat_region: Optional[Box] = None, *,
                     region_method: str = CONFIGURED_SEAT, face_trusted: bool = True) -> DriverAssociation:
    """Pick the driver's person box from this tick's person boxes (pixels).

    ``face_box`` is the face the module is measuring; ``seat_region`` the
    driver region in pixels, configured or calibrated (``region_method`` says
    which). ``face_trusted`` says the face was SELECTED as the driver's (M10
    face selection); with a region and an untrusted face (one face, possibly a
    passenger's) the face never overrides the region.
    """
    people = len(person_boxes)
    if not person_boxes:
        return DriverAssociation(NO_PERSON, None, 0, None, "no person box", UNKNOWN)
    big = largest(person_boxes)
    if seat_region is not None:
        seated = [b for b in person_boxes if box_contains_point(seat_region, box_center(b))]
        if len(seated) == 1:
            return DriverAssociation(region_method, seated[0], people, seated[0] == big,
                                     "person centred in the driver region", IDENTIFIED)
        if seated:
            chosen, why = _seated_choice(seated, face_box if face_trusted else None)
            if chosen is None:
                return DriverAssociation(AMBIGUOUS, None, people, None, why, UNCERTAIN)
            return DriverAssociation(region_method, chosen, people, chosen == big, why, IDENTIFIED)
        # Nobody centred in the region. The driver may be leaning out of it: the
        # person whose head is at the (selected) driver face is still the driver.
        if face_box is not None and face_trusted:
            centre = box_center(face_box)
            fits = sorted((fit, i) for i, b in enumerate(person_boxes) if (fit := face_fit(b, centre)) is not None)
            if len(fits) == 1 or (len(fits) > 1 and fits[1][0] - fits[0][0] >= FACE_FIT_MARGIN):
                chosen = person_boxes[fits[0][1]]
                return DriverAssociation(FACE_ANCHORED, chosen, people, chosen == big,
                                         "nobody centred in the driver region; person at the driver's face",
                                         UNCERTAIN)
        return DriverAssociation(NOT_IN_REGION, None, people, None,
                                 "nobody in the driver region", NOT_VISIBLE)
    if face_box is not None:
        centre = box_center(face_box)
        holding = [b for b in person_boxes if box_contains_point(b, centre)]
        identity = IDENTIFIED if people == 1 else UNCERTAIN
        if len(holding) == 1:
            return DriverAssociation(FACE_ANCHORED, holding[0], people, holding[0] == big,
                                     "person box containing the measured face", identity)
        if len(holding) > 1:
            chosen = min(holding, key=box_area)  # the tightest box around the face
            if big in holding and chosen != big:
                return DriverAssociation(AMBIGUOUS, None, people, None,
                                         "measured face lies in several person boxes", UNCERTAIN)
            return DriverAssociation(FACE_ANCHORED, chosen, people, chosen == big,
                                     "tightest person box containing the measured face", identity)
    if people == 1:
        return DriverAssociation(ASSUMED_LARGEST, big, people, True,
                                 "the only person box (no face to anchor to)", IDENTIFIED)
    return DriverAssociation(UNRESOLVED, None, people, None,
                             "several people, no driver region and no face on any of them: "
                             "nobody is taken as the driver", UNCERTAIN)


__all__ = [
    "AMBIGUOUS",
    "ASSUMED_LARGEST",
    "CALIBRATED_SEAT",
    "CONFIGURED_SEAT",
    "FACE_ANCHORED",
    "IDENTIFIED",
    "NOT_IN_REGION",
    "NOT_VISIBLE",
    "NO_PERSON",
    "UNCERTAIN",
    "UNKNOWN",
    "UNRESOLVED",
    "DriverAssociation",
    "associate_driver",
    "face_fit",
]
