"""Driver identity (M10): whose face the eye/gaze/head-pose signals come from, and
where the driver sits.

Three pieces, all deterministic and synthetic-tested only (no in-cabin data has
evaluated any of them; docs/DRIVER_ASSOCIATION.md):

  RegionCalibration   During the existing calibration window (driver seated,
                      looking ahead) record the driver's face box and person box;
                      derive a normalised driver region (their union plus a
                      margin) and persist it next to the calibration file,
                      versioned. Recomputed on ``--recalibrate``; not derived, or
                      ignored on load, when implausible (another person in view
                      during calibration, a region too wide to tell seats apart,
                      a face not on the detected person, a different frame shape).

  DriverFaceSelector  With a driver region known, FaceMesh runs with two faces and
                      this picks the driver's: the face continuing the driver's
                      face track, else the face in the driver region, else (all
                      faces outside it) a face on the person centred in the
                      driver's seat (a driver leaning / slumping out of it), else
                      a face on no detected person. A face on ANOTHER person's box
                      is a passenger's and is never the driver's: with only such
                      faces the driver's face is not visible, so the face-lost
                      path (coasting, attention_lost) runs for the driver instead
                      of a passenger's eyes being measured.

  DriverIdentity      The facade the module uses: region precedence (configured
                      ``behavior.driver_region`` > calibrated > none), the
                      FaceMesh face budget, person association
                      (core/behavior/association.py) and the identity state
                      reported in telemetry: identified / uncertain / not_visible
                      / unknown.

Without a region nothing here changes what the module measures: FaceMesh keeps
``max_num_faces=1`` and its face is used, as before.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.behavior.association import (
    CALIBRATED_SEAT,
    CONFIGURED_SEAT,
    IDENTIFIED,
    NO_PERSON,
    NOT_VISIBLE,
    UNCERTAIN,
    UNKNOWN,
    DriverAssociation,
    associate_driver,
    face_fit,
)
from core.behavior.evidence import Box, box_center, box_contains_point

REGION_CONFIGURED = "configured"
REGION_CALIBRATED = "calibrated"

FACE_SELECTION_MODES = ("auto", "off")

#: Margin added around the union of the driver's face and person box, in face
#: widths (x) / face heights (y). Room for ordinary leaning; not a seat map.
REGION_MARGIN_FACE_WIDTHS = 0.5
#: A derived region wider than this fraction of the frame cannot separate the
#: driver from a front passenger, so it is not used.
MAX_REGION_WIDTH = 0.70
#: A face narrower than this fraction of the frame is too far / too small to
#: anchor a seat region.
MIN_FACE_WIDTH = 0.03
#: Calibration evidence floor: face samples and detector ticks in the trailing
#: window, and the share of ticks that must see exactly one person.
MIN_FACE_SAMPLES = 5
MIN_PERSON_TICKS = 3
SINGLE_PERSON_RATIO = 0.8
#: The driver's face track: a face within this many (last) face widths of the
#: last driver face, seen within this many seconds, is the same face.
CONTINUITY_S = 1.0
CONTINUITY_FACE_WIDTHS = 1.5
#: Frames whose aspect ratio differs from the calibration's by more than this
#: (relative) cannot reuse a normalised region.
ASPECT_TOLERANCE = 0.02

NormBox = tuple[float, float, float, float]

#: MediaPipe Face Mesh landmarks spanning the face (forehead 10, chin 152, the
#: face edges 234 / 454, eye corners 33 / 263, mouth corners 61 / 291, nose 1).
#: Selection reads only these: iterating all 478 protobuf landmarks of every
#: face cost ~0.25 ms per face (Mac), for a box this one approximates.
FACE_BOX_LANDMARKS = (10, 152, 234, 454, 33, 263, 61, 291, 1)


def landmark_box(landmarks: Any, w: int, h: int) -> Box:
    """A face's pixel box from FACE_BOX_LANDMARKS (all landmarks if the mesh is shorter)."""
    points = ([landmarks[i] for i in FACE_BOX_LANDMARKS] if len(landmarks) > max(FACE_BOX_LANDMARKS)
              else list(landmarks))
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)


# ---------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DriverIdentityConfig:
    """``modules.driver_attention.behavior.driver_identity`` (all optional)."""

    #: auto: with a driver region known at start-up, FaceMesh runs with two
    #: faces and the driver's is selected; off: one face, as before M10.
    face_selection: str = "auto"
    #: Derive, persist and use the installation-calibrated driver region.
    calibrated_region: bool = True


def validate_driver_identity(section: Any) -> list[str]:
    where = "modules.driver_attention.behavior.driver_identity"
    if section is None:
        return []
    if not isinstance(section, dict):
        return [f"`{where}:` must be a mapping, got {section!r}"]
    errors = []
    for key in section:
        if key not in ("face_selection", "calibrated_region"):
            errors.append(f"`{where}.{key}:` unknown key (known: ['calibrated_region', 'face_selection'])")
    if "face_selection" in section and section["face_selection"] not in FACE_SELECTION_MODES:
        errors.append(f"`{where}.face_selection:` must be one of {list(FACE_SELECTION_MODES)}, "
                      f"got {section['face_selection']!r}")
    if "calibrated_region" in section and not isinstance(section["calibrated_region"], bool):
        errors.append(f"`{where}.calibrated_region:` must be true or false")
    return errors


def parse_driver_identity(module_cfg: dict) -> DriverIdentityConfig:
    """Tolerant: config_schema validates; a bad value here falls back to the default."""
    behavior = module_cfg.get("behavior")
    section = behavior.get("driver_identity") if isinstance(behavior, dict) else None
    if not isinstance(section, dict):
        section = {}
    return DriverIdentityConfig(
        face_selection="off" if section.get("face_selection") == "off" else "auto",
        calibrated_region=section.get("calibrated_region") is not False,
    )


# ---------------------------------------------------------------------------- region
def _finite(values: Any, n: int) -> bool:
    return (isinstance(values, (list, tuple)) and len(values) == n
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values))


def _valid_norm_box(box: Any) -> bool:
    return _finite(box, 4) and 0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0


@dataclass(frozen=True)
class DriverRegion:
    box: NormBox  # 0..1 frame coordinates
    source: str  # REGION_CONFIGURED | REGION_CALIBRATED
    #: Calibrated only: the driver's face centre and width (0..1), where a face
    #: that is the driver's is expected; None for a configured region.
    face_anchor: Optional[tuple[float, float, float]] = None
    #: Calibrated only: width / height of the frames it was learned on.
    aspect: Optional[float] = None

    def fits(self, w: int, h: int) -> bool:
        return self.aspect is None or (h > 0 and math.isclose(w / h, self.aspect, rel_tol=ASPECT_TOLERANCE))

    def to_px(self, w: int, h: int) -> Box:
        x1, y1, x2, y2 = self.box
        return (x1 * w, y1 * h, x2 * w, y2 * h)

    def anchor_px(self, w: int, h: int) -> Optional[tuple[float, float, float]]:
        if self.face_anchor is None:
            return None
        cx, cy, fw = self.face_anchor
        return (cx * w, cy * h, fw * w)

    def to_dict(self) -> dict:
        return {"source": self.source, "box": [round(v, 4) for v in self.box],
                "face_anchor": None if self.face_anchor is None else [round(v, 4) for v in self.face_anchor]}


def _median_box(boxes: list[NormBox]) -> NormBox:
    return (statistics.median(b[0] for b in boxes), statistics.median(b[1] for b in boxes),
            statistics.median(b[2] for b in boxes), statistics.median(b[3] for b in boxes))


def derive_region(faces: list[NormBox], person_ticks: list[list[NormBox]],
                  aspect: Optional[float]) -> tuple[Optional[DriverRegion], str]:
    """The driver region from one calibration window, or None with why not.

    ``faces``: the measured face box per frame; ``person_ticks``: the person
    boxes of each detector tick (all normalised 0..1)."""
    if len(faces) < MIN_FACE_SAMPLES:
        return None, (f"not derived: {len(faces)} face sample(s) in the calibration window "
                      f"(need {MIN_FACE_SAMPLES})")
    if len(person_ticks) < MIN_PERSON_TICKS:
        return None, (f"not derived: the object detector ran on {len(person_ticks)} tick(s) of the "
                      f"calibration window (need {MIN_PERSON_TICKS}); no person evidence")
    single = [people[0] for people in person_ticks if len(people) == 1]
    ratio = len(single) / len(person_ticks)
    if ratio < SINGLE_PERSON_RATIO:
        return None, (f"not derived: exactly one person on only {ratio:.0%} of detector ticks "
                      f"(need {SINGLE_PERSON_RATIO:.0%}); calibrate with the driver alone in view")
    face = _median_box(faces)
    person = _median_box(single)
    if not box_contains_point(person, box_center(face)):
        return None, "not derived: the measured face is not on the detected person"
    fw, fh = face[2] - face[0], face[3] - face[1]
    if fw < MIN_FACE_WIDTH:
        return None, f"not derived: face width {fw:.3f} of the frame is below {MIN_FACE_WIDTH}"
    mx, my = REGION_MARGIN_FACE_WIDTHS * fw, REGION_MARGIN_FACE_WIDTHS * fh
    box = (max(0.0, min(face[0], person[0]) - mx), max(0.0, min(face[1], person[1]) - my),
           min(1.0, max(face[2], person[2]) + mx), min(1.0, max(face[3], person[3]) + my))
    if box[2] - box[0] > MAX_REGION_WIDTH:
        return None, (f"not derived: the region would span {box[2] - box[0]:.0%} of the frame width "
                      f"(max {MAX_REGION_WIDTH:.0%}); it could not tell the driver from a passenger")
    cx, cy = box_center(face)
    return DriverRegion(box, REGION_CALIBRATED, (cx, cy, fw), aspect), "derived from the installation calibration"


def region_path_for(calibration_path: Path) -> Path:
    """``data/calibration.json`` -> ``data/calibration.driver_region.json``."""
    return calibration_path.with_name(calibration_path.stem + ".driver_region.json")


class RegionCalibration:
    """Collects the calibration window's face / person boxes; derives, persists, loads the region."""

    FORMAT_VERSION = 1

    def __init__(self, path: Path, *, enabled: bool, load: bool, window_s: float) -> None:
        self.path = path
        self.enabled = enabled
        self.window_s = float(window_s)
        self.region: Optional[DriverRegion] = None
        self.status = "disabled (behavior.driver_identity.calibrated_region: false)" if not enabled else (
            "not calibrated yet")
        self._faces: deque[tuple[float, NormBox]] = deque()
        self._ticks: deque[tuple[float, list[NormBox]]] = deque()
        self._last_tick: Optional[int] = None
        self._aspect: Optional[float] = None
        self._done = not enabled
        if enabled and load:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            self.status = "no calibrated driver region on file (run --recalibrate with the driver alone)"
            return
        except (OSError, ValueError):
            self.status = "calibrated driver region unreadable: ignored"
            return
        if not isinstance(data, dict) or data.get("version") != self.FORMAT_VERSION:
            self.status = "calibrated driver region in an unknown format: ignored"
            return
        region, anchor, aspect = data.get("region"), data.get("face_anchor"), data.get("aspect")
        if region is None:
            self.status = str(data.get("status") or "no driver region was derived at calibration")
            return
        if anchor is None or aspect is None or not (
                _valid_norm_box(region) and _finite(anchor, 3) and _finite([aspect], 1)):
            self.status = "calibrated driver region implausible: ignored"
            return
        box: NormBox = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
        face_anchor = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
        if not (box[2] - box[0] <= MAX_REGION_WIDTH and face_anchor[2] >= MIN_FACE_WIDTH
                and box_contains_point(box, (face_anchor[0], face_anchor[1])) and float(aspect) > 0):
            self.status = "calibrated driver region implausible: ignored"
            return
        self.region = DriverRegion(box, REGION_CALIBRATED, face_anchor, float(aspect))
        self.status = "loaded from the installation calibration"
        self._done = True

    @property
    def finished(self) -> bool:
        return self._done

    def observe(self, now: float, *, calibrated: bool, face: Optional[NormBox], people: list[NormBox],
                tick: Optional[int], aspect: float) -> None:
        """One frame. While the attention calibration runs, record; when it completes, derive."""
        if self._done:
            return
        if not calibrated:
            self._aspect = aspect
            if face is not None:
                self._faces.append((now, face))
            if tick is not None and tick != self._last_tick:
                self._last_tick = tick
                self._ticks.append((now, list(people)))
            cutoff = now - self.window_s
            for samples in (self._faces, self._ticks):
                while samples and samples[0][0] < cutoff:
                    samples.popleft()
            return
        self._done = True
        if not self._faces and not self._ticks:
            return  # calibrated from file, nothing collected: the load status stands
        self.region, self.status = derive_region([f for _, f in self._faces], [p for _, p in self._ticks],
                                                 self._aspect)
        record = {
            "version": self.FORMAT_VERSION, "status": self.status,
            "region": None if self.region is None else list(self.region.box),
            "face_anchor": None if self.region is None else list(self.region.face_anchor or ()),
            "aspect": self._aspect,
            "samples": {"faces": len(self._faces), "detector_ticks": len(self._ticks)},
            "margin_face_widths": REGION_MARGIN_FACE_WIDTHS,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(self.path)
        except OSError as exc:
            self.status += f" (not persisted: {exc})"


# ---------------------------------------------------------------------------- face selection
@dataclass(frozen=True)
class FaceChoice:
    index: Optional[int]  # which face of the FaceMesh result; None = the driver's face is not seen
    faces: int
    identity: str
    reason: str
    verified: bool = False  # continuity, or the only face in the driver region


class DriverFaceSelector:
    def __init__(self, continuity_s: float = CONTINUITY_S, continuity_face_widths: float = CONTINUITY_FACE_WIDTHS):
        self.continuity_s = continuity_s
        self.continuity_face_widths = continuity_face_widths
        self._track: Optional[tuple[float, float, float, float, bool]] = None  # cx, cy, width, t, verified

    def reset(self) -> None:
        self._track = None

    @staticmethod
    def _rank(indices: list[int], faces: list[Box], anchor: Optional[tuple[float, float, float]],
              region: Optional[Box] = None) -> int:
        """Best of several candidate faces: nearest the calibrated driver face
        (position and size); without an anchor, nearest the region's centre when
        given (faces outside it), else the largest (the nearest the lens)."""
        def score(i: int) -> float:
            f = faces[i]
            width = max(f[2] - f[0], 1e-6)
            cx, cy = box_center(f)
            if anchor is None:
                if region is None:
                    return -width
                rx, ry = box_center(region)
                return math.hypot(cx - rx, cy - ry)
            ax, ay, aw = anchor
            return math.hypot(cx - ax, cy - ay) / max(aw, 1e-6) + 2.0 * abs(math.log(width / max(aw, 1e-6)))
        return min(indices, key=score)

    def choose(self, faces: list[Box], now: float, region: Box, *, anchor: Optional[tuple[float, float, float]],
               people: Optional[list[Box]]) -> FaceChoice:
        """``people``: this detector tick's person boxes (pixels), or None when
        the detector cannot say (stale / failed / not started)."""
        n = len(faces)
        if n == 0:
            return FaceChoice(None, 0, NOT_VISIBLE, "no face")
        centres = [box_center(f) for f in faces]
        choice: Optional[FaceChoice] = None
        track = self._track
        if track is not None and now - track[3] <= self.continuity_s:
            tx, ty, tw, _, verified = track
            near = [(math.hypot(c[0] - tx, c[1] - ty), i) for i, c in enumerate(centres)]
            near = [(d, i) for d, i in near if d <= self.continuity_face_widths * max(tw, 1.0)]
            if near:
                index = min(near)[1]
                # An unverified track (a leaning driver) is confirmed once its face
                # is the only one back in the driver region.
                if not verified:
                    inside = [i for i, c in enumerate(centres) if box_contains_point(region, c)]
                    verified = inside == [index]
                choice = FaceChoice(index, n, IDENTIFIED if verified else UNCERTAIN,
                                    "continues the driver's face track", verified)
        if choice is None:
            inside = [i for i, c in enumerate(centres) if box_contains_point(region, c)]
            if len(inside) == 1:
                choice = FaceChoice(inside[0], n, IDENTIFIED, "the face in the driver region", True)
            elif inside:
                choice = FaceChoice(self._rank(inside, faces, anchor), n, UNCERTAIN,
                                    f"{len(inside)} faces in the driver region; chose the one "
                                    + ("nearest the calibrated driver face" if anchor else "nearest the lens"))
            else:
                # Every face is outside the driver region. A face on the person
                # seated in the region is the driver's (leaning / slumping out of
                # it); a face on another person is a passenger's and is never
                # measured as the driver's; a face on nobody (person missed, or
                # the detector cannot say) is kept: there is no evidence it is not
                # the driver's, and dropping it could only delay a driver alert.
                seat = [p for p in people or [] if box_contains_point(region, box_center(p))]
                others = [p for p in people or [] if p not in seat]

                def on(boxes: list[Box], c: tuple[float, float]) -> bool:
                    return any(face_fit(p, c) is not None for p in boxes)

                on_seat = [i for i, c in enumerate(centres) if on(seat, c)]
                unclaimed = [i for i, c in enumerate(centres) if not on(seat, c) and not on(others, c)]
                if on_seat:
                    choice = FaceChoice(self._rank(on_seat, faces, anchor, region), n, UNCERTAIN,
                                        "outside the driver region, on the person in the driver's seat (leaning)")
                elif unclaimed:
                    why = ("people are not counted (object detector not running)" if people is None
                           else "it is on no detected person")
                    choice = FaceChoice(self._rank(unclaimed, faces, anchor, region), n, UNCERTAIN,
                                        f"outside the driver region and {why}: kept as the driver's")
                else:
                    return FaceChoice(None, n, NOT_VISIBLE, f"{n} face(s), none in the driver region; each is on "
                                      "a person outside it (a passenger's)")
        assert choice.index is not None
        f = faces[choice.index]
        cx, cy = centres[choice.index]
        self._track = (cx, cy, f[2] - f[0], now, choice.verified)
        return choice


# ---------------------------------------------------------------------------- facade
class DriverIdentity:
    """Region precedence, the FaceMesh face budget, face selection, association, reporting."""

    def __init__(self, config: DriverIdentityConfig, *, configured_region: Optional[NormBox],
                 calibration_path: Path, force_recalibrate: bool, window_s: float,
                 attention_calibrated: bool) -> None:
        self.config = config
        self._configured = (DriverRegion(configured_region, REGION_CONFIGURED)
                            if configured_region is not None and _valid_norm_box(configured_region) else None)
        # A stored region is only as current as the attention calibration it was
        # learned with: without that calibration (missing, stale, --recalibrate)
        # a fresh window runs and rewrites both.
        self.calibration = RegionCalibration(
            region_path_for(calibration_path), enabled=config.calibrated_region,
            load=attention_calibrated and not force_recalibrate, window_s=window_s)
        #: Fixed at construction: FaceMesh's face budget cannot change under a
        #: built graph. A region first derived mid-session (first boot) is used
        #: for association at once and for face selection from the next start.
        self.selection_active = config.face_selection == "auto" and self.region is not None
        self.max_faces = 2 if self.selection_active else 1
        self._selector = DriverFaceSelector()
        self.last_choice: Optional[FaceChoice] = None

    @property
    def region(self) -> Optional[DriverRegion]:
        if self._configured is not None:
            return self._configured
        if self.config.calibrated_region:
            return self.calibration.region
        return None

    def _region_px(self, w: int, h: int) -> Optional[Box]:
        region = self.region
        return region.to_px(w, h) if region is not None and region.fits(w, h) else None

    def select_face(self, faces: list[Box], w: int, h: int, now: float,
                    person_boxes: Optional[list[Box]]) -> FaceChoice:
        """Which of this frame's faces (pixel boxes, FaceMesh order) is the driver's.

        ``person_boxes`` is None when the object detector cannot currently say."""
        region_px = self._region_px(w, h) if self.selection_active else None
        if not faces:
            choice = FaceChoice(None, 0, NOT_VISIBLE, "no face")
        elif region_px is None:
            return self.first_face(len(faces), "the driver region was calibrated on a differently shaped frame"
                                   if self.selection_active else "")
        else:
            region = self.region
            assert region is not None
            choice = self._selector.choose(faces, now, region_px, anchor=region.anchor_px(w, h), people=person_boxes)
        self.last_choice = choice
        return choice

    def first_face(self, faces: int, why: str = "") -> FaceChoice:
        """No selection (as before M10): the face model's first face, if any."""
        if faces == 0:
            choice = FaceChoice(None, 0, NOT_VISIBLE, "no face")
        else:
            why = why or ("face selection off" if self.config.face_selection == "off" else
                          "no driver region" if self.region is None else
                          "driver region derived this session; face selection starts at the next start-up")
            choice = FaceChoice(0, faces, UNKNOWN, f"{why}: the face model's first face")
        self.last_choice = choice
        return choice

    def associate(self, person_boxes: list[Box], face_box: Optional[Box], w: int, h: int) -> DriverAssociation:
        region_px = self._region_px(w, h)
        method = CONFIGURED_SEAT if self._configured is not None else CALIBRATED_SEAT
        # The measured face speaks for the driver only if it was SELECTED as the
        # driver's; with one face and no selection it may be a passenger's.
        return associate_driver(person_boxes, face_box, region_px, region_method=method,
                                face_trusted=self.selection_active)

    def observe_calibration(self, now: float, *, calibrated: bool, face_landmarks: Any,
                            person_boxes: list[Box], detection_tick: Optional[int], w: int, h: int) -> None:
        """One face frame of the attention calibration window (the measured face's landmarks)."""
        if w <= 0 or h <= 0 or self.calibration.finished:
            return

        def norm(b: Box) -> NormBox:
            return (b[0] / w, b[1] / h, b[2] / w, b[3] / h)

        face = None if face_landmarks is None else norm(landmark_box(face_landmarks, w, h))
        self.calibration.observe(now, calibrated=calibrated, face=face,
                                 people=[norm(b) for b in person_boxes], tick=detection_tick, aspect=w / h)

    def state(self, association: Optional[DriverAssociation]) -> tuple[str, str]:
        """(identified | uncertain | not_visible | unknown, why) for the measured face and associated person."""
        choice = self.last_choice
        if choice is None or choice.index is None:
            return NOT_VISIBLE, "no frame yet" if choice is None else choice.reason
        if association is None or association.method in ("unknown", NO_PERSON):
            if choice.verified:
                return IDENTIFIED, choice.reason
            return UNKNOWN, "people not counted" if association is None else association.reason
        if choice.identity == UNCERTAIN:
            return UNCERTAIN, choice.reason
        if association.identity in (UNCERTAIN, NOT_VISIBLE):
            return UNCERTAIN, association.reason
        return IDENTIFIED, association.reason

    def snapshot(self, association: Optional[DriverAssociation]) -> dict:
        """Flat telemetry fields."""
        state, _ = self.state(association)
        region = self.region
        return {
            "driver_identity": state,
            "driver_identity_method": association.method if association is not None else "unknown",
            "driver_faces": 0 if self.last_choice is None else self.last_choice.faces,
            "driver_region_source": "none" if region is None else region.source,
        }

    def report(self, association: Optional[DriverAssociation]) -> dict:
        state, reason = self.state(association)
        region = self.region
        return {
            "state": state, "reason": reason,
            "face": None if self.last_choice is None else {
                "faces": self.last_choice.faces, "selected": self.last_choice.index,
                "reason": self.last_choice.reason},
            "person": None if association is None else association.to_dict(),
            "face_selection": {"mode": self.config.face_selection, "active": self.selection_active,
                               "max_faces": self.max_faces},
            "region": None if region is None else region.to_dict(),
            "calibrated_region": self.calibration.status,
            "accuracy": "unevaluated",
        }


__all__ = [
    "REGION_CALIBRATED",
    "REGION_CONFIGURED",
    "DriverFaceSelector",
    "DriverIdentity",
    "DriverIdentityConfig",
    "DriverRegion",
    "FaceChoice",
    "RegionCalibration",
    "derive_region",
    "parse_driver_identity",
    "region_path_for",
    "validate_driver_identity",
]
