"""Hand-anchored object perception: "where are the hands, and what is
associated with them" instead of "what objects exist anywhere in the frame".

Generic full-frame object detection treats every pixel equally, which wastes
the majority of a Pi's perception budget on background, and struggles
specifically on the case that matters most for a safety product: a SMALL
object held close to the body, often partially occluded by fingers. This
module inverts the priority order:

    full frame -> person -> hands -> hand ROI -> object candidates -> identity

The hand becomes an attention prior. Once MediaPipe Pose (already running
every frame in any module that wants flailing/posture too -- no new
inference for hand LOCATION) reports a wrist, a region of interest is built
around it and the SAME already-loaded object detector (core.detectors.
predict_roi) is run on just that crop -- which is free of any new model, and
measurably more sensitive to a small object than the same detector's
full-frame pass, the same lever a security camera's digital-PTZ zoom uses.

WHAT COUNTS AS "IN THE HAND" is deliberately not "object overlaps hand box".
association_score below reasons about the RELATIONSHIP: spatial proximity/
overlap, whether the object's own tracked trajectory (core.tracking.
BoxTracker gives both hands and objects real identity, velocity, and a
trail) moves WITH the hand's rather than independently of it, and detector
confidence for both sides. A hand moving past a stationary object scores low
on the motion term even at close range; an object moving in lock-step with a
hand scores high even before they visibly overlap.

TEMPORAL CONFIRMATION is deliberately NOT reinvented here: a single frame's
association_score, however high, does not mean "held" -- the calling module
feeds the boolean (score >= threshold) into core.event_engine.EventEngine
exactly like every other sustained condition in this codebase, so "held"
requires the SAME kind of short, configurable, tested persistence window as
holding_object always has, not a bespoke state machine.

UNKNOWN OBJECTS are a first-class outcome, not a failure. If a hand-
associated candidate is real (an actual detector box, not a hallucination)
but its class either doesn't map to a configured label or clears the
detector's floor without clearing its confident-classification bar, this
reports UNKNOWN_OBJECT with the detector's own (lower) confidence -- "I
cannot tell you what it is, but something is there and moving with the
hand" is honest signal a safety product should surface, not silence in
favor of a guess.

Deliberately product-agnostic: this module knows about hands, boxes, and
scores, never about "baby" or "driver" or an event type. See
modules/baby_monitoring.py for how a specific product wires this into its
own event/notification meaning; modules/driver_attention.py's existing
hand-box tracking (erratic-hand-movement detection, already on
core.tracking.BoxTracker) is the natural next caller of this exact engine,
documented as a follow-up rather than wired in this pass -- see the final
report for why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from core.detectors import Box, iou, predict_roi
from core.tracking import BoxTracker, TrackedBox

# MediaPipe Pose landmark indices this module reads. Shared with
# core/activity.py's constants (same physical points), duplicated rather than
# imported: this module has no other dependency on core.activity, and two
# independent, tiny, stable index constants are lower-risk than a cross-import
# for what they're worth.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

UNKNOWN_OBJECT = "unknown_object"

#: Below this MediaPipe visibility, a wrist is treated as absent, not guessed.
_MIN_WRIST_VISIBILITY = 0.5

#: Hand ROI sizing, in torso lengths (scale-invariant -- works at any distance
#: from the camera, the same reasoning core/activity.py's motion thresholds use).
ROI_BASE_RADIUS_TORSO = 0.55
#: How much the ROI widens per unit of the hand's own recent speed (box-
#: diagonals/second) -- a fast-moving hand needs a wider net, both for motion
#: blur and because whatever it carries moves with it between the frame the
#: ROI was sized for and the frame it is used on.
ROI_SPEED_EXPANSION = 0.5
#: Floor so a degenerate/not-yet-measured torso reading never collapses the
#: ROI to a point.
ROI_MIN_RADIUS_FRAC = 0.05

#: Half-size of the box that stands for the HAND itself (not the search
#: region), in torso lengths: a hand is roughly a third of a torso length
#: long, so +-0.2 torso around the palm centre covers it. Association scores
#: spatial agreement against THIS box. The ROI (above) is only where to look:
#: when association scored against the ROI, every object the crop found
#: "overlapped the hand" by construction and the spatial term stopped
#: discriminating -- a still object 0.6 torso lengths from a still wrist
#: scored as held.
HAND_BOX_RADIUS_TORSO = 0.2
#: The palm centre sits beyond the wrist along the forearm by about this
#: fraction of the elbow-to-wrist length (hand ~0.11 body heights from wrist
#: to fingertip, forearm ~0.15).
_PALM_EXTRAPOLATION = 0.35

#: Below this speed (box-diagonals/second), a tracked box counts as "not
#: really moving" for the purposes of velocity-agreement scoring.
_STILL_SPEED = 0.08


@dataclass(frozen=True)
class HandAnchor:
    """One hand's raw position this frame, from Pose wrist landmarks."""

    side: str  # "left" | "right"
    wrist: tuple[float, float]  # normalized (0..1) frame coordinates
    elbow: Optional[tuple[float, float]]
    visibility: float


def detect_hand_anchors(landmarks, min_visibility: float = _MIN_WRIST_VISIBILITY) -> list[HandAnchor]:
    """Wrist positions for whichever hands are visible this frame.

    Reads MediaPipe Pose landmarks already being computed for other reasons
    (flailing, posture) -- no new inference for hand LOCATION. A wrist below
    `min_visibility` is treated as absent, never guessed: a confident "I
    can't see that hand" is worth more than a wrong position for it.
    """
    anchors = []
    for side, wrist_idx, elbow_idx in (
        ("left", LEFT_WRIST, LEFT_ELBOW),
        ("right", RIGHT_WRIST, RIGHT_ELBOW),
    ):
        try:
            wrist_lm = landmarks[wrist_idx]
        except (IndexError, TypeError):
            continue
        visibility = float(getattr(wrist_lm, "visibility", 1.0))
        if visibility < min_visibility:
            continue
        elbow_point = None
        try:
            elbow_lm = landmarks[elbow_idx]
            if float(getattr(elbow_lm, "visibility", 1.0)) >= min_visibility:
                elbow_point = (float(elbow_lm.x), float(elbow_lm.y))
        except (IndexError, TypeError):
            pass
        anchors.append(HandAnchor(side=side, wrist=(float(wrist_lm.x), float(wrist_lm.y)),
                                   elbow=elbow_point, visibility=visibility))
    return anchors


def torso_length_from_landmarks(landmarks, aspect: float = 1.0) -> Optional[float]:
    """Shoulder-to-hip distance in aspect-corrected normalized units -- the
    SAME scale reference core/activity.py's ActivityRecognizer uses, so a
    threshold expressed "in torso lengths" means the same thing regardless of
    how close the subject is to the camera."""
    try:
        ls, rs = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
        lh, rh = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
    except (IndexError, TypeError):
        return None
    if min(getattr(ls, "visibility", 1.0), getattr(rs, "visibility", 1.0),
           getattr(lh, "visibility", 1.0), getattr(rh, "visibility", 1.0)) < _MIN_WRIST_VISIBILITY:
        return None
    shoulder = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    hip = ((lh.x + rh.x) / 2, (lh.y + rh.y) / 2)
    length = math.hypot((hip[0] - shoulder[0]) * aspect, hip[1] - shoulder[1])
    return length if length > 1e-4 else None


def hand_roi_box(
    anchor: HandAnchor, torso_length_norm: float, frame_w: int, frame_h: int,
    aspect: float, hand_speed_diag_per_s: float = 0.0,
    base_radius_torso: float = ROI_BASE_RADIUS_TORSO,
    speed_expansion: float = ROI_SPEED_EXPANSION,
    min_radius_frac: float = ROI_MIN_RADIUS_FRAC,
) -> Box:
    """A dynamic region of interest around one hand, in pixel xyxy.

    NOT a tiny fixed box around the wrist point: sized in torso lengths (so
    it means the same thing near or far from the camera) and widened by the
    hand's own recent speed, so a fast-moving hand still has its held object
    inside the searched region rather than trailing just outside a
    too-small, too-static box.
    """
    radius_y_units = max(
        base_radius_torso * torso_length_norm + speed_expansion * hand_speed_diag_per_s * torso_length_norm,
        min_radius_frac,
    )
    # torso_length_norm is a distance in "aspect-corrected" units (x already
    # scaled by `aspect` before being combined with y) -- dividing by aspect
    # again converts the radius back to plain normalized-x units for the
    # horizontal extent, while the vertical extent uses it directly.
    radius_x_units = radius_y_units / max(aspect, 1e-6)
    wx, wy = anchor.wrist
    x1 = int((wx - radius_x_units) * frame_w)
    y1 = int((wy - radius_y_units) * frame_h)
    x2 = int((wx + radius_x_units) * frame_w)
    y2 = int((wy + radius_y_units) * frame_h)
    return (max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2))


def hand_box(
    anchor: HandAnchor, torso_length_norm: float, frame_w: int, frame_h: int, aspect: float,
    radius_torso: float = HAND_BOX_RADIUS_TORSO,
) -> Box:
    """The hand's own extent, in pixel xyxy -- what association is scored
    against (the ROI from hand_roi_box is only the search region).

    Centred on the palm: the wrist extrapolated along the forearm when the
    elbow is visible, else the wrist itself. Sized in torso lengths, with the
    same aspect handling as hand_roi_box.
    """
    wx, wy = anchor.wrist
    if anchor.elbow is not None:
        ex, ey = anchor.elbow
        wx, wy = wx + (wx - ex) * _PALM_EXTRAPOLATION, wy + (wy - ey) * _PALM_EXTRAPOLATION
    radius_y = max(radius_torso * torso_length_norm, ROI_MIN_RADIUS_FRAC / 2.0)
    radius_x = radius_y / max(aspect, 1e-6)
    x1 = int((wx - radius_x) * frame_w)
    y1 = int((wy - radius_y) * frame_h)
    x2 = int((wx + radius_x) * frame_w)
    y2 = int((wy + radius_y) * frame_h)
    return (max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2))


# ------------------------------------------------------------------ association
def _velocity_vector(track: TrackedBox, aspect: float = 1.0) -> Optional[tuple[float, float]]:
    """(dx, dy) over the track's full trail, in box-diagonals/second --
    direction AND magnitude, unlike TrackedBox.velocity() (speed only)."""
    if len(track.trail) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = track.trail[0], track.trail[-1]
    elapsed = t1 - t0
    diagonal = track.diagonal
    if elapsed <= 0 or diagonal <= 1e-6:
        return None
    return ((x1 - x0) * aspect / diagonal / elapsed, (y1 - y0) / diagonal / elapsed)


def _cosine_similarity(a: tuple[float, float], b: tuple[float, float]) -> Optional[float]:
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < 1e-6 or nb < 1e-6:
        return None
    return max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))


def _velocity_agreement(hand: TrackedBox, obj: TrackedBox, aspect: float) -> float:
    """[0, 1]: how consistent hand and object motion are with the object
    being carried BY the hand, not merely near it.

    The nearest-innocent-motion case this exists to reject: a hand passing
    close to a stationary object. Both tracks near-stationary is treated as
    mildly supportive (a held object can be held still); one moving while
    the other stays put is treated as contrary evidence; moving together --
    similar direction AND similar speed -- is the strongest positive signal
    available short of an actual grip sensor.
    """
    hand_speed = hand.velocity(aspect) or 0.0
    obj_speed = obj.velocity(aspect) or 0.0
    hand_still, obj_still = hand_speed < _STILL_SPEED, obj_speed < _STILL_SPEED
    if hand_still and obj_still:
        return 0.6
    if hand_still != obj_still:
        return 0.15
    hand_vec, obj_vec = _velocity_vector(hand, aspect), _velocity_vector(obj, aspect)
    if hand_vec is None or obj_vec is None:
        return 0.5
    cosine = _cosine_similarity(hand_vec, obj_vec)
    if cosine is None:
        return 0.5
    direction_term = (cosine + 1) / 2
    speed_ratio = min(hand_speed, obj_speed) / max(hand_speed, obj_speed)
    return max(0.0, min(1.0, 0.6 * direction_term + 0.4 * speed_ratio))


def _spatial_agreement(
    hand: TrackedBox, obj: TrackedBox, aspect: float, proximity_range_diag: float,
    hand_extent: Optional[Box] = None,
) -> float:
    """[0, 1]: overlap where it exists, graceful falloff by distance
    (in hand-diagonal units) otherwise -- an object beside or below the
    hand, or between fingers, is not always overlapping its box.

    `hand_extent`, when given, is the hand's own box (see hand_box) and is
    what overlap/distance are measured against instead of the track's box --
    HandPerceptionEngine's hand track carries the ROI, which every object the
    ROI crop finds overlaps by construction."""
    box = hand_extent if hand_extent is not None else hand.box
    overlap = iou(box, obj.box)
    if overlap > 0:
        return max(overlap, 0.85)  # any real overlap is already strong evidence
    x1, y1, x2, y2 = box
    scale = max(math.hypot((x2 - x1) * aspect, y2 - y1), 1e-6)
    hx, hy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    ox, oy = obj.centroid
    distance = math.hypot((ox - hx) * aspect, oy - hy) / scale
    return max(0.0, 1.0 - distance / proximity_range_diag)


def association_score(
    hand: TrackedBox, obj: TrackedBox, aspect: float = 1.0, proximity_range_diag: float = 2.5,
    hand_extent: Optional[Box] = None,
) -> float:
    """A single [0, 1] score for "is `obj` being carried by `hand`",
    combining spatial agreement, trajectory agreement, and detector
    confidence on both sides -- reasoning about a RELATIONSHIP, not just
    "did these two boxes touch this frame". See the module docstring for the
    false-positive case (a hand passing a stationary object) this is built
    to reject, and _velocity_agreement's docstring for the motion term.

    Spatial and motion agreement are combined MULTIPLICATIVELY (spatial
    scales a motion-derived factor), not summed: a weighted SUM lets one
    strong term paper over the other being near zero -- two confidently
    detected but stationary, far-apart boxes both being "still" (motion's
    generous case) must not partially rescue a spatial reading of
    essentially zero. Detector confidence is applied as a final multiplier
    for the same reason: an uncertain detection should scale the whole
    claim down, not add independent baseline evidence for a relationship
    the boxes' own geometry doesn't support.

    `aspect` is only for NORMALISED box coordinates; pixel boxes (everything
    HandPerceptionEngine tracks) are isotropic and take the default 1.0.
    `hand_extent`: see _spatial_agreement.
    """
    spatial = _spatial_agreement(hand, obj, aspect, proximity_range_diag, hand_extent)
    motion = _velocity_agreement(hand, obj, aspect)
    core = spatial * (0.4 + 0.6 * motion)
    confidence_factor = 0.7 + 0.3 * (0.5 * min(1.0, hand.confidence) + 0.5 * min(1.0, obj.confidence))
    return round(max(0.0, min(1.0, core * confidence_factor)), 3)


# ------------------------------------------------------------------ the engine
@dataclass
class HandPerceptionResult:
    """Everything believed about hands and hand-associated objects this frame."""

    hands: list[TrackedBox] = field(default_factory=list)
    objects: list[TrackedBox] = field(default_factory=list)
    #: hand label ("left_hand"/"right_hand") -> (object track, association score)
    associations: dict[str, tuple[TrackedBox, float]] = field(default_factory=dict)
    #: hand label -> the hand's own extent (pixel xyxy) this frame. `hands`
    #: tracks carry the search ROI as their box; association is scored here.
    hand_boxes: dict[str, Box] = field(default_factory=dict)


class HandPerceptionEngine:
    """Ties hand anchoring, ROI object search, tracking, and association
    scoring into one per-frame call. Owns two BoxTrackers (hands, objects);
    knows nothing about events, alerts, or which product is calling it.
    """

    def __init__(
        self,
        object_labels: dict[int, str],
        association_threshold: float = 0.55,
        detection_confidence: float = 0.35,
        unknown_object_confidence_floor: float = 0.15,
        roi_min_move_diag: float = 0.4,
        roi_max_interval_s: float = 1.0,
    ) -> None:
        """
        Args:
            object_labels: COCO class id -> label name for classes this
                engine should recognize confidently. A detection whose class
                isn't in here, or whose confidence is below
                `detection_confidence`, is still reported -- as UNKNOWN_OBJECT
                -- rather than dropped, as long as it clears
                `unknown_object_confidence_floor`.
            association_threshold: minimum per-frame association_score to
                count as "associated" at all. The calling module is expected
                to require this SUSTAINED over time (via EventEngine), not
                treat a single frame crossing this bar as "held".
            roi_min_move_diag: a hand must have moved at least this many of
                its OWN diagonals since the last ROI inference for adaptive
                scheduling to trigger a fresh one early.
            roi_max_interval_s: even a stationary hand gets a fresh ROI check
                at least this often (catches an object being set into an
                already-still hand's grip without the hand itself moving).
        """
        self._hand_tracker = BoxTracker()
        # A wider match gate than BoxTracker's own default: a held object's
        # frame-to-frame displacement is bounded by how far the HAND carrying
        # it can move, not by the object's own (often small) size -- matching
        # against the object's own diagonal alone (the default) fragments a
        # small item's track into a new id on every detection once adaptive
        # ROI scheduling means "the next detection" is several real frames
        # (and real hand movement) away, not the very next one.
        self._object_tracker = BoxTracker(max_distance=8.0)
        self._object_labels = object_labels
        self._association_threshold = association_threshold
        self._detection_confidence = detection_confidence
        self._unknown_floor = unknown_object_confidence_floor
        self._roi_min_move_diag = roi_min_move_diag
        self._roi_max_interval_s = roi_max_interval_s
        self._last_roi_at: dict[str, tuple[float, tuple[float, float]]] = {}  # label -> (t, centroid)

    def reset(self) -> None:
        self._hand_tracker.reset()
        self._object_tracker.reset()
        self._last_roi_at.clear()

    def _should_run_roi(self, hand: TrackedBox, now: float) -> bool:
        """Adaptive scheduling: skip the (relatively) expensive ROI-crop
        inference pass for a hand that hasn't moved and was checked recently
        -- "lightweight tracking every frame, ROI inference only when
        something has actually changed" rather than a fixed clock regardless
        of motion."""
        last = self._last_roi_at.get(hand.label)
        if last is None:
            return True
        last_t, last_centroid = last
        if now - last_t >= self._roi_max_interval_s:
            return True
        moved = math.hypot(hand.centroid[0] - last_centroid[0], hand.centroid[1] - last_centroid[1])
        return moved / max(hand.diagonal, 1e-6) >= self._roi_min_move_diag

    def update(
        self, landmarks, frame, model, now: float, aspect: float, coco_class_filter: Optional[list[int]] = None,
    ) -> HandPerceptionResult:
        """Advance one frame. `model` is whatever core.detectors.get_yolo_model
        already returned to the caller (shared, not reloaded here)."""
        h, w = frame.shape[:2]
        anchors = detect_hand_anchors(landmarks)
        # `aspect` corrects NORMALISED landmark coordinates (x spans w, y spans
        # h). Every box from here on is in PIXELS, which are already isotropic,
        # so the trackers and association below run at aspect 1.0 -- passing
        # the frame aspect there inflated horizontal distances and speeds by w/h.
        torso = torso_length_from_landmarks(landmarks, aspect)

        hand_detections = []
        rois: dict[str, Box] = {}
        hand_extents: dict[str, Box] = {}
        for anchor in anchors:
            label = f"{anchor.side}_hand"
            prior = next((t for t in self._hand_tracker.tracks if t.label == label), None)
            speed = (prior.velocity(1.0) or 0.0) if prior is not None else 0.0
            roi = hand_roi_box(anchor, torso or ROI_MIN_RADIUS_FRAC, w, h, aspect, speed)
            rois[label] = roi
            hand_extents[label] = hand_box(anchor, torso or ROI_MIN_RADIUS_FRAC, w, h, aspect)
            # The hand TRACK keeps the ROI as its box (callers read its size and
            # draw it); association below scores against the hand's own extent.
            hand_detections.append((label, roi, anchor.visibility))
        hand_tracks = self._hand_tracker.update(hand_detections, now, 1.0)

        object_detections: list[tuple[str, Box, float]] = []
        for hand in hand_tracks:
            if not self._should_run_roi(hand, now):
                continue
            self._last_roi_at[hand.label] = (now, hand.centroid)
            for box, class_id, score in predict_roi(
                model, frame, rois.get(hand.label, hand.box),
                classes=coco_class_filter, conf=self._unknown_floor,
            ):
                # Belt and braces: a real backend already applies `conf` inside
                # the call above, but object identity here matters enough
                # (this is what stands between a real detection and
                # UNKNOWN_OBJECT) not to depend entirely on every current and
                # future detector backend filtering identically.
                if score < self._unknown_floor:
                    continue
                label = self._object_labels.get(class_id)
                if label is not None and score >= self._detection_confidence:
                    object_detections.append((label, box, score))
                else:
                    object_detections.append((UNKNOWN_OBJECT, box, score))
        self._object_tracker.update(object_detections, now, 1.0)
        # ALL currently-alive object tracks, not just this-frame's matched-or-
        # new ones: adaptive ROI scheduling deliberately skips re-running
        # detection for a hand that hasn't moved much (see _should_run_roi),
        # which means an already-tracked held object may go several frames
        # with no fresh detection feeding it. Unlike modules/baby_monitoring.py's
        # overlap test (which uses only this-frame's matches, because ANY
        # object box might be stale there), this skip is specifically
        # conditioned on the HAND not having moved -- so a stale object
        # position is still a good estimate of where things actually are,
        # and treating the object as having vanished the moment inference is
        # skipped would defeat the entire point of scheduling it adaptively.
        object_tracks = self._object_tracker.tracks

        associations: dict[str, tuple[TrackedBox, float]] = {}
        for hand in hand_tracks:
            best_obj, best_score = None, 0.0
            for obj in object_tracks:
                score = association_score(hand, obj, 1.0, hand_extent=hand_extents.get(hand.label))
                if score > best_score:
                    best_obj, best_score = obj, score
            if best_obj is not None and best_score >= self._association_threshold:
                associations[hand.label] = (best_obj, best_score)

        return HandPerceptionResult(
            hands=hand_tracks, objects=object_tracks, associations=associations, hand_boxes=hand_extents,
        )
