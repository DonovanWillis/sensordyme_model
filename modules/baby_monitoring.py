"""Baby monitoring module: presence, safe/restricted zones, and stillness — honestly.

What this module actually detects (and what it refuses to fake) is documented in
core/taxonomy.py under `baby_monitoring`. The short version:

  SUPPORTED: child present/absent, leaving a configured safe zone (crib/play
  area), entering configured restricted zones (stairs/pool/stove — the USER
  draws these; a camera cannot know what a pool is), unusual stillness,
  person-count changes, activity bursts (logged observations), a bottle
  overlapping the child's box (an approximation of "holding", not proof —
  the SAME box-overlap approximation core/rules.py's `holding` predicate
  already uses and discloses elsewhere in the product; nothing infant-model-
  specific about it, which is why it's safe to claim where face state isn't),
  and a hand-held object on a closing trajectory toward the baby's mouth
  region (choking-hazard risk scoring — see _assess_choking_hazard below).
  The "mouth region" is core/detectors.py's upper_region geometric proxy
  (the same one `holding`/`near_mouth` already use elsewhere), not a facial
  landmark: this module still does not claim to see an actual mouth, on the
  same infant-face-model grounds as the DELIBERATELY NOT CLAIMED list below.

  DELIBERATELY NOT CLAIMED: face-covered / face-down detection (adult face
  models are not reliable enough on infants for a safety-critical claim),
  sleeping vs awake, crying (no microphone), or NAMING which small object is
  in a hand — COCO has no "coin"/"button"/"battery"/"bead" class, and a
  curated subset would create false confidence about coverage it doesn't
  have (see core/capabilities.py's objects.small_personal for the same
  reasoning applied product-wide). Choking-hazard risk scoring works around
  this rather than pretending it isn't true: it never asks "what is this
  object", only "how big is it relative to the hand holding it, and is it
  closing on the mouth region" — questions a box detector CAN honestly
  answer even for an object with no COCO class at all (core/hand_perception.
  py's UNKNOWN_OBJECT). Faking identification would be worse than omitting it.

All temporal reasoning goes through core/event_engine.py: nothing here alerts
on a single frame, brief occlusions don't reset progress, and every alert has
an observed → developing → confirmed lifecycle with a resolution.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core.accel import COCO_NAMES
from core.activity import (
    ActivityRecognizer,
    ActivityState,
    FLAIL_MIN_PEAK_ACCEL,
    FLAIL_MIN_PEAK_SPEED,
    FLAIL_MIN_REVERSAL_AMPLITUDE,
    FLAIL_MIN_REVERSALS,
)
from core.detections import Detection, DetectionFrame
from core.detectors import get_yolo_model, iou, upper_region
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.hand_perception import HandPerceptionEngine, HandPerceptionResult, UNKNOWN_OBJECT, association_score
from core.tracking import BoxTracker
from core.zones import Zone, load_zones
from modules.base import VisionModule, register_module

_PERSON_CLASS = 0  # COCO
_BOTTLE_CLASS = next(cid for cid, name in COCO_NAMES.items() if name == "bottle")

# How many recent per-frame classifications a held object's label is smoothed
# over, and how many of them must agree before it's trusted (see
# _smoothed_hold_label). A tight, close-up hand-ROI crop classifies right at
# the confidence margin, so a single frame's opinion is not reliable evidence
# on its own -- but consistent agreement across several frames is.
_HOLD_LABEL_HISTORY_WINDOW = 5
_HOLD_LABEL_MIN_VOTES = 3


def _smoothed_hold_label(history: "deque[str]") -> str:
    """The label recent observations actually agree on, or UNKNOWN_OBJECT.

    Deliberately a plurality-with-a-floor, not "whatever the last frame said":
    UNKNOWN_OBJECT is itself a real, honest observation (the classifier
    wasn't confident that frame) and must be able to win the vote same as any
    class name, rather than being filtered out before counting -- a hand
    genuinely holding something the model can't confidently name should
    report as unknown, not flicker to whichever class happened to win a
    single frame.
    """
    if not history:
        return UNKNOWN_OBJECT
    counts: dict[str, int] = {}
    for label in history:
        counts[label] = counts.get(label, 0) + 1
    label, votes = max(counts.items(), key=lambda item: item[1])
    return label if votes >= _HOLD_LABEL_MIN_VOTES else UNKNOWN_OBJECT


# --- choking-hazard risk scoring --------------------------------------------
# "object detected + held in hand + closing on the mouth + small relative to
# the hand = high risk" (product spec), built ENTIRELY from signals this
# module already computes elsewhere -- no new inference, no object
# classification, no infant facial landmarks. See _assess_choking_hazard.

#: An object's box diagonal at or below this fraction of the holding hand's
#: own diagonal counts as fully "small" for risk purposes -- a coin, button,
#: battery, or bead pinched between fingers is much smaller than the hand
#: itself. A ratio, not a pixel size, so it means the same thing near or far
#: from the camera (the same reasoning core/hand_perception.py's ROI sizing
#: and core/activity.py's motion thresholds use).
SMALL_OBJECT_MAX_DIAGONAL_RATIO = 0.7
#: _size_factor's floor for a large object -- never zero. A toy too big to
#: swallow moving into an infant's mouth is still worth flagging, just less
#: urgently than something that actually fits, so "large" fades the risk
#: rather than erasing it.
_SIZE_FACTOR_FLOOR = 0.35
#: How close an object must get to the mouth region, in units of the baby's
#: own box diagonal (scale-invariant the same way as above), before
#: proximity alone starts contributing meaningful risk. 1.0 means "roughly
#: one baby-box-diagonal away or closer starts to matter"; risk fades to
#: zero beyond that, same shape as core/hand_perception.py's proximity term.
MOUTH_PROXIMITY_RANGE = 1.4
#: The fraction of the (possibly inferred) baby's box treated as the mouth
#: region -- see core/detectors.py's upper_region. Wider than that
#: function's own 0.35 default because a reclined/lying infant (the common
#: crib-cam framing) doesn't present the same tight, upright head-in-top-
#: third geometry an adult standing at a driver/security camera does.
MOUTH_ZONE_FRACTION = 0.4


@dataclass(frozen=True)
class ChokingHazardAssessment:
    """One frame's read on one hand's held object, for choking-hazard risk.

    Carries every signal the risk score was built from, not just the score
    itself -- "why did/didn't this fire" (an explicit product requirement)
    needs the breakdown, not a single opaque number. See get_debug_state().
    """

    side: str
    held: bool
    object_label: str = UNKNOWN_OBJECT
    object_track_id: Optional[int] = None
    distance_to_mouth_norm: Optional[float] = None
    approaching: Optional[bool] = None
    size_factor: float = 0.0
    association_confidence: float = 0.0
    risk_score: float = 0.0
    reason: str = "nothing held"


def _infer_baby_box(
    boxes: list[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    """Which detected person box is (most likely) the baby's, when more than
    one person is in frame.

    This module has never distinguished a caregiver from the child by
    detection alone (unusual_stillness, zone checks, "holding" — all already
    apply to "the person in frame" generically; core/taxonomy.py's own
    `unattended` entry discloses this exact limitation). For mouth-proximity
    risk specifically the choice matters, so: the SMALLEST box is treated as
    the baby's, since an infant occupies a much smaller fraction of frame
    than an adult at any camera distance/mounting this product ships with.
    An honest geometric approximation, not a classifier — if it guesses
    wrong (e.g. a caregiver crouched far from camera reads smaller than a
    baby close to it), risk scoring is computed against the wrong box for
    that frame rather than failing safe silently; disclosed in
    core/taxonomy.py alongside every other box-only heuristic here.
    """
    if not boxes:
        return None
    def _area(box: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)
    return min(boxes, key=_area)


def _size_factor(object_diagonal: float, hand_diagonal: float) -> float:
    """[_SIZE_FACTOR_FLOOR, 1.0]: how much a held object's size, RELATIVE to
    the hand holding it, should weight choking-hazard risk.

    No object classification involved: a coin and an unclassified
    UNKNOWN_OBJECT of the same relative size score identically, which is the
    point (see the module docstring on why this repo won't claim to NAME
    small objects even though it can flag their presence and size honestly).
    """
    if hand_diagonal <= 1e-6:
        return _SIZE_FACTOR_FLOOR
    ratio = object_diagonal / hand_diagonal
    if ratio <= SMALL_OBJECT_MAX_DIAGONAL_RATIO:
        return 1.0
    return max(_SIZE_FACTOR_FLOOR, SMALL_OBJECT_MAX_DIAGONAL_RATIO / ratio)


def _point_to_box_distance(
    point: tuple[float, float], box: tuple[float, float, float, float], aspect: float = 1.0,
) -> float:
    """Straight-line distance from `point` to the nearest edge of `box` (0 if
    `point` is already inside it) — the standard clamped point-to-rectangle
    distance, aspect-corrected the same way core/tracking.py's velocity/
    distance math already is elsewhere in this product."""
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx * aspect, dy)


def _assess_choking_hazard(
    side: str,
    assoc: Optional[tuple],
    hand_track,
    baby_box: Optional[tuple[int, int, int, int]],
    aspect: float,
) -> ChokingHazardAssessment:
    """Everything about one hand's held object that matters for choking-hazard
    risk, computed from signals this module already has (hand-object
    association, box trackers with trail/velocity) — no new inference.

    Deliberately multiplicative once every factor is known, matching
    core/hand_perception.py's association_score for the same reason: this is
    a claim that several things are true AT ONCE (held AND small AND closing
    on the mouth), not a claim that a strong reading on one axis should be
    able to paper over a weak one on another. A large, stationary,
    far-from-the-mouth object must not score high just because it happens to
    be confidently tracked.
    """
    if assoc is None:
        return ChokingHazardAssessment(side=side, held=False, reason="nothing held")
    object_track, association_confidence = assoc
    if baby_box is None:
        return ChokingHazardAssessment(
            side=side, held=True, object_label=object_track.label, object_track_id=object_track.track_id,
            association_confidence=association_confidence, reason="no baby detected in frame this frame",
        )
    x1, y1, x2, y2 = baby_box
    baby_diagonal = math.hypot((x2 - x1) * aspect, y2 - y1)
    if baby_diagonal <= 1e-6:
        return ChokingHazardAssessment(
            side=side, held=True, object_label=object_track.label, object_track_id=object_track.track_id,
            association_confidence=association_confidence, reason="degenerate baby box",
        )
    mouth_zone = upper_region(baby_box, MOUTH_ZONE_FRACTION)
    mx1, my1, mx2, my2 = mouth_zone
    mouth_center = ((mx1 + mx2) / 2.0, (my1 + my2) / 2.0)

    distance = _point_to_box_distance(object_track.centroid, mouth_zone, aspect)
    distance_norm = distance / baby_diagonal
    proximity_factor = max(0.0, 1.0 - distance_norm / MOUTH_PROXIMITY_RANGE)

    approaching = object_track.approaching(mouth_center, aspect)
    trajectory_factor = 1.0 if approaching else (0.5 if approaching is None else 0.2)

    hand_diagonal = hand_track.diagonal if hand_track is not None else 0.0
    size_factor = _size_factor(object_track.diagonal, hand_diagonal)

    risk = association_confidence * size_factor * proximity_factor * trajectory_factor
    reason = (
        f"distance_norm={distance_norm:.2f} approaching={approaching} "
        f"size_factor={size_factor:.2f} proximity_factor={proximity_factor:.2f}"
    )
    return ChokingHazardAssessment(
        side=side, held=True, object_label=object_track.label, object_track_id=object_track.track_id,
        distance_to_mouth_norm=round(distance_norm, 3), approaching=approaching,
        size_factor=round(size_factor, 3), association_confidence=round(association_confidence, 3),
        risk_score=round(max(0.0, min(1.0, risk)), 3), reason=reason,
    )


# COCO class name -> label used on the generic "holding_object" event's detail
# text. Configurable (`object_classes` in config.yaml, mirrors
# driver_attention.py's `yolo_classes`) and shares the SAME detector pass
# already running for person/bottle -- no added inference cost. Bottle keeps
# its own dedicated "holding_bottle" event (feeding is the single most
# safety-relevant "holding" signal in this mode, and existing rules/tests
# depend on that exact name); everything here fires the shared, generic
# event instead of one event type per class.
#
# "keys" is deliberately absent: COCO has no such class, and this module's
# own principle (see the module docstring's DELIBERATELY NOT CLAIMED list) is
# to refuse claiming detection capability the model doesn't actually have,
# rather than silently no-op on a class nothing will ever match.
DEFAULT_OBJECT_CLASSES: dict[str, str] = {
    "cell phone": "phone",
    "cup": "cup",
    "wine glass": "cup",
    "bowl": "bowl",
    "banana": "food",
    "apple": "food",
    "orange": "food",
    "sandwich": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    "fork": "utensil",
    "knife": "utensil",
    "spoon": "utensil",
    "scissors": "hazard",
    "remote": "object",
}


@register_module("baby_monitoring")
class BabyMonitoringModule(VisionModule):
    """Child-safety monitoring: zones + presence + stillness with staged confirmation."""

    name = "baby_monitoring"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._confidence_threshold = float(config.get("confidence_threshold", 0.4))
        self._detect_every_n = max(1, int(config.get("detect_every_n_frames", 2)))
        self._still_max_motion = float(config.get("still_max_motion", 0.015))
        self._expect_presence = bool(config.get("expect_presence", False))
        self._weights = config.get("model_weights", "yolo11n.pt")
        self._model = None  # lazy: loaded on first frame (keeps tests light)

        object_class_map: dict[str, str] = dict(config.get("object_classes", DEFAULT_OBJECT_CLASSES))
        self._object_class_ids: dict[int, str] = {}
        for cid, coco_name in COCO_NAMES.items():
            if coco_name in object_class_map:
                self._object_class_ids[cid] = object_class_map[coco_name]
        unmatched = set(object_class_map) - {COCO_NAMES[cid] for cid in self._object_class_ids}
        if unmatched:
            print(
                f"baby_monitoring: object_classes entries {sorted(unmatched)} don't match any "
                f"'{self._weights}' class name -- check config.yaml spelling"
            )

        self._zones: list[Zone] = load_zones(config.get("zones", {}))
        # The distinct labels this module was configured to recognize -- used both to
        # register one holding_object rule PER LABEL below and to make sure every
        # registered rule gets a present/absent update every frame (see _reason),
        # even on a frame where that label wasn't detected at all.
        self._object_labels: list[str] = sorted(set(self._object_class_ids.values()))

        # --- event lifecycle rules --------------------------------------------
        self._engine = EventEngine({})
        still_s = float(config.get("still_warning_s", 120.0))
        absence_s = float(config.get("absence_warning_s", 30.0))
        zone_s = float(config.get("zone_confirm_s", 2.0))
        for zone in self._zones:
            if zone.kind == "restricted":
                self._engine.add_rule(RuleSpec(
                    detection=f"zone_violation:{zone.name}", severity="critical",
                    persist_s=zone_s, min_frames=3, resolve_after_s=5.0,
                ))
        if any(z.kind == "safe" for z in self._zones):
            self._engine.add_rule(RuleSpec(
                detection="safe_zone_exit", severity="critical",
                persist_s=zone_s, min_frames=3, resolve_after_s=5.0,
            ))
        self._engine.add_rule(RuleSpec(
            detection="unusual_stillness", severity="warning",
            persist_s=still_s, min_frames=10, occlusion_grace_s=3.0, resolve_after_s=10.0,
        ))
        if self._expect_presence:
            self._engine.add_rule(RuleSpec(
                detection="person_absent", severity="warning",
                persist_s=absence_s, min_frames=5, resolve_after_s=5.0,
            ))
        self._engine.add_rule(RuleSpec(
            detection="activity_burst", severity="info",
            persist_s=0.5, min_frames=2, resolve_after_s=3.0, cooldown_s=10.0,
        ))
        # "holding" cooldown is deliberately much shorter than the 30s the engine
        # defaults to elsewhere (zone/presence rules, where a long cooldown avoids
        # re-alerting on flapping): a child sets an object down and picks up a
        # DIFFERENT one constantly, and that is a legitimate new observation each
        # time, not repeat noise about the same event. See holding_object's
        # per-label keying below for the other half of this fix -- a shared
        # "holding_object" key across every class meant one class's cooldown could
        # silently swallow a completely different class picked up moments later.
        holding_cooldown_s = float(config.get("holding_cooldown_s", 3.0))
        self._engine.add_rule(RuleSpec(
            detection="holding_bottle", severity="info",
            persist_s=float(config.get("holding_bottle_confirm_s", 1.0)),
            min_frames=3, resolve_after_s=2.0, cooldown_s=holding_cooldown_s,
        ))
        for label in self._object_labels:
            self._engine.add_rule(RuleSpec(
                detection=f"holding_object:{label}", severity="info",
                persist_s=float(config.get("holding_object_confirm_s", 1.0)),
                min_frames=3, resolve_after_s=2.0, cooldown_s=holding_cooldown_s,
            ))

        # --- object identity: a real tracker (core.tracking.BoxTracker), not just
        # this frame's boxes -- gives every bottle/object a persistent id, velocity,
        # and a trajectory, which is what "approaching" below needs and what
        # `holding_*`'s confidence is now fused from (detection confidence + track
        # stability), replacing a flat 0.7. Same tracker instance handles every
        # class: BoxTracker's matching is class-scoped internally, so a phone can
        # never be confused with a cup.
        self._item_tracker = BoxTracker()
        self._all_item_labels = ["bottle", *self._object_labels]
        self._approaching_min_speed = float(config.get("object_approaching_min_speed", 0.15))
        for label in self._all_item_labels:
            self._engine.add_rule(RuleSpec(
                detection=f"object_approaching:{label}", severity="info",
                persist_s=float(config.get("object_approaching_confirm_s", 0.3)),
                min_frames=2, resolve_after_s=1.0,
                cooldown_s=float(config.get("object_approaching_cooldown_s", 2.0)),
            ))

        # --- arm/body motion: MediaPipe Pose, for flailing/frantic-movement detection.
        # Off by default nowhere -- this module previously had NO pose signal at all
        # (person presence and "holding" were both bounding-box-only), so rapid,
        # erratic arm motion was invisible regardless of any object-detection
        # threshold. Reuses the exact landmark model modules/fall_detection.py
        # already runs every frame elsewhere in the product (proven affordable on a
        # Pi); `enable_pose: false` is a full opt-out for a deployment that can't
        # spare the extra inference.
        self._enable_pose = bool(config.get("enable_pose", True))
        self._pose_every_n = max(1, int(config.get("pose_every_n_frames", 1)))
        self._pose = (
            mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if self._enable_pose else None
        )
        self._activity = ActivityRecognizer(
            flail_min_peak_speed=float(config.get("flail_min_peak_speed", FLAIL_MIN_PEAK_SPEED)),
            flail_min_peak_accel=float(config.get("flail_min_peak_accel", FLAIL_MIN_PEAK_ACCEL)),
            flail_min_reversals=int(config.get("flail_min_reversals", FLAIL_MIN_REVERSALS)),
            flail_min_reversal_amplitude=float(
                config.get("flail_min_reversal_amplitude", FLAIL_MIN_REVERSAL_AMPLITUDE)
            ),
        ) if self._enable_pose else None
        self._engine.add_rule(RuleSpec(
            detection="flailing", severity="warning",
            persist_s=float(config.get("flailing_confirm_s", 0.5)),
            min_frames=4, resolve_after_s=1.5,
            cooldown_s=float(config.get("flailing_cooldown_s", 5.0)),
        ))

        # --- hand-anchored object perception: hands (from the SAME Pose pass
        # above) as an attention prior for object search, replacing "does a
        # full-frame box overlap the child's whole-body box" with "is this
        # object moving WITH a specific hand" -- see core/hand_perception.py
        # for the full reasoning. Shared, product-agnostic engine (Driver Mode
        # is documented there as the next caller); this module owns the
        # baby-specific event/transfer semantics built on top of it.
        hand_object_labels: dict[int, str] = dict(self._object_class_ids)
        hand_object_labels[_BOTTLE_CLASS] = "bottle"
        # Deliberately every COCO class EXCEPT person, not just the configured
        # object_classes vocabulary: filtering the ROI search down to
        # hand_object_labels would mean an object outside that curated list
        # produces ZERO detections in the crop, not UNKNOWN_OBJECT -- silently
        # defeating the entire "honest unknown, not silent miss" point of
        # UNKNOWN_OBJECT (core/hand_perception.py's own docstring). A class
        # outside hand_object_labels still becomes UNKNOWN_OBJECT downstream
        # (HandPerceptionEngine.update's own label-mapping); only "definitely
        # not an object" (a hand crop that catches the person's own torso/face)
        # is excluded here.
        self._hand_class_filter = sorted(cid for cid in COCO_NAMES if cid != _PERSON_CLASS)
        self._hand_engine = HandPerceptionEngine(
            object_labels=hand_object_labels,
            association_threshold=float(config.get("hand_association_threshold", 0.55)),
            detection_confidence=float(config.get("hand_detection_confidence", 0.35)),
            unknown_object_confidence_floor=float(config.get("hand_unknown_object_floor", 0.15)),
        )
        # object_in_hand:{side} is the primary hand-anchored signal (drives
        # object_picked_up/object_put_down/object_transferred below); it is
        # DELIBERATELY separate from holding_bottle/holding_object:{label},
        # which stay driven by whichever of the legacy overlap test or this
        # engine's associations is more confident this frame (see
        # _combined_holding in _reason) -- fusion, not a replacement, per the
        # "keep the current detector as a secondary signal" requirement.
        for side in ("left", "right"):
            self._engine.add_rule(RuleSpec(
                detection=f"object_in_hand:{side}", severity="info",
                persist_s=float(config.get("hand_hold_confirm_s", 1.0)),
                min_frames=3, resolve_after_s=2.0, cooldown_s=holding_cooldown_s,
            ))

        # --- choking hazard: a held object on a closing trajectory toward the
        # baby's mouth region -- see _assess_choking_hazard for the risk model.
        # A SEPARATE, critical-severity signal from object_in_hand above (which
        # fires for ANY held object, held anywhere): this one requires the
        # object to actually be approaching the mouth, weighted toward small
        # objects, and is meant to interrupt someone immediately, not log
        # quietly like ordinary holding does.
        self._choking_hazard_risk_threshold = float(config.get("choking_hazard_risk_threshold", 0.4))
        for side in ("left", "right"):
            self._engine.add_rule(RuleSpec(
                detection=f"choking_hazard:{side}", severity="critical",
                persist_s=float(config.get("choking_hazard_confirm_s", 0.4)),
                min_frames=3, resolve_after_s=3.0,
                cooldown_s=float(config.get("choking_hazard_cooldown_s", 6.0)),
            ))
        # This frame's per-hand risk breakdown, for get_debug_state() -- see
        # ChokingHazardAssessment for why the full breakdown is kept, not just
        # a pass/fail bit.
        self._last_choking_assessments: dict[str, ChokingHazardAssessment] = {}

        # Per-hand memory the engine itself doesn't keep (it's stateless about
        # "events" on purpose -- see its module docstring): what's currently
        # believed to be in each hand, and a short-lived record of what was
        # JUST released, so a release immediately followed by the OTHER hand
        # confirming the SAME track id reports as one object_transferred
        # rather than a spurious put_down+pick_up pair.
        self._current_hold_label: dict[str, str] = {}
        self._current_hold_track_id: dict[str, Optional[int]] = {}
        self._recent_release: dict[int, tuple[str, float, str]] = {}
        self._transfer_window_s = float(config.get("object_transfer_window_s", 2.0))
        # Per-hand history of raw per-frame classifications, for smoothing (see
        # _smoothed_hold_label): a tight, close-up hand ROI crop classifies
        # right at the confidence margin, and a single frame's opinion flickers
        # between plausible classes (bottle/phone/cup) for what is one held
        # object -- found live, holding one bottle produced object_picked_up
        # events labelled phone, cup, and object across one session. The
        # per-hand key (not per-track) is deliberate: core.tracking.BoxTracker
        # only matches a new detection to an existing track when their labels
        # are EQUAL, so a classification flip doesn't relabel a track, it
        # fragments into a new one -- a per-track history would restart on
        # every flip, which is exactly the problem this exists to smooth over.
        self._hold_label_history: dict[str, deque[str]] = {
            "left": deque(maxlen=_HOLD_LABEL_HISTORY_WINDOW), "right": deque(maxlen=_HOLD_LABEL_HISTORY_WINDOW),
        }

        # --- per-frame state ---------------------------------------------------
        self._frame_index = 0
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._last_confidences: list[float] = []
        self._last_bottle_boxes: list[tuple[int, int, int, int]] = []
        self._last_bottle_confidences: list[float] = []
        self._last_object_boxes: dict[str, list[tuple[int, int, int, int]]] = {}
        self._last_object_confidences: dict[str, list[float]] = {}
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._last_count = -1
        self._prev_gray: Optional[np.ndarray] = None
        self._last_motion = 0.0
        self._last_pose_proto = None
        self._last_pose_landmarks = None
        self._last_activity: Optional[ActivityState] = None
        self._last_hand_result: Optional[HandPerceptionResult] = None
        self._hand_debug_snapshot_saved = False  # TEMPORARY, see process()
        self._status = "watching"

    # ------------------------------------------------------------------ perception
    def _detect_people(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """YOLO person boxes (overridable in tests).

        Bottle boxes and any configured `object_classes` boxes ride along on
        the SAME detector pass (no added inference cost) and are recorded as
        a side effect (self._last_bottle_boxes / self._last_object_boxes) --
        the same pattern _last_confidences already uses, so subclasses and
        tests that override this method for its person-box return value keep
        working unchanged.
        """
        if self._model is None:
            self._model = get_yolo_model(self._weights)
        classes = sorted({_PERSON_CLASS, _BOTTLE_CLASS, *self._object_class_ids})
        results = self._model.predict(frame, verbose=False, classes=classes, conf=self._confidence_threshold)
        boxes, confidences, bottle_boxes, bottle_confidences = [], [], [], []
        object_boxes: dict[str, list[tuple[int, int, int, int]]] = {}
        object_confidences: dict[str, list[float]] = {}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else _PERSON_CLASS
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                if class_id == _BOTTLE_CLASS:
                    bottle_boxes.append(xyxy)
                    bottle_confidences.append(conf)
                elif class_id in self._object_class_ids:
                    label = self._object_class_ids[class_id]
                    object_boxes.setdefault(label, []).append(xyxy)
                    object_confidences.setdefault(label, []).append(conf)
                else:
                    boxes.append(xyxy)
                    confidences.append(conf)
        # Recorded alongside rather than returned, so subclasses and tests that
        # override this method for its boxes keep working unchanged.
        self._last_confidences = confidences
        self._last_bottle_boxes = bottle_boxes
        self._last_bottle_confidences = bottle_confidences
        self._last_object_boxes = object_boxes
        self._last_object_confidences = object_confidences
        return boxes

    def _motion_energy(self, frame: np.ndarray) -> float:
        """Mean absolute inter-frame difference on a downsampled grayscale (0..1)."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        motion = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray
        return motion

    def _update_activity(self, frame: np.ndarray, w: int, h: int, now: float) -> Optional[ActivityState]:
        """Run MediaPipe Pose and advance the rolling-window activity classifier
        (currently used for arm-flailing detection; overridable in tests)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            self._last_pose_proto = None
            self._last_pose_landmarks = None
            # Tolerant of a brief dropout (core/activity.py's pose_lost, up to
            # POSE_LOST_GRACE_S) rather than an immediate reset(): fast, chaotic
            # motion is both what flailing detection exists to catch AND a
            # common cause of MediaPipe losing lock for a frame or two, so
            # treating every dropout as "the subject left" would blind the
            # window exactly when genuine distress motion needs it most.
            self._activity.pose_lost(now)
            return None
        self._last_pose_proto = results.pose_landmarks
        self._last_pose_landmarks = results.pose_landmarks.landmark
        aspect = w / max(h, 1)
        return self._activity.update(results.pose_landmarks.landmark, now=now, aspect=aspect)

    # ------------------------------------------------------------------ reasoning
    def process(self, frame: np.ndarray) -> list[Event]:
        now = time.monotonic()
        self._frame_index += 1
        if self._frame_index % self._detect_every_n == 0 or self._last_count < 0:
            self._last_boxes = self._detect_people(frame)
        boxes = self._last_boxes
        self._last_motion = self._motion_energy(frame)
        h, w = frame.shape[:2]
        self._last_frame_size = (w, h)

        hand_result: Optional[HandPerceptionResult] = None
        if self._pose is not None and self._frame_index % self._pose_every_n == 0:
            self._last_activity = self._update_activity(frame, w, h, now)
            # Requires self._model, loaded lazily by _detect_people above -- on
            # the very first frame that always runs (self._last_count starts at
            # -1), so by the time we get here it's set unless a test/subclass
            # overrides _detect_people entirely (see its own docstring), in
            # which case hand perception simply doesn't run this frame and
            # holding_bottle/holding_object fall back to the legacy signal.
            if self._last_pose_landmarks is not None and self._model is not None:
                aspect = w / max(h, 1)
                hand_result = self._hand_engine.update(
                    self._last_pose_landmarks, frame, self._model, now, aspect,
                    coco_class_filter=self._hand_class_filter,
                )
                # TEMPORARY diagnostic (remove once confirmed): ground truth for
                # what the ROI detector is actually looking at -- saved once
                # (not every frame) so this can't fill the disk on a long-running
                # device.
                if not self._hand_debug_snapshot_saved and hand_result.hands:
                    self._hand_debug_snapshot_saved = True
                    try:
                        import os
                        os.makedirs("data", exist_ok=True)
                        snap = frame.copy()
                        for hand in hand_result.hands:
                            hx1, hy1, hx2, hy2 = (int(v) for v in hand.box)
                            cv2.rectangle(snap, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
                            cv2.putText(snap, hand.label, (hx1, max(12, hy1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        cv2.imwrite("data/hand_debug_frame.jpg", snap)
                        for hand in hand_result.hands:
                            hx1, hy1, hx2, hy2 = (int(v) for v in hand.box)
                            crop = frame[max(0, hy1):hy2, max(0, hx1):hx2]
                            if crop.size > 0:
                                cv2.imwrite(f"data/hand_debug_crop_{hand.label}.jpg", crop)
                    except Exception:  # noqa: BLE001 -- diagnostics must never crash the service
                        pass
                # TEMPORARY diagnostic (remove once live association behavior is
                # confirmed): shows what the ROI detector actually found per
                # hand and the best (even sub-threshold) association score --
                # "nothing held" and "held but scored too low" look identical
                # from get_telemetry_snapshot's objects_in_hand count alone.
                if hand_result.hands and self._frame_index % 15 == 0:
                    per_hand = []
                    for hand in hand_result.hands:
                        best = max(
                            (association_score(hand, o, aspect) for o in hand_result.objects),
                            default=None,
                        )
                        per_hand.append(f"{hand.label} roi_diag={hand.diagonal:.0f}px best_score={best}")
                    objs = [(o.label, round(o.confidence, 2)) for o in hand_result.objects]
                    print(f"sensordyme.hand_debug: {per_hand} objects={objs}", flush=True)
        self._last_hand_result = hand_result

        return self._reason(
            boxes, self._last_motion, w, h, now, self._last_bottle_boxes, self._last_object_boxes,
            self._last_activity, hand_result,
        )

    def _reason(self, boxes: list, motion: float, w: int, h: int, now: float,
                bottle_boxes: list = (), object_boxes: dict = None,
                activity_state: Optional[ActivityState] = None,
                hand_result: Optional[HandPerceptionResult] = None) -> list[Event]:
        """Pure decision layer (unit-testable without camera or model)."""
        object_boxes = object_boxes or {}
        events: list[Event] = []
        present = len(boxes) > 0

        # Person count changes (log-only context, mirrors security mode).
        if len(boxes) != self._last_count and self._last_count >= 0:
            events.append(Event.now(
                "person_count_change", 0.0, 0.9, self.name, f"count {self._last_count} -> {len(boxes)}"
            ))
        self._last_count = len(boxes)

        transitions = []
        # Restricted zones: someone's feet inside the zone.
        for zone in self._zones:
            if zone.kind != "restricted":
                continue
            inside = any(zone.contains_box(b, w, h) for b in boxes)
            transitions += self._engine.update(f"zone_violation:{zone.name}", inside, 0.9, now)

        # Safe zones: subject visible but outside every safe zone.
        safe_zones = [z for z in self._zones if z.kind == "safe"]
        if safe_zones:
            outside = present and not any(
                z.contains_box(b, w, h) for z in safe_zones for b in boxes
            )
            transitions += self._engine.update("safe_zone_exit", outside, 0.85, now)

        # Stillness: subject visible with very little motion.
        transitions += self._engine.update(
            "unusual_stillness", present and motion < self._still_max_motion, 0.8, now
        )
        # Activity bursts: large motion while visible (logged, never an alarm).
        transitions += self._engine.update("activity_burst", present and motion > 6 * self._still_max_motion, 0.7, now)

        # --- object identity: feed every bottle/object box through the tracker
        # first, so "holding" reasons about a persistent, identified item (with
        # a confidence built from detection quality + track stability) rather
        # than a bare box that resets every frame. Tracked boxes also persist
        # through a brief miss (within the tracker's own lost_grace_s), which
        # smooths a one-frame detector dropout the same way occlusion_grace_s
        # already smooths it at the event-lifecycle level -- two independent
        # layers of the same idea, not a substitute for either.
        aspect = w / max(h, 1)
        detections = [
            ("bottle", tuple(box), self._last_bottle_confidences[i] if i < len(self._last_bottle_confidences) else 1.0)
            for i, box in enumerate(bottle_boxes)
        ]
        for label in self._object_labels:
            label_confidences = self._last_object_confidences.get(label, [])
            for i, box in enumerate(object_boxes.get(label, ())):
                detections.append((
                    label, tuple(box), label_confidences[i] if i < len(label_confidences) else 1.0,
                ))
        # Only THIS frame's matched-or-new tracks, not `.tracks` (every track
        # still alive within the tracker's own lost_grace_s): a track not
        # matched this frame keeps its last box FROZEN in place, and testing
        # overlap against a frozen box would keep "holding" true using a
        # position that hasn't actually been re-observed -- exactly the kind
        # of stale-state bug the per-label cooldown fix above was written to
        # eliminate, reintroduced at a different layer if this weren't careful.
        matched_this_frame = self._item_tracker.update(detections, now, aspect)
        tracks_by_label: dict[str, list] = {}
        for track in matched_this_frame:
            tracks_by_label.setdefault(track.label, []).append(track)

        def _holding(label: str) -> tuple[bool, float]:
            """(is holding, fused confidence) for one label's tracked items --
            overlap with the child decides presence; detection confidence AND
            track stability (age/hit-count, see TrackedBox.stability) decide
            HOW confident, replacing what used to be a flat 0.7 regardless of
            whether the item had been reliably tracked for a second or had
            just barely crossed the detector's own confidence floor."""
            holding = False
            confidence = 0.5
            for track in tracks_by_label.get(label, []):
                if any(iou(b, track.box) > 0 for b in boxes):
                    holding = True
                    confidence = max(confidence, min(1.0, 0.3 + 0.4 * track.confidence + 0.3 * track.stability))
            return holding, confidence

        def _corroborated(label: str, box) -> bool:
            """Does the EXISTING full-frame detector also see this class near
            here this frame? Sensor fusion between the two independent object
            signals, not a replacement of one by the other."""
            candidates = bottle_boxes if label == "bottle" else object_boxes.get(label, ())
            return any(iou(b, box) > 0 for b in candidates)

        def _combined_holding(label: str) -> tuple[bool, float]:
            """OR-fused "is holding" from two independent signals: the ORIGINAL
            full-frame-detection-overlaps-child's-box approximation (works even
            when Pose/hands aren't visible), and the NEW hand-anchored
            association (higher precision -- knows WHICH hand, doesn't depend
            on the child's whole-body box). Either alone is enough to flag
            holding -- both already document that "holding" here is an
            approximation, not proof of grip; when both agree, the stronger
            confidence wins, boosted further by _corroborated's bonus."""
            legacy_held, legacy_confidence = _holding(label)
            hand_held, hand_confidence = False, 0.0
            if hand_result is not None:
                for obj_track, score in hand_result.associations.values():
                    if obj_track.label != label:
                        continue
                    confidence = min(1.0, score + 0.15) if _corroborated(label, obj_track.box) else score
                    if not hand_held or confidence > hand_confidence:
                        hand_held, hand_confidence = True, confidence
            held = legacy_held or hand_held
            confidence = max(legacy_confidence if legacy_held else 0.0, hand_confidence if hand_held else 0.0)
            return held, (confidence if held else 0.5)

        # Bottle overlapping the child's box (or associated with a hand): an
        # approximation of "holding", not proof (a bottle propped beside them
        # in a bouncer would also match the legacy signal) -- the same
        # box-overlap approximation core/rules.py's `holding` predicate already
        # uses and discloses. Bare bottle presence elsewhere in frame is NOT
        # this signal on purpose: "bottle detected" alone isn't meaningful.
        holding_bottle, bottle_confidence = _combined_holding("bottle")
        transitions += self._engine.update("holding_bottle", holding_bottle, bottle_confidence, now)

        # Same fused signal, generalized to every other configured class
        # (phone, cup, food, ...) -- see DEFAULT_OBJECT_CLASSES. Each label gets
        # its OWN engine key (holding_object:<label>) and is updated every frame
        # regardless of whether that label was seen at all this frame -- a phone
        # AND a cup can both be held at once, each independently confirms/
        # resolves/cools down, and picking up a different object right after
        # putting one down is a fresh observation, not suppressed by the first
        # object's own cooldown (see holding_cooldown_s above).
        for label in self._object_labels:
            holding, confidence = _combined_holding(label)
            transitions += self._engine.update(f"holding_object:{label}", holding, confidence, now)

        # Hand-anchored pickup/put-down/transfer: a strictly per-HAND signal
        # (which hand, honest UNKNOWN_OBJECT when classification isn't
        # confident) that holding_bottle/holding_object above cannot express --
        # those are per-LABEL and silent about which hand, and never fire at
        # all for an unrecognized class. With no hand_result this frame (pose
        # disabled, or no hand visible), this section reports nothing and
        # holding_bottle/holding_object still work via their legacy fallback.
        self._recent_release = {
            track_id: v for track_id, v in self._recent_release.items()
            if now - v[1] <= self._transfer_window_s
        }
        # Computed ONCE per frame, shared by both hands below -- see
        # _infer_baby_box for why the smallest box stands in for the baby's
        # when more than one person is in frame.
        baby_box = _infer_baby_box(boxes)
        choking_assessments: dict[str, ChokingHazardAssessment] = {}
        for side in ("left", "right"):
            assoc = hand_result.associations.get(f"{side}_hand") if hand_result is not None else None
            held = assoc is not None
            confidence = 0.0
            if held:
                obj_track, confidence = assoc
                self._hold_label_history[side].append(obj_track.label)
                label = _smoothed_hold_label(self._hold_label_history[side])
                if label != UNKNOWN_OBJECT and _corroborated(label, obj_track.box):
                    confidence = min(1.0, confidence + 0.15)
                self._current_hold_label[side] = label
                self._current_hold_track_id[side] = obj_track.track_id
            else:
                # Nothing held: stale classification history must not bias
                # whatever gets picked up next, same reasoning as this
                # module's other per-subject resets on loss of the subject.
                self._hold_label_history[side].clear()
            transitions += self._engine.update(f"object_in_hand:{side}", held, confidence, now)

            # Choking hazard: reuses the SAME association this frame (no
            # re-detection), scored against the baby's mouth region. Gated on
            # `held` up front -- an empty hand near the face never reaches
            # risk scoring at all, which is the product's core distinction
            # ("do not simply alert whenever a hand is near the baby's face").
            hand_track = None
            if held and hand_result is not None:
                hand_track = next((h for h in hand_result.hands if h.label == f"{side}_hand"), None)
            assessment = _assess_choking_hazard(side, assoc, hand_track, baby_box, aspect)
            choking_assessments[side] = assessment
            transitions += self._engine.update(
                f"choking_hazard:{side}",
                assessment.held and assessment.risk_score >= self._choking_hazard_risk_threshold,
                assessment.risk_score,
                now,
            )
        self._last_choking_assessments = choking_assessments

        # Human-object interaction beyond simple overlap: a tracked item's
        # TRAJECTORY closing on the child, before it ever overlaps, is a real,
        # lower-confidence, log-only signal ("about to be picked up") that a
        # single-frame overlap test can't see at all. Requires actual motion
        # toward the child (TrackedBox.approaching), not just proximity, and a
        # minimum speed so a barely-jittering stationary item never qualifies.
        if boxes:
            child_x = sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes)
            child_y = sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)
            for label in self._all_item_labels:
                approaching = False
                for track in tracks_by_label.get(label, []):
                    if any(iou(b, track.box) > 0 for b in boxes):
                        continue  # already overlapping -- that's holding_*'s signal, not this one
                    speed = track.velocity(aspect)
                    if (
                        speed is not None and speed > self._approaching_min_speed
                        and track.approaching((child_x, child_y), aspect)
                    ):
                        approaching = True
                transitions += self._engine.update(f"object_approaching:{label}", approaching, 0.5, now)
        else:
            for label in self._all_item_labels:
                transitions += self._engine.update(f"object_approaching:{label}", False, 0.0, now)

        if self._expect_presence:
            transitions += self._engine.update("person_absent", not present, 0.9, now)

        # Frantic/flailing arm movement: independent of the object/zone/stillness
        # reasoning above, fed from the SAME rolling-window pose classifier
        # fall_detection.py's activity recognizer already uses for waving/hand-
        # raised -- see core/activity.py's flailing thresholds for what actually
        # gates it (peak wrist speed/acceleration AND a direction-reversal count,
        # not a two-frame position delta).
        ready = bool(activity_state and activity_state.ready)
        flailing_present = ready and activity_state.flailing.present
        flailing_confidence = activity_state.flailing.confidence if ready else 0.0
        transitions += self._engine.update("flailing", flailing_present, flailing_confidence, now)

        _object_typed = {"holding_object", "object_approaching"}
        for tr in transitions:
            base, _, detail_suffix = tr.detection.partition(":")
            if base == "object_in_hand":
                side = detail_suffix
                label = self._current_hold_label.get(side, UNKNOWN_OBJECT)
                track_id = self._current_hold_track_id.get(side)
                if tr.state == CONFIRMED:
                    recent = self._recent_release.get(track_id) if track_id is not None else None
                    if recent is not None and recent[0] != side:
                        from_side = recent[0]
                        events.append(Event.now(
                            "object_transferred", tr.duration_s, tr.confidence, self.name,
                            f"object={label} {from_side}->{side}",
                        ))
                        self._recent_release.pop(track_id, None)
                    else:
                        events.append(Event.now(
                            "object_picked_up", tr.duration_s, tr.confidence, self.name,
                            f"object={label} hand={side}",
                        ))
                elif tr.state == RESOLVED:
                    if track_id is not None:
                        self._recent_release[track_id] = (side, now, label)
                    events.append(Event.now(
                        "object_put_down", tr.duration_s, tr.confidence, self.name,
                        f"object={label} hand={side}",
                    ))
                continue
            if base == "choking_hazard":
                side = detail_suffix
                assessment = self._last_choking_assessments.get(side)
                label = assessment.object_label if assessment else UNKNOWN_OBJECT
                if tr.state == CONFIRMED:
                    # Kept short on purpose: this text goes verbatim into the push/SMS
                    # notification body (cloud/app.py's _notification_detail prefers a
                    # module's own signals text over its generic templates), which is
                    # read in a moment of real urgency, not a debug log. The full
                    # risk/distance breakdown lives in get_debug_state() instead.
                    events.append(Event.now(
                        "choking_hazard", tr.duration_s, tr.confidence, self.name,
                        f"Small object detected near baby's mouth (object={label}, {side} hand)",
                    ))
                elif tr.state == RESOLVED:
                    events.append(Event.now(
                        "choking_hazard_resolved", tr.duration_s, tr.confidence, self.name,
                        f"object={label} hand={side}",
                    ))
                continue
            if tr.state == CONFIRMED:
                if base in _object_typed:
                    prefix = f"object={detail_suffix} "
                elif detail_suffix:
                    prefix = f"zone={detail_suffix} "
                else:
                    prefix = ""
                events.append(Event.now(
                    base, tr.duration_s, tr.confidence, self.name,
                    prefix + f"confirmed after {tr.duration_s:.1f}s",
                ))
            elif tr.state == RESOLVED:
                if base in _object_typed:
                    detail = f"object={detail_suffix}"
                else:
                    detail = f"zone={detail_suffix}" if detail_suffix else ""
                events.append(Event.now(f"{base}_resolved", tr.duration_s, tr.confidence, self.name, detail))
            # observed/developing transitions are engine-internal breadcrumbs; the
            # dashboard timeline gets only meaningful events.

        active = self._engine.active_detections()
        self._status = ("ALERT: " + ", ".join(a.split(":")[-1] for a in active)) if active else (
            "watching" if present else "no one in view"
        )
        return events

    # ------------------------------------------------------------------ display
    def annotate(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        for zone in self._zones:
            pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], dtype=np.int32)
            color = (0, 0, 220) if zone.kind == "restricted" else (0, 200, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for (x1, y1, x2, y2) in self._last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(frame, self._status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
        return frame

    def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
        """Adds the pose skeleton and live flailing kinematics on top of `annotate`."""
        frame = self.annotate(frame)
        if self._last_pose_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self._last_pose_proto, mp.solutions.pose.POSE_CONNECTIONS
            )
        if self._last_activity is not None and self._last_activity.ready:
            state = self._last_activity
            color = (0, 0, 255) if state.flailing.present else (0, 220, 220)
            signals = ", ".join(
                f"{k}={v}" for k, v in state.signals.items()
                if "peak_speed" in k or "reversals" in k
            )
            cv2.putText(frame, f"flailing={state.flailing.present} {signals}",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if state.flailing.present:
                cv2.putText(frame, "FLAILING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for track in self._item_tracker.tracks:
            x1, y1, x2, y2 = track.box
            speed = track.velocity()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            label = f"#{track.track_id} {track.label} s={track.stability:.2f}"
            if speed is not None:
                label += f" v={speed:.2f}"
            cv2.putText(frame, label, (x1, min(frame.shape[0] - 4, y2 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        if self._last_hand_result is not None:
            held_object_ids = {obj.track_id for obj, _ in self._last_hand_result.associations.values()}
            for hand in self._last_hand_result.hands:
                x1, y1, x2, y2 = (int(v) for v in hand.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"{hand.label}#{hand.track_id}", (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            for obj in self._last_hand_result.objects:
                x1, y1, x2, y2 = (int(v) for v in obj.box)
                held = obj.track_id in held_object_ids
                color = (0, 255, 0) if held else ((0, 165, 255) if obj.label == UNKNOWN_OBJECT else (0, 120, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if held else 1)
                cv2.putText(frame, f"{obj.label}#{obj.track_id} c={obj.confidence:.2f}",
                            (x1, min(frame.shape[0] - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            for hand_label, (obj_track, score) in self._last_hand_result.associations.items():
                hand = next((h for h in self._last_hand_result.hands if h.label == hand_label), None)
                if hand is None:
                    continue
                hx, hy = (int(v) for v in hand.centroid)
                ox, oy = (int(v) for v in obj_track.centroid)
                cv2.line(frame, (hx, hy), (ox, oy), (0, 255, 255), 2)
                mx, my = (hx + ox) // 2, (hy + oy) // 2
                cv2.putText(frame, f"HELD {obj_track.label} {score:.2f}", (mx, max(12, my - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return frame

    def is_alert_active(self) -> bool:
        return bool(self._engine.active_detections())

    def close(self) -> None:
        """Release the MediaPipe Pose graph (leaked per preset switch otherwise)."""
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:  # noqa: BLE001 - closing twice must be harmless
                pass

    def get_detections(self) -> DetectionFrame:
        """People seen in the most recent *detection* frame, for custom rules.

        This module only runs YOLO every `detect_every_n_frames`, reusing the last
        boxes in between, so on a skipped frame these are the boxes the module is
        itself reasoning about — deliberately the same view, not a fresher one.

        Confidence falls back to 1.0 for boxes set without it (tests and
        subclasses that stub `_detect_people`), matching how the module treats
        its own detections: as certain enough to act on.
        """
        width, height = self._last_frame_size
        detections = [
            Detection(
                label="person",
                confidence=self._last_confidences[i] if i < len(self._last_confidences) else 1.0,
                box=tuple(float(v) for v in box),
            )
            for i, box in enumerate(self._last_boxes)
        ]
        # Activity labels (currently just "flailing"), same pattern
        # fall_detection.py uses: a custom rule can target them exactly like an
        # object class, with no special case in the rule engine.
        if self._last_activity is not None and self._last_activity.ready:
            for attribute in self._last_activity.active():
                detections.append(Detection(label=attribute.name, confidence=attribute.confidence))
        return DetectionFrame(width=width, height=height, detections=tuple(detections))

    def get_telemetry_snapshot(self) -> Optional[dict]:
        flailing = self._last_activity.flailing if self._last_activity else None
        choking_risk_max = max(
            (a.risk_score for a in self._last_choking_assessments.values()), default=0.0,
        )
        return {
            "person_count": max(self._last_count, 0),
            "motion_energy": round(self._last_motion, 4),
            "active_alerts": len(self._engine.active_detections()),
            "flailing": bool(flailing and flailing.present),
            "flailing_confidence": round(flailing.confidence, 3) if flailing else 0.0,
            "hands_visible": len(self._last_hand_result.hands) if self._last_hand_result else 0,
            "objects_in_hand": len(self._last_hand_result.associations) if self._last_hand_result else 0,
            "choking_hazard_risk_max": round(choking_risk_max, 3),
            "choking_hazard_active": any(
                d.startswith("choking_hazard:") for d in self._engine.active_detections()
            ),
        }

    def get_debug_state(self) -> Optional[dict]:
        """Full, human-readable snapshot of choking-hazard risk scoring: what
        object was seen, whether it was associated with a hand, distance to
        the mouth region, movement direction, the risk score, and why an
        alert was or wasn't triggered -- one entry per hand.

        Developer/debug use only (main.py --debug), same as every other
        module's get_debug_state -- never sent to the cloud or end-user
        notification text. get_telemetry_snapshot above carries the single
        max risk number into the logged CSV stream; this carries the full
        per-hand reasoning behind it.
        """
        active = set(self._engine.active_detections())
        per_hand = {}
        for side, assessment in self._last_choking_assessments.items():
            track = None
            if self._last_hand_result is not None:
                for obj_track, _score in self._last_hand_result.associations.values():
                    if obj_track.track_id == assessment.object_track_id:
                        track = obj_track
                        break
            if assessment.approaching is True:
                direction = "toward mouth"
            elif assessment.approaching is False:
                speed = track.velocity() if track is not None else None
                direction = "away from / not toward mouth" if (speed or 0) > 0.05 else "stationary"
            else:
                direction = "unknown (insufficient trail)"
            triggered = f"choking_hazard:{side}" in active
            per_hand[side] = {
                "object_detected": assessment.held,
                "object_label": assessment.object_label,
                "associated_with_hand": assessment.held,
                "distance_to_mouth_norm": assessment.distance_to_mouth_norm,
                "movement_direction": direction,
                "size_factor": assessment.size_factor,
                "association_confidence": assessment.association_confidence,
                "risk_score": assessment.risk_score,
                "risk_threshold": self._choking_hazard_risk_threshold,
                "alert_triggered": triggered,
                "reason": assessment.reason if not triggered else "risk >= threshold, sustained",
            }
        return {"choking_hazard": per_hand, "active_alerts": sorted(active)}
