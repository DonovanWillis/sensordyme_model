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
  specific about it, which is why it's safe to claim where face state isn't).

  DELIBERATELY NOT CLAIMED: face-covered / face-down detection (adult face
  models are not reliable enough on infants for a safety-critical claim),
  sleeping vs awake, crying (no microphone), or anything requiring a
  small-object vocabulary beyond what YOLO actually has classes for (COCO has
  no "choking hazard" class, and a curated subset would create false
  confidence about coverage it doesn't have — see core/capabilities.py's
  objects.small_personal for the same reasoning applied product-wide).
  Faking any of these would be worse than omitting them.

All temporal reasoning goes through core/event_engine.py: nothing here alerts
on a single frame, brief occlusions don't reset progress, and every alert has
an observed → developing → confirmed lifecycle with a resolution.
"""

from __future__ import annotations

import time
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
from core.detectors import get_yolo_model, iou
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.tracking import BoxTracker
from core.zones import Zone, load_zones
from modules.base import VisionModule, register_module

_PERSON_CLASS = 0  # COCO
_BOTTLE_CLASS = next(cid for cid, name in COCO_NAMES.items() if name == "bottle")

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
        self._last_activity: Optional[ActivityState] = None
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
            # The subject left (or the pose model lost them): motion history
            # describes someone who may no longer be there and would misclassify
            # whoever appears next -- same reasoning as fall_detection.py's reset.
            self._activity.reset()
            return None
        self._last_pose_proto = results.pose_landmarks
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

        if self._pose is not None and self._frame_index % self._pose_every_n == 0:
            self._last_activity = self._update_activity(frame, w, h, now)

        return self._reason(
            boxes, self._last_motion, w, h, now, self._last_bottle_boxes, self._last_object_boxes,
            self._last_activity,
        )

    def _reason(self, boxes: list, motion: float, w: int, h: int, now: float,
                bottle_boxes: list = (), object_boxes: dict = None,
                activity_state: Optional[ActivityState] = None) -> list[Event]:
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

        # Bottle overlapping the child's box: an approximation of "holding", not
        # proof (a bottle propped beside them in a bouncer would also match) --
        # the same box-overlap approximation core/rules.py's `holding` predicate
        # already uses and discloses. Bare bottle presence elsewhere in frame is
        # NOT this signal on purpose: "bottle detected" alone isn't meaningful.
        holding_bottle, bottle_confidence = _holding("bottle")
        transitions += self._engine.update("holding_bottle", holding_bottle, bottle_confidence, now)

        # Same overlap approximation, generalized to every other configured class
        # (phone, cup, food, ...) -- see DEFAULT_OBJECT_CLASSES. Each label gets
        # its OWN engine key (holding_object:<label>) and is updated every frame
        # regardless of whether that label was seen at all this frame -- a phone
        # AND a cup can both be held at once, each independently confirms/
        # resolves/cools down, and picking up a different object right after
        # putting one down is a fresh observation, not suppressed by the first
        # object's own cooldown (see holding_cooldown_s above).
        for label in self._object_labels:
            holding, confidence = _holding(label)
            transitions += self._engine.update(f"holding_object:{label}", holding, confidence, now)

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
        return {
            "person_count": max(self._last_count, 0),
            "motion_energy": round(self._last_motion, 4),
            "active_alerts": len(self._engine.active_detections()),
            "flailing": bool(flailing and flailing.present),
            "flailing_confidence": round(flailing.confidence, 3) if flailing else 0.0,
        }
