"""Occupancy module: person counting plus zone security events.

Counting uses YOLO person detection with confirm_frames debounce (as before).
Optional user-drawn zones (config `zones:`) add security events through the
shared event lifecycle (core/event_engine.py):
  * zone_violation — any person's feet inside a `restricted` zone, sustained;
  * person_lingering — one SPECIFIC tracked person's feet in the zone past
    `linger_s`.
Both resolve when their subject leaves, and brief occlusions don't reset progress.

Lingering is per-person via core/tracking.py's PersonTracker, reused here on a
coarse box-derived Skeleton (see core/keypoints.from_detection_box) rather than
running a real pose model -- this module stays the cheap, always-on detector
by design (modules/fall_detection.py is where real pose already runs). Without a
tracked identity, "lingering" is really "has this ZONE been continuously
occupied", which conflates two different four-second visits by two different
people into one eight-second "someone lingered" report the moment the second
person's arrival overlaps the first person's departure by even one frame.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from core.accel import COCO_NAMES
from core.detections import Detection, DetectionFrame
from core.detectors import get_yolo_model
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.keypoints import from_detection_box
from core.tracking import PersonTracker
from core.zones import load_zones
from modules.base import VisionModule, register_module

_PERSON_CLASS = 0  # COCO


@register_module("occupancy")
class OccupancyModule(VisionModule):
    """Detects people with YOLO11-nano and emits an event whenever the confirmed count changes."""

    name = "occupancy"

    def __init__(self, config: dict) -> None:
        """Load YOLO11-nano weights and read thresholds from config."""
        super().__init__(config)
        self._confidence_threshold = float(config["confidence_threshold"])
        self._confirm_frames = int(config["confirm_frames"])
        weights = config.get("model_weights", "yolo11n.pt")
        self._model = get_yolo_model(weights)

        self._last_emitted_count = -1
        self._pending_count: int | None = None
        self._pending_streak = 0
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._last_confidences: list[float] = []
        self._last_frame_size: tuple[int, int] = (0, 0)
        # Person (COCO 0) is this module's own business; AI-configured rules can
        # add classes on top via set_requested_labels().
        self._last_labels: list[str] = []
        self._last_person_boxes: list[tuple[int, int, int, int]] = []
        self._requested_class_ids: list[int] = [_PERSON_CLASS]
        self._extra_labels: set[str] = set()

        # Optional security zones (feet-point membership; see core/zones.py).
        self._zones = load_zones(config.get("zones", {}))
        self._engine = EventEngine({})
        self._linger_s = float(config.get("linger_s", 30.0))
        self._zone_confirm_s = float(config.get("zone_confirm_s", 2.0))
        # The zones lingering/violation logic actually watches -- "safe" zones
        # are someone else's business (modules/baby_monitoring.py).
        self._watched_zones = [z for z in self._zones if z.kind in ("restricted", "boundary")]
        for zone in self._watched_zones:
            self._engine.add_rule(RuleSpec(
                detection=f"zone_violation:{zone.name}", severity="critical",
                persist_s=self._zone_confirm_s, min_frames=3, resolve_after_s=5.0,
            ))
        # Per-person lingering identity (see module docstring). Track ids are
        # only ever added; a retired id is pruned once every zone's lingering
        # rule for it has itself resolved or been dropped, so long-running
        # lingering that outlives a brief detector dropout isn't cut short.
        self._tracker = PersonTracker()
        self._known_track_ids: set[int] = set()
        self._last_tracked_boxes: dict[int, tuple[int, int, int, int]] = {}

    def process(self, frame: np.ndarray) -> list[Event]:
        """Detect people in `frame`; emit occupancy_change once a new count is confirmed."""
        events: list[Event] = []
        results = self._model.predict(
            frame, verbose=False, classes=self._requested_class_ids, conf=self._confidence_threshold
        )

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        labels: list[str] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                boxes.append(tuple(int(v) for v in xyxy))
                confidences.append(float(box.conf[0]))
                # Class ids are only meaningful once extra classes are requested;
                # a detector shim without .cls still yields people.
                class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else _PERSON_CLASS
                labels.append(COCO_NAMES.get(class_id, str(class_id)))

        self._last_boxes = boxes
        self._last_confidences = confidences
        self._last_labels = labels
        self._last_frame_size = (frame.shape[1], frame.shape[0])

        # Occupancy means PEOPLE. Once custom rules widen the detector to dogs or
        # couches, counting every box would report a dog as an extra occupant and
        # fire occupancy_change at it.
        person_indices = [i for i, label in enumerate(labels) if label == "person"]
        self._last_person_boxes = [boxes[i] for i in person_indices]
        count = len(person_indices)
        confidences_for_count = [confidences[i] for i in person_indices]

        if count == self._pending_count:
            self._pending_streak += 1
        else:
            self._pending_count = count
            self._pending_streak = 1

        if self._pending_streak >= self._confirm_frames and count != self._last_emitted_count:
            confidence = (
                sum(confidences_for_count) / len(confidences_for_count)
                if confidences_for_count else 1.0
            )
            events.append(Event.now("occupancy_change", 0.0, confidence, self.name))
            self._last_emitted_count = count

        # Zone security events (only when a restricted/boundary zone is configured
        # -- occupancy has nothing to do with a "safe" zone, see baby_monitoring.py).
        if self._watched_zones:
            now = time.monotonic()
            h, w = frame.shape[:2]

            # Per-person identity for lingering (see module docstring): a coarse
            # box-only Skeleton reuses PersonTracker's torso-length-normalized
            # matching without paying for a pose model this module doesn't
            # otherwise run.
            skeletons = [from_detection_box(b) for b in self._last_person_boxes]
            box_by_skeleton_id = {id(s): b for s, b in zip(skeletons, self._last_person_boxes)}
            tracked = {t.track_id: t for t in self._tracker.update(skeletons, now)}
            self._known_track_ids |= set(tracked)
            self._last_tracked_boxes = {
                tid: box_by_skeleton_id[id(t.skeleton)] for tid, t in tracked.items()
            }

            transitions = []
            for zone in self._watched_zones:
                # People only: zone_violation is a person event, and a dog in the
                # zone must not raise one.
                inside = any(zone.contains_box(b, w, h) for b in self._last_person_boxes)
                transitions += self._engine.update(f"zone_violation:{zone.name}", inside, 0.85, now)

                for track_id in self._known_track_ids:
                    key = f"person_lingering:{zone.name}:{track_id}"
                    track = tracked.get(track_id)
                    if track is not None:
                        # (Re)registering every frame a track is live is cheap
                        # (a dict assignment) and means a brand-new id's first
                        # sighting always has a rule to update against.
                        self._engine.add_rule(RuleSpec(
                            detection=key, severity="warning",
                            persist_s=self._linger_s, min_frames=10, resolve_after_s=5.0,
                        ))
                        box = box_by_skeleton_id[id(track.skeleton)]
                        track_inside = zone.contains_box(box, w, h)
                    else:
                        # Gone from this frame (occluded within the tracker's own
                        # grace, or retired outright): definitely not lingering.
                        # Keep feeding False rather than skipping, so a CONFIRMED
                        # track still gets its resolve_after_s and a real
                        # person_lingering_resolved event, instead of the rule
                        # being pruned out from under it below.
                        track_inside = False
                    transitions += self._engine.update(key, track_inside, 0.85, now)

            # Prune ids the engine has itself fully forgotten (resolved-then-
            # deleted, or dropped without ever confirming) so `_known_track_ids`
            # and the engine's own rule table don't grow for the life of the
            # process as visitors come and go.
            for track_id in list(self._known_track_ids):
                if track_id in tracked:
                    continue
                if all(
                    self._engine.state_of(f"person_lingering:{z.name}:{track_id}") is None
                    for z in self._watched_zones
                ):
                    for z in self._watched_zones:
                        self._engine.remove_rule(f"person_lingering:{z.name}:{track_id}")
                    self._known_track_ids.discard(track_id)

            for tr in transitions:
                base, _, rest = tr.detection.partition(":")
                zone_name = rest.split(":", 1)[0]
                if tr.state == CONFIRMED:
                    events.append(Event.now(
                        base, tr.duration_s, tr.confidence, self.name,
                        f"zone={zone_name} confirmed after {tr.duration_s:.1f}s",
                    ))
                elif tr.state == RESOLVED:
                    events.append(Event.now(f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                                            f"zone={zone_name}"))

        return events

    def is_alert_active(self) -> bool:
        """True while any zone violation is confirmed and unresolved."""
        return bool(self._engine.active_detections())

    def set_requested_labels(self, labels: set[str]) -> None:
        """Ask the detector for these classes too, on top of people.

        Unknown labels are dropped rather than raising: the capability layer has
        already refused anything the detector cannot name, and a stray label
        must not take the camera down.
        """
        name_to_id = {name: class_id for class_id, name in COCO_NAMES.items()}
        extra = {name_to_id[label] for label in labels if label in name_to_id}
        self._extra_labels = {label for label in labels if label in name_to_id}
        self._requested_class_ids = sorted({_PERSON_CLASS, *extra})

    def get_detections(self) -> DetectionFrame:
        """This frame's detections, with boxes, so zone/count/relation rules can evaluate."""
        width, height = self._last_frame_size
        return DetectionFrame(
            width=width,
            height=height,
            detections=tuple(
                Detection(label=label, confidence=conf, box=tuple(float(v) for v in box))
                # zip truncates if a detector ever returns mismatched lists, which
                # is the safe direction: a missing score must not fabricate a box.
                for box, conf, label in zip(self._last_boxes, self._last_confidences, self._last_labels)
            ),
        )

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw zones, and each detection labelled with what it actually is.

        Every box used to be drawn in one colour under a "People: N" header,
        because this began as a person counter. It is now also the detector
        behind AI custom monitoring, where the whole point is that someone asked
        about bottles or phones — so a bottle appeared in a box captioned
        "People", which is the picture contradicting the notification sent about
        the same frame.

        The labels were always available; annotate simply threw them away.
        """
        h, w = frame.shape[:2]
        for zone in self._zones:
            pts = np.array([[int(x * w), int(y * h)] for x, y in zone.polygon], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 220), thickness=2)
            cv2.putText(frame, zone.name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

        # Per-person track id (only populated when zones are configured — see
        # process()), so "3 people entered" reads as Person 1/2/3 rather than
        # three identical, indistinguishable boxes.
        for track_id, (x1, y1, x2, y2) in self._last_tracked_boxes.items():
            cv2.putText(frame, f"#{track_id}", (x1, min(h - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        labels = self._last_labels or []
        for index, (x1, y1, x2, y2) in enumerate(self._last_boxes):
            label = labels[index] if index < len(labels) else "object"
            # What someone explicitly asked about is drawn in the alert colour,
            # so the frame agrees with the alert rather than needing to be read
            # alongside it.
            requested = label in self._extra_labels
            colour = (0, 0, 235) if requested else (0, 200, 0)   # BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if requested else 2)
            caption = label.title()
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            # Filled plate behind the text: a thin label over a busy scene is
            # unreadable exactly when someone is trying to check what fired.
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), colour, -1)
            cv2.putText(frame, caption, (x1 + 4, max(th, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.putText(frame, self._headline(labels), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        return frame

    def _headline(self, labels: list[str]) -> str:
        """A caption describing what is actually on screen.

        "People: 0" while a bottle sits in a box is worse than no caption: it is
        a specific, confident, wrong claim about the picture underneath it.
        """
        if self._extra_labels:
            counts = {
                name: sum(1 for label in labels if label == name)
                for name in sorted(self._extra_labels)
            }
            return "  ".join(f"{name.title()}: {n}" for name, n in counts.items())
        return f"People: {sum(1 for label in labels if label == 'person')}"
