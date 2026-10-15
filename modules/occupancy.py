"""Occupancy module: person counting plus zone security events.

Counting uses YOLO person detection with confirm_frames debounce (as before).
Optional user-drawn zones (config `zones:`) add security events through the
shared event lifecycle (core/event_engine.py):
  * zone_violation — any person's feet inside a `restricted` zone, sustained;
  * person_lingering — one SPECIFIC tracked person's feet in the zone past
    `linger_s`;
  * vehicle_in_zone / animal_in_zone — a car/truck/... or a dog/cat/... inside
    the zone, sustained. The vocabulary comes from core.capabilities'
    `expand("vehicle")`/`expand("animal")` category words, the SAME lookup
    Jupiter uses, so a class added there is picked up here with no separate
    list to maintain -- see core/taxonomy.py's former "not wired into the
    security event set yet" caveat on both.
All resolve when their subject leaves, and brief occlusions don't reset progress.

crowd is the one event in this module that does NOT need a zone: core/taxonomy.py
has always documented it as "person-count threshold", but no code here ever
implemented it -- the taxonomy (and therefore docs/DETECTIONS.md, generated from
it) claimed a capability that silently never fired. `crowd_threshold` confirmed
people on screen at once, sustained `crowd_confirm_s`, resolves once the count
drops back below threshold -- the same confirm/resolve lifecycle as the zone
events above, just not gated on anyone having drawn an area.

Lingering is per-person via core/tracking.py's PersonTracker, reused here on a
coarse box-derived Skeleton (see core/keypoints.from_detection_box) rather than
running a real pose model -- this module stays the cheap, always-on detector
by design (modules/fall_detection.py is where real pose already runs). Without a
tracked identity, "lingering" is really "has this ZONE been continuously
occupied", which conflates two different four-second visits by two different
people into one eight-second "someone lingered" report the moment the second
person's arrival overlaps the first person's departure by even one frame.

Optional cross-model verification (`verify_new_tracks_with_pose`, off by
default): the first time a new track appears, a second, independent model
(MediaPipe Pose, on just that box's crop) checks for a real human pose --
genuine corroborating evidence for the detector's own "person" label, from a
signal a box-only classifier cannot produce about itself. It is surfaced as
an annotation on `person_lingering`'s event detail, never used to suppress an
alert: a person facing away or poorly lit can genuinely have no detectable
pose, and gating a security alert on that absence would trade a real signal
against a false-negative risk with no measured data behind it to tune
against. See _pose_confirms_person.
"""

from __future__ import annotations

import contextlib
import time

import cv2
import numpy as np

from core.accel import COCO_NAMES
from core.capabilities import registry as capability_registry
from core.detections import Detection, DetectionFrame
from core.detectors import get_yolo_model
from core.event_engine import CONFIRMED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.keypoints import from_detection_box
from core.tracking import PersonTracker
from core.zones import BoundaryCrossing, load_zones
from modules.base import VisionModule, register_module

_PERSON_CLASS = 0  # COCO
# Zone-watched object categories beyond people. Kept to two: these are the
# ones core/taxonomy.py already documented as "the classes exist, just not
# wired into the security event set" -- everything else PARTIAL/UNSUPPORTED
# there (package, door state, smoke/fire, ...) needs a model this repo
# doesn't have, not more wiring.
_ZONE_CATEGORIES = ("vehicle", "animal")


@register_module("occupancy")
class OccupancyModule(VisionModule):
    """Detects people with YOLO11-nano and emits an event whenever the confirmed count changes."""

    name = "occupancy"

    def __init__(self, config: dict) -> None:
        """Load YOLO11-nano weights and read thresholds from config."""
        super().__init__(config)
        self._confidence_threshold = float(config["confidence_threshold"])
        self._confirm_frames = int(config["confirm_frames"])
        # Loaded on first use, not here. Every other module in this repo already
        # defers its detector load; occupancy was the lone eager one, and that
        # inconsistency had teeth: a detector failure inside __init__ happens
        # before DeviceRuntime exists, so it bypasses the degraded-mode handling
        # entirely and becomes an un-degradable startup crash. Loading lazily
        # puts occupancy on the same path as the rest, where a missing backend is
        # reported as "monitoring NOT active" instead of restart-looping.
        self._weights = config.get("model_weights", "yolo11n.pt")
        self._model = None

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
        # What the detector is asked for is three independent contributions,
        # recombined by _refresh_requested_classes: people (always), the
        # vehicle/animal classes the zone rules need (only while a zone exists),
        # and whatever AI-configured rules asked for (set_requested_labels).
        # set_requested_labels used to OVERWRITE the list with person + rule
        # labels, which dropped the zone classes. After any AI-rule session on
        # the Security preset (or a rules update to `[]`), vehicle_in_zone and
        # animal_in_zone silently never fired again until a restart.
        self._zone_class_ids: set[int] = set()
        self._extra_class_ids: set[int] = set()

        # Optional security zones (feet-point membership; see core/zones.py).
        self._zones: list = []
        self._watched_zones: list = []
        self._boundary_watch: dict[str, BoundaryCrossing] = {}
        self._label_to_zone_category: dict[str, str] = {}
        self._engine = EventEngine({})
        self._linger_s = float(config.get("linger_s", 30.0))
        self._zone_confirm_s = float(config.get("zone_confirm_s", 2.0))

        # crowd: unusual person count, sustained -- independent of any zone
        # (see module docstring). Always registered, not gated on zones being
        # configured, since "how many people are here at once" needs nothing
        # a user has to draw first.
        self._crowd_threshold = int(config.get("crowd_threshold", 4))
        self._crowd_confirm_s = float(config.get("crowd_confirm_s", 5.0))
        self._engine.add_rule(RuleSpec(
            detection="crowd", severity="warning",
            persist_s=self._crowd_confirm_s, min_frames=3, resolve_after_s=5.0,
        ))
        # Per-person lingering identity (see module docstring). Track ids are
        # only ever added; a retired id is pruned once every zone's lingering
        # rule for it has itself resolved or been dropped, so long-running
        # lingering that outlives a brief detector dropout isn't cut short.
        self._tracker = PersonTracker()
        self._known_track_ids: set[int] = set()
        self._last_tracked_boxes: dict[int, tuple[int, int, int, int]] = {}

        # Zone rules: restricted -> zone_violation / vehicle_in_zone /
        # animal_in_zone / person_lingering; boundary -> a crossing watch
        # (BoundaryCrossing, reported as zone_violation "crossed": a line
        # alerts on CROSSING it, not on standing inside it -- it used to be
        # treated as one more restricted zone). "safe" zones are someone
        # else's business (modules/baby_monitoring.py). Installed here so
        # set_zones() can replace them in place.
        self._install_zones(load_zones(config.get("zones", {})))

        # Optional cross-model verification (off by default -- see
        # _pose_confirms_person). A brand-new track, the first time it's
        # seen, gets ONE MediaPipe Pose check on its own crop: a second,
        # independent model corroborating (or not) the detector's own
        # "person" label. Event-driven -- once per track, not every frame --
        # and opt-in, since it's a real model this module was deliberately
        # built without (see module docstring); nothing about its footprint
        # changes unless a deployment explicitly asks for the corroboration.
        self._verify_new_tracks = bool(config.get("verify_new_tracks_with_pose", False))
        self._pose_verifier = None
        self._track_pose_confirmed: dict[int, bool] = {}

    # ------------------------------------------------------------------ zones
    def _install_zones(self, zones: list) -> None:
        """Make `zones` this module's zones, replacing the previous set in place.

        A restricted zone whose name, kind and polygon are unchanged keeps all
        its rules and state. A removed or redrawn one has every rule under it
        dropped from the engine, with tracks and cooldowns: zone_violation,
        vehicle/animal_in_zone, and person_lingering for every known track.
        Nothing stays latched. Counting, crowd and the person tracker are
        untouched.
        """
        old_zones, new_zones = list(self._zones), list(zones)
        old_watched = {z for z in old_zones if z.kind == "restricted"}
        new_watched = {z for z in new_zones if z.kind == "restricted"}
        for zone in old_watched - new_watched:
            self._retire_zone_rule(f"zone_violation:{zone.name}")
            for category in _ZONE_CATEGORIES:
                self._retire_zone_rule(f"{category}_in_zone:{zone.name}")
            for track_id in self._known_track_ids:
                self._retire_zone_rule(f"person_lingering:{zone.name}:{track_id}")
        for zone in new_watched - old_watched:
            self._engine.add_rule(RuleSpec(
                detection=f"zone_violation:{zone.name}", severity="critical",
                persist_s=self._zone_confirm_s, min_frames=3, resolve_after_s=5.0,
            ))
            for category in _ZONE_CATEGORIES:
                self._engine.add_rule(RuleSpec(
                    detection=f"{category}_in_zone:{zone.name}", severity="warning",
                    persist_s=self._zone_confirm_s, min_frames=3, resolve_after_s=5.0,
                ))
        self._boundary_watch = {
            z.name: (
                self._boundary_watch[z.name]
                if z in old_zones and z.name in self._boundary_watch
                else BoundaryCrossing(self._zone_confirm_s)
            )
            for z in new_zones if z.kind == "boundary"
        }
        self._zones = new_zones
        self._watched_zones = [z for z in new_zones if z.kind == "restricted"]
        if not self._watched_zones:
            self._known_track_ids.clear()
            self._last_tracked_boxes = {}

        # Vehicle/animal-in-zone: same zone engine, a category word's real
        # labels resolved from the live capability registry rather than a
        # second hardcoded list (see module docstring). Only requested from
        # the detector when there's a zone to test them against -- a bare
        # person-counting config pays nothing for this.
        self._label_to_zone_category = {}
        self._zone_class_ids = set()
        if self._watched_zones:
            reg = capability_registry()
            for category in _ZONE_CATEGORIES:
                for label in reg.expand(category):
                    self._label_to_zone_category[label] = category
            name_to_id = {name: class_id for class_id, name in COCO_NAMES.items()}
            self._zone_class_ids = {
                name_to_id[label] for label in self._label_to_zone_category if label in name_to_id
            }
        self._refresh_requested_classes()

    def _retire_zone_rule(self, detection: str) -> None:
        """Withdraw a zone's rule. An alert confirmed under it is RESOLVED, not
        silently dropped -- reported on the next frame by process() -- so a zone
        redrawn in the app mid-violation does not leave the alert "active" in
        the app and dashboard for good."""
        transition = self._engine.retire_rule(detection)
        if transition is not None:
            self.__dict__.setdefault("_retired_zone_transitions", []).append(transition)

    def _take_retired_zone_events(self) -> list[Event]:
        """The resolutions queued by _retire_zone_rule, as events, once."""
        transitions = self.__dict__.pop("_retired_zone_transitions", None) or []
        events = []
        for tr in transitions:
            # "zone_violation:pool", "safe_zone_exit", "person_lingering:pool:7"
            base, _, rest = tr.detection.partition(":")
            zone = rest.split(":", 1)[0]
            events.append(Event.now(
                f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                f"zone={zone}" if zone else "",
            ))
        return events

    def _refresh_requested_classes(self) -> None:
        # getattr: tolerant of an instance built without __init__ (tests do that
        # to avoid loading a detector); absent means "nothing requested".
        zone_ids = getattr(self, "_zone_class_ids", set())
        extra_ids = getattr(self, "_extra_class_ids", set())
        self._requested_class_ids = sorted({_PERSON_CLASS, *zone_ids, *extra_ids})

    def set_zones(self, zones: list) -> None:
        """Replace the configured zones with the app-drawn ones. See VisionModule.set_zones."""
        self._install_zones(list(zones or []))

    def _ensure_pose_verifier(self) -> None:
        if self._pose_verifier is None:
            import mediapipe as mp

            # static_image_mode=True: each call below is an isolated crop of
            # a different person at a different moment, never a continuous
            # stream -- the opposite of fall_detection.py's own Pose usage,
            # which deliberately leaves this False to track one subject
            # across frames. Sharing that setting here would let MediaPipe's
            # internal frame-to-frame smoothing carry state between unrelated
            # people, which is wrong for what this is.
            self._pose_verifier = mp.solutions.pose.Pose(
                static_image_mode=True, min_detection_confidence=0.5,
            )

    def _pose_confirms_person(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        """Whether a SECOND, independent model finds a human pose in `box`.

        Corroborating evidence for the detector's own "person" label, from a
        completely different signal (body-landmark geometry, not the object
        detector's own classifier) -- the actual "detector A says person,
        other evidence disagrees" case a box-only detector cannot answer
        about itself. A margin is added around the box because a tight YOLO
        box can crop through a limb MediaPipe would otherwise have found.

        Absence of a pose is real evidence, not proof: a person facing away,
        poorly lit, or partly out of frame can genuinely have no detectable
        pose. That asymmetry is why this is surfaced as an annotation on
        `person_lingering`'s event detail rather than used to suppress an
        alert outright -- gating a security alert on this would trade a
        real, if imperfect, signal against a false-negative risk with no
        measured data behind it to tune against. See the module docstring.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        margin_x, margin_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return False
        self._ensure_pose_verifier()
        crop_rgb = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        return self._pose_verifier.process(crop_rgb).pose_landmarks is not None

    def process(self, frame: np.ndarray) -> list[Event]:
        """Detect people in `frame`; emit occupancy_change once a new count is confirmed."""
        return self._take_retired_zone_events() + self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> list[Event]:
        """One frame's work; see process()."""
        events: list[Event] = []
        if self._model is None:
            self._model = get_yolo_model(self._weights)
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

        category_boxes: dict[str, list[tuple[int, int, int, int]]] = {}
        for i, label in enumerate(labels):
            category = self._label_to_zone_category.get(label)
            if category is not None:
                category_boxes.setdefault(category, []).append(boxes[i])

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

        now = time.monotonic()

        # crowd: sustained person count over threshold, independent of any zone.
        for tr in self._engine.update("crowd", count >= self._crowd_threshold, 0.85, now):
            if tr.state == CONFIRMED:
                events.append(Event.now(
                    "crowd", tr.duration_s, tr.confidence, self.name,
                    f"{count} people present (threshold {self._crowd_threshold})",
                ))
            elif tr.state == RESOLVED:
                events.append(Event.now("crowd_resolved", tr.duration_s, tr.confidence, self.name))

        # Boundary zones: a settled change of side, in either direction, by
        # anyone's feet. Nobody in view is no observation, not a crossing.
        if self._boundary_watch:
            h, w = frame.shape[:2]
            for zone in self._zones:
                if zone.kind != "boundary":
                    continue
                inside = (
                    any(zone.contains_box(b, w, h) for b in self._last_person_boxes)
                    if self._last_person_boxes else None
                )
                crossing = self._boundary_watch[zone.name].update(inside, now)
                if crossing is not None:
                    events.append(Event.now(
                        "zone_violation", 0.0, 0.85, self.name, f"zone={zone.name} crossed ({crossing})",
                    ))

        # Zone security events (only when a restricted zone is configured --
        # occupancy has nothing to do with a "safe" zone, see baby_monitoring.py).
        if self._watched_zones:
            h, w = frame.shape[:2]

            # Per-person identity for lingering (see module docstring): a coarse
            # box-only Skeleton reuses PersonTracker's torso-length-normalized
            # matching without paying for a pose model this module doesn't
            # otherwise run.
            skeletons = [from_detection_box(b) for b in self._last_person_boxes]
            box_by_skeleton_id = {
                id(s): b for s, b in zip(skeletons, self._last_person_boxes, strict=True)
            }
            tracked = {t.track_id: t for t in self._tracker.update(skeletons, now)}
            newly_appeared_tracks = set(tracked) - self._known_track_ids
            self._known_track_ids |= set(tracked)
            self._last_tracked_boxes = {
                tid: box_by_skeleton_id[id(t.skeleton)] for tid, t in tracked.items()
            }
            if self._verify_new_tracks:
                for track_id in newly_appeared_tracks:
                    box = box_by_skeleton_id[id(tracked[track_id].skeleton)]
                    self._track_pose_confirmed[track_id] = self._pose_confirms_person(frame, box)

            transitions = []
            for zone in self._watched_zones:
                # People only: zone_violation is a person event, and a dog in the
                # zone must not raise one.
                inside = any(zone.contains_box(b, w, h) for b in self._last_person_boxes)
                transitions += self._engine.update(f"zone_violation:{zone.name}", inside, 0.85, now)

                for category in _ZONE_CATEGORIES:
                    cat_inside = any(
                        zone.contains_box(b, w, h) for b in category_boxes.get(category, [])
                    )
                    transitions += self._engine.update(
                        f"{category}_in_zone:{zone.name}", cat_inside, 0.8, now
                    )

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
                    self._track_pose_confirmed.pop(track_id, None)

            for tr in transitions:
                base, _, rest = tr.detection.partition(":")
                parts = rest.split(":", 1)
                zone_name = parts[0]
                if tr.state == CONFIRMED:
                    detail = f"zone={zone_name} confirmed after {tr.duration_s:.1f}s"
                    # Only person_lingering has a track_id in `rest` (zone:id) --
                    # zone_violation/vehicle_in_zone/animal_in_zone are zone-
                    # aggregate and have no single subject to attribute this to.
                    if base == "person_lingering" and len(parts) > 1 and self._verify_new_tracks:
                        track_id = int(parts[1])
                        if self._track_pose_confirmed.get(track_id) is False:
                            detail += " (no body pose confirmed for this detection — worth a manual check)"
                    events.append(Event.now(base, tr.duration_s, tr.confidence, self.name, detail))
                elif tr.state == RESOLVED:
                    events.append(Event.now(f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                                            f"zone={zone_name}"))

        return events

    def is_alert_active(self) -> bool:
        """True while any zone violation is confirmed and unresolved."""
        return bool(self._engine.active_detections())

    def close(self) -> None:
        """Release the pose-verification MediaPipe graph, if it was ever created
        (opt-in via verify_new_tracks_with_pose -- most deployments never build
        one). Leaked per preset/mode switch otherwise: found in a repo-wide
        audit -- this is one of the two modules _apply_monitoring_mode/
        _switch_preset hot-swap between, and both already call
        old_module.close(), but this override was missing."""
        if self._pose_verifier is not None:
            with contextlib.suppress(Exception):  # closing twice must be harmless
                self._pose_verifier.close()

    def set_requested_labels(self, labels: set[str]) -> None:
        """Ask the detector for these classes too, on top of people.

        Unknown labels are dropped rather than raising: the capability layer has
        already refused anything the detector cannot name, and a stray label
        must not take the camera down.
        """
        name_to_id = {name: class_id for class_id, name in COCO_NAMES.items()}
        self._extra_class_ids = {name_to_id[label] for label in labels if label in name_to_id}
        self._extra_labels = {label for label in labels if label in name_to_id}
        # Added to, never replacing, the zone rules' own vehicle/animal classes.
        self._refresh_requested_classes()

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
                for box, conf, label in zip(
                    self._last_boxes, self._last_confidences, self._last_labels, strict=False
                )
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
        for track_id, (x1, _y1, _x2, y2) in self._last_tracked_boxes.items():
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
