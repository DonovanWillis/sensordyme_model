"""Keeping the same person the same person across frames.

Without this every frame is a stranger, and nothing temporal is possible: "this
person has been on the floor for 30 seconds" requires knowing it is one person
for 30 seconds rather than 900 unrelated observations.

The matcher is deliberately simple — nearest centroid, gated by a distance
expressed in torso lengths. Appearance-based re-identification is a much larger
problem, needs a model, and would buy accuracy SensorDyme does not yet need: a
home camera sees one to four people, well separated, and the failure mode of
this matcher (two IDs swapping when people cross) is visible and bounded.

Occlusion is handled by patience. A track whose person vanishes is kept alive
for `lost_grace_s` before being retired, because someone walking behind a sofa
is not someone leaving — and retiring the track would restart every timer that
matters.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Optional

from core.keypoints import Skeleton

#: How far a person may move between frames and still be the same person,
#: measured in torso lengths so it holds at any distance from the camera.
MAX_MATCH_DISTANCE_TORSO = 1.5

#: How long a vanished person stays tracked. Long enough to walk behind
#: furniture; short enough that the ID does not outlive the visit.
LOST_GRACE_S = 2.0


@dataclass
class TrackedPerson:
    """One person, followed across frames."""

    track_id: int
    skeleton: Skeleton
    first_seen: float
    last_seen: float
    #: Frames this track has been matched in — a crude confidence in its reality.
    hits: int = 1
    #: Recent centroids, for velocity and for anything wanting a short history.
    trail: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def centroid(self) -> Optional[tuple[float, float]]:
        return self.skeleton.center_of_mass()

    def velocity(self, aspect: float = 1.0) -> Optional[float]:
        """Centroid speed in torso lengths per second, or None if unknowable."""
        if len(self.trail) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = self.trail[0], self.trail[-1]
        elapsed = t1 - t0
        torso = self.skeleton.torso_length()
        if elapsed <= 0 or not torso:
            return None
        return round(math.hypot((x1 - x0) * aspect, y1 - y0) / torso / elapsed, 2)

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "age_s": round(self.age_s, 1),
            "hits": self.hits,
            "coverage": self.skeleton.coverage,
            "geometry": self.skeleton.geometry(),
        }


class PersonTracker:
    """Assigns stable ids to skeletons across frames."""

    def __init__(
        self,
        max_distance: float = MAX_MATCH_DISTANCE_TORSO,
        lost_grace_s: float = LOST_GRACE_S,
    ) -> None:
        self.max_distance = max_distance
        self.lost_grace_s = lost_grace_s
        self._tracks: dict[int, TrackedPerson] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[TrackedPerson]:
        return list(self._tracks.values())

    def active(self, now: Optional[float] = None) -> list[TrackedPerson]:
        """Tracks seen in this frame, newest activity first."""
        now = time.monotonic() if now is None else now
        return sorted(
            (t for t in self._tracks.values() if now - t.last_seen < 1e-6),
            key=lambda t: t.track_id,
        )

    def update(self, skeletons: list[Skeleton], now: Optional[float] = None) -> list[TrackedPerson]:
        """Match this frame's skeletons to existing tracks; return the matched set."""
        now = time.monotonic() if now is None else now
        self._retire(now)

        # Only skeletons with a usable centroid can be matched at all.
        candidates = [(s, s.center_of_mass()) for s in skeletons]
        candidates = [(s, c) for s, c in candidates if c is not None]

        unmatched_tracks = dict(self._tracks)
        matched: list[TrackedPerson] = []

        # Greedy nearest-neighbour over every (track, skeleton) pair, closest
        # first. With a handful of people this is exact enough and avoids
        # dragging in an assignment solver for a four-element problem.
        pairs = []
        for skeleton, centroid in candidates:
            for track_id, track in unmatched_tracks.items():
                distance = self._distance(track, skeleton, centroid)
                if distance is not None and distance <= self.max_distance:
                    pairs.append((distance, track_id, id(skeleton)))
        pairs.sort()

        used_tracks: set[int] = set()
        used_skeletons: set[int] = set()
        by_identity = {id(s): (s, c) for s, c in candidates}

        for _distance, track_id, skeleton_key in pairs:
            if track_id in used_tracks or skeleton_key in used_skeletons:
                continue
            skeleton, centroid = by_identity[skeleton_key]
            track = self._tracks[track_id]
            track.skeleton = skeleton
            track.last_seen = now
            track.hits += 1
            track.trail.append((now, centroid[0], centroid[1]))
            # Bounded: enough for a velocity estimate, not a recording.
            if len(track.trail) > 30:
                track.trail.pop(0)
            used_tracks.add(track_id)
            used_skeletons.add(skeleton_key)
            matched.append(track)

        # Anything left is someone new.
        for skeleton, centroid in candidates:
            if id(skeleton) in used_skeletons:
                continue
            track = TrackedPerson(
                track_id=self._next_id, skeleton=skeleton,
                first_seen=now, last_seen=now,
                trail=[(now, centroid[0], centroid[1])],
            )
            self._tracks[self._next_id] = track
            self._next_id += 1
            matched.append(track)

        return sorted(matched, key=lambda t: t.track_id)

    def _distance(self, track: TrackedPerson, skeleton: Skeleton,
                  centroid: tuple[float, float]) -> Optional[float]:
        """Centroid separation in torso lengths, or None if not comparable."""
        previous = track.centroid
        if previous is None:
            return None
        # Scale by whichever torso is measurable; a person mid-fall may have a
        # foreshortened one, and using the larger is the forgiving choice.
        torso = max(
            track.skeleton.torso_length() or 0.0,
            skeleton.torso_length() or 0.0,
        )
        if torso <= 1e-4:
            return None
        aspect = skeleton.aspect
        return math.hypot(
            (centroid[0] - previous[0]) * aspect, centroid[1] - previous[1]
        ) / torso

    def _retire(self, now: float) -> None:
        """Drop tracks whose person has been gone longer than the grace period."""
        for track_id in [
            tid for tid, t in self._tracks.items() if now - t.last_seen > self.lost_grace_s
        ]:
            del self._tracks[track_id]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


#: BoxTracker's matching gate, in box-diagonal lengths (the size reference
#: available with no pose skeleton) -- looser than PersonTracker's 1.5 torso-
#: lengths since held/handled objects can move faster relative to their own
#: size than a walking person does relative to their torso.
MAX_MATCH_DISTANCE_DIAGONAL = 1.2
#: The same gate for a BARE HAND, which is the fastest thing any of these
#: trackers has to follow relative to its own size: a hand is small (its box
#: diagonal is a fraction of a torso's) while its reachable envelope is a whole
#: arm, so a genuinely frantic hand covers well over one of its own diagonals
#: between two samples -- and Driver Mode samples hands only on the object
#: detector's stride (yolo_every_n_frames), which multiplies the gap further.
#: At the default 1.2, exactly the motion "erratic hand movement" exists to
#: catch was the motion that broke the association: every tick spawned a fresh
#: track with an empty trail, so the trajectory this feature reasons about
#: never existed. The looser gate's cost is the documented failure mode of a
#: centroid matcher -- two same-class boxes can swap ids when they cross --
#: bounded here by greedy nearest-first assignment (with both hands detected,
#: the closest pairing still wins) and by there being at most two of them.
MAX_MATCH_DISTANCE_DIAGONAL_HAND = 3.0
#: How long a vanished object stays tracked before its id is retired.
BOX_LOST_GRACE_S = 2.0

Box = tuple[float, float, float, float]


@dataclass
class TrackedBox:
    """One detected object (any class), followed across frames.

    The object-identity counterpart to TrackedPerson, for modules that reason
    about held/nearby items (a phone, a bottle, a cup) where no pose skeleton
    exists to supply a torso-length scale reference -- box diagonal stands in.
    """

    track_id: int
    label: str
    box: Box
    confidence: float
    first_seen: float
    last_seen: float
    #: Frames this track has been matched in -- feeds `stability` below.
    hits: int = 1
    #: Recent centroids, for velocity/direction and a short history.
    trail: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def age_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def diagonal(self) -> float:
        x1, y1, x2, y2 = self.box
        return math.hypot(x2 - x1, y2 - y1)

    @property
    def stability(self) -> float:
        """A crude [0, 1] confidence that this track is real and not a single-
        frame flicker -- rises with both how many frames it's matched in and
        how long it's persisted, saturating quickly so a brand-new track isn't
        zero (min_frames/persist_s in the event engine already gate whether a
        FRESH track can confirm anything; this is a continuous complement to
        that binary gate, for confidence fusion -- see modules/baby_monitoring.py)."""
        return round(min(1.0, 0.3 + 0.1 * min(self.hits, 5) + 0.2 * min(self.age_s, 1.0)), 2)

    def velocity(self, aspect: float = 1.0) -> Optional[float]:
        """Centroid speed in box-diagonals per second, or None if unknowable."""
        if len(self.trail) < 2:
            return None
        (t0, x0, y0), (t1, x1, y1) = self.trail[0], self.trail[-1]
        elapsed = t1 - t0
        diagonal = self.diagonal
        if elapsed <= 0 or diagonal <= 1e-6:
            return None
        return round(math.hypot((x1 - x0) * aspect, y1 - y0) / diagonal / elapsed, 3)

    def peak_speed(self, aspect: float = 1.0) -> Optional[float]:
        """Highest INSTANTANEOUS centroid speed over the trail, in box-diagonals
        per second -- the fastest single step, not `velocity`'s net displacement
        from trail start to trail end divided by the elapsed time.

        The two answer different questions and only one of them can describe
        oscillation. Net displacement is the right measure for "is this thing
        travelling somewhere" (velocity's callers: is a held object closing on a
        mouth, is a person crossing a room). It is structurally blind to motion
        that returns to where it started -- which is the DEFINITION of the
        chaotic back-and-forth that `reversals` counts, so pairing the two in
        one AND condition (fast AND reversing) made the pair near-mutually-
        exclusive: whatever reversed enough to clear the reversal bar had, by
        reversing, cancelled its own net displacement to nearly zero.

        This mirrors core/activity.py's flailing detector, which has always
        measured intensity as the max of per-step speeds for exactly this
        reason; it is that primitive made available to any tracked box.
        """
        if len(self.trail) < 2:
            return None
        diagonal = self.diagonal
        if diagonal <= 1e-6:
            return None
        peak: Optional[float] = None
        for (t0, x0, y0), (t1, x1, y1) in pairwise(self.trail):
            dt = t1 - t0
            if dt <= 0:
                continue
            step = math.hypot((x1 - x0) * aspect, y1 - y0) / diagonal / dt
            if peak is None or step > peak:
                peak = step
        return None if peak is None else round(peak, 3)

    def approaching(self, point: tuple[float, float], aspect: float = 1.0) -> Optional[bool]:
        """Whether this track's recent motion has been heading TOWARD `point`
        (e.g. a person's box centroid) rather than away from or around it --
        the trajectory half of "being picked up", distinct from a merely
        overlapping-right-now box with no direction behind it."""
        if len(self.trail) < 2:
            return None
        (_, x0, y0), (_, x1, y1) = self.trail[0], self.trail[-1]
        if math.hypot((x1 - x0) * aspect, y1 - y0) < 1e-6:
            return None  # not moving enough to have a direction at all
        before = math.hypot((point[0] - x0) * aspect, point[1] - y0)
        after = math.hypot((point[0] - x1) * aspect, point[1] - y1)
        return after < before

    def reversals(self, min_amplitude: float, aspect: float = 1.0) -> int:
        """Direction reversals over the full trail, in the box's own diagonal
        units -- the same chaotic-vs-smooth-motion primitive
        core/activity.py's flailing detector uses for wrist trajectories,
        generalized to any tracked box (a hand, a held item): a wide, smooth
        reach has ~0 reversals; fast, chaotic motion has several. Travel in
        roughly one direction accumulates (dot product with the running
        direction stays positive); a reversal counts each time the track
        commits to a new direction after covering at least `min_amplitude`
        diagonals in the old one, so landmark jitter (near-zero-amplitude
        direction flips) can't inflate the count.
        """
        if len(self.trail) < 3:
            return 0
        diagonal = self.diagonal
        if diagonal <= 1e-6:
            return 0
        count = 0
        seg_dx = seg_dy = 0.0
        prev_dir: Optional[tuple[float, float]] = None
        for (_, x0, y0), (_, x1, y1) in pairwise(self.trail):
            dx, dy = (x1 - x0) * aspect / diagonal, (y1 - y0) / diagonal
            step = math.hypot(dx, dy)
            if step < 1e-6:
                continue
            direction = (dx / step, dy / step)
            if prev_dir is None:
                seg_dx, seg_dy = dx, dy
            elif direction[0] * prev_dir[0] + direction[1] * prev_dir[1] > 0.3:
                seg_dx += dx
                seg_dy += dy
            else:
                # An actual reversal: the PRIOR segment (now ending) counts if
                # it was wide enough. The final still-in-progress segment is
                # deliberately NOT counted after the loop -- unlike a swing
                # counter, a reversal is a direction CHANGE, and the last leg
                # hasn't changed direction (yet) merely by ending mid-trail.
                if math.hypot(seg_dx, seg_dy) >= min_amplitude:
                    count += 1
                seg_dx, seg_dy = dx, dy
            prev_dir = direction
        return count

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id, "label": self.label, "age_s": round(self.age_s, 1),
            "hits": self.hits, "confidence": round(self.confidence, 2), "stability": self.stability,
        }


class BoxTracker:
    """Assigns stable ids to arbitrary-class detector boxes across frames.

    Matching is centroid distance in box-diagonal lengths (PersonTracker's
    torso-length approach, generalized) AND class-scoped: a detection can only
    ever match a track of the SAME label, which a plain person tracker doesn't
    get to assume -- a phone's track can never absorb a cup's, a stronger
    identity signal than centroid distance alone provides for same-class people.
    Appearance re-identification remains out of scope for the same reason
    PersonTracker doesn't have it: a home/car camera sees a handful of items at
    once, well separated, and this matcher's failure mode (two same-class items
    swapping ids when they cross) is visible and bounded.
    """

    def __init__(
        self,
        max_distance: float = MAX_MATCH_DISTANCE_DIAGONAL,
        lost_grace_s: float = BOX_LOST_GRACE_S,
    ) -> None:
        self.max_distance = max_distance
        self.lost_grace_s = lost_grace_s
        self._tracks: dict[int, TrackedBox] = {}
        self._next_id = 1

    @property
    def tracks(self) -> list[TrackedBox]:
        return list(self._tracks.values())

    def update(
        self,
        detections: list[tuple[str, Box, float]],
        now: Optional[float] = None,
        aspect: float = 1.0,
    ) -> list[TrackedBox]:
        """Match this frame's (label, box, confidence) detections to existing
        tracks; return the matched+new set, newly-created tracks included."""
        now = time.monotonic() if now is None else now
        self._retire(now)

        unmatched_tracks = dict(self._tracks)
        pairs = []
        for i, (label, box, _conf) in enumerate(detections):
            for track_id, track in unmatched_tracks.items():
                if track.label != label:
                    continue
                distance = self._distance(track, box, aspect)
                if distance is not None and distance <= self.max_distance:
                    pairs.append((distance, track_id, i))
        pairs.sort()

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matched: list[TrackedBox] = []

        for _distance, track_id, i in pairs:
            if track_id in used_tracks or i in used_detections:
                continue
            _label, box, conf = detections[i]
            track = self._tracks[track_id]
            track.box = box
            track.confidence = conf
            track.last_seen = now
            track.hits += 1
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            track.trail.append((now, cx, cy))
            if len(track.trail) > 30:  # bounded: enough for velocity, not a recording
                track.trail.pop(0)
            used_tracks.add(track_id)
            used_detections.add(i)
            matched.append(track)

        for i, (label, box, conf) in enumerate(detections):
            if i in used_detections:
                continue
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            track = TrackedBox(
                track_id=self._next_id, label=label, box=box, confidence=conf,
                first_seen=now, last_seen=now, trail=[(now, cx, cy)],
            )
            self._tracks[self._next_id] = track
            self._next_id += 1
            matched.append(track)

        return sorted(matched, key=lambda t: t.track_id)

    def _distance(self, track: TrackedBox, box: Box, aspect: float) -> Optional[float]:
        diagonal = max(track.diagonal, math.hypot(box[2] - box[0], box[3] - box[1]))
        if diagonal <= 1e-6:
            return None
        tx, ty = track.centroid
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return math.hypot((cx - tx) * aspect, cy - ty) / diagonal

    def _retire(self, now: float) -> None:
        for track_id in [tid for tid, t in self._tracks.items() if now - t.last_seen > self.lost_grace_s]:
            del self._tracks[track_id]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ====================================================================== shared object tracking (M8)
# ObjectTracker is the ONE tracker the shared perception engine runs
# (core/perception.py). It follows every box the shared object detector
# returns, once per detector run, and every monitoring node and custom rule
# reads the same track ids from the frame's Observation. No node owns a
# tracker of its own. (PersonTracker / BoxTracker above stay what they were:
# internal helpers of the legacy preset modules -- occupancy's per-person
# lingering, baby monitoring's hand/object trajectories.)
#
# Association, per detector run:
#   1. retire tracks unseen for longer than `lost_grace_s` (occlusion patience)
#   2. score every (live track, detection) pair of the SAME label: IoU when the
#      boxes overlap at least `iou_min` (cost 1 - IoU), otherwise centre
#      distance in box diagonals up to `max_centre_distance` (cost 1 + d), so an
#      overlapping match always beats a distance-only one
#   3. greedy assignment, cheapest first, ties broken by track id then detection
#      index -- deterministic: no randomness, no wall clock (time is passed in)
#   4. unmatched detections start new tracks; at capacity the stalest track is
#      evicted (oldest last_seen, then fewest hits, then lowest id)
#
# Stability:
#   - `confirmed` once matched in `confirm_hits` detector runs, and it stays
#     confirmed until retired (a one-run flicker never becomes a track anyone
#     reports; a confirmed track never flaps back to tentative)
#   - `confidence` is an exponential moving average of the matched detections'
#     confidences (`raw_confidence` is the latest). ConfidenceGate adds an
#     enter/exit hysteresis band on top for consumers with a floor
#
# Failure mode, stated: no appearance model. Two same-label boxes that cross
# can swap ids, and someone who leaves for longer than lost_grace_s comes back
# as a new id. Nothing here is re-identification.
#
# Cost (bounded): at most `max_detections` (64) boxes and `max_tracks` (64)
# live tracks, so one update is at most 64 x 64 = 4096 box comparisons (a few
# float operations each) plus a sort of the candidate pairs; a typical room
# (1-5 people) is ~25 comparisons. It runs once per detector run -- never per
# consumer, never on frames the detector skipped. Memory is O(max_tracks).
# The mean update time is reported as `perception.tracking.update_ms`
# (tests/test_m8_tracking.py measures the worst case).

SHARED_MAX_TRACKS = 64
SHARED_MAX_DETECTIONS = 64
SHARED_IOU_MIN = 0.2
SHARED_MAX_CENTRE_DISTANCE = 1.0
SHARED_CONFIRM_HITS = 2
SHARED_LOST_GRACE_S = 2.0
SHARED_CONFIDENCE_ALPHA = 0.4
_UPDATE_MS_EMA = 0.2


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two pixel-space (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _box_centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _box_diagonal(box: Box) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _as_box(box: Any) -> Box:
    x1, y1, x2, y2 = (float(v) for v in tuple(box)[:4])
    return x1, y1, x2, y2


@dataclass(frozen=True)
class Track:
    """One followed object as a frame's Observation carries it (immutable)."""

    track_id: int
    label: str
    box: Box
    #: Smoothed (EMA) detector confidence; `raw_confidence` is the latest match.
    confidence: float
    raw_confidence: float
    first_seen: float
    last_seen: float
    hits: int
    confirmed: bool
    #: Matched in the latest detector run (False: coasting through a miss).
    visible: bool

    @property
    def age_s(self) -> float:
        """How long this object has been followed, first to latest sighting."""
        return self.last_seen - self.first_seen

    def to_dict(self) -> dict:
        return {"track_id": self.track_id, "label": self.label,
                "box": [round(float(v), 1) for v in self.box],
                "confidence": round(self.confidence, 3), "age_s": round(self.age_s, 2),
                "hits": self.hits, "confirmed": self.confirmed, "visible": self.visible}


@dataclass
class _LiveTrack:
    track_id: int
    label: str
    box: Box
    confidence: float
    raw_confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed: bool = False
    visible: bool = True

    def frozen(self) -> Track:
        return Track(self.track_id, self.label, self.box, self.confidence, self.raw_confidence,
                     self.first_seen, self.last_seen, self.hits, self.confirmed, self.visible)


class ObjectTracker:
    """Persistent identity for the shared detector's boxes (see the section comment)."""

    def __init__(self, *, iou_min: float = SHARED_IOU_MIN,
                 max_centre_distance: float = SHARED_MAX_CENTRE_DISTANCE,
                 confirm_hits: int = SHARED_CONFIRM_HITS, lost_grace_s: float = SHARED_LOST_GRACE_S,
                 max_tracks: int = SHARED_MAX_TRACKS, max_detections: int = SHARED_MAX_DETECTIONS,
                 confidence_alpha: float = SHARED_CONFIDENCE_ALPHA) -> None:
        if max_tracks < 1 or max_detections < 1 or confirm_hits < 1:
            raise ValueError("max_tracks, max_detections and confirm_hits must be >= 1")
        if not 0.0 < confidence_alpha <= 1.0:
            raise ValueError("confidence_alpha must be in (0, 1]")
        self.iou_min = float(iou_min)
        self.max_centre_distance = float(max_centre_distance)
        self.confirm_hits = int(confirm_hits)
        self.lost_grace_s = float(lost_grace_s)
        self.max_tracks = int(max_tracks)
        self.max_detections = int(max_detections)
        self.alpha = float(confidence_alpha)
        self._tracks: dict[int, _LiveTrack] = {}
        self._next_id = 1
        # Counters for the perception report (never accuracy claims).
        self.updates = 0
        self.created = 0
        self.retired = 0
        self.evicted = 0
        self.dropped_detections = 0
        self.update_ms: Optional[float] = None

    # ------------------------------------------------------------------ the update
    def update(self, detections: list[tuple[str, Any, float]], now: float) -> list[Optional[int]]:
        """Associate one detector run's (label, box, confidence) boxes.

        Returns the track id given to each detection, in input order (None for
        a detection dropped by the `max_detections` bound).
        """
        started = time.perf_counter()
        self.updates += 1
        self._retire(now)
        for track in self._tracks.values():
            track.visible = False
        boxes = [_as_box(box) for _label, box, _conf in detections]

        kept = list(range(len(detections)))
        if len(kept) > self.max_detections:
            kept = sorted(kept, key=lambda i: (-float(detections[i][2]), i))[:self.max_detections]
            kept.sort()
            self.dropped_detections += len(detections) - len(kept)

        pairs: list[tuple[float, int, int]] = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            for i in kept:
                if str(detections[i][0]) != track.label:
                    continue
                cost = self._cost(track.box, boxes[i])
                if cost is not None:
                    pairs.append((cost, track_id, i))
        pairs.sort()

        ids: list[Optional[int]] = [None] * len(detections)
        used_tracks: set[int] = set()
        for _cost, track_id, i in pairs:
            if track_id in used_tracks or ids[i] is not None:
                continue
            conf = float(detections[i][2])
            track = self._tracks[track_id]
            track.box = boxes[i]
            track.raw_confidence = conf
            track.confidence = self.alpha * conf + (1.0 - self.alpha) * track.confidence
            track.last_seen = now
            track.hits += 1
            track.visible = True
            if track.hits >= self.confirm_hits:
                track.confirmed = True
            used_tracks.add(track_id)
            ids[i] = track_id

        for i in kept:
            if ids[i] is not None:
                continue
            if len(self._tracks) >= self.max_tracks:
                self._evict()
            conf = float(detections[i][2])
            track = _LiveTrack(
                track_id=self._next_id, label=str(detections[i][0]), box=boxes[i],
                confidence=conf, raw_confidence=conf, first_seen=now, last_seen=now,
                confirmed=self.confirm_hits <= 1)
            self._tracks[track.track_id] = track
            self._next_id += 1
            self.created += 1
            ids[i] = track.track_id

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.update_ms = elapsed_ms if self.update_ms is None else (
            (1 - _UPDATE_MS_EMA) * self.update_ms + _UPDATE_MS_EMA * elapsed_ms)
        return ids

    def _cost(self, track_box: Box, box: Box) -> Optional[float]:
        overlap = box_iou(track_box, box)
        if overlap >= self.iou_min:
            return 1.0 - overlap
        diagonal = max(_box_diagonal(track_box), _box_diagonal(box))
        if diagonal <= 1e-6:
            return None
        (tx, ty), (cx, cy) = _box_centre(track_box), _box_centre(box)
        distance = math.hypot(cx - tx, cy - ty) / diagonal
        return 1.0 + distance if distance <= self.max_centre_distance else None

    def _retire(self, now: float) -> None:
        for track_id in [tid for tid, t in self._tracks.items() if now - t.last_seen > self.lost_grace_s]:
            del self._tracks[track_id]
            self.retired += 1

    def _evict(self) -> None:
        stalest = min(self._tracks.values(), key=lambda t: (t.last_seen, t.hits, t.track_id))
        del self._tracks[stalest.track_id]
        self.evicted += 1

    # ------------------------------------------------------------------ views
    def tracks(self) -> tuple[Track, ...]:
        """Every live track (visible or coasting), by id."""
        return tuple(self._tracks[tid].frozen() for tid in sorted(self._tracks))

    def __len__(self) -> int:
        return len(self._tracks)

    def stats(self) -> dict:
        live = list(self._tracks.values())
        return {
            "live": len(live), "confirmed": sum(1 for t in live if t.confirmed),
            "max_tracks": self.max_tracks, "updates": self.updates, "created": self.created,
            "retired": self.retired, "evicted": self.evicted, "dropped_detections": self.dropped_detections,
            "update_ms": None if self.update_ms is None else round(self.update_ms, 3),
            "lost_grace_s": self.lost_grace_s, "confirm_hits": self.confirm_hits,
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


class ConfidenceGate:
    """Enter/exit hysteresis on a track's smoothed confidence.

    A track is admitted once its confidence reaches `enter`, and stays admitted
    until it falls below `enter - margin`: a person hovering around a node's
    confidence floor is counted once, not on and off every detector run.
    Memory is pruned to the tracks it was last shown (bounded by the tracker).
    """

    def __init__(self, enter: float, margin: float = 0.1) -> None:
        self.enter = float(enter)
        self.exit = max(0.0, self.enter - float(margin))
        self._admitted: set[int] = set()

    def admit(self, tracks: Iterable[Track]) -> list[Track]:
        out = []
        for track in tracks:
            floor = self.exit if track.track_id in self._admitted else self.enter
            if track.confidence >= floor:
                out.append(track)
        self._admitted = {t.track_id for t in out}
        return out


@dataclass(frozen=True)
class ZoneTransition:
    """A tracked object crossing into or out of a drawn area."""

    kind: str  # "entered" | "exited"
    zone: str
    track_id: int
    at: float
    #: For "exited": how long it was inside (first sighting inside -> last).
    dwell_s: float = 0.0


@dataclass
class _Membership:
    inside: bool = False
    candidate_since: Optional[float] = None  # first sighting inside (not yet settled)
    entered_at: Optional[float] = None
    outside_since: Optional[float] = None
    last_inside_at: Optional[float] = None


class ZoneMembership:
    """Which tracks are inside which drawn areas, with hysteresis, and for how long.

    Built on the shared tracker's ids: a node keeps one of these (state, not a
    tracker -- it never associates boxes). A visible track whose feet point is
    inside an area for `settle_s` has ENTERED it, its dwell clock starting at
    its first sighting inside; outside for `settle_s` it has EXITED. A coasting
    track (missed this run) changes nothing; a retired track that was inside
    exits at its last sighting inside. Bounded by live tracks x areas.
    """

    def __init__(self, settle_s: float = 1.0) -> None:
        self.settle_s = float(settle_s)
        self._state: dict[tuple[str, int], _Membership] = {}

    def update(self, tracks: Iterable[Track], zones: Iterable[Any], width: int, height: int,
               now: float, *, live: Optional[Iterable[int]] = None) -> list[ZoneTransition]:
        """Advance on one detector run. `tracks` are the tracks to place (a
        consumer may have filtered them, e.g. by confidence); `live` is every
        track id the tracker still holds (default: those of `tracks`). A track
        that is live but not placed this run is treated like a coasting one."""
        tracks = list(tracks)
        zones = list(zones)
        live = {t.track_id for t in tracks} if live is None else set(live)
        names = [str(z.name) for z in zones]
        out: list[ZoneTransition] = []
        # Retired tracks and removed areas: close what was open, forget the rest.
        for key in sorted(k for k in self._state if k[1] not in live or k[0] not in names):
            state = self._state.pop(key)
            if state.inside and key[0] in names and state.entered_at is not None:
                last = state.last_inside_at if state.last_inside_at is not None else state.entered_at
                out.append(ZoneTransition("exited", key[0], key[1], last, max(0.0, last - state.entered_at)))
        for zone, name in zip(zones, names, strict=True):
            for track in tracks:
                if track.visible:
                    out += self._observe(zone, name, track, width, height, now)
        return out

    def _observe(self, zone: Any, name: str, track: Track, width: int, height: int,
                 now: float) -> list[ZoneTransition]:
        key = (name, track.track_id)
        inside = bool(zone.contains_box(track.box, width, height))
        state = self._state.get(key)
        if state is None:
            if not inside:
                return []
            state = self._state[key] = _Membership()
        if inside:
            state.last_inside_at = now
            state.outside_since = None
            if state.inside:
                return []
            if state.candidate_since is None:
                state.candidate_since = now
            if now - state.candidate_since < self.settle_s:
                return []
            state.inside, state.entered_at = True, state.candidate_since
            return [ZoneTransition("entered", name, track.track_id, now)]
        state.candidate_since = None
        if not state.inside:
            del self._state[key]
            return []
        if state.outside_since is None:
            state.outside_since = now
        if now - state.outside_since < self.settle_s:
            return []
        entered = state.entered_at if state.entered_at is not None else now
        last = state.last_inside_at if state.last_inside_at is not None else entered
        del self._state[key]
        return [ZoneTransition("exited", name, track.track_id, now, max(0.0, last - entered))]

    def members(self, zone: str) -> dict[int, float]:
        """Track id -> when its visit started, for every track inside `zone` now."""
        return {tid: s.entered_at for (name, tid), s in sorted(self._state.items())
                if name == zone and s.inside and s.entered_at is not None}

    def dwell(self, zone: str, now: float) -> dict[int, float]:
        """Track id -> seconds inside `zone` so far, for every track inside now."""
        return {tid: max(0.0, now - since) for tid, since in self.members(zone).items()}

    def __len__(self) -> int:
        return len(self._state)

    def clear(self) -> None:
        self._state.clear()
