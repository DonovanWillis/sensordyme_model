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
from dataclasses import dataclass, field
from typing import Optional

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
        for (_, x0, y0), (_, x1, y1) in zip(self.trail, self.trail[1:]):
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
