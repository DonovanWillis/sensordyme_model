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
