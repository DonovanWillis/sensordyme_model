"""Zones: named polygon regions of the camera frame, for restricted-area detections.

Zones are configured in normalized coordinates (0..1 relative to frame width and
height) so one config works at any capture resolution:

    modules:
      security_zones:
        zones:
          front_door:
            polygon: [[0.6, 0.2], [0.95, 0.2], [0.95, 0.9], [0.6, 0.9]]
            kind: restricted    # restricted | safe | boundary
          crib:
            polygon: [[0.1, 0.1], [0.5, 0.1], [0.5, 0.7], [0.1, 0.7]]
            kind: safe

Detection semantics used by modules:
  * restricted zone: a person INSIDE it is the event;
  * safe zone: a tracked subject LEAVING it is the event;
  * boundary: crossing it (inside state change in either direction) is the event
    (see BoundaryCrossing; reported as a `zone_violation` with "crossed" detail).

The membership test point for a person is the bottom-center of their bounding
box — feet position — which matches "standing in an area" far better than the
box center (a tall person's center crosses a floor zone long before they do).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Point = tuple[float, float]


@dataclass(frozen=True)
class Zone:
    """A named polygon in normalized (0..1) frame coordinates."""

    name: str
    polygon: tuple[Point, ...]
    kind: str = "restricted"  # restricted | safe | boundary

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon (ray casting) for a normalized point."""
        inside = False
        pts = self.polygon
        n = len(pts)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def contains_box(self, box: tuple[float, float, float, float],
                     frame_w: int, frame_h: int) -> bool:
        """Whether a pixel-space (x1, y1, x2, y2) box's feet point is in the zone.

        The feet point is clamped just inside the frame. Both detector backends
        clip boxes to the image, so a person whose legs run off the bottom of
        the frame arrives with y2 == frame_h exactly, which normalizes to 1.0.
        The ray cast in `contains` is half-open, so y == 1.0 is outside EVERY
        polygon, including one drawn right down to the bottom edge (the
        shipped "stairs" examples). Without the clamp, the person nearest the
        camera was never in any zone.
        """
        x1, _y1, x2, y2 = box
        foot_x = ((x1 + x2) / 2) / max(frame_w, 1)
        foot_y = y2 / max(frame_h, 1)
        return self.contains(_clamp_unit(foot_x), _clamp_unit(foot_y))


#: The largest normalized coordinate treated as still inside the frame. See
#: Zone.contains_box for why 1.0 itself has to be pulled back from the edge.
_INSIDE_EDGE = 1.0 - 1e-6


def _clamp_unit(value: float) -> float:
    return min(max(value, 0.0), _INSIDE_EDGE)


class BoundaryCrossing:
    """Debounced membership for one `boundary` zone, reporting each crossing.

    A boundary zone alerts when someone CROSSES it in either direction (the
    app's "Line" kind), not while someone stands inside it. That is a change
    in membership, which the sustained-condition EventEngine cannot express,
    so it is kept here. Each frame takes `inside` (True/False), or None when
    there is no observation (nobody visible, feet not visible). None freezes
    the state, because not seeing anyone is not evidence that anyone crossed.

    A new membership has to hold for `confirm_s` before it counts, so a
    detector wobbling on the edge cannot fire. The first settled membership
    is the baseline and is never reported. The baseline is dropped after
    `forget_after_s` with no observation, so the next person's starting side
    is not compared with where the previous person was last seen.
    """

    def __init__(self, confirm_s: float, forget_after_s: float = 5.0) -> None:
        self.confirm_s = confirm_s
        self.forget_after_s = forget_after_s
        self.settled: Optional[bool] = None
        self._candidate: Optional[bool] = None
        self._candidate_since = 0.0
        self._last_observed: Optional[float] = None

    def update(self, inside: Optional[bool], now: float) -> Optional[str]:
        """Advance one frame. Returns "entered"/"left" on a settled crossing, else None."""
        if inside is None:
            self._candidate = None
            if self._last_observed is not None and now - self._last_observed > self.forget_after_s:
                self.settled = None
            return None
        self._last_observed = now
        if inside == self.settled:
            self._candidate = None
            return None
        if self._candidate != inside:
            self._candidate, self._candidate_since = inside, now
        if now - self._candidate_since < self.confirm_s:
            return None
        previous, self.settled, self._candidate = self.settled, inside, None
        if previous is None:
            return None  # the baseline, not a crossing
        return "entered" if inside else "left"


def load_zones(config: dict) -> list[Zone]:
    """Parse a `zones:` config mapping into Zone objects; invalid entries are skipped
    with the reason attached (never fatal — a typo'd zone must not kill the camera)."""
    zones: list[Zone] = []
    if not isinstance(config, dict):
        # A list-shaped `zones:` (an easy YAML slip) used to raise
        # AttributeError inside a module's __init__, which takes the camera
        # down. Here it is treated like any other invalid entry and dropped.
        return zones
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        polygon = spec.get("polygon") or []
        try:
            points = tuple((float(p[0]), float(p[1])) for p in polygon)
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) < 3:
            continue
        if not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points):
            continue
        kind = str(spec.get("kind", "restricted"))
        if kind not in ("restricted", "safe", "boundary"):
            kind = "restricted"
        zones.append(Zone(name=str(name), polygon=points, kind=kind))
    return zones
