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
  * boundary: crossing it (inside state change in either direction) is the event.

The membership test point for a person is the bottom-center of their bounding
box — feet position — which matches "standing in an area" far better than the
box center (a tall person's center crosses a floor zone long before they do).
"""

from __future__ import annotations

from dataclasses import dataclass

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
        """Whether a pixel-space (x1, y1, x2, y2) box's feet point is in the zone."""
        x1, y1, x2, y2 = box
        foot_x = ((x1 + x2) / 2) / max(frame_w, 1)
        foot_y = y2 / max(frame_h, 1)
        return self.contains(foot_x, foot_y)


def load_zones(config: dict) -> list[Zone]:
    """Parse a `zones:` config mapping into Zone objects; invalid entries are skipped
    with the reason attached (never fatal — a typo'd zone must not kill the camera)."""
    zones: list[Zone] = []
    for name, spec in (config or {}).items():
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
