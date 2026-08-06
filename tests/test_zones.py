"""Tests for polygon zones and the feet-point membership rule."""

from core.zones import Zone, load_zones

SQUARE = ((0.4, 0.4), (0.8, 0.4), (0.8, 0.9), (0.4, 0.9))


def test_point_in_polygon():
    zone = Zone("pool", SQUARE)
    assert zone.contains(0.6, 0.6)
    assert not zone.contains(0.1, 0.1)
    assert not zone.contains(0.6, 0.95)  # just below


def test_feet_point_membership_not_center():
    zone = Zone("pool", SQUARE)
    # A tall person: box center is inside the zone band, feet are below it.
    box = (500, 100, 700, 1000)  # feet y = 1000/1000 = 1.0 -> outside (zone ends at 0.9)
    assert not zone.contains_box(box, frame_w=1000, frame_h=1000)
    # Feet standing inside the zone.
    box = (500, 300, 700, 800)  # feet at (0.6, 0.8) -> inside
    assert zone.contains_box(box, frame_w=1000, frame_h=1000)


def test_load_zones_parses_and_skips_invalid():
    zones = load_zones({
        "pool": {"polygon": [[0.4, 0.4], [0.8, 0.4], [0.8, 0.9], [0.4, 0.9]], "kind": "restricted"},
        "crib": {"polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]], "kind": "safe"},
        "bad_two_points": {"polygon": [[0.1, 0.1], [0.2, 0.2]]},
        "bad_out_of_range": {"polygon": [[0.1, 0.1], [1.5, 0.1], [0.5, 0.5]]},
        "bad_kind_defaults": {"polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]], "kind": "wat"},
        "bad_shape": "not a dict",
    })
    by_name = {z.name: z for z in zones}
    assert set(by_name) == {"pool", "crib", "bad_kind_defaults"}
    assert by_name["pool"].kind == "restricted"
    assert by_name["crib"].kind == "safe"
    assert by_name["bad_kind_defaults"].kind == "restricted"  # unknown kind -> safe default
