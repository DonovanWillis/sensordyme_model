"""M10.1 Priority 3: the two driver-association edge cases M10 left open
(docs/DRIVER_ASSOCIATION.md section 8), plus tracking through occlusion, seat
changes and temporary face loss.

SYNTHETIC ONLY, through the real DriverAttentionModule (tests/test_m10_driver_scenarios.py
runner: an installation session with the driver alone, then the drive session).
Positions only: no face is recognised and nothing is enrolled.

Before M10.1 (measured on 629f906 with this file's scenarios):

| scenario | truth | 629f906 | M10.1 |
|---|---|---|---|
| passenger leans in (own box outside the region) while the driver is turned away | distraction | **nothing (masked)**, identified | distraction, not_visible |
| same, the passenger's box centred in the region too | distraction | **nothing (masked)**, identified | distraction, not_visible |
| driver slumps onto the passenger's box, eyes shut, passenger visible | drowsiness | **nothing (missed)**, not_visible | drowsiness, uncertain |
"""

from __future__ import annotations

import pytest
from test_m10_driver_scenarios import (
    DRIVER,
    DRIVER_BOX,
    DRIVER_SHUT,
    INTO_REGION_BOX,
    PASSENGER,
    PASSENGER_BOX,
    Phase,
    Scenario,
    face,
    run_scenario,
)

#: A passenger leaning across: face inside the driver region (x 180), left of the
#: driver's person box, looking ahead (so measured as the driver it reads attentive).
LEAN_IN = face(dx=-140)
LEAN_IN_BOX = (0, 40, 260, 480)  # the passenger's box, centred outside the region
LEAN_IN_FAR = face(dx=-140, dy=-10)
#: A front passenger sitting further over, and the driver slumped onto their box.
FAR_PASSENGER = face(dx=-260, dy=-20, scale=0.9)
FAR_PASSENGER_BOX = (0, 100, 150, 470)
SLUMPED_SHUT = face(dx=-185, dy=60, lid=1.0)  # centre (135, 317): outside the region, on the passenger's box
TURNED = face(yaw=45.0)

EDGES: dict[str, tuple[Scenario, set[str], set[str], str]] = {s.name: (s, must, must_not, ident)
                                                              for s, must, must_not, ident in (
    (Scenario("passenger_leans_in_while_driver_turned_away",
              "The driver turns away (face lost); the passenger leans into the driver region, looking ahead",
              (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(1.0, (PASSENGER, TURNED), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(4.0, (LEAN_IN,), (DRIVER_BOX, LEAN_IN_BOX)))),
     {"distraction"}, {"drowsiness", "eyes_down"}, "not_visible"),
    (Scenario("passenger_leans_in_box_centred_in_region",
              "As above, the leaning passenger's box centred inside the driver region too",
              (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(1.0, (PASSENGER, TURNED), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(4.0, (LEAN_IN_FAR,), (DRIVER_BOX, INTO_REGION_BOX)))),
     {"distraction"}, {"drowsiness", "eyes_down"}, "not_visible"),
    (Scenario("driver_slumps_onto_passenger_box",
              "Eyes shut, the driver slumps toward the passenger: face lost 1.5 s, re-found on the "
              "passenger's box; the passenger's own face stays visible",
              (Phase(1.0, (FAR_PASSENGER, DRIVER), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(0.5, (FAR_PASSENGER, DRIVER_SHUT), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(1.5, (FAR_PASSENGER,), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(3.0, (FAR_PASSENGER, SLUMPED_SHUT), (DRIVER_BOX, FAR_PASSENGER_BOX)))),
     {"drowsiness"}, {"distraction"}, "uncertain"),
    # ------------------------------------------------ the uncertain cases stay uncertain (nothing guessed)
    (Scenario("slumped_face_without_the_passengers_face",
              "As above but the passenger's face is not visible: a face on the passenger's box could be "
              "theirs, so it is not measured",
              (Phase(1.0, (DRIVER,), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(0.5, (DRIVER_SHUT,), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(1.5, (), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(3.0, (SLUMPED_SHUT,), (DRIVER_BOX, FAR_PASSENGER_BOX)))),
     set(), {"drowsiness"}, "not_visible"),
    (Scenario("passenger_eyes_shut_driver_turned_away",
              "The driver turned away (face lost); the passenger beside, eyes shut: never the driver's",
              (Phase(1.0, (FAR_PASSENGER, DRIVER), (DRIVER_BOX, FAR_PASSENGER_BOX)),
               Phase(5.0, (face(dx=-260, dy=-20, scale=0.9, lid=1.0),), (DRIVER_BOX, FAR_PASSENGER_BOX)))),
     set(), {"drowsiness"}, "not_visible"),
    # ------------------------------------------------ occlusion, seat change, temporary loss
    (Scenario("driver_occluded_then_back",
              "The driver's face occluded 0.8 s (hand), passenger visible, then the driver's eyes shut 3 s",
              (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(0.8, (PASSENGER,), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(3.0, (PASSENGER, DRIVER_SHUT), (DRIVER_BOX, PASSENGER_BOX)))),
     {"drowsiness"}, set(), "identified"),
    (Scenario("driver_lost_3s_returns_to_seat",
              "The driver's face lost 3 s (turned fully away, body in the seat), back in the seat, eyes shut",
              (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(3.0, (PASSENGER,), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(3.0, (PASSENGER, DRIVER_SHUT), (DRIVER_BOX, PASSENGER_BOX)))),
     {"drowsiness"}, set(), "identified"),
    (Scenario("seat_change_driver_leaves_new_driver_sits",
              "The driver leaves (5 s empty seat), another person sits and closes their eyes",
              (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
               Phase(5.0, (PASSENGER,), (PASSENGER_BOX,)),
               Phase(4.0, (PASSENGER, face(scale=1.1, lid=1.0)), (DRIVER_BOX, PASSENGER_BOX)))),
     {"drowsiness"}, set(), "identified"),  # attention_lost: 8 s with any person in view (section 8)
)}


@pytest.mark.parametrize("name", sorted(EDGES))
def test_edge_case(name, tmp_path):
    scenario, must, must_not, identity = EDGES[name]
    result = run_scenario(scenario, tmp_path)
    assert must <= set(result["types"]), (name, result["events"])
    assert not (must_not & set(result["types"])), (name, result["events"])
    assert result["identity"] == identity, (name, result["identity"], result["events"])
