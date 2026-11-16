"""M10 multi-person driver scenarios through the REAL DriverAttentionModule.

SYNTHETIC ONLY. Faces are the geometric landmark generator of
tests/test_behavior_integration.py, translated / scaled to stand where a
passenger, a leaning passenger or a rear passenger would; FaceMesh returns them
in a chosen order (the passenger FIRST unless stated: what a one-face FaceMesh
does when it locks onto the passenger). Person / phone boxes come from a fake
detector, hands from a fake MediaPipe Hands, time from a deterministic clock.
Everything between those seams is the real module. Passing says the logic does
what it was designed to do on constructed inputs; it is NOT a measured accuracy
and says nothing about real cabins.

Every scenario is two sessions, as on a device: an INSTALLATION session (driver
alone, seated, looking ahead: the calibration window), then a DRIVE session that
loads that calibration and runs the scenario's phases.

`run_scenario` deliberately uses only what the pre-M10 module has (no import of
M10 code) so the same scenarios can be run against base 4dfd761 for the
before/after table in docs/DRIVER_ASSOCIATION.md.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import pytest
from test_behavior_integration import _FakeHands, landmarks
from test_driver_attention_scenarios import _CONFIG, H, W, _FakeLandmark
from test_phase1_driver_fixes import _FakeBox, _FakeResult

import modules.driver_attention as driver_attention_module
from modules.driver_attention import DriverAttentionModule

STEP = 0.05
NEUTRAL_FACE_CENTRE = (320.0, 249.0)  # the generator's neutral face: x 274..366, y 190..308

DRIVER_BOX = (200, 60, 460, 470)
DRIVER_SLUMPED_BOX = (90, 60, 460, 470)  # the detector's box follows the head that left the seat
PASSENGER_BOX = (0, 120, 190, 470)  # smaller than the driver's
LEANING_BOX = (0, 0, 290, 480)  # LARGER than the driver's (wide lens, leaning toward it)
INTO_REGION_BOX = (40, 0, 330, 480)  # leaning across: box centre inside the driver region
REAR_BOX = (470, 90, 590, 300)

LAP_PHONE, LAP_HAND = (295, 370, 345, 440), (285, 385, 355, 460)
PASSENGER_PHONE, PASSENGER_HAND = (20, 300, 70, 380), (10, 310, 80, 400)
LEANING_PHONE, LEANING_HAND = (150, 300, 195, 380), (140, 310, 199, 400)
REAR_PHONE, REAR_HAND = (495, 220, 535, 270), (485, 225, 545, 280)

#: An installer-drawn region (0..1): the driver's seat, head to lap.
CONFIGURED_REGION = [0.27, 0.0, 0.85, 1.0]


def face(dx: float = 0.0, dy: float = 0.0, scale: float = 1.0, **pose: Any) -> list:
    """Generator landmarks for `pose`, moved by (dx, dy) px and scaled about the face centre."""
    cx, cy = NEUTRAL_FACE_CENTRE
    return [_FakeLandmark((cx + scale * (p.x * W - cx) + dx) / W, (cy + scale * (p.y * H - cy) + dy) / H)
            for p in landmarks(**pose)]


DRIVER = face()
DRIVER_SHUT = face(lid=1.0)
DRIVER_DOWN = face(pitch=-30.0)
PASSENGER = face(dx=-225)
PASSENGER_SHUT = face(dx=-225, lid=1.0)
LEANING = face(dx=-185, scale=1.3)
LEANING_DOWN = face(dx=-185, scale=1.3, pitch=-30.0)
INTO_REGION = face(dx=-130, yaw=40.0)
REAR_SHUT = face(dx=210, dy=-80, scale=0.6, lid=1.0)
OTHER_DRIVER_SHUT = face(scale=1.1, lid=1.0)  # a different (larger) person in the driver's seat


# ---------------------------------------------------------------------------- fakes
class _Proto:
    def __init__(self, landmark: list) -> None:
        self.landmark = landmark


class _MultiFaceResult:
    def __init__(self, faces: tuple) -> None:
        self.multi_face_landmarks = [_Proto(f) for f in faces] or None


class _Detector:
    names: ClassVar[dict[int, str]] = {0: "person", 67: "cell phone"}

    def __init__(self) -> None:
        self.persons: tuple = ()
        self.phones: tuple = ()

    def predict(self, frame, **kwargs):
        return [_FakeResult([_FakeBox(0, b, 0.9) for b in self.persons]
                            + [_FakeBox(67, b, 0.8) for b in self.phones])]


@dataclass(frozen=True)
class Phase:
    seconds: float
    faces: tuple = ()  # FaceMesh's faces, in its order
    persons: tuple = ()
    phones: tuple = ()
    hands: tuple = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    phases: tuple[Phase, ...]
    config: dict = field(default_factory=dict)


def _slump(seconds_each: float = 0.1, steps: int = 10, to_dx: float = -200.0) -> tuple[Phase, ...]:
    """The driver's face drifts continuously toward the passenger side, eyes shut."""
    return tuple(Phase(seconds_each, (face(dx=to_dx * (i + 1) / steps, lid=1.0),), (DRIVER_SLUMPED_BOX,))
                 for i in range(steps))


SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("driver_alone", "Driver alone, attentive",
             (Phase(6.0, (DRIVER,), (DRIVER_BOX,)),)),
    Scenario("driver_alone_eyes_closed", "Driver alone, eyes shut 3 s",
             (Phase(1.0, (DRIVER,), (DRIVER_BOX,)), Phase(3.0, (DRIVER_SHUT,), (DRIVER_BOX,)))),
    Scenario("larger_passenger", "Driver attentive; a LARGER (leaning) passenger, listed first",
             (Phase(6.0, (LEANING, DRIVER), (DRIVER_BOX, LEANING_BOX)),)),
    Scenario("passenger_phone_driver_attentive",
             "The larger passenger looks down at the phone in their hand; the driver watches the road",
             (Phase(6.0, (LEANING_DOWN, DRIVER), (DRIVER_BOX, LEANING_BOX), (LEANING_PHONE,), (LEANING_HAND,)),)),
    Scenario("passenger_phone_configured_region",
             "As above, with an installer-configured driver_region instead of the calibrated one",
             (Phase(6.0, (LEANING_DOWN, DRIVER), (DRIVER_BOX, LEANING_BOX), (LEANING_PHONE,), (LEANING_HAND,)),),
             {"behavior": {"driver_region": CONFIGURED_REGION}}),
    Scenario("driver_phone_passenger_present",
             "The driver looks down at the phone in their hand; a passenger sits beside, listed first",
             (Phase(4.0, (PASSENGER, DRIVER_DOWN), (DRIVER_BOX, PASSENGER_BOX), (LAP_PHONE,), (LAP_HAND,)),)),
    Scenario("driver_phone_passenger_present_no_region",
             "As above with no driver region (calibrated_region off) and the driver's face listed first",
             (Phase(4.0, (DRIVER_DOWN, PASSENGER), (DRIVER_BOX, PASSENGER_BOX), (LAP_PHONE,), (LAP_HAND,)),),
             {"behavior": {"driver_identity": {"calibrated_region": False}}}),
    Scenario("passenger_leaning_into_region",
             "A passenger leans across into the driver region, turned to the driver, phone in hand",
             (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(5.0, (INTO_REGION, DRIVER), (DRIVER_BOX, INTO_REGION_BOX), (LEANING_PHONE,),
                    (LEANING_HAND,)))),
    Scenario("driver_absent_passenger_present",
             "The driver leaves; the passenger stays, eyes shut, for 10 s",
             (Phase(2.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(10.0, (PASSENGER_SHUT,), (PASSENGER_BOX,)))),
    Scenario("rear_passenger",
             "A rear passenger asleep with a phone in hand, listed first; the driver attentive",
             (Phase(6.0, (REAR_SHUT, DRIVER), (DRIVER_BOX, REAR_BOX), (REAR_PHONE,), (REAR_HAND,)),)),
    Scenario("seat_swap_after_calibration",
             "After calibration the two swap seats; the new driver (another face) closes their eyes",
             (Phase(2.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(1.5, (), ()),
              Phase(4.0, (PASSENGER, OTHER_DRIVER_SHUT), (DRIVER_BOX, PASSENGER_BOX)))),
    Scenario("passenger_eyes_closed",
             "The passenger sleeps (listed first); the driver is attentive",
             (Phase(5.0, (PASSENGER_SHUT, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),)),
    Scenario("driver_eyes_closed_passenger_present",
             "The driver's eyes shut; the passenger (listed first) is attentive",
             (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(4.0, (PASSENGER, DRIVER_SHUT), (DRIVER_BOX, PASSENGER_BOX)))),
    Scenario("driver_turns_away_passenger_visible",
             "The driver turns 45 deg, then fully away (face lost) for 4 s; the passenger stays visible",
             (Phase(1.0, (PASSENGER, DRIVER), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(1.0, (PASSENGER, face(yaw=45.0)), (DRIVER_BOX, PASSENGER_BOX)),
              Phase(4.0, (PASSENGER,), (DRIVER_BOX, PASSENGER_BOX)))),
    Scenario("driver_slumps_out_of_region",
             "Driver alone, eyes shut, slumps sideways out of the driver region (continuous)",
             (Phase(0.5, (DRIVER_SHUT,), (DRIVER_BOX,)), *_slump(),
              Phase(3.0, (face(dx=-200, lid=1.0),), (DRIVER_SLUMPED_BOX,)))),
    Scenario("driver_slumps_face_lost_then_outside",
             "Driver alone slumps; face lost 1.5 s, re-found outside the region, eyes shut",
             (Phase(0.5, (DRIVER_SHUT,), (DRIVER_BOX,)), Phase(1.5, (), (DRIVER_SLUMPED_BOX,)),
              Phase(3.0, (face(dx=-200, lid=1.0),), (DRIVER_SLUMPED_BOX,)))),
    Scenario("driver_slumps_box_misses_head",
             "As above, but the detector's box stays on the seat and misses the head (face on nobody)",
             (Phase(0.5, (DRIVER_SHUT,), (DRIVER_BOX,)), Phase(1.5, (), (DRIVER_BOX,)),
              Phase(3.0, (face(dx=-200, lid=1.0),), (DRIVER_BOX,)))),
)}


# ---------------------------------------------------------------------------- runner
@contextlib.contextmanager
def _patched(detector: _Detector):
    clock = {"now": 0.0}
    real_monotonic, real_loader = time.monotonic, driver_attention_module.get_yolo_model
    time.monotonic = lambda: clock["now"]  # type: ignore[assignment]
    driver_attention_module.get_yolo_model = lambda weights: detector  # type: ignore[assignment]
    try:
        yield clock
    finally:
        time.monotonic = real_monotonic  # type: ignore[assignment]
        driver_attention_module.get_yolo_model = real_loader  # type: ignore[assignment]


def _drive(module: Any, clock: dict, detector: _Detector, hands: _FakeHands, frame: np.ndarray,
           phase: Phase, t0: float, out: list) -> None:
    detector.persons, detector.phones = phase.persons, phase.phones
    hands.boxes = list(phase.hands)
    module._face_mesh.process = lambda rgb, faces=phase.faces: _MultiFaceResult(faces)
    for _ in range(max(1, round(phase.seconds / STEP))):
        for event in module.process(frame):
            out.append((round(clock["now"] - t0, 2), event.type))
        clock["now"] += STEP


def run_scenario(scenario: Scenario, tmp_dir: Path) -> dict:
    """Install session (driver alone) then the drive session; events (time, type) and final identity."""
    detector, hands = _Detector(), _FakeHands()
    frame = np.random.default_rng(7).integers(60, 200, (H, W, 3), dtype=np.uint8)
    config = {**_CONFIG, "calibration_path": str(tmp_dir / "calibration.json"), "yolo_every_n_frames": 1,
              **scenario.config}
    with _patched(detector) as clock:
        install: Any = DriverAttentionModule({**config, "recalibrate": True})
        install._hands = hands
        _drive(install, clock, detector, hands, frame, Phase(2.0, (DRIVER,), (DRIVER_BOX,)), 0.0, [])
        assert install._calibrator.is_calibrated
        install.close()
        module: Any = DriverAttentionModule(config)
        module._hands = hands
        events: list = []
        t0 = clock["now"]
        for phase in scenario.phases:
            _drive(module, clock, detector, hands, frame, phase, t0, events)
        snapshot = module.get_telemetry_snapshot() or {}
        identity = getattr(module, "_identity", None)
        module.close()
    return {
        "events": events, "types": sorted({t for _, t in events}),
        "identity": snapshot.get("driver_identity", "n/a"),
        "max_faces": identity.max_faces if identity is not None else 1,
    }


#: What the scenario must and must not raise (M10), and the identity it ends in.
#: drowsiness / distraction / eyes_down / phone_use / attention_lost are the
#: driver alerts in play; risk-fusion and fatigue events are not asserted.
EXPECTED: dict[str, tuple[set[str], set[str], str]] = {
    "driver_alone": (set(), {"drowsiness", "distraction", "eyes_down", "phone_use", "attention_lost"},
                     "identified"),
    "driver_alone_eyes_closed": ({"drowsiness"}, {"attention_lost"}, "identified"),
    "larger_passenger": (set(), {"drowsiness", "distraction", "eyes_down", "phone_use", "attention_lost"},
                         "identified"),
    "passenger_phone_driver_attentive": (set(), {"phone_use", "eyes_down", "distraction"}, "identified"),
    "passenger_phone_configured_region": (set(), {"phone_use", "eyes_down", "distraction"}, "identified"),
    "driver_phone_passenger_present": ({"phone_use"}, set(), "identified"),
    "driver_phone_passenger_present_no_region": ({"phone_use"}, set(), "uncertain"),
    "passenger_leaning_into_region": (set(), {"phone_use", "distraction", "eyes_down", "drowsiness"},
                                      "identified"),
    "driver_absent_passenger_present": ({"attention_lost"}, {"drowsiness", "distraction"}, "not_visible"),
    "rear_passenger": (set(), {"drowsiness", "phone_use", "distraction"}, "identified"),
    "seat_swap_after_calibration": ({"drowsiness"}, set(), "identified"),
    "passenger_eyes_closed": (set(), {"drowsiness"}, "identified"),
    "driver_eyes_closed_passenger_present": ({"drowsiness"}, set(), "identified"),
    "driver_turns_away_passenger_visible": ({"distraction"}, set(), "not_visible"),
    "driver_slumps_out_of_region": ({"drowsiness"}, set(), "identified"),
    "driver_slumps_face_lost_then_outside": ({"drowsiness"}, set(), "uncertain"),
    "driver_slumps_box_misses_head": ({"drowsiness"}, set(), "uncertain"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario(name, tmp_path):
    must, must_not, identity = EXPECTED[name]
    result = run_scenario(SCENARIOS[name], tmp_path)
    assert must <= set(result["types"]), (name, result["events"])
    assert not (must_not & set(result["types"])), (name, result["events"])
    assert result["identity"] == identity, (name, result)


def test_every_scenario_has_an_expectation():
    assert set(EXPECTED) == set(SCENARIOS)


def test_two_faces_only_with_a_region(tmp_path):
    """FaceMesh's face budget: 2 with the calibrated region on file, 1 without one."""
    assert run_scenario(SCENARIOS["driver_alone"], tmp_path / "a")["max_faces"] == 2
    off = Scenario("x", "", SCENARIOS["driver_alone"].phases, {"behavior": {"driver_identity": {"calibrated_region": False}}})
    assert run_scenario(off, tmp_path / "b")["max_faces"] == 1


def test_driver_drowsiness_alert_timing_is_unchanged_by_a_passenger(tmp_path):
    """The safety argument, measured: a passenger in frame does not delay the
    driver's drowsiness alert (same onset as the driver alone, within a frame)."""
    alone = run_scenario(SCENARIOS["driver_alone_eyes_closed"], tmp_path / "alone")
    shared = run_scenario(SCENARIOS["driver_eyes_closed_passenger_present"], tmp_path / "shared")
    t_alone = next(t for t, kind in alone["events"] if kind == "drowsiness")
    t_shared = next(t for t, kind in shared["events"] if kind == "drowsiness")
    assert t_shared == pytest.approx(t_alone, abs=STEP + 1e-9)


def first(events: list, kind: str) -> Optional[float]:
    return next((t for t, k in events if k == kind), None)
