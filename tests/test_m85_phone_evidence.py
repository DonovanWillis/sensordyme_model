"""M8.5 phone-handling evidence: the shared, pure pieces of the production phone_use gate.

core/behavior/phone_evidence.py (PhonePlacement, handling_cue,
classify_handling, supports_phone_use) and modules/driver_attention's driver
relation (phones_for_associated_driver). Deterministic unit tests on
constructed boxes; the end-to-end behaviour is tests/test_m85_phone_scenarios.py.
The cues are UNVALIDATED heuristics; these tests pin what they do, not that
they are right in a real cabin.
"""

from __future__ import annotations

import pytest
from test_behavior_integration import (
    DRIVER_BOX,
    LAP_PHONE,
    STEP,
    _FakeHands,
    _make,
    _types,
    clock,  # noqa: F401 - fixture
    landmarks,
    yolo,  # noqa: F401 - fixture
)
from test_driver_attention_scenarios import _FakeFaceMeshResult

from core.behavior.association import AMBIGUOUS, DriverAssociation, associate_driver
from core.behavior.phone_evidence import (
    ACTIVE,
    CUE_AT_FACE,
    CUE_LAP,
    CUE_MOVING,
    HELD,
    INSUFFICIENT,
    INTERACTION,
    MOUNTED,
    NEAR_DRIVER,
    NO_PHONE,
    STILL_S,
    STILL_TOLERANCE,
    VISIBLE,
    PhoneEvidence,
    PhoneEvidenceInput,
    PhonePlacement,
    classify_handling,
    classify_phone,
    handling_cue,
    phone_risk_for,
    supports_phone_use,
)
from modules.driver_attention import PhoneHoldEvidence, phones_for_associated_driver

#: Neutral face of the scripted generators (x 274..366, y 190..308).
FACE = (274.0, 190.0, 366.0, 308.0)
MOUNT = (360, 330, 410, 400)  # below the chin, centre x 385: outside the face's x-span
LAP = (295, 370, 345, 440)  # below the chin, centre x 320: inside it
EAR = (370, 220, 410, 290)  # at face height, beside the face


# ============================================================ constants are the latch's own
def test_the_motion_bar_is_the_not_held_latchs_stillness_bar():
    hold = PhoneHoldEvidence()
    assert STILL_S == hold._static_s == 2.0
    assert STILL_TOLERANCE == hold._static_tolerance == 0.10


# ============================================================ PhonePlacement
def _feed(placement, boxes, t0=0.0, dt=0.1):
    t = t0
    for box in boxes:
        placement.update(t, box)
        t = round(t + dt, 6)
    return t


def test_first_sighting_is_not_a_move_and_a_still_phone_never_moves():
    p = PhonePlacement()
    t = _feed(p, [MOUNT] * 50)
    assert not p.moving(t) and p.moves == 0


def test_jitter_under_the_tolerance_is_still():
    p = PhonePlacement()
    diag = ((50 ** 2) + (70 ** 2)) ** 0.5
    nudge = 0.9 * STILL_TOLERANCE * diag
    t = _feed(p, [MOUNT, (MOUNT[0] + nudge, MOUNT[1], MOUNT[2] + nudge, MOUNT[3])] * 10)
    assert not p.moving(t)


def test_a_move_counts_for_still_s_then_the_phone_is_still_again():
    p = PhonePlacement()
    t = _feed(p, [MOUNT] * 10)  # t = 1.0
    p.update(t, LAP)  # moved > 10% of the diagonal
    assert p.moving(t) and p.moving(t + STILL_S - 0.01)
    assert not p.moving(t + STILL_S)


def test_flicker_neither_invents_nor_hides_a_move():
    p = PhonePlacement()
    t = _feed(p, [MOUNT, None, None, MOUNT, None, MOUNT] * 5)  # last seen at 2.9 s
    assert p.moves == 0, "reappearing in place is not a move"
    p.update(t, None)
    p.update(t + 0.5, LAP)  # gone for 0.6 s, back somewhere else: a move
    assert p.moves == 1 and p.moving(t + 0.5)


def test_a_long_absence_forgets_the_anchor():
    p = PhonePlacement()
    p.update(0.0, MOUNT)
    p.update(STILL_S + 0.5, None)  # gone longer than memory_s
    p.update(STILL_S + 0.6, LAP)
    assert not p.moving(STILL_S + 0.6), "a fresh sighting, not a move"


# ============================================================ handling_cue
def test_moving_is_a_cue_whatever_the_geometry():
    assert handling_cue(MOUNT, None, moving=True, looking_down=False) == CUE_MOVING


def test_without_a_face_only_motion_can_speak():
    assert handling_cue(LAP, None, moving=False, looking_down=True) is None


def test_a_phone_at_the_ear_is_at_the_face():
    assert handling_cue(EAR, FACE, moving=False, looking_down=False) == CUE_AT_FACE


def test_the_lap_zone_needs_the_gaze_down():
    assert handling_cue(LAP, FACE, moving=False, looking_down=True) == CUE_LAP
    assert handling_cue(LAP, FACE, moving=False, looking_down=False) is None


def test_a_still_mount_below_and_beside_the_face_has_no_cue():
    assert handling_cue(MOUNT, FACE, moving=False, looking_down=True) is None
    assert handling_cue(MOUNT, FACE, moving=False, looking_down=False) is None


def test_far_from_the_face_at_face_height_is_not_at_the_face():
    far = (480, 220, 520, 290)  # centre x 500: 1.5 face widths right of the face
    assert handling_cue(far, FACE, moving=False, looking_down=False) is None


# ============================================================ levels and the gate
@pytest.mark.parametrize(("hold", "cue", "level", "supports"), [
    ("held", None, HELD, True),
    ("not_held", None, MOUNTED, False),
    ("not_held", CUE_LAP, MOUNTED, False),  # the set-down latch wins, as before
    ("unknown", None, NEAR_DRIVER, False),
    ("unknown", CUE_MOVING, NEAR_DRIVER, True),
    ("unknown", CUE_AT_FACE, NEAR_DRIVER, True),
    ("unknown", CUE_LAP, NEAR_DRIVER, True),
])
def test_classify_handling(hold, cue, level, supports):
    evidence = classify_handling(hold, cue)
    assert evidence.level == level and supports_phone_use(evidence) is supports
    if level == NEAR_DRIVER and cue is not None:
        assert evidence.handling_cue == cue and "unvalidated" in evidence.reason
        assert evidence.to_dict()["handling_cue"] == cue


@pytest.mark.parametrize("level", [NO_PHONE, VISIBLE, MOUNTED, INSUFFICIENT])
def test_the_gate_refuses_every_non_handling_level(level):
    assert supports_phone_use(PhoneEvidence(level, "")) is False
    assert supports_phone_use(None) is False


@pytest.mark.parametrize("level", [HELD, INTERACTION, ACTIVE])
def test_the_gate_accepts_every_held_level(level):
    assert supports_phone_use(PhoneEvidence(level, "")) is True


def test_a_cue_adds_no_risk_decision_b_is_unchanged():
    evidence = classify_phone(PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True,
                                                 hold="unknown", eyes_off_road=True, handling_cue=CUE_LAP))
    assert evidence.level == NEAR_DRIVER and evidence.handling_cue == CUE_LAP
    assert phone_risk_for(evidence, True) == 0.0


def test_without_a_cue_the_shadow_levels_and_reasons_are_the_m84_ones():
    def level(hold):
        return classify_phone(PhoneEvidenceInput(detector_ok=True, any_phone=True, driver_phone=True, hold=hold))
    assert level("unknown").to_dict() == {"level": NEAR_DRIVER,
                                          "reason": "in the driver's region; holding not established"}
    assert level("not_held").to_dict() == {"level": MOUNTED, "reason": "still, hands seen elsewhere, none on it"}


# ============================================================ which person's phone
PASSENGER_LEANING = (0, 0, 300, 480)
PASSENGER_PHONE = (20, 300, 70, 380)


def test_a_larger_passengers_phone_is_not_the_drivers():
    persons = [DRIVER_BOX, PASSENGER_LEANING]
    association = associate_driver(persons, FACE)
    assert association.driver_box == DRIVER_BOX
    assert phones_for_associated_driver([PASSENGER_PHONE, LAP], persons, association) == [LAP]


def test_without_a_face_the_shipped_largest_person_proxy_applies():
    persons = [DRIVER_BOX, PASSENGER_LEANING]
    association = associate_driver(persons, None)
    assert association.driver_box == PASSENGER_LEANING
    assert phones_for_associated_driver([PASSENGER_PHONE], persons, association) == [PASSENGER_PHONE]


def test_an_ambiguous_driver_relates_no_phone_and_no_person_relates_every_phone():
    ambiguous = DriverAssociation(AMBIGUOUS, None, 2, None, "measured face lies in several person boxes")
    assert phones_for_associated_driver([LAP], [DRIVER_BOX, PASSENGER_LEANING], ambiguous) == []
    assert phones_for_associated_driver([LAP], [], associate_driver([], FACE)) == [LAP]


# ============================================================ the preset says what the module does
def test_the_driver_safety_preset_is_v5_and_labels_the_handling_evidence_unvalidated():
    from pathlib import Path

    import yaml

    preset = yaml.safe_load((Path(__file__).resolve().parent.parent / "presets.yaml").read_text())[
        "presets"]["driver_safety"]
    assert preset["version"] == 5
    phone = next(p for p in preset["temporal_policies"] if p["event"] == "phone_use")
    assert "handled" in phone["description"] and "never phone use" in phone["description"]
    assert "unvalidated" in phone["description"]
    limitations = " ".join(preset["limitations"])
    assert "Unvalidated phone-handling evidence (M8.5)" in limitations
    assert "still judged as in use" not in limitations


# ============================================================ in the module
def _run(module, frame, clock, seconds, lms):  # noqa: F811 - clock is the fixture value
    module._face_mesh.process = lambda rgb, lm=lms: _FakeFaceMeshResult(lm)
    events = []
    for _ in range(max(1, round(seconds / STEP))):
        events += module.process(frame)
        clock["now"] += STEP
    return events


def test_the_module_exposes_the_gate_and_labels_it_unvalidated(tmp_path, clock, yolo):  # noqa: F811
    yolo.phones = [MOUNT]
    module, frame = _make(tmp_path, clock, v2=False, hands=_FakeHands())
    _run(module, frame, clock, 3.0, landmarks())
    events = _run(module, frame, clock, 3.0, landmarks(pitch=-30.0))
    assert "phone_use" not in _types(events) and "eyes_down" in _types(events)
    handling = module.get_debug_state()["phone_handling"]
    assert handling["level"] == NEAR_DRIVER and handling["supports_phone_use"] is False
    assert handling["evidence"] == "unvalidated"


def test_a_lap_phone_picked_up_without_a_hand_seen_is_phone_use(tmp_path, clock, yolo):  # noqa: F811
    yolo.phones = [LAP_PHONE]
    module, frame = _make(tmp_path, clock, v2=False, hands=_FakeHands())
    events = _run(module, frame, clock, 3.0, landmarks(pitch=-30.0))
    phone = [e for e in events if e.type == "phone_use"]
    assert len(phone) == 1 and phone[0].duration == pytest.approx(2.0, abs=0.1)
    assert module.get_debug_state()["phone_handling"]["handling_cue"] == CUE_LAP
