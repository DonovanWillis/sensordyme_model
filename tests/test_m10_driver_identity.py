"""M10 driver identity: region calibration, face selection, association, reporting.

Deterministic unit tests on constructed boxes (core/behavior/driver_identity.py,
core/behavior/association.py) plus a few module-level checks. The end-to-end
multi-person scenarios are tests/test_m10_driver_scenarios.py. Synthetic only:
these pin what the logic does, not that it is right in a real cabin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from test_m10_driver_scenarios import (
    DRIVER,
    DRIVER_BOX,
    LEANING,
    LEANING_BOX,
    PASSENGER,
    PASSENGER_BOX,
    SCENARIOS,
    STEP,
    _MultiFaceResult,
    run_scenario,
)

from core.behavior.association import (
    AMBIGUOUS,
    CALIBRATED_SEAT,
    CONFIGURED_SEAT,
    FACE_ANCHORED,
    IDENTIFIED,
    NOT_IN_REGION,
    NOT_VISIBLE,
    UNCERTAIN,
    associate_driver,
)
from core.behavior.config import validate_behavior
from core.behavior.driver_identity import (
    MAX_REGION_WIDTH,
    REGION_CALIBRATED,
    REGION_CONFIGURED,
    DriverFaceSelector,
    DriverIdentity,
    DriverIdentityConfig,
    RegionCalibration,
    derive_region,
    parse_driver_identity,
    region_path_for,
)

W, H = 640, 480
FACE = (274.0, 190.0, 366.0, 308.0)  # the generator's neutral driver face (pixels)
PASSENGER_FACE = (49.0, 190.0, 141.0, 308.0)
REGION_PX = (154.0, 1.0, 506.0, 480.0)  # what the install session derives (see test below)


def _n(box):
    return (box[0] / W, box[1] / H, box[2] / W, box[3] / H)


# ============================================================ region derivation
def test_a_single_seated_driver_gives_a_region_with_margins():
    region, status = derive_region([_n(FACE)] * 10, [[_n(DRIVER_BOX)]] * 5, W / H)
    assert region is not None and status == "derived from the installation calibration"
    assert region.source == REGION_CALIBRATED
    # union of face and person box, plus half a face width / height
    assert region.box[0] == pytest.approx((200 - 46) / W) and region.box[2] == pytest.approx((460 + 46) / W)
    assert region.box[1] == pytest.approx((60 - 59) / H) and region.box[3] == 1.0
    assert region.face_anchor == pytest.approx((320 / W, 249 / H, 92 / W))


@pytest.mark.parametrize(("faces", "ticks", "why"), [
    ([_n(FACE)] * 3, [[_n(DRIVER_BOX)]] * 5, "face sample"),
    ([_n(FACE)] * 10, [[_n(DRIVER_BOX)]] * 2, "no person evidence"),
    ([_n(FACE)] * 10, [[_n(DRIVER_BOX), _n(PASSENGER_BOX)]] * 5, "calibrate with the driver alone"),
    ([_n(FACE)] * 10, [[_n(PASSENGER_BOX)]] * 5, "not on the detected person"),
    ([_n(FACE)] * 10, [[(0.0, 0.0, 0.9, 1.0)]] * 5, "could not tell the driver from a passenger"),
    ([(0.5, 0.5, 0.51, 0.52)] * 10, [[(0.4, 0.3, 0.6, 1.0)]] * 5, "below"),
])
def test_an_implausible_calibration_derives_no_region(faces, ticks, why):
    region, status = derive_region(faces, ticks, W / H)
    assert region is None and status.startswith("not derived") and why in status


def _calibrate(path: Path, *, people=(DRIVER_BOX,), frames=30) -> RegionCalibration:
    cal = RegionCalibration(path, enabled=True, load=False, window_s=1.0)
    for i in range(frames):
        cal.observe(i * STEP, calibrated=False, face=_n(FACE), people=[_n(p) for p in people], tick=i, aspect=W / H)
    cal.observe(frames * STEP, calibrated=True, face=_n(FACE), people=[], tick=None, aspect=W / H)
    return cal


def test_the_region_is_persisted_versioned_and_reloaded(tmp_path):
    path = region_path_for(tmp_path / "calibration.json")
    assert path.name == "calibration.driver_region.json"
    cal = _calibrate(path)
    assert cal.region is not None
    stored = json.loads(path.read_text())
    assert stored["version"] == RegionCalibration.FORMAT_VERSION and stored["region"] == list(cal.region.box)
    again = RegionCalibration(path, enabled=True, load=True, window_s=1.0)
    assert again.region == cal.region and again.status.startswith("loaded")
    # --recalibrate (load=False) never uses the stored region, and rewrites it
    assert RegionCalibration(path, enabled=True, load=False, window_s=1.0).region is None


def test_a_calibration_with_a_passenger_in_view_stores_why_and_no_region(tmp_path):
    path = tmp_path / "r.json"
    cal = _calibrate(path, people=(DRIVER_BOX, PASSENGER_BOX))
    assert cal.region is None and "driver alone" in cal.status
    again = RegionCalibration(path, enabled=True, load=True, window_s=1.0)
    assert again.region is None and "driver alone" in again.status


def _stored(region, anchor, aspect: Any = 1.33) -> str:
    return json.dumps({"version": 1, "region": region, "face_anchor": anchor, "aspect": aspect})


@pytest.mark.parametrize("damage", [
    "", "\x00garbage", "[1, 2]", "null", '{"version": 99}',
    _stored([0.9, 0, 0.1, 1], [0.5, 0.5, 0.1]),  # x1 > x2
    _stored([0.0, 0, 0.95, 1], [0.5, 0.5, 0.1]),  # too wide to tell seats apart
    _stored([0.2, 0, 0.7, 1], [0.9, 0.5, 0.1]),  # the face anchor outside the region
    _stored([0.2, 0, 0.7, 1], None),
    _stored([0.2, 0, 0.7, 1], [0.5, 0.5, 0.1], aspect=None),
])
def test_a_damaged_or_implausible_region_file_is_ignored(tmp_path, damage):
    path = tmp_path / "r.json"
    path.write_text(damage)
    cal = RegionCalibration(path, enabled=True, load=True, window_s=1.0)
    assert cal.region is None and "ignored" in cal.status


def _identity(tmp_path, *, configured=None, calibrated=True, face_selection="auto", stored=True,
              attention_calibrated=True) -> DriverIdentity:
    cal_path = tmp_path / "calibration.json"
    if stored:
        _calibrate(region_path_for(cal_path))
    return DriverIdentity(DriverIdentityConfig(face_selection, calibrated), configured_region=configured,
                          calibration_path=cal_path, force_recalibrate=False, window_s=1.0,
                          attention_calibrated=attention_calibrated)


def test_region_precedence_and_the_face_budget(tmp_path):
    calibrated = _identity(tmp_path / "a")
    assert calibrated.region is not None and calibrated.region.source == REGION_CALIBRATED
    assert calibrated.max_faces == 2
    configured = _identity(tmp_path / "b", configured=(0.3, 0.0, 1.0, 1.0))
    assert configured.region is not None and configured.region.source == REGION_CONFIGURED
    assert _identity(tmp_path / "c", calibrated=False).region is None
    assert _identity(tmp_path / "c2", calibrated=False).max_faces == 1
    assert _identity(tmp_path / "d", face_selection="off").max_faces == 1
    assert _identity(tmp_path / "e", stored=False).max_faces == 1  # no region anywhere: as before M10
    # a region stored by an attention calibration that is no longer valid is not used
    assert _identity(tmp_path / "f", attention_calibrated=False).region is None


def test_a_region_calibrated_on_another_frame_shape_is_not_used(tmp_path):
    identity = _identity(tmp_path)
    assert identity.associate([DRIVER_BOX], FACE, 640, 480).method == CALIBRATED_SEAT
    assert identity.associate([DRIVER_BOX], FACE, 640, 360).method == FACE_ANCHORED  # 16:9 vs 4:3
    choice = identity.select_face([PASSENGER_FACE, FACE], 640, 360, 0.0, [DRIVER_BOX])
    assert choice.index == 0 and "differently shaped" in choice.reason


# ============================================================ face selection
def test_the_selection_box_reads_a_few_landmarks_and_falls_back_to_all():
    from types import SimpleNamespace

    from core.behavior.driver_identity import FACE_BOX_LANDMARKS, landmark_box

    mesh = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    mesh[FACE_BOX_LANDMARKS[0]] = SimpleNamespace(x=0.4, y=0.2)
    mesh[FACE_BOX_LANDMARKS[1]] = SimpleNamespace(x=0.6, y=0.7)
    mesh[400] = SimpleNamespace(x=0.0, y=0.0)  # not a box landmark: ignored
    assert landmark_box(mesh, 100, 100) == pytest.approx((40.0, 20.0, 60.0, 70.0))
    short = [SimpleNamespace(x=0.1, y=0.2), SimpleNamespace(x=0.3, y=0.4)]
    assert landmark_box(short, 100, 100) == pytest.approx((10.0, 20.0, 30.0, 40.0))


def _choose(selector, faces, t=0.0, people=(DRIVER_BOX, PASSENGER_BOX), anchor=(320.0, 249.0, 92.0)):
    return selector.choose(list(faces), t, REGION_PX, anchor=anchor,
                           people=None if people is None else list(people))


def test_the_face_in_the_region_is_the_drivers_whatever_the_order():
    selector = DriverFaceSelector()
    choice = _choose(selector, [PASSENGER_FACE, FACE])
    assert choice.index == 1 and choice.identity == IDENTIFIED and choice.verified


def test_a_passengers_face_alone_is_not_the_drivers():
    choice = _choose(DriverFaceSelector(), [PASSENGER_FACE])
    assert choice.index is None and choice.identity == NOT_VISIBLE and "passenger" in choice.reason


def test_a_face_on_nobody_or_with_the_detector_down_is_kept():
    """No evidence it is a passenger's: dropping it could only delay a driver alert."""
    on_nobody = _choose(DriverFaceSelector(), [PASSENGER_FACE], people=(DRIVER_BOX,))
    assert on_nobody.index == 0 and on_nobody.identity == UNCERTAIN
    blind = _choose(DriverFaceSelector(), [PASSENGER_FACE], people=None)
    assert blind.index == 0 and blind.identity == UNCERTAIN and "not counted" in blind.reason


def test_the_face_track_keeps_the_driver_while_a_passenger_leans_into_the_region():
    selector = DriverFaceSelector()
    assert _choose(selector, [PASSENGER_FACE, FACE], 0.0).index == 1
    leaning_in = (150.0, 180.0, 250.0, 300.0)  # centre x 200: inside the region
    choice = _choose(selector, [leaning_in, FACE], 0.05)
    assert choice.index == 1 and "track" in choice.reason
    # without the track (a fresh start) both faces are in the region: the
    # calibrated anchor decides, and the answer is reported uncertain
    fresh = _choose(DriverFaceSelector(), [leaning_in, FACE], 0.0)
    assert fresh.index == 1 and fresh.identity == UNCERTAIN


def test_the_track_follows_a_slumping_driver_out_of_the_region_and_expires():
    selector = DriverFaceSelector()
    t = 0.0
    choices = []
    for dx in range(0, -220, -20):  # ends at centre x 120: outside the region, on the passenger's box
        choices.append(_choose(selector, [(FACE[0] + dx, FACE[1], FACE[2] + dx, FACE[3])], t,
                               people=(PASSENGER_BOX,)))
        t += STEP
    assert all(c.index == 0 and c.verified for c in choices)
    # 1.5 s later the same face is a fresh face, judged by where it is
    late = _choose(selector, [(54.0, 190.0, 146.0, 308.0)], t + 1.5, people=(PASSENGER_BOX,))
    assert late.index is None


def test_without_an_anchor_the_largest_face_in_the_region_wins():
    big, small = (300.0, 150.0, 420.0, 300.0), (420.0, 100.0, 470.0, 160.0)
    choice = _choose(DriverFaceSelector(), [small, big], anchor=None)
    assert choice.index == 1 and choice.identity == UNCERTAIN


# ============================================================ association (M10 precedence)
def test_nobody_in_the_region_is_driver_not_visible():
    absent = associate_driver([PASSENGER_BOX], None, REGION_PX, region_method=CALIBRATED_SEAT)
    assert absent.method == NOT_IN_REGION and absent.driver_box is None and absent.identity == NOT_VISIBLE
    # ... even when the passenger is larger and the only face is theirs (a one-face
    # FaceMesh locked onto them: the face was not selected, so it is not trusted)
    assert associate_driver([LEANING_BOX], PASSENGER_FACE, REGION_PX, face_trusted=False).driver_box is None
    two = associate_driver([(40, 0, 330, 480), DRIVER_BOX], PASSENGER_FACE, REGION_PX, face_trusted=False)
    assert two.method == AMBIGUOUS


def test_two_people_in_the_region_are_told_apart_by_the_drivers_face():
    into = (40, 0, 330, 480)  # a passenger leaning across: box centre x 185, in the region
    chosen = associate_driver([into, DRIVER_BOX], FACE, REGION_PX, region_method=CALIBRATED_SEAT)
    assert chosen.method == CALIBRATED_SEAT and chosen.driver_box == DRIVER_BOX and chosen.identity == IDENTIFIED
    no_face = associate_driver([into, DRIVER_BOX], None, REGION_PX)
    assert no_face.method == AMBIGUOUS and no_face.driver_box is None and no_face.identity == UNCERTAIN


def test_a_rear_box_reaching_behind_the_drivers_head_does_not_claim_the_face():
    rear = (260, 40, 380, 260)  # small box, high in the frame: the driver's face is at its bottom
    chosen = associate_driver([rear, DRIVER_BOX], FACE, (154.0, 0.0, 506.0, 480.0))
    assert chosen.driver_box == DRIVER_BOX


def test_a_driver_leaning_out_of_the_region_is_anchored_by_the_face_and_uncertain():
    leaning_driver_box = (60, 60, 300, 470)  # centre x 180 -> outside a narrow configured region
    face = (110.0, 190.0, 202.0, 308.0)
    out = associate_driver([leaning_driver_box], face, (200.0, 0.0, 640.0, 480.0))
    assert out.method == FACE_ANCHORED and out.driver_box == leaning_driver_box and out.identity == UNCERTAIN


def test_a_configured_region_labels_itself():
    assert associate_driver([DRIVER_BOX], FACE, REGION_PX).method == CONFIGURED_SEAT


# ============================================================ config
def test_driver_identity_config_is_validated_and_parsed():
    assert validate_behavior({"behavior": {"driver_identity": {"face_selection": "off",
                                                               "calibrated_region": False}}}) == []
    bad = validate_behavior({"behavior": {"driver_identity": {"face_selection": "always", "typo": 1,
                                                              "calibrated_region": "yes"}}})
    assert any("face_selection:` must be one of" in e for e in bad)
    assert any("driver_identity.typo:` unknown key" in e for e in bad)
    assert any("calibrated_region:` must be true or false" in e for e in bad)
    assert validate_behavior({"behavior": {"driver_identity": []}})
    assert parse_driver_identity({}) == DriverIdentityConfig("auto", True)
    assert parse_driver_identity({"behavior": {"driver_identity": {"face_selection": "off"}}}).face_selection == "off"
    assert MAX_REGION_WIDTH < 1.0


# ============================================================ in the module
def test_telemetry_and_debug_state_say_who_the_signals_are_about(tmp_path):
    result = run_scenario(SCENARIOS["driver_absent_passenger_present"], tmp_path)
    assert result["identity"] == NOT_VISIBLE


def test_the_module_reports_uncertain_with_two_people_and_no_region(tmp_path, monkeypatch):
    import test_m10_driver_scenarios as sc

    seen: dict[str, Any] = {}
    real = sc.run_scenario

    def spy(scenario, tmp_dir):
        out = real(scenario, tmp_dir)
        seen.update(out)
        return out

    monkeypatch.setattr(sc, "run_scenario", spy)
    scenario = sc.Scenario("two_no_region", "", (sc.Phase(1.0, (DRIVER, PASSENGER), (DRIVER_BOX, PASSENGER_BOX)),),
                           {"behavior": {"driver_identity": {"calibrated_region": False}}})
    assert sc.run_scenario(scenario, tmp_path)["identity"] == UNCERTAIN
    assert seen["max_faces"] == 1


class _BottleDetector:
    names: ClassVar[dict[int, str]] = {0: "person", 39: "bottle"}

    def __init__(self, persons, bottles) -> None:
        self.persons, self.bottles = persons, bottles

    def predict(self, frame, **kwargs):
        from test_phase1_driver_fixes import _FakeBox, _FakeResult

        return [_FakeResult([_FakeBox(0, b, 0.9) for b in self.persons] + [_FakeBox(39, b, 0.8) for b in self.bottles])]


@pytest.mark.parametrize(("faces", "bottle", "expected"), [
    ((DRIVER, LEANING), (60, 40, 110, 140), False),  # at the larger passenger's face; the driver is measured
    ((), (60, 40, 110, 140), False),  # no face: two people, nobody is the driver (was: the largest)
    ((DRIVER, LEANING), (300, 100, 350, 190), True),  # at the driver's face: still the driver's drink
])
def test_food_at_a_larger_passengers_face_is_not_the_drivers(tmp_path, monkeypatch, faces, bottle, expected):
    from test_driver_attention_scenarios import _CONFIG

    import modules.driver_attention as da

    clock = {"now": 0.0}
    monkeypatch.setattr(da.time, "monotonic", lambda: clock["now"])
    detector = _BottleDetector([DRIVER_BOX, LEANING_BOX], [bottle])
    monkeypatch.setattr(da, "get_yolo_model", lambda weights: detector)
    module: Any = da.DriverAttentionModule({
        **_CONFIG, "calibration_path": str(tmp_path / "c.json"), "yolo_every_n_frames": 1, "enable_hands": False,
        "yolo_classes": {"bottle": "eating_drinking"},
        "behavior": {"driver_identity": {"calibrated_region": False}}})
    module._face_mesh.process = lambda rgb: _MultiFaceResult(faces)
    frame = np.random.default_rng(7).integers(60, 200, (H, W, 3), dtype=np.uint8)
    events = []
    for _ in range(10):
        events += module.process(frame)
        clock["now"] += STEP
    assert ("eating_drinking" in [e.type for e in events]) is expected
    module.close()


def test_the_first_calibration_derives_the_region_and_uses_it_at_once(tmp_path):
    """First boot / --recalibrate: the region is derived when the attention
    calibration completes, persisted next to it, and used for person
    association at once; FaceMesh's face budget (fixed when the graph was
    built) moves to two faces from the next start-up."""
    import test_m10_driver_scenarios as sc

    detector, hands = sc._Detector(), sc._FakeHands()
    frame = np.random.default_rng(7).integers(60, 200, (H, W, 3), dtype=np.uint8)
    config = {**sc._CONFIG, "calibration_path": str(tmp_path / "calibration.json"), "yolo_every_n_frames": 1,
              "recalibrate": True}
    with sc._patched(detector) as clock:
        module: Any = sc.DriverAttentionModule(config)
        module._hands = hands
        assert module._identity.region is None and module._identity.max_faces == 1
        sc._drive(module, clock, detector, hands, frame, sc.Phase(2.0, (DRIVER,), (DRIVER_BOX,)), 0.0, [])
        assert module._calibrator.is_calibrated
        assert module._identity.region is not None and module._identity.max_faces == 1
        assert module._driver.method == CALIBRATED_SEAT
        assert "next start-up" in module._identity.last_choice.reason
        assert (tmp_path / "calibration.driver_region.json").exists()
        debug = module.get_debug_state()["driver_identity"]
        assert debug["state"] == IDENTIFIED and debug["accuracy"] == "unevaluated"
        assert debug["region"]["source"] == REGION_CALIBRATED
        module.close()
        again: Any = sc.DriverAttentionModule({**config, "recalibrate": False})
        assert again._identity.max_faces == 2
        again.close()
