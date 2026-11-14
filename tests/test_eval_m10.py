"""Mission 10 (EVAL stream): the leakage-proof, event-level evaluation workflow.

Everything here is synthetic: annotations, predictions and clips are built by
the tests with known answers. No recording, no dataset and no real-world number
exists or is implied; the point is to pin the arithmetic, the refusals, the
split lock and the power check that stands between a small sample and a claim.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from evaluation import agreement as agr
from evaluation import annotations_v2 as av2
from evaluation import event_metrics as em
from evaluation import leakage, scenarios
from evaluation import manifest as mf
from evaluation import report as rp
from evaluation.annotations import Span
from evaluation.replay import replay

MATRIX = scenarios.load()
SUSTAIN = {"drowsiness": 2.0, "distraction": 3.0, "eyes_down": 2.5, "eyes_up": 2.5, "phone_use": 2.0,
           "attention_lost": 2.5}
SMALL = scenarios.Minimums(subjects=2, clips=2, events=2, negative_events=1, exposure_hours=0.5)


def _small_matrix() -> scenarios.Matrix:
    """The shipped matrix with minimums a two-subject synthetic set can meet (tests only)."""
    cells = {k: dataclasses.replace(c, minimums=SMALL) for k, c in MATRIX.cells.items()}
    return dataclasses.replace(MATRIX, overall=SMALL, cells=cells)


def _doc(**over) -> dict:
    doc = {
        "schema": "sensordyme.incabin.annotation", "version": 2, "item": "c1", "annotator": "A1",
        "clip": {"subject": "S01", "session": "S01-a", "vehicle": "V01", "recorded_on": "2026-12-05",
                 "seat": "driver", "duration_s": 3600.0},
        "scenarios": ["E3_sustained_closure"],
        "tags": {"lighting": "day", "eyewear": "none", "occupancy": "driver_only", "mount": "dash_centre",
                 "eye_closure": "long_closure"},
        "events": [], "states": {}, "ignore": [], "uncertain": [],
    }
    for k, v in over.items():
        if k == "clip":
            doc["clip"] = {**doc["clip"], **v}
        else:
            doc[k] = v
    return doc


def _ann(**over) -> av2.AnnotationV2:
    parsed, problems = av2.parse(_doc(**over), matrix=MATRIX)
    assert parsed is not None, problems
    return parsed


# -- 1. scenario matrix ----------------------------------------------------------------------

def test_matrix_covers_every_required_scenario_with_minimums_and_safety():
    axes = MATRIX.axes
    required = {
        "occupancy": {"driver_plus_front_passenger", "driver_plus_rear_passenger"},
        "seat_position": {"forward_high", "rearward_low"},
        "eyewear": {"clear", "sunglasses_ir_pass", "sunglasses_ir_block"},
        "lighting": {"day", "dusk", "night_ir", "direct_sun_glare"},
        "phone": {"handheld_hand", "handheld_lap", "handheld_ear", "mounted", "passenger_phone"},
        "head_motion": {"yaw_sweep", "pitch_sweep"},
        "vibration": {"door_close", "mount_tap"},
        "occlusion": {"hand", "hat_brim", "lens_partial", "lens_full"},
        "presence": {"absent"},
        "eye_closure": {"natural_blink", "short_closure", "long_closure"},
        "glance": {"mirrors", "cluster", "passenger"},
    }
    for axis, values in required.items():
        assert values <= set(axes[axis]), axis
        covered = {v for c in MATRIX.cells.values() for v in c.tags.get(axis, ())}
        assert values <= covered, f"no cell covers {axis} {values - covered}"
    assert MATRIX.safety["stationary_vehicle_only"] and MATRIX.safety["consenting_adults_only"]
    assert MATRIX.safety["children_allowed"] is False and MATRIX.safety["driving_allowed"] is False
    for cell in MATRIX.cells.values():
        assert cell.minimums.subjects >= 30 and cell.minimums.events >= 60 and cell.minimums.exposure_hours >= 3
        assert cell.checklist, f"{cell.id} has no recording checklist"
    # Blink vs microsleep vs sustained closure are three different roles.
    assert MATRIX.rules["long_closure"].role == "positive"
    assert MATRIX.rules["short_closure"].role == "boundary"
    assert "E1_natural_blinks" in MATRIX.cells
    assert MATRIX.rules["glance_mirror_left"].role == "negative"
    assert MATRIX.rules["passenger_phone_use"].alerts == {"phone_use"}


@pytest.mark.parametrize(("mutate", "expected"), [
    (lambda r: r["safety"].update(children_allowed=True), "children_allowed"),
    (lambda r: r["safety"].update(stationary_vehicle_only=False), "stationary_vehicle_only"),
    (lambda r: r["cells"][0].update(safety_exempt=True), "exempt"),
    (lambda r: r["cells"][0].update(positives=["glance_mirror_left"]), "scored as negative"),
    (lambda r: r["cells"][0].update(tags={"lighting": ["moonlight"]}), "not in the axis vocabulary"),
    (lambda r: r["scoring"]["events"].update(texting={"role": "positive", "alerts": ["phone_use"]}),
     "not an annotation v2 event type"),
    (lambda r: r["defaults"]["minimums"].update(subjects=0), "positive number"),
])
def test_an_unsafe_or_inconsistent_matrix_is_refused(mutate, expected):
    raw = yaml.safe_load(scenarios.MATRIX_PATH.read_text())
    mutate(raw)
    with pytest.raises(scenarios.MatrixError, match=expected):
        scenarios.parse(raw)


def test_clip_tags_must_match_the_cells_they_claim():
    tags = {"lighting": "night_ir", "eyewear": "none", "occupancy": "driver_only", "mount": "dash_centre"}
    assert MATRIX.clip_tag_problems(["L3_night_ir"], {**tags, "ir_illuminator": "850nm"}) == []
    problems = MATRIX.clip_tag_problems(["L1_day", "X9"], {**tags, "weather": "rain"})
    assert any("cell L1_day needs tags.lighting" in p for p in problems)
    assert any("unknown cell 'X9'" in p for p in problems)
    assert any("tags.weather" in p for p in problems)
    assert any("tags.mount is required" in p for p in MATRIX.clip_tag_problems(["L1_day"], {"lighting": "day"}))


# -- 2. annotation schema v2 -----------------------------------------------------------------

def test_v2_parses_clip_identity_scenarios_uncertain_spans_and_annotator():
    a = _ann(events=[{"type": "long_closure", "start_s": 10, "end_s": 13}],
             uncertain=[{"start_s": 50, "end_s": 55, "reason": "ambiguous_eye_state", "types": ["long_closure"]}],
             clip={"occupants": ["S07"]})
    assert a.clip.vehicle_day == "V01@2026-12-05" and a.clip.people == ("S01", "S07")
    assert a.annotator == "A1" and a.scenarios == ("E3_sustained_closure",)
    assert a.uncertain[0].covers("long_closure") and not a.uncertain[0].covers("eyes_off_road")
    assert a.excluded_for({"eyes_off_road"}) == [] and a.excluded_for({"long_closure"}) == [Span(50, 55)]
    assert a.as_v1().events == a.events


@pytest.mark.parametrize(("over", "expected"), [
    ({"annotator": ""}, "annotator is required"),
    ({"clip": {"vehicle": ""}}, "clip.vehicle is required"),
    ({"clip": {"recorded_on": "05/12/2026"}}, "not YYYY-MM-DD"),
    ({"clip": {"seat": "roof"}}, "clip.seat"),
    ({"states": {"sleepy": [[1, 2]]}}, "unknown state"),
    ({"ignore": [{"start_s": 0, "end_s": 1, "reason": "lunch"}]}, "reason 'lunch'"),
    ({"uncertain": [{"start_s": 0, "end_s": 1, "reason": "dunno"}]}, "reason 'dunno'"),
    ({"events": [{"type": "texting", "start_s": 1, "end_s": 2}]}, "unknown event type"),
    ({"events": [{"type": "long_closure", "start_s": 3590, "end_s": 3700}]}, "after the clip"),
    ({"events": [{"type": "long_closure", "start_s": 10, "end_s": 12}],
      "uncertain": [{"start_s": 9, "end_s": 13, "reason": "motion_blur", "types": ["long_closure"]}]},
     "wholly inside an uncertain span"),
    ({"scenarios": ["L3_night_ir"]}, "cell L3_night_ir needs tags.lighting"),
    ({"version": 1}, "not a v2 annotation"),
])
def test_v2_problems_are_reported_not_guessed(over, expected):
    parsed, problems = av2.parse(_doc(**over), matrix=MATRIX)
    assert parsed is None
    assert any(expected in p for p in problems), problems


def _item(n, subject, split=None, *, vehicle="V01", day="2026-12-05", occupants="", session=None,
          scenarios_="E3_sustained_closure") -> mf.Item:
    cond = {"vehicle": vehicle, "recorded_on": day, "seat": "driver", "lighting": "day", "eyewear": "none",
            "occupancy": "driver_only", "mount": "dash_centre", "eye_closure": "long_closure",
            "scenarios": scenarios_}
    if occupants:
        cond["occupants"] = occupants
    return mf.Item(id=f"c{n}", path=f"v/{n}.avi", sha256=f"{n:064x}", bytes=1, subject=subject,
                   session=session or f"{subject}-{day}", split=split, conditions=cond)


def test_v1_documents_migrate_from_manifest_metadata_or_are_refused(tmp_path):
    v1 = {"item": "c1", "events": [{"type": "long_closure", "start_s": 4, "end_s": 6}],
          "ignore": [{"start_s": 0, "end_s": 1, "reason": "calibration"}]}
    path = tmp_path / "c1.json"
    path.write_text(json.dumps(v1))
    item = _item(1, "S01")
    migrated, problems = av2.load_any(path, item, matrix=MATRIX)
    assert migrated is not None, problems
    assert migrated.migrated_from == 1 and migrated.annotator == av2.MIGRATED_ANNOTATOR
    assert (migrated.clip.subject, migrated.clip.vehicle, migrated.scenarios) == ("S01", "V01", ("E3_sustained_closure",))
    assert migrated.tags["lighting"] == "day" and "vehicle" not in migrated.tags
    assert av2.consistency_problems(migrated, item) == []
    bare = _item(1, "S01")
    del bare.conditions["vehicle"]
    none, problems = av2.load_any(path, bare, matrix=MATRIX)
    assert none is None and any("conditions.vehicle" in p for p in problems)
    assert av2.load_any(path, None)[0] is None, "v1 without its manifest item cannot be migrated"


def test_annotation_and_manifest_must_agree_on_who_and_where():
    a = _ann()
    item = _item(1, "S01", session="S01-a")
    assert av2.consistency_problems(a, item) == []
    other = _item(1, "S02", vehicle="V02", session="S01-a")
    problems = av2.consistency_problems(a, other)
    assert any("clip.subject" in p for p in problems) and any("clip.vehicle" in p for p in problems)
    assert any("clip.occupants" in p for p in av2.consistency_problems(_ann(clip={"occupants": ["S09"]}), item))


# -- agreement -------------------------------------------------------------------------------

def test_event_iou_onset_agreement_and_state_kappa_known_answers():
    assert agr.iou(Span(0, 2), Span(1, 3)) == pytest.approx(1 / 3)
    m = agr.event_agreement([Span(10, 12), Span(20, 22)], [Span(10.2, 12.2), Span(30, 31)], iou_min=0.5)
    assert (m["matched"], m["only_a"], m["only_b"]) == (1, 1, 1)
    assert m["f1"] == pytest.approx(0.5) and m["mean_iou"] == pytest.approx(1.8 / 2.2)
    assert m["onset_p50_s"] == pytest.approx(0.2) and m["onset_max_s"] == pytest.approx(0.2)
    # 10 samples: a on 0-4, b on 0-2 -> observed 0.8, expected 0.5*0.3+0.5*0.7=0.5, kappa 0.6
    assert agr.cohen_kappa([Span(0, 0.45)], [Span(0, 0.25)], 1.0, 0.1) == pytest.approx(0.6)
    assert agr.cohen_kappa([], [], 1.0, 0.1) == 1.0, "both say never: perfect agreement"


def test_double_annotation_pass_fail_and_refusals():
    a = _ann(events=[{"type": "long_closure", "start_s": 10, "end_s": 13}], states={"eyes_closed": [[10, 13]]})
    b = _ann(annotator="A2", events=[{"type": "long_closure", "start_s": 10.1, "end_s": 13}],
             states={"eyes_closed": [[10.1, 13]]})
    good = agr.compare(a, b, duration_s=60)
    assert good["passes"], good["failures"]
    late = _ann(annotator="A2", events=[{"type": "long_closure", "start_s": 11.0, "end_s": 13.5}],
                states={"eyes_closed": [[11, 13]]})
    bad = agr.compare(a, late, duration_s=60)
    assert not bad["passes"] and any("onset p90" in f for f in bad["failures"])
    assert not agr.compare(a, a)["passes"] and "two annotators" in agr.compare(a, a)["refused"][0]


# -- 3. leakage and the split lock -----------------------------------------------------------

def _manifest(items) -> mf.Manifest:
    return mf.Manifest(name="incabin", version="1", licence={"class": "own_collection", "id": "SD consent v1"},
                       items=items, consent={"protocol": "SD-CONSENT-1", "retention_until": "2099-01-01"})


def test_subject_session_vehicle_day_and_co_occupant_leaks_are_refused():
    clean = _manifest([_item(1, "S01", "train", vehicle="V01"), _item(2, "S02", "test", vehicle="V02")])
    assert leakage.check(clean) == [] and mf.validate(clean) == []
    same_car_day = _manifest([_item(1, "S01", "train"), _item(2, "S02", "test")])
    assert any("vehicle-day 'V01@2026-12-05'" in p for p in leakage.check(same_car_day))
    passenger = _manifest([_item(1, "S01", "train", vehicle="V01", occupants="S02"),
                           _item(2, "S02", "test", vehicle="V02")])
    assert any("person 'S02'" in p for p in leakage.check(passenger))
    no_vehicle = _manifest([_item(1, "S01", "val")])
    del no_vehicle.items[0].conditions["vehicle"]
    assert any("vehicle-day leakage cannot be ruled out" in p for p in leakage.check(no_vehicle))
    unsplit = _manifest([_item(1, "S01")])
    assert any("no split assigned" in p for p in leakage.check(unsplit))


def test_split_unit_is_the_linked_subject_group_and_placement_never_moves_anything():
    items = [_item(n, f"S{n:02d}", vehicle=f"V{n % 9}", day=f"2026-12-{1 + n % 9:02d}") for n in range(40)]
    items.append(_item(40, "S40", vehicle="V3", day="2026-12-04"))  # shares a vehicle-day with S03
    items.append(_item(41, "S41", vehicle="V99", occupants="S05"))  # S05 rides along
    items[7].split = "test"
    m = _manifest(items)
    leakage.assign_subject_splits(m, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=3)
    assert leakage.check(m) == [], "linked subjects land together"
    by = {i.subject: i.split for i in m.items}
    assert by["S07"] == "test" and by["S40"] == by["S03"] and by["S41"] == by["S05"]
    groups = leakage.subject_groups(m)
    assert groups["S40"] == groups["S03"]
    again = _manifest([dataclasses.replace(i, split=None, conditions=dict(i.conditions)) for i in items])
    again.items[7].split = "test"
    leakage.assign_subject_splits(again, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=3)
    assert [i.split for i in again.items] == [i.split for i in m.items], "deterministic"
    spread = _manifest([_item(1, "S01", "train"), _item(2, "S02", "test"), _item(3, "S03")])
    with pytest.raises(leakage.LeakageError, match="already span splits"):
        leakage.assign_subject_splits(spread, {"train": 0.5, "val": 0.25, "test": 0.25})


def _square_clip(path: Path, positions: list[int], frames: int = 60) -> Path:
    """64x48 MJPG; frame n shows a white square at x = positions[n * len // frames]."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 10, (64, 48))
    for n in range(frames):
        frame = np.full((48, 64, 3), 30, np.uint8)
        x = positions[n * len(positions) // frames]
        frame[10:38, x:x + 20] = 255
        frame[20:28, 0:64:8] = 120  # texture, so dHash is not degenerate
        writer.write(frame)
    writer.release()
    return path


def test_near_duplicate_frames_anywhere_in_a_video_are_caught(tmp_path):
    _square_clip(tmp_path / "a.avi", [4])  # square on the left throughout
    _square_clip(tmp_path / "b.avi", [4, 40, 40, 40, 40, 40])  # opens on the same shot, then differs
    _square_clip(tmp_path / "c.avi", [22])  # a different shot throughout
    rows = [{"id": n, "path": f"{n}.avi", "subject": n, "session": n, "split": s}
            for n, s in (("a", "train"), ("b", "test"), ("c", "val"))]
    m = mf.build(tmp_path, rows, name="t", version="1", licence={"class": "evaluation_only", "id": "x"})
    assert mf.near_duplicates(m, tmp_path) == [], "the middle-frame check alone misses the shared opening"
    found = leakage.near_duplicate_frames(m, tmp_path, frames=5)
    assert len(found) == 1 and "a (train) ~ b (test)" in found[0]


def test_a_training_manifest_may_not_share_people_sessions_vehicle_days_or_files():
    ev = _manifest([_item(1, "S01", "val", vehicle="V01"), _item(2, "S02", "test", vehicle="V02")])
    tr = _manifest([_item(3, "S03", "train", vehicle="V03"), _item(4, "S04", "train", vehicle="V02")])
    problems = leakage.cross_manifest(tr, ev)
    assert len(problems) == 1 and "vehicle-day 'V02@2026-12-05'" in problems[0]
    assert leakage.cross_manifest(_manifest([_item(5, "S05", "train", vehicle="V05")]), ev) == []


def test_the_test_split_lock_is_hash_stamped_and_enforced(tmp_path):
    m = _manifest([_item(1, "S01", "train", vehicle="V1"), _item(2, "S02", "val", vehicle="V2"),
                   _item(3, "S03", "test", vehicle="V3")])
    lock_path = tmp_path / "split_lock.json"
    digest = leakage.write_lock(m, lock_path, created_at="2026-11-14T00:00:00+00:00")
    lock, problems = leakage.load_lock(lock_path)
    assert problems == [] and lock is not None and lock["lock_sha256"] == digest
    assert lock["test"]["people"] == ["S03"] and lock["test"]["vehicle_days"] == ["V3@2026-12-05"]
    assert leakage.verify_lock(m, lock) == ([], [])
    with pytest.raises(FileExistsError):
        leakage.write_lock(m, lock_path)

    tampered = json.loads(lock_path.read_text())
    tampered["test"]["items"] = []
    (tmp_path / "t.json").write_text(json.dumps(tampered))
    assert "edited after locking" in leakage.load_lock(tmp_path / "t.json")[1][0]

    moved = copy.deepcopy(m)
    moved.items[2].split = "val"
    assert any("moved from test to val" in p for p in leakage.verify_lock(moved, lock)[0])
    grown = copy.deepcopy(m)
    grown.items.append(_item(4, "S04", "test", vehicle="V4"))
    assert any("joined the locked test split" in p for p in leakage.verify_lock(grown, lock)[0])
    sneaky = copy.deepcopy(m)
    sneaky.items.append(_item(5, "S03", "train", vehicle="V9", session="S03-b"))
    assert any("shares a person or vehicle-day" in p for p in leakage.verify_lock(sneaky, lock)[0])
    withdrawn = copy.deepcopy(m)
    withdrawn.items.pop(0)
    problems, notes = leakage.verify_lock(withdrawn, lock)
    assert problems == [] and any("removed since the lock" in n for n in notes), "withdrawal is allowed"
    leaky = _manifest([_item(1, "S01", "train"), _item(2, "S02", "test")])
    with pytest.raises(leakage.LeakageError, match="vehicle-day"):
        leakage.write_lock(leaky, tmp_path / "other.json")


def test_scoring_the_test_split_needs_the_flag_and_every_access_is_recorded(tmp_path):
    m = _manifest([_item(1, "S01", "val", vehicle="V1"), _item(3, "S03", "test", vehicle="V3")])
    lock_path = tmp_path / "lock.json"
    leakage.write_lock(m, lock_path)
    with pytest.raises(leakage.TestAccessRefusedError, match="--frozen-test"):
        leakage.authorize_test_access(lock_path, frozen_test=False, purpose="x", commit="c", config_sha256="h")
    with pytest.raises(leakage.TestAccessRefusedError, match="purpose"):
        leakage.authorize_test_access(lock_path, frozen_test=True, purpose=" ", commit="c", config_sha256="h")
    first = leakage.authorize_test_access(lock_path, frozen_test=True, purpose="release 1", commit="c1",
                                          config_sha256="h1", now="2026-11-14T00:00:00+00:00")
    second = leakage.authorize_test_access(lock_path, frozen_test=True, purpose="release 2", commit="c2",
                                           config_sha256="h2")
    assert (first["access_number"], second["access_number"]) == (1, 2)
    log = [json.loads(x) for x in Path(str(lock_path) + ".access.jsonl").read_text().splitlines()]
    assert [r["purpose"] for r in log] == ["release 1", "release 2"] and log[0]["lock_sha256"] == first["lock_sha256"]


# -- 4. event-level metrics (known answers) --------------------------------------------------

def _scored_clip() -> em.ClipInput:
    a = _ann(events=[{"type": "long_closure", "start_s": 10, "end_s": 13},
                     {"type": "long_closure", "start_s": 50, "end_s": 53},
                     {"type": "eyes_off_road", "start_s": 100, "end_s": 104},
                     {"type": "glance_mirror_left", "start_s": 200, "end_s": 201},
                     {"type": "short_closure", "start_s": 300, "end_s": 301}],
             ignore=[{"start_s": 0, "end_s": 5, "reason": "calibration"}],
             uncertain=[{"start_s": 400, "end_s": 410, "reason": "ambiguous_eye_state", "types": ["long_closure"]}])
    predictions = [(12.1, "drowsiness"), (13.5, "drowsiness"), (103.0, "distraction"), (200.5, "distraction"),
                   (300.9, "drowsiness"), (405.0, "drowsiness"), (2.0, "drowsiness"), (1000.0, "eyes_down"),
                   (20.0, "phone_detected")]
    return em.ClipInput("c1", "S01", a, 3600.0, predictions, {"drowsiness": [Span(49, 54)]})


def test_score_clip_counts_every_outcome_with_known_answers():
    s = em.score_clip(_scored_clip(), MATRIX, SUSTAIN)
    lc, eor = s.detection["long_closure"], s.detection["eyes_off_road"]
    assert (lc.events, lc.caught, lc.missed, lc.abstained) == (2, 1, 1, 1)
    assert lc.latencies == [pytest.approx(2.1)] and lc.excess == [pytest.approx(0.1)]
    assert (eor.events, eor.caught, eor.latencies, eor.excess) == (1, 1, [3.0], [0.0])
    d = s.alerts["drowsiness"]
    assert (d.predictions, d.true_first, d.repeats, d.boundary, d.not_scored, d.false_alerts) == (3, 1, 1, 1, 2, 0)
    assert d.exposure_s == pytest.approx(3600 - 5 - 10) and d.unavailable_s == pytest.approx(5.0)
    x = s.alerts["distraction"]
    assert (x.predictions, x.true_first, x.false_alerts, x.false_on_negative) == (2, 1, 1, 1)
    assert x.exposure_s == pytest.approx(3595), "an uncertain span for long_closure does not cut distraction time"
    assert s.alerts["eyes_down"].false_alerts == 1 and s.alerts["phone_use"].predictions == 0
    assert (s.negatives["glance_mirror_left"].events, s.negatives["glance_mirror_left"].alerted) == (1, 1)
    assert s.boundary_events == {"short_closure": 1} and s.unscored_types == {"phone_detected": 1}
    assert s.uncertain_s == pytest.approx(10.0)


def test_one_alert_can_answer_concurrent_positives_and_abstention_needs_every_alert_unavailable():
    a = _ann(events=[{"type": "phone_use", "start_s": 10, "end_s": 15},
                     {"type": "eyes_off_road", "start_s": 10, "end_s": 15},
                     {"type": "eyes_off_road", "start_s": 100, "end_s": 104}])
    s = em.score_clip(em.ClipInput("c", "S01", a, 600.0, [(12.0, "phone_use")],
                                   {"distraction": [Span(99, 105)], "eyes_down": [Span(99, 105)]}), MATRIX, SUSTAIN)
    assert s.detection["phone_use"].caught == 1 and s.detection["eyes_off_road"].caught == 1
    assert s.alerts["phone_use"].true_first == 1 and s.alerts["phone_use"].false_alerts == 0
    assert s.detection["eyes_off_road"].abstained == 0, "eyes_up and phone_use were still available"
    s2 = em.score_clip(em.ClipInput("c", "S01", a, 600.0, [], None), MATRIX, SUSTAIN)
    assert s2.alerts["distraction"].unavailable_s is None and not s2.detection["eyes_off_road"].abstention_known


def test_interval_helpers():
    assert em.union([Span(5, 7), Span(0, 2), Span(1, 3)], 0, 6) == [Span(0, 3), Span(5, 6)]
    assert em.length([Span(0, 2), Span(1, 3)]) == 3
    assert em.overlap([Span(0, 10)], [Span(2, 3), Span(5, 20)]) == 6


def _two_subjects() -> dict[str, em.Scores]:
    def subject(predictions, extra_fa):
        a = _ann(events=[{"type": "long_closure", "start_s": 10, "end_s": 13},
                         {"type": "long_closure", "start_s": 60, "end_s": 63},
                         {"type": "glance_mirror_left", "start_s": 100, "end_s": 101}])
        preds = predictions + [(500.0 + k, "drowsiness") for k in range(extra_fa)]
        return em.score_clip(em.ClipInput("c", "S", a, 1800.0, preds, {}), MATRIX, SUSTAIN)
    s1 = subject([(12.0, "drowsiness"), (62.5, "drowsiness")], extra_fa=1)  # 2/2 caught, 1 FA in 0.5 h
    s2 = subject([(13.0, "drowsiness")], extra_fa=3)  # 1/2 caught, 3 FA in 0.5 h
    return {"S01": s1, "S02": s2}


def test_summary_reports_every_metric_when_the_power_check_passes():
    per = _two_subjects()
    out = em.summarise(per, matrix=_small_matrix(), minimums=SMALL, bootstrap_n=200)
    lc = out["detection"]["long_closure"]
    assert lc["status"] == em.OK and lc["n"] == {"subjects": 2, "events": 4, "excluded": 0}
    c = lc["claim"]
    assert c["recall"] == pytest.approx(0.75) and c["missed"] == 1
    assert (c["missed_abstained"], c["missed_silent"]) == (0, 1)
    assert c["latency_s"] == {"n": 3, "p50": pytest.approx(2.5), "p90": pytest.approx(3.0), "max": pytest.approx(3.0)}
    assert c["latency_minus_sustain_s"]["p50"] == pytest.approx(0.5)
    assert c["recall_wilson95"] is not None and c["recall_subject_bootstrap95"] is not None
    dz = out["alerts"]["drowsiness"]
    assert dz["status"] == em.OK
    assert dz["claim"]["false_alerts"] == 4 and dz["claim"]["false_alerts_per_hour"] == pytest.approx(4.0)
    assert dz["claim"]["false_alerts_per_hour_by_subject"] == {"S01": pytest.approx(2.0), "S02": pytest.approx(6.0)}
    assert dz["claim"]["precision"] == pytest.approx(3 / 7) and dz["claim"]["unknown_rate"] == 0.0
    lo, hi = dz["claim"]["false_alerts_per_hour_subject_bootstrap95"]
    assert 2.0 <= lo <= hi <= 6.0
    neg = out["negatives"]["glance_mirror_left"]
    assert neg["status"] == em.OK and neg["claim"]["alert_rate"] == 0.0


def test_the_power_check_refuses_a_claim_below_the_cell_minimum():
    per = _two_subjects()
    out = em.summarise(per, matrix=MATRIX, minimums=MATRIX.overall, bootstrap_n=50)
    lc = out["detection"]["long_closure"]
    assert lc["status"] == em.INSUFFICIENT and lc["claim"] is None
    assert lc["reasons"] == ["2/30 subjects", "2/30 clips", "4/60 events"]
    assert "diagnostic_not_a_claim" not in lc
    assert out["alerts"]["drowsiness"]["claim"] is None
    assert "1.00/3.0 exposure hours" in out["alerts"]["drowsiness"]["reasons"]
    diag = em.summarise(per, matrix=MATRIX, minimums=MATRIX.overall, include_diagnostics=True, bootstrap_n=50)
    assert diag["detection"]["long_closure"]["diagnostic_not_a_claim"]["recall"] == pytest.approx(0.75)
    assert diag["detection"]["long_closure"]["claim"] is None, "a diagnostic is never promoted to a claim"
    empty = em.summarise({}, matrix=MATRIX, minimums=MATRIX.overall)
    assert {m["status"] for g in ("detection", "alerts", "negatives") for m in empty[g].values()} == {em.NO_DATA}
    assert em.power(subjects=40, clips=40, minimums=MATRIX.overall, events=0)[0] == em.NO_DATA
    assert em.power(subjects=40, clips=40, minimums=MATRIX.overall, events=60) == (em.OK, [])


# -- 5. the report ---------------------------------------------------------------------------

def test_the_no_data_report_says_so_everywhere(tmp_path):
    assert rp.main(["--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "eval_report.json").read_text())
    md = (tmp_path / "eval_report.md").read_text()
    assert report["status"] == "NO DATA" and report["data"]["clips"] == 0
    assert report["provenance"]["matrix"]["sha256"] == MATRIX.sha256
    assert report["provenance"]["config_sha256"] and report["provenance"]["model_artifacts"]
    statuses = {m["status"] for s in [report["overall"], *(c["summary"] for c in report["cells"].values())]
                for g in ("detection", "alerts", "negatives") for m in s[g].values()}
    assert statuses == {em.NO_DATA}
    assert "**Status: NO DATA**" in md and set(report["cells"]) == set(MATRIX.cells)
    metric_rows = [r for r in md.splitlines() if r.startswith("| ") and not r.startswith(("| field", "| event", "| alert"))
                   and "---" not in r and "|" in r[2:] and not r.startswith(("| generated", "| commit", "| config",
                                                                            "| module", "| sustain", "| scenario",
                                                                            "| annotation", "| split", "| test",
                                                                            "| manifest", "| model"))]
    assert metric_rows and all("NO DATA" in r for r in metric_rows)


def _write_dataset(tmp_path: Path, *, subjects=("S01", "S02"), test_subject="S09") -> tuple[mf.Manifest, Path]:
    """Tiny files (never decoded), v2 annotations, a replay run directory and a split lock."""
    rows, run_preds, run_avail = [], [], []
    plan = [(s, "val", f"V{n}") for n, s in enumerate(subjects)] + [(test_subject, "test", "V9")]
    for n, (subject, split, vehicle) in enumerate(plan):
        (tmp_path / f"{n}.avi").write_bytes(f"clip {n}".encode())
        doc = _doc(item=f"c{n}", clip={"subject": subject, "session": f"{subject}-a", "vehicle": vehicle,
                                       "duration_s": 1800.0},
                   events=[{"type": "long_closure", "start_s": 10, "end_s": 13},
                           {"type": "long_closure", "start_s": 60, "end_s": 63}])
        (tmp_path / f"{n}.json").write_text(json.dumps(doc))
        rows.append({"id": f"c{n}", "path": f"{n}.avi", "subject": subject, "session": f"{subject}-a",
                     "split": split, "annotations": f"{n}.json", "cond_vehicle": vehicle,
                     "cond_recorded_on": "2026-12-05", "cond_seat": "driver"})
        run_preds += [{"item": f"c{n}", "t": 12.0, "type": "drowsiness"}, {"item": f"c{n}", "t": 900.0,
                                                                          "type": "drowsiness"}]
        run_avail.append({"item": f"c{n}", "duration_s": 1800.0, "frames": 18000,
                          "unavailable": {"phone_use": [[0.0, 1800.0]]}})
    m = mf.build(tmp_path, rows, name="incabin", version="1", licence={"class": "own_collection", "id": "SD v1"},
                 consent={"protocol": "SD-CONSENT-1", "retention_until": "2099-01-01"})
    leakage.write_lock(m, tmp_path / "lock.json")
    for split in ("val", "test"):
        run = tmp_path / f"run_{split}"
        run.mkdir()
        ids = {i.id for i in m.split(split)}
        (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run_preds if r["item"] in ids))
        (run / "availability.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run_avail if r["item"] in ids))
        (run / "report.json").write_text(json.dumps({
            "dataset": {"manifest_id": m.id, "split": split}, "commit": "abc", "config_sha256": "cfg",
            "overrides": {}, "backend": {"detector": "desktop surrogate"},
            "split_lock_sha256": leakage.load_lock(tmp_path / "lock.json")[0]["lock_sha256"]}))  # type: ignore[index]
    return m, tmp_path


def _generate(tmp_path, m, **kw):
    args = {"matrix": MATRIX, "manifest": m, "root": tmp_path, "split": "val", "run_dir": tmp_path / "run_val",
            "lock_path": tmp_path / "lock.json", "config_sha256": "cfg", "commit": "abc", "bootstrap_n": 100,
            "today": "2026-11-14T00:00:00+00:00"}
    args.update(kw)
    return rp.generate(**args)


def test_report_end_to_end_records_provenance_and_refuses_claims_below_minimum(tmp_path):
    m, root = _write_dataset(tmp_path)
    report = _generate(root, m, config={"modules": {"driver_attention": {"drowsy_seconds": 2.0}}},
                       artifacts=[{"source": "--model", "path": "x.hef", "sha256": "f" * 64}])
    assert report["status"] == "INSUFFICIENT DATA"
    p = report["provenance"]
    assert p["split_lock_sha256"] == leakage.load_lock(root / "lock.json")[0]["lock_sha256"]  # type: ignore[index]
    assert p["manifest"]["id"] == m.id and p["run"]["backend"]["detector"] == "desktop surrogate"
    assert p["sustain_s"] == {"drowsiness": 2.0} and p["model_artifacts"][0]["sha256"] == "f" * 64
    assert report["data"]["clips"] == 2 and report["data"]["subjects"] == 2
    lc = report["overall"]["detection"]["long_closure"]
    assert lc["status"] == em.INSUFFICIENT and lc["claim"] is None and lc["n"]["events"] == 4
    md = rp.render_markdown(report)
    assert "INSUFFICIENT DATA (2/30 subjects, 2/30 clips, 4/60 events)" in md
    assert "NO DATA (not recorded)" not in md
    small = _generate(root, m, matrix=_small_matrix())
    lc = small["overall"]["detection"]["long_closure"]
    assert small["status"].endswith("CLAIMABLE METRIC(S)") and lc["claim"]["recall"] == 0.5
    assert small["overall"]["alerts"]["drowsiness"]["claim"]["false_alerts_per_hour"] == pytest.approx(2.0)
    assert small["overall"]["alerts"]["phone_use"]["claim"]["unknown_rate"] == pytest.approx(1.0)
    assert small["cells"]["E3_sustained_closure"]["summary"]["clips"] == 2
    assert small["cells"]["L1_day"]["summary"]["clips"] == 0
    assert "| S01 | 2.000 |" in rp.render_markdown(small)


def test_report_refusals(tmp_path):
    m, root = _write_dataset(tmp_path)
    with pytest.raises(rp.ReportRefusedError, match="split lock is required"):
        _generate(root, m, lock_path=None)
    with pytest.raises(rp.ReportRefusedError, match="--frozen-test"):
        _generate(root, m, split="test", run_dir=root / "run_test")
    with pytest.raises(rp.ReportRefusedError, match="replayed split 'val', not 'test'"):
        _generate(root, m, split="test", frozen_test=True, purpose="final")
    with pytest.raises(rp.ReportRefusedError, match="config sha256 differs"):
        _generate(root, m, config_sha256="other")
    moved = copy.deepcopy(m)
    moved.items[0].split = "test"
    with pytest.raises(rp.ReportRefusedError, match="moved from val to test"):
        _generate(root, moved)
    leaky = copy.deepcopy(m)
    leaky.items[2].conditions["vehicle"] = "V0"  # the test subject recorded in S01's car on S01's day
    with pytest.raises(rp.ReportRefusedError, match="vehicle-day 'V0@2026-12-05'"):
        _generate(root, leaky)
    (tmp_path / "x").mkdir()
    other, _ = _write_dataset(tmp_path / "x", subjects=("S05", "S06"))
    with pytest.raises(rp.ReportRefusedError, match="the run replayed manifest"):
        _generate(root, other, root=tmp_path / "x", lock_path=tmp_path / "x" / "lock.json")
    doc = json.loads((root / "0.json").read_text())
    doc["clip"]["subject"] = "S77"
    (root / "0.json").write_text(json.dumps(doc))
    with pytest.raises(rp.ReportRefusedError, match=r"clip\.subject 'S77'"):
        _generate(root, m)
    assert not Path(str(root / "lock.json") + ".access.jsonl").exists(), "refused runs are not test accesses"


def test_report_on_the_locked_test_split_records_the_access(tmp_path):
    m, root = _write_dataset(tmp_path)
    report = _generate(root, m, split="test", run_dir=root / "run_test", frozen_test=True, purpose="M10 dry run")
    assert report["provenance"]["test_access"]["access_number"] == 1
    assert report["provenance"]["test_access"]["purpose"] == "M10 dry run"
    assert report["data"]["subjects"] == 1 and report["status"] == "INSUFFICIENT DATA"


# -- 6. end to end through replay.py with the real driver-attention module ------------------

def test_replay_with_the_real_module_feeds_latency_and_abstention(tmp_path):
    from test_driver_attention_scenarios import _CONFIG, _FakeFaceMeshResult, make_fake_landmarks

    from modules.driver_attention import DriverAttentionModule

    fps, w, h = 10, 640, 480
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (w, h))
    for n in range(100):  # 10 s: neutral, head down 3.0-7.0 s (bright frames), neutral
        writer.write(np.full((h, w, 3), 200 if 30 <= n < 70 else 100, np.uint8))
    writer.release()
    neutral, down = make_fake_landmarks(0.0, 0.0), make_fake_landmarks(pitch_deg=30.0)
    config = {**_CONFIG, "calibration_path": str(tmp_path / "cal.json"), "enable_hands": False}

    def factory():
        module = DriverAttentionModule(dict(config))
        module._face_mesh.process = lambda rgb: _FakeFaceMeshResult(down if rgb.mean() > 150 else neutral)  # type: ignore[method-assign]
        return module

    result = replay(factory, str(path), resize=None)
    assert result.settings["availability_declared"] is True
    spans = result.unavailable_spans()
    assert spans is not None
    assert spans["phone_use"] == [(0.0, pytest.approx(10.0))], "no object detector ran: phone_use never evaluable"
    assert spans["eyes_down"][0][0] == 0.0 and 0.5 <= spans["eyes_down"][0][1] <= 2.0, "calibrating first"
    assert "drowsiness" not in spans, "drowsiness runs on the fallback threshold while calibrating"

    annotation = _ann(clip={"duration_s": 10.0}, scenarios=["H1_head_orientation"],
                      tags={"lighting": "day", "eyewear": "none", "occupancy": "driver_only", "mount": "dash_centre",
                            "head_motion": "pitch_sweep"},
                      events=[{"type": "eyes_off_road", "start_s": 3.0, "end_s": 7.0}])
    unavailable = {k: [Span(a, b) for a, b in v] for k, v in spans.items()}
    clip = em.ClipInput("clip", "S01", annotation, result.duration_s, [(e.t, e.type) for e in result.events],
                        unavailable)
    s = em.score_clip(clip, MATRIX, SUSTAIN)
    det = s.detection["eyes_off_road"]
    assert det.caught == 1, [(e.t, e.type) for e in result.events]
    assert 2.5 <= det.latencies[0] <= 3.5 and 0.0 <= det.excess[0] <= 1.0
    assert s.alerts["eyes_down"].false_alerts == 0
    assert s.alerts["phone_use"].unavailable_s == pytest.approx(10.0)
    again = replay(factory, str(path), resize=None)
    assert [(e.t, e.type) for e in again.events] == [(e.t, e.type) for e in result.events], "deterministic"


def test_run_cli_scores_the_test_split_only_against_a_verified_lock(tmp_path, capsys):
    from evaluation import run

    m, root = _write_dataset(tmp_path)
    mf.save(m, root / "m.json")
    (root / "config.yaml").write_text("module: driver_attention\nmodules: {driver_attention: {}}\n")
    base = ["--manifest", str(root / "m.json"), "--root", str(root), "--config", str(root / "config.yaml"),
            "--out", str(root / "out")]
    assert run.main([*base, "--split", "test", "--frozen-test"]) == 2
    assert "--split-lock" in capsys.readouterr().err
    tampered = json.loads((root / "lock.json").read_text())
    tampered["test"]["people"] = []
    (root / "bad_lock.json").write_text(json.dumps(tampered))
    assert run.main([*base, "--split", "val", "--split-lock", str(root / "bad_lock.json")]) == 2
    assert "edited after locking" in capsys.readouterr().err
    moved = copy.deepcopy(m)
    moved.items[0].split = "test"
    mf.save(moved, root / "moved.json")
    args = [*base, "--split", "val", "--split-lock", str(root / "lock.json")]
    args[1] = str(root / "moved.json")
    assert run.main(args) == 2
    assert "moved from val to test" in capsys.readouterr().err
    assert not (root / "out").exists(), "nothing is written for a refused run"
