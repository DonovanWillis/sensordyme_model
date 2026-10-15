"""Regression tests for the safety-module review (findings F1-F14).

Each test was written against the shipped code first and failed there for
the reason its docstring gives. Most are MISSED-DETECTION cases: an alert
that silently never fired, or fired once and then latched or resolved while
the person was still in danger. Where a fix could plausibly over-correct, the
paired negative case sits next to it.

The fakes mirror the existing suites (test_phase1_safety_fall.py,
test_occupancy_zones.py, test_phase3_baby_infant_identification.py). No camera,
no MediaPipe inference and no detector are involved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest

import modules.baby_monitoring as baby_monitoring_module
import modules.fall_detection as fall_detection_module
import modules.occupancy as occupancy_module
from core.accel import COCO_NAMES
from core.activity import ActivityState, Attribute
from core.detectors import upper_region
from core.hand_perception import HandPerceptionResult
from core.tracking import TrackedBox
from core.zones import Zone, load_zones
from modules.baby_monitoring import MOUTH_ZONE_FRACTION, BabyMonitoringModule, InfantIdentifier
from modules.base import VisionModule
from modules.fall_detection import FallDetectionModule
from modules.occupancy import OccupancyModule

# ============================================================== shared fakes
W, H = 1000, 1000
CRIB = {"crib": {"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.9], [0.2, 0.9]], "kind": "safe"}}
IN_CRIB = (400, 300, 600, 800)
OUT_OF_CRIB = (850, 300, 980, 800)
INFANT = (500, 400, 700, 850)


@dataclass
class _Landmark:
    x: float
    y: float
    visibility: float = 1.0


class _FakeLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _FakeResults:
    def __init__(self, landmarks):
        self.pose_landmarks = _FakeLandmarkList(landmarks) if landmarks is not None else None


class _FakePose:
    def __init__(self):
        self.next_landmarks = None

    def process(self, rgb):
        return _FakeResults(self.next_landmarks)

    def close(self):
        pass


def _blank() -> list[_Landmark]:
    return [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]


def _standing(x: float = 0.3, y_hip: float = 0.3, ankles_visible: bool = True) -> list[_Landmark]:
    marks = _blank()
    marks[0] = _Landmark(x, y_hip - 0.35)
    marks[11], marks[12] = _Landmark(x - 0.05, y_hip - 0.25), _Landmark(x + 0.05, y_hip - 0.25)
    marks[23], marks[24] = _Landmark(x - 0.05, y_hip), _Landmark(x + 0.05, y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.2), _Landmark(x + 0.05, y_hip + 0.2)
    vis = 1.0 if ankles_visible else 0.0
    marks[27] = _Landmark(x - 0.05, y_hip + 0.4, vis)
    marks[28] = _Landmark(x + 0.05, y_hip + 0.4, vis)
    return marks


def _lying(x: float = 0.3, y_center: float = 0.85) -> list[_Landmark]:
    marks = _blank()
    marks[0] = _Landmark(x - 0.4, y_center)
    marks[11], marks[12] = _Landmark(x - 0.2, y_center - 0.02), _Landmark(x - 0.2, y_center + 0.02)
    marks[23], marks[24] = _Landmark(x + 0.2, y_center - 0.02), _Landmark(x + 0.2, y_center + 0.02)
    marks[25], marks[26] = _Landmark(x + 0.35, y_center - 0.02), _Landmark(x + 0.35, y_center + 0.02)
    marks[27], marks[28] = _Landmark(x + 0.5, y_center - 0.02), _Landmark(x + 0.5, y_center + 0.02)
    return marks


def _sitting(x: float = 0.3, y_hip: float = 0.5) -> list[_Landmark]:
    """Upright torso with the legs folded toward the camera (core/posture.py
    reads a leg/torso span ratio under 0.85 as sitting)."""
    marks = _standing(x=x, y_hip=y_hip)
    marks[25], marks[26] = _Landmark(x - 0.05, y_hip + 0.08), _Landmark(x + 0.05, y_hip + 0.08)
    marks[27], marks[28] = _Landmark(x - 0.05, y_hip + 0.12), _Landmark(x + 0.05, y_hip + 0.12)
    return marks


def _low_torso_visibility(landmarks: list[_Landmark], visibility: float = 0.3) -> list[_Landmark]:
    out = [_Landmark(lm.x, lm.y, lm.visibility) for lm in landmarks]
    for i in (11, 12, 23, 24):
        out[i] = _Landmark(out[i].x, out[i].y, visibility)
    return out


FALL_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _fall_config(**overrides) -> dict:
    """The SHIPPED config.yaml fall_detection values."""
    config = {
        "drop_velocity_threshold": 0.6, "stillness_duration_s": 1.5,
        "stillness_velocity_threshold": 0.02, "still_max_motion": 0.01,
        "unusual_stillness_s": 600.0, "fall_pose_recovery_max_gap_s": 2.5,
        "fall_upright_recency_s": 0.5, "fall_pending_timeout_s": 6.0, "fall_resolve_after_s": 4.0,
    }
    config.update(overrides)
    return config


@pytest.fixture()
def make_fall(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(fall_detection_module.time, "monotonic", lambda: clock["now"])

    def _make(**overrides):
        module = FallDetectionModule(_fall_config(**overrides))
        module._pose.close()
        module._pose = _FakePose()
        module._clock = clock
        return module

    return _make


def _feed(module, landmarks, frames: int, dt: float = 0.1) -> list:
    events = []
    module._pose.next_landmarks = landmarks
    for _ in range(frames):
        events += module.process(FALL_FRAME)
        module._clock["now"] += dt
    return events


def _feed_sequence(module, frames: list, dt: float = 0.1) -> list:
    events = []
    for landmarks in frames:
        module._pose.next_landmarks = landmarks
        events += module.process(FALL_FRAME)
        module._clock["now"] += dt
    return events


def _types(events) -> list[str]:
    return [e.type for e in events]


def _confirmed_fall(module) -> list:
    events = _feed(module, _standing(), 5)
    events += _feed(module, _lying(), 1)
    events += _feed(module, _lying(), 25)
    assert "fall" in _types(events)
    assert module.is_alert_active()
    return events


def _baby(**overrides) -> BabyMonitoringModule:
    config = {"zone_confirm_s": 2.0}
    config.update(overrides)
    return BabyMonitoringModule(config)


def _run_baby(module, boxes, seconds, start=0.0, step=0.1, motion=0.05, hand_result_fn=None):
    events, t = [], start
    while t < start + seconds - 1e-9:
        events += module._reason(
            list(boxes), motion, W, H, t, (), None,
            hand_result=hand_result_fn(t) if hand_result_fn else None,
        )
        t = round(t + step, 6)
    return events, t


# ============================================== F1: app-drawn zones reach modules
def test_base_set_zones_is_a_harmless_default():
    class _NoZones(VisionModule):
        def process(self, frame):
            return []

    _NoZones({}).set_zones(load_zones(CRIB))  # must exist and do nothing


def test_baby_crib_drawn_in_the_app_raises_safe_zone_exit():
    """The app's "Stay inside" crib reached only the AI-rule engine. The
    module read zones once from config.yaml (shipped `{}`), so a baby leaving
    an app-drawn crib raised nothing, ever."""
    module = _baby()  # no config.yaml zones, the shipped default
    module.set_zones(load_zones(CRIB))
    events, t = _run_baby(module, [IN_CRIB], 3.0)
    assert "safe_zone_exit" not in _types(events)
    events, _ = _run_baby(module, [OUT_OF_CRIB], 4.0, start=t)
    assert "safe_zone_exit" in _types(events)


def test_baby_app_crib_feeds_infant_identification():
    """The crib zone is InfantIdentifier's strongest signal. It used to be
    reachable only by hand-editing config.yaml."""
    module = _baby()
    module.set_zones(load_zones(CRIB))
    _run_baby(module, [INFANT], 0.5)  # zone-qualified on the first frames, no 3 s warm-up
    assert module._infant_identifier.safe_zones
    assert module._last_infant_identification.identified is True


def test_set_zones_empty_list_removes_rules_and_unlatches_an_active_alert():
    stairs = {"stairs": {"polygon": [[0.0, 0.5], [0.15, 0.5], [0.15, 1.0], [0.0, 1.0]], "kind": "restricted"}}
    module = _baby(zones=stairs)
    events, t = _run_baby(module, [(20, 600, 120, 950)], 4.0)
    assert "zone_violation" in _types(events)
    assert module.is_alert_active()

    module.set_zones([])
    assert not module.is_alert_active(), "an alert for a deleted zone must not stay latched"
    events, _ = _run_baby(module, [(20, 600, 120, 950)], 4.0, start=t)
    assert "zone_violation" not in _types(events)


def test_resending_identical_zones_keeps_in_progress_state():
    module = _baby()
    module.set_zones(load_zones(CRIB))
    events, t = _run_baby(module, [OUT_OF_CRIB], 4.0)
    assert _types(events).count("safe_zone_exit") == 1
    module.set_zones(load_zones(CRIB))  # the uplink re-delivers the same zones
    events, _ = _run_baby(module, [OUT_OF_CRIB], 4.0, start=t)
    assert "safe_zone_exit" not in _types(events), "no reset and re-alert from a no-op update"
    assert module.is_alert_active()


def test_fall_bed_zone_drawn_in_the_app_raises_safe_zone_exit(make_fall):
    module = make_fall(zone_confirm_s=0.5)
    module.set_zones(load_zones({"bed": {"polygon": [[0.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]],
                                         "kind": "safe"}}))
    assert "safe_zone_exit" not in _types(_feed(module, _standing(x=0.3, y_hip=0.5), 10))
    assert "safe_zone_exit" in _types(_feed(module, _standing(x=0.9, y_hip=0.5), 10))
    module.set_zones([])
    assert not module.is_alert_active()


class _FakeBox:
    def __init__(self, xyxy, cls=0):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.float32(0.9)]
        self.cls = [np.float32(cls)]


class _FakeModel:
    def __init__(self):
        self.boxes_to_return = []
        self.last_classes = None

    def predict(self, frame, **kwargs):
        self.last_classes = kwargs.get("classes")
        return [type("R", (), {"boxes": [_FakeBox(b) for b in self.boxes_to_return]})()]


@pytest.fixture()
def occupancy(monkeypatch):
    fake = _FakeModel()
    clock = {"now": 0.0}
    monkeypatch.setattr(occupancy_module, "get_yolo_model", lambda weights: fake)
    monkeypatch.setattr(occupancy_module.time, "monotonic", lambda: clock["now"])

    def _make(zones=None):
        module = OccupancyModule({"confidence_threshold": 0.5, "confirm_frames": 2,
                                  "zones": zones or {}, "zone_confirm_s": 0.5, "linger_s": 3.0})
        module._fake, module._clock = fake, clock
        return module

    return _make


OCC_FRAME = np.zeros((1000, 1000, 3), dtype=np.uint8)
ENTRY = {"entry": {"polygon": [[0.6, 0.2], [0.95, 0.2], [0.95, 0.9], [0.6, 0.9]], "kind": "restricted"}}


def _run_occ(module, boxes, frames, dt=0.1):
    events = []
    module._fake.boxes_to_return = boxes
    for _ in range(frames):
        events += module.process(OCC_FRAME)
        module._clock["now"] += dt
    return events


def test_occupancy_restricted_zone_drawn_in_the_app_fires_and_can_be_removed(occupancy):
    module = occupancy()
    module.set_zones(load_zones(ENTRY))
    assert "zone_violation" in _types(_run_occ(module, [(700, 300, 900, 800)], 10))
    assert any(COCO_NAMES[c] == "car" for c in module._requested_class_ids)

    module.set_zones([])
    assert not module.is_alert_active()
    assert module._requested_class_ids == [0], "no zone, so no vehicle/animal classes requested"
    assert "zone_violation" not in _types(_run_occ(module, [(700, 300, 900, 800)], 10))


def test_removing_a_zone_mid_violation_reports_its_resolution(occupancy):
    """Deleting or redrawing a zone while its alert is live used to drop the
    rule with no *_resolved event, so the app and dashboard showed the alert
    as active for good."""
    module = occupancy()
    module.set_zones(load_zones(ENTRY))
    _run_occ(module, [(700, 300, 900, 800)], 10)
    assert module.is_alert_active()

    module.set_zones([])
    events = _run_occ(module, [], 1)

    assert "zone_violation_resolved" in _types(events)
    assert _types(_run_occ(module, [], 3)).count("zone_violation_resolved") == 0, "reported once"


def test_removing_a_bed_zone_mid_exit_reports_its_resolution(make_fall):
    module = make_fall(zone_confirm_s=0.5)
    module.set_zones(load_zones({"bed": {"polygon": [[0.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]],
                                         "kind": "safe"}}))
    _feed(module, _standing(x=0.3, y_hip=0.5), 10)
    assert "safe_zone_exit" in _types(_feed(module, _standing(x=0.9, y_hip=0.5), 10))

    module.set_zones([])

    assert "safe_zone_exit_resolved" in _types(_feed(module, _standing(x=0.9, y_hip=0.5), 1))


# ============================================== F2: confirmed fall stays down
def test_confirmed_fall_is_not_resolved_by_struggling_on_the_floor(make_fall):
    """config.yaml: a confirmed fall clears only once the person is "seen NOT
    lying down" for fall_resolve_after_s. The code also required stillness, so
    4 s of struggling on the floor produced `fall_resolved`. When they then
    lost consciousness, nothing alerted again."""
    module = make_fall()
    _confirmed_fall(module)
    struggle = [_lying(y_center=0.85 if i % 2 == 0 else 0.80) for i in range(50)]
    events = _feed_sequence(module, struggle)
    assert "fall_resolved" not in _types(events)
    assert module.is_alert_active()
    events = _feed(module, _lying(), 300)  # then motionless on the floor
    assert module.is_alert_active()
    assert "fall_resolved" not in _types(events)


def test_fall_found_still_down_after_tracking_loss_alerts_again(make_fall):
    """MediaPipe often loses someone lying on the floor. Losing them for
    resolve_after_s resolves the fall (BUG 3's fix, kept), but the pending
    state was wiped with it, so finding them still lying there afterwards
    produced nothing at all."""
    module = make_fall()
    _confirmed_fall(module)
    assert "fall_resolved" in _types(_feed(module, None, 50))
    events = _feed(module, _lying(), 60)
    assert "fall" in _types(events)
    assert module.is_alert_active()


def test_getting_up_while_untracked_is_still_a_recovery(make_fall):
    """Paired negative: resolved during a dropout and then seen confidently
    standing means they got up. No new fall."""
    module = make_fall()
    _confirmed_fall(module)
    _feed(module, None, 50)
    events = _feed(module, _standing(), 100)
    assert "fall" not in _types(events)
    assert module._fall_pending_since is None


def test_sitting_up_after_a_fall_ends_it_without_re_arming(make_fall):
    """Seen sitting up (in a chair, on the bed edge), the fall is over. Re-arming
    the pending state here made lying down still on a sofa within
    fall_pending_timeout_s a brand-new "fall" with no drop at all."""
    module = make_fall()
    _confirmed_fall(module)
    events = _feed(module, _sitting(), 60)
    assert "fall_resolved" in _types(events)
    assert module._fall_pending_since is None
    assert not module.is_alert_active()


# ============================================== F3: low-visibility torso gap
def test_fall_during_a_low_torso_visibility_gap_is_detected(make_fall):
    """A pose that is still returned, but whose torso landmarks are too
    low-visibility to measure (the gating added for blurry falls), used to be
    neither a 'tracking gap' nor a measurable drop. The same 1 s as a full pose
    loss was detected."""
    module = make_fall()
    events = _feed(module, _standing(y_hip=0.45), 5)
    events += _feed(module, _low_torso_visibility(_standing(y_hip=0.45)), 10)
    events += _feed(module, _lying(y_center=0.70), 60)
    assert "fall" in _types(events)


def test_low_visibility_gap_while_already_lying_is_not_a_fall(make_fall):
    """Paired negative: no confident upright reading right before the gap,
    so the recovery path must not open."""
    module = make_fall()
    events = _feed(module, _lying(), 30)
    events += _feed(module, _low_torso_visibility(_lying()), 10)
    events += _feed(module, _lying(), 60)
    assert "fall" not in _types(events)


# ============================================== F4: infant identity wedges
LEANING = (450, 250, 800, 980)
CAREGIVER_BESIDE = (700, 180, 900, 940)
HOLDING = (480, 150, 760, 960)
STEPPED_AWAY = (820, 150, 990, 960)


def test_a_spurious_duplicate_box_does_not_bar_the_infant_in_its_own_crib():
    """A duplicate box on the infant was recorded as 'somebody else'. When the
    infant's next box matched that duplicate's track, the infant was rejected
    for the track's whole life, even fully inside the crib zone."""
    module = _baby(zones=CRIB)
    _, t = _run_baby(module, [INFANT], 3.0)
    _, t = _run_baby(module, [INFANT, (520, 430, 660, 700)], 0.3, start=t)
    _, t = _run_baby(module, [(510, 420, 670, 720)], 60.0, start=t)  # legs under a blanket
    assert module._last_infant_identification.identified is True


def test_pick_up_and_put_down_never_identifies_the_caregiver_as_the_infant():
    """No crib zone (the shipped default). After a pick-up and put-down the
    caregiver's fresh track qualified on 3 s of persistence while the infant's
    own track was quarantined. The ADULT became 'the infant', the real infant
    was marked not-the-infant for good, and no warning was raised."""
    module = _baby()
    _, t = _run_baby(module, [INFANT], 6.0)
    _, t = _run_baby(module, [INFANT, CAREGIVER_BESIDE], 3.0, start=t)
    _, t = _run_baby(module, [HOLDING], 4.0, start=t)
    events = []
    end = t + 60.0
    while t < end:
        events += module._reason([INFANT, STEPPED_AWAY], 0.05, W, H, t)
        assert module._last_infant_identification.box != STEPPED_AWAY, f"caregiver identified at t={t}"
        t = round(t + 0.1, 6)
    assert "infant_unidentified" in _types(events), "declining must be reported"


def test_an_identity_break_clears_the_not_the_infant_mark():
    identifier = InfantIdentifier([])
    now = 0.0
    for _ in range(60):  # infant alone, identified
        identifier.update([INFANT], [], W, H, now)
        now += 0.1
    for _ in range(10):  # caregiver alongside -> recorded as somebody else
        identifier.update([INFANT, CAREGIVER_BESIDE], [], W, H, now)
        now += 0.1
    caregiver_track = next(tr.track_id for tr in identifier._tracker.tracks if tr.box == CAREGIVER_BESIDE)
    assert caregiver_track in identifier._not_the_infant
    # The infant's box is lost, and an infant-sized box lands on the caregiver's
    # track (same centroid): a different subject now holds that track id.
    identifier.update([(760, 500, 840, 620)], [], W, H, now)
    assert caregiver_track not in identifier._not_the_infant


def test_a_caregiver_leaning_in_is_still_not_mistaken_for_a_duplicate():
    """Paired negative for the duplicate exemption: a caregiver overlapping the
    infant (but mostly outside the infant's box) is still somebody else."""
    module = _baby()
    _, t = _run_baby(module, [INFANT], 5.0)
    _, t = _run_baby(module, [INFANT, LEANING], 5.0, start=t)
    leaning_track = next(tr.track_id for tr in module._infant_identifier._tracker.tracks if tr.box == LEANING)
    assert leaning_track in module._infant_identifier._not_the_infant


# ============================================== F5: identity check vs frame rate
def _jittered_infant_run(fps: float, jitter: float, seconds: float = 60.0, seed: int = 0) -> float:
    rnd = random.Random(seed)
    module = _baby(detect_every_n_frames=2)
    base = (300, 250, 400, 400)
    box, identified, n, t = base, 0, 0, 0.0
    for i in range(int(seconds * fps)):
        if i % 2 == 0:  # detection every 2nd frame; stale boxes in between, as process() does
            bw, bh = base[2] - base[0], base[3] - base[1]
            box = tuple(round(v + rnd.gauss(0, jitter * (bw if k % 2 == 0 else bh)))
                        for k, v in enumerate(base))
        module._reason([box], 0.05, 640, 480, t)
        identified += module._last_infant_identification.identified
        n += 1
        t += 1.0 / fps
    return identified / n


@pytest.mark.parametrize("fps", [15, 30])
def test_a_lone_infant_stays_identified_with_ordinary_detector_jitter_at_high_fps(fps):
    """The identity-break rates divided a per-detection box change by ONE frame
    interval. 2-3% box jitter at 30 fps broke the infant's identity over and
    over: identified 0-9% of the time with nobody else in frame. At 10 fps it
    was 99.5%. (The ceiling is 95%: the first 3 s of 60 are the no-zone
    observation warm-up.)"""
    assert _jittered_infant_run(fps, 0.03) > 0.9


def test_a_real_box_swap_still_breaks_identity_at_30fps():
    """Paired negative: a caregiver grafted onto the infant's track must still break it."""
    module = _baby()
    t, dt = 0.0, 1 / 30
    for _ in range(int(6 / dt)):
        module._reason([INFANT], 0.05, W, H, t)
        t += dt
    assert module._last_infant_identification.identified
    for _ in range(int(2 / dt)):
        module._reason([LEANING], 0.05, W, H, t)
        t += dt
    assert module._last_infant_identification.identified is False


# ============================================== F6: frame-bottom edge
def test_a_box_clipped_at_the_frame_bottom_is_inside_a_zone_that_reaches_it():
    """Detectors clip y2 to exactly the frame height, which normalizes to 1.0,
    and the ray cast is half-open, so such a person was outside EVERY zone,
    including config.yaml's own bottom-edge 'stairs' example."""
    stairs = Zone("stairs", ((0.0, 0.5), (0.15, 0.5), (0.15, 1.0), (0.0, 1.0)))
    assert stairs.contains_box((10, 300, 80, 480), 640, 480)
    whole = Zone("whole", ((0, 0), (1, 0), (1, 1), (0, 1)), "safe")
    assert whole.contains_box((200, 100, 400, 480), 640, 480)


def test_a_person_on_bottom_edge_stairs_raises_zone_violation_end_to_end():
    module = BabyMonitoringModule({"zone_confirm_s": 1.0, "zones": {
        "stairs": {"polygon": [[0.0, 0.5], [0.15, 0.5], [0.15, 1.0], [0.0, 1.0]], "kind": "restricted"}}})
    events = []
    for i in range(30):
        events += module._reason([(10, 300, 80, 480)], 0.05, 640, 480, i * 0.1)
    assert "zone_violation" in _types(events)


def test_feet_below_a_zone_that_stops_short_of_the_edge_are_still_outside():
    zone = Zone("pool", ((0.4, 0.4), (0.8, 0.4), (0.8, 0.9), (0.4, 0.9)))
    assert not zone.contains_box((500, 100, 700, 1000), 1000, 1000)


# ============================================== F7: intermittent detection
@pytest.mark.parametrize(("on_s", "off_s"), [(4, 4), (2, 6), (10, 10), (1.5, 20)])
def test_a_motionless_baby_detected_only_intermittently_still_alerts(on_s, off_s):
    """Stillness reset on any detection gap over 3 s. person_absent reset on
    any sighting over 1 s. So a motionless baby detected in runs alerted on
    NEITHER, measured over 30 minutes."""
    module = _baby()  # shipped defaults: still_warning_s 120, absence 30
    types, t = set(), 0.0
    while t < 600:
        present = (t % (on_s + off_s)) < on_s
        for event in module._reason([IN_CRIB] if present else [], 0.0, W, H, t):
            types.add(event.type)
        t = round(t + 0.1, 6)
    assert types & {"unusual_stillness", "person_absent"}


def test_an_active_baby_with_dropouts_still_never_reads_as_unusually_still():
    """Paired negative: widening the stillness grace must not let real motion be bridged."""
    module = _baby(still_warning_s=20.0)
    types, t = set(), 0.0
    while t < 300:
        phase = t % 12.0
        if phase < 1.0:
            boxes, motion = [IN_CRIB], 0.12   # moves
        elif phase < 4.0:
            boxes, motion = [IN_CRIB], 0.001  # holds still
        else:
            boxes, motion = [], 0.0           # detection dropout
        for event in module._reason(boxes, motion, W, H, t):
            types.add(event.type)
        t = round(t + 0.1, 6)
    assert "unusual_stillness" not in types


# ============================================== F8: zone alerts must not latch
BED = {"bed": {"polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], "kind": "safe"}}


def test_a_second_bed_exit_is_reported_after_feet_were_hidden_in_bed(make_fall):
    """With the ankles invisible the zone rules were skipped rather than fed,
    so a confirmed bed exit never resolved, kept the alert 'active' all
    night, and swallowed the next real bed exit."""
    module = make_fall(zones=BED)
    _feed(module, _standing(x=0.25), 30)
    assert "safe_zone_exit" in _types(_feed(module, _standing(x=0.8), 40))
    hidden = _feed(module, _standing(x=0.25, ankles_visible=False), 600)
    assert "safe_zone_exit_resolved" in _types(hidden)
    assert "safe_zone_exit" in _types(_feed(module, _standing(x=0.8), 100))


def test_a_confirmed_running_episode_does_not_latch_through_a_pose_dropout(make_fall, monkeypatch):
    module = make_fall()
    running = ActivityState(ready=True, running=Attribute("running", True, 0.9))
    monkeypatch.setattr(module._activity, "update", lambda *a, **k: running)
    assert "running" in _types(_feed(module, _standing(), 30))
    assert "running_ended" in _types(_feed(module, None, 40))


# ============================================== F9: infant_unidentified notifies
def test_infant_unidentified_is_an_alert_type_of_the_baby_preset():
    from main import load_presets

    assert "infant_unidentified" in load_presets()["baby_monitoring"]["alert_types"]


# ============================================== F10: no cooldown on safety rules
def test_a_second_crib_exit_alerts_promptly():
    """The engine's default 30 s cooldown refused even to start tracking a
    second exit, so it took ~29 s to alert instead of 2 s."""
    module = _baby(zones=CRIB)
    log, t = [], 0.0
    for boxes, secs in (([IN_CRIB], 5), ([OUT_OF_CRIB], 5), ([IN_CRIB], 8), ([OUT_OF_CRIB], 10)):
        events, t = _run_baby(module, boxes, secs, start=t)
        log += [(round(t, 1), e.type) for e in events if e.type == "safe_zone_exit"]
    assert len(log) == 2


def test_a_second_absence_alerts_after_absence_warning_s_not_a_cooldown_later():
    module = _baby()
    _, t = _run_baby(module, [IN_CRIB], 2)
    events, t = _run_baby(module, [], 35, start=t)
    assert "person_absent" in _types(events)
    _, t = _run_baby(module, [IN_CRIB], 6, start=t)
    events, _ = _run_baby(module, [], 32, start=t)
    assert "person_absent" in _types(events)


def test_a_second_stairs_entry_in_elderly_mode_alerts_promptly(make_fall):
    stairs = {"stairs": {"polygon": [[0.0, 0.6], [0.6, 0.6], [0.6, 1.0], [0.0, 1.0]], "kind": "restricted"}}
    module = make_fall(zones=stairs, zone_confirm_s=0.5)
    assert "zone_violation" in _types(_feed(module, _standing(x=0.3, y_hip=0.5), 10))
    assert "zone_violation_resolved" in _types(_feed(module, _standing(x=0.9, y_hip=0.5), 60))
    assert "zone_violation" in _types(_feed(module, _standing(x=0.3, y_hip=0.5), 15))


# ============================================== F11: stillness vs glitches
def test_fall_unusual_stillness_survives_small_landmark_jitter(make_fall):
    """One frame of 0.01 centroid movement locked the feed and restarted the
    600 s clock. sigma=0.004 jitter (about 2 px) on a motionless person on the
    floor restarted it 29 times in 15 minutes, and it never fired."""
    module = make_fall()
    rnd = random.Random(1)
    base = _lying()
    events, fps = [], 15
    for _ in range(int(700 * fps)):
        module._pose.next_landmarks = [
            _Landmark(lm.x + rnd.gauss(0, 0.004), lm.y + rnd.gauss(0, 0.004), lm.visibility) for lm in base
        ]
        events += module.process(FALL_FRAME)
        module._clock["now"] += 1.0 / fps
    assert "unusual_stillness" in _types(events)


def test_fall_stillness_ignores_landmarks_mediapipe_is_only_guessing_at(make_fall):
    module = make_fall(unusual_stillness_s=20.0)
    rnd = random.Random(2)
    events = []
    for _ in range(400):
        marks = _lying()
        for i in (23, 24):  # hips occluded: low visibility, wildly unstable guesses
            marks[i] = _Landmark(rnd.uniform(0, 1), rnd.uniform(0, 1), 0.1)
        events += _feed(module, marks, 1)
    assert "unusual_stillness" in _types(events)


def test_baby_motion_energy_ignores_a_single_frame_glitch():
    module = _baby()
    still = np.full((480, 640, 3), 100, dtype=np.uint8)
    flash = np.full((480, 640, 3), 220, dtype=np.uint8)
    readings = [module._motion_energy(f) for f in (still, still, still, flash, still, still, still)]
    assert max(readings) < 0.015, readings


def test_baby_motion_energy_still_sees_a_change_that_persists():
    module = _baby()
    dark = np.full((480, 640, 3), 60, dtype=np.uint8)
    lit = np.full((480, 640, 3), 200, dtype=np.uint8)
    readings = [module._motion_energy(f) for f in (dark, dark, lit, lit, lit)]
    assert max(readings) > 0.3


def test_baby_motion_outside_the_person_boxes_does_not_count():
    """A ceiling fan or crib mobile outside the baby's box kept whole-frame
    motion above the floor, so unusual_stillness could never accumulate."""
    module = _baby()
    rnd = np.random.default_rng(0)
    box = [(100, 100, 300, 400)]
    readings = []
    for _ in range(8):
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        frame[:, 400:] = rnd.integers(0, 255, size=(480, 240, 3), dtype=np.uint8)  # fan region
        readings.append(module._motion_energy(frame, box))
    assert max(readings) < 0.015, readings


# ============================================== F12: occupancy requested classes
def test_ai_rule_labels_do_not_drop_the_zone_vehicle_and_animal_classes(occupancy):
    module = occupancy(zones={"drive": {"polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                                        "kind": "restricted"}})
    names = lambda: {COCO_NAMES[c] for c in module._requested_class_ids}  # noqa: E731
    module.set_requested_labels({"bottle"})
    assert {"car", "dog", "bottle", "person"} <= names()
    module.set_requested_labels(set())
    assert {"car", "dog", "person"} <= names()
    assert "bottle" not in names()


# ============================================== F13: boundary zones, load_zones
def test_load_zones_tolerates_a_list_shaped_config():
    assert load_zones([{"polygon": [[0, 0], [1, 0], [1, 1]]}]) == []


def test_boundary_crossing_debounces_and_ignores_the_baseline():
    from core.zones import BoundaryCrossing

    watch = BoundaryCrossing(confirm_s=1.0)
    t, results = 0.0, []
    for inside in [True] * 20 + [False, True] * 3 + [False] * 20 + [None] * 5 + [True] * 20:
        results.append(watch.update(inside, t))
        t += 0.1
    assert [r for r in results if r] == ["left", "entered"]


LINE = {"line": {"polygon": [[0.6, 0.2], [0.95, 0.2], [0.95, 0.9], [0.6, 0.9]], "kind": "boundary"}}


def test_occupancy_boundary_alerts_on_crossing_not_on_standing_inside(occupancy):
    """A boundary ("Line" in the app) was treated as one more restricted zone:
    standing inside fired, crossing out did not."""
    module = occupancy(zones=LINE)
    assert not [e for e in _run_occ(module, [(100, 300, 300, 800)], 20) if e.type == "zone_violation"]
    entered = [e for e in _run_occ(module, [(700, 300, 900, 800)], 40) if e.type == "zone_violation"]
    assert len(entered) == 1 and "crossed (entered)" in entered[0].contributing_signals
    left = [e for e in _run_occ(module, [(100, 300, 300, 800)], 20) if e.type == "zone_violation"]
    assert len(left) == 1 and "crossed (left)" in left[0].contributing_signals


def test_occupancy_person_already_inside_a_boundary_at_start_is_not_a_crossing(occupancy):
    module = occupancy(zones=LINE)
    assert not [e for e in _run_occ(module, [(700, 300, 900, 800)], 60) if e.type == "zone_violation"]


def test_baby_boundary_zone_is_no_longer_silently_ignored():
    module = _baby(zones={"door": {"polygon": [[0.6, 0.2], [0.95, 0.2], [0.95, 0.9], [0.6, 0.9]],
                                   "kind": "boundary"}})
    events, t = _run_baby(module, [IN_CRIB], 3.0)
    events, _ = _run_baby(module, [OUT_OF_CRIB], 4.0, start=t)
    assert any(e.type == "zone_violation" and "crossed (entered)" in e.contributing_signals for e in events)


def test_fall_boundary_zone_reports_crossings(make_fall):
    module = make_fall(zones={"door": {"polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                                       "kind": "boundary"}}, zone_confirm_s=0.5)
    _feed(module, _standing(x=0.2), 20)
    events = _feed(module, _standing(x=0.8), 20)
    assert any(e.type == "zone_violation" and "crossed (entered)" in e.contributing_signals for e in events)


# ============================================== F14: no frames on disk by default
def _hand_result():
    hand = TrackedBox(track_id=1, label="right_hand", box=(100, 100, 200, 200), confidence=0.9,
                      first_seen=0.0, last_seen=0.0)
    return HandPerceptionResult(hands=[hand], objects=[], associations={})


@pytest.mark.parametrize("enabled", [False, True])
def test_hand_debug_snapshot_is_written_only_when_opted_in(enabled, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    written = []
    monkeypatch.setattr(baby_monitoring_module.cv2, "imwrite", lambda path, img: written.append(path) or True)
    module = _baby(hand_debug=enabled)
    module._detect_people = lambda frame: []
    module._model = object()

    def _fake_activity(frame, w, h, now):
        module._last_pose_landmarks = object()

    module._update_activity = _fake_activity
    module._hand_engine.update = lambda *a, **k: _hand_result()
    module.process(np.zeros((480, 640, 3), dtype=np.uint8))
    assert bool(written) is enabled


# ============================================== choking scoring sanity after F4/F5
def test_a_real_hazard_still_fires_for_an_app_drawn_crib():
    """End-to-end: a crib zone delivered through set_zones() lets a real hazard fire."""
    module = _baby()
    module.set_zones(load_zones(CRIB))
    mouth = upper_region(INFANT, MOUTH_ZONE_FRACTION)
    target = ((mouth[0] + mouth[2]) / 2.0, (mouth[1] + mouth[3]) / 2.0)
    start = (target[0] + 250, target[1] + 250)
    trail: list = []

    def hazard(t):
        frac = min(1.0, t / 0.6)
        x, y = start[0] + (target[0] - start[0]) * frac, start[1] + (target[1] - start[1]) * frac
        trail.append((t, x, y))
        del trail[:-15]
        obj = TrackedBox(track_id=1, label="bottle", box=(int(x - 20), int(y - 20), int(x + 20), int(y + 20)),
                         confidence=0.38, first_seen=trail[0][0], last_seen=t, hits=5, trail=list(trail))
        hand = TrackedBox(track_id=100, label="right_hand", box=(440, 330, 560, 460), confidence=0.9,
                          first_seen=0.0, last_seen=t, hits=5)
        return HandPerceptionResult(hands=[hand], objects=[obj], associations={"right_hand": (obj, 0.38)})

    events, _ = _run_baby(module, [INFANT], 2.4, hand_result_fn=hazard)
    assert "choking_hazard" in _types(events)
