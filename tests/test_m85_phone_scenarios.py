"""M8.5: phone HANDLING vs prolonged OFF-ROAD ATTENTION, through the REAL module.

sim.scenario.M85_PHONE_SCENARIOS (and the four M8.4 cabin scenarios the change
touches) run by sim.perception.run_cabin: FaceMesh, the detector and MediaPipe
Hands are scripted stand-ins; calibration, every tracker, the phone-handling
gate, events and the engine are the real code at the shipped config.yaml
thresholds. SYNTHETIC REGRESSION EVIDENCE ONLY -- not real-world accuracy.

The production rule (docs/BEHAVIOR_ENGINE.md section 10): a sustained
eyes-off-road episode is ``phone_use`` only while the driver's phone is HANDLED
-- a hand on it (unchanged), or, with holding unobserved, a handling cue (it
moved within phone_static_s, it is at the face, or it sits in the lap zone under
a downward gaze). A mounted / unheld / passenger's phone leaves the look to
eyes_down / distraction, whose rules and timings are untouched.

On 87d91b3 the tests marked BEFORE-FAILS fail (phone_use fired); every other
pin here is also the 87d91b3 outcome (~/sensordyme-audit/2026-11-10/
m85-evidence/phone/scenarios_before_87d91b3.json).
"""

from __future__ import annotations

import pytest

from sim.perception import run_cabin
from sim.scenario import CABIN_SCENARIOS, M85_PHONE_SCENARIOS

VALIDATED = {"drowsiness", "distraction", "eyes_down", "eyes_up", "phone_use", "attention_lost",
             "fatigue_warning", "frequent_glancing"}
OFF_ROAD = {"eyes_down", "distraction", "eyes_up"}

#: The four M8.4 cabin scenarios with a production-only phone_use on 87d91b3, plus
#: the lap phone the rule must keep.
M84_TOUCHED = ("phone_mounted_look_before_latch", "phone_mounted_look_down", "passenger_leaning_with_phone",
               "hands_out_of_view_phone", "held_phone_eyes_off_road", "phone_mounted_unused",
               "phone_mounted_hands_unseen", "passenger_holding_phone", "detector_failure_look_down",
               "fluctuating_confidence")
SCENARIOS = {**{n: CABIN_SCENARIOS[n] for n in M84_TOUCHED}, **M85_PHONE_SCENARIOS}

#: Validated alerts after M8.5 (scenario time, s). Comment: the 87d91b3 outcome when it differs.
AFTER = {
    # ---- mounted / unheld: no phone_use; the look is still alerted by its own rule
    "phone_mounted_unused": [],
    "phone_mounted_hands_unseen": [],
    "m85_mount_unseen_short_glance": [],
    "phone_mounted_look_before_latch": ["eyes_down@2.7"],  # was phone_use@2.2, eyes_down@2.7
    "phone_mounted_look_down": ["eyes_down@7.7"],  # was phone_use@7.2, eyes_down@7.7
    "m85_mount_unseen_long_look_side": ["distraction@8.2"],  # was phone_use@7.2, distraction@8.2
    "m85_mount_flicker_long_look": ["eyes_down@7.7"],  # was eyes_down@7.7, phone_use@8.2
    "m85_sunglasses_mount_long_look": ["eyes_down@7.7"],  # was phone_use@7.2, eyes_down@7.7
    "m85_stall_mount_long_look": ["eyes_down@14.4"],  # was phone_use@12.9, eyes_down@14.4
    "m85_mount_detector_fails_mid_look": ["eyes_down@7.7"],
    "detector_failure_look_down": ["eyes_down@4.7"],
    "fluctuating_confidence": [],
    # ---- someone else's phone
    "passenger_holding_phone": [],
    "passenger_leaning_with_phone": ["distraction@8.2"],  # was phone_use@7.2, distraction@8.2
    # ---- real handling: phone_use at the unchanged time
    "held_phone_eyes_off_road": ["phone_use@4.2", "eyes_down@4.7"],
    "m85_held_phone_eyes_on_road": [],
    "m85_sunglasses_held_phone": ["phone_use@4.2", "eyes_down@4.7"],
    "m85_stall_held_phone": ["phone_use@9.0", "eyes_down@11.5"],
    "m85_mount_then_picked_up": ["phone_use@7.2", "eyes_down@7.7"],
    "m85_mount_look_then_pick_up": ["eyes_down@7.7", "phone_use@8.5"],
    "m85_held_then_mounted_mid_look": ["phone_use@4.2", "eyes_down@4.7"],
    # ---- holding unobserved, a handling cue: phone_use kept (same time as before)
    "hands_out_of_view_phone": ["phone_use@4.2", "eyes_down@4.7"],  # lap zone + gaze down
    "m85_mount_then_picked_up_hands_unseen": ["phone_use@7.2", "eyes_down@7.7"],  # moved, then lap zone
    "m85_moving_phone_hands_unseen": ["phone_use@5.2", "eyes_down@5.7"],  # moving
    "m85_phone_at_ear_look_away": ["phone_use@6.2", "distraction@7.2"],  # at the face
}

#: Scenarios where 87d91b3 fired phone_use and M8.5 must not (BEFORE-FAILS).
NO_LONGER_PHONE_USE = ("phone_mounted_look_before_latch", "phone_mounted_look_down",
                       "m85_mount_unseen_long_look_side", "m85_mount_flicker_long_look",
                       "m85_sunglasses_mount_long_look", "m85_stall_mount_long_look",
                       "passenger_leaning_with_phone")


def _alerts(trace, types=VALIDATED) -> list[str]:
    return [f"{e['type']}@{e['t']}" for e in trace["events"] if e["type"] in types]


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    return {name: run_cabin(scenario, str(tmp_path_factory.mktemp(name))) for name, scenario in SCENARIOS.items()}


def test_every_scenario_is_pinned():
    assert set(AFTER) == set(SCENARIOS)


@pytest.mark.parametrize("name", sorted(AFTER))
def test_validated_alerts(traces, name):
    assert _alerts(traces[name]) == AFTER[name]


# ============================================================ the owner-requested fix (BEFORE-FAILS)
@pytest.mark.parametrize("name", NO_LONGER_PHONE_USE)
def test_a_mounted_unheld_or_passenger_phone_is_never_the_drivers_phone_use(traces, name):
    assert "phone_use" not in [e["type"] for e in traces[name]["events"]]


@pytest.mark.parametrize("name", NO_LONGER_PHONE_USE)
def test_the_long_look_is_never_silenced(traces, name):
    """The episode that used to be phone_use is still alerted, by the off-road
    rule the situation warrants, inside the scripted look."""
    scenario = SCENARIOS[name]
    window = next(e for e in scenario.label.expected if e[0] in OFF_ROAD)
    fired = [e for e in traces[name]["events"] if e["type"] in OFF_ROAD]
    assert fired, name
    assert window[1] <= fired[0]["t"] <= window[2] + 0.5, (name, fired[0], window)


def test_mount_before_latch_is_eyes_down_not_phone_use(traces):
    """SUNDAY B3: the look starts before PhoneHoldEvidence's ~2 s set-down latch."""
    assert _alerts(traces["phone_mounted_look_before_latch"]) == ["eyes_down@2.7"]


def test_mount_with_hands_unseen_is_eyes_down_not_phone_use(traces):
    """Hold 'unknown' (MediaPipe Hands sees no hand at all): no handling cue."""
    assert _alerts(traces["phone_mounted_look_down"]) == ["eyes_down@7.7"]
    assert traces["phone_mounted_look_down"]["engine"]["phone_evidence"]["level"] == "phone_near_driver"


# ============================================================ real phone use: unchanged timing
@pytest.mark.parametrize(("name", "onset"), [
    ("held_phone_eyes_off_road", 4.2), ("m85_sunglasses_held_phone", 4.2), ("m85_stall_held_phone", 9.0),
    ("m85_mount_then_picked_up", 7.2), ("m85_mount_look_then_pick_up", 8.5),
])
def test_a_held_phone_with_the_eyes_off_the_road_is_phone_use_at_the_same_time(traces, name, onset):
    phone = [e for e in traces[name]["events"] if e["type"] == "phone_use"]
    assert [e["t"] for e in phone] == [onset]
    assert phone[0]["signals"] == "hand overlapping the phone (experimental hand tracking)"


def test_a_held_phone_with_the_eyes_on_the_road_is_not_phone_use(traces):
    assert _alerts(traces["m85_held_phone_eyes_on_road"]) == []


# ============================================================ holding unobserved: the cues
@pytest.mark.parametrize(("name", "cue"), [
    ("hands_out_of_view_phone", "phone in lap zone"),
    ("m85_mount_then_picked_up_hands_unseen", "phone in lap zone"),
    ("m85_moving_phone_hands_unseen", "phone moving"),
    ("m85_phone_at_ear_look_away", "phone at face"),
    ("m85_held_then_mounted_mid_look", "phone moving"),
])
def test_a_handling_cue_keeps_phone_use_and_says_it_is_unvalidated(traces, name, cue):
    phone = [e for e in traces[name]["events"] if e["type"] == "phone_use"]
    assert len(phone) == 1
    detail = phone[0]["signals"]
    assert detail.startswith("phone visible; no hand confirmed on it; possible handling: ")
    assert cue in detail and "unvalidated" in detail


def test_the_lap_phone_with_hands_unseen_decision(traces):
    """Phone in the lap, no hand found, looking down at it: phone_use at 2.0 s
    (as before) AND eyes_down at 2.5 s. Justification (docs/BEHAVIOR_ENGINE.md
    section 10): directly below the face is where a hand-held phone is read, not
    where a mount sits, and the hand model's recall is too low (8/152 web images)
    for "no hand" to clear it. The shadow engine still abstains (holding unknown)."""
    trace = traces["hands_out_of_view_phone"]
    assert _alerts(trace) == ["phone_use@4.2", "eyes_down@4.7"]
    assert trace["engine"]["comparison"]["phone_use"]["shadow"] == 0


def test_hand_tracking_off_mount_is_eyes_down_lap_is_phone_use(tmp_path):
    mount = run_cabin(CABIN_SCENARIOS["phone_mounted_look_before_latch"], str(tmp_path / "m"), enable_hands=False)
    assert _alerts(mount) == ["eyes_down@2.7"]
    lap = run_cabin(CABIN_SCENARIOS["hands_out_of_view_phone"], str(tmp_path / "l"), enable_hands=False)
    assert _alerts(lap) == ["phone_use@4.2", "eyes_down@4.7"]
    assert "hand tracking unavailable" in next(e["signals"] for e in lap["events"] if e["type"] == "phone_use")


def test_the_hands_policy_changes_no_phone_decision(traces, tmp_path_factory):
    for name in ("phone_mounted_look_before_latch", "m85_mount_then_picked_up", "m85_mount_look_then_pick_up",
                 "m85_mount_unseen_long_look_side"):
        every = run_cabin(SCENARIOS[name], str(tmp_path_factory.mktemp(f"every_{name}")), hands_policy="every_tick")
        assert _alerts(every) == _alerts(traces[name]), name


# ============================================================ nothing else moved
def test_experimental_observations_and_risk_unchanged_for_the_mounted_look(traces):
    """Decision B already made the mounted phone 0 risk; M8.5 leaves that as is."""
    for name in ("phone_mounted_look_down", "passenger_leaning_with_phone"):
        assert traces[name]["risk_peak"] == pytest.approx(0.3, abs=0.01)
        assert not [e for e in traces[name]["events"] if e["type"] in ("elevated_risk", "high_risk")]


def test_the_shadow_engine_never_calls_a_mounted_phone_phone_use(traces):
    for name in NO_LONGER_PHONE_USE:
        assert "phone_use" not in [e["type"] for e in traces[name]["shadow_events"]], name
