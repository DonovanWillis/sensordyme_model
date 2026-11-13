"""M8.4 behavioural regression scenarios through the REAL DriverAttentionModule.

sim.scenario.CABIN_SCENARIOS run by sim.perception.run_cabin: FaceMesh, the
detector and MediaPipe Hands are scripted stand-ins; calibration, trackers,
risk fusion, events and the M8.4 engine are the real code at the shipped
config.yaml thresholds. SYNTHETIC REGRESSION EVIDENCE ONLY -- not real-world
accuracy.

What is pinned:
  * every VALIDATED alert fires exactly as on c9029fa (same types, same times),
    in every scenario, under both hands policies (owner rule: no hidden change);
  * decision B: a visible, mounted, passenger's or unheld phone adds nothing to
    the risk score, and the experimental composite events it used to trigger
    no longer appear; a held phone with the eyes off the road still counts;
  * the engine abstains -- and says why -- on sunglasses, a dark cabin, a lost
    face, a failed detector, an unheld phone, a moved camera, an empty vehicle;
  * its shadow decisions match production on clean evidence (same onsets);
  * B5: Hands runs far less, and never on a tick that could change phone_use.
"""

from __future__ import annotations

import pytest

from core.behavior.experimental import EXPERIMENTAL_ALERT_TYPES, EXPERIMENTAL_DETAIL_PREFIX
from sim.perception import run_cabin
from sim.scenario import CABIN_SCENARIOS

VALIDATED = {"drowsiness", "distraction", "eyes_down", "eyes_up", "phone_use", "attention_lost",
             "fatigue_warning", "frequent_glancing"}

#: Validated alerts per scenario on c9029fa (scripts/eval_behavior.py run on that
#: commit; ~/sensordyme-audit/2026-11-10/m84-evidence/alpha/eval_before_c9029fa.json).
BASELINE_ALERTS = {
    "camera_blocked": ["attention_lost@4.5"],
    "camera_shifted": ["distraction@3.2"],
    "dashboard_glances": [],
    "detector_failure_look_down": ["eyes_down@4.7"],
    "empty_vehicle": ["attention_lost@4.5"],
    "eyes_closed_3s": ["drowsiness@4.2", "fatigue_warning@4.8"],
    "fluctuating_confidence": [],
    "hands_out_of_view_phone": ["phone_use@4.2", "eyes_down@4.7"],
    "held_phone_eyes_off_road": ["phone_use@4.2", "eyes_down@4.7"],
    "intersection_look": [],
    "mirror_checks": [],
    "partial_face_occlusion": [],
    "passenger_holding_phone": [],
    "passenger_leaning_with_phone": ["phone_use@7.2", "distraction@8.2"],
    "phone_mounted_hands_unseen": [],
    "phone_mounted_long_look_latched": ["eyes_down@12.7"],
    "phone_mounted_look_before_latch": ["phone_use@2.2", "eyes_down@2.7"],
    "phone_mounted_look_down": ["phone_use@7.2", "eyes_down@7.7"],
    "phone_mounted_unused": [],
    "poor_lighting": [],
    "stream_stall": [],
    "sunglasses": [],
    "sustained_look_away": ["distraction@5.2"],
    "sustained_look_down": ["eyes_down@4.7"],
}


def _alerts(trace) -> list[str]:
    return [f"{e['type']}@{e['t']}" for e in trace["events"] if e["type"] in VALIDATED]


@pytest.fixture(scope="module")
def traces(tmp_path_factory):
    return {name: run_cabin(scenario, str(tmp_path_factory.mktemp(name)))
            for name, scenario in CABIN_SCENARIOS.items()}


def test_every_scenario_is_pinned():
    assert set(BASELINE_ALERTS) == set(CABIN_SCENARIOS)


# ============================================================ validated alerts unchanged
@pytest.mark.parametrize("name", sorted(BASELINE_ALERTS))
def test_validated_alerts_fire_exactly_as_before(traces, name):
    assert _alerts(traces[name]) == BASELINE_ALERTS[name]


def test_true_scenarios_still_alert(traces):
    assert "drowsiness@4.2" in _alerts(traces["eyes_closed_3s"])  # eyes closed 3 s
    assert "distraction@5.2" in _alerts(traces["sustained_look_away"])  # sustained look-away
    held = _alerts(traces["held_phone_eyes_off_road"])  # held phone + eyes off road
    assert "phone_use@4.2" in held and "eyes_down@4.7" in held


# ============================================================ decision A in the module
def test_experimental_events_are_labelled_observations(traces):
    seen = [e for t in traces.values() for e in t["events"] if e["type"] in EXPERIMENTAL_ALERT_TYPES]
    assert seen, "the research signals are still computed and logged"
    assert all(e["signals"].startswith(EXPERIMENTAL_DETAIL_PREFIX) for e in seen)


def test_the_module_no_longer_flashes_for_experimental_conditions(tmp_path):
    from modules.driver_attention import DriverAttentionModule
    from sim.perception import driver_module_config

    module = DriverAttentionModule(driver_module_config(str(tmp_path)))
    try:
        module._erratic_hands_active = True
        assert module.is_alert_active() is False
        assert "erratic_hand_movement" in module.active_alert_conditions()  # still reported as a condition
    finally:
        module.close()


# ============================================================ decision B: a visible phone is not risk
@pytest.mark.parametrize("name", ["phone_mounted_unused", "phone_mounted_hands_unseen", "passenger_holding_phone"])
def test_a_phone_with_the_eyes_on_the_road_adds_no_risk(traces, name):
    # c9029fa: 0.066 for both mounted cases (phone_risk 0.3 x weight 0.22).
    assert traces[name]["risk_peak"] <= 0.03 + 1e-9


@pytest.mark.parametrize("name", ["hands_out_of_view_phone", "phone_mounted_look_down", "passenger_leaning_with_phone"])
def test_an_unheld_or_passenger_phone_no_longer_lifts_a_glance_into_elevated_risk(traces, name):
    # c9029fa: risk peak 0.52 and an elevated_risk event in each (phone_risk 1.0
    # while phone_use was active on a visible phone); the glance alone is 0.30.
    assert traces[name]["risk_peak"] == pytest.approx(0.3, abs=0.01)
    assert not [e for e in traces[name]["events"] if e["type"] in ("elevated_risk", "high_risk")]


def test_a_held_phone_with_the_eyes_off_the_road_still_counts(traces):
    trace = traces["held_phone_eyes_off_road"]
    assert trace["risk_peak"] == pytest.approx(0.52, abs=0.01)
    assert trace["engine"]["phone_evidence"]["level"] in ("phone_held", "possible_interaction",
                                                           "active_distraction_evidence")


def test_phone_evidence_levels_in_the_cabin(traces):
    level = {n: t["engine"]["phone_evidence"]["level"] for n, t in traces.items()}
    assert level["phone_mounted_unused"] == "phone_mounted"
    assert level["phone_mounted_hands_unseen"] == "phone_near_driver"
    assert level["passenger_holding_phone"] == "phone_visible"
    assert level["passenger_leaning_with_phone"] == "phone_visible"
    assert level["detector_failure_look_down"] == "insufficient_evidence"


# ============================================================ abstention, reported
def _abstain_rate(trace, claim) -> float:
    return trace["engine"]["comparison"][claim]["abstain_rate"]


def test_sunglasses_and_a_dark_cabin_abstain_on_drowsiness(traces):
    for name in ("sunglasses", "poor_lighting"):
        assert _abstain_rate(traces[name], "drowsiness") > 0.8
        assert "eyes occluded" in traces[name]["engine"]["abstaining"]["drowsiness"]


def test_unheld_phone_while_looking_down_is_an_abstention_not_phone_use(traces):
    for name in ("hands_out_of_view_phone", "phone_mounted_look_down"):
        comparison = traces[name]["engine"]["comparison"]["phone_use"]
        assert comparison["shadow"] == 0 and comparison["production_only"] == 1
        assert "holding unknown" in " ".join(comparison["abstain_reasons"])


def test_mounted_phone_is_never_shadow_phone_use_and_the_look_is_still_eyes_down(traces):
    """SUNDAY B1/B3/B4: a long look at a MOUNTED phone, hands visible. Production
    fires phone_use when the look starts before the set-down latch (validated
    behaviour, unchanged); the engine never calls it phone use; eyes_down fires
    in both by its own rule."""
    for name in ("phone_mounted_look_before_latch", "phone_mounted_long_look_latched"):
        shadow = [e["type"] for e in traces[name]["shadow_events"]]
        assert "phone_use" not in shadow and "eyes_down" in shadow
        assert traces[name]["engine"]["phone_evidence"]["level"] == "phone_mounted"


def test_a_failed_detector_makes_phone_evidence_insufficient(traces):
    assert _abstain_rate(traces["detector_failure_look_down"], "phone_use") > 0.9


def test_a_moved_camera_and_an_empty_vehicle_are_reported(traces):
    assert "camera may have moved" in traces["camera_shifted"]["engine"]["abstaining"]["distraction"]
    assert "vehicle may be empty" in traces["empty_vehicle"]["engine"]["abstaining"]["attention_lost"]
    blocked = traces["camera_blocked"]["engine"]["comparison"]["attention_lost"]
    assert blocked["matched"] == 1


def test_noise_dropouts_and_stalls_raise_no_alert_and_no_shadow_event(traces):
    for name in ("fluctuating_confidence", "partial_face_occlusion", "stream_stall", "mirror_checks",
                 "dashboard_glances", "intersection_look", "sunglasses", "poor_lighting"):
        assert _alerts(traces[name]) == [] and traces[name]["shadow_events"] == [], name


# ============================================================ shadow vs production
def test_shadow_matches_production_on_clean_evidence(traces):
    for name in ("eyes_closed_3s", "sustained_look_away", "sustained_look_down", "held_phone_eyes_off_road"):
        for claim, comp in traces[name]["engine"]["comparison"].items():
            assert comp["production_only"] == 0 and comp["shadow_only"] == 0, (name, claim)
            if comp["matched"]:
                assert comp["mean_shadow_minus_production_s"] == 0.0


def test_driver_identity_is_stated(traces):
    assert traces["sustained_look_away"]["engine"]["driver"]["driver_identity"] == "face_anchored_person"
    leaning = traces["passenger_leaning_with_phone"]["engine"]["driver"]
    assert leaning["agrees_with_largest_person"] is False and leaning["accuracy"] == "unevaluated"


# ============================================================ B5: conditional hands
def test_hands_run_far_less_and_change_no_validated_alert(traces, tmp_path_factory):
    every = {}
    for name in ("sustained_look_away", "held_phone_eyes_off_road", "phone_mounted_unused",
                 "phone_mounted_look_before_latch", "passenger_leaning_with_phone"):
        every[name] = run_cabin(CABIN_SCENARIOS[name], str(tmp_path_factory.mktemp(f"every_{name}")),
                                hands_policy="every_tick")
        assert _alerts(every[name]) == _alerts(traces[name]), name
        assert traces[name]["hands_calls"] <= every[name]["hands_calls"]
    assert every["sustained_look_away"]["hands_calls"] == every["sustained_look_away"]["detector_calls"]
    # No phone: Hands runs only as a liveness sample, at least every 0.5 s of
    # evidence time (under the perception engine's 1.0 s freshness bar). At this
    # harness's 10 fps the detector ticks every 0.3 s, so about every other tick.
    assert traces["sustained_look_away"]["hands_calls"] * 1.8 < every["sustained_look_away"]["hands_calls"]
    # A phone in view: every tick, exactly as before (phone_use reads hands there).
    assert traces["phone_mounted_unused"]["hands_calls"] >= 0.8 * every["phone_mounted_unused"]["hands_calls"]


def test_at_the_device_frame_rate_hands_runs_about_a_fifth_as_often(tmp_path):
    """30 fps (the device's measured rate) with the shipped stride of 3: detector
    ticks every 0.1 s, Hands every 0.5 s without a phone."""
    scenario = CABIN_SCENARIOS["sustained_look_away"]
    gated = run_cabin(scenario, str(tmp_path / "g"), fps=30.0)
    every = run_cabin(scenario, str(tmp_path / "e"), fps=30.0, hands_policy="every_tick")
    assert _alerts(gated) == _alerts(every)
    assert gated["hands_calls"] * 4 < every["hands_calls"]
