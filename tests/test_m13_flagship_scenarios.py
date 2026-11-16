"""M13 flagship distracted-driving scenarios: the 10 demonstrations, pinned. SIMULATED.

Scenarios 1-8 run the REAL DriverAttentionModule at the SHIPPED config.yaml
thresholds (sim.perception.run_cabin; scripted FaceMesh / detector / hands),
and read its overlay half through the real core/overlay.py OverlayBuilder.
Scenarios 9-10 run the virtual device with the real backend through a camera
freeze and its recovery. sim/flagship.py holds the scenarios and the checks;
scripts/demo/flagship_rehearsal.py prints the same runs as a table.

For each scenario the system must: alert only when the configured policy is
satisfied; distinguish the behaviour; never attribute a passenger's phone to
the driver; report insufficient evidence where it has it; keep the role's
mandatory safety policy; and say why an event occurred. A requirement the
product does not meet is pinned as a strict xfail naming its KNOWN_GAPS id,
with the measured behaviour -- never tuned away.

Constructed inputs only: not accuracy on real cabins, faces, lenses or a Pi.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from sim import flagship
from sim.flagship import CASES, KNOWN_GAPS, Problem
from sim.perception import driver_module_config

pytest.importorskip("mediapipe")

CORE_RATES = (10.0, 15.0, 30.0)


@pytest.fixture(scope="module")
def results():
    """Every module-harness scenario at the table's rate (10 fps)."""
    return {case.number: flagship.run_case(case) for case in CASES}


@pytest.fixture(scope="module")
def device():
    """Scenarios 9 and 10: one virtual-device run."""
    pytest.importorskip("fastapi")
    return flagship.run_device()


def _unexpected(problems: list[Problem]) -> list[str]:
    return [p.text for p in problems if p.gap is None]


def _gapped(problems: list[Problem], gap: str) -> list[str]:
    return [p.text for p in problems if p.gap == gap]


# ============================================================ the harness is the shipped product
def test_the_harness_runs_the_shipped_thresholds_and_role_floor(tmp_path):
    shipped = yaml.safe_load((flagship.REPO / "config.yaml").read_text())["modules"]["driver_attention"]
    used = driver_module_config(str(tmp_path))
    changed = {k for k in set(shipped) | set(used) if shipped.get(k) != used.get(k)}
    assert changed <= {"calibration_path", "recalibrate"}, changed  # file location + a fresh calibration only
    policy = flagship.shipped_alert_policy()
    assert policy["floor_violations"] == []
    assert policy["mandatory"] <= policy["loud"], "every mandatory driver alert is loud in the shipped config"


def test_every_scenario_is_numbered_once_and_labelled_simulated():
    assert [c.number for c in CASES] == list(range(1, 9))
    assert flagship.MODULE_HARNESS.startswith("SIMULATED") and flagship.VIRTUAL_DEVICE.startswith("SIMULATED")


# ============================================================ 1-8: every required behaviour, every rate
@pytest.mark.parametrize("fps", CORE_RATES)
@pytest.mark.parametrize("case", CASES, ids=[c.key for c in CASES])
def test_module_scenario_meets_every_requirement_except_its_recorded_gaps(case, fps, tmp_path):
    result = flagship.run_case(case, str(tmp_path), fps=fps)
    assert _unexpected(result.problems) == [], (case.key, fps, [e["type"] for e in result.alerts])


# -- 1. normal forward attention
def test_1_normal_attention_raises_nothing_and_measures_the_face_throughout(results):
    r = results[1]
    assert r.alerts == [] and r.problems == []
    assert all(f.face_visible and f.head_direction == "Forward" and f.behaviours == () for f in r.frames)


# -- 2. a short glance is a developing distraction that never confirms
def test_2_short_glance_develops_and_is_never_confirmed(results):
    r = results[2]
    assert r.alerts == [] and r.problems == []
    phases = {phase for f in r.frames for kind, phase, _ in f.behaviours if kind == "distraction"}
    assert phases == {"developing"}
    longest = max(for_s or 0.0 for f in r.frames for kind, _, for_s in f.behaviours if kind == "distraction")
    assert longest < 3.0, "distraction_seconds (3.0) was never reached"


# -- 3. prolonged attention away
def test_3_prolonged_look_away_confirms_and_alerts_once_at_the_shipped_time(results):
    r = results[3]
    assert [e["type"] for e in r.alerts] == ["distraction"]
    assert 3.0 <= r.latency("distraction") <= 3.0 + flagship.lag(flagship.FPS)
    confirmed = [f for f in r.frames if ("distraction", "confirmed") in {(k, p) for k, p, _ in f.behaviours}]
    assert confirmed and confirmed[0].state == "DISTRACTED" and confirmed[0].head_direction in ("Left", "Right")


# -- 4. mounted phone
def test_4_a_mounted_phone_never_reads_as_phone_use_developing(results):
    r = results[4]
    assert r.alerts == [] and r.problems == []
    assert all(kind != "phone_use" for f in r.frames for kind, _, _ in f.behaviours)
    assert {e["type"] for e in r.events} == {"phone_detected"}, "the phone is recorded, log-only"


# -- 5. phone held
def test_5_a_held_phone_is_phone_use_with_the_hand_as_its_reason(results):
    r = results[5]
    assert [e["type"] for e in r.alerts] == ["phone_use", "eyes_down"]
    phone = next(e for e in r.alerts if e["type"] == "phone_use")
    assert phone["signals"] == "hand overlapping the phone (experimental hand tracking)"
    assert 2.0 <= r.latency("phone_use") <= 2.0 + flagship.lag(flagship.FPS)
    assert {role for f in r.frames for role in f.phone_roles} == {"driver"}


# -- 6. passenger's phone
def test_6_the_passengers_phone_is_never_the_drivers(results):
    r = results[6]
    assert r.alerts == []
    assert all(f.driver_phones == 0 for f in r.frames), "never one of the driver's phone boxes"
    assert all(kind != "phone_use" for f in r.frames for kind, _, _ in f.behaviours)
    assert all(role != "driver" for f in r.frames for role in f.phone_roles)
    assert _unexpected(r.problems) == []


@pytest.mark.xfail(strict=True, reason="G1: the overlay sends a passenger's phone with NO role, not 'other' "
                                       "(core/overlay.py only assigns phone roles when the driver has a phone)")
def test_6_known_gap_the_passengers_phone_is_marked_other_on_the_overlay(results):
    assert {role for f in results[6].frames for role in f.phone_roles} == {"other"}


# -- 7. eyes closed
def test_7_eyes_closed_is_drowsiness_loud_and_critical_for_the_role(results):
    r = results[7]
    drowsy = [e for e in r.alerts if e["type"] == "drowsiness"]
    assert len(drowsy) == 1 and 2.0 <= r.latency("drowsiness") <= 2.0 + flagship.lag(flagship.FPS)
    assert {e["type"] for e in r.alerts} <= {"drowsiness", "fatigue_warning"}
    policy = flagship.shipped_alert_policy()
    assert "drowsiness" in policy["critical"] and flagship.severity("drowsiness", policy) == "loud"


# -- 8. brief occlusion
def test_8_brief_occlusion_reports_no_face_abstains_and_alerts_nothing(results):
    r = results[8]
    assert r.alerts == [] and r.problems == []
    hidden = [f for f in r.frames if 2.15 <= f.t < 5.0]
    assert hidden and all(f.face_visible is False and f.head_direction is None for f in hidden)
    assert all({"distraction", "drowsiness", "eyes_down"} <= set(f.abstaining.split(";")) for f in hidden)
    assert all(f.driver_identity == "identified" for f in hidden), "the body still places the driver"


# -- why: the event's own contributing_signals
@pytest.mark.xfail(strict=True, reason="G2: distraction / drowsiness / eyes_down events carry an empty "
                                       "contributing_signals; the reason is only their type and duration")
def test_known_gap_every_flagship_alert_says_why_in_its_own_signals(results):
    fired = [e for n in (3, 5, 7) for e in results[n].alerts if e["type"] != "fatigue_warning"]
    assert fired and all(str(e["signals"]).strip() for e in fired), [(e["type"], e["signals"]) for e in fired]


def test_the_gap_records_match_what_was_measured(results):
    """The table's 'known gap' tags are exactly the measured gaps, nothing hidden behind them."""
    assert _gapped(results[3].problems, "G2") == ["distraction carries no contributing_signals"]
    assert _gapped(results[5].problems, "G2") == ["eyes_down carries no contributing_signals"]
    assert _gapped(results[7].problems, "G2") == ["drowsiness carries no contributing_signals"]
    assert _gapped(results[6].problems, "G1") == ["phone role(s) [None], expected 'other'"]
    assert set(KNOWN_GAPS) == {"G1", "G2"}


def test_the_checks_catch_a_regression(results):
    """The judge is not a rubber stamp: a passenger phone attributed to the driver, an
    alert in the glance scenario and a missing occlusion report each FAIL unexpectedly."""
    six = copy.deepcopy(results[6])
    six.frames[-1].driver_phones = 1
    six.frames[-1].phone_roles = ("driver",)
    assert {"the passenger's phone was counted as the driver's phone"} <= set(_unexpected(flagship.check_case(six)))
    two = copy.deepcopy(results[2])
    two.events.append({"t": 5.0, "type": "distraction", "signals": "x"})
    assert any("unexpected alert" in p for p in _unexpected(flagship.check_case(two)))
    eight = copy.deepcopy(results[8])
    for f in eight.frames:
        f.face_visible = True
    assert _unexpected(flagship.check_case(eight))


# ============================================================ 9-10: camera unavailable, restored
def test_9_camera_unavailable_is_failed_not_protecting_and_named(device):
    assert device.problems9 == []
    detected = device.first(lambda s: s.device_state == "failed", flagship.FREEZE_AT)
    assert detected is not None and detected - flagship.FREEZE_AT <= flagship.DETECT_BOUND_S
    outage = [s for s in device.samples if flagship.FREEZE_AT + flagship.APP_BOUND_S <= s.t < flagship.RESTORE_AT]
    assert all(s.app_state == "failed" and s.app_protection == "not_protecting" for s in outage)
    assert all(any("camera" in reason for reason in s.reasons) for s in outage)


def test_9_no_false_protecting_at_any_sample(device):
    assert not [s.t for s in device.samples if s.app_protection == "protecting" and s.app_state != "ready"]


def test_10_monitoring_restored_and_the_buzzer_path_resumes(device):
    assert _unexpected(device.problems10) == []
    drowsy = [e for e in device.events if e["type"] == "drowsiness"]
    assert len(drowsy) == 1 and drowsy[0]["severity"] == "loud" and drowsy[0]["simulated"] is True
    assert drowsy[0]["alert"]["decision"] == "alerted"
    assert any(b[0] == "drowsiness" for b in device.buzzer)
    local = [e for e in device.local_events if e["type"] == "drowsiness"]
    assert local and local[0]["event_key"] == drowsy[0]["event_key"]
    latency = flagship.drowsy_latency(device)
    assert latency is not None and 2.0 <= latency <= 3.0


def test_the_rows_print_simulated(results, device):
    rows = [flagship.describe_case(results[n]) for n in range(1, 9)] + flagship.describe_device(device)
    assert [r["number"] for r in rows] == list(range(1, 11))
    assert all(r["harness"].startswith("SIMULATED") for r in rows)
    assert {r["verdict"] for r in rows} <= {"PASS", "FAIL (known gap G1)", "FAIL (known gap G2)"}
