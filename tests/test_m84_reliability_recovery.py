"""M8.4 fault recovery on the digital twin (D2): what M8.3 and earlier did not
already prove (docs/DIGITAL_TWIN.md maps every D2 item to its test).

  * repeated inference failure: the module raises on every frame -> FAILED,
    never protecting, the loop restarted like the supervisor does, and READY
    again once frames analyse;
  * no false protecting state through a seeded storm of overlapping faults,
    checked against the fault actually in force at every monitoring check;
  * no duplicate physical alert after a reconnect: the buzzer actuates an
    episode the same number of times with or without a network flap;
  * a restart shortly after a confirmed resume (a state-model edge found here).

Deterministic: one SimClock, seeded schedules, the truth InvariantMonitor attached.
"""

from __future__ import annotations

import random

import pytest

from core import state_model
from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import PROTECTING_STATES, InvariantMonitor
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, *rest) for t, scene, *rest in segments)))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


# ---------------------------------------------------------------------- repeated inference failure
def test_a_module_that_fails_on_every_frame_is_failed_never_protecting_and_recovers(stack, monkeypatch):
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"), supervise_inference=True)
    stack.claim()
    device.advance(C + 2)
    assert stack.status()["monitoring_state"] == "ready"
    module = device.runtime.module
    real = module.process
    broken = {"on": True}

    def process(frame, *args, **kwargs):
        if broken["on"]:
            raise RuntimeError("SIMULATED: the model graph returns garbage on every frame")
        return real(frame, *args, **kwargs)

    monkeypatch.setattr(module, "process", process)
    seen: set[str] = set()
    for _ in range(30):
        device.advance(2.0)
        seen.add(device.runtime.protection_state()["monitoring_state"])
    status = stack.status()
    assert status["monitoring_state"] == "failed" and status["protection"] == "not_protecting", status
    impaired = {i["name"]: i.get("reason") or i.get("detail") or str(i) for i in status["capabilities"]["impaired"]}
    assert "inference" in impaired and "analysis failing" in impaired["inference"], impaired
    assert not set(status["capabilities"]["available"]) & {"face", "objects", "attention"}
    assert device.inference_restarts >= 5, "past its failure budget the loop is handed to the supervisor"
    assert seen <= {"failed"}, f"while every frame failed the device also reported {seen - {'failed'}}"

    broken["on"] = False
    device.advance(30.0)
    assert stack.status()["monitoring_state"] == "ready", stack.status()
    monitor.assert_clean()


# ---------------------------------------------------------------------- no false protecting state
def _storm(seed: int, minutes: int) -> Scenario:
    """Overlapping faults on a seeded schedule; scenes change underneath."""
    rng = random.Random(seed)
    kinds = [Faults(camera="frozen"), Faults(camera="stalled"), Faults(camera="dropping"),
             Faults(detector_failing=True), Faults(network_down=True), Faults(soc_temp_c=84.0),
             Faults(camera="frozen", network_down=True), Faults(detector_failing=True, network_down=True)]
    segments = [Segment(0.0)]
    t = C + 5
    while t < minutes * 60:
        faults = rng.choice(kinds) if rng.random() < 0.6 else Faults()
        scene = ASLEEP if rng.random() < 0.15 else ATTENTIVE
        segments.append(Segment(t, scene, faults))
        t += rng.uniform(4.0, 40.0)
    segments.append(Segment(t, ATTENTIVE, Faults()))
    return Scenario(f"storm{seed}", "seeded fault storm", tuple(segments))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_false_protecting_state_through_a_fault_storm(stack, seed):
    """At every monitoring check, against the fault in force at that moment:
    a camera delivering nothing for longer than its detection window is never
    protecting, and a failing detector is never reported READY with objects."""
    monitor = InvariantMonitor(stack)
    scenario = _storm(seed, minutes=10)
    device = stack.boot(scenario)
    stack.claim()
    runtime = device.runtime
    origin = device.scenario_clock.origin
    violations: list[str] = []
    check = runtime._monitoring_check

    def watched():
        result = check()
        t = stack.clock.monotonic() - origin
        segment = scenario.at(t)
        held = t - segment.start
        prot = runtime.protection_state()
        if segment.faults.camera == "frozen" and held >= 6.0 and prot["state"] == "protecting":
            violations.append(f"t={t:.1f}: protecting {held:.1f}s into a frozen camera")
        if segment.faults.detector_failing and held >= 6.0 and prot["monitoring_state"] == "ready":
            violations.append(f"t={t:.1f}: READY {held:.1f}s into a detector failure")
        return result

    runtime._monitoring_check = watched
    device.advance(10 * 60 + 60)
    assert violations == [], violations[:5]
    assert stack.status()["monitoring_state"] == "ready"
    assert {"faulted", "degraded", "ready"} <= set(monitor.user_states), monitor.user_states
    monitor.assert_clean()


# ---------------------------------------------------------------------- no duplicate physical alert
def _episode_actuations(tmp_path, *, flap: bool) -> tuple[list, list]:
    down = Faults(network_down=True)
    segments = [(C + 3, ASLEEP, down if flap else Faults()), (C + 9, ATTENTIVE, down if flap else Faults()),
                (C + 20, ATTENTIVE, Faults())]
    with SimStack(tmp_path) as stack:
        monitor = InvariantMonitor(stack)
        device = stack.boot(_scenario("episode", *segments))
        stack.claim()
        device.advance(C + 40)
        monitor.assert_clean()
        return [p for p in device.buzzer.played if p[0] == "drowsiness"], \
            [e for e in stack.events() if e["type"] == "drowsiness"]


def test_a_reconnect_never_re_sounds_an_alert_or_stores_it_twice(tmp_path):
    """The eyes close while the link is down and it comes back mid-episode. The
    buzzer is a local decision: it plays exactly what the same episode plays with
    the link up -- the reconnect and the backlog flush add no actuation -- and the
    app gets one alert."""
    steady_played, steady_rows = _episode_actuations(tmp_path / "steady", flap=False)
    flap_played, flap_rows = _episode_actuations(tmp_path / "flap", flap=True)
    assert steady_played, "the episode never reached the buzzer"
    assert flap_played == steady_played, (flap_played, steady_played)
    assert len(flap_rows) == len(steady_rows) == 1


# ---------------------------------------------------------------------- state-model edge found by M8.4
# M8.4 finding, fixed by the integrator: the resume is tied to the boot it took effect on
# (protection_command.applied_boot), so a restart inside RESUME_SETTLE_S reads as starting.
def test_a_restart_shortly_after_a_resume_reads_as_starting_not_resuming(stack):
    monitor = InvariantMonitor(stack)
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C + 2)
    assert stack.protection("stop", confirm=True).status_code == 200
    device.advance(6.0)
    assert stack.protection("resume").status_code == 200
    device.advance(12.0)
    assert state_model.derive(stack.row(), stack.clock.wall()).state == "ready"
    origin = device.scenario_clock.origin
    rebooted = stack.restart(sc.get("attentive"))
    rebooted.scenario_clock.origin = origin
    rebooted.advance(2.0)
    assert state_model.derive(stack.row(), stack.clock.wall()).state != state_model.RESUMING
    monitor.assert_clean()


def test_the_monitor_s_protecting_set_matches_the_state_model():
    assert frozenset({"ready", "degraded"}) == PROTECTING_STATES == state_model.WATCHING
