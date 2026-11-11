"""Deep reliability: multi-stage fault scenarios on the virtual device, invariants at every step.

Each test drives the real runtime + real backend (sim/, one SimClock) through a
sequence of faults and recoveries, with sim.invariants.InvariantMonitor checking
after every monitoring check, command application and uplink cycle that:

  a failed device never reports protecting; a stopped device never reports ready;
  a stale device is never healthy; a refused rule set never runs; one command id
  has one effect and one alert one stored row; the driver_safety floor is never
  replaced by custom rules; the uplink backlog stays bounded.

End-state assertions below are scenario-specific; the monitor covers the states
in between.
"""

from __future__ import annotations

import pytest

from sim import scenario as sc
from sim.harness import SimStack
from sim.invariants import InvariantMonitor
from sim.perception import Scene
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
DOWN = Faults(network_down=True)
UP = Faults()
PHONE_VISIBLE = Scene(phone=True, label="phone on the dash")


def _timeline(name: str, *segments: tuple, first: Faults = UP) -> Scenario:
    return Scenario(name, name, (Segment(0.0, ATTENTIVE, first),
                                 *(Segment(float(t), scene, faults) for t, scene, faults in segments)))


def _compose(name: str, scenes: list[tuple[float, float, Scene]], faults: list[tuple[float, float, Faults]],
             ) -> Scenario:
    """Scene windows and fault windows, independent of each other (start, end, what).

    A fault window never hides a scene window that overlaps it, which hand-written
    segments with a fault and a scene each do (the later segment wins outright).
    """
    def at(windows, t, default):
        return next((w for s, e, w in windows if s <= t < e), default)

    cuts = sorted({0.0, *(x for s, e, _ in scenes + faults for x in (s, e))})
    return _timeline(name, *((t, at(scenes, t, ATTENTIVE), at(faults, t, UP)) for t in cuts if t > 0))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


@pytest.fixture
def monitor(stack):
    return InvariantMonitor(stack)


class _CommandTap:
    """A transport that can rewrite or replay the command in ingest responses.

    `inject` adds command keys to the next response (as an older or misbehaving
    backend, or a replayed response, would deliver them); `drop_next_requests`
    loses the next N requests before they reach the backend.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.inject: list[dict] = []
        self.drop_next_requests = 0
        self.commands: list[dict] = []

    def post(self, url, **kwargs):
        if self.drop_next_requests > 0:
            self.drop_next_requests -= 1
            import requests

            raise requests.ConnectionError("SIMULATED: request lost on the way to the backend")
        response = self._inner.post(url, **kwargs)
        body = response.json()
        command = dict(body.get("command") or {})
        if self.inject:
            command.update(self.inject.pop(0))
        if command:
            self.commands.append(command)
        return _Rewritten(response, {**body, "command": command or None})


class _Rewritten:
    def __init__(self, response, body) -> None:
        self._response, self._body = response, body
        self.status_code, self.headers, self.text = response.status_code, response.headers, response.text

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


# ---------------------------------------------------------------------- start offline
def test_a_device_that_boots_with_no_backend_protects_locally_then_reconnects(stack, monitor):
    script = _timeline("boot_offline", (C + 5, ASLEEP, DOWN), (C + 10, ATTENTIVE, DOWN),
                       (C + 30, ATTENTIVE, Faults()), first=DOWN)
    stack.claim()  # paired in the app; the device has never reported
    device = stack.boot(script)
    device.advance(C + 4)
    status = stack.status()
    assert status["stale"] is True and status["monitoring_state"] == "unknown"
    assert device.runtime.protection_state()["state"] == "protecting", "offline is not unprotected"

    device.advance(10.0)  # asleep at C+5, offline: the buzzer is local
    assert any(p[0] == "drowsiness" for p in device.buzzer.played)
    assert stack.events() == []

    device.advance(20.0)  # the backend is reachable from C+30
    status = stack.status()
    assert status["stale"] is False and status["monitoring_state"] == "ready"
    assert stack.event_types().count("drowsiness") == 1, "the queued alert arrives, once"
    monitor.assert_clean()
    assert monitor.backend_checks > 5


# ---------------------------------------------------------------------- commands
def test_a_replayed_stop_after_a_local_resume_has_no_effect(stack, monitor):
    tap = _CommandTap(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=tap)
    stack.claim()
    device.advance(C)

    stop_id = stack.protection("stop", confirm=True).json()["command_id"]
    device.advance(1.0)  # delivered and applied
    assert device.runtime._stopped is not None
    # The report that would acknowledge the stop is lost, so the backend re-sends it.
    tap.drop_next_requests = 2
    device.runtime._apply_protection({"action": "resume"}, source="dashboard")  # resumed locally
    device.advance(8.0)
    resent = [c["set_protection"]["id"] for c in tap.commands if "set_protection" in c]
    assert resent.count(stop_id) >= 1
    assert device.runtime._stopped is None, "the re-delivered stop must not stop monitoring again"
    assert stack.status()["protection"] == "protecting"
    assert stack.row()["health"]["protection"]["last_command_id"] == stop_id
    monitor.assert_clean()
    assert monitor.protection_effects[stop_id] == 1


def test_stop_then_resume_before_delivery_converges_on_the_last_request(stack, monitor):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    stack.protection("stop", confirm=True)
    resume_id = stack.protection("resume").json()["command_id"]  # overwrites the undelivered stop
    device.advance(6.0)
    row = stack.row()
    assert row["status"]["protection"] == "protecting" and row["status"]["monitoring_state"] == "ready"
    assert row["health"]["protection"]["last_command_id"] == resume_id
    assert monitor.transitions.count(("stopped", "stopped")) == 0, "an overwritten stop never took effect"

    resume_id = stack.protection("resume").json()["command_id"]
    stop_id = stack.protection("stop", confirm=True).json()["command_id"]
    device.advance(6.0)
    status = stack.status()
    assert status["protection"] == "stopped" and status["monitoring_state"] == "stopped"
    assert stack.row()["health"]["protection"]["last_command_id"] == stop_id
    assert resume_id != stop_id
    monitor.assert_clean()


def test_lost_responses_and_lost_requests_still_apply_each_command_once(stack, monitor):
    tap = _CommandTap(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=tap)
    stack.claim()
    device.advance(C)
    ids = []
    for action in ("stop", "resume", "stop", "resume"):
        ids.append(stack.protection(action, confirm=True).json()["command_id"])
        tap.drop_next_requests = 1  # the next poll never arrives: the command waits a cycle
        device.advance(6.0)
    status = stack.status()
    assert status["protection"] == "protecting" and status["monitoring_state"] == "ready"
    assert [monitor.protection_effects[i] for i in ids] == [1, 1, 1, 1]
    assert stack.row()["health"]["protection"]["last_command_id"] == ids[-1]
    monitor.assert_clean()


def test_malformed_commands_are_ignored_and_change_nothing(stack, monitor):
    tap = _CommandTap(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=tap)
    stack.claim()
    device.advance(C)
    preset = device.runtime.active_preset
    tap.inject = [
        {"set_preset": "x" * 500},                        # oversized name
        {"set_preset": 42},                                # wrong type
        {"set_protection": {"action": "explode", "id": "m1"}},
        {"set_protection": {"action": "stop", "id": "m2"}},   # unconfirmed stop
        {"set_protection": "stop"},
    ]
    device.advance(12.0)
    assert device.runtime.active_preset == preset
    assert device.runtime._stopped is None
    status = stack.status()
    assert status["protection"] == "protecting" and status["monitoring_state"] == "ready"
    assert status["last_rejection"]["reasons"] == ["stopping monitoring requires confirm: true"]
    monitor.assert_clean()


def test_a_role_incompatible_preset_delivered_anyway_is_refused_by_the_device(stack, monitor):
    """An older backend (no role check) sends a preset the driver role forbids."""
    tap = _CommandTap(stack.transport)
    device = stack.boot(sc.get("attentive"), transport=tap)
    stack.claim()
    device.advance(C)
    tap.inject = [{"set_preset": "security_monitoring"}]
    device.advance(6.0)
    assert device.runtime.module_name == "driver_attention"
    status = stack.status()
    assert status["protection"] == "protecting" and status["monitoring_state"] == "ready"
    assert status["last_rejection"]["kind"] == "preset", status["last_rejection"]
    monitor.assert_clean()


def test_an_invalid_policy_is_refused_whole_and_the_running_rules_are_kept(stack, monitor):
    script = _timeline("rules_then_phone", (C + 30, PHONE_VISIBLE, Faults()), (C + 36, ATTENTIVE, Faults()))
    tap = _CommandTap(stack.transport)
    device = stack.boot(script, transport=tap)
    stack.claim()
    device.advance(C)
    two = [
        {"target": "cell phone", "predicate": "present", "severity": "warning", "persist_s": 1.0,
         "notify": "push", "description": "phone visible"},
        {"target": "person", "predicate": "absent", "severity": "warning", "persist_s": 3.0,
         "notify": "push", "description": "nobody in the seat"},
    ]
    preview = stack.ai_configure("tell me when a phone is visible")
    assert stack.set_preset("", draft_version=preview.json()["draft_version"], rules=two).status_code == 200
    device.advance(6.0)
    running = [r.detection for r in device.runtime.rule_engine.rules]
    assert sorted(running) == ["cell phone", "person"], running
    assert stack.status()["mode"] == "standard_plus_custom"

    for bad in ([{"target": "cell phone", "predicate": "bogus"}],
                [{"target": "dog", "predicate": "present"}, two[0]],
                [{"target": "cell phone"}, "not a rule"]):
        tap.inject = [{"set_rules": bad, "rules_version": f"bad-{len(tap.commands)}"}]
        device.advance(4.0)
        assert sorted(r.detection for r in device.runtime.rule_engine.rules) == ["cell phone", "person"], bad
        assert stack.status()["last_rejection"]["kind"] == "rules"
        assert device.runtime.module_name == "driver_attention"

    device.advance(20.0)  # C+38: the phone was on the dash from C+30, the kept rule fired
    custom = [e["type"] for e in stack.events() if e["source"] == "custom_rule"]
    assert custom == ["cell phone"], "only the phone rule's condition was ever true: " + repr(custom)
    monitor.assert_clean()


# ---------------------------------------------------------------------- network flapping
def _flapping(name: str, down_s: float, up_s: float, until: float, asleep: tuple[float, ...]) -> Scenario:
    """Network down `down_s` / up `up_s` from C to `until`; eyes closed 4 s at each `asleep` time."""
    def network_down(t: float) -> bool:
        return C <= t < until and (t - C) % (down_s + up_s) < down_s

    def scene(t: float) -> Scene:
        return ASLEEP if any(a <= t < a + 4 for a in asleep) else ATTENTIVE

    cuts, t = {C, until, *asleep, *(a + 4 for a in asleep)}, C
    while t < until:
        cuts |= {t, t + down_s}
        t += down_s + up_s
    return _timeline(name, *((c, scene(c), DOWN if network_down(c) else Faults())
                             for c in sorted(x for x in cuts if x <= until)))


# Not (1.0, 1.0): a 2.0 s flap period equals the deterministic sync cadence, so every sync
# lands in a down window (phase lock). Production syncs on a jittered backoff
# (CloudUplink.next_delay); test_outage_recovery_on_the_production_backoff covers that path.
@pytest.mark.parametrize(("down_s", "up_s"), [(0.5, 0.9), (3.0, 2.0), (12.0, 4.0)])
def test_network_flapping_never_goes_stale_and_loses_nothing(stack, monitor, down_s, up_s):
    device = stack.boot(_flapping(f"flap_{down_s}_{up_s}", down_s, up_s, C + 90, (C + 20, C + 60)))
    stack.claim()
    stale = []
    for _ in range(int((C + 100) / 2)):
        device.advance(2.0)
        stale.append(stack.status()["stale"])
    assert not any(stale[int(C):]), "flaps shorter than the stale window must not read as offline"
    assert stack.event_types().count("drowsiness") == 2, stack.event_types()
    assert stack.status()["monitoring_state"] == "ready"
    assert device.network.refused > 0
    monitor.assert_clean()


# ---------------------------------------------------------------------- episodes vs cooldown
def test_two_close_episodes_are_two_alerts_not_one_lost_to_a_cooldown(stack, monitor):
    script = _timeline("drowsy_twice_fast", (C + 3, ASLEEP, Faults()), (C + 7, ATTENTIVE, Faults()),
                       (C + 9, ASLEEP, Faults()), (C + 13, ATTENTIVE, Faults()))
    device = stack.boot(script)
    stack.claim()
    device.advance(C + 20)
    drowsy = [e for e in stack.events() if e["type"] == "drowsiness"]
    assert len(drowsy) == 2, stack.event_types()
    played = [p for p in device.buzzer.played if p[0] == "drowsiness"]
    assert len(played) >= 2, device.buzzer.played
    monitor.assert_clean()


# ---------------------------------------------------------------------- compound hardware faults
class _FlakyBuzzer:
    """A buzzer that raises on play() while `broken` (a GPIO driver error)."""

    name, kind, audible = "sim_buzzer", "buzzer", True

    def __init__(self) -> None:
        self.broken = False
        self.played: list = []
        self.raised = 0

    def readiness(self):
        return (not self.broken), ("SIMULATED buzzer fault" if self.broken else "SIMULATED buzzer")

    def play(self, pattern, event, category) -> None:
        if self.broken:
            self.raised += 1
            raise OSError("SIMULATED GPIO write failed")
        self.played.append((event.type, category.name, pattern.name))

    def emit(self, event, level) -> None:
        self.play(None, event, type("L", (), {"name": level.value})())


def test_detector_then_camera_then_actuator_faults_each_recover(stack, monitor):
    script = _timeline(
        "compound",
        (C + 5, ATTENTIVE, Faults(detector_failing=True)), (C + 25, ATTENTIVE, Faults()),
        (C + 60, ATTENTIVE, Faults(camera="frozen")), (C + 75, ATTENTIVE, Faults()),
        (C + 100, ASLEEP, Faults()), (C + 104, ATTENTIVE, Faults()),
        (C + 130, ASLEEP, Faults()), (C + 134, ATTENTIVE, Faults()),
    )
    device = stack.boot(script)
    buzzer = _FlakyBuzzer()
    device.buzzer = buzzer
    _swap_buzzer(device.runtime, buzzer)
    stack.claim()
    device.advance(C + 10)
    assert stack.status()["monitoring_state"] == "degraded"
    device.advance(45.0)  # detector back at C+25, then the module's retry backoff
    assert stack.status()["monitoring_state"] == "ready"
    device.advance(10.0)  # frozen since C+60
    assert stack.status()["monitoring_state"] == "failed"
    device.advance(25.0)  # sensor back at C+75
    assert stack.status()["monitoring_state"] == "ready"

    buzzer.broken = True  # the actuator fails before the driver falls asleep
    device.advance(20.0)  # asleep at C+100 with no working buzzer
    status = stack.status()
    assert status["monitoring_state"] == "failed", status
    assert any("audible_alert" in r for r in status["reasons"]), status["reasons"]
    assert "drowsiness" in stack.event_types(), "the alert is still recorded and sent"

    buzzer.broken = False
    device.advance(25.0)  # repaired; asleep again at C+130
    assert stack.status()["monitoring_state"] == "ready"
    assert any(p[0] == "drowsiness" for p in buzzer.played)
    assert device.runtime.module_name == "driver_attention"
    monitor.assert_clean()


def _swap_buzzer(runtime, buzzer) -> None:
    sinks = runtime.alert_manager._sinks if hasattr(runtime.alert_manager, "_sinks") else None
    assert sinks is not None, "alert manager layout changed"
    for i, sink in enumerate(sinks):
        if getattr(sink, "name", "") == "sim_buzzer":
            sinks[i] = buzzer
            return
    raise AssertionError("no sim_buzzer sink")


# ---------------------------------------------------------------------- person rules on the driver role
def _activate(stack, rules: list[dict], prompt: str = "tell me when a phone is visible") -> None:
    preview = stack.ai_configure(prompt)
    activated = stack.set_preset("", draft_version=preview.json()["draft_version"], rules=rules)
    assert activated.status_code == 200, activated.text


SEAT_EMPTY = Scene(face=False, person=False, label="seat empty")
NOBODY = {"target": "person", "predicate": "absent", "severity": "warning", "persist_s": 3.0,
          "notify": "push", "description": "nobody in the seat"}


def test_a_person_rule_the_driver_role_accepts_sees_the_driver(stack, monitor):
    """driver_attention declares `person` (available_labels), so the device accepts person
    rules beside driver safety; the rule must then see the person boxes the detector finds."""
    device = stack.boot(_timeline("driver_leaves", (C + 20, SEAT_EMPTY, Faults()), (C + 30, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C)
    _activate(stack, [NOBODY])
    device.advance(6.0)
    assert [r.detection for r in device.runtime.rule_engine.rules] == ["person"]
    assert "person" in device.runtime.module.get_detections().labels(), "the driver is in frame"
    device.advance(12.0)  # C+18: the driver is still seated
    custom = [e["type"] for e in stack.events() if e["source"] == "custom_rule"]
    assert custom == [], f"'nobody in the seat' fired with the driver in the seat: {custom}"
    device.advance(10.0)  # C+28: the seat has been empty since C+20
    custom = [e["type"] for e in stack.events() if e["source"] == "custom_rule"]
    assert custom == ["person"], custom
    monitor.assert_clean()


# ---------------------------------------------------------------------- restart and persistence
@pytest.mark.parametrize("damage", ["truncated", "not_json", "wrong_shape", "null", "empty"])
def test_a_restart_over_damaged_state_files_boots_protecting_without_a_reboot(stack, monitor, damage):
    device = stack.boot(sc.get("attentive"))
    stack.claim()
    device.advance(C)
    assert stack.set_preset("driver_safety:strict").status_code == 200
    device.advance(4.0)
    _activate(stack, [NOBODY])
    device.advance(6.0)
    files = ["device_state.json", "monitoring.json", "calibration.json", "reboot_state.json"]
    for name in files:
        path = stack.state_dir / name
        original = path.read_text() if path.exists() else '{"x": 1}'
        path.write_text({"truncated": original[: len(original) // 2], "not_json": "\x00\xffgarbage",
                         "wrong_shape": "[1, 2, 3]", "null": "null", "empty": ""}[damage])

    reboots = []
    device = stack.restart(sc.get("attentive"))
    device.runtime._issue_reboot = lambda: reboots.append(1) or False
    device.advance(C + 4)  # a damaged calibration is redone from scratch
    status = stack.status()
    assert status["monitoring_state"] == "ready" and status["protection"] == "protecting", status
    assert device.runtime.module_name == "driver_attention", "the role floor survives lost state"
    assert reboots == [], "damaged state files must never escalate to a reboot"
    monitor.assert_clean()


def test_a_malformed_config_stops_once_and_never_loops(tmp_path):
    """main.py exits 78 (EX_CONFIG) for every kind of bad config, and the unit stops on 78."""
    import configparser

    import main
    from sim.device import REPO

    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.read(REPO / "sensordyme.service")
    prevent = {int(x) for x in unit["Service"]["RestartPreventExitStatus"].split()}
    assert unit["Service"]["Restart"] == "always" and main.CONFIG_ERROR_EXIT in prevent
    assert unit["Unit"]["StartLimitAction"] == "none", "a start-limit hit must not reboot by itself"

    bad = {"unreadable.yaml": "module: [unclosed\n", "list.yaml": "- 1\n- 2\n",
           "schema.yaml": "module: driver_attention\nmodules: 7\n", "empty.yaml": ""}
    for name, text in bad.items():
        path = tmp_path / name
        path.write_text(text)
        starts = 0
        for _attempt in range(10):  # systemd: Restart=always unless the status is prevented
            starts += 1
            with pytest.raises(SystemExit) as exited:
                main.build_runtime(["--config", str(path), "--no-display"])
            if exited.value.code in prevent:
                break
        assert starts == 1, f"{name}: exit {exited.value.code} would be restarted {starts} times"


# ---------------------------------------------------------------------- long-running monitoring
def test_twenty_minutes_of_monitoring_with_faults_keeps_every_episode(stack, monitor):
    import gc
    import threading

    minute = 60.0
    # An eyes-closed episode every 2.5 min from minute 1; three of them start as a fault
    # starts (outage at 6:00, detector failure at 11:00) or as one ends (stall to 16:00).
    asleep = [(minute + k * 150.0, minute + k * 150.0 + 4.5, ASLEEP) for k in range(8)]
    faults = [(6 * minute, 7 * minute, DOWN), (11 * minute, 12 * minute, Faults(detector_failing=True)),
              (15 * minute + 30, 16 * minute, Faults(camera="stalled"))]
    script = _compose("twenty_minutes", asleep, faults)
    device = stack.boot(script)
    stack.claim()
    samples = []
    for _ in range(20):
        device.advance(minute)
        gc.collect()
        samples.append({"objects": len(gc.get_objects()), "threads": threading.active_count(),
                        "queue": len(device.runtime.uplink._queue)})
    drowsy = stack.event_types().count("drowsiness")
    assert drowsy == 8, f"{drowsy} of 8 episodes stored"
    assert stack.status()["monitoring_state"] == "ready"
    growth = samples[-1]["objects"] - samples[4]["objects"]
    assert growth < 20_000, f"{growth} objects retained over 15 minutes: a leak"
    assert len({s["threads"] for s in samples}) == 1, samples
    monitor.assert_clean()
    print(f"20 min: objects min5={samples[4]['objects']} min20={samples[-1]['objects']} (+{growth}); "
          f"threads {samples[-1]['threads']}; max backlog {monitor.max_queue}; "
          f"checks device={monitor.device_checks} app={monitor.backend_checks}; "
          f"transitions {len(monitor.transitions)}")


# ---------------------------------------------------------------------- production reconnection cadence
def test_outage_recovery_on_the_production_backoff(tmp_path):
    """The deterministic harness normally syncs every poll interval; here it waits what the
    production worker waits (CloudUplink.next_delay: jittered exponential backoff, cap 30 s).
    Measured: attempts during a 120 s outage, and how long the app stays stale after it."""
    import random

    from core.uplink import BACKOFF_CAP_S

    random.seed(20261111)
    outage = _timeline("long_outage", (C + 5, ATTENTIVE, DOWN), (C + 125, ATTENTIVE, Faults()))
    with SimStack(tmp_path) as stack:
        monitor = InvariantMonitor(stack)
        device = stack.boot(outage, honour_backoff=True)
        stack.claim()
        device.advance(C + 5)
        refused_before = device.network.refused
        device.advance(120.0)
        attempts = device.network.refused - refused_before
        assert attempts <= 10, f"{attempts} attempts in a 120 s outage: no backoff"
        back_at = device.clock.monotonic()
        waited = 0.0
        while stack.status()["stale"]:
            device.advance(0.5)
            waited = device.clock.monotonic() - back_at
            assert waited <= BACKOFF_CAP_S + 2.0, "reconnection took longer than the backoff cap"
        assert stack.status()["monitoring_state"] == "ready"
        monitor.assert_clean()
        print(f"backoff: {attempts} attempts in 120 s outage; fresh {waited:.1f} s after the network returned")
