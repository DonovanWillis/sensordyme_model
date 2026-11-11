"""Mission 7 on the virtual device: one shared perception engine, several monitoring nodes.

Everything here runs the real DeviceRuntime, real modules, the real uplink and the
real backend (sim/, one SimClock). The object detector is the simulator's one
scripted detector, shared by every module build and node exactly as the device
shares its one loaded model, so `sim.detector.calls` counts every real inference
on the device whoever asked. Commands the backend stream has not shipped yet
(set_nodes) are delivered the way an ingest response carries them (_CommandTap).

Simulated evidence only: these prove the plumbing and the call counts, never
accuracy on real footage.
"""

from __future__ import annotations

import pytest
from test_sim_reliability import _CommandTap

from core import nodes as monitoring_nodes
from core import perception as pc
from sim import scenario as sc
from sim.harness import SimStack
from sim.scenario import ABSENT, ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

C = CALIBRATION_S
STEP = 0.05  # one camera frame and one inference pass per step (20 fps)
CABIN = {"name": "cabin", "polygon": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]], "kind": "restricted"}
PERSON_IN_CABIN = {"target": "person", "predicate": "in_zone", "zone": "cabin", "severity": "warning",
                   "notify": "push", "persist_s": 1.0, "description": "Someone in the cabin"}


def _scenario(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0, ATTENTIVE), *(Segment(float(t), scene, faults or Faults())
                                                            for t, scene, faults in segments)))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _boot(stack, scenario, **kwargs):
    tap = _CommandTap(stack.transport)
    stack.claim()
    device = stack.boot(scenario, transport=tap, **kwargs)
    return device, tap


def _perception(device) -> dict:
    return device.runtime.monitoring_report().to_dict()["perception"]


def _node(device, node_id) -> dict:
    return next(n for n in _perception(device)["nodes"] if n["id"] == node_id)


def _frames(device) -> int:
    return device.runtime.perception.frames


# ---------------------------------------------------------------------- shared inference
def test_driver_safety_a_custom_rule_and_a_companion_share_one_detector_run_per_frame(stack):
    script = _scenario("cabin", (C + 15, ABSENT, None), (C + 24, ATTENTIVE, None))
    device, tap = _boot(stack, script)
    device.advance(C)  # calibration
    tap.inject.append({"set_zones": [CABIN], "zones_version": "z1",
                       "set_rules": [PERSON_IN_CABIN], "rules_version": "r1",
                       "set_nodes": {"people_counter": {"enabled": True, "settings": {"settle_s": 1.0}}}})
    device.advance(2.0)
    runtime = device.runtime
    assert runtime.module_name == "driver_attention" and set(runtime._companions) == {"people_counter"}

    calls0, frames0 = device.sim.detector.calls, _frames(device)
    device.advance(30.0)
    calls, frames = device.sim.detector.calls - calls0, _frames(device) - frames0
    # At most once per analysed frame, whoever asked. driver_attention runs it on
    # every 3rd frame; the counter's own 0.5 s demand is always answered from
    # those runs, so the node costs no inference at all.
    assert 0 < calls <= frames
    assert calls == pytest.approx(frames / 3, abs=2), (calls, frames)
    separate_pipelines = frames / 3 + 30.0 / 0.5  # what a second, model-owning node would have run
    print(f"\n[M7 call counts] frames={frames} detector_runs={calls} "
          f"(separate pipelines would run ~{separate_pipelines:.0f})")

    report = _perception(device)
    detector = report["components"]["object_detector"]
    assert detector["state"] == "ok" and set(detector["consumers"]) >= {"driver_attention", "people_counter"}
    assert {n["id"]: n["kind"] for n in report["nodes"]} == {
        "driver_safety": "preset", "rules": "rules", "people_counter": "companion"}
    # All three consumers got the same observations:
    types = stack.event_types()
    assert "person" in types, "the custom person-in-cabin rule fired"
    assert {"person_exited", "person_entered"} <= set(types), "the companion counted the seat emptying and filling"
    events = {e["type"]: e for e in stack.events()}
    assert events["person_exited"]["source"] == "node:people_counter"
    assert any(t in types for t in ("attention_lost",)), "the driver module kept watching (empty seat)"
    counter = _node(device, "people_counter")
    assert counter["state"] == "active" and counter["metrics"]["count"] == 1
    assert counter["validation"] == "synthetic_tests" and counter["enabled_by"] == "owner"


def test_disabling_one_node_leaves_the_others_running(stack):
    device, tap = _boot(stack, _scenario("steady"))
    device.advance(C)
    tap.inject.append({"set_zones": [CABIN], "zones_version": "z1", "set_rules": [PERSON_IN_CABIN],
                       "rules_version": "r1",
                       "set_nodes": {"people_counter": {"enabled": True}, "zone_watch": {"enabled": True}}})
    device.advance(3.0)
    assert set(device.runtime._companions) == {"people_counter", "zone_watch"}
    tap.inject.append({"set_nodes": {"zone_watch": {"enabled": False}}})
    device.advance(2.0)
    runtime = device.runtime
    assert set(runtime._companions) == {"people_counter"}, "only the named node was disabled"
    assert {d.consumer for d in runtime.perception.demands()} == {"people_counter"}
    observed = runtime._companions["people_counter"].observations
    rules_alerts = stack.event_types().count("person")
    device.advance(5.0)
    assert runtime._companions["people_counter"].observations > observed, "the counter keeps consuming"
    assert runtime.module_name == "driver_attention" and runtime.monitoring_report().state.value == "ready"
    assert stack.event_types().count("person") >= rules_alerts and runtime.rule_engine.rules
    assert "zone_watch" not in {n["id"] for n in _perception(device)["nodes"]}


# ---------------------------------------------------------------------- failure isolation
def test_a_detector_failure_degrades_objects_while_face_monitoring_keeps_alerting_then_recovers(stack):
    fail = Faults(detector_failing=True)
    script = _scenario("detector_down", (C + 5, ASLEEP, fail), (C + 15, ATTENTIVE, fail), (C + 20, ATTENTIVE, None))
    device, tap = _boot(stack, script)
    device.advance(C)
    tap.inject.append({"set_nodes": {"people_counter": {"enabled": True}}})
    device.advance(13.0)  # C+13: asleep with the detector failing
    report = device.runtime.monitoring_report()
    component = report.perception["components"]["object_detector"]
    assert component["state"] == "failed" and "SIMULATED detector fault" in component["reason"]
    assert report.capabilities["objects"].impaired and report.capabilities["face"].available
    assert report.state.value == "degraded", "face-based safety still runs: degraded, not failed"
    counter = _node(device, "people_counter")
    assert counter["state"] == "failed" and "objects" in counter["missing_capabilities"]
    assert "drowsiness" in {p[0] for p in device.buzzer.played}, "drowsiness still alerts on face alone"
    assert device.runtime._last_observation.detections is None, "no boxes are carried past a failure"
    device.advance(40.0)  # detector back since C+20; the module's own retry backoff has expired
    report = device.runtime.monitoring_report()
    assert report.perception["components"]["object_detector"]["state"] == "ok"
    assert report.capabilities["objects"].available and _node(device, "people_counter")["state"] == "active"


def test_a_frozen_camera_makes_components_stale_and_fabricates_nothing(stack):
    script = _scenario("frozen", (C + 5, ATTENTIVE, Faults(camera="frozen")), (C + 15, ATTENTIVE, None))
    device, tap = _boot(stack, script)
    device.advance(C)
    tap.inject.append({"set_nodes": {"people_counter": {"enabled": True}}})
    device.advance(6.0)  # C+6: frozen since C+5
    last_frame = device.runtime._last_observation.frame_id
    device.advance(3.0)  # C+9: no new frame for 4 s
    assert device.runtime._last_observation.frame_id == last_frame, "no frame, no observation"
    perception = _perception(device)
    assert perception["components"]["object_detector"]["state"] == "stale"
    assert perception["components"]["face_landmarks"]["state"] == "stale"
    counter = _node(device, "people_counter")
    assert counter["state"] == "failed" and "stale" in counter["reason"]
    assert device.runtime.monitoring_report().state.value == "failed"  # the camera is required
    device.advance(10.0)
    assert _perception(device)["components"]["object_detector"]["state"] == "ok"


# ---------------------------------------------------------------------- commands
def test_a_repeated_set_nodes_has_no_second_effect(stack, monkeypatch):
    device, tap = _boot(stack, _scenario("steady"))
    device.advance(C)
    command = {"set_nodes": {"people_counter": {"enabled": True, "settings": {"settle_s": 1.0}}}}
    tap.inject.append(dict(command))
    device.advance(3.0)
    node = device.runtime._companions["people_counter"]
    seen = node.observations
    from core import device_state

    writes = []
    real_update = device_state.update
    monkeypatch.setattr(device_state, "update", lambda **f: writes.append(f) or real_update(**f))
    for _ in range(3):
        tap.inject.append(dict(command))
        device.advance(2.0)
    assert device.runtime._companions["people_counter"] is node, "the running node was not rebuilt"
    assert node.observations > seen and node.count == 1, "its state carried on"
    assert writes == [], "nothing new to persist"
    assert [d.consumer for d in device.runtime.perception.demands()] == ["people_counter"]


@pytest.mark.parametrize(("command", "code"), [
    ({"driver_off": {"enabled": True}}, "unknown_node"),
    ({"people_counter": {"enabled": True, "settings": {"settle_s": 99}}}, "invalid_setting"),
    ({"people_counter": {"enabled": "yes"}}, "malformed"),
])
def test_an_unrunnable_node_is_refused_whole_with_a_code(stack, command, code):
    device, tap = _boot(stack, _scenario("steady"))
    device.advance(C)
    tap.inject.append({"set_nodes": {"zone_watch": {"enabled": True}, **command}})
    device.advance(3.0)
    rejection = device.runtime._last_rejection
    assert rejection["kind"] == "nodes" and rejection["code"] == code and rejection["reasons"]
    assert device.runtime._companions == {}, "all or nothing: zone_watch did not start either"


def test_no_node_or_preset_can_remove_driver_attention_on_a_driver_safety_device(stack):
    device, tap = _boot(stack, _scenario("steady"))
    device.advance(C)
    tap.inject.append({"set_preset": "occupancy"})
    device.advance(2.0)
    runtime = device.runtime
    assert runtime._last_rejection["kind"] == "preset" and runtime._last_rejection["subject"] == "occupancy"
    tap.inject.append({"set_nodes": {"people_counter": {"enabled": True}, "zone_watch": {"enabled": True}}})
    device.advance(3.0)
    assert runtime.module_name == "driver_attention" and runtime.active_preset == "driver_safety"
    assert set(runtime._companions) == {"people_counter", "zone_watch"}, "companions are additive"
    assert runtime.protection_state()["state"] == "protecting"
    preset = _node(device, "driver_safety")
    assert preset["state"] == "active" and preset["experimental_features"][0]["enabled"] is False


# ---------------------------------------------------------------------- persistence
def test_nodes_and_rules_survive_a_restart(stack):
    script = _scenario("steady")
    device, tap = _boot(stack, script)
    device.advance(C)
    tap.inject.append({"set_zones": [CABIN], "zones_version": "z1", "set_rules": [PERSON_IN_CABIN],
                       "rules_version": "r1",
                       "set_nodes": {"people_counter": {"enabled": True, "settings": {"settle_s": 1.5}}}})
    device.advance(3.0)
    device = stack.restart(script, transport=tap)
    runtime = device.runtime
    assert set(runtime._companions) == {"people_counter"}
    assert runtime._companions["people_counter"].settings["settle_s"] == 1.5
    assert [r.detection for r in runtime.rule_engine.rules] == ["person"]
    calls0, frames0 = device.sim.detector.calls, _frames(device)
    device.advance(C + 5.0)
    assert device.sim.detector.calls - calls0 <= _frames(device) - frames0
    assert _node(device, "people_counter")["metrics"]["count"] == 1


# ---------------------------------------------------------------------- node-backed presets
GENERAL = {"role": "general", "module": "shared_perception"}


def test_the_occupancy_preset_runs_on_shared_perception_alone(stack):
    script = _scenario("room", (C + 10, ABSENT, None), (C + 20, ATTENTIVE, None))
    device, _tap = _boot(stack, script, config_overrides=GENERAL)
    runtime = device.runtime
    assert runtime.module_name == "shared_perception" and runtime.active_preset == "occupancy"
    assert set(runtime._preset_nodes) == {"people_counter"}
    calls0, frames0 = device.sim.detector.calls, _frames(device)
    device.advance(C + 30.0)
    assert device.sim.detector.calls - calls0 <= _frames(device) - frames0
    assert {"person_exited", "person_entered"} <= set(stack.event_types())
    nodes = {n["id"]: n for n in _perception(device)["nodes"]}
    assert nodes["occupancy"]["kind"] == "preset" and nodes["people_counter"]["enabled_by"] == "preset"
    assert nodes["people_counter"]["metrics"]["count"] == 1
    assert runtime.monitoring_report().state.value == "ready"
    # Nothing on this device ran a face or hand model.
    components = _perception(device)["components"]
    assert components["face_landmarks"]["runs"] == 0 and components["hand_landmarks"]["runs"] == 0


def test_workplace_safety_alerts_on_a_person_in_a_restricted_area(stack):
    device, tap = _boot(stack, _scenario("site"), config_overrides=GENERAL)
    device.advance(2.0)
    assert stack.set_preset("workplace_safety").status_code == 200
    device.advance(4.0)
    assert device.runtime.active_preset == "workplace_safety"
    tap.inject.append({"set_zones": [CABIN], "zones_version": "z1"})
    device.advance(6.0)
    events = [e for e in stack.events() if e["type"] == "restricted_area"]
    assert events and events[0]["source"] == "node:zone_watch" and "cabin" in events[0]["signals"]
    assert _node(device, "zone_watch")["metrics"]["areas"] == ["cabin"]


# ---------------------------------------------------------------------- profile
def test_profile_component_costs_on_the_virtual_device(stack):
    """Per-component call counts and run time (simulated models: the times are the
    scripted stand-ins', not a Pi's; the call counts are the point)."""
    device, tap = _boot(stack, sc.get("attentive"))
    device.advance(C)
    tap.inject.append({"set_nodes": {"people_counter": {"enabled": True}}})
    frames0 = _frames(device)
    runs0 = {k: v["runs"] for k, v in _perception(device)["components"].items()}
    device.advance(20.0)
    frames = _frames(device) - frames0
    components = _perception(device)["components"]
    runs = {k: components[k]["runs"] - runs0[k] for k in components}
    print(f"\n[M7 profile] frames={frames} " + " ".join(
        f"{k}={runs[k]} ({components[k]['run_ms']} ms)" for k in pc.COMPONENTS))
    assert runs["face_landmarks"] == frames, "face landmarks once per frame"
    assert runs["object_detector"] <= frames / 3 + 2, "detector on the driver stride, shared with the node"
    assert runs["hand_landmarks"] == runs["object_detector"], "hands ride the detector ticks"
    assert runs["pose"] == 0
    for name in pc.COMPONENTS:
        assert components[name]["state"] in pc.STATES


def test_every_companion_declares_the_node_contract():
    for node_id, cls in monitoring_nodes.COMPANIONS.items():
        spec = cls.spec
        assert spec.id == node_id and spec.kind == "companion" and spec.version >= 1
        assert {"camera", "inference"} <= set(spec.required_capabilities)
        assert spec.components and spec.events and spec.limitations and spec.temporal_logic
        assert spec.settings and all(s.default is not None for s in spec.settings)
        assert monitoring_nodes.validation_tier(node_id) == "synthetic_tests"
