"""M8 monitoring modes on the virtual device: Occupancy, Property, Workplace Safety, Child.

Each mode is driven the way the app drives it -- the real backend API (preset,
zones, nodes), the real uplink, the real DeviceRuntime -- on one SimClock, with
the simulator's one scripted detector shared by every module build and node
(sim/perception.py), so `detector.calls` counts every inference on the device.
Scenes hold people at given positions (Scene.people), so counting, the shared
tracker, area crossings and dwell run on boxes that move.

Per mode: declared requirements, real activation, observation delivery,
events, state transitions, restart persistence (same state dir), refusals with
a code, and compatible concurrency on ONE detector run per frame.

Simulated evidence only: plumbing, identities and call counts, never accuracy
on real footage. Set SD_M8_EVIDENCE_DIR to keep each test's API transcript.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_sim_reliability import _CommandTap

from sim.harness import SimStack
from sim.perception import Scene
from sim.scenario import Faults, Scenario, Segment

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

GENERAL = {"role": "general", "module": "shared_perception"}
EMPTY = Scene(face=False, person=False, label="empty room")
LEFT = {"name": "left", "polygon": [[0.0, 0.5], [0.4, 0.5], [0.4, 1.0], [0.0, 1.0]], "kind": "restricted"}


def people(*xs: float) -> Scene:
    """Standing people at these horizontal positions (feet at y = 0.95)."""
    return Scene(face=False, person=False, people=tuple(xs), label=f"{len(xs)} people")


def walk(start: float, x0: float, x1: float, steps: int = 4) -> list[tuple]:
    """One person walking from x0 to x1 over `steps` seconds, a step a second
    (the simulator places boxes; it does not interpolate between segments)."""
    return [(start + i, people(x0 + (x1 - x0) * (i + 1) / steps)) for i in range(steps)]


def room(name: str, *segments: tuple) -> Scenario:
    """(start, scene[, faults]) after an empty room at t = 0."""
    return Scenario(name, name, (Segment(0.0, EMPTY), *(Segment(float(s[0]), s[1], s[2] if len(s) > 2 else Faults())
                                                       for s in segments)))


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def boot(stack: SimStack, script: Scenario, **kwargs):
    tap = _CommandTap(stack.transport)
    if stack.device_id is None:
        stack.claim()
    device = stack.boot(script, transport=tap, config_overrides=GENERAL, **kwargs)
    return device, tap


def restart(stack: SimStack, script: Scenario, tap):
    return stack.restart(script, transport=tap, config_overrides=GENERAL)


def activate(stack: SimStack, device, preset: str) -> dict:
    response = stack.set_preset(preset)
    assert response.status_code == 200, response.text
    device.advance(2.5)
    assert device.runtime.active_preset == preset
    return response.json()


def draw(stack: SimStack, device, *zones: dict) -> None:
    response = stack.api.post(f"/v1/devices/{stack.device_id}/zones", json={"zones": list(zones)},
                              headers=stack.headers)
    assert response.status_code == 200, response.text
    device.advance(2.5)


def enable_node(stack: SimStack, node: str, enabled: bool = True, settings=None):
    body = {"node": node, "enabled": enabled, **({"settings": settings} if settings else {})}
    return stack.api.post(f"/v1/devices/{stack.device_id}/nodes", json=body, headers=stack.headers)


def perception(device) -> dict:
    return device.runtime.perception_report()


def node(device, node_id: str) -> dict:
    return next(n for n in perception(device)["nodes"] if n["id"] == node_id)


def catalogue(stack: SimStack) -> dict:
    response = stack.api.get("/v1/presets", headers=stack.headers)
    assert response.status_code == 200
    body = response.json()
    presets = body["presets"] if isinstance(body, dict) else body
    return {p["name"] if "name" in p else p["id"]: p for p in presets}


def counters(device) -> tuple[int, int]:
    return device.sim.detector.calls, device.runtime.perception.frames


def events(stack: SimStack, source: str | None = None) -> list[dict]:
    return [e for e in stack.events() if source is None or e.get("source") == source]


def record(name: str, payload: dict) -> None:
    """Keep an API/state transcript when SD_M8_EVIDENCE_DIR is set (never by default)."""
    target = os.environ.get("SD_M8_EVIDENCE_DIR")
    if target:
        path = Path(target).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))


# ====================================================================== Occupancy
def test_occupancy_counts_tracked_people_entries_exits_areas_and_dwell(stack):
    """A enters the left area and stays 35 s (dwell), walks right; B arrives in the
    area; both leave. Every event names the person by the shared track id."""
    script = room("occupancy", (5, people(0.2)), *walk(45, 0.2, 0.8), (55, people(0.8, 0.2)), (70, EMPTY))
    device, _tap = boot(stack, script)
    runtime = device.runtime
    assert runtime.active_preset == "occupancy" and runtime.module_name == "shared_perception"
    preset = catalogue(stack)["occupancy"]
    assert set(preset["required_capabilities"]) >= {"camera", "inference", "objects"}
    draw(stack, device, LEFT)

    timeline = []
    for _ in range(16):
        device.advance(5.0)
        counter = node(device, "people_counter")
        timeline.append((round(device.sim.state.clock.now()), counter["state"], counter["metrics"]["count"],
                         counter["metrics"]["zones"]["left"]["count"]))
    counter_node = runtime._preset_nodes["people_counter"]
    assert counter_node.observations > 50, "the node received the shared observations"
    assert runtime._last_observation.tracks is not None, "every observation carries the shared tracks"

    produced = [(e["type"], e.get("signals")) for e in events(stack, "node:people_counter")]
    kinds = [k for k, _ in produced]
    assert kinds.count("person_entered") == 2 and kinds.count("person_exited") == 2
    assert kinds.count("zone_entered") == 2 and kinds.count("zone_exited") == 2
    assert kinds.count("zone_dwell") == 1, "only A stayed 30 s; B left after 15 s"
    details = " | ".join(str(d) for _, d in produced)
    assert "person #1 entered" in details and "person #2 entered" in details
    assert "person #1 in 'left' for 30s" in details, "dwell is per person, from the shared track"
    assert {state for _, state, _, _ in timeline} == {"active"}
    counts = [c for _, _, c, _ in timeline]
    assert 1 in counts and 2 in counts and counts[-1] == 0
    assert stack.status()["monitoring_state"] == "ready"
    calls, frames = counters(device)
    assert calls <= frames
    record("occupancy_lifecycle", {"timeline(t,state,count,left)": timeline, "events": produced,
                                   "tracking": perception(device)["tracking"], "calls": calls, "frames": frames})


def test_occupancy_survives_a_restart_without_phantom_entries(stack):
    # Two people from the first frame: the scenario clock starts over at every boot.
    script = Scenario("occupancy_restart", "two people", (Segment(0.0, people(0.2, 0.7)),))
    device, tap = boot(stack, script)
    draw(stack, device, LEFT)
    device.advance(15.0)
    before = [e["type"] for e in events(stack, "node:people_counter")]
    assert "person_entered" not in before, "people present when counting starts are the baseline"
    assert node(device, "people_counter")["metrics"]["count"] == 2
    device = restart(stack, script, tap)
    runtime = device.runtime
    assert runtime.active_preset == "occupancy" and set(runtime._preset_nodes) == {"people_counter"}
    assert [z.name for z in runtime._node_zones()] == ["left"], "the drawn area survived the restart"
    device.advance(15.0)
    after = [e["type"] for e in events(stack, "node:people_counter")][len(before):]
    assert not {"person_entered", "person_exited", "zone_entered", "zone_exited"} & set(after), \
        "the two people present at boot are the new baseline, not two new arrivals"
    assert node(device, "people_counter")["metrics"]["count"] == 2


def test_a_detector_outage_degrades_then_recounts_without_inventing_exits(stack):
    fail = Faults(detector_failing=True)
    script = room("occupancy_outage", (2, people(0.3)), (12, people(0.3), fail), (20, people(0.3)))
    device, _tap = boot(stack, script)
    device.advance(11.0)
    assert node(device, "people_counter")["state"] == "active"
    device.advance(5.0)  # t=16: failing since 12
    during = node(device, "people_counter")
    assert during["state"] == "failed" and "objects" in during["missing_capabilities"]
    assert device.runtime._last_observation.tracks is None, "no tracks are carried past a failure"
    device.advance(40.0)
    after = node(device, "people_counter")
    assert after["state"] == "active" and after["metrics"]["count"] == 1
    kinds = [e["type"] for e in events(stack, "node:people_counter")]
    assert kinds == ["person_entered"], "the person never left: no exit and no second entry"


# ====================================================================== Property (security_monitoring)
def test_property_monitoring_activates_alerts_and_shares_one_detector_with_companions(stack):
    script = room("property", (8, people(0.2)), *walk(50, 0.2, 0.8), (70, EMPTY))
    device, _tap = boot(stack, script)
    preset = catalogue(stack)["security_monitoring"]
    assert set(preset["required_capabilities"]) >= {"camera", "inference", "objects", "alert_output"}
    activate(stack, device, "security_monitoring")
    runtime = device.runtime
    assert runtime.module_name == "occupancy"
    draw(stack, device, LEFT)
    for node_id in ("people_counter", "zone_watch"):
        assert enable_node(stack, node_id).json()["pending"] is True
    device.advance(4.0)
    assert set(runtime._companions) == {"people_counter", "zone_watch"}

    calls0, frames0 = counters(device)
    device.advance(70.0)
    calls, frames = (a - b for a, b in zip(counters(device), (calls0, frames0), strict=True))
    assert 0 < calls <= frames, "module + two companions: at most one detector run per frame"
    report = perception(device)
    detector = report["components"]["object_detector"]
    assert set(detector["consumers"]) >= {"occupancy", "people_counter", "zone_watch"}
    assert report["tracking"]["updates"] == device.sim.detector.calls, "one association per detector run"
    kinds = [(e["type"], e["source"]) for e in stack.events()]
    assert ("zone_violation", "module") in kinds and ("person_lingering", "module") in kinds
    assert ("restricted_area", "node:zone_watch") in kinds
    assert ("person_entered", "node:people_counter") in kinds and ("person_exited", "node:people_counter") in kinds
    entry = node(device, "security_monitoring")
    assert entry["kind"] == "preset" and entry["state"] == "active" and entry["validation"] == "synthetic_tests"
    status = stack.status()
    assert status["monitoring_state"] == "ready" and {n["id"] for n in status["perception"]["nodes"]} == {
        "security_monitoring", "people_counter", "zone_watch"}
    record("property_concurrency", {"frames": frames, "detector_runs": calls, "consumers": detector["consumers"],
                                    "events": kinds, "backend_nodes": status["perception"]["nodes"]})


def test_property_monitoring_and_its_companions_survive_a_restart(stack):
    script = room("property_restart", (3, people(0.2)))
    device, tap = boot(stack, script)
    activate(stack, device, "security_monitoring")
    draw(stack, device, LEFT)
    enable_node(stack, "people_counter")
    device.advance(4.0)
    device = restart(stack, script, tap)
    runtime = device.runtime
    assert runtime.active_preset == "security_monitoring" and runtime.module_name == "occupancy"
    assert set(runtime._companions) == {"people_counter"}
    assert [z.name for z in runtime.module._watched_zones] == ["left"]
    violations = len([e for e in stack.events() if e["type"] == "zone_violation"])
    device.advance(10.0)
    assert len([e for e in stack.events() if e["type"] == "zone_violation"]) == violations + 1, \
        "the area alert runs again after the restart"


# ====================================================================== Workplace Safety
def test_workplace_safety_restricted_area_with_a_counter_on_one_detector(stack):
    script = room("site", (6, people(0.8)), *walk(12, 0.8, 0.2), *walk(30, 0.2, 0.8))
    device, _tap = boot(stack, script)
    activate(stack, device, "workplace_safety")
    runtime = device.runtime
    assert runtime.module_name == "shared_perception" and set(runtime._preset_nodes) == {"zone_watch"}
    draw(stack, device, LEFT)
    enable_node(stack, "people_counter", settings={"settle_s": 1.0})
    device.advance(4.0)
    calls0, frames0 = counters(device)
    device.advance(30.0)
    calls, frames = device.sim.detector.calls - calls0, runtime.perception.frames - frames0
    assert calls <= frames / 2 + 2, "the module's every-2nd-frame cadence answers both nodes' demands"
    restricted = events(stack, "node:zone_watch")
    assert [e["type"] for e in restricted] == ["restricted_area"] and "left" in restricted[0]["signals"]
    assert node(device, "zone_watch")["metrics"]["areas"] == ["left"]
    assert node(device, "people_counter")["enabled_by"] == "owner"
    device = restart(stack, script, _tap)
    assert device.runtime.active_preset == "workplace_safety"
    assert set(device.runtime._preset_nodes) == {"zone_watch"} and set(device.runtime._companions) == {
        "people_counter"}


def test_workplace_safety_claims_no_ppe_or_vehicle_detection():
    from core.presets import load_presets

    preset = load_presets()["workplace_safety"]
    text = " ".join(preset["limitations"]).lower() + preset["description"].lower()
    assert "no ppe" in text and "forklift" in text
    assert {p["event"] for p in preset["temporal_policies"]} == {"restricted_area"}


# ====================================================================== Child (baby_monitoring)
CRIB = {"name": "crib", "polygon": [[0.0, 0.5], [0.4, 0.5], [0.4, 1.0], [0.0, 1.0]], "kind": "safe"}
STAIRS = {"name": "stairs", "polygon": [[0.6, 0.5], [1.0, 0.5], [1.0, 1.0], [0.6, 1.0]], "kind": "restricted"}


def test_child_monitoring_runs_its_supported_alerts_and_reports_pose_features_as_experimental(stack):
    script = room("nursery", (3, people(0.2)), *walk(20, 0.2, 0.8), (40, EMPTY))
    device, _tap = boot(stack, script)
    preset = catalogue(stack)["baby_monitoring"]
    assert "objects" in preset["required_capabilities"] and "pose" in preset["optional_capabilities"]
    activate(stack, device, "baby_monitoring")
    draw(stack, device, CRIB, STAIRS)
    enable_node(stack, "people_counter")
    calls0, frames0 = counters(device)
    device.advance(80.0)
    calls, frames = device.sim.detector.calls - calls0, device.runtime.perception.frames - frames0
    assert 0 < calls <= frames, "the baby module and the counter share each detector run"
    kinds = [e["type"] for e in events(stack, "module")]
    assert "safe_zone_exit" in kinds and "zone_violation" in kinds and "person_absent" in kinds
    assert "flailing" not in kinds and "choking_hazard" not in kinds, \
        "the simulator scripts no pose: pose-based signals never fire and are never claimed"
    entry = node(device, "baby_monitoring")
    features = {f["id"]: f for f in entry["experimental_features"]}
    assert set(features) == {"flailing", "choking_hazard"}
    # M8: off by default; an owner opts in (below). Reported truthfully either way.
    assert all(not f["enabled"] and f["status"].startswith("experimental") for f in features.values())
    assert device.runtime.module._pose is None, "no pose inference unless the owner opts in"
    assert entry["validation"] == "synthetic_tests" and entry["validation"] not in ("device_verified",
                                                                                     "production_validated")
    backend_entry = next(n for n in stack.status()["perception"]["nodes"] if n["id"] == "baby_monitoring")
    assert {f["id"]: f["enabled"] for f in backend_entry["experimental_features"]} == {
        "flailing": False, "choking_hazard": False}, "the app is told which parts are experimental"
    # The owner opts in through the preset setting: the module now runs pose, and
    # the report follows the config the running module was built from (not the
    # bare config.yaml block), on the device and in the backend status.
    opted = stack.set_preset("baby_monitoring", settings={"enable_pose": True})
    assert opted.status_code == 200, opted.text
    device.advance(2.5)
    assert device.runtime.module._pose is not None
    entry = node(device, "baby_monitoring")
    assert all(f["enabled"] for f in entry["experimental_features"])
    device.advance(2.5)
    backend_entry = next(n for n in stack.status()["perception"]["nodes"] if n["id"] == "baby_monitoring")
    assert {f["id"]: f["enabled"] for f in backend_entry["experimental_features"]} == {
        "flailing": True, "choking_hazard": True}
    limitations = " ".join(preset["limitations"]).lower()
    assert "not a medical device" in limitations and "does not detect choking" in limitations
    device = restart(stack, script, _tap)
    assert device.runtime.active_preset == "baby_monitoring@enable_pose=true", "the opt-in survives a restart"
    assert device.runtime.module._pose is not None
    assert sorted(z.name for z in device.runtime._node_zones()) == ["crib", "stairs"]
    record("child_monitoring", {"events": kinds, "experimental_features": entry["experimental_features"],
                                "frames": frames, "detector_runs": calls})


@pytest.mark.parametrize("prompt", [
    "tell me if my baby stops breathing",
    "alert me if the baby is choking",
    "let me know if the baby rolls onto her stomach",
    "alert me if my baby is face down",
])
def test_child_emergency_prompts_are_refused_not_approximated(stack, prompt):
    stack.claim()
    response = stack.ai_configure(prompt)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rules"] == [], "nothing is set up"
    assert body["unsupported"] and body["unsupported"][0]["category"] == "medical"
