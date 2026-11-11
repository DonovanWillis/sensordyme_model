"""Monitoring nodes (core/nodes.py) and their registry tiers, without a runtime."""

from __future__ import annotations

import dataclasses

import pytest

from core import model_registry as mr
from core import nodes as N
from core import perception as pc
from core.monitoring_state import IMPAIRED, Capability
from core.presets import PresetError, load_companion_nodes
from core.zones import Zone

W, H = 640, 480
PERSON = (160.0, 40.0, 480.0, 470.0)  # feet at (0.5, 0.98)
ROOM_FLOOR = Zone("floor", ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)), "restricted")


def _obs(people: int, *, fresh: bool = True, available: bool = True, fid: int = 1) -> pc.Observation:
    detections = tuple(pc.ObjectDetection("person", 0.9, PERSON, 0) for _ in range(people))
    status = pc.ComponentStatus(pc.OBJECT_DETECTOR, pc.OK if available else pc.FAILED,
                                reason=None if available else "NPU stream aborted")
    return pc.Observation(
        frame_id=fid, captured_at=0.0, source="sim", simulated=True, width=W, height=H,
        detections=detections if available else None, detections_age_s=0.0 if available else None,
        fresh_detections=fresh and available, detection_consumers=frozenset(), face_visible=None, face=None,
        hands=None, components={pc.OBJECT_DETECTOR: status})


def _run(node, plan, step=0.25):
    """plan: [(seconds, people)] -> every event type produced."""
    t, out = 0.0, []
    for seconds, people in plan:
        for _ in range(round(seconds / step)):
            out += [e.event.type for e in node.consume(_obs(people), t)]
            t += step
    return out


def test_people_counter_settles_counts_then_reports_entry_and_exit():
    node = N.OccupancyNode({"settle_s": 1.0})
    events = _run(node, [(2.0, 1), (0.5, 2), (0.5, 1), (2.0, 2), (2.0, 0)])
    assert events == ["person_entered", "person_exited"], "a half-second flicker is not a change"
    assert node.count == 0


def test_capacity_and_dwell_alerts_fire_once():
    node = N.OccupancyNode({"settle_s": 0.5, "capacity": 1, "dwell_s": 5})
    node.set_zones([ROOM_FLOOR])
    events = _run(node, [(2.0, 1), (8.0, 2)])
    assert events.count("capacity_exceeded") == 1
    assert events.count("zone_occupied") == 1 and events.count("zone_dwell") == 1
    assert node.metrics()["zones"]["floor"] == {"count": 2, "occupied": True}


def test_unavailable_detections_are_never_counted_as_an_empty_room():
    node = N.OccupancyNode({"settle_s": 0.5})
    _run(node, [(1.0, 1)])
    for t in range(20):
        assert node.consume(_obs(0, available=False), 10.0 + t) == []
    assert node.count == 1 and "NPU stream aborted" in node.degraded_reason
    assert node.consume(_obs(0, fresh=False), 40.0) == [], "between detector runs nothing new is observed"


def test_zone_watch_is_a_typed_rule_bundle_on_the_rule_engine():
    node = N.ZoneWatchNode({"persist_s": 0.5})
    node.set_zones([ROOM_FLOOR, Zone("door", ROOM_FLOOR.polygon, "safe")])
    assert [r["zone"] for r in node.rules()] == ["floor"], "only restricted areas are watched"
    assert node.problems == []
    events = []
    for i in range(20):
        events += node.consume(_obs(1, fid=i), i * 0.25)
    assert [e.event.type for e in events] == ["restricted_area"]
    assert events[0].severity == "soft" and events[0].notify == "push"


def test_settings_are_validated_against_the_node_schema():
    with pytest.raises(PresetError, match="settle_s"):
        N.OccupancyNode({"settle_s": 99})
    with pytest.raises(PresetError, match="not a setting"):
        N.OccupancyNode({"nope": 1})
    assert N.ZoneWatchNode({"severity": "critical"}).settings["severity"] == "critical"
    with pytest.raises(KeyError):
        N.build_companion("driver_off")


def test_node_state_from_capabilities():
    spec = N.OccupancyNode.spec
    ok = {"camera": Capability(), "inference": Capability(), "objects": Capability()}
    assert N.node_state(spec, ok) == (N.ACTIVE, [], None)
    lost = {**ok, "objects": Capability(IMPAIRED, "stale")}
    state, missing, reason = N.node_state(spec, lost)
    assert state == N.FAILED and missing == ["objects"] and "stale" in reason
    assert N.node_state(spec, ok, stopped=True)[0] == N.INACTIVE


def test_the_catalogue_the_backend_reads():
    catalogue = {n["id"]: n for n in load_companion_nodes()}
    assert set(catalogue) == {"people_counter", "zone_watch"}
    assert catalogue["people_counter"]["required_capabilities"] == ["camera", "inference", "objects"]
    assert {s["key"] for s in catalogue["zone_watch"]["settings"]} >= {"persist_s", "severity"}


# ---------------------------------------------------------------------- registry
def test_every_node_tier_is_capped_by_its_evidence():
    registry = mr.load_registry()
    assert mr.node_violations(registry) == [] and mr.component_violations(registry) == []
    for node in registry.nodes:
        tier = mr.node_tier(registry, node.id)
        weakest = min(mr.tier_rank(registry.product_capability(c).claimed_tier) for c in node.capabilities)
        assert mr.tier_rank(tier) <= weakest, node.id
        assert mr.tier_rank(tier) < mr.tier_rank("device_verified"), "nothing is device-verified"
    assert mr.node_tier(registry, "unknown_node") == "exists_in_code"
    served = mr.capability_tiers(registry)["nodes"]
    assert {n["node"] for n in served} >= {"driver_safety", "occupancy", "people_counter", "zone_watch", "rules"}
    assert not any(n["production_validated"] or n["device_verified"] for n in served)


def test_a_node_claiming_more_than_its_capabilities_is_refused():
    registry = mr.load_registry()
    inflated = dataclasses.replace(registry.node("people_counter"), claimed_tier="device_verified")
    bad = dataclasses.replace(registry, nodes=tuple(inflated if n.id == "people_counter" else n
                                                    for n in registry.nodes))
    assert any("can never claim more" in v for v in mr.node_violations(bad))
    unknown = dataclasses.replace(registry.node("zone_watch"), components=("lidar",))
    bad = dataclasses.replace(registry, nodes=(*registry.nodes, dataclasses.replace(unknown, id="lidar_watch")))
    assert any("not declared in inference_components" in v for v in mr.node_violations(bad))
