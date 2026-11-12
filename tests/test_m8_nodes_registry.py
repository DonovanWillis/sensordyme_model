"""M8 node registry: a new companion is one class + one registry entry, never a model.

Proves the extension path core/nodes.py documents: a class registered with
@register_companion is runnable by id (set_nodes), listed in the catalogue the
backend validates against, reports `exists_in_code` until core/model_registry
.yaml gives it a tier, and asks the SHARED detector through the default
demand -- so on a virtual device it adds no inference at all.
"""

from __future__ import annotations

import pytest
from test_m8_nodes_modes import GENERAL, boot, counters, node, people, room

from core import nodes as cn
from core import perception as pc
from core.presets import load_companion_nodes


class _LoiterCount(cn.MonitoringNode):
    """A toy companion: how many confirmed person tracks have been seen for 3 s."""

    spec = cn.NodeSpec(
        id="m8_test_loiter", version=1, kind="companion", title="Test loiter count",
        required_capabilities=("camera", "inference", "objects"), components=(pc.OBJECT_DETECTOR,),
        settings=cn._settings([{"key": "interval_s", "title": "Interval", "type": "number", "default": 0.5,
                                "min": 0.1, "max": 5.0, "step": 0.1, "unit": "s"}]),
        events={"loitering": frozenset({"objects"})}, temporal_logic=("a track followed for 3 s",),
        limitations=("test node",),
    )
    detect_labels = frozenset({"person"})

    def __init__(self, settings=None):
        super().__init__(settings)
        self.seen: set[int] = set()

    def consume(self, observation, now):
        if not observation.fresh_detections or observation.tracks is None:
            return []
        self.observations += 1
        out = []
        for track in observation.tracks:
            if track.label == "person" and track.confirmed and track.age_s >= 3.0 and track.track_id not in self.seen:
                self.seen.add(track.track_id)
                out.append(cn.NodeEvent(cn.Event.now("loitering", track.age_s, track.confidence,
                                                     f"node:{self.id}", f"person #{track.track_id}"), "log"))
        return out

    def metrics(self):
        return {"seen": len(self.seen)}


@pytest.fixture
def registered():
    cn.register_companion(_LoiterCount)
    try:
        yield _LoiterCount
    finally:
        cn.COMPANIONS.pop(_LoiterCount.spec.id, None)


def test_registering_a_node_makes_it_runnable_and_listed(registered):
    assert cn.COMPANIONS["m8_test_loiter"] is registered
    assert "m8_test_loiter" in {n["id"] for n in load_companion_nodes()}
    built = cn.build_companion("m8_test_loiter", {"interval_s": 1.0})
    (demand,) = built.demands()
    assert demand.component == pc.OBJECT_DETECTOR and demand.labels == {"person"} and demand.min_interval_s == 1.0
    assert cn.validation_tier("m8_test_loiter") == "exists_in_code", "no registry entry: no claim"
    assert cn.register_companion(registered) is registered, "re-registering the same class is a no-op"


def test_the_registry_refuses_clashes_and_unknown_components(registered):
    clash = type("Clash", (_LoiterCount,), {})
    with pytest.raises(ValueError, match="already registered"):
        cn.register_companion(clash)
    import dataclasses

    lidar = type("Lidar", (_LoiterCount,), {"spec": dataclasses.replace(_LoiterCount.spec, id="m8_lidar",
                                                                          components=("lidar",))})
    with pytest.raises(ValueError, match="unknown perception component"):
        cn.register_companion(lidar)
    preset_kind = type("P", (_LoiterCount,), {"spec": dataclasses.replace(_LoiterCount.spec, id="m8_p",
                                                                            kind="preset")})
    with pytest.raises(ValueError, match="kind 'companion'"):
        cn.register_companion(preset_kind)
    assert set(cn.COMPANIONS) == {"people_counter", "zone_watch", "m8_test_loiter"}


def test_a_new_node_on_the_virtual_device_costs_no_inference(tmp_path, registered):
    from sim.harness import SimStack

    with SimStack(tmp_path) as stack:
        device, tap = boot(stack, room("loiter", (2, people(0.3)), (4, people(0.3, 0.7))))
        assert GENERAL["module"] == device.runtime.module_name
        device.advance(5.0)
        calls0, frames0 = counters(device)
        device.advance(10.0)
        baseline_rate = (device.sim.detector.calls - calls0) / (device.runtime.perception.frames - frames0)
        tap.inject.append({"set_nodes": {"m8_test_loiter": {"enabled": True}}})
        device.advance(2.0)
        assert set(device.runtime._companions) == {"m8_test_loiter"}
        calls0, frames0 = counters(device)
        device.advance(10.0)
        rate = (device.sim.detector.calls - calls0) / (device.runtime.perception.frames - frames0)
        assert rate == pytest.approx(baseline_rate, abs=0.02), "the new node rode the existing detector runs"
        entry = node(device, "m8_test_loiter")
        assert entry["state"] == "active" and entry["validation"] == "exists_in_code"
        assert entry["metrics"]["seen"] == 2, "it reads the shared tracker's two ids"
        consumers = device.runtime.perception_report()["components"]["object_detector"]["consumers"]
        assert {"m8_test_loiter", "people_counter", "shared_perception"} <= set(consumers)
        assert [e["type"] for e in stack.events() if e["source"] == "node:m8_test_loiter"] == ["loitering"] * 2
