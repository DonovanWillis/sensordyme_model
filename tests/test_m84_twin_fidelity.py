"""M8.4: the fidelity map is explicit, complete and carried by every twin report."""

from __future__ import annotations

from pathlib import Path

import pytest

from sim import fidelity
from sim import scenario as sc
from sim.harness import SimStack

REPO = Path(__file__).resolve().parent.parent
DOC = (REPO / "docs" / "DIGITAL_TWIN.md").read_text()
PRODUCT = ("camera", "capture", "frame_bus", "perception_engine", "face_landmarks", "hand_landmarks",
           "object_detector", "hailo_decoder", "monitoring_state", "uplink", "backend", "dashboard_stream",
           "ble_provisioning", "buzzer", "thermal", "storage")


def test_every_product_component_has_an_honest_entry():
    for name in PRODUCT:
        entry = fidelity.FIDELITY[name]
        assert entry["code"] in ("real", "real+seam", "stand-in", "absent"), name
        if entry["code"] != "real":
            assert entry["fakes"] and entry["fakes"] != "nothing", f"{name}: a stand-in must say what it fakes"
    # The models and the actuator are never real in the twin: no claim may say so.
    for name in ("camera", "face_landmarks", "hand_landmarks", "object_detector", "buzzer"):
        assert fidelity.FIDELITY[name]["code"] == "stand-in", name


def test_the_doc_table_matches_the_module():
    rows = {line.split("|")[1].strip(): line for line in DOC.splitlines() if line.startswith("| ") and "**" in line}
    words = {"frame_bus": "FrameBus", "perception_engine": "perception engine", "face_landmarks": "FaceMesh",
             "hand_landmarks": "Hands", "object_detector": "object detector", "hailo_decoder": "Hailo decoder",
             "driver_attention_logic": "driver_attention logic", "monitoring_state": "monitoring state",
             "alert_policy": "alert policy", "buzzer": "buzzer / actuators", "dashboard_stream": "dashboard / `/stream`",
             "ble_provisioning": "BLE provisioning", "config_boot": "config / boot", "health_ladder": "health ladder",
             "pose": "Pose"}
    kind = {"real": "**real**", "real+seam": "**real + seam**", "stand-in": "**stand-in**"}
    for name, entry in fidelity.FIDELITY.items():
        label = words.get(name, name)
        assert label in rows, f"docs/DIGITAL_TWIN.md has no fidelity row for {label!r}"
        assert kind[entry["code"]] in rows[label], f"{label}: the doc and sim/fidelity.py disagree"


@pytest.mark.parametrize("_", [None])
def test_the_device_health_report_names_its_stand_ins(tmp_path, _):
    pytest.importorskip("fastapi")
    pytest.importorskip("mediapipe")
    with SimStack(tmp_path) as stack:
        device = stack.boot(sc.get("attentive"))
        stack.claim()
        device.advance(4.0)
        simulation = stack.row()["health"]["simulation"]
        assert set(simulation["stand_ins"]) == set(fidelity.STAND_INS)
        assert "camera" in simulation["stand_ins"] and "object_detector" in simulation["stand_ins"]
