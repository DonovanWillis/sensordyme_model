"""M8.2: a device whose monitoring was stopped ON PURPOSE reports its capabilities
as `paused`, not as failing or stale; real faults found while stopped stay faults.

Before M8.2 a stopped device's components went stale (nothing runs), so the app
read "Not working — face landmarks stale" under a correct "Stopped" headline.
Runs the real runtime, uplink and backend on the virtual device (sim/).
"""

from __future__ import annotations

import pytest

from core import monitoring_state as ms
from sim.harness import SimStack
from sim.scenario import ASLEEP, ATTENTIVE, CALIBRATION_S, Faults, Scenario, Segment

C = CALIBRATION_S


@pytest.fixture
def stack(tmp_path):
    with SimStack(tmp_path) as s:
        yield s


def _timeline(name: str, *segments: tuple) -> Scenario:
    return Scenario(name, name, (Segment(0.0), *(Segment(float(t), scene, faults)
                                                  for t, scene, faults in segments)))


def _stop(stack, device) -> dict:
    assert stack.protection("stop", confirm=True).status_code == 200
    device.advance(5.0)
    return stack.status()


def test_a_stopped_device_reports_paused_not_failing_and_resumes_ready(stack):
    device = stack.boot(_timeline("paused", (C + 60, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C)
    assert stack.status()["monitoring_state"] == "ready"

    device.advance(40.0)  # stopped well past every staleness and start-up grace
    status = _stop(stack, device)
    device.advance(40.0)
    status = stack.status()
    assert status["monitoring_state"] == "stopped" and status["protection"] == "stopped"
    caps = status["capabilities"]
    assert caps["impaired"] == [], f"nothing is broken, only stopped: {caps['impaired']}"
    assert {"face", "inference"} <= set(caps["paused"]), caps
    assert "camera" in caps["available"], "the camera keeps capturing while stopped"
    # Nothing that analyses is "producing results" while stopped, including the
    # driver's calibrated attention; the hardware outputs keep their real state.
    assert not ({"face", "objects", "hands", "inference", "attention"} & set(caps["available"])), caps
    assert "attention" in caps["paused"], caps
    assert not ({"camera", "audible_alert", "alert_output"} & set(caps["paused"])), caps
    reasons = device.runtime.monitoring_report().to_dict()["reasons"]
    assert reasons == ["monitoring stopped from the app"], reasons

    report = device.runtime.monitoring_report().to_dict()
    for name in caps["paused"]:
        cap = report["capabilities"][name]
        assert cap["state"] == ms.PAUSED and "stopped" in cap["reason"], (name, cap)
    components = status["perception"]["components"]
    for name in ("face_landmarks", "object_detector"):
        assert components[name]["state"] == "paused", (name, components[name])
        assert components[name]["reason"] == "monitoring is stopped"

    # The preset readiness the app reads says paused, not impaired or unsupported.
    presets = stack.api.get(f"/v1/devices/{stack.device_id}/presets", headers=stack.headers).json()
    driver = next(p for p in presets["presets"] if p["id"] == "driver_safety")
    face = next(c for c in driver["capabilities"] if c["name"] == "face")
    assert face["state"] == "paused", face

    # Resume: everything comes back, with no stale result counted as fresh.
    assert stack.protection("resume").status_code == 200
    device.advance(C + 10.0)
    status = stack.status()
    assert status["monitoring_state"] == "ready", status
    assert status["capabilities"]["paused"] == []
    assert "face" in status["capabilities"]["available"]


def test_a_real_fault_while_stopped_is_still_reported(stack):
    """Paused never hides a fault: a camera that stops delivering frames while the
    device is stopped is impaired, while the analysis capabilities are paused."""
    device = stack.boot(_timeline("paused_fault", (C + 8, ATTENTIVE, Faults(camera="frozen")),
                                  (C + 90, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C)
    status = _stop(stack, device)
    device.advance(40.0)
    status = stack.status()
    assert status["monitoring_state"] == "stopped"
    impaired = {item["name"]: item["reason"] for item in status["capabilities"]["impaired"]}
    assert "camera" in impaired, status["capabilities"]
    assert "face" in status["capabilities"]["paused"]


def test_stopping_a_driver_unit_still_needs_confirmation_and_alerts_resume(stack):
    """The driver-safety rules are unchanged: a stop needs an explicit confirmation,
    nothing alerts while stopped, and drowsiness alerts again after resume."""
    device = stack.boot(_timeline("paused_driver", (C + 20, ASLEEP, Faults()), (C + 26, ATTENTIVE, Faults()),
                                  (C + 60, ASLEEP, Faults()), (C + 66, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C)
    refused = stack.protection("stop")
    assert refused.status_code == 400 and refused.json()["code"] == "confirmation_required"
    _stop(stack, device)
    device.advance(25.0)  # asleep while stopped
    assert "drowsiness" not in stack.event_types()
    assert stack.protection("resume").status_code == 200
    device.advance(40.0)  # asleep again after resume
    assert "drowsiness" in stack.event_types()


def test_old_apps_ignore_paused_and_never_see_a_stopped_capability_as_available(stack):
    """Compatibility: `paused` is additive. An app that reads only available/
    impaired/starting finds the paused capabilities in none of them, so it never
    shows them as working; the headline still says stopped."""
    device = stack.boot(_timeline("paused_compat", (C + 60, ATTENTIVE, Faults())))
    stack.claim()
    device.advance(C)
    _stop(stack, device)
    device.advance(40.0)
    caps = stack.status()["capabilities"]
    old_view = set(caps["available"]) | {i["name"] for i in caps["impaired"]} | set(caps["starting"])
    assert not (old_view & set(caps["paused"])), caps
