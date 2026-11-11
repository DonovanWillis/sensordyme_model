"""The virtual device's own parts: scripted perception, SimCamera, scenarios, identity, CLI."""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from core.camera import CameraError
from sim import identity as sim_identity
from sim import scenario as sc
from sim.camera import BANNER, SimCamera, in_stall_gap, render
from sim.clock import SimClock
from sim.perception import (
    Scene,
    ScriptedDetector,
    ScriptedFaceMesh,
    ScriptedHands,
    decode_scene,
    face_landmarks,
)
from sim.scenario import Faults, Scenario, ScenarioClock, ScenarioState, Segment


class _Faults:
    def __init__(self, **kw):
        self.faults = Faults(**kw)

    def current(self):
        return self.faults


def _state(scenario, clock):
    return ScenarioState(scenario, ScenarioClock(clock.monotonic))


# ---------------------------------------------------------------------- perception
@pytest.mark.parametrize("scene", [
    Scene(), Scene(eyes_closed=True), Scene(face=False, person=False), Scene(head_down_deg=30.0, phone=True,
                                                                             hand_on_phone=True),
    Scene(yaw_deg=-45.0), Scene(head_down_deg=-127.0, yaw_deg=127.0),
])
def test_a_scene_survives_the_frame_including_bgr_to_rgb(scene):
    import cv2

    frame = render(scene, 320, 240)
    for image in (frame, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)):
        decoded = decode_scene(image)
        assert decoded == Scene(**{**scene.__dict__, "label": ""})


def test_a_frame_the_sim_did_not_render_carries_no_scene():
    assert decode_scene(np.zeros((48, 64, 3), dtype=np.uint8)) is None


def _ear(points, w, h, ids):
    p = [np.array([points[i][0] * w, points[i][1] * h]) for i in ids]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]))


@pytest.mark.parametrize(("width", "height"), [(640, 480), (320, 240)])
def test_closed_eyes_are_closed_by_both_measures_the_module_uses(width, height):
    from core.eye_signal import iris_openness
    from modules.driver_attention import LEFT_EYE, RIGHT_EYE

    open_pts = face_landmarks(width, height)
    closed_pts = face_landmarks(width, height, eyes_closed=True)
    for ids in (LEFT_EYE, RIGHT_EYE):
        assert _ear(open_pts, width, height, ids) == pytest.approx(0.4, abs=0.02)
        assert _ear(closed_pts, width, height, ids) < 0.1
    scale = lambda pts: [(x * width, y * height) for x, y in pts]  # noqa: E731
    assert iris_openness(scale(open_pts)) > 0.7
    assert iris_openness(scale(closed_pts)) < 0.2  # below the iris veto: a real closure


def test_scripted_models_answer_from_the_frame():
    face_mesh, hands = ScriptedFaceMesh(), ScriptedHands()
    faults = _Faults()
    detector = ScriptedDetector(faults)
    with_phone = render(Scene(head_down_deg=30.0, phone=True, hand_on_phone=True), 640, 480)
    assert face_mesh.process(with_phone).multi_face_landmarks[0].landmark[1].x > 0
    labels = sorted(ScriptedDetector.names[int(b.cls[0])] for b in detector.predict(with_phone)[0].boxes)
    assert labels == ["cell phone", "person"]
    assert hands.process(with_phone).multi_hand_landmarks
    empty = render(Scene(face=False, person=False), 640, 480)
    assert face_mesh.process(empty).multi_face_landmarks is None
    assert detector.predict(empty)[0].boxes == []
    faults.faults = Faults(detector_failing=True)
    with pytest.raises(RuntimeError, match="SIMULATED detector fault"):
        detector.predict(with_phone)


def test_install_rejects_a_module_it_cannot_script():
    from modules.base import UnavailableModule
    from sim.perception import install

    with pytest.raises(TypeError):
        install(UnavailableModule("x", "y"), _Faults())


# ---------------------------------------------------------------------- camera
def test_every_frame_says_simulated_in_its_pixels():
    frame = render(Scene(), 640, 480)
    assert frame.shape == (480, 640, 3)
    red = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 30) & (frame[:, :, 1] < 30)
    assert red[-10:, :].mean() > 0.5, "bottom SIMULATED bar"
    assert red[:30, -60:].any(), "top-right SIMULATED tag"
    assert BANNER == "SIMULATED"


def test_camera_faults():
    clock = SimClock()
    script = Scenario("t", "t", (Segment(0.0), Segment(10.0, faults=Faults(camera="frozen")),
                                 Segment(20.0, faults=Faults(camera="dropping")),
                                 Segment(30.0, faults=Faults(camera="stalled"))))
    camera = SimCamera(_state(script, clock), width=64, height=48, sleep=clock.advance)
    assert camera.read() is not None and camera.is_open()
    clock.advance(10.0)
    with pytest.raises(CameraError, match="SIMULATED camera fault: frozen"):
        camera.read()
    with pytest.raises(CameraError):
        camera.reopen()
    assert camera.grab() is None and not camera.is_open()
    clock.advance(10.0)
    got = [camera.grab() is not None for _ in range(6)]
    assert got.count(True) == 3, got
    clock.advance(10.0)
    assert camera.grab() is not None  # 30.0: inside the flow window
    clock.advance(1.5)
    assert in_stall_gap(camera._state.clock.now()) and camera.grab() is None  # 31.5: in a gap
    assert camera.read() is not None, "read() holds through the gap, as a stalled sensor does"


# ---------------------------------------------------------------------- scenarios
def test_the_catalogue_is_well_formed():
    for name, scenario in sc.SCENARIOS.items():
        assert scenario.name == name and scenario.description
        starts = [s.start for s in scenario.segments]
        assert starts == sorted(starts) and starts[0] == 0.0
        assert scenario.at(0.0).scene.face, f"{name}: calibration needs a face at boot"
        first_change = next((s.start for s in scenario.segments if s.start > 0), sc.CALIBRATION_S)
        assert first_change >= sc.CALIBRATION_S, f"{name} changes the scene during calibration"
    with pytest.raises(KeyError, match="known:"):
        sc.get("nope")


def test_scenario_clock_and_overrides():
    clock = SimClock()
    state = ScenarioState(sc.get("backend_outage"), ScenarioClock(clock.monotonic, speed=2.0))
    assert state.current().network_down is False
    clock.advance((sc.CALIBRATION_S + 6) / 2)
    assert state.clock.now() == pytest.approx(sc.CALIBRATION_S + 6)
    assert state.current().network_down is True
    state.fault_overrides = {"network_down": False}
    assert state.current().network_down is False
    with pytest.raises(ValueError, match="speed must be"):
        ScenarioClock(clock.monotonic, speed=0)


# ---------------------------------------------------------------------- identity
def test_identity_is_created_once_and_survives(tmp_path):
    with pytest.raises(ValueError, match="no virtual device identity"):
        sim_identity.load_or_create(tmp_path)
    first = sim_identity.load_or_create(tmp_path, device_key="dk_1", claim_code="AA-BB")
    assert first.hardware_id.startswith("SD-SIM-") and len(first.hardware_id) == len("SD-SIM-") + 8
    assert sim_identity.load_or_create(tmp_path) == first
    rekeyed = sim_identity.load_or_create(tmp_path, device_key="dk_2", claim_code="CC-DD")
    assert rekeyed.hardware_id == first.hardware_id and rekeyed.device_key == "dk_2"


# ---------------------------------------------------------------------- local API
def test_the_device_local_api_says_simulated(tmp_path):
    pytest.importorskip("mediapipe")
    from sim.harness import SimStack

    with SimStack(tmp_path) as stack:
        device = stack.boot(sc.get("attentive"), dashboard_port=0)
        device.advance(3.0)
        url = device.runtime.dashboard.url  # the sim binds 127.0.0.1
        with urllib.request.urlopen(f"{url}/api/health?token={device.dashboard_token}", timeout=5) as r:  # noqa: S310
            health = json.loads(r.read())
        assert health["device"]["simulated"] is True
        assert health["device"]["device_id"] == device.identity.hardware_id
        assert "camera sensor" in health["device"]["simulation"]["not_simulated"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{url}/api/health?token=wrong", timeout=5)  # noqa: S310


# ---------------------------------------------------------------------- CLI
def test_cli_lists_scenarios(capsys):
    from sim.virtual_device import main

    assert main(["--list-scenarios"]) == 0
    out = capsys.readouterr().out
    for name in ("drowsy", "camera_freeze", "backend_outage", "tour"):
        assert name in out


def test_cli_reads_the_factory_file(tmp_path):
    from sim.virtual_device import _from_factory

    path = tmp_path / "factory.json"
    path.write_text(json.dumps({"email": "e", "password": "p", "devices": [
        {"name": "a", "device_key": "dk_a", "claim_code": "A"},
        {"name": "b", "device_key": "dk_b", "claim_code": "B"}]}))
    assert _from_factory(path, None)["device_key"] == "dk_a"
    assert _from_factory(path, "B")["device_key"] == "dk_b"
    with pytest.raises(SystemExit):
        _from_factory(path, "Z")
