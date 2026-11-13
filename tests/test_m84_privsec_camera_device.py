"""M8.4 owner Decision C, device side: no remote live camera while monitoring is stopped.

M8.3 added a camera fallback: whenever the runtime was not drawing --
monitoring stopped included -- /stream sent the raw camera to anyone holding
the device token, on by default, with no owner switch. Decision C reverses the
default:

  * stopped + camera-only OFF (the default): a REMOTE viewer is sent no frame
    at all -- not the camera, not a picture left in the slot from before the
    stop -- and heartbeats say `camera_off_while_stopped` (the app can then
    say "Camera off -- monitoring stopped", not "stalled");
  * a loopback viewer (on the device itself) keeps the camera for local
    maintenance, labelled `camera_local`, unless the config turns that off;
  * camera-only ON (only the owner, via the backend's signed command) lets
    the camera through, labelled `camera_only`;
  * the setting survives a restart, and every device upgraded from M8.3 (no
    saved setting) starts OFF;
  * the uplink refuses an unsigned command that turns camera-only ON.

Remote viewers are simulated by making the dashboard's locality check answer
"not local" (the test client necessarily connects over loopback). On c9029fa
that check does not exist and every viewer gets the camera, so the remote
tests fail there on the behaviour itself.
"""

from __future__ import annotations

import json
import threading
import time
import types
import urllib.request

import numpy as np
import pytest

import core.dashboard as dashboard_module
from core.dashboard import DashboardServer
from core.frame_bus import FrameBus
from sim.stream_client import StreamClient

TOKEN = "m84-privsec-token"
STOPPED = {"protection": {"state": "stopped"}}
RUNNING = {"protection": {"state": "protecting"}}


def _frame(value: int) -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 1] = value % 256
    frame[0, 0, 0] = value % 256
    return frame


class _Camera:
    """Publishes to a FrameBus like CaptureService does."""

    def __init__(self, bus: FrameBus, fps: float = 30.0) -> None:
        self.bus, self.interval = bus, 1.0 / fps
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        n = 0
        while not self.stop.is_set():
            n += 1
            self.bus.publish(_frame(n))
            self.stop.wait(self.interval)

    def __enter__(self) -> "_Camera":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


@pytest.fixture()
def remote(monkeypatch):
    """Every /stream viewer counts as remote (as a phone on the LAN would)."""
    cp = getattr(dashboard_module, "cp", None)
    if cp is not None:
        monkeypatch.setattr(cp, "peer_is_local", lambda host: False)


def _server(bus: FrameBus, **kwargs) -> DashboardServer:
    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN, stream_fps=50, **kwargs)
    server.attach_frame_source(bus)
    server.start()
    return server


def _camera_only_on():
    from core.camera_privacy import CameraPrivacy, CameraPrivacySetting

    return CameraPrivacy(CameraPrivacySetting(camera_only=True), persist=False)


# ------------------------------------------------------------------ the default: off while stopped
def test_stopped_remote_viewer_gets_no_frames_and_an_explicit_reason(remote):
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(STOPPED)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            client.read_for(2.6)
        assert client.status == 200
        assert client.jpeg_parts() == [], "a remote viewer saw the camera while monitoring was stopped"
        beats = client.heartbeats()
        assert len(beats) >= 2
        assert {b.headers["x-stream-status"] for b in beats} == {"camera_off_while_stopped"}
        assert client.headers.get("x-stream-camera") == "off_while_stopped"
        # Not even encoded: the camera is never read for a viewer who may not see it.
        assert server.stream_stats()["camera_fallback_encodes"] == 0
    finally:
        server.close()


def test_stopped_remote_viewer_without_heartbeats_gets_nothing(remote):
    """An old app or a browser <img>: silence, never a frame."""
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(STOPPED)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN) as client:
            client.read_for(1.5)
        assert client.status == 200 and client.parts == []
    finally:
        server.close()


def test_a_picture_from_before_the_stop_is_not_sent_either(remote):
    """The slot still holds the last analysed frame (< 5 s old) at the moment of the stop."""
    bus = FrameBus()
    server = _server(bus)
    server.push_frame(_frame(7), captured_at=time.monotonic())
    server.set_policy_state(STOPPED)
    try:
        with StreamClient(server._port, TOKEN, heartbeat=True) as client:
            client.read_for(1.3)
        assert client.jpeg_parts() == []
    finally:
        server.close()


def test_a_stop_mid_stream_cuts_frames_and_resume_restores_them_on_the_same_connection(remote):
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(RUNNING)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 5, timeout=5)
            server.set_policy_state(STOPPED)
            client.read_for(0.3)  # anything already in flight
            before = len(client.jpeg_parts())
            assert client.read_until(lambda c: any(
                b.headers.get("x-stream-status") == "camera_off_while_stopped" for b in c.heartbeats()), timeout=5)
            client.read_for(1.0)
            assert len(client.jpeg_parts()) == before, "frames kept flowing after the stop"
            server.set_policy_state(RUNNING)
            assert client.read_until(lambda c: len(c.jpeg_parts()) > before + 3, timeout=5), (
                "frames did not come back on resume")
    finally:
        server.close()


def test_monitoring_running_but_not_drawing_still_streams_the_camera(remote):
    """Not a stop: a degraded module. Monitoring is on, the camera is in use for it; unchanged."""
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(RUNNING)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 5, timeout=5)
        assert {p.headers["x-frame-source"] for p in client.jpeg_parts()} == {"camera"}
        assert client.headers.get("x-stream-camera") == "monitoring"
    finally:
        server.close()


# ------------------------------------------------------------------ owner opt-in: camera only
def test_camera_only_on_delivers_frames_labelled_not_monitoring(remote):
    bus = FrameBus()
    server = _server(bus, camera_privacy=_camera_only_on())
    server.set_policy_state(STOPPED)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 5, timeout=5)
        assert {p.headers["x-frame-source"] for p in client.jpeg_parts()} == {"camera_only"}
        assert client.headers.get("x-stream-camera") == "camera_only"
        state = server.state()["camera"]
        assert state["camera_only"] is True and state["remote_stream"] == "camera_only"
    finally:
        server.close()


def test_camera_only_does_not_change_anything_while_monitoring_runs(remote):
    bus = FrameBus()
    server = _server(bus, camera_privacy=_camera_only_on())
    server.set_policy_state(RUNNING)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 3, timeout=5)
        assert {p.headers["x-frame-source"] for p in client.jpeg_parts()} == {"camera"}
    finally:
        server.close()


# ------------------------------------------------------------------ local maintenance (loopback)
def test_a_loopback_viewer_keeps_the_camera_for_local_maintenance():
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(STOPPED)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            assert client.read_until(lambda c: len(c.jpeg_parts()) >= 5, timeout=5)
        assert {p.headers["x-frame-source"] for p in client.jpeg_parts()} == {"camera_local"}
        assert client.headers.get("x-stream-camera") == "local_maintenance"
    finally:
        server.close()


def test_local_maintenance_view_can_be_switched_off():
    bus = FrameBus()
    server = _server(bus, local_camera_while_stopped=False)
    server.set_policy_state(STOPPED)
    try:
        with _Camera(bus), StreamClient(server._port, TOKEN, heartbeat=True) as client:
            client.read_for(1.5)
        assert client.jpeg_parts() == []
        assert client.heartbeats()[0].headers["x-stream-status"] == "camera_off_while_stopped"
    finally:
        server.close()


def test_the_token_is_still_required_for_a_loopback_viewer():
    bus = FrameBus()
    server = _server(bus)
    server.set_policy_state(STOPPED)
    try:
        with StreamClient(server._port, None) as anon:
            anon.read_for(0.5)
        assert anon.status == 401
    finally:
        server.close()


def test_peer_locality():
    from core.camera_privacy import peer_is_local

    assert peer_is_local("127.0.0.1") and peer_is_local("::1") and peer_is_local("::ffff:127.0.0.1")
    assert not peer_is_local("192.168.1.20") and not peer_is_local("fe80::1%en0")
    assert not peer_is_local("") and not peer_is_local(None) and not peer_is_local("localhost")


# ------------------------------------------------------------------ reported state
def test_api_state_reports_what_remote_viewers_get():
    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN)
    server.start()
    try:
        server.set_policy_state(STOPPED)
        request = urllib.request.Request(f"http://127.0.0.1:{server._port}/api/state",
                                         headers={"X-Device-Token": TOKEN})
        with urllib.request.urlopen(request, timeout=5) as reply:  # noqa: S310 - loopback test server
            camera = json.loads(reply.read())["camera"]
        assert camera["camera_only"] is False and camera["remote_stream"] == "off_while_stopped"
        assert camera["schema"] == 1
        assert server.stream_stats()["remote_stream"] == "off_while_stopped"
        server.set_policy_state(RUNNING)
        assert server.state()["camera"]["remote_stream"] == "monitoring"
    finally:
        server.close()


def test_there_is_no_dashboard_endpoint_to_turn_the_camera_on():
    """The dashboard token may be shared with non-owners: it must not reach this setting."""
    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN)
    server.start()
    try:
        for path in ("/api/camera-mode", "/api/camera", "/api/privacy"):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server._port}{path}", method="POST",
                data=json.dumps({"camera_only": True}).encode(),
                headers={"X-Device-Token": TOKEN, "Content-Type": "application/json"})
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(request, timeout=5)  # noqa: S310 - loopback test server
            assert refused.value.code == 404
        assert server.camera_privacy.camera_only is False
    finally:
        server.close()


# ------------------------------------------------------------------ persistence + migration
def test_the_setting_survives_a_restart(tmp_path):
    from core.camera_privacy import CameraPrivacy

    privacy = CameraPrivacy.load()
    assert privacy.camera_only is False
    assert privacy.apply(True, command_id="c0ffee000001", by="owner") is True
    reloaded = CameraPrivacy.load()  # a new process
    assert reloaded.camera_only is True and reloaded.setting.command_id == "c0ffee000001"
    reloaded.apply(False, command_id="c0ffee000002")
    assert CameraPrivacy.load().camera_only is False


def test_a_device_upgraded_from_m83_starts_with_camera_only_off(tmp_path, monkeypatch):
    from core import device_state
    from core.camera_privacy import CameraPrivacy

    # What an M8.3 device has on disk: a preset and zones, no camera setting.
    device_state.update(preset="driver_safety", config_module="driver_attention", zones=[])
    assert CameraPrivacy.load().camera_only is False


@pytest.mark.parametrize("stored", [
    {"schema": 1, "camera_only": "true"}, {"schema": 1, "camera_only": 1}, {"schema": 2, "camera_only": True},
    {"camera_only": True}, ["camera_only"], "on", None,
])
def test_anything_but_an_exact_saved_true_is_off(stored):
    from core import device_state
    from core.camera_privacy import CameraPrivacy

    device_state.update(camera_privacy=stored)
    assert CameraPrivacy.load().camera_only is False


def test_an_unreadable_state_file_is_off(tmp_path, monkeypatch):
    from core.camera_privacy import CameraPrivacy

    path = tmp_path / "broken.json"
    path.write_text("{not json")
    monkeypatch.setenv("SENSORDYME_DEVICE_STATE_PATH", str(path))
    assert CameraPrivacy.load().camera_only is False


def test_apply_refuses_a_non_boolean():
    from core.camera_privacy import CameraPrivacy

    with pytest.raises(TypeError):
        CameraPrivacy(persist=False).apply("yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(("raw", "ok"), [
    ({"camera_only": True, "id": "a1"}, True), ({"camera_only": False, "id": None}, True),
    ({"camera_only": "true"}, False), ({"camera_only": 1}, False), ({}, False), ("on", False),
    ({"camera_only": True, "id": 5}, False), ({"camera_only": True, "id": "x" * 65}, False),
])
def test_command_parsing_is_strict(raw, ok):
    from core.camera_privacy import parse_command

    assert (parse_command(raw) is not None) is ok


# ------------------------------------------------------------------ the uplink: signed to turn ON
KEY = "dk_" + "84" * 24
HEALTH = {"device_id": "SD-0A1B2C3D", "state": "ok"}


def _uplink(reply, **cloud):
    from core.uplink import CloudUplink

    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": KEY, "poll_interval_s": 0, **cloud})

    def post(url, json=None, data=None, headers=None, **_kwargs):
        from core import command_auth as a

        headers = dict(headers or {})
        nonce = (a.parse_request_header(headers["X-SD-Auth"]).nonce if "X-SD-Auth" in headers
                 else headers.get("X-SD-Nonce"))
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: reply(nonce))

    uplink._requests = types.SimpleNamespace(post=post)
    uplink.set_health(HEALTH)
    return uplink


def _signed(command):
    from core import command_auth as a

    return lambda nonce: {"command": command,
                          "command_auth": a.sign_command(a.DeviceKey(KEY), command, device="SD-0A1B2C3D",
                                                         request_nonce=nonce)}


ON = {"set_camera_mode": {"camera_only": True, "id": "c4m3r40n0001"}}
OFF = {"set_camera_mode": {"camera_only": False, "id": "c4m3r40ff001"}}


@pytest.mark.parametrize("mode", ["off", "permissive"])
def test_an_unsigned_command_cannot_turn_the_camera_on(mode):
    uplink = _uplink(lambda nonce: {"command": ON}, command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_camera_mode() is None
    assert uplink.command_auth_rejections == 1


@pytest.mark.parametrize("mode", ["off", "permissive"])
def test_an_unsigned_command_may_turn_it_off(mode):
    uplink = _uplink(lambda nonce: {"command": OFF}, command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_camera_mode() == {"camera_only": False, "id": "c4m3r40ff001"}


@pytest.mark.parametrize("mode", ["permissive", "required"])
def test_a_signed_command_turns_it_on(mode):
    uplink = _uplink(_signed(ON), command_auth=mode)
    uplink.sync_once()
    assert uplink.pop_pending_camera_mode() == {"camera_only": True, "id": "c4m3r40n0001"}
    assert uplink.pop_pending_camera_mode() is None


def test_a_signed_envelope_for_a_different_body_is_refused():
    from core import command_auth as a

    def reply(nonce):
        envelope = a.sign_command(a.DeviceKey(KEY), OFF, device="SD-0A1B2C3D", request_nonce=nonce)
        return {"command": ON, "command_auth": envelope}

    uplink = _uplink(reply, command_auth="permissive")
    uplink.sync_once()
    assert uplink.pop_pending_camera_mode() is None


# ------------------------------------------------------------------ the runtime applies + reports
def test_the_runtime_applies_reports_and_persists_the_owner_choice():
    from core.camera_privacy import CameraPrivacy
    from services.runtime import DeviceRuntime

    runtime = object.__new__(DeviceRuntime)
    runtime.camera_privacy = CameraPrivacy.load()
    runtime.dashboard = None
    published = []
    runtime._publish_monitoring = lambda: published.append(runtime._camera_report())
    assert runtime._camera_report()["camera_only"] is False
    runtime._apply_camera_mode({"camera_only": True, "id": "c4m3r40n0001"}, source="owner")
    assert published[-1]["camera_only"] is True and published[-1]["last_command_id"] == "c4m3r40n0001"
    assert published[-1]["remote_stream"] == "unavailable"  # no dashboard: no stream at all
    assert CameraPrivacy.load().camera_only is True  # what the next boot reads
    runtime._stopped = {"since": 0, "by": "app"}
    runtime.dashboard = types.SimpleNamespace(_local_camera_while_stopped=True)
    assert runtime._camera_report()["remote_stream"] == "camera_only"
