"""M8.4 Sunday: scripts/hw_validation/check_live_status.py on recorded fixtures and a loopback stream.

The fixtures in tests/fixtures_m84_sunday/ were RECORDED from the virtual
device (sim/: the real runtime and backend on a Mac, scripted perception):
GET /api/health while ready and while stopped, and GET /v1/devices while
ready, stopped (a confirmed stop), resumed, and offline. They are virtual-
device data, so the checker must call them SIMULATED and never PASS.

The stream checks run against the real core/dashboard.py DashboardServer on
loopback, so the header parser reads exactly what the device writes.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures_m84_sunday"
sys.path.insert(0, str(REPO / "scripts" / "hw_validation"))

import check_live_status as cls  # noqa: E402

from core import state_model  # noqa: E402
from core.dashboard import DashboardServer  # noqa: E402
from core.frame_bus import FrameBus  # noqa: E402

TOKEN = "m84-sunday-token-0123456789"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


def _row(name: str) -> dict:
    return _load(name)["devices"][0]


def _args(**kw) -> Namespace:
    base = {"device": None, "backend": None, "device_id": None, "health_file": None, "devices_file": None,
            "token_env": "SENSORDYME_TOKEN", "bearer_env": "SENSORDYME_BEARER", "samples": 1, "interval": 0.0,
            "stream_seconds": 0.0, "camera_while_stopped": False, "out": None}
    base.update(kw)
    return Namespace(**base)


# ------------------------------------------------------------------ recorded device samples
@pytest.mark.parametrize("name", ["device_health_ready.json", "device_health_stopped.json"])
def test_recorded_device_samples_are_coherent_and_marked_simulated(name):
    result = cls.check_device(_load(name))
    assert result["violations"] == []
    assert result["summary"]["simulated"] is True
    assert result["summary"]["device_id"].startswith("SD-SIM-")


def test_the_stopped_sample_reports_stopped_paused_and_not_active():
    s = cls.check_device(_load("device_health_stopped.json"))["summary"]
    assert (s["monitoring_state"], s["protection"], s["monitoring_active"]) == ("stopped", "stopped", False)
    assert {"face", "objects", "attention"} <= set(s["paused"])


def test_a_torn_device_sample_is_a_violation():
    health = _load("device_health_ready.json")
    health["device"]["monitoring"]["state"] = "stopped"  # active and protecting kept: contradictory
    found = {v["invariant"] for v in cls.check_device(health)["violations"]}
    assert {"stopped_coherent", "active_matches_state"} <= found


def test_the_sim_id_alone_marks_data_simulated():
    health = _load("device_health_ready.json")
    health["device"].pop("simulated")
    health["device"]["monitoring"]["perception"]["simulated"] = False
    assert cls.check_device(health)["summary"]["simulated"] is True


# ------------------------------------------------------------------ recorded backend rows
@pytest.mark.parametrize(("name", "state"), [("devices_ready.json", state_model.READY),
                                        ("devices_stopped.json", state_model.STOPPED),
                                        ("devices_resumed.json", state_model.READY),
                                        ("devices_offline.json", state_model.OFFLINE)])
def test_recorded_backend_rows_have_no_violations_and_the_expected_user_state(name, state):
    row = _row(name)
    result = cls.check_row(row, cls._row_clock(row))
    assert result["violations"] == []
    assert result["summary"]["user_state"] == state
    assert result["summary"]["simulated"] is True


def test_a_stale_row_that_still_claims_ready_is_a_violation():
    row = copy.deepcopy(_row("devices_ready.json"))
    row["status"]["stale"] = True
    row["status"]["link_stale"] = True
    found = {v["invariant"] for v in cls.check_row(row)["violations"]}
    assert "stale_withholds" in found


def test_a_confirmed_stop_without_the_device_echo_is_a_violation():
    row = copy.deepcopy(_row("devices_stopped.json"))
    row["status"]["protection_command"]["applied_at"] = None
    found = {v["invariant"] for v in cls.check_row(row)["violations"]}
    assert "ack_before_success" in found


def test_recorded_transitions_follow_the_state_model():
    rows = [cls.check_row(_row(n)) for n in ("devices_ready.json", "devices_stopped.json",
                                            "devices_resumed.json", "devices_offline.json")]
    assert cls.transition_findings(rows) == []


def test_stopped_without_an_applied_stop_is_a_transition_finding():
    ready = cls.check_row(_row("devices_ready.json"))
    stopped_row = copy.deepcopy(_row("devices_stopped.json"))
    stopped_row["status"].pop("protection_command")
    findings = cls.transition_findings([ready, cls.check_row(stopped_row)])
    assert [f["invariant"] for f in findings] == ["state_transition"]


# ------------------------------------------------------------------ the CLI on recorded files
def test_cli_on_recorded_files_is_simulated_never_pass(tmp_path, capsys):
    out = tmp_path / "live.json"
    code = cls.main(["--health-file", str(FIX / "device_health_ready.json"),
                     "--devices-file", str(FIX / "devices_ready.json"), "--out", str(out)])
    assert code == cls.SIMULATED
    report = json.loads(out.read_text())
    assert report["verdict"] == "SIMULATED" and report["simulated_data"] is True
    assert "NOT hardware evidence" in capsys.readouterr().out


def test_cli_fails_on_a_contradiction(tmp_path):
    health = _load("device_health_ready.json")
    health["device"]["monitoring_active"] = True
    health["device"]["monitoring"]["state"] = "failed"
    path = tmp_path / "torn.json"
    path.write_text(json.dumps(health))
    assert cls.main(["--health-file", str(path)]) == cls.FAIL


def test_without_a_token_the_device_check_is_blocked_not_failed():
    report = cls.run(_args(device="http://127.0.0.1:9"), env={})
    assert report["verdict"] == "BLOCKED"
    assert "SENSORDYME_TOKEN" in report["results"][0]["blocked"]


def test_an_unreachable_device_is_blocked(tmp_path):
    report = cls.run(_args(device="http://127.0.0.1:9", stream_seconds=0.5), env={"SENSORDYME_TOKEN": TOKEN})
    assert report["verdict"] == "BLOCKED"
    assert TOKEN not in json.dumps(report)


def test_device_backend_disagreement_fails_only_when_it_persists(tmp_path):
    stopped = tmp_path / "rows.json"
    stopped.write_text((FIX / "devices_stopped.json").read_text())
    one = cls.run(_args(health_file=str(FIX / "device_health_ready.json"), devices_file=str(stopped)), env={})
    assert not any(r.get("violations") for r in one["results"] if r["source"] == "agreement")
    assert any("one-sample disagreement" in str(r.get("note")) for r in one["results"])
    two = cls.run(_args(health_file=str(FIX / "device_health_ready.json"), devices_file=str(stopped), samples=2),
                  env={}, sleep=lambda s: None)
    assert two["verdict"] == "FAIL"
    assert any(v["invariant"] == "device_backend_agree" for r in two["results"] for v in r.get("violations", []))


# ------------------------------------------------------------------ the live stream (loopback DashboardServer)
def _server(**kw) -> DashboardServer:
    server = DashboardServer(host="127.0.0.1", port=0, token=TOKEN, stream_fps=30, **kw)
    server.start()
    return server


class _Feed:
    """Annotated frames (the runtime drawing) or camera frames on a FrameBus."""

    def __init__(self, server: DashboardServer, bus: FrameBus | None = None, fps: float = 20.0) -> None:
        self.server, self.bus, self.interval = server, bus, 1.0 / fps
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        n = 0
        while not self.stop.is_set():
            n += 1
            frame = np.full((48, 64, 3), n % 256, dtype=np.uint8)
            if self.bus is None:
                self.server.push_frame(frame, captured_at=time.monotonic())
            else:
                self.bus.publish(frame)
            self.stop.wait(self.interval)

    def __enter__(self) -> "_Feed":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def _base(server: DashboardServer) -> str:
    return f"http://127.0.0.1:{server._port}"


def test_stream_headers_of_a_monitoring_device_pass():
    server = _server()
    try:
        with _Feed(server):
            probe = cls.read_stream_parts(_base(server), TOKEN, 2.5)
        result = cls.check_stream(probe, monitoring_running=True)
        assert all(c["ok"] for c in result["checks"]), result["checks"]
        assert result["values"]["sources"] == ["annotated"] and result["values"]["parts"] >= 5
        assert cls.http_status(_base(server) + "/stream") == 401
        stats = cls.fetch_stream_stats(_base(server), TOKEN)
        assert stats["connections_total"] >= 1
    finally:
        server.close()


def test_a_stalled_stream_while_monitoring_fails():
    server = _server()
    try:
        with _Feed(server):
            time.sleep(0.3)
        probe = cls.read_stream_parts(_base(server), TOKEN, 1.5)  # nothing new is pushed now
        result = cls.check_stream(probe, monitoring_running=True)
        assert not all(c["ok"] for c in result["checks"])
    finally:
        server.close()


def test_decision_c_flags_the_camera_fallback_while_stopped():
    """At c9029fa the dashboard streams the camera unannotated while stopped (camera_fallback
    defaults on). Mission 8.4 Decision C says it must not by default: the check FAILS on that
    behaviour, and passes only when the owner enabled the camera-only view."""
    bus = FrameBus()
    server = _server()
    server.attach_frame_source(bus)
    try:
        with _Feed(server, bus=bus):
            probe = cls.read_stream_parts(_base(server), TOKEN, 2.0)
        default = cls.check_stream(probe, monitoring_running=False)
        assert default["values"]["sources"] == ["camera"]
        failed = [c["check"] for c in default["checks"] if not c["ok"]]
        assert failed and failed[0].startswith("Decision C")
        enabled = cls.check_stream(probe, monitoring_running=False, camera_while_stopped=True)
        assert all(c["ok"] for c in enabled["checks"])
    finally:
        server.close()


def test_no_camera_frames_while_stopped_passes_decision_c():
    bus = FrameBus()
    server = _server(camera_fallback=False)
    server.attach_frame_source(bus)
    try:
        with _Feed(server, bus=bus):
            probe = cls.read_stream_parts(_base(server), TOKEN, 1.5)
        result = cls.check_stream(probe, monitoring_running=False)
        assert result["values"]["parts"] == 0
        assert all(c["ok"] for c in result["checks"]), result["checks"]
    finally:
        server.close()
