"""M8.3 truth invariants on the backend: GET /v1/devices never emits contradictory fields.

docs/STATE_MODEL.md. The checker is sim/invariants.py (`status_violations`,
`report_violations`), the same one the InvariantMonitor runs at every uplink
cycle of a virtual device. Here it runs over GENERATED device reports, sent
through the real ingest endpoint -- coherent ones a current device sends, and
contradictory ones an older device, a torn sample or a damaged report can --
on a controllable backend clock.
"""

from __future__ import annotations

import importlib
import json
import random
import secrets
import sys
import time

import pytest

from core import state_model
from sim.invariants import report_violations, status_violations

NAMES = ("camera", "inference", "face", "objects", "hands", "pose", "attention", "audible_alert", "alert_output")
COMPONENTS = ("object_detector", "face_landmarks", "hand_landmarks", "pose")


class _Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def time(self) -> float:
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    with app_module.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Car", "AA-BB-CC", "dk_test", time.time()))
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    clock = _Clock()
    monkeypatch.setattr(app_module, "time", clock)
    return client, headers, app_module, clock


def _row(client, headers) -> dict:
    devices = client.get("/v1/devices", headers=headers).json()["devices"]
    assert len(devices) == 1
    return devices[0]


def _ingest(client, health=None, monitoring=None) -> dict:
    resp = client.post("/v1/ingest", json={"events": [], "active_preset": "driver_safety", "health": health,
                                           "monitoring": monitoring}, headers={"X-Device-Key": "dk_test"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stamp(boot: str, generation: int, age: float) -> dict:
    return {"schema": "sd-health-v1", "boot_id": boot, "generation": generation, "revision": 0,
            "generated_wall": 1.0, "generated_mono": 1.0, "age_s": age}


def _sample(*, state="ready", protection="protecting", caps=None, components=None, torn=None,
            boot=None, age=1.0, command_id=None, active=None) -> dict:
    """A health sample in the shape services/runtime.py _collect_health sends."""
    caps = {"camera": "available", "inference": "available", "face": "available"} if caps is None else caps
    monitoring = {
        "state": state, "monitoring": state in ("ready", "degraded") if active is None else active,
        "reasons": [] if state == "ready" else [f"{state}: why"], "role": "driver_safety",
        "mode": "standard", "custom_rules": [],
        "capabilities": {n: {"state": s, "reason": None if s == "available" else f"{n} {s}"} for n, s in caps.items()},
    }
    if components is not None:
        monitoring["perception"] = {"schema": "sd-perception/1", "simulated": True, "nodes": [],
                                    "components": {n: {"state": s, "reason": f"{n} {s}"} for n, s in components.items()}}
    block = {"state": protection, "role": "driver_safety", "reasons": [] if protection == "protecting" else ["why"],
             "monitoring_state": torn or state, "last_command_id": command_id}
    health = {"monitoring": monitoring, "monitoring_active": monitoring["monitoring"], "protection": block,
              "camera": {"state": "ok", "detail": "", "metrics": {}}}
    health["stamp"] = _stamp(boot or secrets.token_hex(8), 1, age)
    return health


def _pushed(protection="protecting", state="ready", command_id=None) -> dict:
    return {"instruction": "", "rule_count": 0, "active": False, "mode": "standard",
            "role": {"name": "driver_safety", "title": "Driver safety", "allowed_presets": ["driver_safety"]},
            "protection": {"state": protection, "role": "driver_safety", "reasons": [],
                           "last_command_id": command_id},
            "last_rejection": None, "state": state}


# ---------------------------------------------------------------------- the checker fires
def test_the_checker_names_each_contradiction():
    """A checker that never fires proves nothing."""
    def row(**status) -> dict:
        base = {"monitoring_state": "ready", "protection": "protecting", "monitoring_active": True, "stale": False,
                "link_stale": False, "health_stale": False, "capabilities": None, "perception": None,
                "health_freshness": {"state": "fresh", "verified": True, "age_s": 1.0, "threshold_s": 30}}
        base.update(status)
        return {"online": True, "health": {}, "status": base}

    def caught(r) -> set[str]:
        return {name for name, _ in status_violations(r)}

    assert caught(row()) == set()
    assert "protecting_requires_monitoring" in caught(row(monitoring_state="failed", monitoring_active=False))
    assert "stopped_coherent" in caught(row(protection="stopped"))
    assert "paused_not_analysing" in caught(row(
        monitoring_state="stopped", protection="stopped", monitoring_active=False,
        capabilities={"available": ["face", "camera"], "impaired": [], "starting": [], "paused": []}))
    assert "pipeline_down_not_detecting" in caught(row(
        capabilities={"available": ["face"], "impaired": [{"name": "inference", "reason": "dead"}],
                      "starting": [], "paused": []}))
    assert "model_missing_not_available" in caught(row(
        capabilities={"available": ["objects"], "impaired": [], "starting": [], "paused": []},
        perception={"components": {"object_detector": {"state": "failed"}}}))
    assert "stale_definition" in caught(row(
        health_freshness={"state": "stale", "verified": True, "age_s": 95.0, "threshold_s": 30}))
    assert "stale_withholds" in caught(row(stale=True, link_stale=True))
    assert "active_matches_state" in caught(row(monitoring_state="starting", protection="not_protecting"))
    assert "ack_before_success" in caught(row(protection_command={
        "action": "stop", "outcome": "confirmed", "applied_at": None, "reported_state": None}))
    assert "capability_lists_disjoint" in caught(row(capabilities={
        "available": ["face"], "impaired": [{"name": "face", "reason": "x"}], "starting": [], "paused": []}))

    sample = _sample(state="failed", protection="protecting", torn="ready")
    assert {name for name, _ in report_violations(sample)} >= {"protecting_requires_monitoring"}


# ---------------------------------------------------------------------- each correction
def test_a_torn_sample_never_reaches_the_app_as_protecting(backend):
    """monitoring and protection evaluated a moment apart: failed beside protecting."""
    client, headers, _, _ = backend
    _ingest(client, _sample(state="failed", protection="protecting", torn="ready",
                            caps={"camera": "available", "inference": "available", "face": "impaired"}), _pushed())
    row = _row(client, headers)
    status = row["status"]
    assert status["monitoring_state"] == "failed" and status["protection"] == "not_protecting"
    assert status["monitoring_active"] is False
    assert any("protecting while failed" in c for c in status["corrections"])
    assert status_violations(row) == []


def test_a_stopped_device_from_before_m82_shows_nothing_analysing(backend):
    client, headers, _, _ = backend
    _ingest(client, _sample(state="stopped", protection="stopped",
                            caps={"camera": "available", "inference": "available", "face": "available"},
                            components={"face_landmarks": "ok"}), _pushed("stopped", "stopped"))
    row = _row(client, headers)
    caps = row["status"]["capabilities"]
    assert caps["available"] == ["camera"] and sorted(caps["paused"]) == ["face", "inference"]
    assert row["status"]["perception"]["components"]["face_landmarks"]["state"] == "paused"
    assert status_violations(row) == []


def test_ready_with_the_camera_impaired_is_failed_and_nothing_detects(backend):
    client, headers, _, _ = backend
    _ingest(client, _sample(state="ready", caps={"camera": "impaired", "inference": "available",
                                                  "face": "available", "objects": "available"}), _pushed())
    row = _row(client, headers)
    status = row["status"]
    assert status["monitoring_state"] == "failed" and status["protection"] == "not_protecting"
    assert status["capabilities"]["available"] == ["inference"]  # the loop is alive; it has no frames
    assert {i["name"] for i in status["capabilities"]["impaired"]} == {"camera", "face", "objects"}
    assert any("camera" in r for r in status["reasons"])
    assert status_violations(row) == []


def test_a_failed_model_is_never_available_and_a_missing_one_is_unsupported(backend):
    client, headers, app_module, _ = backend
    _ingest(client, _sample(state="degraded", caps={"camera": "available", "inference": "available",
                                                     "face": "available", "objects": "available"},
                            components={"object_detector": "failed", "face_landmarks": "unavailable"}), _pushed())
    row = _row(client, headers)
    caps = row["status"]["capabilities"]
    assert "objects" not in caps["available"] and "face" not in caps["available"]
    assert any(i["name"] == "objects" for i in caps["impaired"])
    states = app_module.capability_states(row["status"])
    assert states["face"]["state"] == "unsupported" and states["objects"]["state"] == "impaired"
    assert status_violations(row) == []


def test_online_with_old_health_is_stale_and_claims_nothing(backend):
    client, headers, _, clock = backend
    boot = secrets.token_hex(8)
    _ingest(client, _sample(boot=boot, age=1.0), _pushed())
    for _ in range(20):  # the device keeps polling every 2 s, re-sending one frozen sample
        clock.now += 2.0
        frozen = _sample(boot=boot, age=1.0)
        frozen["stamp"]["age_s"] = 1.0 + (clock.now - 1_800_000_000.0)
        _ingest(client, frozen, _pushed())
    row = _row(client, headers)
    assert row["online"] is True and row["status"]["stale"] is True and row["health"] is None
    assert row["status"]["monitoring_state"] == "unknown" and row["status"]["capabilities"] is None
    assert state_model.derive(row, clock.now).state == state_model.STALE
    assert status_violations(row) == []


# ---------------------------------------------------------------------- property test
def _random_report(rng: random.Random, coherent: bool) -> tuple[dict, dict]:
    state = rng.choice(["starting", "ready", "degraded", "stopped", "failed", "bogus"])
    if coherent:
        protection = {"stopped": "stopped", "ready": "protecting", "degraded": "protecting"}.get(state, "not_protecting")
    else:
        protection = rng.choice(["protecting", "not_protecting", "stopped", "bogus"])
    caps = {n: rng.choice(["available", "available", "impaired", "starting", "paused", "bogus"])
            for n in NAMES if rng.random() < 0.8}
    if coherent and state == "stopped":
        caps = {n: ("paused" if s == "available" and n not in ("camera", "audible_alert", "alert_output") else s)
                for n, s in caps.items()}
    if coherent and state in ("ready", "degraded"):
        caps.update(camera="available", inference="available")
    components = ({c: rng.choice(["ok", "idle", "stale", "failed", "unavailable", "paused", "bogus"])
                   for c in COMPONENTS if rng.random() < 0.6} if rng.random() < 0.7 else None)
    torn = None if coherent else rng.choice([None, "ready", "failed", "stopped"])
    sample = _sample(state=state, protection=protection, caps=caps, components=components, torn=torn,
                     age=rng.choice([0.5, 5.0, 29.0, 31.0, 120.0]),
                     command_id=rng.choice([None, "c0"]), active=None if coherent else rng.choice([True, False]))
    if rng.random() < 0.1:
        sample.pop("stamp")  # firmware before M8.1
    pushed = _pushed(rng.choice([protection, "protecting", "stopped"]) if not coherent else protection, state)
    return sample, pushed


@pytest.mark.parametrize("coherent", [True, False], ids=["coherent", "contradictory"])
def test_generated_reports_never_yield_a_contradictory_status(backend, coherent):
    client, headers, _, clock = backend
    rng = random.Random(8300 + coherent)
    seen_states: set[str] = set()
    corrected = 0
    for case in range(400):
        sample, pushed = _random_report(rng, coherent)
        _ingest(client, sample, pushed if rng.random() < 0.9 else None)
        clock.now += rng.choice([0.0, 1.0, 10.0, 29.0, 31.0, 45.0])
        row = _row(client, headers)
        problems = status_violations(row)
        assert problems == [], (case, problems, json.dumps(sample)[:1500], json.dumps(row["status"])[:1500])
        derived = state_model.derive(row, clock.now)
        if derived.protecting:
            assert derived.state in state_model.WATCHING and not row["status"]["stale"]
        seen_states.add(derived.state)
        corrected += bool(row["status"].get("corrections"))
        if rng.random() < 0.2:  # the link goes quiet for a while
            clock.now += 31.0
            quiet = _row(client, headers)
            assert status_violations(quiet) == []
            assert state_model.derive(quiet, clock.now).state == state_model.OFFLINE
    # The generator reached every live state (and the contradictory half needed corrections).
    assert {state_model.READY, state_model.DEGRADED, state_model.FAULTED, state_model.STOPPED,
            state_model.ACTIVATION_PENDING, state_model.STALE} <= seen_states | {state_model.OFFLINE}, seen_states
    if not coherent:
        assert corrected > 50, corrected
