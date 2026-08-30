"""Zones drawn in the app must reach the device that enforces them.

Rules about places — "the kitchen", "the pool area", "this room" — are
meaningless without a shape someone drew and named. Before this path existed the
AI told users to "draw the zone in the app" and there was no way to draw one, so
every area rule was accepted, stored, and permanently inert.
"""

import importlib
import json
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

DEVICE = {"X-Device-Key": "dk_test"}
SQUARE = [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    import time as _time
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Test unit", "AA-BB-CC", "dk_test", _time.time()),
        )
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return app_module, client, headers


def poll(client, zones_version=""):
    resp = client.post("/v1/ingest", json={"events": [], "zones_version": zones_version}, headers=DEVICE)
    assert resp.status_code == 200
    return resp.json()["command"]


def zone(name="pool", kind="restricted", polygon=None):
    return {"name": name, "kind": kind, "polygon": polygon or SQUARE}


# --------------------------------------------------------------------- storage
def test_zones_round_trip(backend):
    _, client, headers = backend
    resp = client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)
    assert resp.status_code == 200

    stored = client.get("/v1/devices/1/zones", headers=headers).json()["zones"]
    assert len(stored) == 1
    assert stored[0]["name"] == "pool"


def test_a_device_with_no_zones_returns_an_empty_list(backend):
    _, client, headers = backend
    assert client.get("/v1/devices/1/zones", headers=headers).json()["zones"] == []


def test_another_user_cannot_read_or_set_zones(backend):
    _, client, headers = backend
    client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)

    other = client.post("/v1/signup", json={"email": "x@x.com", "password": "password2"}).json()["token"]
    stranger = {"Authorization": f"Bearer {other}"}
    assert client.get("/v1/devices/1/zones", headers=stranger).status_code == 404
    assert client.post("/v1/devices/1/zones", json={"zones": []}, headers=stranger).status_code == 404


# --------------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", [
    {"name": "", "kind": "restricted", "polygon": SQUARE},           # unnamed
    {"name": "x", "kind": "restricted", "polygon": [[0.1, 0.1]]},    # not an area
    {"name": "x", "kind": "nonsense", "polygon": SQUARE},            # unknown kind
    {"name": "x", "kind": "restricted", "polygon": [[5, 5], [6, 6], [7, 7]]},  # not normalised
    {"name": "x", "kind": "restricted", "polygon": "not a polygon"},
])
def test_malformed_zones_are_dropped_not_repaired(backend, bad):
    """A repaired zone would watch the wrong part of a room while looking fine."""
    _, client, headers = backend
    resp = client.post("/v1/devices/1/zones", json={"zones": [bad]}, headers=headers)
    assert resp.json()["zones"] == []
    assert resp.json()["rejected"] == 1


def test_good_zones_survive_alongside_bad_ones(backend):
    _, client, headers = backend
    resp = client.post(
        "/v1/devices/1/zones",
        json={"zones": [zone(name="pool"), {"name": "", "polygon": SQUARE}]},
        headers=headers,
    )
    assert [z["name"] for z in resp.json()["zones"]] == ["pool"]
    assert resp.json()["rejected"] == 1


# --------------------------------------------------------------------- delivery
def test_zones_reach_the_device(backend):
    _, client, headers = backend
    client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)

    command = poll(client)
    assert command is not None
    assert command["set_zones"][0]["name"] == "pool"
    assert command["zones_version"]


def test_zones_are_not_redelivered_once_acknowledged(backend):
    _, client, headers = backend
    client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)
    version = poll(client)["zones_version"]
    assert poll(client, zones_version=version) is None


def test_editing_a_zone_redelivers_it(backend):
    _, client, headers = backend
    client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)
    version = poll(client)["zones_version"]

    moved = zone(polygon=[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]])
    client.post("/v1/devices/1/zones", json={"zones": [moved]}, headers=headers)

    command = poll(client, zones_version=version)
    assert command["zones_version"] != version
    assert command["set_zones"][0]["polygon"][0] == [0.0, 0.0]


def test_zones_and_rules_ride_the_same_response(backend):
    """A rule naming an area is inert without it, so neither may crowd out the other."""
    app_module, client, headers = backend
    client.post("/v1/devices/1/zones", json={"zones": [zone()]}, headers=headers)
    with app_module.db() as conn:
        # custom_monitoring_wanted=1: this test is about zones+rules riding
        # together once a rule set is applied, not about whether it is
        # currently requested (that gating has its own tests) — set directly
        # since /v1/devices/{id}/preset is what normally sets it.
        conn.execute("UPDATE devices SET ai_rules = ?, custom_monitoring_wanted = 1 WHERE id = 1", (json.dumps({
            "rules": [{"target": "person", "predicate": "in_zone", "zone": "pool",
                       "description": "someone in the pool area", "secondary": "",
                       "persist_s": 3, "min_count": 1, "confidence": 0.6,
                       "severity": "critical", "cooldown_s": 60, "schedule": "", "notify": "push"}]
        }),))

    command = poll(client)
    assert command["set_zones"][0]["name"] == "pool"
    assert command["set_rules"][0]["zone"] == "pool"
