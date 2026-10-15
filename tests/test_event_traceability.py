"""An event's confidence and model identity, end to end.

Device Event -> CloudUplink payload -> /v1/ingest -> SQLite -> GET events ->
the JSON the iOS app decodes. Each hop could silently drop or coerce what the
previous one meant: None confidence becoming 0.0 ("certainly did not happen"),
or the model identity lost, so "which exact model produced this event?" had no
answer anywhere. These tests run the real device payload builder into the real
backend.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core.events import Event
from core.uplink import CloudUplink

REPO = Path(__file__).resolve().parents[1]
SHA = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
IDENTITY = {
    "module": "driver_attention",
    "detector": {"name": "damoyolo_tinynasL20_T.hef", "sha256": SHA, "backend": "hailo"},
}


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("SENSORDYME_MDNS", "0")
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)
    import time as _time

    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Trace unit", "TR-AC-E1", "dk_trace", _time.time()),
        )
    token = client.post("/v1/signup", json={"email": "trace@example.com", "password": "supersecret1"}).json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    device_id = client.post("/v1/devices/claim", json={"claim_code": "TR-AC-E1"}, headers=auth).json()["device_id"]
    return app_module, client, auth, device_id


def _payload(events_and_models):
    """What the device would POST: built by the real CloudUplink.push_event."""
    uplink = CloudUplink({"base_url": "http://unused.invalid", "device_key": "dk_trace"})
    for event, model in events_and_models:
        uplink.push_event(event, "soft", model=model)
    return {"events": list(uplink._queue)}


def _stored(client, auth, device_id):
    return {e["type"]: e for e in client.get(f"/v1/devices/{device_id}/events", headers=auth).json()["events"]}


def test_confidence_and_model_survive_every_hop(backend):
    _, client, auth, device_id = backend
    body = _payload([
        (Event.now("eyes_down", 2.6, 0.973, "driver_attention"), IDENTITY),
        (Event.now("frequent_glancing", 0.0, None, "driver_attention"), IDENTITY),
    ])
    assert body["events"][0]["model"] == IDENTITY
    assert body["events"][1]["confidence"] is None, "the device sends null, not 0"

    resp = client.post("/v1/ingest", json=body, headers={"X-Device-Key": "dk_trace"})
    assert resp.status_code == 200
    events = _stored(client, auth, device_id)
    assert events["eyes_down"]["confidence"] == pytest.approx(0.973)
    assert events["eyes_down"]["model"] == IDENTITY
    assert events["frequent_glancing"]["confidence"] is None, "stored as NULL, returned as null"


def test_history_from_before_model_tracking_still_reads(backend):
    """Older firmware sends no model and float confidences; nothing changes for it."""
    _, client, auth, device_id = backend
    client.post("/v1/ingest", json={"events": [{"type": "drowsiness", "confidence": 0.7}]},
                headers={"X-Device-Key": "dk_trace"})
    old = _stored(client, auth, device_id)["drowsiness"]
    assert old["model"] is None
    assert old["confidence"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("not a dict", None),
        ({"module": "m", "detector": {"name": "x.hef", "sha256": "not-a-hash", "backend": "hailo"}},
         {"module": "m", "detector": {"name": "x.hef", "sha256": None, "backend": "hailo"}}),
        ({"module": "m" * 500, "detector": None, "extra": "dropped"}, {"module": "m" * 64, "detector": None}),
    ],
)
def test_a_malformed_identity_is_normalised_never_fatal(backend, raw, expected):
    """Validated at the boundary: a bad identity must not cost the event."""
    _, client, auth, device_id = backend
    resp = client.post("/v1/ingest", json={"events": [{"type": "t", "confidence": 0.9, "model": raw}]},
                       headers={"X-Device-Key": "dk_trace"})
    assert resp.status_code == 200
    stored = _stored(client, auth, device_id)["t"]
    assert stored["model"] == expected
    assert stored["confidence"] == pytest.approx(0.9)


def test_an_out_of_range_confidence_is_clamped_not_stored_raw(backend):
    """Older firmware could send |pitch|/90-style values; the field's range holds."""
    _, client, auth, device_id = backend
    client.post("/v1/ingest", json={"events": [{"type": "x", "confidence": 3.5}]},
                headers={"X-Device-Key": "dk_trace"})
    assert _stored(client, auth, device_id)["x"]["confidence"] == 1.0


def test_the_explanation_names_the_model(backend):
    _, client, auth, device_id = backend
    client.post("/v1/ingest", json=_payload([(Event.now("eyes_down", 2.6, 0.97, "driver_attention"), IDENTITY)]),
                headers={"X-Device-Key": "dk_trace"})
    event_id = _stored(client, auth, device_id)["eyes_down"]["id"]
    facts = client.get(f"/v1/devices/{device_id}/events/{event_id}/explain", headers=auth).json()["facts"]
    assert {"label": "Model", "detail": f"damoyolo_tinynasL20_T.hef (sha256 {SHA[:12]})"} in [
        {"label": f["label"], "detail": f["detail"]} for f in facts
    ]
    assert any(f["label"] == "Confidence" for f in facts)


def test_the_ios_model_decodes_exactly_the_keys_the_backend_returns():
    """ios/Sources/EventModel.swift and cloud/app.py's _model_identity are two
    halves of one contract; a renamed key on either side decodes as nil in the
    app, silently. ios/Tests/EventModelTests.swift checks the Swift half runs."""
    swift = (REPO / "ios" / "Sources" / "EventModel.swift").read_text()
    detector_block = swift[swift.index("struct Detector") : swift.index("/// The vision module")]
    detector_keys = set(re.findall(r"let (\w+):", detector_block))
    top_keys = set(re.findall(r"^    let (\w+):", swift, re.MULTILINE))
    assert top_keys == set(IDENTITY)
    assert detector_keys == set(IDENTITY["detector"])
    models = (REPO / "ios" / "Sources" / "Models.swift").read_text()
    assert "case timestamp, type, confidence, signals, severity, source, model" in models
    assert "let confidence: Double?" in models, "null confidence must decode"


def test_the_uplink_records_the_identity_at_queue_time():
    """An event queued before a module switch keeps the model that produced it."""
    uplink = CloudUplink({"base_url": "http://unused.invalid", "device_key": "k"})
    identity = json.loads(json.dumps(IDENTITY))
    uplink.push_event(Event.now("eyes_down", 1.0, 0.9, "driver_attention"), "soft", model=identity)
    identity["detector"]["name"] = "something_else.hef"  # the runtime moves on
    assert uplink._queue[0]["model"]["detector"]["name"] == "damoyolo_tinynasL20_T.hef"


def test_each_device_reports_and_returns_its_own_hardware_id(backend):
    """The app shows a device its own live feed only if it can tell which camera
    that device is (ios FeedLink.swift). The device reports its SD- id in health;
    the backend keeps it — even once health goes stale — and returns it."""
    from core import device_health, identity

    app_module, client, auth, device_id = backend
    report = device_health.collect()
    assert identity.is_device_id(report["device_id"])
    client.post("/v1/ingest", json={"events": [], "health": report}, headers={"X-Device-Key": "dk_trace"})
    listed = client.get("/v1/devices", headers=auth).json()["devices"]
    assert listed[0]["hardware_id"] == report["device_id"]

    # A malformed id is not stored, and does not erase the good one.
    client.post("/v1/ingest", json={"events": [], "health": {**report, "device_id": "x'; drop"}},
                headers={"X-Device-Key": "dk_trace"})
    # Health from firmware that predates the field leaves it alone too.
    client.post("/v1/ingest", json={"events": [], "health": {"state": "ok"}}, headers={"X-Device-Key": "dk_trace"})
    with app_module.db() as conn:
        conn.execute("UPDATE devices SET last_seen = 0 WHERE id = ?", (device_id,))  # gone quiet: health withheld
    listed = client.get("/v1/devices", headers=auth).json()["devices"]
    assert listed[0]["health"] is None
    assert listed[0]["hardware_id"] == report["device_id"], "the id identifies the camera even when health is stale"


def test_a_re_paired_camera_belongs_to_one_device_row_only(backend):
    """The same Pi claimed again (a new device row) must not leave its id on the
    old row: the app would then show the live feed on the dead card too."""
    import time as _time

    app_module, client, auth, old_id = backend
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            ("Re-paired", "RE-PA-IR", "dk_repaired", _time.time()),
        )
    new_id = client.post("/v1/devices/claim", json={"claim_code": "RE-PA-IR"}, headers=auth).json()["device_id"]
    health = {"state": "ok", "device_id": "SD-3D248632"}
    client.post("/v1/ingest", json={"events": [], "health": health}, headers={"X-Device-Key": "dk_trace"})
    client.post("/v1/ingest", json={"events": [], "health": health}, headers={"X-Device-Key": "dk_repaired"})
    ids = {d["id"]: d["hardware_id"] for d in client.get("/v1/devices", headers=auth).json()["devices"]}
    assert ids == {old_id: None, new_id: "SD-3D248632"}
