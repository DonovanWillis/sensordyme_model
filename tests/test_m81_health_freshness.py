"""M8.1 health freshness: units (stamp, uplink, backend decision, status contract).

The defect: a device kept polling while re-sending a frozen health snapshot,
and the backend took each copy as current (staleness was judged from
`last_seen`, the time of the last poll). Samples are now stamped where they are
generated (boot id, generation, monotonic time), the uplink adds the sample's
own age at send time, and the backend judges freshness from receipt time minus
that age -- never from the device's wall clock. The end-to-end versions, on the
virtual device and the real backend, are tests/test_m81_health_freshness_sim.py.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import types

import pytest

from core import health_stamp as hs
from core.uplink import CloudUplink


# ---------------------------------------------------------------------- stamping
class _Clock:
    def __init__(self) -> None:
        self.mono, self.wall = 100.0, 1_700_000_000.0


def _stamper(clock: _Clock, boot_id: str = "boot-a") -> hs.HealthStamper:
    return hs.HealthStamper(boot_id=boot_id, monotonic=lambda: clock.mono, wall=lambda: clock.wall)


def test_each_sample_gets_the_next_generation_of_its_boot_and_its_generation_times():
    clock = _Clock()
    stamper = _stamper(clock)
    first = stamper.stamp({"camera": {}})["stamp"]
    clock.mono += 5
    clock.wall -= 3600  # an NTP step backwards changes nothing but the informational wall time
    second = stamper.stamp({"camera": {}})["stamp"]
    assert (first["boot_id"], first["generation"], first["revision"]) == ("boot-a", 1, 0)
    assert second["generation"] == 2 and second["generated_mono"] == 105.0
    assert second["generated_wall"] == 1_700_000_000.0 - 3600
    assert first["schema"] == second["schema"] == hs.SCHEMA


def test_every_process_start_is_a_new_boot_id():
    assert hs.HealthStamper().boot_id != hs.HealthStamper().boot_id


def test_the_send_age_is_monotonic_and_never_touches_the_held_sample():
    clock = _Clock()
    held = _stamper(clock).stamp({"camera": {}})
    sent = hs.with_send_age(held, clock.mono + 42.5)
    assert sent["stamp"]["age_s"] == 42.5
    assert "age_s" not in held["stamp"], "the held snapshot must stay as generated"
    assert hs.with_send_age(held, clock.mono - 10)["stamp"]["age_s"] == 0.0, "never negative"
    unstamped = {"camera": {}}
    assert hs.with_send_age(unstamped, 1.0) is unstamped and hs.with_send_age(None, 1.0) is None


def test_a_monitoring_refresh_is_a_revision_not_a_new_sample():
    held = _stamper(_Clock()).stamp({"camera": {}})
    refreshed = hs.revised(hs.revised(held))
    assert refreshed["stamp"]["generation"] == 1 and refreshed["stamp"]["revision"] == 2
    assert refreshed["stamp"]["generated_mono"] == held["stamp"]["generated_mono"]


@pytest.mark.parametrize("stamp", [
    "not an object",
    {"schema": "sd-health-v0", "boot_id": "b", "generation": 1},
    {"schema": hs.SCHEMA, "boot_id": "", "generation": 1},
    {"schema": hs.SCHEMA, "boot_id": "x" * 65, "generation": 1},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": 0},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": True},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": "7"},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": 1, "revision": -1},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": 1, "age_s": -5},
    {"schema": hs.SCHEMA, "boot_id": "b", "generation": 1, "age_s": "soon"},
])
def test_a_stamp_that_cannot_be_trusted_is_refused(stamp):
    with pytest.raises(hs.StampError):
        hs.parse({"stamp": stamp})


def test_parse_reads_a_good_stamp_and_older_firmware_has_none():
    stamp = hs.parse({"stamp": {"schema": hs.SCHEMA, "boot_id": "b", "generation": 3, "revision": 1,
                                "age_s": 1e12, "generated_wall": 5.0}})
    assert stamp is not None and stamp.order() == (3, 1) and stamp.generated_wall == 5.0
    assert stamp.age_s == pytest.approx(315_360_000.0), "an absurd age is clamped, still stale"
    assert hs.parse({"camera": {}}) is None and hs.parse(None) is None


# ---------------------------------------------------------------------- the uplink
class _Wire:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, url, json=None, **_kwargs):
        self.bodies.append(json)
        return types.SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                     json=lambda: {"command": None})


def _uplink(wire: _Wire) -> CloudUplink:
    uplink = CloudUplink({"base_url": "https://backend.example", "device_key": "dk_" + "x" * 32,
                          "command_auth": "off"})
    uplink._requests = wire
    return uplink


def test_a_frozen_sample_is_resent_with_a_growing_age_and_the_same_generation(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", lambda: clock.mono)
    wire = _Wire()
    uplink = _uplink(wire)
    uplink.set_health(_stamper(clock).stamp({"camera": {"state": "ok"}}))
    for _ in range(3):
        uplink.sync_once()
        clock.mono += 2.0
    stamps = [body["health"]["stamp"] for body in wire.bodies]
    assert [s["generation"] for s in stamps] == [1, 1, 1]
    assert [s["age_s"] for s in stamps] == [0.0, 2.0, 4.0]


def test_refresh_health_monitoring_bumps_the_revision_and_keeps_the_age(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", lambda: clock.mono)
    wire = _Wire()
    uplink = _uplink(wire)
    uplink.set_health(_stamper(clock).stamp({"camera": {"state": "ok"}, "monitoring": {"state": "ready"}}))
    clock.mono += 7.0
    uplink.refresh_health_monitoring({"state": "stopped"}, False, {"state": "stopped"})
    uplink.sync_once()
    sent = wire.bodies[-1]["health"]
    assert sent["monitoring"] == {"state": "stopped"}
    assert (sent["stamp"]["generation"], sent["stamp"]["revision"], sent["stamp"]["age_s"]) == (1, 1, 7.0)


def test_unstamped_health_from_an_older_runtime_is_sent_unchanged():
    wire = _Wire()
    uplink = _uplink(wire)
    uplink.set_health({"camera": {"state": "ok"}})
    uplink.sync_once()
    assert wire.bodies[-1]["health"] == {"camera": {"state": "ok"}}


def test_the_runtime_stamps_where_the_sample_is_generated():
    from services.runtime import DeviceRuntime

    runtime = DeviceRuntime.__new__(DeviceRuntime)  # no hardware: only the stamping path
    runtime._collect_component_health = lambda caller: {"camera": {"state": "ok"}}
    runtime.monitoring_report = lambda: types.SimpleNamespace(to_dict=lambda: {"state": "ready"}, monitoring=True)
    runtime.protection_state = lambda: {"state": "protecting"}
    runtime._config_report = lambda: {}
    runtime.uplink = None
    first = runtime._collect_health("uplink")["stamp"]
    second = runtime._collect_health("uplink")["stamp"]
    assert first["boot_id"] == second["boot_id"]
    assert (first["generation"], second["generation"]) == (1, 2)
    assert DeviceRuntime.__new__(DeviceRuntime)._stamper().boot_id != first["boot_id"], "a new process, a new boot"


# ---------------------------------------------------------------------- backend decision
@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    return importlib.import_module("cloud.app")


def _health(boot="boot-a", generation=1, revision=0, age=0.5, **extra) -> dict:
    return {"camera": {"state": "ok"}, **extra,
            "stamp": {"schema": hs.SCHEMA, "boot_id": boot, "generation": generation, "revision": revision,
                      "generated_wall": 123.0, "generated_mono": 9.0, "age_s": age}}


def _apply(app, stored: dict, health: dict, now: float) -> tuple[str, dict]:
    outcome, updates = app._health_decision(stored, health, now)
    merged = {**stored, **updates}
    return outcome, merged


def test_backend_decisions_new_repeat_revised_out_of_order_reboot_retired(app_module):
    app, now = app_module, 1000.0
    outcome, row = _apply(app, {}, _health(generation=5, age=2.0), now)
    assert outcome == "new" and row["health_observed_at"] == 998.0 and row["health_received_at"] == 1000.0

    outcome, row = _apply(app, row, _health(generation=5, age=12.0), now + 10)
    assert outcome == "repeat" and row["health_observed_at"] == 998.0 and row["health_repeats"] == 1
    assert row["health_received_at"] == 1000.0, "a re-sent copy does not move the receipt time"

    outcome, row2 = _apply(app, row, _health(generation=5, age=1.0), now + 20)
    assert outcome == "repeat" and row2["health_observed_at"] == 998.0, "a copy can only make it older"
    outcome, row2 = _apply(app, row, _health(generation=5, age=40.0), now + 20)
    assert row2["health_observed_at"] == 980.0, "a later copy that implies an older sample is believed"

    outcome, row = _apply(app, row, _health(generation=5, revision=1, age=14.0), now + 12)
    assert outcome == "revised" and row["health_observed_at"] == 998.0 and row["health_revision"] == 1
    assert "health" in app._health_decision(row, _health(generation=5, revision=2), now)[1], "content stored"

    for older in (_health(generation=4), _health(generation=5, revision=0)):
        outcome, unchanged = _apply(app, row, older, now + 13)
        assert outcome == "out_of_order" and unchanged["health_generation"] == 5
        assert "health" not in app._health_decision(row, older, now)[1], "older content is never stored"
        assert json.loads(unchanged["health_ignored"])["reason"] == "out_of_order"

    outcome, row = _apply(app, row, _health(boot="boot-b", generation=1), now + 30)
    assert outcome == "reboot" and row["health_boot"] == "boot-b" and row["health_generation"] == 1
    assert json.loads(row["health_boots"]) == ["boot-a"]

    outcome, row = _apply(app, row, _health(boot="boot-a", generation=99), now + 31)
    assert outcome == "retired_boot" and row["health_boot"] == "boot-b", "a pre-reboot report is not a reboot"


def test_unstamped_and_invalid_samples(app_module):
    app = app_module
    outcome, updates = app._health_decision({}, {"camera": {}}, 50.0)
    assert outcome == "unverified" and updates["health_boot"] is None and "health" in updates
    outcome, updates = app._health_decision({"health_boot": "boot-a", "health_generation": 3},
                                            {"stamp": {"schema": "nope"}}, 50.0)
    assert outcome == "invalid" and "health" not in updates
    assert json.loads(updates["health_ignored"])["reason"] == "invalid"


def test_the_retired_boot_list_is_bounded(app_module):
    app, row = app_module, {}
    for n in range(20):
        _, row = _apply(app, row, _health(boot=f"boot-{n}"), float(n))
    assert len(json.loads(row["health_boots"])) == app._HEALTH_BOOTS_KEPT


# ---------------------------------------------------------------------- the status contract
@pytest.fixture()
def backend(app_module, monkeypatch):
    from fastapi.testclient import TestClient

    app = app_module
    clock = {"now": float(int(time.time()))}
    monkeypatch.setattr(app, "time", types.SimpleNamespace(**{**vars(time), "time": lambda: clock["now"]}))
    client = TestClient(app.app)
    with app.db() as conn:
        conn.execute("INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
                     ("Car", "AA-BB-CC", "dk_test", clock["now"]))
    token = client.post("/v1/signup", json={"email": "t@t.com", "password": "password1"}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    client.post("/v1/devices/claim", json={"claim_code": "AA-BB-CC"}, headers=headers)
    return client, headers, clock


def _ready_health(**stamp) -> dict:
    return _health(**stamp, monitoring={"state": "ready", "reasons": [], "capabilities": {}},
                   monitoring_active=True, protection={"state": "protecting", "reasons": []})


def _post(client, health, events=()):
    resp = client.post("/v1/ingest", json={"events": list(events), "health": health},
                       headers={"X-Device-Key": "dk_test"})
    assert resp.status_code == 200, resp.text
    return resp


def _row(client, headers) -> dict:
    return client.get("/v1/devices", headers=headers).json()["devices"][0]


def test_a_frozen_sample_on_a_live_link_falls_to_unknown_and_the_poll_is_still_answered(backend):
    client, headers, clock = backend
    _post(client, _ready_health(generation=1, age=0.2))
    status = _row(client, headers)["status"]
    assert status["protection"] == "protecting" and status["health_freshness"]["state"] == "fresh"
    assert (status["stale"], status["link_stale"], status["health_stale"]) == (False, False, False)

    for n in range(20):  # 40 s of polls, every one re-sending generation 1
        clock["now"] += 2
        _post(client, _ready_health(generation=1, age=0.2 + 2 * (n + 1)),
              events=[{"type": "drowsiness", "event_key": f"k{n}", "severity": "loud"}])
    row = _row(client, headers)
    status = row["status"]
    assert row["online"] is True and status["link_stale"] is False, "the device is reachable"
    assert status["health_stale"] is True and status["stale"] is True
    assert status["monitoring_state"] == "unknown" and status["protection"] == "unknown"
    assert status["monitoring_active"] is None and status["capabilities"] is None
    assert status["reasons"] == ["the device is connected, but its monitoring status has not been updated for 40 s"]
    assert status["health_freshness"]["repeats"] == 20 and status["health_freshness"]["age_s"] == 40.2
    assert status["reported_at"] == clock["now"], "reported_at is link receipt time"
    assert status["health_freshness"]["observed_at"] == pytest.approx(clock["now"] - 40 - 0.2)
    assert row["health"] is None, "the frozen sample is withheld like a silent device's"
    events = client.get(f"/v1/devices/{row['id']}/events", headers=headers).json()["events"]
    assert len(events) == 20, "events are delivered whatever the health"

    _post(client, _ready_health(generation=2, age=0.3))
    status = _row(client, headers)["status"]
    assert status["protection"] == "protecting" and status["stale"] is False, "a new sample recovers at once"


def test_an_older_device_without_stamps_keeps_the_link_only_behaviour(backend):
    client, headers, clock = backend
    health = _ready_health()
    del health["stamp"]
    _post(client, health)
    clock["now"] += 25
    _post(client, health)
    clock["now"] += 25
    status = _row(client, headers)["status"]
    assert status["health_freshness"]["state"] == "unverified" and status["health_freshness"]["verified"] is False
    assert status["stale"] is False and status["health_stale"] is False and status["protection"] == "protecting"
    clock["now"] += 31
    assert _row(client, headers)["status"]["stale"] is True, "the link rule still applies"


def test_a_device_reporting_a_long_held_sample_is_stale_on_first_receipt(backend):
    """Delayed observation: a never-seen generation whose own age is past the threshold."""
    client, headers, _ = backend
    _post(client, _ready_health(generation=7, age=95.0))
    status = _row(client, headers)["status"]
    assert status["health_stale"] is True and status["link_stale"] is False
    assert status["health_freshness"]["age_s"] == 95.0 and status["protection"] == "unknown"


def test_an_invalid_stamp_never_refreshes_and_the_poll_is_not_refused(backend):
    client, headers, clock = backend
    _post(client, _ready_health(generation=1))
    clock["now"] += 31
    bad = _ready_health(generation=2)
    bad["stamp"]["schema"] = "forged"
    _post(client, bad)
    status = _row(client, headers)["status"]
    assert status["health_stale"] is True and status["link_stale"] is False
    assert status["health_freshness"]["last_ignored"]["reason"] == "invalid"
    assert status["health_freshness"]["generation"] == 1
