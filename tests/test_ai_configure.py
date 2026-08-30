"""Tests for the rebuilt Configure-with-AI: validation, honesty, fallback, persistence."""

import importlib
import json
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    client = TestClient(app_module.app)

    # Provision + claim a device so /v1/ai/configure authorizes.
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


def test_fallback_produces_valid_structured_proposal(backend):
    """"Valid" has to mean runnable. This used to assert
    `rules[0]["detection"] == "unusual_stillness"` -- an EVENT TYPE, not a
    detection label, so the rule passed validation, reached the device and
    could never fire. See tests/test_phase1_cloud_ai_fallback.py, which checks every
    fallback branch against the capability registry and builds each rule
    through the device's own ConfiguredRule."""
    _, client, headers = backend
    resp = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "watch my baby in the crib"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["preset"] == "baby_monitoring"
    assert data["source"] == "fallback"
    assert data["rules"]
    rule = data["rules"][0]
    assert (rule["target"], rule["predicate"]) == ("person", "absent")
    assert 0.2 <= rule["confidence"] <= 0.95
    assert data["unsupported"] == []


def test_unmatched_prompt_is_honest_not_fake(backend):
    _, client, headers = backend
    resp = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "alert me when the stock market drops"},
                       headers=headers)
    data = resp.json()
    assert data["rules"] == []           # nothing pretended
    assert data["unsupported"]           # the gap is stated
    assert "requires" in data["unsupported"][0]


def test_validation_moves_unsupported_detections_out_of_rules(backend):
    app_module, client, headers = backend

    def rule(**overrides):
        base = {"target": "person", "predicate": "present", "secondary": "", "description": "d",
                "zone": "", "persist_s": 2, "min_count": 1, "confidence": 0.6,
                "severity": "warning", "cooldown_s": 60, "schedule": "", "notify": "push"}
        base.update(overrides)
        return base

    def fake_llm(prompt):
        return {
            "preset": "", "variant": "", "explanation": "test",
            "settings": {}, "unsupported": [], "source": "claude",
            "rules": [
                rule(target="person", predicate="present", description="someone is here"),
                # A behaviour no installed model can recognise:
                rule(target="person", predicate="eating", description="alert on eating"),
                # An object the detector has never heard of:
                rule(target="quantum_entanglement", predicate="present", description="??"),
            ],
        }

    app_module._ai_configure_llm = fake_llm
    resp = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "watch the room"}, headers=headers)
    data = resp.json()
    kept = [r["target"] for r in data["rules"]]
    assert kept == ["person"]                            # only the executable rule survived
    blocked = " ".join(u["request"] + u["reason"] for u in data["unsupported"])
    assert "eating" in blocked and "quantum_entanglement" in blocked
    # Each rejection carries the specific gap, not a generic apology.
    assert all(u.get("requires") for u in data["unsupported"])


def test_validation_clamps_out_of_range_values(backend):
    app_module, client, headers = backend
    app_module._ai_configure_llm = lambda prompt: {
        "preset": "security_monitoring", "variant": "", "explanation": "t", "settings": {},
        "unsupported": [], "source": "claude",
        # A RUNNABLE rule with out-of-range numbers, which is what this test
        # is about. It used to name `detection: "zone_violation"` -- an event
        # type no detector emits as a label -- so it was really asserting
        # that clamping happens on a rule the device would silently never
        # run. Validation now rejects that shape outright (see
        # tests/test_phase1_cloud_ai_fallback.py), which would leave nothing here
        # to clamp.
        "rules": [{"target": "person", "predicate": "in_zone", "secondary": "",
                   "description": "someone in the entry", "zone": "entry",
                   "persist_s": -5, "min_count": 0, "confidence": 99, "severity": "LOUD",
                   "cooldown_s": 1, "schedule": "", "notify": "push"}],
    }
    data = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "guard the door"}, headers=headers).json()
    assert data["rules"], data["unsupported"]
    rule = data["rules"][0]
    assert rule["persist_s"] >= 0.5
    assert rule["min_count"] >= 1
    assert rule["confidence"] <= 0.95
    assert rule["severity"] == "warning"   # bogus severity normalized
    assert rule["cooldown_s"] >= 10


def test_proposal_is_persisted_on_the_device_as_a_draft(backend):
    """The validated proposal is kept, so Activate has something to apply —
    but as a DRAFT, not as the running configuration.

    This endpoint is the preview: AIConfigureView calls it from `configure()`,
    before the person has read the result. It used to write straight into
    `ai_rules`, which is the column the ingest handler delivers from, so on a
    device already in AI Custom the next 2-second poll shipped and committed
    rules nobody had accepted. `ai_rules` now changes only when set_preset
    (Activate) promotes the draft. See tests/test_phase1_cloud_ai_draft.py.
    """
    app_module, client, headers = backend
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    with app_module.db() as conn:
        row = conn.execute(
            "SELECT ai_rules, ai_rules_draft FROM devices WHERE id = 1"
        ).fetchone()
    assert row["ai_rules"] is None, "a preview must never become the running configuration"
    saved = json.loads(row["ai_rules_draft"])
    assert saved["preset"] == "elderly_monitoring"
    assert (saved["rules"][0]["target"], saved["rules"][0]["predicate"]) == ("on_floor", "present")

    # Activate promotes it, and that is the only thing that does.
    client.post("/v1/devices/1/preset", json={"preset": ""}, headers=headers)
    with app_module.db() as conn:
        row = conn.execute(
            "SELECT ai_rules, ai_rules_draft FROM devices WHERE id = 1"
        ).fetchone()
    assert json.loads(row["ai_rules"]) == saved
    assert row["ai_rules_draft"] is None


def test_taxonomy_integrity():
    from core import taxonomy

    # Derived from the presets, not hardcoded. The capability sheet describes
    # the modes on offer, so the two cannot be allowed to drift: a mode with no
    # sheet has no honest answer about what it can do, and a sheet with no mode
    # documents something nobody can select.
    from main import load_presets

    assert set(taxonomy.TAXONOMY) == set(load_presets())
    for mode, detections in taxonomy.TAXONOMY.items():
        ids = [d.id for d in detections]
        assert len(ids) == len(set(ids)), f"duplicate detection ids in {mode}"
        for det in detections:
            assert det.support in ("supported", "partial", "planned", "unsupported")
            if det.support in ("planned", "unsupported"):
                assert det.requires, f"{mode}.{det.id} must say what it requires"
            if det.support in ("supported", "partial"):
                assert det.how, f"{mode}.{det.id} must say how it works"
    sheet = taxonomy.summary_for_llm()
    assert "baby_monitoring" in sheet and "UNSUPPORTED" in sheet
