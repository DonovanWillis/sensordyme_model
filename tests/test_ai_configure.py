"""Tests for the rebuilt Configure-with-AI: validation, honesty, fallback, persistence."""

import importlib
import json
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud"):
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
    _, client, headers = backend
    resp = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "watch my baby in the crib"}, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["preset"] == "baby_monitoring"
    assert data["source"] == "fallback"
    assert data["rules"] and data["rules"][0]["detection"] == "unusual_stillness"
    assert 0.2 <= data["rules"][0]["confidence"] <= 0.95


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

    def fake_llm(prompt):
        return {
            "preset": "baby_monitoring", "variant": "", "explanation": "test",
            "settings": {}, "unsupported": [], "source": "claude",
            "rules": [
                {"detection": "unusual_stillness", "description": "still too long", "zone": "",
                 "persist_s": 120, "min_count": 1, "confidence": 0.6, "severity": "warning",
                 "cooldown_s": 60, "schedule": "", "notify": "push"},
                # The model hallucinating an unsupported capability:
                {"detection": "crying", "description": "alert on crying", "zone": "",
                 "persist_s": 2, "min_count": 1, "confidence": 0.6, "severity": "critical",
                 "cooldown_s": 60, "schedule": "", "notify": "push"},
                # And a detection id that doesn't exist at all:
                {"detection": "quantum_entanglement", "description": "??", "zone": "",
                 "persist_s": 2, "min_count": 1, "confidence": 0.6, "severity": "info",
                 "cooldown_s": 60, "schedule": "", "notify": "push"},
            ],
        }

    app_module._ai_configure_llm = fake_llm
    resp = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "watch the baby"}, headers=headers)
    data = resp.json()
    kept = [r["detection"] for r in data["rules"]]
    assert kept == ["unusual_stillness"]                 # only the real capability survived
    reasons = " ".join(u["reason"] for u in data["unsupported"])
    assert "crying" in reasons and "quantum_entanglement" in reasons


def test_validation_clamps_out_of_range_values(backend):
    app_module, client, headers = backend
    app_module._ai_configure_llm = lambda prompt: {
        "preset": "security_monitoring", "variant": "", "explanation": "t", "settings": {},
        "unsupported": [], "source": "claude",
        "rules": [{"detection": "zone_violation", "description": "zone", "zone": "entry",
                   "persist_s": -5, "min_count": 0, "confidence": 99, "severity": "LOUD",
                   "cooldown_s": 1, "schedule": "", "notify": "push"}],
    }
    data = client.post("/v1/ai/configure",
                       json={"device_id": 1, "prompt": "guard the door"}, headers=headers).json()
    rule = data["rules"][0]
    assert rule["persist_s"] >= 0.5
    assert rule["min_count"] >= 1
    assert rule["confidence"] <= 0.95
    assert rule["severity"] == "warning"   # bogus severity normalized
    assert rule["cooldown_s"] >= 10


def test_proposal_is_persisted_on_the_device(backend):
    app_module, client, headers = backend
    client.post("/v1/ai/configure",
                json={"device_id": 1, "prompt": "watch grandpa for falls"}, headers=headers)
    with app_module.db() as conn:
        stored = conn.execute("SELECT ai_rules FROM devices WHERE id = 1").fetchone()["ai_rules"]
    saved = json.loads(stored)
    assert saved["preset"] == "elderly_monitoring"
    assert saved["rules"][0]["detection"] == "fall"


def test_taxonomy_integrity():
    from core import taxonomy

    assert set(taxonomy.TAXONOMY) == {
        "driver_distraction", "driver_distraction_openpilot", "baby_monitoring",
        "elderly_monitoring", "security_monitoring",
    }
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
