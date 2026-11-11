"""M6.5 §8: product capability tiers bound to evidence (core/model_registry.py).

Pins today's honest tiers, that every inconsistent claim is refused, that the
API and the app can never present a research or experiment-only capability as
production-validated, and that no competition-only data is cited anywhere.
"""

from __future__ import annotations

import copy
import importlib
import re
import sys
from pathlib import Path

import pytest
import yaml

from core import model_registry as mr

REPO = Path(__file__).resolve().parent.parent
HEF_SHA = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
ONNX_SHA = "02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657"
FORBIDDEN = re.compile(r"state[ _-]?farm|kaggle|competition", re.IGNORECASE)

# What the evidence supports today. A change here is a claim: it needs a binding.
TODAY = {
    "person_detection": "image_level_benchmark",
    "phone_detection": "image_level_benchmark",
    "drowsiness": "synthetic_tests",
    "gaze_distraction": "synthetic_tests",
    "hand_object_association": "synthetic_tests",
    "fall_detection": "synthetic_tests",
    "hand_box_detection": "exists_in_code",
}


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load(mr.DEFAULT_PATH.read_text())


def violations(data) -> list[str]:
    return mr.registry_violations(mr.parse_registry(data))


def binding(**kw):
    base = {"id": "t", "capability": "phone_detection", "tier": "image_level_benchmark",
            "model": "damo_yolo_tinynasl20_t_coco80", "model_sha256": ONNX_SHA, "decoder": "legacy",
            "test_id": "t", "dataset_manifest": "b878e8429d221f3f", "hardware_target": "oryn_gpu_float",
            "status": "current", "date": "2026-11-11", "metrics": {"phone_ap50": 0.7}}
    return {**base, **kw}


def with_changes(doc, capability=None, cap_changes=None, add=(), card_changes=None):
    data = copy.deepcopy(doc)
    if capability:
        data["product_capabilities"][capability].update(cap_changes or {})
    data["evidence_bindings"] += list(add)
    for card in data["models"]:
        if card_changes and card["id"] in card_changes:
            card_changes[card["id"]](card)
    return data


# ---------------------------------------------------------------------- today
def test_the_registry_is_consistent_and_says_what_the_evidence_supports():
    registry = mr.load_registry()
    assert {c.id: c.claimed_tier for c in registry.capabilities} == TODAY
    out = mr.capability_tiers(registry)
    assert out["tiers"] == list(mr.TIERS)
    for entry in out["capabilities"]:
        assert entry["production_validated"] is False and entry["device_verified"] is False
        if entry["tier"] != "exists_in_code":
            assert entry["evidence"], f"{entry['capability']}: a claim with no supporting binding"


def test_experiment_only_and_pending_evidence_supports_nothing():
    registry = mr.load_registry()
    out = {e["capability"]: e for e in mr.capability_tiers(registry)["capabilities"]}
    assert out["phone_detection"]["experiment_evidence"] == ["phone_detection.oi7_val.reference"]
    assert "phone_detection.oi7_val.reference" not in out["phone_detection"]["evidence"]
    hand = out["hand_box_detection"]
    assert hand["evidence"] == [] and hand["model_deployment"] == "candidate"
    assert set(hand["experiment_evidence"]) == {"hand_box_detection.exp08a_rerun", "hand_box_detection.exp08_full60"}


def test_the_60_epoch_run_is_recorded_and_stays_experiment_only(doc):
    slot = next(b for b in doc["evidence_bindings"] if b["id"] == "hand_box_detection.exp08_full60")
    assert slot["status"] == "experiment_only" and slot["dataset_manifest"] == "b878e8429d221f3f"
    assert len(slot["model_sha256"]) == 64 and slot["metrics"]["hand_ap50"] > 0
    assert violations(doc) == []
    data = copy.deepcopy(doc)
    next(b for b in data["evidence_bindings"] if b["id"] == slot["id"])["status"] = "current"
    assert any("candidate model" in v for v in violations(data)), "a candidate checkpoint can never be current"


# ---------------------------------------------------------------------- refusals
@pytest.mark.parametrize(("capability", "tier", "add", "expect"), [
    ("drowsiness", "image_level_benchmark", (), "without a current evidence binding"),
    ("phone_detection", "video_evidence", (), "without a current evidence binding"),
    # A float benchmark on the GPU is not a device: wrong hardware target.
    ("phone_detection", "device_verified", (binding(tier="device_verified", metrics={"events": 1}),),
     "on hailo8l_int8 with a pinned artifact hash"),
    # The right target, but not the hash of the deployed .hef.
    ("phone_detection", "device_verified",
     (binding(tier="device_verified", hardware_target="hailo8l_int8", model_sha256="b" * 64),),
     "pinned artifact hash"),
    ("hand_box_detection", "image_level_benchmark", (), "without a current evidence binding"),
])
def test_a_claim_beyond_the_evidence_is_refused(doc, capability, tier, add, expect):
    problems = violations(with_changes(doc, capability, {"claimed_tier": tier}, add))
    assert any(expect in p for p in problems), problems


def test_device_verified_needs_the_deployed_target_and_artifact(doc):
    ok = binding(id="dv", tier="device_verified", hardware_target="hailo8l_int8", model_sha256=HEF_SHA)
    assert violations(with_changes(doc, "phone_detection", {"claimed_tier": "device_verified"}, [ok])) == []


def test_production_needs_device_verified_a_validated_card_and_an_approval(doc):
    dv = binding(id="dv", tier="device_verified", hardware_target="hailo8l_int8", model_sha256=HEF_SHA)
    claim = {"claimed_tier": "production_permitted"}
    problems = violations(with_changes(doc, "phone_detection", claim, [dv]))
    assert any("validated_on_device" in p for p in problems)
    assert any("recorded approval" in p for p in problems)
    problems = violations(with_changes(doc, "phone_detection", claim))
    assert any("reaching device_verified" in p for p in problems)
    approved = {**claim, "approval": {"by": "owner", "date": "2026-12-01", "reference": "ticket"}}
    assert any("validated_on_device" in p for p in violations(with_changes(doc, "phone_detection", approved, [dv])))


def test_an_approval_without_a_production_claim_is_refused(doc):
    data = with_changes(doc, "drowsiness", {"approval": {"by": "x", "date": "2026-12-01", "reference": "y"}})
    assert any("only recorded with a production_permitted" in p for p in violations(data))


def test_the_reference_decoder_cannot_be_device_verified_without_int8_evidence(doc):
    float_dv = binding(id="r", decoder="reference", tier="device_verified", hardware_target="oryn_gpu_float",
                       model_sha256=HEF_SHA)
    data = with_changes(doc, "phone_detection", {"decoder": "reference", "claimed_tier": "device_verified"},
                        [float_dv])
    problems = violations(data)
    assert any("unvalidated on the INT8 .hef" in p for p in problems), problems


def test_reference_decoder_evidence_does_not_lift_the_legacy_capability(doc):
    current_ref = binding(id="r", decoder="reference", tier="video_evidence", metrics={"x": 1})
    problems = violations(with_changes(doc, "phone_detection", {"claimed_tier": "video_evidence"}, [current_ref]))
    assert any("without a current evidence binding" in p for p in problems)


@pytest.mark.parametrize(("changes", "expect"), [
    ({"tier": "production_permitted"}, "approval on the capability"),
    ({"model_sha256": None}, "must name the sha256"),
    ({"dataset_manifest": "synthetic:x"}, "needs real data"),
    ({"metrics": {}}, "no metrics"),
    ({"hardware_target": "laptop"}, "hardware_target must be"),
    ({"decoder": "fast"}, "decoder must be"),
    ({"status": "maybe"}, "status must be"),
    ({"capability": "teleportation"}, "not in product_capabilities"),
    ({"model": "nope"}, "is not registered"),
    ({"date": "soon"}, "date must be"),
    ({"dataset_manifest": "statefarm-distracted-driver"}, "competition-only"),
    ({"test_id": "Kaggle State Farm leaderboard"}, "competition-only"),
])
def test_a_malformed_binding_is_refused(doc, changes, expect):
    problems = violations(with_changes(doc, add=[binding(id="bad", **changes)]))
    assert any(expect in p for p in problems), problems


def test_duplicate_binding_ids_are_refused(doc):
    data = with_changes(doc, add=[binding(id="dup"), binding(id="dup")])
    assert any("duplicate id" in p for p in violations(data))


# ---------------------------------------------------------------------- no competition data
def _strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif node is not None:
        yield str(node)


def test_no_competition_only_data_is_cited_anywhere_in_the_registry(doc):
    cited = [s for s in _strings(doc) if FORBIDDEN.search(s)]
    assert cited == [], cited  # the YAML may explain the exclusion in a comment, never in data
    for card in mr.load_registry().models:
        for e in card.evaluation.evidence:
            assert not FORBIDDEN.search(" ".join((e.report_id, e.manifest_id, e.protocol, e.notes)))


# ---------------------------------------------------------------------- backend API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    app_module = importlib.import_module("cloud.app")
    c = TestClient(app_module.app)
    token = c.post("/v1/signup", json={"email": "tiers@example.com", "password": "password1"}).json()["token"]
    return c, {"Authorization": f"Bearer {token}"}


def test_the_api_never_calls_a_research_capability_production_validated(client):
    c, headers = client
    assert c.get("/v1/capability-tiers").status_code == 401
    body = c.get("/v1/capability-tiers", headers=headers).json()
    assert {e["capability"]: e["tier"] for e in body["capabilities"]} == TODAY
    for e in body["capabilities"]:
        assert e["production_validated"] is False and e["device_verified"] is False
        if e["model_deployment"] != "shipping":
            assert e["evidence"] == [], "a research model's evidence supports no product claim"
    presets = c.get("/v1/presets", headers=headers).json()["presets"]
    assert all(p["validation"]["status"] != "validated_on_device" for p in presets)


# ---------------------------------------------------------------------- the app
def test_only_the_tier_mapping_can_say_validated_for_production():
    sources = REPO / "ios" / "Sources"
    users = sorted(p.name for p in sources.glob("*.swift")
                   if re.search(r"validated for production|production_validated", p.read_text(), re.IGNORECASE))
    assert users == ["CapabilityTier.swift"], users
    text = (sources / "CapabilityTier.swift").read_text()
    # The one green path needs all three: the tier, the backend's flag, and a shipping model.
    assert 'case .productionPermitted where entry.productionValidated:' in text
    assert 'if !shipping {' in text.split("switch entry.tier")[0]
