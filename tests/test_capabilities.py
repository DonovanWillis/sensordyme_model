"""The capability layer, and the promise it exists to keep.

Every test here is really one question: does SensorDyme tell the truth about
what it can watch for? The failure this replaces was a request for something
unsupported quietly becoming "Security Monitoring" and reporting success.
"""

import importlib
import sys

import pytest

from core.capabilities import (
    Capability,
    CapabilityRegistry,
    Support,
    default_registry,
)


@pytest.fixture()
def reg():
    return default_registry()


# --------------------------------------------------------------- what really exists
def test_the_object_vocabulary_is_the_coco_class_list(reg):
    labels = reg.labels()
    for present in ("person", "dog", "couch", "bottle", "cup", "cell phone", "car"):
        assert present in labels
    # The absences matter more than the presences: these are the words users
    # reach for that the model genuinely does not know.
    for absent in ("can", "package", "food", "drink", "smoke"):
        assert absent not in labels


# --------------------------------------------------------------- supported requests
@pytest.mark.parametrize("target,predicate", [
    ("person", "present"),
    ("person", "absent"),
    ("dog", "present"),
    ("couch", "present"),
    ("person", "in_zone"),
    ("person", "still"),
])
def test_requests_the_device_can_genuinely_serve(reg, target, predicate):
    assert reg.assess(target, predicate).support is Support.SUPPORTED


def test_dog_on_the_couch_is_genuinely_supported(reg):
    """Both objects are real COCO classes, so this one actually works today."""
    assert reg.assess("dog", "near").ok
    assert reg.assess("couch", "present").support is Support.SUPPORTED


# --------------------------------------------------------------- honest refusals
def test_can_is_refused_with_a_training_data_route(reg):
    """The example from the brief: 'monitor for a can' must not become Security."""
    verdict = reg.assess("can", "present")
    assert verdict.support is Support.NEEDS_TRAINING_DATA
    assert not verdict.ok
    assert "does not recognise 'can'" in verdict.reason
    assert "labelled examples" in verdict.requires


def test_package_is_refused_rather_than_approximated(reg):
    assert reg.assess("package", "present").support is Support.NEEDS_TRAINING_DATA


def test_smoke_reports_its_own_specific_model_gap(reg):
    """A declared-but-uninstalled capability explains itself precisely."""
    verdict = reg.assess("smoke", "present")
    assert verdict.support is Support.NEEDS_MODEL
    assert "smoke/fire detection model" in verdict.requires


def test_behaviours_need_an_action_model(reg):
    for behaviour in ("eating", "drinking", "running", "sleeping"):
        verdict = reg.assess("person", behaviour)
        assert not verdict.ok
        assert "action-recognition" in verdict.requires


def test_posture_is_available_as_labels_now_that_pose_is_wired(reg):
    """Postures are labels, asked for the same way objects are."""
    if "lying_down" not in reg.labels():
        pytest.skip("mediapipe not installed in this environment")
    for pose_label in ("standing", "sitting", "lying_down", "on_floor"):
        assert reg.assess(pose_label, "present").support is Support.SUPPORTED


def test_climbing_still_needs_an_action_model(reg):
    """Posture is not action: standing on a chair and climbing look identical."""
    verdict = reg.assess("person", "climbing")
    assert not verdict.ok
    assert "action-recognition" in verdict.requires


def test_entry_and_exit_need_tracking(reg):
    """Without identity across frames, 'entered' and 'left' are indistinguishable."""
    for predicate in ("enters", "exits", "picks_up"):
        verdict = reg.assess("person", predicate)
        assert not verdict.ok
        assert "tracker" in verdict.requires


def test_unknowable_requests_are_refused_outright(reg):
    verdict = reg.assess("person", "thinking")
    assert verdict.support is Support.UNSUPPORTED


# --------------------------------------------------------------- approximation
def test_category_words_are_approximated_and_labelled_as_such(reg):
    """'food' isn't a class, but the concrete foods are — offered as PARTIAL."""
    verdict = reg.assess("food", "present")
    assert verdict.support is Support.PARTIAL
    assert "sandwich" in verdict.alternatives and "pizza" in verdict.alternatives


def test_drink_approximates_to_real_containers(reg):
    verdict = reg.assess("drink", "present")
    assert verdict.support is Support.PARTIAL
    assert set(verdict.alternatives) == {"bottle", "cup", "wine glass"}


def test_proximity_is_partial_because_it_is_not_proof_of_interaction(reg):
    assert reg.assess("person", "holding").support is Support.PARTIAL


# --------------------------------------------------------------- extensibility
def test_registering_a_model_makes_its_requests_supported(reg):
    """The whole point: new model in, new vocabulary out, no logic changes."""
    assert reg.assess("can", "present").support is Support.NEEDS_TRAINING_DATA

    reg.register(Capability(
        id="objects.beverage_v1", kind="objects",
        summary="a fine-tuned beverage detector",
        labels=frozenset({"can", "carton"}),
        predicates=frozenset({"present", "absent", "count"}),
    ))

    assert reg.assess("can", "present").support is Support.SUPPORTED
    assert "can" in reg.labels()


def test_installing_a_tracker_unlocks_entry_and_exit(reg):
    assert not reg.assess("person", "enters").ok
    reg.register(Capability(
        id="tracking.identity", kind="tracking",
        summary="multi-object tracking",
        predicates=frozenset({"enters", "exits", "direction"}),
    ))
    assert reg.assess("person", "enters").support is Support.SUPPORTED


def test_removing_a_capability_withdraws_its_claims():
    registry = CapabilityRegistry([
        Capability(id="objects.coco", kind="objects", summary="detector",
                   labels=frozenset({"person"}), predicates=frozenset({"present"}))
    ])
    assert registry.assess("person", "present").ok
    registry.unregister("objects.coco")
    assert not registry.assess("person", "present").ok


def test_capability_sheet_lists_real_labels_for_the_model(reg):
    sheet = reg.describe_for_llm()
    assert "person" in sheet and "couch" in sheet
    # Unavailable capabilities are declared, not hidden — but AFTER the usable
    # ones. Printing them first measurably cost accuracy: they were the most
    # prominent entries on the page and none of them work.
    assert "NOT AVAILABLE" in sheet
    assert sheet.index("CONDITIONS") < sheet.index("NOT AVAILABLE")


def test_object_labels_are_scannable_not_one_long_line(reg):
    """84 labels on a single comma-separated line measurably broke a 1.5B
    model's ability to find objects it can actually detect.

    Only the LABEL lines are constrained. Prose in the not-available section is
    allowed to run long — the model reads it once, rather than searching it for
    a word.
    """
    sheet = reg.describe_for_llm()
    objects_section = sheet.split("CONDITIONS")[0]
    longest = max(len(line) for line in objects_section.splitlines())
    assert longest < 140, f"a {longest}-char label line is back; the model cannot scan it"


# --------------------------------------------------------------- the endpoint
@pytest.fixture()
def backend(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # Pin the AI provider: without this the suite passes or fails depending
    # on whether a local model server is running, which is not a property
    # a test should have. "unavailable" is not a registered provider, so
    # _ai_configure_llm falls through to the deterministic path.
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for mod in ("cloud.app", "cloud"):
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


def configure(client, headers, prompt):
    return client.post(
        "/v1/ai/configure", json={"device_id": 1, "prompt": prompt}, headers=headers
    ).json()


def test_unmatched_prompt_no_longer_becomes_security_monitoring(backend):
    """The reported bug, as a test. 'Monitor for a can' picked Security and said
    it had chosen a 'safe default'; it had chosen something never asked for."""
    _, client, headers = backend
    data = configure(client, headers, "monitor for a can")

    assert data["preset"] != "security_monitoring"
    assert data["preset"] == ""
    assert "safe default" not in data["explanation"].lower()
    assert data["rules"] == []


def test_missing_language_service_is_stated_not_papered_over(backend):
    _, client, headers = backend
    data = configure(client, headers, "alert me if my dog gets on the couch")

    assert data["source"] == "unavailable"
    assert "ANTHROPIC_API_KEY" in data["unsupported"][0]["requires"]
    assert "nothing has been changed" in data["explanation"].lower()


def test_custom_rules_do_not_require_a_preset(backend):
    """A configuration belonging to none of the five modes is a normal outcome."""
    app_module, client, headers = backend

    app_module._ai_configure_llm = lambda prompt: {
        "preset": "", "variant": "", "explanation": "Watching for your dog on the couch.",
        "settings": {}, "unsupported": [], "source": "claude",
        "rules": [{
            "target": "dog", "predicate": "near", "secondary": "couch",
            "description": "dog on the couch", "zone": "", "persist_s": 3, "min_count": 1,
            "confidence": 0.6, "severity": "warning", "cooldown_s": 120,
            "schedule": "", "notify": "push",
        }],
    }

    data = configure(client, headers, "alert me if my dog gets on the couch")
    assert data["mode"] == "custom"
    assert data["preset"] == ""
    assert len(data["rules"]) == 1
    assert data["rules"][0]["target"] == "dog"
    assert data["rules"][0]["secondary"] == "couch"


def test_a_rule_whose_second_object_is_undetectable_is_rejected(backend):
    app_module, client, headers = backend

    app_module._ai_configure_llm = lambda prompt: {
        "preset": "", "variant": "", "explanation": "x", "settings": {},
        "unsupported": [], "source": "claude",
        "rules": [{
            "target": "person", "predicate": "holding", "secondary": "can",
            "description": "someone holding a can", "zone": "", "persist_s": 3,
            "min_count": 1, "confidence": 0.6, "severity": "warning",
            "cooldown_s": 60, "schedule": "", "notify": "push",
        }],
    }

    data = configure(client, headers, "alert me if someone is holding a can")
    assert data["rules"] == []
    assert "can" in data["unsupported"][0]["reason"]


def test_partial_rules_carry_their_caveat_into_the_rule(backend):
    """The approximation has to survive to the review screen, not just the reply."""
    app_module, client, headers = backend

    app_module._ai_configure_llm = lambda prompt: {
        "preset": "", "variant": "", "explanation": "x", "settings": {},
        "unsupported": [], "source": "claude",
        "rules": [{
            "target": "person", "predicate": "holding", "secondary": "bottle",
            "description": "someone holding a bottle", "zone": "", "persist_s": 3,
            "min_count": 1, "confidence": 0.6, "severity": "warning",
            "cooldown_s": 60, "schedule": "", "notify": "push",
        }],
    }

    data = configure(client, headers, "alert me if someone is holding a bottle")
    assert len(data["rules"]) == 1
    assert "caveat" in data["rules"][0]
    assert "not proof" in data["rules"][0]["caveat"]
