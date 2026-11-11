"""core/model_registry.py: honest model cards, the evaluation ladder, and what presets inherit.

The registry is only worth having if it refuses to overclaim. So most of these
tests are refusals: each builds an otherwise-valid registry, breaks one thing,
and checks validation names it.
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import pytest
import yaml

from core import model_registry as mr
from core.model_registry import EvaluationStatus as S
from core.roles import ROLES

REGISTRY_PATH = Path(mr.DEFAULT_PATH)
DAMO_SHA = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
OTHER_SHA = "a" * 64

PRESETS = {
    "driver_safety": {"module": "driver_attention"},
    "elderly_monitoring": {"module": "fall_detection"},
    "security_monitoring": {"module": "occupancy"},
    "baby_monitoring": {"module": "baby_monitoring"},
}


# ---------------------------------------------------------------------- fixtures
def _evidence(**over):
    e = {
        "kind": "offline_benchmark", "report_id": "bench:x", "manifest_id": "m1", "split": "val",
        "protocol": "AP50", "metrics": {"ap50": 0.7}, "date": "2026-11-11", "granularity": "image",
    }
    e.update(over)
    return e


def _card(**over):
    c = {
        "id": "det", "version": "1", "family": "damo-yolo", "task": "object_detection",
        "artifacts": [{"path": "models/det.hef", "sha256": DAMO_SHA, "size": 10}],
        "classes": [{"source": "cell phone", "index": 67, "label": "cell phone", "capability": "objects"}],
        "provides_capabilities": ["objects"],
        "compute": {"targets": ["hailo8l"]},
        "licence": {"code": "Apache-2.0", "weights": "Apache-2.0", "training_data": "COCO",
                    "commercial_status": "cleared_with_caveats"},
        "evaluation": {"status": "benchmarked_offline", "evidence": [_evidence()]},
        "known_limitations": [],
        "deployment": "shipping",
    }
    c.update(over)
    return c


def _face_card(**over):
    c = _card(
        id="face", family="mediapipe", task="face_landmarks",
        artifacts=[{"path": "face.tflite", "sha256": "b" * 64}],
        classes=[{"source": "face_landmarks_478", "label": "face_landmarks", "capability": "face"}],
        provides_capabilities=["face"],
    )
    c.update(over)
    return c


def _doc(*cards, providers=None, modules=None):
    return {
        "schema_version": 1,
        "registry_version": "t",
        "providers": providers if providers is not None else {"objects": "det"},
        "modules": modules if modules is not None else {"occupancy": {"uses": ["objects"]}},
        "models": list(cards) or [_card()],
    }


def _violations(doc, today=None):
    return mr.registry_violations(mr.parse_registry(doc), today)


def _assert_refused(doc, fragment, today=None):
    problems = _violations(doc, today)
    assert any(fragment in p for p in problems), f"expected {fragment!r} in {problems}"


def _validated_card(**over):
    return _card(evaluation={"status": "validated_on_device", "evidence": [
        _evidence(kind="on_device_hardware", device="pi5+hailo8l", granularity="event",
                  artifact_sha256=DAMO_SHA, manifest_id="cabin-1", split="test",
                  metrics={"event_recall": 0.9}),
    ]}, **over)


# ---------------------------------------------------------------------- the file
class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate keys {sorted(map(str, dupes))} at line {node.start_mark.line + 1}")
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique)


def test_registry_file_parses_without_duplicate_keys():
    with REGISTRY_PATH.open() as f:
        yaml.load(f, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass


def test_registry_file_passes_validation_today():
    # `today` makes a lapsed shipping_exception fail here: an exception is a
    # dated promise to evaluate, not a permanent pass.
    registry = mr.load_registry(today=dt.datetime.now(dt.UTC).date())
    assert registry.registry_version
    assert {m.id for m in registry.models} >= {
        "damo_yolo_tinynasl20_t_coco80", "mediapipe_face_mesh", "mediapipe_hands",
        "mediapipe_pose", "ultralytics_yolo11n_coco80", "damo_yolo_t_exp08_person_phone_hand",
    }


def test_registry_file_records_the_honest_facts():
    registry = mr.load_registry()
    damo = registry.card("damo_yolo_tinynasl20_t_coco80")
    assert damo.deployment == "shipping"
    assert damo.evaluation.status is S.BENCHMARKED_OFFLINE
    assert damo.has_artifact(DAMO_SHA)
    shipped = [e for e in damo.evaluation.evidence if e.shipped_config and e.manifest_id == "b878e8429d221f3f"]
    assert shipped[0].metrics_dict["phone_ap50"] == 0.728
    assert shipped[0].metrics_dict["person_ap50"] == 0.656
    # The decoder fix is recorded but supports nothing until it ships.
    fix = [e for e in damo.evaluation.evidence if not e.shipped_config]
    assert fix and fix[0].metrics_dict["phone_ap50"] == 0.795
    assert damo.compute.published_fps[0].value == 124
    assert "not a raspberry pi" in damo.compute.published_fps[0].conditions.lower()
    for mp_id in ("mediapipe_face_mesh", "mediapipe_hands", "mediapipe_pose"):
        card = registry.card(mp_id)
        assert card.deployment == "shipping"
        assert card.evaluation.status is S.UNEVALUATED
        assert card.evaluation.shipping_exception is not None
    assert "life-critical" in registry.card("mediapipe_face_mesh").licence.notes
    yolo = registry.card("ultralytics_yolo11n_coco80")
    assert yolo.deployment == "dev_only" and "AGPL" in yolo.licence.weights
    exp08 = registry.card("damo_yolo_t_exp08_person_phone_hand")
    assert exp08.deployment == "candidate" and exp08.evaluation.status is S.UNEVALUATED
    assert not any(m.validated for m in registry.models)


def test_registry_file_matches_the_config_pin():
    config = yaml.safe_load((REGISTRY_PATH.parent.parent / "config.yaml").read_text())
    registry = mr.load_registry()
    assert registry.provider_of("objects").has_artifact(config["accelerator"]["yolo_hef_sha256"])


def test_every_preset_module_is_described():
    from core.presets import load_presets

    registry = mr.load_registry()
    for name, spec in load_presets().items():
        assert spec["module"] in registry.module_map, f"preset {name}: module {spec['module']} undescribed"


def test_role_capabilities_share_the_registry_vocabulary():
    for role in ROLES.values():
        assert set(role.required_capabilities) <= mr.CAPABILITY_KINDS


def test_driver_attention_declarations_match_the_registry():
    pytest.importorskip("mediapipe")
    from modules.driver_attention import DriverAttentionModule

    declared = {k for needs in DriverAttentionModule._EVENT_REQUIREMENTS.values() for k in needs}
    assert declared == set(mr.load_registry().module_map["driver_attention"].uses)


# ---------------------------------------------------------------------- the ladder
def test_ladder_is_strictly_ordered():
    assert [s.value for s in mr.LADDER] == [
        "unevaluated", "benchmarked_offline", "evaluated_in_domain", "validated_on_device"]
    assert S.UNEVALUATED < S.BENCHMARKED_OFFLINE < S.EVALUATED_IN_DOMAIN < S.VALIDATED_ON_DEVICE
    assert S.VALIDATED_ON_DEVICE >= S.VALIDATED_ON_DEVICE > S.EVALUATED_IN_DOMAIN
    shuffled = [S.VALIDATED_ON_DEVICE, S.UNEVALUATED, S.EVALUATED_IN_DOMAIN, S.BENCHMARKED_OFFLINE]
    assert sorted(shuffled) == list(mr.LADDER)
    # Alphabetical order would put benchmarked_offline first; rank must win.
    assert min(mr.LADDER) is S.UNEVALUATED


def test_weakest_picks_the_lowest_rung():
    assert mr.weakest([S.VALIDATED_ON_DEVICE, S.BENCHMARKED_OFFLINE, S.EVALUATED_IN_DOMAIN]) \
        is S.BENCHMARKED_OFFLINE
    assert mr.weakest([]) is None


def test_status_serialises_as_its_value():
    assert S.BENCHMARKED_OFFLINE == "benchmarked_offline"
    assert S("validated_on_device") is S.VALIDATED_ON_DEVICE


def test_unknown_status_cannot_be_parsed():
    with pytest.raises(mr.RegistryError, match="unknown evaluation status"):
        mr.parse_registry(_doc(_card(evaluation={"status": "validated", "evidence": []})))


# ---------------------------------------------------------------------- refusals
def test_a_valid_card_passes():
    assert _violations(_doc()) == []
    assert _violations(_doc(_validated_card())) == []


def test_refuses_missing_licence():
    card = _card()
    del card["licence"]
    _assert_refused(_doc(card), "licence.weights is missing")
    _assert_refused(_doc(_card(licence={"code": "Apache-2.0"})), "licence.commercial_status is missing")


@pytest.mark.parametrize("sha", ["E69136D3" + "0" * 56, "abc", DAMO_SHA + "0", "sha256:" + DAMO_SHA])
def test_refuses_malformed_sha256(sha):
    _assert_refused(_doc(_card(artifacts=[{"path": "x.hef", "sha256": sha}])), "64 lowercase hex")


def test_refuses_shipping_artifact_without_a_pin():
    _assert_refused(_doc(_card(artifacts=[{"path": "x.hef"}])), "has no sha256 pin")
    _assert_refused(_doc(_card(artifacts=[])), "must list its artifacts")


def test_refuses_shipping_below_benchmarked_offline():
    _assert_refused(_doc(_card(evaluation={"status": "unevaluated", "evidence": []})),
                    "must be at least benchmarked_offline")


def test_shipping_exception_is_dated_and_expires():
    exc = {"reason": "shipped before the registry", "recorded": "2026-11-11", "review_by": "2027-01-31"}
    doc = _doc(_card(evaluation={"status": "unevaluated", "evidence": [], "shipping_exception": exc}))
    assert _violations(doc, today=dt.date(2027, 1, 31)) == []
    _assert_refused(doc, "expired on 2027-01-31", today=dt.date(2027, 2, 1))
    no_dates = {"reason": "because"}
    _assert_refused(_doc(_card(evaluation={"status": "unevaluated", "shipping_exception": no_dates})),
                    "needs recorded and review_by dates")
    # Only for a shipping model below the floor: anywhere else it is stale.
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline", "evidence": [_evidence()],
                                           "shipping_exception": exc})), "remove it")


@pytest.mark.parametrize("status", ["benchmarked_offline", "evaluated_in_domain", "validated_on_device"])
def test_refuses_status_claims_without_evidence(status):
    _assert_refused(_doc(_card(deployment="candidate", evaluation={"status": status, "evidence": []})),
                    "claimed without supporting evidence")


def test_refuses_a_claim_above_what_the_evidence_reaches():
    _assert_refused(_doc(_card(evaluation={"status": "evaluated_in_domain", "evidence": [_evidence()]})),
                    "claimed without supporting evidence")


def test_evidence_for_an_unshipped_configuration_supports_nothing():
    ev = {"status": "benchmarked_offline", "evidence": [_evidence(shipped_config=False)]}
    _assert_refused(_doc(_card(evaluation=ev)), "claimed without supporting evidence")


def test_evidence_must_be_complete():
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline",
                                           "evidence": [_evidence(metrics={})]})), "no metrics")
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline",
                                           "evidence": [_evidence(manifest_id="")]})), "manifest_id/split")
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline",
                                           "evidence": [_evidence(date="last week")]})), "YYYY-MM-DD")
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline",
                                           "evidence": [_evidence(split="train")]})), "training split")
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline",
                                           "evidence": [_evidence(kind="vibes")]})), "unknown evidence kind")


def test_in_domain_evidence_needs_consent():
    ev = {"status": "evaluated_in_domain", "evidence": [_evidence(kind="in_domain_harness")]}
    _assert_refused(_doc(_card(evaluation=ev)), "consented footage")
    ev["evidence"][0]["consented"] = True
    assert _violations(_doc(_card(evaluation=ev))) == []


def test_on_device_validation_needs_event_level_evidence_on_the_pinned_binary():
    frame_level = _validated_card()
    frame_level["evaluation"]["evidence"][0]["granularity"] = "frame"
    _assert_refused(_doc(frame_level), "must be event-level")
    _assert_refused(_doc(frame_level), "validated_on_device needs event-level")

    other_binary = _validated_card()
    other_binary["evaluation"]["evidence"][0]["artifact_sha256"] = OTHER_SHA
    _assert_refused(_doc(other_binary), "a different binary proves nothing")

    no_device = _validated_card()
    no_device["evaluation"]["evidence"][0]["device"] = ""
    _assert_refused(_doc(no_device), "must name the device")


@pytest.mark.parametrize(("field", "value", "fragment"), [
    ("provides_capabilities", ["objects", "smell"], "unknown capability kind 'smell'"),
    ("provides_capabilities", ["camera"], "runtime/hardware capability"),
    ("classes", [{"source": "x", "label": "x", "capability": "telepathy"}], "unknown capability kind"),
    ("classes", [{"source": "x", "label": "x", "capability": "face"}], "does not list in provides"),
])
def test_refuses_unknown_or_misplaced_capability_kinds(field, value, fragment):
    _assert_refused(_doc(_card(**{field: value})), fragment)


def test_refuses_unknown_kinds_in_modules_and_providers():
    _assert_refused(_doc(modules={"occupancy": {"uses": ["objects", "sonar"]}}), "unknown capability kind 'sonar'")
    _assert_refused(_doc(modules={"fall_detection": {"uses": ["pose"]}}), "no provider is registered")
    _assert_refused(_doc(providers={"objects": "det", "camera": "det"}), "not a model-provided")
    _assert_refused(_doc(providers={"objects": "nope"}), "not registered")


@pytest.mark.parametrize(("licence", "fragment"), [
    ({"code": "Apache-2.0", "weights": "AGPL-3.0"}, "copyleft"),
    ({"code": "AGPL-3.0", "weights": "Apache-2.0"}, "copyleft"),
    ({"code": "GPL-3.0-only", "weights": "Apache-2.0"}, "copyleft"),
    ({"code": "Apache-2.0", "weights": "CC BY-NC 4.0"}, "non-commercial"),
    ({"code": "Apache-2.0", "weights": "research only, non-commercial"}, "non-commercial"),
])
def test_refuses_copyleft_or_non_commercial_weights_marked_shipping(licence, fragment):
    lic = {"training_data": "COCO", "commercial_status": "cleared_with_caveats", **licence}
    _assert_refused(_doc(_card(licence=lic)), fragment)
    # The same model is fine off the device.
    dev = _card(licence=lic, deployment="dev_only")
    assert _violations(_doc(dev, _card(id="det2"), providers={"objects": "det2"})) == []


def test_lgpl_is_not_mistaken_for_gpl():
    lic = {"code": "LGPL-2.1", "weights": "Apache-2.0", "training_data": "x",
           "commercial_status": "cleared_with_caveats"}
    assert _violations(_doc(_card(licence=lic))) == []


@pytest.mark.parametrize("status", ["restricted", "non_commercial", "unreviewed"])
def test_refuses_shipping_without_commercial_clearance(status):
    lic = {"code": "Apache-2.0", "weights": "Apache-2.0", "training_data": "x", "commercial_status": status}
    _assert_refused(_doc(_card(licence=lic)), "cannot be shipping")


def test_only_shipping_models_may_be_providers():
    dev = _card(id="yolo", deployment="dev_only")
    _assert_refused(_doc(_card(), dev, providers={"objects": "yolo"}), "only a shipping model")


def test_refuses_duplicate_ids():
    _assert_refused(_doc(_card(), _card()), "duplicate id")


def test_load_registry_raises_with_every_problem(tmp_path):
    bad = _doc(_card(licence={}, artifacts=[{"path": "x", "sha256": "nope"}]))
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(mr.RegistryError) as info:
        mr.load_registry(path)
    assert len(info.value.problems) >= 2


# ---------------------------------------------------------------------- loads != validated
def test_load_check_evidence_never_lifts_status():
    load = _evidence(kind="load_check", protocol="model loaded and produced boxes", metrics={})
    _assert_refused(_doc(_card(evaluation={"status": "benchmarked_offline", "evidence": [load]})),
                    "claimed without supporting evidence")
    # Recording the load is fine; it just counts for nothing.
    candidate = _card(deployment="candidate", evaluation={"status": "unevaluated", "evidence": [load]})
    assert _violations(_doc(_card(), candidate.copy() | {"id": "c"})) == []


def test_a_running_model_is_available_not_validated():
    registry = mr.parse_registry(_doc())
    link = mr.capability_link(registry, "objects")
    assert link.evaluation_status is S.BENCHMARKED_OFFLINE and not link.validated
    assert mr.display_state("available", link) == mr.DISPLAY_AVAILABLE_NOT_VALIDATED
    assert mr.display_state("starting", link) == mr.DISPLAY_STARTING
    assert mr.display_state(None, link) == mr.DISPLAY_NOT_REPORTED


def test_validated_needs_the_top_rung_and_liveness_can_only_lower_it():
    registry = mr.parse_registry(_doc(_validated_card()))
    link = mr.capability_link(registry, "objects")
    assert link.validated
    assert mr.display_state("available", link) == mr.DISPLAY_VALIDATED
    assert mr.display_state("impaired", link) == mr.DISPLAY_IMPAIRED


def test_in_domain_evaluation_is_still_not_validated():
    ev = {"status": "evaluated_in_domain",
          "evidence": [_evidence(kind="in_domain_harness", consented=True)]}
    registry = mr.parse_registry(_doc(_card(evaluation=ev)))
    assert mr.display_state("available", mr.capability_link(registry, "objects")) \
        == mr.DISPLAY_AVAILABLE_NOT_VALIDATED


def test_a_binary_the_registry_does_not_know_borrows_no_status():
    registry = mr.parse_registry(_doc(_validated_card()))
    link = mr.capability_link(registry, "objects", running_sha256=OTHER_SHA)
    assert link.provided_by == "unregistered"
    assert link.evaluation_status is S.UNEVALUATED and not link.validated
    assert mr.capability_link(registry, "objects", running_sha256=DAMO_SHA).validated


# ---------------------------------------------------------------------- derivation
def _real():
    return mr.load_registry()


def test_driver_safety_preset_chain():
    view = mr.preset_view(_real(), "driver_safety:strict", PRESETS)
    by_kind = {link.capability: link for link in view.links}
    assert set(by_kind) == {"camera", "inference", "face", "objects", "hands"}
    assert by_kind["face"].provided_by == "mediapipe_face_mesh"
    assert by_kind["objects"].provided_by == "damo_yolo_tinynasl20_t_coco80"
    assert by_kind["hands"].provided_by == "mediapipe_hands"
    assert not by_kind["camera"].model_backed
    # DAMO is benchmarked_offline, but FaceMesh and Hands are unevaluated: the chain is
    # only as strong as its weakest model.
    assert view.weakest is S.UNEVALUATED
    assert not view.validated


def test_optional_capabilities_do_not_set_the_weakest_rung():
    view = mr.preset_view(_real(), "security_monitoring", PRESETS)
    assert view.weakest is S.BENCHMARKED_OFFLINE
    pose = next(link for link in view.links if link.capability == "pose")
    assert pose.optional and pose.evaluation_status is S.UNEVALUATED


def test_all_preset_views():
    views = mr.preset_views(_real(), PRESETS)
    assert views["elderly_monitoring"].models == ("mediapipe_pose",)
    assert views["baby_monitoring"].models == ("damo_yolo_tinynasl20_t_coco80", "mediapipe_pose")
    assert all(not v.validated for v in views.values())
    assert views["driver_safety"].to_dict()["weakest_evaluation_status"] == "unevaluated"


def test_unknown_preset_or_undescribed_module_is_an_error_not_an_empty_chain():
    with pytest.raises(KeyError):
        mr.preset_view(_real(), "exam_proctoring", PRESETS)
    with pytest.raises(KeyError):
        mr.preset_view(_real(), "x", {"x": {"module": "brand_new_module"}})


def test_role_chains():
    views = mr.role_views(_real(), ROLES)
    assert views["driver_safety"].models == ("mediapipe_face_mesh",)
    assert views["driver_safety"].weakest is S.UNEVALUATED
    assert views["occupancy"].weakest is S.BENCHMARKED_OFFLINE
    assert views["child_monitoring"].weakest is None  # needs camera + inference only
    assert not views["child_monitoring"].validated  # no model-backed link: nothing to call validated


def test_custom_policy_chain():
    view = mr.policy_view(_real(), "r1", ["objects"])
    assert view.subject == "policy:r1"
    assert view.weakest is S.BENCHMARKED_OFFLINE


def test_capability_evidence_block():
    block = mr.capability_evidence(
        _real(), preset="driver_safety", role=ROLES["driver_safety"], presets=PRESETS,
        live={"face": "available", "objects": "impaired", "hands": "available", "camera": "available"},
        running_sha256={"objects": DAMO_SHA},
    )
    by_kind = {e["capability"]: e for e in block["capabilities"]}
    assert by_kind["face"]["display"] == "available_not_validated"
    assert by_kind["objects"]["display"] == "impaired"
    assert by_kind["objects"]["evaluation_status"] == "benchmarked_offline"
    assert by_kind["audible_alert"]["model_backed"] is False  # from the role
    assert block["weakest_evaluation_status"] == "unevaluated"
    assert block["validated"] is False
    unknown = mr.capability_evidence(_real(), preset="nonexistent", presets=PRESETS)
    assert unknown["preset_known"] is False and unknown["capabilities"] == []


# ---------------------------------------------------------------------- re-evaluation
def _raw():
    with REGISTRY_PATH.open() as f:
        return yaml.safe_load(f)


def _with(edit):
    raw = copy.deepcopy(_raw())
    edit(raw)
    return mr.parse_registry(raw)


def _model(raw, model_id):
    return next(m for m in raw["models"] if m["id"] == model_id)


def test_no_change_no_reevaluation():
    assert mr.reevaluation_impact(_real(), _real(), PRESETS, ROLES) == []


def test_a_new_detector_binary_reopens_every_preset_that_uses_objects():
    def edit(raw):
        _model(raw, "damo_yolo_tinynasl20_t_coco80")["artifacts"][0]["sha256"] = OTHER_SHA

    impacts = mr.reevaluation_impact(_real(), _with(edit), PRESETS, ROLES)
    assert [(i.model_id, i.change) for i in impacts] == [("damo_yolo_tinynasl20_t_coco80", "changed")]
    assert impacts[0].presets == ("baby_monitoring", "driver_safety", "security_monitoring")
    assert impacts[0].roles == ("occupancy", "property")


def test_a_face_model_change_reopens_driver_safety_only():
    def edit(raw):
        _model(raw, "mediapipe_face_mesh")["version"] = "mediapipe-0.10.21"

    (impact,) = mr.reevaluation_impact(_real(), _with(edit), PRESETS, ROLES)
    assert impact.presets == ("driver_safety",)
    assert impact.roles == ("driver_safety",)


def test_a_pose_change_includes_presets_that_use_it_optionally():
    def edit(raw):
        _model(raw, "mediapipe_pose")["known_limitations"].append("new finding")

    (impact,) = mr.reevaluation_impact(_real(), _with(edit), PRESETS, ROLES)
    assert impact.presets == ("baby_monitoring", "elderly_monitoring", "security_monitoring")


def test_a_provider_swap_names_both_models():
    def edit(raw):
        cand = _model(raw, "damo_yolo_t_exp08_person_phone_hand")
        cand.update(_card(id=cand["id"], provides_capabilities=["objects", "hands"], classes=cand["classes"]))
        raw["providers"]["objects"] = cand["id"]

    new = _with(edit)
    impacts = {i.model_id: i for i in mr.reevaluation_impact(_real(), new, PRESETS, ROLES)}
    assert impacts["damo_yolo_tinynasl20_t_coco80"].change == "provider_changed"
    assert impacts["damo_yolo_t_exp08_person_phone_hand"].change == "changed"
    assert impacts["damo_yolo_t_exp08_person_phone_hand"].presets == (
        "baby_monitoring", "driver_safety", "security_monitoring")


def test_a_dev_only_change_reopens_nothing_on_the_device():
    def edit(raw):
        _model(raw, "ultralytics_yolo11n_coco80")["version"] = "ultralytics-8.4"

    (impact,) = mr.reevaluation_impact(_real(), _with(edit), PRESETS, ROLES)
    assert impact.presets == () and impact.roles == ()
