"""What Configure-with-AI does when there is no language model to ask.

Two separate defects lived here, and they compounded: the fallback is the ONLY
path taken when the model is unreachable, i.e. exactly the moment it has to be
trustworthy.

1. `_MODES` was a hardcoded tuple listing `driver_distraction` and
   `driver_distraction_openpilot` — consolidated away, now only aliases — and
   NOT listing `driver_safety`, the mode that exists. It is both the enum in
   the structured-output schema (so the model could not emit `driver_safety`)
   and the server-side filter `preset if preset in _MODES else ""` (so a model
   that emitted it anyway was downgraded to "no preset"). The fallback
   meanwhile returned the superseded alias, which the iOS `Preset.catalog` has
   no entry for, so the mode picker rendered the raw string at the user.

2. The fallback's rules were keyed `detection` with EVENT-TYPE names ("fall",
   "phone_use", "zone_violation", "unusual_stillness") — names no detector
   emits as a label. `_validate_proposal` read `target`, found nothing, and
   assessed "", which the capability layer accepts as a whole-frame condition.
   So the rule passed validation, was delivered to the device, was stored
   there, and could never fire. Silently: nothing anywhere said it was inert.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("fastapi")

from cloud import app as cloud_app  # noqa: E402
from core.capabilities import registry  # noqa: E402
from core.presets import load_presets, resolve_preset  # noqa: E402
from core.rules import ConfiguredRule  # noqa: E402

# One prompt per fallback branch, phrased the way somebody actually types.
FALLBACK_PROMPTS = [
    "watch my baby in the crib",
    "let me know if grandma falls",
    "tell me if there is an intruder at the door",
    "watch me while I am driving in case I use my phone",
]


# ------------------------------------------------------------------ the mode list
def test_the_mode_list_is_exactly_what_presets_yaml_defines():
    """Derived, not transcribed. The hardcoded tuple had drifted in BOTH
    directions -- two modes that no longer exist, one that does missing."""
    assert set(cloud_app._modes()) == set(load_presets())


def test_the_model_can_emit_every_mode_that_exists(monkeypatch):
    """The filter must accept every real mode. `driver_safety` was silently
    downgraded to "" -- the model's correct answer, discarded."""
    for mode in load_presets():
        assert cloud_app._resolve_mode(mode) == mode, f"{mode} was rejected by the server filter"


def test_a_superseded_mode_name_is_corrected_rather_than_discarded():
    """A model trained on, or a prompt quoting, an old name means the mode
    that name now refers to. Answering "" throws away a correct request."""
    assert cloud_app._resolve_mode("driver_distraction") == "driver_safety"
    assert cloud_app._resolve_mode("driver_distraction_openpilot") == "driver_safety"


def test_something_that_is_not_a_mode_is_still_refused():
    """The filter still has to filter -- this must not become a passthrough."""
    assert cloud_app._resolve_mode("watch_the_stock_market") == ""
    assert cloud_app._resolve_mode("") == ""
    assert cloud_app._resolve_mode(None) == ""


def test_the_structured_output_enum_offers_the_real_modes():
    """The schema handed to the model is built from the same list, so the
    model is never asked to choose between names that do not exist."""
    enum = cloud_app._PROPOSAL_SCHEMA["properties"]["preset"]["enum"]
    assert set(enum) == {"", *load_presets()}


# ------------------------------------------------------------------ the fallback
@pytest.mark.parametrize("prompt", FALLBACK_PROMPTS)
def test_a_fallback_names_a_mode_the_app_can_render(prompt):
    """The iOS Preset.catalog is keyed on presets.yaml's names. A superseded
    alias has no entry, so the mode row rendered `driver_distraction` --
    the internal string, on screen, to a customer."""
    proposal = cloud_app._validate_proposal(cloud_app._ai_fallback(prompt))
    preset = proposal["preset"]
    assert preset in load_presets(), f"{preset!r} is not a mode the app can render"
    assert resolve_preset(preset) == preset, f"{preset!r} is a superseded name"


@pytest.mark.parametrize("prompt", FALLBACK_PROMPTS)
def test_every_fallback_rule_survives_the_backend_s_own_validation(prompt):
    """Nothing the fallback proposes may be quietly moved to `unsupported` --
    the fallback is the path used when nothing else is available."""
    proposal = cloud_app._validate_proposal(cloud_app._ai_fallback(prompt))
    assert proposal["rules"], f"the fallback for {prompt!r} proposed no runnable rule"
    assert proposal["unsupported"] == [], proposal["unsupported"]


@pytest.mark.parametrize("prompt", FALLBACK_PROMPTS)
def test_every_fallback_rule_is_one_the_capability_registry_accepts(prompt):
    """Assessed on the target the DEVICE will use, which is the check that
    was missing: validation looked at `target` while core/rules.py resolves
    `target or detection`."""
    reg = registry()
    proposal = cloud_app._validate_proposal(cloud_app._ai_fallback(prompt))
    for rule in proposal["rules"]:
        target = rule.get("target") or rule.get("detection") or ""
        verdict = reg.assess(target, rule.get("predicate") or "present")
        assert verdict.ok, f"{prompt!r} -> {target!r}/{rule.get('predicate')!r}: {verdict.reason}"


@pytest.mark.parametrize("prompt", FALLBACK_PROMPTS)
def test_every_fallback_rule_actually_builds_on_the_device(prompt):
    """The end of the line: ConfiguredRule.from_dict is what the device runs
    the rule through. It returns None for a rule with no usable detection,
    and that is where every one of the old fallback rules died -- or worse,
    built with an event-type name it would then look for as a label and
    never find."""
    reg = registry()
    proposal = cloud_app._validate_proposal(cloud_app._ai_fallback(prompt))
    for index, rule in enumerate(proposal["rules"]):
        built = ConfiguredRule.from_dict(index, rule)
        assert built is not None, f"{prompt!r}: the device would discard this rule outright"
        # And what it will look for really is a label some detector produces.
        assert built.detection in reg.labels(), (
            f"{prompt!r}: the device would watch for {built.detection!r}, "
            "which no detector emits -- the rule can never fire"
        )


def test_an_unmatched_prompt_still_refuses_rather_than_inventing():
    """Unchanged behaviour, pinned: no keyword match means no configuration
    and a stated reason, not a plausible-looking guess."""
    proposal = cloud_app._ai_fallback("alert me when the stock market drops")
    assert proposal["preset"] == "" and proposal["rules"] == []
    assert proposal["unsupported"] and proposal["unsupported"][0]["requires"]


# ------------------------------------------------- the loophole that allowed it
def test_a_rule_that_names_nothing_to_watch_is_refused_not_kept():
    """The loophole itself. `reg.assess("", "present")` is legitimately True
    -- the capability layer treats an empty target as a whole-frame condition
    -- but core/rules.py drops such a rule outright, so it can never fire.
    Validation must not hand back a rule the device will throw away."""
    assert registry().assess("", "present").ok, "premise: the capability layer accepts it"
    assert ConfiguredRule.from_dict(0, {"target": "", "predicate": "present"}) is None, (
        "premise: the device discards it"
    )

    validated = cloud_app._validate_proposal({
        "preset": "", "rules": [{"target": "", "predicate": "present", "description": "watch"}],
    })
    assert validated["rules"] == [], "a rule the device discards was reported as configured"
    assert validated["unsupported"], "and the user was not told why"


def test_a_legacy_detection_keyed_rule_is_assessed_on_the_name_the_device_uses():
    """core/rules.py accepts `detection` as a fallback for `target`, so that
    is the string the device watches for. Assessing "" instead is what let
    every event-type-named rule through."""
    validated = cloud_app._validate_proposal({
        "preset": "", "rules": [{"detection": "phone_use", "description": "sustained phone use"}],
    })
    assert validated["rules"] == []
    assert "phone_use" in validated["unsupported"][0]["reason"]

    # A legacy rule naming a REAL label is still accepted -- this must reject
    # dead rules, not the legacy shape.
    validated = cloud_app._validate_proposal({
        "preset": "", "rules": [{"detection": "person", "description": "someone is here"}],
    })
    assert [r["target"] for r in validated["rules"]] == ["person"]
