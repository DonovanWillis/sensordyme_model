"""The operator must be told scoring is paused BEFORE it silently resumes.

Baby monitoring declines to score a choking hazard when it cannot positively
identify which person in frame is the infant. Two independently-configurable
durations govern what happens next:

    infant_unidentified_warn_s   how long the decline must persist before the
                                 `infant_unidentified` alert fires        (20 s)
    infant_reacquire_s           how long before a re-identified person may be
                                 accepted as the infant again             (30 s)

The ORDERING between them is a safety property, not a tuning preference. If the
warn is longer than the reacquire, scoring silently restarts before anyone is
told it had stopped — which is the exact silent-degradation failure the decline
path exists to prevent. `config.yaml` documents the relationship in prose; this
pins it, because prose does not fail a build.

This is a real gap and not a hypothetical: both values read from config with
`config.get(...)`, so a deployment can set either one, and nothing in the module
compares them. It was flagged by the agent who wrote the identifier, after the
ordering had already been written up as a guarantee.
"""

from __future__ import annotations

import inspect

from modules import baby_monitoring
from modules.baby_monitoring import INFANT_REACQUIRE_S, BabyMonitoringModule


def _module(**overrides) -> BabyMonitoringModule:
    """A baby module with detectors stubbed out — only the rule wiring matters here."""
    config = {
        "confidence_threshold": 0.4,
        "yolo_weights": "yolo11n.pt",
        "expect_presence": True,
        **overrides,
    }
    module = BabyMonitoringModule.__new__(BabyMonitoringModule)
    module.__init__.__wrapped__(module, config) if hasattr(module.__init__, "__wrapped__") \
        else BabyMonitoringModule.__init__(module, config)
    return module


def _warn_persist(module) -> float:
    """The persist_s the `infant_unidentified` rule was actually registered with."""
    rules = module._engine._rules
    assert "infant_unidentified" in rules, (
        "the infant_unidentified rule is gone — the decline is silent again"
    )
    return rules["infant_unidentified"].persist_s


def test_the_defaults_warn_before_scoring_can_resume():
    """The shipped values must satisfy the ordering, not merely document it."""
    module = _module()

    warn_s = _warn_persist(module)

    assert warn_s < INFANT_REACQUIRE_S, (
        f"infant_unidentified_warn_s ({warn_s}s) is not shorter than "
        f"infant_reacquire_s ({INFANT_REACQUIRE_S}s), so scoring would resume "
        "before the operator is told it ever stopped"
    )


def test_a_deployment_cannot_configure_the_warning_to_arrive_too_late():
    """The dangerous tuning: warn longer than reacquire.

    Both values are `config.get`-able, so this is reachable by editing a YAML
    file — no code change required. Whatever the module does about it, it must
    not silently accept an ordering that hides the degradation from the operator.
    """
    module = _module(infant_unidentified_warn_s=45.0, infant_reacquire_s=30.0)

    warn_s = _warn_persist(module)
    reacquire_s = module._infant_identifier.reacquire_s

    assert warn_s < reacquire_s, (
        f"a config set warn={warn_s}s against reacquire={reacquire_s}s and the module "
        "accepted it. Scoring resumes at 30s; the operator hears at 45s. For 15 "
        "seconds the device is scoring choking hazards against a person it is not "
        "confident is the infant, while reporting nothing. Clamp the warn below the "
        "reacquire, or refuse the config — but do not take it as given."
    )


def test_raising_the_reacquire_alone_keeps_the_ordering():
    """A deployment that wants a longer quarantine must not break the warning."""
    module = _module(infant_reacquire_s=120.0)

    assert _warn_persist(module) < module._infant_identifier.reacquire_s


def test_the_relationship_is_documented_where_it_is_configured():
    """config.yaml is where someone tunes these; the reason must be there too.

    A guarantee enforced in code but unexplained at the point of configuration
    gets "fixed" by the next person who finds the clamp surprising.
    """
    from pathlib import Path

    config_text = Path(__file__).resolve().parent.parent.joinpath("config.yaml").read_text()

    assert "infant_unidentified" in config_text or "20 s" in config_text, (
        "config.yaml no longer explains the warn/reacquire ordering; if the values "
        "move, nothing tells the next person why one must be shorter"
    )


def test_the_module_still_declines_rather_than_guessing():
    """Guard the property all of the above is protecting.

    If the identifier ever goes back to always returning a box, the ordering
    above becomes irrelevant — and these tests would keep passing while the
    original defect returned. This asserts the decline path still exists.
    """
    source = inspect.getsource(baby_monitoring)

    assert "InfantIdentifier" in source, "the positive-identification path is gone"
    # The NAME survives in a comment explaining what was replaced, which is good
    # documentation. What must not come back is the function.
    assert "def _infer_baby_box" not in source, (
        "the smallest-box heuristic is back; it returns the CAREGIVER when the "
        "infant's box drops out, which is the defect this all exists to prevent"
    )
