"""One authoritative answer to "what is this device monitoring".

There were two answers and they could disagree. `active_preset` said which
preset was selected; the stored custom configuration said whether rules had
replaced it. The app read the first, the device acted on the second, and a
person who had configured something was shown a mode it was not running.

Worse, "override" meant only that the base module's alerts were thrown away.
The module still ran — MediaPipe FaceMesh, Hands and Pose, 85% of one core —
producing drowsiness and gaze estimates that were computed, discarded, and paid
for in heat. Suppressing an alert is not the same as not doing the work.

So the mode decides which detector runs:

    OFF        nothing
    STANDARD   the preset's own module (driver safety, baby, elderly, security)
    AI_CUSTOM  a general object detector, and only the user's rules

AI_CUSTOM deliberately does NOT run the preset's module. A rule like "tell me
when you see a phone" needs object detection and nothing else; face landmarks
and pose estimation answer questions nobody asked.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Optional


class MonitoringMode(str, Enum):  # noqa: UP042 - StrEnum would change str()/format() of members
    OFF = "off"
    STANDARD = "standard"
    AI_CUSTOM = "ai_custom"
    #: Custom rules evaluated BESIDE the preset's module, which keeps running and
    #: alerting. Chosen when the device's role requires that module (M0.1,
    #: core/roles.py): custom rules may add to a mandatory safeguard, never
    #: replace it.
    ADDITIVE = "standard_plus_custom"


# The module that serves custom rules: general object detection, no MediaPipe.
#
# Chosen because rules target the *things* a detector names — "person",
# "cell phone", "bottle" — rather than any conclusion a preset draws from them.
# It implements get_detections(), which is what the rule engine consumes.
CUSTOM_DETECTOR_MODULE = "occupancy"

# The other module custom rules can run against: MediaPipe Pose, for rules that
# name a posture or an activity label (lying_down, sitting, running, ...) rather
# than an object. core/capabilities.py's `pose.body` / `activity.gait` are what
# make "alert if I'm sleeping" composable at all (see jupiter/prompt.py) — this
# is the other half: if the active detector never runs pose estimation, that
# composed rule validates, looks active in the app, and never fires, which is
# worse than Jupiter refusing the request outright.
POSE_DETECTOR_MODULE = "fall_detection"

# Which capability IDs (core/capabilities.py) each candidate module actually
# backs. Not derived from the registry automatically — the registry knows what
# a capability can name, not which module's code produces it, so this is the
# one place that link is stated. Keep in step with occupancy.py/fall_detection.py's
# `get_detections()` if either widens what it reports.
_MODULE_CAPABILITIES: dict[str, frozenset[str]] = {
    CUSTOM_DETECTOR_MODULE: frozenset({"objects.coco"}),
    POSE_DETECTOR_MODULE: frozenset({"pose.body", "activity.gait", "gesture.arm", "state.resting"}),
}


def resolve(
    *, monitoring_active: bool, preset_module: Optional[str], additive: bool = False
) -> MonitoringMode:
    """The one place the active mode is decided.

    `additive` is the role's verdict that the preset module is mandatory, so custom
    rules must run beside it instead of replacing it.
    """
    if monitoring_active:
        if additive and preset_module:
            return MonitoringMode.ADDITIVE
        return MonitoringMode.AI_CUSTOM
    if preset_module:
        return MonitoringMode.STANDARD
    return MonitoringMode.OFF


def select_custom_module(requested_labels: Iterable[str], reg=None) -> str:
    """Which detector actually covers today's custom rules.

    Defaults to `CUSTOM_DETECTOR_MODULE` (occupancy) — the common case, and the
    cheap one: no MediaPipe, matching `test_the_general_detector_carries_no_mediapipe`.
    Only switches to the pose detector when EVERY requested label needs it and
    NONE need plain object detection, so a rule set that mixes "sleeping" with
    "see a phone" does not silently stop serving the phone half. See
    `capabilities_covered_by` for the honest half of that mixed case.
    """
    from core.capabilities import registry as default_registry_fn

    reg = reg or default_registry_fn()
    labels = [label for label in requested_labels if label]
    if not labels:
        return CUSTOM_DETECTOR_MODULE

    needed_capabilities: set[str] = set()
    for label in labels:
        providers = {c.id for c in reg.providers_of_label(label) if c.available}
        if not providers:
            # An unrecognised label shouldn't be possible post-validation, but
            # guessing a module for it would be worse than the safe default.
            return CUSTOM_DETECTOR_MODULE
        needed_capabilities |= providers

    if needed_capabilities and needed_capabilities <= _MODULE_CAPABILITIES[POSE_DETECTOR_MODULE]:
        return POSE_DETECTOR_MODULE
    return CUSTOM_DETECTOR_MODULE


def capabilities_covered_by(module: str) -> frozenset[str]:
    """The capability IDs `module` actually backs, for the mismatch warning in
    services/runtime.py's `_apply_requested_labels`."""
    return _MODULE_CAPABILITIES.get(module, frozenset())


def module_for(
    mode: MonitoringMode,
    preset_module: Optional[str],
    *,
    requested_labels: Optional[Iterable[str]] = None,
    reg=None,
) -> Optional[str]:
    """Which detection module a mode should actually run.

    `requested_labels` is optional and AI_CUSTOM-only: omitting it (every
    existing caller before this parameter existed) keeps today's behaviour —
    always `CUSTOM_DETECTOR_MODULE`. Passing the current rule set's labels lets
    AI_CUSTOM pick `POSE_DETECTOR_MODULE` when that is genuinely what is needed.
    """
    if mode is MonitoringMode.OFF:
        return None
    if mode is MonitoringMode.AI_CUSTOM:
        if requested_labels is not None:
            return select_custom_module(requested_labels, reg=reg)
        return CUSTOM_DETECTOR_MODULE
    return preset_module  # STANDARD and ADDITIVE both run the preset's module


def describe(mode: MonitoringMode, module: Optional[str], instruction: str = "") -> dict:
    """What is running and what is deliberately not, for logs and the app.

    Naming the disabled half matters: "AI custom monitoring is on" and "driver
    distraction is off" are different claims, and only stating the first is how
    someone ends up believing both systems are running at once.
    """
    return {
        "mode": mode.value,
        "module": module,
        "instruction": instruction,
        "base_monitoring_disabled": mode is MonitoringMode.AI_CUSTOM,
    }
