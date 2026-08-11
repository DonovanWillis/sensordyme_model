"""One authoritative answer to what the device is monitoring.

There were two answers that could disagree: `active_preset` said which preset
was selected, the stored configuration said whether rules had replaced it. The
app read the first and the device acted on the second.

And "override" meant only that the base module's alerts were discarded — the
module still ran. MediaPipe FaceMesh, Hands and Pose cost 85% of one core on
this device, computing drowsiness and gaze for someone who asked about phones.
Suppressing an alert is not the same as not doing the work.
"""

from core.monitoring_mode import (
    CUSTOM_DETECTOR_MODULE,
    POSE_DETECTOR_MODULE,
    MonitoringMode,
    capabilities_covered_by,
    describe,
    module_for,
    resolve,
    select_custom_module,
)


def test_a_custom_configuration_wins():
    assert resolve(monitoring_active=True, preset_module="driver_attention") is MonitoringMode.AI_CUSTOM


def test_without_one_the_preset_is_in_charge():
    assert resolve(monitoring_active=False, preset_module="driver_attention") is MonitoringMode.STANDARD


def test_no_preset_and_no_configuration_is_off():
    assert resolve(monitoring_active=False, preset_module=None) is MonitoringMode.OFF


def test_custom_monitoring_runs_a_general_detector_not_the_preset():
    """The point. Running driver_attention and throwing its alerts away pays
    for MediaPipe to answer questions nobody asked."""
    assert module_for(MonitoringMode.AI_CUSTOM, "driver_attention") == CUSTOM_DETECTOR_MODULE
    assert module_for(MonitoringMode.AI_CUSTOM, "driver_attention") != "driver_attention"


def test_the_general_detector_carries_no_mediapipe():
    """If this ever stops being true, the saving disappears silently."""
    from pathlib import Path
    source = Path(f"modules/{CUSTOM_DETECTOR_MODULE}.py").read_text()
    assert "mediapipe" not in source.lower(), \
        f"{CUSTOM_DETECTOR_MODULE} imports mediapipe; custom monitoring would cost as much as the preset"


def test_the_general_detector_can_actually_feed_the_rules():
    """A detector that reports nothing makes every custom rule silently dead."""
    from pathlib import Path
    source = Path(f"modules/{CUSTOM_DETECTOR_MODULE}.py").read_text()
    assert "def get_detections" in source


def test_standard_mode_runs_the_presets_own_module():
    assert module_for(MonitoringMode.STANDARD, "fall_detection") == "fall_detection"


def test_off_runs_nothing():
    assert module_for(MonitoringMode.OFF, "driver_attention") is None


class TestSelectCustomModule:
    """A "sleeping" rule composes to `target: lying_down` (jupiter/prompt.py),
    which validates fine against the capability registry — but occupancy, the
    module AI_CUSTOM always ran, never produces that label. The rule would
    activate in the app and never fire. This is the fix: AI_CUSTOM picks the
    module that actually covers what today's rules ask for."""

    def test_object_only_rules_stay_on_the_cheap_detector(self):
        assert select_custom_module(["bottle", "person"]) == CUSTOM_DETECTOR_MODULE

    def test_pose_only_rules_switch_to_the_pose_detector(self):
        assert select_custom_module(["lying_down"]) == POSE_DETECTOR_MODULE

    def test_activity_labels_also_route_to_the_pose_detector(self):
        assert select_custom_module(["running"]) == POSE_DETECTOR_MODULE

    def test_mixed_object_and_pose_rules_default_to_the_object_detector(self):
        """Neither module covers both. Defaulting to occupancy — rather than
        fall_detection — keeps the common half (an object rule) working; the
        pose half is left to the mismatch warning in
        services/runtime.py's `_apply_requested_labels`, not silently dropped."""
        assert select_custom_module(["lying_down", "bottle"]) == CUSTOM_DETECTOR_MODULE

    def test_no_labels_is_the_safe_default(self):
        assert select_custom_module([]) == CUSTOM_DETECTOR_MODULE

    def test_an_unrecognised_label_is_the_safe_default(self):
        """Should not be reachable post-validation, but guessing a module for a
        label nothing provides would be worse than the existing behaviour."""
        assert select_custom_module(["unicorn"]) == CUSTOM_DETECTOR_MODULE


class TestModuleForWithRequestedLabels:
    def test_omitting_requested_labels_keeps_the_old_behaviour(self):
        """Every caller that predates this parameter must be unaffected."""
        assert module_for(MonitoringMode.AI_CUSTOM, "driver_attention") == CUSTOM_DETECTOR_MODULE

    def test_pose_labels_route_ai_custom_to_the_pose_detector(self):
        assert module_for(
            MonitoringMode.AI_CUSTOM, "driver_attention", requested_labels={"lying_down"}
        ) == POSE_DETECTOR_MODULE

    def test_standard_mode_ignores_requested_labels(self):
        """STANDARD always runs the preset's own module — requested_labels is
        an AI_CUSTOM-only concept and must not leak into this path."""
        assert module_for(
            MonitoringMode.STANDARD, "fall_detection", requested_labels={"bottle"}
        ) == "fall_detection"


def test_capabilities_covered_by_the_pose_detector_excludes_objects():
    """Sanity check on the map the mismatch warning reads: the pose detector
    must not claim to cover object labels it cannot produce."""
    assert "objects.coco" not in capabilities_covered_by(POSE_DETECTOR_MODULE)
    assert "pose.body" in capabilities_covered_by(POSE_DETECTOR_MODULE)


def test_the_description_names_what_is_disabled_not_only_what_is_on():
    """"AI custom monitoring is on" and "driver distraction is off" are
    different claims, and stating only the first is how someone ends up
    believing both are running."""
    described = describe(MonitoringMode.AI_CUSTOM, CUSTOM_DETECTOR_MODULE, "Alert me about phones.")
    assert described["base_monitoring_disabled"] is True
    assert described["instruction"] == "Alert me about phones."

    standard = describe(MonitoringMode.STANDARD, "driver_attention")
    assert standard["base_monitoring_disabled"] is False
