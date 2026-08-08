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
    MonitoringMode,
    describe,
    module_for,
    resolve,
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


def test_the_description_names_what_is_disabled_not_only_what_is_on():
    """"AI custom monitoring is on" and "driver distraction is off" are
    different claims, and stating only the first is how someone ends up
    believing both are running."""
    described = describe(MonitoringMode.AI_CUSTOM, CUSTOM_DETECTOR_MODULE, "Alert me about phones.")
    assert described["base_monitoring_disabled"] is True
    assert described["instruction"] == "Alert me about phones."

    standard = describe(MonitoringMode.STANDARD, "driver_attention")
    assert standard["base_monitoring_disabled"] is False
