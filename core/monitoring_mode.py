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

from enum import Enum
from typing import Optional


class MonitoringMode(str, Enum):
    OFF = "off"
    STANDARD = "standard"
    AI_CUSTOM = "ai_custom"


# The module that serves custom rules: general object detection, no MediaPipe.
#
# Chosen because rules target the *things* a detector names — "person",
# "cell phone", "bottle" — rather than any conclusion a preset draws from them.
# It implements get_detections(), which is what the rule engine consumes.
CUSTOM_DETECTOR_MODULE = "occupancy"


def resolve(*, monitoring_active: bool, preset_module: Optional[str]) -> MonitoringMode:
    """The one place the active mode is decided."""
    if monitoring_active:
        return MonitoringMode.AI_CUSTOM
    if preset_module:
        return MonitoringMode.STANDARD
    return MonitoringMode.OFF


def module_for(mode: MonitoringMode, preset_module: Optional[str]) -> Optional[str]:
    """Which detection module a mode should actually run."""
    if mode is MonitoringMode.OFF:
        return None
    if mode is MonitoringMode.AI_CUSTOM:
        return CUSTOM_DETECTOR_MODULE
    return preset_module


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
