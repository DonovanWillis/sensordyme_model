"""The iPhone's alert sound names the same driver-safety alerts the device does.

ios/Sources/AlertSound.swift decides which new alerts play the iPhone's
notification sound. Its list of driver-safety alerts mirrors the driver_safety
role's mandatory alert types, and its log-only list must never name a type the
device alerts on. Read from the Swift source, so the mirror cannot drift
silently (the same approach as tests/test_app_preset_parity.py).
"""

import re
from pathlib import Path

from core.roles import ROLES
from main import ALERT_EVENT_TYPES_PI

SOURCE = Path("ios/Sources/AlertSound.swift")


def swift_set(name: str) -> set[str]:
    text = SOURCE.read_text()
    start = text.index(f"static let {name}: Set<String> = [")
    body = text[start:text.index("]", start)]
    return set(re.findall(r'"([a-z_]+)"', body))


def test_driver_safety_alert_types_mirror_the_role():
    assert swift_set("driverSafetyAlertTypes") == set(ROLES["driver_safety"].mandatory_alert_types)


def test_log_only_types_are_never_alert_types():
    log_only = swift_set("logOnlyTypes")
    assert "phone_detected" in log_only
    assert not log_only & set(ALERT_EVENT_TYPES_PI)
    assert not log_only & set(ROLES["driver_safety"].mandatory_alert_types)
