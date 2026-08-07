"""What the device's temperature and cooling are actually doing.

Every value here is read from the kernel. Nothing is inferred from what
hardware was ordered: "AI HAT+ fitted" is a fact about a purchase order, and
"the fan is spinning at 7100 RPM" is a fact about this device right now.

Thresholds are the Raspberry Pi 5's published ones, not invented:

    80 C   soft throttle — the firmware begins reducing clocks
    85 C   hard throttle
    110 C  critical trip

The Pi 5's own firmware runs the fan against trip points at 50/60/67.5/75 C.
That control loop is in the firmware and works without help; this module
observes it rather than competing with it. A second fan controller fighting the
first is a good way to end up with a device that is both noisier and hotter.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

# Raspberry Pi 5 published limits.
SOFT_THROTTLE_C = 80.0
HARD_THROTTLE_C = 85.0

# Where to start telling someone. Below the throttle point, because a warning
# that arrives as performance is already dropping is a notification, not a
# warning.
WARN_C = 75.0


def _read_int(path: str) -> Optional[int]:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def cpu_temperature_c() -> Optional[float]:
    value = _read_int("/sys/class/thermal/thermal_zone0/temp")
    return value / 1000.0 if value is not None else None


def _pwmfan_dir() -> Optional[str]:
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if Path(f"{hwmon}/name").read_text().strip() == "pwmfan":
                return hwmon
        except OSError:
            continue
    return None


def fan() -> dict:
    """The fan's real state: RPM, duty cycle, and the firmware's cooling level.

    RPM is the one that matters. `pwm1` is what the firmware asked for; a fan
    that is unplugged, jammed or wired to nothing reports a perfectly healthy
    PWM value and 0 RPM — which is exactly the failure someone means when they
    say the cooler is not coming on.
    """
    hwmon = _pwmfan_dir()
    rpm = _read_int(f"{hwmon}/fan1_input") if hwmon else None
    pwm = _read_int(f"{hwmon}/pwm1") if hwmon else None
    level = _read_int("/sys/class/thermal/cooling_device0/cur_state")
    level_max = _read_int("/sys/class/thermal/cooling_device0/max_state")
    return {
        "present": hwmon is not None,
        "rpm": rpm,
        "duty": pwm,
        "level": level,
        "level_max": level_max,
        "spinning": bool(rpm),
    }


def throttling() -> dict:
    """Whether the firmware has ever had to reduce clocks.

    `get_throttled` reports both current state (low bits) and a sticky
    since-boot record (bits 16+). The sticky half is the useful one: a device
    that threw a throttle event an hour ago under a load you have since stopped
    reproducing looks perfectly healthy in the moment.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5
        )
        raw = result.stdout.strip().split("=")[-1]
        bits = int(raw, 16)
    except Exception:  # noqa: BLE001 - absent off-Pi
        return {"available": False}
    return {
        "available": True,
        "raw": raw,
        "throttled_now": bool(bits & 0x4),
        "capped_now": bool(bits & 0x2),
        "under_voltage_now": bool(bits & 0x1),
        "throttled_since_boot": bool(bits & 0x40000),
        "under_voltage_since_boot": bool(bits & 0x10000),
    }


def snapshot() -> dict:
    """Everything, in one read, for the health monitor and the app."""
    temp = cpu_temperature_c()
    state = fan()
    limits = throttling()
    return {
        "cpu_temp_c": round(temp, 1) if temp is not None else None,
        "fan": state,
        "throttling": limits,
        "headroom_c": round(SOFT_THROTTLE_C - temp, 1) if temp is not None else None,
        "warn_at_c": WARN_C,
        "throttle_at_c": SOFT_THROTTLE_C,
    }


def status() -> tuple[bool, str]:
    """(healthy, detail) for the health monitor's `thermal` component.

    Non-critical: a hot device is still a working device, and the firmware
    protects the hardware on its own. This reports so a person can see it, not
    so anything is shut down behind their back.
    """
    temp = cpu_temperature_c()
    if temp is None:
        return True, "no temperature sensor on this platform"

    state = fan()
    limits = throttling()

    # A fan that should be spinning and is not is the one genuinely actionable
    # fault here — and it is silent, because the firmware keeps asking.
    if state["present"] and not state["spinning"] and temp >= 60:
        return False, (
            f"{temp:.1f}°C and the fan is not turning "
            f"(duty {state['duty']}) — check it is plugged into the fan header"
        )

    if limits.get("throttled_now"):
        return False, f"{temp:.1f}°C — the firmware is throttling to protect the hardware"

    detail = f"{temp:.1f}°C"
    if state["rpm"]:
        detail += f", fan {state['rpm']} rpm (level {state['level']}/{state['level_max']})"
    if temp >= WARN_C:
        return True, detail + f" — warm; throttling starts at {SOFT_THROTTLE_C:.0f}°C"
    if limits.get("throttled_since_boot"):
        return True, detail + " — has throttled since boot"
    return True, detail + f", {SOFT_THROTTLE_C - temp:.0f}°C below throttling"
