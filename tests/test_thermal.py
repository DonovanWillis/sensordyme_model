"""Reporting the device's temperature and cooling honestly.

The question that prompted this was "the Pi is very hot and the fan doesn't
seem to come on". Answering it needs the difference between a fan the firmware
is asking to spin and a fan that is actually spinning — those look identical
from every value except RPM, and the failure they hide is the one that matters.
"""

import core.thermal as thermal
from core.thermal import SOFT_THROTTLE_C, status


def fake(monkeypatch, *, temp=68.0, rpm=7100, duty=175, level=3,
         throttled_now=False, throttled_since_boot=False, present=True):
    monkeypatch.setattr(thermal, "cpu_temperature_c", lambda: temp)
    monkeypatch.setattr(thermal, "fan", lambda: {
        "present": present, "rpm": rpm, "duty": duty,
        "level": level, "level_max": 4, "spinning": bool(rpm),
    })
    monkeypatch.setattr(thermal, "throttling", lambda: {
        "available": True, "raw": "0x0",
        "throttled_now": throttled_now, "capped_now": False,
        "under_voltage_now": False,
        "throttled_since_boot": throttled_since_boot,
        "under_voltage_since_boot": False,
    })


def test_a_warm_but_healthy_device_is_reported_healthy(monkeypatch):
    """Measured on the real device: 68.8C peak, fan level 3 of 4, never
    throttled. Hot to the touch and entirely fine — saying otherwise sends
    someone to fix working hardware."""
    fake(monkeypatch, temp=68.8)
    healthy, detail = status()
    assert healthy
    assert "68.8" in detail and "7100 rpm" in detail
    assert "below throttling" in detail


def test_a_stopped_fan_on_a_hot_device_is_a_fault(monkeypatch):
    """The one genuinely actionable fault, and it is silent: the firmware keeps
    asking for a duty cycle whether or not anything is turning."""
    fake(monkeypatch, temp=72.0, rpm=0, duty=200)
    healthy, detail = status()
    assert healthy is False
    assert "not turning" in detail
    assert "fan header" in detail


def test_a_stopped_fan_on_a_cold_device_is_not_a_fault(monkeypatch):
    """Below the first trip point the firmware deliberately leaves it off."""
    fake(monkeypatch, temp=42.0, rpm=0, duty=0, level=0)
    assert status()[0] is True


def test_active_throttling_is_a_fault(monkeypatch):
    fake(monkeypatch, temp=81.0, throttled_now=True)
    healthy, detail = status()
    assert healthy is False
    assert "throttling" in detail


def test_a_past_throttle_event_is_reported_without_failing(monkeypatch):
    """Sticky since-boot bits matter: a device that threw an event under a load
    you have stopped reproducing looks perfectly healthy in the moment."""
    fake(monkeypatch, temp=70.0, throttled_since_boot=True)
    healthy, detail = status()
    assert healthy is True
    assert "throttled since boot" in detail


def test_a_warm_device_says_where_the_limit_is(monkeypatch):
    fake(monkeypatch, temp=76.0)
    _, detail = status()
    assert str(int(SOFT_THROTTLE_C)) in detail


def test_no_sensor_is_not_a_failure(monkeypatch):
    """A dev Mac has none of this; provisioning never runs there anyway."""
    monkeypatch.setattr(thermal, "cpu_temperature_c", lambda: None)
    assert status()[0] is True


def test_the_snapshot_reports_headroom_not_just_temperature(monkeypatch):
    fake(monkeypatch, temp=68.0)
    snap = thermal.snapshot()
    assert snap["cpu_temp_c"] == 68.0
    assert snap["headroom_c"] == 12.0
    assert snap["fan"]["spinning"] is True


def test_under_voltage_is_reported_and_outranks_temperature(monkeypatch):
    """These bits were parsed and never read by anything, so the one fault a
    power-hungry peripheral actually causes was the one the device could not
    report. Brownout-corrupted SD cards are this build's known field killer."""
    fake(monkeypatch, temp=55.0)
    monkeypatch.setattr(thermal, "throttling", lambda: {
        "available": True, "raw": "0x1", "throttled_now": False, "capped_now": False,
        "under_voltage_now": True, "throttled_since_boot": False,
        "under_voltage_since_boot": True,
    })
    healthy, detail = status()
    assert healthy is False
    assert "power supply" in detail.lower() or "sagging" in detail.lower()


def test_a_past_voltage_sag_is_reported_without_failing(monkeypatch):
    fake(monkeypatch, temp=60.0)
    monkeypatch.setattr(thermal, "throttling", lambda: {
        "available": True, "raw": "0x10000", "throttled_now": False, "capped_now": False,
        "under_voltage_now": False, "throttled_since_boot": False,
        "under_voltage_since_boot": True,
    })
    healthy, detail = status()
    assert healthy is True
    assert "sagged" in detail
