"""Tests for the hardware abstraction layer."""

from hardware.hal import HardwareSuite, build_hardware


def test_build_hardware_is_fully_populated_with_nulls():
    suite = build_hardware({})
    assert isinstance(suite, HardwareSuite)
    # The runtime can call every interface with nothing connected:
    assert suite.imu.read() is None
    assert suite.gps.read() is None
    assert suite.io_expander.read_pin(0) is False
    suite.io_expander.write_pin(0, True)
    suite.audio.play("/tmp/nope.wav")
    suite.leds.set("alert", True)
    suite.buttons.on_press("mode", lambda: None)
    suite.esp32.send({"cmd": "ping"})
    assert suite.esp32.poll() == []


def test_health_checks_report_not_fitted_as_healthy():
    checks = build_hardware({}).health_checks()
    assert set(checks) == {
        "hw_imu", "hw_gps", "hw_io_expander", "hw_audio", "hw_leds", "hw_buttons", "hw_esp32",
    }
    for check in checks.values():
        healthy, detail = check()
        assert healthy is True  # absent optional hardware is never an error
        assert detail == "not fitted"
