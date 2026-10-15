"""Tests for the health monitor: checks, restarts, escalation, watchdog gating."""

from core.health import DEGRADED, DOWN, OK, HealthMonitor


def _drive_checks(monitor: HealthMonitor, name: str, times: int) -> None:
    """Run a component's check synchronously `times` times (no monitor thread)."""
    comp = monitor._components[name]
    for _ in range(times):
        comp.last_checked = 0  # force due
        monitor._run_check(comp)


def test_passing_check_is_ok():
    monitor = HealthMonitor()
    monitor.register("camera", critical=True, check=lambda: (True, "fresh"))
    _drive_checks(monitor, "camera", 1)
    snap = monitor.snapshot()
    assert snap["state"] == OK
    assert snap["components"]["camera"]["state"] == OK


def test_failures_escalate_to_restart_then_recovery_resets():
    restarts = []
    healthy = {"value": False}
    monitor = HealthMonitor()
    monitor.register(
        "camera",
        critical=True,
        check=lambda: (healthy["value"], "detail"),
        restart=lambda: restarts.append(1),
    )
    _drive_checks(monitor, "camera", 3)  # threshold is 3 consecutive failures
    assert restarts == [1]
    assert monitor.snapshot()["components"]["camera"]["total_restarts"] == 1

    healthy["value"] = True  # restart "worked"
    _drive_checks(monitor, "camera", 1)
    assert monitor.snapshot()["components"]["camera"]["state"] == OK
    assert monitor.snapshot()["components"]["camera"]["consecutive_failures"] == 0


def test_last_resort_fires_once_after_restart_budget_spent():
    last_resort = []
    monitor = HealthMonitor(on_last_resort=last_resort.append)
    monitor.register(
        "inference",
        critical=True,
        check=lambda: (False, "hung"),
        restart=lambda: None,
        max_restarts=2,
    )
    _drive_checks(monitor, "inference", 12)
    assert last_resort == ["inference"]  # fired exactly once
    assert monitor.snapshot()["components"]["inference"]["state"] == DOWN
    assert monitor.snapshot()["components"]["inference"]["total_restarts"] == 2


def test_noncritical_component_only_degrades_and_never_escalates():
    last_resort = []
    monitor = HealthMonitor(on_last_resort=last_resort.append)
    monitor.register("cloud_uplink", critical=False, check=lambda: (False, "offline"))
    _drive_checks(monitor, "cloud_uplink", 10)
    snap = monitor.snapshot()
    assert snap["components"]["cloud_uplink"]["state"] == DEGRADED
    assert snap["state"] == DEGRADED  # overall reflects it, but...
    assert last_resort == []  # ...no escalation for non-critical components


def test_watchdog_gated_on_critical_down():
    monitor = HealthMonitor()
    monitor.register("camera", critical=True, check=lambda: (False, "dead"))
    assert monitor._critical_component_down() is False  # not DOWN until threshold
    _drive_checks(monitor, "camera", 3)
    assert monitor._critical_component_down() is True  # watchdog pings stop now


def test_push_style_reporting():
    monitor = HealthMonitor()
    monitor.register("capture", critical=True)
    monitor.report_failure("capture", "read failed")
    assert monitor.snapshot()["components"]["capture"]["state"] == DEGRADED
    monitor.report_ok("capture", "recovered")
    assert monitor.snapshot()["components"]["capture"]["state"] == OK


def test_broken_check_counts_as_failure():
    def exploding_check():
        raise ValueError("boom")

    monitor = HealthMonitor()
    monitor.register("hw_imu", check=exploding_check)
    _drive_checks(monitor, "hw_imu", 1)
    assert "check raised" in monitor.snapshot()["components"]["hw_imu"]["detail"]


def test_a_slow_restart_does_not_block_health_reporting():
    """Restarts used to run under the monitor's lock, so /api/health, every
    component's report_failure and the watchdog gate all waited on them."""
    import threading
    import time

    in_restart = threading.Event()
    release = threading.Event()

    def slow_restart():
        in_restart.set()
        release.wait(5)

    monitor = HealthMonitor()
    monitor.register("camera", critical=True, check=lambda: (False, "dead"), restart=slow_restart)
    worker = threading.Thread(target=_drive_checks, args=(monitor, "camera", 5), daemon=True)
    worker.start()
    try:
        assert in_restart.wait(5)
        started = time.monotonic()
        monitor.snapshot()
        monitor.report_ok("camera")
        assert time.monotonic() - started < 0.5, "health reporting waited on a restart"
    finally:
        release.set()
        worker.join(5)
