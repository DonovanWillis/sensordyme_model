"""Unit tests for TelemetryLogger's throttling and CSV persistence."""

import csv

from core.telemetry import TelemetryLogger

_SNAPSHOT = {
    "ear": 0.31,
    "yaw_deg": 4.2,
    "pitch_deg": -1.5,
    "perclos": 0.05,
    "risk_score": 0.12,
    "phone_visible": False,
    "hand_count": 1,
    "eating_visible": False,
}


def test_first_row_always_written(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=1.0)
    wrote = logger.log_if_due(now=0.0, timestamp_iso="2026-01-01T00:00:00+00:00", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger.close()
    assert wrote is True

    with path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "timestamp"
    assert len(rows) == 2  # header + one data row


def test_throttles_writes_within_interval(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=1.0)
    logger.log_if_due(now=0.0, timestamp_iso="t0", module_name="driver_attention", snapshot=_SNAPSHOT)
    wrote = logger.log_if_due(now=0.3, timestamp_iso="t1", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger.close()
    assert wrote is False

    with path.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + only the first row


def test_writes_again_once_interval_elapses(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=1.0)
    logger.log_if_due(now=0.0, timestamp_iso="t0", module_name="driver_attention", snapshot=_SNAPSHOT)
    wrote = logger.log_if_due(now=1.5, timestamp_iso="t1", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger.close()
    assert wrote is True

    with path.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + two data rows


def test_appends_across_instances_without_rewriting_header(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger1 = TelemetryLogger(path, interval_s=0.0)
    logger1.log_if_due(now=0.0, timestamp_iso="t0", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger1.close()

    logger2 = TelemetryLogger(path, interval_s=0.0)
    logger2.log_if_due(now=0.0, timestamp_iso="t1", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger2.close()

    with path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "timestamp"
    assert sum(1 for r in rows if r and r[0] == "timestamp") == 1  # header written only once
    assert len(rows) == 3  # header + two data rows


# ----------------------------------------------------------------- non-driver modules
# A real hardware session (2026-08-15) logged nothing but zero rows for an
# entire baby_monitoring run: the driver-family schema has no flailing/
# motion_energy/hands_visible columns, and every module's snapshot used to be
# forced through it regardless. Non-driver modules now get their own sibling
# file with their own schema instead of being silently zeroed.
_BABY_SNAPSHOT = {
    "person_count": 1,
    "motion_energy": 0.42,
    "active_alerts": 2,
    "flailing": True,
    "flailing_confidence": 0.81,
    "hands_visible": 2,
    "objects_in_hand": 1,
}


def test_a_non_driver_modules_own_signals_are_not_silently_zeroed(tmp_path):
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=0.0)
    logger.log_if_due(now=0.0, timestamp_iso="t0", module_name="baby_monitoring", snapshot=_BABY_SNAPSHOT)
    logger.close()

    # The shared driver file is untouched -- no phantom zero row for a module
    # that never ran through it.
    with path.open() as f:
        driver_rows = list(csv.reader(f))
    assert len(driver_rows) == 1  # header only

    sibling = tmp_path / "telemetry_baby_monitoring.csv"
    assert sibling.exists()
    with sibling.open() as f:
        rows = list(csv.reader(f))
    header, row = rows[0], rows[1]
    assert "flailing" in header and "motion_energy" in header
    as_dict = dict(zip(header, row))
    assert as_dict["flailing"] == "1"
    assert as_dict["flailing_confidence"] == "0.8100"
    assert as_dict["hands_visible"] == "2"


def test_driver_family_and_other_modules_coexist_in_the_same_run(tmp_path):
    """Switching from a preset to AI Custom (occupancy) and back, in one
    process lifetime, must not corrupt either file's schema."""
    path = tmp_path / "telemetry.csv"
    logger = TelemetryLogger(path, interval_s=0.0)
    logger.log_if_due(now=0.0, timestamp_iso="t0", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger.log_if_due(now=1.0, timestamp_iso="t1", module_name="baby_monitoring", snapshot=_BABY_SNAPSHOT)
    logger.log_if_due(now=2.0, timestamp_iso="t2", module_name="driver_attention", snapshot=_SNAPSHOT)
    logger.close()

    with path.open() as f:
        driver_rows = list(csv.reader(f))
    assert len(driver_rows) == 3  # header + two driver rows, the baby row never lands here

    with (tmp_path / "telemetry_baby_monitoring.csv").open() as f:
        baby_rows = list(csv.reader(f))
    assert len(baby_rows) == 2  # header + one row
