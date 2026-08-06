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
