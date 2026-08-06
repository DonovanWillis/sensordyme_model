"""Throttled raw-signal logging, distinct from the sparse event log in core/logger.py.

`core/logger.py` records discrete *events* (drowsiness fired, phone_use fired, ...).
This module records a continuous, throttled stream of the *raw signals* a module is
watching (EAR, yaw, pitch, PERCLOS, composite risk score, ...) regardless of whether
anything fired. That stream is the raw material a future learned model would need:
real driving sessions build a labeled(-able) dataset to eventually replace hand-tuned
thresholds with something trained on real behavior, per the project's stated roadmap.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

_HEADER = [
    "timestamp",
    "module",
    "ear",
    "yaw_deg",
    "pitch_deg",
    "perclos",
    "risk_score",
    "phone_visible",
    "hand_count",
    "eating_visible",
]


class TelemetryLogger:
    """Appends throttled raw-signal snapshots to a CSV, flushing after each write."""

    def __init__(self, csv_path: Union[str, Path], interval_s: float = 0.5) -> None:
        """Open (creating if needed) the telemetry CSV, writing a header if new.

        Args:
            csv_path: Path to the telemetry CSV log.
            interval_s: Minimum seconds between two written rows (throttles file growth).
        """
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._interval_s = interval_s
        self._last_write_time: Optional[float] = None
        is_new = not self._path.exists()
        self._file = self._path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(_HEADER)
            self._file.flush()

    def log_if_due(self, now: float, timestamp_iso: str, module_name: str, snapshot: dict) -> bool:
        """Write one row if `interval_s` has elapsed since the last write.

        Args:
            now: Current monotonic time, used only to throttle write frequency.
            timestamp_iso: ISO-8601 wall-clock timestamp to record in the row.
            module_name: Name of the module the snapshot came from.
            snapshot: Dict with keys matching the telemetry schema (ear, yaw_deg,
                pitch_deg, perclos, risk_score, phone_visible, hand_count, eating_visible).

        Returns:
            Whether a row was actually written (False if throttled).
        """
        if self._last_write_time is not None and (now - self._last_write_time) < self._interval_s:
            return False
        self._last_write_time = now
        # Tolerant of per-module snapshot shapes: the schema is driver-centric
        # (this file predates other modes), so absent keys log as zeros rather
        # than crashing the inference loop when e.g. baby_monitoring supplies
        # {person_count, motion_energy, ...} instead.
        self._writer.writerow(
            [
                timestamp_iso,
                module_name,
                f"{float(snapshot.get('ear', 0.0)):.4f}",
                f"{float(snapshot.get('yaw_deg', 0.0)):.2f}",
                f"{float(snapshot.get('pitch_deg', 0.0)):.2f}",
                f"{float(snapshot.get('perclos', 0.0)):.4f}",
                f"{float(snapshot.get('risk_score', 0.0)):.4f}",
                int(snapshot.get("phone_visible", 0)),
                snapshot.get("hand_count", 0),
                int(snapshot.get("eating_visible", 0)),
            ]
        )
        self._file.flush()
        return True

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
