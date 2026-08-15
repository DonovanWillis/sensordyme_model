"""Throttled raw-signal logging, distinct from the sparse event log in core/logger.py.

`core/logger.py` records discrete *events* (drowsiness fired, phone_use fired, ...).
This module records a continuous, throttled stream of the *raw signals* a module is
watching (EAR, yaw, pitch, PERCLOS, composite risk score, ...) regardless of whether
anything fired. That stream is the raw material a future learned model would need:
real driving sessions build a labeled(-able) dataset to eventually replace hand-tuned
thresholds with something trained on real behavior, per the project's stated roadmap.

The driver-attention family (driver_attention, and the *_max/_skinnemoen/_dashsentinel
variants benchmarked against it) deliberately share ONE fixed schema — ear/yaw_deg/
pitch_deg/perclos/risk_score/phone_visible/hand_count/eating_visible — so their logged
sessions are directly comparable column-for-column; that sharing is intentional and
this module must not disturb it. Every other module's signals are a genuinely
different shape (baby_monitoring's flailing/motion_energy/hands_visible have no driver
equivalent), so forcing them through the same 8 columns silently zeroed all of them —
found live, a real hardware session logged nothing but zero rows for an entire
baby_monitoring run. Each non-driver module instead gets its own sibling CSV, its
header derived from its own snapshot's keys, so nothing is dropped and the driver
family's shared file stays byte-for-byte what it always was.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

_DRIVER_HEADER = [
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

# Modules sharing the fixed schema above, for direct cross-model comparison
# (see module docstring). Matched by prefix so a future variant
# (driver_attention_whatever) is included without a code change here.
_DRIVER_FAMILY_PREFIX = "driver_attention"


def _format_value(value: object) -> str:
    """A CSV cell for one raw signal, formatted by its own type.

    Floats get a fixed precision (readable, and stable for a diff/model-
    training pipeline); bools log as 0/1, matching the driver schema's own
    convention for phone_visible/eating_visible rather than "True"/"False".
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


class TelemetryLogger:
    """Appends throttled raw-signal snapshots to CSVs, flushing after each write."""

    def __init__(self, csv_path: Union[str, Path], interval_s: float = 0.5) -> None:
        """Open (creating if needed) the driver-family telemetry CSV, header if new.

        Args:
            csv_path: Path to the driver-family telemetry CSV log. Non-driver
                modules get sibling files next to it (see module docstring),
                derived from this same path.
            interval_s: Minimum seconds between two written rows (throttles file growth).
        """
        self._base_path = Path(csv_path)
        self._base_path.parent.mkdir(parents=True, exist_ok=True)
        self._interval_s = interval_s
        self._last_write_time: Optional[float] = None
        self._driver_file, self._driver_writer = self._open(self._base_path, _DRIVER_HEADER)
        # module_name -> (file, writer), opened lazily the first time each
        # non-driver-family module's snapshot is actually seen — most
        # deployments only ever run one such module at a time.
        self._other: dict[str, tuple] = {}

    @staticmethod
    def _open(path: Path, header: list[str]):
        is_new = not path.exists()
        file = path.open("a", newline="")
        writer = csv.writer(file)
        if is_new:
            writer.writerow(header)
            file.flush()
        return file, writer

    def _sibling_path(self, module_name: str) -> Path:
        return self._base_path.with_name(f"{self._base_path.stem}_{module_name}{self._base_path.suffix}")

    def _writer_for(self, module_name: str, snapshot: dict):
        entry = self._other.get(module_name)
        if entry is None:
            header = ["timestamp", "module"] + sorted(snapshot.keys())
            entry = self._open(self._sibling_path(module_name), header)
            self._other[module_name] = entry
        return entry

    def log_if_due(self, now: float, timestamp_iso: str, module_name: str, snapshot: dict) -> bool:
        """Write one row if `interval_s` has elapsed since the last write.

        Args:
            now: Current monotonic time, used only to throttle write frequency.
            timestamp_iso: ISO-8601 wall-clock timestamp to record in the row.
            module_name: Name of the module the snapshot came from.
            snapshot: The module's own get_telemetry_snapshot() dict — the
                driver family's fixed keys, or any other module's own shape.

        Returns:
            Whether a row was actually written (False if throttled).
        """
        if self._last_write_time is not None and (now - self._last_write_time) < self._interval_s:
            return False
        self._last_write_time = now
        if module_name.startswith(_DRIVER_FAMILY_PREFIX):
            # Tolerant of the driver family's own snapshot shapes varying slightly
            # (an extra field one variant tracks and another doesn't): absent keys
            # log as zero, same as always, so the shared file's schema never moves.
            self._driver_writer.writerow([
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
            ])
            self._driver_file.flush()
        else:
            file, writer = self._writer_for(module_name, snapshot)
            keys = sorted(snapshot.keys())
            writer.writerow([timestamp_iso, module_name] + [_format_value(snapshot[k]) for k in keys])
            file.flush()
        return True

    def close(self) -> None:
        """Flush and close every underlying file, driver and per-module alike."""
        self._driver_file.close()
        for file, _writer in self._other.values():
            file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
