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

Two failure properties, both because the caller is the inference loop:

  * A failed write is logged (rate-limited) and dropped, never raised. A full or
    read-only card makes every write fail; an exception here used to escape the
    loop on each telemetry tick, so detection stopped within half a second.
  * Each file is bounded. At two rows a second this stream grows by roughly
    15-26 MB a day and nothing ever trimmed it, so on a long-running device it was
    the thing that filled the card. A file past `max_bytes` is renamed to
    `<name>.1<suffix>` (replacing the previous one) and a fresh file with the same
    header is started, so at most about twice `max_bytes` is kept per file.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Union

from core.logger import _close_quietly, _is_empty_or_missing, _WriteErrors

#: Default size at which a telemetry file is rotated. ~25 MB is about a day and
#: a half of driver-family rows: plenty of training data per file, and with one
#: rotated generation kept, the stream can never use more than ~50 MB per file.
DEFAULT_MAX_BYTES = 25 * 1024 * 1024

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

    def __init__(self, csv_path: Union[str, Path], interval_s: float = 0.5,
                 max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        """Open (creating if needed) the driver-family telemetry CSV, header if new.

        Args:
            csv_path: Path to the driver-family telemetry CSV log. Non-driver
                modules get sibling files next to it (see module docstring),
                derived from this same path.
            interval_s: Minimum seconds between two written rows (throttles file growth).
            max_bytes: Size past which a file is rotated to `<name>.1<suffix>`;
                0 or less disables rotation.
        """
        self._base_path = Path(csv_path)
        self._interval_s = interval_s
        self._max_bytes = int(max_bytes)
        self._last_write_time: Optional[float] = None
        self._errors = _WriteErrors(f"telemetry log {self._base_path}")
        try:
            self._base_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._errors.report(exc)
        # path -> (file, writer, header). The driver file is opened eagerly (as
        # it always was); a file that cannot be opened is retried on the next
        # due write instead of failing the device's boot.
        self._files: dict[Path, tuple] = {}
        # module_name -> (path, header) for non-driver-family modules, fixed the
        # first time each one's snapshot is seen -- most deployments only ever
        # run one such module at a time.
        self._other: dict[str, tuple[Path, list[str]]] = {}
        self._entry(self._base_path, _DRIVER_HEADER)

    @staticmethod
    def _existing_header(path: Path) -> Optional[list[str]]:
        """The first line of an already-written CSV, or None if there isn't one."""
        try:
            with path.open("r", newline="") as f:
                row = next(csv.reader(f), None)
            return row
        except (FileNotFoundError, StopIteration):
            return None

    @staticmethod
    def _open(path: Path, header: list[str]):
        # An existing but EMPTY file (a power cut right after creation) needs
        # the header too; `not exists()` alone appended headerless rows to it.
        needs_header = _is_empty_or_missing(path)
        file = path.open("a", newline="")
        writer = csv.writer(file)
        if needs_header:
            try:
                writer.writerow(header)
                file.flush()
            except OSError:
                _close_quietly(file)
                raise
        return file, writer

    def _entry(self, path: Path, header: list[str]):
        """The open (file, writer, header) for `path`, opening it if needed; None if it cannot be."""
        entry = self._files.get(path)
        if entry is not None:
            return entry
        try:
            file, writer = self._open(path, header)
        except OSError as exc:
            self._errors.report(exc)
            return None
        entry = (file, writer, header)
        self._files[path] = entry
        return entry

    def _sibling_path(self, module_name: str) -> Path:
        return self._base_path.with_name(f"{self._base_path.stem}_{module_name}{self._base_path.suffix}")

    def _sibling_for(self, module_name: str, snapshot: dict) -> tuple[Path, list[str]]:
        known = self._other.get(module_name)
        if known is not None:
            return known
        header = ["timestamp", "module", *sorted(snapshot.keys())]
        path = self._sibling_path(module_name)
        # A module's snapshot shape is fixed by its code, but a device that
        # deployed an older version of that code (or switched module
        # variants) can already have a file on disk from a DIFFERENT
        # shape. Appending under a stale header — found live, this exact
        # file — silently misaligns every row after it for anyone reading
        # by column name. Rather than guess how to migrate old rows, this
        # rotates the mismatched file aside (its data isn't lost, just not
        # extended) and starts a fresh one with today's header.
        existing = self._existing_header(path)
        if existing is not None and existing != header:
            for n in range(1, 100):
                stamp = path.with_name(f"{path.stem}.old{n}{path.suffix}")
                if not stamp.exists():
                    path.rename(stamp)
                    break
        self._other[module_name] = (path, header)
        return path, header

    def _rotate_if_large(self, path: Path) -> None:
        """Rename `path` to `<name>.1<suffix>` once it passes max_bytes, and start afresh."""
        if self._max_bytes <= 0:
            return
        entry = self._files.get(path)
        if entry is None:
            return
        file, _writer, header = entry
        if os.fstat(file.fileno()).st_size < self._max_bytes:
            return
        del self._files[path]
        _close_quietly(file)
        os.replace(path, path.with_name(f"{path.stem}.1{path.suffix}"))
        self._entry(path, header)

    def log_if_due(self, now: float, timestamp_iso: str, module_name: str, snapshot: dict) -> bool:
        """Write one row if `interval_s` has elapsed since the last write.

        Args:
            now: Current monotonic time, used only to throttle write frequency.
            timestamp_iso: ISO-8601 wall-clock timestamp to record in the row.
            module_name: Name of the module the snapshot came from.
            snapshot: The module's own get_telemetry_snapshot() dict — the
                driver family's fixed keys, or any other module's own shape.

        Returns:
            Whether a row was actually written (False if throttled, or if the
            write failed -- which is logged, never raised).
        """
        if self._last_write_time is not None and (now - self._last_write_time) < self._interval_s:
            return False
        self._last_write_time = now
        try:
            if module_name.startswith(_DRIVER_FAMILY_PREFIX):
                path, header = self._base_path, _DRIVER_HEADER
                # Tolerant of the driver family's own snapshot shapes varying slightly
                # (an extra field one variant tracks and another doesn't): absent keys
                # log as zero, same as always, so the shared file's schema never moves.
                row = [
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
            else:
                path, header = self._sibling_for(module_name, snapshot)
                keys = header[2:]
                row = [timestamp_iso, module_name] + [_format_value(snapshot.get(k, "")) for k in keys]
            entry = self._entry(path, header)
            if entry is None:
                return False
            file, writer, _header = entry
            writer.writerow(row)
            file.flush()
            self._rotate_if_large(path)
        except OSError as exc:
            self._errors.report(exc)
            return False
        return True

    def close(self) -> None:
        """Flush and close every underlying file, driver and per-module alike."""
        for file, _writer, _header in self._files.values():
            _close_quietly(file)
        self._files.clear()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
