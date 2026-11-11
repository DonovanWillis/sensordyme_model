"""Appends detected events to a CSV file. Never writes frames or video to disk.

Durability matters here more than it looks. This file is the device's only
permanent record of what it saw: the upload queue is bounded and in memory, so
for anything beyond the last few hundred events this CSV *is* the history.

`flush()` is not enough for that. It hands bytes to the operating system, which
is free to extend the file's length before the data blocks reach the card. Cut
the power in that window and the file comes back longer, with the gap reading as
NUL bytes — which is exactly what was found on the device: 944 NUL bytes across
3 lines. `fsync()` closes that window.

The cost is a real disk sync per event, which would be unacceptable for
telemetry at two rows a second but is nothing here: a device producing thirty
events an hour syncs once every two minutes.

A write that fails is logged and dropped, never raised. A full card (ENOSPC)
or one the kernel remounted read-only (EROFS) makes every write fail, and the
caller is the inference loop: an exception here used to escape it before the
alert for the very event being logged was dispatched, and then kill the loop
on each restart. Losing a log row is bad; losing detection is worse.
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Union

from core.events import Event
from core.logging_setup import get_logger

log = get_logger("logger")

_HEADER = ["timestamp", "type", "duration_s", "confidence", "module", "contributing_signals"]

#: How often a persistently failing write is re-reported. Every event would
#: otherwise log a stack of identical errors on a card that is simply full.
_ERROR_LOG_EVERY_S = 60.0


class _WriteErrors:
    """Rate-limited reporting for a write path that may fail on every call."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._last_logged = -math.inf
        self._suppressed = 0

    def report(self, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_logged < _ERROR_LOG_EVERY_S:
            self._suppressed += 1
            return
        extra = f" ({self._suppressed} more since the last report)" if self._suppressed else ""
        log.error("could not write the %s: %s%s", self._what, exc, extra)
        self._last_logged = now
        self._suppressed = 0


def _is_empty_or_missing(path: Path) -> bool:
    """Whether a CSV needs its header written.

    `not exists()` alone is wrong for an existing EMPTY file -- what repair()
    leaves when the only line was damaged, or what a power cut leaves right
    after creation -- and rows appended to it have no header at all, so every
    reader keyed by column name takes the first event as the header.
    """
    try:
        return path.stat().st_size == 0
    except FileNotFoundError:
        return True


def _close_quietly(handle) -> None:
    """Close a file whose final flush may fail (a full card): the close must
    still release the descriptor and must not raise into a shutdown path."""
    try:
        handle.close()
    except OSError as exc:
        log.warning("closing a log file lost buffered data: %s", exc)
        with contextlib.suppress(OSError):
            handle.close()  # buffered data is discarded; the descriptor is released


def repair(path: Union[str, Path]) -> int:
    """Drop unreadable rows from an event log, keeping everything else.

    Returns the number of rows removed.

    Deliberately conservative: a row containing a NUL byte is one the previous
    write never finished, and nothing can recover its contents. Everything
    around it is intact and is the history someone may care about, so this
    removes the damage rather than the file. Rewritten through a temporary file
    so an interruption during the repair cannot make things worse.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return 0
    if b"\x00" not in raw:
        return 0

    lines = raw.split(b"\n")
    kept = [line for line in lines if b"\x00" not in line]
    removed = len(lines) - len(kept)

    temp = path.with_suffix(path.suffix + ".repair")
    try:
        with temp.open("wb") as f:
            f.write(b"\n".join(kept))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except OSError as exc:
        log.error("could not repair %s: %s", path, exc)
        temp.unlink(missing_ok=True)
        return 0

    log.warning(
        "%s: removed %d row(s) damaged by an interrupted write; %d kept",
        path.name, removed, len(kept),
    )
    return removed


class EventLogger:
    """Appends :class:`~core.events.Event` rows to a CSV file, synced after each write."""

    def __init__(self, csv_path: Union[str, Path]) -> None:
        """Open (creating if needed) the CSV file at `csv_path`, writing a header if new.

        Repairs damage from a previous unclean shutdown on the way in. Doing it
        here rather than on read means every consumer — the dashboard, the
        exporter, a person with a spreadsheet — sees a clean file, instead of
        each having to tolerate corruption separately.
        """
        self._path = Path(csv_path)
        # The inference loop and the health monitor (monitoring-state changes)
        # both write rows; one writer at a time keeps every row whole.
        self._write_lock = threading.Lock()
        self._errors = _WriteErrors(f"event log {self._path}")
        self._file: Any = None    # open text file, or None while it cannot be opened
        self._writer: Any = None
        self._last_open_attempt = -math.inf
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._errors.report(exc)
        repair(self._path)
        self._open()

    def _open(self) -> bool:
        """Open for append, writing the header to a new or empty file.

        False (and logged) when the file cannot be opened -- a read-only card
        must not stop the device booting; log() retries now and then.
        """
        self._last_open_attempt = time.monotonic()
        handle = None
        try:
            needs_header = _is_empty_or_missing(self._path)
            handle = self._path.open("a", newline="")
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(_HEADER)
                handle.flush()
        except OSError as exc:
            self._errors.report(exc)
            if handle is not None:
                _close_quietly(handle)
            return False
        self._file, self._writer = handle, writer
        if needs_header:
            self._sync()
        return True

    def _ensure_open(self) -> bool:
        """Whether the file is open, retrying an earlier failure once a minute."""
        if self._file is not None:
            return True
        if time.monotonic() - self._last_open_attempt < _ERROR_LOG_EVERY_S:
            return False
        return self._open()

    def log(self, event: Event) -> None:
        """Append one event as a CSV row and sync it to the card. Never raises
        on a write failure: the row is dropped and the failure logged."""
        with self._write_lock:
            self._log(event)

    def _log(self, event: Event) -> None:
        if not self._ensure_open():
            return
        try:
            self._writer.writerow(
                [
                    event.timestamp.isoformat(),
                    event.type,
                    f"{event.duration:.3f}",
                    "" if event.confidence is None else f"{event.confidence:.3f}",
                    event.module_name,
                    event.contributing_signals,
                ]
            )
            self._file.flush()
        except OSError as exc:
            self._errors.report(exc)
            return
        self._fsync()

    def _sync(self) -> None:
        """Flush to the OS and then to the card.

        Both halves are needed: flush() alone leaves the data in a buffer the
        kernel may write after it has already extended the file, which is how a
        power cut produces a longer file full of NULs.
        """
        try:
            self._file.flush()
        except OSError as exc:
            self._errors.report(exc)
            return
        self._fsync()

    def _fsync(self) -> None:
        try:
            os.fsync(self._file.fileno())
        except OSError as exc:
            # A failed sync is worth knowing about — it usually means the card
            # is failing — but it must not stop the device monitoring.
            log.warning("could not sync the event log: %s", exc)

    def close(self) -> None:
        """Flush, sync and close the underlying file. Never raises on I/O errors."""
        if self._file is None:
            return
        try:
            self._sync()
        finally:
            _close_quietly(self._file)
            self._file = None

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
