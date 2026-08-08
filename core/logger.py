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
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Union

from core.events import Event
from core.logging_setup import get_logger

log = get_logger("logger")

_HEADER = ["timestamp", "type", "duration_s", "confidence", "module", "contributing_signals"]


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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        repair(self._path)
        is_new = not self._path.exists()
        self._file = self._path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(_HEADER)
            self._sync()

    def log(self, event: Event) -> None:
        """Append one event as a CSV row and sync it to the card."""
        self._writer.writerow(
            [
                event.timestamp.isoformat(),
                event.type,
                f"{event.duration:.3f}",
                f"{event.confidence:.3f}",
                event.module_name,
                event.contributing_signals,
            ]
        )
        self._sync()

    def _sync(self) -> None:
        """Flush to the OS and then to the card.

        Both halves are needed: flush() alone leaves the data in a buffer the
        kernel may write after it has already extended the file, which is how a
        power cut produces a longer file full of NULs.
        """
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError as exc:
            # A failed sync is worth knowing about — it usually means the card
            # is failing — but it must not stop the device monitoring.
            log.warning("could not sync the event log: %s", exc)

    def close(self) -> None:
        """Flush, sync and close the underlying file."""
        try:
            self._sync()
        finally:
            self._file.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
