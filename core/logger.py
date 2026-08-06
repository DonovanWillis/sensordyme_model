"""Appends detected events to a CSV file. Never writes frames or video to disk."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from core.events import Event

_HEADER = ["timestamp", "type", "duration_s", "confidence", "module", "contributing_signals"]


class EventLogger:
    """Appends :class:`~core.events.Event` rows to a CSV file, flushing after each write."""

    def __init__(self, csv_path: Union[str, Path]) -> None:
        """Open (creating if needed) the CSV file at `csv_path`, writing a header if new."""
        self._path = Path(csv_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self._path.exists()
        self._file = self._path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(_HEADER)
            self._file.flush()

    def log(self, event: Event) -> None:
        """Append one event as a CSV row and flush immediately."""
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
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
