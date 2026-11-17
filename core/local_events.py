"""The device's own bounded event journal, served to enrolled phones (sd-local-v1).

Why a journal and not `data/events.csv`: the CSV has no event identity
(docs/EVENT_PIPELINE.md §7), so a phone reading it could never merge local
history with the backend's without duplicates. Every Event already carries
`event_id`, which the uplink sends as `event_key` and the backend stores once;
this journal records the SAME key, so the app can show one row per occurrence
whether it learnt of it from the device or from the account.

Bounded on purpose (an SD card, a device that runs for months):
* at most `max_events` rows are kept; the file is compacted atomically past
  1.25x that, or past `max_bytes`;
* one short append per event; events are seconds to minutes apart.

Privacy: the same metadata the uplink already carries (type, times,
confidence, signals text, duration, severity, alert-output report). No frame,
no image, no path to one -- pinned by tests/test_m12_local_events.py.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Optional

from core.config_lkg import atomic_write_bytes

SCHEMA = "sd-local-events/1"
DEFAULT_MAX_EVENTS = 2000
DEFAULT_MAX_BYTES = 1_000_000
#: Alert deliveries kept in memory so a buzzer report that lands after the event
#: (up to ~3.5 s) still reaches the row that is served.
_DELIVERY_MEMORY = 200
#: Fields a row may carry. Anything else is dropped at write and at read.
_FIELDS = ("seq", "event_key", "type", "timestamp", "severity", "confidence", "signals", "duration",
           "module", "experimental", "simulated", "alert", "boot", "mono")


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return value


class LocalEventJournal:
    def __init__(self, path: Optional[str | os.PathLike], *, max_events: int = DEFAULT_MAX_EVENTS,
                 max_bytes: int = DEFAULT_MAX_BYTES, boot_id: Optional[str] = None,
                 clock=time.monotonic) -> None:
        self.path = Path(path) if path else None
        self.max_events = max(10, int(max_events))
        self.max_bytes = max(10_000, int(max_bytes))
        self.boot_id = boot_id or os.urandom(4).hex()
        self._clock = clock
        self._lock = threading.Lock()
        self._rows: deque[dict] = deque(maxlen=self.max_events)
        self._deliveries: OrderedDict[str, Any] = OrderedDict()
        self._seq = 0
        self._lines_on_disk = 0
        self.fault: Optional[str] = None
        #: Set by the runtime: every row from a simulated device says so.
        self.simulated = False
        self._load()

    # ------------------------------------------------------------------ storage
    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with self.path.open("rb") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            self.fault = f"the local event journal could not be read ({type(exc).__name__})"
            return
        for raw in lines:
            try:
                row = json.loads(raw)
            except ValueError:
                continue  # a torn last line after a power cut
            if isinstance(row, dict) and isinstance(row.get("seq"), int) and isinstance(row.get("event_key"), str):
                self._rows.append({k: row[k] for k in _FIELDS if k in row})
                self._seq = max(self._seq, row["seq"])
        self._lines_on_disk = len(lines)

    def _append_line(self, row: dict) -> None:
        if self.path is None:
            return
        line = (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            self._lines_on_disk += 1
            self.fault = None
        except OSError as exc:
            self.fault = f"the local event journal could not be written ({type(exc).__name__})"
            return
        size = 0
        with contextlib.suppress(OSError):
            size = self.path.stat().st_size
        if self._lines_on_disk > self.max_events * 1.25 or size > self.max_bytes:
            self._compact()

    def _compact(self) -> None:
        if self.path is None:
            return
        rows = list(self._rows)
        body = b""
        while rows:
            body = b"".join((json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
                            for r in rows)
            if len(body) <= self.max_bytes * 0.8 or len(rows) <= 10:
                break
            rows = rows[len(rows) // 4:]
        if atomic_write_bytes(self.path, body, mode=0o600):
            self._lines_on_disk = len(rows)
            while len(self._rows) > len(rows):
                self._rows.popleft()

    # ------------------------------------------------------------------ writing
    def record(self, event, severity: str, *, delivery: Any = None) -> Optional[dict]:
        """Journal one Event under its own `event_id`. Never raises into the caller."""
        try:
            from core.behavior.experimental import is_experimental

            key = getattr(event, "event_id", None)
            if not isinstance(key, str) or not 0 < len(key) <= 64:
                return None
            experimental = bool(is_experimental(event.type))
            with self._lock:
                self._seq += 1
                row = {
                    "seq": self._seq,
                    "event_key": key,
                    "type": str(event.type)[:64],
                    "timestamp": event.timestamp.isoformat(),
                    # An experimental family is an OBSERVATION: always "log" (M8.4 decision A).
                    "severity": "log" if experimental else str(severity)[:16],
                    "confidence": None if event.confidence is None else round(float(_finite(event.confidence)
                                                                                    or 0.0), 3),
                    "signals": str(event.contributing_signals or "")[:500],
                    "duration": round(float(_finite(event.duration) or 0.0), 3),
                    "module": str(getattr(event, "module_name", "") or "")[:64],
                    "experimental": experimental,
                    "simulated": bool(self.simulated),
                    "boot": self.boot_id,
                    "mono": round(self._clock(), 3),
                }
                summary = _summary(delivery)
                if summary is not None:
                    row["alert"] = summary
                if delivery is not None:
                    self._deliveries[key] = delivery
                    while len(self._deliveries) > _DELIVERY_MEMORY:
                        self._deliveries.popitem(last=False)
                self._rows.append(row)
                self._append_line(row)
                return dict(row)
        except Exception as exc:  # noqa: BLE001 - history must never cost the event or the alert
            self.fault = f"an event could not be journaled ({type(exc).__name__})"
            return None

    # ------------------------------------------------------------------ reading
    def page(self, *, after: Optional[int] = None, limit: int = 100,
             pending_keys: Optional[set[str]] = None, uplink: bool = False) -> dict:
        """`after` None: the newest `limit` rows, newest first. `after` N: rows with
        seq > N, oldest first (what a phone polls for new alerts)."""
        limit = max(1, min(500, int(limit)))
        now = self._clock()
        with self._lock:
            rows = list(self._rows)
            deliveries = dict(self._deliveries)
            latest = self._seq
        if after is None:
            chosen = rows[-limit:][::-1]
        else:
            chosen = [r for r in rows if r["seq"] > after][:limit]
        pending = pending_keys or set()
        out = []
        for r in chosen:
            item = {k: r.get(k) for k in ("seq", "event_key", "type", "timestamp", "severity", "confidence",
                                          "signals", "duration", "experimental", "simulated")}
            live = _summary(deliveries.get(r["event_key"]))
            item["alert"] = live if live is not None else r.get("alert")
            # Seconds since it happened, by the device's monotonic clock: right
            # whatever the wall clock says, but only within the boot that made it.
            item["age_s"] = round(max(0.0, now - float(r.get("mono") or 0.0)), 1) \
                if r.get("boot") == self.boot_id else None
            if not uplink:
                item["sync"] = "local_only"
            elif r["event_key"] in pending:
                item["sync"] = "pending"
            else:
                item["sync"] = "sent"
            out.append(item)
        oldest = rows[0]["timestamp"] if rows else None
        return {"schema": SCHEMA, "events": out, "latest_seq": latest, "boot_id": self.boot_id,
                "retention": {"max_events": self.max_events, "kept": len(rows), "oldest": oldest},
                "fault": self.fault}


def _summary(delivery: Any) -> Optional[dict]:
    if delivery is None:
        return None
    summary = getattr(delivery, "summary", None)
    if not callable(summary):
        return None
    try:
        value = summary()
    except Exception:  # noqa: BLE001 - a report that cannot be read is simply absent
        return None
    if not isinstance(value, dict):
        return None
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return None
