"""M8.4 fault injection for the digital twin: a bad link, a full disk, a damaged file.

Everything here wraps a real seam and is deterministic (seeded, or driven by the
SimClock); nothing sleeps. What each fault does to the PRODUCTION code is real:
the uplink's own retry, dedupe and spool, the backend's own ingest, the event
log's own error path. Only the fault itself is simulated.

  LinkFaults        a `requests`-shaped transport between the device's uplink and
                    the backend that loses requests, loses answers after the
                    backend stored them, duplicates deliveries, and holds
                    requests past the device's timeout to deliver them LATE
                    (after newer ones: reordering). Seeded, so a run is repeatable.
  EventLedger       every event the device queued (its event_key), for comparing
                    "generated on the device" with "stored by the backend".
  small_volume      a real, tiny filesystem (macOS hdiutil) to fill up: genuine
                    ENOSPC from the kernel, not a mocked exception.
  read_only         a directory (and its files) made unwritable for a block.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------- the link
@dataclass
class LinkStats:
    sent: int = 0
    lost_requests: int = 0
    lost_answers: int = 0
    duplicated: int = 0
    held: int = 0
    delivered_late: int = 0
    #: Each POST the backend actually received: (path, number of events in it).
    deliveries: list[tuple[str, int]] = field(default_factory=list)


class LinkFaults:
    """A lossy, duplicating, reordering link (deterministic: seeded RNG).

    Rates are per request, in [0, 1]:
      loss          the request never reaches the backend (ConnectionError)
      answer_loss   the backend stores it, the answer is lost (ConnectionError)
      duplicate     the request is delivered twice (a retransmitting proxy);
                    the device sees the second answer
      hold          the request is held past the device's timeout (Timeout):
                    it reaches the backend LATER, at the first request sent
                    at least `hold_s` after it -- after newer requests
    `clock` (anything with monotonic()) times the holds; the SimClock in tests.
    `only_events` limits the faults to requests that carry events.
    """

    def __init__(self, inner: Any, *, seed: int = 0, loss: float = 0.0, answer_loss: float = 0.0,
                 duplicate: float = 0.0, hold: float = 0.0, hold_s: float = 10.0,
                 clock: Optional[Any] = None, only_events: bool = False) -> None:
        self._inner = inner
        self._rng = random.Random(seed)  # noqa: S311 - reproducible fault injection, not crypto
        self.loss, self.answer_loss, self.duplicate, self.hold = loss, answer_loss, duplicate, hold
        self.hold_s = float(hold_s)
        self._clock = clock
        self.only_events = only_events
        self.enabled = True
        self.stats = LinkStats()
        self._held: list[tuple[float, str, dict]] = []

    def _now(self) -> float:
        import time

        return float(self._clock.monotonic()) if self._clock is not None else time.monotonic()

    def _deliver(self, url: str, kwargs: dict):
        events = (kwargs.get("json") or {}).get("events") if isinstance(kwargs.get("json"), dict) else None
        self.stats.deliveries.append((url.rsplit("/", 1)[-1], len(events or [])))
        return self._inner.post(url, **kwargs)

    def release_held(self, *, force: bool = False) -> int:
        """Deliver every held request whose time has come (all of them with force)."""
        now = self._now()
        due = [h for h in self._held if force or now - h[0] >= self.hold_s]
        self._held = [h for h in self._held if h not in due]
        for _, url, kwargs in due:
            self.stats.delivered_late += 1
            with contextlib.suppress(Exception):  # nobody is waiting for this answer
                self._deliver(url, kwargs)
        return len(due)

    @property
    def pending(self) -> int:
        return len(self._held)

    def post(self, url: str, **kwargs):
        import requests

        self.release_held()
        self.stats.sent += 1
        body = kwargs.get("json")
        carries_events = isinstance(body, dict) and bool(body.get("events"))
        if not self.enabled or (self.only_events and not carries_events):
            return self._deliver(url, kwargs)
        roll = self._rng.random()
        if roll < self.loss:
            self.stats.lost_requests += 1
            raise requests.ConnectionError(f"SIMULATED lossy link: request to {url} lost")
        roll -= self.loss
        if roll < self.hold:
            self.stats.held += 1
            # The device's own copy is re-sent after its timeout; this one is delivered later.
            self._held.append((self._now(), url, copy.deepcopy(kwargs)))
            raise requests.Timeout(f"SIMULATED slow link: no answer from {url} within the timeout")
        roll -= self.hold
        if roll < self.duplicate:
            self.stats.duplicated += 1
            self._deliver(url, kwargs)
            return self._deliver(url, kwargs)
        roll -= self.duplicate
        response = self._deliver(url, kwargs)
        if roll < self.answer_loss:
            self.stats.lost_answers += 1
            raise requests.ConnectionError(f"SIMULATED lossy link: answer from {url} lost after it was stored")
        return response


# ---------------------------------------------------------------------- what the device generated
class _LedgerDeque(collections.deque):
    """The uplink queue, recording every NEW event appended (push_event appends;
    a restore from the spool or a re-queue extends, and is not new)."""

    ledger: Optional["EventLedger"] = None

    def append(self, item) -> None:  # type: ignore[override]
        ledger = self.ledger
        if ledger is not None and isinstance(item, dict):
            ledger.record(item)
        super().append(item)


class EventLedger:
    """Every event the device queued for the backend, by event_key.

    In memory, and appended to `path` (JSON lines, flushed per line) when given,
    so a ledger survives a killed device process.
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[dict] = []
        self._lock = threading.Lock()

    def record(self, item: dict) -> None:
        entry = {"event_key": item.get("event_key"), "type": item.get("type"),
                 "timestamp": item.get("timestamp"), "source": item.get("source")}
        with self._lock:
            self.entries.append(entry)
            if self.path is not None:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()

    def attach(self, uplink) -> None:
        """Record every event `uplink` (a CloudUplink) queues from now on."""
        queue = _LedgerDeque(uplink._queue, maxlen=uplink._queue.maxlen)
        queue.ledger = self
        uplink._queue = queue

    def keys(self) -> list[str]:
        return [e["event_key"] for e in self.entries]

    @staticmethod
    def read(path: Path | str) -> list[dict]:
        out = []
        with contextlib.suppress(FileNotFoundError):
            for line in Path(path).read_text().splitlines():
                with contextlib.suppress(ValueError):
                    out.append(json.loads(line))
        return out


def compare(generated: list[dict], stored: list[dict]) -> dict:
    """Device-generated events vs backend-stored rows, by event_key."""
    made = collections.Counter(e["event_key"] for e in generated if e.get("event_key"))
    kept = collections.Counter(r.get("event_key") for r in stored if r.get("event_key"))
    by_time = collections.Counter((r.get("type"), r.get("timestamp")) for r in stored)
    return {
        "generated": sum(made.values()), "generated_unique": len(made),
        "stored": sum(kept.values()), "stored_unique": len(kept),
        "lost": sorted(set(made) - set(kept)),
        "unexpected": sorted(set(kept) - set(made)),
        "duplicate_keys": sorted(k for k, n in kept.items() if n > 1),
        "duplicate_type_timestamp": sorted(f"{t}@{ts}" for (t, ts), n in by_time.items() if n > 1),
    }


# ---------------------------------------------------------------------- storage
HDIUTIL = "/usr/bin/hdiutil"


def can_make_volumes() -> bool:
    return sys.platform == "darwin" and os.path.exists(HDIUTIL)


@contextlib.contextmanager
def small_volume(size_kb: int = 2048, *, root: Optional[Path] = None) -> Iterator["Volume"]:
    """A real HFS+ volume of `size_kb`, mounted for the block (macOS only).

    Writes past its end fail in the kernel with ENOSPC, exactly as on a full SD
    card; nothing is mocked. Always detached on the way out.
    """
    if not can_make_volumes():
        raise RuntimeError("small_volume needs macOS hdiutil")
    base = Path(tempfile.mkdtemp(prefix="sdvol-", dir=str(root) if root else None))
    image, mount = base / "vol.dmg", base / "mnt"
    subprocess.run([HDIUTIL, "create",  # noqa: S603 - fixed argv, our own temp paths "-size", f"{int(size_kb)}k", "-fs", "HFS+", "-volname", "sdtwin",
                    "-layout", "NONE", str(image), "-quiet"], check=True, timeout=60)
    mount.mkdir()
    subprocess.run([HDIUTIL, "attach", str(image), "-mountpoint", str(mount),  # noqa: S603 - fixed argv
                    "-nobrowse", "-quiet"],
                   check=True, timeout=60)
    try:
        yield Volume(mount)
    finally:
        subprocess.run([HDIUTIL, "detach", str(mount), "-force", "-quiet"],  # noqa: S603 - fixed argv
                       check=False, timeout=60)
        shutil.rmtree(base, ignore_errors=True)


class Volume:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._ballast = path / ".ballast"

    def free_bytes(self) -> int:
        st = os.statvfs(self.path)
        return st.f_bavail * st.f_frsize

    def fill(self, leave: int = 0) -> int:
        """Fill the volume, leaving about `leave` bytes; returns what is left."""
        want = max(0, self.free_bytes() - int(leave))
        with open(self._ballast, "ab") as handle, contextlib.suppress(OSError):
            handle.write(b"\0" * want)
            handle.flush()
            os.fsync(handle.fileno())
        return self.free_bytes()

    def free(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._ballast.unlink()


@contextlib.contextmanager
def read_only(directory: Path, *, files: bool = True) -> Iterator[None]:
    """`directory` (and, with files, every file in it) unwritable for the block."""
    directory = Path(directory)
    saved: dict[Path, int] = {}
    targets = [directory] + ([p for p in directory.iterdir() if p.is_file()] if files else [])
    for path in targets:
        saved[path] = stat.S_IMODE(path.stat().st_mode)
        path.chmod(saved[path] & ~0o222)
    try:
        yield
    finally:
        for path, mode in saved.items():
            with contextlib.suppress(OSError):
                path.chmod(mode)


def running_as_root() -> bool:
    geteuid: Optional[Callable[[], int]] = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0
