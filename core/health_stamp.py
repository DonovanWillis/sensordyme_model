"""Provenance for health snapshots (M8.1): which sample this is, and how old.

A device kept polling while re-sending a frozen health snapshot, and the
backend took every copy as current: `last_seen` moved, the snapshot did not,
and the app said "protecting" on evidence minutes old. A successful poll proves
the link, not the monitoring.

Each snapshot is stamped where it is GENERATED (services/runtime.py
`_collect_health`), not where it is sent:

    "stamp": {"schema": "sd-health-v1", "boot_id": "9f1c...", "generation": 42,
              "revision": 0, "generated_wall": 1731400000.1, "generated_mono": 812.4}

* `boot_id`: random per process start. A reboot is a new boot_id, so the
  generation counter restarting at 1 is a reboot, not an old report.
* `generation`: +1 per sample within a boot. The same generation sent twice is
  the same evidence, however many polls carry it.
* `revision`: +1 each time the uplink refreshes only the monitoring keys of the
  same sample (`CloudUplink.refresh_health_monitoring`). Newer content, but NOT
  a newer sample: freshness stays that of the sample.
* `generated_mono`: the device's monotonic clock at generation. Only ever
  compared with the same process's monotonic clock.
* `generated_wall`: the device's wall clock at generation. Informational: a Pi
  has no RTC and may boot with a wrong clock, so nothing judges freshness by it.

At send time the uplink adds `age_s = monotonic_now - generated_mono` to the
copy it sends (never to the held snapshot). The backend turns that into an
observation time on ITS clock (`received_at - age_s`), so neither side needs
the device's wall clock to be right. docs/APP_STATUS_CONTRACT.md "Health
freshness" has the backend half.

Shared by the device and the backend, so standard library only.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

SCHEMA = "sd-health-v1"

#: Longest boot id accepted; ours are 16 hex characters.
_MAX_BOOT_ID = 64
#: A generation or revision beyond this is not a counter, it is garbage.
_MAX_COUNTER = 2**53
#: An age beyond ~10 years is clamped: it is stale either way.
_MAX_AGE_S = 315_360_000.0
#: A slightly negative age (float rounding) is read as 0; anything below this
#: means the stamp is not from a monotonic clock and is refused.
_NEGATIVE_AGE_TOLERANCE_S = 1.0


def _now_mono() -> float:
    # Looked up at call time: the deterministic tests replace time.monotonic.
    return time.monotonic()


def _now_wall() -> float:
    return time.time()


class HealthStamper:
    """Stamps the snapshots of one process (one boot)."""

    def __init__(self, *, boot_id: Optional[str] = None,
                 monotonic: Callable[[], float] = _now_mono,
                 wall: Callable[[], float] = _now_wall) -> None:
        self.boot_id = boot_id or secrets.token_hex(8)
        self._monotonic = monotonic
        #: Replaceable: the clock-jump tests move the device's wall clock.
        self.wall = wall
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def generation(self) -> int:
        return self._generation

    def stamp(self, report: dict) -> dict:
        """`report` with a new stamp (the report is updated in place and returned)."""
        with self._lock:
            self._generation += 1
            generation = self._generation
        report["stamp"] = {
            "schema": SCHEMA,
            "boot_id": self.boot_id,
            "generation": generation,
            "revision": 0,
            "generated_wall": _finite_or_none(self.wall()),
            "generated_mono": float(self._monotonic()),
        }
        return report


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def with_send_age(health: Any, monotonic_now: float) -> Any:
    """The copy of `health` to send: its stamp gains `age_s` (device monotonic).

    Unstamped or malformed health is returned unchanged; the held snapshot is
    never modified.
    """
    if not isinstance(health, dict):
        return health
    stamp = health.get("stamp")
    if not isinstance(stamp, dict):
        return health
    generated = _finite_or_none(stamp.get("generated_mono"))
    if generated is None:
        return health
    age = max(0.0, float(monotonic_now) - generated)
    return {**health, "stamp": {**stamp, "age_s": round(age, 3)}}


def revised(health: Any) -> Any:
    """`health` with its stamp's revision raised (the monitoring keys changed)."""
    if not isinstance(health, dict):
        return health
    stamp = health.get("stamp")
    if not isinstance(stamp, dict):
        return health
    revision = stamp.get("revision")
    revision = revision if isinstance(revision, int) and not isinstance(revision, bool) else 0
    return {**health, "stamp": {**stamp, "revision": revision + 1}}


@dataclass(frozen=True)
class Stamp:
    boot_id: str
    generation: int
    revision: int
    #: Seconds since generation by the device's monotonic clock, at send time.
    #: None when the sender did not say (receipt time alone then ages it).
    age_s: Optional[float]
    #: Device wall clock at generation; informational only.
    generated_wall: Optional[float]

    def order(self) -> tuple[int, int]:
        return (self.generation, self.revision)


class StampError(ValueError):
    """A stamp is present but unusable: the snapshot's provenance is unknown."""


def _counter(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_COUNTER:
        raise StampError(f"{name} is not a counter")
    return value


def parse(health: Any) -> Optional[Stamp]:
    """The stamp of a health report. None when there is none (older firmware).

    Raises StampError when a stamp is present but cannot be trusted: a wrong
    schema, a missing boot id, a counter that is not a counter, a negative age.
    """
    if not isinstance(health, dict) or "stamp" not in health:
        return None
    raw = health.get("stamp")
    if not isinstance(raw, dict):
        raise StampError("stamp is not an object")
    if raw.get("schema") != SCHEMA:
        raise StampError("unknown stamp schema")
    boot_id = raw.get("boot_id")
    if not isinstance(boot_id, str) or not boot_id or len(boot_id) > _MAX_BOOT_ID:
        raise StampError("boot_id missing")
    generation = _counter(raw.get("generation"), "generation", minimum=1)
    revision = _counter(raw.get("revision", 0), "revision", minimum=0)
    age: Optional[float] = None
    if raw.get("age_s") is not None:
        age = _finite_or_none(raw.get("age_s"))
        if age is None or age < -_NEGATIVE_AGE_TOLERANCE_S:
            raise StampError("age_s is not a non-negative number")
        age = min(max(age, 0.0), _MAX_AGE_S)
    return Stamp(boot_id=boot_id, generation=generation, revision=revision, age_s=age,
                 generated_wall=_finite_or_none(raw.get("generated_wall")))
