"""Always-on, bounded per-stage latency for the device's frame pipeline (M10 OBSERVE).

Every stage of one analysed frame is timed with the monotonic clock and kept in a
fixed-size ring per stage, so the device can answer "where does the time go" from
its own running loop, not only from an out-of-process harness
(scripts/hw_validation/stage_timing.py `stages`, which stops the service).

Bounded on purpose:

  * each stage owns ONE preallocated `array('d')` ring of `WINDOW` samples; a
    sample overwrites the oldest, so nothing grows with uptime;
  * the set of stages is fixed (`STAGES`); an unknown name is refused, never
    added, so a typo cannot leak a new ring per call;
  * `add()` is a lock, two stores and three integer updates. Percentiles are
    computed only when somebody reads (`/api/health`), from a sorted copy of
    the window, off the inference thread.

Measured cost of `add()` is in docs/DIAGNOSTICS.md (`tests/test_m10_stage_timing.py
::test_the_instrumentation_is_cheap` re-measures it). Numbers from a Mac are
labelled as such: `platform_tag()` says which machine produced the figures, and
`/api/health` carries it next to them, so a desktop number can never be read as
a Raspberry Pi number.

The stage names are the contract (docs/DIAGNOSTICS.md §2):

  camera_read          one Camera.read() in the capture thread (sensor + ISP + copy)
  frame_age            capture -> the inference loop picks the frame up
  detector_total       one shared detector run (predict), whatever its backend
  hailo_preprocess     letterbox + colour conversion before the chip (HailoYOLO)
  hailo_inference      pipeline.infer(): host -> chip -> host (HailoYOLO only)
  decoder_cpu          off-chip decode + NMS of the raw heads (HailoYOLO only)
  face_landmarks       one MediaPipe Face Mesh process()
  hands                one MediaPipe Hands process()
  tracking             the shared tracker's association for one detector run
  behaviour_reasoning  module.process() minus every model call and the tracker
  alert_dispatch       the alert policy for one frame's events (command issued)
  alert_output         alert command -> GPIO pin driven (gpio_buzzer only)
  frame_to_decision    capture -> the frame's alert decision is made
  frame_interval       between two analysed frames (-> processed FPS)
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
from array import array
from typing import Optional

SCHEMA = "sd-stage-timing/1"
#: Samples kept per stage. 256 x 8 bytes x 14 stages = 28 KiB, allocated once.
WINDOW = 256

STAGES = (
    "camera_read",
    "frame_age",
    "detector_total",
    "hailo_preprocess",
    "hailo_inference",
    "decoder_cpu",
    "face_landmarks",
    "hands",
    "tracking",
    "behaviour_reasoning",
    "alert_dispatch",
    "alert_output",
    "frame_to_decision",
    "frame_interval",
)


class RollingStat:
    """One stage: the last `window` samples in milliseconds, plus lifetime counters."""

    __slots__ = ("_buf", "_count", "_idx", "_last", "_window", "max_ever")

    def __init__(self, window: int = WINDOW) -> None:
        self._window = int(window)
        self._buf = array("d", bytes(8 * self._window))  # zero-filled, never resized
        self._idx = 0
        self._count = 0
        self._last = 0.0
        self.max_ever = 0.0

    def add(self, ms: float) -> None:
        self._buf[self._idx] = ms
        self._idx = (self._idx + 1) % self._window
        self._count += 1
        self._last = ms
        if ms > self.max_ever:
            self.max_ever = ms

    @property
    def count(self) -> int:
        return self._count

    def window_values(self) -> list[float]:
        n = min(self._count, self._window)
        if n < self._window:
            return list(self._buf[:n])
        return list(self._buf)

    def summary(self) -> Optional[dict]:
        """p50/p95/max/mean over the window, or None before the first sample."""
        values = self.window_values()
        if not values:
            return None
        values.sort()
        n = len(values)
        return {
            "n": n,
            "p50_ms": round(_quantile(values, 0.50), 3),
            "p95_ms": round(_quantile(values, 0.95), 3),
            "max_ms": round(values[-1], 3),
            "mean_ms": round(sum(values) / n, 3),
            "last_ms": round(self._last, 3),
            "count": self._count,
        }


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted, non-empty list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class StageTimings:
    """The fixed set of stage rings. Thread-safe: writers are the capture thread,
    the inference loop and the alert player; the reader is the HTTP thread."""

    def __init__(self, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._stats = {name: RollingStat(window) for name in STAGES}
        self.window = int(window)
        self.started_at = time.monotonic()

    def add(self, stage: str, ms: float) -> bool:
        """Record one sample; False (and nothing recorded) for an unknown stage or a
        non-finite / negative value."""
        stat = self._stats.get(stage)
        if stat is None or not (ms >= 0.0) or ms == float("inf"):
            return False
        with self._lock:
            stat.add(float(ms))
        return True

    def add_since(self, stage: str, started: float) -> float:
        """Record `time.perf_counter() - started` (seconds) as milliseconds; returns it."""
        ms = (time.perf_counter() - started) * 1000.0
        self.add(stage, ms)
        return ms

    def count(self, stage: str) -> int:
        stat = self._stats.get(stage)
        return 0 if stat is None else stat.count

    def reset(self) -> None:
        with self._lock:
            self._stats = {name: RollingStat(self.window) for name in STAGES}
            self.started_at = time.monotonic()

    def snapshot(self) -> dict:
        """Every stage's summary (None for a stage with no sample yet) + processed FPS."""
        with self._lock:
            stages = {name: stat.summary() for name, stat in self._stats.items()}
        interval = stages.get("frame_interval")
        fps = None
        if interval and interval["mean_ms"] > 0:
            fps = round(1000.0 / interval["mean_ms"], 2)
        return {"stages": stages, "processed_fps": fps}


#: The process-wide registry. core.accel and core.alerts record into it without
#: plumbing (like core.accel's inference counters); the runtime reads it.
TIMINGS = StageTimings()


def record(stage: str, ms: float) -> None:
    TIMINGS.add(stage, ms)


# ---------------------------------------------------------------------- the host
def _read_text(path: str, limit: int = 256) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit).decode("utf-8", "replace").strip("\x00\n ") or None
    except OSError:
        return None


_PLATFORM_CACHE: Optional[dict] = None


def platform_tag(*, simulated: bool = False) -> dict:
    """Which machine produced these timings, so a Mac figure is never read as a Pi's.

    `class` is the one field to look at:
      pi            a Raspberry Pi (from /proc/device-tree/model)
      desktop       anything else (a Mac, a Linux workstation): NOT a Pi measurement
      simulated     the frames or the accelerator are stand-ins (sim/, fake_hailo)
    `hailo` says whether a real Hailo runtime is loaded (the fake reports "simulated").
    """
    global _PLATFORM_CACHE  # noqa: PLW0603 - the host does not change while the process runs
    if _PLATFORM_CACHE is None:
        model = _read_text("/proc/device-tree/model")
        _PLATFORM_CACHE = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "board": model,
            "is_raspberry_pi": bool(model and "raspberry pi" in model.lower()),
        }
    tag = dict(_PLATFORM_CACHE)
    hailo_mod = sys.modules.get("hailo_platform")
    if hailo_mod is None:
        hailo = "not loaded"
    elif getattr(hailo_mod, "SIMULATED", False):
        hailo = "simulated"
        simulated = True
    else:
        hailo = "loaded"
    tag["hailo_runtime"] = hailo
    tag["simulated"] = bool(simulated)
    tag["class"] = "simulated" if simulated else ("pi" if tag["is_raspberry_pi"] else "desktop")
    tag["note"] = {
        "pi": "measured on a Raspberry Pi",
        "desktop": "measured on a desktop host: NOT a Raspberry Pi measurement",
        "simulated": "SIMULATED inputs or accelerator: not a hardware measurement",
    }[tag["class"]]
    return tag


def rss_mb() -> tuple[Optional[float], str]:
    """This process's resident memory (MiB) and how it was read.

    Linux: current RSS from /proc/self/statm. Elsewhere: the PEAK RSS from
    getrusage (macOS has no cheap current-RSS read without psutil), said so.
    """
    statm = _read_text("/proc/self/statm")
    if statm:
        try:
            pages = int(statm.split()[1])
            return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1), "current (/proc/self/statm)"
        except (IndexError, ValueError, OSError):
            pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # bytes on macOS, KiB on Linux/BSD
        mb = peak / 2**20 if sys.platform == "darwin" else peak / 1024
        return round(mb, 1), "peak (getrusage)"
    except (ImportError, OSError, ValueError):
        return None, "unavailable"


def report(*, system_metrics: Optional[dict] = None, simulated: bool = False,
           timings: Optional[StageTimings] = None) -> dict:
    """The `performance` block of /api/health (schema sd-stage-timing/1).

    `system_metrics` is device_health's `system.metrics` from the same health
    sample (temperature and throttle state are read there once, not twice).
    """
    timings = TIMINGS if timings is None else timings
    snap = timings.snapshot()
    rss, rss_source = rss_mb()
    sysm = system_metrics or {}
    return {
        "schema": SCHEMA,
        "platform": platform_tag(simulated=simulated),
        "window_samples": timings.window,
        "clock": "time.perf_counter / time.monotonic (ms)",
        "stages": snap["stages"],
        "processed_fps": snap["processed_fps"],
        "rss_mb": rss,
        "rss_source": rss_source,
        "soc_temperature_c": sysm.get("temperature_c"),
        "throttle_now": sysm.get("throttle_now"),
        "throttled_raw": sysm.get("throttled"),
        "uptime_s": round(time.monotonic() - timings.started_at, 1),
    }
