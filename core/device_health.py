"""Device health: what this Pi can honestly say about itself, in consumer terms.

The app shows a customer (and, at a demo, an investor) whether the camera, the
AI and the network are working. Those indicators are worth nothing unless they
come from the device — a green light driven by "a row exists in the database"
says only that the device was once registered, and it stays green after the Pi
is unplugged.

So every field here is derived from something observed on the device this
second:

  camera        frames actually arriving on the FrameBus, by age
  inference     the inference loop's heartbeat, by age
  accelerator   completed Hailo inferences, not merely a seated HAT
  network       the live Wi-Fi association and signal
  system        temperature, load and memory read from /proc and vcgencmd

Anything that cannot be determined is reported as "unknown" rather than
defaulted to healthy. A monitoring product that guesses in its own favour is
worse than one that admits it doesn't know.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from core import accel
from core.thermal import SOFT_THROTTLE_C

# States the app understands. Kept deliberately small so every surface can
# render the same vocabulary.
OK = "ok"
WARNING = "warning"
FAILED = "failed"
UNKNOWN = "unknown"

# A frame or heartbeat older than this means the pipeline has stalled, not that
# it is merely between ticks.
_STALE_S = 5.0


@dataclass
class Component:
    """One subsystem's state plus the sentence explaining it."""

    state: str = UNKNOWN
    detail: str = ""
    metrics: dict = field(default_factory=dict)


def _read_temperature_c() -> Optional[float]:
    """SoC temperature, from sysfs first (cheap) then vcgencmd."""
    try:
        raw = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        return round(int(raw) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.run(
                ["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2
            ).stdout
            return round(float(out.strip().replace("temp=", "").replace("'C", "")), 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None


def _read_throttled() -> Optional[str]:
    """Raspberry Pi throttling flags — 0x0 means no undervoltage or capping."""
    if not shutil.which("vcgencmd"):
        return None
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        ).stdout
        return out.strip().replace("throttled=", "") or None
    except (OSError, subprocess.SubprocessError):
        return None


def _read_memory() -> tuple[Optional[int], Optional[int]]:
    """(used_mb, total_mb) from /proc/meminfo."""
    try:
        values = {}
        for line in open("/proc/meminfo"):
            key, _, rest = line.partition(":")
            values[key] = int(rest.strip().split()[0])  # kB
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total is None or available is None:
            return None, None
        return (total - available) // 1024, total // 1024
    except (OSError, ValueError, IndexError):
        return None, None


# Previous (sequence, monotonic) sample, for measuring real throughput. Reported
# FPS is frames the bus actually published between two health samples — not the
# configured target, which stays reassuringly at 30 while a camera dies.
_last_fps_sample: Optional[tuple[int, float]] = None


def _measure_fps(bus) -> Optional[float]:
    global _last_fps_sample
    now = time.monotonic()
    sequence = bus.sequence
    previous, _last_fps_sample = _last_fps_sample, (sequence, now)
    if previous is None:
        return None  # first sample establishes the baseline; no honest rate yet
    frames = sequence - previous[0]
    elapsed = now - previous[1]
    if elapsed <= 0 or frames < 0:
        return None
    return round(frames / elapsed, 1)


def _camera(bus, capture) -> Component:
    """Frames arriving is the only proof a camera is working."""
    if bus is None:
        return Component(UNKNOWN, "camera pipeline not started")
    age = bus.age_s()
    if age == float("inf"):
        return Component(FAILED, "no frames captured yet")
    metrics = {"frame_age_s": round(age, 2)}
    fps = _measure_fps(bus)
    if fps is not None:
        metrics["fps"] = fps
    # CaptureService.is_running is a property, not a method.
    if capture is not None and not capture.is_running:
        return Component(FAILED, "capture service stopped", metrics)
    if age > _STALE_S:
        return Component(FAILED, f"no new frame for {age:.1f}s", metrics)
    return Component(OK, "capturing", metrics)


def _inference(heartbeat_age_s: Optional[float], module_name: str) -> Component:
    if heartbeat_age_s is None:
        return Component(UNKNOWN, "inference loop has not reported yet")
    metrics = {"loop_age_s": round(heartbeat_age_s, 2)}
    if heartbeat_age_s > _STALE_S * 2:
        return Component(FAILED, f"inference stalled {heartbeat_age_s:.1f}s ago", metrics)
    return Component(OK, f"analysing ({module_name})", metrics)


def _accelerator() -> Component:
    """Reports the accelerator as in use only when inferences have completed.

    `accel.status()` answers "is the configured intent being met", which is a
    config question. It returns healthy for a CPU-only device by design. That is
    the right answer for the health monitor and the wrong one for a customer
    reading "AI running on the Hailo-8", so completed-inference evidence decides
    the state here and status() supplies the explanation.
    """
    healthy, detail = accel.status()
    stats = accel.inference_stats()
    info = accel._accel_info or accel.detect()
    metrics = {
        "hailo_present": info.present,
        "runtime_available": info.runtime_available,
        "inferences": stats["count"],
    }
    if stats["age_s"] is not None:
        metrics["last_inference_s"] = stats["age_s"]

    if stats["live"]:
        return Component(OK, "running on the Hailo-8 accelerator", metrics)
    if stats["count"] > 0:
        return Component(WARNING, "accelerator idle — no recent inference", metrics)
    if info.present and not healthy:
        return Component(WARNING, detail, metrics)
    # Present or not, nothing has run on it. CPU inference is a supported mode,
    # so this is not a failure — but it must never read as "AI on the HAT".
    return Component(OK if healthy else WARNING, f"running on CPU — {detail}", metrics)


def _network(wifi) -> Component:
    if wifi is None:
        return Component(UNKNOWN, "network manager unavailable")
    try:
        ssid = wifi.current_ssid()
    except Exception:  # noqa: BLE001 - health must never raise into the caller
        return Component(UNKNOWN, "could not read Wi-Fi state")
    if not ssid:
        return Component(FAILED, "not connected to Wi-Fi")
    return Component(OK, f"connected to {ssid}", {"ssid": ssid})


def collect(
    *,
    bus=None,
    capture=None,
    wifi=None,
    heartbeat_age_s: Optional[float] = None,
    module_name: str = "",
    uplink_connected: Optional[bool] = None,
) -> dict:
    """Assemble the health report the device sends upstream each sync."""
    used_mb, total_mb = _read_memory()
    system_metrics = {}
    temperature = _read_temperature_c()
    if temperature is not None:
        system_metrics["temperature_c"] = temperature
    throttled = _read_throttled()
    if throttled is not None:
        system_metrics["throttled"] = throttled
    if used_mb is not None:
        system_metrics["memory_used_mb"] = used_mb
        system_metrics["memory_total_mb"] = total_mb
    try:
        system_metrics["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass

    system_state = OK
    system_detail = "nominal"
    if temperature is not None and temperature >= SOFT_THROTTLE_C:
        system_state, system_detail = WARNING, f"running hot ({temperature}°C)"
    elif throttled not in (None, "0x0"):
        system_state, system_detail = WARNING, f"throttled ({throttled})"

    report = {
        "reported_at": time.time(),
        "camera": asdict(_camera(bus, capture)),
        "inference": asdict(_inference(heartbeat_age_s, module_name or "unknown")),
        "accelerator": asdict(_accelerator()),
        "network": asdict(_network(wifi)),
        "system": asdict(Component(system_state, system_detail, system_metrics)),
    }
    if uplink_connected is not None:
        report["cloud_connected"] = bool(uplink_connected)
    return report
