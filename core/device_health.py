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

import functools
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from core import accel, identity
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
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            raw = f.read().strip()
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


#: Raspberry Pi `get_throttled` bits. The low nibble is what is happening NOW;
#: bits 16-19 are "has happened at some point since boot" and STAY SET until the
#: Pi is power-cycled. Conflating the two makes a device report a hardware fault
#: forever because of one brown-out during setup -- which is exactly what this
#: device did, reporting `warning: throttled (0x50000)` continuously while every
#: live bit was clear and every measurement was nominal.
_THROTTLE_LIVE = {
    0: "under-voltage",
    1: "ARM frequency capped",
    2: "throttled",
    3: "soft temperature limit",
}
_THROTTLE_STICKY = {
    16: "under-voltage",
    17: "ARM frequency capping",
    18: "throttling",
    19: "soft temperature limit",
}


def _decode_throttled(raw: Optional[str]) -> tuple[list[str], list[str]]:
    """(happening now, has happened since boot) from a `get_throttled` value.

    Returns two empty lists when the value cannot be parsed, because an
    unreadable register is not evidence of a fault.
    """
    if not raw:
        return [], []
    try:
        value = int(raw, 16)
    except (TypeError, ValueError):
        return [], []
    live = [name for bit, name in _THROTTLE_LIVE.items() if value >> bit & 1]
    sticky = [name for bit, name in _THROTTLE_STICKY.items() if value >> bit & 1]
    return live, sticky


def _read_memory() -> tuple[Optional[int], Optional[int]]:
    """(used_mb, total_mb) from /proc/meminfo."""
    try:
        values = {}
        with open("/proc/meminfo") as f:
            for line in f:
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
#: Previous (sequence, monotonic) sample PER CALLER. It used to be one global,
#: which two callers shared: the inference loop samples every ~5s, and every
#: /api/health request samples again on the dashboard thread. Each consumed the
#: other's baseline, so a request arriving just after a loop sample measured
#: almost no elapsed time and reported ~0 fps on a perfectly healthy 30 fps
#: camera -- and reset the window the loop was mid-way through. Measured: loop
#: 30.0, dashboard 0.0, loop 29.9, on the same bus in the same second.
_last_fps_sample: dict[str, tuple[int, float]] = {}


def _measure_fps(bus, caller: str = "default") -> Optional[float]:
    now = time.monotonic()
    sequence = bus.sequence
    previous = _last_fps_sample.get(caller)
    _last_fps_sample[caller] = (sequence, now)
    if previous is None:
        return None  # first sample establishes the baseline; no honest rate yet
    frames = sequence - previous[0]
    elapsed = now - previous[1]
    if elapsed <= 0 or frames < 0:
        return None
    return round(frames / elapsed, 1)


def _camera(bus, capture, caller: str = "default") -> Component:
    """Frames arriving is the only proof a camera is working."""
    if bus is None:
        return Component(UNKNOWN, "camera pipeline not started")
    age = bus.age_s()
    if age == float("inf"):
        return Component(FAILED, "no frames captured yet")
    metrics = {"frame_age_s": round(age, 2)}
    fps = _measure_fps(bus, caller)
    if fps is not None:
        metrics["fps"] = fps
    # A mean rate cannot answer the question that actually matters — did the
    # pipeline ever STALL. EventEngine discards an unconfirmed track when two
    # processed frames land more than occlusion_grace_s apart (1.0s by default,
    # and left at default by fall, every zone rule, flailing and choking), so a
    # device averaging 30 fps with one 1.2s hiccup silently loses in-progress
    # evidence while reporting a healthy rate. The distribution shows it; the
    # mean hides it.
    intervals = bus.interval_stats(caller) if hasattr(bus, "interval_stats") else {}
    metrics.update(intervals)
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


@functools.cache
def _device_id() -> str:
    """identity.device_id(), computed once: it reads /proc and hashes."""
    return identity.device_id()


def collect(
    *,
    bus=None,
    capture=None,
    wifi=None,
    heartbeat_age_s: Optional[float] = None,
    module_name: str = "",
    uplink_connected: Optional[bool] = None,
    detection_degraded: Optional[str] = None,
    caller: str = "default",
    detector_required: bool = True,
) -> dict:
    """Assemble the health report the device sends upstream each sync.

    `detection_degraded` is the reason object detection is unavailable, or None.
    When set, this report must never read as though monitoring is working: the
    inference component is FAILED, and `monitoring_active` is False. A safety
    product that shows green while it cannot see is worse than one that is
    plainly off, because only the second one gets fixed.
    """
    used_mb, total_mb = _read_memory()
    system_metrics = {}
    temperature = _read_temperature_c()
    if temperature is not None:
        system_metrics["temperature_c"] = temperature
    throttled = _read_throttled()
    live_throttle: list[str] = []
    sticky_throttle: list[str] = []
    if throttled is not None:
        system_metrics["throttled"] = throttled  # raw, so the value stays traceable
        live_throttle, sticky_throttle = _decode_throttled(throttled)
        system_metrics["throttle_now"] = live_throttle
        system_metrics["throttle_since_boot"] = sticky_throttle
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
    elif live_throttle:
        # Only a LIVE bit is a fault. This is the condition that is happening
        # right now and that a person can act on by changing the supply.
        system_state = WARNING
        system_detail = f"{' and '.join(live_throttle)} happening now ({throttled})"
    elif sticky_throttle:
        # History, not a fault. Reported so it is not hidden -- a past brown-out
        # is worth knowing about -- but it must not colour the device unhealthy
        # for the rest of its uptime, or the warning stops meaning anything.
        system_detail = (
            f"nominal ({' and '.join(sticky_throttle)} occurred earlier this boot)"
        )

    accelerator = _accelerator()
    # A detector that dies at frame 10,000 must flip this too, not just one that
    # was missing at boot. `detection_degraded` is set once, during startup, so on
    # its own it would leave monitoring_active True for a device whose NPU stopped
    # answering an hour ago -- and this field is documented as the one a consumer
    # should trust. The accelerator's "idle" state is the runtime half of that
    # signal: inferences HAVE completed and then stopped, which is the shape of a
    # detector dying mid-run.
    #
    # Keyed on the idle branch specifically, not on "accelerator is not ok" --
    # and ONLY for a module that actually uses the accelerator.
    #
    # `detector_required` is load-bearing, not defensive. A Pose-only module
    # (fall_detection consumes no YOLO) legitimately runs no NPU inference. On a
    # FRESH BOOT into it the count is 0, which lands on the "running on CPU"
    # branch and reads fine. But on a LIVE SWITCH into it -- exactly what the
    # demo does -- the counter still holds the previous module's inferences, so
    # it ages past the liveness window and looks like "inferences happened, then
    # stopped": the signature of a detector dying. The device then reported
    # monitoring_active FALSE while fall detection worked perfectly, which is the
    # false-degraded mirror of the bug this whole field exists to prevent.
    accelerator_died = (
        detector_required and accelerator.state == WARNING and "idle" in accelerator.detail
    )

    # A device whose camera stopped is not monitoring, whatever the inference
    # loop says. The loop stamps its heartbeat on every no-fresh-frame timeout --
    # correctly, so a camera outage is not misread as a dead loop -- which means
    # `inference` reports "analysing" while no frame has arrived for a minute.
    # monitoring_active is documented as the one field a consumer should trust,
    # and it was answering yes with nothing to look at.
    #
    # Keyed on the STALE-FRAME branch only, not on FAILED generally: `_camera`
    # also reports FAILED for "no frames captured yet", which is the ordinary
    # state during the first second of boot. Flipping on that would make a
    # healthy device announce it is not monitoring every time it starts.
    camera_component = _camera(bus, capture, caller)
    camera_stopped = (
        camera_component.state == FAILED and "no new frame" in camera_component.detail
    )

    if detection_degraded is None:
        inference = _inference(heartbeat_age_s, module_name or "unknown")
    else:
        # The loop is alive (its heartbeat still ticks) but it is deliberately
        # not analysing. That is a FAILURE of the product's purpose even though
        # no thread has died, so it is reported as one.
        inference = Component(
            FAILED,
            f"monitoring NOT active — {detection_degraded}",
            {"degraded": True, "reason": detection_degraded},
        )

    report = {
        "reported_at": time.time(),
        # Which camera this is: the same SD- id it advertises over Bonjour. The
        # backend stores it so the app can show each device its OWN live feed
        # rather than whichever camera the phone linked (ios FeedLink.swift).
        "device_id": _device_id(),
        # The single field a consumer should trust for "is this thing watching?".
        # Everything else describes components; this describes the product.
        "monitoring_active": (
            detection_degraded is None and not accelerator_died and not camera_stopped
        ),
        "camera": asdict(camera_component),
        "inference": asdict(inference),
        "accelerator": asdict(accelerator),
        "network": asdict(_network(wifi)),
        "system": asdict(Component(system_state, system_detail, system_metrics)),
    }
    if uplink_connected is not None:
        report["cloud_connected"] = bool(uplink_connected)
    return report
