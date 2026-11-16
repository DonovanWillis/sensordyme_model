#!/usr/bin/env python3
"""ONE read-only command that collects a shareable, sanitized hardware evidence bundle (M11 Phase C).

    PY=$D/.venv/bin/python   V=~/hwval/sensordyme_model/scripts/hw_validation
    export SENSORDYME_TOKEN=...          # dashboard token, read from the environment only (never argv)
    $PY $V/collect_evidence.py --install-dir $D --expect-commit <release sha>          # on the Pi
    $PY $V/collect_evidence.py --fake                                                   # Mac dry run: SIMULATED

What it does, in order, changing nothing on the device:
  1. runs the read-only preflight (sunday_preflight.Preflight; never the stopped stages, never the buzzer)
     into a private 0700 temp directory;
  2. samples the RUNNING service for --minutes (default 2): the runtime's own stage timings and processed
     fps from /api/health `performance` (stage_timing.record_row), the camera's measured rate, and the
     host's CPU utilisation, memory, SoC temperature, ARM clock and throttle state;
  3. rolls up every result it finds under --results (hwv_stage.py stage reports, earlier preflights,
     cabin-session scores): one PASS/FAIL/BLOCKED/SIMULATED line each, with the file's sha256;
  4. builds the report from ALLOW-LISTED fields only (no config dump, no full /api/health, no SSID, no
     hostname, no serial), then scans every string for secrets (device keys, tokens, bearer values, PSK /
     password assignments, claim and setup codes, e-mail, MAC and non-loopback IP addresses, the SoC
     serial, the exact token values in the environment). Hits are replaced by [REDACTED:<kind>] and
     counted in the report's `sanitizer` block; the raw value is never written anywhere;
  5. writes evidence_<device-id>_<UTC>/ (report.json, report.md, SHA256SUMS) and a .tgz of it, and prints
     the bundle's sha256 so the owner can confirm the file they receive is the one produced.

No camera frame, video or audio is collected. config.yaml is read for a fixed list of non-secret keys and
hashed before/after by the preflight (FAIL if anything changed it).

Exit codes as the other hw_validation tools: 0 PASS, 1 FAIL, 2 BLOCKED, 3 SIMULATED. A --fake run never
exits 0, and off a Raspberry Pi a hardware check can never PASS.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hwv_common as hc  # noqa: E402
import stage_timing as st  # noqa: E402
import sunday_preflight as sp  # noqa: E402

PASS, FAIL, BLOCKED, SIMULATED, NOT_RUN = sp.PASS, sp.FAIL, sp.BLOCKED, sp.SIMULATED, sp.NOT_RUN
EXIT = sp.EXIT
SCHEMA = "sensordyme.hw_evidence/1"
DOC = "docs/FRIEND_HARDWARE_SETUP.md"

#: config.yaml keys copied into the report: configuration that explains a measurement, never a credential.
CONFIG_ALLOW = {
    "role": None, "module": None, "profile": None, "deployment": None,
    "camera": ("width", "height", "fps", "rotate_180"),
    "accelerator": ("enabled", "yolo_decoder", "yolo_hef", "yolo_hef_sha256", "yolo_extra_classes"),
    "alerts": ("gpio_buzzer_pin", "cooldown_s", "experimental_alerts_audible"),
    "cloud": ("enabled", "command_auth"),
    "network": ("provisioning",),
}
#: (section, subsection, key) read from nested module config.
CONFIG_NESTED = (("modules", "driver_attention", "yolo_every_n_frames"),
                 ("modules", "driver_attention", "eye_timing_min_fps"))
#: words that make a key a credential or a network identity: dropped even inside an allowed section.
CONFIG_DENY_WORDS = frozenset({"key", "token", "secret", "password", "passphrase", "psk", "ssid", "code", "codes",
                               "bearer", "url", "host", "hostname", "email", "cert", "pem"})


def _denied(key: str) -> bool:
    """A key is denied when any of its `_`-separated words is a credential word (not substrings: 'decoder')."""
    return bool(set(str(key).lower().split("_")) & CONFIG_DENY_WORDS)

# ------------------------------------------------------------------ sanitizer
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("device_key", re.compile(r"\bdk_[A-Za-z0-9_\-]{6,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*")),
    ("assignment", re.compile(r"(?i)\b(psk|password|passwd|passphrase|secret|token|device_key)\b\s*[=:]\s*[^\s,;\"']+")),
    ("hex_token", re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")),
    ("pairing_code", re.compile(r"\b[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("mac", re.compile(r"\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\b")),
    ("ipv4", re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b")),
]


class Sanitizer:
    """Replaces secrets in every string of a JSON-like value; counts what it replaced (never the values)."""

    def __init__(self, exact: Optional[dict[str, str]] = None) -> None:
        # exact values (the token in the environment, the SoC serial): longest first, at least 6 characters
        self.exact = sorted(((k, v) for k, v in (exact or {}).items() if v and len(v) >= 6),
                            key=lambda kv: -len(kv[1]))
        self.hits: dict[str, int] = {}

    def text(self, s: str) -> str:
        for kind, value in self.exact:
            if value in s:
                self.hits[kind] = self.hits.get(kind, 0) + s.count(value)
                s = s.replace(value, f"[REDACTED:{kind}]")
        for kind, pattern in SECRET_PATTERNS:
            s, n = pattern.subn(f"[REDACTED:{kind}]", s)
            if n:
                self.hits[kind] = self.hits.get(kind, 0) + n
        return s

    def clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(str(k)): self.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        return value


def residual_secrets(text: str, exact: dict[str, str]) -> list[str]:
    """Kinds of secret still present in a final text (must be empty before anything is written)."""
    kinds = [k for k, v in exact.items() if v and len(v) >= 6 and v in text]
    kinds += [kind for kind, p in SECRET_PATTERNS if p.search(text.replace("[REDACTED:", "[R:"))]
    return kinds


# ------------------------------------------------------------------ the machine, extended for sampling
class EvidenceFakeSystem(sp.FakeSystem):
    """Dry-run stand-ins for the sampled values. Every number here is SIMULATED, never evidence."""

    def __init__(self, install_dir: Path) -> None:
        super().__init__(install_dir)
        self._tick = 0
        self.outputs.update({"vcgencmd measure_temp": "temp=51.2'C", "vcgencmd measure_clock": "frequency(0)=2400000000",
                             "modinfo": "4.20.0 (SIMULATED)", "vcgencmd bootloader_version": "2024/11/12 (SIMULATED)"})

    def run(self, cmd: list[str], timeout: float = 20.0) -> Optional[str]:
        if cmd[:2] == ["systemctl", "show"] and "MainPID" in cmd:
            return "MainPID=4242"
        return super().run(cmd, timeout)

    def read(self, path: str) -> Optional[str]:
        if path == "/proc/stat":
            self._tick += 1
            busy, idle = 4000 + 900 * self._tick, 20000 + 2100 * self._tick
            return f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n"
        if path == "/proc/meminfo":
            return "MemTotal:        4046272 kB\nMemAvailable:    2893440 kB\n"
        if path == "/proc/4242/stat":
            return "4242 (python) S " + " ".join(["0"] * 10) + f" {300 * self._tick} {60 * self._tick} 0 0"
        if path == "/proc/4242/status":
            return "Name:\tpython\nVmRSS:\t  412000 kB\n"
        return super().read(path)

    def health(self, stopped: bool = False) -> dict:
        h = super().health(stopped)
        dev = h["device"]
        dev["camera"] = {"state": "ok", "detail": "capturing (SIMULATED)",
                         "metrics": {"frame_age_s": 0.04, "fps": 29.8, "interval_p95_ms": 36.0}}
        dev["system"] = {"state": "ok", "detail": "nominal", "metrics": {"temperature_c": 51.2, "throttled": "0x0"}}
        h["performance"] = {"platform": {"class": "simulated"}, "processed_fps": 9.9, "rss_mb": 402.0,
                            "soc_temperature_c": 51.2, "throttle_now": [], "throttled_raw": "0x0",
                            "stages": {"hailo_inference": {"p50_ms": 22.0, "p95_ms": 24.0, "max_ms": 30.0, "count": 300},
                                       "decoder_cpu": {"p50_ms": 1.2, "p95_ms": 1.9, "max_ms": 3.0, "count": 300},
                                       "face_landmarks": {"p50_ms": 11.0, "p95_ms": 15.0, "max_ms": 22.0, "count": 900},
                                       "frame_to_decision": {"p50_ms": 48.0, "p95_ms": 71.0, "max_ms": 95.0,
                                                             "count": 900}}}
        return h


# ------------------------------------------------------------------ sampling
def _cpu_jiffies(system: sp.RealSystem) -> Optional[tuple[int, int]]:
    """(busy, total) jiffies from /proc/stat's first line."""
    line = (system.read("/proc/stat") or "").splitlines()[:1]
    if not line or not line[0].startswith("cpu "):
        return None
    vals = [int(x) for x in line[0].split()[1:] if x.isdigit()]
    if len(vals) < 4:
        return None
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals) - idle, sum(vals)


def _meminfo(system: sp.RealSystem) -> dict:
    out: dict[str, float] = {}
    for line in (system.read("/proc/meminfo") or "").splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable") and rest.strip().endswith("kB"):
            out[key] = round(int(rest.split()[0]) / 1024, 1)
    if "MemTotal" in out and "MemAvailable" in out:
        out["MemUsed"] = round(out["MemTotal"] - out["MemAvailable"], 1)
    return {"total_mb": out.get("MemTotal"), "available_mb": out.get("MemAvailable"), "used_mb": out.get("MemUsed")}


def _service_pid(system: sp.RealSystem) -> Optional[int]:
    text = system.run(["systemctl", "show", sp.SERVICE, "-p", "MainPID"]) or ""
    pid = sp._field(text, "MainPID", sep="=")
    return int(pid) if pid.isdigit() and int(pid) > 0 else None


def _proc_cpu_ticks(system: sp.RealSystem, pid: int) -> Optional[int]:
    stat = system.read(f"/proc/{pid}/stat") or ""
    fields = stat.rpartition(")")[2].split()
    try:
        return int(fields[11]) + int(fields[12])  # utime + stime (fields 14 and 15 of stat(5))
    except (IndexError, ValueError):
        return None


def _proc_rss_mb(system: sp.RealSystem, pid: int) -> Optional[float]:
    for line in (system.read(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def _vc(system: sp.RealSystem, *args: str) -> Optional[str]:
    out = system.run(["vcgencmd", *args], timeout=5)
    return out.strip() if out else None


def _num(text: Optional[str], pattern: str) -> Optional[float]:
    m = re.search(pattern, text or "")
    return float(m.group(1)) if m else None


def _pct(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}

    def q(p: float) -> float:
        return round(vals[min(len(vals) - 1, round(p * (len(vals) - 1)))], 2)

    return {"n": len(vals), "min": round(vals[0], 2), "p50": q(0.5), "p95": q(0.95), "max": round(vals[-1], 2)}


def sample(system: sp.RealSystem, device_url: str, token: str, minutes: float, interval: float,
           sleep=time.sleep) -> dict:
    """Poll the running service and the host. Returns raw-free summaries (no /api/health dump)."""
    clk = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    ncpu = os.cpu_count() or 1
    rows: list[dict] = []
    host: dict[str, list[float]] = {"cpu_util_pct": [], "service_cpu_pct_of_one_core": [], "service_rss_mb": [],
                                    "mem_used_mb": [], "soc_temp_c": [], "arm_clock_mhz": []}
    camera: dict[str, list[float]] = {"fps": [], "frame_age_s": []}
    throttles: set[str] = set()
    states: set[str] = set()
    errors: list[str] = []
    performance_missing = False
    pid = _service_pid(system)
    prev_cpu, prev_proc, prev_t = _cpu_jiffies(system), (_proc_cpu_ticks(system, pid) if pid else None), time.monotonic()
    polls = max(1, round(minutes * 60 / max(interval, 0.1)))
    for _ in range(polls):
        sleep(interval)
        now = time.monotonic()
        cpu = _cpu_jiffies(system)
        if cpu and prev_cpu and cpu[1] > prev_cpu[1]:
            host["cpu_util_pct"].append(100.0 * (cpu[0] - prev_cpu[0]) / (cpu[1] - prev_cpu[1]))
        prev_cpu = cpu
        if pid:
            ticks = _proc_cpu_ticks(system, pid)
            if ticks is not None and prev_proc is not None and now > prev_t:
                elapsed = (now - prev_t) if not system.fake else interval
                host["service_cpu_pct_of_one_core"].append(100.0 * (ticks - prev_proc) / clk / max(elapsed, 1e-6))
            prev_proc = ticks
            rss = _proc_rss_mb(system, pid)
            if rss is not None:
                host["service_rss_mb"].append(rss)
        prev_t = now
        mem = _meminfo(system)
        if mem["used_mb"] is not None:
            host["mem_used_mb"].append(mem["used_mb"])
        temp = _num(_vc(system, "measure_temp"), r"temp=([\d.]+)")
        if temp is not None:
            host["soc_temp_c"].append(temp)
        clock = _num(_vc(system, "measure_clock", "arm"), r"=(\d+)")
        if clock is not None:
            host["arm_clock_mhz"].append(clock / 1e6)
        thr = _vc(system, "get_throttled")
        if thr:
            throttles.add(thr.replace("throttled=", ""))
        if not token:
            continue
        try:
            payload = system.get_json(device_url.rstrip("/") + "/api/health", {"X-Device-Token": token})
        except Exception as exc:  # noqa: BLE001 - a failed poll is a data point
            errors.append(type(exc).__name__)
            continue
        dev = (payload or {}).get("device") or {}
        states.add(str((dev.get("monitoring") or {}).get("state")))
        cam = (dev.get("camera") or {}).get("metrics") or {}
        for key in ("fps", "frame_age_s"):
            if isinstance(cam.get(key), (int, float)):
                camera[key].append(float(cam[key]))
        try:
            rows.append(st.record_row(payload, now))
        except ValueError:
            performance_missing = True
    runtime = st.summarize_record(rows) if rows else {}
    return {
        "polls": polls, "interval_s": interval, "minutes": minutes, "health_polls_ok": len(rows),
        "health_poll_errors": sorted(set(errors)), "performance_block_missing": performance_missing,
        "monitoring_states_seen": sorted(states),
        "runtime": {k: runtime.get(k) for k in ("platform_classes", "is_pi_measurement", "processed_fps", "rss_mb",
                                                 "soc_temperature_c", "throttled_any", "stages")} if runtime else {},
        "camera_measured": {k: _pct(v) for k, v in camera.items()},
        "host": {**{k: _pct(v) for k, v in host.items()}, "cpu_count": ncpu,
                 "memory_now": _meminfo(system), "throttled_values": sorted(throttles)},
    }


# ------------------------------------------------------------------ static facts
def config_facts(path: Path) -> dict:
    """The allow-listed, non-secret configuration that explains the measurements."""
    import yaml

    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"error": f"cannot read {path.name}: {type(exc).__name__}"}
    out: dict[str, Any] = {}
    for section, keys in CONFIG_ALLOW.items():
        value = cfg.get(section)
        if keys is None:
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[section] = value
            continue
        if not isinstance(value, dict):
            continue
        picked = {}
        for k in keys:
            if k not in value or _denied(k):
                continue
            v = value[k]
            if isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if not _denied(str(kk)) and not isinstance(vv, (dict, list))}
            picked[k] = v
        out[section] = picked
    for section, sub, key in CONFIG_NESTED:
        node = (cfg.get(section) or {}).get(sub) if isinstance(cfg.get(section), dict) else None
        if isinstance(node, dict) and key in node:
            out.setdefault(f"{section}.{sub}", {})[key] = node[key]
    out["config_sha256"] = hc.sha256_file(path)
    return out


def driver_versions(system: sp.RealSystem, preflight: dict) -> dict:
    packages = next((c["values"] for c in preflight.get("checks", []) if c["id"] == "packages"), {}) or {}
    os_check = next((c["values"] for c in preflight.get("checks", []) if c["id"] == "os"), {}) or {}
    module = system.run(["modinfo", "-F", "version", "hailo_pci"], timeout=5)
    return {"kernel_arch": os_check.get("kernel_arch"), "os_codename": os_check.get("codename"),
            "venv_python": os_check.get("venv_python"), "firmware": os_check.get("firmware"),
            "bootloader": _vc(system, "bootloader_version"),
            "hailo_pci_module": (module or "").strip() or None, "hailortcli": packages.get("hailortcli"),
            "debian_packages": packages.get("debs")}


def results_rollup(dirs: list[Path]) -> list[dict]:
    """Every stage report / preflight / cabin score under the given directories: one line each."""
    rows: list[dict] = []
    seen: set[Path] = set()
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if path in seen or path.stat().st_size > 5_000_000:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if "stage" in data and "verdict" in data:
                kind, name, verdict = "hwv_stage", data["stage"], data["verdict"]
                failed = [c.get("check") for c in data.get("checks") or [] if not c.get("ok")][:5]
            elif data.get("tool") == "sunday_preflight":
                kind, name, verdict = "preflight", "sunday_preflight", data.get("verdict")
                failed = [c["id"] for c in data.get("checks") or [] if c.get("status") in (FAIL, BLOCKED)][:8]
            elif str(data.get("schema", "")).startswith("sensordyme.incabin"):
                kind, name = "cabin_session", data.get("session_id") or path.stem
                verdict = data.get("evidence_level") or data.get("verdict") or "recorded"
                failed = []
            else:
                continue
            rows.append({"kind": kind, "name": name, "verdict": str(verdict), "simulated": bool(data.get("simulated")),
                         "utc": data.get("utc") or data.get("finished"), "file": path.name,
                         "sha256": hc.sha256_file(path), "failed_checks": failed})
    return rows


# ------------------------------------------------------------------ remote vs local
REMOTE_LOCAL = [
    ("Photos of the assembly, ribbon, cooler, buzzer wiring", "LOCAL (physical)"),
    ("Reseating any ribbon, the HAT, the SD card; power-supply swap", "LOCAL (physical, power OFF)"),
    ("Hearing the buzzer at 50 cm and 1 m; SPL reading", "LOCAL (a person must listen)"),
    ("Aiming/mounting the camera; checkerboard calibration frames", "LOCAL (physical)"),
    ("Controlled event tests with a consenting participant", "LOCAL (participant + observer present)"),
    ("Power-cycle / unplug tests; SD-card imaging (RB-E5)", "LOCAL (physical)"),
    ("BLE provisioning from the iPhone", "LOCAL (phone within Bluetooth range)"),
    ("Live video in the app", "same Wi-Fi as the device (LAN-only by design)"),
    ("This evidence command, preflight, stage_timing, thermal soak, parity/latency stages",
     "REMOTE possible over owner-approved SSH (or Tailscale, approved separately); never required for operation"),
    ("Hash verification of the transfer package and installed tree", "REMOTE possible (SSH); the owner compares hashes"),
    ("Install of an owner-approved release (setup_pi.sh / release_guard.py)",
     "REMOTE possible, but someone LOCAL must be able to power-cycle and to stop it"),
]


# ------------------------------------------------------------------ report
def build_report(args: argparse.Namespace, system: sp.RealSystem, preflight: dict, sampled: dict,
                 config: dict, results: list[dict]) -> dict:
    checks = {c["id"]: c for c in preflight.get("checks", [])}

    def val(cid: str, key: str) -> Any:
        return (checks.get(cid) or {}).get("values", {}).get(key)

    health = checks.get("health_truth") or {}
    hv = health.get("values") or {}
    perception = checks.get("perception") or {}
    pv = perception.get("values") or {}
    runtime = sampled.get("runtime") or {}
    stages = runtime.get("stages") or {}
    camera_line = val("camera_listing", "camera")
    install = Path(args.install_dir)
    release_link = None
    with contextlib.suppress(OSError):
        release_link = os.readlink(install) if install.is_symlink() else None
    verdict = preflight.get("verdict", BLOCKED)
    if system.fake:
        verdict = SIMULATED if verdict != FAIL else FAIL
    elif not sampled.get("health_polls_ok") and verdict == PASS:
        verdict = BLOCKED
    return {
        "schema": SCHEMA, "tool": "collect_evidence", "generated_utc": hc.utc_now(),
        "verdict": verdict, "exit_code": EXIT.get(verdict, 1),
        "simulated": system.fake, "on_pi": system.on_pi(),
        "evidence_level": ("SIMULATED dry run: NOT hardware evidence" if system.fake else
                           "device measurement" if system.on_pi() else "not a Raspberry Pi: NOT hardware evidence"),
        "platform": {"board": val("identity", "board"), "revision": val("identity", "revision"),
                     "device_id": val("identity", "device_id"), "os_codename": val("os", "codename"),
                     "kernel_arch": val("os", "kernel_arch"), "venv_python": val("os", "venv_python")},
        "power": {"throttled": val("power", "throttled"), "ext5v": val("power", "ext5v")},
        "accelerator": {"lspci": val("hailo", "lspci"), "dev_hailo0": val("hailo", "dev_hailo0"),
                        "scan": val("hailo", "scan"), "parse_hef": (checks.get("parse_hef") or {}).get("detail"),
                        "health": (hv.get("accelerator") or {})},
        "camera": {"listing": camera_line or (checks.get("camera_listing") or {}).get("detail"),
                   "configured": config.get("camera"), "measured": sampled.get("camera_measured")},
        "drivers": driver_versions(system, preflight),
        "software": {"package_commit": preflight.get("package_commit"), "expected_commit": args.expect_commit,
                     "revision_check": (checks.get("software_revision") or {}).get("status"),
                     "installed_release_link": release_link,
                     "device_versions": hv.get("versions") if isinstance(hv.get("versions"), dict) else None},
        "model": {"sha256": val("model_artifact", "sha256"), "pin": val("model_artifact", "pin"),
                  "bytes": val("model_artifact", "bytes"),
                  "artifact_check": (checks.get("model_artifact") or {}).get("status"),
                  "pins_check": (checks.get("pins") or {}).get("status")},
        "decoder": {"configured": (config.get("accelerator") or {}).get("yolo_decoder"),
                    "every_n_frames": (config.get("modules.driver_attention") or {}).get("yolo_every_n_frames"),
                    "extra_classes": (config.get("accelerator") or {}).get("yolo_extra_classes")},
        "configuration": config,
        "latency_ms": {name: stages.get(name) for name in ("hailo_inference", "decoder_cpu", "face_landmarks",
                                                           "frame_to_decision") if name in stages} or
                      {"detector_run_ms_from_perception": (pv.get("object_detector") or {}).get("run_ms")},
        "processing_fps": runtime.get("processed_fps"),
        "is_pi_measurement": bool(runtime.get("is_pi_measurement")),
        "cpu_memory_thermal": sampled.get("host"),
        "monitoring": {"state": hv.get("monitoring_state"), "reasons": hv.get("reasons"),
                       "states_seen_while_sampling": sampled.get("monitoring_states_seen"),
                       "config_source": hv.get("config_source"), "simulated": hv.get("simulated"),
                       "invariant_check": health.get("status"),
                       "perception": {"object_detector": pv.get("object_detector"),
                                      "face_landmarks": pv.get("face_landmarks")}},
        "sampling": {k: sampled.get(k) for k in ("polls", "interval_s", "minutes", "health_polls_ok",
                                                 "health_poll_errors", "performance_block_missing")},
        "tests": {"preflight": {"verdict": preflight.get("verdict"), "counts": preflight.get("counts"),
                                "next_manual_step": preflight.get("next_manual_step"),
                                "checks": [{"id": c["id"], "step": c["step"], "status": c["status"],
                                            "detail": str(c["detail"])[:240]} for c in preflight.get("checks", [])]},
                  "collected_results": results},
        "remote_vs_local": [{"task": t, "where": w} for t, w in REMOTE_LOCAL],
        "privacy": {"frames_collected": False, "video_or_audio_collected": False,
                    "config_dump": False, "health_dump": False, "hostname_recorded": False},
    }


def markdown(report: dict) -> str:
    r = report
    lines = [f"# SensorDyme hardware evidence: {r['verdict']}", "",
             f"- generated {r['generated_utc']} · evidence level: **{r['evidence_level']}**",
             (f"- device `{r['platform'].get('device_id')}` · board {r['platform'].get('board')} · "
              f"{r['platform'].get('os_codename')} {r['platform'].get('kernel_arch')}"),
             (f"- package commit `{r['software'].get('package_commit')}` (expected `{r['software'].get('expected_commit')}`, "
              f"check {r['software'].get('revision_check')}); installed release {r['software'].get('installed_release_link')}"),
             (f"- model sha256 `{r['model'].get('sha256')}` vs pin `{r['model'].get('pin')}` ({r['model'].get('artifact_check')}); "
              f"decoder **{r['decoder'].get('configured')}** every {r['decoder'].get('every_n_frames')} frames"),
             f"- accelerator: {r['accelerator'].get('lspci')} · /dev/hailo0 {r['accelerator'].get('dev_hailo0')}",
             (f"- camera: {r['camera'].get('listing')} · configured {r['camera'].get('configured')} · "
              f"measured fps {r['camera'].get('measured', {}).get('fps')}"),
             f"- processing fps {r.get('processing_fps')} · Pi measurement: {r.get('is_pi_measurement')}",
             ""]
    if r["simulated"]:
        lines += ["> DRY RUN with stand-in answers. Nothing in this file is hardware evidence.", ""]
    lines += ["## Latency (ms, the runtime's own windowed percentiles)", "", "```json",
              json.dumps(r["latency_ms"], indent=1, default=str)[:3000], "```", "",
              "## CPU, memory, thermal", "", "```json",
              json.dumps(r["cpu_memory_thermal"], indent=1, default=str)[:3000], "```", "",
              "## Drivers", "", "```json", json.dumps(r["drivers"], indent=1, default=str)[:2000], "```", "",
              "## Preflight checks", "", "| Step | Check | Result | Detail |", "|---|---|---|---|"]
    for c in r["tests"]["preflight"]["checks"]:
        lines.append(f"| {c['step']} | {c['id']} | **{c['status']}** | {c['detail'].replace('|', '/')[:200]} |")
    lines += ["", "## Collected stage / session results", "", "| Kind | Name | Verdict | Simulated | File | sha256 |",
              "|---|---|---|---|---|---|"]
    for x in r["tests"]["collected_results"]:
        lines.append(f"| {x['kind']} | {x['name']} | **{x['verdict']}** | {x['simulated']} | {x['file']} | "
                     f"`{x['sha256'][:16]}…` |")
    lines += ["", "## Remote vs local", "", "| Task | Where |", "|---|---|"]
    lines += [f"| {x['task']} | {x['where']} |" for x in r["remote_vs_local"]]
    lines += ["", f"Sanitizer: {r.get('sanitizer')}", ""]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR)
    p.add_argument("--device", default="http://localhost:8899")
    p.add_argument("--expect-commit", default=None, help="the approved release commit")
    p.add_argument("--token-env", default="SENSORDYME_TOKEN")
    p.add_argument("--minutes", type=float, default=2.0, help="sampling window of the running service")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--results", type=Path, nargs="*", default=None,
                   help="directories with stage / preflight / cabin results (default ~/hwval)")
    p.add_argument("--out", type=Path, default=None, help="default ~/hwval/evidence")
    p.add_argument("--fake", action="store_true", help="dry run with stand-in answers (SIMULATED, never PASS)")
    return p


def main(argv: Optional[list[str]] = None, *, system: Optional[sp.RealSystem] = None,
         env: Optional[dict[str, str]] = None, sleep=time.sleep) -> int:
    args = build_parser().parse_args(argv)
    sys.dont_write_bytecode = True
    env = dict(os.environ if env is None else env)
    if args.out is None:
        args.out = (Path(tempfile.gettempdir()) / "hwval-evidence-fake") if args.fake else Path.home() / "hwval" / "evidence"
    ok, why = hc.ensure_outside(args.out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return EXIT[BLOCKED]
    if system is None:
        system = EvidenceFakeSystem(args.install_dir) if args.fake else sp.RealSystem()
    token = env.get(args.token_env, "") or ("simulated-token" if system.fake else "")
    results_dirs = args.results if args.results is not None else [Path.home() / "hwval"]

    print("== 1/4 read-only preflight", flush=True)
    with tempfile.TemporaryDirectory(prefix="sd-evidence-") as private:
        os.chmod(private, 0o700)
        pf_args = sp.build_parser().parse_args(
            ["--install-dir", str(args.install_dir), "--device", args.device, "--token-env", args.token_env,
             "--out", private, *(["--expect-commit", args.expect_commit] if args.expect_commit else []),
             *(["--fake"] if system.fake else [])])
        runner = sp.Preflight(pf_args, system, env=env)
        preflight = runner.run()
        config_path = runner.config_path()
    print(f"== 2/4 sampling the running service for {args.minutes} min every {args.interval} s", flush=True)
    sampled = sample(system, args.device, token, args.minutes, args.interval, sleep=sleep)
    print("== 3/4 collecting earlier results", flush=True)
    results = results_rollup([d for d in results_dirs if d.resolve() != args.out.resolve()])
    config = config_facts(config_path)

    print("== 4/4 sanitizing and writing", flush=True)
    serial = sp._field(system.read("/proc/cpuinfo") or "", "Serial")
    exact = {"token": env.get(args.token_env, ""), "bearer": env.get("SENSORDYME_BEARER", ""),
             "soc_serial": serial if serial and set(serial) != {"0"} else ""}
    sanitizer = Sanitizer(exact)
    report = sanitizer.clean(build_report(args, system, preflight, sampled, config, results))
    report["sanitizer"] = {"redactions": dict(sorted(sanitizer.hits.items())),
                           "rule": "allow-listed fields only; secret patterns redacted; raw values never written"}
    text_json = json.dumps(report, indent=2, default=str)
    text_md = markdown(report)
    leftover = residual_secrets(text_json + text_md, exact)
    if leftover:  # the sanitizer's own output still matches: refuse rather than risk a leak
        print(f"REFUSED: secrets still detected after redaction ({sorted(set(leftover))}); nothing written",
              file=sys.stderr)
        return EXIT[FAIL]

    device = str((report.get("platform") or {}).get("device_id") or "unknown").replace("/", "_")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    folder = args.out / f"evidence_{device}_{stamp}{'_SIMULATED' if system.fake else ''}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(text_json)
    (folder / "report.md").write_text(text_md)
    sums = "".join(f"{hashlib.sha256((folder / n).read_bytes()).hexdigest()}  {n}\n" for n in ("report.json", "report.md"))
    (folder / "SHA256SUMS").write_text(sums)
    bundle = folder.with_suffix(".tgz")
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    digest = hc.sha256_file(bundle)
    (args.out / f"{bundle.name}.sha256").write_text(f"{digest}  {bundle.name}\n")
    print(f"\nEVIDENCE: {report['verdict']}  ({report['evidence_level']})")
    print(f"  folder  {folder}\n  bundle  {bundle}\n  sha256  {digest}")
    print(f"  redactions: {report['sanitizer']['redactions'] or 'none'}")
    print("  Share the .tgz through the agreed private channel and send ONLY its sha256 in chat.")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
