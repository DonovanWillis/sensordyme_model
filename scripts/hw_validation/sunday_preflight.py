#!/usr/bin/env python3
"""Sunday device activation: ONE read-only preflight, one evidence summary, the next step.

    PY=$D/.venv/bin/python   V=~/hwval/sensordyme_model/scripts/hw_validation
    export SENSORDYME_TOKEN=...                       # dashboard token (RB-0); never on the command line
    $PY $V/sunday_preflight.py --install-dir $D --expect-commit <rc sha>      # read-only, any time
    $PY $V/sunday_preflight.py --fake                 # dry run on a Mac: SIMULATED, never PASS

Runs the read-only checks of docs/SUNDAY_DEVICE_ACTIVATION.md in order --
identity, power, OS, packages, software/model revisions, Hailo enumeration,
camera listing, service, auth, health + state truth, perception, the live
stream's headers, the backend view -- and writes ONE JSON + Markdown summary
with PASS / FAIL / BLOCKED / SIMULATED / NOT RUN per check, then prints the
next manual step.

It changes nothing by default. It never edits config.yaml (it hashes the file
before and after and FAILs if anything changed it). Steps that change state
run only with an explicit flag AND a typed confirmation at a terminal:

    --with-stopped-stages   stops sensordyme.service (monitoring OFF), runs the
                            hwv_stage.py stages model, capture, parity, latency,
                            then starts the service again and checks it is active
    --with-buzzer           adds the buzzer stage (loud; warn the people present)

The 30-minute thermal soak is never run from here (stage `thermal`, step 11).

Exit codes (as hwv_stage.py): 0 PASS, 1 FAIL, 2 BLOCKED, 3 SIMULATED. A
--fake run never exits 0, and off a Raspberry Pi a hardware check can never
be PASS (it is BLOCKED), so a dry run cannot be mistaken for validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_live_status as cls  # noqa: E402
import hwv_common as hc  # noqa: E402

PASS, FAIL, BLOCKED, SIMULATED, NOT_RUN = "PASS", "FAIL", "BLOCKED", "SIMULATED", "NOT RUN"
EXIT = {PASS: 0, FAIL: 1, BLOCKED: 2, SIMULATED: 3}
DOC = "docs/SUNDAY_DEVICE_ACTIVATION.md"
SERVICE = hc.SERVICE
HEF_NAME = "damoyolo_tinynasL20_T.hef"
#: Debian packages the device needs (setup_pi.sh / RB-C3); recorded with versions.
DEB_PACKAGES = ("hailo-all", "hailort", "rpicam-apps", "python3-picamera2", "python3-libcamera", "bluez",
                "network-manager")


# ------------------------------------------------------------------ the machine, real or simulated
class RealSystem:
    """Read-only access to this machine. Every method returns None when the thing is absent."""

    fake = False

    def on_pi(self) -> bool:
        return "raspberry pi" in (self.read("/proc/device-tree/model") or "").lower()

    def read(self, path: str) -> Optional[str]:
        try:
            return Path(path).read_bytes().rstrip(b"\x00").decode(errors="replace")
        except OSError:
            return None

    def exists(self, path: str) -> bool:
        return Path(path).exists()

    def run(self, cmd: list[str], timeout: float = 20.0) -> Optional[str]:
        return hc.run_text(cmd, timeout=timeout)

    def which(self, tool: str) -> bool:
        return shutil.which(tool) is not None

    def identity(self) -> dict:
        from core import identity

        did = identity.device_id()
        return {"device_id": did, "display_name": identity.display_name(did)}

    def get_json(self, url: str, headers: dict[str, str]) -> Any:
        return cls._get_json(url, headers)

    def http_status(self, url: str, headers: Optional[dict[str, str]] = None) -> Optional[int]:
        return cls.http_status(url, headers)

    def stream(self, base: str, token: str, seconds: float) -> dict:
        return cls.read_stream_parts(base, token, seconds)

    def venv_versions(self, python: Path, names: list[str]) -> Optional[dict[str, Optional[str]]]:
        if not python.exists():
            return None
        code = ("import json,sys,importlib.metadata as m\nout={}\nfor n in sys.argv[1:]:\n"
                "    try: out[n]=m.version(n)\n    except m.PackageNotFoundError: out[n]=None\nprint(json.dumps(out))")
        text = hc.run_text([str(python), "-c", code, *names], timeout=60)
        try:
            return json.loads((text or "").splitlines()[-1])
        except (ValueError, IndexError):
            return None


class FakeSystem(RealSystem):
    """Canned stand-in answers for a dry run. NOT evidence: every hardware check reports SIMULATED."""

    fake = True
    TAG = "(SIMULATED)"

    def __init__(self, install_dir: Path) -> None:
        self.install_dir = install_dir
        hef = install_dir / "models" / HEF_NAME
        self.files = {
            "/proc/device-tree/model": f"Raspberry Pi 5 Model B Rev 1.0 {self.TAG}",
            "/proc/cpuinfo": "Revision\t: d04170\nSerial\t\t: 5151515151515151\nModel\t\t: Raspberry Pi 5 Model B",
            "/etc/os-release": 'PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nVERSION_CODENAME=bookworm\nID=debian',
            "/etc/avahi/services/sensordyme.service": "<txt-record>id=SD-5113D0DE</txt-record>",
        }
        self.paths = {"/dev/hailo0", str(hef)}
        self.outputs = {
            "uname": "6.6.51+rpt-rpi-2712 aarch64",
            "vcgencmd get_throttled": "throttled=0x0",
            "vcgencmd pmic_read_adc": "EXT5V_V volt(24)=5.10000000V",
            "vcgencmd version": f"firmware {self.TAG}",
            "dpkg-query": "\n".join(f"{p}\t4.20.0 {self.TAG}" for p in DEB_PACKAGES),
            "hailortcli --version": f"HailoRT-CLI version 4.20.0 {self.TAG}",
            "lspci": f"0001:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor (rev 01) {self.TAG}",
            "hailortcli scan": "Hailo Devices:\n[-] Device: 0001:01:00.0",
            "rpicam-hello": "Available cameras\n-----------------\n0 : imx708_wide_noir [4608x2592 10-bit RGGB] "
                            f"(/base/axi/pcie@120000/rp1/i2c@88000/imx708@1a) {self.TAG}",
            "systemctl show": "ActiveState=active\nSubState=running\nNRestarts=0\nUser=pi\n"
                              f"WorkingDirectory={install_dir}\nOnFailure=sensordyme-recover.service",
        }

    def on_pi(self) -> bool:
        return False  # a dry run is never on the device

    def read(self, path: str) -> Optional[str]:
        return self.files.get(path)

    def exists(self, path: str) -> bool:
        return path in self.paths

    def which(self, tool: str) -> bool:
        return any(key.split()[0] == tool for key in self.outputs)

    def run(self, cmd: list[str], timeout: float = 20.0) -> Optional[str]:
        joined = " ".join(cmd)
        for key, text in self.outputs.items():
            if joined.startswith(key) or (key == "uname" and cmd[0] == "uname"):
                return text
        return None

    def identity(self) -> dict:
        return {"device_id": "SD-5113D0DE", "display_name": "SensorDyme-D0DE"}

    def health(self, stopped: bool = False) -> dict:
        perception = {"schema": "sd-perception/1", "simulated": True, "components": {
            "object_detector": {"state": "ok", "runs": 120, "run_ms": 23.0, "consumers": ["driver_attention"]},
            "face_landmarks": {"state": "ok", "runs": 360, "run_ms": 9.0, "consumers": ["driver_attention"]}},
            "nodes": [{"id": "driver_safety", "state": "active", "validation": "synthetic_tests"}]}
        return {"state": "ok", "device": {
            "device_id": "SD-5113D0DE", "simulated": True, "monitoring_active": not stopped,
            "monitoring": {"state": "stopped" if stopped else "ready", "monitoring": not stopped,
                           "capabilities": {}, "available": [], "impaired": [], "paused": [],
                           "versions": {"software": "0.2", "model": f"damoyolo_tinynasL20_T@{'0' * 12}"},
                           "model": {"detector": {"name": "damoyolo_tinynasL20_T", "sha256": None}},
                           "perception": perception},
            "protection": {"state": "stopped" if stopped else "protecting",
                           "monitoring_state": "stopped" if stopped else "ready"},
            "accelerator": {"state": "ok", "detail": f"running on the Hailo-8 accelerator {self.TAG}",
                            "metrics": {"hailo_present": True, "runtime_available": True, "inferences": 1200}},
            "config": {"source": "config", "fault": None}, "saved_state": {"faults": []},
            "stamp": {"schema": "sd-health-v1", "boot_id": "simulated", "generation": 1, "age_s": 0.0}}}

    def get_json(self, url: str, headers: dict[str, str]) -> Any:
        if url.endswith("/api/health"):
            return self.health()
        if url.endswith("/api/stream/stats"):
            return {"viewers": 1, "frames_sent": 50, "camera_fallback_encodes": 0, "simulated": True}
        raise cls.UnavailableError(f"{url}: no stand-in")

    def http_status(self, url: str, headers: Optional[dict[str, str]] = None) -> Optional[int]:
        return 200 if headers and (headers.get("X-Device-Token") or headers.get("Authorization")) else 401

    def stream(self, base: str, token: str, seconds: float) -> dict:
        parts = [{"t": 0.1 * i, "bytes": 18000, "jpeg": True,
                  "headers": {"content-type": "image/jpeg", "x-frame-seq": str(100 + i), "x-frame-age-ms": "80",
                              "x-frame-source": "annotated", "x-stream-id": "simulated"}} for i in range(40)]
        return {"status": 200, "headers": {"X-Stream-Id": "simulated"}, "parts": parts}

    def venv_versions(self, python: Path, names: list[str]) -> Optional[dict[str, Optional[str]]]:
        pins = _requirement_pins()
        return {n: pins.get(n) for n in names}


# ------------------------------------------------------------------ results
@dataclass
class Check:
    id: str
    step: int  # docs/SUNDAY_DEVICE_ACTIVATION.md step
    title: str
    hardware: bool
    status: str = NOT_RUN
    detail: str = ""
    values: dict = field(default_factory=dict)
    next: str = ""  # what the operator does about a non-PASS

    def to_dict(self) -> dict:
        return {"id": self.id, "step": self.step, "title": self.title, "hardware": self.hardware,
                "status": self.status, "detail": self.detail, "values": self.values, "next": self.next}


class Preflight:
    def __init__(self, args: argparse.Namespace, system: RealSystem, *,
                 env: Optional[dict[str, str]] = None,
                 confirm: Optional[Callable[[str], str]] = None,
                 call: Optional[Callable[[list[str]], int]] = None) -> None:
        self.args, self.sys = args, system
        self.env = dict(os.environ if env is None else env)
        self.confirm = confirm
        self.call = call or subprocess.call
        self.checks: list[Check] = []
        self.install = Path(args.install_dir)
        self.out_dir = Path(args.out)
        # A dry run talks to stand-ins only; it never needs (or sees) the real token.
        self.token = self.env.get(args.token_env, "") or ("simulated-token" if system.fake else "")
        self.device_url = args.device.rstrip("/")
        self.health: Optional[dict] = None
        self.identity: dict = {}

    # ---------------------------------------------------------------- helpers
    def add(self, check: Check, status: str, detail: str = "", next_step: str = "", **values: Any) -> Check:
        """Record a result. Off the Pi a hardware check can never PASS; under --fake it is SIMULATED."""
        if check.hardware and status == PASS:
            if self.sys.fake:
                status, detail = SIMULATED, f"{detail} [stand-in answers, NOT evidence]".strip()
            elif not self.sys.on_pi():
                status, detail = BLOCKED, f"not a Raspberry Pi: a hardware check cannot pass here ({detail})"
        if self.sys.fake and status == PASS and not check.hardware:
            status = SIMULATED
        check.status, check.detail, check.values, check.next = status, detail, values, next_step
        self.checks.append(check)
        mark = {PASS: "ok  ", FAIL: "FAIL", BLOCKED: "BLKD", SIMULATED: "SIM ", NOT_RUN: "----"}[status]
        print(f"  [{mark}] step {check.step:>2} {check.id}: {detail}", flush=True)
        return check

    def blocked_off_pi(self, check: Check, what: str) -> Optional[Check]:
        if not self.sys.fake and not self.sys.on_pi():
            return self.add(check, BLOCKED, f"not a Raspberry Pi: {what} needs the device",
                            "run this on the Pi (docs/SUNDAY_DEVICE_ACTIVATION.md §0)")
        return None

    def config_path(self) -> Path:
        path = self.install / "config.yaml"
        # A dry run on a Mac has no install tree: the repo's config stands in (read only).
        return REPO_ROOT / "config.yaml" if self.sys.fake and not path.exists() else path

    # ---------------------------------------------------------------- read-only checks, in order
    def identity_check(self) -> None:
        c = Check("identity", 1, "board, hardware revision and device identity", True)
        if self.blocked_off_pi(c, "the board identity") is not None:
            return
        model = self.sys.read("/proc/device-tree/model") or ""
        cpuinfo = self.sys.read("/proc/cpuinfo") or ""
        revision = _field(cpuinfo, "Revision")
        serial = _field(cpuinfo, "Serial")
        self.identity = self.sys.identity()
        avahi = re.search(r"id=(SD-[0-9A-F]{8})", self.sys.read("/etc/avahi/services/sensordyme.service") or "")
        values = {"board": model, "revision": revision, "serial_present": bool(serial and set(serial) != {"0"}),
                  "device_id": self.identity.get("device_id"), "display_name": self.identity.get("display_name"),
                  "avahi_id": avahi.group(1) if avahi else None}
        problems = []
        if not model.startswith("Raspberry Pi 5"):
            problems.append(f"board is {model!r}, expected Raspberry Pi 5")
        if not values["serial_present"]:
            problems.append("no SoC serial: the id falls back to a MAC (record it as a finding, RB-C2)")
        if not re.fullmatch(r"SD-[0-9A-F]{8}", str(values["device_id"])):
            problems.append(f"device id {values['device_id']!r} is not SD-XXXXXXXX")
        if values["avahi_id"] and values["avahi_id"] != values["device_id"]:
            problems.append(f"avahi advertises {values['avahi_id']}, the device computes {values['device_id']}")
        hard = [p for p in problems if not p.startswith("no SoC serial")]
        self.add(c, FAIL if hard else PASS,
                 "; ".join(problems) or f"{model}; rev {revision}; {values['device_id']}",
                 "RB-B1/RB-C2: record the id; stop on a mismatch" if hard else "", **values)

    def power_check(self) -> None:
        c = Check("power", 1, "power supply: never throttled, EXT5V", True)
        if self.blocked_off_pi(c, "vcgencmd") is not None:
            return
        throttled = self.sys.run(["vcgencmd", "get_throttled"])
        volts = self.sys.run(["vcgencmd", "pmic_read_adc", "EXT5V_V"])
        if throttled is None:
            self.add(c, BLOCKED, "vcgencmd not available", "RB-B2")
            return
        ok = throttled.strip().endswith("=0x0")
        self.add(c, PASS if ok else FAIL, f"{throttled.strip()}; {(volts or '').strip()}",
                 "" if ok else "fix power (27 W PSU) before measuring anything: RB-B2",
                 throttled=throttled.strip(), ext5v=(volts or "").strip())

    def os_check(self) -> None:
        c = Check("os", 3, "OS / runtime compatibility (Bookworm 64-bit, Python 3.11)", True)
        if self.blocked_off_pi(c, "the OS check") is not None:
            return
        release = self.sys.read("/etc/os-release") or ""
        codename = _field(release, "VERSION_CODENAME", sep="=").strip('"')
        uname = self.sys.run(["uname", "-rm"]) or ""
        py = self.sys.run([str(self.venv_python()), "--version"]) if not self.sys.fake else "Python 3.11.2 (SIMULATED)"
        values = {"codename": codename, "kernel_arch": uname.strip(), "venv_python": (py or "").strip(),
                  "firmware": (self.sys.run(["vcgencmd", "version"]) or "").strip()[:200]}
        problems = []
        if codename != "bookworm":
            problems.append(f"OS codename {codename!r}, expected bookworm (RB-E6 clean-install criterion)")
        if "aarch64" not in uname:
            problems.append(f"not 64-bit: {uname.strip()!r}")
        if not py or "3.11" not in py:
            problems.append(f"install venv python is {py!r}, expected 3.11.x ({self.venv_python()})")
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or f"{codename}, {uname.strip()}, {py.strip()}",
                 "RB-C1/RB-E6" if problems else "", **values)

    def venv_python(self) -> Path:
        return self.install / ".venv" / "bin" / "python"

    def packages_check(self) -> None:
        c = Check("packages", 3, "system packages and the install venv's pinned Python packages", True)
        if self.blocked_off_pi(c, "the package inventory") is not None:
            return
        deb = self.sys.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n", *DEB_PACKAGES]) or ""
        debs = dict(line.split("\t", 1) for line in deb.splitlines() if "\t" in line)
        hailortcli = (self.sys.run(["hailortcli", "--version"]) or "").strip()
        pins = _requirement_pins()
        installed = self.sys.venv_versions(self.venv_python(), sorted(pins))
        if installed is None:
            self.add(c, BLOCKED, f"no install venv at {self.venv_python()}: pass --install-dir (RB-C1)",
                     "find the service's WorkingDirectory: systemctl show sensordyme.service -p WorkingDirectory",
                     debs=debs, hailortcli=hailortcli)
            return
        mismatched = {n: [want, installed.get(n)] for n, want in pins.items() if installed.get(n) != want}
        problems = []
        if "hailo-all" not in debs and "hailort" not in debs:
            problems.append("neither hailo-all nor hailort is installed (sudo apt install hailo-all)")
        if mismatched:
            problems.append(f"venv differs from requirements-device.txt: {mismatched}")
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or
                 f"{len(pins)} pins match; {hailortcli or 'hailortcli ?'}",
                 "RB-C3 / RB-F2 (pip install --require-hashes --only-binary=:all: -r requirements-device.lock; docs/DEPLOYMENT.md) -- a deliberate CHANGE" if problems else "",
                 debs=debs, hailortcli=hailortcli, pins_checked=len(pins), mismatched=mismatched)

    def revision_check(self) -> None:
        c = Check("software_revision", 5, "validation package commit == the approved release candidate", False)
        commit = hc.git_commit(REPO_ROOT)
        expected = self.args.expect_commit
        values = {"package_commit": commit, "expected": expected}
        if self.sys.fake:
            self.add(c, SIMULATED, f"package {commit}; expected {expected or '(not given)'}", **values)
        elif not expected:
            self.add(c, BLOCKED, f"package is {commit}; pass --expect-commit $(git rev-parse $RC) to compare",
                     "record the RC commit (docs/SUNDAY_DEVICE_ACTIVATION.md §2)", **values)
        else:
            same = bool(commit) and str(commit).removeprefix("file:").startswith(expected)
            self.add(c, PASS if same else FAIL, f"package {commit} vs expected {expected}",
                     "" if same else "re-copy the package from the RC (RB-A4)", **values)

    def pins_check(self) -> None:
        c = Check("pins", 5, "model pins agree (config / setup_pi.sh / model_registry.yaml); decoder legacy", False)
        code, out = self.stage("pins")
        status = {0: PASS, 1: FAIL, 2: BLOCKED, 3: SIMULATED}.get(code, FAIL)
        self.add(c, status, out, "" if status in (PASS, SIMULATED) else "docs/MONDAY_HARDWARE_SESSION.md §9",
                 exit_code=code)

    def model_artifact_check(self) -> None:
        c = Check("model_artifact", 5, "installed .hef sha256 == accelerator.yolo_hef_sha256 (hash only, no load)", True)
        if self.blocked_off_pi(c, "the installed model") is not None:
            return
        import yaml

        hef = self.install / "models" / HEF_NAME
        try:
            config = yaml.safe_load(self.config_path().read_text()) or {}
        except OSError as exc:
            self.add(c, BLOCKED, f"cannot read {self.config_path()}: {exc.strerror}", "pass --install-dir (RB-C1)")
            return
        pin = str((config.get("accelerator") or {}).get("yolo_hef_sha256") or "").lower()
        if self.sys.fake:
            self.add(c, PASS, f"{hef} (stand-in path) vs pin {pin[:12]}...", pin=pin)
            return
        if not hef.exists():
            self.add(c, BLOCKED, f"{hef} does not exist", "setup_pi.sh provisions it (RB-C3)")
            return
        actual = hc.sha256_file(hef)
        ok, why = hc.verify_pin(actual, pin)
        self.add(c, PASS if ok else FAIL, why, "" if ok else "stop: wrong model (docs/MONDAY_HARDWARE_SESSION.md §9)",
                 sha256=actual, pin=pin, bytes=hef.stat().st_size)

    def hailo_check(self) -> None:
        c = Check("hailo", 7, "Hailo-8L enumerated (PCIe, /dev/hailo0, hailortcli scan)", True)
        if self.blocked_off_pi(c, "the accelerator") is not None:
            return
        lspci = self.sys.run(["lspci"]) or ""
        scan = self.sys.run(["hailortcli", "scan"]) or ""
        line = next((x for x in lspci.splitlines() if "hailo" in x.lower()), "")
        dev = self.sys.exists("/dev/hailo0")
        problems = [p for p, bad in (("no Hailo on PCIe (lspci)", not line), ("no /dev/hailo0", not dev),
                                     ("hailortcli scan lists no device", "Device:" not in scan)) if bad]
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or line.strip(),
                 "power off; reseat the ribbon; dtparam=pciex1 (RB-D1)" if problems else "",
                 lspci=line.strip(), dev_hailo0=dev, scan=scan.strip()[:300])

    def parse_hef_check(self) -> None:
        c = Check("parse_hef", 8, "hailortcli parse-hef: HAILO8L, six outputs, no NMS", True)
        code, out = self.stage("parse-hef")
        status = {0: PASS, 1: FAIL, 2: BLOCKED, 3: SIMULATED}.get(code, FAIL)
        self.add(c, status, out, "" if status in (PASS, SIMULATED) else "docs/MONDAY_HARDWARE_SESSION.md §9",
                 exit_code=code)

    def camera_check(self) -> None:
        c = Check("camera_listing", 6, "Camera Module 3 Wide NoIR listed (imx708_wide_noir)", True)
        if self.blocked_off_pi(c, "the camera") is not None:
            return
        tool = next((t for t in ("rpicam-hello", "libcamera-hello") if self.sys.which(t)), None)
        if tool is None:
            self.add(c, BLOCKED, "rpicam-hello not installed (sudo apt install rpicam-apps)", "RB-D2")
            return
        text = self.sys.run([tool, "--list-cameras"], timeout=30) or ""
        line = next((x.strip() for x in text.splitlines() if re.match(r"\s*\d+\s*:", x)), "")
        if "imx708" in line:
            wide_noir = "wide_noir" in line
            self.add(c, PASS if wide_noir else FAIL, line,
                     "" if wide_noir else "a different imx708 variant: record it (the Wide NoIR is expected)",
                     camera=line)
        elif self.service_state() == "active":
            self.add(c, BLOCKED, "no camera listed while the service holds it; re-run in the stopped window",
                     "RB-D2; or --with-stopped-stages", raw=text[-400:])
        else:
            self.add(c, FAIL, f"no camera listed with the service stopped: {text.strip()[-200:]}",
                     "power off, reseat the CSI ribbon (RB-D2)", raw=text[-400:])

    def service_state(self) -> Optional[str]:
        text = self.sys.run(["systemctl", "show", SERVICE, "-p", "ActiveState"]) or ""
        return _field(text, "ActiveState", sep="=") or None

    def service_check(self) -> None:
        c = Check("service", 4, "sensordyme.service active, no restarts, expected user/dir", True)
        if self.blocked_off_pi(c, "systemd") is not None:
            return
        text = self.sys.run(["systemctl", "show", SERVICE, "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
                             "-p", "User", "-p", "WorkingDirectory", "-p", "OnFailure"])
        if text is None:
            self.add(c, BLOCKED, "systemctl not available", "RB-C1")
            return
        v = {k: _field(text, k, sep="=") for k in ("ActiveState", "SubState", "NRestarts", "User",
                                                   "WorkingDirectory", "OnFailure")}
        problems = []
        if v["ActiveState"] != "active":
            problems.append(f"service is {v['ActiveState']!r}")
        if v["NRestarts"] not in ("0", ""):
            problems.append(f"NRestarts={v['NRestarts']}")
        if v["WorkingDirectory"] and Path(v["WorkingDirectory"]) != self.install:
            problems.append(f"service runs from {v['WorkingDirectory']}, not --install-dir {self.install}")
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or
                 f"active, NRestarts=0, user {v['User']}, {v['WorkingDirectory']}",
                 "RB-C1 (set U and D from the unit), RB-F4" if problems else "", **v)

    def auth_check(self) -> None:
        c = Check("auth", 4, "dashboard: no token -> 401, header token -> 200, query token -> 401", True)
        if self.blocked_off_pi(c, "the device dashboard") is not None:
            return
        if not self.token:
            self.add(c, BLOCKED, f"${self.args.token_env} is empty", "export the dashboard token (RB-0)")
            return
        url = self.device_url + "/api/health"
        codes = {"none": self.sys.http_status(url),
                 "header": self.sys.http_status(url, {"X-Device-Token": self.token}),
                 "query": self.sys.http_status(f"{url}?token={self.token}")}
        if codes["header"] is None:
            self.add(c, BLOCKED, f"{url} unreachable", "is the service running? RB-C5", **codes)
            return
        ok = codes == {"none": 401, "header": 200, "query": 401}
        self.add(c, PASS if ok else FAIL, f"no token {codes['none']}, header {codes['header']}, query {codes['query']}",
                 "" if ok else "RB-L1 (fail-closed auth)", **codes)

    def health_check(self) -> None:
        c = Check("health_truth", 15, "/api/health: ready, Hailo live, config source, no contradictions", True)
        if self.blocked_off_pi(c, "the device health") is not None:
            return
        if not self.token:
            self.add(c, BLOCKED, f"${self.args.token_env} is empty", "export the dashboard token (RB-0)")
            return
        try:
            self.health = self.sys.get_json(self.device_url + "/api/health", {"X-Device-Token": self.token})
        except cls.UnavailableError as exc:
            self.add(c, BLOCKED, str(exc), "RB-C5")
            return
        result = cls.check_device(self.health or {})
        s = result["summary"]
        accel = s["accelerator"] or {}
        metrics = accel.get("metrics") or {}
        problems = [f"{v['invariant']}: {v['detail']}" for v in result["violations"]]
        if s["monitoring_state"] not in ("ready", "stopped"):
            problems.append(f"monitoring {s['monitoring_state']} ({s['reasons']})")
        if accel.get("state") != "ok" or not metrics.get("hailo_present"):
            problems.append(f"accelerator {accel.get('state')}: {accel.get('detail')}")
        if s["config_source"] != "config":
            problems.append(f"config.source={s['config_source']} (RB-R1)")
        if s["saved_state_faults"]:
            problems.append(f"saved_state faults {s['saved_state_faults']}")
        if self.identity.get("device_id") and s["device_id"] != self.identity["device_id"]:
            problems.append(f"health says {s['device_id']}, identity says {self.identity['device_id']}")
        if s["simulated"] and not self.sys.fake:
            problems.append("the device reports simulated: this is not the hardware path")
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or
                 f"{s['monitoring_state']}, {accel.get('detail')}, inferences {metrics.get('inferences')}",
                 "docs/MONDAY_HARDWARE_SESSION.md §9; RB-G1/RB-R1" if problems else "", **s)

    def perception_check(self) -> None:
        c = Check("perception", 10, "shared perception report real: detector + face components ok", True)
        if self.health is None:
            self.add(c, NOT_RUN if self.sys.fake or self.sys.on_pi() else BLOCKED, "needs the health check",
                     "RB-G3")
            return
        p = ((self.health.get("device") or {}).get("monitoring") or {}).get("perception") or {}
        comps = p.get("components") or {}
        det, face = comps.get("object_detector") or {}, comps.get("face_landmarks") or {}
        problems = []
        if p.get("schema") != "sd-perception/1":
            problems.append(f"schema {p.get('schema')!r}")
        if p.get("simulated") and not self.sys.fake:
            problems.append("perception simulated: True")
        stopped = ((self.health.get("device") or {}).get("protection") or {}).get("state") == "stopped"
        if not stopped:
            if det.get("state") != "ok" or "driver_attention" not in (det.get("consumers") or []):
                problems.append(f"object_detector {det.get('state')} consumers {det.get('consumers')}")
            if face.get("state") != "ok":
                problems.append(f"face_landmarks {face.get('state')}")
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or
                 f"detector runs {det.get('runs')} run_ms {det.get('run_ms')}; face ok",
                 "RB-G3" if problems else "", schema=p.get("schema"), object_detector=det, face_landmarks=face,
                 nodes=p.get("nodes"), tracking=p.get("tracking"))

    def stream_check(self) -> None:
        c = Check("stream", 13, "authenticated live stream: seq rising, age <= 2 s, analysed frames", True)
        if self.blocked_off_pi(c, "the live stream") is not None:
            return
        if not self.token:
            self.add(c, BLOCKED, f"${self.args.token_env} is empty", "export the dashboard token (RB-0)")
            return
        prot = (((self.health or {}).get("device") or {}).get("protection") or {}).get("state")
        if prot == "stopped":
            self.add(c, NOT_RUN, "monitoring is stopped: see the privacy check (step 22)", "resume, then re-run")
            return
        probe = self.sys.stream(self.device_url, self.token, float(self.args.stream_seconds))
        if probe.get("status") is None:
            self.add(c, BLOCKED, f"GET /stream unreachable: {probe.get('error')}", "RB-C5")
            return
        result = cls.check_stream(probe, monitoring_running=True if self.health else None)
        unauth = self.sys.http_status(self.device_url + "/stream")
        result["checks"].insert(0, {"check": "GET /stream without a token answers 401", "ok": unauth == 401,
                                    "detail": unauth})
        try:
            stats = self.sys.get_json(self.device_url + "/api/stream/stats", {"X-Device-Token": self.token})
        except cls.UnavailableError as exc:
            stats = {"error": str(exc)}
        failed = [f"{x['check']}: {x['detail']}" for x in result["checks"] if not x["ok"]]
        v = result["values"]
        self.add(c, FAIL if failed else PASS, "; ".join(failed) or
                 f"{v['parts']} parts in {self.args.stream_seconds}s, seq {v['seq_first_last']}, "
                 f"age p50/max {v['age_ms_p50']}/{v['age_ms_max']} ms, {v['sources']}",
                 "docs/VIDEO_TRANSPORT.md; docs/SUNDAY_DEVICE_ACTIVATION.md step 13" if failed else "",
                 **{k: v[k] for k in ("parts", "seq_first_last", "age_ms_p50", "age_ms_max", "sources", "stream_ids")},
                 stats=stats)

    def privacy_check(self) -> None:
        c = Check("privacy_stopped", 22, "Decision C: no remote camera while stopped (unless owner-enabled)", True)
        if self.blocked_off_pi(c, "the stopped-camera check") is not None:
            return
        prot = (((self.health or {}).get("device") or {}).get("protection") or {}).get("state")
        if self.sys.fake:
            probe = {"status": 200, "parts": []}
            result = cls.check_stream(probe, monitoring_running=False)
            ok = all(x["ok"] for x in result["checks"])
            self.add(c, SIMULATED if ok else FAIL, "stand-in: a stopped device sending no camera parts")
            return
        if prot != "stopped":
            self.add(c, NOT_RUN, "monitoring is running; this check needs it stopped",
                     "in the app: Stop monitoring (step 16), then re-run with --only privacy_stopped")
            return
        probe = self.sys.stream(self.device_url, self.token, float(self.args.stream_seconds))
        if probe.get("status") is None:
            self.add(c, BLOCKED, f"GET /stream unreachable: {probe.get('error')}", "RB-C5")
            return
        result = cls.check_stream(probe, monitoring_running=False,
                                  camera_while_stopped=bool(self.args.camera_while_stopped))
        failed = [f"{x['check']}: {x['detail']}" for x in result["checks"] if not x["ok"]]
        self.add(c, FAIL if failed else PASS, "; ".join(failed) or f"{result['values']['parts']} parts while stopped",
                 "TODO-INTEGRATOR: PRIVSEC's camera-while-stopped setting (docs/SUNDAY_DEVICE_ACTIVATION.md step 22)"
                 if failed else "", **{k: result["values"][k] for k in ("parts", "sources")})

    def backend_check(self) -> None:
        c = Check("backend_truth", 12, "GET /v1/devices row: invariants, user state, agrees with the device", True)
        if not self.args.backend:
            self.add(c, NOT_RUN, "no --backend given", "step 12: re-run with --backend <url> and $SENSORDYME_BEARER")
            return
        if self.sys.fake:
            self.add(c, SIMULATED, "stand-in: backend not contacted in a dry run")
            return
        bearer = self.env.get(self.args.bearer_env, "")
        if not bearer:
            self.add(c, BLOCKED, f"${self.args.bearer_env} is empty", "sign in (POST /v1/login) and export the token")
            return
        try:
            devices = self.sys.get_json(self.args.backend.rstrip("/") + "/v1/devices",
                                        {"Authorization": f"Bearer {bearer}"})
            row = cls.pick_row(devices, self.args.device_id, None if self.args.device_id is not None
                               else self.identity.get("device_id"))
        except cls.UnavailableError as exc:
            self.add(c, BLOCKED, str(exc), "step 12")
            return
        result = cls.check_row(row)
        problems = [f"{v['invariant']}: {v['detail']}" for v in result["violations"]]
        if self.health:
            disagree = cls.agreement(cls.check_device(self.health), result)
            if disagree:
                problems.append(f"device/backend disagree (re-run once: uplink lag up to ~5 s): {disagree}")
        s = result["summary"]
        self.add(c, FAIL if problems else PASS, "; ".join(problems) or
                 f"{s['user_state']}, online {s['online']}, freshness {s['freshness'].get('state')}",
                 "check_live_status.py --samples 3; docs/STATE_MODEL.md" if problems else "", **s)

    # ---------------------------------------------------------------- state-changing (flag + confirmation)
    def stage(self, name: str, *extra: str) -> tuple[int, str]:
        """Run one hwv_stage.py stage; (exit code, its STAGE line)."""
        argv = [sys.executable, str(HERE / "hwv_stage.py"), name, "--install-dir", str(self.install),
                "--out", str(self.out_dir / "stages"), *extra]
        if self.sys.fake:
            argv.append("--fake")
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600, check=False)
        line = next((x for x in reversed(proc.stdout.splitlines()) if x.startswith("STAGE ")), "")
        blocked = next((x.strip() for x in proc.stdout.splitlines() if x.strip().startswith("blocked:")), "")
        return proc.returncode, " ".join(filter(None, (line, blocked))) or (proc.stderr.strip()[-300:])

    def ask(self, text: str, word: str) -> bool:
        if self.confirm is None:
            if not sys.stdin.isatty():
                print("  refused: the confirmation needs a terminal (no stdin tty)")
                return False
            self.confirm = input
        print(text)
        return self.confirm(f"Type {word} to continue, anything else to skip: ").strip() == word

    def stopped_stages(self) -> None:
        names = ["model", "capture", "parity", "latency"] + (["buzzer"] if self.args.with_buzzer else [])
        steps = {"model": 8, "capture": 9, "parity": 9, "latency": 10, "buzzer": 19}
        if not self.args.with_stopped_stages:
            for n in names:
                self.add(Check(f"stage_{n}", steps[n], f"hwv_stage.py {n} (service STOPPED)", True), NOT_RUN,
                         "state-changing: needs --with-stopped-stages and a typed confirmation",
                         f"approved window: re-run with --with-stopped-stages{' --with-buzzer' if n == 'buzzer' else ''}"
                         " (or run the stage by hand: docs/MONDAY_HARDWARE_SESSION.md §6.2)")
            return
        if self.sys.fake:
            for n in names:
                extra = ("--bundle", str(self.out_dir / "capture_kg")) if n in ("capture", "parity") else ()
                code, out = self.stage(n, *extra)
                status = {0: PASS, 1: FAIL, 2: BLOCKED, 3: SIMULATED}.get(code, FAIL)
                self.add(Check(f"stage_{n}", steps[n], f"hwv_stage.py {n} --fake", True), status, out)
            return
        before = self.service_state()
        warn = ("\n  This STOPS sensordyme.service: the device monitors NOTHING until it is started again.\n"
                "  Never on a unit protecting a driver. Stages: " + ", ".join(names) +
                ("\n  The buzzer stage is LOUD: warn everyone present." if self.args.with_buzzer else ""))
        if not self.ask(warn, "STOP"):
            for n in names:
                self.add(Check(f"stage_{n}", steps[n], f"hwv_stage.py {n}", True), NOT_RUN, "operator declined")
            return
        stopped_here = False
        try:
            if before == "active":
                self.call(["sudo", "systemctl", "stop", SERVICE])
                stopped_here = True
            bundle = self.out_dir / f"capture_kg_{time.strftime('%Y%m%dT%H%M%S')}"
            for n in names:
                extra = ("--bundle", str(bundle)) if n in ("capture", "parity") else ()
                code, out = self.stage(n, *extra)
                status = {0: PASS, 1: FAIL, 2: BLOCKED, 3: SIMULATED}.get(code, FAIL)
                self.add(Check(f"stage_{n}", steps[n], f"hwv_stage.py {n}", True), status, out,
                         "" if status == PASS else "docs/MONDAY_HARDWARE_SESSION.md §9", exit_code=code)
        finally:
            if stopped_here:
                self.call(["sudo", "systemctl", "start", SERVICE])
                time.sleep(5)
                after = self.service_state()
                c = Check("service_restored", 4, "sensordyme.service started again after the stopped stages", True)
                self.add(c, PASS if after == "active" else FAIL, f"is-active: {after}",
                         "" if after == "active" else "start it now: sudo systemctl start sensordyme.service; RB-O1")

    def thermal_note(self) -> None:
        self.add(Check("stage_thermal", 11, "30-min soak under the running service", True), NOT_RUN,
                 "never run from the preflight (30 minutes)",
                 "SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py thermal --minutes 30 --install-dir $D")

    # ---------------------------------------------------------------- the run
    def run(self) -> dict:
        before = _sha(self.config_path())
        started = hc.utc_now()
        print(f"== Sunday preflight{' [--fake: stand-ins, NOT evidence]' if self.sys.fake else ''} "
              f"install={self.install} device={self.device_url}", flush=True)
        order: list[Callable[[], None]] = [
            self.identity_check, self.power_check, self.os_check, self.packages_check, self.revision_check,
            self.pins_check, self.model_artifact_check, self.hailo_check, self.parse_hef_check, self.camera_check,
            self.service_check, self.auth_check, self.health_check, self.perception_check, self.stream_check,
            self.backend_check, self.privacy_check, self.stopped_stages, self.thermal_note,
        ]
        only = set(self.args.only or [])
        for fn in order:
            name = fn.__name__.removesuffix("_check")
            if only and name not in only and not (name == "health" and only & {"privacy", "perception", "stream"}):
                continue
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
                self.add(Check(name, 0, f"{name} crashed", True), FAIL, repr(exc), "a bug in the preflight: report it")
        after = _sha(self.config_path())
        cfg = Check("config_unchanged", 0, "config.yaml byte-identical before and after (never edited)", False)
        if before is None:
            self.add(cfg, NOT_RUN, f"no {self.config_path()}")
        else:
            self.add(cfg, PASS if before == after else FAIL,
                     "unchanged" if before == after else "config.yaml CHANGED during the preflight",
                     "" if before == after else "investigate: nothing here writes it", sha256_before=before,
                     sha256_after=after)
        return self.summary(started)

    def summary(self, started: str) -> dict:
        statuses = [c.status for c in self.checks]
        overall = FAIL if FAIL in statuses else BLOCKED if BLOCKED in statuses else \
            SIMULATED if SIMULATED in statuses else PASS
        if self.sys.fake and overall == PASS:
            overall = SIMULATED
        first = next((c for c in self.checks if c.status in (FAIL, BLOCKED)), None)
        if first is not None:
            nxt = f"step {first.step} ({first.id}): {first.next or first.detail}"
        elif self.sys.fake:
            nxt = "this was a dry run: on the Pi run it without --fake (docs/SUNDAY_DEVICE_ACTIVATION.md §0)"
        else:
            pending = next((c for c in self.checks if c.status == NOT_RUN and c.next), None)
            nxt = (f"step {pending.step} ({pending.id}): {pending.next}" if pending else
                   f"read-only preflight complete: continue at {DOC} step 6")
        return {"tool": "sunday_preflight", "verdict": overall, "exit_code": EXIT[overall],
                "simulated": self.sys.fake, "on_pi": self.sys.on_pi(), "started": started, "finished": hc.utc_now(),
                "install_dir": str(self.install), "device": self.device_url, "backend": self.args.backend,
                "package_commit": hc.git_commit(REPO_ROOT), "host": platform.node(),
                "counts": {s: statuses.count(s) for s in (PASS, FAIL, BLOCKED, SIMULATED, NOT_RUN)},
                "next_manual_step": nxt, "checks": [c.to_dict() for c in self.checks]}


# ------------------------------------------------------------------ helpers
def _field(text: str, key: str, sep: str = ":") -> str:
    for line in text.splitlines():
        k, s, v = line.partition(sep)
        if s and k.strip() == key:
            return v.strip()
    return ""


def _sha(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _requirement_pins() -> dict[str, str]:
    # M8.5: the pins that apply ON THE DEVICE (markers evaluated for Linux/aarch64/3.11):
    # simpleaudio is a non-Linux pin, gpiozero/lgpio are device pins.
    from scripts.release.check_device_deps import device_pins

    return device_pins(REPO_ROOT / "requirements-device.txt")


def markdown(report: dict) -> str:
    lines = [f"# Sunday preflight: {report['verdict']}", "",
             (f"- started {report['started']}, finished {report['finished']}, host `{report['host']}`, "
              f"on a Pi: {report['on_pi']}, simulated: {report['simulated']}"),
             (f"- package commit `{report['package_commit']}`, install `{report['install_dir']}`, device "
              f"`{report['device']}`, backend `{report['backend']}`"),
             f"- counts: {report['counts']}", f"- **next manual step:** {report['next_manual_step']}", ""]
    if report["simulated"]:
        lines += ["> DRY RUN with stand-in answers. Nothing here is hardware evidence.", ""]
    lines += ["| Step | Check | HW | Result | Detail | Next |", "|---|---|---|---|---|---|"]
    for c in report["checks"]:
        detail = str(c["detail"]).replace("|", "\\|")[:300]
        lines.append(f"| {c['step']} | {c['id']} | {'yes' if c['hardware'] else 'no'} | **{c['status']}** | "
                     f"{detail} | {str(c['next']).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--install-dir", type=Path, default=hc.DEFAULT_INSTALL_DIR)
    p.add_argument("--device", default="http://localhost:8899", help="the device dashboard base URL")
    p.add_argument("--backend", default=None, help="backend base URL for the GET /v1/devices check")
    p.add_argument("--device-id", type=int, default=None, help="backend device id (default: match the identity)")
    p.add_argument("--expect-commit", default=None, help="the approved RC commit (git rev-parse $RC)")
    p.add_argument("--token-env", default="SENSORDYME_TOKEN")
    p.add_argument("--bearer-env", default="SENSORDYME_BEARER")
    p.add_argument("--stream-seconds", type=float, default=5.0)
    p.add_argument("--camera-while-stopped", action="store_true",
                   help="the owner ENABLED the camera-only view while stopped (Decision C)")
    p.add_argument("--only", nargs="*", default=None,
                   help="run only these checks (identity power os packages revision pins model_artifact hailo "
                        "parse_hef camera service auth health perception stream backend privacy)")
    p.add_argument("--with-stopped-stages", action="store_true", help="CHANGE: stop the service, run stages")
    p.add_argument("--with-buzzer", action="store_true", help="CHANGE: also the buzzer stage (loud)")
    p.add_argument("--fake", action="store_true", help="dry run with stand-in answers (SIMULATED, never PASS)")
    p.add_argument("--out", type=Path, default=None, help="evidence directory (default ~/hwval/sunday)")
    return p


def main(argv: Optional[list[str]] = None, *, system: Optional[RealSystem] = None,
         env: Optional[dict[str, str]] = None, confirm: Optional[Callable[[str], str]] = None,
         call: Optional[Callable[[list[str]], int]] = None) -> int:
    args = build_parser().parse_args(argv)
    sys.dont_write_bytecode = True
    if args.with_buzzer:
        args.with_stopped_stages = True
    if args.out is None:
        # As hwv_stage.py: a dry run's reports go to a temp dir, never beside real evidence.
        args.out = Path(tempfile.gettempdir()) / "hwval-sunday-fake" if args.fake else Path.home() / "hwval" / "sunday"
    ok, why = hc.ensure_outside(args.out, args.install_dir)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return EXIT[BLOCKED]
    args.out.mkdir(parents=True, exist_ok=True)
    if system is None:
        system = FakeSystem(args.install_dir) if args.fake else RealSystem()
    report = Preflight(args, system, env=env, confirm=confirm, call=call).run()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    json_path = args.out / f"sunday_preflight_{stamp}.json"
    md_path = args.out / f"sunday_preflight_{stamp}.md"
    hc.write_json(json_path, report)
    md_path.write_text(markdown(report))
    print(f"\nPREFLIGHT: {report['verdict']}  {report['counts']}")
    if report["simulated"]:
        print("  (dry run with stand-in answers: NOT hardware evidence)")
    print(f"  evidence: {json_path}\n            {md_path}")
    print(f"NEXT: {report['next_manual_step']}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
