#!/usr/bin/env python3
"""Demo preflight: is this device ready to be demonstrated, right now?

DELIBERATELY DIFFERENT FROM scripts/preflight.py. That one runs as the service's
ExecStartPre, before anything is up, and it opens the camera itself. This one
answers a different question -- "is the thing that is ALREADY RUNNING fit to
demo" -- so it must never open the camera or the NPU, because the service holds
both. Every hardware fact here is read from the live /api/health endpoint, which
is the same evidence the app and the dashboard see. A preflight that proved the
camera worked by taking it away from the running service would be worse than no
preflight.

It reports DEMO READY or DEMO NOT READY. There is no third answer, and a
degraded device is never READY -- the entire point is that a device which cannot
see must not be walked on stage looking green.

Usage:
    python scripts/demo_preflight.py                # check the local device
    python scripts/demo_preflight.py --host pi.local
    python scripts/demo_preflight.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

#: Packages that must NOT be installed on a device we convey. `ultralytics` is
#: AGPL-3.0; torch/torchvision only ever arrived as its dependencies. Phase 5
#: found all three installed on this very device after Phase 4 had "removed"
#: them from the repo's requirements -- the repo was fixed and the hardware was
#: never re-provisioned. That is why this is a preflight check and not a comment.
FORBIDDEN_PACKAGES = ("ultralytics", "torch", "torchvision")

#: A frame older than this means the pipeline has stalled, not that we caught it
#: between ticks.
FRESH_FRAME_S = 5.0


def _oneline(text: str, limit: int = 88) -> str:
    """Collapse a detail to one readable line.

    `/api/health` can carry an entire multi-line RuntimeError as a component's
    detail -- core.detectors' "no backend available" message is nine lines of
    remediation advice. Printed raw it buries the actual finding under a wall of
    text, which is exactly the wrong failure mode for something read under demo
    pressure. The full text is still in the journal and in --json.
    """
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass
class Check:
    name: str
    state: str
    detail: str
    fix: str = ""

    def __post_init__(self) -> None:
        self.detail = _oneline(self.detail)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, fix: str = "", warn_only: bool = False) -> None:
        state = PASS if ok else (WARN if warn_only else FAIL)
        self.checks.append(Check(name, state, detail, "" if ok else fix))

    @property
    def ready(self) -> bool:
        return not any(c.state == FAIL for c in self.checks)


def _health(host: str, port: int, timeout: float = 8.0) -> dict | None:
    url = f"http://{host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _state(host: str, port: int, timeout: float = 8.0) -> dict | None:
    """The dashboard's own view, fetched independently of /api/health."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/state", timeout=timeout
        ) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _device(health: dict) -> dict:
    """The device half of the health payload, which is nested under "device"."""
    return health.get("device", health)


def run(host: str, port: int) -> Report:
    report = Report()
    config_path = REPO / "config.yaml"

    # ---- 16. files and directories, first: everything below reads them -------
    import yaml

    if not config_path.exists():
        report.add("config present", False, f"{config_path} missing", "restore config.yaml")
        return report
    config = yaml.safe_load(config_path.read_text())
    for directory in ("data",):
        path = REPO / directory
        report.add(
            f"directory {directory}/",
            path.is_dir() and os.access(path, os.W_OK),
            f"{path} {'writable' if path.is_dir() else 'MISSING'}",
            f"mkdir -p {path}",
        )

    # ---- 11 / 12. the dashboard, and whether it claims to be monitoring ------
    dash = config.get("dashboard", {}) or {}
    port = port or int(dash.get("port", 8899))
    # TWO reads, deliberately. fps and the frame-interval percentiles are both
    # WINDOW measurements keyed per caller: the first call from a new caller
    # registers a baseline and honestly reports nothing, because there is no
    # previous read to measure from. A preflight run right after a service
    # restart -- which is precisely when an operator runs it -- therefore saw
    # `fps: None` and no stall counters, and failed a healthy device.
    #
    # The priming read is discarded; the second is the measurement. The gap
    # between them IS the window, so it must be long enough to contain frames.
    if _health(host, port) is not None:
        time.sleep(2.0)
    health = _health(host, port)
    if health is None:
        report.add(
            "dashboard reachable", False, f"no /api/health on {host}:{port}",
            "is the service running?  systemctl status sensordyme",
        )
        # Without health there is nothing live to check; report what we have.
        return report
    report.add("dashboard reachable", True, f"{host}:{port} answered")
    device = _device(health)

    monitoring_active = device.get("monitoring_active")
    report.add(
        "monitoring active", monitoring_active is True,
        f"monitoring_active={monitoring_active}"
        + (f" — {device.get('inference', {}).get('detail', '')}" if monitoring_active is not True else ""),
        "the device is DEGRADED: it is not analysing anything. Check the "
        "accelerator and the model before demonstrating.",
    )

    # ---- the supervisor's own verdict, and WHICH component it blames --------
    # /api/health is {**health.snapshot(), "device": ...}. Everything else here
    # reads the "device" half; the snapshot half carries a top-level `state` and
    # a per-component breakdown covering things the device half never mentions
    # (cloud uplink, thermal, wifi, provisioning, the hardware peripherals).
    # Without this the preflight could report DEMO READY while the supervisor
    # said "degraded" — which is exactly what happened on this device, where the
    # cloud uplink is offline because the backend is not running.
    #
    # WARN, not FAIL: an offline uplink does not stop a local demo, and failing
    # the gate for it would train the operator to ignore the gate. The rule is
    # that the operator must SEE it, not that it must block.
    supervisor_state = health.get("state")
    components = health.get("components") or {}
    unhealthy = [
        f"{name} ({c.get('detail', '')[:40]})"
        for name, c in components.items()
        if isinstance(c, dict) and str(c.get("state")) not in ("ok", "healthy")
    ]
    report.add(
        "supervisor state", not unhealthy,
        f"{supervisor_state}"
        + (f" — {', '.join(unhealthy)}" if unhealthy else " — all components healthy"),
        "a supervised component is unhealthy; check whether it matters for the "
        "demo you are about to give",
        warn_only=True,
    )

    # ---- 1 / 2. camera present and actually producing frames ----------------
    camera = device.get("camera", {})
    cam_metrics = camera.get("metrics", {})
    report.add(
        "camera capturing", camera.get("state") == "ok",
        f"{camera.get('state')} — {camera.get('detail')}",
        "check the ribbon cable and `libcamera-hello --list-cameras`",
    )
    frame_age = cam_metrics.get("frame_age_s")
    report.add(
        "frames are fresh", frame_age is not None and frame_age < FRESH_FRAME_S,
        f"newest frame {frame_age}s old" if frame_age is not None else "no frame age reported",
        "frames have stopped arriving even though capture reports running",
    )
    fps = cam_metrics.get("fps")
    report.add(
        "frame rate", fps is not None and fps >= 10.0,
        f"{fps} fps",
        "below 10 fps starves every module's timing assumptions",
        warn_only=fps is not None and fps >= 5.0,
    )

    # ---- 14. stalls that would DISCARD in-progress evidence ------------------
    stalls = cam_metrics.get("evidence_losing_stalls_1s")
    report.add(
        "no evidence-losing stalls", stalls == 0,
        f"{stalls} gaps over 1.0s since the last health sample",
        "EventEngine discards an unconfirmed track across a >1s gap, so a fall "
        "or a choking event in progress can be silently dropped",
    )

    # ---- 3 / 4. the accelerator, by completed work not by configuration -----
    accel_component = device.get("accelerator", {})
    accel_metrics = accel_component.get("metrics", {})
    report.add(
        "Hailo present", accel_metrics.get("hailo_present") is True,
        f"hailo_present={accel_metrics.get('hailo_present')}, "
        f"runtime={accel_metrics.get('runtime_available')}",
        "reseat the AI HAT+ and check `hailortcli fw-control identify`",
    )
    inferences = accel_metrics.get("inferences") or 0
    last_s = accel_metrics.get("last_inference_s")
    npu_live = inferences > 0 and last_s is not None and last_s < 10.0
    report.add(
        "NPU actually inferring", npu_live,
        f"{inferences} inferences"
        + (f", last {last_s}s ago" if last_s is not None else ", none ever"),
        "the accelerator is present but nothing is running on it — this is the "
        "difference between a seated HAT and a working one",
    )

    # ---- 5 / 6. the model, by hash, not by filename -------------------------
    accel_cfg = config.get("accelerator", {}) or {}
    hef_name = accel_cfg.get("yolo_hef")
    pinned = accel_cfg.get("yolo_hef_sha256")
    hef = (REPO / hef_name) if hef_name else None
    if not hef_name:
        report.add("model configured", False, "accelerator.yolo_hef is unset",
                   "set accelerator.yolo_hef to the provisioned .hef")
    elif hef is None or not hef.exists():
        report.add("model present", False, f"{hef_name} not found",
                   "re-run setup_pi.sh to provision the pinned model")
    else:
        digest = hashlib.sha256(hef.read_bytes()).hexdigest()
        report.add(
            "model hash matches pin", bool(pinned) and digest == pinned,
            f"{hef.name} sha256 {digest[:16]}… "
            + ("matches" if digest == pinned else f"EXPECTED {str(pinned)[:16]}…"),
            "this device is NOT running the binary the config and the benchmarks "
            "describe — re-provision with setup_pi.sh",
        )
        # Exactly one active .hef: two is an invitation to demo the wrong one.
        actives = sorted(p.name for p in hef.parent.glob("*.hef"))
        report.add(
            "exactly one active model", len(actives) == 1,
            f"{len(actives)} .hef in {hef.parent.name}/: {actives}",
            "move retired models into an archive subdirectory",
        )

    # ---- 7. no forbidden CPU-fallback packages ------------------------------
    present = [p for p in FORBIDDEN_PACKAGES if importlib.util.find_spec(p) is not None]
    report.add(
        "no AGPL/CPU-fallback packages", not present,
        "none installed" if not present else f"INSTALLED: {', '.join(present)}",
        "ultralytics is AGPL-3.0 and its presence also re-enables a silent CPU "
        "fallback: pip uninstall ultralytics torch torchvision",
    )
    stray = [
        p.name for p in REPO.rglob("*.pt")
        if "site-packages" not in str(p) and ".venv" not in str(p)
    ]
    report.add(
        "no stray .pt weights", not stray,
        "none" if not stray else f"found: {stray}",
        "YOLO .pt checkpoints carry AGPL-3.0; delete them from the device",
    )

    # ---- 8 / 18. the mode is the intended one, and nothing overrides it -----
    configured = config.get("module")
    running = (device.get("inference", {}).get("detail") or "")
    if monitoring_active is not True:
        # Degraded: `running` is the degradation reason, not a module name.
        # Failing this check too would report one problem as two and bury the
        # real one under the longer message.
        report.add(
            "mode is the configured one", True,
            f"not applicable — device is degraded (configured: '{configured}')",
        )
    else:
        report.add(
            "mode is the configured one",
            bool(configured) and configured in running,
            f"config says '{configured}', device reports '{running}'",
            "an AI-configured rule set or a preset is overriding config.yaml",
        )
    override = _override_state()
    report.add(
        "no unexpected override", not override,
        override or "no AI-custom rule set is overriding the configured mode",
        "clear the custom rule set from the app, or accept that the demo runs "
        "AI-Custom mode instead of the configured module",
    )

    # ---- 9. the thresholds that will actually be applied --------------------
    driver = (config.get("modules", {}) or {}).get("driver_attention", {}) or {}
    phone = driver.get("yolo_confidence_threshold")
    person = driver.get("person_confidence_threshold")
    report.add(
        "thresholds are expressible", phone is not None and phone >= 0.20,
        f"phone {phone}, person {person}",
        "an NMS-on-chip .hef compiles its score threshold in and silently floors "
        "anything under 0.20 — the config would stop describing the detector",
    )
    report.add(
        "person/phone thresholds are separate",
        phone is not None and person is not None and person > phone,
        f"person {person} > phone {phone}" if person and phone else "one is unset",
        "person and phone need opposite ends of the confidence range; sharing "
        "one threshold reintroduces spurious person boxes",
    )

    # ---- 10. alerts are configured at all -----------------------------------
    alerts = config.get("alerts", {}) or {}
    loud = alerts.get("loud_types") or []
    report.add(
        "alerts configured", bool(loud),
        f"{len(loud)} loud alert types, {len(alerts.get('soft_types') or [])} soft",
        "no alert types are configured, so nothing will fire during the demo",
    )

    # ---- the two surfaces an operator reads must agree with each other -----
    # /api/health drives the status card; /api/state drives the live page the
    # audience actually sees. They are computed independently, so they can
    # disagree -- and a dashboard showing one module while health reports another
    # is precisely the "config and runtime disagree" failure, in the one place a
    # judge is looking. Cheap to check, and neither surface can self-detect it.
    state = _state(host, port)
    if state is None:
        report.add(
            "dashboard state API", False, f"no /api/state on {host}:{port}",
            "the live page cannot render; health may still be fine",
        )
    else:
        state_module = str(state.get("module") or "")
        health_module = (device.get("inference", {}).get("detail") or "")
        report.add(
            "health and dashboard agree",
            bool(state_module) and state_module in health_module,
            f"/api/state says '{state_module}', /api/health says '{health_module}'",
            "the live page and the status card disagree about what is running",
        )
        report.add(
            "event surface reachable", "total_events" in state,
            f"total_events={state.get('total_events')}, "
            f"{len(state.get('events') or [])} in the recent buffer",
            "the dashboard cannot report events, so nothing will appear on stage "
            "even if the pipeline produces them",
        )

    # ---- 13. throttling that is happening NOW, not once, historically -------
    system = device.get("system", {})
    now_bits = system.get("metrics", {}).get("throttle_now")
    report.add(
        "no live throttling", not now_bits,
        f"throttle_now={now_bits}"
        + (f" (since boot: {system.get('metrics', {}).get('throttle_since_boot')})"
           if system.get("metrics", {}).get("throttle_since_boot") else ""),
        "the supply is sagging under load right now — use a 5A PSU",
    )

    # ---- 15. the event pipeline can actually record something ---------------
    events_csv = REPO / (config.get("logging", {}) or {}).get("csv_path", "data/events.csv")
    report.add(
        "event log writable",
        events_csv.parent.is_dir() and os.access(events_csv.parent, os.W_OK),
        f"{events_csv}"
        + (f" ({sum(1 for _ in events_csv.open()) - 1} events)" if events_csv.exists() else " (not yet created)"),
        "events cannot be persisted, so nothing will appear in history",
    )

    # ---- 17. clock and logging ----------------------------------------------
    report.add(
        "clock is sane", time.time() > 1_700_000_000,
        time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "an unset clock makes every event timestamp meaningless",
    )
    if shutil.which("systemctl"):
        active = _run("systemctl is-active sensordyme")
        report.add(
            "service active", active == "active", f"systemd reports '{active}'",
            "systemctl status sensordyme",
        )
        restarts = _run("systemctl show sensordyme -p NRestarts --value")
        report.add(
            "service is stable", restarts in ("0", ""),
            f"NRestarts={restarts}",
            "the service has been restarting — check the journal before demoing",
            warn_only=True,
        )
    return report


def _override_state() -> str:
    """Whether a persisted AI-configured rule set outranks config.yaml's module.

    Phase 5 found a rule saved thirteen days earlier silently running occupancy
    on a device configured for driver_attention, with nothing in the logs saying
    so. The demo must not be able to start in a mode nobody chose.
    """
    path = REPO / "data" / "monitoring.json"
    if not path.exists():
        return ""
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return f"data/monitoring.json is unreadable ({exc}) — behaviour undefined"
    if not (saved.get("rules") and saved.get("overrides_mode")):
        return ""
    age = ""
    if saved.get("updated_at"):
        days = (time.time() - float(saved["updated_at"])) / 86400.0
        age = f", saved {days:.1f} days ago"
    return (
        f"an AI-configured rule set is ACTIVE and overrides config.yaml's module"
        f"{age}: {saved.get('instruction') or '(no instruction recorded)'!r}"
    )


def _run(command: str) -> str:
    try:
        return subprocess.run(
            command.split(), capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=0, help="default: config.yaml's dashboard port")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(args.host, args.port)

    if args.json:
        print(json.dumps({
            "ready": report.ready,
            "checks": [vars(c) for c in report.checks],
        }, indent=2))
        return 0 if report.ready else 1

    width = max(len(c.name) for c in report.checks) + 2
    print("=" * 72)
    print("SensorDyme demo preflight")
    print("=" * 72)
    for check in report.checks:
        print(f"  [{check.state:4}] {check.name:{width}} {check.detail}")
        if check.fix:
            for line in _wrap(check.fix, 66):
                print(f"          -> {line}")
    print("-" * 72)
    failed = [c for c in report.checks if c.state == FAIL]
    if report.ready:
        warned = [c for c in report.checks if c.state == WARN]
        print("DEMO READY" + (f"  ({len(warned)} warning(s) above)" if warned else ""))
        return 0
    print(f"DEMO NOT READY — {len(failed)} check(s) failed:")
    for check in failed:
        print(f"  * {check.name}: {check.detail}")
    return 1


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
