#!/usr/bin/env python3
"""Boot preflight: verify the platform before the main service starts inference.

Run by sensordyme.service as ExecStartPre. Checks, in order:

  1. config.yaml parses and validates          -> FATAL on error (exit 1)
  2. camera present and produces a frame       -> reported; service still starts
                                                  (the runtime retries and the
                                                  health monitor keeps trying)
  3. AI HAT+ / Hailo present + runtime usable  -> reported; FATAL only when
                                                  config sets accelerator.required
  4. disk space for logs                       -> reported

Results go to stdout (journalctl) and data/preflight.json so the dashboard and
app can show what boot looked like. Philosophy: only an unusable *config* stops
boot — a missing camera or HAT boots into degraded mode where health monitoring,
provisioning, and the app all still work and can say what's wrong.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root


def main() -> int:
    from core.config_schema import validate
    from main import load_config

    results: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": {}}
    fatal = False

    # 1. config -----------------------------------------------------------------
    config: dict = {}
    try:
        config = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
        errors, warnings = validate(config)
        results["checks"]["config"] = {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
        }
        if errors:
            fatal = True
    except Exception as exc:  # noqa: BLE001
        results["checks"]["config"] = {"ok": False, "errors": [str(exc)], "warnings": []}
        fatal = True

    # 2. camera -----------------------------------------------------------------
    try:
        from core.camera import Camera

        camera = Camera(config.get("camera", {}))
        frame = camera.read()  # prove frames flow, not just that open succeeded
        camera.release()
        results["checks"]["camera"] = {
            "ok": True,
            "detail": f"frame {frame.shape[1]}x{frame.shape[0]}",
        }
    except Exception as exc:  # noqa: BLE001
        results["checks"]["camera"] = {"ok": False, "detail": str(exc)}

    # 3. accelerator ------------------------------------------------------------
    from core import accel

    info = accel.detect()
    accel_cfg = config.get("accelerator", {})
    accel_ok = info.usable or not accel_cfg.get("required", False)
    results["checks"]["accelerator"] = {
        "ok": accel_ok,
        "present": info.present,
        "runtime": info.runtime_available,
        "detail": info.detail,
    }
    if not accel_ok:
        fatal = True

    # 3b. detection backend -----------------------------------------------------
    # Since Phase 4 a device ships WITHOUT ultralytics (AGPL-3.0), so the Hailo
    # path is not an optimisation any more — it is the only detector present. A
    # missing .hef therefore stops object detection completely, and does it deep
    # inside the inference loop where it reads as a warm-up warning and a restart
    # loop rather than one clear sentence at boot. This says the sentence.
    #
    # PROBES, deliberately; it does not call core.detectors.require_backend.
    # That would LOAD the model, which constructs a Hailo VDevice inside
    # ExecStartPre — and the service starts microseconds after this process
    # exits, so it would be racing its own preflight for the NPU on every boot.
    # A weaker claim that cannot break startup beats a stronger one that can.
    #
    # Non-fatal by design, matching the camera and HAT checks above: a device
    # that cannot detect should still boot so health, provisioning and the app
    # come up and can SAY it cannot detect. Booting into a silent restart loop
    # tells the owner nothing.
    hef = accel_cfg.get("yolo_hef")
    hef_exists = bool(hef) and Path(hef).exists()
    try:
        import importlib.util

        cpu_fallback = importlib.util.find_spec("ultralytics") is not None
    except (ImportError, ValueError):
        cpu_fallback = False

    if hef_exists and info.usable:
        backend_detail = f"Hailo ({hef})"
    elif cpu_fallback:
        backend_detail = "CPU (ultralytics) — a dev machine, or a device that should not ship as-is"
    elif hef and not hef_exists:
        backend_detail = (
            f"NONE — accelerator.yolo_hef points at a missing file ({hef}) and "
            "ultralytics is not installed. Object detection will not run. "
            "Re-run setup_pi.sh to provision the model."
        )
    elif not hef:
        backend_detail = (
            "NONE — accelerator.yolo_hef is unset and ultralytics is not installed. "
            "Object detection will not run. Set accelerator.yolo_hef in config.yaml."
        )
    else:
        backend_detail = (
            f"NONE — the accelerator is unusable ({info.detail}) and there is no CPU "
            "fallback installed. Object detection will not run. Reseat the AI HAT+."
        )
    results["checks"]["detection_backend"] = {
        "ok": bool((hef_exists and info.usable) or cpu_fallback),
        "detail": backend_detail,
    }

    # 4. disk -------------------------------------------------------------------
    usage = shutil.disk_usage(".")
    free_mb = usage.free // (1024 * 1024)
    results["checks"]["disk"] = {"ok": free_mb > 200, "free_mb": free_mb}

    results["fatal"] = fatal
    out = Path("data/preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    for name, check in results["checks"].items():
        status = "OK " if check.get("ok") else "FAIL"
        print(f"preflight {status} {name}: {check.get('detail', check)}")
    if fatal:
        print("preflight: FATAL problem above — refusing to start the service")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
