#!/usr/bin/env python3
"""Deployment preflight: refuse to deploy onto a device whose configuration
cannot do the job its role says it does. Run BEFORE code reaches a device.

`make deploy` excludes config.yaml (it is device state: keys, camera
orientation, its role), so new code lands on whatever config the unit already
has. This checks that config against the code about to be deployed:

  1. it parses, and the schema accepts it (core/config_schema.validate, which
     includes the role's safety floor)
  2. the role it declares is the role this deployment is for (--expect-role).
     A config with no `role:` runs as "general" with no safety floor; that is
     legitimate for a general-purpose unit and is never changed here, but it is
     reported, and refused when a protected role was expected
  3. what the role needs is configured: an audible alert output, a detector
     for roles that need object detection, the module the role requires
  4. the object-detection model the device holds is the one the config pins
     (--hef-sha256, measured on the device). setup_pi.sh accepts an "evaluated"
     build that the runtime then refuses (audit F1): caught here instead.

It never edits a config. A failure prints what to change and exits 1.

    python scripts/deploy_preflight.py --config device-config.yaml \
        --expect-role driver_safety --hef-sha256 <sha256 of the device's .hef>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from core import roles
from core.config_schema import validate

#: Config keys that name an object-detection model for a module.
_WEIGHTS_KEYS = ("yolo_weights", "model_weights", "phone_weights", "weights")


@dataclass
class Report:
    role: str
    role_declared: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check(
    config_text: str,
    *,
    expect_role: Optional[str] = None,
    hef_sha256: Optional[str] = None,
) -> Report:
    """Every reason this config should not receive a deployment (pure; no I/O)."""
    try:
        config = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        return Report("unknown", False, errors=[f"config.yaml does not parse: {exc}"])
    if not isinstance(config, dict):
        return Report("unknown", False, errors=["config.yaml must be a YAML mapping"])

    declared = config.get("role")
    report = Report(str(declared or roles.DEFAULT_ROLE), declared is not None)
    errors, warnings = validate(config)
    report.errors += [f"config: {e}" for e in errors]
    report.warnings += [f"config: {w}" for w in warnings]
    if declared is not None and declared not in roles.ROLES:
        return report  # validate() already explains the unknown role

    role = roles.ROLES[report.role]
    if expect_role is not None:
        if expect_role not in roles.ROLES:
            report.errors.append(f"--expect-role {expect_role!r} is not a role ({', '.join(sorted(roles.ROLES))})")
        elif expect_role != role.name:
            how = "declares no role, so it runs as 'general' (no safety floor)" if declared is None \
                else f"declares role '{role.name}'"
            report.errors.append(
                f"this deployment is for role '{expect_role}', but the device's config.yaml {how}. "
                f"MIGRATION: add `role: {expect_role}` to that device's config.yaml (deliberately, by a "
                f"person: deploys never change a device's role), then re-run this preflight."
            )
    elif declared is None:
        report.warnings.append(
            "no `role:` declared: this device runs as 'general' with no safety floor. Correct for a "
            "general-purpose unit; if it protects a driver, add `role: driver_safety` to its config.yaml."
        )

    report.errors += _role_requirements(config, role)
    report.errors += _model_compatibility(config, hef_sha256, role)
    return report


def _role_requirements(config: dict, role: roles.Role) -> list[str]:
    """What the role's required capabilities need from the configuration."""
    errors: list[str] = []
    needs = role.required_capabilities
    alerts = config.get("alerts") if isinstance(config.get("alerts"), dict) else {}
    sinks = alerts.get("sinks", ["console"]) if isinstance(alerts, dict) else []
    if "audible_alert" in needs and not roles.AUDIBLE_SINKS & set(sinks or []):
        errors.append(f"role '{role.name}' needs an audible alert output; add gpio_buzzer to alerts.sinks")
    if "objects" in needs:
        module = config.get("module")
        module_cfg = (config.get("modules") or {}).get(module) or {}
        accel_cfg = config.get("accelerator") or {}
        if not any(module_cfg.get(k) for k in _WEIGHTS_KEYS):
            errors.append(f"role '{role.name}' needs object detection, but module '{module}' configures no detector")
        elif not (accel_cfg.get("enabled", True) and accel_cfg.get("yolo_hef")):
            errors.append(
                f"role '{role.name}' needs object detection, and on the device the Hailo model is the only "
                "detector (ultralytics is not installed): set accelerator.yolo_hef"
            )
    return errors


def _model_compatibility(config: dict, hef_sha256: Optional[str], role: roles.Role) -> list[str]:
    accel_cfg = config.get("accelerator") or {}
    pin = str(accel_cfg.get("yolo_hef_sha256") or "").strip().lower()
    if hef_sha256 is None or not accel_cfg.get("yolo_hef"):
        return []
    have = hef_sha256.strip().lower()
    if not have:
        consequence = "FAILED" if "objects" in role.required_capabilities else "DEGRADED (no object detection)"
        return [(f"the device has no object-detection model at {accel_cfg['yolo_hef']}: it would run "
                 f"{consequence}. Provision it with setup_pi.sh.")]
    if pin and have != pin:
        return [(
            f"the device's model ({have[:12]}…) is not the build config.yaml pins ({pin[:12]}…); the "
            "runtime refuses it at load, so object detection would be unavailable. Re-provision the "
            "pinned build (setup_pi.sh), or update yolo_hef_sha256 only for an intended model change."
        )]
    return []


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # -OO strips docstrings
    parser.add_argument("--config", required=True, help="the DEVICE's config.yaml (fetched from it)")
    parser.add_argument("--expect-role", default=None, help="the role this deployment is for")
    parser.add_argument("--hef-sha256", default=None, help="sha256 of the .hef on the device ('' = none)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--print-hef-path", action="store_true",
                        help="print accelerator.yolo_hef from the config and exit (for the Makefile)")
    args = parser.parse_args(argv)
    try:
        text = Path(args.config).read_text()
    except OSError as exc:
        print(f"deploy preflight: cannot read {args.config}: {exc}", file=sys.stderr)
        return 1
    if args.print_hef_path:
        with contextlib.suppress(yaml.YAMLError, AttributeError):
            print(((yaml.safe_load(text) or {}).get("accelerator") or {}).get("yolo_hef") or "")
        return 0
    report = check(text, expect_role=args.expect_role, hef_sha256=args.hef_sha256)
    if args.json:
        print(json.dumps({**asdict(report), "ok": report.ok}, indent=2))
    else:
        print(f"deploy preflight: role '{report.role}'{'' if report.role_declared else ' (implicit)'}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        for error in report.errors:
            print(f"  REFUSED: {error}")
        print("  OK to deploy" if report.ok else f"  {len(report.errors)} problem(s): not deploying")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
