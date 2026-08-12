"""SensorDyme entrypoint: loads config.yaml, starts the camera, runs the selected module.

Usage:
    python main.py                      # uses config.yaml, display per config
    python main.py --no-display         # force headless (e.g. on a Pi service)
    python main.py --debug              # full tuning overlay (EAR/yaw/pitch/PERCLOS/boxes/state)
    python main.py --recalibrate        # redo the driver_attention EAR calibration on this run
    python main.py --config other.yaml  # use an alternate config file

Profiles (config.yaml's `profile:` key):
    profile: pi      -- the shipping product (default). Unaffected by anything below.
    profile: mac_max -- additive, Mac-only, GPU-backed high-accuracy mode. Requires
                        Apple Silicon + torch MPS; refuses to start otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import yaml

from core.alerts import AlertLevel, AlertManager, ConsoleSink, SoundSink
from modules import available_modules, get_module_class

# Event types that reach the AlertManager at all, per profile. Anything else (e.g.
# "occupancy_change", "phone_detected") is still written to the CSV log but never alerts.
#
# These are DELIBERATELY kept as two separate sets, not one union: "eating_drinking" (and
# a few others) are log-only for the pi profile but alert-worthy for mac_max -- the same
# event-type string means different things to each profile's alert policy, so a shared
# set would silently change the pi profile's behavior the moment mac_max's types were
# added to it. Keeping them separate is what makes mac_max provably additive.
ALERT_EVENT_TYPES_PI = {
    "drowsiness",
    "distraction",
    "fall",
    "fatigue_warning",
    "frequent_glancing",
    "eyes_down",
    "eyes_up",
    "phone_use",
    "attention_lost",
    "elevated_risk",
    "high_risk",
}
# openpilot's own alert names (modules/driver_attention_openpilot.py): the orange
# "prompt" tier chimes (soft), the red terminal tier is continuous (loud); the green
# "pre" tier is display-only, exactly like the comma device.
ALERT_EVENT_TYPES_OPENPILOT = {
    "promptDriverDistracted",
    "driverDistracted",
    "promptDriverUnresponsive",
    "driverUnresponsive",
    "tooDistracted",
    "distractionDetected",  # additive instant chime (not stock openpilot; see the module)
    "phoneUseDetected",  # additive: YOLO phone + hand overlap, sustained (loud)
    "eyesClosedDetected",  # additive: eyes closed > eyes_closed_seconds (loud)
}
# The vendored skinnemoen module: every alert is audible (an addition — the original
# only beeped on Drowsy and calibration-complete; DISTRACTED was silent text).
ALERT_EVENT_TYPES_SKINNEMOEN = {
    "drowsiness",
    "skinnemoen_distracted",
    "skinnemoen_phone_use",
    "skinnemoen_calibration_complete",
}
# DashSentinel's statuses, each audible (DROWSY loud; the rest chime).
ALERT_EVENT_TYPES_DASHSENTINEL = {
    "dashsentinel_drowsy",
    "dashsentinel_warning",
    "dashsentinel_no_face",
    "dashsentinel_vision_degraded",
}
ALERT_EVENT_TYPES_BY_MODULE = {
    "driver_attention_openpilot": ALERT_EVENT_TYPES_OPENPILOT,
    "driver_attention_skinnemoen": ALERT_EVENT_TYPES_SKINNEMOEN,
    "driver_attention_dashsentinel": ALERT_EVENT_TYPES_DASHSENTINEL,
}
ALERT_EVENT_TYPES_MAC_MAX = {
    "drowsiness",
    "eyes_down",
    "microsleep_warning",
    "fatigue",
    "fatigue_critical",
    "eyes_off_road",
    "excessive_glancing",
    "phone_in_hand",
    "phone_use_while_distracted",
    "eating_drinking",
    # VLM-confirmed composite alerts (mac_max only; "vlm_judgment" itself is log-only)
    "vlm_confirmed_drowsiness",
    "vlm_confirmed_distraction",
    "vlm_confirmed_phone_use",
    "vlm_confirmed_eating",
}
WINDOW_NAME = "SensorDyme"


def load_config(path: Union[str, Path]) -> dict:
    """Load and parse a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# Preset loading and alias resolution live in core/presets.py so the backend can
# use them too — it runs on machines that have never had mediapipe installed,
# and importing main.py there is impossible. Re-exported here because callers
# and tests already reach for them by these names.
from core.presets import (  # noqa: E402
    load_preset_aliases,
    load_presets,
    resolve_preset,
    same_preset,
)


def alert_types_for(module_name: str, preset: Optional[dict] = None) -> set:
    """The alertable event types for a module, with optional per-preset override."""
    if preset and preset.get("alert_types"):
        return set(preset["alert_types"])
    return ALERT_EVENT_TYPES_BY_MODULE.get(module_name, ALERT_EVENT_TYPES_PI)


def build_alert_manager(alert_cfg: dict, is_pi: bool) -> AlertManager:
    """Construct the AlertManager from an `alerts:`-shaped config block.

    Callers pick which block to pass: `config["alerts"]` for the pi profile,
    `config["mac_max"]["alerts"]` for mac_max -- the two profiles never share
    alert configuration (cooldowns, escalation types, etc).
    """
    sinks = []
    for sink_name in alert_cfg.get("sinks", ["console"]):
        if sink_name == "console":
            sinks.append(ConsoleSink())
        elif sink_name == "sound":
            sinks.append(SoundSink())
        elif sink_name == "gpio_buzzer":
            if is_pi:
                from core.alerts import GPIOBuzzerSink

                sinks.append(GPIOBuzzerSink(int(alert_cfg.get("gpio_buzzer_pin", 17))))
            else:
                print("SensorDyme: 'gpio_buzzer' sink skipped (not running on a Pi)")
        else:
            print(f"SensorDyme: unknown alert sink '{sink_name}', skipping")

    fixed_levels = {}
    for event_type in alert_cfg.get("loud_types", []):
        fixed_levels[event_type] = AlertLevel.CONTINUOUS
    for event_type in alert_cfg.get("soft_types", []):
        fixed_levels[event_type] = AlertLevel.SOFT

    return AlertManager(
        sinks=sinks,
        cooldown_s=float(alert_cfg.get("cooldown_s", 10.0)),
        escalation_window_s=float(alert_cfg.get("escalation_window_s", 60.0)),
        fixed_levels=fixed_levels,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="SensorDyme edge vision runner")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--display", dest="display", action="store_true", default=None, help="Force the live debug window on")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Force the live debug window off")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Show the full tuning overlay (EAR/yaw/pitch/PERCLOS/YOLO boxes/state) and force the display on",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        default=False,
        help="Redo driver_attention's per-user EAR calibration instead of loading data/calibration.json",
    )
    parser.add_argument(
        "--profile",
        choices=["pi", "mac_max"],
        default=None,
        help="Override config.yaml's `profile:` for this run (pi = shipping product, mac_max = Mac-only high-accuracy mode)",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Override config.yaml's `module:` for this run (e.g. driver_attention_openpilot); uses the standard (pi) profile path",
    )
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None) -> None:
    """Resolve profile/module/alert policy from config + CLI, then hand the wired
    pieces to services.runtime.DeviceRuntime, which owns the process lifetime
    (camera capture, inference, dashboard, uplink, provisioning, health, shutdown)."""
    args = parse_args(argv)
    config = load_config(args.config)

    from core.config_schema import validate

    errors, warnings = validate(config)
    for warning in warnings:
        print(f"SensorDyme: config warning: {warning}")
    if errors:
        for error in errors:
            print(f"SensorDyme: config ERROR: {error}")
        raise SystemExit(f"invalid config ({len(errors)} error(s) above) — refusing to start")

    profile = args.profile if args.profile is not None else config.get("profile", "pi")

    module_name = args.module if args.module is not None else config["module"]
    if args.module is not None:
        profile = "pi"  # --module runs through the standard module path

    if profile == "mac_max":
        from modules.driver_attention_max import require_mps

        require_mps()  # refuses to start rather than silently falling back to CPU

        module_cls = get_module_class(f"{module_name}_max")
        mac_max_config = config["mac_max"]
        module_config = {
            **mac_max_config[module_name],
            "device": mac_max_config.get("device", "mps"),
            "recalibrate": args.recalibrate,
        }
        alert_cfg = mac_max_config["alerts"]
        alert_event_types = ALERT_EVENT_TYPES_MAC_MAX
    else:
        module_cls = get_module_class(module_name)
        module_config = {**config["modules"][module_name], "recalibrate": args.recalibrate}
        alert_cfg = config.get("alerts", {})
        alert_event_types = ALERT_EVENT_TYPES_BY_MODULE.get(module_name, ALERT_EVENT_TYPES_PI)

    # Presets: app/dashboard-selectable modes (standard module path only — the
    # mac_max profile keeps its own launch flow and is never hot-swapped).
    presets = load_presets() if profile != "mac_max" else {}
    active_preset = next((n for n, p in presets.items() if p.get("module") == module_name), None)
    if active_preset is not None:
        alert_event_types = alert_types_for(module_name, presets[active_preset])

    module = module_cls(module_config)
    print(f"SensorDyme: profile '{profile}', running module '{module_name}' (available: {available_modules()})")

    debug = args.debug
    display = True if debug else (args.display if args.display is not None else bool(config.get("display", False)))

    from core.camera import detect_pi
    from services.runtime import DeviceRuntime

    DeviceRuntime(
        config=config,
        profile=profile,
        module=module,
        module_name=module_name,
        alert_manager=build_alert_manager(alert_cfg, detect_pi()),
        alert_cfg=alert_cfg,
        alert_event_types=alert_event_types,
        alert_types_for=alert_types_for,
        presets=presets,
        active_preset=active_preset,
        display=display,
        debug=debug,
        window_name=WINDOW_NAME,
    ).run()


if __name__ == "__main__":
    run()
