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

from core.alerts import AlertLevel, AlertManager, ConsoleSink, SoundSink, UnavailableSink
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
    "erratic_hand_movement",
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
#: Exit status for an invalid config (sysexits.h EX_CONFIG); see run().
CONFIG_ERROR_EXIT = 78


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

# Deliberate re-exports. Nothing inside main.py calls the three names above
# besides load_presets, so an "unused import" autofix will delete them and break
# every caller that imports them from here -- which is exactly what happened
# during Phase 7 when `pre-commit run --all-files` ran ruff's safe autofix over
# the tree. __all__ makes the intent machine-readable so the next autofix keeps
# them.
__all__ = ["load_preset_aliases", "load_presets", "resolve_preset", "same_preset"]


def restored_preset(config_module: Optional[str], modules_cfg: dict,
                    presets: dict) -> Optional[tuple[str, dict, dict]]:
    """The preset the owner last chose, if it still applies: (name, preset, overrides).

    None when nothing was saved, when config.yaml's `module:` has changed since
    (a newer, deliberate choice — see core/device_state.py), or when the saved
    name no longer resolves to a preset and variant this build can run.
    """
    from core import device_state

    name = device_state.saved_preset(config_module)
    if not name:
        return None
    from core.presets import PresetError, preset_overrides

    # Variant overrides plus any owner settings the saved name carries
    # ("preset:variant@key=value"); a name this build cannot apply is not restored.
    try:
        name, preset, overrides = preset_overrides(name, presets)
    except PresetError:
        return None
    if preset.get("module") not in (modules_cfg or {}):
        return None
    return name, preset, overrides


def alert_types_for(module_name: str, preset: Optional[dict] = None) -> set:
    """The alertable event types for a module, with optional per-preset override."""
    if preset and preset.get("alert_types"):
        return set(preset["alert_types"])
    return ALERT_EVENT_TYPES_BY_MODULE.get(module_name, ALERT_EVENT_TYPES_PI)


def build_alert_manager(alert_cfg: dict, is_pi: bool,
                        sink_overrides: Optional[dict] = None) -> AlertManager:
    """Construct the AlertManager from an `alerts:`-shaped config block.

    Callers pick which block to pass: `config["alerts"]` for the pi profile,
    `config["mac_max"]["alerts"]` for mac_max -- the two profiles never share
    alert configuration (cooldowns, escalation types, etc).

    `sink_overrides` maps a configured sink name to a zero-argument factory used
    INSTEAD of the built-in construction (sim/'s simulated buzzer stands in for
    `gpio_buzzer`). None -- every production caller -- changes nothing.
    """
    sinks = []
    for sink_name in alert_cfg.get("sinks", ["console"]):
        if sink_overrides and sink_name in sink_overrides:
            sinks.append(sink_overrides[sink_name]())
        elif sink_name == "console":
            sinks.append(ConsoleSink())
        elif sink_name == "sound":
            sinks.append(SoundSink())
        elif sink_name == "gpio_buzzer":
            if is_pi:
                from core.alerts import GPIOBuzzerSink

                # Losing the buzzer must not cost detection. gpiozero/lgpio come
                # from setup_pi.sh, not requirements-device.txt, and a busy pin
                # or a permissions problem raises here too — unguarded, that
                # exited the process before the runtime existed, and systemd's
                # StartLimitAction=reboot turned it into a reboot loop.
                try:
                    sinks.append(GPIOBuzzerSink(int(alert_cfg.get("gpio_buzzer_pin", 17))))
                except Exception as exc:  # noqa: BLE001 - any failure means "no buzzer"
                    print(f"SensorDyme: 'gpio_buzzer' sink unavailable ({exc}); continuing without it")
                    # Kept, as unavailable: health must report the missing buzzer
                    # (a driver_safety device without one is FAILED, not protecting).
                    sinks.append(UnavailableSink("gpio_buzzer", "buzzer", True, f"could not be set up: {exc}"))
            else:
                print("SensorDyme: 'gpio_buzzer' sink skipped (not running on a Pi)")
                sinks.append(UnavailableSink("gpio_buzzer", "buzzer", True, "not running on a Raspberry Pi"))
        else:
            print(f"SensorDyme: unknown alert sink '{sink_name}', skipping")
            sinks.append(UnavailableSink(str(sink_name), "unknown", False, "unknown alert sink type"))

    from core.behavior.experimental import opted_in as experimental_opted_in

    fixed_levels = {}
    type_categories: dict[str, str] = {}
    for event_type in alert_cfg.get("loud_types", []):
        fixed_levels[event_type] = AlertLevel.CONTINUOUS
        type_categories[event_type] = "warning"
    for event_type in alert_cfg.get("soft_types", []):
        fixed_levels[event_type] = AlertLevel.SOFT
        type_categories[event_type] = "notice"
    # The most urgent, distinct pattern (sleep / unresponsive). A critical type
    # is always loud.
    for event_type in alert_cfg.get("critical_types", []):
        fixed_levels[event_type] = AlertLevel.CONTINUOUS
        type_categories[event_type] = "critical"

    return AlertManager(
        sinks=sinks,
        cooldown_s=float(alert_cfg.get("cooldown_s", 10.0)),
        escalation_window_s=float(alert_cfg.get("escalation_window_s", 60.0)),
        fixed_levels=fixed_levels,
        categories=alert_categories(alert_cfg),
        type_categories=type_categories,
        duplicate_window_s=float(alert_cfg.get("duplicate_window_s", 2.0)),
        # M8.4 owner decision A: experimental families sound only on an explicit
        # boolean opt-in (core/behavior/experimental.py), whatever the lists say.
        experimental_audible=experimental_opted_in(alert_cfg),
    )


def alert_categories(alert_cfg: dict) -> dict:
    """DEFAULT_CATEGORIES with any `alerts.categories:` overrides applied.

    Overrides are bounded by AlertCategory itself (known patterns, repeat leaves
    room for the pattern, max 120 repeats); config_schema reports violations as
    errors before this runs.
    """
    from dataclasses import replace

    from core.alerts import DEFAULT_CATEGORIES

    categories = dict(DEFAULT_CATEGORIES)
    for name, override in (alert_cfg.get("categories") or {}).items():
        if name in categories and isinstance(override, dict):
            categories[name] = replace(categories[name], **override)
    return categories


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


def load_device_config(path: Union[str, Path]):
    """config.yaml, validated -- or the last config that reached READY, with the fault.

    Returns (config, fault): fault is None when config.yaml itself is in use,
    else a core.config_lkg.ConfigFault the runtime reports in every health
    report. Exits 78 (EX_CONFIG) when neither can be used: sensordyme.service
    lists 78 in RestartPreventExitStatus, so a bad config stops the service with
    the reason in the journal instead of a restart or reboot loop over a file no
    restart can fix (audit D12).
    """
    from core import config_lkg
    from core.config_schema import validate
    from core.secret_store import config_permission_warnings

    config, problem = config_lkg.read(path)
    if config is None:
        # Before this was guarded, an unreadable or malformed file raised straight
        # out (exit 1), which Restart=always and StartLimitAction=reboot turned
        # into a reboot loop (audit D12).
        errors = [str(problem)]
        print(f"SensorDyme: config ERROR: {problem}")
    else:
        errors, warnings = validate(config)
        # A config holding the device key or dashboard token must not be readable by
        # other local users (docs/CREDENTIAL_STORAGE.md). Warned, not refused: a
        # refusal here would take a working device offline over a file mode.
        warnings = warnings + config_permission_warnings(path, config)
        for warning in warnings:
            print(f"SensorDyme: config warning: {warning}")
        torn = None if errors else config_lkg.torn_write(path)
        if torn:
            errors = [torn]
        for error in errors:
            print(f"SensorDyme: config ERROR: {error}")
    if not errors:
        return config, None

    # M8: a config.yaml that cannot be used no longer has to mean a dark device.
    # The last config that brought this device to READY is the same device (same
    # key, same token, same role); run on it and say so everywhere.
    fallback, fallback_path = config_lkg.load_last_good(path)
    if fallback is None:
        print(f"SensorDyme: invalid config ({len(errors)} error(s) above) and no usable "
              f"last-known-good copy ({config_lkg.last_good_path(path)}) — refusing to start")
        raise SystemExit(CONFIG_ERROR_EXIT)
    reason = "; ".join(errors)
    if len(reason) > 500:
        reason = reason[:497] + "..."
    print(f"SensorDyme: config FAULT: {path} cannot be used ({len(errors)} error(s) above). "
          f"Running on the last configuration that reached READY ({fallback_path}). "
          f"{path} was NOT changed: fix it and restart the service.")
    return fallback, config_lkg.ConfigFault(str(path), reason, str(fallback_path))


def run(argv: Optional[list[str]] = None) -> None:
    """Resolve profile/module/alert policy from config + CLI, then hand the wired
    pieces to services.runtime.DeviceRuntime, which owns the process lifetime
    (camera capture, inference, dashboard, uplink, provisioning, health, shutdown)."""
    build_runtime(argv).run()


def build_runtime(argv: Optional[list[str]] = None, *, runtime_cls=None, module_factory=None,
                  sink_overrides: Optional[dict] = None):
    """Everything run() does before DeviceRuntime.run(): the wired, unstarted runtime.

    The keyword arguments are seams for sim/ (the virtual device), all None in
    production, where this is exactly what run() always did:
      runtime_cls     a DeviceRuntime subclass to construct instead
      module_factory  (module_cls, module_config) -> module, instead of
                      module_cls(module_config); a failure still boots FAILED
      sink_overrides  see build_alert_manager
    """
    args = parse_args(argv)
    config, config_fault = load_device_config(args.config)

    profile = args.profile if args.profile is not None else config.get("profile", "pi")

    module_name = args.module if args.module is not None else config["module"]
    if args.module is not None:
        profile = "pi"  # --module runs through the standard module path

    # Presets: app/dashboard-selectable modes (standard module path only — the
    # mac_max profile keeps its own launch flow and is never hot-swapped).
    presets = load_presets() if profile != "mac_max" else {}

    # The mode the owner picked survives a restart. An explicit --module is a
    # developer asking for something specific, so it wins.
    restored = None
    if profile != "mac_max" and args.module is None:
        restored = restored_preset(config.get("module"), config.get("modules", {}), presets)
    if restored is not None:
        # A saved preset the device's role forbids (saved before the role was set,
        # or under another role) is not restored: the role's own module runs.
        from core.roles import preset_violation, role_for_config

        refusal = preset_violation(role_for_config(config), restored[0], presets)
        if refusal:
            print(f"SensorDyme: not restoring saved mode '{restored[0]}': {refusal}")
            restored = None
    if restored is not None:
        module_name = restored[1]["module"]

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
        overrides = restored[2] if restored is not None else {}
        module_config = {
            **config["modules"][module_name], **overrides, "recalibrate": args.recalibrate,
        }
        alert_cfg = config.get("alerts", {})
        alert_event_types = ALERT_EVENT_TYPES_BY_MODULE.get(module_name, ALERT_EVENT_TYPES_PI)

    if restored is not None:
        active_preset: Optional[str] = restored[0]
        alert_event_types = alert_types_for(module_name, restored[1])
        print(f"SensorDyme: restored the last selected mode '{active_preset}'")
    else:
        active_preset = next(
            (n for n, p in presets.items() if p.get("module") == module_name), None
        )
        if active_preset is not None:
            alert_event_types = alert_types_for(module_name, presets[active_preset])

    # Install the accelerator config BEFORE constructing the module. This is an
    # ordering requirement, not a tidiness one: core.accel.maybe_hailo_yolo reads
    # the `accelerator:` block from module-level state that configure() installs,
    # so any module that builds a detector in its __init__ (occupancy does) would
    # otherwise ask for a model while `yolo_hef` is still unset, get None, and
    # fall through to the CPU path.
    #
    # On a device that is the whole failure: ultralytics is deliberately absent
    # (AGPL-3.0), so the fall-through raises, main.py exits non-zero, systemd's
    # Restart=always retries, and StartLimitAction=reboot power-cycles the Pi.
    # Measured in Phase 5 — occupancy alone put this device into a sustained
    # reboot loop while the other three modules were fine, purely because they
    # happen to load their detectors lazily, after DeviceRuntime calls
    # configure(). DeviceRuntime still calls it; configure() is idempotent.
    from core import accel as _accel

    _accel.configure(config.get("accelerator", {}))

    try:
        module = module_cls(module_config) if module_factory is None else module_factory(module_cls, module_config)
    except Exception as exc:  # noqa: BLE001 - any failure: boot FAILED with the reason, never crash
        # Raising here was exit 1 -> Restart=always -> StartLimitAction=reboot:
        # a reboot loop over a missing model or a broken install (audit D12),
        # with health, the dashboard, the app link and BLE setup all dark.
        # The placeholder analyses nothing and the device reports FAILED, why,
        # and stays reachable so a person (or a preset switch) can fix it.
        from modules.base import UnavailableModule

        print(f"SensorDyme: module '{module_name}' could not be built ({exc}); "
              "starting FAILED so the device stays reachable")
        module = UnavailableModule(module_name, str(exc))
    print(f"SensorDyme: profile '{profile}', running module '{module_name}' (available: {available_modules()})")

    debug = args.debug
    display = True if debug else (args.display if args.display is not None else bool(config.get("display", False)))

    from core.camera import detect_pi
    from services.runtime import DeviceRuntime

    runtime = (runtime_cls or DeviceRuntime)(
        config=config,
        profile=profile,
        module=module,
        module_name=module_name,
        alert_manager=build_alert_manager(alert_cfg, detect_pi(), sink_overrides),
        alert_cfg=alert_cfg,
        alert_event_types=alert_event_types,
        alert_types_for=alert_types_for,
        presets=presets,
        active_preset=active_preset,
        display=display,
        debug=debug,
        window_name=WINDOW_NAME,
        module_config=module_config,
    )
    # Where this config came from: the runtime records it as last-known-good once
    # it reaches READY, and reports a fault while it runs from the saved copy.
    runtime.config_path = str(args.config)
    runtime.config_fault = config_fault
    return runtime


if __name__ == "__main__":
    run()
