"""config.yaml validation: catch bad configs at startup with readable errors.

Deliberately hand-rolled (no pydantic/jsonschema dependency): the known keys are
type- and range-checked, unknown top-level sections produce warnings (typos like
`dashbord:` used to fail silently), and modules keep validating their own
thresholds — this covers the runtime-level shape.

`validate(config)` returns (errors, warnings). Errors abort startup in
services/runtime (a wrong-typed port must not take the camera down mid-boot);
warnings are logged and the device runs.
"""

from __future__ import annotations

_KNOWN_TOP_LEVEL = {
    "module", "profile", "display", "camera", "dashboard", "cloud", "alerts",
    "logging", "modules", "mac_max", "accelerator", "network", "health", "hardware",
    # `training:` drives the CaptureStore (services/runtime.py builds it from this
    # block). It was missing here, so every device printed "unknown top-level section
    # 'training:' (typo?)" on every boot for a section that is real and is read --
    # which is how an operator learns to ignore this validator's output entirely.
    "training",
    # Deployment role and its safety floor (core/roles.py, M0.1).
    "role",
    # The shared perception engine (core/perception.py, M7): max_age_s, detector_weights.
    "perception",
    # development (default) | production: production refuses insecure fallbacks,
    # e.g. unauthenticated Bluetooth setup (services/ble_pop.py, M6.5).
    "deployment",
    # M12 sd-local-v1: per-phone local access (core/local_auth.py, docs/LOCAL_ACCESS.md).
    "local_access",
}


#: The class channels every product decoder already reads (COCO-80, core/accel.py COCO_NAMES).
_COCO_CHANNELS = 80


def parse_yolo_extra_classes(value) -> dict[int, str]:
    """`accelerator.yolo_extra_classes` -> {class-head channel: class name}; {} when absent.

    M10, opt-in and experiment-only. A raw-head .hef whose class head carries a
    class beyond COCO-80 (the EXP-08 device-layout export puts `hand` in channel
    80, research/m84/device_export.py) is decoded by both HailoYOLO decoders as
    if channel 80 were padding, so that class is never reported. This names
    which extra channels ARE classes. Absent or null keeps today's decode bit
    for bit.

    Refused (ValueError, the message names the key):
      * a channel below 80: COCO ids are fixed, remapping one would relabel a
        class every module and preset already depends on;
      * a non-integer channel (YAML `80:` and JSON `"80"` are both accepted);
      * an empty, non-string or COCO-colliding name, or one name twice: a
        consumer selecting by name must find exactly one channel.
    Whether the loaded .hef has the channel is checked when it is loaded
    (core/accel.py), because only the HEF knows its class-head width.
    """
    if value is None:
        return {}
    key = "`accelerator.yolo_extra_classes:`"
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping of class-head channel -> class name like {{80: hand}}, "
                         f"got {value!r}")
    from core.accel import COCO_NAMES  # imported here: config validation must not need it otherwise

    coco_names = set(COCO_NAMES.values())
    out: dict[int, str] = {}
    for raw_channel, name in value.items():
        if isinstance(raw_channel, int) and not isinstance(raw_channel, bool):
            channel = raw_channel
        elif isinstance(raw_channel, str) and raw_channel.strip().isdigit():
            channel = int(raw_channel.strip())
        else:
            raise ValueError(f"{key} channel {raw_channel!r} is not an integer")
        if channel < _COCO_CHANNELS:
            raise ValueError(f"{key} channel {channel} is a COCO class ({COCO_NAMES.get(channel)!r}); only "
                             f"channels >= {_COCO_CHANNELS} can be mapped, COCO ids are never relabelled")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{key} channel {channel} needs a non-empty class name, got {name!r}")
        label = name.strip()
        if label in coco_names:
            raise ValueError(f"{key} name {label!r} (channel {channel}) collides with a COCO class name")
        if label in out.values():
            raise ValueError(f"{key} name {label!r} is mapped to more than one channel")
        if channel in out:
            raise ValueError(f"{key} channel {channel} is mapped twice")
        out[channel] = label
    return dict(sorted(out.items()))


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


#: Sink names main.build_alert_manager knows how to build.
_KNOWN_SINKS = ("console", "sound", "gpio_buzzer")


def _validate_alerts(alerts: dict, where: str, errors: list[str], warnings: list[str]) -> None:
    """An `alerts:`-shaped block (the pi profile's, or mac_max's own).

    The list keys matter most. YAML reads `loud_types: fall` as the STRING
    "fall", and every consumer iterates it -- character by character -- so
    nothing was loud any more (no push for any event) and `sinks: gpio_buzzer`
    built no sink at all, with no error and no warning.
    """
    if "cooldown_s" in alerts:
        _check(errors, _is_num(alerts["cooldown_s"]) and alerts["cooldown_s"] >= 0,
               f"`{where}.cooldown_s:` must be a non-negative number, got {alerts['cooldown_s']!r}")
    if "escalation_window_s" in alerts:
        value = alerts["escalation_window_s"]
        _check(errors, _is_num(value) and value >= 0,
               f"`{where}.escalation_window_s:` must be a non-negative number, got {value!r}")
    if "duplicate_window_s" in alerts:
        value = alerts["duplicate_window_s"]
        _check(errors, _is_num(value) and 0 <= value <= 60,
               f"`{where}.duplicate_window_s:` must be a number in [0, 60], got {value!r}")
    categories = alerts.get("categories")
    if categories is not None:
        _validate_categories(categories, where, errors)
    relapse = alerts.get("relapse_escalation")
    if relapse is not None:
        if not isinstance(relapse, dict):
            errors.append(f"`{where}.relapse_escalation:` must be a mapping, got {relapse!r}")
        else:
            enabled = relapse.get("enabled", False)
            _check(errors, isinstance(enabled, bool),
                   f"`{where}.relapse_escalation.enabled:` must be true or false, got {enabled!r}")
            types = relapse.get("types", [])
            _check(errors, isinstance(types, list) and all(isinstance(t, str) and t for t in types),
                   f"`{where}.relapse_escalation.types:` must be a list of names like [a, b], got {types!r}")
            recovery = relapse.get("recovery_s", 60)
            _check(errors, _is_num(recovery) and 5 <= recovery <= 600,
                   f"`{where}.relapse_escalation.recovery_s:` must be a number in [5, 600], got {recovery!r}")
    for key in ("sinks", "loud_types", "soft_types", "critical_types"):
        if key not in alerts:
            continue
        value = alerts[key]
        if not (isinstance(value, list) and all(isinstance(item, str) and item for item in value)):
            errors.append(f"`{where}.{key}:` must be a list of names like [a, b], got {value!r}")
    sinks = alerts.get("sinks")
    if isinstance(sinks, list):
        for sink in sinks:
            if isinstance(sink, str) and sink not in _KNOWN_SINKS:
                warnings.append(f"`{where}.sinks:` unknown sink {sink!r} will be skipped "
                                f"(known: {', '.join(_KNOWN_SINKS)})")
    loud, soft = alerts.get("loud_types"), alerts.get("soft_types")
    if isinstance(loud, list) and isinstance(soft, list):
        both = sorted({t for t in loud if isinstance(t, str)} & {t for t in soft if isinstance(t, str)})
        if both:
            warnings.append(
                f"`{where}`: {both} listed in both loud_types and soft_types -- the buzzer "
                "plays them soft while the event is tagged loud"
            )
    if "gpio_buzzer_pin" in alerts:
        pin = alerts["gpio_buzzer_pin"]
        _check(errors, isinstance(pin, int) and not isinstance(pin, bool) and 0 <= pin <= 27,
               f"`{where}.gpio_buzzer_pin:` must be a BCM GPIO number 0-27, got {pin!r}")
    if "evidence_max_files" in alerts:
        value = alerts["evidence_max_files"]
        _check(errors, isinstance(value, int) and not isinstance(value, bool) and value >= 0,
               f"`{where}.evidence_max_files:` must be a non-negative integer, got {value!r}")


def _validate_categories(categories, where: str, errors: list[str]) -> None:
    """`alerts.categories:` overrides, bounded exactly as AlertCategory bounds them."""
    from dataclasses import replace

    from core.alerts import DEFAULT_CATEGORIES

    if not isinstance(categories, dict):
        errors.append(f"`{where}.categories:` must be a mapping of category -> settings")
        return
    allowed = {"initial_pattern", "escalated_pattern", "repeat_s", "escalate_after_s", "max_repeats"}
    for name, override in categories.items():
        if name not in DEFAULT_CATEGORIES:
            errors.append(f"`{where}.categories.{name}:` unknown category (known: {sorted(DEFAULT_CATEGORIES)})")
            continue
        if not isinstance(override, dict) or set(override) - allowed:
            errors.append(f"`{where}.categories.{name}:` may only set {sorted(allowed)}")
            continue
        try:
            replace(DEFAULT_CATEGORIES[name], **override)
        except (TypeError, ValueError) as exc:
            errors.append(f"`{where}.categories.{name}:` {exc}")


#: `network.direct_mode:` keys (core/direct_mode.DirectModeConfig).
_DIRECT_MODE_KEYS = {
    "enabled", "interface", "security", "channel", "ap_timeout_s", "psk_path", "state_path",
    "fallback_enabled", "fallback_after_s", "return_check_s",
}


def _validate_direct_mode(direct, config: dict, errors: list[str], warnings: list[str]) -> None:
    """`network.direct_mode:` (docs/DIRECT_MODE.md). Off unless `enabled: true`.

    The radio security has no open value, and a passphrase in config.yaml is
    refused outright: it lives only in the 0600 file behind the device label.
    """
    where = "network.direct_mode"
    if not isinstance(direct, dict):
        errors.append(f"`{where}:` must be a mapping, got {direct!r}")
        return
    for key in ("psk", "password", "passphrase"):
        if key in direct:
            errors.append(f"`{where}.{key}:` is refused: the direct-mode passphrase is never set in config.yaml; "
                          "it is generated per device into `psk_path` (scripts/direct_mode_psk.py)")
    for key in sorted(set(direct) - _DIRECT_MODE_KEYS - {"psk", "password", "passphrase"}):
        warnings.append(f"`{where}.{key}:` unknown key (typo?) is ignored")
    for key in ("enabled", "fallback_enabled"):
        value = direct.get(key, False)
        _check(errors, isinstance(value, bool), f"`{where}.{key}:` must be true or false, got {value!r}")
    security = direct.get("security", "wpa2")
    _check(errors, security in ("wpa2", "wpa3"),
           f"`{where}.security:` must be wpa2 or wpa3 (there is no open access point), got {security!r}")
    interface = direct.get("interface", "wlan0")
    _check(errors, isinstance(interface, str) and 0 < len(interface) <= 15
           and all(c.isalnum() or c in "_.-" for c in interface),
           f"`{where}.interface:` must be a network interface name like wlan0, got {interface!r}")
    channel = direct.get("channel", 6)
    _check(errors, isinstance(channel, int) and not isinstance(channel, bool) and 1 <= channel <= 11,
           f"`{where}.channel:` must be a 2.4 GHz channel 1-11, got {channel!r}")
    for key, low, high, default in (("ap_timeout_s", 5, 120, 30), ("fallback_after_s", 60, 86400, 600),
                                    ("return_check_s", 30, 86400, 300)):
        value = direct.get(key, default)
        _check(errors, _is_num(value) and low <= value <= high,
               f"`{where}.{key}:` must be a number in [{low}, {high}], got {value!r}")
    for key in ("psk_path", "state_path"):
        if key in direct:
            _check(errors, isinstance(direct[key], str) and direct[key] != "",
                   f"`{where}.{key}:` must be a path string, got {direct[key]!r}")
    if direct.get("enabled") is not True:
        return
    network = config.get("network") or {}
    prov = network.get("provisioning") if isinstance(network, dict) else None
    method = prov.get("method", "ble") if isinstance(prov, dict) else "ble"
    if method != "ble":
        warnings.append(f"{where} is enabled with network.provisioning.method: {method}: Bluetooth setup is "
                        "the recovery path for a broken access point; without it only a power cycle recovers")
    dashboard = config.get("dashboard")
    if not (isinstance(dashboard, dict) and dashboard.get("token")):
        warnings.append(f"{where} is enabled but dashboard.token is not set: a phone on the access point "
                        "gets 401 on every request (the dashboard fails closed)")


def validate(config: dict) -> tuple[list[str], list[str]]:
    """Validate the top-level runtime config. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(config, dict):
        return ["config.yaml did not parse to a mapping"], []

    for key in config:
        if key not in _KNOWN_TOP_LEVEL:
            warnings.append(f"unknown top-level section '{key}:' (typo?)")

    _check(errors, isinstance(config.get("module"), str) and config.get("module"),
           "`module:` must be a non-empty string")
    profile = config.get("profile", "pi")
    _check(errors, profile in ("pi", "mac_max"), f"`profile:` must be 'pi' or 'mac_max', got {profile!r}")
    deployment = config.get("deployment", "development")
    _check(errors, deployment in ("development", "production"),
           f"`deployment:` must be development|production, got {deployment!r}")

    camera = config.get("camera", {})
    if isinstance(camera, dict):
        for key in ("width", "height"):
            value = camera.get(key, 640)
            _check(errors, isinstance(value, int) and 16 <= value <= 8192,
                   f"`camera.{key}:` must be an integer pixel size, got {value!r}")
        fps = camera.get("fps", 30)
        _check(errors, _is_num(fps) and 0 < fps <= 240, f"`camera.fps:` must be in (0, 240], got {fps!r}")
    else:
        errors.append("`camera:` must be a mapping")

    local_access = config.get("local_access", {})
    if isinstance(local_access, dict):
        enabled = local_access.get("enabled", True)
        _check(errors, isinstance(enabled, bool),
               f"`local_access.enabled:` must be true or false, got {enabled!r}")
        days = local_access.get("idle_expiry_days", 180)
        _check(errors, _is_num(days) and 0 <= days <= 3650,
               f"`local_access.idle_expiry_days:` must be 0-3650 (0 = never), got {days!r}")
        events_max = local_access.get("events_max", 2000)
        _check(errors, isinstance(events_max, int) and 10 <= events_max <= 100_000,
               f"`local_access.events_max:` must be an integer 10-100000, got {events_max!r}")
    elif local_access is not None:
        errors.append("`local_access:` must be a mapping")

    dashboard = config.get("dashboard", {})
    if isinstance(dashboard, dict):
        port = dashboard.get("port", 8899)
        _check(errors, isinstance(port, int) and 0 <= port <= 65535,
               f"`dashboard.port:` must be a valid TCP port, got {port!r}")
        quality = dashboard.get("jpeg_quality", 60)
        _check(errors, isinstance(quality, int) and 1 <= quality <= 100,
               f"`dashboard.jpeg_quality:` must be 1-100, got {quality!r}")
        stream_fps = dashboard.get("stream_fps", 12)
        _check(errors, _is_num(stream_fps) and 0 < stream_fps <= 60,
               f"`dashboard.stream_fps:` must be in (0, 60], got {stream_fps!r}")
        host = dashboard.get("host", "127.0.0.1")
        _check(errors, isinstance(host, str) and host != "",
               f"`dashboard.host:` must be an IPv4 address or hostname, got {host!r}")
        # M10 (docs/DIAGNOSTICS.md): developer diagnostics, OFF by default. A bool and
        # nothing else: "yes" or 1 must not switch on a view of what the camera sees.
        diagnostics = dashboard.get("developer_diagnostics", False)
        _check(errors, isinstance(diagnostics, bool),
               f"`dashboard.developer_diagnostics:` must be true or false, got {diagnostics!r}")
        if diagnostics is True and not dashboard.get("token"):
            warnings.append(
                "dashboard.developer_diagnostics is true but dashboard.token is not set: "
                "/api/diagnostics refuses every request (403) until a token is configured"
            )
        if dashboard.get("token") and str(dashboard.get("host")) not in ("127.0.0.1", "localhost"):
            warnings.append(
                "dashboard.token is set with a non-localhost host: make sure the app "
                "has the same token configured, or the live feed will show 401/spinner"
            )
    else:
        errors.append("`dashboard:` must be a mapping")

    accel_cfg = config.get("accelerator", {})
    if isinstance(accel_cfg, dict):
        hef = accel_cfg.get("yolo_hef")
        _check(errors, hef is None or isinstance(hef, str),
               f"`accelerator.yolo_hef:` must be a path string or null, got {hef!r}")
        decoder = accel_cfg.get("yolo_decoder", "legacy")
        _check(errors, decoder in ("legacy", "reference"),
               f"`accelerator.yolo_decoder:` must be legacy|reference, got {decoder!r}")
        try:
            parse_yolo_extra_classes(accel_cfg.get("yolo_extra_classes"))
        except ValueError as exc:
            errors.append(str(exc))
    else:
        errors.append("`accelerator:` must be a mapping")

    network = config.get("network", {})
    if isinstance(network, dict):
        prov = network.get("provisioning", {})
        if isinstance(prov, dict):
            method = prov.get("method", "ble")
            _check(errors, method in ("ble", "hotspot", "off"),
                   f"`network.provisioning.method:` must be ble|hotspot|off, got {method!r}")
            auth = prov.get("auth", "legacy")
            _check(errors, auth in ("legacy", "pop"),
                   f"`network.provisioning.auth:` must be legacy|pop, got {auth!r}")
            if auth == "legacy" and config.get("deployment") == "production" and method == "ble":
                errors.append("`network.provisioning.auth: legacy` (unauthenticated Bluetooth setup) is refused "
                              "with `deployment: production`; set `auth: pop` (services/ble_pop.py)")
        else:
            errors.append("`network.provisioning:` must be a mapping")
        if "direct_mode" in network:
            _validate_direct_mode(network["direct_mode"], config, errors, warnings)
    else:
        errors.append("`network:` must be a mapping")

    health = config.get("health", {})
    if isinstance(health, dict):
        interval = health.get("check_interval_s", 2.0)
        _check(errors, _is_num(interval) and interval > 0,
               f"`health.check_interval_s:` must be a positive number, got {interval!r}")
    else:
        errors.append("`health:` must be a mapping")

    # Previously entirely unvalidated (found in a repo-wide audit) despite
    # holding the security-relevant TLS settings: a typo'd or wrong-typed key
    # here didn't error OR warn at boot, it just silently never connected --
    # exactly the kind of failure that's confusing to diagnose right before a
    # launch, when "did the config even load" shouldn't be in question.
    cloud = config.get("cloud", {})
    if isinstance(cloud, dict):
        if "enabled" in cloud:
            _check(errors, isinstance(cloud["enabled"], bool),
                   f"`cloud.enabled:` must be true or false, got {cloud['enabled']!r}")
        if cloud.get("base_url"):
            _check(errors, isinstance(cloud["base_url"], str),
                   f"`cloud.base_url:` must be a string, got {cloud['base_url']!r}")
        if "device_key" in cloud:
            _check(errors, isinstance(cloud["device_key"], str),
                   f"`cloud.device_key:` must be a string, got {cloud['device_key']!r}")
        if "ca_cert_path" in cloud:
            _check(errors, isinstance(cloud["ca_cert_path"], str),
                   f"`cloud.ca_cert_path:` must be a string, got {cloud['ca_cert_path']!r}")
        skip_verify = cloud.get("tls_insecure_skip_verify", False)
        _check(errors, isinstance(skip_verify, bool),
               f"`cloud.tls_insecure_skip_verify:` must be true or false, got {skip_verify!r}")
        if "queue_max" in cloud:
            queue_max = cloud["queue_max"]
            # Above the backend's MAX_INGEST_EVENTS (1000) every flush after an
            # outage is refused; 0 silently discards every event.
            _check(errors, isinstance(queue_max, int) and not isinstance(queue_max, bool)
                   and 1 <= queue_max <= 1000,
                   f"`cloud.queue_max:` must be an integer in 1-1000, got {queue_max!r}")
        # sd-auth-v1 (core/command_auth.py) and the 0600 key file (core/secret_store.py).
        if "command_auth" in cloud:
            _check(errors, cloud["command_auth"] in ("off", "permissive", "required"),
                   f"`cloud.command_auth:` must be off, permissive or required, got {cloud['command_auth']!r}")
        for key in ("previous_device_key", "device_key_file"):
            if cloud.get(key) is not None:
                _check(errors, isinstance(cloud[key], str), f"`cloud.{key}:` must be a string")
        revoked = cloud.get("revoked_key_ids")
        if revoked is not None:
            _check(errors, isinstance(revoked, list) and all(
                isinstance(k, str) and len(k) == 16 and all(c in "0123456789abcdef" for c in k) for k in revoked),
                "`cloud.revoked_key_ids:` must be a list of 16-hex-digit key ids")
        if cloud.get("device_key") and cloud.get("device_key_file"):
            errors.append("`cloud.device_key` and `cloud.device_key_file` are both set: keep only the file "
                          "(docs/CREDENTIAL_STORAGE.md)")
        if cloud.get("enabled") is True and not cloud.get("device_key") and not cloud.get("device_key_file"):
            warnings.append(
                "cloud.enabled is true but cloud.device_key is empty: the uplink will not "
                "start and this device will never reach the backend"
            )
        if "poll_interval_s" in cloud:
            poll = cloud["poll_interval_s"]
            _check(errors, _is_num(poll) and poll > 0,
                   f"`cloud.poll_interval_s:` must be a positive number, got {poll!r}")
        if cloud.get("enabled") and not str(cloud.get("base_url", "")).startswith(("http://", "https://")):
            errors.append(
                f"`cloud.base_url:` must start with http:// or https:// when cloud.enabled is true, "
                f"got {cloud.get('base_url')!r}"
            )
        if skip_verify and str(cloud.get("base_url", "")).startswith("https://"):
            warnings.append(
                "cloud.tls_insecure_skip_verify is true with an https:// base_url: certificate "
                "verification is OFF -- any attacker on the network path can intercept every event; "
                "only ever use this for a self-signed dev backend, never in the field"
            )
    else:
        errors.append("`cloud:` must be a mapping")

    # Sections the runtime reads with `.get(...)` on the assumption they are
    # mappings. `logging:` with nothing under it parses as null, which passed
    # validation and then crashed boot with an AttributeError.
    for section in ("logging", "alerts", "modules", "training", "hardware", "mac_max"):
        if section in config and not isinstance(config[section], dict):
            errors.append(f"`{section}:` must be a mapping, got {config[section]!r}")

    alerts = config.get("alerts")
    if isinstance(alerts, dict):
        _validate_alerts(alerts, "alerts", errors, warnings)
    mac_max = config.get("mac_max")
    if isinstance(mac_max, dict) and isinstance(mac_max.get("alerts"), dict):
        _validate_alerts(mac_max["alerts"], "mac_max.alerts", errors, warnings)

    if isinstance(health, dict):
        for key in ("reboot_max_total",):
            if key in health:
                value = health[key]
                _check(errors, isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                       f"`health.{key}:` must be a non-negative integer, got {value!r}")

    module_name = config.get("module")
    modules_cfg = config.get("modules", {})
    if isinstance(modules_cfg, dict):
        for name, section in modules_cfg.items():
            if not isinstance(section, dict):
                errors.append(f"`modules.{name}:` must be a mapping, got {section!r}")
        driver = modules_cfg.get("driver_attention")
        if isinstance(driver, dict):
            # Experimental behaviour reasoning (core/behavior/, off by default):
            # a typo'd or wrong-typed key there must fail loudly, not run defaults.
            from core.behavior.camera_model import validate_intrinsics
            from core.behavior.config import validate_behavior

            errors.extend(validate_behavior(driver))
            errors.extend(validate_intrinsics(driver.get("camera_intrinsics")))
            if driver.get("camera_intrinsics") is not None and driver.get("camera_hfov_deg") is not None:
                errors.append("`modules.driver_attention:` set camera_hfov_deg OR camera_intrinsics, not both "
                              "(calibrated intrinsics already fix the field of view)")
    if profile != "mac_max" and isinstance(module_name, str) and isinstance(modules_cfg, dict):
        _check(errors, module_name in modules_cfg,
               f"`module: {module_name}` has no `modules.{module_name}:` config section")

    # Deployment role: the safety floor this config must not weaken.
    from core.roles import ROLES, config_violations

    errors.extend(config_violations(config))
    role_name = config.get("role")
    if (
        isinstance(role_name, str) and role_name in ROLES and ROLES[role_name].has_floor
        and config.get("profile", "pi") == "mac_max"
    ):
        warnings.append(
            f"role '{role_name}' is not enforced on the mac_max developer profile; "
            "it never reports protection"
        )

    return errors, warnings
