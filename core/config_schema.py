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
}


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    else:
        errors.append("`accelerator:` must be a mapping")

    network = config.get("network", {})
    if isinstance(network, dict):
        prov = network.get("provisioning", {})
        if isinstance(prov, dict):
            method = prov.get("method", "ble")
            _check(errors, method in ("ble", "hotspot", "off"),
                   f"`network.provisioning.method:` must be ble|hotspot|off, got {method!r}")
        else:
            errors.append("`network.provisioning:` must be a mapping")
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
        if cloud.get("enabled") is True and not cloud.get("device_key"):
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
    if isinstance(alerts, dict) and "cooldown_s" in alerts:
        _check(errors, _is_num(alerts["cooldown_s"]) and alerts["cooldown_s"] >= 0,
               f"`alerts.cooldown_s:` must be a non-negative number, got {alerts['cooldown_s']!r}")

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
    if profile != "mac_max" and isinstance(module_name, str) and isinstance(modules_cfg, dict):
        _check(errors, module_name in modules_cfg,
               f"`module: {module_name}` has no `modules.{module_name}:` config section")

    return errors, warnings
