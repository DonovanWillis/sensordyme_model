"""Phase 6 — what the device does when things break.

Every failure here is one that can be simulated honestly in software. The ones
that cannot (a real fall, a real infant, a 48-hour soak, a sagging power supply)
are NOT simulated and NOT asserted about; they are recorded as unverified in
docs/PHASE6_DEMO_READINESS.md instead.

The single property under test throughout: **the device must never claim to be
monitoring when it is not.** A safety product that shows green while blind is
worse than one that is plainly off, because only the second one gets fixed.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import device_health
from services.runtime import DeviceRuntime

REPO = Path(__file__).resolve().parent.parent


def _load_preflight():
    """Import scripts/demo_preflight.py as a module.

    It must be REGISTERED in sys.modules before exec: @dataclass resolves
    `sys.modules[cls.__module__]` while creating the class, and that lookup
    returns None for a module that was exec'd but never registered.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "sensordyme_demo_preflight", REPO / "scripts" / "demo_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _no_intervals(caller: str = "default") -> dict:
    """Stub for FrameBus.interval_stats, which takes a caller key.

    Deliberately a named function, not `lambda: {}` and not `dict`. A lambda
    gets rewritten to `dict` by ruff's unnecessary-lambda autofix, and `dict`
    accepts the caller string as a mapping -> ValueError deep inside
    device_health. That is exactly how this broke once.
    """
    return {}


def _runtime(module_name: str = "driver_attention", cfg: dict | None = None) -> Any:
    runtime = types.SimpleNamespace(
        config={"modules": {module_name: cfg if cfg is not None else {"yolo_weights": "yolo11n.pt"}}},
        module_name=module_name,
        _WEIGHTS_KEYS=DeviceRuntime._WEIGHTS_KEYS,
        _detection_degraded=None,
        _model_identity=None,
        # No separate boot config: the running module's config is the one in
        # config["modules"], as on the pi profile.
        module=None,
        _boot_module=None,
        _boot_module_config=None,
    )
    # The real helpers, bound, so the method under test reads config exactly
    # as it does in DeviceRuntime rather than through a stub.
    runtime.__dict__["_active_module_config"] = types.MethodType(DeviceRuntime._active_module_config, runtime)
    runtime.__dict__["_module_identity_name"] = types.MethodType(DeviceRuntime._module_identity_name, runtime)
    return runtime


# ----------------------------------------------------------------- accelerator

@pytest.mark.parametrize(
    "reason",
    [
        "no object-detection backend available for 'yolo11n.pt'",   # nothing installed
        "accelerator.yolo_hef points to a missing file: models/x.hef",  # missing HEF
        "Hailo init failed (device busy) — running on CPU",         # init failure
    ],
)
def test_every_accelerator_failure_degrades_rather_than_lies(monkeypatch, reason):
    """Missing device, missing HEF, invalid HEF, init failure — one policy."""
    from core import detectors

    def explode(weights, device=None):
        raise RuntimeError(reason)

    monkeypatch.setattr(detectors, "require_backend", explode)
    runtime = _runtime()
    DeviceRuntime._verify_detection_backend(runtime)  # must not raise

    assert runtime._detection_degraded is not None
    report = device_health.collect(
        module_name="driver_attention", detection_degraded=runtime._detection_degraded
    )
    assert report["monitoring_active"] is False
    assert report["inference"]["state"] == device_health.FAILED
    assert reason.split("(")[0].strip()[:30] in report["inference"]["detail"]


def test_a_detector_that_dies_mid_run_also_flips_monitoring(monkeypatch):
    """Not only a boot-time failure. Inferences happened, then stopped."""
    monkeypatch.setattr(
        device_health.accel, "inference_stats",
        lambda: {"count": 900, "age_s": 300.0, "live": False},
    )
    assert device_health.collect(module_name="driver_attention")["monitoring_active"] is False


def test_recovery_returns_the_device_to_active_monitoring(monkeypatch):
    """Degradation must not latch. A fixed accelerator means a working device."""
    monkeypatch.setattr(
        device_health.accel, "inference_stats",
        lambda: {"count": 900, "age_s": 0.2, "live": True},
    )
    report = device_health.collect(module_name="driver_attention")
    assert report["monitoring_active"] is True
    assert report["accelerator"]["state"] == device_health.OK


# ---------------------------------------------------------------------- camera

def test_a_missing_camera_is_reported_not_assumed_healthy():
    """No bus at all — the state before capture ever starts."""
    report = device_health.collect(bus=None, module_name="driver_attention")
    assert report["camera"]["state"] == device_health.UNKNOWN
    assert "not started" in report["camera"]["detail"]


def test_a_camera_that_never_produced_a_frame_is_FAILED():
    bus = types.SimpleNamespace(
        age_s=lambda: float("inf"), sequence=0, interval_stats=_no_intervals
    )
    report = device_health.collect(bus=bus, module_name="driver_attention")
    assert report["camera"]["state"] == device_health.FAILED
    assert "no frames" in report["camera"]["detail"]


def test_stale_frames_are_a_failure_even_while_capture_claims_to_run():
    """The exact shape of a wedged camera: the service is up, frames are not.

    `capture.is_running` says the thread is alive, which is why it cannot be the
    signal. Frame age is the only evidence that frames are actually arriving.
    """
    bus = types.SimpleNamespace(
        age_s=lambda: 45.0, sequence=1200, interval_stats=_no_intervals
    )
    capture = types.SimpleNamespace(is_running=True)
    report = device_health.collect(bus=bus, capture=capture, module_name="driver_attention")
    assert report["camera"]["state"] == device_health.FAILED
    assert "no new frame" in report["camera"]["detail"]


def test_a_stopped_capture_service_is_reported_as_failed():
    bus = types.SimpleNamespace(age_s=lambda: 0.1, sequence=10, interval_stats=_no_intervals)
    capture = types.SimpleNamespace(is_running=False)
    report = device_health.collect(bus=bus, capture=capture, module_name="driver_attention")
    assert report["camera"]["state"] == device_health.FAILED


# --------------------------------------------------------------- configuration

def test_a_stale_ai_override_is_detectable_without_running_the_device():
    """The Phase 5 failure, as a unit test.

    A rule set saved thirteen days earlier silently ran occupancy on a device
    configured for driver_attention. The detector for that condition must not
    require a running service to notice.
    """
    preflight = _load_preflight()

    saved = REPO / "data" / "monitoring.json"
    original = saved.read_text() if saved.exists() else None
    try:
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps({
            "instruction": "Watching for a bottle present in the scene.",
            "rules": [{"target": "bottle"}],
            "overrides_mode": True,
            "updated_at": 1_786_934_483.0,
        }))
        message = preflight._override_state()
        assert message, "an active override must be detected"
        assert "overrides config.yaml" in message
        assert "bottle" in message, "the operator must be told WHAT it is watching for"

        # And the inverse: an inactive override is not reported as one.
        saved.write_text(json.dumps({"rules": [{"target": "bottle"}], "overrides_mode": False}))
        assert preflight._override_state() == ""

        # A corrupt file must be reported, never silently treated as "no override".
        saved.write_text("{not json")
        assert "unreadable" in preflight._override_state()
    finally:
        if original is None:
            saved.unlink(missing_ok=True)
        else:
            saved.write_text(original)


def test_shipped_thresholds_are_expressible_by_the_shipped_detector():
    """A threshold the hardware cannot represent is not a setting, it is a lie."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    driver = config["modules"]["driver_attention"]
    assert driver["yolo_confidence_threshold"] >= 0.20
    assert driver["person_confidence_threshold"] > driver["yolo_confidence_threshold"]


def test_every_demo_mode_is_actually_configured():
    """A mode the demo script cycles must exist in config, or the cycle fails
    at the device rather than here."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    cycle = (REPO / "scripts" / "demo_cycle.sh").read_text()
    modes = cycle.split('MODES:-"')[1].split('"')[0].split()
    assert len(modes) == 4
    for mode in modes:
        assert mode in config["modules"], f"demo_cycle.sh cycles '{mode}' but config.yaml has no such module"


# -------------------------------------------------------------- the preflight

def test_the_preflight_cannot_report_ready_while_degraded():
    """The one thing this tool must never do."""
    preflight = _load_preflight()

    report = preflight.Report()
    report.add("monitoring active", False, "monitoring_active=False")
    assert report.ready is False

    # A WARN must not block a demo; only a FAIL may.
    ok = preflight.Report()
    ok.add("frame rate", False, "12 fps", warn_only=True)
    assert ok.ready is True


def test_preflight_details_are_collapsed_to_one_line():
    """/api/health can carry a nine-line RuntimeError as a component detail.
    Printed raw it buries the finding under remediation advice."""
    preflight = _load_preflight()

    check = preflight.Check("x", preflight.FAIL, "line one\n\nline two\n   line three")
    assert "\n" not in check.detail


# ------------------------------------------------------------------- deployment

def test_deploy_ships_no_data_bearing_files():
    """`--exclude='data'` covers the data/ DIRECTORY and nothing else.

    A dry run of the real rsync was carrying `cloud/sensordyme.db` (the
    backend's own database, 639KB) plus `e.csv`, `events.csv`, `t.csv` and
    `telemetry.csv` from the repo root onto every device. Shipping a developer's
    event history and a backend DB to a customer is a privacy problem before it
    is a tidiness one, and nothing about it was visible in `make deploy`'s
    output.
    """
    makefile = (REPO / "Makefile").read_text()
    deploy = makefile[makefile.index("deploy:") :]
    deploy = deploy[: deploy.index("\n\n")] if "\n\n" in deploy else deploy
    for pattern in ("'*.db'", "'*.sqlite3'", "'*.csv'"):
        assert f"--exclude={pattern}" in deploy, (
            f"make deploy does not exclude {pattern} — developer data would ship "
            "to customer hardware"
        )


def test_no_shipped_comment_promises_a_cpu_fallback():
    """The device has no CPU fallback. A comment saying otherwise tells an
    installer their failed Hailo install was harmless, when it means the device
    cannot detect anything at all."""
    for name in ("setup_pi.sh", "config.yaml"):
        text = (REPO / name).read_text()
        for line_no, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if "falls back to cpu automatically" in lowered or "still runs on cpu" in lowered:
                # A historical reference is fine; a current claim is not.
                assert any(
                    marker in lowered
                    for marker in ("used to", "was true", "before phase", "previously")
                ), f"{name}:{line_no} claims a CPU fallback that no longer exists: {line.strip()!r}"


def test_every_module_swap_re_verifies_the_detection_backend():
    """Verified once at boot is not enough.

    A device can boot on Elderly Monitoring — fall_detection is MediaPipe-Pose-
    only, declares no weights key, and so degrades nothing — and then be switched
    in the app to Security or Baby, whose modules load YOLO lazily inside
    process(). On a device with no usable model that raises every frame while
    `monitoring_active` stays true and the banner says "active".

    Both swap sites must re-check, and the check must be idempotent so switching
    OUT of a broken state clears the reason too.
    """
    import ast

    tree = ast.parse((REPO / "services" / "runtime.py").read_text())
    # The bodies; the public names wrap them in the swap-in-progress marker.
    swaps = {"_apply_monitoring_mode_now", "_switch_preset_now"}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in swaps):
            continue
        calls = {
            getattr(c.func, "attr", "")
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        }
        assert "_verify_detection_backend" in calls, (
            f"{node.name} constructs a new module but never re-verifies the "
            "detection backend — a mode switch can go blind with health still green"
        )
        swaps.discard(node.name)
    assert not swaps, f"could not find swap method(s): {swaps}"


def test_the_backend_check_clears_a_stale_degraded_reason(monkeypatch):
    """Switching to a module that needs no detector must un-degrade the device."""
    from core import detectors

    monkeypatch.setattr(detectors, "require_backend", lambda w, d=None: "hailo")
    runtime = _runtime("fall_detection", cfg={})       # declares no weights key
    runtime._detection_degraded = "left over from a previous module"
    DeviceRuntime._verify_detection_backend(runtime)
    assert runtime._detection_degraded is None, (
        "a stale degraded reason survived a swap to a module that needs no detector"
    )


def test_a_live_switch_into_a_pose_only_module_is_not_falsely_degraded(monkeypatch):
    """The case the Phase 6 test missed, found by Phase 7's runtime-switch test.

    The Phase 6 guard covered a FRESH BOOT into fall_detection, where the NPU
    counter is 0 — that lands on the "running on CPU" branch and reads fine. But
    the demo SWITCHES modes live, and after a switch the counter still holds the
    previous module's inferences. Those age past the liveness window and look
    exactly like "inferences happened, then stopped" — the signature of a dying
    detector. The device reported `monitoring_active: false` while fall detection
    worked perfectly: the false-degraded mirror of the bug the field exists for.

    Observed on hardware: driver_safety -> baby_monitoring -> elderly_monitoring
    reported monitoring_active False with inference `ok` and 29.7 fps.
    """
    monkeypatch.setattr(
        device_health.accel, "inference_stats",
        lambda: {"count": 2748, "age_s": 45.0, "live": False},   # the previous module's
    )
    assert device_health.collect(
        module_name="fall_detection", detector_required=False
    )["monitoring_active"] is True, (
        "a Pose-only module was marked degraded because an EARLIER module's "
        "inferences went stale"
    )
    # The same stale-counter state MUST still degrade a module that needs the NPU.
    assert device_health.collect(
        module_name="driver_attention", detector_required=True
    )["monitoring_active"] is False


def test_health_and_the_boot_check_agree_on_which_modules_need_a_detector():
    """Two places decide 'does this module use the detector'. They must not drift.

    `_verify_detection_backend` uses _WEIGHTS_KEYS to decide whether to verify a
    backend; `_module_uses_a_detector` uses it to decide whether an idle NPU is a
    failure. If those ever disagreed, a module would either be verified and then
    reported falsely degraded, or skipped and then reported falsely healthy.
    """
    import ast

    source = (REPO / "services" / "runtime.py").read_text()
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_module_uses_a_detector"
    )
    assert "_WEIGHTS_KEYS" in ast.dump(fn), (
        "_module_uses_a_detector must use the same _WEIGHTS_KEYS signal as "
        "_verify_detection_backend, or health and the boot check can disagree"
    )


def test_every_configured_module_is_classified_consistently():
    """Behavioural companion to the AST check above: the classification must
    match what each module actually does."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    keys = DeviceRuntime._WEIGHTS_KEYS
    needs = {
        name: any((cfg or {}).get(k) for k in keys)
        for name, cfg in config["modules"].items()
        if isinstance(cfg, dict)
    }
    assert needs["fall_detection"] is False, "fall_detection is MediaPipe-Pose-only"
    assert needs["occupancy"] is True
    assert needs["baby_monitoring"] is True
    assert needs["driver_attention"] is True


# ------------------------------------------------- Phase 7: alive-but-useless

def test_app_drawn_zones_survive_the_swap_their_own_rules_trigger():
    """The Phase 6 failure class, on a different axis.

    The app sends zones and rules in ONE poll: zones are applied first, then the
    rules — and the rules are what trigger the swap into AI-custom mode. Taking
    the new module's config.yaml zones during that swap replaced the polygon the
    user had just drawn (with `{}` in the shipped config), so the rule
    referencing it could never fire. The uplink had already acked
    `zones_version`, so the app never re-sent and nothing recovered.

    Verified before the fix: 400 frames of a person squarely inside the drawn
    polygon produced zero alerts, with the video correct, people being detected,
    `monitoring_active` True, and the app still listing the rule as active.
    """
    import ast

    source = (REPO / "services" / "runtime.py").read_text()
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_monitoring_mode_now"
    )
    # The zone reset must be conditional, and conditional on the MODE — not on
    # tracking how the zones arrived, because in AI-custom they always come from
    # the app by definition.
    resets = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "set_zones"
    ]
    assert resets, "_apply_monitoring_mode no longer resets zones at all"
    guarded = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "set_zones"
            for c in ast.walk(node)
        )
        and "AI_CUSTOM" in ast.dump(node.test)
    ]
    assert guarded, (
        "the zone reset in _apply_monitoring_mode is not guarded on AI_CUSTOM — "
        "a swap into custom monitoring will discard app-delivered zones and the "
        "rule that needed them can never fire"
    )


def test_leaving_custom_monitoring_restores_the_zones_in_force():
    """The other direction: dropping the custom rules re-points the rule engine
    at the zones in force for the running module.

    Those are the app's zones when it has sent any — they are the device's
    zones, drawn in the app's editor for a crib or a keep-out area, and must
    keep protecting the preset the owner returns to — otherwise the module's
    own config.yaml zones. (tests/test_runtime_coordinator_fixes.py covers the
    behaviour end to end.)
    """
    source = (REPO / "services" / "runtime.py").read_text()
    body = source[source.index("def _exit_custom_monitoring") :]
    body = body[: body.index("\n    def ", 1)]
    assert "set_zones(self._effective_zones(self.module_name))" in body, (
        "_exit_custom_monitoring must reset the rule engine to the effective "
        "zones when it drops the custom rules"
    )
