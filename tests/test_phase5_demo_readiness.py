"""Phase 5 — guards on the things that decide whether a demo is honest.

Two failures motivate this file, both of which the suite could not previously
see:

  * The device reported `system: warning — throttled (0x50000)` continuously,
    for hours, while every live throttle bit was clear and every measurement was
    nominal. 0x50000 is bits 16 and 18: "under-voltage HAS occurred" and
    "throttling HAS occurred" — sticky flags that persist until the Pi is
    power-cycled. A warning that cannot turn off is not a warning.

  * The detection model binary is the one artifact whose contents decide whether
    a fall is seen. config.yaml now names its exact hash, and that claim needs a
    test or it becomes a comment that drifts.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

import pytest
import yaml

from core import device_health

REPO = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((REPO / "config.yaml").read_text())


# --------------------------------------------------------------- throttle bits

@pytest.mark.parametrize(
    ("raw", "live", "sticky"),
    [
        ("0x0", [], []),
        # The exact value this device reported for hours. NOTHING is happening.
        ("0x50000", [], ["under-voltage", "throttling"]),
        ("0x1", ["under-voltage"], []),
        ("0x4", ["throttled"], []),
        # Both at once: a live fault AND a history of one.
        ("0x50005", ["under-voltage", "throttled"], ["under-voltage", "throttling"]),
        # An unreadable register is not evidence of a fault.
        (None, [], []),
        ("not-hex", [], []),
    ],
)
def test_throttle_bits_separate_now_from_ever(raw, live, sticky):
    got_live, got_sticky = device_health._decode_throttled(raw)
    assert got_live == live
    assert got_sticky == sticky


def test_sticky_throttle_alone_does_not_make_the_device_unhealthy(monkeypatch):
    """A brown-out during setup must not mark the device faulty for its whole uptime."""
    monkeypatch.setattr(device_health, "_read_throttled", lambda: "0x50000")
    monkeypatch.setattr(device_health, "_read_temperature_c", lambda: 55.0)
    report = device_health.collect(module_name="driver_attention")

    assert report["system"]["state"] == device_health.OK, (
        "sticky-only throttle bits describe history, not a current fault"
    )
    # ...but it must still be VISIBLE. Not-a-fault is not the same as not-worth-saying.
    assert "occurred earlier" in report["system"]["detail"]
    assert report["system"]["metrics"]["throttle_since_boot"] == [
        "under-voltage",
        "throttling",
    ]
    assert report["system"]["metrics"]["throttle_now"] == []
    # The raw value stays, so the reading is traceable to the register.
    assert report["system"]["metrics"]["throttled"] == "0x50000"


def test_a_live_throttle_bit_does_make_the_device_unhealthy(monkeypatch):
    """The other direction: a real, current fault must still warn."""
    monkeypatch.setattr(device_health, "_read_throttled", lambda: "0x1")
    monkeypatch.setattr(device_health, "_read_temperature_c", lambda: 55.0)
    report = device_health.collect(module_name="driver_attention")
    assert report["system"]["state"] == device_health.WARNING
    assert "happening now" in report["system"]["detail"]


# ------------------------------------------------------------- the demo binary

def test_config_names_an_exact_detector_binary():
    """There must be ONE authoritative answer to 'what detector is this running?'."""
    accel = CONFIG["accelerator"]
    assert accel["yolo_hef"], "the shipped default must name a .hef, not be null"
    sha = accel.get("yolo_hef_sha256", "")
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
        "accelerator.yolo_hef_sha256 must be a full sha256 — it is the only "
        "machine-checkable statement of which binary produced the accuracy numbers"
    )


def test_setup_provisions_the_same_binary_the_config_names():
    """setup_pi.sh and config.yaml must not drift apart about the model."""
    setup = (REPO / "setup_pi.sh").read_text()
    sha = CONFIG["accelerator"]["yolo_hef_sha256"]
    assert sha in setup, (
        f"config.yaml pins {sha[:16]}... but setup_pi.sh does not provision that "
        "hash — a fresh install would run a different detector from the one the "
        "config claims"
    )
    assert 'MODEL_SHA256="' in setup and "sha256sum" in setup, (
        "setup_pi.sh must verify the model checksum, not merely download a URL"
    )


@pytest.mark.skipif(
    not (REPO / "config.yaml").exists(), reason="config missing"
)
def test_the_local_model_if_present_matches_the_pin():
    """On a machine that HAS the model, prove the file is the pinned one.

    Skipped where the model is absent (it is gitignored, ~18MB) rather than
    asserting a weaker thing — the real enforcement is setup_pi.sh, tested above.
    """
    hef = REPO / CONFIG["accelerator"]["yolo_hef"]
    if not hef.exists():
        pytest.skip(f"{hef} not present on this machine (gitignored)")
    got = hashlib.sha256(hef.read_bytes()).hexdigest()
    assert got == CONFIG["accelerator"]["yolo_hef_sha256"], (
        f"{hef} does not match the pinned hash — this device is NOT running the "
        "binary the config and the benchmarks describe"
    )


# ------------------------------------------- accelerator/module startup order

def test_accelerator_is_configured_before_any_module_is_constructed():
    """An ORDERING bug that reboots the device, so it gets an ordering test.

    core.accel.maybe_hailo_yolo reads the `accelerator:` block from module-level
    state that configure() installs. A module building its detector in __init__
    before that call sees no `yolo_hef`, falls through to the CPU path, and on a
    device (where ultralytics is deliberately absent) raises — main.py exits
    non-zero, systemd retries, and StartLimitAction=reboot power-cycles the Pi.

    Measured in Phase 5: occupancy alone put the device into a sustained reboot
    loop for exactly this reason.
    """
    source = (REPO / "main.py").read_text()
    configure_at = source.find("_accel.configure(")
    construct_at = source.find("module = module_cls(module_config)")
    assert configure_at != -1, "main.py must configure the accelerator explicitly"
    assert construct_at != -1, "could not find module construction in main.py"
    assert configure_at < construct_at, (
        "accel.configure() must run BEFORE module_cls(...) — otherwise any module "
        "that loads a detector in __init__ asks for one while yolo_hef is unset"
    )


def test_no_module_loads_a_detector_in_its_constructor():
    """Eager loading bypasses degraded mode entirely.

    A detector failure inside __init__ happens before DeviceRuntime exists, so
    the Phase 5 degraded path — boot, serve health, report monitoring NOT active
    — never runs. It becomes an un-degradable startup crash instead. Every module
    must defer, so that a missing backend is always reportable rather than fatal.
    """
    import ast

    offenders = []
    for path in sorted((REPO / "modules").glob("*.py")):
        tree = ast.parse(path.read_text())
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                if fn.name != "__init__":
                    continue
                for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                    name = getattr(call.func, "id", getattr(call.func, "attr", ""))
                    if name == "get_yolo_model":
                        offenders.append(f"{path.name}:{call.lineno} {cls.name}.__init__")
    assert not offenders, (
        "these modules load a detector in __init__, which runs before "
        f"DeviceRuntime and so cannot be degraded gracefully: {offenders}"
    )


# --------------------------------------------------- deploy licence + provenance

def test_deploy_ships_no_model_weights_of_either_format():
    """`.pt` was excluded in Phase 4; `.hef` is the format the device can RUN.

    Excluding only .pt is a licence control aimed at the one format the Pi cannot
    execute. docs/ACCELERATOR.md used to direct readers to download
    Ultralytics-derived .hef files into models/, which is gitignored — and rsync
    copies the working tree, not the index.

    It is also a determinism control: the device's model must come only from
    setup_pi.sh, which pins a URL and verifies sha256, never from whatever
    happened to be in a developer's tree.
    """
    makefile = (REPO / "Makefile").read_text()
    deploy = makefile[makefile.index("deploy:") :]
    deploy = deploy[: deploy.index("\n\n")] if "\n\n" in deploy else deploy
    for pattern in ("'*.pt'", "'*.hef'"):
        assert f"--exclude={pattern}" in deploy, (
            f"make deploy does not exclude {pattern} — a model can reach the "
            "device from a working tree instead of from checksum-pinned provisioning"
        )


def test_model_version_reads_every_weights_key_a_module_can_use():
    """A training sample stamped with a model that did not produce it is worse
    than an unlabelled one — you cannot tell an old model's mistakes from a new
    one's, which is the only reason this field exists.

    driver_attention (the flagship) declares `yolo_weights`, so a helper reading
    only `model_weights` silently stamped every one of its samples with the
    hardcoded default.
    """
    source = (REPO / "services" / "runtime.py").read_text()
    body = source[source.index("def _model_version") :]
    body = body[: body.index("def _capture_training_sample")]
    assert "_WEIGHTS_KEYS" in body, (
        "_model_version must consult every weights key a module can declare, "
        "the same tuple _verify_detection_backend uses — not model_weights alone"
    )
    assert '"model_weights", "yolo11n.pt"' not in body, (
        "_model_version still falls back to a hardcoded yolo11n.pt default"
    )


def test_the_startup_banner_can_name_the_real_model_on_the_hailo_path():
    """The banner is the line an operator greps to confirm DAMO-YOLO-on-NPU.

    driver_attention prints `getattr(model, "model_name", weights)`. `model_name`
    is an *ultralytics* attribute; until Phase 5 HailoYOLO did not define it, so
    on the Hailo path the getattr fell through to the configured `.pt` name and
    the banner announced "yolo11n.pt" while a DAMO-YOLO .hef did the work — the
    exact confidently-wrong log line that code was written to stop printing.
    """
    from core.accel import HailoYOLO

    assert isinstance(getattr(HailoYOLO, "model_name", None), property), (
        "HailoYOLO must expose model_name, or the banner silently reports the "
        "ultralytics weights name on a device that is not running ultralytics"
    )
    stub = types.SimpleNamespace(_hef_path="models/damoyolo_tinynasL20_T.hef")
    getter = HailoYOLO.model_name.fget
    assert getter is not None
    assert getter(stub) == "damoyolo_tinynasL20_T.hef"


def test_monitoring_active_flips_when_the_detector_dies_mid_run(monkeypatch):
    """Not just when it was missing at boot.

    `detection_degraded` is set once, during startup. On its own it would leave
    monitoring_active True for a device whose NPU stopped answering an hour ago —
    and this field is documented as the one a consumer should trust.
    """
    # Patch through device_health.accel, and patch the INTERFACE, for two
    # separate reasons this test learned the hard way:
    #
    #   * `accel._last_inference_at` does not exist (it is
    #     `_last_inference_monotonic`). With `raising=False` monkeypatch happily
    #     created a dead attribute and the test passed by luck in isolation.
    #   * `from core import accel` in a test does NOT always yield the same module
    #     object `core.device_health` holds — under the full suite `core.accel`
    #     exists twice in sys.modules, so patching one leaves the other untouched.
    #     Reaching through `device_health.accel` patches the one actually called.
    monkeypatch.setattr(
        device_health.accel,
        "inference_stats",
        lambda: {"count": 500, "age_s": 120.0, "live": False},
    )
    assert device_health.collect(module_name="driver_attention")["monitoring_active"] is False


def test_a_pose_only_module_is_not_reported_as_degraded(monkeypatch):
    """The other half: fall_detection consumes no YOLO and runs zero NPU
    inferences forever by design. Treating that as a dead detector would declare
    a working fall detector blind."""
    monkeypatch.setattr(
        device_health.accel,
        "inference_stats",
        lambda: {"count": 0, "age_s": None, "live": False},
    )
    assert device_health.collect(module_name="fall_detection")["monitoring_active"] is True
