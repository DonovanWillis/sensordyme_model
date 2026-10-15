"""Phase 4 — the boot-time detection-backend check.

This runs on every start, before the camera opens, and it can REFUSE to boot.
The existing runtime smoke tests all use synthetic modules ("counting",
"flaky") that declare no detector weights, so they only ever exercise the
early-return path. The branch that actually loads a model — the one that can
stop a device from starting — had no coverage at all. That is the gap here.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from services.runtime import DeviceRuntime


def _runtime(module_name: str, module_cfg: dict) -> Any:
    """The smallest object `_verify_detection_backend` actually reads from."""
    runtime = types.SimpleNamespace(
        config={"modules": {module_name: module_cfg}},
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


def test_a_module_with_no_detector_does_not_load_one(monkeypatch):
    """fall_detection is MediaPipe-Pose-only. Booting it must not touch a model.

    Not a micro-optimisation: loading a model here would cost boot time and NPU
    memory to verify a dependency the module does not have, and would make a
    Pose-only module fail to start because an object detector was missing.
    """
    from core import detectors

    called = []
    monkeypatch.setattr(detectors, "require_backend", lambda *a, **k: called.append(a))
    DeviceRuntime._verify_detection_backend(_runtime("fall_detection", {}))
    assert called == [], "a module declaring no weights must not load a detector"


@pytest.mark.parametrize(
    ("key", "weights"),
    [
        ("yolo_weights", "yolo11n.pt"),        # driver_attention
        ("model_weights", "yolo11n.pt"),       # occupancy, baby_monitoring
        ("phone_weights", "yolo11n.pt"),       # driver_attention_openpilot/skinnemoen
        ("detector_weights", "yolo11x.pt"),    # driver_attention_max
    ],
)
def test_every_weights_key_in_use_is_recognised(monkeypatch, key, weights):
    """Each key a shipped module actually uses must resolve to a backend check.

    A key missing from _WEIGHTS_KEYS fails OPEN — the module would boot claiming
    no detector, then fail lazily inside the inference loop, which is precisely
    the failure mode this check exists to remove.
    """
    from core import detectors

    seen = []
    monkeypatch.setattr(detectors, "require_backend", lambda w, d=None: seen.append(w) or "hailo")
    DeviceRuntime._verify_detection_backend(_runtime("some_module", {key: weights}))
    assert seen == [weights], f"{key} was not recognised as naming detector weights"


def test_a_missing_backend_degrades_loudly_instead_of_crash_looping(monkeypatch):
    """POLICY CHANGED IN PHASE 5, deliberately. Read this before "fixing" it.

    Phase 4 made a missing backend raise out of _build. That was better than the
    restart loop it replaced, but it was still wrong: raising here takes down the
    health endpoint, the dashboard and BLE provisioning along with detection, so
    the owner of a device whose HAT came loose gets a unit that says nothing at
    all. systemd restarts it, it fails again, forever.

    Phase 5 degrades instead — boot, serve health, and state plainly that
    monitoring is NOT active. The assertions below are STRONGER than the old
    `pytest.raises`, not weaker: they require the reason to be recorded so it can
    reach the app, which a raise never did.
    """
    from core import detectors

    def explode(weights, device=None):
        raise RuntimeError(f"no object-detection backend available for {weights!r}")

    monkeypatch.setattr(detectors, "require_backend", explode)
    rt = _runtime("driver_attention", {"yolo_weights": "yolo11n.pt"})
    DeviceRuntime._verify_detection_backend(rt)  # must NOT raise

    assert rt._detection_degraded is not None, (
        "a missing backend must be RECORDED, not swallowed — otherwise the device "
        "boots looking healthy while unable to detect anything"
    )
    assert "no object-detection backend" in rt._detection_degraded, (
        "the recorded reason must be the real one, so the app can show it"
    )


def test_a_working_backend_leaves_the_device_undegraded(monkeypatch):
    """The other half: degradation must not be the default state."""
    from core import detectors

    monkeypatch.setattr(detectors, "require_backend", lambda w, d=None: "hailo")
    rt = _runtime("driver_attention", {"yolo_weights": "yolo11n.pt"})
    DeviceRuntime._verify_detection_backend(rt)
    assert rt._detection_degraded is None


def test_degraded_health_never_claims_monitoring_is_active():
    """The claim that matters to a person looking at the app."""
    from core import device_health

    report = device_health.collect(
        module_name="driver_attention", detection_degraded="Hailo init failed"
    )
    assert report["monitoring_active"] is False
    assert report["inference"]["state"] == device_health.FAILED
    assert "Hailo init failed" in report["inference"]["detail"]
    # And it must not be quietly true elsewhere in the same report.
    assert device_health.collect(module_name="driver_attention")["monitoring_active"] is True


def test_the_device_pin_is_passed_through(monkeypatch):
    """mac_max pins device=mps and never consults the accelerator; dropping the
    pin here would check a backend the module will not use."""
    from core import detectors

    seen = {}
    monkeypatch.setattr(
        detectors,
        "require_backend",
        lambda w, d=None: seen.update(weights=w, device=d) or "ultralytics",
    )
    DeviceRuntime._verify_detection_backend(
        _runtime("driver_attention_max", {"detector_weights": "yolo11x.pt", "device": "mps"})
    )
    assert seen == {"weights": "yolo11x.pt", "device": "mps"}


def test_weights_keys_cover_every_shipped_module():
    """Guard against a new module inventing a fifth key name."""
    from pathlib import Path

    import yaml

    modules = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.yaml").read_text()
    )["modules"]
    suspicious = {
        f"{name}.{key}"
        for name, cfg in modules.items()
        if isinstance(cfg, dict)
        for key in cfg
        if key.endswith("_weights") and key not in DeviceRuntime._WEIGHTS_KEYS
    }
    assert not suspicious, (
        f"these config keys name model weights but are not in _WEIGHTS_KEYS, so the "
        f"boot check would silently skip them: {sorted(suspicious)}"
    )


@pytest.mark.parametrize("module_name", ["driver_attention_openpilot", "driver_attention_skinnemoen"])
def test_the_phone_detector_device_pin_is_passed_through(monkeypatch, module_name):
    """The comparison modules load their phone detector with `phone_device`, not
    `device`. Reading only `device` verified (yolo11n.pt, None) at boot while the
    module asked for (yolo11n.pt, "mps") — a different cache entry and possibly a
    different backend — so health could report a detector the module never got."""
    from core import detectors

    seen = {}
    monkeypatch.setattr(
        detectors,
        "require_backend",
        lambda w, d=None: seen.update(weights=w, device=d) or "ultralytics",
    )
    DeviceRuntime._verify_detection_backend(
        _runtime(module_name, {"phone_weights": "yolo11n.pt", "phone_device": "mps"})
    )
    assert seen == {"weights": "yolo11n.pt", "device": "mps"}


def test_mac_max_verifies_and_labels_the_detector_its_module_actually_runs(monkeypatch):
    """mac_max hands the runtime module_name="driver_attention", whose
    config["modules"] entry is the PI detector (yolo11n.pt, Hailo/CPU). The
    module itself was built from config["mac_max"] (yolo11x.pt on MPS). Reading
    config["modules"] verified — and stamped every event with — a model that
    never ran. The runtime now uses the config the boot module was built from."""
    from core import detectors

    seen = {}
    monkeypatch.setattr(
        detectors, "require_backend",
        lambda w, d=None: seen.update(weights=w, device=d) or "ultralytics",
    )
    monkeypatch.setattr(
        detectors, "model_identity",
        lambda w, d=None: {"name": w, "sha256": None, "backend": "ultralytics"},
    )
    boot_module = types.SimpleNamespace(name="driver_attention_max")
    runtime = _runtime("driver_attention", {"yolo_weights": "yolo11n.pt"})
    runtime.module = boot_module
    runtime._boot_module = boot_module
    runtime._boot_module_config = {"detector_weights": "yolo11x.pt", "device": "mps"}

    DeviceRuntime._verify_detection_backend(runtime)

    assert seen == {"weights": "yolo11x.pt", "device": "mps"}
    assert runtime._model_identity == {
        "module": "driver_attention_max",
        "detector": {"name": "yolo11x.pt", "sha256": None, "backend": "ultralytics"},
    }


def test_events_never_see_a_half_built_or_previous_model_identity(monkeypatch):
    """Verification can take seconds (a CPU load, hashing weights). During it
    the identity is unknown (None) — never the previous module's model, and
    never this module with detector=None — then assigned once, complete."""
    from core import detectors

    runtime = _runtime("driver_attention", {"yolo_weights": "yolo11n.pt"})
    runtime._model_identity = {"module": "fall_detection", "detector": None}  # the previous module's
    during = {}

    def loading(w, d=None):
        during["identity"] = runtime._model_identity
        return "ultralytics"

    monkeypatch.setattr(detectors, "require_backend", loading)
    monkeypatch.setattr(detectors, "model_identity", lambda w, d=None: {"name": w, "sha256": None, "backend": "u"})
    DeviceRuntime._verify_detection_backend(runtime)
    assert during["identity"] is None
    assert runtime._model_identity["detector"] == {"name": "yolo11n.pt", "sha256": None, "backend": "u"}
