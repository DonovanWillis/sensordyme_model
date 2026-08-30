"""/api/health must report what is COMPUTING, not what the config asked for.

`accel.status()` answers a configuration question — "is the configured intent
being met" — and returns "Hailo in use for object detection (<hef>)" whenever a
HAT is seated, the runtime loads, and a `.hef` path is set. None of that requires
a single frame to have reached the NPU.

That is the silent-fallback hole: on a Pi with a working HAT, a caller pinning
`device=`, a per-frame inference failure swallowed by a broad handler, or a model
loaded before `configure()` all leave the configuration intact while the work
happens on CPU — and the device's own /api/health would still say "Hailo in use".

`core/device_health.py::_accelerator` already grounded the APP-facing payload in
completed inferences. `accel.live_status()` applies the same evidence to the
DEVICE-facing endpoint, which had only the config answer. These tests pin the
three states an operator needs to tell apart.

Verified against real hardware on 2026-08-29 (Pi 5 + Hailo-8L, HailoRT 4.20.0):
SensorDyme's own code path drove 30 inferences through `HailoYOLO`, taking
`inference_stats()` from count=0/live=False to count=30/live=True.
"""

from __future__ import annotations

import pytest

from core import accel


@pytest.fixture(autouse=True)
def _reset_accel_state(monkeypatch):
    """Isolate the module-level counters; they are process-global by design."""
    monkeypatch.setattr(accel, "_inference_count", 0, raising=False)
    monkeypatch.setattr(accel, "_last_inference_monotonic", None, raising=False)
    monkeypatch.setattr(accel, "_hailo_failed_reason", None, raising=False)
    yield


def _hat(present=True, runtime=True, detail="Hailo device at /dev/hailo0"):
    """`usable` is a derived property (present and runtime_available), not a field."""
    return accel.AcceleratorInfo(
        present=present,
        runtime_available=runtime,
        kind="hailo" if present else "none",
        detail=detail,
    )


def test_live_inference_reports_the_npu_and_says_so_unambiguously(monkeypatch):
    """The good case: work has actually landed on the accelerator."""
    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": "/m/yolov11n.hef"})
    monkeypatch.setattr(accel, "_accel_info", _hat())
    accel.record_inference()

    healthy, detail = accel.live_status()

    assert healthy is True
    assert "Hailo NPU in use" in detail
    assert "1 inferences" in detail


def test_configured_but_never_executed_is_reported_as_cpu(monkeypatch):
    """THE REGRESSION. Seated HAT, good runtime, valid .hef path, zero inferences.

    `status()` calls this healthy and says "Hailo in use". It is the exact state a
    silent CPU fallback produces, and reporting it as Hailo is how a device ships
    claiming NPU acceleration it never had.
    """
    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": "/m/yolov11n.hef"})
    monkeypatch.setattr(accel, "_accel_info", _hat())

    config_healthy, config_detail = accel.status()
    assert config_healthy is True and "Hailo in use" in config_detail, (
        "precondition: status() reports the configured intent as met"
    )

    healthy, detail = accel.live_status()

    assert healthy is False, "a configured-but-never-used accelerator read as healthy"
    assert "running on CPU" in detail
    assert "no inference has executed" in detail


def test_a_stalled_accelerator_is_not_reported_as_in_use(monkeypatch):
    """Inferences happened, then stopped. Distinct from both 'in use' and 'never ran'."""
    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": "/m/yolov11n.hef"})
    monkeypatch.setattr(accel, "_accel_info", _hat())
    accel.record_inference()
    # Push the last inference outside inference_stats()' 10s liveness window.
    monkeypatch.setattr(
        accel, "_last_inference_monotonic", accel.time.monotonic() - 45.0, raising=False
    )

    healthy, detail = accel.live_status()

    assert healthy is False
    assert "idle" in detail
    assert "Hailo NPU in use" not in detail


def test_cpu_only_device_is_healthy_and_says_cpu(monkeypatch):
    """No HAT at all is a supported shipping mode — healthy, but never 'Hailo'."""
    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": None})
    monkeypatch.setattr(accel, "_accel_info", _hat(present=False, runtime=False,
                                                   detail="no Hailo device (running on CPU)"))

    healthy, detail = accel.live_status()

    assert healthy is True, "CPU-only is a supported mode, not a failure"
    assert "running on CPU" in detail
    assert "Hailo NPU in use" not in detail


def test_an_init_failure_surfaces_its_reason(monkeypatch):
    """A HAT that could not be used must report WHY, not a generic CPU message."""
    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": "/m/missing.hef"})
    monkeypatch.setattr(accel, "_accel_info", _hat())
    monkeypatch.setattr(
        accel, "_hailo_failed_reason",
        "accelerator.yolo_hef points to a missing file: /m/missing.hef", raising=False,
    )

    healthy, detail = accel.live_status()

    assert healthy is False
    assert "missing file" in detail


def test_the_three_operator_states_are_mutually_distinguishable(monkeypatch):
    """The whole point: 'hailo' / 'cpu' / 'error' must never read the same.

    Asserted as a set so a future wording change that collapses two of them into
    the same string fails here rather than in a support ticket.
    """
    monkeypatch.setattr(accel, "_accel_info", _hat())

    monkeypatch.setattr(accel, "_accel_cfg", {"enabled": True, "yolo_hef": "/m/y.hef"})
    accel.record_inference()
    in_use = accel.live_status()[1]

    monkeypatch.setattr(accel, "_inference_count", 0, raising=False)
    monkeypatch.setattr(accel, "_last_inference_monotonic", None, raising=False)
    on_cpu = accel.live_status()[1]

    monkeypatch.setattr(accel, "_hailo_failed_reason", "HailoRT init failed", raising=False)
    errored = accel.live_status()[1]

    assert len({in_use, on_cpu, errored}) == 3, (
        f"states are not distinguishable: {in_use!r} / {on_cpu!r} / {errored!r}"
    )
    assert "Hailo NPU in use" in in_use
    assert "running on CPU" in on_cpu
    assert "HailoRT init failed" in errored
