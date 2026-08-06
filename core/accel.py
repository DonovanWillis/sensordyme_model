"""AI accelerator detection and backend selection (Raspberry Pi AI HAT+ / Hailo).

The AI HAT+ (13 TOPS = Hailo-8L, 26 TOPS = Hailo-8) shows up as a PCIe device with
a kernel driver exposing /dev/hailo0, and userspace reaches it through HailoRT
(`hailo_platform` python package, installed on the Pi via the `hailo-all` apt
package). Nothing about it is usable from a stock .pt/.onnx checkpoint: models
must be compiled ahead of time to a .hef with Hailo's Dataflow Compiler.

What this module provides:

  * `detect()` — is a Hailo device physically present and is the runtime importable?
    Runs at startup (scripts/preflight.py) and feeds the health monitor, so "HAT
    seated but unused/missing driver" is visible instead of silent.
  * `configure(accel_cfg)` — called once by services/runtime.py with config.yaml's
    `accelerator:` block. After this, `core.detectors.get_yolo_model()` returns a
    Hailo-backed model automatically whenever a compiled .hef is configured and
    the runtime is available — every module (occupancy, driver_attention, the
    openpilot phone detector) picks it up without code changes.
  * `HailoYOLO` — an adapter exposing the minimal ultralytics surface the modules
    actually use (`.predict(frame, verbose=, classes=, conf=)` -> results with
    `.boxes` / `box.xyxy[0]` / `box.cls[0]` / `box.conf[0]`, plus `.names`), backed
    by a .hef compiled with NMS on-device (the Hailo Model Zoo default for YOLO).

Honest limits (also in the health detail): MediaPipe Face Mesh has no Hailo
build — face landmarks stay on CPU. The openpilot dmonitoring .onnx would also
need offline compilation before it could run here.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.logging_setup import get_logger

log = get_logger("accel")

_DEVICE_PATH = "/dev/hailo0"

# COCO class names, index-aligned with the Hailo Model Zoo YOLO .hefs (same 80
# classes and order as the ultralytics COCO checkpoints the modules configure
# class maps against).
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


@dataclass(frozen=True)
class AcceleratorInfo:
    """The result of probing for an AI accelerator."""

    present: bool          # a Hailo device node exists (HAT is seated + driver loaded)
    runtime_available: bool  # the hailo_platform python runtime imports
    kind: str              # "hailo" or "none"
    detail: str            # human-readable summary for logs / health / preflight

    @property
    def usable(self) -> bool:
        return self.present and self.runtime_available


def detect() -> AcceleratorInfo:
    """Probe for the AI HAT+: device node first, then the HailoRT python runtime."""
    present = os.path.exists(_DEVICE_PATH)
    runtime = False
    if present:
        try:
            import hailo_platform  # noqa: F401  (installed by `apt install hailo-all` on Pi OS)

            runtime = True
        except ImportError:
            pass

    if present and runtime:
        detail = f"Hailo device at {_DEVICE_PATH}, HailoRT runtime available"
    elif present:
        detail = (
            f"Hailo device at {_DEVICE_PATH} but the hailo_platform python runtime is "
            "missing — install it with `sudo apt install hailo-all` and reboot"
        )
    elif shutil.which("hailortcli"):
        detail = "hailortcli installed but no /dev/hailo0 — HAT not seated or driver not loaded"
    else:
        detail = "no Hailo device (running on CPU)"
    return AcceleratorInfo(
        present=present,
        runtime_available=runtime,
        kind="hailo" if present else "none",
        detail=detail,
    )


# ------------------------------------------------------------------ backend selection
# Set once by configure(); consulted by core.detectors.get_yolo_model on every load.
_accel_cfg: dict = {}
_accel_info: Optional[AcceleratorInfo] = None
_hailo_failed_reason: Optional[str] = None

# Proof of work, not proof of configuration. A seated HAT with a runtime and a
# .hef path means the accelerator *should* be in use; only a completed inference
# means it *is*. The app tells a customer (and an investor) "AI running on the
# Hailo-8", so that claim has to rest on something that actually executed.
_inference_count = 0
_last_inference_monotonic: Optional[float] = None


def record_inference() -> None:
    """Called by HailoYOLO after each successful on-device inference."""
    global _inference_count, _last_inference_monotonic
    _inference_count += 1
    _last_inference_monotonic = time.monotonic()


def inference_stats() -> dict:
    """What the accelerator has actually done, for health reporting."""
    age = None if _last_inference_monotonic is None else time.monotonic() - _last_inference_monotonic
    return {
        "count": _inference_count,
        "age_s": None if age is None else round(age, 2),
        # Inference is only "live" if one landed recently. A count from five
        # minutes ago means the pipeline has since stopped or fallen back.
        "live": age is not None and age < 10.0,
    }


def configure(accel_cfg: dict, info: Optional[AcceleratorInfo] = None) -> AcceleratorInfo:
    """Install the `accelerator:` config block; returns the detection result."""
    global _accel_cfg, _accel_info, _hailo_failed_reason
    _accel_cfg = dict(accel_cfg or {})
    _accel_info = info if info is not None else detect()
    _hailo_failed_reason = None
    if _accel_cfg.get("enabled", True):
        log.info("accelerator: %s", _accel_info.detail)
    return _accel_info


def status() -> tuple[bool, str]:
    """(healthy, detail) for the health monitor's `accelerator` component.

    "Healthy" means: the configured intent is being met. CPU-only with no HAT is
    healthy; a seated HAT that can't be used is degraded (visible, not fatal)
    unless `accelerator.required: true`, which preflight treats as a hard stop.
    """
    info = _accel_info or detect()
    cfg = _accel_cfg
    if not cfg.get("enabled", True):
        return True, "accelerator disabled in config"
    if _hailo_failed_reason:
        return False, _hailo_failed_reason
    if info.usable and cfg.get("yolo_hef"):
        return True, f"Hailo in use for object detection ({cfg['yolo_hef']})"
    if info.usable:
        return False, (
            "Hailo present and runtime OK, but no compiled model configured "
            "(accelerator.yolo_hef) — object detection is running on CPU"
        )
    if info.present:
        return False, info.detail
    return True, info.detail  # no HAT at all: CPU-only is the intended mode


def maybe_hailo_yolo(weights: str):
    """The Hailo-backed model for `weights`, or None to fall back to ultralytics/CPU.

    Called by core.detectors.get_yolo_model. Fallback is always safe and always
    logged; the reason is surfaced through status() so the app's health card
    shows *why* the accelerator isn't being used.
    """
    global _hailo_failed_reason
    if not _accel_cfg.get("enabled", True):
        return None
    hef_path = _accel_cfg.get("yolo_hef")
    if not hef_path:
        return None
    info = _accel_info or detect()
    if not info.usable:
        return None
    if not os.path.exists(hef_path):
        _hailo_failed_reason = f"accelerator.yolo_hef points to a missing file: {hef_path}"
        log.warning("%s — falling back to CPU for '%s'", _hailo_failed_reason, weights)
        return None
    try:
        model = HailoYOLO(hef_path)
        log.info("object detection '%s' -> Hailo (%s)", weights, hef_path)
        return model
    except Exception as exc:  # noqa: BLE001 - never let the accelerator kill startup
        _hailo_failed_reason = f"Hailo init failed ({exc}) — running on CPU"
        log.warning("%s", _hailo_failed_reason)
        return None


# ------------------------------------------------------------------ ultralytics-shaped shims
class _ShimBox:
    """One detection, shaped like an ultralytics Box (`xyxy[0]`, `cls[0]`, `conf[0]`)."""

    def __init__(self, xyxy: np.ndarray, cls_id: int, conf: float) -> None:
        self.xyxy = np.asarray([xyxy], dtype=np.float32)
        self.cls = np.asarray([cls_id], dtype=np.float32)
        self.conf = np.asarray([conf], dtype=np.float32)


class _ShimResult:
    """One frame's detections, shaped like an ultralytics Result (`.boxes` iterable)."""

    def __init__(self, boxes: list[_ShimBox]) -> None:
        self.boxes = boxes


class HailoYOLO:
    """Runs a Model-Zoo YOLO .hef on the AI HAT+ behind the ultralytics call surface.

    Assumes the .hef was compiled with NMS on-device (the Hailo Model Zoo default
    for yolov8/yolo11), whose output is one list per class of
    [ymin, xmin, ymax, xmax, score] rows normalized to [0, 1].
    """

    def __init__(self, hef_path: str) -> None:
        from hailo_platform import (  # imported here: only exists on the Pi
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self.names = dict(COCO_NAMES)
        self._hef = HEF(hef_path)
        self._vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()
        self._input_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        self._infer_pipeline = InferVStreams(
            self._network_group, self._input_params, self._output_params
        )
        input_info = self._hef.get_input_vstream_infos()[0]
        self._input_name = input_info.name
        # Model Zoo YOLO inputs are NHWC uint8, e.g. (640, 640, 3).
        self._input_h, self._input_w = input_info.shape[0], input_info.shape[1]

    def predict(self, frame: np.ndarray, verbose: bool = False, classes=None, conf: float = 0.25):
        """Ultralytics-shaped predict: BGR frame in, [Result] out, boxes in pixel xyxy."""
        import cv2

        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self._input_w, self._input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, axis=0)

        with self._infer_pipeline as pipeline:
            with self._network_group.activate(self._network_group_params):
                raw = pipeline.infer({self._input_name: batch})
        # Recorded only after the device returned a result, so the health report
        # reflects work the Hailo actually completed.
        record_inference()
        # One output tensor (NMS postprocess): per image, a list per class of Nx5 rows.
        per_class = raw[next(iter(raw))][0]

        wanted = set(int(c) for c in classes) if classes is not None else None
        boxes: list[_ShimBox] = []
        for cls_id, rows in enumerate(per_class):
            if wanted is not None and cls_id not in wanted:
                continue
            for row in np.asarray(rows).reshape(-1, 5):
                ymin, xmin, ymax, xmax, score = (float(v) for v in row)
                if score < conf:
                    continue
                boxes.append(
                    _ShimBox(
                        np.asarray([xmin * w, ymin * h, xmax * w, ymax * h]),
                        cls_id,
                        score,
                    )
                )
        return [_ShimResult(boxes)]

    def close(self) -> None:
        """Release the Hailo device."""
        try:
            self._vdevice.release()
        except Exception:  # noqa: BLE001
            pass
