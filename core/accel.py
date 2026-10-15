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

import contextlib
import hashlib
import os
import shutil
import threading
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
    global _inference_count, _last_inference_monotonic  # noqa: PLW0603 - process-wide inference counters
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


def file_sha256(path: str) -> str:
    """sha256 of a file, streamed: a .hef is megabytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def failure_reason() -> Optional[str]:
    """Why the accelerator was refused, if it was (missing file, bad checksum,
    init failure) — for the error a person sees when nothing else can run."""
    return _hailo_failed_reason


def configure(accel_cfg: dict, info: Optional[AcceleratorInfo] = None) -> AcceleratorInfo:
    """Install the `accelerator:` config block; returns the detection result."""
    global _accel_cfg, _accel_info, _hailo_failed_reason  # noqa: PLW0603 - process-wide accelerator state
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


def live_status() -> tuple[bool, str]:
    """(healthy, detail) for /api/health — grounded in COMPLETED INFERENCES.

    `status()` above answers a config question: "is the configured intent being
    met". That is the right answer at boot, and the wrong one for anyone reading
    /api/health to find out what is actually computing — a seated HAT plus a .hef
    path makes it say "Hailo in use" whether or not a single frame ever reached
    the NPU. On real hardware the two can diverge: a caller pinning `device=`, a
    per-frame inference failure inside a broad handler, or a model loaded before
    configure() all leave the config intact while the work happens on CPU.

    So this reports what RAN. `core/device_health.py::_accelerator` already made
    this distinction for the payload the app sees; this applies the same evidence
    to the device's own endpoint, which had only the config answer.

    The three states an operator needs are worded so they can never be confused:
        "Hailo NPU in use"  -- inference completed on-device, recently
        "running on CPU"    -- nothing has executed on the accelerator
        "<failure reason>"  -- configured but unusable
    """
    healthy, detail = status()
    stats = inference_stats()
    info = _accel_info or detect()

    if stats["live"]:
        return True, f"Hailo NPU in use — {stats['count']} inferences, last {stats['age_s']}s ago"
    if stats["count"] > 0:
        # It worked and then stopped. Not a config problem, and not healthy
        # either: something upstream stopped feeding it.
        return False, (
            f"Hailo NPU idle — {stats['count']} inferences but none for "
            f"{stats['age_s']}s (pipeline stalled or fell back to CPU)"
        )
    if info.present and not healthy:
        return False, detail
    if _accel_cfg.get("yolo_hef") and info.usable:
        # The dangerous case this exists for: fully configured, HAT seated,
        # runtime fine -- and nothing has ever executed on it.
        return False, (
            "running on CPU — accelerator is configured "
            f"({_accel_cfg['yolo_hef']}) but no inference has executed on it"
        )
    return healthy, f"running on CPU — {detail}"


def _refuse(reason: str, *, error: bool = False) -> None:
    """Record why the accelerator is not being used (status() reports it) and log it."""
    global _hailo_failed_reason  # noqa: PLW0603 - process-wide accelerator state
    _hailo_failed_reason = reason
    (log.error if error else log.warning)("%s", reason)


#: One HailoYOLO per .hef file, by real path. The chip is one exclusive device
#: (a second VDevice in the same process fails to open it), and config names
#: exactly one .hef, so every weights name a module asks for — yolo11n.pt here,
#: yolo11s.pt there — is a cache key for the SAME compiled model. Keying the
#: device by weights name, as core.detectors does for the CPU path, made a second
#: name try to open the chip a second time, fail, and fall back to a CPU backend
#: the device does not have.
_hailo_models: dict[str, "HailoYOLO"] = {}
_hailo_models_lock = threading.Lock()


def maybe_hailo_yolo(weights: str):
    """The Hailo-backed model for `weights`, or None to fall back to ultralytics/CPU.

    Called by core.detectors.get_yolo_model. Fallback is always safe and always
    logged; the reason is surfaced through status() so the app's health card
    shows *why* the accelerator isn't being used.

    Integrity is checked HERE, on every load — boot, a preset switch that needs
    a detector, a reload after release_hailo_models() — and never per frame.
    The file's sha256 must equal `accelerator.yolo_hef_sha256` (the pin
    setup_pi.sh verified at install). A mismatch refuses the model outright:
    a .hef replaced or corrupted after install is exactly the file that must not
    run, and on a device there is no CPU detector to fall back to, so the
    runtime goes DEGRADED with this reason rather than detecting with it. An
    unpinned config still loads, loudly, with the computed hash recorded as the
    model's identity so every event it produces stays traceable.
    """
    global _hailo_failed_reason  # noqa: PLW0603 - process-wide accelerator state
    if not _accel_cfg.get("enabled", True):
        return None
    hef_path = _accel_cfg.get("yolo_hef")
    if not hef_path:
        return None
    info = _accel_info or detect()
    if not info.usable:
        return None
    if not os.path.exists(hef_path):
        _refuse(f"accelerator.yolo_hef points to a missing file: {hef_path} (wanted for '{weights}')")
        return None
    key = os.path.realpath(hef_path)
    with _hailo_models_lock:
        shared = _hailo_models.get(key)
        if shared is not None:
            log.info("object detection '%s' -> Hailo (%s, shared)", weights, hef_path)
            return shared
        try:
            actual = file_sha256(hef_path)
        except OSError as exc:
            _refuse(f"accelerator.yolo_hef could not be read ({exc}) — refusing to run it", error=True)
            return None
        pinned = str(_accel_cfg.get("yolo_hef_sha256") or "").strip().lower()
        if pinned and actual != pinned:
            _refuse(
                f"model integrity check FAILED for {hef_path}: sha256 {actual} does not match the "
                f"pinned accelerator.yolo_hef_sha256 {pinned}. Refusing to run it. Re-provision "
                "the model (setup_pi.sh verifies it) or, if the change is intended, update the pin.",
                error=True,
            )
            return None
        if not pinned:
            log.warning(
                "accelerator.yolo_hef_sha256 is not set: running %s UNVERIFIED (sha256 %s). "
                "Pin it so a replaced or corrupted model is refused.", hef_path, actual,
            )
        try:
            model = HailoYOLO(hef_path, sha256=actual)
        except Exception as exc:  # noqa: BLE001 - never let the accelerator kill startup
            _refuse(f"Hailo init failed ({exc}) — running on CPU")
            return None
        # Re-hash after the load: the file was read twice (once to hash, once
        # by HEF()), so a swap between the two reads would otherwise pass.
        try:
            reloaded = file_sha256(hef_path)
        except OSError:
            reloaded = None
        if reloaded != actual:
            model.close()
            _refuse(f"model integrity check FAILED for {hef_path}: the file changed while it was being "
                    "loaded. Refusing to run it.", error=True)
            return None
        _hailo_models[key] = model
        # A verified model is now on the chip, so any earlier refusal (e.g. a
        # transient "Hailo init failed" at boot that a lazy retry has since
        # recovered from) no longer describes this device. Left in place,
        # status() kept reporting that stale failure while inference ran.
        _hailo_failed_reason = None
        log.info("object detection '%s' -> Hailo (%s, sha256 %s)", weights, hef_path, actual[:12])
        return model


def release_hailo_models() -> None:
    """Close every Hailo model and free the chip. The next load re-verifies."""
    with _hailo_models_lock:
        models = list(_hailo_models.values())
        _hailo_models.clear()
    for model in models:
        model.close()


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


def letterbox_transform(frame_w: int, frame_h: int, input_w: int, input_h: int) -> tuple[float, int, int]:
    """Scale + left/top pad that fits (frame_w, frame_h) into (input_w, input_h)
    WITHOUT changing aspect ratio -- the same preprocessing ultralytics' own CPU/
    .pt path already does by default.

    A plain `cv2.resize` straight to a square model input (the common case: the
    Pi camera is 640x480, 4:3, while most Hailo Model Zoo YOLO .hefs are 640x640)
    stretches every object horizontally, which measurably hurts thin/small
    classes -- a phone held at an angle is exactly the shape most distorted by a
    non-uniform stretch. Letterboxing (scale to fit, pad the rest) keeps what the
    model actually sees geometrically consistent with the CPU/.pt backend.
    """
    scale = min(input_w / frame_w, input_h / frame_h)
    new_w, new_h = round(frame_w * scale), round(frame_h * scale)
    pad_w, pad_h = input_w - new_w, input_h - new_h
    return scale, pad_w // 2, pad_h // 2


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-max suppression over xyxy boxes. Returns indices INTO `boxes`.

    Written out rather than pulled from torchvision/ultralytics: this runs on the
    Pi inside the no-NMS decode path, where the whole point is not to depend on a
    heavyweight (and AGPL) inference stack just to suppress overlapping boxes.
    """
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        # A zero-area box would divide by zero; treat it as non-overlapping rather
        # than letting a NaN silently keep or drop a detection.
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


class HailoYOLO:
    """Runs a Model-Zoo YOLO .hef on the AI HAT+ behind the ultralytics call surface.

    Assumes the .hef was compiled with NMS on-device (the Hailo Model Zoo default
    for yolov8/yolo11), whose output is one list per class of
    [ymin, xmin, ymax, xmax, score] rows normalized to [0, 1].
    """

    #: Number of DFL bins per box side. A head emitting 4*(reg_max+1) channels is
    #: doing distribution-based regression (GFocal/DFL): each side's distance is a
    #: softmax over `_DFL_BINS` bins, and the distance is that distribution's
    #: expectation. 68 channels = 4 sides x 17 bins. Verified against the real
    #: damoyolo_tinynasL20_T.hef, whose 68-channel outputs carry raw logits
    #: (range -3.3..9.6, mean ~0) while its 81-channel outputs are already
    #: sigmoid'd probabilities (range 0..1).
    _DFL_BINS = 17

    #: IoU above which two same-class boxes are treated as one detection in the
    #: off-chip NMS. 0.50 rather than the 0.70 the Model Zoo's on-chip yolov8/
    #: yolo11 builds use, from a measured sweep on 200 real driver frames:
    #:
    #:     IoU   0.70   0.60   0.50   0.45   0.40
    #:     multi 55.0%  53.5%  51.5%  50.0%  48.0%   (single-person images with >1 box)
    #:
    #: Note what that sweep says: tightening IoU barely helps, because the extra
    #: boxes are NOT overlapping duplicates — they are spatially separate
    #: low-confidence false positives, which NMS cannot touch. The lever that
    #: actually works is the CONFIDENCE threshold (0.5% multi-person at conf
    #: 0.60, still 100% person availability). 0.50 is taken here for the small
    #: free improvement; the real control lives in each module's configured
    #: confidence, and this comment exists so the next person does not spend a
    #: day tuning IoU expecting it to fix duplicate boxes.
    _DEFAULT_NMS_IOU = 0.50

    def __init__(self, hef_path: str, nms_iou: Optional[float] = None, sha256: Optional[str] = None) -> None:
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

        self._hef_path = hef_path  # kept so model_name() can name what actually ran
        self.sha256 = sha256  # the verified hash of the file that was loaded
        # Inference is serialised: one chip, one activated network group, and
        # HailoRT vstreams are not safe to drive from two threads at once.
        self._lock = threading.Lock()
        # The pipeline's two contexts (vstreams open, network group active),
        # entered ONCE and held — see _ensure_ready. None until the first frame.
        self._session: Optional[contextlib.ExitStack] = None
        self._pipeline = None
        self._closed = False
        self._nms_iou = self._DEFAULT_NMS_IOU if nms_iou is None else float(nms_iou)

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

        # Which decode path this .hef needs, decided from the HEF itself rather
        # than from the filename or config. Two kinds ship in the Model Zoo:
        #
        #   NMS-on-chip  one output, already decoded and non-max-suppressed. The
        #                score threshold and IoU are COMPILED IN and cannot be
        #                changed at runtime -- measured on yolov11n.hef, whose
        #                0.200 floor caps phone recall at 25.4% against CPU's
        #                63.3% because `conf` below that is silently ignored.
        #   raw tensors  several outputs per FPN scale, no NMS. Decoding is ours
        #                to do, which costs CPU -- and buys back full control of
        #                the threshold, which is the thing the on-chip builds
        #                take away.
        #
        # The distinction is load-bearing for accuracy, not a detail: it is the
        # difference between a detector that can see a phone and one that cannot.
        out_infos = self._hef.get_output_vstream_infos()
        self._raw_outputs = len(out_infos) > 1
        self._nms_score_threshold = None
        if self._raw_outputs:
            # Group the per-scale heads. Each FPN level emits a box head
            # (4*_DFL_BINS channels) and a class head (num_classes, possibly
            # padded), at a grid whose stride is input_size / grid_size.
            self._scales = self._group_raw_outputs(out_infos)
            log.info(
                "accel: %s has %d raw output tensors and NO on-chip NMS — decoding "
                "off-chip at strides %s (the confidence threshold is fully ours)",
                os.path.basename(hef_path), len(out_infos),
                [s["stride"] for s in self._scales],
            )
        else:
            log.info(
                "accel: %s has on-chip NMS — its compiled-in score threshold bounds "
                "recall regardless of the `conf` passed to predict()",
                os.path.basename(hef_path),
            )

    def _group_raw_outputs(self, out_infos) -> list[dict]:
        """Pair each scale's box head with its class head, by grid size.

        Output names are model-specific (`conv53`/`conv54` on damoyolo), so the
        pairing is done on SHAPE, which is a property of the architecture rather
        than of one vendor's layer naming: at each grid size exactly one head has
        4*_DFL_BINS channels (boxes) and one has the rest (classes).
        """
        by_grid: dict[int, dict] = {}
        box_channels = 4 * self._DFL_BINS
        for info in out_infos:
            h, w, c = info.shape
            if h != w:
                raise RuntimeError(f"non-square detection grid {info.shape} on {info.name}")
            entry = by_grid.setdefault(h, {"grid": h, "stride": self._input_h // h})
            entry["box" if c == box_channels else "cls"] = info.name
            if c != box_channels:
                entry["num_classes"] = c
        scales = [by_grid[g] for g in sorted(by_grid, reverse=True)]
        for s in scales:
            if "box" not in s or "cls" not in s:
                raise RuntimeError(
                    f"grid {s['grid']} is missing a box or class head: {s}. This .hef's "
                    "head layout is not the 4*(reg_max+1) + classes shape this decoder "
                    "understands, and guessing would silently produce wrong boxes."
                )
        return scales

    def predict(self, frame: np.ndarray, verbose: bool = False, classes=None, conf: float = 0.25, **_ignored):
        """Ultralytics-shaped predict: BGR frame in, [Result] out, boxes in pixel xyxy.

        Extra ultralytics keywords (`device=`, `imgsz=`, ...) are accepted and
        ignored: the comparison modules pass `device=` on every call, and
        without this a null device pin -- the only way they can reach the
        accelerator -- raised TypeError and disabled their phone detection.
        """
        import cv2

        h, w = frame.shape[:2]
        scale, left, top = letterbox_transform(w, h, self._input_w, self._input_h)
        new_w, new_h = round(w * scale), round(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = self._input_w - new_w, self._input_h - new_h
        letterboxed = cv2.copyMakeBorder(
            resized, top, pad_h - top, left, pad_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, axis=0)

        with self._lock:
            pipeline = self._ensure_ready()
            try:
                raw = pipeline.infer({self._input_name: batch})
            except Exception:
                # The device may have reset under us: drop the held contexts so
                # the next frame rebuilds them instead of reusing dead streams.
                # The exception still propagates — the runtime counts failures
                # and escalates exactly as before.
                self._teardown()
                raise
        # Recorded only after the device returned a result, so the health report
        # reflects work the Hailo actually completed.
        record_inference()

        if self._raw_outputs:
            return self._decode_raw(raw, scale, left, top, w, h, classes, conf)

        # One output tensor (NMS postprocess): per image, a list per class of Nx5 rows.
        per_class = raw[next(iter(raw))][0]

        wanted = {int(c) for c in classes} if classes is not None else None
        boxes: list[_ShimBox] = []
        for cls_id, rows in enumerate(per_class):
            if wanted is not None and cls_id not in wanted:
                continue
            for row in np.asarray(rows).reshape(-1, 5):
                ymin, xmin, ymax, xmax, score = (float(v) for v in row)
                if score < conf:
                    continue
                # Normalized coords are relative to the LETTERBOXED (padded, scaled)
                # input; undo padding then scale to get back to original-frame pixels.
                x1 = min(max((xmin * self._input_w - left) / scale, 0.0), w)
                y1 = min(max((ymin * self._input_h - top) / scale, 0.0), h)
                x2 = min(max((xmax * self._input_w - left) / scale, 0.0), w)
                y2 = min(max((ymax * self._input_h - top) / scale, 0.0), h)
                boxes.append(_ShimBox(np.asarray([x1, y1, x2, y2]), cls_id, score))
        return [_ShimResult(boxes)]

    # -------------------------------------------------------------- raw decode
    def _decode_raw(self, raw, scale, left, top, w, h, classes, conf):
        """Decode DFL box logits + class scores from a no-NMS .hef, then suppress.

        Everything the on-chip NMS would have done, done here instead: distance
        decoding, anchor placement, thresholding and non-max suppression. The
        reason to want this is the threshold — an on-chip build bakes one in at
        compile time (0.200 on the Model Zoo YOLO builds), and below it `conf` is
        silently a no-op. Here `conf` is the only gate there is.

        Layout was established empirically against damoyolo_tinynasL20_T.hef
        rather than assumed from the architecture paper:
          * box head, 4*17 channels, RAW LOGITS -> softmax over 17 bins per side,
            expectation = distance from the anchor point in stride units;
          * class head, already sigmoid'd probabilities in [0,1] — verified by
            feeding a real driver image and finding channel 0 (COCO `person`)
            at 0.85, channel 56 (`chair`) and 67 (`cell phone`) plausibly lit.
            Channels beyond the COCO 80 are padding and are dropped.
        """
        wanted = {int(c) for c in classes} if classes is not None else None
        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []

        for spec in self._scales:
            box_raw = np.asarray(raw[spec["box"]])[0]   # (G, G, 4*bins)
            cls_raw = np.asarray(raw[spec["cls"]])[0]   # (G, G, C)
            grid, stride = spec["grid"], spec["stride"]

            # Only the real COCO classes; anything past 80 is padding, and letting
            # it through would invent detections for classes that do not exist.
            scores = cls_raw.reshape(-1, cls_raw.shape[-1])[:, : len(COCO_NAMES)]

            best_cls = scores.argmax(axis=1)
            best_score = scores.max(axis=1)
            keep = best_score >= conf
            if wanted is not None:
                keep &= np.isin(best_cls, list(wanted))
            if not keep.any():
                continue

            # DFL: softmax over the bins of each side, then take the expectation.
            dfl = box_raw.reshape(-1, 4, self._DFL_BINS)[keep]
            dfl = dfl - dfl.max(axis=2, keepdims=True)          # stabilise the exp
            prob = np.exp(dfl)
            prob /= prob.sum(axis=2, keepdims=True)
            dist = (prob * np.arange(self._DFL_BINS, dtype=np.float32)).sum(axis=2)  # (N,4) l,t,r,b

            # Anchor points at cell centres, in letterboxed-input pixels.
            #
            # OPEN QUESTION (2026-10 review, not yet settled on hardware): the
            # +0.5 cell-centre offset below is an ASSUMPTION. Upstream DAMO-YOLO
            # (ZeroHead.get_single_level_center_priors, the NanoDet-Plus
            # convention) and, as recalled, the Hailo Model Zoo's damoyolo
            # postprocess config (nanodet_split, scale_factors [0.0, 0.0]) place
            # priors at the cell's top-left corner, i*stride. If so, every box
            # here is shifted right/down by stride/2 (+4/+8/+16 px at P3/P4/P5
            # in letterbox pixels). Nothing in the repo measures box POSITION
            # (the device numbers are presence/recall metrics, which a constant
            # shift cannot move), so the code is left as is until checked.
            # The check: run the same frames through this decoder and through a
            # reference (DAMO-YOLO PyTorch, or the Model Zoo's own eval
            # postprocess on the same .hef) and compare mean (dx, dy) of matched
            # person boxes. A constant ~+stride/2 bias means change this to
            # `xs * stride` / `ys * stride` (and the two hand-computed
            # expectations in tests/test_phase3_raw_decode.py). See
            # docs/ACCELERATOR.md, "Open question: anchor offset".
            ys, xs = np.divmod(np.nonzero(keep)[0], grid)
            cx = (xs + 0.5) * stride
            cy = (ys + 0.5) * stride

            all_boxes.append(np.stack([
                cx - dist[:, 0] * stride, cy - dist[:, 1] * stride,
                cx + dist[:, 2] * stride, cy + dist[:, 3] * stride,
            ], axis=1))
            all_scores.append(best_score[keep])
            all_classes.append(best_cls[keep])

        if not all_boxes:
            return [_ShimResult([])]

        boxes_xyxy = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        class_ids = np.concatenate(all_classes)

        # Class-wise NMS, matching the on-chip builds' "cross classes: false" —
        # a phone overlapping the hand holding it must not suppress either one.
        keep_idx = []
        for cls_id in np.unique(class_ids):
            sel = np.nonzero(class_ids == cls_id)[0]
            keep_idx.extend(sel[_nms(boxes_xyxy[sel], scores[sel], self._nms_iou)])

        shim: list[_ShimBox] = []
        for i in sorted(keep_idx, key=lambda j: -scores[j]):
            x1, y1, x2, y2 = boxes_xyxy[i]
            # Undo the letterbox: remove padding, then rescale to original pixels.
            shim.append(_ShimBox(
                np.asarray([
                    min(max((x1 - left) / scale, 0.0), w),
                    min(max((y1 - top) / scale, 0.0), h),
                    min(max((x2 - left) / scale, 0.0), w),
                    min(max((y2 - top) / scale, 0.0), h),
                ]),
                int(class_ids[i]), float(scores[i]),
            ))
        return [_ShimResult(shim)]

    @property
    def model_name(self) -> str:
        """The .hef this instance actually loaded, for logs and captured samples.

        Exists so callers can name the running detector without caring which
        backend they hold: ultralytics' YOLO sets `model_name` to its weights
        string, so `getattr(model, "model_name", weights)` now returns something
        TRUE on both paths. Before this property it returned the ultralytics
        weights name on the Hailo path too -- the driver_attention banner
        announced "yolo11n.pt" while a DAMO-YOLO .hef was doing the work, which
        is precisely the confidently-wrong log line that property was added to
        stop printing.
        """
        return os.path.basename(self._hef_path)

    def _ensure_ready(self):
        """The open, activated pipeline — set up on first use, then reused.

        Opening the vstreams and activating the network group used to happen
        inside predict(), i.e. on EVERY frame: two context entries and two
        exits around each inference. HailoRT's own usage is to enter both once
        and infer in a loop; holding them is what this does. Caller holds _lock.
        """
        if self._closed:
            raise RuntimeError("HailoYOLO used after close()")
        if self._pipeline is None:
            session = contextlib.ExitStack()
            try:
                pipeline = session.enter_context(self._infer_pipeline)
                session.enter_context(self._network_group.activate(self._network_group_params))
            except Exception:
                session.close()
                raise
            self._session, self._pipeline = session, pipeline
        return self._pipeline

    def _teardown(self) -> None:
        """Exit the held contexts. Safe to call when nothing is held."""
        session, self._session, self._pipeline = self._session, None, None
        if session is not None:
            try:
                session.close()
            except Exception:  # releasing a dead device must not raise; logged with traceback
                log.warning("accel: error while closing the Hailo pipeline", exc_info=True)

    @property
    def identity(self) -> dict:
        """Which exact model this is — see core.detectors.model_identity."""
        return {"name": self.model_name, "sha256": self.sha256, "backend": "hailo"}

    #: How long close() waits for an in-flight inference before giving up.
    _CLOSE_TIMEOUT_S = 2.0

    def close(self) -> None:
        """Release the pipeline and the Hailo device. Idempotent, and bounded.

        A wedged device can leave an inference thread inside infer() holding
        the lock for ever; waiting on it unconditionally hung shutdown until
        systemd killed the process. After the timeout the instance is marked
        closed and left: process exit releases the chip.

        Once closed (including by a timed-out close), further calls return at
        once: release_all() calls close() once per cache key aliasing this
        instance plus once more via release_hailo_models(), and re-waiting the
        full timeout on each made a wedged shutdown stall 2s per alias.
        """
        if self._closed:
            return
        if not self._lock.acquire(timeout=self._CLOSE_TIMEOUT_S):
            self._closed = True
            log.warning("accel: inference still in progress after %.1fs; not waiting to close the Hailo device",
                        self._CLOSE_TIMEOUT_S)
            return
        try:
            if self._closed:
                return
            self._closed = True
            self._teardown()
            try:
                self._vdevice.release()
            except Exception as exc:  # noqa: BLE001
                log.debug("accel: VDevice.release() failed during close: %r", exc)
        finally:
            self._lock.release()
