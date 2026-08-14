"""Shared object-detection resources so multiple modules don't each load their own copy.

`occupancy.py` and `driver_attention.py` both want a YOLO model for person/object
detection; loading the same weights twice wastes memory and startup time (and
matters concretely when `benchmarks/run_benchmarks.py` instantiates several
modules back-to-back in one process). `get_yolo_model` caches by (weights, device)
so the second caller with the same arguments gets the already-loaded instance.

The optional `device` argument exists for the mac_max profile (which pins detection
to `device="mps"`); every existing caller omits it and gets identical behavior to
before this was added -- the Pi profile's `device=None` path is unchanged.
"""

from __future__ import annotations

from typing import Optional

Box = tuple[int, int, int, int]

_model_cache: dict[tuple[str, Optional[str]], object] = {}


def get_yolo_model(weights: str, device: Optional[str] = None):
    """Return a shared detector instance for `weights`, loading it once.

    When the AI HAT+ is present, HailoRT works, and `accelerator.yolo_hef` is
    configured (see core/accel.py — services/runtime.py calls accel.configure at
    startup), this transparently returns a Hailo-backed model with the same call
    surface, so every module offloads to the accelerator without code changes.
    Otherwise: any ultralytics-loadable checkpoint (YOLO or RT-DETR share the
    same `YOLO(...)` loader, which dispatches on the checkpoint's own
    architecture info).

    Args:
        weights: Path or name of the weights file (e.g. "yolo11n.pt", "yolo11x.pt",
            "rtdetr-l.pt"). Ultralytics auto-downloads it on first use if not cached.
        device: Torch device string (e.g. "mps", "cuda", "cpu"). None (default)
            leaves ultralytics to pick automatically, matching prior behavior.
            An explicit device pin (mac_max's "mps") always bypasses the Hailo path.
    """
    cache_key = (weights, device)
    if cache_key not in _model_cache:
        if device is None:
            from core import accel

            hailo_model = accel.maybe_hailo_yolo(weights)
            if hailo_model is not None:
                _model_cache[cache_key] = hailo_model
                return hailo_model
        from ultralytics import YOLO

        print(
            f"core.detectors: loading YOLO weights '{weights}' "
            "(ultralytics auto-downloads on first use if not cached; needs internet once)"
        )
        model = YOLO(weights)
        if device is not None:
            model.to(device)
            print(f"core.detectors: '{weights}' loaded on device={_model_device(model)}")
        _model_cache[cache_key] = model
    return _model_cache[cache_key]


def _model_device(model) -> str:
    """Best-effort readable device name for a loaded ultralytics model."""
    try:
        return str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        return "unknown"


def predict_roi(
    model, frame, roi: Box, classes: Optional[list[int]] = None, conf: float = 0.25
) -> list[tuple[Box, int, float]]:
    """Run `model` on a CROPPED sub-image of `frame` (a region of interest,
    e.g. around a detected hand) instead of the whole frame, returning boxes
    already translated back into full-frame pixel coordinates.

    A crop makes a small object take up a much larger fraction of the model's
    input resolution than it would in the full frame -- the same lever a
    security camera's "digital PTZ" zoom uses for small/distant subjects, and
    free of any new model: it's the SAME loaded detector (`model`, from
    get_yolo_model), called on fewer pixels. Works with either backend behind
    get_yolo_model (ultralytics YOLO or the Hailo shim), since both share the
    same `.predict(frame, verbose=, classes=, conf=)` -> per-box `.xyxy`/
    `.cls`/`.conf` call surface.

    Returns (box_in_frame_coords, class_id, confidence) tuples; empty if the
    roi is degenerate (zero width/height after clamping to the frame).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in roi)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return []
    crop = frame[y1:y2, x1:x2]
    results = model.predict(crop, verbose=False, classes=classes, conf=conf)
    out: list[tuple[Box, int, float]] = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            cx1, cy1, cx2, cy2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(((int(cx1 + x1), int(cy1 + y1), int(cx2 + x1), int(cy2 + y1)), class_id, score))
    return out


def iou(box_a: Box, box_b: Box) -> float:
    """Compute the Intersection-over-Union of two (x1, y1, x2, y2) pixel boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0
