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

import os
from typing import Any, Optional

Box = tuple[int, int, int, int]

# Any: either an ultralytics YOLO or core.accel.HailoYOLO — one call surface, no shared base class.
_model_cache: dict[tuple[str, Optional[str]], Any] = {}
#: sha256 per resolved weights path, computed once per load (never per frame).
_weights_sha256: dict[str, Optional[str]] = {}


def get_yolo_model(weights: str, device: Optional[str] = None) -> Any:
    """Return a shared detector instance for `weights`, loading it once.

    When the AI HAT+ is present, HailoRT works, and `accelerator.yolo_hef` is
    configured (see core/accel.py — services/runtime.py calls accel.configure at
    startup), this transparently returns a Hailo-backed model with the same call
    surface, so every module offloads to the accelerator without code changes.
    Otherwise it falls back to the CPU backend, which needs `ultralytics` — an
    AGPL-3.0 package that is DELIBERATELY NOT part of the device install (see
    `require_backend` below and docs/LICENSING.md). On a developer machine that
    installed requirements.txt it loads any ultralytics-loadable checkpoint (YOLO
    or RT-DETR share the same `YOLO(...)` loader, which dispatches on the
    checkpoint's own architecture info).

    Args:
        weights: Path or name of the weights file (e.g. "yolo11n.pt", "yolo11x.pt",
            "rtdetr-l.pt"). Ultralytics auto-downloads it on first use if not cached.
        device: Torch device string (e.g. "mps", "cuda", "cpu"). None (default)
            leaves ultralytics to pick automatically, matching prior behavior.
            An explicit device pin (mac_max's "mps") always bypasses the Hailo path.

    Raises:
        RuntimeError: when neither backend is available — no usable accelerator
            and no `ultralytics` installed. Never returns a stub: a detector that
            silently finds nothing is worse than one that refuses to start.
    """
    cache_key = (weights, device)
    if cache_key not in _model_cache:
        if device is None:
            from core import accel

            hailo_model = accel.maybe_hailo_yolo(weights)
            if hailo_model is not None:
                _model_cache[cache_key] = hailo_model
                return hailo_model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            message = _no_backend_message(weights, device)
            if device is None:
                from core import accel

                # Say WHY the accelerator was not used when it was refused —
                # above all a failed integrity check, which must never read as a
                # generic "install something" problem.
                if accel.failure_reason():
                    message = f"{message}\n\nAccelerator refused: {accel.failure_reason()}"
            raise RuntimeError(message) from exc

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


def _no_backend_message(weights: str, device: Optional[str]) -> str:
    """The text a person sees when there is no detector at all.

    Written out at length on purpose. The person hitting this is usually
    installing a device, not reading this file, and the two fixes are completely
    different depending on WHICH machine they are on — so a bare
    "ModuleNotFoundError: No module named 'ultralytics'" is close to useless
    here. It names the weights, both fixes, and why the AGPL package is not
    simply installed for them.
    """
    if device is not None:
        # An explicit device pin is the mac_max profile; it never consults the
        # accelerator, so pointing at a Hailo HAT would be a wrong instruction.
        return (
            f"core.detectors: cannot load object-detection weights '{weights}' on "
            f"device='{device}'. This profile runs detection on the CPU/GPU via the "
            "`ultralytics` package, which is not installed.\n"
            "  Fix: pip install -r requirements.txt (the developer set).\n"
            "Note that `ultralytics` is AGPL-3.0 and is therefore NOT part of the "
            "device install (requirements-device.txt) — see docs/LICENSING.md."
        )
    return (
        f"core.detectors: no object-detection backend available for '{weights}'.\n"
        "\n"
        "The AI HAT+ accelerator is not usable (not seated, HailoRT missing, or "
        "`accelerator.yolo_hef` unset/pointing at a missing file — /api/health "
        "reports which), and the CPU fallback needs the `ultralytics` package, "
        "which is not installed.\n"
        "\n"
        "`ultralytics` is AGPL-3.0, so it is deliberately excluded from the device "
        "install (requirements-device.txt). See docs/LICENSING.md.\n"
        "\n"
        "Fix ONE of these:\n"
        "  * Device (the shipping path): seat the AI HAT+, `sudo apt install "
        "hailo-all`, and set `accelerator.yolo_hef` in config.yaml to a compiled "
        ".hef.\n"
        "  * Developer machine ONLY: pip install -r requirements.txt. Do not do "
        "this on a shipped device — it puts an AGPL-3.0 package on hardware you "
        "convey to a customer."
    )


def require_backend(weights: str, device: Optional[str] = None) -> str:
    """Preflight: confirm a detector backend exists, or raise the same actionable error.

    WHY A SEPARATE ENTRY POINT. `get_yolo_model` already fails correctly, but not
    all callers reach it at the same time: occupancy.py loads its model in
    `__init__` (so a missing backend surfaces during startup), while
    driver_attention.py loads lazily on the first stride tick — inside the
    inference loop, where services/runtime.py catches per-frame exceptions and
    only escalates after several consecutive failures. That path still fails, but
    it fails as a warm-up warning followed by a restart loop rather than as one
    clear message at boot.

    Call this once at startup, after `accel.configure`, to collapse both cases
    into the same immediate, readable failure. Returns the backend name in use
    ("hailo" or "ultralytics") so a caller can log it.

    This LOADS the model (through `get_yolo_model`, so it lands in the shared
    cache and the lazy caller gets it for free) rather than merely probing for
    one. Probing separately would construct a second Hailo VDevice for the same
    .hef just to throw it away, and "the import works" is a weaker claim than
    "the weights actually loaded" anyway.
    """
    model = get_yolo_model(weights, device)
    return "hailo" if type(model).__name__ == "HailoYOLO" else "ultralytics"


def model_identity(weights: str, device: Optional[str] = None) -> Optional[dict]:
    """Which exact model `get_yolo_model(weights, device)` loaded, or None if not loaded.

    {"name", "sha256", "backend"}: the file actually executing (the .hef on the
    accelerator, not the .pt name that was only a cache key), the sha256 of the
    file as loaded, and which runtime ran it. A name alone cannot answer "which
    exact model produced this event" — two different files can share one.
    """
    model = _model_cache.get((weights, device))
    if model is None:
        return None
    identity = getattr(model, "identity", None)
    if isinstance(identity, dict):
        return dict(identity)
    path = getattr(model, "ckpt_path", None) or weights
    return {
        "name": os.path.basename(str(path)),
        "sha256": _sha256_once(str(path)),
        "backend": "ultralytics",
    }


def _sha256_once(path: str) -> Optional[str]:
    """sha256 of a local weights file, cached per path; None if it is not a file."""
    if path not in _weights_sha256:
        from core.accel import file_sha256

        try:
            _weights_sha256[path] = file_sha256(path) if os.path.isfile(path) else None
        except OSError:
            _weights_sha256[path] = None
    return _weights_sha256[path]


def release_all() -> None:
    """Close every loaded detector and forget it. Called at shutdown.

    The Hailo models hold the chip; process exit would free it too, but a clean
    close is what lets a reload in the same process work — and every reload
    passes through the integrity check again.
    """
    from core import accel

    for model in list(_model_cache.values()):
        close = getattr(model, "close", None)
        if callable(close):
            close()
    _model_cache.clear()
    _weights_sha256.clear()
    accel.release_hailo_models()


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


def upper_region(box: Box, fraction: float = 0.35) -> Box:
    """The top `fraction` of a box — where a standing or seated person's face is.

    The evidence upgrade behind core/rules.py's `near_mouth` and
    modules/baby_monitoring.py's choking-hazard risk scoring: an object
    overlapping a person anywhere is "holding" at best (it also matches an
    object at their feet in box terms); an object overlapping the region
    their face occupies has been raised toward the mouth. Still an
    approximation — a box knows nothing about an actual mouth, and this
    module has no opinion on WHOSE face, standing/seated/reclined, or infant
    vs adult — but it is the difference between "has a drink" and "is
    drinking from it" at the fidelity a box detector can support, without
    claiming a facial-landmark capability this repo does not have (see
    modules/baby_monitoring.py's module docstring on infant face models).
    """
    x1, y1, x2, y2 = box
    return (x1, y1, x2, int(y1 + (y2 - y1) * fraction))


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
