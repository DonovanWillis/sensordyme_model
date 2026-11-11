"""Calibrated pinhole intrinsics for head pose, when hardware calibration provides them.

modules/driver_attention.py's solvePnP assumes focal length = image width (a
~53 degree horizontal FOV) and zero lens distortion unless ``camera_hfov_deg``
is set -- and even then distortion stays zero. The deployed lens (Camera
Module 3 Wide, ~102 degrees) is neither. Measuring the real values is a
hardware-calibration job (a checkerboard run with cv2.calibrateCamera); this
module only makes the code ACCEPT them:

    camera_intrinsics:
      width: 1536            # resolution the calibration was measured at
      height: 864
      fx: 615.0              # pixels at that resolution
      fy: 615.0
      cx: 768.0
      cy: 432.0
      distortion: [k1, k2, p1, p2, k3]   # OpenCV order; 4, 5 or 8 values; optional

Values are rescaled to the frame actually processed (same aspect ratio
required). ``null`` (shipped) keeps the historical behaviour exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

_REQUIRED = ("width", "height", "fx", "fy", "cx", "cy")


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_intrinsics(value: Any, where: str = "modules.driver_attention.camera_intrinsics") -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return [f"`{where}:` must be null or a mapping, got {value!r}"]
    errors = []
    unknown = sorted(set(value) - {*_REQUIRED, "distortion"})
    if unknown:
        errors.append(f"`{where}:` unknown keys {unknown}")
    for key in _REQUIRED:
        if key not in value:
            errors.append(f"`{where}.{key}:` is required")
        elif not (_is_num(value[key]) and value[key] > 0):
            errors.append(f"`{where}.{key}:` must be a positive number, got {value[key]!r}")
    dist = value.get("distortion")
    if dist is not None and not (isinstance(dist, list) and len(dist) in (4, 5, 8) and all(_is_num(v) for v in dist)):
        errors.append(f"`{where}.distortion:` must be a list of 4, 5 or 8 numbers (OpenCV order), got {dist!r}")
    return errors


@dataclass(frozen=True)
class CameraIntrinsics:
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @classmethod
    def from_config(cls, value: Any) -> Optional[CameraIntrinsics]:
        if value is None:
            return None
        errors = validate_intrinsics(value)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(
            width=float(value["width"]), height=float(value["height"]),
            fx=float(value["fx"]), fy=float(value["fy"]), cx=float(value["cx"]), cy=float(value["cy"]),
            distortion=tuple(float(v) for v in value.get("distortion") or ()),
        )

    def matrix(self, image_size: tuple[int, int]) -> np.ndarray:
        """3x3 camera matrix for an (h, w) frame, rescaled from the calibration resolution."""
        h, w = image_size
        sx, sy = w / self.width, h / self.height
        if not math.isclose(sx, sy, rel_tol=0.02):
            raise ValueError(
                f"frame {w}x{h} does not share the aspect ratio of the intrinsics' "
                f"{self.width:.0f}x{self.height:.0f}; a crop cannot be rescaled")
        return np.array(
            [[self.fx * sx, 0.0, self.cx * sx], [0.0, self.fy * sy, self.cy * sy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def distortion_coefficients(self) -> np.ndarray:
        """OpenCV distortion vector (zeros when none was calibrated). Scale-free."""
        coeffs = self.distortion or (0.0, 0.0, 0.0, 0.0)
        return np.array(coeffs, dtype=np.float64).reshape(-1, 1)

    def fingerprint(self) -> list[float]:
        """What a stored head-pose calibration must match to be reused."""
        return [self.width, self.height, self.fx, self.fy, self.cx, self.cy, *self.distortion]
