"""Frame quality checks for real-time driver monitoring.

These checks are deliberately lightweight so they can run locally before any
landmark or optional deep-learning model inference. They help the app avoid
making high-confidence claims from blurry, dark, washed-out, or low-contrast
frames.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import clamp


@dataclass(frozen=True)
class FrameQuality:
    """Summarizes whether a frame is suitable for driver-state scoring."""

    brightness: float
    contrast: float
    blur: float
    score: float
    usable: bool
    reason: str

    def as_features(self):
        """Return quality fields in the same dict style used by feature extraction."""
        return {
            "frame_brightness": float(self.brightness),
            "frame_contrast": float(self.contrast),
            "frame_blur": float(self.blur),
            "frame_quality": float(self.score),
            "frame_usable": float(1.0 if self.usable else 0.0),
            "frame_quality_reason": self.reason,
        }


DEFAULT_FRAME_QUALITY = FrameQuality(
    brightness=128.0,
    contrast=64.0,
    blur=250.0,
    score=1.0,
    usable=True,
    reason="ok",
)

def _brightness_score(brightness, min_brightness, max_brightness):
    """Calculate a brightness score between 0 and 1 
    based on how the brightness compares to thresholds."""
    if brightness < min_brightness:
        return clamp(brightness / max(min_brightness, 1.0), 0.0, 1.0)

    if brightness > max_brightness:
        denominator = max(255.0 - max_brightness, 1.0)
        return clamp((255.0 - brightness) / denominator, 0.0, 1.0)

    return 1.0

# pylint: disable=too-many-arguments, too-many-positional-arguments
def _quality_reasons(brightness, contrast, blur, min_brightness,
                     max_brightness, min_contrast, min_blur):
    """Generate a list of reasons why a frame might be considered low quality."""
    reasons = []

    if brightness < min_brightness:
        reasons.append("poor lighting: too dark")
    if brightness > max_brightness:
        reasons.append("poor lighting: overexposed")
    if contrast < min_contrast:
        reasons.append("low contrast")
    if blur < min_blur:
        reasons.append("unclear/blurry frame")

    return reasons


def measure_frame_quality(
    frame,
    min_brightness=35.0,
    max_brightness=225.0,
    min_contrast=18.0,
    min_blur=45.0,
):
    """Measure brightness, contrast, and blur for a BGR frame."""
    if frame is None or frame.size == 0:
        return FrameQuality(0.0, 0.0, 0.0, 0.0, False, "empty frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    score = min(
        _brightness_score(brightness, min_brightness, max_brightness),
        clamp(contrast / max(min_contrast, 1.0), 0.0, 1.0),
        clamp(blur / max(min_blur, 1.0), 0.0, 1.0),
    )

    reasons = _quality_reasons(
        brightness,
        contrast,
        blur,
        min_brightness,
        max_brightness,
        min_contrast,
        min_blur,
    )

    return FrameQuality(
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        score=float(score),
        usable=not reasons,
        reason="; ".join(reasons) if reasons else "ok"
    )
