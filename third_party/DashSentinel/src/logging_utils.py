"""handles logging of status and features"""

import os
import time

from .utils import ensure_parent, now_ts


CSV_FIELDS = (
    "timestamp",
    "status",
    "confidence",
    "drowsy_score",
    "attentiveness",
    "ear",
    "mar",
    "blink_rate",
    "roll_deg",
    "yaw_ratio",
    "pitch_ratio",
    "closed_frames_norm",
    "yawn_flag",
    "posture_flag",
    "yawn_count",
    "eyes_visible",
    "mouth_visible",
    "visibility_score",
    "frame_quality",
    "frame_brightness",
    "frame_contrast",
    "frame_blur",
    "face_count",
    "model_label",
    "model_confidence",
    "model_drowsy_score",
)


# pylint: disable=too-few-public-methods
class EventLogger:
    """handle logging and features to a CSV file"""

    def __init__(self, path: str, enabled: bool):
        self.path = path
        self.enabled = enabled
        self.last_write = 0.0
        if enabled:
            # create log file with header if it doesn't exist
            ensure_parent(path)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(CSV_FIELDS) + "\n")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def write_periodic(self, status: str, confidence: float, drowsy_score: float,
                       attentiveness: float, features: dict, every_seconds: float = 1.0):
        """periodically update logs with current status and features"""
        if not self.enabled:
            return
        # current timestamp used for fps or timing logic
        now = time.time()
        if now - self.last_write < every_seconds:
            return
        self.last_write = now

        model_score = features.get("model_drowsy_score")
        row = (
            now_ts(),
            status,
            f"{confidence:.4f}",
            f"{drowsy_score:.4f}",
            f"{attentiveness:.2f}",
            f"{features['ear']:.4f}",
            f"{features['mar']:.4f}",
            f"{features['blink_rate']:.2f}",
            f"{features['roll_deg']:.2f}",
            f"{features['yaw_ratio']:.4f}",
            f"{features['pitch_ratio']:.4f}",
            f"{features['closed_frames_norm']:.4f}",
            f"{features['yawn_flag']:.0f}",
            f"{features['posture_flag']:.0f}",
            f"{features['yawn_count']}",
            f"{features.get('eyes_visible', 1.0):.0f}",
            f"{features.get('mouth_visible', 1.0):.0f}",
            f"{features.get('visibility_score', 1.0):.4f}",
            f"{features.get('frame_quality', 1.0):.4f}",
            f"{features.get('frame_brightness', 0.0):.2f}",
            f"{features.get('frame_contrast', 0.0):.2f}",
            f"{features.get('frame_blur', 0.0):.2f}",
            f"{features.get('face_count', 1)}",
            str(features.get("model_label", "unavailable")),
            f"{features.get('model_confidence', 0.0):.4f}",
            "" if model_score is None else f"{model_score:.4f}",
        )

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")
