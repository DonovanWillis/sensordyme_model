"""Optional local deep-learning model adapter for DashSentinel.

No cloud calls are made. If the user provides an ONNX model path, OpenCV DNN
loads it locally and fuses its output with the landmark-based scorer.
"""

import cv2
import numpy as np

from .utils import clamp


def _softmax(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = float(np.sum(exp_values))
    if denom <= 0.0:
        return np.zeros_like(values)
    return exp_values / denom


def _sigmoid(value):
    return 1.0 / (1.0 + float(np.exp(-value)))


class LocalDMSModel:
    """Thin wrapper around a local ONNX classifier.

    Supported output contracts:
    - one output value: interpreted as drowsy/unsafe logit or probability
    - two output values: interpreted as [alert, unsafe]
    - three or more output values: interpreted as [alert, warning, drowsy, ...]
    """

    def __init__(self, model_path=None, input_size=224, enabled=True):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.net = None
        self.load_error = None

        if not enabled or not model_path:
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
        except cv2.error as exc:
            self.load_error = str(exc)
            self.net = None

    @property
    def available(self):
        """True when a local model loaded successfully."""
        return self.net is not None

    def predict(self, face_crop):
        """Run local inference on a selected face crop.

        Returns fields that can be merged into the frame feature dict. Empty dict
        means no usable local model prediction was available.
        """
        if self.net is None or face_crop is None or face_crop.size == 0:
            return {}

        try:
            resized = cv2.resize(face_crop, (self.input_size, self.input_size))
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0 / 255.0,
                size=(self.input_size, self.input_size),
                mean=(0.0, 0.0, 0.0),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            output = np.asarray(self.net.forward(), dtype=np.float32).reshape(-1)
        except cv2.error as exc:
            self.load_error = str(exc)
            return {}

        if output.size == 0:
            return {}

        if output.size == 1:
            raw_value = float(output[0])
            unsafe_score = raw_value if 0.0 <= raw_value <= 1.0 else _sigmoid(raw_value)
            label = "unsafe" if unsafe_score >= 0.5 else "alert"
            confidence = max(unsafe_score, 1.0 - unsafe_score)
        elif output.size == 2:
            probs = _softmax(output)
            unsafe_score = float(probs[1])
            label = "unsafe" if unsafe_score >= float(probs[0]) else "alert"
            confidence = float(np.max(probs))
        else:
            probs = _softmax(output[:3])
            unsafe_score = float(0.50 * probs[1] + probs[2])
            label_idx = int(np.argmax(probs))
            label = ["alert", "warning", "drowsy"][label_idx]
            confidence = float(probs[label_idx])

        return {
            "model_drowsy_score": float(clamp(unsafe_score, 0.0, 1.0)),
            "model_confidence": float(clamp(confidence, 0.0, 1.0)),
            "model_label": label,
        }
