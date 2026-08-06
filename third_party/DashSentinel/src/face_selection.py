"""Face selection helpers for multi-face driver-monitoring scenes."""

from dataclasses import dataclass

from .utils import clamp


@dataclass(frozen=True)
class FaceSelection:
    """Selected face plus metadata used for scoring and debugging."""

    landmarks: object
    bbox: tuple
    face_count: int
    selected_index: int
    area_ratio: float
    center_offset: float

    def as_features(self):
        """Return selection metadata as a feature dict."""
        return {
            "face_count": int(self.face_count),
            "selected_face_index": int(self.selected_index),
            "selected_face_area": float(self.area_ratio),
            "selected_face_center_offset": float(self.center_offset),
        }


def landmark_bbox(landmarks, width, height):
    """Return a clamped pixel-space bounding box for one MediaPipe face."""
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x1 = int(clamp(min(xs), 0, width - 1))
    y1 = int(clamp(min(ys), 0, height - 1))
    x2 = int(clamp(max(xs), 0, width - 1))
    y2 = int(clamp(max(ys), 0, height - 1))
    return x1, y1, x2, y2


def expand_bbox(bbox, width, height, padding=0.18):
    """Expand a bounding box while keeping it inside the frame."""
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    return (
        int(clamp(x1 - pad_x, 0, width - 1)),
        int(clamp(y1 - pad_y, 0, height - 1)),
        int(clamp(x2 + pad_x, 0, width - 1)),
        int(clamp(y2 + pad_y, 0, height - 1)),
    )


def crop_bbox(frame, bbox):
    """Crop a frame to a bounding box. Returns None for invalid boxes."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# pylint: disable=too-many-locals
def choose_driver_face(face_landmarks, width, height):
    """Pick the most likely driver face from one or more detected faces.

    A real DMS should eventually track identity over time. For this prototype,
    the safest deterministic rule is to prefer the largest face near the frame
    center because dash-mounted cameras usually put the driver closest and most
    central.
    """
    if not face_landmarks:
        return None

    frame_area = float(max(width * height, 1))
    best = None
    best_score = -1.0
    face_count = len(face_landmarks)

    for index, face in enumerate(face_landmarks):
        landmarks = face.landmark
        bbox = landmark_bbox(landmarks, width, height)
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        area_ratio = clamp((box_w * box_h) / frame_area, 0.0, 1.0)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        norm_dx = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        norm_dy = abs(center_y - height / 2.0) / max(height / 2.0, 1.0)
        center_offset = clamp((norm_dx + norm_dy) / 2.0, 0.0, 1.0)

        score = (0.75 * area_ratio) + (0.25 * (1.0 - center_offset))
        if score > best_score:
            best_score = score
            best = FaceSelection(
                landmarks=landmarks,
                bbox=bbox,
                face_count=face_count,
                selected_index=index,
                area_ratio=float(area_ratio),
                center_offset=float(center_offset),
            )

    return best
