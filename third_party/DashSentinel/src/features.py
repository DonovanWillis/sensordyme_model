"""handles feature extraction from facial landmarks"""

import time
from collections import deque

import cv2
import numpy as np

from .constants import (
    LEFT_CHEEK,
    LEFT_EYE,
    LEFT_EYE_OUTER,
    LEFT_MOUTH,
    MOUTH_BOTTOM,
    MOUTH_TOP,
    NOSE_TIP,
    RIGHT_CHEEK,
    RIGHT_EYE,
    RIGHT_EYE_OUTER,
    RIGHT_MOUTH,
)
from .utils import clamp, euclidean
from .vision_quality import DEFAULT_FRAME_QUALITY


def get_point(landmarks, idx, w, h):
    """convert landmark to pixel coordinates"""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Get eye aspect ratio for given eye landmarks."""
    pts = [get_point(landmarks, idx, w, h) for idx in eye_indices]

    if len(pts) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    """get mouth aspect ratio for given mouth landmarks"""
    top = get_point(landmarks, MOUTH_TOP, w, h)
    bottom = get_point(landmarks, MOUTH_BOTTOM, w, h)
    left = get_point(landmarks, LEFT_MOUTH, w, h)
    right = get_point(landmarks, RIGHT_MOUTH, w, h)

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    # return 0 to avoid division by zero and indicate mouth is closed
    if horizontal == 0:
        return 0.0

    # higher ratio indicates mouth is more open.
    return vertical / horizontal


def estimate_head_pose(landmarks, w, h):
    """estimate head pose: roll, yaw, and pitch."""
    points = {
        "left_eye": get_point(landmarks, LEFT_EYE_OUTER, w, h),
        "right_eye": get_point(landmarks, RIGHT_EYE_OUTER, w, h),
        "nose": get_point(landmarks, NOSE_TIP, w, h),
        "mouth_top": get_point(landmarks, MOUTH_TOP, w, h),
        "mouth_bottom": get_point(landmarks, MOUTH_BOTTOM, w, h),
        "left_cheek": get_point(landmarks, LEFT_CHEEK, w, h),
        "right_cheek": get_point(landmarks, RIGHT_CHEEK, w, h),
    }

    eye_center = (points["left_eye"] + points["right_eye"]) / 2.0
    mouth_center = (points["mouth_top"] + points["mouth_bottom"]) / 2.0

    dx = points["right_eye"][0] - points["left_eye"][0]
    dy = points["right_eye"][1] - points["left_eye"][1]
    roll_deg = np.degrees(np.arctan2(dy, dx)) if abs(dx) > 1e-6 else 0.0

    left_dist = euclidean(points["nose"], points["left_cheek"])
    right_dist = euclidean(points["nose"], points["right_cheek"])
    total_dist = left_dist + right_dist

    yaw_ratio = 0.0
    if total_dist > 1e-6:
        yaw_ratio = (right_dist - left_dist) / total_dist

    face_vertical = max(mouth_center[1] - eye_center[1], 1e-6)
    pitch_ratio = (points["nose"][1] - eye_center[1]) / face_vertical

    return float(roll_deg), float(yaw_ratio), float(pitch_ratio)


def enhance_lighting(frame):
    """enhance lighting and contrast of the input frame using CLAHE and gamma correction"""
    # convert image color space (bgr <-> rgb/lab)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))

    # convert image color space (bgr <-> rgb/lab)
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gamma = 1.12
    table = np.array(
        [(i / 255.0) ** (1.0 / gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
    return enhanced


def _landmark_group_visible(landmarks, indices, width, height, min_width_ratio=0.025):
    """Estimate whether a landmark group is visible enough to trust.

    MediaPipe FaceMesh often has no useful visibility score, so this combines
    in-frame checks and minimum landmark spread. This is not a perfect occlusion
    detector, but it prevents obvious out-of-frame/too-small landmark failures
    from driving eye or mouth decisions.
    """
    pts = [get_point(landmarks, idx, width, height) for idx in indices]
    if not pts:
        return False, 0.0

    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    margin_x = width * 0.015
    margin_y = height * 0.015
    in_frame = [
        margin_x <= x <= width - margin_x and margin_y <= y <= height - margin_y
        for x, y in zip(xs, ys)
    ]
    in_frame_ratio = sum(in_frame) / len(in_frame)
    spread_ratio = (max(xs) - min(xs)) / max(float(width), 1.0)
    spread_ok = spread_ratio >= min_width_ratio

    confidence = clamp(
        0.70 * in_frame_ratio + 0.30 * clamp(spread_ratio / min_width_ratio, 0, 1),
        0,
        1,
    )
    return bool(in_frame_ratio >= 0.90 and spread_ok), float(confidence)


# pylint: disable=too-many-instance-attributes
class FeatureExtractor:
    """feature extractor for all facial features"""

    # initialize thresholds and state for feature extraction
    def __init__(self, ear_threshold=0.23, yawn_mar_threshold=0.4, yawn_frames_threshold=30):
        # compute eye aspect ratio to detect eye closure
        self.ear_threshold = ear_threshold
        self.yawn_mar_threshold = yawn_mar_threshold
        self.yawn_frames_threshold = yawn_frames_threshold

        # compute eye aspect ratio to detect eye closure
        self.ear_history = deque(maxlen=10)
        self.blink_timestamps = deque(maxlen=90)
        self.eye_closed = False
        self.closed_frames = 0
        self.mouth_open_frames = 0
        self.last_yawn_time = 0.0
        self.yawn_count = 0
        self.last_ear = 0.28
        self.last_mar = 0.16

        # for head motion
        self.mar_history = deque(maxlen=30)
        self.roll_history = deque(maxlen=30)
        self.yaw_history = deque(maxlen=30)
        self.pitch_history = deque(maxlen=30)

        self.bad_pose_frames = 0
        self.look_away_frames = 0
        self.head_tilt_frames = 0
        self.head_back_frames = 0

    def reset(self):
        """reset all states/counters for a new session"""
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self.mar_history.clear()
        self.roll_history.clear()
        self.yaw_history.clear()
        self.pitch_history.clear()

        self.eye_closed = False
        self.closed_frames = 0
        self.mouth_open_frames = 0
        self.bad_pose_frames = 0
        self.look_away_frames = 0
        self.head_tilt_frames = 0
        self.head_back_frames = 0

    # pylint: disable=too-many-locals
    def _visibility_features(self, landmarks, w, h, face_bbox):
        left_eye_visible, left_eye_conf = _landmark_group_visible(landmarks, LEFT_EYE, w, h)
        right_eye_visible, right_eye_conf = _landmark_group_visible(landmarks, RIGHT_EYE, w, h)
        mouth_visible, mouth_conf = _landmark_group_visible(
            landmarks,
            [LEFT_MOUTH, RIGHT_MOUTH, MOUTH_TOP, MOUTH_BOTTOM],
            w,
            h,
            min_width_ratio=0.035,
        )

        face_area_ratio = 0.0
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_area_ratio = clamp(((x2 - x1) * (y2 - y1)) / max(float(w * h), 1.0), 0.0, 1.0)

        face_large_enough = face_area_ratio >= 0.035
        eyes_visible = left_eye_visible and right_eye_visible and face_large_enough
        landmark_visibility = (left_eye_conf + right_eye_conf + mouth_conf) / 3.0
        visibility_score = clamp(
            0.60 * landmark_visibility + 0.40 * clamp(face_area_ratio / 0.12, 0.0, 1.0), 0.0, 1.0
        )

        return {
            "eyes_visible": float(1.0 if eyes_visible else 0.0),
            "mouth_visible": float(1.0 if mouth_visible and face_large_enough else 0.0),
            "left_eye_visible": float(1.0 if left_eye_visible else 0.0),
            "right_eye_visible": float(1.0 if right_eye_visible else 0.0),
            "landmark_visibility": float(landmark_visibility),
            "visibility_score": float(visibility_score),
            "face_area_ratio": float(face_area_ratio),
        }

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-arguments,too-many-positional-arguments
    def extract(
        self,
        landmarks,
        w,
        h,
        frame_quality=None,
        face_selection=None,
        model_output=None,
    ):
        """compute features from face landmarks (ear, mar, blink, pose)"""
        frame_quality = frame_quality or DEFAULT_FRAME_QUALITY
        model_output = model_output or {}
        face_bbox = face_selection.bbox if face_selection is not None else None
        visibility = self._visibility_features(landmarks, w, h, face_bbox)
        eyes_visible = visibility["eyes_visible"] >= 0.5
        mouth_visible = visibility["mouth_visible"] >= 0.5

        # current timestamp used for fps or timing logic
        now = time.time()

        # compute eye aspect ratio only when eyes are visible enough to trust
        if eyes_visible:
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            self.last_ear = ear
            self.ear_history.append(ear)
            ear_smoothed = sum(self.ear_history) / len(self.ear_history)

            if ear_smoothed < self.ear_threshold:
                if not self.eye_closed:
                    self.eye_closed = True
                self.closed_frames += 1
            else:
                if self.eye_closed:
                    self.blink_timestamps.append(now)
                    self.eye_closed = False
                self.closed_frames = max(0, self.closed_frames - 2)
        else:
            ear_smoothed = self.last_ear
            self.eye_closed = False
            self.closed_frames = max(0, self.closed_frames - 3)

        # remove old blink timestamps beyond 60 seconds to compute recent blink rate
        while self.blink_timestamps and now - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()
        blink_rate = float(len(self.blink_timestamps)) if eyes_visible else 0.0

        # compute mouth aspect ratio only when mouth is visible enough to trust
        if mouth_visible:
            mar = mouth_aspect_ratio(landmarks, w, h)
            self.last_mar = mar
            self.mar_history.append(mar)
            mar_smoothed = sum(self.mar_history) / len(self.mar_history)
            if mar_smoothed > self.yawn_mar_threshold:
                self.mouth_open_frames += 1
            else:
                if (
                    self.mouth_open_frames >= self.yawn_frames_threshold
                    and now - self.last_yawn_time > 2.0
                ):
                    self.yawn_count += 1
                    self.last_yawn_time = now
                self.mouth_open_frames = 0
        else:
            mar_smoothed = self.last_mar
            self.mouth_open_frames = max(0, self.mouth_open_frames - 3)

        # head pose estimation to detect posture issues (looking away, tilting, etc.)
        yawn_flag = (
            1.0
            if mouth_visible and self.mouth_open_frames >= self.yawn_frames_threshold
            else 0.0
        )
        roll_deg, yaw_ratio, pitch_ratio = estimate_head_pose(landmarks, w, h)

        self.roll_history.append(roll_deg)
        self.yaw_history.append(yaw_ratio)
        self.pitch_history.append(pitch_ratio)

        roll_smooth = sum(self.roll_history) / len(self.roll_history)
        yaw_smooth = sum(self.yaw_history) / len(self.yaw_history)
        pitch_smooth = sum(self.pitch_history) / len(self.pitch_history)

        # detect sustained unsafe head posture
        looking_away = abs(yaw_smooth) > 0.7
        head_tilted = abs(roll_smooth) > 15.0
        head_back_or_down = pitch_smooth < 0.20 or pitch_smooth > 0.85
        bad_pose_now = looking_away or head_tilted or head_back_or_down

        if looking_away:
            self.look_away_frames += 1
        else:
            self.look_away_frames = max(0, self.look_away_frames - 2)

        if head_tilted:
            self.head_tilt_frames += 1
        else:
            self.head_tilt_frames = max(0, self.head_tilt_frames - 2)

        if head_back_or_down:
            self.head_back_frames += 1
        else:
            self.head_back_frames = max(0, self.head_back_frames - 2)

        if bad_pose_now:
            self.bad_pose_frames += 1
        else:
            self.bad_pose_frames = max(0, self.bad_pose_frames - 2)

        bad_pose_norm = clamp(self.bad_pose_frames / 24.0, 0.0, 1.0)
        look_away_norm = clamp(self.look_away_frames / 18.0, 0.0, 1.0)
        head_tilt_norm = clamp(self.head_tilt_frames / 18.0, 0.0, 1.0)
        head_back_norm = clamp(self.head_back_frames / 18.0, 0.0, 1.0)

        posture_flag = 1.0 if self.bad_pose_frames >= 8 else 0.0
        closed_frames_norm = clamp(self.closed_frames / 30.0, 0.0, 1.0)

        features = {
            "ear": float(ear_smoothed),
            "mar": float(mar_smoothed),
            "blink_rate": float(blink_rate),
            "closed_frames_norm": float(closed_frames_norm),
            "yawn_flag": float(yawn_flag),
            "yawn_count": int(self.yawn_count),
            "roll_deg": float(roll_smooth),
            "yaw_ratio": float(yaw_smooth),
            "pitch_ratio": float(pitch_smooth),
            "posture_flag": float(posture_flag),
            "bad_pose_norm": float(bad_pose_norm),
            "look_away_norm": float(look_away_norm),
            "head_tilt_norm": float(head_tilt_norm),
            "head_back_norm": float(head_back_norm),
            "looking_away": float(1.0 if looking_away else 0.0),
            "head_tilted": float(1.0 if head_tilted else 0.0),
            "head_back_or_down": float(1.0 if head_back_or_down else 0.0),
        }
        features.update(visibility)
        features.update(frame_quality.as_features())
        if face_selection is not None:
            features.update(face_selection.as_features())
        else:
            features.update({"face_count": 1, "selected_face_index": 0, "selected_face_area": 0.0})
        features.update(model_output)
        features.setdefault("model_drowsy_score", None)
        features.setdefault("model_confidence", 0.0)
        features.setdefault("model_label", "unavailable")
        return features

    def reset_stats(self):
        """resets all states/counters"""
        self.reset()
        self.yawn_count = 0
        self.last_yawn_time = 0.0
