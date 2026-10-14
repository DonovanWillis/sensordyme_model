"""mac_max profile: high-accuracy driver attention, GPU-backed (Apple Silicon / MPS only).

This is an ADDITIVE, Mac-only module. It never runs on the Pi, never reads the
`modules:` section of config.yaml, and does not affect `modules/driver_attention.py`
in any way (they don't import from each other for behavior; this module imports a
handful of pure, side-effect-free helpers from it — eye_aspect_ratio, head-pose
estimation, and the generic rolling trackers — purely for reuse, not coupling).

It exists for demos, generating ground-truth comparisons, and establishing a
GPU-backed accuracy ceiling. The Raspberry Pi profile (`driver_attention.py`) is
the shipping product.

Three upgrades over the Pi profile, in order of how much they change the signal
quality (not just the compute budget):

  1. Gaze, not head pose. `GazeCalibrator` and `_compute_gaze_components` fuse
     head yaw/pitch with iris position (MediaPipe Face Mesh's refined iris
     landmarks) into an actual gaze vector, then calibrate a neutral "looking at
     the screen center" baseline the same way the Pi profile calibrates head
     pose — but gaze is a materially better signal for eyes-off-road than head
     angle alone (a driver can turn their head only slightly and still look far
     off-axis with their eyes).
  2. Tiered drowsiness with a real blink-rate fatigue signal (BlinkRateTracker),
     not just a single sustained-closure threshold.
  3. Confidence-stacked phone/object fusion — three distinct tiers (phone alone,
     phone in hand, phone in hand while distracted) instead of one binary check —
     plus repeated hand-to-face motion for eating/drinking.

Every threshold's real-world basis is cited in config.yaml under `mac_max:`, not
here — this docstring only explains the mechanisms.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core.detectors import Box, get_yolo_model, iou
from core.events import Event
from core.vlm_judge import VLM_EVENT_BY_STATE, VLMConfirmTracker, VLMJudge, vlm_is_local
from modules.base import SustainedConditionTracker, VisionModule, register_module
from modules.driver_attention import (
    LEFT_EYE,
    POSE_LANDMARK_IDS,
    RIGHT_EYE,
    GlanceCounter,
    PerclosTracker,
    estimate_head_pose_deg,
    eye_aspect_ratio,
)

# MediaPipe Face Mesh iris landmark indices (only populated when refine_landmarks=True).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Approximate ratio-to-degrees scale for eye-in-head rotation. Not physically
# calibrated (there's no rig here to derive it precisely); the systematic
# scale/offset error this introduces is absorbed by GazeCalibrator's baseline
# subtraction, since we calibrate the *combined* gaze signal, not this in isolation.
EYE_GAZE_SCALE_DEG = 40.0


def require_mps() -> None:
    """Raise a clear error if this machine can't run the mac_max profile.

    Called once at startup (from main.py) before anything else in the mac_max
    path — refuses to run rather than silently falling back to CPU.
    """
    import torch

    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise RuntimeError(
            "profile 'mac_max' requires Apple Silicon + a torch build with MPS support, "
            "which is not available on this machine (torch.backends.mps.is_available() "
            "is False). Use profile: pi instead, or run this on a Mac with Apple Silicon."
        )


class RollingMedianFilter:
    """Smooths a per-frame numeric signal with a short rolling median.

    A single flickered detector reading (a landmark jitter, a missed YOLO frame)
    shouldn't be able to flip a state on its own; the median filter absorbs it.
    Also usable for booleans (as 0.0/1.0) as a majority-vote-over-N-frames filter.
    """

    def __init__(self, size: int = 5) -> None:
        self._buffer: deque[float] = deque(maxlen=size)

    def update(self, value: float) -> float:
        """Push one raw value and return the current median of the last `size` values."""
        self._buffer.append(float(value))
        return statistics.median(self._buffer)


class GazeCalibrator:
    """Calibrates a personal EAR threshold AND a neutral forward-gaze baseline together.

    One combined ~3-second "look at the screen center" prompt at startup captures
    both: your resting open-eye EAR (for the drowsiness threshold) and the total
    gaze angle (head + eye-in-head) that corresponds to looking straight ahead at
    this camera's mounting position. Persisted separately from the Pi profile's
    data/calibration.json — the two profiles never share calibration state.
    """

    def __init__(
        self,
        calibration_path: Path,
        duration_s: float,
        ear_threshold_ratio: float,
        ear_threshold_fallback: float,
        force_recalibrate: bool,
    ) -> None:
        self._path = calibration_path
        self._duration_s = duration_s
        self._ear_threshold_ratio = ear_threshold_ratio
        self._ear_threshold_fallback = ear_threshold_fallback
        self._ear_samples: list[float] = []
        self._gaze_yaw_samples: list[float] = []
        self._gaze_pitch_samples: list[float] = []
        self._start_time: Optional[float] = None
        self._last_now: float = 0.0
        self._ear_threshold: Optional[float] = None
        self._baseline_gaze_yaw: Optional[float] = None
        self._baseline_gaze_pitch: Optional[float] = None

        if not force_recalibrate:
            self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._ear_threshold = float(data["ear_threshold"])
            self._baseline_gaze_yaw = float(data["baseline_gaze_yaw"])
            self._baseline_gaze_pitch = float(data["baseline_gaze_pitch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            self._ear_threshold = None
            self._baseline_gaze_yaw = None
            self._baseline_gaze_pitch = None

    @property
    def is_calibrated(self) -> bool:
        """Whether a personal threshold + gaze baseline (loaded or freshly computed) is in effect."""
        return self._ear_threshold is not None

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold if self._ear_threshold is not None else self._ear_threshold_fallback

    @property
    def baseline_gaze_yaw(self) -> float:
        return self._baseline_gaze_yaw if self._baseline_gaze_yaw is not None else 0.0

    @property
    def baseline_gaze_pitch(self) -> float:
        return self._baseline_gaze_pitch if self._baseline_gaze_pitch is not None else 0.0

    @property
    def seconds_remaining(self) -> float:
        """Seconds left in the calibration prompt, for the on-screen countdown."""
        if self.is_calibrated:
            return 0.0
        if self._start_time is None:
            return self._duration_s
        return max(0.0, self._duration_s - (self._last_now - self._start_time))

    def observe(self, ear: float, gaze_yaw: Optional[float], gaze_pitch: Optional[float], now: float) -> None:
        """Feed one frame's readings during the calibration window; no-op once calibrated."""
        if self.is_calibrated:
            return
        if self._start_time is None:
            self._start_time = now
        self._last_now = now
        self._ear_samples.append(ear)
        if gaze_yaw is not None:
            self._gaze_yaw_samples.append(gaze_yaw)
        if gaze_pitch is not None:
            self._gaze_pitch_samples.append(gaze_pitch)

        if now - self._start_time >= self._duration_s and self._ear_samples:
            baseline_ear = statistics.fmean(self._ear_samples)
            self._ear_threshold = baseline_ear * self._ear_threshold_ratio
            self._baseline_gaze_yaw = statistics.fmean(self._gaze_yaw_samples) if self._gaze_yaw_samples else 0.0
            self._baseline_gaze_pitch = (
                statistics.fmean(self._gaze_pitch_samples) if self._gaze_pitch_samples else 0.0
            )
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "baseline_ear": baseline_ear,
                        "ear_threshold": self._ear_threshold,
                        "baseline_gaze_yaw": self._baseline_gaze_yaw,
                        "baseline_gaze_pitch": self._baseline_gaze_pitch,
                    },
                    indent=2,
                )
            )


class CumulativeConditionTracker:
    """Tracks actual cumulative seconds a condition has been true within a rolling window.

    This is deliberately distinct from `PerclosTracker` (imported from the pi
    profile), which returns a *fraction* of elapsed history — correct for a
    percentage like PERCLOS, but wrong for a fixed time *budget* like
    excessive_glancing's 4.5s-within-12s threshold: multiplying that fraction by
    the full window size overestimates cumulative seconds during the ramp-up
    period before a full window's worth of history has accumulated (e.g. after
    only 2 seconds of driving, a fraction of 1.0 does NOT mean 12 off-road
    seconds have elapsed — it means all 2 of the 2 elapsed seconds were off-road).
    This class returns the raw cumulative seconds directly, sidestepping that.
    """

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._samples: deque[tuple[float, bool]] = deque()

    def update(self, condition: bool, now: float) -> float:
        """Record the current state and return cumulative seconds `condition` was
        True across the retained (rolling-window) samples."""
        self._samples.append((now, condition))
        cutoff = now - self._window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if len(self._samples) < 2:
            return 0.0

        cumulative = 0.0
        prev_t, prev_condition = self._samples[0]
        for t, condition_state in list(self._samples)[1:]:
            if prev_condition:
                cumulative += t - prev_t
            prev_t, prev_condition = t, condition_state
        return cumulative


class BlinkRateTracker:
    """Blinks/minute over a rolling window, compared against the user's own session baseline.

    A sustained drop or rise in blink rate relative to how the user normally
    blinks is itself a fatigue signal, independent of any single closure's
    duration. See config.yaml's citations for the specific rate thresholds.
    """

    def __init__(
        self,
        window_s: float,
        baseline_seconds: float,
        low_per_min: float,
        high_per_min: float,
        confirm_s: float,
    ) -> None:
        self._window_s = window_s
        self._baseline_seconds = baseline_seconds
        self._low_per_min = low_per_min
        self._high_per_min = high_per_min
        self._blink_times: deque[float] = deque()
        self._baseline_blink_count = 0
        self._session_start: Optional[float] = None
        self._baseline_rate: Optional[float] = None
        self._out_of_range_tracker = SustainedConditionTracker(confirm_s)
        self._last_rate = 0.0

    def record_blink(self, now: float) -> None:
        """Call once per completed short (non-microsleep) eye closure."""
        if self._session_start is None:
            self._session_start = now
        self._blink_times.append(now)
        if self._baseline_rate is None:
            self._baseline_blink_count += 1

    def update(self, now: float) -> Optional[str]:
        """Call once per frame. Returns "fatigue_pattern" the moment it fires, else None."""
        if self._session_start is None:
            return None

        if self._baseline_rate is None:
            if now - self._session_start >= self._baseline_seconds:
                self._baseline_rate = self._baseline_blink_count / (self._baseline_seconds / 60.0)
            return None  # no anomaly detection until the baseline is established

        cutoff = now - self._window_s
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        window_minutes = min(self._window_s, now - self._session_start) / 60.0
        rate = len(self._blink_times) / window_minutes if window_minutes > 0 else 0.0
        self._last_rate = rate

        out_of_range = rate < self._low_per_min or rate > self._high_per_min
        fired = self._out_of_range_tracker.update(out_of_range, now)
        return "fatigue_pattern" if fired is not None else None

    @property
    def rate_per_min(self) -> float:
        """Current blinks/min over the rolling window, for the debug overlay."""
        return self._last_rate

    @property
    def baseline_rate_per_min(self) -> Optional[float]:
        """The user's own first-baseline_seconds blink rate, once established."""
        return self._baseline_rate


@register_module("driver_attention_max")
class DriverAttentionMaxModule(VisionModule):
    """mac_max profile: gaze-based, confidence-stacked, temporally-smoothed driver attention."""

    name = "driver_attention_max"

    def __init__(self, config: dict) -> None:
        """Read mac_max thresholds from config and eagerly initialize MediaPipe."""
        super().__init__(config)
        self._device = config.get("device", "mps")

        self._calibrator = GazeCalibrator(
            calibration_path=Path(config.get("calibration_path", "data/gaze_calibration.json")),
            duration_s=float(config.get("calibration_seconds", 3.0)),
            ear_threshold_ratio=float(config.get("ear_threshold_ratio", 0.75)),
            ear_threshold_fallback=float(config.get("ear_threshold_fallback", 0.20)),
            force_recalibrate=bool(config.get("recalibrate", False)),
        )

        # --- drowsiness tiers ---
        self._blink_ignore_max_s = float(config.get("blink_ignore_max_s", 0.5))
        self._microsleep_tracker = SustainedConditionTracker(float(config.get("microsleep_seconds", 0.5)))
        self._drowsy_tracker = SustainedConditionTracker(float(config.get("drowsiness_seconds", 1.5)))
        self._perclos_tracker = PerclosTracker(float(config.get("perclos_window_s", 60.0)))
        self._perclos_fatigue_threshold = float(config.get("perclos_fatigue_threshold", 0.15))
        self._perclos_escalate_threshold = float(config.get("perclos_escalate_threshold", 0.25))
        self._blink_rate_tracker = BlinkRateTracker(
            window_s=float(config.get("blink_rate_window_s", 120.0)),
            baseline_seconds=float(config.get("blink_rate_baseline_seconds", 300.0)),
            low_per_min=float(config.get("blink_rate_low_per_min", 8.0)),
            high_per_min=float(config.get("blink_rate_high_per_min", 30.0)),
            confirm_s=float(config.get("fatigue_pattern_confirm_s", 30.0)),
        )
        self._fatigue_warned = False
        self._fatigue_critical_warned = False

        # --- gaze / eyes-off-road ---
        self._gaze_yaw_threshold_deg = float(config.get("gaze_yaw_threshold_deg", 20.0))
        self._gaze_pitch_down_threshold_deg = float(config.get("gaze_pitch_down_threshold_deg", -15.0))
        self._off_road_tracker = SustainedConditionTracker(float(config.get("gaze_off_road_seconds", 2.0)))
        self._excessive_glance_window_s = float(config.get("excessive_glance_window_s", 12.0))
        self._excessive_glance_budget_s = float(config.get("excessive_glance_budget_s", 4.5))
        self._excessive_glance_tracker = CumulativeConditionTracker(self._excessive_glance_window_s)
        self._excessive_glancing_warned = False
        self._eyes_down_tracker = SustainedConditionTracker(float(config.get("eyes_down_seconds", 1.5)))

        # --- object detection: full frame, every frame, lazily loaded ---
        self._detector_weights = config.get("detector_weights", "yolo11x.pt")
        self._detector_confidence_threshold = float(config.get("detector_confidence_threshold", 0.4))
        self._detector_class_map: dict[str, str] = dict(config.get("detector_classes", {}))
        self._enable_hands = bool(config.get("enable_hands", True))
        self._detector_model = None
        self._detector_class_ids: dict[int, str] = {}
        self._hands = None
        self._object_detection_ready = False
        self._object_detection_error: Optional[str] = None
        self._last_phone_boxes: list[Box] = []
        self._last_eating_boxes: list[Box] = []
        self._last_hand_boxes: list[Box] = []
        self._last_overlay_boxes: list[tuple[Box, str, float]] = []
        self._phone_visible_prev = False
        self._hand_overlaps_phone_prev = False
        self._hand_overlaps_phone = False
        self._pending_phone_alone_edge = False
        self._pending_phone_in_hand_edge = False

        # --- phone / object fusion (confidence stacking) ---
        self._hand_phone_iou_threshold = float(config.get("hand_phone_iou_threshold", 0.2))
        self._phone_alone_confidence = float(config.get("phone_alone_confidence", 0.3))
        self._phone_in_hand_confidence = float(config.get("phone_in_hand_confidence", 0.7))
        self._phone_use_distracted_confidence = float(config.get("phone_use_distracted_confidence", 0.95))
        self._phone_use_distracted_tracker = SustainedConditionTracker(
            float(config.get("phone_use_distracted_seconds", 1.5))
        )
        self._hand_face_proximity_window_s = float(config.get("hand_face_proximity_window_s", 30.0))
        self._hand_face_distance_ratio = float(config.get("hand_face_distance_ratio", 0.6))
        self._hand_face_counter = GlanceCounter(
            window_s=self._hand_face_proximity_window_s,
            min_count=int(config.get("hand_face_proximity_min_count", 2)),
        )

        # --- temporal smoothing ---
        filter_size = int(config.get("median_filter_frames", 5))
        self._ear_filter = RollingMedianFilter(filter_size)
        self._gaze_yaw_filter = RollingMedianFilter(filter_size)
        self._gaze_pitch_filter = RollingMedianFilter(filter_size)
        self._phone_visible_filter = RollingMedianFilter(filter_size)
        self._hand_overlap_filter = RollingMedianFilter(filter_size)

        # --- frontier VLM judge (async second opinion; see core/vlm_judge.py) ---
        vlm_config = config.get("vlm") or {}
        self._vlm: Optional[VLMJudge] = None
        self._vlm_confirm = VLMConfirmTracker(
            min_confidence=float(vlm_config.get("min_confidence", 0.7)),
            needed=int(vlm_config.get("confirm_judgments", 2)),
        )
        self._vlm_last_seq = 0
        if vlm_config.get("enabled", False):
            self._vlm = VLMJudge(vlm_config)
            if vlm_is_local(vlm_config):
                privacy_note = "frames stay ON-DEVICE (local model; nothing is transmitted)."
            else:
                privacy_note = (
                    "NOTE: frames are transmitted in-memory to the provider's API for "
                    "classification (still never written to disk). Set "
                    "mac_max.driver_attention.vlm.enabled: false to keep everything on-device."
                )
            print(
                "driver_attention_max: VLM judge ENABLED "
                f"(provider={self._vlm.provider}, model={self._vlm.model}, "
                f"every {float(vlm_config.get('interval_s', 4.0)):.0f}s). " + privacy_note
            )

        # --- MediaPipe: eager, every frame (no stride -- Mac can afford it) ---
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # required for iris landmarks 468-477
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("driver_attention_max: MediaPipe Face Mesh ready on device=cpu (XNNPACK delegate)")
        if self._enable_hands:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            print("driver_attention_max: MediaPipe Hands ready on device=cpu (XNNPACK delegate)")

        # --- live state, for annotate()/annotate_debug()/is_alert_active() ---
        self._last_face_proto = None
        self._last_ear = 0.0
        self._last_gaze_yaw = 0.0
        self._last_gaze_pitch = 0.0
        self._last_perclos = 0.0
        self._last_excessive_glance_seconds = 0.0
        self._blink_count = 0
        self._eyes_closed_prev = False
        self._closure_start: Optional[float] = None
        self._microsleep_active = False
        self._drowsy_active = False
        self._fatigue_active = False
        self._fatigue_critical_active = False
        self._off_road_active = False
        self._excessive_glancing_active = False
        self._eyes_down_active = False
        self._phone_use_distracted_active = False

    def _ensure_object_detectors_loaded(self) -> None:
        """Lazily load the detector (YOLO11-x or RT-DETR-l, per config) on the first frame."""
        if self._detector_model is None:
            self._detector_model = get_yolo_model(self._detector_weights, device=self._device)
            self._detector_class_ids = {
                class_id: self._detector_class_map[name]
                for class_id, name in self._detector_model.names.items()
                if name in self._detector_class_map
            }
            if not self._detector_class_ids:
                raise RuntimeError(
                    f"none of detector_classes {list(self._detector_class_map)} matched a class name "
                    f"the '{self._detector_weights}' model knows about — check config.yaml spelling"
                )
        if not self._object_detection_ready:
            self._object_detection_ready = True
            print(
                f"driver_attention_max: object detection ready "
                f"(weights={self._detector_weights}, device={self._device}, "
                f"classes={list(self._detector_class_map)}, hands={self._enable_hands})"
            )

    def _update_object_detections(self, frame: np.ndarray) -> None:
        """Run the detector (+ optional Hands) on the FULL frame, every frame."""
        if self._object_detection_error is not None:
            return
        try:
            self._ensure_object_detectors_loaded()
            self._run_object_detections(frame)
        except Exception as exc:  # noqa: BLE001 - never crash the video loop
            self._object_detection_error = str(exc)
            print(f"driver_attention_max: object detection disabled after an error: {exc}")

    def _run_object_detections(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        results = self._detector_model.predict(
            frame,
            verbose=False,
            classes=list(self._detector_class_ids.keys()),
            conf=self._detector_confidence_threshold,
            device=self._device,
        )
        phone_boxes: list[Box] = []
        eating_boxes: list[Box] = []
        overlay_boxes: list[tuple[Box, str, float]] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                event_type = self._detector_class_ids.get(class_id)
                if event_type is None:
                    continue
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                overlay_boxes.append((xyxy, self._detector_model.names[class_id], confidence))
                if event_type == "phone_detected":
                    phone_boxes.append(xyxy)
                elif event_type == "eating_drinking":
                    eating_boxes.append(xyxy)

        hand_boxes: list[Box] = []
        if self._hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self._hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    xs = [lm.x * w for lm in hand_landmarks.landmark]
                    ys = [lm.y * h for lm in hand_landmarks.landmark]
                    hand_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

        self._last_phone_boxes = phone_boxes
        self._last_eating_boxes = eating_boxes
        self._last_hand_boxes = hand_boxes
        self._last_overlay_boxes = overlay_boxes

        phone_now_visible = self._phone_visible_filter.update(1.0 if phone_boxes else 0.0) >= 0.5
        hand_overlaps_phone_raw = any(
            iou(hand_box, phone_box) > self._hand_phone_iou_threshold
            for hand_box in hand_boxes
            for phone_box in phone_boxes
        )
        hand_overlaps_phone_now = self._hand_overlap_filter.update(1.0 if hand_overlaps_phone_raw else 0.0) >= 0.5
        self._hand_overlaps_phone = hand_overlaps_phone_now

        if phone_now_visible and not self._phone_visible_prev and not hand_overlaps_phone_now:
            self._pending_phone_alone_edge = True
            print("driver_attention_max: phone detected in frame")
        if hand_overlaps_phone_now and not self._hand_overlaps_phone_prev:
            self._pending_phone_in_hand_edge = True
        self._phone_visible_prev = phone_now_visible
        self._hand_overlaps_phone_prev = hand_overlaps_phone_now

    def _compute_gaze_components(self, points: list[tuple[float, float]]) -> tuple[float, float]:
        """Eye-in-head deviation (yaw, pitch) in pseudo-degrees, from iris position
        relative to each eye's own socket center and size, averaged across both eyes."""

        def eye_component(eye_indices: list[int], iris_index: int) -> tuple[float, float]:
            eye_points = [points[i] for i in eye_indices]
            xs = [p[0] for p in eye_points]
            ys = [p[1] for p in eye_points]
            center_x, center_y = sum(xs) / len(xs), sum(ys) / len(ys)
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            iris = points[iris_index]
            h_ratio = (iris[0] - center_x) / width if width > 0 else 0.0
            v_ratio = (iris[1] - center_y) / height if height > 0 else 0.0
            return h_ratio, v_ratio

        right_h, right_v = eye_component(RIGHT_EYE, RIGHT_IRIS_CENTER)
        left_h, left_v = eye_component(LEFT_EYE, LEFT_IRIS_CENTER)
        h_ratio = (right_h + left_h) / 2.0
        v_ratio = (right_v + left_v) / 2.0
        # v_ratio is POSITIVE when the iris sits toward the LOWER eyelid (larger
        # image y = physically looking down) -- but estimate_head_pose_deg's own
        # verified convention (its docstring, and tests/test_head_pose_geometry.py)
        # is NEGATIVE = down, and gaze_pitch_down_threshold_deg above is negative
        # too. Un-negated, "head level, eyes cast down" would ADD a positive
        # value onto head_pitch and partially CANCEL it instead of reinforcing
        # it -- confirmed by direct comparison against estimate_head_pose_deg's
        # sign for an equivalent physical rotation, not assumed (see
        # core/eye_signal.py's eye_gaze_offset_deg, the pi-profile port of this
        # same fusion, and tests/test_eye_gaze.py, which found and fixed the
        # identical issue there). h_ratio needs no such flip: image-right
        # already matches this module's yaw convention directly.
        return h_ratio * EYE_GAZE_SCALE_DEG, -v_ratio * EYE_GAZE_SCALE_DEG

    def _hand_near_face(self, hand_boxes: list[Box], face_bbox: Box) -> bool:
        """Whether any hand box is within `hand_face_distance_ratio` of the face box."""
        fx1, fy1, fx2, fy2 = face_bbox
        face_center = ((fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0)
        face_size = ((fx2 - fx1) + (fy2 - fy1)) / 2.0
        if face_size <= 0:
            return False
        radius = face_size * (1.0 + self._hand_face_distance_ratio)
        for hx1, hy1, hx2, hy2 in hand_boxes:
            hand_center = ((hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0)
            distance = math.hypot(hand_center[0] - face_center[0], hand_center[1] - face_center[1])
            if distance < radius:
                return True
        return False

    def _poll_vlm(self, events: list[Event]) -> None:
        """Publish any new VLM verdict as events (log-only judgment + confirmed alerts)."""
        if self._vlm is None:
            return
        verdict = self._vlm.latest_verdict()
        if verdict is None or verdict.seq == self._vlm_last_seq:
            return
        self._vlm_last_seq = verdict.seq
        # Every judgment is logged (analyzable CSV trail), but never alerts on its own.
        events.append(
            Event.now(
                "vlm_judgment", 0.0, verdict.confidence, self.name, f"vlm:{verdict.state} — {verdict.evidence}"
            )
        )
        confirmed_state = self._vlm_confirm.update(verdict)
        if confirmed_state is not None:
            events.append(
                Event.now(
                    VLM_EVENT_BY_STATE[confirmed_state],
                    0.0,
                    verdict.confidence,
                    self.name,
                    f"vlm confirmed x{self._vlm_confirm.needed}: {verdict.evidence}",
                )
            )

    def process(self, frame: np.ndarray) -> list[Event]:
        """Run Face Mesh + gaze + (every-frame) object detection; return fired events."""
        now = time.monotonic()
        events: list[Event] = []
        h, w = frame.shape[:2]

        self._update_object_detections(frame)

        # VLM tier: offer this frame (self-throttled, async) and fold in any new verdict.
        # Runs on BOTH the face-visible and face-lost paths -- the judge sees the whole
        # scene, so it still works when Face Mesh has nothing.
        if self._vlm is not None:
            self._vlm.submit(frame)
        self._poll_vlm(events)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._last_face_proto = None
            # Reset the trackers, not just the flags. A tracker keeps its start
            # time until it sees the condition go false, so clearing only the
            # flags let the whole absence count toward the first reading after
            # the face returned: one closed frame then fired drowsiness with a
            # "duration" that was really time the face was not seen at all.
            # driver_attention.py's face-lost path does the same.
            for tracker in (
                self._microsleep_tracker,
                self._drowsy_tracker,
                self._off_road_tracker,
                self._eyes_down_tracker,
                self._phone_use_distracted_tracker,
            ):
                tracker.update(False, now)
            self._eyes_closed_prev = False
            self._closure_start = None
            self._microsleep_active = False
            self._drowsy_active = False
            self._off_road_active = False
            self._eyes_down_active = False
            self._phone_use_distracted_active = False
            return events

        face_proto = results.multi_face_landmarks[0]
        self._last_face_proto = face_proto
        landmarks = face_proto.landmark
        points = [(lm.x * w, lm.y * h) for lm in landmarks]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        face_bbox: Box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

        # --- raw signals ---
        left_ear = eye_aspect_ratio(*(points[i] for i in LEFT_EYE))
        right_ear = eye_aspect_ratio(*(points[i] for i in RIGHT_EYE))
        raw_ear = (left_ear + right_ear) / 2.0

        image_points = np.array(
            [
                points[POSE_LANDMARK_IDS[key]]
                for key in ("nose_tip", "chin", "left_eye_corner", "right_eye_corner", "left_mouth", "right_mouth")
            ],
            dtype=np.float64,
        )
        pose = estimate_head_pose_deg(image_points, (h, w))
        if pose is not None:
            head_yaw, head_pitch = pose
            eye_yaw, eye_pitch = self._compute_gaze_components(points)
            raw_gaze_yaw: Optional[float] = head_yaw + eye_yaw
            raw_gaze_pitch: Optional[float] = head_pitch + eye_pitch
        else:
            raw_gaze_yaw = raw_gaze_pitch = None

        self._calibrator.observe(raw_ear, raw_gaze_yaw, raw_gaze_pitch, now)
        if not self._calibrator.is_calibrated:
            return events  # still calibrating -- annotate() shows the prompt

        # --- temporal smoothing ---
        ear = self._ear_filter.update(raw_ear)
        self._last_ear = ear
        if raw_gaze_yaw is not None:
            gaze_yaw = self._gaze_yaw_filter.update(raw_gaze_yaw - self._calibrator.baseline_gaze_yaw)
            gaze_pitch = self._gaze_pitch_filter.update(raw_gaze_pitch - self._calibrator.baseline_gaze_pitch)
            self._last_gaze_yaw, self._last_gaze_pitch = gaze_yaw, gaze_pitch
        else:
            gaze_yaw, gaze_pitch = self._last_gaze_yaw, self._last_gaze_pitch

        # --- drowsiness: blink vs. microsleep vs. drowsiness, PERCLOS, blink rate ---
        eyes_closed = ear < self._calibrator.ear_threshold
        if eyes_closed and not self._eyes_closed_prev:
            self._closure_start = now
        if not eyes_closed and self._eyes_closed_prev and self._closure_start is not None:
            closure_duration = now - self._closure_start
            if closure_duration < self._blink_ignore_max_s:
                self._blink_rate_tracker.record_blink(now)
                self._blink_count += 1
            self._closure_start = None
        self._eyes_closed_prev = eyes_closed

        fired = self._microsleep_tracker.update(eyes_closed, now)
        self._microsleep_active = self._microsleep_tracker.is_active
        if fired is not None:
            events.append(Event.now("microsleep_warning", fired, 0.6, self.name, "sustained_eye_closure_0.5-1.5s"))

        fired = self._drowsy_tracker.update(eyes_closed, now)
        self._drowsy_active = self._drowsy_tracker.is_active
        if fired is not None:
            events.append(Event.now("drowsiness", fired, 0.9, self.name, "sustained_eye_closure_gt_1.5s"))

        perclos = self._perclos_tracker.update(eyes_closed, now)
        self._last_perclos = perclos
        self._fatigue_active = perclos > self._perclos_fatigue_threshold
        self._fatigue_critical_active = perclos > self._perclos_escalate_threshold
        if self._fatigue_critical_active and not self._fatigue_critical_warned:
            events.append(Event.now("fatigue_critical", 0.0, perclos, self.name, "perclos_gt_0.25"))
            self._fatigue_critical_warned = True
        elif perclos <= self._perclos_escalate_threshold * 0.8:
            self._fatigue_critical_warned = False
        if self._fatigue_active and not self._fatigue_warned:
            events.append(Event.now("fatigue", 0.0, perclos, self.name, "perclos_gt_0.15"))
            self._fatigue_warned = True
        elif perclos <= self._perclos_fatigue_threshold * 0.8:
            self._fatigue_warned = False

        fatigue_pattern_event = self._blink_rate_tracker.update(now)
        if fatigue_pattern_event:
            events.append(Event.now("fatigue_pattern", 0.0, 0.5, self.name, "blink_rate_anomaly"))

        # --- gaze / eyes-off-road ---
        off_road_condition = abs(gaze_yaw) > self._gaze_yaw_threshold_deg
        fired = self._off_road_tracker.update(off_road_condition, now)
        self._off_road_active = self._off_road_tracker.is_active
        if fired is not None:
            events.append(Event.now("eyes_off_road", fired, min(1.0, abs(gaze_yaw) / 90.0), self.name, "gaze_yaw"))

        cumulative_seconds = self._excessive_glance_tracker.update(off_road_condition, now)
        self._last_excessive_glance_seconds = cumulative_seconds
        self._excessive_glancing_active = cumulative_seconds > self._excessive_glance_budget_s
        if self._excessive_glancing_active and not self._excessive_glancing_warned:
            events.append(
                Event.now(
                    "excessive_glancing",
                    0.0,
                    min(1.0, cumulative_seconds / self._excessive_glance_window_s),
                    self.name,
                    "cumulative_gaze_off_12s_window",
                )
            )
            self._excessive_glancing_warned = True
        elif cumulative_seconds < self._excessive_glance_budget_s * 0.7:
            self._excessive_glancing_warned = False

        eyes_down_condition = gaze_pitch < self._gaze_pitch_down_threshold_deg
        fired = self._eyes_down_tracker.update(eyes_down_condition, now)
        self._eyes_down_active = self._eyes_down_tracker.is_active
        if fired is not None:
            events.append(
                Event.now("eyes_down", fired, min(1.0, abs(gaze_pitch) / 90.0), self.name, "gaze_pitch_down")
            )

        eyes_off_road_now = off_road_condition or eyes_down_condition

        # --- phone / object fusion: confidence stacking ---
        phone_visible = bool(self._last_phone_boxes)
        eating_visible = bool(self._last_eating_boxes)

        phone_use_distracted_condition = phone_visible and self._hand_overlaps_phone and eyes_off_road_now
        fired = self._phone_use_distracted_tracker.update(phone_use_distracted_condition, now)
        self._phone_use_distracted_active = self._phone_use_distracted_tracker.is_active
        if fired is not None:
            events.append(
                Event.now(
                    "phone_use_while_distracted",
                    fired,
                    self._phone_use_distracted_confidence,
                    self.name,
                    "phone + hand_overlap + gaze_off",
                )
            )

        if self._pending_phone_alone_edge:
            self._pending_phone_alone_edge = False
            events.append(
                Event.now("phone_detected", 0.0, self._phone_alone_confidence, self.name, "phone_only_eyes_forward")
            )
        if self._pending_phone_in_hand_edge:
            self._pending_phone_in_hand_edge = False
            events.append(
                Event.now("phone_in_hand", 0.0, self._phone_in_hand_confidence, self.name, "phone + hand_overlap")
            )

        # --- eating/drinking: food/drink class + hand overlap + repeated hand-to-face motion ---
        hand_overlaps_food = any(
            iou(hand_box, food_box) > self._hand_phone_iou_threshold
            for hand_box in self._last_hand_boxes
            for food_box in self._last_eating_boxes
        )
        hand_near_face = self._hand_near_face(self._last_hand_boxes, face_bbox)
        approach_fired = self._hand_face_counter.update(hand_near_face, now)
        if approach_fired and eating_visible and hand_overlaps_food:
            events.append(
                Event.now(
                    "eating_drinking", 0.0, 0.6, self.name, "food + hand_overlap + repeated_hand_to_face_motion"
                )
            )

        return events

    def _resolve_state(self) -> str:
        """The single fused driver state: ATTENTIVE / DISTRACTED / DROWSY / PHONE_USE / FATIGUED."""
        if not self._calibrator.is_calibrated:
            return "CALIBRATING"
        if self._drowsy_active:
            return "DROWSY"
        if self._phone_use_distracted_active:
            return "PHONE_USE"
        if self._vlm_confirm.active_state is not None:
            return f"VLM:{self._vlm_confirm.active_state.upper()}"
        if self._fatigue_critical_active or self._fatigue_active:
            return "FATIGUED"
        if self._off_road_active or self._eyes_down_active or self._excessive_glancing_active:
            return "DISTRACTED"
        return "ATTENTIVE"

    def get_telemetry_snapshot(self) -> Optional[dict]:
        """Raw per-frame signal snapshot, reusing the same schema as the Pi profile's telemetry."""
        return {
            "ear": self._last_ear,
            "yaw_deg": self._last_gaze_yaw,
            "pitch_deg": self._last_gaze_pitch,
            "perclos": self._last_perclos,
            "risk_score": 1.0 if self._drowsy_active else self._last_perclos,
            "phone_visible": bool(self._last_phone_boxes),
            "hand_count": len(self._last_hand_boxes),
            "eating_visible": bool(self._last_eating_boxes),
        }

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Minimal overlay: face mesh + fused state label."""
        if self._last_face_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=self._last_face_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        if not self._calibrator.is_calibrated:
            self._draw_calibration_prompt(frame)
        else:
            cv2.putText(
                frame, f"State: {self._resolve_state()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        return frame

    def _draw_calibration_prompt(self, frame: np.ndarray) -> None:
        """Big, unmissable "look at the screen center" countdown during calibration."""
        h, w = frame.shape[:2]
        text = "LOOK AT THE SCREEN CENTER"
        cv2.putText(
            frame, text, (max(10, w // 2 - 260), h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Calibrating... {self._calibrator.seconds_remaining:.1f}s",
            (max(10, w // 2 - 150), h // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
        """Full tuning overlay: gaze arrow, on/off-road status, EAR/blinks/PERCLOS, YOLO boxes
        with confidence, hand-phone IoU, fused state, and a rolling 12s glance-budget bar."""
        frame = self.annotate(frame)
        if not self._calibrator.is_calibrated:
            return frame

        h, w = frame.shape[:2]
        y = 55
        cv2.putText(
            frame,
            f"EAR: {self._last_ear:.2f}  Blinks: {self._blink_count}  "
            f"Rate: {self._blink_rate_tracker.rate_per_min:.1f}/min",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        y += 25
        cv2.putText(
            frame, f"PERCLOS: {self._last_perclos * 100:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )
        y += 25
        on_road = not (self._off_road_active or self._eyes_down_active)
        status_color = (0, 200, 0) if on_road else (0, 0, 255)
        cv2.putText(
            frame,
            f"Gaze: yaw={self._last_gaze_yaw:.1f} pitch={self._last_gaze_pitch:.1f} "
            f"[{'ON-ROAD' if on_road else 'OFF-ROAD'}]",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
        )

        # Gaze vector arrow, drawn from each eye's approximate center.
        if self._last_face_proto is not None:
            landmarks = self._last_face_proto.landmark
            for eye_indices in (LEFT_EYE, RIGHT_EYE):
                xs = [landmarks[i].x * w for i in eye_indices]
                ys = [landmarks[i].y * h for i in eye_indices]
                eye_center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
                arrow_len = 40
                dx = int(arrow_len * math.sin(math.radians(self._last_gaze_yaw)))
                dy = int(arrow_len * math.sin(math.radians(-self._last_gaze_pitch)))
                cv2.arrowedLine(frame, eye_center, (eye_center[0] + dx, eye_center[1] + dy), status_color, 2, tipLength=0.35)

        # YOLO boxes with confidence.
        for (x1, y1, x2, y2), label, conf in self._last_overlay_boxes:
            is_hot_phone = label == "cell phone" and self._phone_use_distracted_active
            color = (0, 0, 255) if is_hot_phone else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        for x1, y1, x2, y2 in self._last_hand_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Hand-phone IoU, when nonzero.
        max_iou = 0.0
        for hand_box in self._last_hand_boxes:
            for phone_box in self._last_phone_boxes:
                max_iou = max(max_iou, iou(hand_box, phone_box))
        if max_iou > 0.0:
            y += 25
            cv2.putText(
                frame, f"Hand-phone IoU: {max_iou:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2
            )

        y += 30
        cv2.putText(
            frame, f"State: {self._resolve_state()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )

        # VLM judge status line
        if self._vlm is not None:
            y += 25
            if self._vlm.disabled_reason is not None:
                vlm_line, vlm_color = "VLM: OFF (see terminal)", (0, 0, 255)
            else:
                verdict = self._vlm.latest_verdict()
                if verdict is None:
                    vlm_line, vlm_color = "VLM: warming up...", (140, 160, 160)
                else:
                    age = time.monotonic() - verdict.at
                    danger = verdict.state != "attentive"
                    vlm_color = (0, 165, 255) if danger else (0, 200, 0)
                    vlm_line = f"VLM: {verdict.state.upper()} {verdict.confidence:.2f} ({age:.0f}s ago)"
            cv2.putText(frame, vlm_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vlm_color, 2)

        self._draw_glance_budget_bar(frame)

        if self._object_detection_error is not None:
            cv2.putText(
                frame,
                "object detection: OFF (see terminal)",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
        return frame

    def _draw_glance_budget_bar(self, frame: np.ndarray) -> None:
        """Rolling 12s cumulative off-road glance budget, as a gas-gauge-style bar."""
        used_fraction = min(1.0, self._last_excessive_glance_seconds / self._excessive_glance_budget_s)
        color = (0, 200, 0) if used_fraction < 0.6 else (0, 165, 255) if used_fraction < 1.0 else (0, 0, 255)
        x, y, w, h = 10, 200, 220, 18
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.rectangle(frame, (x, y), (x + int(w * used_fraction), y + h), color, -1)
        cv2.putText(
            frame,
            f"Glance budget: {self._last_excessive_glance_seconds:.1f}/{self._excessive_glance_budget_s:.1f}s "
            f"(12s window)",
            (x, y + h + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    def is_alert_active(self) -> bool:
        """True while any currently-active condition warrants the full-screen alert flash."""
        return (
            self._drowsy_active
            or self._fatigue_critical_active
            or self._phone_use_distracted_active
            or self._eyes_down_active
            or self._off_road_active
            or self._vlm_confirm.active_state is not None
        )
