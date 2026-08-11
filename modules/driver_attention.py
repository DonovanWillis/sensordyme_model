"""Driver attention module.

Combines three signal families, each with its own debounce so short, ordinary
behavior (blinks, mirror checks) never fires an event:

  * Drowsiness — Eye Aspect Ratio (EAR) from MediaPipe Face Mesh, evaluated
    against a per-user calibrated threshold. A sustained closure fires
    "drowsiness"; a rolling PERCLOS (% time closed over 60s) fires the softer
    "fatigue_warning" for a driver who blinks heavily but never has one long
    closure.
  * Distraction — head yaw/pitch from solvePnP on 6 canonical landmarks,
    corrected against a calibrated neutral pose (see AttentionCalibrator)
    so a camera mounted off to the side of the dashboard doesn't read
    ordinary forward driving as a head turn. A sustained sideways turn
    fires "distraction"; a sustained downward tilt (phone in lap) fires
    the separate, more sensitive "eyes_down"; several short glances within
    a rolling window fires "frequent_glancing" (texting while still
    looking up periodically).
  * Object context — YOLO11-nano (shared via core/detectors.py) run on the
    full frame on a stride, covering everything a driver's hands might be
    holding (phone, drinks, food), plus optional MediaPipe Hands, fired to
    distinguish a dashboard-mounted phone (benign, log-only "phone_detected")
    from actual phone use (eyes off the road while a phone is visible ->
    "phone_use").
  * Face lost — if the camera can't find a face at all (turned fully away,
    slumped over, obstructed), that absence is itself a risk signal rather
    than "nothing to report"; sustained loss fires "attention_lost".

On top of these atomic, independently-debounced signals sits RiskFusion: a
cross-signal reasoning layer that catches compound danger no single
threshold would (mild drowsiness *and* a glance away is worse than either
alone), producing a smoothed composite score and its own "elevated_risk" /
"high_risk" events. See RiskFusion's docstring for the full rationale.

Every frame's raw signals are also available via get_telemetry_snapshot()
for main.py to log to data/telemetry.csv — real driving sessions build the
labeled dataset needed to eventually replace these hand-tuned thresholds
with a learned model.

Signal conditioning (core/eye_signal.py) sits between the landmarks and every
threshold above, because the raw readings are noisy measurements — especially
with glasses, whose frames and glare corrupt the eyelid contour for single
frames at a time. Three layers, all arithmetic, no new inference:

  * a rolling median on EAR/yaw/pitch, so one bad frame moves nothing
    (onset of a real state change lags by ~half the window — ~0.3s at 8fps —
    which the sustain thresholds already dwarf);
  * release-grace on the condition trackers (GracedConditionTracker), so one
    bad frame can no longer reset an accumulating drowsiness clock or drop a
    live alert — the mechanism that made detection DEGRADE as the driver got
    drowsier, since marginal landmarks are exactly when noise frames appear;
  * a session-adaptive open-eye reference banded to the calibration, plus an
    iris-visibility veto (the iris landmarks were always computed and never
    read), so a threshold calibrated bare-faced still means the right thing
    the day glasses go on.
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

from core.detections import Detection, DetectionFrame
from core.detectors import Box, get_yolo_model, iou
from core.events import Event
from core.eye_signal import AdaptiveBaseline, SignalSmoother, iris_openness
from modules.base import GracedConditionTracker, SustainedConditionTracker, VisionModule, register_module

# MediaPipe Face Mesh landmark indices for the 6-point EAR eye model.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Iris-openness (eyelid gap / iris diameter, see core/eye_signal.py) above which
# the eye is unambiguously open: a closed lid physically cannot leave this much
# iris visible. Comfortably-open eyes measure ~0.7-1.1; half-lidded ~0.3-0.5.
# 0.5 is deliberately above the ambiguous band so the veto only fires on
# contradictions, never on genuinely droopy eyes.
_IRIS_CLEARLY_OPEN = 0.5

# The 6 canonical landmarks used for solvePnP head-pose estimation.
POSE_LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Generic 3D face model points (mm), matched to POSE_LANDMARK_IDS order.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -63.6, -12.5),  # chin
        (-43.3, 32.7, -26.0),  # left eye corner
        (43.3, 32.7, -26.0),  # right eye corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)

Point = tuple[float, float]


def angle_delta_deg(angle: float, baseline: float) -> float:
    """Signed difference between two angles, wrapped to (-180, 180].

    Euler angles from Rodrigues live on a circle, and with this 6-point face
    model the pitch of an ordinary upright face rides near ±180 — so plain
    subtraction produced deltas like +249° and -84° on CONSECUTIVE frames of
    an unchanged pose (observed in the device's own telemetry). Every
    threshold downstream compares the delta, so the wrap must be resolved
    here, once, rather than survived by each consumer.
    """
    delta = (angle - baseline + 180.0) % 360.0 - 180.0
    # The modulo maps +180 to -180; keep the documented half-open interval.
    return 180.0 if delta == -180.0 else delta


def circular_mean_deg(angles: list[float]) -> float:
    """Mean of angles in degrees, computed on the circle.

    fmean(+179, -179) is 0 — the exact opposite direction of both inputs.
    That is not hypothetical: the deployed device's calibration file held
    baseline_pitch = -90.3, an arithmetic mean over wrap-straddling samples,
    pointing nowhere any head ever faced.
    """
    if not angles:
        return 0.0
    sines = sum(math.sin(math.radians(a)) for a in angles)
    cosines = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sines, cosines))


def eye_aspect_ratio(p1: Point, p2: Point, p3: Point, p4: Point, p5: Point, p6: Point) -> float:
    """Compute the Eye Aspect Ratio from 6 (x, y) eye landmarks.

    Points follow the standard 6-point eye model: p1/p4 are the horizontal
    corners, p2/p3 and p5/p6 are the upper/lower eyelid pairs.
    """

    def dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    vertical = dist(p2, p6) + dist(p3, p5)
    horizontal = 2.0 * dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_head_pose_deg(image_points: np.ndarray, image_size: tuple[int, int]) -> Optional[tuple[float, float]]:
    """Estimate (yaw, pitch) in degrees from 6 canonical facial landmarks via solvePnP.

    Positive yaw = turned toward the frame's right. Negative pitch = looking down.
    Returns None if pose estimation fails to converge.
    """
    h, w = image_size
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
    else:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
    return math.degrees(yaw), math.degrees(pitch)


class AttentionCalibrator:
    """Learns a per-user EAR threshold AND a per-install neutral head pose.

    Two different things vary by install and must be calibrated rather than
    assumed:

      * Resting EAR varies by face shape, which is why a single fixed EAR
        threshold misfires across users.
      * "Looking at the road" is not yaw=0/pitch=0 unless the camera happens
        to be mounted dead-center on the dashboard facing the driver. A
        camera off to the side (common when the phone/Pi is mounted next to
        the wheel rather than in the middle) sees the driver's natural
        forward gaze at some nonzero yaw/pitch — without correcting for
        that, ordinary forward driving reads as a sustained head turn.

    Both are learned from the first `duration_s` of driving (the user should
    look at the road as they normally would) and persisted to
    `calibration_path` so restarts skip recalibration unless
    `force_recalibrate` is set (wired to `main.py --recalibrate`).
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
        self._yaw_samples: list[float] = []
        self._pitch_samples: list[float] = []
        self._start_time: Optional[float] = None
        self._baseline_ear: Optional[float] = None
        self._ear_threshold: Optional[float] = None
        self._baseline_yaw: Optional[float] = None
        self._baseline_pitch: Optional[float] = None

        if not force_recalibrate:
            self._load()

    #: Bumped when the stored numbers' MEANING changes. Version 2: pose
    #: baselines are circular means (v1 used an arithmetic mean, which for a
    #: wrap-straddling pitch produced a direction no head ever faced — the
    #: deployed device's file held baseline_pitch=-90.3 this way). A v1 file
    #: is rejected so the device recalibrates once with the fixed math,
    #: rather than keep thresholding against a corrupt baseline forever.
    FORMAT_VERSION = 2

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if int(data.get("version", 1)) != self.FORMAT_VERSION:
                return  # stale format: leave uncalibrated; a fresh window will rewrite it
            self._baseline_ear = float(data["baseline_ear"])
            self._ear_threshold = float(data["ear_threshold"])
            self._baseline_yaw = float(data["baseline_yaw"])
            self._baseline_pitch = float(data["baseline_pitch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            self._baseline_ear = None
            self._ear_threshold = None
            self._baseline_yaw = None
            self._baseline_pitch = None

    @property
    def is_calibrated(self) -> bool:
        """Whether a personal EAR threshold (loaded or freshly computed) is in effect."""
        return self._ear_threshold is not None

    @property
    def baseline_ear(self) -> Optional[float]:
        """The calibrated open-eye EAR, or None before calibration completes.

        Exposed so the session-adaptive baseline can band its estimate around
        the calibrated value rather than replace it — see AdaptiveBaseline.
        """
        return self._baseline_ear

    @property
    def ear_threshold(self) -> float:
        """The EAR threshold to use right now: personal if calibrated, else the fallback."""
        return self._ear_threshold if self._ear_threshold is not None else self._ear_threshold_fallback

    @property
    def baseline_yaw(self) -> float:
        """The yaw (degrees) that counts as "looking at the road" for this camera mount."""
        return self._baseline_yaw if self._baseline_yaw is not None else 0.0

    @property
    def baseline_pitch(self) -> float:
        """The pitch (degrees) that counts as "looking at the road" for this camera mount."""
        return self._baseline_pitch if self._baseline_pitch is not None else 0.0

    def observe(self, ear: float, yaw: Optional[float], pitch: Optional[float], now: float) -> None:
        """Feed one frame's readings during the calibration window; no-op once calibrated."""
        if self.is_calibrated:
            return
        if self._start_time is None:
            self._start_time = now
        self._ear_samples.append(ear)
        if yaw is not None:
            self._yaw_samples.append(yaw)
        if pitch is not None:
            self._pitch_samples.append(pitch)

        if now - self._start_time >= self._duration_s and self._ear_samples:
            self._baseline_ear = statistics.fmean(self._ear_samples)
            self._ear_threshold = self._baseline_ear * self._ear_threshold_ratio
            # Circular means: pose angles live on a circle, and with this face
            # model an upright face's pitch rides near ±180, where fmean points
            # the opposite way — see circular_mean_deg.
            self._baseline_yaw = circular_mean_deg(self._yaw_samples)
            self._baseline_pitch = circular_mean_deg(self._pitch_samples)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {
                        "version": self.FORMAT_VERSION,
                        "baseline_ear": self._baseline_ear,
                        "ear_threshold": self._ear_threshold,
                        "baseline_yaw": self._baseline_yaw,
                        "baseline_pitch": self._baseline_pitch,
                    },
                    indent=2,
                )
            )


class PerclosTracker:
    """Tracks the rolling fraction of elapsed time a condition (eye closure) has been true."""

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._samples: deque[tuple[float, bool]] = deque()

    def update(self, closed: bool, now: float) -> float:
        """Record the current closed/open state and return the PERCLOS fraction in [0, 1]."""
        self._samples.append((now, closed))
        cutoff = now - self._window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if len(self._samples) < 2:
            return 0.0

        closed_time = 0.0
        total_time = 0.0
        prev_t, prev_closed = self._samples[0]
        for t, closed_state in list(self._samples)[1:]:
            dt = t - prev_t
            total_time += dt
            if prev_closed:
                closed_time += dt
            prev_t, prev_closed = t, closed_state

        return closed_time / total_time if total_time > 0 else 0.0


class GlanceCounter:
    """Counts distinct off-center glance episodes within a rolling window.

    Fires once when `min_count` episodes have occurred within `window_s`,
    re-arming only once the count drops back below `min_count` (as old
    episodes age out of the window).
    """

    def __init__(self, window_s: float, min_count: int) -> None:
        self._window_s = window_s
        self._min_count = min_count
        self._episode_starts: deque[float] = deque()
        self._was_off_center = False
        self._warned = False

    def update(self, off_center: bool, now: float) -> bool:
        """Feed the current off-center boolean. Returns True the moment `min_count` is reached."""
        if off_center and not self._was_off_center:
            self._episode_starts.append(now)
        self._was_off_center = off_center

        cutoff = now - self._window_s
        while self._episode_starts and self._episode_starts[0] < cutoff:
            self._episode_starts.popleft()

        count = len(self._episode_starts)
        if count >= self._min_count:
            if not self._warned:
                self._warned = True
                return True
            return False
        self._warned = False
        return False

    @property
    def count(self) -> int:
        """Glance episodes currently within the rolling window, as of the last `update()` call.

        Exposed so RiskFusion can give partial credit for "building toward" frequent
        glancing (e.g. 2 of 4 needed) rather than only reacting once the full count fires.
        """
        return len(self._episode_starts)


class RiskFusion:
    """Combines the module's individual signals into one continuous, smoothed risk score.

    Real-world rationale: independent per-signal thresholds each catch their own
    failure mode, but they miss *compound* danger. A driver who is a little drowsy
    AND glancing away is materially more dangerous than either fact alone — a
    microsleep is far more likely to become a real hazard if it happens while
    attention is already split, and a glance away lasts longer / registers less
    urgently if reflexes are already dulled. Treating every signal in isolation
    misses that interaction entirely.

    This layer:
      * Weights each signal by real-world severity (a closed-eyes signal outweighs
        a sideways-glance signal, which outweighs snacking).
      * Adds a corroboration bonus when eye risk and gaze risk are *both*
        meaningfully elevated at once, rather than just summing them.
      * Smooths the result with an EMA so the number is stable enough to read and
        alert on, instead of flickering with every frame's noise.
      * Debounces the smoothed score itself (via a SustainedConditionTracker)
        before firing a composite alert, so a brief spike doesn't fire it.

    This is supplementary to, not a replacement for, the atomic drowsiness /
    distraction / phone_use / eyes_down events above, which keep firing (and
    logging) independently. Composite events ("elevated_risk", "high_risk") exist
    to catch the cases where no single signal crosses its own bar but the
    combination clearly should.
    """

    # Severity weights: how much each fully-active signal contributes to risk.
    # Ranked by real-world danger: eyes closed > eyes off the road > phone > food.
    EYE_WEIGHT = 0.42
    GAZE_WEIGHT = 0.30
    PHONE_WEIGHT = 0.22
    EATING_WEIGHT = 0.06
    # The face isn't a normal "signal" -- it's the absence of any signal at all. A face
    # that's been unreadable by the camera for a couple of seconds (slumped over, turned
    # fully away, camera blocked) is at least as concerning as a confirmed distraction,
    # so it gets its own dominant weight rather than being folded into eye/gaze risk.
    FACE_LOST_WEIGHT = 0.40
    # Extra risk added when eye risk and gaze risk are both already meaningful --
    # "drowsy AND looking away" is worse than the sum of its parts.
    CORROBORATION_BONUS = 0.25
    CORROBORATION_INPUT_THRESHOLD = 0.5

    def __init__(
        self,
        ema_alpha: float = 0.3,
        elevated_threshold: float = 0.4,
        high_threshold: float = 0.75,
        confirm_seconds: float = 1.0,
    ) -> None:
        self._ema_alpha = ema_alpha
        self._elevated_threshold = elevated_threshold
        self._high_threshold = high_threshold
        self._score = 0.0
        self._high_tracker = SustainedConditionTracker(confirm_seconds)
        self._elevated_warned = False

    def update(
        self,
        *,
        eye_risk: float,
        gaze_risk: float,
        phone_risk: float,
        eating_risk: float,
        face_lost_risk: float,
        now: float,
    ) -> tuple[float, Optional[str]]:
        """Combine weighted signals + corroboration bonus, smooth, and maybe fire a composite event.

        Args:
            eye_risk, gaze_risk, phone_risk, eating_risk, face_lost_risk: Each in
                [0, 1] — how much of that signal's own danger is currently present
                (1.0 = its own threshold has fully fired; partial credit below
                that, e.g. rising PERCLOS or an accumulating glance count).
            now: Current monotonic time.

        Returns:
            (smoothed_score, event_type) where event_type is "high_risk",
            "elevated_risk", or None.
        """
        base = (
            self.EYE_WEIGHT * eye_risk
            + self.GAZE_WEIGHT * gaze_risk
            + self.PHONE_WEIGHT * phone_risk
            + self.EATING_WEIGHT * eating_risk
            + self.FACE_LOST_WEIGHT * face_lost_risk
        )
        corroboration = (
            self.CORROBORATION_BONUS
            if (eye_risk >= self.CORROBORATION_INPUT_THRESHOLD and gaze_risk >= self.CORROBORATION_INPUT_THRESHOLD)
            else 0.0
        )
        raw_score = min(1.0, base + corroboration)
        self._score = self._ema_alpha * raw_score + (1.0 - self._ema_alpha) * self._score

        fired = self._high_tracker.update(self._score >= self._high_threshold, now)
        if fired is not None:
            return self._score, "high_risk"
        if self._score >= self._elevated_threshold:
            if not self._elevated_warned:
                self._elevated_warned = True
                return self._score, "elevated_risk"
        elif self._score < self._elevated_threshold * 0.7:
            self._elevated_warned = False
        return self._score, None

    @property
    def score(self) -> float:
        """The current smoothed composite risk score in [0, 1]."""
        return self._score

    @property
    def is_high_risk_active(self) -> bool:
        """True from the moment sustained high risk fires until the score drops back down."""
        return self._high_tracker.is_active


@register_module("driver_attention")
class DriverAttentionModule(VisionModule):
    """Detects drowsiness, distraction, and phone/food use while tolerating normal behavior."""

    name = "driver_attention"

    def __init__(self, config: dict) -> None:
        """Read thresholds from config and initialize MediaPipe Face Mesh + calibration."""
        super().__init__(config)

        # --- calibration (EAR baseline + neutral "looking at the road" yaw/pitch) ---
        self._calibrator = AttentionCalibrator(
            calibration_path=Path(config.get("calibration_path", "data/calibration.json")),
            duration_s=float(config.get("calibration_duration_s", 10.0)),
            ear_threshold_ratio=float(config.get("ear_threshold_ratio", 0.75)),
            ear_threshold_fallback=float(config.get("ear_threshold_fallback", 0.20)),
            force_recalibrate=bool(config.get("recalibrate", False)),
        )

        # --- signal conditioning: raw landmark-derived measurements are NOISY,
        # especially with glasses (frames and glare degrade the eyelid contour),
        # and every threshold below assumed they were clean. A rolling median
        # means no single frame can flip a boolean; `release_s` grace on the
        # trackers means no single frame can reset a timer or drop an alert.
        smoothing_window = int(config.get("signal_median_window", 5))
        release_s = float(config.get("signal_release_s", 0.4))
        self._ear_smoother = SignalSmoother(smoothing_window)
        self._yaw_smoother = SignalSmoother(smoothing_window)
        self._pitch_smoother = SignalSmoother(smoothing_window)
        # Session open-eye reference, so the effective closure threshold follows
        # the face in front of the camera today (glasses on or off) instead of
        # only the face that calibrated. Banded to the calibrated baseline so a
        # genuinely deteriorating driver cannot drag the threshold with them.
        self._ear_baseline = AdaptiveBaseline(
            window_s=float(config.get("ear_baseline_window_s", 60.0)),
            band_low=float(config.get("ear_baseline_band_low", 0.6)),
            band_high=float(config.get("ear_baseline_band_high", 1.4)),
        )
        self._ear_threshold_ratio = float(config.get("ear_threshold_ratio", 0.75))
        self._iris_veto = bool(config.get("iris_veto", True))

        # --- drowsiness ---
        self._drowsy_tracker = GracedConditionTracker(float(config["drowsy_seconds"]), release_s)
        self._perclos_tracker = PerclosTracker(float(config.get("perclos_window_s", 60.0)))
        self._perclos_threshold = float(config.get("perclos_threshold", 0.15))
        self._fatigue_warned = False

        # --- distraction ---
        self._yaw_threshold_deg = float(config["yaw_threshold_deg"])
        self._distract_tracker = GracedConditionTracker(float(config["distraction_seconds"]), release_s)
        self._glance_min_count = int(config.get("glance_min_count", 5))
        self._glance_counter = GlanceCounter(
            window_s=float(config.get("glance_window_s", 45.0)),
            min_count=self._glance_min_count,
        )
        self._pitch_threshold_deg = float(config.get("pitch_threshold_deg", -20.0))
        self._eyes_down_tracker = GracedConditionTracker(
            float(config.get("eyes_down_seconds", 2.5)), release_s
        )

        # --- face lost: the camera can't see the driver's face at all (turned fully away,
        # slumped over, obstructed) -- a real gap in naive logic, which otherwise treats
        # "no landmarks" as "nothing to report" instead of a risk signal in its own right.
        self._face_lost_tracker = GracedConditionTracker(
            float(config.get("face_lost_seconds", 2.5)), release_s
        )
        self._face_lost_active = False

        # --- object detection (phone / food), lazily loaded on first stride tick ---
        self._yolo_weights = config.get("yolo_weights", "yolo11n.pt")
        self._yolo_every_n_frames = max(1, int(config.get("yolo_every_n_frames", 3)))
        self._yolo_confidence_threshold = float(config.get("yolo_confidence_threshold", 0.35))
        self._yolo_class_map: dict[str, str] = dict(config.get("yolo_classes", {}))
        self._enable_hands = bool(config.get("enable_hands", True))
        self._hand_phone_iou_threshold = float(config.get("hand_phone_iou_threshold", 0.2))
        self._phone_use_tracker = GracedConditionTracker(
            float(config.get("phone_use_seconds", 2.0)), release_s
        )

        # --- cross-signal risk fusion (reasons about combinations, not just individual signals) ---
        self._risk_fusion = RiskFusion(
            ema_alpha=float(config.get("risk_ema_alpha", 0.3)),
            elevated_threshold=float(config.get("risk_elevated_threshold", 0.4)),
            high_threshold=float(config.get("risk_high_threshold", 0.75)),
            confirm_seconds=float(config.get("risk_confirm_seconds", 1.0)),
        )
        self._last_risk_score = 0.0

        self._yolo_model = None
        self._yolo_class_ids: dict[int, str] = {}
        self._hands = None
        self._object_detection_ready = False
        self._object_detection_error: Optional[str] = None
        self._frame_index = 0
        self._last_phone_boxes: list[Box] = []
        self._last_eating_boxes: list[Box] = []
        self._last_hand_boxes: list[Box] = []
        self._last_overlay_boxes: list[tuple[Box, str]] = []
        # Parallel to _last_overlay_boxes. Kept separate rather than widening that
        # tuple, which the annotate path unpacks as a pair.
        self._last_overlay_confidences: list[float] = []
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._phone_visible_prev = False
        self._eating_visible_prev = False
        self._pending_phone_detected_edge = False
        self._pending_eating_detected_edge = False
        self._hand_overlaps_phone = False

        # --- live state, for annotate()/annotate_debug()/is_alert_active() ---
        # Confidences are configurable because glasses wearers sit close to the
        # tracking margin: lowering min_tracking_confidence a notch in the
        # field trades a little landmark quality for far fewer dropouts, and
        # that trade should not require a code change to make.
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(config.get("face_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("face_tracking_confidence", 0.5)),
        )
        self._last_face_proto = None
        self._last_ear = 0.0
        self._last_iris: Optional[float] = None
        self._last_ear_threshold = 0.0
        self._last_yaw_raw = 0.0
        self._last_pitch_raw = 0.0
        self._last_yaw = 0.0  # calibration-corrected: 0 == "looking at the road" for this mount
        self._last_pitch = 0.0
        self._last_perclos = 0.0
        self._drowsy_active = False
        self._distracted_active = False
        self._eyes_down_active = False
        self._phone_use_active = False

    def close(self) -> None:
        """Release the mediapipe FaceMesh graph (leaked per preset switch otherwise)."""
        try:
            self._face_mesh.close()
        except Exception:  # noqa: BLE001 - closing twice must be harmless
            pass

    def _ensure_object_detectors_loaded(self) -> None:
        """Lazily load YOLO (shared) and MediaPipe Hands on the first stride tick."""
        if self._yolo_model is None:
            self._yolo_model = get_yolo_model(self._yolo_weights)
            self._yolo_class_ids = {
                class_id: self._yolo_class_map[name]
                for class_id, name in self._yolo_model.names.items()
                if name in self._yolo_class_map
            }
            if not self._yolo_class_ids:
                raise RuntimeError(
                    f"none of yolo_classes {list(self._yolo_class_map)} matched a class name "
                    f"the '{self._yolo_weights}' model knows about — check config.yaml spelling"
                )
        if self._enable_hands and self._hands is None:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        if not self._object_detection_ready:
            self._object_detection_ready = True
            print(
                f"driver_attention: object detection ready "
                f"(YOLO11-nano classes={list(self._yolo_class_map)}, hands={self._enable_hands}, "
                f"conf>={self._yolo_confidence_threshold}, every {self._yolo_every_n_frames} frames)"
            )

    def _update_object_detections(self, frame: np.ndarray) -> None:
        """Run YOLO (+ optional Hands) on `frame` and refresh cached box state.

        Any failure here (bad weights, a broken MediaPipe Hands install, ...) is caught,
        printed once, and disables further attempts for this run rather than crashing the
        whole live-video loop or failing silently.
        """
        if self._object_detection_error is not None:
            return
        try:
            self._ensure_object_detectors_loaded()
            self._run_object_detections(frame)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: never crash the video loop
            self._object_detection_error = str(exc)
            # Clear the cached boxes along with disabling: a phone box frozen
            # from the tick before the error would otherwise keep phone_visible
            # true for the rest of the run — feeding permanent phone risk into
            # the fusion and arming phone_use on any sustained glance away,
            # with no detector running to ever contradict it.
            self._last_phone_boxes = []
            self._last_eating_boxes = []
            self._last_hand_boxes = []
            self._last_overlay_boxes = []
            self._last_overlay_confidences = []
            self._hand_overlaps_phone = False
            print(f"driver_attention: object detection disabled after an error: {exc}")

    def _run_object_detections(self, frame: np.ndarray) -> None:
        """The actual YOLO + Hands inference and box bookkeeping for one stride tick."""
        h, w = frame.shape[:2]

        results = self._yolo_model.predict(
            frame, verbose=False, classes=list(self._yolo_class_ids.keys()), conf=self._yolo_confidence_threshold
        )
        phone_boxes: list[Box] = []
        eating_boxes: list[Box] = []
        overlay_boxes: list[tuple[Box, str]] = []
        overlay_confidences: list[float] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                event_type = self._yolo_class_ids.get(class_id)
                if event_type is None:
                    continue
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                overlay_boxes.append((xyxy, self._yolo_model.names[class_id]))
                overlay_confidences.append(float(box.conf[0]))
                if event_type == "phone_detected":
                    phone_boxes.append(xyxy)
                else:
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
        self._last_overlay_confidences = overlay_confidences
        self._last_frame_size = (w, h)
        self._hand_overlaps_phone = any(
            iou(hand_box, phone_box) > self._hand_phone_iou_threshold
            for hand_box in hand_boxes
            for phone_box in phone_boxes
        )

        phone_now_visible = bool(phone_boxes)
        if phone_now_visible and not self._phone_visible_prev:
            self._pending_phone_detected_edge = True
            print("driver_attention: phone detected in frame")
        self._phone_visible_prev = phone_now_visible

        eating_now_visible = bool(eating_boxes)
        if eating_now_visible and not self._eating_visible_prev:
            self._pending_eating_detected_edge = True
        self._eating_visible_prev = eating_now_visible

    def process(self, frame: np.ndarray) -> list[Event]:
        """Run face mesh, head pose, and (strided) object detection; return fired events."""
        now = time.monotonic()
        events: list[Event] = []
        h, w = frame.shape[:2]

        self._frame_index += 1
        if self._frame_index % self._yolo_every_n_frames == 0:
            self._update_object_detections(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        phone_visible = bool(self._last_phone_boxes)
        eating_visible = bool(self._last_eating_boxes)
        eating_risk = 1.0 if eating_visible else 0.0

        if not results.multi_face_landmarks:
            self._last_face_proto = None
            # Advance the condition trackers with False rather than forcing the
            # active flags off: their release grace is what stops a one-frame
            # tracking dropout (common with glasses glare) from flickering the
            # alert screen and restarting every timer. A sustained loss still
            # releases them within release_s, and face_lost takes over as the
            # active signal for it.
            self._drowsy_tracker.update(False, now)
            self._distract_tracker.update(False, now)
            self._eyes_down_tracker.update(False, now)
            self._drowsy_active = self._drowsy_tracker.is_active
            self._distracted_active = self._distract_tracker.is_active
            self._eyes_down_active = self._eyes_down_tracker.is_active

            # A face the camera can't see at all is its own risk signal, not "nothing to
            # report" -- see the class docstring on _face_lost_tracker for why.
            fired = self._face_lost_tracker.update(True, now)
            self._face_lost_active = self._face_lost_tracker.is_active
            if fired is not None:
                events.append(Event.now("attention_lost", fired, 0.9, self.name))

            # Can't confirm eyes-off-road without a face reading, so a visible phone here
            # gets partial (not full) phone_risk credit -- it might just be sitting in a
            # mount while the driver's head is turned for an unrelated reason.
            score, risk_event = self._risk_fusion.update(
                eye_risk=0.0,
                gaze_risk=0.0,
                phone_risk=0.5 if phone_visible else 0.0,
                eating_risk=eating_risk,
                face_lost_risk=1.0 if self._face_lost_active else 0.0,
                now=now,
            )
            self._last_risk_score = score
            if risk_event is not None:
                events.append(Event.now(risk_event, 0.0, score, self.name))
            return events

        self._face_lost_tracker.update(False, now)
        # The face returning after a SUSTAINED absence is a scene change: the
        # smoothers' samples describe whatever was there before, so they start
        # fresh. A one-frame dropout never reaches here with _face_lost_active
        # set (the tracker's grace holds), so brief noise keeps its history.
        if self._face_lost_active:
            self._ear_smoother.reset()
            self._yaw_smoother.reset()
            self._pitch_smoother.reset()
        self._face_lost_active = False

        face_proto = results.multi_face_landmarks[0]
        self._last_face_proto = face_proto
        landmarks = face_proto.landmark
        points = [(lm.x * w, lm.y * h) for lm in landmarks]

        # --- raw signals: EAR + head pose (both feed calibration before any thresholding) ---
        left_ear = eye_aspect_ratio(*(points[i] for i in LEFT_EYE))
        right_ear = eye_aspect_ratio(*(points[i] for i in RIGHT_EYE))
        # MAX of the two eyes, not the mean. Drowsy eyes close TOGETHER, so a
        # real closure drops both and the max still reads closed. Glare on one
        # spectacle lens corrupts ONE eye's landmarks — under a mean that read
        # as half-closed eyes; under the max the clean eye simply outvotes it.
        # (A wink also reads open, which is correct: a wink is not drowsiness.)
        raw_ear = max(left_ear, right_ear)
        # Median-smoothed: a single glare/glasses frame cannot move the value
        # past the recent majority, so no downstream boolean sees the spike.
        ear = self._ear_smoother.update(raw_ear)
        self._last_ear = ear
        # Second opinion from the iris landmarks (already computed — FaceMesh
        # runs with refine_landmarks=True; the indices were simply never read).
        # The iris is fit from pupil/limbus contrast, which survives spectacle
        # frames better than the eyelid contour this EAR is built from.
        iris_ratio = iris_openness(points)
        self._last_iris = iris_ratio

        image_points = np.array(
            [
                points[POSE_LANDMARK_IDS[key]]
                for key in ("nose_tip", "chin", "left_eye_corner", "right_eye_corner", "left_mouth", "right_mouth")
            ],
            dtype=np.float64,
        )
        pose = estimate_head_pose_deg(image_points, (h, w))
        raw_yaw, raw_pitch = pose if pose is not None else (None, None)
        if raw_yaw is not None:
            self._last_yaw_raw = raw_yaw
            self._last_pitch_raw = raw_pitch

        self._calibrator.observe(ear, raw_yaw, raw_pitch, now)

        # --- drowsiness: calibrated EAR against a session-adaptive reference ---
        # The stored calibration answers "what does an open eye measure" for
        # whatever face calibrated — possibly bare-faced. Glasses shift that
        # distribution, so the session's own open-eye level (upper percentile
        # over the last minute, banded to the calibration) sets the effective
        # threshold: same ratio, live reference.
        self._ear_baseline.observe(ear, now)
        reference = self._ear_baseline.reference(self._calibrator.baseline_ear)
        if reference is not None:
            effective_threshold = reference * self._ear_threshold_ratio
        else:
            effective_threshold = self._calibrator.ear_threshold
        self._last_ear_threshold = effective_threshold

        drowsy_condition = ear < effective_threshold
        # Iris veto: when the eyelid-contour EAR says "closed" but the iris is
        # plainly, measurably visible, the EAR reading is a glasses/glare
        # artifact — a visible iris and a closed eye cannot both be true. The
        # veto only ever suppresses a closure claim, never invents one, so it
        # cannot cost a real drowsiness detection where the iris genuinely
        # disappears behind the lids.
        if (
            drowsy_condition
            and self._iris_veto
            and iris_ratio is not None
            and iris_ratio >= _IRIS_CLEARLY_OPEN
        ):
            drowsy_condition = False
        fired = self._drowsy_tracker.update(drowsy_condition, now)
        # is_active (not the raw per-frame drowsy_condition) drives the live alert: a
        # blink makes drowsy_condition briefly True on every frame, but is_active only
        # becomes True once the closure has actually been sustained for drowsy_seconds.
        self._drowsy_active = self._drowsy_tracker.is_active
        if fired is not None:
            events.append(Event.now("drowsiness", fired, max(0.0, 1.0 - ear), self.name))

        perclos = self._perclos_tracker.update(drowsy_condition, now)
        self._last_perclos = perclos
        if perclos > self._perclos_threshold and not self._fatigue_warned:
            events.append(Event.now("fatigue_warning", 0.0, perclos, self.name))
            self._fatigue_warned = True
        elif perclos < self._perclos_threshold * 0.8:
            self._fatigue_warned = False

        # --- distraction: yaw (sideways) + pitch (down), corrected for the camera's mounting
        # angle -- "0" here means the calibrated neutral pose, not raw yaw=0/pitch=0, so an
        # off-center-mounted camera doesn't misread ordinary forward driving as a head turn.
        eyes_off_road = False
        if pose is not None:
            # Wrap-safe delta FIRST, median second. The delta (angle_delta_deg)
            # is continuous around the baseline even when the raw angle rides
            # the ±180 wrap — where raw values flip sign frame-to-frame and a
            # median of them is as meaningless as their mean. Then the median,
            # like EAR's: solvePnP on 6 landmarks (two of them eye corners,
            # exactly where spectacle frames sit) spikes by tens of degrees on
            # a single bad frame, and a spike was enough to seed a false
            # glance "episode" in the counter below.
            yaw = self._yaw_smoother.update(angle_delta_deg(raw_yaw, self._calibrator.baseline_yaw))
            pitch = self._pitch_smoother.update(
                angle_delta_deg(raw_pitch, self._calibrator.baseline_pitch)
            )
            self._last_yaw = yaw
            self._last_pitch = pitch

            distract_condition = abs(yaw) > self._yaw_threshold_deg
            fired = self._distract_tracker.update(distract_condition, now)
            # Same fix as drowsiness: the live alert only turns on once the turn has
            # actually been sustained for distraction_seconds, not on every glance.
            self._distracted_active = self._distract_tracker.is_active
            if fired is not None:
                events.append(Event.now("distraction", fired, min(1.0, abs(yaw) / 90.0), self.name))

            if self._glance_counter.update(distract_condition, now):
                events.append(Event.now("frequent_glancing", 0.0, 1.0, self.name))

            eyes_down_condition = pitch < self._pitch_threshold_deg
            fired = self._eyes_down_tracker.update(eyes_down_condition, now)
            self._eyes_down_active = self._eyes_down_tracker.is_active
            if fired is not None:
                events.append(Event.now("eyes_down", fired, min(1.0, abs(pitch) / 90.0), self.name))

            eyes_off_road = distract_condition or eyes_down_condition

        # --- object context: phone / eating, fused with where the eyes are ---
        phone_use_condition = phone_visible and eyes_off_road
        fired = self._phone_use_tracker.update(phone_use_condition, now)
        if fired is not None:
            confidence = 0.95 if self._hand_overlaps_phone else 0.6
            events.append(Event.now("phone_use", fired, confidence, self.name))
            self._phone_use_active = True
        elif not phone_use_condition:
            self._phone_use_active = False

        if self._pending_phone_detected_edge:
            self._pending_phone_detected_edge = False
            if not phone_use_condition:
                # Phone visible but eyes on the road: a mounted phone/GPS is legitimate.
                # Log it for the record; do not alert.
                events.append(Event.now("phone_detected", 0.0, 0.6, self.name))

        if self._pending_eating_detected_edge:
            self._pending_eating_detected_edge = False
            events.append(Event.now("eating_drinking", 0.0, 0.6, self.name))

        # --- risk fusion: reason about the signals above *together*, not just individually ---
        eye_risk = 1.0 if self._drowsy_active else min(1.0, perclos / self._perclos_threshold) * 0.6
        gaze_risk = (
            1.0
            if (self._distracted_active or self._eyes_down_active)
            else min(1.0, self._glance_counter.count / self._glance_min_count) * 0.5
        )
        phone_risk = (
            1.0
            if self._phone_use_active
            else 0.6
            if (phone_visible and self._hand_overlaps_phone)
            else 0.3
            if phone_visible
            else 0.0
        )
        score, risk_event = self._risk_fusion.update(
            eye_risk=eye_risk,
            gaze_risk=gaze_risk,
            phone_risk=phone_risk,
            eating_risk=eating_risk,
            face_lost_risk=0.0,
            now=now,
        )
        self._last_risk_score = score
        if risk_event is not None:
            events.append(Event.now(risk_event, 0.0, score, self.name))

        return events

    def get_detections(self) -> DetectionFrame:
        """Objects seen this stride tick, labelled with their YOLO class name.

        Labels are the detector's own names ("cell phone", "bottle", ...) rather
        than this module's event names, because a custom rule targets the *thing*
        the user described, not the conclusion this module happens to draw from it.

        Hands are excluded: they come from MediaPipe landmarks rather than a
        detector with a comparable confidence, and a hand box means something
        quite different from an object box.
        """
        width, height = self._last_frame_size
        return DetectionFrame(
            width=width,
            height=height,
            detections=tuple(
                Detection(
                    label=label,
                    confidence=(
                        self._last_overlay_confidences[i]
                        if i < len(self._last_overlay_confidences) else 1.0
                    ),
                    box=tuple(float(v) for v in box),
                )
                for i, (box, label) in enumerate(self._last_overlay_boxes)
            ),
        )

    def get_telemetry_snapshot(self) -> Optional[dict]:
        """Raw per-frame signal snapshot for telemetry logging (future model-training data)."""
        return {
            "ear": self._last_ear,
            "iris_openness": self._last_iris,
            "ear_threshold": self._last_ear_threshold,
            "yaw_deg": self._last_yaw,
            "pitch_deg": self._last_pitch,
            "perclos": self._last_perclos,
            "risk_score": self._last_risk_score,
            "phone_visible": bool(self._last_phone_boxes),
            "hand_count": len(self._last_hand_boxes),
            "eating_visible": bool(self._last_eating_boxes),
        }

    def _state_label(self) -> str:
        """A short human-readable label of the current dominant state, for the debug overlay."""
        if self._face_lost_active:
            return "FACE_LOST"
        if self._drowsy_active:
            return "DROWSY"
        if self._phone_use_active:
            return "PHONE_USE"
        if self._eyes_down_active:
            return "EYES_DOWN"
        if self._distracted_active:
            return "DISTRACTED"
        if self._risk_fusion.is_high_risk_active:
            return "HIGH_RISK"
        return "OK"

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw the face mesh, EAR/yaw readouts, and any detected phone/food/hand boxes on `frame`."""
        if self._last_face_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=self._last_face_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
        cv2.putText(frame, f"EAR: {self._last_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {self._last_yaw:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._draw_risk_bar(frame)

        for (x1, y1, x2, y2), label in self._last_overlay_boxes:
            is_active_phone_use = label == "cell phone" and self._phone_use_active
            color = (0, 0, 255) if is_active_phone_use else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for x1, y1, x2, y2 in self._last_hand_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        if self._face_lost_active:
            cv2.putText(
                frame, "FACE NOT VISIBLE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        if self._object_detection_error is not None:
            cv2.putText(
                frame, "object detection: OFF (see terminal)", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
        return frame

    def _draw_risk_bar(self, frame: np.ndarray) -> None:
        """Draw the composite risk score as a labeled, color-coded bar."""
        risk = self._last_risk_score
        color = (0, 200, 0) if risk < 0.4 else (0, 165, 255) if risk < 0.75 else (0, 0, 255)
        x, y, w, h = 10, 82, 200, 18
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.rectangle(frame, (x, y), (x + int(w * risk), y + h), color, -1)
        cv2.putText(
            frame, f"Risk: {risk * 100:.0f}%", (x + w + 10, y + h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
        """Full tuning overlay: raw + calibrated yaw/pitch, PERCLOS, risk inputs, state label."""
        frame = self.annotate(frame)
        cv2.putText(
            frame, f"Pitch: {self._last_pitch:.1f} deg", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Raw yaw/pitch: {self._last_yaw_raw:.1f} / {self._last_pitch_raw:.1f} deg "
            f"(baseline {self._calibrator.baseline_yaw:.1f} / {self._calibrator.baseline_pitch:.1f})",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"PERCLOS: {self._last_perclos * 100:.1f}%  Glances: {self._glance_counter.count}/{self._glance_min_count}",
            (10, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame, f"State: {self._state_label()}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        return frame

    def is_alert_active(self) -> bool:
        """True while any currently-active condition warrants the full-screen alert flash."""
        return (
            self._drowsy_active
            or self._distracted_active
            or self._eyes_down_active
            or self._phone_use_active
            or self._face_lost_active
            or self._risk_fusion.is_high_risk_active
        )
