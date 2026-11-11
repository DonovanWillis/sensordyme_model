"""Driver attention module.

Combines three signal families, each with its own debounce so short, ordinary
behavior (blinks, mirror checks) never fires an event:

  * Drowsiness — Eye Aspect Ratio (EAR) from MediaPipe Face Mesh, evaluated
    against a per-user calibrated threshold. A sustained closure fires
    "drowsiness"; a rolling PERCLOS (% time closed over 60s) fires the softer
    "fatigue_warning" for a driver who blinks heavily but never has one long
    closure.
  * Distraction — head yaw/pitch from solvePnP on 6 canonical landmarks,
    FUSED with eye-in-head offset from the iris landmarks (core/eye_signal.py's
    eye_gaze_offset_deg) into one gaze vector, so "head forward, eyes down"
    is not invisible the way head-pose-alone would make it -- iris landmarks
    are already computed every frame (refine_landmarks=True below), so this
    is arithmetic on numbers already paid for. `enable_gaze: false` reverts
    to head-pose-only. The fused yaw/pitch is then corrected against a
    calibrated neutral pose (see AttentionCalibrator)
    so a camera mounted off to the side of the dashboard doesn't read
    ordinary forward driving as a head turn -- looking at the camera isn't
    "attentive" and looking away from it isn't "distracted"; only the
    calibrated deviation from wherever this driver's neutral forward gaze
    actually is counts. A sustained sideways turn fires "distraction", but
    a WIDE-angle turn (mirror_check_yaw_deg+, shaped like a rearview/side-
    mirror or blind-spot check rather than an ordinary glance) gets more
    patience before it does -- rearview_tolerance_seconds instead of
    distraction_seconds, via PeakCorroboratedGraceTracker, which grants that
    patience to any episode REACHING the wide angle at some point (a head
    passes through the ordinary-glance range on its way out and back, so
    keying it to the episode's first frame made the setting unreachable).
    A sustained downward tilt (phone in lap) fires the separate,
    more sensitive "eyes_down"; several short glances within a rolling
    window fires "frequent_glancing" (texting while still looking up
    periodically).
  * Object context — YOLO11-nano (shared via core/detectors.py) run on the
    full frame on a stride, covering everything a driver's hands might be
    holding (phone, drinks, food), plus optional MediaPipe Hands, fired to
    distinguish a dashboard-mounted phone (benign, log-only "phone_detected")
    from actual phone use (eyes off the road while a phone is visible ->
    "phone_use"). Food and drink are SPATIALLY GATED rather than reported on
    sight: only a box overlapping the driver's own face region (core/
    detectors.py's upper_region, the proxy `near_mouth` and Baby Mode's
    choking-hazard scoring already use) fires "eating_drinking", because the
    class of an object cannot tell a bottle in a cupholder from a bottle
    being drunk from -- see _eating_boxes_near_face. The same detector's
    "person" class (always requested, never reported on its own) supplies
    that driver box, and is the fallback signal for face loss below.
  * Face lost — if the camera can't find a face at all, that absence is
    itself a risk signal rather than "nothing to report"; sustained loss
    fires "attention_lost" -- but face-unavailable does not mean
    driver-disappeared. A turned head the object detector still finds a
    person for (checking a mirror, looking over a shoulder) gets the same
    CorroboratedGraceTracker treatment as a wide-angle turn: patient
    (face_lost_with_body_seconds) while a body keeps being detected,
    reverting immediately to the short bar (face_lost_seconds) the moment
    it isn't -- slumped over, obstructed, or genuinely gone.

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

import contextlib
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import ClassVar, Optional

import cv2
import mediapipe as mp
import numpy as np

from core import perception as shared_perception
from core.behavior import BehaviorFrame, BehaviorReasoner, parse_behavior
from core.behavior.camera_model import CameraIntrinsics
from core.behavior.eye_state import CLOSED as EYE_CLOSED
from core.behavior.eye_state import EyeFrame, EyeObservation, classify_eye_state, eye_patch_occluded
from core.behavior.phone import DETECTOR_FAILED, DETECTOR_NOT_STARTED, DETECTOR_OK, PhoneDetection
from core.confidence import (
    NoiseEstimator,
    clamp_unit,
    joint,
    margin_confidence,
    proportion_confidence,
)
from core.detections import Detection, DetectionFrame
from core.detectors import Box, get_yolo_model, iou, upper_region
from core.events import Event
from core.eye_signal import AdaptiveBaseline, SignalSmoother, eye_gaze_offset_deg, iris_openness
from core.monitoring_state import IMPAIRED, STARTING, Capability
from core.tracking import MAX_MATCH_DISTANCE_DIAGONAL_HAND, BoxTracker
from modules.base import (
    CorroboratedGraceTracker,
    EvidenceClock,
    GracedConditionTracker,
    SustainedConditionTracker,
    VisionModule,
    register_module,
)

#: Object-detection retry after a failure: 1 s, doubling to at most 60 s.
#: Retrying never stops (a reseated HAT or a cleared transient comes back by
#: itself) but its rate is bounded, and a failed load holds no model, so the
#: retries cannot pile models up in memory.
_DETECTOR_RETRY_INITIAL_S = 1.0
_DETECTOR_RETRY_MAX_S = 60.0

# MediaPipe Face Mesh landmark indices for the 6-point EAR eye model.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

_PERSON_CLASS = 0  # COCO -- used only as a face-loss fallback signal, see below

# Iris-openness (eyelid gap / iris diameter, see core/eye_signal.py) is NOT
# independent of EAR -- both divide the same eyelid gap, by the iris diameter
# and by the eye width respectively -- so the iris veto compares it against
# this face's own open level rather than a fixed number (see process()). The
# absolute value below survives only as a floor: under it the eye is not
# "plainly open" for anyone (comfortably open ~0.7-1.1, half-lidded ~0.3-0.5).
_IRIS_CLEARLY_OPEN = 0.5
# The veto fires when iris-openness is at least (ear_threshold_ratio + this)
# of the face's own open reference: a uniform real closure drops iris-openness
# and EAR together, so it can only clear this bar if EAR's denominator (the
# eye corners) was corrupted -- the glasses/glare artifact the veto exists for.
_IRIS_VETO_MARGIN = 0.15
# Below this iris-openness the eye is closed (core/eye_signal.iris_openness:
# "closed < 0.2"): the iris landmark is the model's guess behind the lid, so
# the eye-in-head gaze offset is not fused for that frame.
_IRIS_CLOSED = 0.2

# A calibrated yaw/pitch delta beyond this cannot be a real gaze angle from a
# windshield-mounted camera -- solvePnP on 6 landmarks degenerates as tracking
# breaks down (observed live: -92deg pitch, -76deg yaw, right as a fast turn
# lost FaceMesh lock) and the rolling median (core/eye_signal.py's
# SignalSmoother) only rejects an ISOLATED bad frame, not several consecutive
# ones. This matters specifically because the face-loss coast path below
# holds whatever self._last_yaw/_last_pitch last were for up to
# face_lost_with_body_seconds and feeds it straight into eyes_down/distraction
# confidence and risk fusion -- an implausible value there does not just look
# wrong in a debug overlay, it can drive a real, misleadingly-confident alert.
# Such a reading is therefore REJECTED (treated like a failed pose fit), not
# clamped: a value saturated at 90 still crosses every threshold.
MAX_PLAUSIBLE_GAZE_DEG = 90.0

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


def _extrapolated(confidence: float) -> float:
    """Confidence for a condition carried across a lost face, not measured.

    The reading is the last one before landmarks were lost, so all it can say
    is that the condition held THEN. Nothing measures it now, which is the
    definition of the 0.5 boundary in core/confidence.py: it may or may not
    still hold. Capped there, never raised.
    """
    return min(0.5, confidence)


def _risk_event(event_type: str, score: float, module_name: str) -> Event:
    """A fused-risk event. The fusion score is a weighted risk index (how bad),
    not a certainty (how sure), so it travels in the signals, not as confidence."""
    return Event.now(event_type, 0.0, None, module_name, f"risk score {score:.2f}")


def _saturating_ratio(value: float, threshold: float) -> float:
    """`value` as a fraction of `threshold`, capped at 1.0 and safe at 0.

    Used to turn a raw measurement into "how far along toward its threshold"
    for RiskFusion's partial credit. A threshold of 0 comes straight from
    config and used to be an unguarded ZeroDivisionError inside the frame
    loop; it means "any amount at all already counts as fully there", which is
    the limit this ratio approaches as the threshold shrinks anyway.
    """
    if threshold <= 0:
        return 1.0 if value > 0 else 0.0
    return min(1.0, value / threshold)


def _camera_matrix(
    image_size: tuple[int, int], hfov_deg: Optional[float] = None,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> np.ndarray:
    """Pinhole intrinsics for an (h, w) frame.

    Calibrated `intrinsics` (config camera_intrinsics, from a hardware
    calibration run) take precedence over everything below; None keeps it.

    With `hfov_deg` None the focal length is the image width -- the historical
    assumption, a ~53 degree horizontal field of view. Given the lens's real
    horizontal FOV, fx = (w/2) / tan(hfov/2): Camera Module 3 Wide is ~102
    degrees (fx ~ 0.4 w), the standard Camera Module 3 ~66 degrees.
    """
    if intrinsics is not None:
        return intrinsics.matrix(image_size)
    h, w = image_size
    if hfov_deg is not None and 0.0 < float(hfov_deg) < 180.0:
        focal_length = (w / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
    else:
        focal_length = float(w)
    return np.array(
        [[focal_length, 0, w / 2.0], [0, focal_length, h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )


def estimate_head_rotation(
    image_points: np.ndarray, image_size: tuple[int, int], hfov_deg: Optional[float] = None,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> Optional[np.ndarray]:
    """The face model's rotation into the camera frame (3x3), via solvePnP on
    the 6 canonical landmarks, or None if pose estimation fails.

    Lens distortion is zero unless calibrated `intrinsics` supply it."""
    distortion = intrinsics.distortion_coefficients() if intrinsics is not None else np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, _camera_matrix(image_size, hfov_deg, intrinsics), distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    if not np.all(np.isfinite(rotation_mat)):
        return None
    return rotation_mat


def rotation_to_yaw_pitch_deg(rotation_mat: np.ndarray) -> tuple[float, float]:
    """estimate_head_pose_deg's raw Euler convention for a rotation matrix (see
    that function's docstring for the sign negation). Kept for display,
    telemetry and the mac_max profile; the pi profile's thresholds use
    relative_yaw_pitch_deg instead."""
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
    else:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
    return -math.degrees(yaw), -math.degrees(pitch)


def relative_yaw_pitch_deg(neutral: np.ndarray, rotation_mat: np.ndarray) -> tuple[float, float]:
    """Where the face points RELATIVE to the calibrated neutral pose, as
    (yaw, pitch) in degrees: +yaw toward the frame's right, -pitch down.

    Composes rotations instead of subtracting Euler angles. Per-axis
    subtraction (raw - baseline) is only valid for a camera level with the
    driver's eyes; a dash camera below the eye line sees a pure head turn
    about the body's vertical axis partly as pitch -- measured: with the
    camera 20 deg low, a 70 deg shoulder check read as +27 deg "eyes up" and
    only 62 deg of yaw. R_rel = R_neutral^T @ R is the head's rotation in the
    neutral face frame, independent of where the camera sits; yaw/pitch are
    read off its forward (nose, model +Z) axis, which also makes them
    insensitive to head roll.
    """
    forward = (neutral.T @ rotation_mat)[:, 2]
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    nose_down = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    return yaw, -nose_down


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of rotation matrices: the arithmetic mean projected
    back onto SO(3) (nearest rotation via SVD). Well-behaved for the tightly
    clustered samples of a settled calibration window."""
    stacked = np.mean(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _, vt = np.linalg.svd(stacked)
    d = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt))) or 1.0])
    return u @ d @ vt


#: The model's rotation for a face looking straight into the camera:
#: rot_x(180) -- the model authors +Y up and +Z out of the nose, the camera
#: frame has +Y down and +Z into the scene. The neutral used before a pose
#: baseline exists.
_R_FACING_CAMERA = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def head_rotation_from_yaw_pitch_deg(yaw: float, pitch: float) -> np.ndarray:
    """The rotation of a face turned `yaw` degrees toward the frame's right
    and pitched `pitch` degrees (negative = down) from facing the camera --
    the inverse of relative_yaw_pitch_deg(_R_FACING_CAMERA, .)."""
    a, b = math.radians(-yaw), math.radians(-pitch)
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return ry @ rx @ _R_FACING_CAMERA


def estimate_head_pose_deg(
    image_points: np.ndarray, image_size: tuple[int, int], hfov_deg: Optional[float] = None,
    intrinsics: Optional[CameraIntrinsics] = None,
) -> Optional[tuple[float, float]]:
    """Estimate (yaw, pitch) in degrees from 6 canonical facial landmarks via solvePnP.

    Positive yaw = turned toward the frame's right. Negative pitch = looking down.
    Returns None if pose estimation fails to converge.

    The raw Rodrigues decomposition below is NEGATED on both axes before being
    returned. Verified empirically (tests/test_head_pose_geometry.py), not just
    reasoned about: with _MODEL_POINTS_3D's authored axes (+Y anatomically up,
    +Z the direction the nose points) and OpenCV's camera convention (+Y down
    the image, +Z into the scene), solving for the rotation that explains a
    KNOWN, physically-constructed "looking down" pose returns a POSITIVE raw
    pitch, and a KNOWN "turned toward the frame's right" pose returns a
    NEGATIVE raw yaw -- the exact opposite of the two sentences above, which
    every caller in this module was written to trust. That inversion meant
    `eyes_down_condition = pitch < threshold` (modules/driver_attention.py)
    could only ever fire on sustained sideways-then-up head tilts, never on
    genuinely looking down at a phone -- the single largest reason a driver
    staring at their lap produced no alert. Negating here, once, at the
    source, makes the two documented sentences above actually true and needs
    no change anywhere they're relied on.

    These are RAW Euler angles (an upright face's pitch rides near +-180).
    The pi profile no longer thresholds them by per-axis subtraction -- see
    relative_yaw_pitch_deg -- but the convention is kept unchanged here for
    display/telemetry and for the mac_max profile, which still uses it.
    """
    rotation_mat = estimate_head_rotation(image_points, image_size, hfov_deg, intrinsics)
    if rotation_mat is None:
        return None
    return rotation_to_yaw_pitch_deg(rotation_mat)


class AttentionCalibrator:
    """Learns a per-user EAR threshold AND a per-install neutral head pose.

    Two different things vary by install and must be calibrated rather than
    assumed:

      * Resting EAR varies by face shape, which is why a single fixed EAR
        threshold misfires across users.
      * "Looking at the road" is not "facing the camera" unless the camera
        happens to be mounted dead-center on the dashboard facing the driver.
        A camera off to the side, or below/above the eye line (any dash
        mount), sees the driver's natural forward gaze at some other angle —
        without correcting for that, ordinary forward driving reads as a
        sustained head turn.

    The neutral pose is learned as a ROTATION (the chordal mean of the
    window's head rotations), and every later reading is the head's rotation
    relative to it (relative_yaw_pitch_deg) -- not a per-axis subtraction of
    Euler angles, which is wrong for any camera not level with the eyes (a
    pure head turn leaked into pitch). The eye-in-head offset gets its own
    baseline (its static offset from the eye-corner line), applied after.

    Both are learned from the first `duration_s` of driving (the user should
    look at the road as they normally would) and persisted to
    `calibration_path` so restarts skip recalibration unless
    `force_recalibrate` is set (wired to `main.py --recalibrate`).

    The window is extended (never shortened) past `duration_s` until the
    TRAILING `duration_s` of samples is settled: a calibration run during
    which the driver glanced at a phone, checked a mirror, or was otherwise
    not steadily facing forward would otherwise get averaged into "neutral" —
    silently teaching the whole distraction system that distraction is the
    baseline. Only the trailing window is judged and averaged, so a glance
    early on ages out instead of pinning the spread above the limit forever
    (judging every sample since the start could never settle once one glance
    was in it, and then averaged the glance in at the ceiling anyway).
    `max_pose_spread_deg` is the largest yaw/pitch deviation from the window's
    mean pose tolerated before that counts as "settled"; `max_duration_s` is a
    hard ceiling so a driver who can never hold still still gets calibrated
    eventually (from the trailing window as it then stands).

    The stored baseline is only meaningful for the signal and intrinsics it
    was learned with, so the file records `gaze_fused`, `eye_gaze_scale_deg`
    and `camera_hfov_deg`, and a file whose values differ from the running
    config is rejected (recalibrated) rather than silently reused.
    """

    def __init__(
        self,
        calibration_path: Path,
        duration_s: float,
        ear_threshold_ratio: float,
        ear_threshold_fallback: float,
        force_recalibrate: bool,
        max_pose_spread_deg: float = 20.0,
        max_duration_s: float = 30.0,
        *,
        gaze_fused: bool = True,
        eye_gaze_scale_deg: float = 40.0,
        camera_hfov_deg: Optional[float] = None,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        self._path = calibration_path
        self._duration_s = duration_s
        self._ear_threshold_ratio = ear_threshold_ratio
        self._ear_threshold_fallback = ear_threshold_fallback
        self._max_pose_spread_deg = max_pose_spread_deg
        self._max_duration_s = max(duration_s, max_duration_s)
        self._gaze_fused = bool(gaze_fused)
        self._eye_gaze_scale_deg = float(eye_gaze_scale_deg)
        self._camera_hfov_deg = None if camera_hfov_deg is None else float(camera_hfov_deg)
        # Calibrated lens intrinsics, if any. Stored in the file ONLY when set, so
        # a file written without them is unchanged and still loads (absent == None).
        self._intrinsics_fp = None if camera_intrinsics is None else camera_intrinsics.fingerprint()
        # (t, ear, rotation or None, eye offset or None), trailing window only.
        self._samples: deque[tuple[float, float, Optional[np.ndarray], Optional[tuple[float, float]]]] = deque()
        self._start_time: Optional[float] = None
        self._baseline_ear: Optional[float] = None
        self._ear_threshold: Optional[float] = None
        self._baseline_rotation: Optional[np.ndarray] = None
        self._baseline_eye: Optional[tuple[float, float]] = None
        self._extended_notice_given = False

        if not force_recalibrate:
            self._load()

    #: Bumped when the stored numbers' MEANING changes. Version 2: pose
    #: baselines are circular means (v1 used an arithmetic mean, which for a
    #: wrap-straddling pitch produced a direction no head ever faced — the
    #: deployed device's file held baseline_pitch=-90.3 this way). Version 3:
    #: estimate_head_pose_deg's yaw/pitch sign was inverted (verified via
    #: tests/test_head_pose_geometry.py). Version 4: the baseline was learned
    #: against the FUSED gaze vector (head pose + eye-in-head offset).
    #: Version 5: the neutral pose is a ROTATION (relative-rotation
    #: calibration, see relative_yaw_pitch_deg) plus a separate eye-offset
    #: baseline, and the eye offset is measured against the eye corners
    #: instead of the eyelids -- a v4 per-axis Euler baseline means nothing
    #: under either change, so it is rejected and the device recalibrates.
    FORMAT_VERSION = 5

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if not isinstance(data, dict):
                # Valid JSON of the wrong shape (a list, null, a number) used to
                # raise AttributeError out of the module constructor: the device
                # booted FAILED on every boot until the file was deleted (M6.5).
                return
            if int(data.get("version", 1)) != self.FORMAT_VERSION:
                return  # stale format: leave uncalibrated; a fresh window will rewrite it
            if bool(data["gaze_fused"]) != self._gaze_fused:
                return  # learned on a different signal: recalibrate rather than reinterpret
            if not math.isclose(float(data["eye_gaze_scale_deg"]), self._eye_gaze_scale_deg):
                return
            stored_hfov = data["camera_hfov_deg"]
            if (stored_hfov is None) != (self._camera_hfov_deg is None) or (
                stored_hfov is not None and not math.isclose(float(stored_hfov), self._camera_hfov_deg)
            ):
                return
            stored_fp = data.get("camera_intrinsics")
            if (stored_fp is None) != (self._intrinsics_fp is None) or (
                stored_fp is not None and not np.allclose(np.asarray(stored_fp, dtype=float),
                                                          np.asarray(self._intrinsics_fp, dtype=float))
            ):
                return  # learned under different intrinsics: recalibrate
            baseline_ear = float(data["baseline_ear"])
            ear_threshold = float(data["ear_threshold"])
            rotation = data["baseline_rotation"]
            eye = data["baseline_eye_offset"]
            baseline_rotation = None
            if rotation is not None:
                baseline_rotation = np.array(rotation, dtype=np.float64).reshape(3, 3)
                if not (
                    np.all(np.isfinite(baseline_rotation))
                    and np.allclose(baseline_rotation.T @ baseline_rotation, np.eye(3), atol=1e-3)
                    and np.linalg.det(baseline_rotation) > 0
                ):
                    return
            baseline_eye = None if eye is None else (float(eye[0]), float(eye[1]))
            if not (math.isfinite(baseline_ear) and math.isfinite(ear_threshold)):
                return
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError):
            return  # unreadable: recalibrate rather than fail to build the module
        self._baseline_ear = baseline_ear
        self._ear_threshold = ear_threshold
        self._baseline_rotation = baseline_rotation
        self._baseline_eye = baseline_eye

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
    def baseline_rotation(self) -> np.ndarray:
        """The head rotation that counts as "looking at the road" for this mount
        (facing the camera if no pose was ever measured)."""
        return self._baseline_rotation if self._baseline_rotation is not None else _R_FACING_CAMERA

    @property
    def baseline_eye_offset(self) -> Optional[tuple[float, float]]:
        """The calibration window's mean eye-in-head offset (yaw, pitch), or None
        if no eye offset was measured -- eye fusion is then skipped."""
        return self._baseline_eye

    @property
    def baseline_yaw(self) -> float:
        """The neutral pose's raw head yaw (estimate_head_pose_deg convention) -- display only."""
        if self._baseline_rotation is None:
            return 0.0
        return rotation_to_yaw_pitch_deg(self._baseline_rotation)[0]

    @property
    def baseline_pitch(self) -> float:
        """The neutral pose's raw head pitch (estimate_head_pose_deg convention) -- display only."""
        if self._baseline_rotation is None:
            return 0.0
        return rotation_to_yaw_pitch_deg(self._baseline_rotation)[1]

    def _window_pose(self) -> tuple[Optional[np.ndarray], Optional[tuple[float, float]], float]:
        """(mean rotation, mean eye offset, spread) over the trailing window.

        Spread is the largest deviation of any sample's gaze (head rotation
        relative to the window mean, plus its eye offset relative to the mean
        eye offset) from that mean -- a settled, steadily-forward-facing window
        shows only ordinary micro-movement.
        """
        rotations = [r for _, _, r, _ in self._samples if r is not None]
        eyes = [e for _, _, _, e in self._samples if e is not None]
        mean_eye = (
            (statistics.fmean(e[0] for e in eyes), statistics.fmean(e[1] for e in eyes)) if eyes else None
        )
        if not rotations:
            return None, mean_eye, 0.0
        neutral = mean_rotation(rotations)
        spread = 0.0
        for _, _, rotation, eye in self._samples:
            if rotation is None:
                continue
            yaw, pitch = relative_yaw_pitch_deg(neutral, rotation)
            if eye is not None and mean_eye is not None:
                yaw += eye[0] - mean_eye[0]
                pitch += eye[1] - mean_eye[1]
            spread = max(spread, abs(yaw), abs(pitch))
        return neutral, mean_eye, spread

    def observe(
        self,
        ear: float,
        rotation: Optional[np.ndarray],
        eye_offset: Optional[tuple[float, float]],
        now: float,
    ) -> None:
        """Feed one frame's readings during the calibration window; no-op once calibrated.

        `rotation` is estimate_head_rotation's head rotation (None if pose
        estimation failed); `eye_offset` is eye_gaze_offset_deg's reading, or
        None when unmeasured or gaze fusion is off.
        """
        if self.is_calibrated:
            return
        if self._start_time is None:
            self._start_time = now
        self._samples.append((now, ear, rotation, eye_offset if self._gaze_fused else None))
        cutoff = now - self._duration_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        elapsed = now - self._start_time
        if elapsed < self._duration_s or not self._samples:
            return

        neutral, mean_eye, spread = self._window_pose()
        settled = elapsed >= self._max_duration_s or spread <= self._max_pose_spread_deg
        if not settled:
            if not self._extended_notice_given:
                self._extended_notice_given = True
                print(
                    "driver_attention: calibration window extended — head pose hasn't "
                    "settled yet (still glancing around). Look at the road normally; "
                    f"calibrating within the next {self._max_duration_s - elapsed:.0f}s regardless."
                )
            return

        self._baseline_ear = statistics.fmean(e for _, e, _, _ in self._samples)
        self._ear_threshold = self._baseline_ear * self._ear_threshold_ratio
        self._baseline_rotation = neutral
        self._baseline_eye = mean_eye
        self._samples.clear()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "version": self.FORMAT_VERSION,
            "baseline_ear": self._baseline_ear,
            "ear_threshold": self._ear_threshold,
            "baseline_rotation": None if neutral is None else [float(v) for v in neutral.ravel()],
            "baseline_eye_offset": None if mean_eye is None else [mean_eye[0], mean_eye[1]],
            # Informational only (estimate_head_pose_deg's raw Euler
            # convention), so the file is still human-readable.
            "baseline_yaw": self.baseline_yaw,
            "baseline_pitch": self.baseline_pitch,
            "gaze_fused": self._gaze_fused,
            "eye_gaze_scale_deg": self._eye_gaze_scale_deg,
            "camera_hfov_deg": self._camera_hfov_deg,
        }
        if self._intrinsics_fp is not None:
            record["camera_intrinsics"] = self._intrinsics_fp
        self._path.write_text(json.dumps(record, indent=2))


#: How much observed time PERCLOS needs before it will report a number at all.
#:
#: PERCLOS is, by its own definition, the PERCENTAGE OF TIME the eyes are
#: closed OVER A WINDOW -- a fatigue measure precisely because it integrates
#: many blinks; the research measure it comes from uses a one-minute window,
#: which is what perclos_window_s defaults to. Below its window the ratio is
#: not a noisy estimate of fatigue, it is undefined: divide closed time by the
#: span of whatever samples happen to be in hand and the FIRST closed frame of
#: a session reports 1.00 against a 0.15 threshold, from a "window" 33ms wide.
#: That is what shipped -- every session opened by declaring the driver
#: fatigued if their first measured frames caught a blink.
#:
#: Ten seconds is the smallest span over which the answer means anything: at
#: the 0.15 threshold it takes 1.5 seconds of accumulated closure to cross,
#: which is a real closure and not a blink. It is a floor on EVIDENCE, not a
#: second window -- once passed, the metric is the full configured window's.
PERCLOS_MIN_WINDOW_S = 10.0

#: The longest stretch after a CLOSED sample that is charged as closure when the
#: next sample reads open. Nothing feeds the tracker while the face is lost, so
#: without a cap the whole outage was charged to whatever the eyes were doing in
#: the last frame before it: a blink just before a 20s face loss read as 40%
#: PERCLOS. 1s is several frames even on a slow device, so a real closure that
#: ends between frames is still counted in full.
PERCLOS_MAX_SAMPLE_GAP_S = 1.0


class PeakCorroboratedGraceTracker(CorroboratedGraceTracker):
    """A CorroboratedGraceTracker whose corroboration reflects the episode's
    PEAK evidence rather than the evidence present on its FIRST frame.

    The base class latches corroboration from the opening frame of an episode
    and can only ever drop it afterward. That is exactly right for the signal
    it was built for -- face-lost corroborated by a still-visible body, where
    the body disappearing mid-episode means the situation got WORSE and the
    patient threshold should be withdrawn.

    It is structurally wrong for the mirror-check yaw signal, to the point of
    making rearview_tolerance_seconds unreachable config. A distraction
    episode BEGINS the instant |yaw| crosses yaw_threshold_deg (35), which is
    by definition BELOW mirror_check_yaw_deg (55): a head does not teleport to
    70 degrees, it passes through 35 on the way. So every real head turn --
    including every blind-spot check the setting exists for -- opened its
    episode uncorroborated and latched there. Measured before this class
    existed: a 0->70 degree sweep held at 70 never corroborated once and fired
    distraction at 3.45s, the short threshold, in every configuration.

    Here corroboration ACCUMULATES within an episode instead: once seen it
    holds until the episode ends. Applied to yaw that reads as "this turn
    reached blind-spot magnitude at some point", which is what makes it a
    mirror check -- the return sweep back through 40 degrees is part of the
    same maneuver, not evidence it was never one. Everything else is the base
    class untouched: the two thresholds, the release grace, and the invariant
    that corroboration only ever buys MORE patience and never less. It does
    not leak between episodes; the base class resets it on release.
    """

    def update(self, condition: bool, corroborated: bool, now: float) -> Optional[float]:
        if condition and self._start is not None:
            # Mid-episode only. On the opening frame the base class's own
            # "set it from this frame" behavior is already correct, and
            # pre-empting it there would hand an ordinary glance -- which has
            # never corroborated anything -- the patient threshold.
            if corroborated:
                self._corroborated = True
            corroborated = corroborated or self._corroborated
        return super().update(condition, corroborated, now)


class PerclosTracker:
    """Tracks the rolling fraction of elapsed time a condition (eye closure) has been true."""

    def __init__(self, window_s: float, min_window_s: float = PERCLOS_MIN_WINDOW_S) -> None:
        self._window_s = window_s
        # A configured window SHORTER than the default minimum is a deliberate
        # statement about how much evidence this deployment wants, so it caps
        # the floor rather than being overruled by it -- otherwise the metric
        # would be permanently zero, which is a worse answer than a short one.
        self._min_window_s = min(min_window_s, window_s)
        self._samples: deque[tuple[float, bool]] = deque()

    @property
    def episodes(self) -> int:
        """Distinct closures in the window: the independent observations PERCLOS rests on.

        Frames are not independent -- one closure spans many -- so this, not the
        frame count, is the sample size behind the fraction's uncertainty.
        """
        count, prev = 0, False
        for _t, closed in self._samples:
            if closed and not prev:
                count += 1
            prev = closed
        return count

    def update(self, closed: bool, now: float) -> float:
        """Record the current closed/open state and return the PERCLOS fraction in [0, 1].

        Returns 0.0 until `min_window_s` of observation has actually
        accumulated -- see PERCLOS_MIN_WINDOW_S.
        """
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
                # Closed on both sides of a long gap is sustained closure (a
                # sleeping driver can drop the face mesh); closed then open is
                # a blink followed by unobserved time, charged only up to the cap.
                closed_time += dt if closed_state else min(dt, PERCLOS_MAX_SAMPLE_GAP_S)
            prev_t, prev_closed = t, closed_state

        if total_time < self._min_window_s:
            return 0.0
        return closed_time / total_time if total_time > 0 else 0.0


class GlanceCounter:
    """Counts distinct off-center glance episodes within a rolling window.

    Fires once when `min_count` episodes have occurred within `window_s`,
    re-arming only once the count drops back below `min_count` (as old
    episodes age out of the window).

    `release_s` is the same release grace the condition trackers use: a new
    episode only starts once the condition has been False for at least this
    long. Without it, one glance whose angle hovers near the threshold
    chattered across it and was counted as several glances (one 2 s glance
    settling at 37 deg against a 35 deg threshold counted 3+ times in 13% of
    simulated trials). 0 keeps the old every-rising-edge behaviour.
    """

    def __init__(self, window_s: float, min_count: int, release_s: float = 0.0) -> None:
        self._window_s = window_s
        self._min_count = min_count
        self._release_s = max(0.0, float(release_s))
        self._episode_starts: deque[float] = deque()
        self._was_off_center = False
        self._last_off_center_at: Optional[float] = None
        self._warned = False

    def update(self, off_center: bool, now: float) -> bool:
        """Feed the current off-center boolean. Returns True the moment `min_count` is reached."""
        if off_center and not self._was_off_center:
            continuation = (
                self._last_off_center_at is not None
                and now - self._last_off_center_at < self._release_s
            )
            if not continuation:
                self._episode_starts.append(now)
        if off_center:
            self._last_off_center_at = now
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


#: The update interval risk_ema_alpha is specified at: one frame at 30 fps,
#: the deployed device's measured frame rate (docs/PHASE7_COMPETITION_FREEZE.md),
#: so the shipped alpha behaves as before on the device and the same in wall-
#: clock time at any other frame rate.
RISK_EMA_REFERENCE_DT_S = 1.0 / 30.0


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
        alert on, instead of flickering with every frame's noise. The EMA is
        time-based: `ema_alpha` is the weight of one update at the reference
        interval RISK_EMA_REFERENCE_DT_S (one frame at 30 fps, the device's
        measured rate), and an update `dt` seconds after the last one uses
        1 - (1 - ema_alpha) ** (dt / RISK_EMA_REFERENCE_DT_S) -- so the score
        reacts in the same wall-clock time at 8 fps as at 30 fps.
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
        self._ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self._elevated_threshold = elevated_threshold
        self._high_threshold = high_threshold
        self._score = 0.0
        self._last_update: Optional[float] = None
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
        alpha = self._ema_alpha
        if self._last_update is not None and now > self._last_update and alpha < 1.0:
            alpha = 1.0 - (1.0 - alpha) ** ((now - self._last_update) / RISK_EMA_REFERENCE_DT_S)
        self._last_update = now
        self._score = alpha * raw_score + (1.0 - alpha) * self._score

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
        # Read before the calibrator: the stored baseline is only valid for
        # the signal (gaze fused or not, its scale) and intrinsics it was
        # learned with, and the calibrator rejects a file that disagrees.
        self._enable_gaze = bool(config.get("enable_gaze", True))
        self._gaze_scale_deg = float(config.get("eye_gaze_scale_deg", 40.0))
        hfov = config.get("camera_hfov_deg")
        self._camera_hfov_deg: Optional[float] = None if hfov is None else float(hfov)
        self._camera_intrinsics = CameraIntrinsics.from_config(config.get("camera_intrinsics"))
        self._calibrator = AttentionCalibrator(
            calibration_path=Path(config.get("calibration_path", "data/calibration.json")),
            duration_s=float(config.get("calibration_duration_s", 10.0)),
            ear_threshold_ratio=float(config.get("ear_threshold_ratio", 0.75)),
            ear_threshold_fallback=float(config.get("ear_threshold_fallback", 0.20)),
            force_recalibrate=bool(config.get("recalibrate", False)),
            max_pose_spread_deg=float(config.get("calibration_max_pose_spread_deg", 20.0)),
            max_duration_s=float(config.get("calibration_max_duration_s", 30.0)),
            gaze_fused=self._enable_gaze,
            eye_gaze_scale_deg=self._gaze_scale_deg,
            camera_hfov_deg=self._camera_hfov_deg,
            camera_intrinsics=self._camera_intrinsics,
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
        # Each decision signal's own noise, measured live, so an event's
        # confidence says how far past its threshold the reading was in units
        # of that noise -- see core/confidence.py. Floors are the signals'
        # resolution: a landmark-fit angle is not meaningful below ~1 degree, an
        # eyelid-contour EAR below ~0.005.
        self._ear_noise = NoiseEstimator(floor=0.005, smoothing_window=smoothing_window)
        self._yaw_noise = NoiseEstimator(floor=1.0, smoothing_window=smoothing_window)
        self._pitch_noise = NoiseEstimator(floor=1.0, smoothing_window=smoothing_window)
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
        # The iris veto compares iris-openness against THIS face's own open
        # level (P90 over the same window as the EAR reference), not a fixed
        # number: iris-openness shares the eyelid landmarks with EAR, so a
        # fixed cutoff was just an absolute EAR floor on top of the personal
        # threshold. No band -- it is only ever used to suppress a closure.
        self._iris_baseline = AdaptiveBaseline(
            window_s=float(config.get("ear_baseline_window_s", 60.0)),
        )
        # PERCLOS counts the eye as closed at a deeper closure than sustained-
        # closure drowsiness does (see perclos_closure_ratio in config.yaml).
        self._perclos_closure_ratio = float(config.get("perclos_closure_ratio", 0.4))

        # --- drowsiness ---
        # Eye closure is the one condition whose False frames can be REAL: the
        # eye reopens between blinks. So its grace bridges only a frame or two
        # of glitch (eye_closure_release_frames), and the sustained duration
        # counts only closed time -- a flurry of short blinks with brief
        # openings is not a continuous drowsy_seconds closure.
        self._drowsy_tracker = GracedConditionTracker(
            float(config["drowsy_seconds"]), release_s,
            release_frames=int(config.get("eye_closure_release_frames", 2)),
            count_true_time_only=True,
        )
        self._perclos_tracker = PerclosTracker(float(config.get("perclos_window_s", 60.0)))
        self._perclos_threshold = float(config.get("perclos_threshold", 0.15))
        self._fatigue_warned = False

        # --- distraction ---
        self._yaw_threshold_deg = float(config["yaw_threshold_deg"])
        # A rearview/side-mirror or blind-spot check turns the head much further
        # than an ordinary off-center glance (at a passenger, the center console,
        # ...), so it gets more patience before counting as distraction. The
        # patience applies to any episode whose yaw REACHES this angle at any
        # point -- see PeakCorroboratedGraceTracker for why keying it to the
        # episode's FIRST frame (the base tracker's behavior) made the setting
        # dead config: an episode necessarily opens at yaw_threshold_deg, below
        # this one. An episode that never reaches it gets distraction_seconds.
        self._mirror_check_yaw_deg = float(config.get("mirror_check_yaw_deg", 55.0))
        self._distract_tracker = PeakCorroboratedGraceTracker(
            short_s=float(config["distraction_seconds"]),
            long_s=float(config.get("rearview_tolerance_seconds", 5.0)),
            release_s=release_s,
        )
        self._glance_min_count = int(config.get("glance_min_count", 5))
        self._glance_counter = GlanceCounter(
            window_s=float(config.get("glance_window_s", 45.0)),
            min_count=self._glance_min_count,
            release_s=release_s,
        )
        self._pitch_threshold_deg = float(config.get("pitch_threshold_deg", -20.0))
        self._eyes_down_tracker = GracedConditionTracker(
            float(config.get("eyes_down_seconds", 2.5)), release_s
        )
        # Symmetric to eyes_down: "looking all the way up" (sun visor, overhead
        # console, or simply not the road) is just as much outside the focused
        # forward zone as looking down is -- the zone is bounded on both sides
        # of the calibrated neutral, not just the downward one.
        self._pitch_up_threshold_deg = float(config.get("pitch_up_threshold_deg", 20.0))
        self._eyes_up_tracker = GracedConditionTracker(
            float(config.get("eyes_up_seconds", 2.5)), release_s
        )

        # --- eye gaze (independent of head pose) -----------------------------
        # Whole-head yaw/pitch (solvePnP) cannot see the eyes move without the
        # head moving, so "head forward, eyes down" was structurally invisible
        # before this: eyes_down/eyes_up/distraction all read the fused gaze
        # vector (head + eye-in-head offset) below, not head pose alone. Iris
        # landmarks 468-477 are already computed every frame (refine_landmarks=
        # True below), so this is arithmetic on numbers already paid for, not
        # new inference. `enable_gaze: false` restores pure head-pose behavior
        # (e.g. for A/B comparison, or a face crop too small for a reliable iris
        # fit) with zero other code changes.
        self._last_head_yaw = 0.0  # head pose alone, uncalibrated -- debug/comparison only
        self._last_head_pitch = 0.0
        self._last_gaze_offset: Optional[tuple[float, float]] = None

        # --- face lost: the camera can't see the driver's face at all (turned fully away,
        # slumped over, obstructed) -- a real gap in naive logic, which otherwise treats
        # "no landmarks" as "nothing to report" instead of a risk signal in its own right.
        # Two thresholds: face lost with no corroborating body detection fires at the
        # short one (today's face_lost_seconds); face lost while the object detector
        # still finds a person in frame (a turned head, not a vanished driver) is more
        # patient, up to face_lost_with_body_seconds -- see CorroboratedGraceTracker.
        self._face_lost_tracker = CorroboratedGraceTracker(
            short_s=float(config.get("face_lost_seconds", 2.5)),
            long_s=float(config.get("face_lost_with_body_seconds", 8.0)),
            release_s=release_s,
        )
        self._face_lost_active = False
        self._last_person_visible = False

        # --- object detection (phone / food), lazily loaded on first stride tick ---
        self._yolo_weights = config.get("yolo_weights", "yolo11n.pt")
        self._yolo_every_n_frames = max(1, int(config.get("yolo_every_n_frames", 3)))
        # Kept in step with config.yaml's shipped value, so a config-less run
        # behaves like the product.
        #
        # The number has moved twice, for different reasons. 0.35 measured 10.8%
        # phone recall at 100% precision on held-out in-car imagery -- it missed
        # close to nine phones in ten. Phase 1 lowered it to 0.10 (51.7% / 95.8%)
        # on CPU YOLO11n. Phase 4 raises it to 0.20 on two findings the CPU
        # measurement could not see:
        #
        #   1. An NMS-on-chip .hef has its score threshold COMPILED IN and
        #      silently ignores anything lower (measured: a 0.200 floor on
        #      yolov11n.hef). So 0.10 meant 0.10 on the raw-decode build and 0.20
        #      on the on-chip build -- one config file describing two different
        #      detectors. 0.20 is the lowest value that means itself on both.
        #   2. On DAMO-YOLO 0.20 is the F1 optimum: 0.848 at 93.8% recall, against
        #      66.2% recall at 0.35. Below it the raw-decode path turns noisy --
        #      14 phone boxes on a single image at 0.10.
        #
        # Precision at 0.20 is ~77%, BELOW CPU YOLO11n's 95.8% at 0.10, and
        # "phone_detected" still fires on the rising edge with no sustain gate.
        # That is accepted deliberately rather than overlooked: a bare
        # phone_detected is log-only by design (no alert tier -- a
        # dash-mounted GPS is legitimate), while the loud tier needs phone + hand
        # overlap + eyes off road SUSTAINED. A false phone costs a log line; a
        # missed phone costs the whole distraction chain. FN is the worse error.
        self._yolo_confidence_threshold = float(config.get("yolo_confidence_threshold", 0.20))
        # Person rides the same predict() pass but CANNOT share its threshold.
        # One predict() takes one `conf`, and phone and person want opposite ends
        # of it: DAMO-YOLO puts a spurious second person box on 51.5% of
        # single-person images at 0.25, falling to 4.5% at 0.50 with person
        # availability still 100%. Left on the phone threshold those phantom boxes
        # reach _eating_boxes_near_face -- which picks the LARGEST person box as
        # "the driver" -- and stretch face-loss patience from 2.5s out to
        # face_lost_with_body_seconds (8.0s). So the pass runs at the low value
        # and person is filtered up to its own threshold afterwards.
        #
        # Capped at 0.50, not DAMO's apparently free 0.60: config ships
        # accelerator.required false, so any Hailo failure falls back to CPU
        # YOLO11n, whose person availability at 0.60 is 86% -- one person in seven
        # unseen. 0.50 is the last value acceptable on BOTH backends, and the
        # threshold must stay correct on the fallback, not just the happy path.
        self._person_confidence_threshold = float(
            config.get("person_confidence_threshold", 0.50)
        )
        self._yolo_class_map: dict[str, str] = dict(config.get("yolo_classes", {}))
        self._enable_hands = bool(config.get("enable_hands", True))
        self._hand_phone_iou_threshold = float(config.get("hand_phone_iou_threshold", 0.2))

        # --- erratic hand movement (hands leaving the wheel / frantic motion) ---
        # Reuses the SAME hand boxes already computed for phone-IoU checking below
        # (no added inference), tracked over time via core.tracking.BoxTracker so
        # "erratic" is judged from trajectory (peak speed AND a direction-reversal
        # count, the same two-signal pattern core/activity.py's Baby Mode flailing
        # detector uses) rather than a two-frame position delta. Driver Mode has no
        # full-body Pose model (only MediaPipe Hands, strided with the object
        # detector), so this is coarser-grained than Baby Mode's every-frame
        # wrist kinematics -- see the module's final report for that limitation.
        # A looser match gate than BoxTracker's default -- see
        # MAX_MATCH_DISTANCE_DIAGONAL_HAND. A hand frantic enough to be worth
        # reporting moves further between two stride ticks than its own box
        # diagonal, so at the default gate the association broke on exactly the
        # motion this feature exists to catch: each tick started a fresh track
        # with an empty trail, leaving no trajectory to judge.
        self._hand_tracker = BoxTracker(max_distance=MAX_MATCH_DISTANCE_DIAGONAL_HAND)
        self._erratic_hand_min_speed = float(config.get("erratic_hand_min_speed", 2.0))
        self._erratic_hand_min_reversals = int(config.get("erratic_hand_min_reversals", 3))
        self._erratic_hands_tracker = GracedConditionTracker(
            float(config.get("erratic_hands_seconds", 1.0)), release_s
        )
        self._erratic_hands_active = False
        self._hands_erratic_raw = False
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
        self._object_detection_failures = 0  # consecutive, reset by a success
        self._object_detection_retry_at: Optional[float] = None
        # A gap between frames longer than this is a stall, not evidence (F4).
        self._evidence_clock = EvidenceClock(float(config.get("max_evidence_gap_s", 0.5)))
        # --- behaviour reasoning v2 (core/behavior/): EXPERIMENTAL, OFF by default ---
        # With behavior_v2 false nothing below changes a single event or flag;
        # with it true, drowsiness / phone_use / phone_detected / frequent_glancing
        # come from the v2 components and long_glance / glance_budget_exceeded are
        # added (log-only). See docs/BEHAVIOUR_REASONING.md.
        self._behavior_v2 = bool(config.get("behavior_v2", False))
        self._behavior: Optional[BehaviorReasoner] = (
            BehaviorReasoner(parse_behavior(config)) if self._behavior_v2 else None
        )
        self._detection_tick = 0  # successful detector runs: "a fresh result exists"
        self._behavior_tick_seen = 0
        self._last_person_boxes: list[Box] = []
        # Parallel to _last_person_boxes: reported by get_detections() so a custom
        # rule on `person` (available_labels declares it) sees the driver.
        self._last_person_confidences: list[float] = []
        self._frame_index = 0
        self._last_phone_boxes: list[Box] = []
        self._last_eating_boxes: list[Box] = []
        # The detector's own best score for each, this tick: the model
        # confidence phone/eating events report (core/confidence.py, "model").
        self._last_phone_score: Optional[float] = None
        self._last_eating_score: Optional[float] = None
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
        # What this frame measured, typed, for the shared observation
        # (perception_signals): only what was actually computed THIS frame.
        self._frame_signals: dict = {}
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
        self._eyes_up_active = False
        self._phone_use_active = False
        self._last_fired_event_types: list[str] = []

    def close(self) -> None:
        """Release the mediapipe graphs (leaked per preset switch otherwise).

        Hands is created lazily, on the first object-detection tick, so it is
        only closed when it exists.
        """
        for graph in (self._face_mesh, getattr(self, "_hands", None)):
            if graph is None:
                continue
            with contextlib.suppress(Exception):  # closing twice must be harmless
                graph.close()

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
            # Through the shared engine: built lazily inside an analysed frame,
            # so the factory (not the runtime's adopt) is what wraps it.
            self._hands = shared_perception.shared_graph(shared_perception.HAND_LANDMARKS, mp.solutions.hands.Hands(
                max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
            ))
        if not self._object_detection_ready:
            self._object_detection_ready = True
            print(
                # Names the model actually loaded, not a hardcoded one. This
                # line said "YOLO11-nano" through the entire switch to DAMO-YOLO
                # on the Hailo, which is exactly the sort of confidently wrong
                # startup banner that makes a log worse than no log.
                f"driver_attention: object detection ready "
                f"({getattr(self._yolo_model, 'model_name', self._yolo_weights)} "
                f"classes={list(self._yolo_class_map)}, hands={self._enable_hands}, "
                f"conf>={self._yolo_confidence_threshold} "
                f"(person>={self._person_confidence_threshold}), "
                f"every {self._yolo_every_n_frames} frames)"
            )

    def _update_object_detections(self, frame: np.ndarray) -> None:
        """Run YOLO (+ optional Hands) on `frame` and refresh cached box state.

        Any failure here (a detector fault, bad weights, a broken MediaPipe Hands
        install, ...) is caught -- it must never crash the video loop -- and
        object detection is reported IMPAIRED and retried with bounded backoff
        (_DETECTOR_RETRY_*). It used to stay disabled for the rest of the run
        after one exception (audit F6), while face-based monitoring carried on
        and nothing said phone detection had stopped.
        """
        wall = time.monotonic()  # backoff runs on real time, not evidence time
        if self._object_detection_retry_at is not None and wall < self._object_detection_retry_at:
            return
        try:
            self._ensure_object_detectors_loaded()
            self._run_object_detections(frame)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: never crash the video loop
            self._object_detection_failures += 1
            backoff = min(_DETECTOR_RETRY_INITIAL_S * 2 ** (self._object_detection_failures - 1),
                          _DETECTOR_RETRY_MAX_S)
            self._object_detection_retry_at = wall + backoff
            first = self._object_detection_error is None
            self._object_detection_error = str(exc)
            # Clear the cached boxes along with disabling: a phone box frozen
            # from the tick before the error would otherwise keep phone_visible
            # true for the rest of the run — feeding permanent phone risk into
            # the fusion and arming phone_use on any sustained glance away,
            # with no detector running to ever contradict it.
            self._last_phone_boxes = []
            self._last_eating_boxes = []
            self._last_phone_score = None
            self._last_eating_score = None
            self._last_hand_boxes = []
            self._last_overlay_boxes = []
            self._last_overlay_confidences = []
            self._hand_overlaps_phone = False
            # Same for the erratic-hands raw flag: process() feeds it to its
            # tracker every frame, so a True left here was an erratic-hands
            # alert stuck on for the rest of the run with no detector running.
            self._hands_erratic_raw = False
            self._hand_tracker.reset()
            # No detector running to corroborate anything -- fail toward the
            # tighter face-lost threshold, not the more patient one.
            self._last_person_visible = False
            self._last_person_boxes = []
            self._last_person_confidences = []
            if first:
                print(f"driver_attention: object detection unavailable after an error "
                      f"(retrying, backoff up to {_DETECTOR_RETRY_MAX_S:.0f}s): {exc}")
            return
        if self._object_detection_error is not None:
            print(f"driver_attention: object detection recovered after "
                  f"{self._object_detection_failures} failed attempt(s)")
        self._object_detection_error = None
        self._object_detection_failures = 0
        self._object_detection_retry_at = None
        self._detection_tick += 1

    def _eating_boxes_near_face(
        self, eating_boxes: list[Box], person_boxes: list[Box]
    ) -> list[Box]:
        """The food/drink boxes actually related to the DRIVER, not merely present.

        Without this, "eating_drinking" fired on any box of a food or drink
        class anywhere in the frame: a bottle standing in a cupholder, a
        takeaway cup in the passenger footwell, and a bottle raised to the
        driver's lips produced identical, indistinguishable events. The class
        of the object was the whole claim, and the class of the object is
        exactly what cannot separate "has a drink" from "is drinking".

        The relation used is core/detectors.upper_region -- the top fraction
        of the driver's own detected box, where a seated person's face is --
        which is the same geometric proxy core/rules.py's `near_mouth` and
        modules/baby_monitoring.py's choking-hazard scoring already stand on.
        It is an approximation and is documented as one in that function: a
        box knows nothing about an actual mouth. It is nonetheless the whole
        difference between an object being in the car and an object being at
        the driver's face, at the fidelity a box detector can support.

        The largest person box is taken as the driver: in a driver-facing
        camera the driver is the nearest and largest person in frame, and a
        rear-seat passenger's snack is not this module's event to fire.

        With NO person detected there is nothing to be near, so nothing
        qualifies. That is the same direction this module already fails in
        when its detector cannot corroborate anything (see the error path in
        _update_object_detections), and it is the honest one: the alternative
        is to keep firing the ungated event and call the missing evidence a
        pass.
        """
        if not eating_boxes or not person_boxes:
            return []
        driver = max(person_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        face_region = upper_region(driver)
        return [box for box in eating_boxes if iou(box, face_region) > 0.0]

    def _run_object_detections(self, frame: np.ndarray) -> None:
        """The actual YOLO + Hands inference and box bookkeeping for one stride tick."""
        h, w = frame.shape[:2]

        # _PERSON_CLASS rides along on the SAME pass (already running every
        # stride tick for phone/food) so face-loss has a fallback signal (see
        # _face_lost_tracker's use of it) and so the eating/drinking spatial
        # gate below has a driver to be spatially related TO. Not added to
        # yolo_class_ids/overlay_boxes: it is a presence check, not a thing
        # this module reports on.
        results = self._yolo_model.predict(
            frame, verbose=False,
            classes=sorted({*self._yolo_class_ids.keys(), _PERSON_CLASS}),
            conf=self._yolo_confidence_threshold,
        )
        phone_boxes: list[Box] = []
        eating_boxes: list[Box] = []
        overlay_boxes: list[tuple[Box, str]] = []
        overlay_confidences: list[float] = []
        person_boxes: list[Box] = []
        person_confidences: list[float] = []
        box_scores: dict[tuple[int, ...], float] = {}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                # Filtered to person's OWN threshold, not the pass's -- see
                # _person_confidence_threshold for why they must differ.
                if (class_id == _PERSON_CLASS
                        and float(box.conf[0]) >= self._person_confidence_threshold):
                    person_boxes.append(tuple(int(v) for v in box.xyxy[0].tolist()))
                    person_confidences.append(float(box.conf[0]))
                event_type = self._yolo_class_ids.get(class_id)
                if event_type is None:
                    continue
                xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                overlay_boxes.append((xyxy, self._yolo_model.names[class_id]))
                overlay_confidences.append(float(box.conf[0]))
                box_scores[xyxy] = max(box_scores.get(xyxy, 0.0), float(box.conf[0]))
                if event_type == "phone_detected":
                    phone_boxes.append(xyxy)
                else:
                    eating_boxes.append(xyxy)
        self._last_person_visible = bool(person_boxes)
        self._last_person_boxes = person_boxes
        self._last_person_confidences = person_confidences
        eating_boxes = self._eating_boxes_near_face(eating_boxes, person_boxes)

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
        self._last_phone_score = max((box_scores[b] for b in phone_boxes), default=None)
        self._last_eating_score = max((box_scores[b] for b in eating_boxes), default=None)
        self._last_hand_boxes = hand_boxes
        self._last_overlay_boxes = overlay_boxes
        self._last_overlay_confidences = overlay_confidences
        self._hand_overlaps_phone = any(
            iou(hand_box, phone_box) > self._hand_phone_iou_threshold
            for hand_box in hand_boxes
            for phone_box in phone_boxes
        )

        # Erratic hand movement: peak speed AND a direction-reversal count over
        # this hand's own tracked trajectory -- a single fast, smooth reach (one
        # direction, no reversals) must not qualify, same false-positive
        # boundary core/activity.py's flailing detector draws for Baby Mode.
        # Sampled at the object-detector's own stride (this method only runs on
        # a stride tick), debounced in process() like every other raw signal here.
        # Hand boxes are in PIXELS, which are already isotropic: BoxTracker's
        # `aspect` multiplier corrects NORMALISED x for a non-square frame, and
        # applied to pixels it inflated horizontal speed and reversal amplitude
        # by w/h (1.33x at 640x480).
        aspect = 1.0
        hand_detections = [("hand", box, 1.0) for box in hand_boxes]
        # Shipped: wall time, while every other tracker here runs on evidence
        # time (EvidenceClock). behavior_v2 puts the hand tracker on the same
        # evidence clock, so a stall cannot read as a fast hand.
        hand_t = self._evidence_clock.now if self._behavior_v2 else time.monotonic()
        matched_hands = self._hand_tracker.update(hand_detections, hand_t, aspect)
        erratic = False
        for track in matched_hands:
            # PEAK instantaneous speed, not net displacement over the trail:
            # oscillation returns to where it started, so a net-displacement
            # reading of chaotic motion is ~0 by construction and could never
            # clear this gate at the same time as the reversal count below.
            # See TrackedBox.peak_speed and core/activity.py's flailing
            # detector, which has always measured intensity this way.
            speed = track.peak_speed(aspect)
            if (speed is not None and speed > self._erratic_hand_min_speed
                    and track.reversals(min_amplitude=0.15, aspect=aspect)
                    >= self._erratic_hand_min_reversals):
                erratic = True
        self._hands_erratic_raw = erratic

        phone_now_visible = bool(phone_boxes)
        if phone_now_visible and not self._phone_visible_prev:
            self._pending_phone_detected_edge = True
            print("driver_attention: phone detected in frame")
        self._phone_visible_prev = phone_now_visible

        eating_now_visible = bool(eating_boxes)
        if eating_now_visible and not self._eating_visible_prev:
            self._pending_eating_detected_edge = True
        self._eating_visible_prev = eating_now_visible

    # ------------------------------------------------------------------ behaviour reasoning v2
    #: Shipped events whose decision moves to core/behavior/ when behavior_v2 is on.
    _BEHAVIOR_V2_REPLACES: ClassVar[frozenset[str]] = frozenset(
        {"drowsiness", "phone_use", "phone_detected", "frequent_glancing"}
    )
    #: Events only v2 raises. Log-only: main.ALERT_EVENT_TYPES_PI does not list
    #: them, and must not until they are validated (docs/BEHAVIOUR_REASONING.md).
    _BEHAVIOR_V2_EVENTS: ClassVar[frozenset[str]] = frozenset({"long_glance", "glance_budget_exceeded"})

    def _eye_patch_occluded(self, gray: np.ndarray, points: list[Point], indices: list[int]) -> Optional[bool]:
        xs = [points[i][0] for i in indices]
        ys = [points[i][1] for i in indices]
        pad = 0.25 * (max(xs) - min(xs))
        h, w = gray.shape[:2]
        x1, x2 = int(max(0.0, min(xs) - pad)), int(min(float(w), max(xs) + pad))
        y1, y2 = int(max(0.0, min(ys) - pad)), int(min(float(h), max(ys) + pad))
        assert self._behavior is not None
        eye_cfg = self._behavior.config.eye
        return eye_patch_occluded(gray[y1:y2, x1:x2], eye_cfg.occlusion_max_mean, eye_cfg.occlusion_max_std)

    def _behavior_eye_frame(
        self, frame: np.ndarray, points: list[Point], *, left_ear: float, right_ear: float, ear: float,
        threshold: float, iris_ratio: Optional[float], iris_veto: bool,
        head_pitch: Optional[float], eye_pitch: Optional[float],
    ) -> EyeFrame:
        """This frame's eye evidence for core/behavior/eye_state.

        With both eyes readable the smoothed max EAR is used, exactly as the
        shipped drowsiness path reads it. With one eye occluded only the other
        eye's (raw) EAR speaks; with both occluded the state is unknown.
        """
        assert self._behavior is not None
        occluded: tuple[Optional[bool], Optional[bool]] = (None, None)
        if self._behavior.config.eye_occlusion_check:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            occluded = (self._eye_patch_occluded(gray, points, LEFT_EYE),
                        self._eye_patch_occluded(gray, points, RIGHT_EYE))
        if not any(occluded):
            eyes: tuple[EyeObservation, ...] = (EyeObservation(ear, False),)
        else:
            eyes = (EyeObservation(left_ear, occluded[0]), EyeObservation(right_ear, occluded[1]))
        deep = threshold * self._perclos_closure_ratio / max(self._ear_threshold_ratio, 1e-6)
        return EyeFrame(
            face_present=True, eyes=eyes, closure_threshold=threshold, deep_threshold=deep,
            iris_openness=iris_ratio, iris_vetoes_closure=iris_veto,
            head_pitch_deg=head_pitch, eye_pitch_deg=eye_pitch,
        )

    def _behavior_step(
        self, frame: np.ndarray, frame_t: float, events: list[Event], *,
        points: Optional[list[Point]], gaze: Optional[tuple[float, float]], gaze_reason: str,
        eye_frame: Optional[EyeFrame], drowsy_confidence: Optional[float],
    ) -> list[Event]:
        """Run the v2 reasoner for this frame and swap its events in for the ones it replaces."""
        assert self._behavior is not None
        h, w = frame.shape[:2]
        face_box = None
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            face_box = (min(xs), min(ys), max(xs), max(ys))
        detection = None
        if self._detection_tick != self._behavior_tick_seen:
            self._behavior_tick_seen = self._detection_tick
            detection = PhoneDetection(
                t=frame_t, phone_boxes=tuple(self._last_phone_boxes), hand_boxes=tuple(self._last_hand_boxes),
                hands_available=self._enable_hands and self._hands is not None, phone_score=self._last_phone_score,
            )
        if self._object_detection_error is not None:
            detector, detector_reason = DETECTOR_FAILED, f"object detection failed: {self._object_detection_error}"
        elif not self._object_detection_ready:
            detector, detector_reason = DETECTOR_NOT_STARTED, "object detector not started"
        else:
            detector, detector_reason = DETECTOR_OK, ""
        region = self._behavior.config.driver_region
        region_px = None if region is None else (region[0] * w, region[1] * h, region[2] * w, region[3] * h)
        produced = self._behavior.update(BehaviorFrame(
            t=frame_t, face_present=points is not None, face_box=face_box, gaze=gaze,
            gaze_unknown_reason=gaze_reason, eye=eye_frame or EyeFrame(face_present=points is not None),
            detector=detector, detector_reason=detector_reason, detection=detection, driver_region=region_px,
        ))
        kept = [e for e in events if e.type not in self._BEHAVIOR_V2_REPLACES]
        for item in produced:
            if item.type == "drowsiness":
                confidence = drowsy_confidence
            elif item.type in ("phone_use", "phone_detected"):
                confidence = clamp_unit(self._last_phone_score)
            else:
                confidence = None  # duration/count rules: no graded evidence to report
            kept.append(Event.now(item.type, item.duration_s, confidence, self.name, f"behavior_v2: {item.detail}"))
        self._drowsy_active = self._behavior.drowsiness.active
        self._phone_use_active = self._behavior.phone.in_use_active
        return kept

    def behavior_uncertainty(self) -> dict[str, str]:
        """v2 components that currently cannot make a claim, with why ({} when v2 is off)."""
        return self._behavior.uncertainty() if self._behavior is not None else {}

    def process(self, frame: np.ndarray) -> list[Event]:
        """Run face mesh, head pose, and (strided) object detection; return fired events."""
        # Evidence time: the frame's capture time with stalls clamped (F4).
        # The raw capture time is kept for behaviour reasoning v2, which must SEE
        # stalls (it credits nothing across them) rather than receive them clamped.
        frame_t = self.frame_time()
        now = self._evidence_clock.advance(frame_t)
        events: list[Event] = []
        h, w = frame.shape[:2]
        # From the ACTUAL frame, every frame -- not only on an object-detector
        # tick. get_detections() reports this size and core/zones.py divides
        # pixel boxes by it to reach the 0..1 space every AI-configured zone
        # polygon lives in. Written only inside _run_object_detections, it
        # stayed (0, 0) whenever detection was disabled or had errored out,
        # making max(frame_w, 1) == 1 and every normalized polygon test fail
        # for every box, forever, with no error and no log line.
        self._last_frame_size = (w, h)
        self._frame_signals = {}

        self._frame_index += 1
        if self._frame_index % self._yolo_every_n_frames == 0:
            self._update_object_detections(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        phone_visible = bool(self._last_phone_boxes)
        eating_visible = bool(self._last_eating_boxes)
        eating_risk = 1.0 if eating_visible else 0.0

        # Independent of face state -- a driver's hands leaving the wheel or
        # moving erratically matters whether or not their face is currently
        # resolvable, so this runs unconditionally every frame, not inside
        # the face-found/face-lost branches below.
        fired = self._erratic_hands_tracker.update(self._hands_erratic_raw, now)
        self._erratic_hands_active = self._erratic_hands_tracker.is_active
        if fired is not None:
            events.append(Event.now(
                "erratic_hand_movement", fired, None, self.name,
                "sustained rapid, chaotic hand motion",
            ))

        if not results.multi_face_landmarks:
            self._last_face_proto = None
            # A significant head turn is a common way to LOSE FaceMesh lock
            # entirely -- it is tuned for near-frontal faces, so "no landmarks"
            # can BE the very thing being measured, not merely "nothing to
            # report". Naively feeding False here resets the distraction/
            # eyes-down/eyes-up clocks to zero the instant that happens, so a
            # fast, wide turn (which breaks landmark tracking almost
            # immediately) would only ever reach the far more patient
            # face_lost/attention_lost path below -- never "distraction".
            #
            # If the object detector still finds a person-shaped body
            # (self._last_person_visible -- the driver hasn't left, just turned
            # far enough that their face isn't resolvable) AND the last
            # calibrated gaze reading before losing lock already showed yaw/
            # pitch past its threshold, that condition is carried forward
            # rather than reset: a head doesn't snap back to centered between
            # frames, so "we can no longer measure the angle" is not evidence
            # the turn ended. This is what lets a fast, wide turn still
            # accumulate toward "distraction" on its own few-second clock
            # instead of silently losing all progress. Bounded, not invented:
            # only coasts an ALREADY-observed condition, never manufactures one,
            # and face_lost's own tracker (below) keeps running in parallel in
            # case the driver has actually left rather than merely turned.
            coast = self._last_person_visible and self._calibrator.is_calibrated
            coasted_distract = coast and abs(self._last_yaw) > self._yaw_threshold_deg
            coasted_mirror = coast and abs(self._last_yaw) > self._mirror_check_yaw_deg
            coasted_eyes_down = coast and self._last_pitch < self._pitch_threshold_deg
            coasted_eyes_up = coast and self._last_pitch > self._pitch_up_threshold_deg

            # Eye state has no such physical continuity to coast on (closed eyes
            # do not imply anything about the next moment), so drowsiness still
            # resets on face loss exactly as before.
            self._drowsy_tracker.update(False, now)
            fired = self._distract_tracker.update(coasted_distract, corroborated=coasted_mirror, now=now)
            self._drowsy_active = self._drowsy_tracker.is_active
            self._distracted_active = self._distract_tracker.is_active
            if fired is not None:
                confidence = _extrapolated(margin_confidence(
                    abs(self._last_yaw) - self._yaw_threshold_deg, self._yaw_noise.sigma))
                events.append(Event.now(
                    "distraction", fired, confidence, self.name,
                    "face lost mid-turn; extrapolated from last known gaze",
                ))
            fired = self._eyes_down_tracker.update(coasted_eyes_down, now)
            self._eyes_down_active = self._eyes_down_tracker.is_active
            if fired is not None:
                events.append(Event.now(
                    "eyes_down", fired,
                    _extrapolated(margin_confidence(
                        self._pitch_threshold_deg - self._last_pitch, self._pitch_noise.sigma)),
                    self.name, "face lost mid-glance; extrapolated from last known gaze",
                ))
            fired = self._eyes_up_tracker.update(coasted_eyes_up, now)
            self._eyes_up_active = self._eyes_up_tracker.is_active
            if fired is not None:
                events.append(Event.now(
                    "eyes_up", fired,
                    _extrapolated(margin_confidence(
                        self._last_pitch - self._pitch_up_threshold_deg, self._pitch_noise.sigma)),
                    self.name, "face lost mid-glance; extrapolated from last known gaze",
                ))

            # phone_use coasts exactly like the gaze conditions it is built on.
            # It used to be skipped entirely here, so its tracker FROZE across
            # the outage: an episode opened before a face loss kept its start
            # time, and the first face frame afterwards (even minutes later)
            # could satisfy phone_use_seconds from time never observed.
            coasted_off_road = coasted_distract or coasted_eyes_down or coasted_eyes_up
            fired = self._phone_use_tracker.update(phone_visible and coasted_off_road, now)
            self._phone_use_active = self._phone_use_tracker.is_active
            if fired is not None:
                off_road_confidence = _extrapolated(max(
                    margin_confidence(abs(self._last_yaw) - self._yaw_threshold_deg, self._yaw_noise.sigma),
                    margin_confidence(self._pitch_threshold_deg - self._last_pitch, self._pitch_noise.sigma),
                    margin_confidence(self._last_pitch - self._pitch_up_threshold_deg, self._pitch_noise.sigma),
                ))
                events.append(Event.now(
                    "phone_use", fired, joint(clamp_unit(self._last_phone_score), off_road_confidence),
                    self.name, "face lost; extrapolated from last known gaze",
                ))

            # A face the camera can't see at all is its own risk signal, not "nothing to
            # report" -- see the class docstring on _face_lost_tracker for why. But a
            # turned head that the object detector still finds a person for is not the
            # same situation as no evidence of the driver at all, so it gets the
            # tracker's more patient threshold -- see CorroboratedGraceTracker.
            fired = self._face_lost_tracker.update(True, corroborated=self._last_person_visible, now=now)
            self._face_lost_active = self._face_lost_tracker.is_active
            if fired is not None:
                # No graded evidence: "no face found for N seconds" is a
                # duration rule, and the corroboration difference is stated in
                # the detail rather than encoded as an invented number.
                detail = "" if not self._face_lost_tracker.corroborated else "person still in frame; face turned away"
                events.append(Event.now("attention_lost", fired, None, self.name, detail))

            if self._behavior is not None:
                events = self._behavior_step(
                    frame, frame_t, events, points=None, gaze=None, gaze_reason="face not detected",
                    eye_frame=None, drowsy_confidence=None,
                )

            # Can't confirm eyes-off-road without a face reading, so a visible phone here
            # gets partial (not full) phone_risk credit -- it might just be sitting in a
            # mount while the driver's head is turned for an unrelated reason. Same idea
            # for face-lost risk itself: a corroborated turn (body still visible) is real
            # but lesser risk, not the full weight of no evidence at all.
            face_lost_risk = 0.0
            if self._face_lost_active:
                face_lost_risk = 1.0 if not self._face_lost_tracker.corroborated else 0.5
            # Coasting (above) can make these active even without a face reading
            # this frame, so gaze risk mirrors the normal path's weighting
            # rather than being hardcoded to zero the moment the face is lost.
            gaze_risk = (
                1.0 if (self._eyes_down_active or self._eyes_up_active)
                else 1.0 if (self._distracted_active and not self._distract_tracker.corroborated)
                else 0.6 if self._distracted_active
                else 0.0
            )
            score, risk_event = self._risk_fusion.update(
                eye_risk=0.0,
                gaze_risk=gaze_risk,
                phone_risk=1.0 if self._phone_use_active else 0.5 if phone_visible else 0.0,
                eating_risk=eating_risk,
                face_lost_risk=face_lost_risk,
                now=now,
            )
            self._last_risk_score = score
            if risk_event is not None:
                events.append(_risk_event(risk_event, score, self.name))
            self._last_fired_event_types = [e.type for e in events]
            return events

        self._face_lost_tracker.update(False, corroborated=self._last_person_visible, now=now)
        # The face returning after a SUSTAINED absence is a scene change: the
        # smoothers' samples describe whatever was there before, so they start
        # fresh. A one-frame dropout never reaches here with _face_lost_active
        # set (the tracker's grace holds), so brief noise keeps its history.
        if self._face_lost_active:
            self._ear_smoother.reset()
            self._yaw_smoother.reset()
            self._pitch_smoother.reset()
            for noise in (self._ear_noise, self._yaw_noise, self._pitch_noise):
                noise.observe(None)  # a gap is motion, not noise
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
        self._ear_noise.observe(raw_ear)
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
        rotation = estimate_head_rotation(image_points, (h, w), self._camera_hfov_deg, self._camera_intrinsics)
        if rotation is not None:
            # Raw Euler angles (estimate_head_pose_deg's convention): display
            # and telemetry only -- thresholds use the rotation relative to the
            # calibrated neutral (relative_yaw_pitch_deg) below.
            raw_yaw, raw_pitch = rotation_to_yaw_pitch_deg(rotation)
            self._last_yaw_raw = raw_yaw
            self._last_pitch_raw = raw_pitch
            self._last_head_yaw = raw_yaw
            self._last_head_pitch = raw_pitch
            head_pose = shared_perception.HeadPose(yaw_deg=float(raw_yaw), pitch_deg=float(raw_pitch),
                                                   calibrated=False)
            if self._calibrator.is_calibrated:
                rel_yaw, rel_pitch = relative_yaw_pitch_deg(self._calibrator.baseline_rotation, rotation)
                if abs(rel_yaw) <= MAX_PLAUSIBLE_GAZE_DEG and abs(rel_pitch) <= MAX_PLAUSIBLE_GAZE_DEG:
                    head_pose = shared_perception.HeadPose(yaw_deg=float(rel_yaw), pitch_deg=float(rel_pitch),
                                                           calibrated=True)
            self._frame_signals["head_pose"] = head_pose
        # Eye-in-head offset from the iris, fused onto head pose -- see the
        # module docstring and core/eye_signal.eye_gaze_offset_deg. Head-pose-
        # only for this frame when the eye is closed or the iris fit is
        # degenerate (iris-openness below _IRIS_CLOSED, or unmeasurable): a
        # closed eye has no gaze, and the iris landmark behind a closed lid is
        # a guess. EAR-based drowsiness covers the closed-eye case itself.
        gaze_offset = None
        if (
            self._enable_gaze
            and rotation is not None
            and iris_ratio is not None
            and iris_ratio >= _IRIS_CLOSED
        ):
            gaze_offset = eye_gaze_offset_deg(points, self._gaze_scale_deg)
        self._last_gaze_offset = gaze_offset
        if gaze_offset is not None:
            self._frame_signals["gaze"] = shared_perception.Gaze(
                yaw_offset_deg=float(gaze_offset[0]), pitch_offset_deg=float(gaze_offset[1]))

        self._calibrator.observe(ear, rotation, gaze_offset, now)

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

        # Iris veto: when the eyelid-contour EAR says "closed" but the iris-
        # normalised lid gap says this face's eye is about as open as it
        # usually is, the EAR reading is an artifact (glasses/glare corrupting
        # the eye-corner landmarks EAR divides by). Both measures share the
        # lid landmarks, so a REAL closure drops them together; the veto
        # therefore compares iris-openness against this face's own open level
        # (P90 of the session), at a fraction above the EAR closure ratio --
        # never a fixed number, which acted as an absolute EAR floor over the
        # personal threshold. Only ever suppresses a closure, never invents one;
        # no veto until the iris reference has warmed up.
        if iris_ratio is not None:
            self._iris_baseline.observe(iris_ratio, now)
        iris_reference = self._iris_baseline.reference(None)
        iris_vetoes_closure = (
            self._iris_veto
            and iris_ratio is not None
            and iris_reference is not None
            and iris_ratio >= max(
                _IRIS_CLEARLY_OPEN,
                min(1.0, self._ear_threshold_ratio + _IRIS_VETO_MARGIN) * iris_reference,
            )
        )
        behavior_eye: Optional[EyeFrame] = None
        if self._behavior is not None:
            # v2 needs the head's own pitch (not the fused gaze) and the iris's,
            # each against the calibrated neutral, to tell looking down from closing.
            head_pitch_delta: Optional[float] = None
            eye_pitch_delta: Optional[float] = None
            if rotation is not None and self._calibrator.is_calibrated:
                head_pitch_delta = relative_yaw_pitch_deg(self._calibrator.baseline_rotation, rotation)[1]
                if abs(head_pitch_delta) > MAX_PLAUSIBLE_GAZE_DEG:
                    head_pitch_delta = None
                eye_baseline = self._calibrator.baseline_eye_offset
                if gaze_offset is not None and eye_baseline is not None:
                    eye_pitch_delta = gaze_offset[1] - eye_baseline[1]
            behavior_eye = self._behavior_eye_frame(
                frame, points, left_ear=left_ear, right_ear=right_ear, ear=ear, threshold=effective_threshold,
                iris_ratio=iris_ratio, iris_veto=iris_vetoes_closure,
                head_pitch=head_pitch_delta, eye_pitch=eye_pitch_delta,
            )

        drowsy_condition = ear < effective_threshold and not iris_vetoes_closure
        self._frame_signals["eye_state"] = shared_perception.EyeState(
            ear=float(ear), iris_openness=None if iris_ratio is None else float(iris_ratio),
            # Closed only against a calibrated threshold; before that there is
            # nothing honest to judge an open eye by.
            closed=bool(drowsy_condition) if self._calibrator.is_calibrated else None,
        )
        fired = self._drowsy_tracker.update(drowsy_condition, now)
        # is_active (not the raw per-frame drowsy_condition) drives the live alert: a
        # blink makes drowsy_condition briefly True on every frame, but is_active only
        # becomes True once the closure has actually been sustained for drowsy_seconds.
        self._drowsy_active = self._drowsy_tracker.is_active
        if fired is not None:
            events.append(Event.now(
                "drowsiness", fired,
                margin_confidence(effective_threshold - ear, self._ear_noise.sigma), self.name,
            ))

        # PERCLOS counts a DEEPER closure than drowsy_condition (see
        # perclos_closure_ratio): the 0.15 cutoff comes from P80 research, and
        # charging every 25%-closed squint or downward glance as "closed"
        # inflated the percentage against a threshold that never meant that.
        perclos_closed = (
            ear < effective_threshold * self._perclos_closure_ratio / max(self._ear_threshold_ratio, 1e-6)
            and not iris_vetoes_closure
        )
        if behavior_eye is not None and self._behavior is not None:
            # v2: a downward look is not closure, and an unreadable eye is not
            # a sample at all (neither open nor closed time).
            eye_state = classify_eye_state(behavior_eye, self._behavior.config.eye)
            perclos_closed = perclos_closed and eye_state.state == EYE_CLOSED
            perclos = self._perclos_tracker.update(perclos_closed, now) if eye_state.known else self._last_perclos
        else:
            perclos = self._perclos_tracker.update(perclos_closed, now)
        self._last_perclos = perclos
        if perclos > self._perclos_threshold and not self._fatigue_warned:
            events.append(Event.now(
                "fatigue_warning", 0.0,
                proportion_confidence(perclos, self._perclos_threshold, self._perclos_tracker.episodes),
                self.name, f"PERCLOS {perclos:.2f} over {self._perclos_threshold:.2f}",
            ))
            self._fatigue_warned = True
        elif perclos < self._perclos_threshold * 0.8:
            self._fatigue_warned = False

        # --- distraction: yaw (sideways) + pitch (down/up), RELATIVE to the calibrated
        # neutral pose -- "0" means wherever this driver looks at the road from this
        # mount, so an off-centre or below-the-eyes camera doesn't misread ordinary
        # forward driving (or a pure head turn) as a turn on another axis.
        eyes_off_road = False
        eyes_off_road_confidence: Optional[float] = None
        gaze_delta: Optional[tuple[float, float]] = None
        if rotation is not None and self._calibrator.is_calibrated:
            yaw_delta, pitch_delta = relative_yaw_pitch_deg(self._calibrator.baseline_rotation, rotation)
            eye_baseline = self._calibrator.baseline_eye_offset
            if gaze_offset is not None and eye_baseline is not None:
                yaw_delta += gaze_offset[0] - eye_baseline[0]
                pitch_delta += gaze_offset[1] - eye_baseline[1]
            # A reading beyond MAX_PLAUSIBLE_GAZE_DEG is a degenerate pose fit,
            # not a gaze: dropped like a failed solvePnP (no tracker update, no
            # noise sample, _last_yaw/_last_pitch keep the last real reading).
            # Clamping it to the limit instead still drove eyes_down/distraction
            # -- and the face-loss coast -- at near-certain confidence.
            if abs(yaw_delta) <= MAX_PLAUSIBLE_GAZE_DEG and abs(pitch_delta) <= MAX_PLAUSIBLE_GAZE_DEG:
                gaze_delta = (yaw_delta, pitch_delta)
        if gaze_delta is None:
            self._yaw_noise.observe(None)
            self._pitch_noise.observe(None)
        if rotation is not None and not self._calibrator.is_calibrated:
            # Not calibrated yet: there is no neutral pose to measure against,
            # so any delta computed now would be against a placeholder and
            # could cross the distraction/eyes_down/eyes_up thresholds on pure
            # calibration noise. Verified via tests/test_driver_attention_
            # scenarios.py: without this branch, a spurious pre-calibration
            # crossing could start a tracker's clock at t=0, so it fired
            # shortly after calibration completed. Feeding False (not skipping
            # the update) matters: the release-grace design means a skipped
            # call leaves whatever state was already there untouched, while an
            # explicit False update is what actually walks it back to clean
            # over the calibration window, comfortably past release_s.
            self._distract_tracker.update(False, corroborated=False, now=now)
            self._glance_counter.update(False, now)
            self._eyes_down_tracker.update(False, now)
            self._eyes_up_tracker.update(False, now)
            self._distracted_active = False
            self._eyes_down_active = False
            self._eyes_up_active = False
        if gaze_delta is not None:
            # Then the median, like EAR's: solvePnP on 6 landmarks (two of them
            # eye corners, exactly where spectacle frames sit) spikes by tens of
            # degrees on a single bad frame, and a spike was enough to seed a
            # false glance "episode" in the counter below.
            yaw_delta, pitch_delta = gaze_delta
            self._yaw_noise.observe(yaw_delta)
            self._pitch_noise.observe(pitch_delta)
            yaw = self._yaw_smoother.update(yaw_delta)
            pitch = self._pitch_smoother.update(pitch_delta)
            self._last_yaw = yaw
            self._last_pitch = pitch

            distract_condition = abs(yaw) > self._yaw_threshold_deg
            mirror_check_angle = abs(yaw) > self._mirror_check_yaw_deg
            fired = self._distract_tracker.update(distract_condition, corroborated=mirror_check_angle, now=now)
            # Same fix as drowsiness: the live alert only turns on once the turn has
            # actually been sustained for distraction_seconds (or, for a wide-angle
            # turn the whole time, rearview_tolerance_seconds) -- not on every glance.
            self._distracted_active = self._distract_tracker.is_active
            if fired is not None:
                confidence = margin_confidence(abs(yaw) - self._yaw_threshold_deg, self._yaw_noise.sigma)
                detail = "" if not self._distract_tracker.corroborated else "sustained wide-angle turn (mirror/blind-spot check taking too long)"
                events.append(Event.now("distraction", fired, confidence, self.name, detail))

            if self._glance_counter.update(distract_condition, now):
                # A count crossing a limit: no graded evidence to report.
                events.append(Event.now("frequent_glancing", 0.0, None, self.name))

            eyes_down_condition = pitch < self._pitch_threshold_deg
            fired = self._eyes_down_tracker.update(eyes_down_condition, now)
            self._eyes_down_active = self._eyes_down_tracker.is_active
            if fired is not None:
                events.append(Event.now(
                    "eyes_down", fired,
                    margin_confidence(self._pitch_threshold_deg - pitch, self._pitch_noise.sigma), self.name,
                ))

            eyes_up_condition = pitch > self._pitch_up_threshold_deg
            fired = self._eyes_up_tracker.update(eyes_up_condition, now)
            self._eyes_up_active = self._eyes_up_tracker.is_active
            if fired is not None:
                events.append(Event.now(
                    "eyes_up", fired,
                    margin_confidence(pitch - self._pitch_up_threshold_deg, self._pitch_noise.sigma), self.name,
                ))

            eyes_off_road = distract_condition or eyes_down_condition or eyes_up_condition
            # Certainty of "off road" is that of whichever criterion holds most
            # firmly -- the OR of the three, bounded below by its strongest term.
            eyes_off_road_confidence = max(
                margin_confidence(abs(yaw) - self._yaw_threshold_deg, self._yaw_noise.sigma),
                margin_confidence(self._pitch_threshold_deg - pitch, self._pitch_noise.sigma),
                margin_confidence(pitch - self._pitch_up_threshold_deg, self._pitch_noise.sigma),
            )

        # --- object context: phone / eating, fused with where the eyes are ---
        phone_use_condition = phone_visible and eyes_off_road
        fired = self._phone_use_tracker.update(phone_use_condition, now)
        # The tracker's own state, like every other condition here: clearing
        # the flag on any single False frame (one stride tick where YOLO missed
        # the phone) dropped the live alert for the rest of the episode, since
        # the still-active tracker never fires a second time to set it again.
        self._phone_use_active = self._phone_use_tracker.is_active
        if fired is not None:
            # Both conditions had to hold: the detector's certainty there is a
            # phone, times the measurement's that the eyes were off the road.
            # A hand on the phone is corroboration, stated, not a number.
            confidence = joint(clamp_unit(self._last_phone_score), eyes_off_road_confidence)
            detail = "hand on phone" if self._hand_overlaps_phone else ""
            events.append(Event.now("phone_use", fired, confidence, self.name, detail))

        if self._pending_phone_detected_edge:
            self._pending_phone_detected_edge = False
            if not phone_use_condition:
                # Phone visible but eyes on the road: a mounted phone/GPS is legitimate.
                # Log it for the record; do not alert.
                events.append(Event.now("phone_detected", 0.0, clamp_unit(self._last_phone_score), self.name))

        if self._pending_eating_detected_edge:
            self._pending_eating_detected_edge = False
            events.append(Event.now("eating_drinking", 0.0, clamp_unit(self._last_eating_score), self.name))

        if self._behavior is not None:
            if not self._calibrator.is_calibrated:
                gaze_reason = "calibration not complete"
            elif rotation is None:
                gaze_reason = "head pose fit failed"
            else:
                gaze_reason = "implausible head pose fit"
            events = self._behavior_step(
                frame, frame_t, events, points=points,
                gaze=(self._last_yaw, self._last_pitch) if gaze_delta is not None else None,
                gaze_reason=gaze_reason, eye_frame=behavior_eye,
                drowsy_confidence=margin_confidence(effective_threshold - ear, self._ear_noise.sigma),
            )

        # --- risk fusion: reason about the signals above *together*, not just individually ---
        # Both of these turn a raw count into "how far along toward the
        # threshold" for partial credit, and both divide by a number that comes
        # straight from config with nothing stopping it being 0 -- a plausible
        # "switch this contribution off" edit, which was instead an unguarded
        # ZeroDivisionError raised once per frame inside the video loop. A
        # threshold of 0 means "any amount at all already counts as fully
        # there", which is the limit these ratios approach anyway.
        eye_risk = 1.0 if self._drowsy_active else _saturating_ratio(perclos, self._perclos_threshold) * 0.6
        gaze_risk = (
            1.0
            if (self._eyes_down_active or self._eyes_up_active)
            # A corroborated (wide-angle, mirror-check-shaped) sustained turn is
            # real but lesser risk than an ordinary moderate-angle distraction --
            # same reasoning as face_lost_risk below.
            else 1.0 if (self._distracted_active and not self._distract_tracker.corroborated)
            else 0.6 if self._distracted_active
            else _saturating_ratio(self._glance_counter.count, self._glance_min_count) * 0.5
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
            events.append(_risk_event(risk_event, score, self.name))

        self._last_fired_event_types = [e.type for e in events]
        return events

    def get_detections(self) -> DetectionFrame:
        """Objects seen this stride tick, labelled with their YOLO class name.

        Labels are the detector's own names ("cell phone", "bottle", ...) rather
        than this module's event names, because a custom rule targets the *thing*
        the user described, not the conclusion this module happens to draw from it.

        Hands are excluded: they come from MediaPipe landmarks rather than a
        detector with a comparable confidence, and a hand box means something
        quite different from an object box.

        Person boxes are included: available_labels() declares `person`, so the
        device accepts person rules beside driver safety, and without them here
        "nobody in the seat" fired with the driver seated and "person present"
        never could (M6.5, tests/test_sim_reliability.py).
        """
        width, height = self._last_frame_size
        objects = tuple(
            Detection(
                label=label,
                confidence=(
                    self._last_overlay_confidences[i]
                    if i < len(self._last_overlay_confidences) else 1.0
                ),
                box=tuple(float(v) for v in box),
            )
            for i, (box, label) in enumerate(self._last_overlay_boxes)
        )
        people = () if any(d.label == "person" for d in objects) else tuple(
            Detection(label="person", confidence=conf, box=tuple(float(v) for v in box))
            for box, conf in zip(self._last_person_boxes, self._last_person_confidences, strict=False)
        )
        return DetectionFrame(width=width, height=height, detections=objects + people)

    def get_telemetry_snapshot(self) -> Optional[dict]:
        """Raw per-frame signal snapshot for telemetry logging (future model-training data)."""
        return {
            "ear": self._last_ear,
            "iris_openness": self._last_iris,
            "ear_threshold": self._last_ear_threshold,
            "yaw_deg": self._last_yaw,
            "pitch_deg": self._last_pitch,
            "head_yaw_deg": self._last_head_yaw,
            "head_pitch_deg": self._last_head_pitch,
            "perclos": self._last_perclos,
            "risk_score": self._last_risk_score,
            "phone_visible": bool(self._last_phone_boxes),
            "hand_count": len(self._last_hand_boxes),
            "eating_visible": bool(self._last_eating_boxes),
            "erratic_hands": self._erratic_hands_active,
            **(self._behavior.snapshot() if self._behavior is not None else {}),
        }

    def _gaze_label(self) -> str:
        """Which side of the focused-driving zone the calibrated gaze currently sits on.

        Frame-relative ("Left"/"Right"), not vehicle-relative ("driver side"/
        "passenger side"): this codebase has no config for vehicle handedness
        or which side of the dashboard the camera sits on, and guessing that
        mapping would print a confidently wrong label instead of an honestly
        neutral one. A future config key (e.g. `camera_side: driver|passenger`)
        could translate this correctly; until one exists, frame-relative is
        the only claim the available geometry actually supports.
        """
        if self._eyes_down_active or self._last_pitch < self._pitch_threshold_deg:
            return "Down"
        if self._eyes_up_active or self._last_pitch > self._pitch_up_threshold_deg:
            return "Up"
        if self._last_yaw > self._yaw_threshold_deg:
            return "Left"
        if self._last_yaw < -self._yaw_threshold_deg:
            return "Right"
        return "Forward"

    @staticmethod
    def _phase_label(condition: bool, tracker) -> str:
        """NONE (condition false) / DEVELOPING (true, not yet sustained) / CONFIRMED (active).

        Mirrors the observed/developing/confirmed vocabulary core/event_engine.py
        uses elsewhere in the product; this module doesn't run on EventEngine
        (see the module docstring), so this is the equivalent concept computed
        from the tracker this module actually uses, not a borrowed enum value.
        """
        if tracker.is_active:
            return "CONFIRMED"
        return "DEVELOPING" if condition else "NONE"

    def get_debug_state(self) -> dict:
        """Full, human-readable snapshot of why this module is (or isn't) about to alert.

        Developer/debug use only (main.py --debug) -- this is intentionally
        more detailed and more implementation-specific than get_telemetry_snapshot
        (raw numbers for CSV logging) or annotate_debug (a fixed on-frame
        overlay): it exists so "why didn't that alert" or "why did that alert"
        has a direct answer instead of a re-read of this file. Never sent
        anywhere (not to the cloud, not to end-user notification text) --
        wire it into a local-only view if one is added.
        """
        now = self._evidence_clock.now  # the trackers' clock
        phone_confidence = None
        if self._last_phone_boxes and self._last_overlay_confidences:
            phone_confidences = [
                conf for (box, label), conf in zip(
                    self._last_overlay_boxes, self._last_overlay_confidences, strict=False
                )
                if label == "cell phone"
            ]
            if phone_confidences:
                phone_confidence = max(phone_confidences)
        state = {
            "face_detected": self._last_face_proto is not None and not self._face_lost_active,
            "focused": not (
                self._distracted_active or self._eyes_down_active or self._eyes_up_active
                or self._phone_use_active or self._drowsy_active or self._face_lost_active
            ),
            "gaze": self._gaze_label(),
            "gaze_yaw_deg": round(self._last_yaw, 1),
            "gaze_pitch_deg": round(self._last_pitch, 1),
            "head_yaw_deg": round(self._last_head_yaw, 1),
            "head_pitch_deg": round(self._last_head_pitch, 1),
            "eye_gaze_offset_deg": (
                (round(self._last_gaze_offset[0], 1), round(self._last_gaze_offset[1], 1))
                if self._last_gaze_offset is not None else None
            ),
            "ear": round(self._last_ear, 3),
            "ear_threshold": round(self._last_ear_threshold, 3),
            "phone_detected": bool(self._last_phone_boxes),
            "phone_confidence": phone_confidence,
            "hand_overlaps_phone": self._hand_overlaps_phone,
            "erratic_hands_state": self._phase_label(self._hands_erratic_raw, self._erratic_hands_tracker),
            "erratic_hands_duration_s": round(self._erratic_hands_tracker.elapsed(now), 1),
            "distraction_state": self._phase_label(abs(self._last_yaw) > self._yaw_threshold_deg, self._distract_tracker),
            "distraction_duration_s": round(self._distract_tracker.elapsed(now), 1),
            "eyes_down_state": self._phase_label(self._last_pitch < self._pitch_threshold_deg, self._eyes_down_tracker),
            "eyes_down_duration_s": round(self._eyes_down_tracker.elapsed(now), 1),
            "eyes_up_state": self._phase_label(self._last_pitch > self._pitch_up_threshold_deg, self._eyes_up_tracker),
            "eyes_up_duration_s": round(self._eyes_up_tracker.elapsed(now), 1),
            "phone_use_state": self._phase_label(bool(self._last_phone_boxes), self._phone_use_tracker),
            "phone_use_duration_s": round(self._phone_use_tracker.elapsed(now), 1),
            "drowsy_state": self._phase_label(self._last_ear < self._last_ear_threshold, self._drowsy_tracker),
            "risk_score": round(self._last_risk_score, 2),
            "state": self._state_label(),
            "alert_active": self.is_alert_active(),
            "events_fired_this_frame": list(self._last_fired_event_types),
        }
        if self._behavior is not None:
            state["behavior"] = {
                **self._behavior.snapshot(),
                "assessments": {k: a.to_dict() for k, a in self._behavior.assessments().items()},
                "uncertain": self._behavior.uncertainty(),
                "last_glance": self._behavior.glance.last_glance,
            }
        return state

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
        if self._eyes_up_active:
            return "EYES_UP"
        if self._distracted_active:
            return "DISTRACTED"
        if self._erratic_hands_active:
            return "ERRATIC_HANDS"
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
        hand_color = (0, 0, 255) if self._erratic_hands_active else (255, 0, 255)
        for x1, y1, x2, y2 in self._last_hand_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), hand_color, 2)
        if self._erratic_hands_active:
            # y=270: below every debug-overlay line annotate_debug() may have
            # already drawn (gaze pitch/raw pose/eye-offset/PERCLOS/state, up
            # to y=240), so this never collides whether or not --debug is on.
            cv2.putText(
                frame, "ERRATIC HAND MOVEMENT", (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        if self._face_lost_active:
            cv2.putText(
                frame, "FACE NOT VISIBLE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        if self._object_detection_error is not None:
            cv2.putText(
                frame, "object detection: RETRYING (see terminal)", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
        return frame

    def _draw_gaze_arrow(self, frame: np.ndarray, eye_yaw: float, eye_pitch: float) -> None:
        """Debug-only: a short arrow from the face center in the eye-in-head
        offset direction -- lets a developer visually confirm the iris signal
        is tracking real eye movement, separate from the head-pose axes."""
        if self._last_face_proto is None:
            return
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in self._last_face_proto.landmark]
        ys = [lm.y * h for lm in self._last_face_proto.landmark]
        center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
        scale = max(1.0, self._gaze_scale_deg)
        tip = (
            int(center[0] + (eye_yaw / scale) * 80),
            int(center[1] + (eye_pitch / scale) * 80),
        )
        cv2.arrowedLine(frame, center, tip, (255, 0, 255), 2, tipLength=0.3)

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
            frame, f"Gaze pitch: {self._last_pitch:.1f} deg (head {self._last_head_pitch:.1f})",
            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            frame,
            f"Raw head yaw/pitch: {self._last_yaw_raw:.1f} / {self._last_pitch_raw:.1f} deg "
            f"(baseline {self._calibrator.baseline_yaw:.1f} / {self._calibrator.baseline_pitch:.1f})",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        if self._last_gaze_offset is not None:
            eye_yaw, eye_pitch = self._last_gaze_offset
            cv2.putText(
                frame, f"Eye-in-head offset: {eye_yaw:+.1f} / {eye_pitch:+.1f} deg",
                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )
            self._draw_gaze_arrow(frame, eye_yaw, eye_pitch)
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

    #: Conditions tracked as live episodes, and the flag that says each is active.
    _EPISODE_FLAGS: ClassVar[dict[str, str]] = {
        "drowsiness": "_drowsy_active",
        "distraction": "_distracted_active",
        "eyes_down": "_eyes_down_active",
        "eyes_up": "_eyes_up_active",
        "phone_use": "_phone_use_active",
        "attention_lost": "_face_lost_active",
        "erratic_hand_movement": "_erratic_hands_active",
    }

    def alert_episode_types(self) -> frozenset[str]:
        """Every condition with a sustained tracker, plus sustained high risk."""
        return frozenset({*self._EPISODE_FLAGS, "high_risk"})

    def active_alert_conditions(self) -> set[str]:
        """The tracked conditions that are active now (the same flags is_alert_active reads)."""
        active = {etype for etype, flag in self._EPISODE_FLAGS.items() if getattr(self, flag, False)}
        if self._risk_fusion.is_high_risk_active:
            active.add("high_risk")
        return active

    #: What each event this module raises depends on (core/monitoring_state.py).
    _FACE = frozenset({"face"})
    _OBJECTS = frozenset({"objects"})
    _EVENT_REQUIREMENTS: ClassVar[dict[str, frozenset[str]]] = {
        **dict.fromkeys(("drowsiness", "fatigue_warning", "distraction", "frequent_glancing",
                         "eyes_down", "eyes_up", "attention_lost", "elevated_risk", "high_risk"), _FACE),
        "phone_detected": _OBJECTS,
        "eating_drinking": _OBJECTS,
        "phone_use": _FACE | _OBJECTS,
        "erratic_hand_movement": frozenset({"hands", "objects"}),
    }

    def event_requirements(self) -> dict[str, frozenset[str]]:
        reqs = dict(self._EVENT_REQUIREMENTS)
        if self._behavior is not None:
            reqs.update(dict.fromkeys(self._BEHAVIOR_V2_EVENTS, self._FACE))
        if not self._enable_hands:
            reqs.pop("erratic_hand_movement")
        return reqs

    def capability_status(self) -> dict[str, Capability]:
        """Face landmarks run on every frame (a FaceMesh failure raises out of
        process(), which the runtime reports); the detector is reported as it is."""
        if self._object_detection_error is not None:
            wait = max(0.0, (self._object_detection_retry_at or 0.0) - time.monotonic())
            objects = Capability(IMPAIRED, f"object detection failed {self._object_detection_failures}x "
                                           f"({self._object_detection_error}); retrying in {wait:.0f}s")
        elif not self._object_detection_ready:
            objects = Capability(STARTING, "object detector loads on the first detection frame")
        else:
            objects = Capability()
        status = {"face": Capability(), "objects": objects}
        if self._enable_hands:
            # Hand landmarks are their own model, loaded with the detector and
            # run on its ticks; the shared engine (core/perception.py) reports
            # whether they actually ran. Never a copy of the detector's state.
            status["hands"] = (Capability() if self._hands is not None
                               else Capability(STARTING, "hand landmarks load with the object detector"))
        return status

    def perception_signals(self) -> dict:
        """Eye state, head pose and gaze measured on the last frame (core/perception.py
        FaceObservation); a key is absent when it was not computed on that frame."""
        return dict(self._frame_signals)

    def available_labels(self) -> frozenset[str]:
        """The detector class names this module asks for (hands are not detections).

        `yolo_classes` names are the detector's own labels, which is what
        get_detections() reports; person is always requested (face-loss fallback).
        """
        return frozenset({*self._yolo_class_map, "person"})

    def is_alert_active(self) -> bool:
        """True while any currently-active condition warrants the full-screen alert flash."""
        return (
            self._drowsy_active
            or self._distracted_active
            or self._eyes_down_active
            or self._eyes_up_active
            or self._phone_use_active
            or self._face_lost_active
            or self._erratic_hands_active
            or self._risk_fusion.is_high_risk_active
        )
