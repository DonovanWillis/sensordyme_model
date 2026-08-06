"""Adaptive scoring system for drowsiness detection"""

import time
from collections import deque
from dataclasses import dataclass

from .utils import clamp


@dataclass
class ScoreOutput:
    """holds scores and related info for a single frame, returned by AdaptiveScorer"""

    phase: str
    status: str
    confidence: float
    drowsy_score: float
    attentiveness: float
    reason: str


# pylint: disable=too-many-instance-attributes
class AdaptiveScorer:
    """main class for scoring drowsiness based on profile and current features"""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        profile,
        calibration_seconds=25,
        attention_window=240,
        warning_threshold=0.42,
        drowsy_threshold=0.58,
        status_hold_frames=12,
        no_face_hold_frames=20,
        low_quality_hold_frames=8,
        min_model_confidence=0.55,
    ):
        self.profile = profile
        self.calibration_seconds = calibration_seconds
        self.warning_threshold = warning_threshold
        self.drowsy_threshold = drowsy_threshold
        self.status_hold_frames = status_hold_frames
        self.no_face_hold_frames = no_face_hold_frames
        self.low_quality_hold_frames = low_quality_hold_frames
        self.min_model_confidence = min_model_confidence

        self.start_time = time.time()

        # keep recent attentiveness scores to smooth out the output and avoid jitter/spikes
        self.attention_history = deque(maxlen=attention_window)
        self.score_history = deque(maxlen=30)

        # clear history and seed with neutral values for a smooth start
        self.state = "CALIBRATING"
        self.pending_state = None
        self.pending_frames = 0
        self.no_face_frames = 0
        self.low_quality_frames = 0

    def seconds_since_start(self):
        """counts seconds since start to manage calibration phase"""
        return time.time() - self.start_time

    def in_calibration(self):
        """determine if we're still in the initial calibration phase"""
        return (
            self.seconds_since_start() < self.calibration_seconds
            or self.profile.total_updates < 150
        )

    def _normalized_delta(self, value, mean, std, floor):
        """helper to compute normalized delta for head pose features"""
        return clamp(abs(value - mean) / max(2.5 * std, floor), 0.0, 1.0)

    def _build_reason(self, components, features):
        """produce a reason for current score based on significant contributing factors"""
        notes = []
        if features.get("face_count", 1) > 1:
            notes.append(
                f"multiple faces detected; selected face {features.get('selected_face_index', 0)}"
            )
        if features.get("eyes_visible", 1.0) < 0.5:
            notes.append("eyes not visible; using pose/model only")
        if features.get("mouth_visible", 1.0) < 0.5:
            notes.append("mouth not visible; yawn disabled")
        if features.get("model_label") not in (None, "unavailable"):
            notes.append(f"local model: {features.get('model_label')}")

        ranked = sorted(components.items(), key=lambda item: item[1], reverse=True)
        top = [name for name, score in ranked if score >= 0.18][:2]
        notes.extend(top)
        if not notes:
            return "normal behavior"
        return "; ".join(notes[:3])

    def _promote_state(self, desired_state):
        """requires a certain number of consecutive frames to transition to a new state"""
        if desired_state == self.state:
            self.pending_state = None
            self.pending_frames = 0
            return

        if self.pending_state != desired_state:
            self.pending_state = desired_state
            self.pending_frames = 1
            if self.status_hold_frames <= 1:
                self.state = desired_state
                self.pending_state = None
                self.pending_frames = 0
            return

        self.pending_frames += 1
        if self.pending_frames >= self.status_hold_frames:
            self.state = desired_state
            self.pending_state = None
            self.pending_frames = 0

    def _last_attentiveness(self):
        return (
            sum(self.attention_history) / len(self.attention_history)
            if self.attention_history
            else 100.0
        )

    def _status_output(self, status, confidence, reason, phase="ACTIVE"):
        return ScoreOutput(
            phase=phase,
            status=status,
            confidence=float(confidence),
            drowsy_score=self.score_history[-1] if self.score_history else 0.0,
            attentiveness=float(self._last_attentiveness()),
            reason=reason,
        )

    def update_no_face(self):
        """if no face is detected within n frames, switch to NO_FACE state"""
        self.no_face_frames += 1
        if self.no_face_frames >= self.no_face_hold_frames:
            self.state = "NO_FACE"
            self.pending_state = None
            self.pending_frames = 0

        return ScoreOutput(
            phase="ACTIVE" if not self.in_calibration() else "CALIBRATING",
            status="NO FACE",
            confidence=0.0,
            drowsy_score=self.score_history[-1] if self.score_history else 0.0,
            attentiveness=float(self._last_attentiveness()),
            reason="face not visible",
        )

    def _quality_gate(self, features):
        """Return a ScoreOutput when frame quality is too poor for a reliable claim."""
        frame_usable = features.get("frame_usable", 1.0) >= 0.5
        visibility_score = features.get("visibility_score", 1.0)
        if frame_usable and visibility_score >= 0.35:
            self.low_quality_frames = max(0, self.low_quality_frames - 2)
            return None

        self.low_quality_frames += 1
        reason = features.get("frame_quality_reason", "low visibility")
        if visibility_score < 0.35:
            reason = f"low face/landmark visibility; {reason}"

        if self.low_quality_frames >= self.low_quality_hold_frames:
            self.state = "VISION DEGRADED"
            self.pending_state = None
            self.pending_frames = 0
            status = "VISION DEGRADED"
        elif self.state in ("DROWSY", "WARNING"):
            status = self.state
            reason = f"holding previous status until clearer frame; {reason}"
        elif self.state == "CALIBRATING":
            status = "LEARNING BASELINE"
            reason = f"waiting for clearer calibration frame; {reason}"
        else:
            status = "ALERT"
            reason = f"waiting for clearer frame; {reason}"

        return self._status_output(
            status=status,
            confidence=1.0 - clamp(features.get("frame_quality", 0.0), 0.0, 1.0),
            reason=reason,
            phase="ACTIVE" if not self.in_calibration() else "CALIBRATING",
        )

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def score(self, features: dict) -> ScoreOutput:
        """main scoring function that compares current features to the profile"""
        self.no_face_frames = 0

        quality_output = self._quality_gate(features)
        if quality_output is not None:
            return quality_output

        eyes_visible = features.get("eyes_visible", 1.0) >= 0.5
        mouth_visible = features.get("mouth_visible", 1.0) >= 0.5

        # compute the various components of the drowsiness score.
        # Eye and mouth components are explicitly gated so occlusion does not look like drowsiness.
        if eyes_visible:
            ear_drop = clamp(
                (self.profile.mean("ear") - features["ear"])
                / max(2.2 * self.profile.std("ear"), 0.03),
                0.0,
                1.0,
            )
            low_blink = clamp(
                (self.profile.mean("blink_rate") - features["blink_rate"])
                / max(2.0 * self.profile.std("blink_rate"), 4.0),
                0.0,
                1.0,
            )
            eye_component = 0.28 * ear_drop + 0.20 * features["closed_frames_norm"]
            blink_component = 0.08 * low_blink
        else:
            eye_component = 0.0
            blink_component = 0.0

        if mouth_visible:
            yawn_mag = clamp(
                (
                    features["mar"]
                    - max(self.profile.mean("mar") + 2.0 * self.profile.std("mar"), 0.34)
                )
                / 0.22,
                0.0,
                1.0,
            )
            yawn_component = 0.10 * yawn_mag + 0.08 * features["yawn_flag"]
        else:
            yawn_component = 0.0

        roll_delta = 1.0 if abs(features["roll_deg"]) > 15.0 else 0.0
        yaw_delta = 1.0 if abs(features["yaw_ratio"]) > 0.8 else 0.0
        pitch_delta = (
            1.0
            if features["pitch_ratio"] < 0.20 or features["pitch_ratio"] > 0.85
            else 0.0
        )
        look_away_duration = features.get("look_away_norm", 0.0)
        head_tilt_duration = features.get("head_tilt_norm", 0.0)
        head_back_duration = features.get("head_back_norm", 0.0)
        bad_pose_duration = features.get("bad_pose_norm", 0.0)

        pose_component = (
            0.10 * roll_delta
            + 0.12 * yaw_delta
            + 0.10 * pitch_delta
            + 0.12 * look_away_duration
            + 0.08 * head_tilt_duration
            + 0.08 * head_back_duration
            + 0.06 * bad_pose_duration
        )

        model_component = 0.0
        model_score = features.get("model_drowsy_score")
        model_confidence = features.get("model_confidence", 0.0)
        if model_score is not None and model_confidence >= self.min_model_confidence:
            model_component = 0.30 * float(model_score) * float(model_confidence)

        components = {
            "eye closure": eye_component,
            "low blink rate": blink_component,
            "yawning": yawn_component,
            "looking away/head pose": pose_component,
            "local model": model_component,
        }

        drowsy_score = clamp(sum(components.values()), 0.0, 1.0)
        self.score_history.append(drowsy_score)
        smoothed_score = sum(self.score_history) / len(self.score_history)

        # pylint: disable=chained-comparison
        stable_alert = (
            smoothed_score < 0.24
            and features.get("frame_usable", 1.0) >= 0.5
            and features.get("visibility_score", 1.0) >= 0.70
            and eyes_visible
            and mouth_visible
            and features["yawn_flag"] < 0.5
            and features["posture_flag"] < 0.5
            and features["closed_frames_norm"] < 0.15
            and features.get("bad_pose_norm", 0.0) < 0.12
        )

        # update the profile only when the image and landmarks are good enough.
        if (self.in_calibration() and stable_alert) or stable_alert:
            self.profile.update_from_alert_frame(features)

        if self.in_calibration():
            self.state = "CALIBRATING"
            confidence = clamp(
                0.50 + 0.50 * (self.seconds_since_start() / max(self.calibration_seconds, 1)),
                0.0,
                1.0,
            )
            reason = "learning normal baseline"
        else:
            effective_drowsy_threshold = self.drowsy_threshold
            if not eyes_visible or not mouth_visible:
                effective_drowsy_threshold += 0.06
            if features.get("face_count", 1) > 1:
                effective_drowsy_threshold += 0.03

            if smoothed_score >= effective_drowsy_threshold:
                desired_state = "DROWSY"
            elif smoothed_score >= self.warning_threshold:
                desired_state = "WARNING"
            else:
                desired_state = "ALERT"

            if self.state in ("CALIBRATING", "NO_FACE", "VISION DEGRADED"):
                self.state = "ALERT"
            self._promote_state(desired_state)

            confidence = (
                smoothed_score
                if self.state in ("WARNING", "DROWSY")
                else 1.0 - smoothed_score
            )
            reason = self._build_reason(components, features)

        attentiveness = (1.0 - smoothed_score) * 100.0
        self.attention_history.append(attentiveness)
        attentiveness_smoothed = sum(self.attention_history) / len(self.attention_history)

        phase = "CALIBRATING" if self.state == "CALIBRATING" else "ACTIVE"
        status = "LEARNING BASELINE" if self.state == "CALIBRATING" else self.state

        return ScoreOutput(
            phase=phase,
            status=status,
            confidence=float(confidence),
            drowsy_score=float(smoothed_score),
            attentiveness=float(attentiveness_smoothed),
            reason=reason,
        )

    def reset_stats(self):
        """reset score history and state machine"""
        self.attention_history.clear()
        self.score_history.clear()

        # seed clean history so attentiveness immediately returns to 100
        self.attention_history.append(100.0)
        self.score_history.append(0.0)

        # force state machine back to alert
        self.state = "ALERT"
        self.pending_state = None
        self.pending_frames = 0
        self.no_face_frames = 0
        self.low_quality_frames = 0
