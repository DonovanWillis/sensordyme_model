"""DashSentinel (github.com/alec-kr/DashSentinel), vendored for head-to-head comparison.

The full repository is vendored under third_party/DashSentinel (commit 3874e05b,
2026-05-20) and this adapter imports its REAL pipeline modules — nothing is
reimplemented. Per frame it runs exactly what their app.py runs, in the same order,
with their CLI defaults:

  1. resize to their 640x360 processing size
  2. `measure_frame_quality` (brightness/contrast/blur gate — poor frames can never
     produce a high-confidence drowsiness claim)
  3. `enhance_lighting` (CLAHE + gamma + bilateral filter)
  4. MediaPipe FaceMesh (up to 3 faces) -> `choose_driver_face` (largest/most central)
  5. `FeatureExtractor.extract` — EAR, MAR/yawn, blink rate, head pose, all gated on
     per-landmark-group visibility
  6. `AdaptiveScorer.score` — weighted component sum compared against a PERSISTENT
     per-driver profile (data/dashsentinel_profile.json, adapting across sessions),
     with hysteresis (status must hold ~10 frames to change) and a 25s calibration

Their statuses map to events on each transition: DROWSY (loud alarm), WARNING
(chime), NO FACE (chime), VISION DEGRADED (chime). The original drives ESP8266
LEDs/OLED over serial — that hardware layer is the only part not used; alerts go
through the framework's sound/console sinks instead.

Run it:  make dev-dashsentinel
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core.events import Event
from modules.base import VisionModule, register_module

# Make the vendored repo importable (it is laid out as a `src` package with
# relative imports). Path insertion is scoped to this module's import.
_DASHSENTINEL_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "DashSentinel"
if str(_DASHSENTINEL_ROOT) not in sys.path:
  sys.path.insert(0, str(_DASHSENTINEL_ROOT))

from src.face_selection import choose_driver_face  # noqa: E402
from src.features import FeatureExtractor, enhance_lighting  # noqa: E402
from src.profile import DriverProfile  # noqa: E402
from src.scoring import AdaptiveScorer  # noqa: E402
from src.vision_quality import measure_frame_quality  # noqa: E402

# status -> (event type, alert-worthy) on each transition into the status
_STATUS_EVENTS = {
    "DROWSY": "dashsentinel_drowsy",
    "WARNING": "dashsentinel_warning",
    "NO FACE": "dashsentinel_no_face",
    "VISION DEGRADED": "dashsentinel_vision_degraded",
}
_STATUS_COLORS = {
    "ALERT": (0, 200, 0),
    "LEARNING BASELINE": (0, 255, 255),
    "WARNING": (0, 140, 255),
    "DROWSY": (0, 0, 255),
    "NO FACE": (140, 160, 160),
    "VISION DEGRADED": (140, 160, 160),
}


@register_module("driver_attention_dashsentinel")
class DriverAttentionDashSentinelModule(VisionModule):
  """DashSentinel's real feature/scoring pipeline behind the standard module interface."""

  name = "driver_attention_dashsentinel"

  def __init__(self, config: dict) -> None:
    super().__init__(config)

    # their persistent, cross-session driver profile (their default: data/driver_profile.json)
    self._profile = DriverProfile(config.get("profile_path", "data/dashsentinel_profile.json"))
    if bool(config.get("rebuild_baseline_on_start", False)):
      self._profile.reset_baseline()
    self._profile.begin_session()

    # their CLI defaults (src/cli.py), overridable from config.yaml
    self._extractor = FeatureExtractor(
        ear_threshold=float(config.get("ear_threshold", 0.23)),
        yawn_mar_threshold=float(config.get("yawn_mar_threshold", 0.45)),
        yawn_frames_threshold=int(config.get("yawn_frames_threshold", 12)),
    )
    self._scorer = AdaptiveScorer(
        profile=self._profile,
        calibration_seconds=int(config.get("calibration_seconds", 25)),
        attention_window=int(config.get("attention_window", 240)),
        warning_threshold=float(config.get("warning_threshold", 0.42)),
        drowsy_threshold=float(config.get("drowsy_threshold", 0.58)),
        status_hold_frames=int(config.get("status_hold_frames", 10)),
        no_face_hold_frames=int(config.get("no_face_hold_frames", 18)),
        low_quality_hold_frames=int(config.get("low_quality_hold_frames", 8)),
        min_model_confidence=float(config.get("min_model_confidence", 0.55)),
    )
    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=int(config.get("max_faces", 3)),
        refine_landmarks=bool(config.get("refine_landmarks", False)),
        min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
    )
    self._proc_width = int(config.get("width", 640))
    self._proc_height = int(config.get("height", 360))
    self._save_profile_every_s = float(config.get("save_profile_every_seconds", 15))
    self._last_profile_save = time.monotonic()

    # display / event state
    self._last_score = None
    self._last_features: Optional[dict] = None
    self._last_bbox = None
    self._prev_status = "LEARNING BASELINE"
    self._crop_map = (0, 0, 1.0, 1.0)  # (x_offset, y_offset, x_scale, y_scale) proc -> display

    print(
        "driver_attention_dashsentinel: DashSentinel pipeline ready (vendored from "
        "github.com/alec-kr/DashSentinel; profile persists across sessions at "
        f"{self._profile.path}). ~25s adaptive calibration starts now — look ahead normally."
    )

  def process(self, frame: np.ndarray) -> list[Event]:
    events: list[Event] = []

    # Their app captures natively at 640x360 (16:9); this framework's camera delivers
    # 4:3. Center-crop to their aspect BEFORE resizing — a squashing resize distorts
    # EAR (a vertical/horizontal ratio) and shrinks the face area that their
    # visibility gating requires, which starves calibration of "good frames".
    disp_h, disp_w = frame.shape[:2]
    target_aspect = self._proc_width / self._proc_height
    crop_x, crop_y, crop_w, crop_h = 0, 0, disp_w, disp_h
    if disp_w / disp_h < target_aspect:  # too tall -> crop rows (e.g. 640x480 -> 640x360)
      crop_h = int(round(disp_w / target_aspect))
      crop_y = (disp_h - crop_h) // 2
    elif disp_w / disp_h > target_aspect:  # too wide -> crop columns
      crop_w = int(round(disp_h * target_aspect))
      crop_x = (disp_w - crop_w) // 2
    self._crop_map = (crop_x, crop_y, crop_w / self._proc_width, crop_h / self._proc_height)
    frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    # their app.py order: resize -> quality gate on RAW frame -> enhance -> FaceMesh
    frame = cv2.resize(frame, (self._proc_width, self._proc_height), interpolation=cv2.INTER_AREA)
    frame_quality = measure_frame_quality(frame)
    enhanced = enhance_lighting(frame)

    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    results = self._face_mesh.process(rgb)

    if results.multi_face_landmarks:
      h, w = enhanced.shape[:2]
      face_selection = choose_driver_face(results.multi_face_landmarks, w, h)
      if face_selection is None:
        self._extractor.reset()
        score = self._scorer.update_no_face()
        self._last_bbox = None
      else:
        features = self._extractor.extract(
            face_selection.landmarks,
            w,
            h,
            frame_quality=frame_quality,
            face_selection=face_selection,
            model_output=None,  # no optional local ONNX model, as in their default run
        )
        score = self._scorer.score(features)
        self._last_features = features
        self._last_bbox = face_selection.bbox
    else:
      score = self._scorer.update_no_face()
      self._last_bbox = None

    self._last_score = score

    # one event per status transition, carrying their own confidence + reason
    if score.status != self._prev_status and score.status in _STATUS_EVENTS:
      events.append(
          Event.now(
              _STATUS_EVENTS[score.status],
              0.0,
              float(score.confidence),
              self.name,
              f"drowsy_score={score.drowsy_score:.2f} attentiveness={score.attentiveness:.0f}%; {score.reason}",
          )
      )
    self._prev_status = score.status

    # their periodic profile persistence (app.py saves every ~15s)
    now = time.monotonic()
    if now - self._last_profile_save >= self._save_profile_every_s:
      self._profile.save()
      self._last_profile_save = now

    return events

  def is_alert_active(self) -> bool:
    return self._last_score is not None and self._last_score.status == "DROWSY"

  def get_telemetry_snapshot(self) -> Optional[dict]:
    if self._last_score is None or self._last_features is None:
      return None
    f = self._last_features
    return {
        "ear": f.get("ear", 0.0),
        "yaw_deg": f.get("yaw_ratio", 0.0) * 45.0,  # their yaw is a ratio; scaled for the shared column
        "pitch_deg": f.get("pitch_ratio", 0.0) * 45.0,
        "perclos": f.get("closed_frames_norm", 0.0),
        "risk_score": self._last_score.drowsy_score,
        "phone_visible": False,  # DashSentinel has no object detection
        "hand_count": 0,
        "eating_visible": False,
    }

  def annotate(self, frame: np.ndarray) -> np.ndarray:
    score = self._last_score
    if score is None:
      return frame
    h, w = frame.shape[:2]
    color = _STATUS_COLORS.get(score.status, (255, 255, 255))

    # map their processing-resolution bbox back onto the (cropped) display frame
    if self._last_bbox is not None:
      ox, oy, sx, sy = self._crop_map
      x1, y1, x2, y2 = self._last_bbox
      cv2.rectangle(
          frame,
          (int(ox + x1 * sx), int(oy + y1 * sy)),
          (int(ox + x2 * sx), int(oy + y2 * sy)),
          color, 2,
      )

    status_line = score.status
    if score.status == "LEARNING BASELINE":
      # calibration needs BOTH 25s elapsed AND 150 profile updates from clearly-visible
      # frames — show progress so a stalled calibration is diagnosable on screen
      status_line += f"  ({self._profile.total_updates}/150 good frames)"
    cv2.putText(frame, status_line, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(
        frame,
        f"attentiveness {score.attentiveness:.0f}%   drowsy {score.drowsy_score:.2f}",
        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )
    # attentiveness bar (their OLED shows this as a percentage)
    frac = max(0.0, min(1.0, score.attentiveness / 100.0))
    cv2.rectangle(frame, (10, 76), (210, 90), (90, 90, 90), 1)
    cv2.rectangle(frame, (10, 76), (10 + int(200 * frac), 90), color, -1)
    return frame

  def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
    frame = self.annotate(frame)
    score, f = self._last_score, self._last_features
    if score is None:
      return frame
    lines = [f"DashSentinel | phase {score.phase} | reason: {score.reason[:70]}"]
    if f is not None:
      lines += [
          f"EAR {f['ear']:.3f} MAR {f['mar']:.3f} blinks/min {f['blink_rate']:.0f} yawns {f['yawn_count']}",
          f"roll {f['roll_deg']:+.1f} yaw_ratio {f['yaw_ratio']:+.2f} pitch_ratio {f['pitch_ratio']:.2f}",
          f"quality {f['frame_quality']:.2f} ({f['frame_quality_reason'][:40]}) visibility {f['visibility_score']:.2f}",
          f"profile updates {self._profile.total_updates} (sessions {self._profile.sessions})",
      ]
    y = frame.shape[0] - 20 * len(lines) - 10
    for line in lines:
      cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      y += 20
    return frame
