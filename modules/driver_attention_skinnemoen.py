"""erlendskinnemoen's MediaPipe FaceMesh DMS, vendored for head-to-head comparison.

Source: github.com/erlendskinnemoen/Driver-Monitoring-Systems-using-Google-s-MediaPipe-FaceMesh
(MIT licensed; a university course project — "TAV assignment 1"). Added as a third
runnable option so its accuracy can be compared against this repo's own pipeline
(`make dev`/`make dev-max`) and openpilot (`make dev-openpilot`) on the same camera
and the same events.csv.

The algorithm is reproduced FAITHFULLY from their driver_surveillance.py — same
landmarks, same math, same constants, same quirks (which are part of what's being
evaluated):

  * EAR computed as eye height/width from 4 landmarks per eye; the drowsiness
    threshold is calibrated at 80% of the way from your open-eye EAR to your
    closed-eye EAR.
  * Three timed calibration phases from launch: 0-3s eyes open looking forward,
    3-6s eyes CLOSED, 6-8s "calibration complete".
  * Head pose from solvePnP over 6 landmarks (their camera matrix — including its
    swapped principal-point coordinates — is kept as-is), plus iris-offset gaze
    scaled by their 30/45 factors (including the left/right asymmetry in the
    original conditionals).
  * DISTRACTED: |eye pitch + calibrated head pitch| > 30 or same for yaw — raw
    per-frame, no temporal debounce (the original just draws text every frame).
  * Drowsy: PERCLOS > 0.8 over a ~10s rolling frame window -> "drowsiness" event
    (which the framework sounds loudly, matching the original's beep).

Framework adaptations: the camera loop, beep, and cv2.imshow are replaced by
process()/annotate()/EventLogger; the original's `time.sleep(1/25)` frame-rate cap
is dropped (main.py's camera already paces the loop).

ADDITIONS on top of the original (config-toggled, clearly marked below):
  * Every alert is audible: `skinnemoen_distracted` (silent text in the original)
    and `skinnemoen_phone_use` sound the loud alarm; calibration-complete chimes
    softly (the original beeped there too).
  * Phone detection (the original has none): YOLO spots a phone in the frame; if
    the driver is simultaneously looking at it (the system's own DISTRACTED
    gaze/head-angle condition), `skinnemoen_phone_use` fires immediately. A
    mounted phone with eyes forward stays silent.

Run it:  make dev-skinnemoen
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core.detectors import Box, get_yolo_model
from core.events import Event
from modules.base import VisionModule, register_module

# Their landmark indices (driver_surveillance.py)
_RIGHT_EYE = {"right": 33, "bottom": 145, "left": 133, "top": 159, "iris": 468}
_LEFT_EYE = {"right": 362, "bottom": 374, "left": 263, "top": 386, "iris": 473}
_POSE_IDS = (33, 263, 1, 61, 291, 199)

# Their thresholds/constants
_CALIB_OPEN_END_S = 3.0
_CALIB_CLOSED_END_S = 6.0
_CALIB_DONE_END_S = 8.0
_EAR_CLOSED_FRACTION = 0.8  # threshold = open - (open - closed) * 0.8
_ANGLE_THRESHOLD_DEG = 30.0
_PERCLOS_WINDOW_S = 10.0
_PERCLOS_THRESHOLD = 0.8


@register_module("driver_attention_skinnemoen")
class DriverAttentionSkinnemoenModule(VisionModule):
  """The MediaPipe FaceMesh course-project DMS, running unmodified logic."""

  name = "driver_attention_skinnemoen"

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # calibration state (theirs is module-lifetime globals; calibrates every run)
    self._start_capture = time.monotonic()
    self._ear_open_samples: list[float] = []
    self._ear_closed_samples: list[float] = []
    self._pitch_samples: list[float] = []
    self._yaw_samples: list[float] = []
    self._threshold_ear = 0.0
    self._pitch_baseline = 0.0
    self._yaw_baseline = 0.0
    self._calibration_beeped = False

    # detection state
    self._frames: list[int] = []  # their PERCLOS ring (1 = eyes closed this frame)
    self._last_process_t: Optional[float] = None
    self._fps = 0.0
    self._distracted = False
    self._distracted_prev = False
    self._drowsy = False
    self._drowsy_prev = False

    # --- ADDITIONS (not in the original repo): phone detection ---
    self._phone_enabled = bool(config.get("phone_detection", True))
    self._phone_weights = config.get("phone_weights", "yolo11n.pt")
    self._phone_device = config.get("phone_device", "mps")
    self._phone_every_n_frames = max(1, int(config.get("phone_every_n_frames", 3)))
    self._phone_conf = float(config.get("phone_confidence_threshold", 0.35))
    self._phone_model = None
    self._phone_class_id: Optional[int] = None
    self._phone_error: Optional[str] = None
    self._frame_count = 0
    self._last_phone_boxes: list[Box] = []
    self._phone_visible = False
    self._phone_use = False
    self._phone_use_prev = False

    # display state
    self._time_elapsed = 0.0
    self._last_ear = 0.0
    self._last_perclos = 0.0
    self._last_face_pitch = 0.0
    self._last_face_yaw = 0.0
    self._draw_points: list[tuple[int, int, tuple[int, int, int], int]] = []
    self._nose_line: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
    self._face_found = False

    print(
        "driver_attention_skinnemoen: MediaPipe FaceMesh DMS (github.com/erlendskinnemoen, vendored). "
        "Timed calibration starts NOW: 3s eyes open looking forward, then 3s eyes CLOSED."
    )

  def _update_phone_detection(self, frame: np.ndarray) -> None:
    """ADDITION: YOLO phone detection every Nth frame (the original has none)."""
    if not self._phone_enabled or self._phone_error is not None:
      return
    self._frame_count += 1
    if self._frame_count % self._phone_every_n_frames != 0:
      return
    try:
      if self._phone_model is None:
        self._phone_model = get_yolo_model(self._phone_weights, device=self._phone_device)
        self._phone_class_id = next(
            cid for cid, cname in self._phone_model.names.items() if cname == "cell phone"
        )
        print(
            f"driver_attention_skinnemoen: phone detection ready "
            f"(weights={self._phone_weights}, device={self._phone_device})"
        )
      results = self._phone_model.predict(
          frame, verbose=False, classes=[self._phone_class_id], conf=self._phone_conf,
          device=self._phone_device,
      )
      self._last_phone_boxes = [
          tuple(int(v) for v in box.xyxy[0].tolist()) for result in results for box in result.boxes
      ]
      self._phone_visible = bool(self._last_phone_boxes)
    except Exception as exc:  # noqa: BLE001 - never crash the video loop
      self._phone_error = str(exc)
      print(f"driver_attention_skinnemoen: phone detection disabled after an error: {exc}")

  def _check_phone_use(self) -> Optional[Event]:
    """ADDITION: phone in frame + driver looking at it (their DISTRACTED condition)
    -> immediate distracted alert, on the rising edge."""
    self._phone_use = self._phone_visible and self._distracted
    fired = self._phone_use and not self._phone_use_prev
    self._phone_use_prev = self._phone_use
    if fired:
      return Event.now(
          "skinnemoen_phone_use", 0.0, 0.8, self.name, "phone_in_frame + looking_at_it"
      )
    return None

  def process(self, frame: np.ndarray) -> list[Event]:
    now = time.monotonic()
    if self._last_process_t is not None and now > self._last_process_t:
      self._fps = 1.0 / (now - self._last_process_t)  # their fps = 1 / loop time
    self._last_process_t = now
    self._time_elapsed = now - self._start_capture
    events: list[Event] = []
    self._update_phone_detection(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self._face_mesh.process(rgb)
    self._draw_points = []
    self._nose_line = None
    self._face_found = bool(results.multi_face_landmarks)
    if not self._face_found:
      return events  # original skips the whole frame when no face is found

    frame_height, frame_width = frame.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    pt = lambda i: (landmarks[i].x * frame_width, landmarks[i].y * frame_height)  # noqa: E731

    # their camera matrix — principal point coordinates swapped, kept verbatim
    focal_length = 1 * frame_width
    cam_matrix = np.array(
        [[focal_length, 0, frame_height / 2], [0, focal_length, frame_width / 2], [0, 0, 1]]
    )
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    face_2d, face_3d = [], []
    for index in _POSE_IDS:
      x, y = int(landmarks[index].x * frame_width), int(landmarks[index].y * frame_height)
      face_2d.append((x, y))
      face_3d.append((x, y, landmarks[index].z))
      if index == 1:
        nose_2d = pt(1)
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    ok, rotation_vector, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not ok:
      return events
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    face_pitch = float(np.rad2deg(euler_angles[0]))
    face_yaw = float(-np.rad2deg(euler_angles[1]))
    self._last_face_pitch, self._last_face_yaw = face_pitch, face_yaw

    # eye geometry, EAR as height/width (their formula)
    r = {k: pt(i) for k, i in _RIGHT_EYE.items()}
    l = {k: pt(i) for k, i in _LEFT_EYE.items()}
    left_eye_width = l["left"][0] - l["right"][0]
    left_eye_height = l["bottom"][1] - l["top"][1]
    left_eye_center = [(l["left"][0] + l["right"][0]) / 2, (l["bottom"][1] + l["top"][1]) / 2]
    right_eye_width = r["left"][0] - r["right"][0]
    right_eye_height = r["bottom"][1] - r["top"][1]
    right_eye_center = [(r["left"][0] + r["right"][0]) / 2, (r["bottom"][1] + r["top"][1]) / 2]

    right_gaze = np.array([r["iris"][0] - right_eye_center[0], r["iris"][1] - right_eye_center[1]])
    left_gaze = np.array([l["iris"][0] - left_eye_center[0], l["iris"][1] - left_eye_center[1]])

    # their gaze scaling, including the original left/right conditional asymmetry
    if right_eye_height > 0:
      pitch_right_eye = (right_gaze[1] / (right_eye_height / 2)) * 30
    else:
      pitch_right_eye = (right_gaze[1] / (right_eye_height / 2)) * 45
    yaw_right_eye = (right_gaze[0] / (right_eye_width / 2)) * 45
    if left_gaze[1] > 0:
      pitch_left_eye = (left_gaze[1] / (left_eye_height / 2)) * 30
    else:
      pitch_left_eye = (left_gaze[1] / (left_eye_height / 2)) * 45
    yaw_left_eye = (left_gaze[0] / (left_eye_width / 2)) * 45

    right_eye_EAR = right_eye_height / right_eye_width
    left_eye_EAR = left_eye_height / left_eye_width
    EAR = (right_eye_EAR + left_eye_EAR) / 2
    self._last_ear = EAR

    # overlay bookkeeping (their red landmark dots, iris/eye centers, nose line)
    for k in ("right", "bottom", "left", "top"):
      self._draw_points.append((int(r[k][0]), int(r[k][1]), (0, 0, 255), 5))
      self._draw_points.append((int(l[k][0]), int(l[k][1]), (0, 0, 255), 5))
    self._draw_points.append((int(l["iris"][0]), int(l["iris"][1]), (0, 255, 0), 3))
    self._draw_points.append((int(left_eye_center[0]), int(left_eye_center[1]), (128, 128, 128), 2))
    self._draw_points.append((int(r["iris"][0]), int(r["iris"][1]), (0, 0, 255), 3))
    self._draw_points.append((int(right_eye_center[0]), int(right_eye_center[1]), (128, 128, 128), 2))
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + face_yaw * 10), int(nose_2d[1] - face_pitch * 10))
    self._nose_line = (p1, p2)

    # --- their timed calibration state machine ---
    if self._time_elapsed < _CALIB_OPEN_END_S:
      self._ear_open_samples.append(EAR)
      self._pitch_samples.append(face_pitch)
      self._yaw_samples.append(face_yaw)
      self._pitch_baseline = float(np.mean(self._pitch_samples))
      self._yaw_baseline = float(np.mean(self._yaw_samples))
      return events
    if self._time_elapsed < _CALIB_CLOSED_END_S:
      self._ear_closed_samples.append(EAR)
      baseline_open = float(np.mean(self._ear_open_samples)) if self._ear_open_samples else 0.0
      baseline_closed = float(np.mean(self._ear_closed_samples))
      self._threshold_ear = baseline_open - (baseline_open - baseline_closed) * _EAR_CLOSED_FRACTION
      return events
    if self._time_elapsed < _CALIB_DONE_END_S:
      if not self._calibration_beeped:
        self._calibration_beeped = True
        events.append(
            Event.now("skinnemoen_calibration_complete", 0.0, 1.0, self.name,
                      f"threshold_EAR={self._threshold_ear:.3f}")
        )
      return events

    # --- their detection ---
    self._frames.append(1 if EAR < self._threshold_ear else 0)

    self._distracted = (
        abs(np.mean([pitch_right_eye, pitch_left_eye]) + (face_pitch - self._pitch_baseline)) > _ANGLE_THRESHOLD_DEG
        or abs(np.mean([yaw_right_eye, yaw_left_eye]) + (face_yaw - self._yaw_baseline)) > _ANGLE_THRESHOLD_DEG
    )
    if self._distracted and not self._distracted_prev:
      # ADDITION: sounds the loud alarm (the original only drew silent text here)
      events.append(Event.now("skinnemoen_distracted", 0.0, 0.5, self.name, "angle_gt_30deg"))
    self._distracted_prev = self._distracted

    phone_event = self._check_phone_use()
    if phone_event is not None:
      events.append(phone_event)

    self._drowsy = False
    if len(self._frames) > _PERCLOS_WINDOW_S * self._fps:
      PERCLOS = self._frames.count(1) / len(self._frames)
      self._last_perclos = PERCLOS
      self._drowsy = PERCLOS > _PERCLOS_THRESHOLD
      if self._drowsy and not self._drowsy_prev:
        # the original beeps here; "drowsiness" routes through the loud alert, matching it
        events.append(Event.now("drowsiness", 0.0, PERCLOS, self.name, "skinnemoen_perclos_gt_0.8"))
      self._frames.pop(0)
    self._drowsy_prev = self._drowsy
    return events

  def get_telemetry_snapshot(self) -> Optional[dict]:
    return {
        "ear": self._last_ear,
        "yaw_deg": self._last_face_yaw - self._yaw_baseline,
        "pitch_deg": self._last_face_pitch - self._pitch_baseline,
        "perclos": self._last_perclos,
        "risk_score": self._last_perclos,
        "phone_visible": False,  # no object detection in this system
        "hand_count": 0,
        "eating_visible": False,
    }

  def annotate(self, frame: np.ndarray) -> np.ndarray:
    for x, y, color, radius in self._draw_points:
      cv2.circle(frame, (x, y), radius=radius, color=color, thickness=-1)
    if self._nose_line is not None:
      cv2.line(frame, self._nose_line[0], self._nose_line[1], (255, 0, 0), 3)

    # ADDITION: phone boxes + phone-use flag
    for box in self._last_phone_boxes:
      cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255) if self._phone_use else (0, 165, 255), 2)
    if self._phone_use:
      cv2.putText(frame, "PHONE USE", (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # their exact on-screen texts
    if self._time_elapsed < _CALIB_OPEN_END_S:
      cv2.putText(frame, "Calibrating ", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      cv2.putText(frame, "Eyes open and head forward", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif self._time_elapsed < _CALIB_CLOSED_END_S:
      cv2.putText(frame, "Calibrating EAR baseline", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      cv2.putText(frame, "Eyes closed", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif self._time_elapsed < _CALIB_DONE_END_S:
      cv2.putText(frame, "Calibration complete", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
      if self._distracted:
        cv2.putText(frame, "DISTRACTED", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
      if self._drowsy:
        cv2.putText(frame, "Drowsy", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS : {int(self._fps)}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, f"TIME : {round(self._time_elapsed, 2)}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    return frame

  def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
    frame = self.annotate(frame)
    lines = [
        f"skinnemoen DMS | EAR {self._last_ear:.3f} (threshold {self._threshold_ear:.3f})",
        f"pitch {self._last_face_pitch - self._pitch_baseline:+.1f} yaw {self._last_face_yaw - self._yaw_baseline:+.1f} deg (calibrated)",
        f"PERCLOS {self._last_perclos:.2f} (drowsy > {_PERCLOS_THRESHOLD})  window {len(self._frames)} frames",
        f"face: {'YES' if self._face_found else 'NO (frame skipped, as in original)'}",
    ]
    y = 500 if frame.shape[0] > 520 else frame.shape[0] - 90
    for line in lines:
      cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      y += 20
    return frame
