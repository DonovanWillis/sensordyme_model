"""comma.ai openpilot driver monitoring, running unmodified in this framework.

This module gives you EXACTLY what an openpilot driver experiences. Everything
decision-making is vendored VERBATIM from openpilot v0.9.7 (MIT licensed,
github.com/commaai/openpilot):

  * The neural network itself: `dmonitoring_model.onnx` (their real production
    model, auto-downloaded from their repo), with their exact input crop and
    their exact output parsing (`DriverStateResult` ctypes struct, sigmoid,
    REG_SCALE, exp on stds) from selfdrive/modeld/dmonitoringmodeld.py.
  * The whole policy: `DriverMonitoring` from selfdrive/monitoring/helpers.py —
    every threshold, the awareness countdown, pose auto-calibration
    (RunningStatFilter), the distraction filter, and the three-stage alert
    escalation (green "pre" -> orange "prompt" -> red terminal), unchanged,
    ticked at their real DT_DMON = 20Hz.

What is necessarily different (a Mac is not a car):
  * Camera: your webcam instead of their IR global-shutter driver cam. The
    frame is converted to grayscale and cropped with their exact crop math
    (bottom-anchored, horizontally centered) after scaling to cover 1440x960.
  * Car signals are SIMULATED with config values: vEgo (`sim_speed_ms`,
    default 25 m/s so pose calibration runs exactly as it does on the highway),
    engaged state, gear, standstill. These are inputs openpilot receives from
    the car — the monitoring logic consuming them is untouched.
  * Interaction keys (the physical controls a driver has):
      w  = touch the steering wheel
      e  = engage/disengage openpilot

Run it:  make dev-openpilot   (or: python main.py --module driver_attention_openpilot --debug)
"""

from __future__ import annotations

import ctypes
import math
import time
from math import atan2
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

from core.detectors import Box, get_yolo_model, iou
from core.events import Event
from modules.base import SustainedConditionTracker, VisionModule, register_module

# ======================================================================================
# VENDORED from openpilot v0.9.7 — common/filter_simple.py (verbatim)
# ======================================================================================


class FirstOrderFilter:
  # first order filter
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self.dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_alpha(self, rc):
    self.alpha = self.dt / (rc + self.dt)

  def update(self, x):
    if self.initialized:
      self.x = (1. - self.alpha) * self.x + self.alpha * x
    else:
      self.initialized = True
      self.x = x
    return self.x


# ======================================================================================
# VENDORED from openpilot v0.9.7 — common/stat_live.py (verbatim)
# ======================================================================================


class RunningStat:
  # tracks realtime mean and standard deviation without storing any data
  def __init__(self, priors=None, max_trackable=-1):
    self.max_trackable = max_trackable
    if priors is not None:
      # initialize from history
      self.M = priors[0]
      self.S = priors[1]
      self.n = priors[2]
      self.M_last = self.M
      self.S_last = self.S
    else:
      self.reset()

  def reset(self):
    self.M = 0.
    self.S = 0.
    self.M_last = 0.
    self.S_last = 0.
    self.n = 0

  def push_data(self, new_data):
    # short term memory hack
    if self.max_trackable < 0 or self.n < self.max_trackable:
      self.n += 1
    if self.n == 0:
      self.M_last = new_data
      self.M = self.M_last
      self.S_last = 0.
    else:
      self.M = self.M_last + (new_data - self.M_last) / self.n
      self.S = self.S_last + (new_data - self.M_last) * (new_data - self.M)
      self.M_last = self.M
      self.S_last = self.S

  def mean(self):
    return self.M

  def variance(self):
    if self.n >= 2:
      return self.S / (self.n - 1.)
    else:
      return 0

  def std(self):
    return np.sqrt(self.variance())

  def params_to_save(self):
    return [self.M, self.S, self.n]


class RunningStatFilter:
  def __init__(self, raw_priors=None, filtered_priors=None, max_trackable=-1):
    self.raw_stat = RunningStat(raw_priors, -1)
    self.filtered_stat = RunningStat(filtered_priors, max_trackable)

  def reset(self):
    self.raw_stat.reset()
    self.filtered_stat.reset()

  def push_and_update(self, new_data):
    _std_last = self.raw_stat.std()
    self.raw_stat.push_data(new_data)
    _delta_std = self.raw_stat.std() - _std_last
    if _delta_std <= 0:
      self.filtered_stat.push_data(new_data)
    else:
      pass


# ======================================================================================
# Import shims — the ONLY non-verbatim glue. Each replaces an openpilot-internal
# import with an equivalent that carries identical values/behavior.
# ======================================================================================

DT_DMON = 0.05  # openpilot common/realtime.py: driver monitoring runs at 20Hz


def interp(x, xp, fp):  # openpilot common/numpy_fast.py behavior
  return float(np.interp(x, xp, fp))


def sigmoid(x):  # openpilot selfdrive/modeld/models/commonmodel sigmoid
  return 1. / (1. + math.exp(-x))


class EventName:  # cereal car.CarEvent.EventName members used by DriverMonitoring
  preDriverDistracted = "preDriverDistracted"
  promptDriverDistracted = "promptDriverDistracted"
  driverDistracted = "driverDistracted"
  preDriverUnresponsive = "preDriverUnresponsive"
  promptDriverUnresponsive = "promptDriverUnresponsive"
  driverUnresponsive = "driverUnresponsive"
  tooDistracted = "tooDistracted"


class Events:  # openpilot selfdrive/controls/lib/events.py Events container (the parts used)
  def __init__(self):
    self.events = []

  def add(self, event_name):
    self.events.append(event_name)


class _GearShifter:  # cereal car.CarState.GearShifter members used by run_step
  reverse = "reverse"
  park = "park"


car = SimpleNamespace(CarState=SimpleNamespace(GearShifter=_GearShifter))

# The openpilot in-car alert content for each event (selfdrive/controls/lib/events.py),
# so the on-screen banner shows what the driver sees on the comma device.
OPENPILOT_ALERT_TEXT = {
    EventName.preDriverDistracted: ("Pay Attention", "Driver Distracted"),
    EventName.promptDriverDistracted: ("Touch Steering Wheel", "Driver Distracted"),
    EventName.driverDistracted: ("DISENGAGE IMMEDIATELY", "Driver Distracted"),
    EventName.preDriverUnresponsive: ("Touch Steering Wheel", "No Face Detected"),
    EventName.promptDriverUnresponsive: ("Touch Steering Wheel", "Driver Unresponsive"),
    EventName.driverUnresponsive: ("DISENGAGE IMMEDIATELY", "Driver Unresponsive"),
    EventName.tooDistracted: ("Distraction Level Too High", "openpilot will not engage"),
}
_PRE_EVENTS = {EventName.preDriverDistracted, EventName.preDriverUnresponsive}
_PROMPT_EVENTS = {EventName.promptDriverDistracted, EventName.promptDriverUnresponsive}
_TERMINAL_EVENTS = {EventName.driverDistracted, EventName.driverUnresponsive}


# ======================================================================================
# VENDORED from openpilot v0.9.7 — selfdrive/modeld/dmonitoringmodeld.py (verbatim
# structs/constants; the ModelRunner is replaced by onnxruntime below)
# ======================================================================================

CALIB_LEN = 3
REG_SCALE = 0.25
MODEL_WIDTH = 1440
MODEL_HEIGHT = 960
OUTPUT_SIZE = 84


class DriverStateResult(ctypes.Structure):
  _fields_ = [
    ("face_orientation", ctypes.c_float*3),
    ("face_position", ctypes.c_float*3),
    ("face_orientation_std", ctypes.c_float*3),
    ("face_position_std", ctypes.c_float*3),
    ("face_prob", ctypes.c_float),
    ("_unused_a", ctypes.c_float*8),
    ("left_eye_prob", ctypes.c_float),
    ("_unused_b", ctypes.c_float*8),
    ("right_eye_prob", ctypes.c_float),
    ("left_blink_prob", ctypes.c_float),
    ("right_blink_prob", ctypes.c_float),
    ("sunglasses_prob", ctypes.c_float),
    ("occluded_prob", ctypes.c_float),
    ("ready_prob", ctypes.c_float*4),
    ("not_ready_prob", ctypes.c_float*2)]


class DMonitoringModelResult(ctypes.Structure):
  _fields_ = [
    ("driver_state_lhd", DriverStateResult),
    ("driver_state_rhd", DriverStateResult),
    ("poor_vision_prob", ctypes.c_float),
    ("wheel_on_right_prob", ctypes.c_float)]


def fill_driver_state(ds_result: DriverStateResult) -> SimpleNamespace:
  """openpilot's fill_driver_state, filling a namespace instead of a capnp message."""
  return SimpleNamespace(
      faceOrientation=[x * REG_SCALE for x in ds_result.face_orientation],
      faceOrientationStd=[math.exp(x) for x in ds_result.face_orientation_std],
      facePosition=[x * REG_SCALE for x in ds_result.face_position[:2]],
      facePositionStd=[math.exp(x) for x in ds_result.face_position_std[:2]],
      faceProb=sigmoid(ds_result.face_prob),
      leftEyeProb=sigmoid(ds_result.left_eye_prob),
      rightEyeProb=sigmoid(ds_result.right_eye_prob),
      leftBlinkProb=sigmoid(ds_result.left_blink_prob),
      rightBlinkProb=sigmoid(ds_result.right_blink_prob),
      sunglassesProb=sigmoid(ds_result.sunglasses_prob),
      occludedProb=sigmoid(ds_result.occluded_prob),
      readyProb=[sigmoid(x) for x in ds_result.ready_prob],
      notReadyProb=[sigmoid(x) for x in ds_result.not_ready_prob],
  )


def parse_driverstate(model_output: np.ndarray) -> SimpleNamespace:
  """openpilot's get_driverstate_packet, returning a driverStateV2-shaped namespace."""
  model_result = ctypes.cast(model_output.ctypes.data, ctypes.POINTER(DMonitoringModelResult)).contents
  return SimpleNamespace(
      poorVisionProb=sigmoid(model_result.poor_vision_prob),
      wheelOnRightProb=sigmoid(model_result.wheel_on_right_prob),
      leftDriverData=fill_driver_state(model_result.driver_state_lhd),
      rightDriverData=fill_driver_state(model_result.driver_state_rhd),
  )


# ======================================================================================
# VENDORED from openpilot v0.9.7 — selfdrive/monitoring/helpers.py (verbatim; the
# import shims above supply EventName/Events/interp/DT_DMON/filters; DEVICE_CAMERAS
# lookup replaced by its literal result for ("tici", "ar0231"): dcam 1928x1208)
# ======================================================================================

# ******************************************************************************************
#  NOTE: To fork maintainers.
#  Disabling or nerfing safety features will get you and your users banned from our servers.
#  We recommend that you do not change these numbers from the defaults.
# ******************************************************************************************


class DRIVER_MONITOR_SETTINGS:
  def __init__(self):
    self._DT_DMON = DT_DMON
    # ref (page15-16): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:42018X1947&rid=2
    self._AWARENESS_TIME = 30. # passive wheeltouch total timeout
    self._AWARENESS_PRE_TIME_TILL_TERMINAL = 15.
    self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
    self._DISTRACTED_TIME = 11. # active monitoring total timeout
    self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
    self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

    self._FACE_THRESHOLD = 0.7
    self._EYE_THRESHOLD = 0.65
    self._SG_THRESHOLD = 0.9
    self._BLINK_THRESHOLD = 0.865

    self._EE_THRESH11 = 0.25
    self._EE_THRESH12 = 7.5
    self._EE_MAX_OFFSET1 = 0.06
    self._EE_MIN_OFFSET1 = 0.025
    self._EE_THRESH21 = 0.01
    self._EE_THRESH22 = 0.35

    self._POSE_PITCH_THRESHOLD = 0.3133
    self._POSE_PITCH_THRESHOLD_SLACK = 0.3237
    self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
    self._POSE_YAW_THRESHOLD = 0.4020
    self._POSE_YAW_THRESHOLD_SLACK = 0.5042
    self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
    self._PITCH_NATURAL_OFFSET = 0.029 # initial value before offset is learned
    self._PITCH_NATURAL_THRESHOLD = 0.449
    self._YAW_NATURAL_OFFSET = 0.097 # initial value before offset is learned
    self._PITCH_MAX_OFFSET = 0.124
    self._PITCH_MIN_OFFSET = -0.0881
    self._YAW_MAX_OFFSET = 0.289
    self._YAW_MIN_OFFSET = -0.0246

    self._POSESTD_THRESHOLD = 0.3
    self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
    self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz
    self._ALWAYS_ON_ALERT_MIN_SPEED = 7

    self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
    self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
    self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

    self._WHEELPOS_CALIB_MIN_SPEED = 11
    self._WHEELPOS_THRESHOLD = 0.5
    self._WHEELPOS_FILTER_MIN_COUNT = int(15 / self._DT_DMON) # allow 15 seconds to converge wheel side

    self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
    self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

    self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
    self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts


class DistractedType:
  NOT_DISTRACTED = 0
  DISTRACTED_POSE = 1 << 0
  DISTRACTED_BLINK = 1 << 1
  DISTRACTED_E2E = 1 << 2


class DriverPose:
  def __init__(self, max_trackable):
    self.yaw = 0.
    self.pitch = 0.
    self.roll = 0.
    self.yaw_std = 0.
    self.pitch_std = 0.
    self.roll_std = 0.
    self.pitch_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.yaw_offseter = RunningStatFilter(max_trackable=max_trackable)
    self.calibrated = False
    self.low_std = True
    self.cfactor_pitch = 1.
    self.cfactor_yaw = 1.


class DriverBlink:
  def __init__(self):
    self.left = 0.
    self.right = 0.


# model output refers to center of undistorted+leveled image
EFL = 598.0 # focal length in K
W, H = 1928, 1208  # DEVICE_CAMERAS[("tici", "ar0231")].dcam size — corrected image has same size as raw


def face_orientation_from_net(angles_desc, pos_desc, rpy_calib):
  # the output of these angles are in device frame
  # so from driver's perspective, pitch is up and yaw is right

  pitch_net, yaw_net, roll_net = angles_desc

  face_pixel_position = ((pos_desc[0]+0.5)*W, (pos_desc[1]+0.5)*H)
  yaw_focal_angle = atan2(face_pixel_position[0] - W//2, EFL)
  pitch_focal_angle = atan2(face_pixel_position[1] - H//2, EFL)

  pitch = pitch_net + pitch_focal_angle
  yaw = -yaw_net + yaw_focal_angle

  # no calib for roll
  pitch -= rpy_calib[1]
  yaw -= rpy_calib[2]
  return roll_net, pitch, yaw


class DriverMonitoring:
  def __init__(self, rhd_saved=False, settings=None, always_on=False):
    if settings is None:
      settings = DRIVER_MONITOR_SETTINGS()
    # init policy settings
    self.settings = settings

    # init driver status
    self.wheelpos_learner = RunningStatFilter()
    self.pose = DriverPose(self.settings._POSE_OFFSET_MAX_COUNT)
    self.blink = DriverBlink()
    self.eev1 = 0.
    self.eev2 = 1.
    self.ee1_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee2_offseter = RunningStatFilter(max_trackable=self.settings._POSE_OFFSET_MAX_COUNT)
    self.ee1_calibrated = False
    self.ee2_calibrated = False

    self.always_on = always_on
    self.distracted_types = []
    self.driver_distracted = False
    self.driver_distraction_filter = FirstOrderFilter(0., self.settings._DISTRACTED_FILTER_TS, self.settings._DT_DMON)
    self.wheel_on_right = False
    self.wheel_on_right_last = None
    self.wheel_on_right_default = rhd_saved
    self.face_detected = False
    self.terminal_alert_cnt = 0
    self.terminal_time = 0
    self.step_change = 0.
    self.active_monitoring_mode = True
    self.is_model_uncertain = False
    self.hi_stds = 0
    self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
    self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME

    self._reset_awareness()
    self._set_timers(active_monitoring=True)
    self._reset_events()

  def _reset_awareness(self):
    self.awareness = 1.
    self.awareness_active = 1.
    self.awareness_passive = 1.

  def _reset_events(self):
    self.current_events = Events()

  def _set_timers(self, active_monitoring):
    if self.active_monitoring_mode and self.awareness <= self.threshold_prompt:
      if active_monitoring:
        self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      else:
        self.step_change = 0.
      return  # no exploit after orange alert
    elif self.awareness <= 0.:
      return

    if active_monitoring:
      # when falling back from passive mode to active mode, reset awareness to avoid false alert
      if not self.active_monitoring_mode:
        self.awareness_passive = self.awareness
        self.awareness = self.awareness_active

      self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
      self.step_change = self.settings._DT_DMON / self.settings._DISTRACTED_TIME
      self.active_monitoring_mode = True
    else:
      if self.active_monitoring_mode:
        self.awareness_active = self.awareness
        self.awareness = self.awareness_passive

      self.threshold_pre = self.settings._AWARENESS_PRE_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.threshold_prompt = self.settings._AWARENESS_PROMPT_TIME_TILL_TERMINAL / self.settings._AWARENESS_TIME
      self.step_change = self.settings._DT_DMON / self.settings._AWARENESS_TIME
      self.active_monitoring_mode = False

  def _set_policy(self, model_data, car_speed):
    bp = model_data.meta.disengagePredictions.brakeDisengageProbs[0] # brake disengage prob in next 2s
    k1 = max(-0.00156*((car_speed-16)**2)+0.6, 0.2)
    bp_normal = max(min(bp / k1, 0.5),0)
    self.pose.cfactor_pitch = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_PITCH_THRESHOLD_SLACK,
                                            self.settings._POSE_PITCH_THRESHOLD_STRICT]) / self.settings._POSE_PITCH_THRESHOLD
    self.pose.cfactor_yaw = interp(bp_normal, [0, 0.5],
                                           [self.settings._POSE_YAW_THRESHOLD_SLACK,
                                            self.settings._POSE_YAW_THRESHOLD_STRICT]) / self.settings._POSE_YAW_THRESHOLD

  def _get_distracted_types(self):
    distracted_types = []

    if not self.pose.calibrated:
      pitch_error = self.pose.pitch - self.settings._PITCH_NATURAL_OFFSET
      yaw_error = self.pose.yaw - self.settings._YAW_NATURAL_OFFSET
    else:
      pitch_error = self.pose.pitch - min(max(self.pose.pitch_offseter.filtered_stat.mean(),
                                                       self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
      yaw_error = self.pose.yaw - min(max(self.pose.yaw_offseter.filtered_stat.mean(),
                                                    self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)
    pitch_error = 0 if pitch_error > 0 else abs(pitch_error) # no positive pitch limit
    yaw_error = abs(yaw_error)
    if pitch_error > (self.settings._POSE_PITCH_THRESHOLD*self.pose.cfactor_pitch if self.pose.calibrated else self.settings._PITCH_NATURAL_THRESHOLD) or \
       yaw_error > self.settings._POSE_YAW_THRESHOLD*self.pose.cfactor_yaw:
      distracted_types.append(DistractedType.DISTRACTED_POSE)

    if (self.blink.left + self.blink.right)*0.5 > self.settings._BLINK_THRESHOLD:
      distracted_types.append(DistractedType.DISTRACTED_BLINK)

    if self.ee1_calibrated:
      ee1_dist = self.eev1 > max(min(self.ee1_offseter.filtered_stat.M, self.settings._EE_MAX_OFFSET1), self.settings._EE_MIN_OFFSET1) \
                              * self.settings._EE_THRESH12
    else:
      ee1_dist = self.eev1 > self.settings._EE_THRESH11
    if ee1_dist:
      distracted_types.append(DistractedType.DISTRACTED_E2E)

    return distracted_types

  def _update_states(self, driver_state, cal_rpy, car_speed, op_engaged):
    rhd_pred = driver_state.wheelOnRightProb
    # calibrates only when there's movement and either face detected
    if car_speed > self.settings._WHEELPOS_CALIB_MIN_SPEED and (driver_state.leftDriverData.faceProb > self.settings._FACE_THRESHOLD or
                                          driver_state.rightDriverData.faceProb > self.settings._FACE_THRESHOLD):
      self.wheelpos_learner.push_and_update(rhd_pred)
    if self.wheelpos_learner.filtered_stat.n > self.settings._WHEELPOS_FILTER_MIN_COUNT:
      self.wheel_on_right = self.wheelpos_learner.filtered_stat.M > self.settings._WHEELPOS_THRESHOLD
    else:
      self.wheel_on_right = self.wheel_on_right_default # use default/saved if calibration is unfinished
    # make sure no switching when engaged
    if op_engaged and self.wheel_on_right_last is not None and self.wheel_on_right_last != self.wheel_on_right:
      self.wheel_on_right = self.wheel_on_right_last
    driver_data = driver_state.rightDriverData if self.wheel_on_right else driver_state.leftDriverData
    if not all(len(x) > 0 for x in (driver_data.faceOrientation, driver_data.facePosition,
                                    driver_data.faceOrientationStd, driver_data.facePositionStd,
                                    driver_data.readyProb, driver_data.notReadyProb)):
      return

    self.face_detected = driver_data.faceProb > self.settings._FACE_THRESHOLD
    self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_data.faceOrientation, driver_data.facePosition, cal_rpy)
    if self.wheel_on_right:
      self.pose.yaw *= -1
    self.wheel_on_right_last = self.wheel_on_right
    self.pose.pitch_std = driver_data.faceOrientationStd[0]
    self.pose.yaw_std = driver_data.faceOrientationStd[1]
    model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
    self.pose.low_std = model_std_max < self.settings._POSESTD_THRESHOLD
    self.blink.left = driver_data.leftBlinkProb * (driver_data.leftEyeProb > self.settings._EYE_THRESHOLD) \
                                                                  * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.blink.right = driver_data.rightBlinkProb * (driver_data.rightEyeProb > self.settings._EYE_THRESHOLD) \
                                                                  * (driver_data.sunglassesProb < self.settings._SG_THRESHOLD)
    self.eev1 = driver_data.notReadyProb[0]
    self.eev2 = driver_data.readyProb[0]

    self.distracted_types = self._get_distracted_types()
    self.driver_distracted = (DistractedType.DISTRACTED_E2E in self.distracted_types or DistractedType.DISTRACTED_POSE in self.distracted_types
                                or DistractedType.DISTRACTED_BLINK in self.distracted_types) \
                              and driver_data.faceProb > self.settings._FACE_THRESHOLD and self.pose.low_std
    self.driver_distraction_filter.update(self.driver_distracted)

    # update offseter
    # only update when driver is actively driving the car above a certain speed
    if self.face_detected and car_speed > self.settings._POSE_CALIB_MIN_SPEED and self.pose.low_std and (not op_engaged or not self.driver_distracted):
      self.pose.pitch_offseter.push_and_update(self.pose.pitch)
      self.pose.yaw_offseter.push_and_update(self.pose.yaw)
      self.ee1_offseter.push_and_update(self.eev1)
      self.ee2_offseter.push_and_update(self.eev2)

    self.pose.calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
                                       self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee1_calibrated = self.ee1_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT
    self.ee2_calibrated = self.ee2_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

    self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
    self._set_timers(self.face_detected and not self.is_model_uncertain)
    if self.face_detected and not self.pose.low_std and not self.driver_distracted:
      self.hi_stds += 1
    elif self.face_detected and self.pose.low_std:
      self.hi_stds = 0

  def _update_events(self, driver_engaged, op_engaged, standstill, wrong_gear, car_speed):
    self._reset_events()
    # Block engaging after max number of distrations or when alert active
    if self.terminal_alert_cnt >= self.settings._MAX_TERMINAL_ALERTS or \
       self.terminal_time >= self.settings._MAX_TERMINAL_DURATION or \
       self.always_on and self.awareness <= self.threshold_prompt:
      self.current_events.add(EventName.tooDistracted)

    always_on_valid = self.always_on and not wrong_gear
    if (driver_engaged and self.awareness > 0 and not self.active_monitoring_mode) or \
       (not always_on_valid and not op_engaged) or \
       (always_on_valid and not op_engaged and self.awareness <= 0):
      # always reset on disengage with normal mode; disengage resets only on red if always on
      self._reset_awareness()
      return

    driver_attentive = self.driver_distraction_filter.x < 0.37
    awareness_prev = self.awareness

    if (driver_attentive and self.face_detected and self.pose.low_std and self.awareness > 0):
      if driver_engaged:
        self._reset_awareness()
        return
      # only restore awareness when paying attention and alert is not red
      self.awareness = min(self.awareness + ((self.settings._RECOVERY_FACTOR_MAX-self.settings._RECOVERY_FACTOR_MIN)*
                                             (1.-self.awareness)+self.settings._RECOVERY_FACTOR_MIN)*self.step_change, 1.)
      if self.awareness == 1.:
        self.awareness_passive = min(self.awareness_passive + self.step_change, 1.)
      # don't display alert banner when awareness is recovering and has cleared orange
      if self.awareness > self.threshold_prompt:
        return

    _reaching_audible = self.awareness - self.step_change <= self.threshold_prompt
    _reaching_terminal = self.awareness - self.step_change <= 0
    standstill_exemption = standstill and _reaching_audible
    always_on_red_exemption = always_on_valid and not op_engaged and _reaching_terminal
    always_on_lowspeed_exemption = always_on_valid and not op_engaged and car_speed < self.settings._ALWAYS_ON_ALERT_MIN_SPEED and _reaching_audible

    certainly_distracted = self.driver_distraction_filter.x > 0.63 and self.driver_distracted and self.face_detected
    maybe_distracted = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME or not self.face_detected

    if certainly_distracted or maybe_distracted:
      # should always be counting if distracted unless at standstill (lowspeed for always-on) and reaching orange
      # also will not be reaching 0 if DM is active when not engaged
      if not (standstill_exemption or always_on_red_exemption or always_on_lowspeed_exemption):
        self.awareness = max(self.awareness - self.step_change, -0.1)

    alert = None
    if self.awareness <= 0.:
      # terminal red alert: disengagement required
      alert = EventName.driverDistracted if self.active_monitoring_mode else EventName.driverUnresponsive
      self.terminal_time += 1
      if awareness_prev > 0.:
        self.terminal_alert_cnt += 1
    elif self.awareness <= self.threshold_prompt:
      # prompt orange alert
      alert = EventName.promptDriverDistracted if self.active_monitoring_mode else EventName.promptDriverUnresponsive
    elif self.awareness <= self.threshold_pre:
      # pre green alert
      alert = EventName.preDriverDistracted if self.active_monitoring_mode else EventName.preDriverUnresponsive

    if alert is not None:
      self.current_events.add(alert)

  def run_step(self, sm):
    # Set strictness
    self._set_policy(
      model_data=sm['modelV2'],
      car_speed=sm['carState'].vEgo
    )

    # Parse data from dmonitoringmodeld
    self._update_states(
      driver_state=sm['driverStateV2'],
      cal_rpy=sm['liveCalibration'].rpyCalib,
      car_speed=sm['carState'].vEgo,
      op_engaged=sm['controlsState'].enabled
    )

    # Update distraction events
    self._update_events(
      driver_engaged=sm['carState'].steeringPressed or sm['carState'].gasPressed,
      op_engaged=sm['controlsState'].enabled,
      standstill=sm['carState'].standstill,
      wrong_gear=sm['carState'].gearShifter in [car.CarState.GearShifter.reverse, car.CarState.GearShifter.park],
      car_speed=sm['carState'].vEgo
    )


# ======================================================================================
# End of vendored openpilot code. Below: the ONNX runner and framework integration.
# ======================================================================================

_MODEL_URL = (
    "https://media.githubusercontent.com/media/commaai/openpilot/v0.9.7/"
    "selfdrive/modeld/models/dmonitoring_model.onnx"
)


class _OnnxDMModel:
  """Runs openpilot's dmonitoring_model.onnx with their exact crop and parsing."""

  def __init__(self, model_path: str) -> None:
    import onnxruntime as ort  # lazy: never imported on the Pi

    from pathlib import Path
    path = Path(model_path)
    if not path.exists():
      print("driver_attention_openpilot: downloading dmonitoring_model.onnx (~16MB, one-time)")
      import urllib.request
      urllib.request.urlretrieve(_MODEL_URL, path)
    # Provider list comes from the environment so deployments can opt into other
    # execution providers without a code change (default: CPU, the only provider
    # bundled with the pip onnxruntime wheel on both Mac and Pi).
    import os
    providers = [
        p.strip()
        for p in os.environ.get("SENSORDYME_ONNX_PROVIDERS", "CPUExecutionProvider").split(",")
        if p.strip()
    ]
    self._session = ort.InferenceSession(str(path), providers=providers)
    self._calib = np.zeros((1, CALIB_LEN), dtype=np.float32)  # rpyCalib zeros = uncalibrated device

  def run(self, frame_bgr: np.ndarray) -> tuple[SimpleNamespace, float]:
    """One inference: grayscale -> cover-scale -> openpilot's exact crop -> parse."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(MODEL_WIDTH / w, MODEL_HEIGHT / h)
    buf = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))))
    # openpilot's crop math from dmonitoringmodeld.ModelState.run, applied verbatim:
    v_offset = buf.shape[0] - MODEL_HEIGHT
    h_offset = (buf.shape[1] - MODEL_WIDTH) // 2
    input_data = buf[v_offset:v_offset + MODEL_HEIGHT, h_offset:h_offset + MODEL_WIDTH]

    t1 = time.perf_counter()
    output = self._session.run(
        None,
        {
            "input_img": input_data.reshape(1, -1).astype(np.float32),
            "calib": self._calib,
        },
    )[0].reshape(-1).astype(np.float32)
    latency = time.perf_counter() - t1
    return parse_driverstate(output), latency


@register_module("driver_attention_openpilot")
class DriverAttentionOpenpilotModule(VisionModule):
  """openpilot's driver monitoring — real model, real policy, real alert escalation."""

  name = "driver_attention_openpilot"

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self._model = _OnnxDMModel(config.get("model_path", "dmonitoring_model.onnx"))
    self._dm = DriverMonitoring(rhd_saved=False, always_on=bool(config.get("always_on", False)))

    # Simulated car signals (openpilot receives these from the CAN bus).
    self._sim_speed = float(config.get("sim_speed_ms", 25.0))  # m/s; > 13 so pose calibration runs
    self._engaged = bool(config.get("sim_engaged", True))
    self._sim_bp = float(config.get("sim_brake_disengage_prob", 0.0))
    self._wheel_touch = False

    # ADDITION on top of stock openpilot (which stays silent until the orange prompt at
    # ~5s): an immediate chime the moment openpilot's own "certainly distracted" signal
    # fires (driver_distracted with the distraction filter past 0.63 — ~0.3s after the
    # model first flags it, so a normal blink never beeps). Config-toggled; the vendored
    # policy and its escalation are untouched.
    self._instant_sound = bool(config.get("instant_distraction_sound", True))
    self._certainly_distracted_prev = False

    # ADDITION: eyes-closed flag. openpilot's model reports per-eye blink probability
    # every frame (already gated on eye visibility and sunglasses by the vendored
    # _update_states); sustained closure past eyes_closed_seconds flags distracted
    # driving as "eyesClosedDetected" (loud).
    self._eyes_closed_tracker = SustainedConditionTracker(float(config.get("eyes_closed_seconds", 2.0)))
    self._eyes_closed_blink_threshold = float(
        config.get("eyes_closed_blink_threshold", 0.865)  # openpilot's own _BLINK_THRESHOLD
    )

    # ADDITION: phone-use flag. openpilot's published (pre-"LM GT3") model cannot see
    # phones, so this uses the framework's YOLO phone detector (+ optional MediaPipe
    # Hands overlap = "holding it"); sustained use past phone_use_seconds flags
    # "phoneUseDetected" (loud). All lazily loaded and disable-able.
    self._phone_enabled = bool(config.get("phone_detection", True))
    self._phone_weights = config.get("phone_weights", "yolo11n.pt")
    self._phone_device = config.get("phone_device", "mps")
    self._phone_every_n_frames = max(1, int(config.get("phone_every_n_frames", 3)))
    self._phone_conf = float(config.get("phone_confidence_threshold", 0.35))
    self._hand_phone_iou_threshold = float(config.get("hand_phone_iou_threshold", 0.2))
    self._enable_hands = bool(config.get("enable_hands", True))
    self._phone_use_tracker = SustainedConditionTracker(float(config.get("phone_use_seconds", 1.0)))
    self._phone_model = None
    self._phone_class_id: Optional[int] = None
    self._hands = None
    self._phone_error: Optional[str] = None
    self._frame_count = 0
    self._last_phone_boxes: list[Box] = []
    self._last_hand_boxes: list[Box] = []
    self._phone_in_use = False

    self._driver_state: Optional[SimpleNamespace] = None
    self._model_latency = 0.0
    self._last_tick: Optional[float] = None
    self._tick_accum = 0.0
    self._prev_events: set[str] = set()
    self._active_alert: Optional[str] = None

    print(
        "driver_attention_openpilot: openpilot v0.9.7 DM model + verbatim DriverMonitoring policy "
        f"(simulated: vEgo={self._sim_speed} m/s, engaged={self._engaged}). "
        "Keys in the video window: [w] touch steering wheel, [e] engage/disengage."
    )

  # -------------------------------------------------------------- key controls
  def handle_key(self, key: int) -> None:
    """Simulated driver interactions (the physical controls an openpilot driver has)."""
    if key == ord("w"):
      self._wheel_touch = True
    elif key == ord("e"):
      self._engaged = not self._engaged
      print(f"driver_attention_openpilot: openpilot {'ENGAGED' if self._engaged else 'DISENGAGED'}")

  # -------------------------------------------------------------- main loop
  def _build_sm(self) -> dict:
    """The SubMaster dict run_step() reads — car signals simulated, DM data real."""
    return {
        "modelV2": SimpleNamespace(
            meta=SimpleNamespace(
                disengagePredictions=SimpleNamespace(brakeDisengageProbs=[self._sim_bp])
            )
        ),
        "driverStateV2": self._driver_state,
        "liveCalibration": SimpleNamespace(rpyCalib=[0.0, 0.0, 0.0]),
        "carState": SimpleNamespace(
            vEgo=self._sim_speed,
            steeringPressed=self._wheel_touch,
            gasPressed=False,
            standstill=False,
            gearShifter="drive",
        ),
        "controlsState": SimpleNamespace(enabled=self._engaged),
    }

  def _update_phone_detection(self, frame: np.ndarray) -> None:
    """Additive YOLO phone (+ hand overlap) detection, run every Nth frame."""
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
        if self._enable_hands:
          import mediapipe as mp
          self._hands = mp.solutions.hands.Hands(
              max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
          )
        print(
            f"driver_attention_openpilot: phone detection ready "
            f"(weights={self._phone_weights}, device={self._phone_device}, hands={self._enable_hands})"
        )

      results = self._phone_model.predict(
          frame, verbose=False, classes=[self._phone_class_id], conf=self._phone_conf,
          device=self._phone_device,
      )
      phone_boxes: list[Box] = []
      for result in results:
        for box in result.boxes:
          phone_boxes.append(tuple(int(v) for v in box.xyxy[0].tolist()))

      hand_boxes: list[Box] = []
      if self._hands is not None:
        h, w = frame.shape[:2]
        hand_results = self._hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hand_results.multi_hand_landmarks:
          for hand_landmarks in hand_results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            hand_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

      self._last_phone_boxes = phone_boxes
      self._last_hand_boxes = hand_boxes
      if self._hands is not None:
        # "using it" = phone visible AND a hand on it
        self._phone_in_use = any(
            iou(hand, phone) > self._hand_phone_iou_threshold
            for hand in hand_boxes for phone in phone_boxes
        )
      else:
        self._phone_in_use = bool(phone_boxes)
    except Exception as exc:  # noqa: BLE001 - never crash the video loop
      self._phone_error = str(exc)
      print(f"driver_attention_openpilot: phone detection disabled after an error: {exc}")

  def process(self, frame: np.ndarray) -> list[Event]:
    self._driver_state, self._model_latency = self._model.run(frame)
    self._update_phone_detection(frame)

    # Tick DriverMonitoring at openpilot's real 20Hz regardless of webcam FPS.
    now = time.monotonic()
    if self._last_tick is None:
      self._last_tick = now
    self._tick_accum += now - self._last_tick
    self._last_tick = now

    fired: set[str] = set()
    ticks = 0
    while self._tick_accum >= DT_DMON and ticks < 40:
      self._dm.run_step(self._build_sm())
      fired |= set(self._dm.current_events.events)
      self._tick_accum -= DT_DMON
      ticks += 1
    if ticks:
      self._wheel_touch = False  # a touch lasts one policy tick, like a real wheel tap

    current = set(self._dm.current_events.events)
    self._active_alert = next(
        (e for group in (_TERMINAL_EVENTS, _PROMPT_EVENTS, _PRE_EVENTS) for e in current if e in group), None
    )

    # Emit an Event the moment each openpilot alert first appears (rising edge).
    events = []
    for name in fired - self._prev_events:
      events.append(
          Event.now(
              name, 0.0, 1.0 - max(self._dm.awareness, 0.0), self.name,
              f"openpilot distracted_types={self._dm.distracted_types} awareness={self._dm.awareness:.2f}",
          )
      )
    self._prev_events = current

    # Instant chime (addition, see __init__): rising edge of openpilot's own
    # certainly_distracted condition from _update_events.
    certainly_distracted = (
        self._dm.driver_distraction_filter.x > 0.63 and self._dm.driver_distracted and self._dm.face_detected
    )
    if self._instant_sound and certainly_distracted and not self._certainly_distracted_prev:
      types = self._dm.distracted_types
      labels = [
          label for label, flag in (
              ("pose", DistractedType.DISTRACTED_POSE), ("blink", DistractedType.DISTRACTED_BLINK),
              ("e2e", DistractedType.DISTRACTED_E2E)) if flag in types
      ]
      events.append(
          Event.now(
              "distractionDetected", 0.0, self._dm.driver_distraction_filter.x, self.name,
              f"instant: {'+'.join(labels) or 'unknown'}",
          )
      )
    self._certainly_distracted_prev = certainly_distracted

    # Additive flag: phone in the driver's hand, sustained -> distracted driving.
    fired_phone = self._phone_use_tracker.update(self._phone_in_use, now)
    if fired_phone is not None:
      events.append(
          Event.now(
              "phoneUseDetected", fired_phone, 0.9, self.name,
              f"phone + {'hand_overlap' if self._hands is not None else 'visible'} sustained",
          )
      )

    # Additive flag: eyes closed for more than eyes_closed_seconds (openpilot's own
    # blink signal, which _update_states already gates on eye visibility/sunglasses).
    eyes_closed = (
        self._dm.face_detected
        and (self._dm.blink.left + self._dm.blink.right) * 0.5 > self._eyes_closed_blink_threshold
    )
    fired_eyes = self._eyes_closed_tracker.update(eyes_closed, now)
    if fired_eyes is not None:
      events.append(
          Event.now(
              "eyesClosedDetected", fired_eyes, 0.95, self.name,
              f"blink L{self._dm.blink.left:.2f}/R{self._dm.blink.right:.2f} sustained",
          )
      )
    return events

  # -------------------------------------------------------------- display
  def is_alert_active(self) -> bool:
    """Red flash on openpilot's terminal (red) alerts, plus the additive phone/eyes flags."""
    return (
        self._active_alert in _TERMINAL_EVENTS
        or self._phone_use_tracker.is_active
        or self._eyes_closed_tracker.is_active
    )

  def get_telemetry_snapshot(self) -> Optional[dict]:
    return None  # this module's signals don't fit the shared telemetry schema

  def annotate(self, frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    dm = self._dm

    # openpilot-style alert banner with the comma device's exact alert text
    if self._active_alert is not None:
      title, subtitle = OPENPILOT_ALERT_TEXT[self._active_alert]
      color = (0, 0, 255) if self._active_alert in _TERMINAL_EVENTS \
          else (0, 140, 255) if self._active_alert in _PROMPT_EVENTS else (0, 160, 0)
      cv2.rectangle(frame, (0, h - 80), (w, h), color, -1)
      cv2.putText(frame, title, (16, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
      cv2.putText(frame, subtitle, (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # additive distracted-driving flags (phone use / sustained eye closure)
    for box in self._last_phone_boxes:
      color = (0, 0, 255) if self._phone_use_tracker.is_active else (0, 165, 255)
      cv2.rectangle(frame, box[:2], box[2:], color, 2)
    for box in self._last_hand_boxes:
      cv2.rectangle(frame, box[:2], box[2:], (255, 0, 255), 1)
    flag_y = 44
    if self._phone_use_tracker.is_active:
      cv2.putText(frame, "PHONE USE - DISTRACTED DRIVING", (10, flag_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      flag_y += 28
    if self._eyes_closed_tracker.is_active:
      cv2.putText(frame, "EYES CLOSED - DISTRACTED DRIVING", (10, flag_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # awareness bar (openpilot's driver-monitoring arc, linearized)
    frac = max(0.0, min(1.0, dm.awareness))
    bar_color = (0, 200, 0) if dm.awareness > dm.threshold_pre else \
        (0, 140, 255) if dm.awareness > dm.threshold_prompt else (0, 0, 255)
    cv2.rectangle(frame, (10, 10), (210, 26), (90, 90, 90), 1)
    cv2.rectangle(frame, (10, 10), (10 + int(200 * frac), 26), bar_color, -1)
    cv2.putText(frame, f"awareness {dm.awareness:.2f}", (218, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)
    return frame

  def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
    frame = self.annotate(frame)
    dm = self._dm
    types = dm.distracted_types
    lines = [
        f"openpilot DM  |  {'ENGAGED' if self._engaged else 'DISENGAGED (e)'} @ {self._sim_speed:.0f} m/s  |  model {self._model_latency*1000:.0f} ms",
        f"face {'YES' if dm.face_detected else 'NO'}  pitch {math.degrees(dm.pose.pitch):+.1f}  yaw {math.degrees(dm.pose.yaw):+.1f} deg"
        f"  (std p{dm.pose.pitch_std:.2f}/y{dm.pose.yaw_std:.2f} {'ok' if dm.pose.low_std else 'HIGH'})",
        f"blink L{dm.blink.left:.2f} R{dm.blink.right:.2f}  e2e-notready {dm.eev1:.3f}",
        f"distracted: POSE={'Y' if DistractedType.DISTRACTED_POSE in types else '-'} "
        f"BLINK={'Y' if DistractedType.DISTRACTED_BLINK in types else '-'} "
        f"E2E={'Y' if DistractedType.DISTRACTED_E2E in types else '-'}"
        f"  filter {dm.driver_distraction_filter.x:.2f}",
        f"pose calib: {'DONE' if dm.pose.calibrated else f'{dm.pose.pitch_offseter.filtered_stat.n}/{dm.settings._POSE_OFFSET_MIN_COUNT}'}"
        f"  mode {'active' if dm.active_monitoring_mode else 'passive'}  terminal alerts {dm.terminal_alert_cnt}",
        "[w] wheel touch  [e] engage toggle",
    ]
    y = 50
    for line in lines:
      cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
      y += 22
    return frame
