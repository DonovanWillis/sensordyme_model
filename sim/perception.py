"""Scripted perception: what the virtual device's camera "sees", without models.

The virtual device runs the REAL driver_attention module -- calibration, EAR and
iris closure, head pose (solvePnP), gaze, every tracker, risk fusion, episodes --
and replaces only the three model calls at its edge:

  MediaPipe FaceMesh  -> ScriptedFaceMesh: geometrically real landmarks for the
                         scene's head pose and eye state (the generator from
                         tests/test_driver_attention_scenarios.py, made
                         size-aware and able to close the eyes for real)
  MediaPipe Hands     -> ScriptedHands: a hand box on the phone when held
  object detector     -> ScriptedDetector: person / phone boxes; raises while
                         the scenario says the detector is failing

What the scene is travels IN THE FRAME: a few pixels of row 0 carry it, written
by sim.camera when the frame is rendered and read back here. A frozen camera
therefore repeats the scene it froze on, a dropped frame drops its scene, and
nothing here consults a clock -- the module's own frame timing decides.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import cv2
import numpy as np

#: Row-0 header layout (every channel holds the same byte, so BGR->RGB is moot).
_MAGIC = (0xA5, 0x5A)
_HEADER_LEN = 6
_FLAG_FACE, _FLAG_EYES_CLOSED, _FLAG_PHONE, _FLAG_PERSON, _FLAG_HAND = 1, 2, 4, 8, 16


@dataclass(frozen=True)
class Scene:
    """What is in front of the camera at one moment."""

    face: bool = True
    eyes_closed: bool = False
    #: Head pitch, degrees; positive = looking down (towards a lap / phone).
    head_down_deg: float = 0.0
    #: Head yaw, degrees; positive = turned towards the camera's right.
    yaw_deg: float = 0.0
    phone: bool = False
    #: A hand on the phone (MediaPipe Hands would see it holding it).
    hand_on_phone: bool = False
    #: A person box for the object detector (the driver's body).
    person: bool = True
    label: str = "attentive"

    def flags(self) -> int:
        return ((_FLAG_FACE if self.face else 0) | (_FLAG_EYES_CLOSED if self.eyes_closed else 0)
                | (_FLAG_PHONE if self.phone else 0) | (_FLAG_PERSON if self.person else 0)
                | (_FLAG_HAND if self.hand_on_phone else 0))


def _byte(angle: float) -> int:
    return int(max(-127, min(127, round(angle)))) + 128


def encode_scene(frame: np.ndarray, scene: Scene) -> np.ndarray:
    """Write `scene` into row 0 of `frame` (in place) and return the frame."""
    header = (*_MAGIC, scene.flags(), _byte(scene.head_down_deg), _byte(scene.yaw_deg), 0)
    for col, value in enumerate(header):
        frame[0, col] = value
    return frame


def decode_scene(image: np.ndarray) -> Optional[Scene]:
    """The scene a sim frame carries, or None for a frame sim.camera did not render."""
    if image is None or image.ndim != 3 or image.shape[1] < _HEADER_LEN:
        return None
    row = [int(image[0, col, 0]) for col in range(_HEADER_LEN)]
    if tuple(row[:2]) != _MAGIC:
        return None
    flags = row[2]
    return Scene(
        face=bool(flags & _FLAG_FACE), eyes_closed=bool(flags & _FLAG_EYES_CLOSED),
        head_down_deg=float(row[3] - 128), yaw_deg=float(row[4] - 128),
        phone=bool(flags & _FLAG_PHONE), person=bool(flags & _FLAG_PERSON),
        hand_on_phone=bool(flags & _FLAG_HAND), label="",
    )


# ---------------------------------------------------------------------- face geometry
# The same construction as tests/test_driver_attention_scenarios.py (validated in
# tests/test_head_pose_geometry.py): a described 3D head rotation projected
# through the module's own camera model, so the module's solvePnP recovers it.
_R_NEUTRAL = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
_TVEC = np.array([0.0, 0.0, 600.0])
_POSE_ORDER = ("nose_tip", "chin", "left_eye_corner", "right_eye_corner", "left_mouth", "right_mouth")
_MESH_SIZE = 478


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _eye_points(x0: float, y0: float, scale: float, closed: bool) -> dict[str, tuple[float, float]]:
    """One eye's landmarks as offsets from its first corner.

    Open: EAR ~0.4, lid gap / iris diameter ~1.0. Closed: the lids meet (EAR
    ~0.07, iris-normalised gap ~0.17) -- a real closure by both measures the
    module uses, so the iris veto agrees rather than being switched off.
    """
    lid = (1.0 if closed else 6.0) * scale
    return {
        "corner_a": (x0, y0), "corner_b": (x0 + 30 * scale, y0),
        "upper_a": (x0 + 8 * scale, y0 - lid), "lower_a": (x0 + 8 * scale, y0 + lid),
        "upper_b": (x0 + 22 * scale, y0 - lid), "lower_b": (x0 + 22 * scale, y0 + lid),
        "iris_center": (x0 + 15 * scale, y0), "iris_right": (x0 + 21 * scale, y0),
        "iris_left": (x0 + 9 * scale, y0),
    }


def face_landmarks(width: int, height: int, *, head_down_deg: float = 0.0, yaw_deg: float = 0.0,
                   eyes_closed: bool = False) -> list[tuple[float, float]]:
    """A MediaPipe-shaped (478, normalised) landmark list for a described head."""
    from modules.driver_attention import _MODEL_POINTS_3D, POSE_LANDMARK_IDS

    focal = float(width)  # the module's camera model when camera_hfov_deg is unset
    k = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
    rotation = _rot_x(head_down_deg) @ _rot_y(yaw_deg) @ _R_NEUTRAL
    rvec, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, _TVEC, k, np.zeros((4, 1)))
    pose_px = projected.reshape(-1, 2)
    nose = pose_px[0]
    points = [(float(nose[0] / width), float(nose[1] / height))] * _MESH_SIZE
    points = list(points)
    for key, (px, py) in zip(_POSE_ORDER, pose_px, strict=True):
        points[POSE_LANDMARK_IDS[key]] = (px / width, py / height)

    scale = width / 640.0
    # Right eye, anchored on landmark 33 (shared with head pose, as in MediaPipe).
    r0x, r0y = points[33][0] * width, points[33][1] * height
    right = _eye_points(r0x, r0y, scale, eyes_closed)
    for idx, key in ((133, "corner_b"), (160, "upper_a"), (144, "lower_a"), (158, "upper_b"),
                     (153, "lower_b"), (468, "iris_center"), (469, "iris_right"), (471, "iris_left")):
        points[idx] = (right[key][0] / width, right[key][1] / height)
    points[470] = points[472] = points[468]
    # Left eye: landmark 263 is its corner_b.
    l0x, l0y = points[263][0] * width - 30 * scale, points[263][1] * height
    left = _eye_points(l0x, l0y, scale, eyes_closed)
    for idx, key in ((362, "corner_a"), (385, "upper_a"), (380, "lower_a"), (387, "upper_b"),
                     (373, "lower_b"), (473, "iris_center"), (474, "iris_right"), (476, "iris_left")):
        points[idx] = (left[key][0] / width, left[key][1] / height)
    points[475] = points[477] = points[473]
    return points


class _FaceResult:
    def __init__(self, proto: Any) -> None:
        self.multi_face_landmarks = [proto] if proto is not None else None


class ScriptedFaceMesh:
    """Stands in for mediapipe FaceMesh. Returns real NormalizedLandmarkList
    protos, so the module's own drawing code renders them on the live feed."""

    model_name = "sim-scripted-face-mesh"

    def __init__(self) -> None:
        self.calls = 0
        self._cache: dict[tuple, Any] = {}

    def process(self, rgb: np.ndarray) -> _FaceResult:
        self.calls += 1
        scene = decode_scene(rgb)
        if scene is None or not scene.face:
            return _FaceResult(None)
        h, w = rgb.shape[:2]
        key = (w, h, scene.head_down_deg, scene.yaw_deg, scene.eyes_closed)
        proto = self._cache.get(key)
        if proto is None:
            from mediapipe.framework.formats import landmark_pb2

            proto = landmark_pb2.NormalizedLandmarkList(landmark=[
                landmark_pb2.NormalizedLandmark(x=x, y=y, z=0.0)
                for x, y in face_landmarks(w, h, head_down_deg=scene.head_down_deg,
                                           yaw_deg=scene.yaw_deg, eyes_closed=scene.eyes_closed)
            ])
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = proto
        return _FaceResult(proto)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------- object boxes
def phone_box(width: int, height: int) -> tuple[float, float, float, float]:
    """Where a held phone sits: lower middle of the frame, below the face."""
    return (0.42 * width, 0.62 * height, 0.56 * width, 0.86 * height)


def person_box(width: int, height: int) -> tuple[float, float, float, float]:
    return (0.25 * width, 0.08 * height, 0.75 * width, 0.98 * height)


class _Box:
    def __init__(self, class_id: int, xyxy, conf: float) -> None:
        self.cls = [class_id]
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]


class _Result:
    def __init__(self, boxes: list) -> None:
        self.boxes = boxes


class ScriptedDetector:
    """Stands in for the object detector (YOLO/Hailo): boxes from the scene.

    `faults.detector_failing` makes predict() raise the way a wedged NPU stream
    does, so the module's own error path, backoff and recovery all run.
    """

    names: ClassVar[dict[int, str]] = {0: "person", 39: "bottle", 41: "cup", 67: "cell phone"}
    model_name = "sim-scripted-detector"

    def __init__(self, faults) -> None:
        self._faults = faults
        self.calls = 0

    def predict(self, frame: np.ndarray, **_kwargs) -> list[_Result]:
        self.calls += 1
        if self._faults.current().detector_failing:
            raise RuntimeError("SIMULATED detector fault: HAILO_STREAM_ABORTED_BY_USER")
        scene = decode_scene(frame)
        h, w = frame.shape[:2]
        boxes: list[_Box] = []
        if scene is not None and scene.person:
            boxes.append(_Box(0, person_box(w, h), 0.92))
        if scene is not None and scene.phone:
            boxes.append(_Box(67, phone_box(w, h), 0.88))
        return [_Result(boxes)]


class _HandLandmarks:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        self.landmark = [type("L", (), {"x": x, "y": y})() for x, y in points]


class _HandResult:
    def __init__(self, hands: Optional[list]) -> None:
        self.multi_hand_landmarks = hands


class ScriptedHands:
    """Stands in for MediaPipe Hands: one hand wrapped around a held phone."""

    model_name = "sim-scripted-hands"

    def __init__(self) -> None:
        self.calls = 0

    def process(self, rgb: np.ndarray) -> _HandResult:
        self.calls += 1
        scene = decode_scene(rgb)
        if scene is None or not (scene.phone and scene.hand_on_phone):
            return _HandResult(None)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = phone_box(w, h)
        corners = [(x1 / w, y1 / h), (x2 / w, y1 / h), (x1 / w, y2 / h), (x2 / w, y2 / h)]
        return _HandResult([_HandLandmarks(corners)])

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------- installation
def supports(module) -> bool:
    """Whether scripted perception can stand in for this module's models."""
    from modules.driver_attention import DriverAttentionModule
    from modules.shared_perception import SharedPerceptionModule

    return isinstance(module, (DriverAttentionModule, SharedPerceptionModule))


def install(module, faults, detector: Optional[ScriptedDetector] = None) -> Any:
    """Replace a driver_attention module's model calls with scripted ones.

    Everything else on the module is untouched. The real FaceMesh graph the
    constructor built is closed first (it holds native threads until closed).
    Returns the scripted detector, so a caller can count its calls. Pass one
    `detector` to share it across module builds, as the device shares its one
    loaded detector: the runtime's perception engine (core/perception.py)
    wraps it, and `detector.calls` then counts every real inference on the
    device, whoever asked.
    """
    if not supports(module):
        raise TypeError(f"scripted perception supports driver_attention and shared_perception, "
                        f"not {type(module).__name__}")
    from modules.shared_perception import SharedPerceptionModule

    if isinstance(module, SharedPerceptionModule):
        # It owns no graphs: only the detector it would load lazily.
        detector = detector if detector is not None else ScriptedDetector(faults)
        module._yolo_model = detector
        return detector
    real_face_mesh = getattr(module, "_face_mesh", None)
    if real_face_mesh is not None and not isinstance(real_face_mesh, ScriptedFaceMesh):
        with contextlib.suppress(Exception):  # closing a graph we discard must not stop the sim
            real_face_mesh.close()
    module._face_mesh = ScriptedFaceMesh()
    detector = detector if detector is not None else ScriptedDetector(faults)
    # What the module's lazy loader (_ensure_object_detectors_loaded) would
    # have done with a real model, done with the scripted one.
    module._yolo_model = detector
    module._yolo_class_ids = {
        class_id: module._yolo_class_map[name]
        for class_id, name in detector.names.items() if name in module._yolo_class_map
    }
    if getattr(module, "_enable_hands", False):
        module._hands = ScriptedHands()
    return detector
