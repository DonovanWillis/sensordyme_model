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
  MediaPipe Pose      -> ScriptedPose (baby monitoring): never finds a pose. The
                         simulator scripts no body landmarks, so pose-based
                         signals (flailing, choking-hazard scoring) never fire
                         here -- they are unevaluated and stay unclaimed

Besides driver_attention, the same stand-ins run the room modules on a general
device (M8): shared_perception (occupancy, workplace safety, custom rules),
occupancy (security_monitoring) and baby_monitoring. A scene can hold extra
standing people at given horizontal positions (`Scene.people`), so counting,
tracking, area crossings and dwell are exercised with real moving boxes.

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
#: Column 5 holds how many extra people follow, one column each from column 6.
MAX_PEOPLE = 8
PEOPLE_WIDTH = 0.16
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
    #: Extra standing people, by the horizontal centre of each box (0..1 of the
    #: frame width; at most MAX_PEOPLE). Each is a PEOPLE_WIDTH-wide box from
    #: 30% to 95% of the frame height, so its feet are at (x, 0.95).
    people: tuple[float, ...] = ()

    def flags(self) -> int:
        return ((_FLAG_FACE if self.face else 0) | (_FLAG_EYES_CLOSED if self.eyes_closed else 0)
                | (_FLAG_PHONE if self.phone else 0) | (_FLAG_PERSON if self.person else 0)
                | (_FLAG_HAND if self.hand_on_phone else 0))


def _byte(angle: float) -> int:
    return int(max(-127, min(127, round(angle)))) + 128


def encode_scene(frame: np.ndarray, scene: Scene) -> np.ndarray:
    """Write `scene` into row 0 of `frame` (in place) and return the frame."""
    people = [min(250, max(0, round(float(x) * 250))) for x in scene.people[:MAX_PEOPLE]]
    header = (*_MAGIC, scene.flags(), _byte(scene.head_down_deg), _byte(scene.yaw_deg), len(people), *people)
    if frame.shape[1] < len(header):
        raise ValueError("frame too narrow to carry this scene")
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
    count = min(row[5], MAX_PEOPLE, max(0, image.shape[1] - _HEADER_LEN))
    people = tuple(int(image[0, _HEADER_LEN + i, 0]) / 250.0 for i in range(count))
    return Scene(
        face=bool(flags & _FLAG_FACE), eyes_closed=bool(flags & _FLAG_EYES_CLOSED),
        head_down_deg=float(row[3] - 128), yaw_deg=float(row[4] - 128),
        phone=bool(flags & _FLAG_PHONE), person=bool(flags & _FLAG_PERSON),
        hand_on_phone=bool(flags & _FLAG_HAND), label="", people=people,
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


def standing_box(x: float, width: int, height: int) -> tuple[float, float, float, float]:
    """An extra standing person centred at `x` (0..1): feet at (x, 0.95)."""
    half = PEOPLE_WIDTH / 2.0
    return ((x - half) * width, 0.30 * height, (x + half) * width, 0.95 * height)


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
        for x in scene.people if scene is not None else ():
            boxes.append(_Box(0, standing_box(x, w, h), 0.9))
        wanted = _kwargs.get("classes")
        if wanted is not None:
            allowed = {int(c) for c in wanted}
            boxes = [b for b in boxes if b.cls[0] in allowed]
        conf = _kwargs.get("conf")
        if conf is not None:
            boxes = [b for b in boxes if b.conf[0] >= float(conf)]
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


class _PoseResult:
    pose_landmarks = None


class ScriptedPose:
    """Stands in for MediaPipe Pose: it runs, and finds no pose. The simulator
    scripts no body landmarks, so nothing pose-based is exercised (or claimed)."""

    model_name = "sim-scripted-pose-none"

    def __init__(self) -> None:
        self.calls = 0

    def process(self, _rgb: np.ndarray) -> _PoseResult:
        self.calls += 1
        return _PoseResult()

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------- installation
def _scripted_classes() -> tuple[type, ...]:
    from modules.baby_monitoring import BabyMonitoringModule
    from modules.driver_attention import DriverAttentionModule
    from modules.occupancy import OccupancyModule
    from modules.shared_perception import SharedPerceptionModule

    return (DriverAttentionModule, SharedPerceptionModule, OccupancyModule, BabyMonitoringModule)


def supports(module) -> bool:
    """Whether scripted perception can stand in for this module's models."""
    return isinstance(module, _scripted_classes())


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
        raise TypeError(f"scripted perception supports driver_attention, shared_perception, occupancy "
                        f"and baby_monitoring, not {type(module).__name__}")
    from modules.baby_monitoring import BabyMonitoringModule
    from modules.occupancy import OccupancyModule
    from modules.shared_perception import SharedPerceptionModule

    if isinstance(module, SharedPerceptionModule):
        # It owns no graphs: only the detector it would load lazily.
        detector = detector if detector is not None else ScriptedDetector(faults)
        module._yolo_model = detector
        return detector
    if isinstance(module, (OccupancyModule, BabyMonitoringModule)):
        detector = detector if detector is not None else ScriptedDetector(faults)
        module._model = detector  # what get_yolo_model() would have handed it on the first frame
        real_pose = getattr(module, "_pose", None)
        if real_pose is not None and not isinstance(real_pose, ScriptedPose):
            with contextlib.suppress(Exception):  # closing a graph we discard must not stop the sim
                real_pose.close()
            module._pose = ScriptedPose()
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


# ====================================================================== M8.4 cabin scenarios
# A richer, DIRECT scripted cabin for the behavioural false-positive regression
# suite (tests/test_m84_behavior_*.py, scripts/eval_behavior.py). Unlike the
# row-0 Scene above (what the live virtual device renders), a CabinScene is
# handed to the stand-ins directly and can say things a frame header cannot:
# a phone in a dash mount, a passenger holding one, sunglasses, a blocked or
# dark lens, a moved camera, noisy landmark fits, face-model dropouts, stalls.
# The REAL DriverAttentionModule runs between the stand-ins. SYNTHETIC
# REGRESSION EVIDENCE ONLY: it shows the logic does what it is designed to on
# constructed inputs; it says nothing about real cabins, faces or detectors.

CABIN_W, CABIN_H = 640, 480
#: Geometry at 640x480 (the neutral face spans roughly x 274..366, y 190..308).
CABIN_DRIVER_BOX = (200, 60, 460, 470)
CABIN_PASSENGER_BOX = (0, 120, 190, 470)  # smaller than the driver's, frame left
CABIN_PASSENGER_LEANING_BOX = (0, 0, 300, 480)  # larger than the driver's: the largest-person proxy picks it
CABIN_LAP_PHONE = (295, 370, 345, 440)
CABIN_LAP_HAND = (285, 385, 355, 460)
CABIN_MOUNT_PHONE = (360, 330, 410, 400)  # dash mount in front of the driver, inside the driver's box
CABIN_PASSENGER_PHONE = (20, 300, 70, 380)
CABIN_PASSENGER_HAND = (10, 310, 80, 400)
CABIN_WHEEL_HANDS = ((215, 405, 275, 460), (395, 415, 455, 465))


@dataclass(frozen=True)
class CabinScene:
    """What is in the cabin at one moment (sim convention: +head_down = looking down)."""

    face: bool = True
    eyes_closed: bool = False
    sunglasses: bool = False  # eye patches dark and flat (the module's occlusion check)
    head_down_deg: float = 0.0
    yaw_deg: float = 0.0
    pose_jitter_deg: float = 0.0  # per-frame head-pose noise (poor light, partial occlusion)
    face_dropout: float = 0.0  # probability the face model misses the face on a frame
    phone: str = "none"  # none | lap | mount | passenger
    phone_visible_prob: float = 1.0  # detector recall on the phone per tick
    hand_on_phone: bool = False
    wheel_hands: bool = True  # the driver's hands on the wheel, visible to the hand model
    person: bool = True
    passenger: str = "none"  # none | seated | leaning (larger than the driver)
    passenger_hand_on_phone: bool = False
    brightness: int = 130  # frame mean
    camera_blocked: bool = False  # lens covered: a near-black, flat frame; nothing detected
    detector_failing: bool = False
    #: Seconds of frame gap between bursts (0 = none) and the burst length.
    stall_gap_s: float = 0.0
    stall_burst_s: float = 0.4
    label: str = ""


@dataclass(frozen=True)
class CabinSegment:
    start: float
    scene: CabinScene = CabinScene()


@dataclass(frozen=True)
class CabinLabel:
    """Ground truth for one scenario (synthetic: what the SCRIPT did, by construction)."""

    #: (event type, onset_s, end_s): windows in which the event is TRUE
    expected: tuple[tuple[str, float, float], ...] = ()
    #: event types that are acceptable but not required (e.g. attention_lost for a blocked lens)
    allowed: tuple[str, ...] = ()


@dataclass(frozen=True)
class CabinScenario:
    name: str
    description: str
    segments: tuple[CabinSegment, ...]
    duration: float
    label: CabinLabel = CabinLabel()
    kind: str = "false_positive"  # false_positive | true_positive | abstention
    seed: int = 7

    def at(self, t: float) -> CabinScene:
        current = self.segments[0].scene
        for segment in self.segments:
            if segment.start <= t:
                current = segment.scene
            else:
                break
        return current


class _Lm:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


class _CabinFaceProto:
    def __init__(self, landmark: list) -> None:
        self.landmark = landmark


class _CabinFaceResult:
    def __init__(self, proto: Any) -> None:
        self.multi_face_landmarks = [proto] if proto is not None else None


class CabinFaceMesh:
    """FaceMesh stand-in reading the cabin's current scene."""

    model_name = "sim-cabin-face-mesh"

    def __init__(self, cabin: "ScriptedCabin") -> None:
        self.cabin = cabin
        self.calls = 0

    def process(self, _rgb) -> _CabinFaceResult:
        self.calls += 1
        lms = self.cabin.face_landmarks_now()
        return _CabinFaceResult(None if lms is None else _CabinFaceProto(lms))

    def close(self) -> None:
        return None


class CabinDetector:
    """Detector stand-in: person / phone boxes from the cabin's current scene."""

    names: ClassVar[dict[int, str]] = {0: "person", 39: "bottle", 41: "cup", 67: "cell phone"}
    model_name = "sim-cabin-detector"

    def __init__(self, cabin: "ScriptedCabin") -> None:
        self.cabin = cabin
        self.calls = 0

    def predict(self, frame, **kwargs) -> list[_Result]:
        self.calls += 1
        scene = self.cabin.scene
        if scene.detector_failing:
            raise RuntimeError("SIMULATED detector fault")
        boxes: list[_Box] = []
        if not scene.camera_blocked:
            if scene.person:
                boxes.append(_Box(0, CABIN_DRIVER_BOX, 0.92))
            if scene.passenger == "seated":
                boxes.append(_Box(0, CABIN_PASSENGER_BOX, 0.9))
            elif scene.passenger == "leaning":
                boxes.append(_Box(0, CABIN_PASSENGER_LEANING_BOX, 0.9))
            phone = self.cabin.phone_box()
            if phone is not None and self.cabin.rng.random() < scene.phone_visible_prob:
                boxes.append(_Box(67, phone, 0.8))
        conf = kwargs.get("conf")
        if conf is not None:
            boxes = [b for b in boxes if b.conf[0] >= float(conf)]
        return [_Result(boxes)]


class CabinHands:
    """MediaPipe Hands stand-in: hand boxes from the scene; counts its calls (B5 cost)."""

    model_name = "sim-cabin-hands"

    def __init__(self, cabin: "ScriptedCabin") -> None:
        self.cabin = cabin
        self.calls = 0

    def process(self, _rgb) -> _HandResult:
        self.calls += 1
        scene = self.cabin.scene
        if scene.camera_blocked:
            return _HandResult(None)
        boxes: list[tuple] = []
        if scene.hand_on_phone and scene.phone == "lap":
            boxes.append(CABIN_LAP_HAND)
        if scene.wheel_hands and scene.person:
            boxes.extend(CABIN_WHEEL_HANDS)
        if scene.passenger_hand_on_phone and scene.phone == "passenger":
            boxes.append(CABIN_PASSENGER_HAND)
        hands = [_HandLandmarks([(x / CABIN_W, y / CABIN_H) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))])
                 for x1, y1, x2, y2 in boxes]
        return _HandResult(hands or None)

    def close(self) -> None:
        return None


class ScriptedCabin:
    """The current scene, the stand-ins reading it, and the frames it renders."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.scene = CabinScene()
        self._textures: dict[int, np.ndarray] = {}
        self._landmarks: Optional[list] = None

    def set_scene(self, scene: CabinScene) -> None:
        self.scene = scene

    def phone_box(self) -> Optional[tuple]:
        return {"lap": CABIN_LAP_PHONE, "mount": CABIN_MOUNT_PHONE,
                "passenger": CABIN_PASSENGER_PHONE}.get(self.scene.phone)

    def face_landmarks_now(self) -> Optional[list]:
        return self._landmarks

    def _texture(self, brightness: int) -> np.ndarray:
        frame = self._textures.get(brightness)
        if frame is None:
            rng = np.random.default_rng(brightness)
            spread = max(4, min(70, brightness // 2))
            low, high = max(0, brightness - spread), min(255, brightness + spread)
            frame = rng.integers(low, high + 1, (CABIN_H, CABIN_W, 3), dtype=np.uint8)
            self._textures[brightness] = frame
        return frame

    def render(self) -> np.ndarray:
        """This frame: decides the face landmarks (dropout, jitter) and paints lenses."""
        scene = self.scene
        if scene.camera_blocked:
            self._landmarks = None
            return np.full((CABIN_H, CABIN_W, 3), 3, dtype=np.uint8)
        frame = self._texture(int(scene.brightness)).copy()
        if not scene.face or (scene.face_dropout > 0 and self.rng.random() < scene.face_dropout):
            self._landmarks = None
            return frame
        jitter = scene.pose_jitter_deg
        down = scene.head_down_deg + (float(self.rng.normal(0, jitter)) if jitter else 0.0)
        yaw = scene.yaw_deg + (float(self.rng.normal(0, jitter)) if jitter else 0.0)
        points = face_landmarks(CABIN_W, CABIN_H, head_down_deg=down, yaw_deg=yaw, eyes_closed=scene.eyes_closed)
        self._landmarks = [_Lm(x, y) for x, y in points]
        if scene.sunglasses:
            for anchor in (33, 362):
                x0, y0 = points[anchor][0] * CABIN_W, points[anchor][1] * CABIN_H
                frame[max(0, int(y0 - 15)):int(y0 + 15), max(0, int(x0 - 10)):int(x0 + 40)] = 10
        return frame


def install_cabin(module, cabin: ScriptedCabin) -> dict:
    """Replace a DriverAttentionModule's model calls with the cabin's stand-ins."""
    real = getattr(module, "_face_mesh", None)
    if real is not None:
        with contextlib.suppress(Exception):
            real.close()
    face, detector, hands = CabinFaceMesh(cabin), CabinDetector(cabin), CabinHands(cabin)
    module._face_mesh = face
    module._yolo_model = detector
    module._yolo_class_ids = {cid: module._yolo_class_map[name] for cid, name in detector.names.items()
                              if name in module._yolo_class_map}
    if getattr(module, "_enable_hands", False):
        module._hands = hands
    return {"face": face, "detector": detector, "hands": hands}


def driver_module_config(tmp_dir: str, **overrides) -> dict:
    """config.yaml's shipped driver_attention block (the production thresholds)."""
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent
    config = dict(yaml.safe_load((repo / "config.yaml").read_text())["modules"]["driver_attention"])
    config.update({"calibration_path": str(Path(tmp_dir) / "calibration.json"), "recalibrate": True, **overrides})
    return config


#: Seconds of attentive driving before every cabin scenario's script starts
#: (calibration_duration_s is 10 s in config.yaml).
CABIN_LEAD_IN_S = 12.0


def run_cabin(scenario: CabinScenario, tmp_dir: str, *, fps: float = 10.0, hands_factory=None,
              **config_overrides) -> dict:
    """Run one cabin scenario through the REAL module; returns its trace.

    Scenario time 0 is the end of the attentive lead-in; event times are on it.
    ``hands_factory`` (e.g. a real MediaPipe Hands graph) replaces the scripted
    hand model, for measuring its host cost (scripts/bench_hands_policy.py).
    """
    import io
    import time as _time
    from contextlib import redirect_stdout

    from modules.driver_attention import DriverAttentionModule

    cabin = ScriptedCabin(scenario.seed)
    with redirect_stdout(io.StringIO()):
        module: Any = DriverAttentionModule(driver_module_config(tmp_dir, **config_overrides))
        stand_ins = install_cabin(module, cabin)
        if hands_factory is not None and getattr(module, "_enable_hands", False):
            real_hands = hands_factory()
            calls = {"n": 0}

            class _Counted:
                def process(self, rgb, _real=real_hands):
                    calls["n"] += 1
                    return _real.process(rgb)

                def close(self, _real=real_hands):
                    _real.close()

            module._hands = _Counted()
            stand_ins["hands"] = type("C", (), {"calls": property(lambda self: calls["n"])})()
        dt = 1.0 / fps
        events: list[dict] = []
        frames = 0
        risk_peak, risk_sum = 0.0, 0.0
        active_s: dict[str, float] = {}
        shadow: list[dict] = []
        t = -CABIN_LEAD_IN_S
        burst_until: Optional[float] = None
        process_s = 0.0
        lead_in = True
        while t < scenario.duration:
            if lead_in and t >= 0:
                lead_in = False
                reset = getattr(getattr(module, "_engine", None), "reset_stats", None)
                if callable(reset):
                    reset()  # rates are over the scripted scenario, not the calibration lead-in
            scene = CabinScene() if t < 0 else scenario.at(t)
            cabin.set_scene(scene)
            if scene.stall_gap_s > 0:
                if burst_until is None:
                    burst_until = t + scene.stall_burst_s
                elif t >= burst_until:
                    t = round(t + scene.stall_gap_s, 6)  # the stall: no frames at all
                    burst_until = t + scene.stall_burst_s
                    continue
            else:
                burst_until = None
            frame = cabin.render()
            module.set_frame_time(t + 1000.0)
            started = _time.perf_counter()
            produced = module.process(frame)
            process_s += _time.perf_counter() - started
            frames += 1
            engine = getattr(module, "_engine", None)
            if t >= 0:
                for event in produced:
                    events.append({"t": round(t, 2), "type": event.type, "signals": event.contributing_signals})
                if engine is not None:
                    shadow += [{"t": round(t, 2), "type": claim} for claim in engine.started]
                risk = float(module.get_telemetry_snapshot().get("risk_score") or 0.0)
                risk_peak, risk_sum = max(risk_peak, risk), risk_sum + risk
                for condition in module.active_alert_conditions():
                    active_s[condition] = active_s.get(condition, 0.0) + dt
            t = round(t + dt, 6)
        engine = getattr(module, "_engine", None)
        report = engine.report() if engine is not None else None
        module.close()
    return {
        "scenario": scenario.name, "kind": scenario.kind, "events": events, "shadow_events": shadow,
        "frames": frames, "risk_peak": round(risk_peak, 3), "risk_mean": round(risk_sum / max(1, frames), 3),
        "active_s": {k: round(v, 1) for k, v in sorted(active_s.items())},
        "hands_calls": stand_ins["hands"].calls, "detector_calls": stand_ins["detector"].calls,
        "face_calls": stand_ins["face"].calls, "process_ms_per_frame": round(1000 * process_s / max(1, frames), 3),
        "engine": report,
    }
