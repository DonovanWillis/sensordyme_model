"""SimCamera: a core.camera.Camera stand-in that renders the scenario.

Each frame is a drawing of the scene (face, eyes, head pose, phone) with a
"SIMULATED" banner -- the banner is in the pixels, so the device's own MJPEG
/stream shows it whatever the app overlays -- and the scene encoded in row 0
for the scripted perception (sim.perception.decode_scene).

Camera faults come from the scenario: `frozen` raises CameraError on every
read and reopen (as a dead sensor does), `dropping` fails every other read,
`stalled` holds reads for the length of a gap. With --video the frames come
from a file instead (looped) and only the banner is added.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.camera import CameraError
from sim.perception import Scene, encode_scene, phone_box

#: `stalled`: frames flow for this long, then nothing for STALL_GAP_S.
STALL_FLOW_S = 1.0
STALL_GAP_S = 4.0
BANNER = "SIMULATED"


def in_stall_gap(t: float) -> bool:
    return (t % (STALL_FLOW_S + STALL_GAP_S)) >= STALL_FLOW_S


def render(scene: Scene, width: int, height: int, caption: str = "") -> np.ndarray:
    """A cartoon of the scene, banner included, scene encoded in row 0."""
    frame = np.full((height, width, 3), 58, dtype=np.uint8)
    cx = int(width / 2 + scene.yaw_deg / 90.0 * width * 0.25)
    cy = int(height * 0.40 + scene.head_down_deg / 90.0 * height * 0.25)
    radius = int(min(width, height) * 0.18)
    if scene.person:
        cv2.rectangle(frame, (int(width * 0.30), int(height * 0.60)), (int(width * 0.70), height - 1),
                      (90, 90, 110), -1)
    if scene.face:
        cv2.circle(frame, (cx, cy), radius, (150, 175, 205), -1)
        for side in (-1, 1):
            ex, ey = cx + side * radius // 2, cy - radius // 5
            if scene.eyes_closed:
                cv2.line(frame, (ex - radius // 6, ey), (ex + radius // 6, ey), (40, 40, 40), 2)
            else:
                cv2.circle(frame, (ex, ey), max(2, radius // 8), (40, 40, 40), -1)
        cv2.line(frame, (cx - radius // 4, cy + radius // 2), (cx + radius // 4, cy + radius // 2),
                 (60, 60, 120), 2)
    if scene.phone:
        x1, y1, x2, y2 = (int(v) for v in phone_box(width, height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.rectangle(frame, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (200, 160, 60), -1)
    _banner(frame, caption or scene.label)
    return encode_scene(frame, scene)


def _banner(frame: np.ndarray, caption: str) -> None:
    """A red "SIMULATED" bar along the bottom (the module's own overlay owns the
    top-left) and a tag in the top-right corner. Row 0 stays free for the scene."""
    height, width = frame.shape[:2]
    bar = max(18, height // 14)
    scale = bar / 30.0
    cv2.rectangle(frame, (0, height - bar), (width - 1, height - 1), (0, 0, 170), -1)
    text = f"{BANNER}  {caption}" if caption else BANNER
    cv2.putText(frame, text[:60], (8, height - int(bar * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8,
                (255, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(BANNER, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, 2)
    cv2.rectangle(frame, (width - tw - 16, 2), (width - 1, th + 14), (0, 0, 170), -1)
    cv2.putText(frame, BANNER, (width - tw - 8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7,
                (255, 255, 255), 2)


class SimCamera:
    """Duck-types core.camera.Camera for CaptureService and DeviceRuntime."""

    is_pi = False

    def __init__(self, state, *, width: int = 640, height: int = 480, fps: float = 30.0,
                 video_path: Optional[str] = None, sleep=time.sleep) -> None:
        self._state = state
        self.width, self.height = int(width), int(height)
        self._fps = float(fps)
        self._video_path = video_path
        self._video = None
        self._reads = 0
        self._released = False
        self._sleep = sleep
        self._lock = threading.Lock()
        self.frames_rendered = 0

    @property
    def fps(self) -> float:
        return self._fps

    def is_open(self) -> bool:
        return not self._released and self._state.current().camera != "frozen"

    # ------------------------------------------------------------------ frames
    def grab(self) -> Optional[np.ndarray]:
        """The frame for the scenario's current moment, or None when the camera
        fault means no frame arrives now. Never blocks (the deterministic path)."""
        with self._lock:
            self._reads += 1
            faults = self._state.current()
            if faults.camera == "frozen":
                return None
            if faults.camera == "dropping" and self._reads % 2 == 0:
                return None
            if faults.camera == "stalled" and in_stall_gap(self._state.clock.now()):
                return None
            self.frames_rendered += 1
            scene = self._state.scene()
            caption = f"{self._state.scenario.name} t={self._state.clock.now():.0f}s {scene.label}"
            if self._video_path:
                return self._video_frame(caption)
            return render(scene, self.width, self.height, caption)

    def read(self, timeout_s: Optional[float] = None) -> np.ndarray:
        """Camera.read(): a frame, or CameraError like a real sensor fault."""
        faults = self._state.current()
        if faults.camera == "stalled":
            waited, limit = 0.0, (timeout_s if timeout_s is not None else STALL_GAP_S + 1.0)
            while in_stall_gap(self._state.clock.now()) and waited < limit:
                self._sleep(0.05)
                waited += 0.05
        frame = self.grab()
        if frame is None:
            raise CameraError(f"SIMULATED camera fault: {faults.camera}")
        return frame

    def reopen(self) -> None:
        if self._state.current().camera == "frozen":
            raise CameraError("SIMULATED camera fault: sensor not responding")
        self._released = False

    def release(self) -> None:
        self._released = True
        if self._video is not None:
            self._video.release()
            self._video = None

    # ------------------------------------------------------------------ --video
    def _video_frame(self, caption: str) -> np.ndarray:
        if self._video is None:
            self._video = cv2.VideoCapture(str(self._video_path))
        ok, frame = self._video.read()
        if not ok:  # loop the file
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._video.read()
        if not ok or frame is None:
            raise CameraError(f"cannot read video {self._video_path}")
        frame = cv2.resize(frame, (self.width, self.height))
        _banner(frame, caption)
        return frame
