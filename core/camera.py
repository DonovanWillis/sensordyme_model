"""Capture abstraction that auto-detects Raspberry Pi vs. Mac/dev at runtime.

On a Raspberry Pi with Camera Module 3, frames come from Picamera2. On a Mac
(or any machine without Picamera2), frames come from cv2.VideoCapture(0), the
built-in webcam. Callers only ever see :meth:`Camera.frames`.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np


def _detect_pi() -> bool:
    """Return True if Picamera2 is importable, i.e. this is a Raspberry Pi."""
    try:
        from picamera2 import Picamera2  # noqa: F401

        return True
    except ImportError:
        return False


class CameraError(RuntimeError):
    """A frame could not be read (or the device could not be opened).

    Raised by :meth:`Camera.read` so the capture service can retry/reopen and
    report health instead of the whole process dying on one bad read.
    """


class Camera:
    """FPS-throttled frame source: Picamera2 on Pi, cv2 webcam on Mac/dev."""

    def __init__(self, config: dict) -> None:
        """Open the platform-appropriate capture device.

        Args:
            config: The `camera` section of config.yaml (width, height, fps, device_index).
        """
        self._width = int(config.get("width", 640))
        self._height = int(config.get("height", 480))
        self._fps = float(config.get("fps", 30))
        self._device_index = int(config.get("device_index", 0))
        # Device mounted upside down (ribbon exit at the top): flip at the sensor
        # on the Pi (free — done by the ISP), in software on the dev webcam path.
        self._rotate_180 = bool(config.get("rotate_180", False))

        self.is_pi = _detect_pi()
        self._picam: Optional[object] = None
        self._cap: Optional[cv2.VideoCapture] = None

        if self.is_pi:
            print("SensorDyme: detected Raspberry Pi -> using Picamera2 (Camera Module 3)")
            self._start_picamera()
        else:
            print("SensorDyme: Picamera2 not found -> using Mac/dev mode (cv2.VideoCapture webcam)")
            self._start_cv2_capture()

    def _start_picamera(self) -> None:
        """Configure and start the Picamera2 sensor."""
        from picamera2 import Picamera2

        self._picam = Picamera2()
        kwargs = {}
        if self._rotate_180:
            from libcamera import Transform  # Pi-only, like Picamera2 itself

            kwargs["transform"] = Transform(hflip=1, vflip=1)
        video_config = self._picam.create_video_configuration(
            main={"size": (self._width, self._height), "format": "RGB888"}, **kwargs
        )
        self._picam.configure(video_config)
        self._picam.start()
        time.sleep(1.0)  # let auto-exposure/white-balance settle

    def _start_cv2_capture(self) -> None:
        """Open the Mac webcam via OpenCV."""
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open webcam at index {self._device_index}. On macOS, grant "
                "camera access in System Settings > Privacy & Security > Camera."
            )

    def read(self) -> np.ndarray:
        """Capture and return one BGR frame.

        Raises:
            CameraError: on a failed read, so the caller (services/capture.py)
                can retry or :meth:`reopen` instead of the process dying.
        """
        try:
            if self.is_pi:
                frame = self._picam.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise CameraError("Failed to read a frame from the webcam.")
            if self._rotate_180:  # Pi flips at the sensor; webcam path flips here
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame
        except CameraError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize backend-specific errors
            raise CameraError(f"camera read failed: {exc}") from exc

    def reopen(self) -> None:
        """Release and re-open the capture device (recovery path after read failures)."""
        self.release()
        if self.is_pi:
            self._start_picamera()
        else:
            self._start_cv2_capture()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames at the configured FPS, forever, until the caller stops iterating.

        Frames are produced one at a time and handed directly to the caller; nothing
        about the frame's content is written anywhere by this method.
        """
        frame_interval = 1.0 / self._fps if self._fps > 0 else 0.0
        last_time = 0.0
        while True:
            elapsed = time.monotonic() - last_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_time = time.monotonic()

            # PRIVACY: frames are never persisted — this array lives only in process
            # memory for the duration of one loop iteration and is discarded after
            # the caller finishes processing it. No frame is ever written to disk,
            # logged, or transmitted anywhere in this codebase.
            yield self.read()

    @property
    def fps(self) -> float:
        """The configured capture rate (frames/second)."""
        return self._fps

    def release(self) -> None:
        """Release the underlying camera device."""
        if self._picam is not None:
            self._picam.stop()
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
