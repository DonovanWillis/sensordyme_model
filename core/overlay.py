"""sd-overlay/1: what the AI saw on a frame, sent beside that frame (docs/LIVE_OVERLAY.md).

Before M13 the only way to see the device's perception was boxes and text
burned into the JPEG by `module.annotate()`: a fixed look, the face mesh
tessellation and debug readouts in green OpenCV text, the same for every
viewer. The app could not draw a clean consumer view, nor a polished AI view.

sd-overlay/1 sends the CLEAN camera frame and, in the same multipart part,
an `X-SD-Overlay` header carrying a compact JSON description built from the
shared perception engine's Observation for that very frame plus what the
active module derived (driver association, behaviour phases, face outline).
The app draws it.

Truthfulness rules, enforced here and pinned by tests/test_m13_overlay.py:
* Only observations from the inference that produced this frame's analysis.
  Detection boxes come from the Observation; when detection is unavailable
  the field is None ("could not look"), never [] ("looked, nothing there").
* No confidence numbers: a detector score is not a probability that the
  person is right, and a percentage on screen reads like one.
* Head orientation is labelled head pose. Nothing here is eye gaze.
* A simulated source (virtual device, replay) says so on every frame.
* Capabilities that are not working are listed, so the app hides or labels
  them instead of drawing nothing as if nothing were there.
* Performance numbers are measured on this device by this process: the
  frame interval EMA and the models' own run-time EMAs.

Size: about 1-2 KB per frame (68 face points, a few boxes), base64url in one
header line. Only viewers that ask (`X-Stream-Overlay: 1`) receive it.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

SCHEMA = "sd-overlay/1"
#: Labels the overlay draws. Everything the detector reports is drawn by its
#: own label; these only map the detector's names to the app's words.
_LABELS = {"cell phone": "phone"}
_EMA = 0.2
#: Most boxes sent per frame (a crowd is summarised, not streamed).
MAX_BOXES = 24


def _n(v: float, size: int) -> float:
    return round(min(1.0, max(0.0, float(v) / size)), 4) if size else 0.0


def _norm_box(box: Any, w: int, h: int) -> Optional[list[float]]:
    try:
        x1, y1, x2, y2 = (float(b) for b in box)
    except (TypeError, ValueError):
        return None
    return [_n(x1, w), _n(y1, h), _n(x2, w), _n(y2, h)]


def _display_pitch(pitch: float, calibrated: bool) -> float:
    """Pitch for display, in [-90, 90].

    A calibrated pitch is relative to the driver's neutral pose and already in
    range. An uncalibrated one is the raw Euler fit, whose convention puts an
    upright head near +-180; it is wrapped by 180 deg so "forward" reads ~0.
    """
    p = float(pitch)
    if not calibrated:
        if p > 90.0:
            p -= 180.0
        elif p < -90.0:
            p += 180.0
    return round(max(-90.0, min(90.0, p)), 1)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class OverlayBuilder:
    """Builds one sd-overlay/1 document per streamed frame. One per runtime."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._last_captured: Optional[float] = None
        self._fps: Optional[float] = None

    def note_frame(self, captured_at: Optional[float]) -> None:
        """Every ANALYSED frame (not only streamed ones): the processed-fps EMA."""
        if captured_at is None:
            return
        if self._last_captured is not None:
            dt = float(captured_at) - self._last_captured
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps is None else (1 - _EMA) * self._fps + _EMA * inst
        self._last_captured = float(captured_at)

    def build(self, observation: Any, extras: Optional[dict] = None, *, module: str = "",
              mode: Optional[str] = None, last_event: Optional[dict] = None,
              alert_active: bool = False, captured_at: Optional[float] = None) -> dict:
        extras = extras or {}
        doc: dict[str, Any] = {"schema": SCHEMA, "module": module, "mode": mode}
        if observation is None:
            doc.update(analysed=False, sim=False, w=0, h=0, det=None, face=None, unavailable=["perception"])
            return doc
        w, h = int(observation.width), int(observation.height)
        doc.update(analysed=True, sim=bool(observation.simulated), source=observation.source, w=w, h=h,
                   frame_id=int(observation.frame_id))

        driver = extras.get("driver") or {}
        driver_box = driver.get("box")
        driver_phones = [tuple(b) for b in extras.get("driver_phone_boxes") or []]

        # -- detections: the Observation's own boxes, tracker ids, a role where the
        # module decided one. No confidence values (see the module docstring).
        if observation.detections is None:
            doc["det"] = None
        else:
            boxes = []
            for d in list(observation.detections)[:MAX_BOXES]:
                nb = _norm_box(d.box, w, h)
                if nb is None:
                    continue
                item: dict[str, Any] = {"label": _LABELS.get(d.label, d.label), "box": nb}
                if d.track_id is not None:
                    item["track"] = int(d.track_id)
                if d.label == "person" and driver_box is not None and _iou(d.box, driver_box) > 0.5:
                    item["role"] = "driver"
                elif d.label == "cell phone" and (driver_phones or driver_box is not None):
                    # The driver is known: a phone that is not one of the driver's is
                    # someone else's ("other"), never left undecided (M13 G1).
                    item["role"] = "driver" if any(_iou(d.box, p) > 0.5 for p in driver_phones) else "other"
                boxes.append(item)
            doc["det"] = boxes
            doc["det_age_s"] = (None if observation.detections_age_s is None
                                else round(float(observation.detections_age_s), 2))
            doc["det_fresh"] = bool(observation.fresh_detections)

        # -- face: whether the landmark model ran, what it measured.
        face = observation.face
        if observation.face_visible is None:
            doc["face"] = None  # not looked for on this frame
        elif not observation.face_visible or face is None:
            doc["face"] = {"visible": False}
        else:
            f: dict[str, Any] = {"visible": True}
            contours = extras.get("face_contours")
            if isinstance(contours, dict):
                f["contours"] = {k: [list(p) for p in v] for k, v in contours.items()}
            eye = face.eye_state
            if eye is not None and eye.closed is not None:
                f["eyes_closed"] = bool(eye.closed)
            head = extras.get("head")
            if isinstance(head, dict):
                f["head_pose"] = {**head, "kind": "head_pose"}  # never eye gaze
            elif face.head_pose is not None:
                f["head_pose"] = {"yaw_deg": round(face.head_pose.yaw_deg, 1),
                                  "pitch_deg": face.head_pose.pitch_deg,
                                  "calibrated": bool(face.head_pose.calibrated), "kind": "head_pose"}
            hp = f.get("head_pose")
            if hp is not None and hp.get("pitch_deg") is not None:
                hp["pitch_deg"] = _display_pitch(hp["pitch_deg"], bool(hp.get("calibrated")))
            doc["face"] = f

        if observation.hands is not None:
            doc["hands"] = [nb for hb in observation.hands if (nb := _norm_box(hb.box, w, h)) is not None]

        if driver_box is not None:
            doc["driver"] = {"box": _norm_box(driver_box, w, h), "basis": driver.get("method"),
                             "identity": driver.get("identity"), "people": driver.get("people")}
        region = extras.get("driver_region")
        if isinstance(region, dict) and region.get("box") is not None:
            doc["driver_region"] = {"box": [round(float(v), 4) for v in region["box"]],
                                    "source": region.get("source")}

        if "behaviours" in extras:
            doc["behaviours"] = list(extras["behaviours"])
        if "state" in extras:
            doc["state"] = extras["state"]
        doc["alert_active"] = bool(alert_active)
        if last_event:
            doc["last_event"] = last_event

        # -- what is not working: hide or label it, never draw nothing.
        doc["unavailable"] = sorted(name for name, status in observation.components.items()
                                    if getattr(status, "state", "ok") not in ("ok", "idle"))

        # -- measured performance on this device.
        perf: dict[str, Any] = {"fps": None if self._fps is None else round(self._fps, 1)}
        for name, status in observation.components.items():
            run_ms = getattr(status, "run_ms", None)
            if run_ms is not None:
                perf[f"{name}_ms"] = round(float(run_ms), 1)
        stamp = captured_at if captured_at is not None else observation.captured_at
        if stamp is not None:
            perf["frame_age_ms"] = max(0, round((self._clock() - float(stamp)) * 1000))
        doc["perf"] = perf
        return doc


def encode_header(doc: dict) -> str:
    """One header line: base64url (no padding) of compact JSON."""
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_header(value: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
