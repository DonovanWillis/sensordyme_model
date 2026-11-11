"""Eye state (open / closed / looking_down / unknown) and drowsiness evidence.

Two shipped weaknesses this addresses (modules/driver_attention.py):

  * The closure threshold is 0.75 x the open-eye reference EAR, and looking
    DOWN (at the lap, a phone, the cluster) lowers the upper lid by about that
    much: a long downward look reads as a closure and can fire "drowsiness".
    Here an EAR drop that comes WITH a head pitch-down or an iris downshift --
    while the iris is still visible -- is attributed to looking down, not to
    the eyes closing. A drop with the iris covered by the lid is a closure
    whatever the head is doing (a nodding-off head pitches down too).
  * Face loss resets the drowsiness timer, so a closing driver whose face
    mesh drops for half a second starts from zero. ``DrowsinessEvidence``
    COASTS through a bounded loss (``coast_s``): accumulated closure is held,
    not credited, and not reset; past ``coast_s`` it is dropped and the state
    is unknown.

Eyes that cannot be measured (sunglasses, a hand or hair over them, no eye
landmarks) are ``unknown`` -- neither open (which would suppress a real
closure) nor closed (which would invent one). One occluded eye leaves the
other to decide.

Synthetic-scenario tested only; the pitch limits and occlusion thresholds need
in-cabin validation (docs/BEHAVIOUR_REASONING.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.behavior.evidence import Assessment, FrameGapGuard, unknown

OPEN = "open"
CLOSED = "closed"
LOOKING_DOWN = "looking_down"


@dataclass(frozen=True)
class EyeConfig:
    down_head_pitch_deg: float = -10.0  # calibrated head pitch at/below this = head pitched down
    down_iris_pitch_deg: float = -8.0  # calibrated eye-in-head pitch at/below this = iris shifted down
    iris_closed: float = 0.2  # iris-openness below this: the lid covers the iris (a real closure)
    occlusion_max_mean: float = 40.0  # eye patch darker than this (0-255) ...
    occlusion_max_std: float = 8.0  # ... and flatter than this reads as occluded (sunglasses)


@dataclass(frozen=True)
class EyeObservation:
    ear: Optional[float]
    occluded: Optional[bool] = None  # None = not checked


@dataclass(frozen=True)
class EyeFrame:
    face_present: bool
    eyes: tuple[EyeObservation, ...] = ()
    closure_threshold: Optional[float] = None  # EAR below this is a closure-sized drop
    deep_threshold: Optional[float] = None  # EAR below this is a deep closure (used without an iris)
    iris_openness: Optional[float] = None
    iris_vetoes_closure: bool = False
    head_pitch_deg: Optional[float] = None
    eye_pitch_deg: Optional[float] = None


def eye_patch_occluded(patch, max_mean: float, max_std: float) -> Optional[bool]:
    """Dark and featureless: what a sunglasses lens looks like to a visible-light camera.

    ``patch`` is a 2-D grayscale array (numpy). Returns None for an empty
    patch (no opinion). An unvalidated heuristic: an IR-illuminated cabin
    sees through many lenses, and a dark cabin without IR also reads dark --
    which is reported as unknown, the honest answer for an unreadable eye.
    """
    if patch is None or getattr(patch, "size", 0) == 0:
        return None
    return bool(float(patch.mean()) < max_mean and float(patch.std()) < max_std)


def classify_eye_state(frame: EyeFrame, config: Optional[EyeConfig] = None) -> Assessment:
    c = config or EyeConfig()
    if not frame.face_present:
        return unknown("face not detected")
    visible = [e for e in frame.eyes if e.ear is not None and not e.occluded]
    if not visible:
        if any(e.occluded for e in frame.eyes):
            return unknown("eyes occluded (sunglasses or obstruction)")
        return unknown("no eye landmarks")
    if frame.closure_threshold is None:
        return unknown("no closure threshold yet")
    ear = max(e.ear for e in visible if e.ear is not None)
    if ear >= frame.closure_threshold:
        return Assessment(OPEN)
    if frame.iris_vetoes_closure:
        return Assessment(OPEN, "iris veto: eyelid EAR artifact")
    if frame.iris_openness is not None:
        deep = frame.iris_openness < c.iris_closed
    else:
        deep = frame.deep_threshold is not None and ear < frame.deep_threshold
    if deep:
        return Assessment(CLOSED, "lid covers the iris")
    head_down = frame.head_pitch_deg is not None and frame.head_pitch_deg <= c.down_head_pitch_deg
    iris_down = frame.eye_pitch_deg is not None and frame.eye_pitch_deg <= c.down_iris_pitch_deg
    if head_down or iris_down:
        why = "head pitched down" if head_down else "iris shifted down"
        return Assessment(LOOKING_DOWN, f"EAR drop with {why}, iris visible")
    return Assessment(CLOSED, "EAR drop without downward gaze")


class DrowsinessEvidence:
    """Sustained-closure evidence that survives a brief face loss.

    Closed time is credited per frame (FrameGapGuard: a stalled stretch
    credits nothing). ``release_frames`` consecutive open/looking-down frames
    end the episode, like the shipped eye_closure_release_frames. An unknown
    eye state holds the evidence for at most ``coast_s`` of frame time, then
    drops it.
    """

    def __init__(self, sustain_s: float = 2.0, release_frames: int = 2, coast_s: float = 1.0,
                 max_gap_s: float = 0.5) -> None:
        self.sustain_s = float(sustain_s)
        self.release_frames = max(0, int(release_frames))
        self.coast_s = float(coast_s)
        self._gap = FrameGapGuard(max_gap_s)
        self._closed_s = 0.0
        self._open_frames = 0
        self._unknown_since: Optional[float] = None
        self._prev_closed = False
        self._fired = False
        self.assessment: Assessment = unknown("no frame yet")

    @property
    def closed_s(self) -> float:
        return self._closed_s

    @property
    def active(self) -> bool:
        return self._fired

    def _reset(self) -> None:
        self._closed_s = 0.0
        self._prev_closed = False
        self._fired = False

    def update(self, t: float, eye: Assessment) -> Optional[float]:
        """Feed this frame's eye state; returns the closed duration when drowsiness fires."""
        dt, _stale = self._gap.step(t)
        if not eye.known:
            if self._unknown_since is None:
                self._unknown_since = t
            coasted = t - self._unknown_since
            if self._closed_s > 0 and coasted <= self.coast_s:
                self.assessment = Assessment(
                    "coasting", f"eye state unknown ({eye.reason}); holding {self._closed_s:.1f}s of closure")
            else:
                self._reset()
                self.assessment = unknown(eye.reason)
            return None
        self._unknown_since = None
        if eye.state == CLOSED:
            self._open_frames = 0
            if self._prev_closed:
                self._closed_s += dt
            self._prev_closed = True
            if self._closed_s >= self.sustain_s and not self._fired:
                self._fired = True
                self.assessment = Assessment("drowsy")
                return self._closed_s
            self.assessment = Assessment("drowsy" if self._fired else "accumulating")
            return None
        self._open_frames += 1
        if self._open_frames > self.release_frames:
            self._reset()
        self.assessment = Assessment("drowsy" if self._fired else "none",
                                     "looking down" if eye.state == LOOKING_DOWN else "")
        return None
