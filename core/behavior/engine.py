"""BehaviorEngine: evidence -> verdicts with abstention, in SHADOW of the shipped alerts (M8.4).

One call per video frame. The driver module hands the engine a ``FrameEvidence``
(core/behavior/signals.py: every signal typed, with freshness, continuity,
confidence, calibration and capability validity, and an explicit missing
reason) and the engine returns a ``Verdict`` per claim -- SUPPORTED,
NOT_SUPPORTED or ABSTAIN-with-reason -- plus the claims whose SUPPORTED
episode began on this frame.

It does NOT alert. Every claim that maps onto a validated alert (drowsiness,
distraction, eyes_down, phone_use, attention_lost) is computed in SHADOW: the
module keeps firing its shipped trackers exactly as before, the engine's
decisions are logged/reported beside them, and ``record_production`` counts
agreement, production-only and shadow-only episodes and the timing difference
of matched ones. Promoting any shadow decision to an alert is a separate,
evidence-backed change (docs/BEHAVIOR_ENGINE.md).

What the shadow does differently from production, and why (all on missing or
degraded evidence -- on clean evidence the thresholds and timings are the
shipped ones, from the same config keys):

  * unobserved time is never credited (FrameGapGuard); production's
    StallAwareEvidenceClock credits one frame interval per stall;
  * a missing gaze or eye state is ABSTAIN, never "on road"/"open": progress
    is HELD (not grown) for at most ``coast_s``, then dropped. Production
    coasts the last gaze reading while a body is detected and resets eye
    closure on any face loss;
  * occluded eyes (sunglasses) are ABSTAIN for drowsiness; production reads
    whatever EAR the landmarks give;
  * a gaze held off-road, on one side, with almost no variation, for
    ``misalignment_s`` is more likely a moved camera than a driver: ABSTAIN
    ("recalibrate"), where production keeps alerting;
  * no face and no person for ``empty_after_s`` is "vehicle may be empty":
    ABSTAIN, where production's attention_lost keeps repeating; a dark/flat
    frame is "view blocked", reported as the cause rather than "driver absent";
  * phone_use needs the phone HELD (core/behavior/phone_evidence.py); a phone
    near the driver with holding unknown is ABSTAIN, not use; mounted /
    visible is NOT_SUPPORTED.

Pure: standard library + core.behavior only.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.behavior.association import DriverAssociation
from core.behavior.evidence import Assessment, FrameGapGuard, unknown
from core.behavior.eye_state import LOOKING_DOWN, OPEN, DrowsinessEvidence
from core.behavior.phone_evidence import (
    ACTIVE,
    HELD,
    INSUFFICIENT,
    INTERACTION,
    NEAR_DRIVER,
    PhoneEvidence,
    PhoneEvidenceInput,
    classify_phone,
)
from core.behavior.signals import (
    ABSTAIN,
    EYE_STATE,
    FACE,
    FRAME,
    GAZE,
    NOT_SUPPORTED,
    PERSON,
    SUPPORTED,
    SignalEvidence,
    Verdict,
)

DROWSINESS = "drowsiness"
DISTRACTION = "distraction"
EYES_DOWN = "eyes_down"
PHONE_USE = "phone_use"
ATTENTION_LOST = "attention_lost"
CLAIMS: tuple[str, ...] = (DROWSINESS, DISTRACTION, EYES_DOWN, PHONE_USE, ATTENTION_LOST)

#: frame-quality values (signals.FRAME)
FRAME_OK, FRAME_DARK, FRAME_BLOCKED = "ok", "dark", "blocked"


@dataclass(frozen=True)
class EngineConfig:
    """Thresholds: the shipped config keys, so the shadow judges by the same bars."""

    drowsy_seconds: float = 2.0
    eye_release_frames: int = 2
    yaw_threshold_deg: float = 35.0
    mirror_check_yaw_deg: float = 55.0
    distraction_seconds: float = 3.0
    rearview_tolerance_seconds: float = 5.0
    pitch_threshold_deg: float = -20.0
    eyes_down_seconds: float = 2.5
    phone_use_seconds: float = 2.0
    face_lost_seconds: float = 2.5
    face_lost_with_body_seconds: float = 8.0
    release_s: float = 0.4
    max_gap_s: float = 0.5
    #: Shadow-only (documented in docs/BEHAVIOR_ENGINE.md; no production effect).
    coast_s: float = 1.0
    misalignment_s: float = 30.0
    misalignment_spread_deg: float = 3.0
    empty_after_s: float = 30.0
    match_window_s: float = 3.0

    @classmethod
    def from_module_config(cls, config: dict) -> "EngineConfig":
        def num(key: str, default: float) -> float:
            value = config.get(key, default)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

        return cls(
            drowsy_seconds=num("drowsy_seconds", 2.0),
            eye_release_frames=int(num("eye_closure_release_frames", 2)),
            yaw_threshold_deg=num("yaw_threshold_deg", 35.0),
            mirror_check_yaw_deg=num("mirror_check_yaw_deg", 55.0),
            distraction_seconds=num("distraction_seconds", 3.0),
            rearview_tolerance_seconds=num("rearview_tolerance_seconds", 5.0),
            pitch_threshold_deg=num("pitch_threshold_deg", -20.0),
            eyes_down_seconds=num("eyes_down_seconds", 2.5),
            phone_use_seconds=num("phone_use_seconds", 2.0),
            face_lost_seconds=num("face_lost_seconds", 2.5),
            face_lost_with_body_seconds=num("face_lost_with_body_seconds", 8.0),
            release_s=num("signal_release_s", 0.4),
            max_gap_s=num("max_evidence_gap_s", 0.5),
        )


@dataclass(frozen=True)
class FrameEvidence:
    """Everything the engine needs for one frame. Signals are keyed by signals.* names."""

    t: float  # frame capture time (NOT clamped: the engine must see stalls)
    signals: dict[str, SignalEvidence]
    phone: PhoneEvidenceInput
    driver: Optional[DriverAssociation] = None
    eyes_occluded: bool = False


class _Accumulator:
    """Credited seconds a condition has held, with release grace and bounded coast.

    ``update(dt, cond)`` with cond True/False on an observed frame; ``hold(t)``
    on a frame where the condition could not be observed: nothing is credited
    and nothing is reset, until ``coast_s`` of frame time has passed, after
    which the progress is dropped (the caller then reports ABSTAIN).
    """

    def __init__(self, release_s: float, coast_s: float) -> None:
        self.release_s, self.coast_s = release_s, coast_s
        self.seconds = 0.0
        self.active = False
        self.started = False
        self.peak = 0.0  # largest |magnitude| seen in the current episode
        self._false_since: Optional[float] = None
        self._missing_since: Optional[float] = None

    def reset(self) -> None:
        self.seconds, self.active, self.started, self.peak = 0.0, False, False, 0.0
        self._false_since = None

    def update(self, t: float, dt: float, cond: bool, magnitude: float = 0.0) -> None:
        """Like the shipped GracedConditionTracker: an episode's time runs from its
        first True frame (that frame itself credits nothing), through false frames
        bridged by the release grace; a stall (dt == 0 from FrameGapGuard) credits nothing."""
        self._missing_since = None
        if cond:
            self._false_since = None
            if self.started:
                self.seconds += dt
            self.started = True
            self.peak = max(self.peak, abs(magnitude))
            return
        if not self.started:
            return
        if self._false_since is None:
            self._false_since = t
        if t - self._false_since > self.release_s:
            self.reset()
        else:
            self.seconds += dt

    def hold(self, t: float) -> bool:
        """True while coasting; False once the coast is exhausted (progress dropped)."""
        if self._missing_since is None:
            self._missing_since = t
        if t - self._missing_since <= self.coast_s and self.started:
            return True
        self.reset()
        return False


@dataclass
class _ClaimStats:
    production: list[float] = field(default_factory=list)
    shadow: list[float] = field(default_factory=list)
    frames: int = 0
    abstain_frames: int = 0
    abstain_reasons: dict[str, int] = field(default_factory=dict)


class BehaviorEngine:
    """Shadow verdicts + abstention reporting over typed evidence."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = c = config or EngineConfig()
        self._gap = FrameGapGuard(c.max_gap_s)
        self._drowsy = DrowsinessEvidence(sustain_s=c.drowsy_seconds, release_frames=c.eye_release_frames,
                                          coast_s=c.coast_s, max_gap_s=c.max_gap_s)
        self._yaw = _Accumulator(c.release_s, c.coast_s)
        self._pitch = _Accumulator(c.release_s, c.coast_s)
        self._phone_off_road = _Accumulator(c.release_s, c.coast_s)
        self._phone_active = False
        self._face_missing = _Accumulator(c.release_s, c.coast_s)
        self._nobody_s = 0.0
        self._attention_lost_active = False
        self._off_road_poses: deque[tuple[float, float]] = deque()
        self._off_road_since: Optional[float] = None
        self.misalignment_suspected = False
        self.verdicts: dict[str, Verdict] = {}
        self.phone_evidence = PhoneEvidence(INSUFFICIENT, "no frame yet")
        self.driver: Optional[DriverAssociation] = None
        self.started: list[str] = []
        self.stalled_frames = 0
        self.frames = 0
        self.missing_frames: dict[str, int] = {}
        self._stats: dict[str, _ClaimStats] = {claim: _ClaimStats() for claim in CLAIMS}
        self._last_t: Optional[float] = None

    # ------------------------------------------------------------------ per frame
    def update(self, frame: FrameEvidence) -> dict[str, Verdict]:
        t = frame.t
        dt, stalled = self._gap.step(t)
        self.frames += 1
        self.stalled_frames += int(stalled)
        self._last_t = t
        for name, ev in frame.signals.items():
            if ev.missing:
                self.missing_frames[name] = self.missing_frames.get(name, 0) + 1
        self.driver = frame.driver
        before = {k: v.state for k, v in self.verdicts.items()}
        verdicts = {
            DROWSINESS: self._drowsiness(frame),
            **self._gaze(frame, t, dt),
        }
        verdicts[PHONE_USE] = self._phone(frame, t, dt, verdicts)
        verdicts[ATTENTION_LOST] = self._attention_lost(frame, dt)
        self.verdicts = verdicts
        self.started = [claim for claim, v in verdicts.items()
                        if v.state == SUPPORTED and before.get(claim) != SUPPORTED]
        for claim, v in verdicts.items():
            stats = self._stats[claim]
            stats.frames += 1
            if v.state == ABSTAIN:
                stats.abstain_frames += 1
                key = v.reason.split(":")[0][:60]
                stats.abstain_reasons[key] = stats.abstain_reasons.get(key, 0) + 1
        for claim in self.started:
            self._stats[claim].shadow.append(t)
        return verdicts

    # ------------------------------------------------------------------ claims
    def _eye_assessment(self, frame: FrameEvidence) -> Assessment:
        eye = frame.signals.get(EYE_STATE)
        if eye is None:
            return unknown("no eye evidence")
        if not eye.usable():
            return unknown(eye.why_unusable())
        return Assessment(str(eye.value))

    def _drowsiness(self, frame: FrameEvidence) -> Verdict:
        assessment = self._eye_assessment(frame)
        self._drowsy.update(frame.t, assessment)
        state = self._drowsy.assessment
        if self._drowsy.active:
            return Verdict(DROWSINESS, SUPPORTED, f"eyes closed {self._drowsy.closed_s:.1f}s", (EYE_STATE,))
        if state.state == "coasting":
            return Verdict(DROWSINESS, ABSTAIN, f"eye state unknown, holding closure: {state.reason}", (EYE_STATE,))
        if not assessment.known:
            return Verdict(DROWSINESS, ABSTAIN, f"eye state unknown: {assessment.reason}", (EYE_STATE,))
        why = "looking down" if assessment.state == LOOKING_DOWN else (
            "eyes open" if assessment.state == OPEN else f"closed {self._drowsy.closed_s:.1f}s")
        return Verdict(DROWSINESS, NOT_SUPPORTED, why, (EYE_STATE,))

    def _gaze(self, frame: FrameEvidence, t: float, dt: float) -> dict[str, Verdict]:
        c = self.config
        gaze = frame.signals.get(GAZE)
        if gaze is None or not gaze.usable():
            reason = "no gaze evidence" if gaze is None else gaze.why_unusable()
            out = {}
            for claim, acc in ((DISTRACTION, self._yaw), (EYES_DOWN, self._pitch)):
                if acc.active and acc.hold(t):
                    out[claim] = Verdict(claim, SUPPORTED, f"holding through missing gaze: {reason}", (GAZE,))
                elif acc.hold(t):
                    out[claim] = Verdict(claim, ABSTAIN, f"gaze unknown, holding progress: {reason}", (GAZE,))
                else:
                    out[claim] = Verdict(claim, ABSTAIN, f"gaze unknown: {reason}", (GAZE,))
            self._off_road_since = None
            self._off_road_poses.clear()
            return out
        yaw, pitch = gaze.value
        yaw_off = abs(yaw) > c.yaw_threshold_deg
        pitch_off = pitch < c.pitch_threshold_deg
        self._yaw.update(t, dt, yaw_off, yaw)
        self._pitch.update(t, dt, pitch_off, pitch)
        misaligned = self._misalignment(t, yaw, pitch, yaw_off or pitch_off)

        out: dict[str, Verdict] = {}
        need = c.rearview_tolerance_seconds if self._yaw.peak > c.mirror_check_yaw_deg else c.distraction_seconds
        for claim, acc, bar, off in ((DISTRACTION, self._yaw, need, yaw_off),
                                     (EYES_DOWN, self._pitch, c.eyes_down_seconds, pitch_off)):
            if misaligned and off:
                out[claim] = Verdict(claim, ABSTAIN, "camera may have moved: constant off-road pose for "
                                     f"{c.misalignment_s:.0f}s+ (recalibrate)", (GAZE,))
                continue
            if acc.seconds >= bar:
                acc.active = True
            if acc.active:
                out[claim] = Verdict(claim, SUPPORTED, f"off road {acc.seconds:.1f}s (bar {bar:.1f}s)", (GAZE,))
            elif acc.seconds > 0:
                kind = "glance" if acc.seconds < bar else "sustained"
                out[claim] = Verdict(claim, NOT_SUPPORTED, f"{kind} {acc.seconds:.1f}s < {bar:.1f}s", (GAZE,))
            else:
                out[claim] = Verdict(claim, NOT_SUPPORTED, "on road", (GAZE,))
        return out

    def _misalignment(self, t: float, yaw: float, pitch: float, off: bool) -> bool:
        """A long, nearly motionless off-road pose: more likely a moved camera than a driver."""
        c = self.config
        if not off:
            self._off_road_since = None
            self._off_road_poses.clear()
            self.misalignment_suspected = False
            return False
        if self._off_road_since is None:
            self._off_road_since = t
        self._off_road_poses.append((yaw, pitch))
        while len(self._off_road_poses) > 600:
            self._off_road_poses.popleft()
        if t - self._off_road_since < c.misalignment_s or len(self._off_road_poses) < 10:
            return self.misalignment_suspected
        yaws = [p[0] for p in self._off_road_poses]
        pitches = [p[1] for p in self._off_road_poses]
        spread = max(statistics.pstdev(yaws), statistics.pstdev(pitches))
        self.misalignment_suspected = spread <= c.misalignment_spread_deg
        return self.misalignment_suspected

    def _phone(self, frame: FrameEvidence, t: float, dt: float, gaze_verdicts: dict[str, Verdict]) -> Verdict:
        c = self.config
        inp = frame.phone
        gaze = frame.signals.get(GAZE)
        held = inp.detector_ok and inp.any_phone and inp.driver_known and inp.driver_phone and inp.hold == "held"
        if held and inp.eyes_off_road:
            self._phone_off_road.update(t, dt, True)
        elif held and inp.eyes_off_road is False:
            self._phone_off_road.update(t, dt, False)
        elif held:
            self._phone_off_road.hold(t)
        else:
            self._phone_off_road.update(t, dt, False)
        evidence = classify_phone(PhoneEvidenceInput(
            detector_ok=inp.detector_ok, detector_reason=inp.detector_reason, any_phone=inp.any_phone,
            driver_known=inp.driver_known, driver_phone=inp.driver_phone, hold=inp.hold,
            eyes_off_road=inp.eyes_off_road, gaze_toward_phone=inp.gaze_toward_phone,
            held_off_road_s=self._phone_off_road.seconds, sustain_s=c.phone_use_seconds,
        ))
        self.phone_evidence = evidence
        if evidence.level == ACTIVE:
            self._phone_active = True
        elif not (evidence.level in (HELD, INTERACTION) and self._phone_active):
            self._phone_active = False
        if self._phone_active:
            return Verdict(PHONE_USE, SUPPORTED, evidence.reason, ("phone", GAZE))
        if evidence.level == INSUFFICIENT:
            return Verdict(PHONE_USE, ABSTAIN, f"phone evidence insufficient: {evidence.reason}", ("phone",))
        off_road = inp.eyes_off_road
        if evidence.level == NEAR_DRIVER and off_road:
            return Verdict(PHONE_USE, ABSTAIN, "phone near driver, eyes off road, holding unknown", ("phone", GAZE))
        if evidence.level in (HELD, INTERACTION) and off_road is None:
            why = gaze.why_unusable() if gaze is not None else "no gaze evidence"
            return Verdict(PHONE_USE, ABSTAIN, f"phone held, gaze unknown: {why}", ("phone", GAZE))
        return Verdict(PHONE_USE, NOT_SUPPORTED, f"{evidence.level}" + (f" ({evidence.reason})"
                                                                         if evidence.reason else ""), ("phone",))

    def _attention_lost(self, frame: FrameEvidence, dt: float) -> Verdict:
        c = self.config
        face = frame.signals.get(FACE)
        person = frame.signals.get(PERSON)
        quality = frame.signals.get(FRAME)
        face_seen = face is not None and face.value is True
        person_known = person is not None and person.usable(max_age_s=1.0)
        person_seen = person_known and person is not None and bool(person.value)
        if face_seen or person_seen:
            self._nobody_s = 0.0
        else:
            self._nobody_s += dt
        self._face_missing.update(frame.t, dt, not face_seen)
        if not self._face_missing.started:
            self._attention_lost_active = False
            return Verdict(ATTENTION_LOST, NOT_SUPPORTED, "face visible", (FACE,))
        missing_s = self._face_missing.seconds
        bar = c.face_lost_with_body_seconds if person_seen else c.face_lost_seconds
        blocked = quality is not None and quality.value == FRAME_BLOCKED
        if blocked:
            cause = "camera view blocked or dark: cannot tell an absent driver from an obstructed lens"
        elif person_seen:
            cause = "face not visible, person still detected (turned away?)"
        elif not person_known:
            cause = f"face not visible; person evidence unavailable ({person.why_unusable() if person else 'none'})"
        else:
            cause = "no face and no person"
        if not blocked and not person_seen and person_known and self._nobody_s >= c.empty_after_s:
            self._attention_lost_active = False
            return Verdict(ATTENTION_LOST, ABSTAIN, f"vehicle may be empty: nobody seen for "
                           f"{self._nobody_s:.0f}s", (FACE, PERSON))
        if missing_s >= bar:
            self._attention_lost_active = True
        if self._attention_lost_active:
            return Verdict(ATTENTION_LOST, SUPPORTED, cause, (FACE, PERSON, FRAME))
        return Verdict(ATTENTION_LOST, NOT_SUPPORTED, f"face missing {missing_s:.1f}s < {bar:.1f}s",
                       (FACE, PERSON))

    # ------------------------------------------------------------------ shadow comparison
    def reset_stats(self) -> None:
        """Forget the comparison/abstention/missing counters (not the temporal state),
        e.g. at the end of a scenario's calibration lead-in."""
        self._stats = {claim: _ClaimStats() for claim in CLAIMS}
        self.missing_frames = {}
        self.frames = 0
        self.stalled_frames = 0

    def record_production(self, event_types: list[str], t: Optional[float] = None) -> None:
        """The events production fired on this frame (call once per frame, after update)."""
        when = self._last_t if t is None else t
        if when is None:
            return
        for event_type in event_types:
            if event_type in self._stats:
                self._stats[event_type].production.append(when)

    def comparison(self) -> dict:
        """Per claim: production vs shadow episode onsets, matched within match_window_s."""
        out = {}
        window = self.config.match_window_s
        for claim, stats in self._stats.items():
            shadow = list(stats.shadow)
            matched, delays = 0, []
            for p in stats.production:
                best = min(shadow, key=lambda s, p=p: abs(s - p), default=None)
                if best is not None and abs(best - p) <= window:
                    matched += 1
                    delays.append(best - p)
                    shadow.remove(best)
            out[claim] = {
                "production": len(stats.production), "shadow": len(stats.shadow), "matched": matched,
                "production_only": len(stats.production) - matched, "shadow_only": len(shadow),
                "mean_shadow_minus_production_s": round(statistics.fmean(delays), 3) if delays else None,
                "abstain_rate": round(stats.abstain_frames / stats.frames, 4) if stats.frames else None,
                "abstain_reasons": dict(sorted(stats.abstain_reasons.items(), key=lambda kv: -kv[1])[:5]),
            }
        return out

    # ------------------------------------------------------------------ reporting
    def abstentions(self) -> dict[str, str]:
        """Every claim the engine cannot currently decide, with why."""
        return {claim: v.reason for claim, v in self.verdicts.items() if v.state == ABSTAIN}

    def snapshot(self) -> dict:
        """Flat per-frame fields for telemetry (prefixed, CSV-friendly)."""
        out = {f"engine_{claim}": v.state for claim, v in self.verdicts.items()}
        out["engine_phone_evidence"] = self.phone_evidence.level
        out["engine_driver_identity"] = self.driver.method if self.driver is not None else "unknown"
        out["engine_abstaining"] = ";".join(sorted(self.abstentions()))
        out["engine_misalignment_suspected"] = self.misalignment_suspected
        return out

    def report(self) -> dict:
        return {
            "mode": "shadow",
            "verdicts": {claim: v.to_dict() for claim, v in self.verdicts.items()},
            "abstaining": self.abstentions(),
            "phone_evidence": self.phone_evidence.to_dict(),
            "driver": self.driver.to_dict() if self.driver is not None else None,
            "frames": self.frames, "stalled_frames": self.stalled_frames,
            "missing_frames": dict(self.missing_frames),
            "comparison": self.comparison(),
        }


def frame_quality(mean: float, std: float) -> str:
    """Thumbnail statistics -> ok / dark / blocked. ``blocked``: nearly no light
    AND no texture (a lens cap, a hand over the lens, a blacked-out camera);
    ``dark``: little light (night without IR) -- lower confidence, not blocked."""
    if not (math.isfinite(mean) and math.isfinite(std)):
        return FRAME_BLOCKED
    if mean < 12.0 and std < 6.0:
        return FRAME_BLOCKED
    if mean < 40.0:
        return FRAME_DARK
    return FRAME_OK


__all__ = [
    "ATTENTION_LOST",
    "CLAIMS",
    "DISTRACTION",
    "DROWSINESS",
    "EYES_DOWN",
    "FRAME_BLOCKED",
    "FRAME_DARK",
    "FRAME_OK",
    "PHONE_USE",
    "BehaviorEngine",
    "EngineConfig",
    "FrameEvidence",
    "frame_quality",
]
