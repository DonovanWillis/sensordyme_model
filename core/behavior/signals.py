"""The common intermediate evidence representation (M8.4 behavioural engine).

Every per-frame measurement the driver module makes -- face found, eye state,
calibrated gaze, phone boxes, hand overlap, the person box, frame quality --
is reported to the engine as ONE typed ``SignalEvidence``: what was observed,
when, by which capability and stage, how sure the producer is, how noisy the
measurement is, how long the value has held, whether it is calibrated, and
whether the capability that produced it is currently valid at all.

The rule this type exists to enforce: **unavailable evidence is never negative
evidence.** A missing value is ``value=None`` with a required
``missing_reason``; it can never be read as a False. A face that was not
detected is not "eyes open" and not "eyes closed"; a hand that was not looked
for is not "no hand"; a phone box is not "phone use". A ``Verdict`` built on
such evidence is ``ABSTAIN`` with the reason, and abstentions are reported,
never silently scored as safe.

Pure: standard library only; callers pass evidence-clock time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from core.behavior.evidence import FrameGapGuard

# --------------------------------------------------------------- signal names
FACE = "face"  # value: True (face landmarks found this frame); missing when the face model did not run
EYE_STATE = "eye_state"  # value: open | closed | looking_down ; missing: face lost, eyes occluded, no threshold
GAZE = "gaze"  # value: (yaw, pitch) calibrated degrees ; missing: no face, uncalibrated, implausible fit
PHONE = "phone"  # value: core/behavior/phone_evidence level ; missing: detector failed/stale/not started
HAND = "hand"  # value: held | not_held | unknown-hold-but-ran ; missing: hand model did not run this tick
PERSON = "person"  # value: number of person boxes ; missing: detector failed/stale/not started
FRAME = "frame"  # value: "ok" | "dark" | "blocked" ; from a 32x24 thumbnail of the frame
DRIVER = "driver"  # value: association method (core/behavior/association.py)

#: Capability kinds (core/monitoring_state.py) each signal is produced by.
SOURCES = {
    FACE: "face", EYE_STATE: "face", GAZE: "attention", PHONE: "objects", HAND: "hands",
    PERSON: "objects", FRAME: "camera", DRIVER: "objects",
}


@dataclass(frozen=True)
class SignalEvidence:
    """One signal's evidence at one frame.

    observed_at      evidence-clock time the value was measured (None: never)
    now              evidence-clock time of the frame this snapshot belongs to
    confidence       the producer's own certainty in [0, 1]; None = no graded evidence
    uncertainty      measurement noise in the signal's units (e.g. degrees); None = unknown
    continuity_s     credited seconds the CURRENT value (or missing state) has held
    missing_reason   required, and only allowed, when value is None
    calibrated       None when no calibration applies; False = needs one it lacks
    capability_valid the capability producing it is running (not STARTING/IMPAIRED/FAILED)
    """

    name: str
    value: Any
    now: float
    observed_at: Optional[float] = None
    source: str = ""
    provenance: str = ""
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    continuity_s: float = 0.0
    missing_reason: str = ""
    calibrated: Optional[bool] = None
    capability_valid: bool = True

    def __post_init__(self) -> None:
        if self.value is None and not self.missing_reason:
            raise ValueError(f"missing evidence for {self.name!r} must say why")
        if self.value is not None and self.missing_reason:
            raise ValueError(f"{self.name!r} has a value and a missing_reason")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"{self.name!r} confidence {self.confidence} outside [0, 1]")

    @property
    def missing(self) -> bool:
        return self.value is None

    @property
    def freshness_s(self) -> float:
        """Seconds since the value was measured (inf when never)."""
        return math.inf if self.observed_at is None else max(0.0, self.now - self.observed_at)

    def stale(self, max_age_s: float) -> bool:
        return self.freshness_s > max_age_s

    def usable(self, max_age_s: float = math.inf) -> bool:
        """Present, fresh, from a valid capability, and calibrated where that matters."""
        return (not self.missing and self.capability_valid and not self.stale(max_age_s)
                and self.calibrated is not False)

    def why_unusable(self, max_age_s: float = math.inf) -> str:
        if self.missing:
            return self.missing_reason
        if not self.capability_valid:
            return f"{self.source or self.name} capability not running"
        if self.stale(max_age_s):
            return f"{self.name} stale ({self.freshness_s:.1f}s old)"
        if self.calibrated is False:
            return f"{self.name} not calibrated"
        return ""

    def to_dict(self) -> dict:
        value = self.value
        if isinstance(value, tuple):
            value = [round(v, 1) if isinstance(v, float) else v for v in value]
        elif isinstance(value, float):
            value = round(value, 3)
        out = {
            "value": value, "source": self.source, "provenance": self.provenance,
            "observed_at": None if self.observed_at is None else round(self.observed_at, 3),
            "freshness_s": None if self.observed_at is None else round(self.freshness_s, 3),
            "continuity_s": round(self.continuity_s, 2), "missing": self.missing,
            "capability_valid": self.capability_valid,
        }
        if self.missing_reason:
            out["missing_reason"] = self.missing_reason
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 3)
        if self.uncertainty is not None:
            out["uncertainty"] = round(self.uncertainty, 3)
        if self.calibrated is not None:
            out["calibrated"] = self.calibrated
        return out


def observed(name: str, value: Any, now: float, **kwargs) -> SignalEvidence:
    """Evidence measured on this frame."""
    kwargs.setdefault("source", SOURCES.get(name, ""))
    return SignalEvidence(name=name, value=value, now=now, observed_at=now, **kwargs)


def missing(name: str, reason: str, now: float, observed_at: Optional[float] = None, **kwargs) -> SignalEvidence:
    """No usable value this frame, and why. ``observed_at`` is the last time one existed."""
    kwargs.setdefault("source", SOURCES.get(name, ""))
    return SignalEvidence(name=name, value=None, now=now, observed_at=observed_at,
                          missing_reason=reason, **kwargs)


# --------------------------------------------------------------- verdicts
SUPPORTED = "supported"
NOT_SUPPORTED = "not_supported"
ABSTAIN = "abstain"


@dataclass(frozen=True)
class Verdict:
    """The engine's answer about one claim on one frame.

    ``ABSTAIN`` is an answer: the evidence cannot support either way, and
    ``reason`` says which evidence was missing. It never alerts and is never
    counted as "safe".
    """

    claim: str
    state: str
    reason: str = ""
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in (SUPPORTED, NOT_SUPPORTED, ABSTAIN):
            raise ValueError(f"unknown verdict state {self.state!r}")
        if self.state == ABSTAIN and not self.reason:
            raise ValueError("an abstention must say why")

    def to_dict(self) -> dict:
        return {"state": self.state, "reason": self.reason, "evidence": list(self.evidence)}


class Continuity:
    """Credited seconds a signal's value has held unchanged.

    A change of value (including to or from missing) restarts it; a stalled
    stretch (FrameGapGuard) credits nothing but does not reset -- unobserved
    time is neither evidence for nor against continuity.
    """

    def __init__(self, max_gap_s: float = 0.5) -> None:
        self._gap = FrameGapGuard(max_gap_s)
        self._key: Any = object()
        self.seconds = 0.0

    def update(self, t: float, key: Any) -> float:
        dt, _stale = self._gap.step(t)
        if key != self._key:
            self._key = key
            self.seconds = 0.0
        else:
            self.seconds += dt
        return self.seconds


__all__ = [
    "ABSTAIN",
    "DRIVER",
    "EYE_STATE",
    "FACE",
    "FRAME",
    "GAZE",
    "HAND",
    "NOT_SUPPORTED",
    "PERSON",
    "PHONE",
    "SUPPORTED",
    "Continuity",
    "SignalEvidence",
    "Verdict",
    "missing",
    "observed",
]
