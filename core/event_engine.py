"""Event lifecycle engine: turns per-frame observations into staged, debounced events.

Modules feed raw per-frame booleans ("person present in zone X", "face not
visible", "still for a while") into the engine; the engine owns the temporal
reasoning and hands back *state transitions*:

    observed   -- condition seen for the first time (log-only signal)
    developing -- condition persisting, at least half-way to confirmation
    confirmed  -- condition met its persistence + frame-count + confidence
                  requirements; this is the alert-worthy moment
    resolved   -- a previously-confirmed condition has been absent long enough

Design rules encoded here (rather than re-implemented per module):
  * a single uncertain frame never confirms anything (min_frames, persist_s);
  * confidence is smoothed (EMA), so one high-confidence flicker can't spike
    a track over its threshold;
  * brief occlusion doesn't reset progress or resolve an active event
    (occlusion_grace_s): a person walking behind a chair is not "gone";
  * a confirmed event stays active until genuinely absent (resolve_after_s),
    so the app can show ongoing-vs-ended, and repeated notifications are the
    AlertManager's cooldown decision, not a re-detection artifact.

This complements (not replaces) the specialized trackers modules already use
(PERCLOS windows, RiskFusion, SustainedConditionTracker); it's the shared
lifecycle for zone/presence/stillness-style detections and for AI-configured
custom rules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

OBSERVED = "observed"
DEVELOPING = "developing"
CONFIRMED = "confirmed"
RESOLVED = "resolved"

_EMA_ALPHA = 0.3  # smoothing for per-frame confidence


@dataclass(frozen=True)
class RuleSpec:
    """Confirmation requirements for one detection id."""

    detection: str                 # e.g. "zone_violation:pool", "person_absent"
    severity: str = "warning"      # info | warning | critical (alert routing hint)
    min_confidence: float = 0.5    # smoothed confidence needed to confirm
    persist_s: float = 2.0         # continuous seconds before confirmation
    min_frames: int = 3            # minimum sightings before confirmation
    occlusion_grace_s: float = 1.0  # absence shorter than this doesn't reset/resolve
    resolve_after_s: float = 5.0   # absence needed to resolve a confirmed event
    cooldown_s: float = 30.0       # re-confirmation suppression after resolve


@dataclass
class Transition:
    """One state change for one detection, as the module should report it."""

    detection: str
    state: str                     # observed | developing | confirmed | resolved
    confidence: float              # smoothed confidence at transition time
    duration_s: float              # how long the condition has been held
    severity: str


@dataclass
class _Track:
    first_seen: float
    last_seen: float
    frames: int = 1
    confidence: float = 0.0
    state: str = OBSERVED
    confirmed_at: Optional[float] = None
    resolved_at: Optional[float] = None
    emitted: set = field(default_factory=set)


class EventEngine:
    """Shared observation→event state machine for any number of detections."""

    def __init__(self, rules: dict[str, RuleSpec]) -> None:
        self._rules = dict(rules)
        self._tracks: dict[str, _Track] = {}
        self._last_resolved: dict[str, float] = {}  # detection -> resolve time (cooldown)

    def add_rule(self, rule: RuleSpec) -> None:
        self._rules[rule.detection] = rule

    def update(self, detection: str, present: bool, confidence: float = 1.0,
               now: Optional[float] = None) -> list[Transition]:
        """Advance one detection with this frame's observation; return transitions."""
        now = time.monotonic() if now is None else now
        rule = self._rules.get(detection)
        if rule is None:
            return []
        track = self._tracks.get(detection)
        out: list[Transition] = []

        if present:
            if track is None:
                # Cooldown: a just-resolved event shouldn't instantly restart the
                # observe->confirm cycle on detector jitter.
                last = self._last_resolved.get(detection)
                if last is not None and now - last < rule.cooldown_s:
                    return []
                track = _Track(first_seen=now, last_seen=now, confidence=confidence)
                self._tracks[detection] = track
                track.emitted.add(OBSERVED)
                out.append(self._transition(track, rule, OBSERVED, now))
                return out

            # Gap longer than the occlusion grace, before confirmation: start over.
            if track.state in (OBSERVED, DEVELOPING) and now - track.last_seen > rule.occlusion_grace_s:
                track.first_seen = now
                track.frames = 0
                track.confidence = 0.0
                track.state = OBSERVED

            track.last_seen = now
            track.frames += 1
            track.confidence += _EMA_ALPHA * (confidence - track.confidence)
            held = now - track.first_seen

            if track.state in (OBSERVED, DEVELOPING):
                confirmable = (
                    held >= rule.persist_s
                    and track.frames >= rule.min_frames
                    and track.confidence >= rule.min_confidence
                )
                if confirmable:
                    track.state = CONFIRMED
                    track.confirmed_at = now
                    out.append(self._transition(track, rule, CONFIRMED, now))
                elif track.state == OBSERVED and held >= rule.persist_s / 2 and DEVELOPING not in track.emitted:
                    track.state = DEVELOPING
                    track.emitted.add(DEVELOPING)
                    out.append(self._transition(track, rule, DEVELOPING, now))
            return out

        # --- absent this frame -------------------------------------------------
        if track is None:
            return out
        gap = now - track.last_seen
        if track.state == CONFIRMED:
            if gap >= rule.resolve_after_s:
                track.state = RESOLVED
                track.resolved_at = now
                out.append(self._transition(track, rule, RESOLVED, now))
                self._last_resolved[detection] = now
                del self._tracks[detection]
        elif gap > rule.occlusion_grace_s:
            # Never confirmed and genuinely gone: silently drop the track.
            del self._tracks[detection]
        return out

    # ------------------------------------------------------------------ introspection
    def state_of(self, detection: str) -> Optional[str]:
        """Current lifecycle state, or None if not being tracked."""
        track = self._tracks.get(detection)
        return track.state if track else None

    def is_active(self, detection: str) -> bool:
        """True while the detection is CONFIRMED and not yet resolved."""
        return self.state_of(detection) == CONFIRMED

    def active_detections(self) -> list[str]:
        return [d for d, t in self._tracks.items() if t.state == CONFIRMED]

    def _transition(self, track: _Track, rule: RuleSpec, state: str, now: float) -> Transition:
        return Transition(
            detection=rule.detection,
            state=state,
            confidence=round(track.confidence, 3),
            duration_s=round(now - track.first_seen, 3),
            severity=rule.severity,
        )
