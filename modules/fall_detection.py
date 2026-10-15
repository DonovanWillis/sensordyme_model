"""Fall detection module: MediaPipe Pose hip/shoulder tracking for rapid-drop-then-stillness.

The fall detector is a small state machine, not a single-frame threshold:

    upright (seen recently) --[rapid drop OR pose lost right after upright,
    reacquired lying down]--> pending --[sustained, corroborated by posture,
    stillness]--> CONFIRMED (via core/event_engine.py, so resolve/cooldown
    behave exactly like every other detection in this product)

Two failure modes drove this design, both found by reading the previous
single-signal version rather than by guessing at new thresholds:

  1. MISSED falls: the previous version measured velocity from the raw
     landmark average with NO visibility gating, and reset all fall state
     the instant `pose_landmarks` came back None. A real fall is exactly
     the motion most likely to make MediaPipe lose lock for a frame or two
     (motion blur, extreme non-standing pose) -- so a fall whose violent
     phase coincided with a tracking dropout would recover into a
     `_prev_y = None` state, produce a velocity of 0.0 on the first
     recovered frame, and never cross the drop threshold at all. The
     stillness-after-a-fall check only runs once `_fall_pending_since` is
     set, so a fall that never triggers that flag is invisible regardless
     of how still the person is afterward. Fixed by tracking "last time
     this person was confidently upright" and treating "upright, pose lost,
     reacquired lying down within a bounded window" as a second, independent
     way to enter the pending state.
  2. Spurious LATER false positives: `_fall_pending_since`, once set by any
     velocity spike (including an ungated single-frame landmark glitch), was
     never cleared unless a confirmed fall was followed by a sharp upward
     velocity spike. Any unrelated later stillness (standing at the counter,
     reading) could then confirm a "fall" with no relationship to the
     original spike. Fixed by requiring posture to actually read
     LYING_DOWN (not just "not moving") before confirming, by clearing a
     pending state the moment posture reasserts a confident standing
     posture, and by a hard timeout as a backstop.

Also the elderly-mode zone and stillness detector, on the same landmarks:
  * restricted/boundary zone -> "zone_violation" (same event name occupancy.py
    uses, so the existing alert-severity wiring needs no change);
  * safe zone (e.g. a "bed" zone) -> "safe_zone_exit" (same event name and
    meaning baby_monitoring.py uses for a crib -- "getting out of bed", soft
    severity by default since getting up is often benign; a real fall
    afterward is caught separately by the fall detector above);
  * "unusual_stillness" from how little the hip/shoulder centroid has moved.
All three are tested against the ankle midpoint rather than a fabricated
bounding box (see get_detections' docstring on why this module deliberately
never invents one), reusing the same landmarks the fall detector already
reads -- no extra model.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from core import posture
from core.activity import ActivityRecognizer
from core.detections import Detection, DetectionFrame
from core.event_engine import CONFIRMED, DEVELOPING, OBSERVED, RESOLVED, EventEngine, RuleSpec
from core.events import Event
from core.zones import BoundaryCrossing, load_zones
from modules.base import VisionModule, register_module

HIP_LANDMARKS = [23, 24]
SHOULDER_LANDMARKS = [11, 12]
ANKLE_LANDMARKS = [27, 28]
_MIN_ANKLE_VISIBILITY = 0.5
#: Below this, a hip/shoulder landmark is a guess by MediaPipe, not an
#: observation -- excluded from the drop-velocity average rather than
#: silently trusted (see module docstring, missed-fall cause #1).
_MIN_TORSO_LANDMARK_VISIBILITY = 0.5
#: Minimum of the 4 hip/shoulder landmarks that must be visible for a
#: frame's average to be trusted for velocity at all.
_MIN_TORSO_LANDMARKS_FOR_VELOCITY = 2
#: Confidence reported for the "person" entry this module contributes to
#: DetectionFrame. It is a PLACEHOLDER, not a measurement, and is named here so
#: that it stops reading as one. MediaPipe Pose either returns landmarks or does
#: not (gated at min_detection_confidence=0.5); it produces no per-detection
#: score comparable to the one a YOLO box carries -- yet AI-configured rules
#: compare against this number exactly as though it were one.
#:
#: The honest replacement is the mean visibility of the torso landmarks this
#: module already computes, which IS a measured quantity. Deliberately not
#: changed here: user-authored rules compare against this value, so moving it
#: silently changes when their alerts fire. Measure the visibility distribution
#: on real footage first, then move it on purpose.
_PERSON_PRESENCE_CONFIDENCE = 0.9
#: Posture confidence a STANDING reading must EXCEED before it is allowed to
#: cancel a pending fall. Deliberately above core/posture.py's 0.5 partial-
#: observation sentinel: classify() returns exactly Posture(STANDING, 0.5,
#: "upright; lower body not visible") for ANY near-vertical torso whose
#: knees AND ankles are occluded -- which is precisely what a camera sees
#: mid-fall behind a sofa or for a fall along the camera axis. Treating that
#: guess as proof the person caught themselves made the most-occluded falls
#: the least detectable ones. Only the full-body reading (0.8, "upright with
#: extended legs") clears this bar.
_STANDING_CANCEL_MIN_CONFIDENCE = 0.7


@register_module("fall_detection")
class FallDetectionModule(VisionModule):
    """Fires a fall event on a rapid vertical drop of hip/shoulder height followed by stillness."""

    name = "fall_detection"

    def __init__(self, config: dict) -> None:
        """Read thresholds from config and initialize MediaPipe Pose."""
        super().__init__(config)
        self._drop_velocity_threshold = float(config["drop_velocity_threshold"])
        self._stillness_duration_s = float(config["stillness_duration_s"])
        self._stillness_velocity_threshold = float(config["stillness_velocity_threshold"])

        self._pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self._prev_y: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._fall_pending_since: Optional[float] = None
        self._still_since: Optional[float] = None
        self._last_pose_proto = None
        self._last_posture: Optional[posture.Posture] = None
        self._activity = ActivityRecognizer()
        self._last_activity = None

        # --- fall state machine: see module docstring for why both an
        # upright-recency clock and a pose-loss clock exist independently of
        # the raw velocity/stillness numbers below.
        self._last_upright_at: Optional[float] = None
        self._pose_lost_since: Optional[float] = None
        self._fall_via_recovery = False
        self._fall_confidence = 0.0
        self._last_drop_velocity = 0.0
        # How long after losing tracking entirely a reacquired LYING_DOWN
        # posture still counts as "found where the fall put them", not an
        # unrelated later event (missed-fall cause #1's fix).
        self._pose_recovery_max_gap_s = float(config.get("fall_pose_recovery_max_gap_s", 2.5))
        # How recently the person must have been confidently upright,
        # relative to the MOMENT tracking was lost, for a reacquired
        # lying-down posture to count as a fall rather than "was already
        # down, camera glitched".
        self._upright_recency_s = float(config.get("fall_upright_recency_s", 0.5))
        # Backstop: abandon a pending fall that never resolves either way
        # (never confirms, never clearly recovers to standing) so it cannot
        # sit around and confirm against an unrelated later stillness
        # (spurious-later-alert fix, see module docstring). Measured from the
        # last frame the person was actually SEEN DOWN, not from the drop --
        # see _fall_pending_evidence_at below.
        self._fall_pending_timeout_s = float(config.get("fall_pending_timeout_s", 6.0))
        # Last time this pending fall had positive corroborating evidence
        # (an observed LYING_DOWN posture). Backdated to the pending start so
        # a drop that is never followed by a down reading still expires.
        self._fall_pending_evidence_at: Optional[float] = None
        self._fall_engine = EventEngine({})
        self._fall_engine.add_rule(RuleSpec(
            detection="fall", severity="critical",
            persist_s=self._stillness_duration_s, min_frames=3, occlusion_grace_s=1.0,
            resolve_after_s=float(config.get("fall_resolve_after_s", 4.0)),
            # No cooldown by default. EventEngine's cooldown gate returns
            # BEFORE a track is even created, so any non-zero value here
            # means a second fall inside the window never enters the
            # lifecycle at all -- and repeat falls (syncope, a failed
            # attempt to stand) are the clinically most alarming pattern
            # this product can see, not detector jitter to be debounced.
            # resolve_after_s already prevents one fall re-reporting itself:
            # re-confirming requires being seen NOT down for that long and
            # then down-and-still again for a full stillness_duration_s.
            cooldown_s=float(config.get("fall_cooldown_s", 0.0)),
        ))

        # Activity events go through the shared lifecycle, which is what turns
        # "running" into ONE alert with a start and an end rather than one per
        # frame for as long as someone keeps running.
        self._activity_engine = EventEngine({})
        self._activity_engine.add_rule(RuleSpec(
            detection="running", severity="warning",
            persist_s=float(config.get("running_confirm_s", 1.0)),
            min_frames=5, resolve_after_s=2.0, cooldown_s=15.0,
        ))
        self._activity_engine.add_rule(RuleSpec(
            detection="jumping", severity="info",
            persist_s=float(config.get("jump_confirm_s", 0.2)),
            min_frames=2, resolve_after_s=1.5, cooldown_s=5.0,
        ))
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._last_hip_y: Optional[float] = None

        # Elderly-mode zones and standalone stillness, both on the landmarks
        # already being read. Separate engine from _activity_engine so a
        # fall-adjacent bug in one can't cross-contaminate the other's
        # per-detection state. "safe" zones here are bed monitoring: a "bed"
        # zone the person is expected to stay in overnight, same safe_zone_exit
        # semantics baby_monitoring.py already uses for a crib -- "getting out
        # of bed" is "leaving a safe zone" whether the subject is 6 months or
        # 86 years old, so this reuses that meaning rather than inventing a
        # bed-specific one.
        self._zone_engine = EventEngine({})
        self._zone_confirm_s = float(config.get("zone_confirm_s", 2.0))
        # Installed by _install_zones so set_zones() can replace them in place
        # when the user draws zones in the app. restricted -> zone_violation,
        # safe -> safe_zone_exit, boundary -> a crossing watch (BoundaryCrossing,
        # reported as zone_violation "crossed": a line alerts on CROSSING, not
        # on standing inside it).
        self._zones: list = []
        self._watched_zones: list = []
        self._safe_zones: list = []
        self._boundary_watch: dict[str, BoundaryCrossing] = {}
        self._install_zones(load_zones(config.get("zones", {})))
        self._still_max_motion = float(config.get("still_max_motion", 0.01))
        stillness_grace_s, stillness_resolve_s = 3.0, 10.0
        self._stillness_rule = RuleSpec(
            detection="unusual_stillness", severity="warning",
            persist_s=float(config.get("unusual_stillness_s", 600.0)),
            min_frames=10, occlusion_grace_s=stillness_grace_s,
            resolve_after_s=stillness_resolve_s,
            # No cooldown: a second episode needs a full unusual_stillness_s anyway.
            cooldown_s=0.0,
        )
        self._zone_engine.add_rule(self._stillness_rule)
        # occlusion_grace_s bridges ABSENCE without resetting the held-time
        # clock, which is right for zone rules ("walked behind a chair") and
        # WRONG for a stillness rule: there, a False frame means THE SUBJECT
        # MOVED -- contradicting evidence, not missing evidence. Bridged, it
        # let a subject moving 1s in every 3.5s accumulate a continuous
        # ten-minute "stillness" and fire a loud/push alert on someone
        # demonstrably active.
        #
        # The engine itself can't tell the two apart (and isn't this module's
        # to change), so the distinction is made HERE, at the feed. Observed
        # motion drops an unconfirmed stillness track outright, so its clock
        # restarts from zero, and holds the feed False past resolve_after_s so
        # an already-confirmed stillness actually resolves. A pose dropout
        # (real missing evidence) sets no lockout and is bridged.
        self._stillness_motion_lockout_s = stillness_resolve_s + 0.5
        self._stillness_locked_until: Optional[float] = None
        # The last four frames' (visible-landmark set, centroid of those
        # landmarks), for _torso_motion's visibility gating and median filter.
        self._torso_history: deque = deque(maxlen=4)

    # ------------------------------------------------------------------ zones
    def _install_zones(self, zones: list) -> None:
        """Make `zones` this module's zones, replacing the previous set in place.

        Unchanged zones (same name, kind and polygon) keep their rule and
        state. A removed or redrawn zone's rule is dropped from the engine with
        every track and cooldown under it, so nothing stays latched. The fall
        and stillness state machines are untouched.
        """
        old_zones, new_zones = list(self._zones), list(zones)
        old_watched = {z for z in old_zones if z.kind == "restricted"}
        new_watched = {z for z in new_zones if z.kind == "restricted"}
        for zone in old_watched - new_watched:
            self._retire_zone_rule(f"zone_violation:{zone.name}")
        for zone in new_watched - old_watched:
            self._zone_engine.add_rule(RuleSpec(
                detection=f"zone_violation:{zone.name}", severity="critical",
                persist_s=self._zone_confirm_s, min_frames=3, resolve_after_s=5.0,
                # No cooldown on a safety zone: the default 30 s refused even
                # to start tracking a second entry until it had expired.
                cooldown_s=0.0,
            ))
        old_safe = {z for z in old_zones if z.kind == "safe"}
        new_safe = {z for z in new_zones if z.kind == "safe"}
        if old_safe != new_safe:
            self._retire_zone_rule("safe_zone_exit")
            if new_safe:
                self._zone_engine.add_rule(RuleSpec(
                    detection="safe_zone_exit", severity="warning",
                    persist_s=self._zone_confirm_s, min_frames=3, resolve_after_s=5.0,
                    cooldown_s=0.0,  # a second bed exit is a new event, not jitter
                ))
        self._boundary_watch = {
            z.name: (
                self._boundary_watch[z.name]
                if z in old_zones and z.name in self._boundary_watch
                else BoundaryCrossing(self._zone_confirm_s)
            )
            for z in new_zones if z.kind == "boundary"
        }
        self._zones = new_zones
        self._watched_zones = [z for z in new_zones if z.kind == "restricted"]
        self._safe_zones = [z for z in new_zones if z.kind == "safe"]

    def _retire_zone_rule(self, detection: str) -> None:
        """Withdraw a zone's rule. An alert confirmed under it is RESOLVED, not
        silently dropped -- reported on the next frame by process() -- so a zone
        redrawn in the app mid-violation does not leave the alert "active" in
        the app and dashboard for good."""
        transition = self._zone_engine.retire_rule(detection)
        if transition is not None:
            self.__dict__.setdefault("_retired_zone_transitions", []).append(transition)

    def _take_retired_zone_events(self) -> list[Event]:
        """The resolutions queued by _retire_zone_rule, as events, once."""
        transitions = self.__dict__.pop("_retired_zone_transitions", None) or []
        return self._zone_transition_events(transitions)

    def set_zones(self, zones: list) -> None:
        """Replace the configured zones with the app-drawn ones. See VisionModule.set_zones."""
        self._install_zones(list(zones or []))

    def process(self, frame: np.ndarray) -> list[Event]:
        """Run pose estimation on `frame`, updating drop/stillness state and returning events."""
        return self._take_retired_zone_events() + self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> list[Event]:
        """One frame's work; see process()."""
        now = time.monotonic()
        events: list[Event] = []
        h, _w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            self._last_pose_proto = None
            self._last_posture = None
            # Tolerant of a brief dropout (core/activity.py's pose_lost, up to
            # POSE_LOST_GRACE_S) rather than an immediate reset(): fast, chaotic
            # motion is both what flailing/jumping detection exists to catch AND
            # a common cause of MediaPipe losing lock for a frame or two, so
            # treating every dropout as "the subject left" would blind the
            # window exactly when genuine distress motion needs it most. This
            # is scoped to the shared activity window only -- the fall-specific
            # state reset below (hip/centroid tracking) is unrelated and unchanged.
            self._activity.pose_lost(now)
            self._last_activity = None
            # Fed (False) rather than skipped, so a CONFIRMED running/jumping
            # cannot latch for ever through a dropout. EventEngine has no
            # time-based reaper.
            events += self._activity_events(now)
            self._last_hip_y = None
            self._prev_y = None
            self._prev_t = None
            self._torso_history.clear()
            for watch in self._boundary_watch.values():
                watch.update(None, now)
            # Deliberately NOT resetting _fall_pending_since/_still_since here:
            # a fall's violent phase is exactly when pose tracking is most
            # likely to drop out (see module docstring). Recording only WHEN
            # tracking was lost -- not clearing anything -- is what lets the
            # "reacquired lying down shortly after" path below recognize a
            # fall that happened during this exact gap.
            if self._pose_lost_since is None:
                self._pose_lost_since = now
            # Gone, not merely still, and gone, not merely outside the zone --
            # both are real absence, so both get an explicit False rather than
            # being left to freeze mid-confirmation.
            # A CONFIRMED fall must be fed here too. EventEngine advances a
            # confirmed track ONLY on an explicit update -- it has no
            # time-based reaper -- so skipping this call left a confirmed
            # fall latched forever the moment tracking dropped, and while
            # latched every subsequent real fall was silently absorbed into
            # the same never-resolving track with no new alert. False is the
            # right value for the same reason it is for the zone rules: we
            # cannot see the person, and resolve_after_s (not a single
            # frame) is what decides when that becomes a resolution. This
            # deliberately does NOT touch _fall_pending_since -- an
            # unconfirmed pending fall is state-machine state, preserved
            # across the gap exactly as before so a fall that happens DURING
            # the dropout is still recognized by the recovery path below.
            fall_transitions = self._fall_engine.update("fall", False, self._fall_confidence, now)
            events += self._fall_transition_events(fall_transitions, now)
            transitions = []
            for zone in self._watched_zones:
                transitions += self._zone_engine.update(f"zone_violation:{zone.name}", False, 0.85, now)
            # Not observed is not "moved": no motion lockout is set here, so
            # a brief dropout is still bridged by occlusion_grace_s and does
            # not restart a long stillness window (see __init__).
            transitions += self._zone_engine.update("unusual_stillness", False, 0.8, now)
            # Matches baby_monitoring.py's safe_zone_exit convention exactly:
            # "not detected at all" is a DIFFERENT concern (person_absent-style,
            # not currently built here) from "detected, but outside the safe
            # zone" -- so gone-entirely does not itself count as having exited.
            if self._safe_zones:
                transitions += self._zone_engine.update("safe_zone_exit", False, 0.85, now)
            events += self._zone_transition_events(transitions)
            return events

        self._last_pose_proto = results.pose_landmarks
        landmarks = results.pose_landmarks.landmark

        # Posture from the landmarks this module was already computing, so
        # custom rules ("someone on the floor") cost no extra inference.
        frame_h, frame_w = frame.shape[:2]
        self._last_frame_size = (frame_w, frame_h)
        aspect = frame_w / max(frame_h, 1)
        self._last_posture = posture.classify(landmarks, aspect=aspect)
        self._last_hip_y = posture.hip_height(landmarks)
        if self._last_posture.label in (posture.STANDING, posture.SITTING):
            self._last_upright_at = now

        # Visibility-gated drop velocity: an average built from a landmark
        # MediaPipe is guessing at (low visibility) is not evidence, and
        # averaging it in unconditionally is missed-fall cause #1 (see module
        # docstring) -- a fast, blurry fall is exactly when this matters most.
        indices = HIP_LANDMARKS + SHOULDER_LANDMARKS
        visible_ys = [
            landmarks[i].y for i in indices
            if getattr(landmarks[i], "visibility", 1.0) >= _MIN_TORSO_LANDMARK_VISIBILITY
        ]
        avg_y = (
            sum(visible_ys) / len(visible_ys) * h
            if len(visible_ys) >= _MIN_TORSO_LANDMARKS_FOR_VELOCITY else None
        )

        # The tracking gap the recovery path below reasons about. It covers
        # BOTH no pose at all AND a pose whose torso landmarks are too
        # low-visibility to measure (avg_y is None). The gating above made the
        # second kind of gap common in exactly a fast, blurry fall. It
        # used to count as "tracked", so the recovery path never saw a gap, and
        # the velocity after it was averaged over the whole unmeasured stretch
        # and stayed under the drop threshold. Measured: standing, then 1 s of
        # low-visibility torso, then lying gave NO fall, while the identical
        # 1 s as a full pose loss gave one. The gap only ends on a frame whose
        # torso is measurable. Until then the marker stays set, and nothing
        # below reads gap_start.
        if avg_y is None:
            if self._pose_lost_since is None:
                self._pose_lost_since = now
            gap_start = None
        else:
            gap_start = self._pose_lost_since
            self._pose_lost_since = None

        velocity = 0.0
        if avg_y is not None:
            if self._prev_y is not None and self._prev_t is not None and now > self._prev_t:
                velocity = (avg_y - self._prev_y) / (now - self._prev_t)
            self._prev_y = avg_y
            self._prev_t = now

        # --- entering "pending": two independent ways in, see module docstring.
        pending_opened_this_frame = False
        if velocity > self._drop_velocity_threshold * h:
            self._fall_pending_since = now
            self._fall_pending_evidence_at = now
            self._still_since = None
            self._fall_via_recovery = False
            self._last_drop_velocity = velocity
            pending_opened_this_frame = True
        elif (
            gap_start is not None
            and self._fall_pending_since is None
            and self._last_upright_at is not None
            and gap_start - self._last_upright_at <= self._upright_recency_s
            and now - gap_start <= self._pose_recovery_max_gap_s
            and self._last_posture.label == posture.LYING_DOWN
        ):
            # Upright right up until tracking broke, reacquired lying down
            # shortly after: the fall almost certainly happened inside the
            # gap we couldn't see. Backdate to when tracking was lost so
            # duration/confidence reflect the real event, not just the
            # recovery frame.
            self._fall_pending_since = gap_start
            self._fall_pending_evidence_at = now
            self._still_since = now
            self._fall_via_recovery = True
            self._last_drop_velocity = 0.0
            pending_opened_this_frame = True

        # Seeing the person actually down is the pending state's positive
        # corroboration, and it is what the backstop timeout below is
        # measured from.
        if self._fall_pending_since is not None and self._last_posture.label == posture.LYING_DOWN:
            self._fall_pending_evidence_at = now

        # A confident return to standing means they caught themselves --
        # abandon the pending window rather than let it sit around to
        # possibly confirm against unrelated stillness much later (spurious-
        # later-alert fix, see module docstring). Only while not yet
        # CONFIRMED: once an actual fall is active, recovery is handled by
        # `fall_present` going False below and the engine's own
        # resolve_after_s, not by ripping the pending state out from under it.
        #
        # Two guards, both fixing cases where this branch erased real falls:
        #   * NEVER in the frame the pending state was opened. There is no
        #     ordering in which "they dropped at 11x the threshold" and "they
        #     are standing" can both be evidence about the same instant, and
        #     without this the cancellation ran in the SAME process() call as
        #     the drop, erasing the pending window before anything could ever
        #     confirm against it.
        #   * The posture reading must be a real full-body observation
        #     (_STANDING_CANCEL_MIN_CONFIDENCE), not core/posture.py's 0.5
        #     "lower body not visible" sentinel -- which is exactly what a
        #     fall behind furniture or along the camera axis produces.
        if (
            self._fall_pending_since is not None
            and not pending_opened_this_frame
            and self._last_posture.label == posture.STANDING
            and self._last_posture.confidence >= _STANDING_CANCEL_MIN_CONFIDENCE
            and not self._fall_engine.is_active("fall")
        ):
            self._fall_pending_since = None
            self._fall_pending_evidence_at = None
            self._still_since = None
            self._fall_via_recovery = False

        # Backstop timeout: a pending state that never resolves either way.
        # Measured from the last frame the person was SEEN DOWN, not as a
        # wall clock from the drop. Counting wall clock inverted detection
        # probability against severity: someone who fell and lay still for
        # 2s was detected, and someone who fell, struggled to get up for ten
        # seconds and then lost consciousness -- this product's actual use
        # case, and the worse outcome -- was permanently missed, because the
        # pending state was thrown away mid-struggle with no re-entry path.
        # A spike that is never corroborated by an observed lying-down
        # posture still expires on exactly the old schedule.
        if (
            self._fall_pending_since is not None
            and not self._fall_engine.is_active("fall")
            and now - (self._fall_pending_evidence_at or self._fall_pending_since)
            > self._fall_pending_timeout_s
        ):
            self._fall_pending_since = None
            self._fall_pending_evidence_at = None
            self._still_since = None
            self._fall_via_recovery = False

        # --- confirmation/resolution: must be pending AND currently still AND
        # currently observed lying down -- fed through EventEngine every
        # frame exactly like every other sustained detection in this module,
        # so persist_s/resolve_after_s/cooldown_s give this the same
        # anti-spam behavior as zone_violation/unusual_stillness above, and
        # occlusion_grace_s (not a bespoke sticky flag) is what tolerates a
        # single noisy posture frame either during the initial confirmation
        # countdown or while genuinely down -- reusing the SAME mechanism
        # the rest of this module already relies on for that, rather than a
        # second, parallel one. Deliberately CURRENT-frame, not "ever seen
        # lying down while pending": once the person is confirmed down and
        # then gets back up, THIS frame's posture must be able to say so
        # immediately, or resolution could never begin.
        #
        # Stillness gates CONFIRMATION only. Once the fall is active it stays
        # present for as long as the person is seen lying down, still or not,
        # as config.yaml documents for fall_resolve_after_s ("must be seen NOT
        # lying down for this long"). It used to require stillness here too,
        # so a fallen person who struggled on the floor for 4 s got
        # `fall_resolved` while still down. When they then went still (the
        # struggle-then-lose-consciousness case) no new alert was possible.
        fall_present = False
        if self._fall_pending_since is not None:
            is_still = abs(velocity) < self._stillness_velocity_threshold * h
            is_down = self._last_posture.label == posture.LYING_DOWN
            if is_still:
                if self._still_since is None:
                    self._still_since = now
            else:
                self._still_since = None
            fall_present = is_down if self._fall_engine.is_active("fall") else (is_still and is_down)
            self._fall_confidence = self._compute_fall_confidence(h)
        else:
            self._fall_confidence = 0.0
        fall_transitions = self._fall_engine.update("fall", fall_present, self._fall_confidence, now)
        events += self._fall_transition_events(fall_transitions, now)

        # Activity from the same landmarks — no extra inference.
        self._last_activity = self._activity.update(landmarks, now=now, aspect=aspect)
        events += self._activity_events(now)

        zone_transitions = []
        # Ankle midpoint, not a fabricated box: zones.py's own point-in-polygon
        # primitive, on the one point "is this person in this area" actually
        # means for someone who might be lying down as easily as standing.
        # Occluded or invisible ankles (legs behind furniture, under a blanket,
        # off the bottom of the frame) do not count as "in the zone" or as
        # "left the safe zone". They are fed as False (not observed), exactly
        # like the pose-lost branch, rather than skipped. Skipped, a CONFIRMED
        # alert was never fed again (EventEngine has no time-based reaper), so
        # a bed exit stayed "active" all night once the person was back in
        # bed under a blanket, and the NEXT real bed exit produced no event.
        if self._watched_zones or self._safe_zones or self._boundary_watch:
            ankle_vis = [getattr(landmarks[i], "visibility", 1.0) for i in ANKLE_LANDMARKS]
            feet: Optional[tuple[float, float]] = None
            if all(v >= _MIN_ANKLE_VISIBILITY for v in ankle_vis):
                feet = (
                    sum(landmarks[i].x for i in ANKLE_LANDMARKS) / len(ANKLE_LANDMARKS),
                    sum(landmarks[i].y for i in ANKLE_LANDMARKS) / len(ANKLE_LANDMARKS),
                )
            for zone in self._watched_zones:
                inside = feet is not None and zone.contains(*feet)
                zone_transitions += self._zone_engine.update(
                    f"zone_violation:{zone.name}", inside, 0.85, now
                )
            if self._safe_zones:
                outside = feet is not None and not any(zone.contains(*feet) for zone in self._safe_zones)
                zone_transitions += self._zone_engine.update("safe_zone_exit", outside, 0.85, now)
            for zone in self._zones:
                if zone.kind != "boundary":
                    continue
                crossing = self._boundary_watch[zone.name].update(
                    None if feet is None else zone.contains(*feet), now,
                )
                if crossing is not None:
                    events.append(Event.now(
                        "zone_violation", 0.0, 0.85, self.name, f"zone={zone.name} crossed ({crossing})",
                    ))

        # Stillness from how little the hip/shoulder centroid has moved, in
        # normalized landmark coordinates (so the threshold means the same
        # thing at any capture resolution) -- independent of the vertical-only
        # velocity above, which exists to catch a FAST drop, not a slow lack
        # of movement over minutes. None means not measurable this frame
        # (missing evidence, see _torso_motion).
        motion = self._torso_motion(landmarks)
        # Observed motion is contradicting evidence, not missing evidence. It
        # drops an unconfirmed stillness clock at once and holds the feed
        # False past resolve_after_s, so a confirmed one resolves (see
        # _stillness_motion_lockout_s in __init__).
        if motion is not None and motion >= self._still_max_motion:
            self._stillness_locked_until = now + self._stillness_motion_lockout_s
            if self._zone_engine.state_of("unusual_stillness") in (OBSERVED, DEVELOPING):
                self._zone_engine.remove_rule("unusual_stillness")
                self._zone_engine.add_rule(self._stillness_rule)
        elif self._stillness_locked_until is not None and now >= self._stillness_locked_until:
            self._stillness_locked_until = None
        still = motion is not None and motion < self._still_max_motion and self._stillness_locked_until is None
        zone_transitions += self._zone_engine.update("unusual_stillness", still, 0.8, now)
        events += self._zone_transition_events(zone_transitions)

        return events

    def _torso_motion(self, landmarks) -> Optional[float]:
        """Glitch-filtered, visibility-gated torso movement this frame, in
        normalized units. None when it cannot be measured.

        Two defects, both of which kept `unusual_stillness` (the backstop for a
        person who is down and not moving) from ever firing:
          * The centroid averaged all four torso landmarks whatever their
            visibility. MediaPipe's guess at an occluded hip, or a landmark
            moving in or out of view, read as movement. Only landmarks visible
            (>= 0.5) are used now, and only when the same set is visible in all
            four frames compared.
          * A single jittery frame of 0.01 locked the feed for 10.5 s and
            restarted the 600 s clock. Measured: Gaussian jitter of sigma 0.004
            (about 2 px) on a motionless person lying down caused 29 restarts in
            15 minutes, and the rule never fired. Motion is now the change in the
            per-axis MEDIAN of the last three centroids, so a lone one-frame
            excursion never registers, wherever it falls in the window. Real
            movement moves the median and is reported a frame later.
        """
        points = {
            i: (landmarks[i].x, landmarks[i].y)
            for i in HIP_LANDMARKS + SHOULDER_LANDMARKS
            if getattr(landmarks[i], "visibility", 1.0) >= _MIN_TORSO_LANDMARK_VISIBILITY
        }
        if len(points) < _MIN_TORSO_LANDMARKS_FOR_VELOCITY:
            self._torso_history.clear()
            return None
        key = frozenset(points)
        centroid = (
            sum(p[0] for p in points.values()) / len(points),
            sum(p[1] for p in points.values()) / len(points),
        )
        if self._torso_history and self._torso_history[-1][0] != key:
            self._torso_history.clear()  # a different landmark set: not comparable
        self._torso_history.append((key, centroid))
        if len(self._torso_history) < 4:
            return None
        cs = [c for _, c in self._torso_history]

        def _median3(i: int, axis: int) -> float:
            return sorted(c[axis] for c in cs[i:i + 3])[1]

        return math.hypot(_median3(1, 0) - _median3(0, 0), _median3(1, 1) - _median3(0, 1))

    def _zone_transition_events(self, transitions: list) -> list[Event]:
        """CONFIRMED/RESOLVED transitions -> Events, for both the zone and
        stillness engine and the person-gone early-return, so the translation
        (and the event-type naming) is written once."""
        events: list[Event] = []
        for tr in transitions:
            base, _, zone_name = tr.detection.partition(":")
            detail = f"zone={zone_name} confirmed after {tr.duration_s:.1f}s" if zone_name else ""
            if tr.state == CONFIRMED:
                events.append(Event.now(base, tr.duration_s, tr.confidence, self.name, detail))
            elif tr.state == RESOLVED:
                events.append(Event.now(
                    f"{base}_resolved", tr.duration_s, tr.confidence, self.name,
                    f"zone={zone_name}" if zone_name else "",
                ))
        return events

    def _compute_fall_confidence(self, h: float) -> float:
        """How strongly the evidence for an in-progress fall supports it.

        Deliberately lower base confidence for a pose-recovery-inferred fall
        than a directly-measured one: the recovery path never actually saw
        the drop, only "confidently upright, then untracked, then found lying
        down" -- strong circumstantial evidence, not a measurement. Posture
        confidence and how far the measured drop exceeded the configured
        threshold both nudge it, never dominate it -- this is a supporting
        detail for get_debug_state/telemetry, not what gates the alert
        itself (that's the state machine in process()).
        """
        base = 0.55 if self._fall_via_recovery else 0.65
        if not self._fall_via_recovery and self._drop_velocity_threshold > 0:
            excess = max(0.0, (self._last_drop_velocity / (self._drop_velocity_threshold * h)) - 1.0)
            base += min(0.2, excess * 0.15)
        posture_confidence = self._last_posture.confidence if self._last_posture else 0.0
        base += 0.15 * posture_confidence
        return round(min(0.99, base), 3)

    def _fall_transition_events(self, transitions: list, now: float) -> list[Event]:
        """CONFIRMED/RESOLVED -> Events for the fall engine specifically (kept
        separate from _zone_transition_events: "fall" isn't a `label:name`
        keyed detection, and RESOLVED here also clears the state-machine
        fields so a future, independent fall can be detected from scratch)."""
        events: list[Event] = []
        for tr in transitions:
            if tr.state == CONFIRMED:
                trigger = "tracking-loss recovery" if self._fall_via_recovery else "measured rapid drop"
                posture_label = self._last_posture.label if self._last_posture else "unknown"
                events.append(Event.now(
                    "fall", tr.duration_s, tr.confidence, self.name,
                    f"trigger={trigger} posture={posture_label}",
                ))
            elif tr.state == RESOLVED:
                events.append(Event.now("fall_resolved", tr.duration_s, tr.confidence, self.name))
                self._still_since = None
                if self._seen_upright():
                    # Seen back off the floor (on their feet, or sitting up in a
                    # chair): the fall is over, and a future one must be
                    # detected from scratch. Re-arming here made someone who
                    # sat up and then lay down still on a sofa within
                    # fall_pending_timeout_s a brand-new "fall" with no drop.
                    self._fall_pending_since = None
                    self._fall_pending_evidence_at = None
                    self._fall_via_recovery = False
                else:
                    # Resolved WITHOUT anyone seeing them get up: pose tracking
                    # was lost for resolve_after_s (common for someone lying on
                    # the floor), or the last posture was unreadable. Clearing
                    # the pending state here used to make the person still
                    # lying there when tracking came back invisible: no
                    # velocity spike and no "recently upright" left to re-enter
                    # by. It is re-armed instead. Seen lying down again, it
                    # re-confirms after stillness_duration_s. Seen confidently
                    # standing, it is cancelled. Neither within
                    # fall_pending_timeout_s, it expires as usual.
                    self._fall_pending_since = now
                    self._fall_pending_evidence_at = self._fall_pending_since
        return events

    def _seen_upright(self) -> bool:
        """Last seen standing or sitting — torso vertical, i.e. off the floor.

        No extra confidence bar: unlike cancelling a PENDING fall on one frame
        (_STANDING_CANCEL_MIN_CONFIDENCE), this runs once a confirmed fall has
        already resolved through resolve_after_s of not lying down, and the
        classifier only says STANDING/SITTING for an upright torso. Sitting is
        always reported at 0.6, so a 0.7 bar would never count it.
        """
        return self._last_posture is not None and self._last_posture.label in (
            posture.STANDING, posture.SITTING,
        )


    def _activity_events(self, now: float) -> list[Event]:
        """Turn the activity state into lifecycle-managed events.

        Every frame feeds the engine, but only CONFIRMED and RESOLVED
        transitions become events — which is what makes thirty seconds of
        running one alert with a start and an end, not thirty alerts.
        """
        state = self._last_activity
        events: list[Event] = []
        if state is None or not state.ready:
            # Nothing measurable this frame. It is fed as "not present" rather
            # than skipped. Skipped, a CONFIRMED running/jumping track was never
            # updated again (EventEngine has no time-based reaper), so it
            # latched for ever after a pose dropout and swallowed every later
            # episode. resolve_after_s still decides when absence becomes a
            # resolution.
            attributes: tuple = (("running", False, 0.0), ("jumping", False, 0.0))
            evidence = ""
        else:
            attributes = tuple(
                (a.name, a.present, a.confidence or 0.0) for a in (state.running, state.jumping)
            )
            evidence = ", ".join(f"{k}={v}" for k, v in state.signals.items() if k != "reason")

        for name, present, confidence in attributes:
            transitions = self._activity_engine.update(name, present, confidence, now)
            for transition in transitions:
                if transition.state == CONFIRMED:
                    events.append(Event.now(
                        transition.detection, transition.duration_s,
                        transition.confidence, self.name, evidence,
                    ))
                elif transition.state == RESOLVED:
                    events.append(Event.now(
                        f"{transition.detection}_ended", transition.duration_s,
                        transition.confidence, self.name, evidence,
                    ))
        return events

    def get_activity(self) -> Optional[dict]:
        """Every attribute believed about the subject right now, independently.

        Multi-attribute by design: a person can be running and upright and not
        jumping all at once, and collapsing that to one label throws away the
        combinations that matter.
        """
        return self._last_activity.as_dict() if self._last_activity else None

    def get_detections(self) -> DetectionFrame:
        """The tracked person plus their posture, as labels custom rules can use.

        Posture is emitted as its own label ("lying_down", "on_floor") alongside
        "person", so a rule can ask for a posture the same way it asks for an
        object — the rule engine needs no special case, and a future posture
        model can replace this one without the rule layer noticing.

        No bounding box: MediaPipe Pose gives landmarks, not a detection box, and
        inventing one from landmark extents would let zone rules silently key off
        a rectangle that means something different from every other box here.
        """
        width, height = self._last_frame_size
        detections: list[Detection] = []
        if self._last_posture is not None:
            detections.append(
                Detection(label="person", confidence=_PERSON_PRESENCE_CONFIDENCE)
            )
            if self._last_posture.label != posture.UNKNOWN:
                detections.append(
                    Detection(label=self._last_posture.label, confidence=self._last_posture.confidence)
                )
            if posture.is_on_floor(self._last_posture, self._last_hip_y):
                detections.append(Detection(label="on_floor", confidence=self._last_posture.confidence))

        # Activities are labels too, so "alert me if someone runs in the hallway"
        # is expressible with the rule engine exactly as any object rule is.
        if self._last_activity is not None and self._last_activity.ready:
            for attribute in self._last_activity.active():
                if attribute.name not in {d.label for d in detections}:
                    detections.append(Detection(label=attribute.name, confidence=attribute.confidence))
        return DetectionFrame(width=width, height=height, detections=tuple(detections))

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pose skeleton on `frame`."""
        if self._last_pose_proto is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, self._last_pose_proto, mp.solutions.pose.POSE_CONNECTIONS
            )
        return frame

    def is_alert_active(self) -> bool:
        """True while a fall, a zone violation, or unusual stillness is confirmed."""
        return self._fall_engine.is_active("fall") or bool(self._zone_engine.active_detections())

    def get_telemetry_snapshot(self) -> Optional[dict]:
        """Raw per-frame fall-relevant signals, for the CSV telemetry stream
        (core/telemetry.py gives every module its own sibling file keyed off
        this dict's fields -- see that module's docstring)."""
        return {
            "person_detected": self._last_posture is not None,
            "posture": self._last_posture.label if self._last_posture else "unknown",
            "posture_confidence": round(self._last_posture.confidence, 3) if self._last_posture else 0.0,
            "hip_y": round(self._last_hip_y, 3) if self._last_hip_y is not None else None,
            "fall_pending": self._fall_pending_since is not None,
            "fall_active": self._fall_engine.is_active("fall"),
            "fall_confidence": round(self._fall_confidence, 3),
        }

    def get_debug_state(self) -> Optional[dict]:
        """Full, human-readable breakdown of the fall state machine: current
        posture, how long ago the subject was last confidently upright,
        whether a fall is pending and via which trigger, how long they've
        been still, and the resulting confidence -- developer/debug use only
        (main.py --debug), same contract as baby_monitoring.py's
        get_debug_state (never sent to the cloud or a notification)."""
        now = time.monotonic()
        if self._fall_pending_since is None:
            trigger = "none"
        elif self._fall_via_recovery:
            trigger = "tracking_loss_recovery"
        else:
            trigger = "measured_rapid_drop"
        active = list(self._zone_engine.active_detections())
        if self._fall_engine.is_active("fall"):
            active.append("fall")
        return {
            "posture": self._last_posture.label if self._last_posture else "unknown",
            "posture_confidence": round(self._last_posture.confidence, 3) if self._last_posture else 0.0,
            "torso_angle_deg": self._last_posture.torso_angle_deg if self._last_posture else None,
            "seconds_since_last_upright": (
                round(now - self._last_upright_at, 2) if self._last_upright_at is not None else None
            ),
            "fall_pending": self._fall_pending_since is not None,
            "fall_trigger": trigger,
            "seconds_still": round(now - self._still_since, 2) if self._still_since is not None else None,
            "fall_confidence": round(self._fall_confidence, 3),
            "fall_state": self._fall_engine.state_of("fall"),
            "active_alerts": sorted(active),
        }

    def close(self) -> None:
        """Release the MediaPipe Pose graph (leaked per preset/mode switch otherwise --
        found in a repo-wide audit: this is one of the two modules
        _apply_monitoring_mode/_switch_preset hot-swap between, and both already call
        old_module.close(), but this override was missing, so every swap away from
        fall_detection leaked a live Pose graph)."""
        with contextlib.suppress(Exception):  # closing twice must be harmless
            self._pose.close()
