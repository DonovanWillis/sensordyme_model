"""AI-configured rules: the layer that makes a user's plain-language request
actually change what the device watches for.

`Configure with AI` produces validated rule dicts (see `_RULE_SCHEMA` in
cloud/app.py) which the backend stores on the device row. Until this module they
went no further — the proposal was persisted and nothing on the device ever read
it, so a custom rule changed the text in the app and nothing about the camera.

The path is now:

    plain language -> Claude -> validated rule dicts -> device (uplink)
        -> ConfiguredRule (here) -> EventEngine -> Alert -> push

What this layer owns, and the event engine deliberately does not:

  * **schedule gating** — "between 11pm and 5am" is enforced by refusing to
    accumulate observations outside the window, not by suppressing the alert at
    the end. Gating only the alert would let a rule confirm at 22:59 and fire the
    instant the window opened, which is not what the user asked for. Every
    predicate is gated, counting included. Hours are read in the rule's own
    `timezone` (IANA name) when it carries one, else the engine's, else the
    device's local time — the person who typed "after 11pm" meant their clock.
  * **count thresholds** — "more than three people" is a property of one frame's
    observation, not of the lifecycle, so `min_count` is checked before the
    observation reaches the engine. With a zone, only subjects INSIDE the zone
    count: "three people in the driveway" is not "one in the driveway and two
    on the pavement".
  * **zone binding** — a rule names a zone the user draws in the app; if that
    zone does not exist yet the rule is inert rather than firing frame-wide,
    which is the difference between "alert near the pool" and "alert anywhere".
    That holds for every predicate: an absence rule with a zone asks whether
    the zone is empty, and a counting rule counts inside it.
  * **stillness** — `still` asks whether the subject has stopped moving, which
    no single frame can answer; see `_StillnessWatcher`.

Everything temporal (persistence, occlusion grace, resolve, cooldown) stays in
EventEngine, so custom rules get exactly the same debouncing as built-in ones.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo

from core.detections import DetectionFrame
from core.detectors import iou
from core.detectors import upper_region as _upper_region
from core.event_engine import CONFIRMED, EventEngine, RuleSpec
from core.zones import Zone

# Severity -> whether a confirmation should reach the user at all.
_NOTIFY_KINDS = ("push", "sms", "silent")
_SEVERITIES = ("info", "warning", "critical")


def _zone_key(name: str) -> str:
    """Fold a zone name to a comparison key that ignores spacing/casing/punctuation.

    A rule names a zone the way a user or Jupiter phrased it in plain language
    ("front door"); the zone itself is named the way it was drawn ("front_door",
    "Front Door", ...) — same place, different string. Without this, the two
    never compare equal and a correctly-drawn zone looks identical to an
    undrawn one: the rule goes inert with no way to tell the two apart from the
    data model alone. Stripping everything but letters/digits is deliberately
    aggressive — a zone name has no reason to depend on which separator someone
    happened to type.
    """
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


@dataclass(frozen=True)
class Alert:
    """A confirmed, user-visible consequence of one configured rule."""

    rule_id: str
    detection: str
    description: str
    severity: str
    notify: str
    confidence: float
    duration_s: float
    zone: str = ""
    # Set only by counting rules. `count` is how many are visible now;
    # `previous_count` is what it was last reported as, or None the first time.
    # Carried so the message can say what changed rather than only what is.
    count: Optional[int] = None
    previous_count: Optional[int] = None

    def message(self) -> str:
        """One sentence a person can read on a lock screen.

        Counting rules say the number, because the number IS the answer someone
        asked for — "7 bottles visible" rather than "bottle rule triggered".
        """
        if self.count is None:
            return self.description
        noun = self.detection if self.count == 1 else _plural(self.detection)
        if self.previous_count is None:
            return f"{self.count} {noun} visible"
        if self.count > self.previous_count:
            return f"{self.count} {noun} visible (was {self.previous_count})"
        if self.count == 0:
            return f"No {_plural(self.detection)} visible now (was {self.previous_count})"
        return f"{self.count} {noun} visible (was {self.previous_count})"


def _plural(noun: str) -> str:
    """Good enough for the object labels a detector emits."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _any_overlap(boxes_a: list, boxes_b: list, *, upper: bool = False) -> bool:
    """Whether any pixel-space (x1, y1, x2, y2) box in `boxes_a` intersects
    any in `boxes_b` at all -- reuses core/detectors.py's iou() (the same
    overlap math driver_attention.py's hand/phone check and occupancy.py's
    tracker matching already use) rather than a second implementation of
    box intersection.

    `upper` tests the upper (face) region of each `boxes_a` box instead of the
    whole box. A box is never compared with itself: when both lists hold the
    same label ("two people close together"), one person's box trivially
    overlaps itself, which made a single person satisfy a two-person relation.
    Identity, not equality, decides "itself" -- the lists hold the frame's own
    box objects, and two detections at the same coordinates are two subjects.
    """
    for a in boxes_a:
        region = _upper_region(a) if upper else a
        for b in boxes_b:
            if b is a:
                continue
            if iou(region, b) > 0:
                return True
    return False


def _resolve_timezone(name: object) -> Optional[tzinfo]:
    """An IANA timezone ("Europe/London"), or None when absent or unknown.

    None means "use the engine default, else device local time" -- an unknown
    name must not disable the rule or move its hours somewhere arbitrary.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return ZoneInfo(name.strip())
    except (ValueError, OSError, KeyError):  # ZoneInfoNotFoundError is a KeyError
        return None


def _minute_of_day(stamp: datetime, zone: Optional[tzinfo]) -> int:
    """Minutes since local midnight of `stamp`, read in `zone` when given.

    A naive `stamp` is device local time (what `datetime.now()` returns), and
    `astimezone` converts it on that assumption.
    """
    if zone is not None:
        stamp = stamp.astimezone(zone)
    return stamp.hour * 60 + stamp.minute


def _parse_schedule(text: str) -> Optional[tuple[int, int]]:
    """'23:00-06:00' -> (1380, 360) in minutes-since-midnight. None means always on.

    Returns None for empty/malformed input so a bad schedule degrades to
    always-active rather than silently disabling the rule the user asked for.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        start_s, end_s = text.split("-", 1)
        sh, sm = (int(p) for p in start_s.strip().split(":", 1))
        eh, em = (int(p) for p in end_s.strip().split(":", 1))
    except (ValueError, TypeError):
        return None
    if not (0 <= sh < 24 and 0 <= sm < 60 and 0 <= eh < 24 and 0 <= em < 60):
        return None
    start, end = sh * 60 + sm, eh * 60 + em
    if start == end:
        return None  # a zero-length window means "always" far more often than "never"
    return start, end


def _in_window(window: Optional[tuple[int, int]], minute_of_day: int) -> bool:
    if window is None:
        return True
    start, end = window
    if start < end:
        return start <= minute_of_day < end
    return minute_of_day >= start or minute_of_day < end  # wraps past midnight


@dataclass
class ConfiguredRule:
    """One validated AI-configure rule, ready to evaluate."""

    rule_id: str
    detection: str          # the object label to watch ("person", "dog")
    description: str
    predicate: str = "present"  # present | absent | in_zone | near | holding | near_mouth | count
    secondary: str = ""     # second object, for relational predicates
    zone: str = ""
    severity: str = "warning"
    notify: str = "push"
    min_count: int = 1
    schedule: Optional[tuple[int, int]] = None
    # IANA timezone the schedule is written in; None = engine default / device local.
    timezone: Optional[tzinfo] = None

    @classmethod
    def from_dict(cls, index: int, raw: dict) -> Optional["ConfiguredRule"]:
        """Build from a stored proposal rule; None if it isn't usable.

        Tolerant on purpose: these dicts come from a model. A rule missing its
        detection id cannot be evaluated at all, but a rule with a nonsense
        severity is still the rule the user asked for, so it is clamped.
        """
        # `target` is the current shape; `detection` is accepted so proposals
        # stored by an older backend keep working after an upgrade.
        detection = str(raw.get("target") or raw.get("detection") or "").strip().lower()
        if not detection:
            return None
        predicate = str(raw.get("predicate") or "present").strip().lower()
        severity = str(raw.get("severity") or "warning")
        notify = str(raw.get("notify") or "push")
        try:
            min_count = max(1, int(raw.get("min_count") or 1))
        except (TypeError, ValueError):
            min_count = 1
        return cls(
            rule_id=f"rule{index}:{predicate}:{detection}",
            detection=detection,
            predicate=predicate,
            secondary=str(raw.get("secondary") or "").strip().lower(),
            description=str(raw.get("description") or detection),
            zone=str(raw.get("zone") or "").strip(),
            severity=severity if severity in _SEVERITIES else "warning",
            notify=notify if notify in _NOTIFY_KINDS else "push",
            min_count=min_count,
            schedule=_parse_schedule(str(raw.get("schedule") or "")),
            timezone=_resolve_timezone(raw.get("timezone")),
        )

    def spec(self, raw: dict) -> RuleSpec:
        """Confirmation requirements for the shared event engine."""
        def number(key: str, default: float, low: float, high: float) -> float:
            try:
                return max(low, min(high, float(raw.get(key, default))))
            except (TypeError, ValueError):
                return default

        # Floors, not just clamps: the AI is instructed to propose conservative
        # values, but a 0s-persistence rule from a bad generation would alert on
        # a single frame, which is the exact failure this product cannot ship.
        persist_s = number("persist_s", 2.0, 0.5, 3600.0)
        return RuleSpec(
            detection=self.rule_id,
            severity=self.severity,
            min_confidence=number("confidence", 0.5, 0.05, 1.0),
            persist_s=persist_s,
            min_frames=max(2, int(min(persist_s, 10.0))),
            cooldown_s=number("cooldown_s", 30.0, 5.0, 86400.0),
        )


class _CountWatcher:
    """Tracks one counting rule, and reports only when the number settles anew.

    Counting has a different shape from the other predicates. "Is a person
    present" is a condition that becomes true and later false; "how many bottles
    are there" is a value, and a value is worth mentioning when it CHANGES.

    Reporting every frame would be hundreds of identical notifications. Reporting
    every change would fire on detector flicker — one frame where a bottle is
    occluded reads as 4, then 5, then 4 again. So a new number has to hold for
    the rule's persistence window before it counts as real.
    """

    def __init__(self, persist_s: float, threshold: int = 1) -> None:
        self._persist_s = persist_s
        # min_count. Above 1 the rule is "tell me when there are N or more", so
        # a number is news only at or above N -- or when it falls back below N
        # from there, which is the all-clear. Changes entirely below N are not.
        self._threshold = max(1, int(threshold))
        self._reported: Optional[int] = None
        self._candidate: Optional[int] = None
        self._candidate_since: float = 0.0

    def interrupt(self) -> None:
        """Forget a count that had not settled yet (outside the rule's hours, or
        its zone is gone). Otherwise a value seen before the window opened would
        be reported the instant it did, having "settled" while nobody asked."""
        self._candidate = None

    def observe(self, count: int, now: float) -> Optional[tuple[int, Optional[int]]]:
        """Feed one frame's count. Returns (count, previous) when worth saying."""
        if count != self._candidate:
            self._candidate = count
            self._candidate_since = now
            return None
        if now - self._candidate_since < self._persist_s:
            return None
        if count == self._reported:
            return None
        first_report = self._reported is None
        previous, self._reported = self._reported, count
        if first_report and count == 0:
            # Starting up pointed at an empty room is not news. Record it as the
            # baseline so a later arrival reads "2 people visible (was 0)", but
            # do not announce that nothing is there.
            return None
        if self._threshold > 1 and count < self._threshold and (
            previous is None or previous < self._threshold
        ):
            return None
        return count, previous


#: Stillness: how far back a subject's position is compared, and how far its box
#: centre may have moved in that time (in box diagonals) and still be "still".
#: Measured in seconds, not frames, so the answer does not change with frame rate.
_STILL_WINDOW_S = 1.0
_STILL_MAX_SHIFT = 0.1
#: A reference position older than this is from before a gap in view -- whoever
#: is there now cannot be judged against it.
_STILL_MAX_AGE_S = 3.0


def _centre_and_diagonal(box) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, math.hypot(x2 - x1, y2 - y1)


class _StillnessWatcher:
    """Whether a subject's box has stayed put over the last second.

    "Still" is a claim about motion, so it needs two moments: a box now, and a
    box about `_STILL_WINDOW_S` ago at nearly the same place (centre shift under
    `_STILL_MAX_SHIFT` of the box diagonal, so it scales with distance from the
    camera). Detector jitter is a few percent of a diagonal; a person walking
    covers about half a diagonal a second. The rule's own persist_s then asks
    for that to hold continuously ("still for ten minutes"), and any real
    movement breaks the run through the event engine's ordinary reset.

    With several subjects, any one of them being still is enough -- there is no
    identity tracking here, so each current box is matched to the nearest
    reference box. Detections without boxes cannot be judged and read as not
    still.
    """

    def __init__(self) -> None:
        self._history: deque[tuple[float, list[tuple[float, float, float]]]] = deque()

    def observe(self, boxes: list, now: float) -> bool:
        while self._history and now - self._history[0][0] > _STILL_MAX_AGE_S:
            self._history.popleft()
        reference = None
        for stamp, centres in reversed(self._history):
            if now - stamp >= _STILL_WINDOW_S:
                reference = centres
                break
        current = [_centre_and_diagonal(b) for b in boxes]
        if current:
            self._history.append((now, current))
        if not current or not reference:
            return False
        for cx, cy, diagonal in current:
            if diagonal <= 0:
                continue
            nearest = min(math.hypot(cx - rx, cy - ry) for rx, ry, _ in reference)
            if nearest / diagonal <= _STILL_MAX_SHIFT:
                return True
        return False


class RuleEngine:
    """Evaluates AI-configured rules against per-frame observations."""

    def __init__(
        self,
        raw_rules: Optional[list[dict]] = None,
        zones: Optional[list[Zone]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        timezone: Optional[str] = None,
    ) -> None:
        self._clock = clock or datetime.now
        self._zones = {_zone_key(z.name): z for z in (zones or [])}
        self._timezone = _resolve_timezone(timezone)
        self._rules: dict[str, ConfiguredRule] = {}
        self._engine = EventEngine({})
        self._counters: dict[str, _CountWatcher] = {}
        self._stillness: dict[str, _StillnessWatcher] = {}
        self._persist: dict[str, float] = {}
        self.replace_rules(raw_rules or [])

    # ------------------------------------------------------------------ configuration
    def replace_rules(self, raw_rules: list[dict]) -> None:
        """Hot-swap the rule set (a new proposal arrived from the app)."""
        self._rules = {}
        self._engine = EventEngine({})  # drop in-flight tracks; they belong to old rules
        # Counting state belongs to the old rules too. Keeping it would make a
        # brand-new rule report "5 bottles (was 3)" from a previous
        # configuration's history, which is a fact about nothing.
        self._counters = {}
        self._stillness = {}
        self._persist = {}
        for index, raw in enumerate(raw_rules or []):
            if not isinstance(raw, dict):
                continue
            rule = ConfiguredRule.from_dict(index, raw)
            if rule is None:
                continue
            self._rules[rule.rule_id] = rule
            spec = rule.spec(raw)
            self._persist[rule.rule_id] = spec.persist_s
            self._engine.add_rule(spec)

    def _persist_for(self, rule: ConfiguredRule) -> float:
        """How long a new count must hold before it is believed.

        Same window the rule uses for everything else, so one setting governs
        how twitchy the rule is rather than two that can disagree.
        """
        return self._persist.get(rule.rule_id, 2.0)

    def set_zones(self, zones: list[Zone]) -> None:
        self._zones = {_zone_key(z.name): z for z in (zones or [])}

    def set_timezone(self, name: Optional[str]) -> None:
        """Default IANA timezone for rules that do not name their own; None or
        an unknown name means device local time."""
        self._timezone = _resolve_timezone(name)

    @property
    def rules(self) -> list[ConfiguredRule]:
        return list(self._rules.values())

    def rules_for(self, detection: str) -> list[ConfiguredRule]:
        return [r for r in self._rules.values() if r.detection == detection]

    def is_alert_active(self) -> bool:
        """Whether any configured rule is currently in a confirmed, unresolved state.

        Drives the same red-screen state a preset's own alerts do. Someone who
        said "alert me when you see a bottle" asked for an alert, and it should
        look like one — the picture and the notification describing the same
        moment must not disagree about whether it is serious.

        A `silent` rule is the opposite request — track it, do not alert — so it
        never drives the alert state on its own.
        """
        return any(
            rule is not None and rule.notify != "silent"
            for rule in (self._rules.get(d) for d in self._engine.active_detections())
        )

    def pending_zones(self) -> list[str]:
        """Zones rules reference that the user hasn't drawn yet.

        Surfaced so the app can say "draw the pool zone to activate this rule"
        instead of the rule silently never firing.
        """
        return sorted({
            r.zone for r in self._rules.values() if r.zone and _zone_key(r.zone) not in self._zones
        })

    # ------------------------------------------------------------------ evaluation
    def observe(
        self,
        detection: str,
        present: bool,
        confidence: float = 1.0,
        count: int = 1,
        now: Optional[float] = None,
        boxes: Optional[list[tuple[float, float, float, float]]] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> list[Alert]:
        """Feed one frame's observation of `detection`; return confirmed alerts.

        `boxes`/`frame_size` are only needed by zone-bound rules — a detection
        reported without them satisfies a zone rule only if it has no zone.
        """
        now = time.monotonic() if now is None else now
        stamp = self._clock()
        alerts: list[Alert] = []

        for rule in self.rules_for(detection):
            active = present and count >= rule.min_count
            if active and not _in_window(rule.schedule, self._rule_minute(rule, stamp)):
                active = False  # outside its hours the rule observes nothing at all
            if active and (rule.zone or rule.predicate == "in_zone"):
                # `or rule.predicate == "in_zone"` is the whole point: a rule
                # whose predicate is in_zone but whose zone is EMPTY used to
                # skip this check and stay active on bare presence, firing
                # frame-wide. _in_zone("") resolves no zone and returns False,
                # so routing through it makes such a rule inert -- the same
                # direction _in_zone already chose for a zone that is named but
                # never drawn. Until now the only thing standing between a
                # model and a frame-wide alert was jupiter/schema.py:145.
                active = self._in_zone(rule.zone, boxes, frame_size)

            for transition in self._engine.update(rule.rule_id, active, confidence, now=now):
                if transition.state != CONFIRMED or rule.notify == "silent":
                    continue
                alerts.append(
                    Alert(
                        rule_id=rule.rule_id,
                        detection=rule.detection,
                        description=rule.description,
                        severity=rule.severity,
                        notify=rule.notify,
                        confidence=transition.confidence,
                        duration_s=transition.duration_s,
                        zone=rule.zone,
                    )
                )
        return alerts

    def observe_frame(self, frame: "DetectionFrame", now: Optional[float] = None) -> list[Alert]:
        """Evaluate every rule against one frame of detections.

        Labels the frame doesn't contain are reported as absent rather than
        skipped — that is what lets a confirmed event resolve, and what makes
        absence rules ("alert when the room becomes empty") possible at all.
        """
        now = time.monotonic() if now is None else now
        alerts: list[Alert] = []
        stamp = self._clock()

        for rule in self._rules.values():
            seen = rule.detection in frame.labels()
            boxes = frame.boxes(rule.detection)
            in_hours = _in_window(rule.schedule, self._rule_minute(rule, stamp))
            # A zone-bound rule sees only the subjects inside its zone; with the
            # zone undrawn it sees nothing at all (inert, never frame-wide).
            zone_bound = bool(rule.zone) or rule.predicate == "in_zone"
            zone = self._zones.get(_zone_key(rule.zone)) if rule.zone else None
            if zone_bound:
                in_zone = (
                    [b for b in boxes if zone.contains_box(b, frame.width, frame.height)]
                    if zone is not None else []
                )
            else:
                in_zone = boxes

            # Counting is a value, not a condition, so it does not go through
            # the event engine at all. "Is a person present" becomes true and
            # later false; "how many bottles are there" is a number that is
            # worth mentioning when it changes. It is still a rule: it keeps
            # its hours and its zone like every other one.
            if rule.predicate == "count":
                watcher = self._counters.get(rule.rule_id)
                if watcher is None:
                    watcher = self._counters[rule.rule_id] = _CountWatcher(
                        self._persist_for(rule), threshold=rule.min_count
                    )
                if not in_hours or (zone_bound and zone is None):
                    watcher.interrupt()
                    continue
                count = len(in_zone) if zone_bound else frame.count(rule.detection)
                settled = watcher.observe(count, now)
                if settled is not None and rule.notify != "silent":
                    count, previous = settled
                    alerts.append(Alert(
                        rule_id=rule.rule_id,
                        detection=rule.detection,
                        description=rule.description,
                        severity=rule.severity,
                        notify=rule.notify,
                        confidence=frame.mean_confidence(rule.detection) if seen else 1.0,
                        duration_s=0.0,
                        zone=rule.zone,
                        count=count,
                        previous_count=previous,
                    ))
                continue

            active = self._evaluate(rule, frame, seen, boxes, zone_bound, zone, in_zone, now)
            if active and not in_hours:
                active = False  # outside its hours the rule observes nothing at all

            confidence = frame.mean_confidence(rule.detection) if seen else 0.0
            # An absence rule is a confident observation of nothing, so it cannot
            # borrow the (zero) confidence of the thing that is missing.
            if rule.predicate == "absent":
                confidence = 1.0

            for transition in self._engine.update(rule.rule_id, active, confidence, now=now):
                if transition.state != CONFIRMED or rule.notify == "silent":
                    continue
                alerts.append(Alert(
                    rule_id=rule.rule_id,
                    detection=rule.detection,
                    description=rule.description,
                    severity=rule.severity,
                    notify=rule.notify,
                    confidence=transition.confidence,
                    duration_s=transition.duration_s,
                    zone=rule.zone,
                ))
        return alerts

    def _evaluate(self, rule: ConfiguredRule, frame: "DetectionFrame", seen: bool,
                  boxes: list, zone_bound: bool, zone: Optional[Zone], in_zone: list,
                  now: float) -> bool:
        """Whether this rule's condition holds in this frame.

        `in_zone` is `boxes` narrowed to the rule's zone when it is zone-bound
        (empty when that zone is not drawn), and `boxes` otherwise.
        """
        if rule.predicate == "absent":
            if not zone_bound:
                return not seen
            # "Nobody at the pool": the zone must exist, hold none of the target,
            # and every sighting must have a box -- a subject the module saw
            # without a location could be standing in the zone.
            located = len(boxes) == frame.count(rule.detection)
            return zone is not None and not in_zone and located
        if rule.predicate in ("near", "holding", "near_mouth"):
            # Proximity stands in for interaction: overlapping boxes are strong
            # evidence of "with", and no evidence at all of intent. The rule
            # carries a caveat saying so. near_mouth tests the target's FACE
            # region rather than the whole target — what separates "drinking
            # from a bottle" evidence from "standing next to a bottle"; see
            # _upper_region.
            if not seen or not rule.secondary or not in_zone:
                return False
            return _any_overlap(in_zone, frame.boxes(rule.secondary),
                                upper=rule.predicate == "near_mouth")
        if rule.predicate == "still":
            watcher = self._stillness.get(rule.rule_id)
            if watcher is None:
                watcher = self._stillness[rule.rule_id] = _StillnessWatcher()
            return watcher.observe(in_zone, now)
        # present / in_zone / lingering / persists all start from "is it there,
        # enough of it" -- inside the zone, when the rule has one.
        if zone_bound:
            return len(in_zone) >= rule.min_count
        return seen and frame.count(rule.detection) >= rule.min_count

    def active_alerts(self) -> list[str]:
        """rule_ids currently confirmed and unresolved."""
        return self._engine.active_detections()

    # ------------------------------------------------------------------ internals
    def _rule_minute(self, rule: ConfiguredRule, stamp: datetime) -> int:
        """Minute of day for `rule`'s schedule: its own timezone, else the
        engine's, else the device's local time."""
        return _minute_of_day(stamp, rule.timezone or self._timezone)

    def _in_zone(
        self,
        zone_name: str,
        boxes: Optional[list[tuple[float, float, float, float]]],
        frame_size: Optional[tuple[int, int]],
    ) -> bool:
        zone = self._zones.get(_zone_key(zone_name))
        if zone is None or not boxes or not frame_size:
            # An undrawn zone makes the rule inert rather than frame-wide: firing
            # everywhere is worse than not firing, and pending_zones() tells the
            # user exactly what to draw.
            return False
        width, height = frame_size
        return any(zone.contains_box(box, width, height) for box in boxes)
