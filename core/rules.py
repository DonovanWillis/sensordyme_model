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
    instant the window opened, which is not what the user asked for.
  * **count thresholds** — "more than three people" is a property of one frame's
    observation, not of the lifecycle, so `min_count` is checked before the
    observation reaches the engine.
  * **zone binding** — a rule names a zone the user draws in the app; if that
    zone does not exist yet the rule is inert rather than firing frame-wide,
    which is the difference between "alert near the pool" and "alert anywhere".

Everything temporal (persistence, occlusion grace, resolve, cooldown) stays in
EventEngine, so custom rules get exactly the same debouncing as built-in ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from core.detections import DetectionFrame
from core.event_engine import CONFIRMED, EventEngine, RuleSpec
from core.zones import Zone

# Severity -> whether a confirmation should reach the user at all.
_NOTIFY_KINDS = ("push", "sms", "silent")
_SEVERITIES = ("info", "warning", "critical")


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


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    """Whether two pixel-space (x1, y1, x2, y2) boxes intersect at all."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _any_overlap(boxes_a: list, boxes_b: list) -> bool:
    return any(_boxes_overlap(a, b) for a in boxes_a for b in boxes_b)


def _upper_region(box: tuple, fraction: float = 0.35) -> tuple:
    """The top `fraction` of a box — where a standing or seated person's face is.

    The evidence upgrade behind `near_mouth`: a bottle overlapping a person
    anywhere is "holding" at best (it also matches a bottle at their feet in
    box terms); a bottle overlapping the region their face occupies has been
    raised toward the mouth. Still an approximation — a box knows nothing about
    an actual mouth — but it is the difference between "has a drink" and
    "is drinking from it" at the fidelity a box detector can support.
    """
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + (y2 - y1) * fraction)


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

    def __init__(self, persist_s: float) -> None:
        self._persist_s = persist_s
        self._reported: Optional[int] = None
        self._candidate: Optional[int] = None
        self._candidate_since: float = 0.0

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
        return count, previous


class RuleEngine:
    """Evaluates AI-configured rules against per-frame observations."""

    def __init__(
        self,
        raw_rules: Optional[list[dict]] = None,
        zones: Optional[list[Zone]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._clock = clock or datetime.now
        self._zones = {z.name: z for z in (zones or [])}
        self._rules: dict[str, ConfiguredRule] = {}
        self._engine = EventEngine({})
        self._counters: dict[str, _CountWatcher] = {}
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
        self._zones = {z.name: z for z in (zones or [])}

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
        """
        return bool(self._engine.active_detections())

    def pending_zones(self) -> list[str]:
        """Zones rules reference that the user hasn't drawn yet.

        Surfaced so the app can say "draw the pool zone to activate this rule"
        instead of the rule silently never firing.
        """
        return sorted({r.zone for r in self._rules.values() if r.zone and r.zone not in self._zones})

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
        minute_of_day = self._minute_of_day()
        alerts: list[Alert] = []

        for rule in self.rules_for(detection):
            active = present and count >= rule.min_count
            if active and not _in_window(rule.schedule, minute_of_day):
                active = False  # outside its hours the rule observes nothing at all
            if active and rule.zone:
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
        minute_of_day = self._minute_of_day()

        for rule in self._rules.values():
            seen = rule.detection in frame.labels()
            boxes = frame.boxes(rule.detection)

            # Counting is a value, not a condition, so it does not go through
            # the event engine at all. "Is a person present" becomes true and
            # later false; "how many bottles are there" is a number that is
            # worth mentioning when it changes.
            if rule.predicate == "count":
                watcher = self._counters.get(rule.rule_id)
                if watcher is None:
                    watcher = self._counters[rule.rule_id] = _CountWatcher(
                        self._persist_for(rule)
                    )
                settled = watcher.observe(frame.count(rule.detection), now)
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

            active = self._evaluate(rule, frame, seen, boxes)

            if active and not _in_window(rule.schedule, minute_of_day):
                active = False  # outside its hours the rule observes nothing at all
            if active and rule.zone:
                active = self._in_zone(rule.zone, boxes, frame.size)

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

    def _evaluate(self, rule: ConfiguredRule, frame: "DetectionFrame",
                  seen: bool, boxes: list) -> bool:
        """Whether this rule's condition holds in this frame."""
        if rule.predicate == "absent":
            return not seen
        if rule.predicate in ("near", "holding"):
            # Proximity stands in for interaction: overlapping boxes are strong
            # evidence of "with", and no evidence at all of intent. The rule
            # carries a caveat saying so.
            if not seen or not rule.secondary:
                return False
            return _any_overlap(boxes, frame.boxes(rule.secondary))
        if rule.predicate == "near_mouth":
            # The secondary must overlap the target's FACE region, not just the
            # target. This is what separates "drinking from a bottle" evidence
            # from "standing next to a bottle" — see _upper_region.
            if not seen or not rule.secondary:
                return False
            return _any_overlap([_upper_region(b) for b in boxes], frame.boxes(rule.secondary))
        # present / in_zone / count all start from "is it there, enough of it".
        return seen and frame.count(rule.detection) >= rule.min_count

    def active_alerts(self) -> list[str]:
        """rule_ids currently confirmed and unresolved."""
        return self._engine.active_detections()

    # ------------------------------------------------------------------ internals
    def _minute_of_day(self) -> int:
        stamp = self._clock()
        return stamp.hour * 60 + stamp.minute

    def _in_zone(
        self,
        zone_name: str,
        boxes: Optional[list[tuple[float, float, float, float]]],
        frame_size: Optional[tuple[int, int]],
    ) -> bool:
        zone = self._zones.get(zone_name)
        if zone is None or not boxes or not frame_size:
            # An undrawn zone makes the rule inert rather than frame-wide: firing
            # everywhere is worse than not firing, and pending_zones() tells the
            # user exactly what to draw.
            return False
        width, height = frame_size
        return any(zone.contains_box(box, width, height) for box in boxes)
