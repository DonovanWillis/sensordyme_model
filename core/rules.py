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
    detection: str
    description: str
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
        detection = str(raw.get("detection") or "").strip()
        if not detection:
            return None
        severity = str(raw.get("severity") or "warning")
        notify = str(raw.get("notify") or "push")
        try:
            min_count = max(1, int(raw.get("min_count") or 1))
        except (TypeError, ValueError):
            min_count = 1
        return cls(
            rule_id=f"rule{index}:{detection}",
            detection=detection,
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
        self.replace_rules(raw_rules or [])

    # ------------------------------------------------------------------ configuration
    def replace_rules(self, raw_rules: list[dict]) -> None:
        """Hot-swap the rule set (a new proposal arrived from the app)."""
        self._rules = {}
        self._engine = EventEngine({})  # drop in-flight tracks; they belong to old rules
        for index, raw in enumerate(raw_rules or []):
            if not isinstance(raw, dict):
                continue
            rule = ConfiguredRule.from_dict(index, raw)
            if rule is None:
                continue
            self._rules[rule.rule_id] = rule
            self._engine.add_rule(rule.spec(raw))

    def set_zones(self, zones: list[Zone]) -> None:
        self._zones = {z.name: z for z in (zones or [])}

    @property
    def rules(self) -> list[ConfiguredRule]:
        return list(self._rules.values())

    def rules_for(self, detection: str) -> list[ConfiguredRule]:
        return [r for r in self._rules.values() if r.detection == detection]

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
