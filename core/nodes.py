"""Monitoring nodes: structured consumers of the shared perception engine.

A monitoring node is what a mode IS once the camera is shared: it declares what
it needs, consumes the frame's typed Observation (core/perception.py), keeps its
own bounded temporal state, and produces typed events. It never owns the camera
and, for the nodes defined here, never owns a model either — a node that needs
the object detector registers a Demand and reads the detector's output from the
Observation, so two nodes that both want person detection cost one inference.

Three kinds, reported identically (docs/M7_IMPLEMENTATION_MAP.md wire contract):

  preset     the active preset's VisionModule (driver attention, fall
             detection, ...), described by its presets.yaml entry. Its events
             keep their existing path through DeviceRuntime; the node is how it
             is declared, health-checked and reported.
  rules      the owner's custom rules: the existing core/rules.py RuleEngine,
             unchanged, fed the shared detections.
  companion  a node with no model of its own that runs beside the preset:
             `occupancy` (count, entry/exit, zone occupancy, dwell) and
             `zone_watch` (a person in a restricted area — a typed rule bundle
             validated and evaluated by the same RuleEngine, not a second
             policy engine).

Companions are additive: they never replace the preset's module, so on a
driver_safety device nothing here can remove driver attention. Every node's
validation tier comes from core/model_registry.yaml (`monitoring_nodes`) and
can never exceed the evidence behind the capabilities it depends on.

Identity (M8): nodes never track. The perception engine runs the device's one
ObjectTracker (core/tracking.py) on every detector run, and each Observation
carries the live `tracks`; a node keeps only its own per-track state (who it
has counted, who is in which area since when -- core/tracking.ZoneMembership).

Adding a companion node is one class and one registry entry, never a model:

    @register_companion
    class MyNode(MonitoringNode):
        spec = NodeSpec(id="my_node", version=1, kind="companion", ...,
                        components=(pc.OBJECT_DETECTOR,))
        detect_labels = frozenset({"person"})      # what it asks the detector for
        def consume(self, observation, now): ...    # typed events from observation/tracks

plus a `monitoring_nodes.my_node` entry in core/model_registry.yaml (its tier,
capped by its capabilities' evidence; without one it reports
`exists_in_code`). The default demands() asks the SHARED detector for
`detect_labels` at the node's `interval_s` / `min_confidence` settings, so a
new node adds no inference on frames where anyone else already ran the detector
(tests/test_m8_nodes_registry.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from core import perception as pc
from core.events import Event
from core.monitoring_state import Capability
from core.presets import PresetError, Setting, _setting
from core.rules import RuleEngine, _zone_key, validate_rule_set
from core.tracking import SHARED_LOST_GRACE_S, ConfidenceGate, Track, ZoneMembership

KINDS = ("preset", "rules", "companion")
ACTIVE, DEGRADED, FAILED, INACTIVE = "active", "degraded", "failed", "inactive"
#: At most this many companion nodes run beside a preset (bounded work per frame).
MAX_COMPANIONS = 4
#: Tolerance when comparing float durations against settle times.
_TIME_EPS = 1e-6
#: A node event's severity, as the uplink names it.
SEVERITIES = ("loud", "soft", "log")


@dataclass(frozen=True)
class NodeSpec:
    """Everything a node declares about itself."""

    id: str
    version: int
    kind: str
    title: str
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[str, ...] = ()
    #: Perception components (core/perception.py) whose output it consumes.
    components: tuple[str, ...] = ()
    hardware: Mapping[str, str] = field(default_factory=dict)
    settings: tuple[Setting, ...] = ()
    #: event -> the capabilities it needs (feeds core/monitoring_state.py).
    events: Mapping[str, frozenset[str]] = field(default_factory=dict)
    temporal_logic: tuple[str, ...] = ()
    alert_policy: str = ""
    limitations: tuple[str, ...] = ()
    #: Experimental nodes are off by default and never reported as validated.
    experimental: bool = False
    preset: Optional[str] = None

    def settings_schema(self) -> dict[str, Setting]:
        return {s.key: s for s in self.settings}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "version": self.version, "kind": self.kind, "title": self.title,
            "required_capabilities": list(self.required_capabilities),
            "optional_capabilities": list(self.optional_capabilities),
            "components": list(self.components), "hardware": dict(self.hardware),
            "settings": [s.to_dict() for s in self.settings],
            "events": sorted(self.events), "temporal_logic": list(self.temporal_logic),
            "alert_policy": self.alert_policy, "limitations": list(self.limitations),
            "experimental": self.experimental, "preset": self.preset,
        }


@dataclass(frozen=True)
class NodeEvent:
    """One event a node produced, with how prominently it should be delivered."""

    event: Event
    severity: str = "soft"  # loud | soft | log
    notify: Optional[str] = None  # push | silent | None (backend default)


def _settings(raw: Iterable[dict]) -> tuple[Setting, ...]:
    return tuple(_setting(item, "node settings") for item in raw)


def validate_node_settings(spec: NodeSpec, settings: Any) -> dict:
    """The cleaned settings, or PresetError naming the first bad key."""
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise PresetError(f"node '{spec.id}': settings must be an object of key -> value")
    schema = spec.settings_schema()
    cleaned = {key: s.default for key, s in schema.items()}
    for key in sorted(settings):
        setting = schema.get(key)
        if setting is None:
            known = ", ".join(sorted(schema)) or "none"
            raise PresetError(f"'{key}' is not a setting of node '{spec.id}' (settings: {known})", str(key))
        cleaned[key] = setting.coerce(settings[key])
    return cleaned


def node_state(spec: NodeSpec, capabilities: Mapping[str, Capability], *,
               stopped: bool = False) -> tuple[str, list[str], Optional[str]]:
    """(state, missing required capabilities, reason) for one node right now."""
    if stopped:
        return INACTIVE, [], "monitoring stopped"

    def impaired(name: str) -> bool:
        cap = capabilities.get(name)
        return cap is None or cap.impaired

    missing = [c for c in spec.required_capabilities if impaired(c)]
    if missing:
        why = "; ".join(f"{c}: {capabilities[c].reason}" if c in capabilities and capabilities[c].reason
                        else f"{c}: not provided" for c in missing)
        return FAILED, missing, why
    starting = [c for c in spec.required_capabilities
                if c in capabilities and capabilities[c].state == "starting"]
    lost = [c for c in spec.optional_capabilities if c in capabilities and capabilities[c].impaired]
    if starting:
        return DEGRADED, [], "starting: " + ", ".join(starting)
    if lost:
        return DEGRADED, [], "optional capability unavailable: " + ", ".join(lost)
    return ACTIVE, [], None


class MonitoringNode(ABC):
    """A structured consumer of shared perception (see the module docstring)."""

    spec: ClassVar[NodeSpec]
    #: Detector labels the default demands() asks for (empty: whatever runs).
    detect_labels: ClassVar[frozenset[str]] = frozenset()
    #: Detector cadence when no setting `interval_s` says otherwise.
    default_interval_s: ClassVar[float] = 0.5

    def __init__(self, settings: Optional[dict] = None) -> None:
        self.settings = validate_node_settings(self.spec, settings)
        self._zones: list = []
        self.events_emitted = 0
        self.observations = 0

    @property
    def id(self) -> str:
        return self.spec.id

    def demands(self) -> tuple[pc.Demand, ...]:
        """What it needs the engine to run for it: the shared detector for
        `detect_labels`, when its spec consumes the detector (none otherwise)."""
        if pc.OBJECT_DETECTOR not in self.spec.components:
            return ()
        interval = float(self.settings.get("interval_s", self.default_interval_s))
        confidence = float(self.settings.get("min_confidence", 0.25))
        return (pc.Demand(self.id, pc.OBJECT_DETECTOR, frozenset(self.detect_labels),
                          min_interval_s=interval, min_confidence=confidence),)

    def set_zones(self, zones: list) -> None:
        self._zones = list(zones or [])

    @abstractmethod
    def consume(self, observation: pc.Observation, now: float) -> list[NodeEvent]:
        """Advance this node on one frame's observation."""

    def metrics(self) -> dict:
        """Current values worth showing (counts, ...), never accuracy claims."""
        return {}

    def describe(self) -> dict:
        """What survives a restart: {node, enabled, settings}."""
        return {"node": self.id, "enabled": True, "settings": dict(self.settings)}

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------- occupancy
_OCCUPANCY_SPEC = NodeSpec(
    id="people_counter", version=1, kind="companion", title="Occupancy count",
    required_capabilities=("camera", "inference", "objects"),
    components=(pc.OBJECT_DETECTOR,),
    hardware={"camera": "a view of the room or cabin", "accelerator": "object detector (shared)"},
    settings=_settings([
        {"key": "min_confidence", "title": "Person confidence", "type": "number", "default": 0.5,
         "min": 0.2, "max": 0.9, "step": 0.05},
        {"key": "settle_s", "title": "Time a new count must hold", "type": "number", "default": 2.0,
         "min": 0.5, "max": 30.0, "step": 0.5, "unit": "s"},
        {"key": "dwell_s", "title": "Time in an area before a dwell alert", "type": "number", "default": 30,
         "min": 5, "max": 3600, "step": 5, "unit": "s"},
        {"key": "capacity", "title": "Alert above this many people (0 = off)", "type": "number", "default": 0,
         "min": 0, "max": 50, "step": 1, "unit": "people"},
        {"key": "interval_s", "title": "Detection interval when nothing else runs it", "type": "number",
         "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "unit": "s"},
    ]),
    events={
        "person_entered": frozenset({"objects"}), "person_exited": frozenset({"objects"}),
        "capacity_exceeded": frozenset({"objects"}), "zone_occupied": frozenset({"objects"}),
        "zone_entered": frozenset({"objects"}), "zone_exited": frozenset({"objects"}),
        "zone_dwell": frozenset({"objects"}),
    },
    temporal_logic=(
        "people are the shared tracker's confirmed person tracks (core/tracking.py), one id each",
        ("a person counts once their track has been followed for settle_s; the people present when "
        "counting starts are the baseline, not entries"),
        ("an entry is reported when a new person counts; an exit when their track is retired "
        "(unseen for the tracker's grace, 2 s)"),
        ("a person is in an area once their feet have been inside it for settle_s, and has left "
        "after settle_s outside; the area is occupied while anyone is in it"),
        "a dwell alert fires once per person per visit, after dwell_s inside",
        ("after a detection gap longer than the tracker's grace the count is re-established "
        "without entry/exit events (nothing was observed in between)"),
    ),
    alert_policy=("entries/exits and area crossings are logged; capacity and per-person dwell alerts "
                  "are pushed"),
    limitations=(
        "Counts come from person boxes: people hidden behind others are not counted",
        ("Identity is box tracking, not re-identification: two people crossing can swap ids, and "
        "someone out of view for more than 2 s returns as a new person"),
        "Evaluated only on the virtual device (synthetic scenes); no in-domain occupancy evidence",
    ),
)


class OccupancyNode(MonitoringNode):
    """People count, entry/exit, area crossings and per-person dwell, from the
    shared tracker's person tracks (core/tracking.py).

    An Observation without tracks (a replay, a hand-built frame) falls back to
    counting per-frame person boxes with the same settle time.
    """

    spec = _OCCUPANCY_SPEC
    detect_labels = frozenset({"person"})

    def __init__(self, settings: Optional[dict] = None) -> None:
        super().__init__(settings)
        self.count: Optional[int] = None  # the settled count; None until one has held
        self.degraded_reason: Optional[str] = None
        # From tracks: who is counted (id -> latest snapshot), and since when counting runs.
        self._gate = ConfidenceGate(float(self.settings["min_confidence"]))
        self._present: dict[int, Track] = {}
        self._baseline_from: Optional[float] = None
        self._last_fresh: Optional[float] = None
        self._membership = ZoneMembership(settle_s=float(self.settings["settle_s"]))
        self._dwell_fired: set[tuple[str, int]] = set()
        self._tracked = False
        # From per-frame boxes (no tracker behind the observation).
        self._candidate: Optional[tuple[int, float]] = None
        self._zone_since: dict[str, float] = {}  # zone key -> first time seen occupied (this streak)
        self._zone_occupied: dict[str, bool] = {}
        self._zone_dwell_fired: set[str] = set()
        self._zone_counts: dict[str, int] = {}

    def set_zones(self, zones: list) -> None:
        super().set_zones(zones)
        keys = {_zone_key(z.name) for z in self._zones}
        for state in (self._zone_since, self._zone_occupied, self._zone_counts):
            for key in [k for k in state if k not in keys]:
                del state[key]
        self._zone_dwell_fired &= keys
        names = {z.name for z in self._zones}
        self._dwell_fired = {k for k in self._dwell_fired if k[0] in names}

    def _event(self, kind: str, detail: str, confidence: Optional[float]) -> Event:
        return Event.now(kind, 0.0, confidence, f"node:{self.id}", detail)

    def consume(self, observation: pc.Observation, now: float) -> list[NodeEvent]:
        if observation.detections is None:
            status = observation.components.get(pc.OBJECT_DETECTOR)
            self.degraded_reason = (f"object detection {status.state}: {status.reason}"
                                    if status is not None and status.reason else "object detection unavailable")
            self._candidate = None  # a count interrupted by a gap must hold again from scratch
            return []
        self.degraded_reason = None
        if not observation.fresh_detections:
            return []  # nothing new since the last detector run
        self.observations += 1
        if observation.tracks is not None:
            events = self._consume_tracks(observation, observation.tracks, now)
        else:
            events = self._consume_boxes(observation, now)
        self.events_emitted += len(events)
        return events

    # ------------------------------------------------------------------ from the shared tracks
    def _consume_tracks(self, observation: pc.Observation, tracks: tuple[Track, ...],
                        now: float) -> list[NodeEvent]:
        self._tracked = True
        settle = float(self.settings["settle_s"])
        if self._last_fresh is not None and now - self._last_fresh > SHARED_LOST_GRACE_S:
            # A gap the tracker cannot bridge: whoever is here now has a new id.
            # Re-establish the count without claiming entries or exits nobody saw.
            self._baseline_from, self.count = None, None
            self._present.clear()
            self._membership.clear()
            self._dwell_fired.clear()
        self._last_fresh = now
        live = {t.track_id: t for t in tracks}
        people = self._gate.admit(t for t in tracks if t.label == "person" and t.confirmed)
        # (a tolerance for float time: a track first seen when counting began has
        # age == settle exactly when the baseline is taken)
        qualifying = {t.track_id: t for t in people if t.age_s + _TIME_EPS >= settle}
        events: list[NodeEvent] = []
        if self._baseline_from is None:
            self._baseline_from = now
        counting = self.count is not None
        if not counting:
            if now - self._baseline_from + _TIME_EPS >= settle:
                # The people here when counting starts are the baseline, not entries.
                self._present = dict(qualifying)
                self.count = len(self._present)
        else:
            events += self._count_changes(qualifying, live)
        for tid in self._present:
            if tid in live:
                self._present[tid] = live[tid]
        areas = self._area_changes(people, set(live), observation, now)
        # Likewise whoever stands in an area when counting starts did not just walk in.
        return events + areas if counting else events

    def _count_changes(self, qualifying: dict[int, Track], live: dict[int, Track]) -> list[NodeEvent]:
        events: list[NodeEvent] = []
        previous = len(self._present)
        for tid in sorted(t for t in self._present if t not in live):
            last = self._present.pop(tid)
            events.append(NodeEvent(self._event(
                "person_exited", f"person #{tid} left after {last.age_s:.0f}s ({len(self._present)} now)",
                1.0), "log"))
        for tid in sorted(t for t in qualifying if t not in self._present):
            track = self._present[tid] = qualifying[tid]
            events.append(NodeEvent(self._event(
                "person_entered", f"person #{tid} entered ({len(self._present)} now)",
                round(track.confidence, 3)), "log"))
        self.count = len(self._present)
        capacity = int(self.settings["capacity"])
        if capacity and self.count > capacity >= previous:
            confidence = min(t.confidence for t in self._present.values())
            events.append(NodeEvent(self._event(
                "capacity_exceeded", f"{self.count} people (was {previous}); capacity {capacity}",
                round(confidence, 3)), "soft", "push"))
        return events

    def _area_changes(self, people: list[Track], live: set[int], observation: pc.Observation,
                      now: float) -> list[NodeEvent]:
        if not self._zones:
            return []
        events: list[NodeEvent] = []
        occupied = {z.name: bool(self._membership.members(z.name)) for z in self._zones}
        by_id = {t.track_id: t for t in people}
        for change in self._membership.update(people, self._zones, observation.width, observation.height,
                                              now, live=live):
            if change.kind == "entered":
                track = by_id.get(change.track_id)
                confidence = round(track.confidence, 3) if track is not None else None
                if not occupied.get(change.zone):
                    occupied[change.zone] = True
                    events.append(NodeEvent(self._event(
                        "zone_occupied", f"'{change.zone}' occupied (person #{change.track_id})", confidence),
                        "log"))
                events.append(NodeEvent(self._event(
                    "zone_entered", f"person #{change.track_id} entered '{change.zone}'", confidence), "log"))
            else:
                self._dwell_fired.discard((change.zone, change.track_id))
                events.append(NodeEvent(self._event(
                    "zone_exited", f"person #{change.track_id} left '{change.zone}' after {change.dwell_s:.0f}s",
                    1.0), "log"))
        dwell = float(self.settings["dwell_s"])
        for zone in self._zones:
            for tid, seconds in self._membership.dwell(zone.name, now).items():
                track = by_id.get(tid)
                if seconds < dwell or (zone.name, tid) in self._dwell_fired or track is None or not track.visible:
                    continue
                self._dwell_fired.add((zone.name, tid))
                events.append(NodeEvent(self._event(
                    "zone_dwell", f"person #{tid} in '{zone.name}' for {seconds:.0f}s",
                    round(track.confidence, 3)), "soft", "push"))
        return events

    # ------------------------------------------------------------------ from per-frame boxes
    def _consume_boxes(self, observation: pc.Observation, now: float) -> list[NodeEvent]:
        floor = float(self.settings["min_confidence"])
        people = [d for d in observation.detections or () if d.label == "person" and d.confidence >= floor]
        mean_conf = (sum(p.confidence for p in people) / len(people)) if people else None
        events = self._settle_count(len(people), mean_conf, now)
        events += self._zones_update(people, observation, now)
        return events

    def _settle_count(self, count: int, confidence: Optional[float], now: float) -> list[NodeEvent]:
        if self._candidate is None or self._candidate[0] != count:
            self._candidate = (count, now)
        if now - self._candidate[1] < float(self.settings["settle_s"]) or count == self.count:
            return []
        previous, self.count = self.count, count
        if previous is None:
            return []  # the first settled count is a baseline, not an entry or exit
        events: list[NodeEvent] = []
        detail = f"{count} {'person' if count == 1 else 'people'} (was {previous})"
        kind = "person_entered" if count > previous else "person_exited"
        events.append(NodeEvent(self._event(kind, detail, confidence if count else 1.0), "log"))
        capacity = int(self.settings["capacity"])
        if capacity and count > capacity >= previous:
            events.append(NodeEvent(self._event("capacity_exceeded", f"{detail}; capacity {capacity}",
                                                confidence), "soft", "push"))
        return events

    def _zones_update(self, people: list, observation: pc.Observation, now: float) -> list[NodeEvent]:
        events: list[NodeEvent] = []
        settle = float(self.settings["settle_s"])
        dwell = float(self.settings["dwell_s"])
        for zone in self._zones:
            key = _zone_key(zone.name)
            inside = [p for p in people if zone.contains_box(p.box, observation.width, observation.height)]
            self._zone_counts[key] = len(inside)
            if not inside:
                self._zone_since.pop(key, None)
                self._zone_occupied[key] = False
                self._zone_dwell_fired.discard(key)
                continue
            since = self._zone_since.setdefault(key, now)
            if not self._zone_occupied.get(key) and now - since >= settle:
                self._zone_occupied[key] = True
                events.append(NodeEvent(self._event(
                    "zone_occupied", f"{len(inside)} in '{zone.name}'",
                    sum(p.confidence for p in inside) / len(inside)), "log"))
            if key not in self._zone_dwell_fired and now - since >= dwell:
                self._zone_dwell_fired.add(key)
                events.append(NodeEvent(self._event(
                    "zone_dwell", f"someone in '{zone.name}' for {now - since:.0f}s",
                    sum(p.confidence for p in inside) / len(inside)), "soft", "push"))
        return events

    def metrics(self) -> dict:
        if self._tracked:
            zones = {}
            for zone in self._zones:
                dwell = self._membership.dwell(zone.name, self._last_fresh or 0.0)
                zones[zone.name] = {"count": len(dwell), "occupied": bool(dwell),
                                    "longest_dwell_s": round(max(dwell.values()), 1) if dwell else 0.0}
            return {"count": self.count, "tracked": True, "people": sorted(self._present), "zones": zones}
        names = {_zone_key(z.name): z.name for z in self._zones}
        return {"count": self.count,
                "zones": {names.get(k, k): {"count": n, "occupied": bool(self._zone_occupied.get(k))}
                          for k, n in sorted(self._zone_counts.items())}}


# ---------------------------------------------------------------------- zone watch
_ZONE_WATCH_SPEC = NodeSpec(
    id="zone_watch", version=1, kind="companion", title="Restricted areas",
    required_capabilities=("camera", "inference", "objects"),
    components=(pc.OBJECT_DETECTOR,),
    hardware={"camera": "a view of the areas to watch", "accelerator": "object detector (shared)"},
    settings=_settings([
        {"key": "persist_s", "title": "Time in an area before an alert", "type": "number", "default": 2.0,
         "min": 0.5, "max": 60.0, "step": 0.5, "unit": "s"},
        {"key": "dwell_s", "title": "Per-person time in an area (0 = alert on entry)", "type": "number",
         "default": 0, "min": 0, "max": 3600, "step": 1, "unit": "s"},
        {"key": "severity", "title": "Alert severity", "type": "enum", "default": "warning",
         "options": ["warning", "critical"]},
        {"key": "min_confidence", "title": "Person confidence", "type": "number", "default": 0.5,
         "min": 0.2, "max": 0.9, "step": 0.05},
    ]),
    events={"restricted_area": frozenset({"objects"})},
    temporal_logic=(
        "one typed rule per restricted area the owner has drawn (core/rules.py in_zone)",
        "confirmed by the rule engine's own persistence, cooldown and resolve",
        "a per-person dwell (dwell_s > 0) follows the shared tracker's ids, not a second matcher",
    ),
    alert_policy="each confirmed entry is pushed at the chosen severity",
    limitations=(
        "Only areas drawn in the app with kind 'restricted' are watched; an undrawn area watches nothing",
        "Person boxes only: no PPE, vehicle or forklift detection exists",
        "Evaluated only on the virtual device (synthetic scenes)",
    ),
)


class ZoneWatchNode(MonitoringNode):
    """A person in a restricted area: a typed rule bundle on the existing rule engine."""

    spec = _ZONE_WATCH_SPEC
    detect_labels = frozenset({"person"})

    def __init__(self, settings: Optional[dict] = None) -> None:
        super().__init__(settings)
        self.engine = RuleEngine()
        self.problems: list[str] = []

    def rules(self) -> list[dict]:
        """The bundle: one rule per restricted area (the existing typed rule format)."""
        out = []
        for zone in self._zones:
            if getattr(zone, "kind", "restricted") != "restricted":
                continue
            rule = {"target": "person", "predicate": "in_zone", "zone": zone.name,
                    "description": f"Person in restricted area '{zone.name}'",
                    "severity": self.settings["severity"], "notify": "push",
                    "persist_s": float(self.settings["persist_s"]),
                    "confidence": float(self.settings["min_confidence"])}
            if float(self.settings["dwell_s"]) > 0:
                rule["dwell_s"] = float(self.settings["dwell_s"])
            out.append(rule)
        return out

    def set_zones(self, zones: list) -> None:
        super().set_zones(zones)
        rules = self.rules()
        self.problems = validate_rule_set(rules, label_ok=lambda label: label == "person",
                                          zone_names=[z.name for z in self._zones])
        self.engine.set_zones(self._zones)
        self.engine.replace_rules([] if self.problems else rules)

    def consume(self, observation: pc.Observation, now: float) -> list[NodeEvent]:
        if not observation.fresh_detections:
            return []
        frame = observation.detection_frame({"person": float(self.settings["min_confidence"])},
                                            labels=("person",))
        if frame is None:
            return []
        self.observations += 1
        events = []
        for alert in self.engine.observe_frame(frame, now=now):
            event = Event.now("restricted_area", alert.duration_s, alert.confidence, f"node:{self.id}",
                              alert.message())
            events.append(NodeEvent(event, "loud" if alert.severity == "critical" else "soft", alert.notify))
        self.events_emitted += len(events)
        return events

    def metrics(self) -> dict:
        return {"areas": [r["zone"] for r in self.rules()], "problems": list(self.problems)}


#: Every companion node this software can run, by id (filled by register_companion).
COMPANIONS: dict[str, type[MonitoringNode]] = {}


def register_companion(cls: type[MonitoringNode]) -> type[MonitoringNode]:
    """Make a companion node runnable by id (`set_nodes`, presets.yaml `nodes:`,
    the catalogue the backend validates against). Refuses a spec that is not a
    companion, an id already taken by another class, and a component the
    perception engine does not have."""
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, NodeSpec) or spec.kind != "companion":
        raise ValueError(f"{cls.__name__}: a companion node needs a NodeSpec with kind 'companion'")
    unknown = [c for c in spec.components if c not in pc.COMPONENTS]
    if unknown:
        raise ValueError(f"{spec.id}: unknown perception component(s) {unknown}")
    existing = COMPANIONS.get(spec.id)
    if existing is not None and existing is not cls:
        raise ValueError(f"monitoring node id '{spec.id}' is already registered by {existing.__name__}")
    COMPANIONS[spec.id] = cls
    return cls


register_companion(OccupancyNode)
register_companion(ZoneWatchNode)


def companion_spec(node_id: str) -> Optional[NodeSpec]:
    cls = COMPANIONS.get(node_id)
    return cls.spec if cls is not None else None


def build_companion(node_id: str, settings: Optional[dict] = None) -> MonitoringNode:
    """A companion node (raises KeyError for an unknown id, PresetError for bad settings)."""
    cls = COMPANIONS.get(node_id)
    if cls is None:
        raise KeyError(f"unknown monitoring node '{node_id}' (known: {', '.join(sorted(COMPANIONS))})")
    return cls(settings)


# ---------------------------------------------------------------------- preset and rules nodes
#: The driver-safety behaviours that exist only as experiments (core/behavior/,
#: `behavior_v2`, off by default). Reported so the app can show them as such.
EXPERIMENTAL_DRIVER_FEATURES = (
    {"id": "behavior_v2", "config_key": "behavior_v2",
     "includes": ["hand-phone interaction", "long-glance classification", "cumulative glance budgets",
                  "gaze zones", "learned temporal phone/eye reasoning"],
     "status": "experimental: not validated; off by default"},
)


def preset_experimental_features(name: str, preset: Mapping[str, Any],
                                 module_config: Optional[Mapping[str, Any]] = None) -> list[dict]:
    """The experimental parts of a preset, as `perception.nodes[].experimental_features`:
    driver_safety's behaviour_v2 bundle, and every temporal policy presets.yaml
    marks `experimental: true` (e.g. baby monitoring's pose-based flailing and
    choking-hazard scoring). `enabled` says whether THIS device runs it (its
    module config key), so the app can show "experimental, on" truthfully."""
    config = module_config or {}
    if name == "driver_safety":
        return [{**f, "enabled": bool(config.get(f["config_key"], False))} for f in EXPERIMENTAL_DRIVER_FEATURES]
    out = []
    for policy in preset.get("temporal_policies") or []:
        if not isinstance(policy, dict) or policy.get("experimental") is not True or not policy.get("event"):
            continue
        key = policy.get("config_key")
        out.append({
            "id": str(policy["event"]), "config_key": key, "includes": [str(policy["event"])],
            "status": "experimental: unevaluated on real footage; not validated",
            "enabled": bool(config.get(key, True)) if key else True,
        })
    return out


def preset_spec(name: str, preset: Mapping[str, Any]) -> NodeSpec:
    """The node a preset's module runs as, from its presets.yaml entry."""
    from core.presets import settings_schema

    events: dict[str, frozenset[str]] = {}
    for policy in preset.get("temporal_policies") or []:
        if isinstance(policy, dict) and policy.get("event"):
            events[str(policy["event"])] = frozenset(
                c for c in preset.get("required_capabilities") or () if c not in ("audible_alert", "alert_output"))
    return NodeSpec(
        id=name, version=int(preset.get("version") or 1), kind="preset",
        title=str(preset.get("title") or name),
        required_capabilities=tuple(preset.get("required_capabilities") or ()),
        optional_capabilities=tuple(preset.get("optional_capabilities") or ()),
        components=tuple(pc.COMPONENT_OF[c] for c in (*(preset.get("required_capabilities") or ()),
                                                      *(preset.get("optional_capabilities") or ()))
                         if c in pc.COMPONENT_OF),
        hardware={str(k): str(v) for k, v in (preset.get("hardware") or {}).items()},
        settings=tuple(settings_schema(dict(preset)).values()),
        events=events,
        temporal_logic=tuple(str(p.get("description") or p.get("event"))
                             for p in preset.get("temporal_policies") or [] if isinstance(p, dict)),
        alert_policy="; ".join(f"{p.get('tone')}: {', '.join(p.get('events') or [])}"
                               for p in preset.get("alert_patterns") or [] if isinstance(p, dict)),
        limitations=tuple(str(x) for x in preset.get("limitations") or ()),
        preset=name,
    )


RULES_SPEC = NodeSpec(
    id="rules", version=1, kind="rules", title="Custom rules",
    required_capabilities=("camera", "inference", "objects"),
    components=(pc.OBJECT_DETECTOR,),
    alert_policy="each rule's own severity and notify choice",
    temporal_logic=("core/rules.py: persistence, cooldown, dwell and schedules per rule",),
    limitations=("Rules can only target labels this device detects; an undrawn area is inert",),
)


def wire_entry(spec: NodeSpec, *, state: str, missing: list[str], reason: Optional[str],
               validation: str, metrics: Optional[dict] = None, settings: Optional[dict] = None,
               experimental_features: Iterable[dict] = ()) -> dict:
    """One `perception.nodes[]` entry (sd-perception/1)."""
    entry = {
        "id": spec.id, "kind": spec.kind, "version": str(spec.version), "title": spec.title,
        "state": state, "required_capabilities": list(spec.required_capabilities),
        "optional_capabilities": list(spec.optional_capabilities),
        "missing_capabilities": list(missing), "validation": validation,
        "experimental": spec.experimental, "reason": reason,
    }
    if metrics:
        entry["metrics"] = metrics
    if settings is not None:
        entry["settings"] = dict(settings)
    features = list(experimental_features)
    if features:
        entry["experimental_features"] = features
    return entry


def validation_tier(node_id: str, registry: Any = None) -> str:
    """The node's tier from the registry (`monitoring_nodes`); never stronger than
    its capabilities' evidence. Unknown nodes are `exists_in_code`."""
    try:
        from core import model_registry as mr

        registry = registry if registry is not None else _registry()
        return mr.node_tier(registry, node_id)
    except Exception:  # noqa: BLE001 - an unreadable registry claims nothing
        return "exists_in_code"


_REGISTRY_CACHE: dict[str, Any] = {}


def _registry() -> Any:
    if "r" not in _REGISTRY_CACHE:
        from core.model_registry import load_registry

        _REGISTRY_CACHE["r"] = load_registry()
    return _REGISTRY_CACHE["r"]
