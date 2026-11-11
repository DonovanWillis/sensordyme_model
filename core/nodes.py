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

KINDS = ("preset", "rules", "companion")
ACTIVE, DEGRADED, FAILED, INACTIVE = "active", "degraded", "failed", "inactive"
#: At most this many companion nodes run beside a preset (bounded work per frame).
MAX_COMPANIONS = 4
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

    def __init__(self, settings: Optional[dict] = None) -> None:
        self.settings = validate_node_settings(self.spec, settings)
        self._zones: list = []
        self.events_emitted = 0
        self.observations = 0

    @property
    def id(self) -> str:
        return self.spec.id

    def demands(self) -> tuple[pc.Demand, ...]:
        """What it needs the engine to run for it (none for nodes fed by a module)."""
        return ()

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
        "zone_dwell": frozenset({"objects"}),
    },
    temporal_logic=(
        "a new person count must hold for settle_s before it is the count",
        "entry/exit are reported when the settled count rises/falls",
        "an area is occupied once a person box has stayed inside it for settle_s",
        "a dwell alert fires once per occupancy, after dwell_s",
    ),
    alert_policy="entries/exits and area occupancy are logged; capacity and dwell alerts are pushed",
    limitations=(
        "Counts come from per-frame person boxes: people hidden behind others are not counted",
        "No identity: two people swapping places reads as no change",
        "Evaluated only on the virtual device (synthetic scenes); no in-domain occupancy evidence",
    ),
)


class OccupancyNode(MonitoringNode):
    """People count, entry/exit, area occupancy and dwell, from shared person boxes."""

    spec = _OCCUPANCY_SPEC

    def __init__(self, settings: Optional[dict] = None) -> None:
        super().__init__(settings)
        self.count: Optional[int] = None  # the settled count; None until one has held
        self._candidate: Optional[tuple[int, float]] = None
        self._zone_since: dict[str, float] = {}  # zone key -> first time seen occupied (this streak)
        self._zone_occupied: dict[str, bool] = {}
        self._zone_dwell_fired: set[str] = set()
        self._zone_counts: dict[str, int] = {}
        self.degraded_reason: Optional[str] = None

    def demands(self) -> tuple[pc.Demand, ...]:
        return (pc.Demand(self.id, pc.OBJECT_DETECTOR, frozenset({"person"}),
                          min_interval_s=float(self.settings["interval_s"]),
                          min_confidence=float(self.settings["min_confidence"])),)

    def set_zones(self, zones: list) -> None:
        super().set_zones(zones)
        keys = {_zone_key(z.name) for z in self._zones}
        for state in (self._zone_since, self._zone_occupied, self._zone_counts):
            for key in [k for k in state if k not in keys]:
                del state[key]
        self._zone_dwell_fired &= keys

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
        floor = float(self.settings["min_confidence"])
        people = [d for d in observation.detections if d.label == "person" and d.confidence >= floor]
        events: list[NodeEvent] = []
        mean_conf = (sum(p.confidence for p in people) / len(people)) if people else None
        events += self._settle_count(len(people), mean_conf, now)
        events += self._zones_update(people, observation, now)
        self.events_emitted += len(events)
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

    def __init__(self, settings: Optional[dict] = None) -> None:
        super().__init__(settings)
        self.engine = RuleEngine()
        self.problems: list[str] = []

    def demands(self) -> tuple[pc.Demand, ...]:
        return (pc.Demand(self.id, pc.OBJECT_DETECTOR, frozenset({"person"}), min_interval_s=0.5,
                          min_confidence=float(self.settings["min_confidence"])),)

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


#: Every companion node this software can run, by id.
COMPANIONS: dict[str, type[MonitoringNode]] = {
    OccupancyNode.spec.id: OccupancyNode,
    ZoneWatchNode.spec.id: ZoneWatchNode,
}


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
