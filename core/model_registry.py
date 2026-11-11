"""The model registry: which models SensorDyme runs, and what is known about them.

core/capabilities.py says what the device CAN watch for; core/monitoring_state.py
says whether each capability is WORKING right now. Neither says whether the
model behind a capability has ever been shown to be right. This module does,
from the hand-edited data file core/model_registry.yaml, and nothing else.

The evaluation ladder (ordered; see EvaluationStatus):

    unevaluated < benchmarked_offline < evaluated_in_domain < validated_on_device

  unevaluated          nothing measured (a model that loads and runs is here)
  benchmarked_offline  measured on public data, possibly on a surrogate backend
                       (e.g. the float ONNX of a .hef)
  evaluated_in_domain  measured on SensorDyme's own consented footage with the
                       evaluation harness
  validated_on_device  measured event-level on the Pi + Hailo hardware package,
                       on the exact pinned artifact

A capability may be shown to a user as "validated" only at validated_on_device.
Runtime facts (the model loaded, the capability is AVAILABLE) never move a model
up the ladder: no function here accepts them as evidence.

Pure: yaml + the standard library + core.roles/core.presets (both pure), so the
backend can import it exactly as the device does. No clock is read unless the
caller passes `today`.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().with_name("model_registry.yaml")


# ---------------------------------------------------------------------- vocabulary
#: The capability vocabulary of core/monitoring_state.py (and core/roles.py's
#: required_capabilities). Anything else is refused as an unknown kind.
CAPABILITY_KINDS = frozenset({
    "camera", "inference", "face", "objects", "hands", "pose", "module",
    "alert_output", "audible_alert",
})
#: The subset a model can provide. The rest are runtime/hardware facts that the
#: hardware package, not this registry, is responsible for.
PERCEPTION_KINDS = frozenset({"face", "objects", "hands", "pose"})

DEPLOYMENTS = ("shipping", "dev_only", "candidate")
COMMERCIAL_STATUSES = ("cleared", "cleared_with_caveats", "restricted", "non_commercial", "unreviewed")
_SHIPPABLE_COMMERCIAL = frozenset({"cleared", "cleared_with_caveats"})
GRANULARITIES = ("image", "frame", "event")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Strong copyleft (GPL/AGPL, not LGPL) and non-commercial terms: never on a
# shipped device. Matched against the code and weights licence strings.
_COPYLEFT = re.compile(r"\bA?GPL", re.IGNORECASE)
_NON_COMMERCIAL = re.compile(r"(\bNC\b|-NC\b|non[- ]?commercial)", re.IGNORECASE)


class EvaluationStatus(str, Enum):  # noqa: UP042 - str() of members is part of the wire format
    """The evaluation ladder. Members compare by rank, lowest first."""

    UNEVALUATED = "unevaluated"
    BENCHMARKED_OFFLINE = "benchmarked_offline"
    EVALUATED_IN_DOMAIN = "evaluated_in_domain"
    VALIDATED_ON_DEVICE = "validated_on_device"

    @property
    def rank(self) -> int:
        return _LADDER.index(self)

    def _cmp(self, other: object) -> int:
        if not isinstance(other, EvaluationStatus):
            return NotImplemented  # type: ignore[return-value]
        return self.rank - other.rank

    def __lt__(self, other: object) -> bool:
        diff = self._cmp(other)
        return diff if diff is NotImplemented else diff < 0

    def __le__(self, other: object) -> bool:
        diff = self._cmp(other)
        return diff if diff is NotImplemented else diff <= 0

    def __gt__(self, other: object) -> bool:
        diff = self._cmp(other)
        return diff if diff is NotImplemented else diff > 0

    def __ge__(self, other: object) -> bool:
        diff = self._cmp(other)
        return diff if diff is NotImplemented else diff >= 0

    __hash__ = str.__hash__


_LADDER: tuple[EvaluationStatus, ...] = (
    EvaluationStatus.UNEVALUATED,
    EvaluationStatus.BENCHMARKED_OFFLINE,
    EvaluationStatus.EVALUATED_IN_DOMAIN,
    EvaluationStatus.VALIDATED_ON_DEVICE,
)
LADDER = _LADDER

#: Evidence kinds and the highest rung each can support. A load check is
#: recordable (it is a fact) but supports nothing above unevaluated.
EVIDENCE_LEVELS: Mapping[str, EvaluationStatus] = {
    "load_check": EvaluationStatus.UNEVALUATED,
    "offline_benchmark": EvaluationStatus.BENCHMARKED_OFFLINE,
    "in_domain_harness": EvaluationStatus.EVALUATED_IN_DOMAIN,
    "on_device_hardware": EvaluationStatus.VALIDATED_ON_DEVICE,
}


def weakest(statuses: Iterable[EvaluationStatus]) -> Optional[EvaluationStatus]:
    """The lowest rung among `statuses`, or None when there are none."""
    statuses = list(statuses)
    return min(statuses, key=lambda s: s.rank) if statuses else None


# ---------------------------------------------------------------------- the card
@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: Optional[str] = None
    url: Optional[str] = None
    size: Optional[int] = None
    format: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ClassMapping:
    """One model output class: its own name, the SensorDyme label, the capability kind."""

    source: str
    label: str
    capability: str
    index: Optional[int] = None


@dataclass(frozen=True)
class PublishedFps:
    value: float
    hardware: str
    conditions: str
    source: str


@dataclass(frozen=True)
class Measurement:
    device: str
    metric: str
    value: float
    evidence: str


@dataclass(frozen=True)
class Compute:
    targets: tuple[str, ...] = ()
    published_fps: tuple[PublishedFps, ...] = ()
    measured: tuple[Measurement, ...] = ()


@dataclass(frozen=True)
class Licence:
    code: str = ""
    weights: str = ""
    training_data: str = ""
    commercial_status: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Evidence:
    """One recorded measurement that supports (or does not support) a status claim."""

    kind: str
    report_id: str
    manifest_id: str
    split: str
    protocol: str
    date: str
    metrics: tuple[tuple[str, Union[float, str]], ...] = ()
    backend: str = ""
    granularity: str = ""
    device: str = ""
    artifact_sha256: Optional[str] = None
    consented: bool = False
    #: False when the measured configuration is not what the device runs (e.g.
    #: a decoder fix not yet deployed). Such evidence supports no claim.
    shipped_config: bool = True
    notes: str = ""

    @property
    def level(self) -> Optional[EvaluationStatus]:
        return EVIDENCE_LEVELS.get(self.kind)

    @property
    def metrics_dict(self) -> dict[str, Union[float, str]]:
        return dict(self.metrics)


@dataclass(frozen=True)
class ShippingException:
    """A recorded, dated reason a shipping model sits below benchmarked_offline."""

    reason: str
    recorded: str
    review_by: str


@dataclass(frozen=True)
class Evaluation:
    status: EvaluationStatus
    evidence: tuple[Evidence, ...] = ()
    shipping_exception: Optional[ShippingException] = None

    @property
    def event_level_on_device(self) -> bool:
        return any(e.kind == "on_device_hardware" and e.granularity == "event" and e.shipped_config
                   for e in self.evidence)


@dataclass(frozen=True)
class ModelCard:
    id: str
    version: str
    family: str
    task: str
    artifacts: tuple[Artifact, ...]
    classes: tuple[ClassMapping, ...]
    provides_capabilities: tuple[str, ...]
    compute: Compute
    licence: Licence
    evaluation: Evaluation
    known_limitations: tuple[str, ...]
    deployment: str
    vocabulary: Optional[str] = None

    @property
    def validated(self) -> bool:
        """The only path to "validated": the top rung, with event-level on-device evidence."""
        return (self.evaluation.status is EvaluationStatus.VALIDATED_ON_DEVICE
                and self.evaluation.event_level_on_device)

    def has_artifact(self, sha256: Optional[str]) -> bool:
        return bool(sha256) and any(a.sha256 == str(sha256).lower() for a in self.artifacts)


@dataclass(frozen=True)
class ModuleUse:
    uses: tuple[str, ...]
    optional: tuple[str, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class ModelRegistry:
    schema_version: int
    registry_version: str
    models: tuple[ModelCard, ...]
    providers: tuple[tuple[str, str], ...]  # (capability kind, model id)
    modules: tuple[tuple[str, ModuleUse], ...]

    def card(self, model_id: str) -> Optional[ModelCard]:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def provider_map(self) -> dict[str, str]:
        return dict(self.providers)

    @property
    def module_map(self) -> dict[str, ModuleUse]:
        return dict(self.modules)

    def provider_of(self, kind: str) -> Optional[ModelCard]:
        model_id = self.provider_map.get(kind)
        return self.card(model_id) if model_id else None


# ---------------------------------------------------------------------- parsing
class RegistryError(ValueError):
    """The registry is malformed or fails validation; `problems` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("model registry invalid:\n  " + "\n  ".join(problems))
        self.problems = problems


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    return str(value).strip()


def _opt_text(value: Any) -> Optional[str]:
    text = _text(value)
    return text or None


def _seq(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _map(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _status(value: Any, where: str, problems: list[str]) -> EvaluationStatus:
    try:
        return EvaluationStatus(_text(value))
    except ValueError:
        problems.append(f"{where}: unknown evaluation status {value!r}; the ladder is "
                        f"{' < '.join(s.value for s in _LADDER)}")
        return EvaluationStatus.UNEVALUATED


def _metrics(value: Any) -> tuple[tuple[str, Union[float, str]], ...]:
    out = []
    for key, val in _map(value).items():
        out.append((str(key), val if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)))
    return tuple(out)


def _evidence(raw: Any) -> Evidence:
    raw = _map(raw)
    sha = _opt_text(raw.get("artifact_sha256"))
    return Evidence(
        kind=_text(raw.get("kind")), report_id=_text(raw.get("report_id")),
        manifest_id=_text(raw.get("manifest_id")), split=_text(raw.get("split")),
        protocol=_text(raw.get("protocol")), date=_text(raw.get("date")),
        metrics=_metrics(raw.get("metrics")), backend=_text(raw.get("backend")),
        granularity=_text(raw.get("granularity")), device=_text(raw.get("device")),
        artifact_sha256=sha.lower() if sha else None,
        consented=raw.get("consented") is True,
        shipped_config=raw.get("shipped_config", True) is not False,
        notes=_text(raw.get("notes")),
    )


def _card(raw: Any, problems: list[str]) -> ModelCard:
    raw = _map(raw)
    model_id = _text(raw.get("id"))
    where = f"model {model_id or '<no id>'}"
    artifacts = []
    for raw_artifact in _seq(raw.get("artifacts")):
        a = _map(raw_artifact)
        size = a.get("size")
        artifacts.append(Artifact(
            path=_text(a.get("path")),
            sha256=_opt_text(a.get("sha256")),
            url=_opt_text(a.get("url")),
            size=size if isinstance(size, int) and not isinstance(size, bool) else (
                -1 if size is not None else None),
            format=_text(a.get("format")), notes=_text(a.get("notes")),
        ))
    classes = []
    for raw_class in _seq(raw.get("classes")):
        c = _map(raw_class)
        index = c.get("index")
        classes.append(ClassMapping(
            source=_text(c.get("source")), label=_text(c.get("label")),
            capability=_text(c.get("capability")),
            index=index if isinstance(index, int) and not isinstance(index, bool) else None,
        ))
    compute = _map(raw.get("compute"))
    lic = _map(raw.get("licence"))
    ev = _map(raw.get("evaluation"))
    exc_raw = ev.get("shipping_exception")
    exception = None
    if exc_raw is not None:
        exc = _map(exc_raw)
        exception = ShippingException(
            reason=_text(exc.get("reason")), recorded=_text(exc.get("recorded")),
            review_by=_text(exc.get("review_by")),
        )
    vocabulary = _map(raw.get("vocabulary"))
    return ModelCard(
        id=model_id,
        version=_text(raw.get("version")),
        family=_text(raw.get("family")),
        task=_text(raw.get("task")),
        artifacts=tuple(artifacts),
        classes=tuple(classes),
        provides_capabilities=tuple(_text(k) for k in _seq(raw.get("provides_capabilities"))),
        compute=Compute(
            targets=tuple(_text(t) for t in _seq(compute.get("targets"))),
            published_fps=tuple(
                PublishedFps(value=float(_map(p).get("value") or 0), hardware=_text(_map(p).get("hardware")),
                             conditions=_text(_map(p).get("conditions")), source=_text(_map(p).get("source")))
                for p in _seq(compute.get("published_fps"))),
            measured=tuple(
                Measurement(device=_text(_map(m).get("device")), metric=_text(_map(m).get("metric")),
                            value=float(_map(m).get("value") or 0), evidence=_text(_map(m).get("evidence")))
                for m in _seq(compute.get("measured"))),
        ),
        licence=Licence(
            code=_text(lic.get("code")), weights=_text(lic.get("weights")),
            training_data=_text(lic.get("training_data")),
            commercial_status=_text(lic.get("commercial_status")), notes=_text(lic.get("notes")),
        ),
        evaluation=Evaluation(
            status=_status(ev.get("status"), where, problems),
            evidence=tuple(_evidence(e) for e in _seq(ev.get("evidence"))),
            shipping_exception=exception,
        ),
        known_limitations=tuple(_text(k) for k in _seq(raw.get("known_limitations"))),
        deployment=_text(raw.get("deployment")),
        vocabulary=_opt_text(vocabulary.get("name")),
    )


def parse_registry(data: Any) -> ModelRegistry:
    """Build the typed registry from the YAML document, without judging it.

    Raises RegistryError only for what cannot be represented at all (not a
    mapping, unknown status string, wrong schema version). Everything a person
    could get wrong in good faith is left to `registry_violations`.
    """
    problems: list[str] = []
    if not isinstance(data, dict):
        raise RegistryError(["the registry document must be a mapping"])
    if data.get("schema_version") != 1:
        problems.append(f"schema_version must be 1, got {data.get('schema_version')!r}")
    models = tuple(_card(m, problems) for m in _seq(data.get("models")))
    providers = tuple((str(k), _text(v)) for k, v in _map(data.get("providers")).items())
    modules = []
    for name, raw_spec in _map(data.get("modules")).items():
        spec = _map(raw_spec)
        modules.append((str(name), ModuleUse(
            uses=tuple(_text(k) for k in _seq(spec.get("uses"))),
            optional=tuple(_text(k) for k in _seq(spec.get("optional"))),
            source=_text(spec.get("source")),
        )))
    if problems:
        raise RegistryError(problems)
    return ModelRegistry(
        schema_version=1, registry_version=_text(data.get("registry_version")),
        models=models, providers=providers, modules=tuple(modules),
    )


# ---------------------------------------------------------------------- validation
def _valid_date(text: str) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _evidence_violations(card: ModelCard, i: int, e: Evidence) -> list[str]:
    where = f"model {card.id}: evidence[{i}] ({e.report_id or 'no report_id'})"
    errors: list[str] = []
    if e.kind not in EVIDENCE_LEVELS:
        errors.append(f"{where}: unknown evidence kind {e.kind!r}; expected one of {sorted(EVIDENCE_LEVELS)}")
        return errors
    for name in ("report_id", "protocol"):
        if not getattr(e, name):
            errors.append(f"{where}: missing {name}")
    if _valid_date(e.date) is None:
        errors.append(f"{where}: date must be YYYY-MM-DD, got {e.date!r}")
    if e.kind == "load_check":
        return errors
    if not e.metrics:
        errors.append(f"{where}: a measurement with no metrics is not evidence")
    if not e.manifest_id or not e.split:
        errors.append(f"{where}: missing manifest_id/split (which frames were measured?)")
    if e.split.lower() == "train":
        errors.append(f"{where}: measured on the training split; that supports no claim")
    if e.granularity and e.granularity not in GRANULARITIES:
        errors.append(f"{where}: granularity must be one of {list(GRANULARITIES)}, got {e.granularity!r}")
    if e.artifact_sha256 is not None and not _SHA256.match(e.artifact_sha256):
        errors.append(f"{where}: artifact_sha256 is not 64 lowercase hex characters")
    if e.kind == "in_domain_harness" and not e.consented:
        errors.append(f"{where}: in-domain evidence must be on consented footage (consented: true)")
    if e.kind == "on_device_hardware":
        if not e.device:
            errors.append(f"{where}: on-device evidence must name the device")
        if e.granularity != "event":
            errors.append(f"{where}: on-device validation must be event-level, got {e.granularity!r}")
        if not card.has_artifact(e.artifact_sha256):
            errors.append(f"{where}: on-device evidence must name the sha256 of one of this card's "
                          f"artifacts (got {e.artifact_sha256!r}); a different binary proves nothing")
    return errors


def card_violations(card: ModelCard, today: Optional[_dt.date] = None) -> list[str]:
    """Every way one card is incomplete or claims more than its evidence."""
    where = f"model {card.id or '<no id>'}"
    errors: list[str] = []
    for name in ("id", "version", "family", "task"):
        if not getattr(card, name):
            errors.append(f"{where}: missing {name}")
    if card.deployment not in DEPLOYMENTS:
        errors.append(f"{where}: deployment must be one of {list(DEPLOYMENTS)}, got {card.deployment!r}")

    # -- capabilities and classes
    if not card.provides_capabilities:
        errors.append(f"{where}: provides_capabilities is empty")
    for kind in card.provides_capabilities:
        if kind not in CAPABILITY_KINDS:
            errors.append(f"{where}: unknown capability kind {kind!r}; known: {sorted(CAPABILITY_KINDS)}")
        elif kind not in PERCEPTION_KINDS:
            errors.append(f"{where}: {kind!r} is a runtime/hardware capability, not one a model provides")
    for c in card.classes:
        if c.capability not in CAPABILITY_KINDS:
            errors.append(f"{where}: class {c.source!r} maps to unknown capability kind {c.capability!r}")
        elif c.capability not in card.provides_capabilities:
            errors.append(f"{where}: class {c.source!r} maps to {c.capability!r}, which the card "
                          f"does not list in provides_capabilities")
        if not c.source or not c.label:
            errors.append(f"{where}: every class needs a source name and a SensorDyme label")

    # -- artifacts
    for a in card.artifacts:
        if not a.path:
            errors.append(f"{where}: an artifact has no path")
        if a.sha256 is not None and not _SHA256.match(a.sha256):
            errors.append(f"{where}: artifact {a.path}: sha256 must be 64 lowercase hex characters, "
                          f"got {a.sha256!r}")
        if a.size is not None and a.size <= 0:
            errors.append(f"{where}: artifact {a.path}: size must be a positive integer of bytes")
    if card.deployment == "shipping":
        if not card.artifacts:
            errors.append(f"{where}: a shipping model must list its artifacts")
        for a in card.artifacts:
            if a.sha256 is None:
                errors.append(f"{where}: shipping artifact {a.path} has no sha256 pin")

    # -- licence
    lic = card.licence
    for name in ("code", "weights", "training_data", "commercial_status"):
        if not getattr(lic, name):
            errors.append(f"{where}: licence.{name} is missing; an unlicensed model cannot be registered")
    if lic.commercial_status and lic.commercial_status not in COMMERCIAL_STATUSES:
        errors.append(f"{where}: licence.commercial_status must be one of {list(COMMERCIAL_STATUSES)}, "
                      f"got {lic.commercial_status!r}")
    if card.deployment == "shipping":
        for name in ("code", "weights"):
            value = getattr(lic, name)
            if _COPYLEFT.search(value):
                errors.append(f"{where}: {name} licence {value!r} is copyleft (GPL/AGPL); "
                              f"it can never be shipping")
            if _NON_COMMERCIAL.search(value):
                errors.append(f"{where}: {name} licence {value!r} is non-commercial; it can never be shipping")
        if lic.commercial_status and lic.commercial_status not in _SHIPPABLE_COMMERCIAL:
            errors.append(f"{where}: commercial_status {lic.commercial_status!r} cannot be shipping "
                          f"(needs {sorted(_SHIPPABLE_COMMERCIAL)})")

    # -- evaluation
    ev = card.evaluation
    for i, e in enumerate(ev.evidence):
        errors += _evidence_violations(card, i, e)
    if ev.status > EvaluationStatus.UNEVALUATED:
        supporting = [e for e in ev.evidence
                      if e.shipped_config and e.level is not None and e.level >= ev.status]
        if not supporting:
            errors.append(f"{where}: status {ev.status.value} claimed without supporting evidence "
                          f"(needs an evidence item of a kind that reaches {ev.status.value}, "
                          f"measured on the shipped configuration)")
    if ev.status is EvaluationStatus.VALIDATED_ON_DEVICE and not ev.event_level_on_device:
        errors.append(f"{where}: validated_on_device needs event-level on_device_hardware evidence")

    exc = ev.shipping_exception
    below_floor = ev.status < EvaluationStatus.BENCHMARKED_OFFLINE
    if card.deployment == "shipping" and below_floor and exc is None:
        errors.append(f"{where}: a shipping model must be at least benchmarked_offline "
                      f"(status is {ev.status.value}; record a shipping_exception with a reason "
                      f"and a review date, or stop shipping it)")
    if exc is not None:
        if card.deployment != "shipping" or not below_floor:
            errors.append(f"{where}: shipping_exception is only for a shipping model below "
                          f"benchmarked_offline; remove it")
        if not exc.reason:
            errors.append(f"{where}: shipping_exception needs a reason")
        recorded, review_by = _valid_date(exc.recorded), _valid_date(exc.review_by)
        if recorded is None or review_by is None:
            errors.append(f"{where}: shipping_exception needs recorded and review_by dates (YYYY-MM-DD)")
        elif today is not None and today > review_by:
            errors.append(f"{where}: shipping_exception expired on {exc.review_by}; evaluate the model "
                          f"or renew the exception with a new reason")
    return errors


def registry_violations(registry: ModelRegistry, today: Optional[_dt.date] = None) -> list[str]:
    """Every reason `registry` must be refused (empty when it is acceptable).

    `today`, when given, also expires shipping exceptions past their review date.
    """
    errors: list[str] = []
    seen: set[str] = set()
    for card in registry.models:
        if card.id in seen:
            errors.append(f"model {card.id}: duplicate id")
        seen.add(card.id)
        errors += card_violations(card, today)
    for kind, model_id in registry.providers:
        if kind not in PERCEPTION_KINDS:
            errors.append(f"providers: {kind!r} is not a model-provided capability kind")
            continue
        card = registry.card(model_id)
        if card is None:
            errors.append(f"providers: {kind} -> {model_id!r}, which is not registered")
        elif card.deployment != "shipping":
            errors.append(f"providers: {kind} -> {model_id} is {card.deployment}; only a shipping "
                          f"model may serve a shipped device")
        elif kind not in card.provides_capabilities:
            errors.append(f"providers: {kind} -> {model_id}, which does not provide {kind}")
    providers = registry.provider_map
    for name, use in registry.modules:
        for kind in (*use.uses, *use.optional):
            if kind not in CAPABILITY_KINDS:
                errors.append(f"modules.{name}: unknown capability kind {kind!r}")
            elif kind in PERCEPTION_KINDS and kind not in providers:
                errors.append(f"modules.{name}: uses {kind}, but no provider is registered for it")
    return errors


def load_registry(path: Union[str, Path, None] = None, *, validate: bool = True,
                  today: Optional[_dt.date] = None) -> ModelRegistry:
    """Load (and by default validate) the registry file. Raises RegistryError."""
    path = Path(path) if path is not None else DEFAULT_PATH
    with path.open() as f:
        registry = parse_registry(yaml.safe_load(f))
    if validate:
        problems = registry_violations(registry, today)
        if problems:
            raise RegistryError(problems)
    return registry


# ---------------------------------------------------------------------- derivation
#: The labels the app shows. Liveness (core/monitoring_state.py) and evidence
#: (this module) combine here, and liveness can only ever LOWER what is shown.
DISPLAY_VALIDATED = "validated"
DISPLAY_AVAILABLE = "available"  # running, and not model-backed (camera, alert output...)
DISPLAY_AVAILABLE_NOT_VALIDATED = "available_not_validated"
DISPLAY_STARTING = "starting"
DISPLAY_IMPAIRED = "impaired"
DISPLAY_NOT_REPORTED = "not_reported"


@dataclass(frozen=True)
class CapabilityLink:
    """One capability a subject needs, the model behind it, and how far that model got."""

    capability: str
    model_backed: bool
    provided_by: Optional[str] = None
    model_version: Optional[str] = None
    evaluation_status: Optional[EvaluationStatus] = None
    validated: bool = False
    shipping_exception: bool = False
    optional: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "model_backed": self.model_backed,
            "provided_by": self.provided_by,
            "model_version": self.model_version,
            "evaluation_status": None if self.evaluation_status is None else self.evaluation_status.value,
            "validated": self.validated,
            "shipping_exception": self.shipping_exception,
            "optional": self.optional,
            "note": self.note,
        }


@dataclass(frozen=True)
class ChainView:
    """What a preset, role, or custom policy needs, and the weakest link in it."""

    subject: str
    links: tuple[CapabilityLink, ...]

    @property
    def weakest(self) -> Optional[EvaluationStatus]:
        """The lowest rung over the required model-backed links (None: none are model-backed)."""
        return weakest(link.evaluation_status for link in self.links
                       if link.model_backed and not link.optional and link.evaluation_status is not None)

    @property
    def validated(self) -> bool:
        """True only when every required model-backed link is validated (and there is one)."""
        required = [link for link in self.links if link.model_backed and not link.optional]
        return bool(required) and all(link.validated for link in required)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({link.provided_by for link in self.links if link.provided_by}))

    def to_dict(self) -> dict:
        w = self.weakest
        return {
            "subject": self.subject,
            "weakest_evaluation_status": None if w is None else w.value,
            "validated": self.validated,
            "models": list(self.models),
            "capabilities": [link.to_dict() for link in self.links],
        }


def capability_link(registry: ModelRegistry, kind: str, *, optional: bool = False,
                    running_sha256: Optional[str] = None) -> CapabilityLink:
    """The link for one capability kind on a shipped device.

    `running_sha256`, when the device reports the hash of what it actually
    loaded, must match the provider's pinned artifact; a mismatch is reported as
    an unregistered, unevaluated model rather than borrowing the card's status.
    """
    if kind not in PERCEPTION_KINDS:
        note = ("not model-backed; covered by the hardware package" if kind in CAPABILITY_KINDS
                else "unknown capability kind")
        return CapabilityLink(kind, model_backed=False, optional=optional, note=note)
    card = registry.provider_of(kind)
    if card is None:
        return CapabilityLink(kind, model_backed=True, evaluation_status=EvaluationStatus.UNEVALUATED,
                              optional=optional, note="no registered provider")
    if running_sha256 is not None and not card.has_artifact(running_sha256):
        return CapabilityLink(
            kind, model_backed=True, provided_by="unregistered", optional=optional,
            evaluation_status=EvaluationStatus.UNEVALUATED,
            note=f"the device runs sha256 {str(running_sha256)[:12]}, not an artifact of {card.id}",
        )
    return CapabilityLink(
        kind, model_backed=True, provided_by=card.id, model_version=card.version,
        evaluation_status=card.evaluation.status, validated=card.validated,
        shipping_exception=card.evaluation.shipping_exception is not None, optional=optional,
    )


def chain_for(registry: ModelRegistry, subject: str, required: Iterable[str],
              optional: Iterable[str] = (), *,
              running_sha256: Optional[Mapping[str, str]] = None) -> ChainView:
    """The chain for an arbitrary set of capability kinds (presets, roles, custom rules)."""
    running = dict(running_sha256 or {})
    required = sorted(set(required))
    optional = sorted(set(optional) - set(required))
    links = [capability_link(registry, k, running_sha256=running.get(k)) for k in required]
    links += [capability_link(registry, k, optional=True, running_sha256=running.get(k)) for k in optional]
    return ChainView(subject, tuple(links))


def preset_view(registry: ModelRegistry, preset: str, presets: Optional[Mapping] = None, *,
                running_sha256: Optional[Mapping[str, str]] = None) -> ChainView:
    """Preset -> module -> capabilities -> models, with the weakest rung.

    Accepts "preset" or "preset:variant" (variants only retune thresholds, so
    they share the preset's chain). Raises KeyError for an unknown preset or a
    module the registry does not describe: an undescribed module must not read
    as "needs nothing".
    """
    if presets is None:
        from core.presets import load_presets

        presets = load_presets()
    base = preset.partition(":")[0]
    module = (presets[base] or {}).get("module")
    use = registry.module_map[module]
    return chain_for(registry, f"preset:{base}", ("camera", "inference", *use.uses), use.optional,
                     running_sha256=running_sha256)


def preset_views(registry: ModelRegistry, presets: Optional[Mapping] = None) -> dict[str, ChainView]:
    if presets is None:
        from core.presets import load_presets

        presets = load_presets()
    return {name: preset_view(registry, name, presets) for name in sorted(presets)}


def role_view(registry: ModelRegistry, role: Any) -> ChainView:
    """A role's required capabilities (core/roles.py) and the models behind them."""
    return chain_for(registry, f"role:{role.name}", role.required_capabilities)


def role_views(registry: ModelRegistry, roles: Optional[Mapping] = None) -> dict[str, ChainView]:
    if roles is None:
        from core.roles import ROLES

        roles = ROLES
    return {name: role_view(registry, role) for name, role in sorted(roles.items())}


def policy_view(registry: ModelRegistry, policy_id: str, kinds: Iterable[str]) -> ChainView:
    """A custom monitoring policy (an AI-configured rule). `kinds` are the
    capability kinds its labels come from -- the runtime already derives them
    (services/runtime.py `_label_needs`, via core/capabilities.py)."""
    return chain_for(registry, f"policy:{policy_id}", ("camera", "inference", *kinds))


def display_state(live_state: Optional[str], link: CapabilityLink) -> str:
    """What the app shows for one capability: liveness first, then evidence.

    `live_state` is core/monitoring_state.py's AVAILABLE/IMPAIRED/STARTING (or
    None when the device has not reported). A capability that is running is
    shown as "validated" only when the registry says so; running is never a
    substitute for evidence.
    """
    if live_state is None:
        return DISPLAY_NOT_REPORTED
    if live_state == "impaired":
        return DISPLAY_IMPAIRED
    if live_state == "starting":
        return DISPLAY_STARTING
    if live_state != "available":
        return DISPLAY_NOT_REPORTED
    if not link.model_backed:
        return DISPLAY_AVAILABLE
    return DISPLAY_VALIDATED if link.validated else DISPLAY_AVAILABLE_NOT_VALIDATED


def capability_evidence(registry: ModelRegistry, *, preset: Optional[str], role: Any = None,
                        live: Optional[Mapping[str, Optional[str]]] = None,
                        running_sha256: Optional[Mapping[str, str]] = None,
                        presets: Optional[Mapping] = None) -> dict:
    """The proposed per-device `capability_evidence` block (docs/MODEL_REGISTRY.md).

    Pure: the backend passes what the device reported (active preset, role,
    capability liveness, the detector's sha256) and gets back one entry per
    capability with its provider, evaluation status and display state.
    """
    view: Optional[ChainView] = None
    if preset:
        try:
            view = preset_view(registry, preset, presets, running_sha256=running_sha256)
        except KeyError:
            view = None
    links = {link.capability: link for link in (view.links if view else ())}
    if role is not None:
        for link in chain_for(registry, "role", role.required_capabilities,
                              running_sha256=running_sha256).links:
            links.setdefault(link.capability, link)
    live = dict(live or {})
    entries = []
    for kind in sorted(links):
        link = links[kind]
        entry = link.to_dict()
        entry["live_state"] = live.get(kind)
        entry["display"] = display_state(live.get(kind), link)
        entries.append(entry)
    combined = ChainView("device", tuple(links[k] for k in sorted(links)))
    w = combined.weakest
    return {
        "registry_version": registry.registry_version,
        "preset_known": view is not None,
        "weakest_evaluation_status": None if w is None else w.value,
        "validated": combined.validated,
        "capabilities": entries,
    }


# ---------------------------------------------------------------------- re-evaluation
@dataclass(frozen=True)
class Impact:
    model_id: str
    change: str  # added | removed | changed | provider_changed
    presets: tuple[str, ...]
    roles: tuple[str, ...]


def presets_using(registry: ModelRegistry, model_id: str, presets: Optional[Mapping] = None,
                  *, include_optional: bool = True) -> tuple[str, ...]:
    """Presets whose chain runs through `model_id` (via the provider map)."""
    if presets is None:
        from core.presets import load_presets

        presets = load_presets()
    kinds = {k for k, m in registry.providers if m == model_id}
    out = []
    for name, spec in sorted(presets.items()):
        use = registry.module_map.get((spec or {}).get("module"))
        if use is None:
            continue
        needed = set(use.uses) | (set(use.optional) if include_optional else set())
        if needed & kinds:
            out.append(name)
    return tuple(out)


def roles_using(registry: ModelRegistry, model_id: str, roles: Optional[Mapping] = None) -> tuple[str, ...]:
    if roles is None:
        from core.roles import ROLES

        roles = ROLES
    kinds = {k for k, m in registry.providers if m == model_id}
    return tuple(name for name, role in sorted(roles.items()) if set(role.required_capabilities) & kinds)


def reevaluation_impact(old: ModelRegistry, new: ModelRegistry, presets: Optional[Mapping] = None,
                        roles: Optional[Mapping] = None) -> list[Impact]:
    """When cards change, which presets and roles must be re-evaluated.

    Any change to a card (artifact, version, classes, licence, evidence) counts:
    the registry cannot tell a cosmetic edit from a new binary, and a
    re-evaluation that turns out unnecessary is cheaper than one that was
    skipped. A provider swap counts for both the outgoing and incoming model.
    """
    old_cards = {m.id: m for m in old.models}
    new_cards = {m.id: m for m in new.models}
    changes: dict[str, str] = {}
    for model_id in sorted(set(old_cards) | set(new_cards)):
        if model_id not in old_cards:
            changes[model_id] = "added"
        elif model_id not in new_cards:
            changes[model_id] = "removed"
        elif old_cards[model_id] != new_cards[model_id]:
            changes[model_id] = "changed"
    old_p, new_p = old.provider_map, new.provider_map
    for kind in sorted(set(old_p) | set(new_p)):
        if old_p.get(kind) != new_p.get(kind):
            for model_id in (old_p.get(kind), new_p.get(kind)):
                if model_id:
                    changes.setdefault(model_id, "provider_changed")
    impacts = []
    for model_id, change in sorted(changes.items()):
        affected_p = set(presets_using(old, model_id, presets)) | set(presets_using(new, model_id, presets))
        affected_r = set(roles_using(old, model_id, roles)) | set(roles_using(new, model_id, roles))
        impacts.append(Impact(model_id, change, tuple(sorted(affected_p)), tuple(sorted(affected_r))))
    return impacts


def as_dict(registry: ModelRegistry) -> dict:
    """A read-only summary (one entry per model) for logs, docs, and a future health block."""
    return {
        "registry_version": registry.registry_version,
        "models": [
            {
                "id": m.id, "version": m.version, "deployment": m.deployment,
                "provides": list(m.provides_capabilities),
                "evaluation_status": m.evaluation.status.value,
                "validated": m.validated,
                "shipping_exception": m.evaluation.shipping_exception is not None,
                "commercial_status": m.licence.commercial_status,
            }
            for m in registry.models
        ],
        "providers": dict(registry.providers),
    }


__all__ = [
    "CAPABILITY_KINDS", "DEFAULT_PATH", "EVIDENCE_LEVELS", "LADDER", "PERCEPTION_KINDS",
    "CapabilityLink", "ChainView", "EvaluationStatus", "Evidence", "Impact", "ModelCard",
    "ModelRegistry", "RegistryError", "as_dict", "capability_evidence", "capability_link",
    "card_violations", "chain_for", "display_state", "load_registry", "parse_registry",
    "policy_view", "preset_view", "preset_views", "presets_using", "reevaluation_impact",
    "registry_violations", "role_view", "role_views", "roles_using", "weakest",
]
