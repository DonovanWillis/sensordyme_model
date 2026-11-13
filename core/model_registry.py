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

Product capabilities and their tiers (M6.5): the ladder above is per MODEL. What
the product claims about a CAPABILITY (phone detection, drowsiness...) is a
second, finer ladder, bound to evidence records that each name the exact model
hash, decoder, test, dataset manifest and hardware target:

    exists_in_code < synthetic_tests < image_level_benchmark < video_evidence
                   < device_verified < production_permitted

A tier is claimed in core/model_registry.yaml and refused unless a `current`
evidence binding for the same capability, model and decoder reaches it.
device_verified also needs the deployed hardware target and a pinned artifact
hash of the model card; production_permitted also needs device_verified, a
shipping card that is validated_on_device, and a recorded approval. Experiment
and pending bindings are kept for the record and support nothing.

Pure: yaml + the standard library + core.roles/core.presets (both pure), so the
backend can import it exactly as the device does. No clock is read unless the
caller passes `today`.

Bundled model files (M8.5, M84-5): `verify_bundled_artifacts()` hashes every
model file the registry pins inside an installed package (the MediaPipe
.tflite files in the mediapipe wheel) against its pin. The staged install runs
it before a release may be switched to (`python -m core.model_registry
verify-artifacts`), and main.py runs `enforce_bundled_artifacts()` at every boot,
before any module builds a graph: a file that does not match keeps its MediaPipe
solution from loading at all (the only point that imports mediapipe) and the
runtime reports that capability impaired with the reason.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
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

    # Each comparison returns NotImplemented (not False) for anything that is not
    # an EvaluationStatus, so Python falls back to the other operand. Spelled out
    # per operator: NotImplemented may be returned from a `-> bool` dunder, but
    # threading it through an `-> int` helper made every one of these return
    # `int | bool`.
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EvaluationStatus):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, EvaluationStatus):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, EvaluationStatus):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, EvaluationStatus):
            return NotImplemented
        return self.rank >= other.rank

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


# ---------------------------------------------------------------------- product capability tiers
#: The product-capability ladder, lowest first (module docstring).
TIERS = ("exists_in_code", "synthetic_tests", "image_level_benchmark", "video_evidence",
         "device_verified", "production_permitted")
#: Where a measurement ran. Only the deployed target can make a capability device_verified.
HARDWARE_TARGETS = ("sim", "dev_cpu", "oryn_gpu_float", "pi5_cpu", "hailo8l_int8")
#: "n/a" is for models without a box decoder (MediaPipe graphs).
DECODERS = ("legacy", "reference", "n/a")
#: current: supports claims. experiment_only: a research result (pilot checkpoints, a
#: candidate decoder) kept for the record. superseded: replaced by a newer record.
#: pending: a reserved slot (e.g. a run still training) whose values are not known yet.
BINDING_STATUSES = ("current", "experiment_only", "superseded", "pending")
#: Data the product may never cite (competition-only licences; Mission 4 dataset audit).
_FORBIDDEN_DATA = re.compile(r"state[ _-]?farm|kaggle|competition", re.IGNORECASE)


def tier_rank(tier: str) -> int:
    """Position on the ladder; -1 for a value that is not a tier."""
    return TIERS.index(tier) if tier in TIERS else -1


@dataclass(frozen=True)
class Approval:
    by: str
    date: str
    reference: str


@dataclass(frozen=True)
class ProductCapability:
    """One capability the product claims, the model behind it, and the tier it claims."""

    id: str
    model: str
    kind: str
    code: str
    deployed_target: str
    decoder: str
    claimed_tier: str
    approval: Optional[Approval] = None
    notes: str = ""


@dataclass(frozen=True)
class EvidenceBinding:
    """One piece of capability evidence, bound to exactly what was measured."""

    id: str
    capability: str
    tier: str
    model: str
    model_sha256: Optional[str]
    decoder: str
    test_id: str
    dataset_manifest: str
    hardware_target: str
    status: str
    date: str
    metrics: tuple[tuple[str, Union[float, str]], ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class InferenceComponent:
    """One shared perception component (core/perception.py) and the model behind it."""

    id: str  # object_detector | face_landmarks | hand_landmarks | pose
    capability: str  # the monitoring capability kind it provides
    provider: str  # model id
    product_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class MonitoringNodeEntry:
    """One monitoring node (core/nodes.py): what it consumes and the tier it claims.

    A node can never claim more than the weakest product capability it depends
    on (`node_tier`): a mode is not validated because its tests pass, and a
    simulated event is not evidence of real-world accuracy.
    """

    id: str
    kind: str  # preset | rules | companion
    components: tuple[str, ...]
    capabilities: tuple[str, ...]  # product capability ids
    claimed_tier: str
    preset: Optional[str] = None
    experimental: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ModelRegistry:
    schema_version: int
    registry_version: str
    models: tuple[ModelCard, ...]
    providers: tuple[tuple[str, str], ...]  # (capability kind, model id)
    modules: tuple[tuple[str, ModuleUse], ...]
    capabilities: tuple[ProductCapability, ...] = ()
    bindings: tuple[EvidenceBinding, ...] = ()
    components: tuple[InferenceComponent, ...] = ()
    nodes: tuple[MonitoringNodeEntry, ...] = ()

    def component(self, component_id: str) -> Optional[InferenceComponent]:
        return next((c for c in self.components if c.id == component_id), None)

    def node(self, node_id: str) -> Optional[MonitoringNodeEntry]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def product_capability(self, capability_id: str) -> Optional[ProductCapability]:
        return next((c for c in self.capabilities if c.id == capability_id), None)

    def bindings_for(self, capability_id: str) -> tuple[EvidenceBinding, ...]:
        return tuple(b for b in self.bindings if b.capability == capability_id)

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
    capabilities = []
    for cap_id, raw_cap in _map(data.get("product_capabilities")).items():
        cap = _map(raw_cap)
        approval = None
        if cap.get("approval") is not None:
            a = _map(cap.get("approval"))
            approval = Approval(by=_text(a.get("by")), date=_text(a.get("date")), reference=_text(a.get("reference")))
        capabilities.append(ProductCapability(
            id=str(cap_id), model=_text(cap.get("model")), kind=_text(cap.get("kind")),
            code=_text(cap.get("code")), deployed_target=_text(cap.get("deployed_target")),
            decoder=_text(cap.get("decoder")), claimed_tier=_text(cap.get("claimed_tier")),
            approval=approval, notes=_text(cap.get("notes")),
        ))
    bindings = []
    for raw_binding in _seq(data.get("evidence_bindings")):
        b = _map(raw_binding)
        sha = _opt_text(b.get("model_sha256"))
        bindings.append(EvidenceBinding(
            id=_text(b.get("id")), capability=_text(b.get("capability")), tier=_text(b.get("tier")),
            model=_text(b.get("model")), model_sha256=sha.lower() if sha else None,
            decoder=_text(b.get("decoder")), test_id=_text(b.get("test_id")),
            dataset_manifest=_text(b.get("dataset_manifest")), hardware_target=_text(b.get("hardware_target")),
            status=_text(b.get("status")), date=_text(b.get("date")), metrics=_metrics(b.get("metrics")),
            notes=_text(b.get("notes")),
        ))
    components = []
    for comp_id, raw_comp in _map(data.get("inference_components")).items():
        comp = _map(raw_comp)
        components.append(InferenceComponent(
            id=str(comp_id), capability=_text(comp.get("capability")), provider=_text(comp.get("provider")),
            product_capabilities=tuple(_text(c) for c in _seq(comp.get("product_capabilities"))),
        ))
    nodes = []
    for node_id, raw_node in _map(data.get("monitoring_nodes")).items():
        node = _map(raw_node)
        nodes.append(MonitoringNodeEntry(
            id=str(node_id), kind=_text(node.get("kind")),
            components=tuple(_text(c) for c in _seq(node.get("components"))),
            capabilities=tuple(_text(c) for c in _seq(
                node.get("product_capabilities") or node.get("requires") or node.get("capabilities"))),
            claimed_tier=_text(node.get("claimed_tier")), preset=_opt_text(node.get("preset")),
            experimental=bool(node.get("experimental", False)), notes=_text(node.get("notes")),
        ))
    if problems:
        raise RegistryError(problems)
    return ModelRegistry(
        schema_version=1, registry_version=_text(data.get("registry_version")),
        models=models, providers=providers, modules=tuple(modules),
        capabilities=tuple(capabilities), bindings=tuple(bindings),
        components=tuple(components), nodes=tuple(nodes),
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


def binding_violations(registry: ModelRegistry, b: EvidenceBinding) -> list[str]:
    """Every way one evidence binding is malformed (independent of any claim)."""
    where = f"evidence_bindings.{b.id or '<no id>'}"
    errors: list[str] = []
    if not b.id:
        errors.append(f"{where}: missing id")
    if b.status not in BINDING_STATUSES:
        errors.append(f"{where}: status must be one of {list(BINDING_STATUSES)}, got {b.status!r}")
    if registry.product_capability(b.capability) is None:
        errors.append(f"{where}: capability {b.capability!r} is not in product_capabilities")
    if tier_rank(b.tier) < 0:
        errors.append(f"{where}: tier must be one of {list(TIERS)}, got {b.tier!r}")
    elif b.tier == "production_permitted":
        errors.append(f"{where}: production_permitted is an approval on the capability, not a measurement")
    card = registry.card(b.model)
    if card is None:
        errors.append(f"{where}: model {b.model!r} is not registered")
    for value in (b.test_id, b.dataset_manifest, b.notes, *(str(v) for _, v in b.metrics)):
        if _FORBIDDEN_DATA.search(value or ""):
            errors.append(f"{where}: cites competition-only data ({value!r}); it can support nothing "
                          f"and must not be recorded as evidence")
    if b.status == "pending":
        return errors  # a reserved slot: the values arrive with the run
    if b.decoder not in DECODERS:
        errors.append(f"{where}: decoder must be one of {list(DECODERS)}, got {b.decoder!r}")
    if b.hardware_target not in HARDWARE_TARGETS:
        errors.append(f"{where}: hardware_target must be one of {list(HARDWARE_TARGETS)}, "
                      f"got {b.hardware_target!r}")
    for name in ("test_id", "dataset_manifest"):
        if not getattr(b, name):
            errors.append(f"{where}: missing {name}")
    if _valid_date(b.date) is None:
        errors.append(f"{where}: date must be YYYY-MM-DD, got {b.date!r}")
    if b.model_sha256 is not None and not _SHA256.match(b.model_sha256):
        errors.append(f"{where}: model_sha256 is not 64 lowercase hex characters")
    if tier_rank(b.tier) >= tier_rank("image_level_benchmark"):
        if b.model_sha256 is None:
            errors.append(f"{where}: a {b.tier} measurement must name the sha256 of the weights it measured")
        if b.dataset_manifest.startswith("synthetic:"):
            errors.append(f"{where}: {b.tier} needs real data; {b.dataset_manifest!r} is synthetic")
        if not b.metrics:
            errors.append(f"{where}: a {b.tier} measurement with no metrics is not evidence")
    if card is not None and card.deployment == "candidate" and b.status == "current":
        errors.append(f"{where}: {card.id} is a candidate model; its evidence is experiment_only until "
                      f"it is promoted")
    return errors


def _supports(registry: ModelRegistry, cap: ProductCapability, b: EvidenceBinding, tier: str) -> bool:
    """Whether `b` supports claiming `tier` for `cap`."""
    if b.capability != cap.id or b.status != "current" or b.model != cap.model or b.decoder != cap.decoder:
        return False
    if tier_rank(b.tier) < tier_rank(tier):
        return False
    if tier_rank(tier) >= tier_rank("device_verified"):
        card = registry.card(cap.model)
        return (b.hardware_target == cap.deployed_target and card is not None
                and card.has_artifact(b.model_sha256))
    return True


def capability_violations(registry: ModelRegistry, cap: ProductCapability) -> list[str]:
    """Every way one product capability claims more than its evidence."""
    where = f"product_capabilities.{cap.id}"
    errors: list[str] = []
    card = registry.card(cap.model)
    if card is None:
        errors.append(f"{where}: model {cap.model!r} is not registered")
    elif cap.kind not in card.provides_capabilities:
        errors.append(f"{where}: {card.id} does not provide {cap.kind!r}")
    if not cap.code:
        errors.append(f"{where}: missing code (where the capability is implemented)")
    if cap.deployed_target not in HARDWARE_TARGETS:
        errors.append(f"{where}: deployed_target must be one of {list(HARDWARE_TARGETS)}, "
                      f"got {cap.deployed_target!r}")
    if cap.decoder not in DECODERS:
        errors.append(f"{where}: decoder must be one of {list(DECODERS)}, got {cap.decoder!r}")
    rank = tier_rank(cap.claimed_tier)
    if rank < 0:
        errors.append(f"{where}: claimed_tier must be one of {list(TIERS)}, got {cap.claimed_tier!r}")
        return errors
    if rank > 0:
        need = "device_verified" if cap.claimed_tier == "production_permitted" else cap.claimed_tier
        if not any(_supports(registry, cap, b, need) for b in registry.bindings_for(cap.id)):
            extra = (f" on {cap.deployed_target} with a pinned artifact hash of {cap.model}"
                     if tier_rank(need) >= tier_rank("device_verified") else "")
            errors.append(f"{where}: claims {cap.claimed_tier} without a current evidence binding for "
                          f"model {cap.model}, decoder {cap.decoder}, reaching {need}{extra}")
    if cap.decoder == "reference" and rank >= tier_rank("device_verified") and not any(
            b.status == "current" and b.decoder == "reference" and b.hardware_target == "hailo8l_int8"
            and tier_rank(b.tier) >= tier_rank("device_verified") for b in registry.bindings_for(cap.id)):
        errors.append(f"{where}: the reference decoder is unvalidated on the INT8 .hef "
                      f"(docs/DECODER_CANDIDATE.md); it cannot be {cap.claimed_tier}")
    if cap.claimed_tier == "production_permitted":
        if card is not None and (card.deployment != "shipping" or not card.validated):
            errors.append(f"{where}: production_permitted needs a shipping model that is validated_on_device")
        a = cap.approval
        if a is None or not a.by or not a.reference or _valid_date(a.date) is None:
            errors.append(f"{where}: production_permitted needs a recorded approval (by, date, reference)")
    elif cap.approval is not None:
        errors.append(f"{where}: an approval is only recorded with a production_permitted claim")
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
    seen_caps: set[str] = set()
    for cap in registry.capabilities:
        if cap.id in seen_caps:
            errors.append(f"product_capabilities.{cap.id}: duplicate id")
        seen_caps.add(cap.id)
        errors += capability_violations(registry, cap)
    seen_bindings: set[str] = set()
    for b in registry.bindings:
        if b.id in seen_bindings:
            errors.append(f"evidence_bindings.{b.id}: duplicate id")
        seen_bindings.add(b.id)
        errors += binding_violations(registry, b)
    errors += component_violations(registry)
    errors += node_violations(registry)
    return errors


#: The shared perception components (core/perception.py COMPONENTS) and what each provides.
COMPONENT_CAPABILITY = {"object_detector": "objects", "face_landmarks": "face",
                        "hand_landmarks": "hands", "pose": "pose"}
NODE_KINDS = ("preset", "rules", "companion")


def component_violations(registry: ModelRegistry) -> list[str]:
    errors: list[str] = []
    for comp in registry.components:
        where = f"inference_components.{comp.id}"
        if comp.id not in COMPONENT_CAPABILITY:
            errors.append(f"{where}: not a perception component ({sorted(COMPONENT_CAPABILITY)})")
            continue
        if comp.capability != COMPONENT_CAPABILITY[comp.id]:
            errors.append(f"{where}: provides {COMPONENT_CAPABILITY[comp.id]}, not {comp.capability!r}")
        if registry.provider_map.get(comp.capability) != comp.provider:
            errors.append(f"{where}: provider {comp.provider!r} is not the registered {comp.capability} provider")
        for cap_id in comp.product_capabilities:
            cap = registry.product_capability(cap_id)
            if cap is None:
                errors.append(f"{where}: product capability {cap_id!r} is not declared")
            elif cap.model != comp.provider:
                errors.append(f"{where}: {cap_id} is served by {cap.model}, not by this component's model")
    return errors


def node_violations(registry: ModelRegistry) -> list[str]:
    """A node naming unknown parts, or claiming more than its capabilities' evidence."""
    errors: list[str] = []
    seen: set[str] = set()
    for node in registry.nodes:
        where = f"monitoring_nodes.{node.id}"
        if node.id in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(node.id)
        if node.kind not in NODE_KINDS:
            errors.append(f"{where}: kind must be one of {NODE_KINDS}")
        if tier_rank(node.claimed_tier) < 0:
            errors.append(f"{where}: claimed_tier {node.claimed_tier!r} is not a tier")
        for comp_id in node.components:
            if registry.component(comp_id) is None:
                errors.append(f"{where}: component {comp_id!r} is not declared in inference_components")
        if not node.capabilities:
            errors.append(f"{where}: a node must name the product capabilities it depends on")
        for cap_id in node.capabilities:
            cap = registry.product_capability(cap_id)
            if cap is None:
                errors.append(f"{where}: product capability {cap_id!r} is not declared")
            elif tier_rank(node.claimed_tier) > tier_rank(cap.claimed_tier):
                errors.append(f"{where}: claims {node.claimed_tier}, above {cap_id}'s {cap.claimed_tier}: "
                              f"a node can never claim more than the evidence behind what it depends on")
        if node.experimental and tier_rank(node.claimed_tier) > tier_rank("synthetic_tests"):
            errors.append(f"{where}: an experimental node claims nothing above synthetic_tests")
        if node.kind == "preset" and not node.preset:
            errors.append(f"{where}: a preset node names its preset")
    return errors


def node_tier(registry: ModelRegistry, node_id: str) -> str:
    """The tier a node may report: its claim, capped by its weakest capability.

    Unknown nodes are `exists_in_code`. Capabilities that do not exist count as
    `exists_in_code`, so a broken entry lowers what is reported, never raises it.
    """
    node = registry.node(node_id)
    if node is None:
        return TIERS[0]
    ranks = [tier_rank(node.claimed_tier)]
    for cap_id in node.capabilities:
        cap = registry.product_capability(cap_id)
        ranks.append(tier_rank(cap.claimed_tier) if cap is not None else 0)
    return TIERS[max(0, min(ranks))]


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


def capability_tiers(registry: ModelRegistry) -> dict:
    """The product-capability tiers as the API serves them (GET /v1/capability-tiers).

    `production_validated` is true only at production_permitted, and
    `device_verified` only at device_verified or above: the app reads these two
    booleans, never the tier string alone, to decide what to call validated.
    """
    entries = []
    for cap in registry.capabilities:
        card = registry.card(cap.model)
        bindings = registry.bindings_for(cap.id)
        rank = tier_rank(cap.claimed_tier)
        entries.append({
            "capability": cap.id,
            "kind": cap.kind,
            "model": cap.model,
            "model_deployment": card.deployment if card else None,
            "decoder": cap.decoder,
            "deployed_target": cap.deployed_target,
            "tier": cap.claimed_tier,
            "device_verified": rank >= tier_rank("device_verified"),
            "production_validated": cap.claimed_tier == "production_permitted",
            "evidence": [b.id for b in bindings if _supports(registry, cap, b, cap.claimed_tier)],
            "experiment_evidence": [b.id for b in bindings if b.status in ("experiment_only", "pending")],
        })
    nodes = [{
        "node": node.id, "kind": node.kind, "preset": node.preset,
        "components": list(node.components), "product_capabilities": list(node.capabilities),
        "claimed_tier": node.claimed_tier,
        # What the node may report: never above its weakest capability.
        "tier": node_tier(registry, node.id),
        "experimental": node.experimental,
        "device_verified": tier_rank(node_tier(registry, node.id)) >= tier_rank("device_verified"),
        "production_validated": node_tier(registry, node.id) == "production_permitted",
    } for node in registry.nodes]
    return {"registry_version": registry.registry_version, "tiers": list(TIERS), "capabilities": entries,
            "nodes": nodes}


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


# ---------------------------------------------------------------------- bundled artifacts (M8.5, M84-5)
#: The MediaPipe solution class that loads each capability's model files, and the
#: fields its result carries (all None = "nothing found", which every module
#: already handles). A capability whose files fail the check gets a stand-in that
#: never loads a model.
MEDIAPIPE_SOLUTIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "face": ("face_mesh", "FaceMesh", ("multi_face_landmarks",)),
    "hands": ("hands", "Hands", ("multi_hand_landmarks", "multi_hand_world_landmarks", "multi_handedness")),
    "pose": ("pose", "Pose", ("pose_landmarks", "pose_world_landmarks", "segmentation_mask")),
}
ARTIFACT_OK, ARTIFACT_MISMATCH, ARTIFACT_MISSING, ARTIFACT_UNREADABLE, ARTIFACT_ABSENT = (
    "ok", "mismatch", "missing", "unreadable", "package_absent")


@dataclass(frozen=True)
class ArtifactCheck:
    """One pinned model file inside an installed package, hashed where it is loaded from."""

    model: str
    path: str
    expected: str
    status: str
    found: Optional[str] = None
    location: Optional[str] = None
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fault(self) -> bool:
        return self.status in (ARTIFACT_MISMATCH, ARTIFACT_MISSING, ARTIFACT_UNREADABLE)

    def reason(self) -> str:
        name = self.path.rsplit("/", 1)[-1]
        if self.status == ARTIFACT_MISMATCH:
            return (f"model integrity: {name} sha256 {str(self.found)[:12]}... is not the pinned "
                    f"{self.expected[:12]}... ({self.model}); refused")
        if self.status == ARTIFACT_MISSING:
            return f"model integrity: {name} is missing from the installed package ({self.model}); refused"
        if self.status == ARTIFACT_UNREADABLE:
            return f"model integrity: {name} could not be read ({self.model}); refused"
        if self.status == ARTIFACT_ABSENT:
            return f"{self.path.split('/', 1)[0]} is not installed"
        return "verified"

    def to_dict(self) -> dict:
        return {"model": self.model, "path": self.path, "expected": self.expected, "found": self.found,
                "status": self.status, "location": self.location, "capabilities": list(self.capabilities)}


def _package_root(package: str) -> Optional[Path]:
    """The directory an installed package's files are read from, WITHOUT importing it."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    return Path(locations[0]).parent if locations else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundled_artifacts(registry: Optional[ModelRegistry] = None, *,
                             site_root: Optional[Path] = None) -> list[ArtifactCheck]:
    """Hash every pinned .tflite the registry lists inside an installed package
    (path = `<package>/...`) where Python would load it from, against its pin.

    `site_root` overrides where packages are looked up (tests). A package that is
    not installed at all is reported ABSENT, not a fault: nothing can load it."""
    registry = registry if registry is not None else load_registry(validate=False)
    roots: dict[str, Optional[Path]] = {}
    checks: list[ArtifactCheck] = []
    for card in registry.models:
        for artifact in card.artifacts:
            if artifact.format != "tflite" or not artifact.sha256:
                continue
            package = artifact.path.split("/", 1)[0]
            if site_root is not None:
                root: Optional[Path] = site_root if (site_root / package).is_dir() else None
            else:
                if package not in roots:
                    roots[package] = _package_root(package)
                root = roots[package]
            expected = artifact.sha256.lower()
            caps = tuple(card.provides_capabilities)
            if root is None:
                checks.append(ArtifactCheck(card.id, artifact.path, expected, ARTIFACT_ABSENT, capabilities=caps))
                continue
            location = root / artifact.path
            if not location.is_file():
                checks.append(ArtifactCheck(card.id, artifact.path, expected, ARTIFACT_MISSING,
                                            location=str(location), capabilities=caps))
                continue
            try:
                found = _file_sha256(location)
            except OSError:
                checks.append(ArtifactCheck(card.id, artifact.path, expected, ARTIFACT_UNREADABLE,
                                            location=str(location), capabilities=caps))
                continue
            status = ARTIFACT_OK if found == expected else ARTIFACT_MISMATCH
            checks.append(ArtifactCheck(card.id, artifact.path, expected, status, found=found,
                                        location=str(location), capabilities=caps))
    return checks


def artifact_faults(checks: Iterable[ArtifactCheck]) -> dict[str, str]:
    """Capability -> why its model is refused (first failing file per capability)."""
    faults: dict[str, str] = {}
    for check in checks:
        if check.fault:
            for capability in check.capabilities:
                faults.setdefault(capability, check.reason())
    return faults


def _refusing_solution(reason: str, fields: tuple[str, ...]) -> type:
    class IntegrityRefusedGraph:
        """Stands in for a MediaPipe solution whose model file failed verification:
        loads nothing, runs nothing, finds nothing. The capability is reported
        impaired with `refused_reason` (services/runtime.py)."""

        refused_reason = reason

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def process(self, image: Any) -> SimpleNamespace:
            del image
            return SimpleNamespace(**dict.fromkeys(fields))

        def close(self) -> None:
            return None

        def __enter__(self) -> "IntegrityRefusedGraph":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    return IntegrityRefusedGraph


_BOOT_CHECKS: list[ArtifactCheck] = []
_BOOT_FAULTS: dict[str, str] = {}


def enforce_bundled_artifacts(checks: Optional[list[ArtifactCheck]] = None) -> dict[str, str]:
    """Boot (main.py, before any module is built): verify, and fail closed for each
    capability whose model file does not match -- its MediaPipe solution class is
    replaced by a stand-in, so the unverified file is never parsed or run. If the
    check itself cannot run, every MediaPipe capability is refused. Returns (and
    remembers, for the runtime's health report) capability -> reason."""
    try:
        found = list(checks) if checks is not None else verify_bundled_artifacts()
        faults = artifact_faults(found)
    except Exception as exc:  # noqa: BLE001 - an unverifiable model is an unverified one
        found = []
        faults = dict.fromkeys(MEDIAPIPE_SOLUTIONS, f"model integrity could not be checked ({exc}); refused")
    _BOOT_CHECKS[:] = found
    _BOOT_FAULTS.clear()
    _BOOT_FAULTS.update(faults)
    for capability, reason in faults.items():
        solution = MEDIAPIPE_SOLUTIONS.get(capability)
        if solution is None:
            continue
        module_name, class_name, fields = solution
        try:
            import mediapipe as mp  # only on a fault: the stand-in must replace the real class

            setattr(getattr(mp.solutions, module_name), class_name, _refusing_solution(reason, fields))
        except Exception:  # noqa: BLE001, S112 - no mediapipe: nothing can load the model anyway
            continue
    return dict(faults)


def boot_artifact_faults() -> dict[str, str]:
    """What enforce_bundled_artifacts() refused at this boot (empty if it never ran)."""
    return dict(_BOOT_FAULTS)


def boot_artifact_checks() -> list[dict]:
    return [c.to_dict() for c in _BOOT_CHECKS]


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m core.model_registry verify-artifacts [--json-out F] [--allow-absent]`:
    exit 1 when a pinned bundled model file is missing or not its pin (or, unless
    --allow-absent, its package is not installed). The staged install runs this in
    the new release's venv before the release may be switched to."""
    parser = argparse.ArgumentParser(prog="python -m core.model_registry")
    parser.add_argument("command", choices=["verify-artifacts"])
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--allow-absent", action="store_true")
    args = parser.parse_args(argv)
    checks = verify_bundled_artifacts()
    bad = [c for c in checks if c.fault or (c.status == ARTIFACT_ABSENT and not args.allow_absent)]
    for check in checks:
        print(f"  [{check.status.upper():>14}] {check.path}  {check.found or '-'}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"checks": [c.to_dict() for c in checks],
                                                   "ok": not bad}, indent=2) + "\n")
    print(f"bundled model artifacts: {'OK' if not bad else 'FAIL'} ({len(checks)} checked, {len(bad)} bad)")
    return 1 if bad else 0


__all__ = [
    "BINDING_STATUSES",
    "CAPABILITY_KINDS",
    "DECODERS",
    "DEFAULT_PATH",
    "EVIDENCE_LEVELS",
    "HARDWARE_TARGETS",
    "LADDER",
    "MEDIAPIPE_SOLUTIONS",
    "PERCEPTION_KINDS",
    "TIERS",
    "Approval",
    "ArtifactCheck",
    "CapabilityLink",
    "ChainView",
    "EvaluationStatus",
    "Evidence",
    "EvidenceBinding",
    "Impact",
    "InferenceComponent",
    "ModelCard",
    "ModelRegistry",
    "MonitoringNodeEntry",
    "ProductCapability",
    "RegistryError",
    "artifact_faults",
    "as_dict",
    "binding_violations",
    "boot_artifact_checks",
    "boot_artifact_faults",
    "capability_evidence",
    "capability_link",
    "capability_tiers",
    "capability_violations",
    "card_violations",
    "chain_for",
    "component_violations",
    "display_state",
    "enforce_bundled_artifacts",
    "load_registry",
    "node_tier",
    "node_violations",
    "parse_registry",
    "policy_view",
    "preset_view",
    "preset_views",
    "presets_using",
    "reevaluation_impact",
    "registry_violations",
    "role_view",
    "role_views",
    "roles_using",
    "tier_rank",
    "verify_bundled_artifacts",
    "weakest",
]


if __name__ == "__main__":
    sys.exit(main())
