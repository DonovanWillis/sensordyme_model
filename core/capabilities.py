"""Vision capability layer: what SensorDyme can actually watch for, and why not.

This sits between a user's plain-language request and the models that run on the
device. It exists so the product can answer three questions honestly:

    "Can you watch for this?"          -> a support verdict, not a guess
    "What would make it possible?"     -> a specific model / data / hardware gap
    "What can you do instead?"         -> the nearest thing that genuinely works

Why a registry rather than a list of prompts
--------------------------------------------
The old Configure-with-AI matched keywords against five presets and fell back to
security monitoring for anything unrecognised, which meant "monitor for a can"
silently became "watch for intruders". Adding prompts to that list would never
have finished: the space of reasonable monitoring requests is open-ended.

So capabilities are *declared*, not enumerated per phrase. Each one states the
labels and predicates it can supply. A request is satisfiable when every label
and predicate it needs is supplied by some registered, available capability.
Register a can-detection model tomorrow and "monitor for a can" starts working
with no change to this file's logic — which is the whole point.

What is deliberately NOT here
-----------------------------
No phrase matching. Turning English into a structured rule is the language
model's job (see cloud/app.py). This layer only answers whether the resulting
structure is executable on the hardware in front of us, and that answer is
derived from what is registered, never from the wording of the request.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class Support(str, Enum):
    """How well the current system can serve a request.

    The gradations matter: a customer told "no" walks away, while a customer
    told "not until you add a model that recognises cans" knows what to buy.
    """

    SUPPORTED = "supported"                      # runs today, as asked
    PARTIAL = "partial"                          # an approximation, stated as such
    NEEDS_MODEL = "needs_model"                  # a model we do not ship
    NEEDS_TRAINING_DATA = "needs_training_data"  # a model we could fine-tune, given examples
    NEEDS_HARDWARE = "needs_hardware"            # no camera-only route exists
    UNSUPPORTED = "unsupported"                  # not determinable from vision at all


@dataclass(frozen=True)
class Capability:
    """One thing the vision stack can do, and what it can name or assert.

    `labels` are nouns the capability can find in a frame ("person", "bottle").
    `predicates` are claims it can evaluate about them ("present", "in_zone").
    Keeping them separate is what lets a new detector widen the vocabulary
    without touching rule logic, and a new tracker widen the logic without
    touching the vocabulary.
    """

    id: str
    kind: str            # objects | pose | face | hands | tracking | temporal | action
    summary: str
    labels: frozenset[str] = frozenset()
    predicates: frozenset[str] = frozenset()
    available: bool = True
    # When unavailable: the specific thing that would enable it.
    requires: str = ""
    requirement_kind: Support = Support.NEEDS_MODEL


@dataclass
class Assessment:
    """The verdict on one requested (target, predicate) pair."""

    support: Support
    reason: str
    requires: str = ""
    alternatives: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.support in (Support.SUPPORTED, Support.PARTIAL)

    def as_dict(self) -> dict:
        return {
            "support": self.support.value,
            "reason": self.reason,
            "requires": self.requires,
            "alternatives": list(self.alternatives),
            "capabilities": list(self.capabilities),
        }


# Category words map to the concrete classes that stand in for them. This is
# vocabulary, not prompt matching: it never inspects a user's sentence, only
# normalises a label the model already chose. Kept deliberately small — the
# language model handles synonyms far better than a table ever will, and a table
# that tried to would be the hardcoded list this design exists to avoid.
_CATEGORY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "food": ("sandwich", "pizza", "hot dog", "donut", "cake", "banana", "apple",
             "orange", "broccoli", "carrot", "bowl"),
    "drink": ("bottle", "cup", "wine glass"),
    "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle", "train"),
    "animal": ("dog", "cat", "bird", "horse", "sheep", "cow", "bear", "elephant", "zebra", "giraffe"),
    "bag": ("backpack", "handbag", "suitcase"),
    "furniture": ("chair", "couch", "bed", "dining table", "bench"),
}


class CapabilityRegistry:
    """The set of capabilities this device currently has."""

    def __init__(self, capabilities: Optional[Iterable[Capability]] = None) -> None:
        self._capabilities: dict[str, Capability] = {}
        for capability in capabilities or ():
            self.register(capability)

    # ------------------------------------------------------------------ registration
    def register(self, capability: Capability) -> None:
        """Add or replace a capability. This is the extension point: shipping a
        new model means registering what it can name, not editing rule logic."""
        self._capabilities[capability.id] = capability

    def unregister(self, capability_id: str) -> None:
        self._capabilities.pop(capability_id, None)

    @property
    def capabilities(self) -> list[Capability]:
        return list(self._capabilities.values())

    # ------------------------------------------------------------------ vocabulary
    def labels(self, available_only: bool = True) -> set[str]:
        return {
            label
            for capability in self._capabilities.values()
            if capability.available or not available_only
            for label in capability.labels
        }

    def predicates(self, available_only: bool = True) -> set[str]:
        return {
            predicate
            for capability in self._capabilities.values()
            if capability.available or not available_only
            for predicate in capability.predicates
        }

    def providers_of_label(self, label: str) -> list[Capability]:
        return [c for c in self._capabilities.values() if label in c.labels]

    def providers_of_predicate(self, predicate: str) -> list[Capability]:
        return [c for c in self._capabilities.values() if predicate in c.predicates]

    def expand(self, label: str) -> tuple[str, ...]:
        """Concrete detectable classes for a category word, if it is one."""
        available = self.labels()
        return tuple(c for c in _CATEGORY_EXPANSIONS.get(label.lower(), ()) if c in available)

    def nearest_labels(self, label: str, limit: int = 3) -> tuple[str, ...]:
        """Closest detectable labels, for "did you mean" guidance.

        Spelling-based, so it catches typos and near-misses. It will not map
        "can" to "bottle" — nothing here pretends to understand meaning, which
        is honest: the suggestion the user sees comes from the language model,
        and this is only the safety net beneath it.
        """
        candidates = sorted(self.labels())
        close = difflib.get_close_matches(label.lower(), candidates, n=limit, cutoff=0.72)
        expanded = self.expand(label)
        return tuple(dict.fromkeys([*expanded, *close]))[:limit]

    # ------------------------------------------------------------------ assessment
    def assess(self, target: str, predicate: str) -> Assessment:
        """Can we watch `target` for `predicate`, and if not, what is missing?"""
        target = (target or "").strip().lower()
        predicate = (predicate or "").strip().lower()

        predicate_caps = self.providers_of_predicate(predicate)
        if not predicate_caps:
            return Assessment(
                Support.UNSUPPORTED,
                f"'{predicate}' is not something SensorDyme can evaluate.",
                requires="a capability that can assess this kind of condition",
            )

        unavailable_predicate = [c for c in predicate_caps if not c.available]
        available_predicate = [c for c in predicate_caps if c.available]
        if not available_predicate:
            blocker = unavailable_predicate[0]
            return Assessment(
                blocker.requirement_kind,
                f"'{predicate}' needs {blocker.summary}, which this device does not have.",
                requires=blocker.requires,
                capabilities=(blocker.id,),
            )

        # A predicate that names no object (e.g. a whole-frame condition) is
        # satisfied by the predicate capability alone.
        if not target:
            best = available_predicate[0]
            return Assessment(Support.SUPPORTED, best.summary, capabilities=(best.id,))

        label_caps = [c for c in self.providers_of_label(target) if c.available]
        if not label_caps:
            expansions = self.expand(target)
            if expansions:
                return Assessment(
                    Support.PARTIAL,
                    f"'{target}' is not a class the detector knows, but it can be approximated "
                    f"by {', '.join(expansions)}.",
                    alternatives=expansions,
                    capabilities=tuple(c.id for c in available_predicate[:1]),
                )
            # A capability that names this label but isn't installed knows
            # exactly what is missing — far more useful than the generic answer.
            declared = [c for c in self.providers_of_label(target) if not c.available]
            if declared:
                blocker = declared[0]
                return Assessment(
                    blocker.requirement_kind,
                    f"'{target}' needs {blocker.summary}, which this device does not have.",
                    requires=blocker.requires,
                    capabilities=(blocker.id,),
                )
            nearest = self.nearest_labels(target)
            return Assessment(
                Support.NEEDS_TRAINING_DATA,
                f"The current vision model does not recognise '{target}'.",
                requires=(
                    f"a detector trained on '{target}' — either a model that includes it, "
                    f"or labelled examples to fine-tune one"
                ),
                alternatives=nearest,
            )

        weakest = min(
            (self._pair_support(lc, pc) for lc in label_caps for pc in available_predicate),
            key=lambda pair: _SUPPORT_ORDER[pair[0]],
        )
        support, reason = weakest
        return Assessment(
            support,
            reason,
            capabilities=tuple({c.id for c in label_caps} | {c.id for c in available_predicate}),
        )

    def _pair_support(self, label_cap: Capability, predicate_cap: Capability) -> tuple[Support, str]:
        """How well a label capability and a predicate capability combine."""
        if predicate_cap.requirement_kind is Support.PARTIAL:
            return Support.PARTIAL, predicate_cap.summary
        return Support.SUPPORTED, f"{label_cap.summary}; {predicate_cap.summary}"

    def describe_for_llm(self) -> str:
        """The capability sheet handed to the language model.

        Layout is not cosmetic here — it was measured. The first version put all
        84 labels on one comma-separated line and listed UNAVAILABLE conditions
        before available ones. A 1.5B model reading that declined objects it
        could actually detect ("banana", "baseball glove") because it could not
        find them, while the most prominent conditions on the page were the ones
        that do not work.

        So: labels are grouped and wrapped so each line is scannable, and
        anything unavailable is moved to its own section at the end, after the
        model has already read what it *can* do.
        """
        available = self.labels()
        lines = ["DETECTABLE OBJECTS — use these labels EXACTLY, and only these:"]

        # Grouped by the category map that already exists, so this introduces no
        # second vocabulary to keep in sync.
        grouped: set[str] = set()
        for word, members in sorted(_CATEGORY_EXPANSIONS.items()):
            usable = [m for m in members if m in available]
            if usable:
                lines.append(f"  {word}: {', '.join(sorted(usable))}")
                grouped.update(usable)

        remaining = sorted(available - grouped)
        if remaining:
            lines.append("  other:")
            for chunk in _chunk(remaining, 8):
                lines.append(f"    {', '.join(chunk)}")

        lines.append("")
        lines.append("CONDITIONS — use these EXACTLY:")
        unavailable: list[str] = []
        for capability in sorted(self._capabilities.values(), key=lambda c: c.id):
            if not capability.predicates:
                continue
            listing = f"  {', '.join(sorted(capability.predicates))} — {capability.summary}"
            if capability.available:
                lines.append(listing)
            else:
                unavailable.append(
                    f"  {', '.join(sorted(capability.predicates))} — needs {capability.requires}"
                )

        if unavailable:
            lines.append("")
            lines.append("NOT AVAILABLE — never use these; put such requests in `unsupported`:")
            lines.extend(unavailable)
        return "\n".join(lines)


def _chunk(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


_SUPPORT_ORDER = {
    Support.SUPPORTED: 0,
    Support.PARTIAL: 1,
    Support.NEEDS_TRAINING_DATA: 2,
    Support.NEEDS_MODEL: 3,
    Support.NEEDS_HARDWARE: 4,
    Support.UNSUPPORTED: 5,
}


# --------------------------------------------------------------------- the real device
# Every entry below reflects something actually in this repository. Nothing is
# aspirational: a capability listed as available must have code behind it, or
# the honesty this layer exists to provide is worthless.

def _coco_labels() -> frozenset[str]:
    try:
        from core.accel import COCO_NAMES

        return frozenset(COCO_NAMES.values())
    except Exception:  # noqa: BLE001 - capability listing must never break startup
        return frozenset()


def _pose_available() -> bool:
    """Whether body-pose estimation can run on this device at all.

    Availability here means the model is installed. Whether the *active* mode
    runs it is a separate question the runtime answers, by warning when a rule
    needs detections the running module does not produce.
    """
    try:
        import mediapipe  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - absence is an answer, not an error
        return False


def default_registry() -> CapabilityRegistry:
    """The capability set of a stock SensorDyme device."""
    return CapabilityRegistry([
        Capability(
            id="objects.coco",
            kind="objects",
            summary="the object detector recognises 80 everyday classes",
            labels=_coco_labels(),
            predicates=frozenset({"present", "absent", "count"}),
        ),
        Capability(
            id="zones.polygon",
            kind="temporal",
            summary="user-drawn areas, tested against where a subject's feet are",
            predicates=frozenset({"in_zone"}),
        ),
        Capability(
            id="temporal.persistence",
            kind="temporal",
            summary="conditions confirmed over time rather than on one frame",
            predicates=frozenset({"persists", "still", "lingering"}),
        ),
        Capability(
            id="spatial.proximity",
            kind="objects",
            summary="two objects overlapping or close together — an approximation of "
                    "interaction, not proof of it",
            predicates=frozenset({"near", "holding"}),
            requirement_kind=Support.PARTIAL,
        ),
        # ---- declared but NOT available on a stock device --------------------
        Capability(
            id="tracking.identity",
            kind="tracking",
            summary="following the same subject across frames",
            predicates=frozenset({"enters", "exits", "direction", "picks_up", "removes"}),
            available=False,
            requires="a multi-object tracker (e.g. ByteTrack/SORT over the detector's boxes)",
            requirement_kind=Support.NEEDS_MODEL,
        ),
        Capability(
            id="pose.body",
            kind="pose",
            summary="body posture from skeletal landmarks, in modes that run pose "
                    "estimation (elderly monitoring / fall detection)",
            # Postures are LABELS, not predicates: a rule asks whether
            # "lying_down" is present exactly as it asks whether "dog" is, so the
            # rule engine needs no special case and a better posture model can
            # replace this one without the rule layer noticing.
            labels=frozenset({"standing", "sitting", "lying_down", "on_floor"}),
            predicates=frozenset(),
            available=_pose_available(),
            requires="mediapipe, and a monitoring mode that runs pose estimation",
            requirement_kind=Support.NEEDS_MODEL,
        ),
        Capability(
            id="pose.climbing",
            kind="pose",
            summary="climbing, reaching and other whole-body actions",
            predicates=frozenset({"climbing", "reaching"}),
            available=False,
            requires="an action-recognition model — posture alone cannot "
                     "distinguish climbing from standing on something",
            requirement_kind=Support.NEEDS_TRAINING_DATA,
        ),
        Capability(
            id="action.recognition",
            kind="action",
            summary="recognising activities from motion over time",
            predicates=frozenset({"eating", "drinking", "running", "sleeping", "falling", "smoking"}),
            available=False,
            requires="an action-recognition model, or labelled clips of the behaviour "
                     "to fine-tune one",
            requirement_kind=Support.NEEDS_TRAINING_DATA,
        ),
        Capability(
            id="face.identity",
            kind="face",
            summary="telling known people from strangers",
            predicates=frozenset({"known_person", "unknown_person"}),
            available=False,
            requires="a face-recognition model plus enrolled faces — a significant "
                     "privacy decision, not just a model choice",
            requirement_kind=Support.NEEDS_MODEL,
        ),
        Capability(
            id="scene.fire_smoke",
            kind="objects",
            summary="visible smoke or flame",
            labels=frozenset({"smoke", "fire"}),
            predicates=frozenset(),
            available=False,
            requires="a smoke/fire detection model — the COCO detector has no such class",
            requirement_kind=Support.NEEDS_MODEL,
        ),
    ])


_REGISTRY: Optional[CapabilityRegistry] = None


def registry() -> CapabilityRegistry:
    """Process-wide registry, built once."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = default_registry()
    return _REGISTRY


def set_registry(new_registry: CapabilityRegistry) -> None:
    """Replace the registry (tests, and devices with extra models installed)."""
    global _REGISTRY
    _REGISTRY = new_registry
