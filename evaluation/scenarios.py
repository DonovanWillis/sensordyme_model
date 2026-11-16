"""The in-cabin scenario coverage matrix: which situations a claim must cover, and how much data each needs.

The matrix is data (``evaluation/schemas/incabin_scenarios_v1.yaml``). This
module loads it, refuses one that is inconsistent or weakens the safety rules,
and answers the questions the report asks of it:

  * which ground-truth event types each cell needs labelled (positives the
    module must catch, negatives it must not alert on, boundary behaviour that
    is outside the validated scope);
  * how the product's alert types map onto those event types, with each
    alert's sustain-time config key;
  * the minimum subjects / clips / events / exposure hours before a cell's
    numbers may be printed as a claim (the power check lives in
    ``evaluation/event_metrics.py``);
  * whether a clip's tags are consistent with the cells it says it belongs to.

    python -m evaluation.scenarios            # validate the shipped matrix, print a summary
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
MATRIX_PATH = SCHEMA_DIR / "incabin_scenarios_v1.yaml"
VOCAB_V2_PATH = SCHEMA_DIR / "incabin_v2.json"
MATRIX_SCHEMA = "sensordyme.incabin.scenario_matrix"
ROLES = ("positive", "negative", "boundary")

#: Safety flags every matrix must carry with exactly these values.
REQUIRED_SAFETY = {
    "stationary_vehicle_only": True,
    "consenting_adults_only": True,
    "children_allowed": False,
    "driving_allowed": False,
    "impairment_allowed": False,
    "operator_present": True,
    "stop_any_time": True,
}
MINIMUM_KEYS = ("subjects", "clips", "events", "negative_events", "exposure_hours")


class MatrixError(ValueError):
    """The matrix is inconsistent or unsafe; nothing may be scored against it."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Minimums:
    subjects: int
    clips: int
    events: int
    negative_events: int
    exposure_hours: float

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in MINIMUM_KEYS}


@dataclass(frozen=True)
class EventRule:
    """How one ground-truth event type is scored."""

    type: str
    role: str  # positive | negative | boundary
    alerts: frozenset[str]
    early_s: float
    late_s: float

    def window(self, start_s: float, end_s: float) -> tuple[float, float]:
        return start_s - self.early_s, end_s + self.late_s


@dataclass(frozen=True)
class Cell:
    id: str
    title: str
    tags: dict[str, tuple[str, ...]]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    boundary: tuple[str, ...]
    minimums: Minimums
    cases: tuple[str, ...]
    checklist: tuple[str, ...]
    safety: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "tags": {k: list(v) for k, v in self.tags.items()},
                "positives": list(self.positives), "negatives": list(self.negatives),
                "boundary": list(self.boundary), "minimums": self.minimums.to_dict(),
                "cases": list(self.cases), "checklist": list(self.checklist), "safety": list(self.safety)}


@dataclass(frozen=True)
class Matrix:
    version: int
    sha256: str
    path: str
    annotation_schema: int
    safety: dict
    axes: dict[str, tuple[str, ...]]
    required_clip_tags: tuple[str, ...]
    cells: dict[str, Cell]
    rules: dict[str, EventRule]
    alert_sustain_keys: dict[str, str]  # alert type -> config key of its sustain time
    module: str
    abstain_fraction: float
    overall: Minimums

    @property
    def alerts(self) -> frozenset[str]:
        return frozenset(self.alert_sustain_keys)

    def types_with_role(self, role: str) -> list[str]:
        return sorted(t for t, r in self.rules.items() if r.role == role)

    def sustain_seconds(self, module_config: dict) -> dict[str, float]:
        """Each scored alert's configured sustain time, where the config has it."""
        out = {}
        for alert, key in self.alert_sustain_keys.items():
            value = module_config.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[alert] = float(value)
        return out

    def clip_tag_problems(self, cell_ids: list[str] | tuple[str, ...], tags: dict) -> list[str]:
        """Why a clip with these tags may not be counted toward these cells (empty: it may)."""
        problems = []
        for axis in self.required_clip_tags:
            if axis not in tags:
                problems.append(f"tags.{axis} is required")
        for axis, value in sorted(tags.items()):
            if axis not in self.axes:
                problems.append(f"tags.{axis}: not an axis of the scenario matrix v{self.version}")
            elif value not in self.axes[axis]:
                problems.append(f"tags.{axis}: {value!r} is not one of {list(self.axes[axis])}")
        if not cell_ids:
            problems.append("scenarios: a clip must name at least one matrix cell")
        for cell_id in cell_ids:
            cell = self.cells.get(cell_id)
            if cell is None:
                problems.append(f"scenarios: unknown cell {cell_id!r}")
                continue
            for axis, allowed in sorted(cell.tags.items()):
                if tags.get(axis) not in allowed:
                    problems.append(f"scenarios: cell {cell_id} needs tags.{axis} in {list(allowed)}, "
                                    f"clip has {tags.get(axis)!r}")
        return problems


def _vocabulary(path: Path = VOCAB_V2_PATH) -> dict:
    return json.loads(path.read_text())


def _minimums(raw, fallback: Optional[Minimums], where: str, problems: list[str]) -> Minimums:
    raw = dict(raw or {})
    values: dict = {}
    for key in MINIMUM_KEYS:
        value = raw.get(key, getattr(fallback, key) if fallback else None)
        cast = float if key == "exposure_hours" else int
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"{where}.minimums.{key} must be a positive number, got {value!r}")
            value = 1
        values[key] = cast(value)
    if unknown := sorted(set(raw) - set(MINIMUM_KEYS)):
        problems.append(f"{where}.minimums: unknown keys {unknown}")
    return Minimums(**values)


def parse(raw: dict, *, vocab: Optional[dict] = None, sha256: str = "", path: str = "") -> Matrix:
    """Validate a matrix document. Raises MatrixError with every problem found."""
    vocab = vocab if vocab is not None else _vocabulary()
    known_events = set(vocab.get("events", {}))
    problems: list[str] = []
    if raw.get("schema") != MATRIX_SCHEMA:
        problems.append(f"schema must be {MATRIX_SCHEMA!r}")
    version = raw.get("version")
    if not isinstance(version, int) or version < 1:
        problems.append("version must be a positive integer")
    if raw.get("annotation_schema") != vocab.get("version"):
        problems.append(f"annotation_schema {raw.get('annotation_schema')!r} does not match the "
                        f"vocabulary version {vocab.get('version')!r}")

    safety = dict(raw.get("safety") or {})
    for key, required in REQUIRED_SAFETY.items():
        if safety.get(key) is not required:
            problems.append(f"safety.{key} must be {required} (stationary, consenting adults, no children)")

    axes = {str(k): tuple(str(v) for v in values) for k, values in (raw.get("axes") or {}).items()}
    if not axes:
        problems.append("axes: no tag vocabulary")
    required_tags = tuple(raw.get("required_clip_tags") or ())
    problems += [f"required_clip_tags: {a!r} is not an axis" for a in required_tags if a not in axes]

    defaults = _minimums((raw.get("defaults") or {}).get("minimums"), None, "defaults", problems)
    overall = _minimums((raw.get("overall") or {}).get("minimums"), defaults, "overall", problems)

    scoring = raw.get("scoring") or {}
    sustain_keys = {str(k): str(v) for k, v in (scoring.get("alerts") or {}).items()}
    if not sustain_keys:
        problems.append("scoring.alerts: no alert types are scored")
    window = scoring.get("window") or {}
    early, late = float(window.get("early_s", 0.0)), float(window.get("late_s", 2.0))
    rules: dict[str, EventRule] = {}
    for gt_type, raw_spec in (scoring.get("events") or {}).items():
        spec = raw_spec or {}
        if gt_type not in known_events:
            problems.append(f"scoring.events.{gt_type}: not an annotation v2 event type")
        role = spec.get("role")
        if role not in ROLES:
            problems.append(f"scoring.events.{gt_type}: role {role!r} is not one of {ROLES}")
        alerts = frozenset(spec.get("alerts") or ())
        if not alerts:
            problems.append(f"scoring.events.{gt_type}: no alerts listed")
        if unknown := sorted(alerts - set(sustain_keys)):
            problems.append(f"scoring.events.{gt_type}: alerts {unknown} are not in scoring.alerts")
        e, lt = float(spec.get("early_s", early)), float(spec.get("late_s", late))
        if e < 0 or lt < 0:
            problems.append(f"scoring.events.{gt_type}: early_s/late_s must be >= 0")
        rules[str(gt_type)] = EventRule(str(gt_type), str(role), alerts, e, lt)
    abstain = float(scoring.get("abstain_fraction", 0.5))
    if not 0 < abstain <= 1:
        problems.append("scoring.abstain_fraction must be in (0, 1]")

    cells: dict[str, Cell] = {}
    for n, c in enumerate(raw.get("cells") or []):
        cid = str(c.get("id") or "")
        where = f"cells[{n}]" + (f" ({cid})" if cid else "")
        if not cid:
            problems.append(f"{where}: no id")
            continue
        if cid in cells:
            problems.append(f"{where}: duplicate cell id")
        tags: dict[str, tuple[str, ...]] = {}
        for axis, raw_values in (c.get("tags") or {}).items():
            values = tuple(raw_values if isinstance(raw_values, list) else [raw_values])
            if axis not in axes:
                problems.append(f"{where}.tags.{axis}: not an axis")
            elif bad := [v for v in values if v not in axes[axis]]:
                problems.append(f"{where}.tags.{axis}: {bad} not in the axis vocabulary")
            tags[str(axis)] = tuple(str(v) for v in values)
        if not tags:
            problems.append(f"{where}: a cell needs at least one tag (which clips count toward it)")
        groups = {}
        for role_key, role in (("positives", "positive"), ("negatives", "negative"), ("boundary", "boundary")):
            types = tuple(str(t) for t in (c.get(role_key) or ()))
            for t in types:
                rule = rules.get(t)
                if rule is None:
                    problems.append(f"{where}.{role_key}: {t!r} has no scoring rule")
                elif rule.role != role:
                    problems.append(f"{where}.{role_key}: {t!r} is scored as {rule.role}, not {role}")
            groups[role_key] = types
        if c.get("safety_exempt") or c.get("exemptions"):
            problems.append(f"{where}: no cell may be exempt from the matrix safety rules")
        cells[cid] = Cell(
            id=cid, title=str(c.get("title") or cid), tags=tags,
            positives=groups["positives"], negatives=groups["negatives"], boundary=groups["boundary"],
            minimums=_minimums(c.get("minimums"), defaults, where, problems),
            cases=tuple(str(x) for x in c.get("cases") or ()),
            checklist=tuple(str(x) for x in c.get("checklist") or ()),
            safety=tuple(str(x) for x in c.get("safety") or ()),
        )
    if not cells:
        problems.append("cells: the matrix has no cells")
    if problems:
        raise MatrixError(problems)
    return Matrix(version=int(version or 0), sha256=sha256, path=path,
                  annotation_schema=int(raw["annotation_schema"]), safety=safety, axes=axes,
                  required_clip_tags=required_tags, cells=cells, rules=rules,
                  alert_sustain_keys=sustain_keys, module=str(scoring.get("module") or ""),
                  abstain_fraction=abstain, overall=overall)


def load(path: str | Path = MATRIX_PATH, *, vocab: Optional[dict] = None) -> Matrix:
    data = Path(path).read_bytes()
    return parse(yaml.safe_load(data), vocab=vocab, sha256=hashlib.sha256(data).hexdigest(), path=str(path))


def main(argv: Optional[list[str]] = None) -> int:
    path = Path(argv[0]) if argv else MATRIX_PATH
    try:
        matrix = load(path)
    except MatrixError as exc:
        for p in exc.problems:
            print(f"  PROBLEM: {p}")
        return 1
    print(f"scenario matrix v{matrix.version} sha256={matrix.sha256[:16]}: {len(matrix.cells)} cells, "
          f"{len(matrix.rules)} scored event types, alerts {sorted(matrix.alerts)}")
    for cell in matrix.cells.values():
        m = cell.minimums
        print(f"  {cell.id:24s} +{list(cell.positives)} -{list(cell.negatives)}  "
              f"min {m.subjects} subjects / {m.events} events / {m.exposure_hours} h")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
