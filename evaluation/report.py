"""The M10 event-level evaluation report: Markdown + JSON, NO DATA where there is none.

    # today (no consented recordings exist): every cell reports NO DATA
    python -m evaluation.report --out reports/m10_nodata

    # after a replay (python -m evaluation.run ... --split-lock lock.json --out runs/r1)
    python -m evaluation.report --manifest m.json --root /data/ds --split val \
        --run runs/r1 --split-lock lock.json --config config.yaml --out reports/r1
    # the held-out split: --split test --frozen-test --purpose "release 1.2 final" (recorded)

Reads the replay's ``predictions.jsonl`` + ``availability.jsonl`` + ``report.json``,
the v2 (or migrated v1) annotations, and the scenario matrix; scores every clip
with ``evaluation.event_metrics``; and writes ``eval_report.json`` and
``eval_report.md``. Every metric passes the matrix's power check first: below a
cell's minimum it prints INSUFFICIENT DATA with the sample size, never a rate;
with nothing recorded it prints NO DATA.

Refuses (exit 2, nothing written) when the data could leak or was tuned on:
manifest problems, a licence that forbids evaluation, any subject / session /
vehicle-day / co-occupant / near-duplicate frame across splits, a training
manifest overlapping the evaluated splits, a missing or violated split lock,
the test split without ``--frozen-test``, annotations that disagree with the
manifest, or a replay run of a different manifest or split.

Recorded with every report: commit, config sha256, model artifact sha256s (or
"missing on this host"), split lock sha256, test-access record, scenario matrix
version + sha256, manifest id, and the replay run's own provenance and backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from evaluation import annotations_v2 as av2
from evaluation import event_metrics as em
from evaluation import leakage, scenarios
from evaluation import manifest as mf
from evaluation.annotations import Span

REPO = Path(__file__).resolve().parent.parent
REPORT_SCHEMA = "sensordyme.eval.report"
NO_DATA_TEXT = "NO DATA"
INSUFFICIENT_TEXT = "INSUFFICIENT DATA"
#: Module-config keys that name a model file, recorded with their sha256.
MODEL_KEYS = ("yolo_weights", "model_path", "phone_weights", "model_weights", "detector_weights", "hand_model_path")


class ReportRefusedError(Exception):
    """A report must not exist for this input (leaky, unlocked, untuned-on-test violated, inconsistent)."""


def _git_commit() -> Optional[str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607
                             timeout=10, check=True, cwd=REPO).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,  # noqa: S607
                               timeout=10, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return sha + ("+dirty" if dirty else "")


def model_artifacts(config: dict, module: str, extra: list[str], base: Path = REPO) -> list[dict]:
    """sha256 of every model file the config names (and any extra), or why there is none."""
    module_cfg = (config.get("modules") or {}).get(module) or {}
    named = []
    if hef := (config.get("accelerator") or {}).get("yolo_hef"):
        named.append(("accelerator.yolo_hef", str(hef)))
    named += [(f"modules.{module}.{k}", str(module_cfg[k])) for k in MODEL_KEYS if module_cfg.get(k)]
    named += [("--model", p) for p in extra]
    out = []
    for source, name in named:
        path = Path(name) if Path(name).is_absolute() else base / name
        entry: dict[str, Any] = {"source": source, "path": name}
        if path.is_file():
            entry["sha256"] = mf.file_sha256(path)
        else:
            entry["sha256"] = None
            entry["status"] = "missing on this host (not hashed)"
        out.append(entry)
    return out


def load_run(run_dir: Path) -> tuple[dict, dict, Optional[dict], list[str]]:
    """(predictions by item, availability by item, run report, problems)."""
    problems: list[str] = []
    predictions: dict[str, list[tuple[float, str]]] = defaultdict(list)
    availability: dict[str, dict] = {}
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.is_file():
        problems.append(f"{pred_path} missing")
    else:
        for n, line in enumerate(pred_path.read_text().splitlines()):
            if line.strip():
                try:
                    row = json.loads(line)
                    predictions[str(row["item"])].append((float(row["t"]), str(row["type"])))
                except (ValueError, KeyError, TypeError):
                    problems.append(f"predictions.jsonl line {n + 1} is not {{item, t, type}}")
    avail_path = run_dir / "availability.jsonl"
    if avail_path.is_file():
        for n, line in enumerate(avail_path.read_text().splitlines()):
            if line.strip():
                try:
                    row = json.loads(line)
                    availability[str(row["item"])] = row
                except (ValueError, KeyError, TypeError):
                    problems.append(f"availability.jsonl line {n + 1} is not {{item, duration_s, unavailable}}")
    run_report = None
    if (run_dir / "report.json").is_file():
        run_report = json.loads((run_dir / "report.json").read_text())
    else:
        problems.append(f"{run_dir / 'report.json'} missing (the replay's provenance)")
    return dict(predictions), availability, run_report, problems


def _durations(availability: dict, run_report: Optional[dict]) -> dict[str, float]:
    out = {i: float(r["duration_s"]) for i, r in availability.items() if r.get("duration_s") is not None}
    for clip in (run_report or {}).get("clips", []):
        out.setdefault(str(clip["item"]), float(clip["duration_s"]))
    return out


def _unavailable(row: Optional[dict]) -> Optional[dict[str, list[Span]]]:
    if row is None or row.get("unavailable") is None:
        return None
    return {k: [Span(float(a), float(b)) for a, b in spans] for k, spans in row["unavailable"].items()}


def _overall_status(summaries: list[dict], clips: int) -> str:
    if clips == 0:
        return NO_DATA_TEXT
    ok = sum(1 for s in summaries for group in ("detection", "alerts", "negatives")
             for m in s[group].values() if m["status"] == em.OK)
    return f"{ok} CLAIMABLE METRIC(S)" if ok else INSUFFICIENT_TEXT


def generate(*, matrix: scenarios.Matrix, manifest: Optional[mf.Manifest] = None, root: Optional[Path] = None,
             split: str = "val", run_dir: Optional[Path] = None, lock_path: Optional[Path] = None,
             frozen_test: bool = False, purpose: str = "", config: Optional[dict] = None,
             config_sha256: Optional[str] = None, artifacts: Optional[list[dict]] = None,
             commit: Optional[str] = None, training_manifests: tuple[mf.Manifest, ...] = (), frames: int = 0,
             include_diagnostics: bool = False, bootstrap_n: int = 2000, seed: int = 0,
             verify_files: bool = True, today: Optional[str] = None) -> dict:
    """The report document. Raises ReportRefusedError instead of producing a number that must not exist."""
    config = config or {}
    module_cfg = (config.get("modules") or {}).get(matrix.module) or {}
    sustain = matrix.sustain_seconds(module_cfg)
    provenance: dict[str, Any] = {
        "commit": commit or "unknown", "config_sha256": config_sha256, "module": matrix.module,
        "sustain_s": sustain, "model_artifacts": artifacts or [],
        "matrix": {"version": matrix.version, "sha256": matrix.sha256, "path": matrix.path},
        "annotation_schema": matrix.annotation_schema, "split_lock_sha256": None, "test_access": None,
        "manifest": None, "run": None,
    }
    data: dict[str, Any] = {"split": split, "clips": 0, "subjects": 0, "hours": 0.0,
                            "leakage_checks": [], "lock_notes": []}
    per_subject: dict[str, em.Scores] = {}
    per_cell: dict[str, dict[str, em.Scores]] = defaultdict(dict)

    if manifest is not None:
        problems = mf.validate(manifest)
        if not manifest.usable_for("evaluation"):
            problems.append(f"licence class {manifest.licence.get('class')!r} does not permit evaluation")
        if root is None:
            problems.append("--root is required with --manifest (annotations and files live there)")
        problems += leakage.check(manifest, root=root, frames=frames if root else 0)
        data["leakage_checks"] = ["subject/co-occupant", "session", "vehicle-day", "sha256 duplicate"] + \
            ([f"near-duplicate frames ({frames}/video dHash)"] if frames and root else [])
        for training in training_manifests:
            problems += leakage.cross_manifest(training, manifest)
            data["leakage_checks"].append(f"training manifest {training.name} {training.version}")
        lock = None
        if lock_path is None:
            problems.append("a split lock is required (python -m evaluation.leakage lock)")
        else:
            lock, lock_problems = leakage.load_lock(lock_path)
            problems += lock_problems
            if lock is not None:
                lock_problems, data["lock_notes"] = leakage.verify_lock(manifest, lock)
                problems += lock_problems
                provenance["split_lock_sha256"] = lock["lock_sha256"]
        if split == "test" and not frozen_test:
            problems.append("the test split is locked: pass --frozen-test (and --purpose) for a final, "
                            "untuned measurement; the access is recorded")
        items = [i for i in manifest.split(split) if i.kind == "video"]
        if not items:
            problems.append(f"split {split!r} has no video items")
        if root is not None and verify_files:
            problems += mf.verify_files(manifest, root, items, decode=False)
        predictions: dict = {}
        availability: dict = {}
        run_report = None
        if run_dir is None:
            problems.append("--run is required with --manifest (the replay's predictions)")
        else:
            predictions, availability, run_report, run_problems = load_run(run_dir)
            problems += run_problems
            if run_report is not None:
                ds = run_report.get("dataset") or {}
                if ds.get("manifest_id") != manifest.id:
                    problems.append(f"the run replayed manifest {ds.get('manifest_id')!r}, not {manifest.id!r}")
                if ds.get("split") != split:
                    problems.append(f"the run replayed split {ds.get('split')!r}, not {split!r}")
                if run_report.get("overrides") and split == "test":
                    problems.append("the run used --set overrides: refusing to report them on the test split")
                run_lock = run_report.get("split_lock_sha256")
                if run_lock and lock is not None and run_lock != lock["lock_sha256"]:
                    problems.append("the run was made against a different split lock")
                provenance["run"] = {k: run_report.get(k) for k in
                                     ("commit", "config_sha256", "module", "overrides", "backend", "replay",
                                      "started_at", "split_lock_sha256")}
                if config_sha256 and run_report.get("config_sha256") not in (None, config_sha256):
                    problems.append("the run's config sha256 differs from --config: report on the config "
                                    "that produced the predictions")
        durations = _durations(availability, run_report)
        clips: list[tuple[mf.Item, av2.AnnotationV2]] = []
        for item in items:
            if not item.annotations:
                problems.append(f"{item.id}: no annotations")
                continue
            if item.id not in durations:
                problems.append(f"{item.id}: not in the replay run (no duration)")
            if root is None:
                continue
            annotation, errs = av2.load_any(root / item.annotations, item, matrix=matrix)
            problems += [f"{item.id}: {e}" for e in errs]
            if annotation is None:
                continue
            problems += [f"{item.id}: {e}" for e in av2.consistency_problems(annotation, item)]
            d = durations.get(item.id)
            if d is not None and annotation.clip.duration_s is not None and abs(annotation.clip.duration_s - d) > 1.0:
                problems.append(f"{item.id}: annotated duration {annotation.clip.duration_s}s != replayed {d:.2f}s")
            clips.append((item, annotation))
        if problems:
            raise ReportRefusedError("; ".join(problems))
        if split == "test":
            try:
                provenance["test_access"] = leakage.authorize_test_access(
                    lock_path or Path(), frozen_test=frozen_test, purpose=purpose,
                    commit=provenance["commit"], config_sha256=config_sha256 or "unknown", now=today)
            except leakage.TestAccessRefusedError as exc:
                raise ReportRefusedError(str(exc)) from exc
        provenance["manifest"] = {"name": manifest.name, "version": manifest.version, "id": manifest.id,
                                  "licence_class": manifest.licence.get("class")}
        for item, annotation in clips:
            clip = em.ClipInput(item.id, item.subject, annotation, durations[item.id],
                                predictions.get(item.id, []), _unavailable(availability.get(item.id)))
            scores = em.score_clip(clip, matrix, sustain)
            per_subject[item.subject] = per_subject.get(item.subject, em.Scores()).add(scores)
            for cell_id in annotation.scenarios:
                cell = per_cell[cell_id]
                cell[item.subject] = cell.get(item.subject, em.Scores()).add(scores)
        data.update(clips=len(clips), subjects=len(per_subject),
                    hours=round(sum(s.seconds for s in per_subject.values()) / 3600, 4))

    kwargs = {"matrix": matrix, "include_diagnostics": include_diagnostics, "bootstrap_n": bootstrap_n, "seed": seed}
    overall = em.summarise(per_subject, minimums=matrix.overall, **kwargs)
    cells = {}
    for cell in matrix.cells.values():
        summary = em.summarise(per_cell.get(cell.id, {}), minimums=cell.minimums, positives=cell.positives,
                               negatives=cell.negatives, **kwargs)
        cells[cell.id] = {"title": cell.title, "cases": list(cell.cases), "minimums": cell.minimums.to_dict(),
                          "tags": {k: list(v) for k, v in cell.tags.items()}, "summary": summary}
    return {
        "schema": REPORT_SCHEMA, "version": 1,
        "generated_at": today or datetime.now(UTC).isoformat(timespec="seconds"),
        "status": _overall_status([overall, *(c["summary"] for c in cells.values())], data["clips"]),
        "provenance": provenance, "data": data, "overall": overall, "cells": cells,
        "notes": [
            "Scores module events on replayed clips; the runtime AlertPolicy (episodes, escalation) is not replayed.",
            "Off the Pi the object detector is a desktop surrogate; see provenance.run.backend.",
            "A 3-5 person pilot cannot meet any cell minimum: its numbers are diagnostics, never claims.",
            "Stationary vehicle, consenting adults only (matrix safety block); nothing here measures driving.",
        ],
    }


# -- Markdown --------------------------------------------------------------------------------

def _num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(ci: Optional[list | tuple]) -> str:
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _dist(d: dict) -> str:
    return "n/a" if not d or not d.get("n") else f"{_num(d['p50'], 2)} / {_num(d['p90'], 2)} / {_num(d['max'], 2)}"


def _gate_text(m: dict) -> Optional[str]:
    if m["status"] == em.NO_DATA:
        return NO_DATA_TEXT
    if m["status"] == em.INSUFFICIENT:
        return f"{INSUFFICIENT_TEXT} ({', '.join(m['reasons'])})"
    return None


def _tables(summary: dict, *, by_subject: bool) -> list[str]:
    lines = ["", "**Detection (positive events)**", "",
             ("| event type | subjects | events | recall | recall Wilson 95% | recall subject-bootstrap 95% "
              "| missed (abstained / silent) | latency s p50/p90/max | latency - sustain s p50/p90/max |"),
             "|---|---|---|---|---|---|---|---|---|"]
    for kind, m in summary["detection"].items():
        n = m["n"]
        if (gate := _gate_text(m)) is not None:
            lines.append(f"| {kind} | {n['subjects']} | {n['events']} | {gate} | | | | | |")
            continue
        c = m["claim"]
        lines.append(f"| {kind} | {n['subjects']} | {n['events']} | {_num(c['recall'])} | {_ci(c['recall_wilson95'])} "
                     f"| {_ci(c['recall_subject_bootstrap95'])} | {c['missed']} ({_num(c['missed_abstained'])} / "
                     f"{_num(c['missed_silent'])}) | {_dist(c['latency_s'])} | {_dist(c['latency_minus_sustain_s'])} |")
    lines += ["", "**False alerts and abstention (per alert type)**", "",
              ("| alert | subjects | exposure h | false alerts | false alerts/h | FA/h subject-bootstrap 95% "
               "| precision | unknown rate | on legitimate behaviour | boundary |"),
              "|---|---|---|---|---|---|---|---|---|---|"]
    for alert, m in summary["alerts"].items():
        n = m["n"]
        if (gate := _gate_text(m)) is not None:
            lines.append(f"| {alert} | {n['subjects']} | {n['exposure_hours']} | {gate} | | | | | | |")
            continue
        c = m["claim"]
        precision = _num(c["precision"]) if c["precision_status"] == em.OK else \
            (NO_DATA_TEXT if c["precision_status"] == em.NO_DATA else INSUFFICIENT_TEXT)
        unknown = _num(c["unknown_rate"]) if n["availability_recorded"] else f"{NO_DATA_TEXT} (not recorded)"
        lines.append(f"| {alert} | {n['subjects']} | {n['exposure_hours']} | {c['false_alerts']} "
                     f"| {_num(c['false_alerts_per_hour'])} | {_ci(c['false_alerts_per_hour_subject_bootstrap95'])} "
                     f"| {precision} | {unknown} | {c['false_alerts_on_legitimate_behaviour']} | {c['boundary_alerts']} |")
    if summary["negatives"]:
        lines += ["", "**Legitimate behaviour (must not alert)**", "",
                  "| event type | subjects | events | alerted rate | Wilson 95% |", "|---|---|---|---|---|"]
        for kind, m in summary["negatives"].items():
            n = m["n"]
            if (gate := _gate_text(m)) is not None:
                lines.append(f"| {kind} | {n['subjects']} | {n['events']} | {gate} | |")
            else:
                c = m["claim"]
                lines.append(f"| {kind} | {n['subjects']} | {n['events']} | {_num(c['alert_rate'])} "
                             f"| {_ci(c['alert_rate_wilson95'])} |")
    if by_subject:
        rows = [(a, s, r) for a, m in summary["alerts"].items() if m["status"] == em.OK
                for s, r in m["claim"]["false_alerts_per_hour_by_subject"].items()]
        lines += ["", "**False alerts per exposure hour, by subject**", ""]
        if rows:
            lines += ["| alert | subject | false alerts/h |", "|---|---|---|"]
            lines += [f"| {a} | {s} | {_num(r)} |" for a, s, r in rows]
        else:
            lines.append(f"{NO_DATA_TEXT if summary['clips'] == 0 else INSUFFICIENT_TEXT}: no alert type "
                         f"passes the power check, so no per-subject rate is printed.")
    return lines


def render_markdown(report: dict) -> str:
    p, d = report["provenance"], report["data"]
    lines = [f"# Event-level evaluation report ({d['split']})", "",
             f"**Status: {report['status']}**", ""]
    if d["clips"] == 0:
        lines += [(f"{NO_DATA_TEXT}: no consented in-cabin recording has been scored. Every metric below is "
                   "NO DATA by construction; this report lists what each scenario cell needs before a number "
                   "may be printed."), ""]
    lines += ["## Provenance", "", "| field | value |", "|---|---|",
              f"| generated | {report['generated_at']} |", f"| commit | {p['commit']} |",
              f"| config sha256 | {p['config_sha256'] or 'not provided'} |",
              f"| module | {p['module']} |",
              f"| sustain times (s) | {', '.join(f'{k} {v}' for k, v in sorted(p['sustain_s'].items())) or 'n/a'} |",
              f"| scenario matrix | v{p['matrix']['version']} sha256 {p['matrix']['sha256']} |",
              f"| annotation schema | v{p['annotation_schema']} |",
              f"| split lock sha256 | {p['split_lock_sha256'] or 'none (no data scored)'} |",
              f"| test access | {json.dumps(p['test_access']) if p['test_access'] else 'none'} |",
              f"| manifest | {json.dumps(p['manifest']) if p['manifest'] else NO_DATA_TEXT} |"]
    if p["run"]:
        backend = (p["run"].get("backend") or {}).get("detector", "unknown")
        lines.append(f"| replay run | commit {p['run'].get('commit')}, config {p['run'].get('config_sha256')}, "
                     f"detector: {backend} |")
    for a in p["model_artifacts"]:
        lines.append(f"| model `{a['source']}` | {a['path']}: {a['sha256'] or a.get('status')} |")
    lines += ["", "## Data and leakage", "",
              f"- clips {d['clips']}, subjects {d['subjects']}, hours {d['hours']}",
              f"- leakage checks passed: {', '.join(d['leakage_checks']) or 'none run (no data)'}"]
    lines += [f"- lock note: {n}" for n in d["lock_notes"]]
    o = report["overall"]
    unlabelable = _num(o["unlabelable_fraction"]) if o["clips"] else NO_DATA_TEXT
    lines += ["", f"## Overall ({d['split']}, all cells pooled)", "",
              f"Unlabelable (uncertain) fraction of clip time: {unlabelable}"]
    lines += _tables(o, by_subject=True)
    lines += ["", "## Scenario cells", ""]
    for cell_id, cell in report["cells"].items():
        s, mins = cell["summary"], cell["minimums"]
        lines += [f"### {cell_id}: {cell['title']}", "",
                  f"clips {s['clips']}, subjects {s['subjects']}, hours {s['hours']}; minimum for a claim: "
                  f"{mins['subjects']} subjects, {mins['clips']} clips, {mins['events']} events per type, "
                  f"{mins['negative_events']} negative events, {mins['exposure_hours']} exposure h"
                  + (f"; behavioural cases {', '.join(cell['cases'])}" if cell["cases"] else "")]
        lines += _tables(s, by_subject=False)
        lines.append("")
    lines += ["## Notes", ""] + [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="M10 event-level evaluation report (Markdown + JSON)")
    p.add_argument("--matrix", default=str(scenarios.MATRIX_PATH))
    p.add_argument("--manifest", default=None, help="omit for the NO DATA report")
    p.add_argument("--root", default=None)
    p.add_argument("--split", default="val", choices=mf.SPLITS)
    p.add_argument("--run", default=None, help="evaluation.run output directory")
    p.add_argument("--split-lock", default=None)
    p.add_argument("--frozen-test", action="store_true")
    p.add_argument("--purpose", default="")
    p.add_argument("--config", default=str(REPO / "config.yaml"))
    p.add_argument("--model", action="append", default=[], help="extra model artifact to hash")
    p.add_argument("--training-manifest", action="append", default=[])
    p.add_argument("--near-duplicate-frames", type=int, default=5, help="frames per video (0: skip)")
    p.add_argument("--include-diagnostics", action="store_true",
                   help="add below-minimum numbers to the JSON, labelled diagnostic_not_a_claim")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    matrix = scenarios.load(args.matrix)
    config_text = Path(args.config).read_text()
    config = yaml.safe_load(config_text)
    try:
        report = generate(
            matrix=matrix, manifest=mf.load(args.manifest) if args.manifest else None,
            root=Path(args.root) if args.root else None, split=args.split,
            run_dir=Path(args.run) if args.run else None,
            lock_path=Path(args.split_lock) if args.split_lock else None, frozen_test=args.frozen_test,
            purpose=args.purpose, config=config, config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
            artifacts=model_artifacts(config, matrix.module, args.model), commit=_git_commit(),
            training_manifests=tuple(mf.load(t) for t in args.training_manifest),
            frames=args.near_duplicate_frames, include_diagnostics=args.include_diagnostics,
            bootstrap_n=args.bootstrap, seed=args.seed)
    except ReportRefusedError as exc:
        print(f"report refused: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (out / "eval_report.md").write_text(render_markdown(report))
    print(f"report {report['status']}: {out / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
