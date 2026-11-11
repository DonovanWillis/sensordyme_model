"""Evaluate a module configuration on one split of a dataset manifest.

    python -m evaluation.run --manifest m.json --root /data/ds --split val \
        --config config.yaml --out results/run1 [--set drowsy_seconds=2.0] [--target-fps 15]

Writes ``predictions.jsonl`` (every event, per clip) and ``report.json``
(metrics plus everything needed to reproduce them: commit, config hash,
overrides, manifest id, backend actually used, replay settings).

Guards:
  * the manifest must validate, its files must match their hashes, and its
    licence must permit evaluation
  * the held-out ``test`` split needs ``--frozen-test``, and refuses ``--set``:
    thresholds are tuned on ``val``, never on ``test``
  * the backend is reported as found. Off the Pi the detector is a desktop
    surrogate (ultralytics/CPU or CUDA), not the Hailo .hef, and the report says so.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for core/ and modules/

from evaluation import annotations as ann
from evaluation import manifest as mf
from evaluation.metrics import (
    EventCounts,
    FrameCounts,
    cluster_bootstrap,
    frame_level,
    match_events,
)
from evaluation.replay import ReplayResult, replay


class EvaluationRefusedError(Exception):
    """The run would produce a number that must not exist (leaky, unlicensed, tuned on test)."""


def _git_commit() -> Optional[str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S607
                             timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,  # noqa: S607
                               timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return sha + ("+dirty" if dirty else "")


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def evaluate(manifest: mf.Manifest, root: str | Path, split: str, module_factory: Callable[[], object], *,
             frozen_test: bool = False, tuning: bool = False, target_fps: Optional[float] = None,
             resize: Optional[tuple[int, int]] = (640, 480), early_s: float = 0.0, late_s: float = 2.0,
             event_types: Optional[set[str]] = None, limit: Optional[int] = None,
             on_item: Optional[Callable[[mf.Item, ReplayResult], None]] = None) -> dict:
    """Replay every clip of ``split`` and score it. Returns the metrics part of the report."""
    root = Path(root)
    if split == "test" and not frozen_test:
        raise EvaluationRefusedError("the test split is held out: pass --frozen-test for a final, untuned measurement")
    if split == "test" and tuning:
        raise EvaluationRefusedError("refusing to evaluate parameter overrides on the test split (tune on val)")
    if not manifest.usable_for("evaluation"):
        raise EvaluationRefusedError(f"licence class {manifest.licence.get('class')!r} does not permit evaluation")
    problems = mf.validate(manifest)
    items = [i for i in manifest.split(split) if i.kind == "video"][:limit]
    if not items:
        problems.append(f"split {split!r} has no video items")
    problems += mf.verify_files(manifest, root, items, decode=False)
    labelled = []
    for item in items:
        if not item.annotations:
            problems.append(f"{item.id}: no annotations")
            continue
        annotation, errs = ann.load(root / item.annotations)
        problems += [f"{item.id}: {e}" for e in errs]
        if annotation is not None:
            labelled.append((item, annotation))
    if problems:
        raise EvaluationRefusedError("; ".join(problems))

    per_type: dict[str, EventCounts] = defaultdict(EventCounts)
    per_type_subject: dict[str, dict[str, EventCounts]] = defaultdict(lambda: defaultdict(EventCounts))
    per_type_stratum: dict[str, dict[str, EventCounts]] = defaultdict(lambda: defaultdict(EventCounts))
    frames: dict[str, FrameCounts] = defaultdict(FrameCounts)
    clips, cost, total_s = [], [], 0.0
    for item, annotation in labelled:
        result = replay(module_factory, str(root / item.path), target_fps=target_fps, resize=resize)
        if on_item:
            on_item(item, result)
        total_s += result.duration_s
        cost += result.process_ms
        predicted = [(e.t, e.type) for e in result.events]
        types = event_types or ({e.type for e in annotation.events} | {t for _, t in predicted})
        clip = {"item": item.id, "subject": item.subject, "duration_s": round(result.duration_s, 3),
                "frames_processed": result.frames_read, "events": {}}
        for kind in sorted(types):
            counts = match_events(predicted, annotation, kind, result.duration_s, early_s=early_s, late_s=late_s)
            per_type[kind] = per_type[kind].add(counts)
            per_type_subject[kind][item.subject] = per_type_subject[kind][item.subject].add(counts)
            for key, value in sorted(item.conditions.items()):
                stratum = f"{key}={value}"
                per_type_stratum[kind][stratum] = per_type_stratum[kind][stratum].add(counts)
            clip["events"][kind] = {"gt": counts.gt, "caught": counts.caught, "false_alarms": counts.false_alarms}
        for state, spans in annotation.states.items():
            predicted_state = [state in active for active in result.active]
            frames[state] = frames[state].add(frame_level(result.times, predicted_state, spans, annotation))
        clips.append(clip)

    events_out = {}
    for kind, counts in sorted(per_type.items()):
        events_out[kind] = {
            **counts.to_dict(),
            "recall_subject_bootstrap95": cluster_bootstrap(per_type_subject[kind], lambda c: c.recall),
            "false_alarms_per_hour_subject_bootstrap95":
                cluster_bootstrap(per_type_subject[kind], lambda c: c.false_alarms_per_hour),
            "strata": {s: c.to_dict() for s, c in sorted(per_type_stratum[kind].items())},
        }
    cost.sort()
    return {
        "clips": clips,
        "subjects": sorted({i.subject for i, _ in labelled}),
        "hours": round(total_s / 3600, 4),
        "events": events_out,
        "frame_states": {k: v.to_dict() for k, v in sorted(frames.items())},
        "host_process_ms": {"p50": cost[len(cost) // 2], "p90": cost[int(0.9 * (len(cost) - 1))],
                            "frames": len(cost), "note": "HOST cost (desktop surrogate), not the Pi"}
        if cost else None,
        "matching": {"early_s": early_s, "late_s": late_s},
    }


def _module_factory(config: dict, module_name: str, overrides: dict) -> Callable[[], object]:
    from main import get_module_class

    cls = get_module_class(module_name)
    module_cfg = {**config["modules"][module_name], **overrides, "recalibrate": False}
    return lambda: cls(dict(module_cfg))


def _backend(config: dict) -> dict:
    from core import accel

    info = accel.configure(config.get("accelerator", {}))
    hef = (config.get("accelerator") or {}).get("yolo_hef")
    hailo = bool(info.usable and hef)
    return {"accelerator": info.kind, "hailo_in_use": hailo, "detail": accel.status()[1],
            "detector": "hailo .hef (shipping path)" if hailo else
            "desktop surrogate (ultralytics on this host) - NOT the shipping Hailo model",
            "hef": hef if hailo else None}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a module on real video")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="val", choices=mf.SPLITS)
    p.add_argument("--frozen-test", action="store_true", help="final measurement on the held-out test split")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--module", default=None, help="module name (default: config's module)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="module config override")
    p.add_argument("--target-fps", type=float, default=None, help="process at the device's rate")
    p.add_argument("--resize", default="640x480", help="WxH, or 'none'")
    p.add_argument("--early-s", type=float, default=0.0)
    p.add_argument("--late-s", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source-commit", default=None, help="commit, when the tree has no .git (fleet jobs)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    config_text = Path(args.config).read_text()
    config = yaml.safe_load(config_text)
    module_name = args.module or config["module"]
    overrides = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in args.set)}
    resize: Optional[tuple[int, int]] = None
    if args.resize != "none":
        width, height = (int(x) for x in args.resize.split("x"))
        resize = (width, height)
    manifest = mf.load(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    backend = _backend(config)

    with (out / "predictions.jsonl").open("w") as pred_file:
        def record(item: mf.Item, result: ReplayResult) -> None:
            for e in result.events:
                pred_file.write(json.dumps({"item": item.id, "t": round(e.t, 4), "type": e.type,
                                            "confidence": e.confidence, "duration": e.duration,
                                            "signals": e.signals}) + "\n")

        try:
            metrics = evaluate(manifest, args.root, args.split, _module_factory(config, module_name, overrides),
                               frozen_test=args.frozen_test, tuning=bool(overrides), target_fps=args.target_fps,
                               resize=resize, early_s=args.early_s, late_s=args.late_s, limit=args.limit,
                               on_item=record)  # type: ignore[arg-type]
        except EvaluationRefusedError as exc:
            print(f"evaluation refused: {exc}", file=sys.stderr)
            return 2

    report = {
        "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit() or args.source_commit or "unknown",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "module": module_name, "overrides": overrides,
        "dataset": {"name": manifest.name, "version": manifest.version, "manifest_id": manifest.id,
                    "licence_class": manifest.licence.get("class"), "split": args.split,
                    "frozen_test_access": args.split == "test"},
        "replay": {"target_fps": args.target_fps, "resize": list(resize) if resize else None,
                   "timebase": "frame_index/fps"},
        "backend": backend,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        **metrics,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{module_name} on {manifest.name}:{args.split} ({metrics['hours']} h, {len(metrics['subjects'])} subjects)")
    for kind, m in metrics["events"].items():
        print(f"  {kind:28s} recall {m['recall']}  false alarms/h {m['false_alarms_per_hour']}  "
              f"latency p50 {m['latency_p50_s']}")
    with contextlib.suppress(Exception):
        from core import accel

        accel.release_hailo_models()
    return 0


if __name__ == "__main__":
    sys.exit(main())
