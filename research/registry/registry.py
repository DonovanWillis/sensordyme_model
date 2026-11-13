"""Experiment registry: one JSON object per line in research/registry/experiments.jsonl.

    python -m research.registry.registry validate          # schema + consistency checks (CI-safe)
    python -m research.registry.registry show [EXP-09]     # human-readable summary

Rules the validator enforces (docs/research/M84_RESEARCH_PACKAGE.md, "Reproducibility"):
  * every entry names its hypothesis, baseline, split, primary metric and pre-registered
    success criteria, its budget, dataset manifest id, model/checkpoint hashes, config,
    seed, hardware, software revision and decision;
  * ``split`` may never be ``test``: the frozen test split is not used for any decision;
  * ``decision`` is one of pending / retain / reject / inconclusive / not_run, and anything other
    than pending / not_run must cite at least one job id and one result artifact;
  * ``preregistered_at`` must not be later than ``started_at`` (criteria are fixed first), unless the
    entry says ``retrospective: true``, which marks a hypothesis written after the run: such an
    entry is a record, never a pre-registered test.
A registry line is a record of what happened, not a claim of production readiness: no entry
promotes a model (that is the model registry under config/, which this file never touches).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PATH = Path(__file__).with_name("experiments.jsonl")
REQUIRED = ("id", "title", "kind", "hypothesis", "baseline", "split", "primary_metric", "success_criteria",
            "budget", "dataset", "model", "config", "seed", "hardware", "software_revision", "jobs",
            "artifacts", "status", "decision", "preregistered_at")
DECISIONS = {"pending", "retain", "reject", "inconclusive", "not_run"}


def load(path: Path = PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(entries: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for n, e in enumerate(entries, 1):
        tag = e.get("id", f"line {n}")
        for k in REQUIRED:
            if k not in e or (e[k] in (None, "", []) and k not in ("jobs", "artifacts")):
                problems.append(f"{tag}: missing {k}")
        if tag in seen:
            problems.append(f"{tag}: duplicate id")
        seen.add(tag)
        if "test" in str(e.get("split", "")).lower().split("/"):
            problems.append(f"{tag}: uses the frozen test split")
        d = e.get("decision")
        if d not in DECISIONS:
            problems.append(f"{tag}: decision {d!r} not in {sorted(DECISIONS)}")
        if d not in ("pending", "not_run") and (not e.get("jobs") or not e.get("artifacts")):
            problems.append(f"{tag}: decision {d} must cite jobs and artifacts")
        pre, start = e.get("preregistered_at"), e.get("started_at")
        if pre and start and pre > start and not e.get("retrospective"):
            problems.append(f"{tag}: criteria registered after the run started (mark it retrospective)")
        if e.get("retrospective") and not (pre and start and pre > start):
            problems.append(f"{tag}: retrospective entries must show criteria recorded after the start")
        if e.get("kind") == "training" and not start and e.get("status") not in ("preregistered", "not_run"):
            problems.append(f"{tag}: a run that started must record started_at")
    return problems


def main(argv: list[str]) -> int:
    entries = load()
    if not argv or argv[0] == "validate":
        problems = validate(entries)
        for p in problems:
            print("PROBLEM", p)
        print(f"{len(entries)} entries, {len(problems)} problems")
        return 1 if problems else 0
    if argv[0] == "show":
        for e in entries:
            if len(argv) > 1 and e["id"] not in argv[1:]:
                continue
            print(f"{e['id']}  [{e['status']} / {e['decision']}]  {e['title']}")
            print(f"  H: {e['hypothesis']}")
            print(f"  split {e['split']}; metric {e['primary_metric']}; jobs {', '.join(e['jobs']) or '-'}")
            if e.get("result"):
                print(f"  result: {json.dumps(e['result'])[:400]}")
        return 0
    print(__doc__)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
