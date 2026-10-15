#!/usr/bin/env python3
"""Benchmark Jupiter — and compare it against anything else.

    # check Jupiter is reachable before spending a run on it
    python scripts/jupiter_eval.py --check

    # score Jupiter on 200 generated cases
    python scripts/jupiter_eval.py --provider jupiter --cases 200

    # head to head against Claude (needs ANTHROPIC_API_KEY and `pip install anthropic`)
    python scripts/jupiter_eval.py --provider jupiter --provider anthropic --cases 200

    # save the numbers so a later change can be compared against them
    python scripts/jupiter_eval.py --provider jupiter --out data/jupiter-baseline.json

Every figure printed is measured during the run. Nothing is estimated, and
nothing is carried over from a previous run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.capabilities import registry
from jupiter.eval import compare, format_report, generate, run, validate_dataset
from jupiter.grammar import grammar_stats
from jupiter.providers import JupiterUnavailable, get_provider
from jupiter.router import role_config


def build_provider(name: str, model: str = "", base_url: str = ""):
    """The provider the backend would actually use for Configure-with-AI.

    get_provider(name) with no config benchmarked qwen2.5:1.5b on localhost,
    ignoring JUPITER_MODEL / JUPITER_BASE_URL / jupiter_config.yaml — so
    `--check` could pass while the backend's real configure model (7b) was not
    even pulled. The router's configure-role config applies whenever it names
    this same provider; --model / --base-url override either way.
    """
    routed_name, config = role_config("configure")
    config = dict(config) if routed_name == name.strip().lower() else {}
    if model:
        config["model"] = model
    if base_url:
        config["base_url"] = base_url
    return get_provider(name, config)


def check(provider_names: list[str], model: str = "", base_url: str = "") -> int:
    """Report whether each provider can actually answer, without scoring."""
    stats = grammar_stats()
    print(f"capability registry : {stats['targets']} targets x {stats['predicates']} "
          f"predicates = {stats['combinations']} executable combinations")
    worst = 0
    for name in provider_names:
        try:
            provider = build_provider(name, model, base_url)
        except JupiterUnavailable as exc:
            print(f"{name:<22} UNAVAILABLE  {exc.message}")
            worst = 1
            continue
        ok, detail = provider.available()
        print(f"{name:<22} {'OK         ' if ok else 'UNAVAILABLE'}  {detail}")
        if not ok:
            worst = 1
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", action="append", default=[],
                        help="provider to score; repeat to compare (default: jupiter)")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-grammar", action="store_true",
                        help="disable constrained decoding, to measure what it is worth")
    parser.add_argument("--check", action="store_true", help="probe providers and exit")
    parser.add_argument("--model", default="",
                        help="model to score (default: what the backend's configure role uses)")
    parser.add_argument("--base-url", default="", help="model server URL override")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    parser.add_argument("--show-failures", type=int, default=10)
    args = parser.parse_args()

    providers = args.provider or ["jupiter"]
    if args.check:
        return check(providers, args.model, args.base_url)

    reg = registry()
    examples = generate(limit=args.cases, reg=reg, seed=args.seed)

    # A dataset whose labels disagree with the registry would produce confident
    # nonsense, so it is checked before anything is scored against it.
    problems = validate_dataset(examples, reg)
    if problems:
        print(f"dataset is inconsistent with the capability registry ({len(problems)} problems):")
        for problem in problems[:10]:
            print(f"  ! {problem}")
        return 2

    refusals = sum(1 for e in examples if not e.expect_supported)
    print(f"{len(examples)} cases ({refusals} refusals, "
          f"{len(examples) - refusals} supported), seed {args.seed}")

    reports = []
    for name in providers:
        try:
            provider = build_provider(name, args.model, args.base_url)
        except JupiterUnavailable as exc:
            print(f"\n{name}: {exc.message}")
            continue
        ok, detail = provider.available()
        if not ok:
            print(f"\n{name}: unavailable — {detail}")
            continue

        print(f"\nrunning {name} ({detail})…")
        started = time.monotonic()

        def progress(done, total, _name=name, _started=started):
            if done % 10 == 0 or done == total:
                elapsed = time.monotonic() - _started
                print(f"  {_name}: {done}/{total}  {elapsed:.0f}s", end="\r", flush=True)

        reports.append(run(examples, provider, reg=reg,
                           use_grammar=not args.no_grammar, progress=progress))
        print()

    if not reports:
        print("\nno provider could be reached — nothing was measured.")
        return 1

    print(format_report(reports))

    for report in reports:
        failures = report.failures(args.show_failures)
        if failures:
            print(f"\n{report.provider} — first {len(failures)} failures:")
            for failure in failures:
                print(f"  {failure['expected']:>9} -> {failure['got']:<9} {failure['request']!r}")
                if failure["error"]:
                    print(f"            {failure['error']}")

    if args.out:
        payload = compare(reports)
        payload["dataset"] = {"cases": len(examples), "refusals": refusals, "seed": args.seed}
        payload["grammar"] = grammar_stats(reg)
        payload["constrained_decoding"] = not args.no_grammar
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
