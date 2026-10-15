"""Scoring Jupiter, and comparing it against anything else.

Built before any model, because without it "is Jupiter good enough?" has no
answer and a switch of provider is a matter of opinion. It runs against any
`Provider`, so Jupiter, Anthropic and a scripted stub are scored by identical
code on identical inputs.

The metrics are chosen so that failing safely and failing dangerously do not
look the same:

  * **parse_rate** — did usable JSON come back at all.
  * **schema_rate** — did every returned rule survive registry validation.
  * **target_accuracy / predicate_accuracy** — did it identify the right thing
    and the right condition. Scored only on cases that should be supported.
  * **refusal_accuracy** — did it refuse what it should refuse. This is the one
    that matters most: a model that invents a capability is worse than one that
    declines a possible request, because the first ships a rule that never fires
    and the user believes they are protected.
  * **hallucination_rate** — how often a rule was proposed for a request that
    cannot be served. Distinct from refusal_accuracy: this counts the dangerous
    direction specifically.
  * **latency** — measured per call, reported as mean and p95.

Nothing here estimates. Every number comes from a call that actually happened.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from core.capabilities import registry
from jupiter.eval.synth import Example
from jupiter.interpret import interpret
from jupiter.providers import Provider
from jupiter.schema import NO_TARGET


@dataclass
class CaseResult:
    """One request, scored."""

    request: str
    expected_supported: bool
    got_supported: bool
    parsed: bool
    schema_ok: bool
    target_match: Optional[bool] = None
    predicate_match: Optional[bool] = None
    latency_s: float = 0.0
    error: str = ""
    tags: tuple = field(default_factory=tuple)

    @property
    def refusal_correct(self) -> bool:
        # A call that failed or returned no usable JSON proposed no rule, but
        # it did not REFUSE anything — counting it as a correct refusal let a
        # dead or garbage-emitting provider score 100% on refusals.
        return self.parsed and self.got_supported == self.expected_supported

    @property
    def hallucinated(self) -> bool:
        """Proposed a rule for something that cannot be served."""
        return self.got_supported and not self.expected_supported


@dataclass
class Report:
    """Aggregate scores for one provider over one dataset."""

    provider: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    wall_clock_s: float = 0.0

    def _rate(self, predicate: Callable[[CaseResult], bool], among=None) -> Optional[float]:
        pool = [c for c in self.cases if among is None or among(c)]
        if not pool:
            return None
        return round(sum(1 for c in pool if predicate(c)) / len(pool), 4)

    def summary(self) -> dict:
        supported_only = lambda c: c.expected_supported  # noqa: E731
        latencies = [c.latency_s for c in self.cases if c.latency_s > 0]
        return {
            "provider": self.provider,
            "model": self.model,
            "cases": len(self.cases),
            "parse_rate": self._rate(lambda c: c.parsed),
            "schema_rate": self._rate(lambda c: c.schema_ok),
            "target_accuracy": self._rate(lambda c: c.target_match is True, supported_only),
            "predicate_accuracy": self._rate(lambda c: c.predicate_match is True, supported_only),
            "refusal_accuracy": self._rate(lambda c: c.refusal_correct),
            "hallucination_rate": self._rate(lambda c: c.hallucinated),
            "latency_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
            # Nearest-rank: the smallest value with at least 95% of samples at
            # or below it. int() rounded the rank down and read low.
            "latency_p95_s": (
                round(sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)], 3)
                if latencies else None
            ),
            "wall_clock_s": round(self.wall_clock_s, 1),
        }

    def failures(self, limit: int = 20) -> list[dict]:
        """The cases that went wrong — the only part worth reading closely."""
        bad = [
            c for c in self.cases
            if not c.refusal_correct or not c.schema_ok
            or (c.expected_supported and c.target_match is False)
        ]
        return [
            {
                "request": c.request,
                "expected": "supported" if c.expected_supported else "refusal",
                "got": "supported" if c.got_supported else "refusal",
                "target_match": c.target_match,
                "predicate_match": c.predicate_match,
                "error": c.error,
            }
            for c in bad[:limit]
        ]


def run(
    examples: list[Example],
    provider: Provider,
    *,
    reg=None,
    use_grammar: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Report:
    """Score one provider over a dataset. Every call is real."""
    reg = reg or registry()
    report = Report(provider=getattr(provider, "name", "unknown"),
                    model=getattr(provider, "model", ""))
    started = time.monotonic()

    for index, example in enumerate(examples, start=1):
        outcome = interpret(
            example.request, provider=provider, reg=reg, use_grammar=use_grammar
        )
        interpretation = outcome.interpretation
        got_supported = bool(interpretation.rules)

        target_match = predicate_match = None
        if example.expect_supported and got_supported:
            first = interpretation.rules[0]
            target_match = first.target == example.target
            # Only scored when the dataset states an expectation; a case that
            # does not pin the predicate should not punish a reasonable choice.
            predicate_match = (
                first.predicate == example.predicate if example.predicate else None
            )

        report.cases.append(CaseResult(
            request=example.request,
            expected_supported=example.expect_supported,
            got_supported=got_supported,
            parsed=outcome.ok,
            # Rules that fail validation never reach `interpretation.rules`, so
            # checking those measured nothing (schema_rate == parse_rate). The
            # question is whether every rule the MODEL proposed survived — the
            # raw payload is what it actually wrote.
            schema_ok=outcome.ok and _all_proposed_rules_valid(interpretation),
            target_match=target_match,
            predicate_match=predicate_match,
            latency_s=outcome.latency_s,
            error=outcome.error,
            tags=example.tags,
        ))
        if progress:
            progress(index, len(examples))

    report.wall_clock_s = time.monotonic() - started
    return report


def _all_proposed_rules_valid(interpretation) -> bool:
    """Every rule the model proposed — other than a deliberate target:"none"
    refusal — was accepted by validation."""
    raw_rules = (interpretation.raw or {}).get("rules") or []
    proposed = sum(
        1 for r in raw_rules
        if isinstance(r, dict) and str(r.get("target", "")).strip().lower() != NO_TARGET
    )
    return len(interpretation.rules) == proposed


def compare(reports: list[Report]) -> dict:
    """Side-by-side summaries. Reports only what was measured."""
    return {
        "providers": [r.summary() for r in reports],
        "note": (
            "All figures measured on the dataset supplied to this run. "
            "No value here is estimated or carried over from another run."
        ),
    }


def format_report(reports: list[Report]) -> str:
    """A plain-text table, for a terminal or a commit message."""
    if not reports:
        return "no results"
    metrics = [
        ("cases", "cases"),
        ("parse_rate", "parsed"),
        ("schema_rate", "schema ok"),
        ("target_accuracy", "target"),
        ("predicate_accuracy", "predicate"),
        ("refusal_accuracy", "refusal"),
        ("hallucination_rate", "hallucination"),
        ("latency_mean_s", "latency mean"),
        ("latency_p95_s", "latency p95"),
    ]
    summaries = [r.summary() for r in reports]
    width = max(len(label) for _, label in metrics) + 2
    lines = ["", " " * width + "".join(f"{s['provider'][:16]:>18}" for s in summaries)]
    lines.append(" " * width + "".join(f"{str(s['model'])[:16]:>18}" for s in summaries))
    lines.append("-" * (width + 18 * len(summaries)))
    for key, label in metrics:
        row = f"{label:<{width}}"
        for summary in summaries:
            value = summary.get(key)
            row += f"{'—' if value is None else value:>18}"
        lines.append(row)
    return "\n".join(lines)
