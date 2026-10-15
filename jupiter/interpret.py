"""The one task Jupiter owns today: request in, validated rules out.

The pipeline is deliberately short, and every stage after the model is
deterministic:

    sanitise -> prompt -> provider -> extract JSON -> clamp -> validate

`interpret()` never raises for a bad model answer. A provider that returns
nonsense, malformed JSON, or a rule the device cannot run produces an
Interpretation with empty `rules` and a populated `unsupported` — the same shape
as a well-formed refusal. The caller cannot tell the difference and does not
need to, because in both cases nothing reaches a camera.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from core.capabilities import registry
from jupiter import confidence as confidence_module
from jupiter import prompt as prompt_module
from jupiter import security
from jupiter.grammar import build_grammar
from jupiter.providers import JupiterUnavailable, Provider, ProviderResult, get_provider
from jupiter.schema import Interpretation, parse_response, schema_for_registry

_DECODER = json.JSONDecoder()


@dataclass
class InterpretOutcome:
    """What happened, for the caller, the logs and the eval harness."""

    interpretation: Interpretation
    provider: str = ""
    model: str = ""
    latency_s: float = 0.0
    ok: bool = True
    error: str = ""
    injection_suspected: bool = False
    meta: dict = field(default_factory=dict)


def extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of model output.

    Small models wrap JSON in prose or fences even when told not to. Refusing
    to parse that would fail a response that is actually correct, so the widest
    balanced-looking block is tried before giving up.
    """
    payload, _ = _extract_json_verbose(text)
    return payload


def _extract_json_verbose(text: str) -> tuple[Optional[dict], bool]:
    """`extract_json`, plus whether the text needed repair to parse.

    `needed_repair` feeds jupiter/confidence.py: a response whose JSON parsed
    on the first, unmodified try is a real, checkable signal that the model
    followed the output contract; one that only parsed after stripping fences
    or falling back to a regex scan is a real signal that it didn't, distinct
    from whether the JSON's *content* turned out to be a valid rule.
    """
    if not text:
        return None, False
    candidate = text.strip()
    fenced = candidate.startswith("```")
    if fenced:
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    try:
        parsed = json.loads(candidate)
        return (parsed if isinstance(parsed, dict) else None), fenced
    except (ValueError, TypeError):
        pass
    # The first object that decodes on its own, wherever it starts. A greedy
    # `{.*}` span failed whenever trailing prose also held a brace ("replace
    # {zone} with ...") or the model wrote two objects.
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            parsed, _end = _DECODER.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed, True
    return None, True


def interpret(
    request: str,
    *,
    provider: Optional[Provider] = None,
    provider_name: Optional[str] = None,
    config: Optional[dict] = None,
    reg=None,
    use_grammar: bool = True,
) -> InterpretOutcome:
    """Turn one plain-language request into validated SensorDyme rules.

    `use_grammar=False` turns constrained decoding off entirely — neither the
    JSON Schema nor the GBNF grammar is sent — which is what the eval's
    `--no-grammar` run needs in order to measure what the constraint is worth.
    """
    reg = reg or registry()

    try:
        cleaned = security.sanitise_request(request)
    except security.RequestRejected as exc:
        return InterpretOutcome(
            interpretation=Interpretation(understood="", rules=[], unsupported=[{
                "request": str(request)[:120],
                "reason": exc.reason,
                "requires": "a shorter, plainer description of one thing to watch for",
                "support": "unsupported",
                "alternatives": [],
            }]),
            ok=False,
            error=exc.reason,
        )

    # Recorded, not blocked. The rule still has to survive validation, and
    # refusing outright would reject legitimate phrasings that happen to match.
    injection = security.looks_like_injection(cleaned)

    # Provider construction is inside the guarded path on purpose. A typo in
    # SENSORDYME_AI_PROVIDER, or an openai_compatible runtime with no base_url,
    # raises here — and outside a try that meant a 500 on every Configure-with-AI
    # request rather than a message saying the AI is misconfigured.
    try:
        backend = provider or get_provider(provider_name, config)
    except JupiterUnavailable as exc:
        return InterpretOutcome(
            interpretation=Interpretation(understood="", rules=[], unsupported=[{
                "request": cleaned,
                "reason": exc.message,
                "requires": "a valid AI provider — see docs/JUPITER.md",
                "support": "unsupported",
                "alternatives": [],
            }]),
            provider=str(provider_name or "jupiter"),
            ok=False,
            error=exc.message,
            injection_suspected=injection,
        )

    system = prompt_module.system_prompt(reg)
    user = prompt_module.user_prompt(cleaned)
    # Ollama enforces the schema through `format`, so leaving it in place made
    # an "unconstrained" run exactly as constrained as a normal one.
    schema = schema_for_registry(reg) if use_grammar else None
    grammar = None
    if use_grammar:
        try:
            grammar = build_grammar(reg)
        except ValueError:
            grammar = None  # empty registry; validation still applies

    try:
        result: ProviderResult = backend.complete(system, user, schema=schema, grammar=grammar)
    except Exception as exc:  # noqa: BLE001 - any provider failure is "unavailable"
        # JupiterUnavailable is the contract, but a third-party provider (or a
        # requests error a built-in one failed to wrap) must still produce an
        # answer the caller can fall back from, never an HTTP 500.
        message = exc.message if isinstance(exc, JupiterUnavailable) else (
            f"the AI provider failed: {exc}"
        )
        return InterpretOutcome(
            interpretation=Interpretation(understood="", rules=[], unsupported=[{
                "request": cleaned,
                "reason": message,
                "requires": "a reachable Jupiter model — see docs/JUPITER.md",
                "support": "unsupported",
                "alternatives": [],
            }]),
            provider=getattr(backend, "name", "unknown"),
            ok=False,
            error=message,
            injection_suspected=injection,
        )

    payload, json_needed_repair = _extract_json_verbose(result.text)
    if payload is None:
        return InterpretOutcome(
            interpretation=Interpretation(understood="", rules=[], unsupported=[{
                "request": cleaned,
                "reason": "the model did not return usable JSON",
                "requires": "a model that supports constrained output, or a retry",
                "support": "unsupported",
                "alternatives": [],
            }]),
            provider=result.provider, model=result.model, latency_s=result.latency_s,
            ok=False, error="unparseable response", injection_suspected=injection,
        )

    bounded = security.clamp_response(payload)
    interpretation = parse_response(bounded, reg)
    constrained = bool(result.meta.get("constrained", False))
    report = confidence_module.score(
        raw_rules=bounded.get("rules") or [],
        rules_accepted=len(interpretation.rules),
        fragments_unsupported=len(interpretation.unsupported),
        json_needed_repair=json_needed_repair,
        constrained=constrained,
    )
    return InterpretOutcome(
        interpretation=interpretation,
        provider=result.provider,
        model=result.model,
        latency_s=result.latency_s,
        ok=True,
        injection_suspected=injection,
        meta={
            "prompt_chars": result.prompt_chars,
            "completion_chars": result.completion_chars,
            "constrained": constrained,
            "rules": len(interpretation.rules),
            "rejected": len(interpretation.unsupported),
            # How confident Jupiter is that it understood the REQUEST, not to
            # be confused with RuleDraft.confidence (an alert-trigger
            # threshold) or Event/Explanation.confidence (vision detection
            # confidence) — see jupiter/confidence.py's module docstring.
            "interpretation_confidence": report.as_dict(),
        },
    )
