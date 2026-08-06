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
import re
from dataclasses import dataclass, field
from typing import Optional

from core.capabilities import registry

from jupiter import prompt as prompt_module
from jupiter import security
from jupiter.grammar import build_grammar
from jupiter.providers import JupiterUnavailable, Provider, ProviderResult, get_provider
from jupiter.schema import Interpretation, parse_response, schema_for_registry

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


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
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[-1] if "\n" in candidate else candidate
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def interpret(
    request: str,
    *,
    provider: Optional[Provider] = None,
    provider_name: Optional[str] = None,
    config: Optional[dict] = None,
    reg=None,
    use_grammar: bool = True,
) -> InterpretOutcome:
    """Turn one plain-language request into validated SensorDyme rules."""
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
    schema = schema_for_registry(reg)
    grammar = None
    if use_grammar:
        try:
            grammar = build_grammar(reg)
        except ValueError:
            grammar = None  # empty registry; validation still applies

    try:
        result: ProviderResult = backend.complete(system, user, schema=schema, grammar=grammar)
    except JupiterUnavailable as exc:
        return InterpretOutcome(
            interpretation=Interpretation(understood="", rules=[], unsupported=[{
                "request": cleaned,
                "reason": exc.message,
                "requires": "a reachable Jupiter model — see docs/JUPITER.md",
                "support": "unsupported",
                "alternatives": [],
            }]),
            provider=getattr(backend, "name", "unknown"),
            ok=False,
            error=exc.message,
            injection_suspected=injection,
        )

    payload = extract_json(result.text)
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
            "constrained": result.meta.get("constrained", False),
            "rules": len(interpretation.rules),
            "rejected": len(interpretation.unsupported),
        },
    )
