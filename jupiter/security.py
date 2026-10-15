"""Treating Jupiter as untrusted.

The threat here is narrow and worth stating precisely, because it determines how
much defence is warranted. Jupiter has no tools, no shell, no filesystem, and no
network. It reads one string and returns JSON. It cannot execute anything.

So prompt injection cannot make Jupiter *do* something. It can only make Jupiter
*say* something — a rule the user did not ask for. And every rule is then
re-checked by `core.capabilities` and clamped by `_validate_proposal` before it
reaches a device.

That leaves three real risks, which is what this module addresses:

  1. **Nuisance rules.** Injected text persuades Jupiter to emit a valid-but-
     unwanted rule (e.g. severity critical, cooldown 10s, on every object).
     Validation cannot catch this — the rule is legal. The review screen is the
     real defence; capping the blast radius here is the cheap one.
  2. **Resource abuse.** A prompt that makes the model emit hundreds of rules,
     or run until timeout.
  3. **Exfiltration through the explanation.** The `understood` string is shown
     to the user; injected content could carry a phishing instruction into the
     app's UI.

Nothing here pretends to make a language model injection-proof. The security
property SensorDyme actually relies on is that Jupiter's output is data, checked
by code, and never an instruction to anything.
"""

from __future__ import annotations

import re

# Long enough for any genuine monitoring request; short enough that a pasted
# document cannot become the prompt.
MAX_REQUEST_CHARS = 600

# A single request should never produce many rules. Beyond this something has
# gone wrong — a confused model or a deliberate attempt to flood the device.
MAX_RULES = 8
MAX_UNSUPPORTED = 8
MAX_EXPLANATION_CHARS = 400
# A rule description is a notification line, not a paragraph.
MAX_DESCRIPTION_CHARS = 200

# Phrases whose only purpose is to redirect the model. Matching them is not a
# security boundary — trivially bypassed by rephrasing — but it is a cheap
# signal worth logging, and it lets the API answer "no" to the blatant cases
# without spending a model call.
_INJECTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|prior|above|earlier)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)",
    r"forget (all |your |the )?(previous|prior|above|instructions|rules)",
    r"you are now\b",
    r"new (system )?(prompt|instructions?)\b",
    r"system\s*:",
    r"</?(system|assistant|instructions?)>",
    r"reveal (your|the) (system )?(prompt|instructions?)",
    r"print (your|the) (system )?(prompt|instructions?)",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Control characters and the delimiter used to fence the request. Stripping the
# fence matters: without it a request can close the block early and the rest
# reads as top-level instruction.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FENCE_RE = re.compile(r"(<<<REQUEST|REQUEST>>>)", re.IGNORECASE)


class RequestRejected(Exception):  # noqa: N818 - established exception name
    """The request was refused before reaching a model."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def sanitise_request(request: str) -> str:
    """Clean a user request, or refuse it.

    Refuses rather than silently truncating an over-long request: quietly
    interpreting the first 600 characters of a 5,000-character paste would
    produce a rule based on text the user cannot see was used.
    """
    if not isinstance(request, str):
        raise RequestRejected("the request must be text")

    cleaned = _CONTROL_RE.sub("", request).strip()
    if not cleaned:
        raise RequestRejected("the request is empty")
    if len(cleaned) > MAX_REQUEST_CHARS:
        raise RequestRejected(
            f"the request is too long ({len(cleaned)} characters, limit {MAX_REQUEST_CHARS}). "
            "Describe one thing to watch for at a time."
        )
    # Neutralise the fence so a request cannot escape its own block. Repeated
    # until stable: one pass re-forms the fence from its own halves
    # ("REQUESTREQUEST>>>>>>" -> "REQUEST>>>").
    while True:
        stripped = _FENCE_RE.sub("", cleaned)
        if stripped == cleaned:
            return cleaned
        cleaned = stripped


def looks_like_injection(request: str) -> bool:
    """Whether the request contains a blatant instruction-override attempt."""
    return bool(_INJECTION_RE.search(request or ""))


def clamp_response(payload: dict) -> dict:
    """Bound the size and shape of whatever a provider returned.

    Runs before schema validation, so a hostile or broken response cannot cost
    more than it should. Truncation is silent here by design — the caller sees a
    normal (if trimmed) proposal, and the review screen shows exactly what will
    be activated.
    """
    if not isinstance(payload, dict):
        return {"understood": "", "rules": [], "unsupported": []}

    understood = payload.get("understood")
    understood = understood if isinstance(understood, str) else ""
    understood = _CONTROL_RE.sub("", understood)[:MAX_EXPLANATION_CHARS]

    rules = payload.get("rules")
    rules = rules[:MAX_RULES] if isinstance(rules, list) else []

    unsupported = payload.get("unsupported")
    unsupported = unsupported[:MAX_UNSUPPORTED] if isinstance(unsupported, list) else []

    return {
        "understood": understood,
        "rules": [
            _bounded_text_fields(r, {"description": MAX_DESCRIPTION_CHARS})
            for r in rules if isinstance(r, dict)
        ],
        "unsupported": [
            _bounded_text_fields(u, {"request": MAX_EXPLANATION_CHARS,
                                     "reason": MAX_EXPLANATION_CHARS})
            for u in unsupported if isinstance(u, dict)
        ],
    }


def _bounded_text_fields(entry: dict, limits: dict) -> dict:
    """A copy of `entry` with its human-facing strings control-stripped and
    capped. A rule's description becomes the push-notification text, so it
    is as exposed as `understood` and gets the same treatment."""
    bounded = dict(entry)
    for key, limit in limits.items():
        value = bounded.get(key)
        if isinstance(value, str):
            bounded[key] = _CONTROL_RE.sub("", value)[:limit]
    return bounded
