"""Role-based model routing for Jupiter.

Jupiter has exactly two things it asks a model to do today, and they have
opposite requirements:

  configure   Turn one sentence into a structured monitoring rule
              (jupiter/interpret.py). Happens once per user action, not per
              frame — accuracy matters more than speed, and a wrong target or
              predicate ships a rule that silently never fires. Measured on
              this repo's own eval harness (40 generated cases, seed 1):
              qwen2.5:1.5b-instruct got the target wrong on 22% of supported
              requests; qwen2.5:7b-instruct got 0 wrong, at a latency cost of
              about 1s -> 5.4s mean. One-shot and not real-time, so that trade
              is worth it — see jupiter/eval/harness.py to re-measure it.

  explain     Paraphrase an already-computed, deterministic fact list into one
              sentence (jupiter/explain.py). Called once per fired event, in a
              notification-adjacent path where latency is user-visible, and
              the model is never the source of truth — explain() already
              returns a complete, correct answer with no model at all. Speed
              matters more than the last percent of fluency here.

A role is a name, not a hardcoded model string at each call site. What
resolves it, in priority order:

  1. A role-specific environment override (JUPITER_ROLE_CONFIGURE_MODEL /
     JUPITER_ROLE_EXPLAIN_MODEL, plus the _PROVIDER/_BASE_URL variants).
  2. An optional YAML file (JUPITER_CONFIG_PATH, default jupiter_config.yaml
     at the repo root) shaped like:

         roles:
           configure: {provider: jupiter, model: qwen2.5:7b-instruct}
           explain:   {provider: jupiter, model: qwen2.5:1.5b-instruct}
         cloud_fallback:
           enabled: false
           provider: anthropic
           model: claude-sonnet-5

  3. The existing repo-wide overrides (SENSORDYME_AI_PROVIDER, JUPITER_MODEL,
     JUPITER_BASE_URL) that cloud/app.py has always read — kept working
     unchanged so a deployment that only ever set one model for everything
     does not have to change anything.
  4. Built-in defaults matching the measurements above.

None of this is a claim that Jupiter routes vision, embeddings, or web search
today — it does not, and giving this file a bigger name than that would be
exactly the kind of unimplemented claim it exists to avoid. It is a text-only
router over the two roles Jupiter actually has.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from jupiter.providers import Provider, get_provider

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "jupiter_config.yaml"

# Built-in defaults — used when neither a role env override nor the YAML file
# names something for this role. Never Opus, and never any paid API by
# default: both roles default to the local, zero-cost provider.
_ROLE_DEFAULTS = {
    "configure": {"provider": "jupiter", "model": "qwen2.5:7b-instruct"},
    "explain": {"provider": "jupiter", "model": "qwen2.5:1.5b-instruct"},
}

# The cloud fallback's own default model. NOT claude-opus-5: Configure-with-AI
# is schema-constrained rule extraction, not open-ended reasoning, and Sonnet
# is materially cheaper for that job — see docs/JUPITER.md for the per-request
# cost comparison. Still off by default either way; see cloud_fallback_enabled().
_CLOUD_FALLBACK_DEFAULT_MODEL = "claude-sonnet-5"

_ROLE_ENV_PREFIX = {
    "configure": "JUPITER_ROLE_CONFIGURE",
    "explain": "JUPITER_ROLE_EXPLAIN",
}


def _load_yaml(path: Optional[Path]) -> dict:
    path = Path(path) if path is not None else Path(
        os.environ.get("JUPITER_CONFIG_PATH", "") or DEFAULT_CONFIG_PATH
    )
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        # A broken routing file must not take Configure-with-AI down; fall
        # back to the same built-in defaults a missing file would use.
        return {}
    # Valid YAML of the wrong shape (a top-level list) is just as broken.
    return data if isinstance(data, dict) else {}


def _mapping(value) -> dict:
    """`value` if it is a mapping, else {} — e.g. `roles: {configure: "qwen"}`
    or `cloud_fallback: true`, which parse fine and are not blocks."""
    return value if isinstance(value, dict) else {}


def role_config(role: str, *, path: Optional[Path] = None) -> tuple[str, dict]:
    """Resolve (provider_name, provider_config) for one Jupiter role.

    Merge order, highest priority first: role-specific env vars, the YAML
    file's `roles.<role>` block, the repo-wide env vars every caller already
    sets, then the built-in default for that role.
    """
    if role not in _ROLE_DEFAULTS:
        raise ValueError(f"unknown Jupiter role {role!r} (known: {sorted(_ROLE_DEFAULTS)})")

    yaml_roles = _mapping(_load_yaml(path).get("roles"))
    yaml_role = _mapping(yaml_roles.get(role))
    default = _ROLE_DEFAULTS[role]
    env_prefix = _ROLE_ENV_PREFIX[role]

    provider_name = (
        os.environ.get(f"{env_prefix}_PROVIDER", "").strip()
        or str(yaml_role.get("provider", "")).strip()
        or os.environ.get("SENSORDYME_AI_PROVIDER", "").strip()
        or default["provider"]
    ).lower()

    # The built-in model belongs to the built-in provider. Pairing it with any
    # other one built e.g. AnthropicProvider(model="qwen2.5:7b-instruct") for a
    # deployment that only set SENSORDYME_AI_PROVIDER=anthropic, so every call
    # failed. Left empty, each provider applies its own default model.
    model = (
        os.environ.get(f"{env_prefix}_MODEL", "").strip()
        or str(yaml_role.get("model", "")).strip()
        or os.environ.get("JUPITER_MODEL", "").strip()
        or (default["model"] if provider_name == default["provider"] else "")
    )

    base_url = (
        os.environ.get(f"{env_prefix}_BASE_URL", "").strip()
        or str(yaml_role.get("base_url", "")).strip()
        or os.environ.get("JUPITER_BASE_URL", "").strip()
    )

    config: dict = {"model": model}
    if base_url:
        config["base_url"] = base_url
    return provider_name, config


def get_role_provider(role: str, *, path: Optional[Path] = None) -> Provider:
    """Build the provider that should answer this role's requests."""
    provider_name, config = role_config(role, path=path)
    return get_provider(provider_name, config)


def cloud_fallback_config(*, path: Optional[Path] = None) -> dict:
    """Whether/how the optional cloud fallback is configured, for the
    `configure` role only — the one place a fallback exists today
    (cloud/app.py's SENSORDYME_ALLOW_EXTERNAL_AI gate).

    Off by default. The env var that already gates this
    (SENSORDYME_ALLOW_EXTERNAL_AI) keeps working unchanged; the YAML file's
    `cloud_fallback.enabled` is an alternative way to turn it on, not a second
    independent switch — either one enables it.
    """
    yaml_cloud = _mapping(_load_yaml(path).get("cloud_fallback"))
    enabled = bool(yaml_cloud.get("enabled", False)) or os.environ.get(
        "SENSORDYME_ALLOW_EXTERNAL_AI", ""
    ).strip().lower() in ("1", "true", "yes")
    provider_name = str(yaml_cloud.get("provider", "")).strip() or "anthropic"
    model = (
        os.environ.get("SENSORDYME_AI_MODEL", "").strip()
        or str(yaml_cloud.get("model", "")).strip()
        or _CLOUD_FALLBACK_DEFAULT_MODEL
    )
    return {"enabled": enabled, "provider": provider_name, "model": model}
