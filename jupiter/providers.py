"""Where Jupiter's reasoning actually runs.

Modelled on `core/vlm_judge.py`'s `_BACKENDS` registry, which already proved the
pattern for vision. Two deliberate departures from it:

  * **Uniform construction.** `vlm_judge` takes `client=` for Anthropic and
    `post=` for everyone else, so adding an SDK-backed provider means a new `if`
    branch at the factory. Here every provider takes `(config)` and exposes one
    method, so a new provider is one class and one registry entry.
  * **Jupiter is the default.** SensorDyme should not need someone else's API
    key to configure a camera the user owns.

The local provider talks HTTP to Ollama or a llama.cpp server rather than
loading weights in-process. That keeps the model out of the API server's memory,
lets it be restarted independently, and means the backend has no heavyweight ML
dependency — `cloud/requirements.txt` stays four lines.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class JupiterUnavailable(Exception):
    """A provider could not answer. Carries a sentence fit to show a user."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


@dataclass
class ProviderResult:
    """One completion, plus what it cost — measured, never estimated."""

    text: str
    provider: str
    model: str
    latency_s: float
    prompt_chars: int = 0
    completion_chars: int = 0
    meta: dict = field(default_factory=dict)


class Provider:
    """One place a prompt can be sent."""

    name = "base"
    #: Whether prompts stay on infrastructure the operator controls.
    local = False

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = dict(config or {})
        self.model = str(self.config.get("model", ""))
        self.timeout_s = float(self.config.get("timeout_s", 60.0))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 1024))

    def complete(self, system: str, user: str, *, schema: Optional[dict] = None,
                 grammar: Optional[str] = None) -> ProviderResult:
        """Return raw model text. Implementations raise JupiterUnavailable."""
        raise NotImplementedError

    def available(self) -> tuple[bool, str]:
        """(reachable, detail) — for health reporting and the eval harness."""
        return True, "assumed available"


# --------------------------------------------------------------------- local
class JupiterLocalProvider(Provider):
    """Jupiter proper: a small instruct model served locally over HTTP.

    Speaks Ollama's /api/chat. A llama.cpp server in Ollama-compatible mode
    works unchanged; for an OpenAI-shaped endpoint use JupiterOpenAICompat.

    Grammar is passed through when the runtime supports it. Ollama exposes
    llama.cpp's GBNF as `options.grammar`; when a runtime ignores it the output
    is still validated downstream, so an unsupported grammar degrades to slower
    and less reliable rather than to broken.
    """

    name = "jupiter"
    local = True

    def __init__(self, config: Optional[dict] = None, post: Optional[Callable] = None) -> None:
        super().__init__(config)
        self.model = self.model or "qwen2.5:1.5b-instruct"
        self.base_url = str(self.config.get("base_url", "http://localhost:11434")).rstrip("/")
        self._post = post or _default_post()
        self._get = self.config.get("_get") or _default_get()

    def available(self) -> tuple[bool, str]:
        try:
            response = self._get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except Exception as exc:  # noqa: BLE001 - absence is an answer
            return False, (
                f"no local model server at {self.base_url} ({exc}). "
                "Install Ollama from ollama.com and run: ollama serve"
            )
        if self.model not in names and not any(n.startswith(self.model) for n in names):
            return False, (
                f"the model server is running but has no '{self.model}' — "
                f"run: ollama pull {self.model}"
            )
        return True, f"{self.model} on {self.base_url}"

    def complete(self, system: str, user: str, *, schema=None, grammar=None) -> ProviderResult:
        options = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if grammar:
            options["grammar"] = grammar

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        # Ollama's own JSON mode, as a second belt when no grammar is supplied.
        if schema is not None and not grammar:
            body["format"] = "json"

        started = time.monotonic()
        try:
            response = self._post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise JupiterUnavailable(
                f"could not reach the local model at {self.base_url}: {exc}"
            ) from exc
        if response.status_code == 404:
            raise JupiterUnavailable(
                f"the local model server has no model '{self.model}' — "
                f"run: ollama pull {self.model}",
                retryable=False,
            )
        response.raise_for_status()
        text = (response.json().get("message") or {}).get("content", "")
        return ProviderResult(
            text=text, provider=self.name, model=self.model,
            latency_s=round(time.monotonic() - started, 3),
            prompt_chars=len(system) + len(user), completion_chars=len(text),
            meta={"grammar": bool(grammar)},
        )


class JupiterOpenAICompat(Provider):
    """Any OpenAI-shaped /chat/completions endpoint — vLLM, LM Studio, llama.cpp.

    `local` is decided by the host, not by the label: a self-hosted vLLM on the
    LAN keeps prompts inside the operator's network, a hosted one does not, and
    the privacy report must not treat them the same.
    """

    name = "jupiter_openai_compat"

    def __init__(self, config: Optional[dict] = None, post: Optional[Callable] = None) -> None:
        super().__init__(config)
        self.base_url = str(self.config.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise JupiterUnavailable(
                "jupiter.base_url is required for the openai_compatible runtime",
                retryable=False,
            )
        self.api_key = self.config.get("api_key") or os.environ.get(
            str(self.config.get("api_key_env", "")), ""
        )
        self._post = post or _default_post()
        self.local = any(
            host in self.base_url.lower() for host in ("localhost", "127.0.0.1", "0.0.0.0")
        )

    def complete(self, system: str, user: str, *, schema=None, grammar=None) -> ProviderResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model or "local-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            response = self._post(
                f"{self.base_url}/chat/completions", json=body,
                headers=headers, timeout=self.timeout_s,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise JupiterUnavailable(f"the model endpoint failed: {exc}") from exc
        return ProviderResult(
            text=text, provider=self.name, model=self.model,
            latency_s=round(time.monotonic() - started, 3),
            prompt_chars=len(system) + len(user), completion_chars=len(text),
        )


# --------------------------------------------------------------------- external
class AnthropicProvider(Provider):
    """Claude. Kept as an OPTIONAL benchmark and fallback, never the default.

    Its purpose now is to be the thing Jupiter is measured against. Prompts
    leave the operator's infrastructure when this is selected, which is why it
    reports `local = False` and why selecting it is explicit.
    """

    name = "anthropic"

    def __init__(self, config: Optional[dict] = None, client: Optional[object] = None) -> None:
        super().__init__(config)
        self.model = self.model or os.environ.get("SENSORDYME_AI_MODEL", "claude-opus-5")
        self._client = client

    def available(self) -> tuple[bool, str]:
        if not (self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")):
            return False, "ANTHROPIC_API_KEY is not set"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            # The exact failure that silently disabled Configure-with-AI on
            # deployed backends: imported by the app, absent from
            # cloud/requirements.txt.
            return False, "the `anthropic` package is not installed (pip install anthropic)"
        return True, self.model

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise JupiterUnavailable(
                    "the `anthropic` package is not installed", retryable=False
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, *, schema=None, grammar=None) -> ProviderResult:
        client = self._ensure_client()
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema is not None:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": schema},
                "effort": "low",
            }
        started = time.monotonic()
        try:
            response = client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001
            raise JupiterUnavailable(f"the Anthropic API failed: {exc}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            raise JupiterUnavailable("the model declined to answer", retryable=False)
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
        return ProviderResult(
            text=text, provider=self.name, model=self.model,
            latency_s=round(time.monotonic() - started, 3),
            prompt_chars=len(system) + len(user), completion_chars=len(text),
        )


class ScriptedProvider(Provider):
    """Returns canned text. For tests and for exercising the pipeline with no
    model installed — the eval harness runs end to end against this."""

    name = "scripted"
    local = True

    def __init__(self, config: Optional[dict] = None, responses: Optional[dict] = None) -> None:
        super().__init__(config)
        self.responses = responses or {}
        self.default = self.config.get("default", '{"understood":"","rules":[],"unsupported":[]}')
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, schema=None, grammar=None) -> ProviderResult:
        self.calls.append((system, user))
        text = self.default
        for needle, canned in self.responses.items():
            if needle.lower() in user.lower():
                text = canned if isinstance(canned, str) else json.dumps(canned)
                break
        return ProviderResult(
            text=text, provider=self.name, model="scripted",
            latency_s=0.0, prompt_chars=len(system) + len(user), completion_chars=len(text),
        )


_PROVIDERS = {
    "jupiter": JupiterLocalProvider,
    "jupiter_openai_compat": JupiterOpenAICompat,
    "anthropic": AnthropicProvider,
    "scripted": ScriptedProvider,
}


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: Optional[str] = None, config: Optional[dict] = None) -> Provider:
    """Build a provider by name. Defaults to Jupiter."""
    name = (name or "jupiter").strip().lower()
    factory = _PROVIDERS.get(name)
    if factory is None:
        raise JupiterUnavailable(
            f"unknown AI provider '{name}' (available: {', '.join(available_providers())})",
            retryable=False,
        )
    return factory(config or {})


def register_provider(name: str, factory) -> None:
    """Extension point: a new runtime is one class and one call."""
    _PROVIDERS[name.strip().lower()] = factory


# --------------------------------------------------------------------- http
def _default_post():
    import requests

    return requests.post


def _default_get():
    import requests

    return requests.get
