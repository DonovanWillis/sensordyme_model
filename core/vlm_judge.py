"""Frontier-VLM judge for the mac_max profile: a vision-language model as the slow, precise tier.

Architecture rationale (the same "VLM as verifier, not hot path" split used by
enterprise driver-monitoring stacks): the local pipeline (Face Mesh, gaze, YOLO)
keeps the fast reflexes — sub-100ms, every frame — while a VLM periodically
judges a whole frame with scene-level understanding ("is that a phone or a
wallet? is the driver actually asleep or just looking down?"). A single VLM
call takes seconds, so it can never sit in the per-frame loop; instead
`VLMJudge` runs in a daemon thread, accepts at most one pending frame at a
time, and publishes its latest verdict for the module to fuse on its own
schedule.

The judge is BACKEND-PLUGGABLE via `vlm.provider` in config.yaml:

  * ``anthropic`` (default) — Claude via the Anthropic API. The accuracy
    ceiling of the three; frames are transmitted to the API.
  * ``ollama`` — a local model served by Ollama (e.g. Qwen2.5-VL), running
    entirely on this machine. Frames never leave the device; slower and less
    accurate than the frontier option, free to run.
  * ``openai_compatible`` — any OpenAI-compatible /chat/completions endpoint:
    NVIDIA NIM (build.nvidia.com), LM Studio, vLLM, etc. Local or cloud
    depending on `base_url`.

Alerting policy lives in `VLMConfirmTracker`: a dangerous verdict must repeat on
consecutive judgments (default 2) at or above a confidence floor before the
module fires a `vlm_confirmed_*` event — one flaky classification never alerts.

Privacy: with a remote provider, enabling this tier TRANSMITS in-memory JPEG
frames to that provider's API for classification. Frames are still never
written to disk anywhere; with a local provider (ollama / a localhost
base_url) everything stays on-device. The module prints the applicable notice
at startup, and the pi profile is entirely unaffected (it never constructs
this class).
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

import cv2
import numpy as np

# The closed set of states the judge may answer with.
VLM_STATES = ("attentive", "distracted", "drowsy", "phone_use", "eating_drinking")
DANGEROUS_STATES = frozenset({"distracted", "drowsy", "phone_use", "eating_drinking"})

# Event type emitted per confirmed dangerous state (wired into alerts in config.yaml).
VLM_EVENT_BY_STATE = {
    "drowsy": "vlm_confirmed_drowsiness",
    "distracted": "vlm_confirmed_distraction",
    "phone_use": "vlm_confirmed_phone_use",
    "eating_drinking": "vlm_confirmed_eating",
}

_SYSTEM_PROMPT = (
    "You are the verification judge inside a driver-monitoring system. Each request contains "
    "one or more frames from a driver-facing in-cabin camera; when there are several, they "
    "are consecutive and in chronological order. Classify the driver's CURRENT state:\n"
    "- attentive: eyes on the road, hands on the task of driving. A phone mounted on the "
    "dashboard that the driver is not touching or looking at counts as attentive.\n"
    "- distracted: gaze or head clearly and substantially away from the road (not a brief "
    "mirror or blindspot check).\n"
    "- drowsy: eyes closed or heavy-lidded, head nodding or slumped.\n"
    "- phone_use: a phone held in the hand, being touched, or being looked at.\n"
    "- eating_drinking: actively bringing food or a drink to the mouth.\n"
    "When several frames are present, use the motion across them: eyes closed in every frame "
    "is a sustained closure (drowsy), eyes closed in only one frame is a normal blink "
    "(attentive); a head turned away in every frame is distraction, in only one frame it is "
    "likely a mirror check.\n"
    "Judge only what is visible. Be conservative: prefer attentive unless the evidence is "
    "clear, and keep the evidence description to one short sentence."
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": list(VLM_STATES)},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["state", "confidence", "evidence"],
    "additionalProperties": False,
}

def _user_question(frame_count: int) -> str:
    """The per-request user text, adjusted for how many frames the burst carries."""
    if frame_count <= 1:
        return "Classify the driver's state in this frame."
    return (
        f"These {frame_count} frames were captured in chronological order, a fraction of a "
        "second apart. Classify the driver's state across the sequence."
    )

# Appended for backends without server-enforced structured output.
_JSON_INSTRUCTION = (
    '\nAnswer with ONLY a JSON object, no prose and no code fences: '
    '{"state": "<one of attentive|distracted|drowsy|phone_use|eating_drinking>", '
    '"confidence": <0..1>, "evidence": "<one short sentence>"}'
)


class VLMBackendAuthError(RuntimeError):
    """Credentials are missing or rejected — disables the judge immediately with its message."""


class VLMBackendRateLimited(RuntimeError):  # noqa: N818 - established exception name
    """The provider rate-limited us; carries how long to back off."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class VLMUnusableVerdict(RuntimeError):  # noqa: N818 - established exception name
    """The backend answered, but not with a usable verdict (empty, not JSON, or a
    state outside VLM_STATES). Counts toward max_consecutive_failures: a judge
    that only ever returns this is broken, and must say so rather than sit on
    "warming up" for ever."""


#: Where the Ollama backend connects when vlm.base_url is not set.
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def _is_loopback_url(url: str) -> bool:
    """Whether `url`'s HOST is this machine. Parsed, not substring-matched:
    "https://localhost.example.com" and "https://api.example.com/?via=localhost"
    both contain "localhost" and both send frames off the machine."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False  # any other name is resolved elsewhere: assume it leaves the machine
    # 0.0.0.0 / :: as a destination reach this host on Linux and macOS.
    return address.is_loopback or address.is_unspecified


def vlm_is_local(config: dict) -> bool:
    """Whether the configured provider keeps frames on this machine (no network egress).

    Decided from the host the backend will actually connect to. Ollama is local
    only at its default address — it is commonly run on a separate GPU box, and
    the Ollama backend honours vlm.base_url, so `provider: ollama` alone used to
    print "frames stay ON-DEVICE" while they went over the network.
    """
    provider = config.get("provider", "anthropic")
    if provider == "ollama":
        return _is_loopback_url(config.get("base_url") or _OLLAMA_DEFAULT_BASE)
    if provider == "openai_compatible":
        return _is_loopback_url(config.get("base_url") or "")
    return False


def _extract_json(text: Optional[str]) -> Optional[dict]:
    """Parse a JSON object out of model output, tolerating code fences and surrounding prose."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _default_post() -> Callable:
    import requests

    return requests.post


# ---------------------------------------------------------------------------- backends
class _AnthropicBackend:
    """Claude via the Anthropic API — structured outputs guarantee a parseable verdict."""

    def __init__(self, config: dict, client: Optional[object] = None) -> None:
        self._model = config.get("model", "claude-opus-4-8")
        # Configurable for the same reason as the openai_compatible backend: a
        # model that thinks before answering spends this budget first, and an
        # exhausted budget yields no verdict text at all.
        self._max_tokens = int(config.get("max_tokens", 300))
        self._client = client
        self._anthropic = None  # the imported module, for typed exception checks

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise VLMBackendAuthError(
                    "the `anthropic` package is not installed — run: pip install anthropic"
                ) from exc
            self._anthropic = anthropic
            # Zero-arg client: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
            # or an `ant auth login` profile from the environment.
            self._client = anthropic.Anthropic()
        return self._client

    def judge(self, images_b64: list[str]) -> Optional[str]:
        """One judgment over a burst of frames. Returns the raw JSON text, or None on a safety refusal."""
        client = self._ensure_client()
        content: list[dict] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
            }
            for image_b64 in images_b64
        ]
        content.append({"type": "text", "text": _user_question(len(images_b64))})
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,  # default 300: the answer is a tiny JSON verdict
                system=_SYSTEM_PROMPT,
                output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            an = self._anthropic
            # Two credential failure shapes: the SDK raises TypeError("Could not resolve
            # authentication method...") client-side when NO credential source exists at
            # all, and AuthenticationError (401) when a credential exists but is invalid.
            if "Could not resolve authentication method" in str(exc) or (
                an is not None and isinstance(exc, an.AuthenticationError)
            ):
                raise VLMBackendAuthError(
                    "no valid Anthropic credentials — set ANTHROPIC_API_KEY or run `ant auth login`, "
                    "then restart. Local detection continues unaffected."
                ) from exc
            if an is not None and isinstance(exc, an.RateLimitError):
                retry_after = 30.0
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    retry_after = float(exc.response.headers.get("retry-after", "30"))
                raise VLMBackendRateLimited("Anthropic API rate limited", retry_after) from exc
            raise
        if getattr(response, "stop_reason", None) == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), "")


class _OllamaBackend:
    """A local model served by Ollama (e.g. `ollama pull qwen2.5vl:7b`) — fully on-device."""

    def __init__(self, config: dict, post: Optional[Callable] = None) -> None:
        self._model = config.get("model", "qwen2.5vl:7b")
        base = (config.get("base_url") or _OLLAMA_DEFAULT_BASE).rstrip("/")
        self._url = base + "/api/chat"
        self._conn_error_cls: Optional[type] = None
        if post is None:
            import requests

            post = requests.post
            self._conn_error_cls = requests.exceptions.ConnectionError
        self._post = post

    def judge(self, images_b64: list[str]) -> Optional[str]:
        payload = {
            "model": self._model,
            "stream": False,
            "format": _OUTPUT_SCHEMA,  # Ollama structured outputs: schema-constrained JSON
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_question(len(images_b64)), "images": images_b64},
            ],
        }
        try:
            # Generous timeout: a 7B VLM on Apple Silicon can take tens of seconds,
            # especially on the first call while the model loads.
            response = self._post(self._url, json=payload, timeout=180)
        except Exception as exc:
            if self._conn_error_cls is not None and isinstance(exc, self._conn_error_cls):
                raise RuntimeError(
                    f"cannot reach Ollama at {self._url} — is it running? "
                    f"(install from ollama.com, then: ollama pull {self._model})"
                ) from exc
            raise
        if response.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model '{self._model}' — run: ollama pull {self._model}"
            )
        response.raise_for_status()
        return response.json()["message"]["content"]


class _OpenAICompatBackend:
    """Any OpenAI-compatible /chat/completions server: NVIDIA NIM, LM Studio, vLLM, ..."""

    def __init__(self, config: dict, post: Optional[Callable] = None) -> None:
        base = (config.get("base_url") or "").rstrip("/")
        if not base:
            raise ValueError(
                "vlm.base_url is required for provider: openai_compatible "
                "(e.g. https://integrate.api.nvidia.com/v1 for NVIDIA NIM)"
            )
        self._url = base + "/chat/completions"
        self._model = config.get("model", "")
        self._key_env = config.get("api_key_env")
        # Models with hidden reasoning (e.g. Gemini 3.5 Flash via the OpenAI-compat
        # endpoint) spend thinking tokens against max_tokens before any visible output;
        # too small a budget yields an empty verdict. Raise via vlm.max_tokens.
        self._max_tokens = int(config.get("max_tokens", 300))
        self._post = post if post is not None else _default_post()

    def judge(self, images_b64: list[str]) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        if self._key_env:
            api_key = os.environ.get(self._key_env)
            if not api_key:
                raise VLMBackendAuthError(
                    f"{self._key_env} is not set — export it before running "
                    "(for NVIDIA NIM, create a key at build.nvidia.com). "
                    "Local detection continues unaffected."
                )
            headers["Authorization"] = f"Bearer {api_key}"
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
            for image_b64 in images_b64
        ]
        content.append({"type": "text", "text": _user_question(len(images_b64))})
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "messages": [
                # No server-enforced schema here, so the JSON contract rides in the prompt
                # and _extract_json tolerates fences/prose on the way back.
                {"role": "system", "content": _SYSTEM_PROMPT + _JSON_INSTRUCTION},
                {"role": "user", "content": content},
            ],
        }
        response = self._post(self._url, json=payload, headers=headers, timeout=120)
        if response.status_code == 401:
            raise VLMBackendAuthError(
                f"authentication rejected by {self._url} — check the "
                f"{self._key_env or 'API key'} value. Local detection continues unaffected."
            )
        if response.status_code == 403:
            # NOT treated as a permanently-bad key: Google's free tier returns transient
            # 403 PERMISSION_DENIED under quota pressure (a truly invalid key is a 400
            # there). Raise as retryable — a real 403 still disables the judge after
            # max_consecutive_failures in a row.
            raise RuntimeError(
                f"403 from {self._url} — transient quota/permission denial "
                f"(or check the {self._key_env or 'API key'} value if it persists)"
            )
        if response.status_code == 429:
            retry_after = 30.0
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                retry_after = float(response.headers.get("retry-after", "30"))
            raise VLMBackendRateLimited("provider rate limited", retry_after)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        content = message.get("content")
        if content is None and message.get("refusal"):
            return None  # an explicit refusal, like Anthropic's stop_reason "refusal"
        if isinstance(content, list):  # some servers return content as typed parts
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        # A null content with no refusal (e.g. hidden reasoning exhausted
        # max_tokens) is an EMPTY answer, which _classify counts as a failure.
        return content if isinstance(content, str) else ""


_BACKENDS = {
    "anthropic": _AnthropicBackend,
    "ollama": _OllamaBackend,
    "openai_compatible": _OpenAICompatBackend,
}


# ---------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class VLMVerdict:
    """One judgment from the VLM.

    Attributes:
        state: One of VLM_STATES.
        confidence: Judge-reported confidence, clamped to [0, 1].
        evidence: One-sentence description of what the judge saw.
        seq: Monotonically increasing verdict counter (for new-verdict detection).
        at: Monotonic time the verdict was received.
        latency_s: Round-trip time of the API call.
    """

    state: str
    confidence: float
    evidence: str
    seq: int
    at: float
    latency_s: float


class VLMConfirmTracker:
    """Requires N consecutive same-state dangerous verdicts before confirming an alert.

    Fires once per continuous episode of a given state; an `attentive` (or
    below-confidence) verdict resets everything, and switching between two
    dangerous states restarts the streak — a judge that flip-flops never alerts.
    """

    def __init__(self, min_confidence: float = 0.7, needed: int = 2) -> None:
        self.min_confidence = min_confidence
        self.needed = needed
        self._streak_state: Optional[str] = None
        self._streak = 0
        self._fired_state: Optional[str] = None
        self.active_state: Optional[str] = None

    def update(self, verdict: VLMVerdict) -> Optional[str]:
        """Feed one verdict. Returns the state name the moment it is first confirmed."""
        if verdict.state in DANGEROUS_STATES and verdict.confidence >= self.min_confidence:
            if verdict.state == self._streak_state:
                self._streak += 1
            else:
                self._streak_state = verdict.state
                self._streak = 1
            if self._streak >= self.needed:
                self.active_state = verdict.state
                if self._fired_state != verdict.state:
                    self._fired_state = verdict.state
                    return verdict.state
        else:
            self.reset()
        return None

    def reset(self) -> None:
        """Forget the streak and any confirmed state.

        Called by the module when there is no CURRENT verdict to stand on (the
        judge disabled itself, or stopped answering): a confirmed state is a
        claim about the driver now, and it must not outlive its evidence.
        """
        self._streak_state = None
        self._streak = 0
        self._fired_state = None
        self.active_state = None


# ---------------------------------------------------------------------------- judge
class VLMJudge:
    """Async VLM judge: submit frames from the video loop, poll verdicts back.

    Never blocks the caller: `submit()` copies at most one frame per `interval_s`
    into a single-slot mailbox; a daemon worker thread encodes it to JPEG in
    memory, calls the configured backend, and publishes the parsed verdict.
    Credential problems disable the judge with a one-line instruction; repeated
    transport errors disable it after `max_consecutive_failures` — the local
    pipeline is never affected either way.
    """

    def __init__(
        self,
        config: dict,
        client: Optional[object] = None,
        post: Optional[Callable] = None,
        autostart: bool = True,
    ) -> None:
        """Create the judge from the `vlm:` config block.

        Args:
            config: The mac_max `vlm` config dict (provider, model, interval_s, ...).
            client: Injected Anthropic-compatible client (tests, anthropic provider only).
            post: Injected HTTP POST callable (tests, ollama/openai_compatible providers).
            autostart: Start the worker thread (tests pass False and drive `_step`).
        """
        provider = config.get("provider", "anthropic")
        if provider not in _BACKENDS:
            raise ValueError(f"unknown vlm.provider '{provider}' — one of {sorted(_BACKENDS)}")
        if provider == "anthropic":
            self._backend = _AnthropicBackend(config, client=client)
        else:
            self._backend = _BACKENDS[provider](config, post=post)
        self.provider = provider
        self.model = config.get("model", "claude-opus-4-8")

        self._interval_s = float(config.get("interval_s", 4.0))
        self._jpeg_quality = int(config.get("jpeg_quality", 80))
        self._max_failures = int(config.get("max_consecutive_failures", 5))
        # Burst mode: send burst_frames frames (captured burst_spacing_s apart) per
        # judgment, so the judge sees motion — a blink vs. a sustained closure, a mirror
        # check vs. a stare. burst_frames=1 is the original single-frame behavior.
        self._burst_frames = max(1, int(config.get("burst_frames", 1)))
        self._burst_spacing_s = float(config.get("burst_spacing_s", 0.7))

        self._lock = threading.Lock()
        self._pending: Optional[list[np.ndarray]] = None
        self._burst: list[np.ndarray] = []
        self._burst_last_t = 0.0
        self._verdict: Optional[VLMVerdict] = None
        self._seq = 0
        self._failures = 0
        self._disabled_reason: Optional[str] = None
        self._last_accept = 0.0

        self._wake = threading.Event()
        self._stop = threading.Event()
        if autostart:
            self._thread = threading.Thread(target=self._run, daemon=True, name="vlm-judge")
            self._thread.start()

    # ------------------------------------------------------------------ public
    @property
    def disabled_reason(self) -> Optional[str]:
        """Why the judge shut itself off, or None while it's operating."""
        return self._disabled_reason

    def submit(self, frame: np.ndarray) -> None:
        """Offer the current frame for judgment. Non-blocking; self-throttled to interval_s.

        With burst_frames > 1, an accepted frame starts a burst: subsequent frames are
        collected burst_spacing_s apart until the burst is full, then the whole burst is
        handed to the worker as one request.
        """
        if self._disabled_reason is not None or self._stop.is_set():
            return
        now = time.monotonic()
        with self._lock:
            if self._pending is not None:
                return  # a full burst is already waiting for the worker
            if not self._burst:
                if now - self._last_accept < self._interval_s:
                    return
                self._last_accept = now
            elif now - self._burst_last_t < self._burst_spacing_s:
                return
            self._burst.append(frame.copy())
            self._burst_last_t = now
            if len(self._burst) >= self._burst_frames:
                self._pending = self._burst
                self._burst = []
                self._wake.set()

    def latest_verdict(self) -> Optional[VLMVerdict]:
        """The most recent verdict, or None before the first one lands."""
        with self._lock:
            return self._verdict

    def verdict_lifetime_s(self, verdict: VLMVerdict) -> float:
        """How long `verdict` stays current: two full judgment cycles (interval,
        burst capture and the call's own latency), so one slow or failed call
        never expires a live verdict but a judge that has stopped answering —
        network down, rate-limit backoff, or disabled — does."""
        burst_span = (self._burst_frames - 1) * self._burst_spacing_s
        return 2.0 * (self._interval_s + burst_span + max(verdict.latency_s, 1.0))

    def is_current(self, verdict: VLMVerdict, now: Optional[float] = None) -> bool:
        """Whether `verdict` still describes the driver: the judge is running and
        the verdict is younger than verdict_lifetime_s()."""
        if self._disabled_reason is not None or self._stop.is_set():
            return False
        now = time.monotonic() if now is None else now
        return now - verdict.at <= self.verdict_lifetime_s(verdict)

    def close(self) -> None:
        """Stop the worker thread. Idempotent; never blocks on an in-flight call
        (the worker is a daemon and exits after it returns)."""
        self._stop.set()
        self._wake.set()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                return
            with self._lock:
                frames = self._pending
                self._pending = None
                self._wake.clear()
            if frames is not None:
                self._step(frames)

    def _step(self, frames) -> None:
        """Judge one burst with full error handling; safe to call directly in tests.

        Accepts a single frame (ndarray) or a list of frames — tests and the
        single-frame path pass one ndarray; burst mode passes a list.
        """
        if isinstance(frames, np.ndarray):
            frames = [frames]
        try:
            verdict = self._classify(frames)
        except Exception as exc:  # noqa: BLE001 - worker thread must never die on an API error
            self._handle_error(exc)
            return
        if verdict is not None:
            with self._lock:
                self._verdict = verdict
            self._failures = 0

    def _classify(self, frames: list[np.ndarray]) -> Optional[VLMVerdict]:
        """One synchronous judgment over a burst.

        Returns None on a refusal (not a failure: the judge declined, it is not
        broken). Raises VLMUnusableVerdict when the backend answered with
        nothing usable — empty text, no JSON, or a state outside VLM_STATES — so
        it counts toward max_consecutive_failures instead of being dropped
        silently for ever.
        """
        images_b64: list[str] = []
        for frame in frames:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if ok:
                images_b64.append(base64.standard_b64encode(buf.tobytes()).decode("utf-8"))
        if not images_b64:
            return None

        t0 = time.perf_counter()
        text = self._backend.judge(images_b64)
        latency = time.perf_counter() - t0
        if text is None:
            return None  # a refusal

        data = _extract_json(text)
        if data is None:
            raise VLMUnusableVerdict(f"no JSON verdict in the answer: {str(text)[:80]!r}")
        state = data.get("state")
        if state not in VLM_STATES:
            raise VLMUnusableVerdict(f"state {state!r} is not one of {', '.join(VLM_STATES)}")
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        self._seq += 1
        return VLMVerdict(
            state=state,
            confidence=confidence,
            evidence=str(data.get("evidence", ""))[:200],
            seq=self._seq,
            at=time.monotonic(),
            latency_s=latency,
        )

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, VLMBackendAuthError):
            self._disable(str(exc))
            return
        if isinstance(exc, VLMBackendRateLimited):
            # Push the next accept out past the rate-limit window instead of hammering.
            self._last_accept = time.monotonic() + min(exc.retry_after, 120.0)
            print(f"vlm_judge: rate limited — backing off {exc.retry_after:.0f}s")
            return
        self._failures += 1
        print(f"vlm_judge: judgment failed ({self._failures}/{self._max_failures}): {exc}")
        if self._failures >= self._max_failures:
            self._disable(f"{self._max_failures} consecutive errors — last: {exc}")

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            self._disabled_reason = reason
            print(f"vlm_judge: DISABLED — {reason}")
