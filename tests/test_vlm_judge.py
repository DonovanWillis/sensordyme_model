"""Unit tests for the mac_max VLM judge — fake client, no network, no credentials needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import main as main_module
from core.vlm_judge import (
    VLM_EVENT_BY_STATE,
    VLM_STATES,
    VLMConfirmTracker,
    VLMJudge,
    VLMVerdict,
    _extract_json,
    vlm_is_local,
)

_FRAME = np.zeros((48, 64, 3), dtype=np.uint8)


def _fake_response(state="phone_use", confidence=0.9, evidence="phone in right hand", stop_reason="end_turn"):
    text = json.dumps({"state": state, "confidence": confidence, "evidence": evidence})
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class _FakeClient:
    """Stands in for anthropic.Anthropic() — records requests, returns scripted responses."""

    def __init__(self, responses=None, error=None):
        self._responses = list(responses or [])
        self._error = error
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._responses.pop(0)


def _make_judge(client, **overrides):
    config = {"model": "claude-opus-4-8", "interval_s": 0.0, "max_consecutive_failures": 3, **overrides}
    return VLMJudge(config, client=client, autostart=False)


# --- verdict parsing ------------------------------------------------------------------


def test_step_parses_a_verdict_and_publishes_it():
    client = _FakeClient(responses=[_fake_response("drowsy", 0.85, "eyes closed, head slumped")])
    judge = _make_judge(client)
    judge._step(_FRAME)

    verdict = judge.latest_verdict()
    assert verdict is not None
    assert verdict.state == "drowsy"
    assert verdict.confidence == 0.85
    assert "eyes closed" in verdict.evidence
    assert verdict.seq == 1


def test_request_carries_image_and_structured_output_schema():
    client = _FakeClient(responses=[_fake_response()])
    judge = _make_judge(client)
    judge._step(_FRAME)

    req = client.requests[0]
    assert req["model"] == "claude-opus-4-8"
    content = req["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    schema = req["output_config"]["format"]["schema"]
    assert set(schema["properties"]["state"]["enum"]) == set(VLM_STATES)


def test_refusal_produces_no_verdict():
    client = _FakeClient(responses=[_fake_response(stop_reason="refusal")])
    judge = _make_judge(client)
    judge._step(_FRAME)
    assert judge.latest_verdict() is None
    assert judge.disabled_reason is None  # a refusal is not a transport failure


def test_out_of_enum_state_is_rejected():
    client = _FakeClient(responses=[_fake_response(state="sleeping")])
    judge = _make_judge(client)
    judge._step(_FRAME)
    assert judge.latest_verdict() is None


def test_confidence_is_clamped_to_unit_interval():
    client = _FakeClient(responses=[_fake_response(confidence=3.7)])
    judge = _make_judge(client)
    judge._step(_FRAME)
    assert judge.latest_verdict().confidence == 1.0


# --- failure handling -----------------------------------------------------------------


def test_missing_credentials_disable_immediately_with_instruction():
    """No credential source at all -> one attempt, clear one-line fix, no retry burn."""
    client = _FakeClient(
        error=TypeError(
            '"Could not resolve authentication method. Expected one of api_key, auth_token, '
            'or credentials to be set."'
        )
    )
    judge = _make_judge(client)
    judge._step(_FRAME)
    assert judge.disabled_reason is not None
    assert "ANTHROPIC_API_KEY" in judge.disabled_reason
    assert "ant auth login" in judge.disabled_reason


def test_judge_disables_after_max_consecutive_failures():
    client = _FakeClient(error=RuntimeError("connection reset"))
    judge = _make_judge(client)
    for _ in range(3):
        judge._step(_FRAME)
    assert judge.disabled_reason is not None
    assert "3 consecutive errors" in judge.disabled_reason


def test_a_success_resets_the_failure_counter():
    client = _FakeClient(responses=[_fake_response()])
    judge = _make_judge(client)
    judge._failures = 2  # two prior failures, one away from disabling
    judge._step(_FRAME)
    assert judge._failures == 0
    assert judge.disabled_reason is None


def test_submit_is_a_noop_once_disabled():
    client = _FakeClient(error=RuntimeError("boom"))
    judge = _make_judge(client, max_consecutive_failures=1)
    judge._step(_FRAME)
    assert judge.disabled_reason is not None
    judge.submit(_FRAME)  # must not raise or queue anything
    assert judge._pending is None


# --- confirmation debounce ------------------------------------------------------------


def _verdict(state, confidence=0.9, seq=1):
    return VLMVerdict(state=state, confidence=confidence, evidence="", seq=seq, at=0.0, latency_s=0.5)


def test_single_dangerous_verdict_does_not_confirm():
    tracker = VLMConfirmTracker(min_confidence=0.7, needed=2)
    assert tracker.update(_verdict("phone_use")) is None
    assert tracker.active_state is None


def test_two_consecutive_same_state_verdicts_confirm_once():
    tracker = VLMConfirmTracker(min_confidence=0.7, needed=2)
    assert tracker.update(_verdict("phone_use", seq=1)) is None
    assert tracker.update(_verdict("phone_use", seq=2)) == "phone_use"
    assert tracker.active_state == "phone_use"
    # A third identical verdict continues the episode but does not re-fire.
    assert tracker.update(_verdict("phone_use", seq=3)) is None


def test_attentive_verdict_resets_the_streak_and_clears_active_state():
    tracker = VLMConfirmTracker(min_confidence=0.7, needed=2)
    tracker.update(_verdict("drowsy", seq=1))
    tracker.update(_verdict("drowsy", seq=2))
    assert tracker.active_state == "drowsy"
    tracker.update(_verdict("attentive", seq=3))
    assert tracker.active_state is None
    # After the reset, confirmation requires a fresh full streak.
    assert tracker.update(_verdict("drowsy", seq=4)) is None
    assert tracker.update(_verdict("drowsy", seq=5)) == "drowsy"


def test_switching_between_dangerous_states_restarts_the_streak():
    tracker = VLMConfirmTracker(min_confidence=0.7, needed=2)
    assert tracker.update(_verdict("phone_use", seq=1)) is None
    assert tracker.update(_verdict("drowsy", seq=2)) is None  # flip-flop never confirms
    assert tracker.update(_verdict("drowsy", seq=3)) == "drowsy"


def test_below_confidence_verdicts_never_confirm():
    tracker = VLMConfirmTracker(min_confidence=0.7, needed=2)
    assert tracker.update(_verdict("phone_use", confidence=0.5, seq=1)) is None
    assert tracker.update(_verdict("phone_use", confidence=0.5, seq=2)) is None
    assert tracker.active_state is None


# --- pluggable backends (ollama / openai_compatible) ----------------------------------


class _FakePost:
    """Stands in for requests.post — records calls, returns a scripted HTTP response."""

    def __init__(self, body, status=200, headers=None):
        self._body = body
        self._status = status
        self._headers = headers or {}
        self.calls = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        body = self._body
        return SimpleNamespace(
            status_code=self._status,
            headers=self._headers,
            json=lambda: body,
            raise_for_status=lambda: None,
        )


def _ollama_body(content):
    return {"message": {"role": "assistant", "content": content}}

def _openai_body(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_ollama_backend_request_shape_and_parse():
    content = json.dumps({"state": "drowsy", "confidence": 0.8, "evidence": "eyes closed"})
    post = _FakePost(_ollama_body(content))
    judge = VLMJudge(
        {"provider": "ollama", "model": "qwen2.5vl:7b", "interval_s": 0.0},
        post=post,
        autostart=False,
    )
    judge._step(_FRAME)

    call = post.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["json"]["model"] == "qwen2.5vl:7b"
    assert call["json"]["stream"] is False
    assert call["json"]["format"]["properties"]["state"]["enum"]  # schema-constrained output
    assert call["json"]["messages"][1]["images"]  # base64 frame attached
    verdict = judge.latest_verdict()
    assert verdict is not None and verdict.state == "drowsy"


def test_ollama_fenced_json_is_still_parsed():
    content = '```json\n{"state": "phone_use", "confidence": 0.9, "evidence": "phone in hand"}\n```'
    post = _FakePost(_ollama_body(content))
    judge = VLMJudge({"provider": "ollama", "interval_s": 0.0}, post=post, autostart=False)
    judge._step(_FRAME)
    assert judge.latest_verdict().state == "phone_use"


def test_openai_compatible_backend_uses_bearer_and_data_uri(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    content = json.dumps({"state": "attentive", "confidence": 0.95, "evidence": "eyes forward"})
    post = _FakePost(_openai_body(content))
    judge = VLMJudge(
        {
            "provider": "openai_compatible",
            "model": "nvidia/some-vision-model",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "interval_s": 0.0,
        },
        post=post,
        autostart=False,
    )
    judge._step(_FRAME)

    call = post.calls[0]
    assert call["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer nvapi-test-key"
    image_part = call["json"]["messages"][1]["content"][0]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert judge.latest_verdict().state == "attentive"


def test_openai_compatible_missing_key_disables_with_instruction(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    post = _FakePost(_openai_body("unused"))
    judge = VLMJudge(
        {
            "provider": "openai_compatible",
            "model": "nvidia/some-vision-model",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "interval_s": 0.0,
        },
        post=post,
        autostart=False,
    )
    judge._step(_FRAME)
    assert judge.disabled_reason is not None
    assert "NVIDIA_API_KEY" in judge.disabled_reason
    assert post.calls == []  # never sent a request without credentials


def test_openai_compatible_401_disables_immediately():
    post = _FakePost(_openai_body("unused"), status=401)
    judge = VLMJudge(
        {"provider": "openai_compatible", "model": "x", "base_url": "http://h/v1", "interval_s": 0.0},
        post=post,
        autostart=False,
    )
    judge._step(_FRAME)
    assert judge.disabled_reason is not None


def test_openai_compatible_403_is_retryable_not_a_dead_key():
    """Google's free tier emits transient 403s under quota pressure — one must not kill the judge."""
    post = _FakePost(_openai_body("unused"), status=403)
    judge = VLMJudge(
        {"provider": "openai_compatible", "model": "x", "base_url": "http://h/v1", "interval_s": 0.0,
         "max_consecutive_failures": 3},
        post=post,
        autostart=False,
    )
    judge._step(_FRAME)
    assert judge.disabled_reason is None  # first 403 -> retry, not disable
    assert judge._failures == 1
    judge._step(_FRAME)
    judge._step(_FRAME)
    assert judge.disabled_reason is not None  # but a persistent 403 still disables eventually


def test_openai_compatible_requires_base_url():
    with pytest.raises(ValueError):
        VLMJudge(
            {"provider": "openai_compatible", "model": "x", "interval_s": 0.0},
            post=_FakePost({}),
            autostart=False,
        )


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        VLMJudge({"provider": "gemini", "interval_s": 0.0}, autostart=False)


def test_vlm_is_local_classification():
    assert vlm_is_local({"provider": "ollama"}) is True
    assert vlm_is_local({"provider": "anthropic"}) is False
    assert vlm_is_local({"provider": "openai_compatible", "base_url": "http://localhost:1234/v1"}) is True
    assert vlm_is_local({"provider": "openai_compatible", "base_url": "https://integrate.api.nvidia.com/v1"}) is False


def test_extract_json_tolerates_prose_around_the_object():
    text = 'Sure! Here is the classification:\n{"state": "drowsy", "confidence": 0.7, "evidence": "x"}\nHope that helps.'
    assert _extract_json(text)["state"] == "drowsy"
    assert _extract_json("no json here") is None
    assert _extract_json(None) is None


# --- burst mode (multi-frame temporal context) -----------------------------------------


def test_burst_request_carries_all_frames_in_order_with_sequence_prompt():
    client = _FakeClient(responses=[_fake_response()])
    judge = _make_judge(client, burst_frames=3)
    judge._step([_FRAME, _FRAME, _FRAME])

    content = client.requests[0]["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image"]
    assert len(image_parts) == 3
    assert content[-1]["type"] == "text"
    assert "3 frames" in content[-1]["text"]
    assert "chronological" in content[-1]["text"]


def test_single_frame_burst_keeps_the_original_prompt():
    client = _FakeClient(responses=[_fake_response()])
    judge = _make_judge(client)
    judge._step(_FRAME)
    content = client.requests[0]["messages"][0]["content"]
    assert len([part for part in content if part["type"] == "image"]) == 1
    assert "this frame" in content[-1]["text"]


def test_submit_collects_a_burst_with_spacing(monkeypatch):
    import core.vlm_judge as vlm_module

    now = [100.0]
    monkeypatch.setattr(vlm_module.time, "monotonic", lambda: now[0])

    client = _FakeClient(responses=[_fake_response()])
    judge = _make_judge(client, interval_s=10.0, burst_frames=3, burst_spacing_s=0.5)
    judge._last_accept = 0.0

    judge.submit(_FRAME)  # t=100.0 -> starts the burst (interval elapsed)
    assert judge._pending is None and len(judge._burst) == 1

    now[0] = 100.2
    judge.submit(_FRAME)  # too soon -> rejected by burst spacing
    assert len(judge._burst) == 1

    now[0] = 100.6
    judge.submit(_FRAME)  # spacing elapsed -> second frame
    assert len(judge._burst) == 2

    now[0] = 101.2
    judge.submit(_FRAME)  # third frame completes the burst
    assert judge._pending is not None and len(judge._pending) == 3
    assert judge._burst == []
    assert judge._wake.is_set()

    now[0] = 101.3
    judge.submit(_FRAME)  # full burst already pending -> no-op
    assert len(judge._pending) == 3 and judge._burst == []


def test_ollama_burst_sends_all_images():
    content = json.dumps({"state": "drowsy", "confidence": 0.8, "evidence": "eyes closed in all frames"})
    post = _FakePost(_ollama_body(content))
    judge = VLMJudge(
        {"provider": "ollama", "interval_s": 0.0, "burst_frames": 3}, post=post, autostart=False
    )
    judge._step([_FRAME, _FRAME, _FRAME])
    assert len(post.calls[0]["json"]["messages"][1]["images"]) == 3


def test_openai_compatible_burst_sends_all_image_parts():
    content = json.dumps({"state": "attentive", "confidence": 0.9, "evidence": "eyes forward throughout"})
    post = _FakePost(_openai_body(content))
    judge = VLMJudge(
        {
            "provider": "openai_compatible",
            "model": "x",
            "base_url": "http://localhost:1234/v1",
            "interval_s": 0.0,
            "burst_frames": 3,
        },
        post=post,
        autostart=False,
    )
    judge._step([_FRAME, _FRAME, _FRAME])
    parts = post.calls[0]["json"]["messages"][1]["content"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 3
    assert parts[-1]["type"] == "text"


# --- wiring ---------------------------------------------------------------------------


def test_every_confirmable_state_maps_to_an_alertable_event_type():
    for state, event_type in VLM_EVENT_BY_STATE.items():
        assert event_type in main_module.ALERT_EVENT_TYPES_MAC_MAX, (state, event_type)
    assert "vlm_judgment" not in main_module.ALERT_EVENT_TYPES_MAC_MAX  # log-only


def test_config_has_the_vlm_block_and_alert_wiring():
    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    vlm = config["mac_max"]["driver_attention"]["vlm"]
    for key in (
        "enabled",
        "model",
        "interval_s",
        "burst_frames",
        "burst_spacing_s",
        "min_confidence",
        "confirm_judgments",
        "max_consecutive_failures",
    ):
        assert key in vlm, f"missing config key: mac_max.driver_attention.vlm.{key}"
    alerts = config["mac_max"]["alerts"]
    assert "vlm_confirmed_phone_use" in alerts["loud_types"]
    assert "vlm_confirmed_drowsiness" in alerts["loud_types"]
    assert "vlm_confirmed_eating" in alerts["soft_types"]
