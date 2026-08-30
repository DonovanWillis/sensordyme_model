"""Phase 8 — the dashboard token must actually be usable, or it protects nothing.

Phase 7 found the dashboard serving the live camera on 0.0.0.0 with no token,
answering 200 to an unauthenticated request from another machine. The config
already supported `dashboard.token` — but turning it on loaded the page and then
401'd every one of the page's own requests, including the video stream, because
the page's JavaScript did not carry the token. A protection nobody can switch on
without breaking the product is not a protection.

These tests pin the parts of that which are checkable off-hardware. The
end-to-end behaviour (401 without / 200 with, from a second machine) was
verified on the device and is recorded in docs/PHASE8_85_POINT_HARDENING.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DASHBOARD = (REPO / "core" / "dashboard.py").read_text()


def test_the_page_carries_the_token_on_its_own_requests():
    """Every fetch the page makes must go through the token-aware helper.

    A bare `fetch("/api/...")` left behind here 401s silently and the card it
    feeds simply stops updating — which looks like a broken dashboard, not like
    a missing token.
    """
    assert "function authFetch" in DASHBOARD, "the token-aware fetch helper is gone"
    bare = re.findall(r'(?<!auth)fetch\("(/api/[^"]+)"', DASHBOARD)
    assert not bare, f"these page requests bypass the token: {bare}"


def test_the_video_stream_carries_the_token_too():
    """An <img> cannot set a header, so the stream needs it as a query param.

    Without this the camera goes black the moment a token is configured — the
    single most visible possible failure, on the element the demo is about.
    """
    assert '"/stream?token=" + encodeURIComponent(TOKEN)' in DASHBOARD, (
        "the MJPEG stream does not carry the token; enabling auth blanks the video"
    )


def test_the_page_opens_with_a_query_string():
    """`/?token=...` is the operator's URL. Matching the raw request line meant
    an exact `== "/"` 404'd it — the one URL the feature requires was the one
    URL that did not work."""
    assert 'bare_path = self.path.split("?", 1)[0]' in DASHBOARD, (
        "the root route compares the raw path, so /?token=... will 404"
    )


def test_both_api_and_stream_are_gated():
    """The stream is the sensitive one: it is the live camera."""
    assert 'self.path.startswith("/api") or self.path.startswith("/stream")' in DASHBOARD, (
        "the token gate no longer covers both /api and /stream"
    )


def test_the_token_is_compared_in_constant_time():
    """A naive == leaks the token's prefix one byte at a time."""
    assert "hmac.compare_digest" in DASHBOARD


def test_the_repo_ships_no_shared_secret():
    """A token committed here would be identical on every unit, which is not a
    secret. It must be null in the repo and set per device."""
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert config["dashboard"]["token"] is None, (
        "config.yaml ships a dashboard token — that value is now public and "
        "identical on every device provisioned from this repo"
    )


def test_the_default_exposure_is_documented_where_someone_will_read_it():
    """Shipping unauthenticated is a defensible default for a bench device and a
    poor one for a conference hall. That trade must be stated at the setting."""
    text = (REPO / "config.yaml").read_text()
    start = text.index("  token: null")
    block = text[start : start + 1200]
    assert "CAMERA STREAM" in block or "camera stream" in block.lower()
    assert "0.0.0.0" in block  # noqa: S104 - asserting the docs mention the bind


def test_the_operator_tooling_reads_the_token():
    """demo-check and demo-cycle poll /api/health. If they do not carry the
    token they report a correctly secured device as broken."""
    preflight = (REPO / "scripts" / "demo_preflight.py").read_text()
    cycle = (REPO / "scripts" / "demo_cycle.sh").read_text()
    assert 'dash.get("token")' in preflight
    assert "X-Device-Token" in preflight
    assert "X-Device-Token" in cycle, "demo_cycle.sh would 401 on a secured device"
