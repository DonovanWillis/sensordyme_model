"""M8.3 security finding (integrator): no caching of device data, no token in a Referer.

* every /api JSON answer and /stream carry `Cache-Control: no-store` -- device
  state, history, health and the camera must not be kept by a browser cache or
  an intermediate proxy;
* the `/` and `/setup` HTML pages, whose URLs carry `?token=` (an operator
  opens them that way), send `Referrer-Policy: no-referrer` so the token never
  leaves in a Referer header.

Auth behaviour is unchanged (tests/test_m6_security_device.py,
tests/test_m83_stream_device.py::test_stream_auth_is_unchanged).
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from core.dashboard import DashboardServer
from sim.stream_client import StreamClient

TOKEN = "m83-headers-token"


@pytest.fixture
def server():
    dashboard = DashboardServer(host="127.0.0.1", port=0, token=TOKEN)
    dashboard.start()
    yield dashboard
    dashboard.close()


def _headers(url: str, token: bool = True, method: str = "GET", body: bytes | None = None):
    headers = {"X-Device-Token": TOKEN} if token else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)  # noqa: S310 - local test server
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status, response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers


@pytest.mark.parametrize("path", ["/api/state", "/api/presets", "/api/health", "/api/history", "/api/stream/stats"])
def test_api_json_is_never_cached(server, path):
    status, headers = _headers(f"http://127.0.0.1:{server._port}{path}")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"


def test_refusals_and_post_answers_are_not_cached_either(server):
    base = f"http://127.0.0.1:{server._port}"
    status, headers = _headers(base + "/api/state", token=False)
    assert status == 401 and headers["Cache-Control"] == "no-store"
    status, headers = _headers(base + "/api/preset", method="POST", body=b'{"preset": "nope"}')
    assert status == 400 and headers["Cache-Control"] == "no-store"


def test_the_stream_is_never_cached(server):
    with StreamClient(server._port, TOKEN) as client:
        client.read_for(0.3)
    assert client.status == 200
    assert client.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", ["/", f"/?token={TOKEN}", "/setup", f"/setup?token={TOKEN}"])
def test_pages_whose_url_can_carry_the_token_send_no_referrer(server, path):
    status, headers = _headers(f"http://127.0.0.1:{server._port}{path}", token=False)
    assert status == 200
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["Cache-Control"] == "no-store"
