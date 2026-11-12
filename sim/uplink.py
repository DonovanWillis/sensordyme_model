"""The virtual device's link to the backend: the real CloudUplink, two transports.

SimUplink IS core.uplink.CloudUplink -- same queue, batching, retry, drop and
command handling, same X-Device-Key auth, same /v1/ingest -- with:

  * a transport in place of the `requests` module: NetworkGate (fails every
    request while the scenario says the network is down, otherwise passes to
    real HTTP) and, in tests, TestClientTransport (the backend in-process);
  * a manual mode in which no thread is started and the caller drives
    `sync_once()` on its own clock;
  * `simulated: true` added to the monitoring report it sends, so the backend's
    copy (GET /v1/devices `monitoring`) says what this device is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional
from urllib.parse import urlsplit

from core.uplink import CloudUplink


class NetworkGate:
    """A `requests`-shaped transport that is down while `is_down()` says so."""

    def __init__(self, inner: Any, is_down: Callable[[], bool]) -> None:
        self._inner = inner
        self._is_down = is_down
        self.refused = 0
        self.sent = 0

    def post(self, url: str, **kwargs):
        if self._is_down():
            import requests

            self.refused += 1
            raise requests.ConnectionError(f"SIMULATED network outage: {url} unreachable")
        self.sent += 1
        return self._inner.post(url, **kwargs)


class _Response:
    """A requests.Response-shaped view of an httpx/TestClient response."""

    def __init__(self, response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = dict(response.headers)
        self.text = response.text

    def json(self):
        return self._response.json()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text[:200]}", response=self)


class TestClientTransport:
    """Sends the uplink's requests to an in-process FastAPI TestClient."""

    __test__ = False  # not a pytest test class

    def __init__(self, client) -> None:
        self._client = client

    def post(self, url: str, *, json=None, data=None, headers=None, timeout=None, verify=None,
             allow_redirects: bool = True):
        del timeout, verify  # meaningless in-process
        parts = urlsplit(url)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        if data is not None:
            # cloud.command_auth: required signs the exact body bytes (X-SD-Auth), so
            # they must reach the backend unchanged, with the caller's Content-Type.
            content = data.encode() if isinstance(data, str) else data
            return _Response(self._client.post(path, content=content, headers=headers,
                                               follow_redirects=allow_redirects))
        return _Response(self._client.post(path, json=json, headers=headers,
                                           follow_redirects=allow_redirects))


class SimUplink(CloudUplink):
    def __init__(self, config: dict, *, transport: Optional[Any] = None, manual: bool = False) -> None:
        super().__init__(config)
        if transport is not None:
            self._requests = transport
        self.manual = manual
        self.syncs = 0

    def start(self) -> None:
        if not self.manual:
            super().start()

    def sync_once(self) -> bool:
        self.syncs += 1
        return super().sync_once()

    def set_monitoring(self, described: Optional[dict]) -> None:
        super().set_monitoring(None if described is None else {**described, "simulated": True})
