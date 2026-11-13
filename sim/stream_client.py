"""A minimal /stream client for the M8.3 stream tests and measurements.

Reads the device's multipart/x-mixed-replace stream the way the app does
(raw socket, one connection, header token) and splits it into parts with
their headers, so a test can assert on X-Frame-Seq / X-Frame-Age-Ms /
X-Stream-Id and on heartbeat parts. Imported by tests/test_m83_stream_*.py and scripts/sim/stream_soak.py.
"""

from __future__ import annotations

import contextlib
import socket
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Part:
    headers: dict[str, str]
    body: bytes
    received_at: float  # time.monotonic() when the part completed

    @property
    def is_jpeg(self) -> bool:
        return self.headers.get("content-type", "").startswith("image/jpeg")

    @property
    def is_heartbeat(self) -> bool:
        return self.headers.get("x-stream-heartbeat") == "1"

    def int_header(self, name: str) -> int:
        """The integer header `name`; KeyError when the part does not carry it."""
        return int(self.headers[name.lower()])


@dataclass
class StreamReply:
    status: int
    headers: dict[str, str]
    parts: list[Part] = field(default_factory=list)
    raw_bytes: int = 0


def _parse_headers(block: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in block.decode("latin-1").split("\r\n"):
        name, sep, value = line.partition(":")
        if sep and name.strip() and not name.startswith("--"):
            headers[name.strip().lower()] = value.strip()
    return headers


class StreamClient:
    """One /stream connection; `read_for(seconds)` collects whole parts."""

    def __init__(
        self,
        port: int,
        token: Optional[str] = None,
        heartbeat: bool = False,
        query_token: Optional[str] = None,
        recv_buffer: Optional[int] = None,
        host: str = "127.0.0.1",
        keep_bodies: bool = True,
    ) -> None:
        #: False for long runs: keep each part's headers and timing, not its JPEG.
        self.keep_bodies = keep_bodies
        #: Largest unparsed buffer seen (the client-side buffer bound).
        self.max_buffer = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if recv_buffer is not None:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer)
        self.sock.settimeout(5)
        self.sock.connect((host, port))
        path = "/stream" if query_token is None else f"/stream?token={query_token}"
        lines = [f"GET {path} HTTP/1.1", "Host: device"]
        if token is not None:
            lines.append(f"X-Device-Token: {token}")
        if heartbeat:
            lines.append("X-Stream-Heartbeat: 1")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        self._buffer = b""
        self.status = 0
        self.headers: dict[str, str] = {}
        self.parts: list[Part] = []
        self.raw_bytes = 0
        self.closed_by_peer = False

    # ------------------------------------------------------------------ reading
    def _fill(self, timeout: float) -> bool:
        self.sock.settimeout(max(0.01, timeout))
        try:
            chunk = self.sock.recv(262144)
        except TimeoutError:
            return True
        except OSError:
            self.closed_by_peer = True
            return False
        if not chunk:
            self.closed_by_peer = True
            return False
        self.raw_bytes += len(chunk)
        self._buffer += chunk
        self.max_buffer = max(self.max_buffer, len(self._buffer))
        return True

    def _drain(self) -> None:
        if not self.status:
            end = self._buffer.find(b"\r\n\r\n")
            if end < 0:
                return
            head = self._buffer[:end].decode("latin-1").split("\r\n")
            self.status = int(head[0].split()[1])
            self.headers = _parse_headers("\r\n".join(head[1:]).encode("latin-1"))
            self._buffer = self._buffer[end + 4:]
        while True:
            start = self._buffer.find(b"--frame")
            if start < 0:
                return
            end = self._buffer.find(b"\r\n\r\n", start)
            if end < 0:
                return
            headers = _parse_headers(self._buffer[start:end])
            length = int(headers.get("content-length", "0"))
            body_start = end + 4
            if len(self._buffer) < body_start + length:
                return
            body = self._buffer[body_start:body_start + length]
            self._buffer = self._buffer[body_start + length:]
            self.parts.append(Part(headers, body if self.keep_bodies else b"", time.monotonic()))

    def read_for(self, seconds: float) -> "StreamClient":
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not self._fill(deadline - time.monotonic()):
                break
            self._drain()
        self._drain()
        return self

    def read_until(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self):
                return True
            if not self._fill(min(0.2, deadline - time.monotonic())):
                break
            self._drain()
        self._drain()
        return predicate(self)

    def jpeg_parts(self) -> list[Part]:
        return [p for p in self.parts if p.is_jpeg]

    def heartbeats(self) -> list[Part]:
        return [p for p in self.parts if p.is_heartbeat]

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()

    def __enter__(self) -> "StreamClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
