"""Push notifications must survive the titles this backend actually writes.

Every entry in cloud/app.py's `_EVENT_TITLES` opens with an emoji, and an HTTP
header value is latin-1 on the wire. `headers={"Title": "🔴 Potential Fall
Detected"}` therefore raised UnicodeEncodeError inside `requests` *before the
request left the process* — so delivery was exactly inverted: all 22 recognised
safety events (fall, drowsiness, zone violation...) failed, and the only pushes
that ever arrived were for UNRECOGNISED types, whose title comes from
`_readable_title`'s plain-ASCII fallback.

The failure was also treated as transient by `_send_with_retry`, which burned
three attempts and ~4.5s of a FastAPI threadpool worker per notification to
reach a conclusion that was already certain on the first one.

These tests go through the real `requests` stack to a real local HTTP server,
rather than re-implementing the latin-1 rule in the test — the rule being
enforced is urllib3's, and a test that models it itself could agree with a
broken implementation.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

requests = pytest.importorskip("requests")

from cloud import app as cloud_app  # noqa: E402


class _CaptureHandler(BaseHTTPRequestHandler):
    """Records the raw Title header bytes of every POST it receives."""

    received: ClassVar[list[bytes]] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        raw = self.headers.get("Title", "")
        # http.client hands the header back as a str it decoded from the
        # wire bytes as latin-1; re-encoding recovers the exact bytes sent.
        _CaptureHandler.received.append(raw.encode("latin-1"))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):  # keep the test output clean
        pass


@pytest.fixture()
def ntfy_server(monkeypatch):
    """A throwaway ntfy stand-in, pointed at by NTFY_BASE."""
    _CaptureHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(cloud_app, "NTFY_BASE", f"http://127.0.0.1:{server.server_port}")
    try:
        yield _CaptureHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_every_event_title_can_actually_be_delivered(ntfy_server):
    """The whole title table, end to end through requests, over a socket.

    Before the fix this failed 22 times out of 22 with UnicodeEncodeError.
    """
    titles = list(cloud_app._EVENT_TITLES.values())
    assert len(titles) >= 22, "the title table shrank; this test guards all of it"

    failed = []
    for title in titles:
        try:
            cloud_app.send_push({"ntfy_topic": "regression-topic"}, title, "body")
        except Exception as exc:  # noqa: BLE001 - the point is to report every failure
            failed.append(f"{title!r}: {type(exc).__name__}: {exc}")

    assert not failed, f"these titles cannot be pushed at all: {failed}"

    # ...and what the server received is the title itself, not mojibake: the
    # bytes on the wire decode as UTF-8 back to exactly what was passed in.
    delivered = [raw.decode("utf-8") for raw in ntfy_server.received]
    assert delivered == titles


def test_an_ascii_title_is_unchanged(ntfy_server):
    """`_readable_title`'s fallback (the only path that worked before) must
    still put the plain ASCII bytes on the wire, byte for byte."""
    plain = cloud_app._readable_title("some_new_module_event")
    assert plain == "SensorDyme: some_new_module_event"
    cloud_app.send_push({"ntfy_topic": "regression-topic"}, plain, "body")
    assert ntfy_server.received == [plain.encode("ascii")]


def test_an_unencodable_value_fails_immediately_instead_of_retrying(monkeypatch):
    """A UnicodeEncodeError is decided by the bytes in hand, so attempts two
    and three are guaranteed to fail identically. Retrying them still raises
    the same error, it just holds a threadpool worker for ~4.5s first.

    Before the fix this saw 3 attempts and slept 4.5s; now it sees 1 and
    does not sleep. It must still RAISE — never report a delivery that did
    not happen.
    """
    attempts = []

    def explode():
        attempts.append(1)
        # The genuine article: exactly what requests raises today for a
        # header value it cannot put on the wire.
        "🔴".encode("latin-1")

    started = time.monotonic()
    with pytest.raises(UnicodeEncodeError):
        cloud_app._send_with_retry(explode)
    elapsed = time.monotonic() - started

    assert attempts == [1], f"a deterministic encoding failure was retried {len(attempts)}x"
    assert elapsed < cloud_app.NOTIFY_RETRY_BACKOFF_S, "it backed off before giving up"


def test_a_genuinely_transient_failure_is_still_retried(monkeypatch):
    """The non-retry above must be narrow: a provider blip is exactly what
    NOTIFY_SEND_ATTEMPTS exists for, and must keep being retried."""
    monkeypatch.setattr(cloud_app, "NOTIFY_RETRY_BACKOFF_S", 0.0)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("ntfy briefly unreachable")

    cloud_app._send_with_retry(flaky)
    assert len(attempts) == 3
