"""Drive Dashboard: the device serves a local web app for live driver monitoring.

The distracted-driving device pushes proactive safety alerts to a clean web app with
a live feed, an alert timeline, and session analytics. The detection engine stays
untouched; this is purely a presentation layer that any driver-monitoring module feeds:

  * live annotated video (MJPEG over localhost)
  * a big current-status card (state, alert flash, risk gauge)
  * a real-time alert timeline (every event, color-coded by severity)
  * session analytics (uptime, alert counts by type, a rolling risk sparkline)

PRIVACY: frames are still never written to disk. Enabling the dashboard streams
annotated frames, in memory, over HTTP to browsers that connect — bound to
127.0.0.1 by default so nothing leaves this machine. (Set dashboard.host to
0.0.0.0 on the Pi to view from your phone on the same network — that is an
explicit opt-in, like the mac_max VLM exception.) Everything is Python stdlib —
no new dependencies.

M8.4 (owner Decision C): while monitoring is STOPPED, /stream sends no camera
frame to a remote viewer unless the owner enabled camera-only mode through the
backend; heartbeats say `camera_off_while_stopped` instead. A viewer on this
machine (loopback) keeps the camera for local maintenance unless
`dashboard.local_camera_while_stopped: false`. See core/camera_privacy.py and
docs/PRIVACY.md.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import secrets
import select
import socket
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from core import camera_privacy as cp
from core.logging_setup import get_logger

#: This module had no logger at all. It needs exactly one — the open-dashboard
#: warning below is the only thing here worth saying out loud.
log = get_logger("dashboard")

_JPEG_QUALITY = 60  # phone-screen quality; every point costs CPU on the Pi's encode path
_STREAM_FPS = 12  # cap on frames pushed to a connected viewer (detection still runs at camera fps)
_MAX_EVENTS = 200
_MAX_RISK_POINTS = 360  # ~6 minutes at 1 sample/s

# How long the "flash red" signal stays true after a new event lands in Recent
# Activity. Deliberately EDGE-triggered (once per event, for this long), not
# tied to is_alert_active()'s continuous "is a condition still ongoing" state:
# an object that remains visible must not keep the screen flashing, but every
# genuinely new event must produce a visible flash even for a page polling at
# 1 sample/s. services/runtime.py tracks the same duration independently for
# the baked-in MJPEG/cv2 overlay; this one drives the web page's own flash.
FLASH_DURATION_S = 1.5

#: How long a /stream write may block before the viewer is dropped. Each viewer
#: has its own handler thread writing from the ONE shared latest-frame slot, so a
#: slow client only ever blocks itself (frames it cannot take are skipped, never
#: queued); this bounds how long a client that stopped reading entirely keeps
#: its thread and its viewer slot. Generous: phones on weak Wi-Fi are slow,
#: not dead. Same value the handler used for every socket operation before.
_STREAM_SEND_TIMEOUT_S = 30.0

#: Most connections served at once. Each one is a thread for as long as it lasts
#: (a /stream viewer for as long as it watches), and the server is reachable --
#: unauthenticated, until the token check -- from the whole LAN: without a cap
#: anyone on the Wi-Fi could open idle sockets until this process, which is also
#: the detector, could not start threads of its own. Far above any real use.
_MAX_CONNECTIONS = 64


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that refuses connections beyond `max_connections`."""

    daemon_threads = True

    def __init__(self, server_address, handler, max_connections: int = _MAX_CONNECTIONS) -> None:
        self._slots = threading.BoundedSemaphore(max(1, int(max_connections)))
        super().__init__(server_address, handler)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)  # full: close it now, hold nothing
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _peer_closed(sock) -> bool:
    """True once the client has closed its end of `sock` (see _Handler._peer_gone).

    One non-blocking peek; writes nothing. Anything unreadable is reported as
    still connected, so a transient condition never hangs up on a healthy viewer.
    """
    try:
        # poll(), not select(): select() cannot take a descriptor numbered 1024
        # or above and raises ValueError -- which read as "peer gone" and ended
        # every stream once the process had that many files open.
        poll = getattr(select, "poll", None)
        if poll is not None:
            poller = poll()
            poller.register(sock, select.POLLIN | select.POLLERR | select.POLLHUP)
            readable = bool(poller.poll(0))
        else:  # pragma: no cover - platforms without poll()
            readable = bool(select.select([sock], [], [], 0)[0])
    except (OSError, ValueError):
        return True  # socket already closed underneath us
    if not readable:
        return False
    try:
        # Readable on a request whose body is complete means EOF. Peek rather
        # than read: nothing may be consumed from the stream.
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (BlockingIOError, InterruptedError):
        return False
    except OSError:
        return True


@dataclass
class _StreamFrame:
    """The one shared latest-frame slot every /stream viewer reads from."""

    jpeg: bytes
    seq: int  # dashboard-wide, strictly increasing across annotated + camera frames
    captured_at: float  # time.monotonic() of the CAMERA capture this frame shows
    source: str  # "annotated" (the runtime's analysed view) | "camera" (fallback, unanalysed)


#: A loopback viewer while monitoring is stopped (local maintenance; never remote).
_ACCESS_LOCAL = "local_maintenance"


def _source_for(frame: _StreamFrame, access: str) -> str:
    """The X-Frame-Source a viewer with `access` is told (M8.4, core/camera_privacy.py).

    While monitoring is stopped nothing is analysed, so whatever the slot holds
    is named for the access that let it out: `camera_only` (the owner's
    camera-only mode) or `camera_local` (a viewer on the device itself).
    """
    if access == cp.STREAM_CAMERA_ONLY:
        return cp.SOURCE_CAMERA_ONLY
    if access == _ACCESS_LOCAL:
        return cp.SOURCE_CAMERA_LOCAL
    return frame.source


@dataclass
class _StreamClient:
    """Per-viewer counters; never holds frames (the slot is shared)."""

    opened_at: float
    frames_sent: int = 0
    frames_skipped: int = 0  # frames published while this viewer was busy: dropped, not queued
    heartbeats: int = 0
    bytes_sent: int = 0
    last_seq: int = 0


def _part_header(frame: _StreamFrame, stream_id: str, now: float, source: Optional[str] = None) -> bytes:
    """One multipart part header for a JPEG.

    The first three lines are exactly what the stream always sent, so a client
    that predates M8.3 (the old iOS reader, a browser <img>) parses it as before
    and ignores the rest. The X- lines are the evidence the app checks before it
    may say LIVE (docs/VIDEO_TRANSPORT.md):

      X-Frame-Seq      strictly increasing for this stream id; a repeat or a
                       stall means nothing new was captured
      X-Frame-Age-Ms   capture -> this write, on the device's own monotonic
                       clock (no clock sync needed); network transit is extra
      X-Frame-Source   "annotated" = the analysed view; "camera" = the camera
                       without analysis while monitoring runs (module degraded
                       or not drawing); M8.4: "camera_only" = monitoring is
                       stopped and the OWNER enabled camera-only mode;
                       "camera_local" = stopped, viewer on the device itself
      X-Stream-Id      random per dashboard process: a change means the device
                       restarted and the seq started again
    """
    age_ms = max(0, int((now - frame.captured_at) * 1000))
    return (
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(frame.jpeg)}\r\n"
        f"X-Frame-Seq: {frame.seq}\r\n"
        f"X-Frame-Age-Ms: {age_ms}\r\n"
        f"X-Frame-Source: {source or frame.source}\r\n"
        f"X-Stream-Id: {stream_id}\r\n"
        "\r\n"
    ).encode()


def _heartbeat_part(stream_id: str, status: str, camera_age_ms: Optional[int], last_seq: int) -> bytes:
    """An empty part that says the connection is alive and why no frame is coming.

    Sent only to a client that asked for it (`X-Stream-Heartbeat: 1` on the
    request): a browser <img> fed a non-image part may show a broken image, and
    the pre-M8.3 app simply discards bytes outside a JPEG, but there is no
    reason to send either of them anything new.
    """
    lines = [
        "--frame",
        "Content-Type: text/plain",
        "Content-Length: 0",
        "X-Stream-Heartbeat: 1",
        f"X-Stream-Id: {stream_id}",
        f"X-Stream-Status: {status}",
        f"X-Frame-Seq: {last_seq}",
    ]
    if camera_age_ms is not None:
        lines.append(f"X-Camera-Age-Ms: {camera_age_ms}")
    return ("\r\n".join(lines) + "\r\n\r\n\r\n").encode()


class DashboardServer:
    """Threaded local web app fed by the main loop; never blocks the camera."""

    #: A frame captured longer ago than this is never sent: not to a new viewer
    #: (the slot can hold a picture from before a stop or a freeze) and not
    #: mid-stream. Generous against a loaded Pi's capture -> annotate -> encode
    #: latency; the APP decides what is fresh enough to call LIVE (2 s,
    #: ios/Sources/StreamLiveness.swift), this only stops the device from
    #: serving pictures nobody should mistake for the present.
    MAX_FRAME_AGE_S = 5.0
    #: The camera itself counts as stale once its newest frame is this old
    #: (heartbeat status `camera_stale`).
    CAMERA_STALE_S = 2.0
    #: With no frame sent for this long, a viewer that opted in gets a heartbeat.
    HEARTBEAT_INTERVAL_S = 1.0
    #: The runtime counts as "not drawing" once it has pushed no annotated frame
    #: for this long; the camera fallback starts then (if a source is attached).
    #: Above the slowest healthy push cadence (the Pi on CPU inference, ~3/s).
    ANNOTATION_GAP_S = 1.5

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8899,
        module_name: str = "",
        presets: Optional[dict] = None,
        active_preset: Optional[str] = None,
        token: Optional[str] = None,
        events_csv_path: str = "data/events.csv",
        jpeg_quality: int = _JPEG_QUALITY,
        stream_fps: float = _STREAM_FPS,
        health_provider=None,
        camera_fallback: bool = True,
        stream_send_timeout_s: float = _STREAM_SEND_TIMEOUT_S,
        camera_privacy: Optional[cp.CameraPrivacy] = None,
        local_camera_while_stopped: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        #: Whether this bind is reachable from anything but this machine.
        #:
        #: The distinction matters because the auth gate below used to treat "no
        #: token configured" as "no auth required", which is a safe default on
        #: loopback and the opposite of safe on 0.0.0.0 — and 0.0.0.0 is what
        #: config.yaml ships. The result was the live camera feed, the event
        #: history, the preset control and the Wi-Fi reprovisioning endpoint all
        #: answering unauthenticated to anyone on the Wi-Fi.
        self._network_reachable = host not in ("127.0.0.1", "::1", "localhost")
        self._module_name = module_name
        self._presets = presets or {}
        self._active_preset = active_preset
        # str(): YAML reads `token: 123456` as an int, and compare_digest(str, int)
        # raises on every request -- an auth check that crashes is not one.
        self._token = str(token) if token not in (None, "") else None
        self._events_csv_path = events_csv_path
        self._health_provider = health_provider  # () -> dict, from core/health.py
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._stream_interval = 1.0 / max(1.0, float(stream_fps))
        self._pending_preset: Optional[str] = None
        self._pending_protection: Optional[dict] = None
        #: Role, protection state and the last refused change (M0.1), from the runtime.
        self._policy_state: dict = {}
        #: Set by the runtime: (preset name) -> reason it would weaken the role, or None.
        self.preset_guard: Optional[Callable[[str], Optional[str]]] = None
        self._lock = threading.Lock()
        #: The ONE latest-frame slot (see _StreamFrame). Replaced, never queued.
        self._frame: Optional[_StreamFrame] = None
        self._jpeg_seq = 0
        #: monotonic time of the runtime's last annotated push (fallback gate).
        self._annotated_at = 0.0
        #: Random per process: lets a viewer see a restart even though the seq
        #: restarts from 1 (docs/VIDEO_TRANSPORT.md).
        self._stream_id = secrets.token_hex(4)
        #: Optional camera source (a FrameBus): heartbeat camera age, and the
        #: unannotated fallback while the runtime is not drawing.
        self._frame_source = None
        self._camera_fallback = bool(camera_fallback)
        self._fallback_lock = threading.Lock()  # one encoder at a time, shared by all viewers
        self._fallback_bus_seq = 0
        self._stream_send_timeout_s = max(0.1, float(stream_send_timeout_s))
        #: M8.4 Decision C: the owner's camera-only setting (OFF unless the
        #: owner turned it on through the backend; core/camera_privacy.py).
        self._camera_privacy = camera_privacy if camera_privacy is not None else cp.CameraPrivacy(persist=False)
        #: Whether a viewer on this machine (loopback) still sees the camera
        #: while monitoring is stopped (`dashboard.local_camera_while_stopped`).
        self._local_camera_while_stopped = bool(local_camera_while_stopped)
        #: Monitoring explicitly stopped, as the runtime reports it
        #: (set_policy_state / set_monitoring_stopped).
        self._monitoring_stopped = False
        #: Set by close(): every open /stream loop ends, so viewers reconnect to
        #: whatever serves next instead of holding a socket to a dead server.
        self._closing = threading.Event()
        self._clients: dict[int, _StreamClient] = {}
        self._stream_counts: Counter = Counter()
        self._viewers = 0
        self._events: deque[dict] = deque(maxlen=_MAX_EVENTS)
        self._counts: Counter = Counter()
        self._status: dict = {}
        self._alert_active = False
        self._flash_until = 0.0
        self._risk_series: deque[tuple[float, float]] = deque(maxlen=_MAX_RISK_POINTS)
        self._last_risk_sample = 0.0
        self._started_at = time.time()
        self._httpd: Optional[ThreadingHTTPServer] = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> str:
        """Start serving in a daemon thread; returns the dashboard URL."""
        dashboard = self

        class _Handler(BaseHTTPRequestHandler):
            # Socket timeout for every read and write on a connection. Without
            # it a client that opens a socket and sends half a request holds a
            # thread for ever; a few hundred of them, unauthenticated, exhaust
            # the process. Generous, because /stream writes to phones on weak
            # Wi-Fi: only a client stalled this long is dropped.
            timeout = 30

            def log_message(self, *args) -> None:  # silence per-request logging
                pass

            def _authorized(self) -> bool:
                """When a device token is configured, require it on every /api and /stream path."""
                if dashboard._token is None:
                    # No token configured. On loopback that is a reasonable
                    # convenience; on a network-reachable bind it is an open
                    # camera, so refuse instead.
                    #
                    # Deliberately NOT solved by binding loopback: the phone
                    # fetches /stream from this dashboard directly, so refusing
                    # to listen on the LAN would delete the live feed rather
                    # than secure it. Failing closed keeps the feature available
                    # to anyone who sets a token and shuts the door for anyone
                    # who has not.
                    return not dashboard._network_reachable
                sent = self.headers.get("X-Device-Token", "")
                if not sent and self.command == "GET" and urlsplit(self.path).path.startswith("/stream"):
                    # Kept for the MJPEG <img> case, which cannot set a header. It is
                    # a real leak -- a token in a URL lands in proxy logs, browser
                    # history and Referer -- so it stays the fallback, never the
                    # documented path, and the app itself uses the header.
                    #
                    # M6: ONLY for GET /stream. It used to be accepted on every
                    # path, POST included, so a link or a history entry carrying
                    # ?token= could change the preset, stop monitoring or move the
                    # device to another Wi-Fi network. The page, the iOS app and
                    # every script send the header on /api and on every POST.
                    #
                    # Parsed, not split: the page sends encodeURIComponent(TOKEN),
                    # so a token containing + / = & or a space only matches once
                    # it is decoded.
                    query = parse_qs(urlsplit(self.path).query)
                    sent = (query.get("token") or [""])[0]
                # compare_digest, not ==: a plain string compare short-circuits on the
                # first differing byte, which leaks the token's prefix to anyone who
                # can time responses. Cheap to get right, awkward to retrofit later.
                # On bytes: compare_digest refuses non-ASCII str, and a header can
                # carry anything -- that used to drop the connection with a traceback.
                return hmac.compare_digest(
                    sent.encode("utf-8", "surrogateescape"),
                    dashboard._token.encode("utf-8", "surrogateescape"),
                )

            def _peer_gone(self) -> bool:
                """True once the client has closed its end of the connection.

                Costs one non-blocking peek and writes nothing, so it works during
                the windows a write-based check cannot see: a stalled camera, a
                degraded detector, or the moment between a viewer connecting and
                the main loop encoding its first frame.

                Why it matters beyond tidiness: `has_viewers()` is the gate the
                inference loop uses to decide whether to annotate and JPEG-encode at
                all, and on the Pi that work competes with the detector for the same
                cores. A viewer slot that is never returned does not merely
                mis-report a number -- it spends inference CPU forever, and parks
                one handler thread per abandoned connection. An app that retries a
                dropped stream (see the iOS MJPEG reader) produces exactly those
                abandoned connections, so this is the device's half of that bug.

                Conservative on purpose: anything unreadable is reported as still
                connected, so a transient condition can never hang up on a healthy
                viewer. Only a genuine EOF or a dead socket ends the stream.
                """
                return _peer_closed(self.connection)

            def _json(self, obj: dict, status: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                # Device state, history and health: never kept by a cache or proxy.
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                """The request body as a JSON object; {} for anything else.

                Bounded: these bodies are a preset name or a Wi-Fi password,
                and an unbounded Content-Length is memory on the device's
                terms. A malformed length used to raise into a traceback.
                """
                try:
                    length = int(self.headers.get("Content-Length", 0))
                except ValueError:
                    return {}
                if not 0 < length <= 64 * 1024:
                    return {}
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError):
                    return {}
                return payload if isinstance(payload, dict) else {}

            def do_POST(self) -> None:
                try:
                    if not self._authorized():
                        self._json({"error": "invalid or missing device token"}, 401)
                        return
                    if self.path.startswith("/setup/wifi"):
                        payload = self._read_json()
                        ok, message = dashboard.apply_wifi(
                            str(payload.get("ssid", "")), str(payload.get("password", ""))
                        )
                        self._json({"ok": ok, "message": message}, 200 if ok else 400)
                    elif self.path.startswith("/api/preset"):
                        payload = self._read_json()
                        name = str(payload.get("preset", ""))
                        base = name.partition(":")[0]
                        if base not in dashboard._presets:
                            self._json({"error": f"unknown preset '{name}'"}, 400)
                            return
                        guard = dashboard.preset_guard
                        refusal = guard(name) if callable(guard) else None
                        if refusal:
                            # The device's role forbids it: say so now, not silently later.
                            self._json({"error": refusal, "refused": True}, 409)
                            return
                        dashboard.request_preset(name)
                        self._json({"ok": True, "switching_to": name}, 202)
                    elif self.path.startswith("/api/protection"):
                        payload = self._read_json()
                        action = str(payload.get("action", "")).lower()
                        if action not in ("stop", "resume"):
                            self._json({"error": "action must be 'stop' or 'resume'"}, 400)
                            return
                        if action == "stop" and payload.get("confirm") is not True:
                            self._json({"error": "stopping monitoring requires \"confirm\": true"}, 400)
                            return
                        dashboard.request_protection({"action": action, "confirm": payload.get("confirm") is True})
                        self._json({"ok": True, "requested": action}, 202)
                    else:
                        self.send_response(404)
                        self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self) -> None:
                try:
                    if (self.path.startswith("/api") or self.path.startswith("/stream")) and not self._authorized():
                        self._json({"error": "invalid or missing device token"}, 401)
                        return
                    # Compare the PATH, not the raw request line. An operator
                    # opens the page as `/?token=...` when a token is configured,
                    # and an exact match on "/" 404s that -- so the one URL the
                    # feature requires was the one URL that did not work.
                    bare_path = self.path.split("?", 1)[0]
                    if bare_path == "/" or bare_path.startswith("/index"):
                        body = _PAGE.encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        # This URL can carry ?token=: never leak it in a Referer.
                        self.send_header("Referrer-Policy", "no-referrer")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    elif self.path.startswith("/api/state"):
                        self._json(dashboard.state())
                    elif self.path.startswith("/api/presets"):
                        self._json(dashboard.presets_state())
                    elif self.path.startswith("/setup"):
                        body = _SETUP_PAGE.encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        # This URL can carry ?token=: never leak it in a Referer.
                        self.send_header("Referrer-Policy", "no-referrer")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    elif self.path.startswith("/api/health"):
                        provider = dashboard._health_provider
                        self._json(provider() if provider is not None else {"state": "unknown"})
                    elif self.path.startswith("/api/history"):
                        limit = 200
                        if "limit=" in self.path:
                            with contextlib.suppress(ValueError):
                                limit = min(2000, int(self.path.rsplit("limit=", 1)[1].split("&")[0]))
                        self._json({"events": dashboard.history(limit)})
                    elif self.path.startswith("/api/stream/stats"):
                        self._json(dashboard.stream_stats())
                    elif self.path.startswith("/stream"):
                        heartbeat = self.headers.get("X-Stream-Heartbeat", "") == "1"
                        local = cp.peer_is_local(self.client_address[0] if self.client_address else None)
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                        )
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Stream-Id", dashboard._stream_id)
                        self.send_header("X-Stream-Max-Frame-Age-Ms", str(int(dashboard.MAX_FRAME_AGE_S * 1000)))
                        # M8.4: what this viewer may see right now (it can change
                        # mid-stream; heartbeats carry the change).
                        self.send_header("X-Stream-Camera", dashboard.stream_access(local))
                        if heartbeat:
                            self.send_header(
                                "X-Stream-Heartbeat-Interval-Ms", str(int(dashboard.HEARTBEAT_INTERVAL_S * 1000))
                            )
                        self.end_headers()
                        # Bounded writes: a client that stops reading blocks only
                        # its own thread, and only this long (_STREAM_SEND_TIMEOUT_S).
                        with contextlib.suppress(OSError):
                            self.connection.settimeout(dashboard._stream_send_timeout_s)
                        dashboard.serve_stream(self.wfile.write, self._peer_gone, heartbeat=heartbeat, local=local)
                    else:
                        self.send_response(404)
                        self.end_headers()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed mid-stream

        if self._network_reachable and self._token is None:
            # Said once, loudly, at the only moment anyone is reading the log.
            # The dashboard still starts — it will simply refuse every /api and
            # /stream request — because a device that fails to boot is worse than
            # one that boots and declines to show a stranger the camera.
            log.error(
                "dashboard is bound to %s with no token configured: every /api and "
                "/stream request will be refused. Set dashboard.token in config.yaml "
                "to use the live feed, or bind 127.0.0.1 if you do not need it.",
                self._host,
            )

        self._closing = threading.Event()  # a restart() serves streams again
        self._httpd = _BoundedThreadingHTTPServer(
            (self._host, self._port), _Handler, max_connections=_MAX_CONNECTIONS
        )
        self._port = self._httpd.server_address[1]  # resolve port 0 -> actual
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="dashboard")
        self._thread = thread  # is_alive() checks this, not merely that _httpd exists
        thread.start()
        return self.url

    @property
    def url(self) -> str:
        host = "localhost" if self._host in ("127.0.0.1", "0.0.0.0") else self._host  # noqa: S104 - a comparison, not a bind
        return f"http://{host}:{self._port}"

    def close(self) -> None:
        # End the open streams too. shutdown() below only stops ACCEPTING: each
        # viewer's handler thread kept looping on a server nobody fed any more,
        # so after a dashboard restart (health monitor) the app held a silent
        # socket until its idle timeout instead of reconnecting at once.
        self._closing.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            # Release the socket and forget the server. Without this, `_httpd`
            # stayed set after close() and is_alive() below reported True
            # forever -- so the registered health check was a constant True and
            # its restart path was unreachable.
            self._httpd.server_close()
            self._httpd = None

    def is_alive(self) -> bool:
        """Whether the HTTP server is actually serving (health monitor check).

        Tests the SERVE THREAD, not merely that an object exists. The previous
        version returned `self._httpd is not None`, which close() never falsified
        and which stayed true if the serving thread died -- the live feed would
        go dark while /api/health reported the dashboard healthy.
        """
        if self._httpd is None:
            return False
        thread = getattr(self, "_thread", None)
        return thread is None or thread.is_alive()

    def restart(self) -> None:
        """Tear the HTTP server down and bring it back on the same host/port."""
        self.close()
        self._httpd = None
        self.start()

    # ------------------------------------------------------------------ viewers
    # The annotate + JPEG-encode path costs real CPU on the Pi (it competes with
    # inference on the same cores), so the main loop skips it entirely when nobody
    # is watching. These three methods are what it asks.
    @property
    def stream_interval(self) -> float:
        return self._stream_interval

    def viewer_joined(self) -> None:
        with self._lock:
            self._viewers += 1

    def viewer_left(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    def has_viewers(self) -> bool:
        """True while at least one client is reading /stream."""
        with self._lock:
            return self._viewers > 0

    # ------------------------------------------------------------------ the live stream
    def attach_frame_source(self, source) -> None:
        """Give the stream the camera itself (a core.frame_bus.FrameBus).

        Two uses, both only while someone is watching: the heartbeat reports
        the camera's own frame age (so the app can say "camera not sending"
        rather than "stream stalled"), and while the runtime is not drawing --
        monitoring stopped, a degraded module, annotate() failing -- the newest
        camera frame is encoded unannotated (X-Frame-Source: camera) instead
        of the stream freezing on its last picture with the connection open.
        """
        self._frame_source = source

    def serve_stream(self, write: Callable[[bytes], object], peer_gone: Callable[[], bool],
                     heartbeat: bool = False, local: bool = False) -> None:
        """Feed one viewer until it leaves. Runs on that viewer's handler thread.

        Latest-frame-only by construction: each pass reads the ONE shared slot
        and sends it if it is newer than what this viewer last got. A viewer
        that is slow (its send blocks) simply misses the frames published
        meanwhile -- counted in `frames_skipped` -- so nothing is queued per
        client and a slow client never slows the producer (push_frame only
        swaps a reference under a lock). A viewer that stops reading entirely
        is dropped by the socket send timeout.

        M8.4 (owner Decision C): every pass first asks what this viewer may see
        (`stream_access`). While monitoring is stopped and the owner has not
        enabled camera-only, a remote viewer is sent NO frame -- not the
        camera, not a picture left in the slot from before the stop -- and,
        if it asked for heartbeats, `camera_off_while_stopped` once a second,
        on the same connection, so frames resume the moment monitoring does.
        `local`: the viewer is on this machine (loopback).
        """
        client = _StreamClient(opened_at=time.monotonic())
        key = id(client)
        with self._lock:
            self._clients[key] = client
            self._stream_counts["connections_total"] += 1
        # Registering as a viewer is what makes the main loop start annotating
        # and encoding at all -- see has_viewers().
        self.viewer_joined()
        last_write = time.monotonic()
        try:
            # `not peer_gone()`, not `while True`: the loop used to learn that a
            # client had left ONLY by having a write fail, and during any window
            # with no fresh frames -- a camera stall, a degraded detector, or the
            # gap before the first frame is encoded for a newly connected viewer
            # -- there is no write to fail. See _Handler._peer_gone().
            while not peer_gone() and not self._closing.is_set():
                access = self.stream_access(local)
                if access == cp.STREAM_OFF:
                    now = time.monotonic()
                    if heartbeat and now - last_write >= self.HEARTBEAT_INTERVAL_S:
                        write(_heartbeat_part(self._stream_id, cp.STATUS_CAMERA_OFF, None, client.last_seq))
                        client.heartbeats += 1
                        last_write = now
                        with self._lock:
                            self._stream_counts["heartbeats"] += 1
                            self._stream_counts["privacy_heartbeats"] += 1
                    time.sleep(self.stream_interval)
                    continue
                frame = self._current_frame()
                now = time.monotonic()
                if frame is not None and frame.seq != client.last_seq:
                    if now - frame.captured_at > self.MAX_FRAME_AGE_S:
                        # Never serve a picture from the past as the present.
                        with self._lock:
                            self._stream_counts["stale_frames_withheld"] += 1
                        client.last_seq = frame.seq  # count it once, not every pass
                    else:
                        # One write per part: header, JPEG and trailer together
                        # (the handler's wfile is unbuffered -- four writes were
                        # four syscalls per frame).
                        header = _part_header(frame, self._stream_id, now, _source_for(frame, access))
                        payload = b"".join((header, frame.jpeg, b"\r\n"))
                        write(payload)
                        skipped = max(0, frame.seq - client.last_seq - 1) if client.last_seq else 0
                        client.last_seq = frame.seq
                        client.frames_sent += 1
                        client.frames_skipped += skipped
                        client.bytes_sent += len(payload)
                        last_write = now
                        with self._lock:
                            self._stream_counts["frames_sent"] += 1
                            self._stream_counts["frames_skipped"] += skipped
                            self._stream_counts["bytes_sent"] += len(payload)
                elif heartbeat and now - last_write >= self.HEARTBEAT_INTERVAL_S:
                    status, camera_age_ms = self._stream_status(now)
                    payload = _heartbeat_part(self._stream_id, status, camera_age_ms, client.last_seq)
                    write(payload)
                    client.heartbeats += 1
                    last_write = now
                    with self._lock:
                        self._stream_counts["heartbeats"] += 1
                time.sleep(self.stream_interval)
        except TimeoutError:
            # The viewer stopped reading for _STREAM_SEND_TIMEOUT_S: drop it,
            # free the thread and the viewer slot.
            with self._lock:
                self._stream_counts["client_send_timeouts"] += 1
        except (BrokenPipeError, ConnectionResetError):
            pass  # the viewer went away mid-write
        finally:
            with self._lock:
                self._clients.pop(key, None)
            self.viewer_left()

    # ------------------------------------------------------------------ camera privacy (M8.4)
    def stream_access(self, local: bool = False) -> str:
        """What a /stream viewer may receive now (docs/PRIVACY.md, Decision C).

        `monitoring` while monitoring is not stopped; while it is stopped,
        `camera_only` if the owner enabled it, `local_maintenance` for a
        loopback viewer (unless disabled), otherwise `off_while_stopped`.
        """
        with self._lock:
            stopped = self._monitoring_stopped
        access = self._camera_privacy.remote_stream(stopped)
        if access == cp.STREAM_OFF and local and self._local_camera_while_stopped:
            return _ACCESS_LOCAL
        return access

    def set_monitoring_stopped(self, stopped: bool) -> None:
        """The runtime's explicit stop (also derived from set_policy_state)."""
        with self._lock:
            self._monitoring_stopped = bool(stopped)

    @property
    def camera_privacy(self) -> cp.CameraPrivacy:
        return self._camera_privacy

    def camera_report(self) -> dict:
        """The camera-access block the device reports (core/camera_privacy.py)."""
        with self._lock:
            stopped = self._monitoring_stopped
        return self._camera_privacy.report(stopped=stopped, dashboard=True,
                                           local_while_stopped=self._local_camera_while_stopped)

    def _current_frame(self) -> Optional[_StreamFrame]:
        """The slot's frame, refreshed from the camera when the runtime is not drawing."""
        with self._lock:
            frame, annotated_at = self._frame, self._annotated_at
        source = self._frame_source
        if source is None or not self._camera_fallback:
            return frame
        now = time.monotonic()
        if annotated_at and now - annotated_at < self.ANNOTATION_GAP_S:
            return frame  # the runtime is drawing: its analysed view wins
        if frame is not None and frame.source == "camera" and now - frame.captured_at < self.stream_interval:
            return frame  # a fallback frame newer than one stream interval: reuse it
        # Shared encoder: whichever viewer thread gets here first encodes, the
        # others serve what is already in the slot. Non-blocking, so a viewer
        # never waits behind another viewer's encode.
        if not self._fallback_lock.acquire(blocking=False):
            return frame
        try:
            camera_frame, bus_seq, published_at = source.latest_stamped()
            if camera_frame is None or bus_seq == self._fallback_bus_seq:
                return frame
            if now - published_at > self.CAMERA_STALE_S:
                return frame  # the camera itself is stale: nothing live to show
            ok, buf = cv2.imencode(".jpg", camera_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if not ok:
                return frame
            self._fallback_bus_seq = bus_seq
            with self._lock:
                if self._annotated_at and time.monotonic() - self._annotated_at < self.ANNOTATION_GAP_S:
                    return self._frame  # the runtime resumed while we encoded
                self._jpeg_seq += 1
                self._frame = _StreamFrame(buf.tobytes(), self._jpeg_seq, published_at, "camera")
                self._stream_counts["camera_fallback_encodes"] += 1
                return self._frame
        finally:
            self._fallback_lock.release()

    def _stream_status(self, now: float) -> tuple[str, Optional[int]]:
        """Why no frame is going out: the heartbeat's X-Stream-Status."""
        source = self._frame_source
        if source is None:
            return "no_fresh_frames", None
        _frame, bus_seq, published_at = source.latest_stamped()
        if bus_seq == 0:
            return "camera_not_started", None
        age = now - published_at
        if age > self.CAMERA_STALE_S:
            return "camera_stale", int(age * 1000)
        return "no_fresh_frames", int(age * 1000)

    def stream_stats(self) -> dict:
        """Live-stream counters (GET /api/stream/stats; tests and soak runs).

        `max_pending_frames_per_client` is structural, not measured: viewers
        read one shared slot, so no client can have more than one frame waiting.
        """
        with self._lock:
            counts = self._stream_counts
            return {
                "stream_id": self._stream_id,
                "viewers": self._viewers,
                "connections_total": counts["connections_total"],
                "frames_sent": counts["frames_sent"],
                "frames_skipped": counts["frames_skipped"],
                "bytes_sent": counts["bytes_sent"],
                "heartbeats": counts["heartbeats"],
                "stale_frames_withheld": counts["stale_frames_withheld"],
                "client_send_timeouts": counts["client_send_timeouts"],
                "camera_fallback_encodes": counts["camera_fallback_encodes"],
                "privacy_heartbeats": counts["privacy_heartbeats"],
                "remote_stream": self._camera_privacy.remote_stream(self._monitoring_stopped),
                "latest_seq": self._jpeg_seq,
                "max_pending_frames_per_client": 1,
                "clients": [
                    {
                        "open_s": round(time.monotonic() - c.opened_at, 1),
                        "frames_sent": c.frames_sent,
                        "frames_skipped": c.frames_skipped,
                        "heartbeats": c.heartbeats,
                        "bytes_sent": c.bytes_sent,
                        "last_seq": c.last_seq,
                    }
                    for c in self._clients.values()
                ],
            }

    # ------------------------------------------------------------------ feeders (main loop)
    def push_frame(self, annotated_bgr: np.ndarray, captured_at: Optional[float] = None) -> None:
        """Encode the annotated frame to an in-memory JPEG for the live stream.

        `captured_at` is the time.monotonic() at which the camera captured the
        frame this view was drawn from (the FrameBus publish time the runtime
        already carries). It becomes X-Frame-Age-Ms. Omitted, the encode time is
        used -- which under-reports the age by the inference latency, so the
        runtime passes it.
        """
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if ok:
            now = time.monotonic()
            with self._lock:
                self._jpeg_seq += 1
                self._frame = _StreamFrame(
                    buf.tobytes(), self._jpeg_seq, now if captured_at is None else float(captured_at), "annotated"
                )
                self._annotated_at = now

    def push_event(self, event, severity: str = "log") -> None:
        """Record one Event for the timeline. severity: 'loud' | 'soft' | 'log'.

        Every call here is a genuinely new event reaching Recent Activity (this
        is THE single choke point every event type -- built-in or AI-configured
        rule -- passes through; see services/runtime.py), so this is also
        where the edge-triggered flash signal is armed. Deliberately NOT
        gated on severity: the product requirement is "every new event
        flashes", not just loud ones.
        """
        with self._lock:
            self._events.appendleft(
                {
                    "time": event.timestamp.astimezone().strftime("%H:%M:%S"),
                    "type": event.type,
                    "confidence": None if event.confidence is None else round(event.confidence, 2),
                    "signals": event.contributing_signals,
                    "severity": severity,
                }
            )
            self._counts[event.type] += 1
            self._flash_until = time.time() + FLASH_DURATION_S

    def flash_active(self, now: Optional[float] = None) -> bool:
        """True for FLASH_DURATION_S after the most recent push_event() call.

        Edge-triggered, not level-triggered: this is what lets the web page
        flash red once per new event and then return to normal, instead of
        flashing continuously for as long as an underlying condition (an
        object still visible, a driver still distracted) remains true.
        """
        with self._lock:
            return (time.time() if now is None else now) < self._flash_until

    def push_status(self, alert_active: bool, snapshot: Optional[dict]) -> None:
        """Update the live status card; samples the risk sparkline at ~1Hz."""
        with self._lock:
            self._alert_active = alert_active
            if snapshot:
                self._status = {k: v for k, v in snapshot.items() if isinstance(v, (int, float, bool))}
            now = time.time()
            if now - self._last_risk_sample >= 1.0:
                self._last_risk_sample = now
                risk = float((snapshot or {}).get("risk_score", 1.0 if alert_active else 0.0))
                self._risk_series.append((now, max(0.0, min(1.0, risk))))

    # ------------------------------------------------------------------ preset switching
    def request_preset(self, name: str) -> None:
        """Queue a preset switch; the main loop applies it on its next iteration."""
        with self._lock:
            self._pending_preset = name

    def pop_pending_preset(self) -> Optional[str]:
        """Called by the main loop: returns and clears any requested preset."""
        with self._lock:
            pending, self._pending_preset = self._pending_preset, None
            return pending

    def request_protection(self, request: dict) -> None:
        """Queue an explicit stop/resume for the main loop."""
        with self._lock:
            self._pending_protection = dict(request)

    def pop_pending_protection(self) -> Optional[dict]:
        with self._lock:
            pending, self._pending_protection = self._pending_protection, None
            return pending

    def set_policy_state(self, state: dict) -> None:
        """Role, protection and last refusal, as the runtime reports them.

        Also the stream's privacy gate (M8.4): `protection.state == "stopped"`
        is the explicit, confirmed stop, and from this call on no remote viewer
        is sent a frame unless the owner enabled camera-only.
        """
        state = dict(state or {})
        protection = state.get("protection")
        stopped = isinstance(protection, dict) and protection.get("state") == "stopped"
        with self._lock:
            self._policy_state = state
            self._monitoring_stopped = stopped

    def set_active_preset(self, name: Optional[str], module_name: str) -> None:
        """Called by the main loop after a switch completes."""
        with self._lock:
            self._active_preset = name
            self._module_name = module_name

    def presets_state(self) -> dict:
        with self._lock:
            return {
                "active": self._active_preset,
                "presets": [
                    {
                        "name": name,
                        "title": p.get("title", name),
                        "description": p.get("description", ""),
                        "module": p.get("module", ""),
                        "active": name == (self._active_preset or "").partition(":")[0],
                        "variants": [
                            {"name": vname, "title": v.get("title", vname)}
                            for vname, v in (p.get("variants") or {}).items()
                        ],
                    }
                    for name, p in self._presets.items()
                ],
            }

    #: How many bytes off the end of the event log to consider for `history`. The log
    #: is append-only and unrotated, so it grows for the life of the device -- reading
    #: all of it to return the last 200 rows made every /api/history request cost
    #: O(file size). That endpoint is reachable over the LAN and, with the shipped
    #: `dashboard.token: null`, unauthenticated, so the cost was an amplification
    #: vector that got worse the longer a device had been deployed. 512KB is far more
    #: than any plausible `limit` needs (rows are ~100 bytes) while staying bounded.
    _HISTORY_TAIL_BYTES = 512 * 1024

    def history(self, limit: int = 200) -> list[dict]:
        """The last `limit` rows of the on-device event log (data/events.csv).

        Reads only the tail of the file, not the whole thing -- see
        `_HISTORY_TAIL_BYTES`. The first (possibly partial) line of the tail is
        dropped, and the header is re-supplied from the file's own first line so the
        DictReader still keys correctly.
        """
        import csv
        from pathlib import Path

        path = Path(self._events_csv_path)
        if not path.exists():
            return []

        with path.open("rb") as handle:
            header = handle.readline().decode("utf-8", "replace").strip()
            if not header:
                return []
            body_start = handle.tell()
            handle.seek(0, 2)
            size = handle.tell()
            if size - body_start > self._HISTORY_TAIL_BYTES:
                handle.seek(size - self._HISTORY_TAIL_BYTES)
                handle.readline()  # discard the partial row we almost certainly landed in
            else:
                handle.seek(body_start)
            tail = handle.read().decode("utf-8", "replace")

        rows = list(csv.DictReader(tail.splitlines(), fieldnames=header.split(",")))
        return [
            {
                "timestamp": r.get("timestamp", ""),
                "type": r.get("type", ""),
                "confidence": r.get("confidence", ""),
                "module": r.get("module", ""),
                "signals": r.get("contributing_signals", ""),
            }
            for r in rows[-limit:][::-1]  # newest first
        ]

    # ------------------------------------------------------------------ Wi-Fi provisioning
    def apply_wifi(self, ssid: str, password: str) -> tuple[bool, str]:
        """Join the device to a Wi-Fi network (the /setup page, used during onboarding).

        Delegates to core.network.WifiManager — the single owner of nmcli — which
        drives NetworkManager on the Pi and dry-runs anywhere without nmcli (so
        the page stays fully testable on a dev Mac).

        provision(), not connect(): the request arrives OVER the network the
        device is on (or its own setup hotspot). connect() let go of that network
        before knowing whether the new one would accept it, so a mistyped
        password cut off the very page it was typed into with no way back;
        provision() validates first and restores the previous network on failure.
        """
        from core.network import WifiManager

        return WifiManager().provision(ssid, password)

    # ------------------------------------------------------------------ readers (handler)
    def latest_jpeg(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            frame = self._frame
            return (None, self._jpeg_seq) if frame is None else (frame.jpeg, frame.seq)

    def state(self) -> dict:
        with self._lock:
            top_counts = self._counts.most_common(8)
            return {
                "module": self._module_name,
                "uptime_s": int(time.time() - self._started_at),
                "alert_active": self._alert_active,
                "flash_active": time.time() < self._flash_until,
                "status": dict(self._status),
                "events": list(self._events)[:60],
                "counts": [{"type": t, "n": n} for t, n in top_counts],
                "total_events": sum(self._counts.values()),
                "risk": [round(r, 3) for _, r in self._risk_series],
                "policy": dict(self._policy_state),
                # M8.4: who may see the camera right now (Decision C).
                "camera": self._camera_privacy.report(stopped=self._monitoring_stopped, dashboard=True,
                                                      local_while_stopped=self._local_camera_while_stopped),
                "generated_at": datetime.now(UTC).isoformat(),
            }


# ---------------------------------------------------------------------------- setup page
# Served at /setup — the page a phone opens (via the app's pairing wizard) while joined
# to the device's first-boot hotspot, to hand the device its home Wi-Fi.
_SETUP_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SensorDyme Setup</title>
<style>
 body{font:16px -apple-system,system-ui;background:#0B0D14;color:#eef1f8;display:grid;place-items:center;min-height:100vh;margin:0}
 .card{background:#141826;border-radius:20px;padding:28px;max-width:340px;width:90%}
 h1{font-size:20px;margin:0 0 6px} p{color:#8b93a7;font-size:14px;margin:0 0 18px}
 input{width:100%;box-sizing:border-box;padding:12px;margin-bottom:10px;border-radius:10px;border:1px solid #2a3040;background:#0B0D14;color:#eef1f8}
 button{width:100%;padding:13px;border:0;border-radius:12px;font-weight:700;color:#fff;
        background:linear-gradient(120deg,#4C6FFF,#9B4DFF,#FF4D9B)}
 #msg{margin-top:12px;font-size:14px}
</style></head><body><div class="card">
<h1>Connect SensorDyme to Wi-Fi</h1>
<p>Enter your home or hotspot network. The device will join it and appear in your app.</p>
<input id="ssid" placeholder="Network name (SSID)">
<input id="pw" type="password" placeholder="Password">
<button onclick="go()">Connect</button>
<div id="msg"></div>
<script>
// POST /setup/wifi is gated by the device token like every other write. This
// page sent none, so on any network bind it was refused every time and showed
// an empty message. The token comes from this page's own URL (/setup?token=...)
// and travels as a header, the same way the main page carries it.
const TOKEN = new URLSearchParams(location.search).get("token") || "";
async function go(){
  const m=document.getElementById('msg'); m.textContent='Connecting…';
  const headers={'Content-Type':'application/json'};
  if(TOKEN){ headers['X-Device-Token']=TOKEN; }
  try{
    const r=await fetch('/setup/wifi',{method:'POST',headers:headers,
      body:JSON.stringify({ssid:document.getElementById('ssid').value,password:document.getElementById('pw').value})});
    const j=await r.json();
    if(r.status===401){
      m.textContent='This device needs its dashboard token: open this page as /setup?token=YOUR_TOKEN.';
      m.style.color='#FF6B60'; return;
    }
    m.textContent=j.message||j.error||('HTTP '+r.status); m.style.color=j.ok?'#2BD9A9':'#FF6B60';
  }catch(e){ m.textContent='No answer from the device: '+e; m.style.color='#FF6B60'; }
}
</script></div></body></html>
"""

# ---------------------------------------------------------------------------- the app page
_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SensorDyme</title>
<style>
  :root { --bg:#f4f6fb; --card:#ffffff; --ink:#1c2333; --muted:#7a8299; --accent:#4c7dff;
          --ok:#1fb970; --warn:#ff9f2e; --bad:#f4453f; --line:#e7eaf3; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151d; --card:#1b202c; --ink:#eef1f8; --muted:#8b93a7; --line:#2a3040; }
  }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink);
         font:15px/1.45 -apple-system, system-ui, "Segoe UI", sans-serif; padding:18px; }
  .wrap { max-width:1080px; margin:0 auto; }
  header { display:flex; align-items:baseline; gap:12px; margin-bottom:14px; }
  header h1 { font-size:20px; font-weight:700; }
  header .sub { color:var(--muted); font-size:13px; }
  .grid { display:grid; grid-template-columns: 1.6fr 1fr; gap:14px; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px;
          padding:14px; box-shadow:0 1px 3px rgba(20,30,60,.06); }
  .video { position:relative; overflow:hidden; padding:0; }
  .video img { width:100%; display:block; border-radius:16px; }
  .pill { position:absolute; top:12px; left:12px; padding:6px 14px; border-radius:999px;
          font-weight:700; font-size:14px; color:#fff; background:var(--ok);
          box-shadow:0 2px 8px rgba(0,0,0,.25); transition:background .2s; }
  .pill.alert { background:var(--bad); animation:pulse 1s infinite; }
  @keyframes pulse { 50% { opacity:.55; } }
  /* Event-driven flash: a brief red wash over the video, once per NEW event
     landing in Recent Activity -- decoupled from .pill.alert above, which
     tracks whether a condition is still ONGOING. Fades in fast, out slower,
     so a rapid string of events reads as "still flashing" rather than
     visibly stuttering. */
  .flash-overlay { position:absolute; inset:0; background:var(--bad); opacity:0;
                    pointer-events:none; transition:opacity .6s ease-out; border-radius:16px; }
  .flash-overlay.on { opacity:.5; transition:opacity .08s ease-in; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted);
       margin-bottom:10px; }
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:12px; }
  .stat { background:var(--bg); border-radius:12px; padding:10px; text-align:center; }
  .stat b { display:block; font-size:20px; }
  .stat span { color:var(--muted); font-size:12px; }
  canvas { width:100%; height:64px; }
  ul.timeline { list-style:none; max-height:420px; overflow-y:auto; }
  ul.timeline li { display:flex; gap:10px; padding:8px 2px; border-bottom:1px solid var(--line);
                   font-size:13.5px; align-items:baseline; }
  .dot { width:9px; height:9px; border-radius:50%; flex:none; position:relative; top:1px; }
  .dot.loud { background:var(--bad); } .dot.soft { background:var(--warn); } .dot.log { background:var(--muted); }
  .t { color:var(--muted); font-variant-numeric:tabular-nums; flex:none; }
  .ev b { font-weight:600; } .ev .sig { color:var(--muted); font-size:12px; display:block; }
  .counts { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { background:var(--bg); border-radius:999px; padding:4px 10px; font-size:12.5px; }
  .chip b { color:var(--accent); }
  .empty { color:var(--muted); font-size:13px; padding:14px 2px; }
</style></head><body><div class="wrap">
<header><h1>SensorDyme</h1><div class="sub" id="sub">connecting…</div></header>
<div class="grid">
  <div>
    <div class="card video">
      <img id="cam" src="/stream" alt="live camera">
      <div class="pill" id="pill">OK</div>
      <div class="flash-overlay" id="flashOverlay"></div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Risk — last 6 minutes</h2>
      <canvas id="spark" width="900" height="120"></canvas>
    </div>
  </div>
  <div>
    <div class="card" id="modecard" style="display:none; margin-bottom:14px">
      <h2>Mode</h2>
      <div class="counts" id="modes"></div>
    </div>
    <div class="card">
      <h2>Session</h2>
      <div class="stats">
        <div class="stat"><b id="uptime">0:00</b><span>uptime</span></div>
        <div class="stat"><b id="total">0</b><span>events</span></div>
        <div class="stat"><b id="risknow">0%</b><span>risk now</span></div>
      </div>
      <div class="counts" id="counts"></div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Alert timeline</h2>
      <ul class="timeline" id="timeline"><li class="empty">no events yet</li></ul>
    </div>
  </div>
</div></div>
<script>
// Carry the dashboard token, when there is one, on every request this page
// makes. Without this, setting `dashboard.token` loaded the page and then 401'd
// every /api call and the video stream -- the page came up blank and the
// protection was unusable, so in practice nobody turned it on and the camera
// stream stayed readable by anyone on the network.
//
// The token is read from this page's own URL (?token=...), which is how an
// operator opens it. The MJPEG <img> gets it as a query parameter because an
// <img> cannot set a header; every fetch gets it as a header, which keeps it
// out of proxy logs and browser history for the calls that can manage it.
const TOKEN = new URLSearchParams(location.search).get("token") || "";
function authFetch(url, opts){
  opts = opts || {};
  if(TOKEN){ opts.headers = Object.assign({}, opts.headers, {"X-Device-Token": TOKEN}); }
  return fetch(url, opts);
}
if(TOKEN){
  const cam = document.getElementById("cam");
  if(cam) cam.src = "/stream?token=" + encodeURIComponent(TOKEN);
}
// Everything below that reaches innerHTML is escaped first. Event signals carry
// rule descriptions, and those come from the AI rule interpreter: text a model
// wrote is not markup this page may execute, on the origin holding TOKEN.
const ESC = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
function esc(v){ return String(v == null ? "" : v).replace(/[&<>"']/g, ch => ESC[ch]); }
function fmt(s){const m=Math.floor(s/60),h=Math.floor(m/60);return h?h+":"+String(m%60).padStart(2,"0")+":"+String(s%60).padStart(2,"0"):m+":"+String(s%60).padStart(2,"0");}
async function tick(){
  try{
    const s = await (await authFetch("/api/state")).json();
    // M8.4: say why the picture is empty instead of leaving a frozen image.
    const cam = s.camera || {};
    document.getElementById("sub").textContent = s.module + (
      cam.remote_stream === "off_while_stopped" ? " · camera off — monitoring stopped"
      : cam.remote_stream === "camera_only" ? " · camera only — not monitoring" : " · live");
    document.getElementById("uptime").textContent = fmt(s.uptime_s);
    document.getElementById("total").textContent = s.total_events;
    const risk = s.risk.length ? s.risk[s.risk.length-1] : 0;
    document.getElementById("risknow").textContent = Math.round(risk*100)+"%";
    const pill = document.getElementById("pill");
    pill.textContent = s.alert_active ? "ALERT" : "OK";
    pill.className = "pill" + (s.alert_active ? " alert" : "");
    document.getElementById("flashOverlay").classList.toggle("on", !!s.flash_active);
    document.getElementById("counts").innerHTML = s.counts.map(c=>`<span class="chip">${esc(c.type)} <b>${esc(c.n)}</b></span>`).join("") || '<span class="empty">no alerts yet</span>';
    document.getElementById("timeline").innerHTML = s.events.length ? s.events.map(e=>
      `<li><span class="dot ${esc(e.severity)}"></span><span class="t">${esc(e.time)}</span>
       <span class="ev"><b>${esc(e.type)}</b><span class="sig">${esc(e.signals)}</span></span></li>`).join("")
      : '<li class="empty">no events yet</li>';
    const cv = document.getElementById("spark"), ctx = cv.getContext("2d");
    ctx.clearRect(0,0,cv.width,cv.height);
    const st = getComputedStyle(document.documentElement);
    ctx.strokeStyle = st.getPropertyValue("--accent"); ctx.lineWidth = 2.5; ctx.beginPath();
    const n = s.risk.length;
    for(let i=0;i<n;i++){const x=i/(Math.max(n-1,1))*cv.width, y=cv.height-6-s.risk[i]*(cv.height-12);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);}
    if(n>1) ctx.stroke();
  }catch(e){ document.getElementById("sub").textContent = "reconnecting…"; }
}
async function loadModes(){
  try{
    const p = await (await authFetch("/api/presets")).json();
    if(!p.presets.length) return;
    const card = document.getElementById("modecard"); card.style.display = "block";
    document.getElementById("modes").innerHTML = p.presets.map(m=>
      `<button class="chip" style="cursor:pointer;border:1px solid ${m.active?'var(--accent)':'var(--line)'};
        ${m.active?'font-weight:700;color:var(--accent);':''}" title="${esc(m.description)}"
        data-name="${esc(m.name)}" onclick="setMode(this.dataset.name)">${esc(m.title)}</button>`).join("");
  }catch(e){}
}
async function setMode(name){
  await authFetch("/api/preset", {method:"POST", headers:{"Content-Type":"application/json"},
                              body:JSON.stringify({preset:name})});
  setTimeout(loadModes, 800);
}
tick(); setInterval(tick, 400); loadModes(); setInterval(loadModes, 5000);
</script></body></html>
"""
