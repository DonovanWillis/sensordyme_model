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
"""

from __future__ import annotations

import hmac
import json
import select
import socket
import threading
import time
from collections import Counter, deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

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


class DashboardServer:
    """Threaded local web app fed by the main loop; never blocks the camera."""

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
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
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
                if not sent:
                    # Kept for the MJPEG <img> case, which cannot set a header. It is
                    # a real leak -- a token in a URL lands in proxy logs, browser
                    # history and Referer -- so it stays the fallback, never the
                    # documented path, and the app itself uses the header.
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
                sock = self.connection
                try:
                    readable = select.select([sock], [], [], 0)[0]
                except (OSError, ValueError):
                    return True  # socket already closed underneath us
                if not readable:
                    return False
                try:
                    # Readable on a request whose body is complete means EOF. Peek
                    # rather than read: nothing may be consumed from the stream.
                    return sock.recv(1, socket.MSG_PEEK) == b""
                except (BlockingIOError, InterruptedError):
                    return False
                except OSError:
                    return True

            def _json(self, obj: dict, status: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
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
                        dashboard.request_preset(name)
                        self._json({"ok": True, "switching_to": name}, 202)
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
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    elif self.path.startswith("/api/health"):
                        provider = dashboard._health_provider
                        self._json(provider() if provider is not None else {"state": "unknown"})
                    elif self.path.startswith("/api/history"):
                        limit = 200
                        if "limit=" in self.path:
                            try:
                                limit = min(2000, int(self.path.rsplit("limit=", 1)[1].split("&")[0]))
                            except ValueError:
                                pass
                        self._json({"events": dashboard.history(limit)})
                    elif self.path.startswith("/stream"):
                        self.send_response(200)
                        self.send_header(
                            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                        )
                        self.end_headers()
                        # Registering as a viewer is what makes the main loop start
                        # annotating and encoding at all -- see has_viewers().
                        dashboard.viewer_joined()
                        try:
                            last_seq = -1
                            # `not self._peer_gone()`, not `while True`: the loop used
                            # to learn that a client had left ONLY by having a write
                            # fail, and during any window with no fresh frames -- a
                            # camera stall, a degraded detector, or simply the gap
                            # before the first frame is encoded for a newly connected
                            # viewer -- there is no write to fail. See _peer_gone().
                            while not self._peer_gone():
                                jpeg, seq = dashboard.latest_jpeg()
                                if jpeg is not None and seq != last_seq:
                                    last_seq = seq
                                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                                    self.wfile.write(jpeg)
                                    self.wfile.write(b"\r\n")
                                time.sleep(dashboard.stream_interval)
                        finally:
                            dashboard.viewer_left()
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

        self._httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._port = self._httpd.server_address[1]  # resolve port 0 -> actual
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True, name="dashboard")
        self._thread = thread  # is_alive() checks this, not merely that _httpd exists
        thread.start()
        return self.url

    @property
    def url(self) -> str:
        host = "localhost" if self._host in ("127.0.0.1", "0.0.0.0") else self._host
        return f"http://{host}:{self._port}"

    def close(self) -> None:
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

    # ------------------------------------------------------------------ feeders (main loop)
    def push_frame(self, annotated_bgr: np.ndarray) -> None:
        """Encode the annotated frame to an in-memory JPEG for the live stream."""
        ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
                self._jpeg_seq += 1

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
        the page stays fully testable on a dev Mac)."""
        from core.network import WifiManager

        return WifiManager().connect(ssid, password)

    # ------------------------------------------------------------------ readers (handler)
    def latest_jpeg(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq

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
async function go(){
  const m=document.getElementById('msg'); m.textContent='Connecting…';
  const r=await fetch('/setup/wifi',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ssid:document.getElementById('ssid').value,password:document.getElementById('pw').value})});
  const j=await r.json(); m.textContent=j.message; m.style.color=j.ok?'#2BD9A9':'#FF6B60';
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
    document.getElementById("sub").textContent = s.module + " · live";
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
