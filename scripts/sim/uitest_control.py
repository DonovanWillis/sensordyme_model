"""Scenario controller for the iOS UI tests (ios/UITests, scripts/sim/ios_uitest.sh).

    python scripts/sim/uitest_control.py --work DIR [--app-port 8787] [--backend-port 8797] [--control-port 8790]

Runs the real backend (sim.backend, throwaway data in DIR) and one virtual
SensorDyme (sim.virtual_device), and puts a small HTTP proxy in front of the
backend on --app-port, the address a Simulator build uses by default
(http://localhost:8787). The device reports to the backend directly, so a fault
injected at the proxy is something only the app sees, exactly like a phone that
loses the server while the device keeps working.

The UI tests drive scenarios through the control port (127.0.0.1 only):

    POST /fresh?claimed=1|0[&tag=T]   EVERY TEST STARTS HERE (ios/UITests/Stack.swift fresh()):
                                      a brand-new world. The device and backend are stopped and
                                      their data deleted; a new backend (new database and JWT secret,
                                      empty rate limiters, no events/rules/zones/drafts) starts with
                                      a NEW account (uitest-<n>@sensordyme.local) and a newly
                                      provisioned device with a new identity; proxy faults are off.
                                      claimed=1 (the default) claims the device for the account;
                                      claimed=0 leaves it unclaimed (the discovery journey pairs it).
                                      Answers once the device has reported to the new backend, so no
                                      test starts against a device that is still booting (~20 s).
    GET  /state                       mode, device process, scenario, claim code, current account
    GET  /session                     a fresh login for the current account (token, email) for -orynSession
    POST /mode?m=MODE[&path=/v1/...]  what the proxy does with app requests under `path` (default: every
                                      /v1/ request except /v1/health):
                                        pass     forward
                                        down     stop listening (server down; `path` ignored)
                                        garbage  200 with invalid JSON
                                        html     200 with a web page (a Wi-Fi sign-in page)
                                        empty    200 with an empty body
                                        error500 500 {"detail": "database is locked"}
                                        gateway  502 with a reverse proxy's HTML error page
                                        freeze   replay the last good GET /v1/devices reply (stale
                                                 telemetry: online=true, last_seen frozen); other paths pass
    POST /device/stop                 kill the virtual device process (it goes offline)
    POST /device/start?scenario=NAME  (re)start it with a scenario from sim/scenario.py
    POST /claim                       claim the device for the demo account (what "Add device" does)
    POST /protection?action=stop|resume
    POST /reset                       (manual use) server answering, custom rules off, plain driver_safety
                                      mode, device running "attentive", protection resumed. Keeps the
                                      account, pairing and event history; the tests use /fresh.

Every request is answered with JSON. Nothing leaves the machine; no camera is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
DEMO_EMAIL = "demo@sensordyme.local"
DEMO_PASSWORD = "sensordyme-demo"  # noqa: S105 - the throwaway sim.backend demo account
GARBAGE = b'{"devices": [ this is not JSON'
PORTAL = b"<!DOCTYPE html><html><head><title>Sign in to Wi-Fi</title></head><body>Accept terms</body></html>"
GATEWAY = (b"<html><head><title>502 Bad Gateway</title></head><body><center><h1>502 Bad Gateway</h1></center>"
           b"<hr><center>nginx</center></body></html>")
MODES = ("pass", "down", "garbage", "html", "empty", "error500", "gateway", "freeze")
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "content-encoding"}
# The controller signs the demo account in once per app launch (see session()),
# far more often than the backend's per-address login limit (10 per 5 minutes)
# allows from 127.0.0.1. The backend under test is told to key its rate limits
# on this header (SENSORDYME_CLIENT_IP_HEADER, cloud/app.py _client_ip), the
# controller's own logins carry a distinct value each, and the proxy strips it
# from the app's requests, whose limits are unchanged (keyed on 127.0.0.1).
CONTROLLER_CLIENT_HEADER = "X-UITest-Controller"


class Stack:
    def __init__(self, work: Path, backend_port: int, python: str) -> None:
        self.work, self.python = work, python
        # The account the tests sign in with. /fresh replaces it with a new one.
        self.email, self.password = DEMO_EMAIL, DEMO_PASSWORD
        self.generation = 0
        self.backend_url = f"http://127.0.0.1:{backend_port}"
        self.backend_port = backend_port
        self.mode = "pass"
        self.fault_path = "/v1/"
        # The last good GET /v1/devices reply the app received (for "freeze").
        self.last_devices: bytes | None = None
        self.scenario: str | None = None
        self.backend: subprocess.Popen | None = None
        self.device: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._token: str | None = None

    # ------------------------------------------------------------------ processes
    def _env(self) -> dict[str, str]:
        return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO),
                "SENSORDYME_CLIENT_IP_HEADER": CONTROLLER_CLIENT_HEADER}

    def start_backend(self) -> None:
        log = open(self.work / "backend.log", "ab")  # noqa: SIM115 - lives as long as the process
        self.backend = subprocess.Popen(
            [self.python, "-m", "sim.backend", "--port", str(self.backend_port), "--data-dir",
             str(self.work / "backend"), "--email", self.email, "--password", self.password],
            cwd=REPO, env=self._env(), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.backend_url}/v1/health", timeout=2):  # noqa: S310
                    return
            except OSError:
                time.sleep(0.3)
        raise SystemExit("backend did not come up; see backend.log")

    def factory(self) -> dict:
        return json.loads((self.work / "backend" / "factory.json").read_text())

    def start_device(self, scenario: str) -> None:
        with self.lock:
            self._stop_device_locked()
            log = open(self.work / "device.log", "ab")  # noqa: SIM115
            self.device = subprocess.Popen(
                [self.python, "-m", "sim.virtual_device", "--backend", self.backend_url,
                 "--factory", str(self.work / "backend" / "factory.json"),
                 "--state-dir", str(self.work / "device"), "--scenario", scenario, "--no-dashboard"],
                cwd=REPO, env=self._env(), stdout=log, stderr=subprocess.STDOUT)
            self.scenario = scenario

    def _stop_device_locked(self) -> None:
        if self.device and self.device.poll() is None:
            self.device.terminate()
            try:
                self.device.wait(10)
            except subprocess.TimeoutExpired:
                self.device.kill()
        self.device = None
        self.scenario = None

    def stop_device(self) -> None:
        with self.lock:
            self._stop_device_locked()

    def stop_backend(self) -> None:
        if self.backend and self.backend.poll() is None:
            self.backend.terminate()
            try:
                self.backend.wait(10)
            except subprocess.TimeoutExpired:
                self.backend.kill()
                self.backend.wait(10)
        self.backend = None

    def close(self) -> None:
        self.stop_device()
        self.stop_backend()

    def fresh(self, claimed: bool, tag: str) -> dict:
        """A new world for one test (POST /fresh in the module docstring).

        Nothing a previous test did can reach the next one: the backend's whole
        data directory (database, JWT secret, factory file) and the device's
        state directory are deleted, and the backend's in-memory rate limiters
        go with its process. The account is new, so the app's own per-account
        reset (AppState.claimLocalState: paired camera, announced alerts, SMS
        number) runs on the next signed-in launch too. The logs are appended
        to, with a marker per test."""
        self.generation += 1
        self.email = f"uitest-{self.generation}@sensordyme.local"
        for name in ("backend.log", "device.log"):
            with open(self.work / name, "a") as f:
                f.write(f"\n===== fresh #{self.generation} for {tag or '?'} "
                        f"({time.strftime('%H:%M:%S')}), account {self.email} =====\n")
        self.stop_device()
        self.stop_backend()
        for name in ("backend", "device"):
            shutil.rmtree(self.work / name, ignore_errors=True)
        self._token = None
        self.last_devices = None
        self.start_backend()
        self.start_device("attentive")
        if claimed:
            status, answer = self.api("POST", "/v1/devices/claim",
                                      {"claim_code": self.factory()["devices"][0]["claim_code"]})
            if status != 200:
                raise ValueError(f"claim failed ({status}): {answer}")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not self.device_reported():
            if not (self.device and self.device.poll() is None):
                raise ValueError("the virtual device exited while starting; see device.log")
            time.sleep(0.5)
        if not self.device_reported():
            raise ValueError("the virtual device did not report to the new backend within 90 s")
        # A claimed device is handed over READY (calibrated, protecting), the state
        # the journeys and the audit start from: a screen audited while the device
        # was still calibrating shows other text than one audited after, and the
        # audit's findings then depended on how fast the previous test finished.
        while claimed and time.monotonic() < deadline and not self.device_ready():
            time.sleep(0.5)
        if claimed and not self.device_ready():
            raise ValueError("the claimed device did not report ready within 90 s; see device.log")
        return self.state()

    def device_ready(self) -> bool:
        status, answer = self.api("GET", "/v1/devices")
        devices = answer.get("devices") if status == 200 and isinstance(answer, dict) else None
        state = (devices[0].get("status") or {}) if devices else {}
        return (bool(devices and devices[0].get("online")) and state.get("monitoring_state") == "ready"
                and state.get("protection") == "protecting")

    def device_reported(self) -> bool:
        """Whether the device has reported to the current backend yet. Read from the
        throwaway database, so it works for an unclaimed device as well."""
        db = self.work / "backend" / "sensordyme.db"
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
            try:
                row = conn.execute("SELECT MAX(last_seen) FROM devices").fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return False
        return bool(row and row[0])

    # ------------------------------------------------------------------ backend API
    def api(self, method: str, path: str, body: dict | None = None, auth: bool = True) -> tuple[int, object]:
        self._calls = getattr(self, "_calls", 0) + 1
        headers = {"Content-Type": "application/json", CONTROLLER_CLIENT_HEADER: f"uitest-controller-{self._calls}"}
        if auth:
            headers["Authorization"] = f"Bearer {self.token()}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.backend_url}{path}", data=data, method=method, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=10) as r:  # noqa: S310
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    def token(self) -> str:
        if self._token is None:
            _, answer = self.api("POST", "/v1/login", {"email": self.email, "password": self.password}, auth=False)
            self._token = answer["token"]  # type: ignore[index]
        return self._token

    def device_id(self) -> int | None:
        status, answer = self.api("GET", "/v1/devices")
        devices = answer.get("devices") if isinstance(answer, dict) else None
        return devices[0]["id"] if status == 200 and devices else None

    def reset_device(self, device_id: int) -> None:
        """Undo what a journey may have left behind. On an additive role, choosing the
        mode again does not turn custom rules off, so each rule is disabled the way the
        app's switch does it (PATCH .../rules/{id})."""
        status, answer = self.api("GET", f"/v1/devices/{device_id}/rules")
        rules = answer.get("rules", []) if status == 200 and isinstance(answer, dict) else []
        for rule in rules:
            if rule.get("enabled"):
                self.api("PATCH", f"/v1/devices/{device_id}/rules/{rule['id']}", {"enabled": False})
        self.api("POST", f"/v1/devices/{device_id}/preset", {"preset": "driver_safety"})
        self.api("POST", f"/v1/devices/{device_id}/protection", {"action": "resume"})

    def session(self) -> dict:
        # A fresh token per app launch: the app's -orynSignOut revokes the token it
        # held, so one shared token would be revoked by the next launch.
        status, answer = self.api("POST", "/v1/login", {"email": self.email, "password": self.password}, auth=False)
        token = answer.get("token") if isinstance(answer, dict) else None
        if not token:
            # Said, not a crashed request: the test then launched the app with an
            # empty session and failed far from the cause.
            raise ValueError(f"demo login failed ({status}): {answer}")
        return {"token": token, "email": self.email}

    def state(self) -> dict:
        device_id = self.device_id()
        events = 0
        if device_id is not None:
            # The backend's largest page: a count capped at 50 stopped growing, and a
            # journey waiting for "more events than before" then waited in vain.
            status, answer = self.api("GET", f"/v1/devices/{device_id}/events?limit=1000")
            events = len(answer.get("events", [])) if status == 200 and isinstance(answer, dict) else 0
        return {"mode": self.mode, "fault_path": self.fault_path, "scenario": self.scenario,
                "device_running": bool(self.device and self.device.poll() is None),
                "claim_code": self.factory()["devices"][0]["claim_code"],
                "email": self.email, "password": self.password, "device_id": device_id, "events": events,
                "generation": self.generation}


class AppPort:
    """The address the app talks to. "down" closes the listening socket, so the app
    sees what a stopped server looks like (connection refused), not a 5xx."""

    def __init__(self, port: int, stack: Stack) -> None:
        self.port, self.stack = port, stack
        self.server: ThreadingHTTPServer | None = None
        self.lock = threading.Lock()

    def up(self) -> None:
        with self.lock:
            if self.server is None:
                self.server = ThreadingHTTPServer(("127.0.0.1", self.port), proxy_handler(self.stack))
                self.server.daemon_threads = True
                threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def down(self) -> None:
        with self.lock:
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
                self.server = None


def proxy_handler(stack: Stack):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet; the backend logs requests
            pass

        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            faulted = self.path.startswith(stack.fault_path) and not self.path.startswith("/v1/health")
            is_devices = self.command == "GET" and urlparse(self.path).path == "/v1/devices"
            if faulted and stack.mode in ("garbage", "html", "empty", "error500", "gateway"):
                self._send(*{
                    "garbage": (200, GARBAGE, "application/json"),
                    "html": (200, PORTAL, "text/html"),
                    "empty": (200, b"", "application/json"),
                    "error500": (500, b'{"detail": "database is locked"}', "application/json"),
                    "gateway": (502, GATEWAY, "text/html"),
                }[stack.mode])
                return
            if faulted and stack.mode == "freeze" and is_devices and stack.last_devices is not None:
                self._send(200, stack.last_devices, "application/json")
                return
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP_HEADERS and k.lower() != CONTROLLER_CLIENT_HEADER.lower()}
            request = urllib.request.Request(f"{stack.backend_url}{self.path}", data=body,  # noqa: S310
                                             method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as r:  # noqa: S310
                    payload = r.read()
                    if is_devices and r.status == 200:
                        stack.last_devices = payload
                    self._send(r.status, payload, r.headers.get("Content-Type"), r.headers.items())
            except urllib.error.HTTPError as e:
                self._send(e.code, e.read(), e.headers.get("Content-Type"), e.headers.items())
            except OSError:
                self._send(502, b'{"detail": "backend unreachable"}', "application/json")

        def _send(self, status: int, payload: bytes, ctype: str | None, extra=()) -> None:
            self.send_response(status)
            for k, v in extra:
                if k.lower() not in HOP_HEADERS and k.lower() != "content-type":
                    self.send_header(k, v)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _forward  # noqa: N815

    return Proxy


def control_handler(stack: Stack, app_port: AppPort, stop: threading.Event):
    scenario_lock = threading.Lock()

    class Control(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            with open(stack.work / "control.log", "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {fmt % args}\n")

        def _reply(self, status: int, payload: object) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            # One request per connection: a reused idle connection that this
            # server had already closed reached the test as -1005.
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/state":
                with scenario_lock:
                    answer = stack.state()
                self._reply(200, answer)
            elif path == "/session":
                try:
                    with scenario_lock:
                        answer = stack.session()
                    self._reply(200, answer)
                except ValueError as e:
                    self._reply(503, {"error": str(e)})
            else:
                self._reply(404, {"error": "unknown"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                # One scenario change at a time: a request arriving while /fresh
                # is replacing the backend would act on half a world.
                with scenario_lock:
                    answer = self._act(url.path, q)
                self._reply(200, answer)
            except (ValueError, KeyError) as e:
                self._reply(400, {"error": str(e)})

        def _act(self, path: str, q: dict[str, str]) -> object:
            if path == "/fresh":
                stack.mode = "pass"
                stack.fault_path = "/v1/"
                app_port.up()
                return stack.fresh(claimed=q.get("claimed", "1") != "0", tag=q.get("tag", ""))
            if path == "/mode":
                if q.get("m") not in MODES:
                    raise ValueError(f"m must be one of {', '.join(MODES)}")
                stack.mode = q["m"]
                stack.fault_path = q.get("path", "/v1/")
                app_port.down() if stack.mode == "down" else app_port.up()
            elif path == "/device/stop":
                stack.stop_device()
            elif path == "/device/start":
                stack.start_device(q.get("scenario", "attentive"))
            elif path == "/claim":
                code = stack.factory()["devices"][0]["claim_code"]
                return dict(zip(("status", "answer"), stack.api("POST", "/v1/devices/claim", {"claim_code": code}),
                                strict=True))
            elif path == "/protection":
                action = q.get("action")
                if action not in ("stop", "resume") or stack.device_id() is None:
                    raise ValueError("action must be stop or resume, on a claimed device")
                body = {"action": action, **({"confirm": True} if action == "stop" else {})}
                status, answer = stack.api("POST", f"/v1/devices/{stack.device_id()}/protection", body)
                return {"status": status, "answer": answer}
            elif path == "/reset":
                stack.mode = "pass"
                stack.fault_path = "/v1/"
                app_port.up()
                if not (stack.device and stack.device.poll() is None and stack.scenario == "attentive"):
                    stack.start_device("attentive")
                device_id = stack.device_id()
                if device_id is not None:
                    stack.reset_device(device_id)
            elif path == "/shutdown":
                stop.set()
            else:
                raise KeyError(f"unknown action {path}")
            return stack.state()

    return Control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--app-port", type=int, default=8787)
    parser.add_argument("--backend-port", type=int, default=8797)
    parser.add_argument("--control-port", type=int, default=8790)
    parser.add_argument("--python", default=sys.executable, help="interpreter with the device + backend requirements")
    parser.add_argument("--claimed", action="store_true", help="claim the device for the demo account at start")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)

    stack = Stack(args.work, args.backend_port, args.python)
    stop = threading.Event()
    app_port = AppPort(args.app_port, stack)
    control = ThreadingHTTPServer(("127.0.0.1", args.control_port), control_handler(stack, app_port, stop))
    control.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    serving = False
    try:
        stack.start_backend()
        stack.start_device("attentive")
        if args.claimed:
            stack.api("POST", "/v1/devices/claim", {"claim_code": stack.factory()["devices"][0]["claim_code"]})
        app_port.up()
        threading.Thread(target=control.serve_forever, daemon=True).start()
        serving = True
        print(f"uitest_control: app proxy :{args.app_port} -> backend :{args.backend_port}; "
              f"control :{args.control_port}", flush=True)
        stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if serving:
            control.shutdown()
        app_port.down()
        stack.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
