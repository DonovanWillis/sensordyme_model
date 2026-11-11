"""Scenario controller for the iOS UI tests (ios/UITests, scripts/sim/ios_uitest.sh).

    python scripts/sim/uitest_control.py --work DIR [--app-port 8787] [--backend-port 8797] [--control-port 8790]

Runs the real backend (sim.backend, throwaway data in DIR) and one virtual
SensorDyme (sim.virtual_device), and puts a small HTTP proxy in front of the
backend on --app-port, the address a Simulator build uses by default
(http://localhost:8787). The device reports to the backend directly, so a fault
injected at the proxy is something only the app sees, exactly like a phone that
loses the server while the device keeps working.

The UI tests drive scenarios through the control port (127.0.0.1 only):

    GET  /state                       mode, device process, scenario, claim code, demo account
    GET  /session                     a fresh login for the demo account (token, email) for -orynSession
    POST /mode?m=pass|down|garbage    proxy: forward / stop listening (server down) / answer invalid JSON
    POST /device/stop                 kill the virtual device process (it goes offline)
    POST /device/start?scenario=NAME  (re)start it with a scenario from sim/scenario.py
    POST /claim                       claim the device for the demo account (what "Add device" does)
    POST /protection?action=stop|resume
    POST /reset                       server answering, custom rules off, plain driver_safety mode,
                                      device running "attentive", protection resumed

Every request is answered with JSON. Nothing leaves the machine; no camera is used.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "content-encoding"}


class Stack:
    def __init__(self, work: Path, backend_port: int, python: str) -> None:
        self.work, self.python = work, python
        self.backend_url = f"http://127.0.0.1:{backend_port}"
        self.backend_port = backend_port
        self.mode = "pass"
        self.scenario: str | None = None
        self.backend: subprocess.Popen | None = None
        self.device: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self._token: str | None = None

    # ------------------------------------------------------------------ processes
    def _env(self) -> dict[str, str]:
        return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO)}

    def start_backend(self) -> None:
        log = open(self.work / "backend.log", "ab")  # noqa: SIM115 - lives as long as the process
        self.backend = subprocess.Popen(
            [self.python, "-m", "sim.backend", "--port", str(self.backend_port), "--data-dir",
             str(self.work / "backend")], cwd=REPO, env=self._env(), stdout=log, stderr=subprocess.STDOUT)
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

    def close(self) -> None:
        self.stop_device()
        if self.backend and self.backend.poll() is None:
            self.backend.terminate()
            self.backend.wait(10)

    # ------------------------------------------------------------------ backend API
    def api(self, method: str, path: str, body: dict | None = None, auth: bool = True) -> tuple[int, object]:
        headers = {"Content-Type": "application/json"}
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
            _, answer = self.api("POST", "/v1/login", {"email": DEMO_EMAIL, "password": DEMO_PASSWORD}, auth=False)
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
        _, answer = self.api("POST", "/v1/login", {"email": DEMO_EMAIL, "password": DEMO_PASSWORD}, auth=False)
        return {"token": answer["token"], "email": DEMO_EMAIL}  # type: ignore[index]

    def state(self) -> dict:
        device_id = self.device_id()
        events = 0
        if device_id is not None:
            status, answer = self.api("GET", f"/v1/devices/{device_id}/events?limit=50")
            events = len(answer.get("events", [])) if status == 200 and isinstance(answer, dict) else 0
        return {"mode": self.mode, "scenario": self.scenario,
                "device_running": bool(self.device and self.device.poll() is None),
                "claim_code": self.factory()["devices"][0]["claim_code"],
                "email": DEMO_EMAIL, "password": DEMO_PASSWORD, "device_id": device_id, "events": events}


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
            if stack.mode == "garbage" and self.path.startswith("/v1/") and self.path != "/v1/health":
                self._send(200, GARBAGE, "application/json")
                return
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            request = urllib.request.Request(f"{stack.backend_url}{self.path}", data=body,  # noqa: S310
                                             method=self.command, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as r:  # noqa: S310
                    self._send(r.status, r.read(), r.headers.get("Content-Type"), r.headers.items())
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
                self._reply(200, stack.state())
            elif path == "/session":
                self._reply(200, stack.session())
            else:
                self._reply(404, {"error": "unknown"})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                self._reply(200, self._act(url.path, q))
            except (ValueError, KeyError) as e:
                self._reply(400, {"error": str(e)})

        def _act(self, path: str, q: dict[str, str]) -> object:
            if path == "/mode":
                if q.get("m") not in ("pass", "down", "garbage"):
                    raise ValueError("m must be pass, down or garbage")
                stack.mode = q["m"]
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
