"""Run a virtual SensorDyme device against a backend.

    python -m sim.virtual_device --with-backend                 # backend + device, one command
    python -m sim.virtual_device --factory DIR/factory.json     # against `python -m sim.backend`
    python -m sim.virtual_device --backend http://127.0.0.1:8787 --device-key dk_... --claim-code AB12-...
    python -m sim.virtual_device --list-scenarios

The device is the real runtime (services/runtime.py via main.build_runtime)
with a simulated camera and scripted perception; it reports to the backend
over the real uplink with its device key, and is paired by typing its claim
code into the app (or --auto-claim with the factory's demo account). Its
state -- identity, pairing key, mode, rules, calibration -- lives in
--state-dir, so stopping and starting it again is a reboot, not a new device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
DEFAULT_STATE = Path.home() / ".sensordyme-sim" / "device"


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m sim.virtual_device",
                                     description="A software SensorDyme (SIMULATED) for the app and tests.")
    parser.add_argument("--backend", default="http://127.0.0.1:8787", help="backend base URL")
    parser.add_argument("--scenario", default="attentive", help="scenario name (--list-scenarios)")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--claim-code", default=None,
                        help="this device's claim code (selects the device in --factory)")
    parser.add_argument("--device-key", default=None, help="device key issued by cloud/manage.py create-device")
    parser.add_argument("--factory", default=None, help="factory.json written by `python -m sim.backend`")
    parser.add_argument("--with-backend", action="store_true",
                        help="also start the real backend locally (throwaway data) on --backend's port")
    parser.add_argument("--auto-claim", action="store_true",
                        help="pair to the factory's demo account (what the app's claim screen does)")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE), help=f"device state (default {DEFAULT_STATE})")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="scenario timeline speed (the device's own thresholds stay real-time)")
    parser.add_argument("--video", default=None,
                        help="feed frames from a video file and run the REAL perception models instead of "
                             "scripted perception (needs the models installed; without a detector the "
                             "device honestly reports DEGRADED)")
    parser.add_argument("--dashboard-port", type=int, default=8899, help="device local API / live feed port")
    parser.add_argument("--dashboard-host", default="127.0.0.1",
                        help="bind address of the local API (0.0.0.0 to reach it from a phone on the LAN)")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--mdns", action="store_true",
                        help="advertise the device as _sensordyme._tcp (what the Pi's avahi does)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def _wait_for_backend(url: str, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/health", timeout=2) as response:  # noqa: S310 - local dev URL
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.3)
    return False


def _start_backend(url: str) -> tuple[subprocess.Popen, Path]:
    parts = urlsplit(url)
    data_dir = Path(tempfile.mkdtemp(prefix="sensordyme-sim-backend-"))
    process = subprocess.Popen(  # noqa: S603 - our own module, fixed argv
        [sys.executable, "-m", "sim.backend", "--host", parts.hostname or "127.0.0.1",
         "--port", str(parts.port or 8787), "--data-dir", str(data_dir)],
        cwd=str(REPO),
    )
    if not _wait_for_backend(url):
        process.terminate()
        raise SystemExit(f"sim: the backend did not come up at {url}")
    return process, data_dir / "factory.json"


def _from_factory(path: Path, claim_code: Optional[str]) -> dict:
    factory = json.loads(Path(path).read_text())
    devices = factory.get("devices") or []
    if claim_code:
        devices = [d for d in devices if d["claim_code"] == claim_code]
    if not devices:
        raise SystemExit(f"sim: no device{' with claim code ' + claim_code if claim_code else ''} in {path}")
    return {**devices[0], "email": factory.get("email"), "password": factory.get("password")}


def _auto_claim(backend: str, email: str, password: str, claim_code: str) -> None:
    import requests

    login = requests.post(f"{backend}/v1/login", json={"email": email, "password": password}, timeout=10)
    login.raise_for_status()
    headers = {"Authorization": f"Bearer {login.json()['token']}"}
    claim = requests.post(f"{backend}/v1/devices/claim", json={"claim_code": claim_code}, headers=headers,
                          timeout=10)
    print(f"sim: auto-claim as {email}: HTTP {claim.status_code} {claim.text[:120]}")


def _advertise(identity, port: int, host: str):
    """_sensordyme._tcp, as deploy/avahi + scripts/publish_mdns.py do on the Pi.

    The address is 127.0.0.1 for a loopback-bound local API (the iOS Simulator
    shares the Mac's network stack, so that is the Mac), else this Mac's LAN
    address (from cloud/mdns.py, the backend's own advertiser).
    """
    try:
        import socket

        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        print("sim: --mdns needs python-zeroconf; not advertising")
        return None
    address = "127.0.0.1"
    if host not in ("127.0.0.1", "localhost"):
        from cloud.mdns import lan_ip

        address = lan_ip() or address
    suffix = identity.hardware_id.removeprefix("SD-SIM-").lower()
    zc = Zeroconf()
    info = ServiceInfo(
        "_sensordyme._tcp.local.", f"SensorDyme SIMULATED {identity.hardware_id}._sensordyme._tcp.local.",
        addresses=[socket.inet_aton(address)], port=port, server=f"sensordyme-sim-{suffix}.local.",
        properties={"path": "/stream", "version": "0.2", "id": identity.hardware_id, "simulated": "1"},
    )
    zc.register_service(info)
    print(f"sim: advertising _sensordyme._tcp at {address}:{port} (id {identity.hardware_id})")
    return zc, info


def _health_freeze_signals(device) -> None:
    """A fault for the iOS UI tests (scripts/sim/uitest_control.py POST /health):
    SIGUSR1 freezes the health this device reports while it keeps polling (the
    sampling thread stops handing samples to the uplink, which re-sends the last
    one), SIGUSR2 lets samples through again. The same fault as
    tests/test_m81_health_freshness_sim.py `_freeze`; a restart clears it.
    The acknowledgement is written with os.write: print() from a signal handler
    can re-enter the buffered stdout the device is printing to."""
    import signal

    if not hasattr(signal, "SIGUSR1"):
        return

    def say(text: str) -> None:
        os.write(sys.stdout.fileno(), f"SensorDyme SIM: {text}\n".encode())

    def freeze(*_) -> None:
        uplink = getattr(device.runtime, "uplink", None)
        if uplink is not None:
            uplink.set_health = lambda report: None
            say("FAULT health frozen (SIGUSR1): polling continues with the last health sample")

    def thaw(*_) -> None:
        uplink = getattr(device.runtime, "uplink", None)
        if uplink is not None:
            vars(uplink).pop("set_health", None)
            say("FAULT health cleared (SIGUSR2): health samples flow again")

    signal.signal(signal.SIGUSR1, freeze)
    signal.signal(signal.SIGUSR2, thaw)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)
    sys.path.insert(0, str(REPO))
    from sim import scenario as scenarios

    if args.list_scenarios:
        for name, scenario in sorted(scenarios.SCENARIOS.items()):
            print(f"{name:18s} {scenario.description}")
        return 0
    scenario = scenarios.get(args.scenario)

    backend_process = None
    factory_path = Path(args.factory) if args.factory else None
    if args.with_backend:
        backend_process, factory_path = _start_backend(args.backend)

    try:
        device_key, claim_code, account = args.device_key, args.claim_code, None
        if factory_path is not None:
            entry = _from_factory(factory_path, claim_code)
            device_key, claim_code = entry["device_key"], entry["claim_code"]
            account = (entry.get("email"), entry.get("password"))

        state_dir = Path(args.state_dir).expanduser()
        if backend_process is not None:
            # A fresh backend has never seen the old device: a fresh device, too.
            state_dir = Path(tempfile.mkdtemp(prefix="sensordyme-sim-device-"))

        from sim.device import VirtualDevice

        device = VirtualDevice(
            state_dir=state_dir, backend_url=args.backend, scenario=scenario, device_key=device_key,
            claim_code=claim_code, speed=args.speed, video=args.video,
            dashboard_port=None if args.no_dashboard else args.dashboard_port,
            dashboard_host=args.dashboard_host, width=args.width, height=args.height, fps=args.fps,
            echo_alerts=True, log_level=args.log_level,
        )
        identity = device.identity
        print("=" * 72)
        print(f"VIRTUAL SensorDyme (SIMULATED) {identity.hardware_id}")
        print(f"  scenario     {scenario.name}: {scenario.description}")
        print(f"  backend      {args.backend}")
        print(f"  claim code   {identity.claim_code or '(not recorded)'}   <- type this in the app")
        if not args.no_dashboard:
            print(f"  local API    http://{args.dashboard_host}:{args.dashboard_port}/api/health "
                  f"(live feed /stream), device token: {device.dashboard_token}")
        print(f"  state dir    {state_dir}")
        print("  Ctrl+C stops it; starting it again with the same --state-dir is a reboot.")
        print("=" * 72, flush=True)

        if args.auto_claim:
            if not account or not account[0]:
                raise SystemExit("sim: --auto-claim needs --factory or --with-backend (the demo account)")
            _auto_claim(args.backend, account[0], account[1], identity.claim_code)

        advert = (_advertise(identity, args.dashboard_port, args.dashboard_host)
                  if args.mdns and not args.no_dashboard else None)
        _health_freeze_signals(device)
        try:
            device.run()
        finally:
            if advert is not None:
                zc, info = advert
                zc.unregister_service(info)
                zc.close()
    finally:
        if backend_process is not None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
