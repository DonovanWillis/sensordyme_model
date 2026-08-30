#!/usr/bin/env python3
"""Connection doctor for the Mac side of SensorDyme: is the cloud backend actually up?

Counterpart to ios/Sources/DiagnosticsView.swift (which checks the chain from the
phone's point of view) and the device's own /api/health (core/health.py, which
reports wifi/cloud_uplink/etc. from the device's point of view). Neither of those
can see the one failure mode that lives entirely on this machine: the backend
process itself not running. That failure looks identical from the phone and the
device to "network problem" -- Wi-Fi, DHCP, and mDNS discovery can all be
completely healthy while nobody ever started `uvicorn`.

Checks, in order: is anything listening on the backend port, does it answer as
the real SensorDyme backend, is the launchd agent (scripts/install_backend_mac.sh)
installed and running so it survives a reboot, is the JWT secret persisted (so
restarts don't invalidate every signed-in session), and -- best-effort -- is the
backend actually announcing itself over mDNS the way the app expects.

Usage:
    .venv/bin/python scripts/diagnose_backend_mac.py [device-hostname]

Passing a device hostname (e.g. sensordyme.local) also fetches that device's own
/api/health and prints its wifi/cloud_uplink component states, so one command can
answer "where in the chain is this actually broken" end to end.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))


def check(name: str, ok: bool, detail: str) -> bool:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}: {detail}")
    return ok


def port_listening(port: int) -> bool:
    try:
        out = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def http_get_json(url: str, timeout: float = 5.0):
    with urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost/LAN diagnostic only
        import json
        return resp.status, json.loads(resp.read().decode())


def launchd_status() -> tuple[bool, str]:
    import os
    domain = f"gui/{os.getuid()}/com.sensordyme.backend"
    try:
        out = subprocess.run(
            ["launchctl", "print", domain], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not query launchd ({exc})"
    if out.returncode != 0:
        return False, "not installed -- run scripts/install_backend_mac.sh so this survives a reboot"
    running = "state = running" in out.stdout
    return running, "running" if running else "installed but not running"


def mdns_advertising(timeout: float = 3.0) -> tuple[bool, str]:
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return False, "zeroconf not installed -- can't check (pip install zeroconf)"

    from cloud.mdns import SERVICE_TYPE

    found: list[str] = []

    class _Listener:
        def add_service(self, zc, type_, name):
            found.append(name)

        def update_service(self, zc, type_, name):
            pass

        def remove_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE_TYPE, _Listener())
        time.sleep(timeout)
    finally:
        zc.close()

    if found:
        return True, f"visible on the network ({found[0]})"
    return False, "not seen -- SENSORDYME_MDNS may be off, or the network blocks multicast"


def main() -> int:
    port = 8787
    all_ok = True

    print("SensorDyme backend diagnostic\n")

    listening = port_listening(port)
    all_ok &= check("Backend process listening", listening, f"port {port}"
                     if listening else f"nothing is listening on :{port} -- it is not running")

    if listening:
        try:
            status, payload = http_get_json(f"http://localhost:{port}/v1/health")
            is_real = status == 200 and payload.get("service") == "sensordyme-cloud"
            all_ok &= check("Backend answers /v1/health", is_real, str(payload))
        except (URLError, OSError, ValueError) as exc:
            all_ok &= check("Backend answers /v1/health", False, str(exc))
    else:
        print("  ✗ Backend answers /v1/health: skipped")
        all_ok = False

    running, detail = launchd_status()
    check("Persists across reboot/crash (launchd)", running, detail)  # advisory, not fatal

    secret_path = REPO_DIR / "cloud" / ".jwt_secret"
    all_ok &= check(
        "JWT secret persisted", secret_path.exists(),
        str(secret_path) if secret_path.exists()
        else f"{secret_path} missing -- every restart will invalidate signed-in sessions "
             f"until the backend has run once and created it",
    )

    check(*(("mDNS advertising",) + mdns_advertising()))  # advisory: Settings entry is a valid fallback

    if len(sys.argv) > 1:
        host = sys.argv[1]
        print(f"\nDevice ({host}):")
        try:
            status, payload = http_get_json(f"http://{host}:8899/api/health", timeout=5)
            comps = payload.get("components", {})
            for key in ("wifi", "cloud_uplink", "provisioning"):
                c = comps.get(key, {})
                ok = c.get("state") == "ok"
                check(key, ok, c.get("detail", "unknown"))
        except (URLError, OSError, ValueError) as exc:
            check("device reachable", False, f"{host}:8899 -- {exc}")

    print()
    if all_ok:
        print("Backend is healthy.")
    else:
        print("Backend has a problem -- see the ✗ lines above.")
        print("Fix and make it durable: ./scripts/install_backend_mac.sh")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
