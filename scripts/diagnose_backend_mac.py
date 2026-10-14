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

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def http_get_json(url: str, timeout: float = 5.0, headers: Optional[dict] = None):
    request = Request(url, headers=headers or {})  # noqa: S310 - localhost/LAN diagnostic only
    with urlopen(request, timeout=timeout) as resp:  # noqa: S310
        import json
        return resp.status, json.loads(resp.read().decode())


def launchd_status() -> tuple[bool, str]:
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


PLIST_TEMPLATE = REPO_DIR / "deploy" / "launchd" / "com.sensordyme.backend.plist.template"
INSTALLED_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.sensordyme.backend.plist"


def launchd_runs_the_backend() -> tuple[bool, str]:
    """Is the installed agent configured to start the BACKEND?

    `launchd_status` above answers "is the job running", which is not the same
    question and was reported as if it were. The installed plist had been
    hand-edited to run `main.py` -- the camera/detector entrypoint, not the API
    server -- while the process still running was a `uvicorn` started before that
    edit. So launchd said "running", the port was open, health answered, every
    check passed, and the next login would have started the detector and left the
    backend simply absent. Comparing what is installed against the template is
    the only check that sees it.
    """
    if not INSTALLED_PLIST.exists():
        return False, f"{INSTALLED_PLIST.name} not installed -- run scripts/install_backend_mac.sh"
    try:
        import plistlib

        installed = plistlib.loads(INSTALLED_PLIST.read_bytes())
    except Exception as exc:  # noqa: BLE001 - a corrupt plist is a finding, not a crash
        return False, f"could not read the installed agent ({exc!r})"

    args = installed.get("ProgramArguments") or []
    joined = " ".join(args)
    if not any(arg.endswith("uvicorn") for arg in args):
        return False, (
            f"the installed agent does NOT start the backend -- it runs `{joined}`. "
            f"Reinstall it: ./scripts/install_backend_mac.sh"
        )
    if "cloud.app:app" not in args:
        return False, (
            f"the installed agent runs uvicorn but not the backend app -- `{joined}`. "
            f"Reinstall it: ./scripts/install_backend_mac.sh"
        )

    # A port mismatch between what uvicorn binds and what the agent announces
    # sends the app somewhere definite and wrong, which is worse than silence.
    announced = (installed.get("EnvironmentVariables") or {}).get("SENSORDYME_PORT")
    port_at = args.index("--port") + 1 if "--port" in args else None
    if port_at is not None and port_at < len(args):
        bound = args[port_at]
        if announced is not None and announced != bound:
            return False, (
                f"the agent binds port {bound} but announces {announced} over mDNS -- "
                f"the app would be sent to a port nothing is on"
            )
    return True, f"runs `{joined}`"


def jwt_secret_persisted() -> tuple[bool, str]:
    """Will the backend sign with the same key after a restart?

    Asked of the key the AGENT will use: cloud/app.py takes
    SENSORDYME_JWT_SECRET, then SENSORDYME_JWT_SECRET_FILE, then a key it
    generates beside the database. Checking only cloud/.jwt_secret reported a
    missing key on a backend configured through either variable.
    """
    env: dict = {}
    if INSTALLED_PLIST.exists():
        try:
            import plistlib

            env = plistlib.loads(INSTALLED_PLIST.read_bytes()).get("EnvironmentVariables") or {}
        except Exception:  # noqa: BLE001 - launchd_runs_the_backend reports a bad plist
            env = {}
    if env.get("SENSORDYME_JWT_SECRET"):
        return True, "set in the agent's environment (SENSORDYME_JWT_SECRET)"
    configured = env.get("SENSORDYME_JWT_SECRET_FILE")
    db_path = env.get("SENSORDYME_DB")
    secret_path = (
        Path(configured) if configured
        else (Path(db_path).parent if db_path else REPO_DIR / "cloud") / ".jwt_secret"
    )
    if secret_path.exists():
        return True, str(secret_path)
    return False, (
        f"{secret_path} missing -- every restart will invalidate signed-in sessions "
        f"until the backend has run once and created it"
    )


def mdns_advertising(timeout: float = 3.0) -> tuple[bool, str]:
    """Is the backend announcing itself, and is what it announces still TRUE?

    The second half is the part that matters and was missing. This used to report
    success on the strength of a record existing, which is exactly the state that
    breaks the app: the address and hostname were captured once at startup, and a
    LaunchAgent keeps the process alive across every Wi-Fi change, tether and DHCP
    move, so a long-running backend sat there advertising an address that had not
    existed for days. From the phone that is indistinguishable from the server
    being switched off -- worse, in fact, because a confident wrong address stops
    the app looking any further.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return False, "zeroconf not installed -- can't check (pip install zeroconf)"

    import socket

    from cloud.mdns import SERVICE_TYPE, lan_ip

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
        if not found:
            return False, "not seen -- SENSORDYME_MDNS may be off, or the network blocks multicast"
        info = zc.get_service_info(SERVICE_TYPE, found[0], timeout=int(timeout * 1000))
    finally:
        zc.close()

    if info is None:
        return False, f"advertised as {found[0]!r} but it would not resolve"

    advertised = [socket.inet_ntoa(a) for a in (info.addresses or []) if len(a) == 4]
    actual = lan_ip()
    if actual is None:
        return False, f"advertised at {advertised} but this machine has no LAN address"
    if actual not in advertised:
        return False, (
            f"STALE: advertised at {advertised or 'no address'} but this machine is at "
            f"{actual} -- the app is being sent to an address that no longer exists. "
            f"Restart the backend so it re-announces: "
            f"launchctl kickstart -k gui/$(id -u)/com.sensordyme.backend"
        )
    return True, f"visible and current ({found[0]} at {actual}:{info.port})"


def main() -> int:
    # Read the port the same way the advertiser does, rather than assuming the
    # default. A diagnostic whose whole job is "is the backend up" reporting a
    # perfectly healthy backend as down — because it is on a port this script
    # hardcoded — is worse than having no diagnostic.
    from cloud.mdns import advertised_port

    port = advertised_port()
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

    # Fatal, unlike the line above: an agent that starts the wrong program means
    # the backend is gone at the next login, and nothing else here would say so.
    all_ok &= check("Agent starts the backend", *launchd_runs_the_backend())

    all_ok &= check("JWT secret persisted", *jwt_secret_persisted())

    # Fatal too. A stale record is not a degraded version of discovery working —
    # it actively stops the app looking anywhere else.
    all_ok &= check("mDNS advertising is current", *mdns_advertising())

    if len(sys.argv) > 1:
        host = sys.argv[1]
        print(f"\nDevice ({host}):")
        # The device's dashboard requires X-Device-Token. Read it from the
        # environment so this tool can report the component states; without it we
        # can still tell "up" from "absent", which is the question that matters.
        token = os.environ.get("SENSORDYME_DEVICE_TOKEN", "")
        headers = {"X-Device-Token": token} if token else {}
        try:
            status, payload = http_get_json(
                f"http://{host}:8899/api/health", timeout=5, headers=headers
            )
            comps = payload.get("components", {})
            for key in ("wifi", "cloud_uplink", "provisioning", "camera", "inference"):
                c = comps.get(key, {})
                ok = c.get("state") == "ok"
                check(key, ok, c.get("detail", "unknown"))
        except HTTPError as exc:
            # A 401 is the device ANSWERING. Reporting that as unreachable is the
            # same false negative this tool exists to avoid: it sends someone to
            # debug a network while the device is up and simply wants its token.
            if exc.code in (401, 403):
                check(
                    "device reachable", True,
                    f"{host}:8899 answered {exc.code} — it is up; set "
                    f"SENSORDYME_DEVICE_TOKEN to read its component health",
                )
            else:
                check("device reachable", False, f"{host}:8899 -- HTTP {exc.code}")
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
