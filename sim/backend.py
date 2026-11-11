"""The real backend (cloud/app.py), locally, on a throwaway data directory.

    python -m sim.backend [--port 8787] [--data-dir DIR] [--devices 1]
                          [--email demo@sensordyme.local --password sensordyme-demo]

starts uvicorn on cloud.app:app with its database, JWT secret and backend id
in DIR (a fresh temp dir by default), creates the demo account, and
"factory-provisions" virtual devices exactly as `python cloud/manage.py
create-device` does for a real one: a devices row with a device key (flashed
into the device's config) and a claim code (printed on the box, typed into the
app). Those are written to DIR/factory.json for `python -m sim.virtual_device
--factory DIR/factory.json`. Nothing is claimed: pairing is the app's job.

The AI provider defaults to "unavailable" (the deterministic keyword fallback,
no model calls) unless SENSORDYME_AI_PROVIDER is already set; mDNS
advertisement is off unless --mdns.

InProcessBackend is the same backend behind a FastAPI TestClient, for the
deterministic end-to-end tests, with its wall clock optionally on a SimClock.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DEMO_EMAIL = "demo@sensordyme.local"
DEMO_PASSWORD = "sensordyme-demo"  # noqa: S105 - a throwaway local demo account, printed on start


def backend_env(data_dir: Path, *, mdns: bool = False) -> dict[str, str]:
    """The environment cloud/app.py reads at import, pointed at `data_dir`."""
    data_dir = Path(data_dir)
    return {
        "SENSORDYME_DB": str(data_dir / "sensordyme.db"),
        "SENSORDYME_JWT_SECRET_FILE": str(data_dir / ".jwt_secret"),
        "SENSORDYME_BACKEND_ID_FILE": str(data_dir / ".backend_id"),
        "SENSORDYME_MDNS": "1" if mdns else "0",
        "SENSORDYME_AI_PROVIDER": os.environ.get("SENSORDYME_AI_PROVIDER", "unavailable"),
    }


def provision(app_module, name: str) -> dict:
    """The factory step (cloud/manage.py create-device): one device row."""
    claim_code = "-".join(secrets.token_hex(2) for _ in range(3)).upper()
    device_key = "dk_" + secrets.token_hex(24)
    with app_module.db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            (name, claim_code, device_key, time.time()),
        )
    return {"name": name, "device_key": device_key, "claim_code": claim_code}


def _fresh_app_module():
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    return importlib.import_module("cloud.app")


class InProcessBackend:
    """cloud/app.py behind a TestClient, on its own database. For tests.

    Sets the backend's environment variables for the life of the object and
    restores them on close(). With a SimClock, the backend's wall time (what
    last_seen and the 30 s staleness window are measured on) follows it.
    """

    def __init__(self, data_dir: Path, clock=None) -> None:
        from fastapi.testclient import TestClient

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        env = backend_env(self.data_dir)
        self._saved_env = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        self.app = _fresh_app_module()
        if clock is not None:
            self.app.time = clock.wall_clock_module()
        self.client = TestClient(self.app.app)
        self.base_url = str(self.client.base_url)

    def close(self) -> None:
        self.client.close()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # ------------------------------------------------------------------ factory + account
    def provision(self, name: str = "Virtual SensorDyme (SIMULATED)") -> dict:
        return provision(self.app, name)

    def signup(self, email: str = DEMO_EMAIL, password: str = DEMO_PASSWORD) -> dict:
        response = self.client.post("/v1/signup", json={"email": email, "password": password})
        response.raise_for_status()
        return {"Authorization": f"Bearer {response.json()['token']}"}


# ---------------------------------------------------------------------- live server
def serve(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.backend", description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (0.0.0.0 to reach it from a phone on the LAN)")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data-dir", default=None, help="state directory (default: a fresh temp dir)")
    parser.add_argument("--devices", type=int, default=1, help="virtual devices to factory-provision")
    parser.add_argument("--email", default=DEMO_EMAIL)
    parser.add_argument("--password", default=DEMO_PASSWORD)
    parser.add_argument("--mdns", action="store_true", help="advertise _sensordyme-api._tcp (cloud/mdns.py)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir or tempfile.mkdtemp(prefix="sensordyme-sim-backend-")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(backend_env(data_dir, mdns=args.mdns))
    sys.path.insert(0, str(REPO))
    app_module = _fresh_app_module()

    factory_path = data_dir / "factory.json"
    try:
        factory = json.loads(factory_path.read_text())
    except (OSError, ValueError):
        factory = {"devices": []}
    while len(factory["devices"]) < args.devices:
        n = len(factory["devices"]) + 1
        factory["devices"].append(provision(app_module, f"Virtual SensorDyme {n} (SIMULATED)"))
    local_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host  # noqa: S104 - a comparison
    factory.update(backend_url=f"http://{local_host}:{args.port}", email=args.email, password=args.password)
    factory_path.write_text(json.dumps(factory, indent=2))

    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as client:
        signup = client.post("/v1/signup", json={"email": args.email, "password": args.password})
        if signup.status_code not in (200, 409):
            print(f"sim.backend: demo account not created: HTTP {signup.status_code} {signup.text}")

    print("=" * 72)
    print("SensorDyme backend (REAL cloud/app.py) — local, throwaway data")
    print(f"  URL           {factory['backend_url']}   (iOS Simulator default: http://localhost:8787)")
    print(f"  data dir      {data_dir}")
    print(f"  demo account  {args.email} / {args.password}")
    for device in factory["devices"]:
        print(f"  device        {device['name']}: claim code {device['claim_code']}")
    print(f"  factory file  {factory_path}")
    print(f"  start a device: python -m sim.virtual_device --factory {factory_path}")
    print("=" * 72, flush=True)

    import uvicorn

    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
