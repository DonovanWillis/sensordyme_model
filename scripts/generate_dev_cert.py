#!/usr/bin/env python3
"""Generate a self-signed TLS cert for running the cloud backend over HTTPS locally.

For anything reachable from outside your own machine (a real deploy, the
investor demo, anything the iOS app needs to trust without extra work), use a
real CA-signed certificate instead — a platform like Fly.io/Railway terminates
TLS for you, or run Caddy/Let's Encrypt in front of uvicorn. See cloud/README.md's
TLS section for why: iOS validates certificates against the system trust store,
and a self-signed cert isn't in it, so the app will not simply trust one at this
path — see that section for what pinning would take, which is not implemented.

This script exists for exercising HTTPS itself in local development (e.g. the
Pi's core/uplink.py talking to a Mac-hosted backend with `ca_cert_path` set to
the .pem this produces) before a real deployment target exists.

Usage:
    .venv/bin/python scripts/generate_dev_cert.py [output-dir] [hostname]

Requires the `openssl` CLI (present by default on macOS and Raspberry Pi OS).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DIR = Path(".secrets")
DEFAULT_HOST = "localhost"


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    hostname = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_HOST

    if shutil.which("openssl") is None:
        print("generate_dev_cert: no `openssl` on PATH — install it and re-run.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    key_path = out_dir / "dev-cert.key"
    cert_path = out_dir / "dev-cert.pem"

    if key_path.exists() or cert_path.exists():
        print(f"generate_dev_cert: {key_path} or {cert_path} already exists — remove it first "
              "if you want a fresh one (a new cert invalidates the old one everywhere it was "
              "trusted via ca_cert_path).", file=sys.stderr)
        return 1

    # 825 days: under the ~825-day CA/Browser Forum ceiling that made this a
    # round, unsurprising number to pick, though nothing here enforces it —
    # this cert is never checked against that policy, only trusted directly
    # by whatever ca_cert_path points at it.
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "825", "-subj", f"/CN={hostname}",
            "-addext", f"subjectAltName=DNS:{hostname},DNS:*.local",
        ],
        check=True,
    )
    key_path.chmod(0o600)

    print(f"wrote {cert_path} and {key_path}")
    print()
    print("Run the backend with it:")
    print(f"  .venv/bin/uvicorn cloud.app:app --host 0.0.0.0 --port 8787 \\")
    print(f"      --ssl-keyfile {key_path} --ssl-certfile {cert_path}")
    print()
    print("Then, on the device, point config.yaml's cloud.base_url at https:// and set:")
    print(f"  cloud.ca_cert_path: {cert_path.resolve()}")
    print()
    print("The iOS app will NOT trust this cert without extra work — see cloud/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
