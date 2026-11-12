#!/usr/bin/env python3
"""First-time onboarding, end to end, on software only (M8 Checkpoint D).

    python scripts/ble_onboarding_e2e.py [--out DIR] [--keep]

What runs, all of it real code except the radio and the board:

  * the backend: cloud/app.py via `python -m sim.backend` (a throwaway
    database, one factory-provisioned device: device key + claim code);
  * the device's Bluetooth side: `python -m sim.ble_peripheral` --
    services/provisioning.py, services/ble_pop.py and BLEProvisioningServer
    with an in-memory GATT table and a simulated Wi-Fi radio; when its Wi-Fi
    join succeeds it starts the virtual device runtime (sim.virtual_device),
    which reports to the backend with its device key like a real unit;
  * the phone: the app's SetupFlow / Onboarding / BLEProvisioner / BLEPoPFlow
    / BLEProofOfPossession, compiled for this Mac (sim/ble_harness.py), over
    the app's DEBUG simulated transport, with an HTTP account standing in
    for AppState.backend (same endpoints, same bodies).

Scenarios: `first-time` (discover, prove the setup code, send Wi-Fi, join,
refused + accepted claim, the device reports in, a monitoring mode is set),
then `reprovision` (the same unit, already claimed and online on another
network, moved to a new one: no claim is asked).

Evidence (default ~/sensordyme-audit/2026-11-10/m8-evidence/onboarding/):
per scenario the app's transitions and API calls (app_evidence.jsonl), the
device's side of every Bluetooth exchange (peripheral_transcript.jsonl) and
the backend's view afterwards (backend_after.json), plus summary.json.
Setup codes, passphrases, claim codes, tokens, nonces, proofs and ciphertext
never appear in any of them (checked before the script exits). NOT a
Bluetooth validation: no radio carried any of this.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sim import ble_harness  # noqa: E402

DEFAULT_OUT = Path.home() / "sensordyme-audit" / "2026-11-10" / "m8-evidence" / "onboarding"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait(url: str, timeout_s: float = 60) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - local URL
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)
    raise SystemExit(f"e2e: {url} did not come up")


def _api(base: str, method: str, path: str, token: str | None = None, body: dict | None = None) -> dict:
    request = urllib.request.Request(f"{base}{path}", method=method,  # noqa: S310 - local URL
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - local URL
        return json.loads(response.read())


def _backend_view(base: str, email: str, password: str) -> list[dict]:
    token = _api(base, "POST", "/v1/login", body={"email": email, "password": password})["token"]
    rows = _api(base, "GET", "/v1/devices", token)["devices"]
    keep = ("id", "name", "hardware_id", "online", "active_preset", "desired_preset")
    return [{k: row.get(k) for k in keep} | {"monitoring_state": (row.get("status") or {}).get("monitoring_state")}
            for row in rows]


def _secrets_absent(directory: Path, secrets: list[str]) -> list[str]:
    """Files under `directory` that contain any of `secrets` (should be none)."""
    leaks = []
    for path in directory.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            leaks += [f"{path.name}: {s[:2]}…" for s in secrets if s and s in text]
    return leaks


def run(out: Path, keep: bool = False) -> dict:
    if ble_harness.swiftc() is None:
        raise SystemExit("e2e: needs swiftc on macOS")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sd-onboarding-e2e-"))
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    backend = subprocess.Popen(
        [sys.executable, "-m", "sim.backend", "--port", str(port), "--data-dir", str(work / "backend")],
        cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    summary: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scenarios": {}}
    try:
        _wait(f"{base}/v1/health")
        factory_path = work / "backend" / "factory.json"
        factory = json.loads(factory_path.read_text())
        email, password = factory["email"], factory["password"]
        claim_code = factory["devices"][0]["claim_code"]
        binary = ble_harness.build(work / "build")
        device_state = work / "device"
        common = ["--factory", str(factory_path), "--start-device", "--backend", base]

        steps = [
            ("first-time", common),
            # The same unit after a reboot: already on "CoffeeShop Guest" and
            # reporting, moved to the lab network inside its reconfigure window.
            ("reprovision", [*common, "--online", "CoffeeShop Guest"]),
        ]
        for name, peripheral_args in steps:
            run_dir = out / name
            if run_dir.exists():
                shutil.rmtree(run_dir)
            result = ble_harness.run_scenario(
                binary, name, run_dir, peripheral_args=peripheral_args, backend=base,
                account=(email, password), claim_code=claim_code, state_dir=device_state, timeout_s=400,
            )
            view = _backend_view(base, email, password)
            (run_dir / "backend_after.json").write_text(json.dumps(view, indent=2))
            checks = result.checks()
            summary["scenarios"][name] = {
                "ok": result.ok, "checks_passed": sum(checks.values()), "checks": checks,
                "app_transitions": [f"{e['from']} --{e['event']}--> {e['to']}" for e in result.evidence
                                    if e.get("kind") == "transition"],
                "api_calls": [f"{e['method']} {e['path']} -> {e['status']}" for e in result.evidence
                              if e.get("kind") == "api"],
                "device_auth_messages": [f"{r['dir']} {r['value'].get('op')}"
                                         + (f" ({r['value'].get('error')})" if r["value"].get("error") else "")
                                         for r in result.transcript
                                         if r.get("char") == "auth" and isinstance(r.get("value"), dict)],
                "backend_after": view,
            }
            print(f"{'PASS' if result.ok else 'FAIL'} {name}: {sum(checks.values())}/{len(checks)} checks")
            if not result.ok:
                break

        setup_code = (device_state / "ble_setup_code").read_text().strip()
        secrets = [setup_code, setup_code.replace("-", ""), ble_harness.LAB_PSK, claim_code, password]
        leaks = _secrets_absent(out, secrets)
        summary["secrets_in_evidence"] = leaks
        summary["ok"] = all(s["ok"] for s in summary["scenarios"].values()) and len(summary["scenarios"]) == 2 \
            and not leaks
        summary["note"] = ("Software only: the app's real onboarding code and the device's real provisioning/"
                           "proof-of-possession code over a simulated link. No Bluetooth radio was involved.")
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=15)
        except subprocess.TimeoutExpired:
            backend.kill()
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--keep", action="store_true", help="keep the throwaway backend/device directories")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    summary = run(Path(args.out).expanduser(), keep=args.keep)
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios"}, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
