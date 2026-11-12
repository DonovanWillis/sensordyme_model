"""Build and run the Swift onboarding harness against the virtual peripheral.

    python -m sim.ble_harness --scenario wrong-code [--evidence-dir DIR]
    python -m sim.ble_harness --all

The harness (ios/Tests/OnboardingHarness) is the app's own onboarding code --
SetupFlow, Onboarding, BLEProvisioner, BLEPoPFlow, BLEProofOfPossession,
ProvisioningStatus and the DEBUG simulated transport -- compiled with swiftc
for this Mac. Each scenario gets a fresh sim/ble_peripheral.py process (the
device's real provisioning and proof-of-possession code) and talks to it
message by message over localhost. Secrets go to both sides in 0600 files,
never on a command line.

Used by tests/test_m8_onboarding_crosslang.py and scripts/ble_onboarding_e2e.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
SOURCES = [
    "BLETransport", "BLESimulatedTransport", "BLEProvisioning", "BLEPoPFlow", "BLEProofOfPossession",
    "ProvisioningStatus", "SetupOnboarding", "SetupFlow",
]
HARNESS = ["Support.swift", "main.swift"]

#: Peripheral settings each scenario needs (beyond the defaults).
SCENARIOS: dict[str, list[str]] = {
    "success": [],
    "wrong-code": [],
    "lockout": ["--max-failures", "3", "--lockout-base", "2"],
    "wrong-label": [],
    "impostor-id": [],
    "tamper-proven": [],
    "tamper-creds": [],
    "timeout": ["--challenge-ttl", "1"],
    "proof-expired": ["--proven-ttl", "1"],
    "wrong-password": [],
}
LAB_SSID = "SensorDyme Lab"
LAB_PSK = "correct-horse-battery"


def sources() -> list[Path]:
    return [REPO / "ios" / "Sources" / f"{name}.swift" for name in SOURCES] + \
        [REPO / "ios" / "Tests" / "OnboardingHarness" / name for name in HARNESS]


def swiftc() -> Optional[str]:
    return shutil.which("swiftc") if sys.platform == "darwin" else None


def build(out_dir: Path) -> Path:
    """Compile the harness (cached by source hash); returns the binary."""
    compiler = swiftc()
    if compiler is None:
        raise RuntimeError("swiftc (macOS) is needed to build the onboarding harness")
    digest = hashlib.sha256()
    for path in sources():
        digest.update(path.read_bytes())
    binary = Path(out_dir) / f"onboarding-harness-{digest.hexdigest()[:12]}"
    if binary.exists():
        return binary
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # noqa: S603 - fixed compiler argv over repository files
        [compiler, "-swift-version", "5", "-D", "DEBUG", "-O", "-o", str(binary), *map(str, sources())],
        capture_output=True, text=True, timeout=600, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"swiftc failed:\n{result.stderr[-4000:]}")
    return binary


def _secret_file(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


@dataclass
class Peripheral:
    process: subprocess.Popen
    port: int
    state_dir: Path
    transcript: Path
    output: list[str] = field(default_factory=list)

    @property
    def setup_code(self) -> str:
        from services import ble_pop

        return ble_pop.load_setup_code(str(self.state_dir / "ble_setup_code"))

    def records(self) -> list[dict]:
        if not self.transcript.exists():
            return []
        return [json.loads(line) for line in self.transcript.read_text().splitlines() if line.strip()]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.process.stdout is not None:
            self.output.extend(self.process.stdout.read().splitlines())


def start_peripheral(state_dir: Path, transcript: Path, extra: list[str]) -> Peripheral:
    """`python -m sim.ble_peripheral` on a free loopback port; waits for READY."""
    process = subprocess.Popen(  # noqa: S603 - our own module, fixed argv
        [sys.executable, "-m", "sim.ble_peripheral", "--port", "0", "--state-dir", str(state_dir),
         "--transcript", str(transcript), "--join-delay", "0.5", "--outcome-linger", "3", *extra],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line.rstrip("\n"))
        if line.startswith("SIMBLE READY"):
            port = int(line.split("port=")[1].split()[0])
            return Peripheral(process, port, state_dir, transcript, output)
    process.wait(timeout=5)
    raise RuntimeError("the virtual peripheral did not start:\n" + "\n".join(output[-40:]))


@dataclass
class Run:
    scenario: str
    returncode: int
    evidence: list[dict]
    transcript: list[dict]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def checks(self) -> dict[str, bool]:
        return {e["name"]: e["ok"] for e in self.evidence if e.get("kind") == "check"}


def run_scenario(binary: Path, scenario: str, work: Path, *, peripheral_args: Optional[list[str]] = None,
                 backend: Optional[str] = None, account: Optional[tuple[str, str]] = None,
                 claim_code: Optional[str] = None, state_dir: Optional[Path] = None,
                 timeout_s: float = 240) -> Run:
    """One scenario: a fresh peripheral process, the harness against it, both records.

    `state_dir` keeps the device's identity and setup code across runs (a
    reboot of the same unit); by default every scenario gets a new device.
    """
    work.mkdir(parents=True, exist_ok=True)
    secrets_dir = Path(tempfile.mkdtemp(prefix="sd-harness-secrets-"))
    peripheral = start_peripheral(state_dir or (work / "device"), work / "peripheral_transcript.jsonl",
                                  SCENARIOS.get(scenario, []) + list(peripheral_args or []))
    try:
        argv = [str(binary), "--endpoint", f"127.0.0.1:{peripheral.port}", "--scenario", scenario,
                "--code-file", str(_secret_file(secrets_dir, "code", peripheral.setup_code)),
                "--ssid", LAB_SSID, "--psk-file", str(_secret_file(secrets_dir, "psk", LAB_PSK)),
                "--evidence", str(work / "app_evidence.jsonl")]
        if backend:
            argv += ["--backend", backend]
        if account:
            argv += ["--account-file", str(_secret_file(secrets_dir, "account", f"{account[0]}\n{account[1]}"))]
        if claim_code:
            argv += ["--claim-file", str(_secret_file(secrets_dir, "claim", claim_code))]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)  # noqa: S603
        (work / "harness_stdout.txt").write_text(result.stdout + result.stderr)
    finally:
        peripheral.stop()
        shutil.rmtree(secrets_dir, ignore_errors=True)
        (work / "peripheral_stdout.txt").write_text("\n".join(peripheral.output))
    evidence_path = work / "app_evidence.jsonl"
    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines()] if evidence_path.exists() else []
    return Run(scenario, result.returncode, evidence, peripheral.records())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim.ble_harness")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--evidence-dir", default=None)
    parser.add_argument("--build-dir", default=str(Path(tempfile.gettempdir()) / "sensordyme-onboarding-harness"))
    args = parser.parse_args(argv)
    names = sorted(SCENARIOS) if args.all else (args.scenario or ["success"])
    binary = build(Path(args.build_dir))
    root = Path(args.evidence_dir or tempfile.mkdtemp(prefix="sd-onboarding-"))
    failed = 0
    for name in names:
        run = run_scenario(binary, name, root / name)
        # The virtual device's state holds its setup code: not evidence.
        shutil.rmtree(root / name / "device", ignore_errors=True)
        checks = run.checks()
        print(f"{'PASS' if run.ok else 'FAIL'} {name}: {sum(checks.values())}/{len(checks)} checks, "
              f"{len(run.transcript)} device-side records ({root / name})")
        failed += 0 if run.ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
