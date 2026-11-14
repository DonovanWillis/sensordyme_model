#!/usr/bin/env python3
"""Disposable install test of a transfer package's bundle (M8.5, checkpoint J).

Runs the REAL `scripts/release/release_guard.py` against the REAL release
tarball + manifest in a throwaway directory tree, and writes an installation
verification report (JSON + Markdown) with one PASS/FAIL per step and the exact
command an operator would run on a device.

  J1  adopt a fake legacy tree into the staged layout
  J2  stage the real bundle: every file verified against the manifest before it
      is written (the guard's real extract_verified over the real tarball)
  J3  activate: config carried + migrated, atomic symlink switch, units,
      restart, confirm
  J4  a tampered copy of the same bundle is REFUSED and nothing changes
  J5  a release whose service crash-loops is rolled back automatically by
      switching back (no pip, no network)
  J6  `switch --to previous`: the operator rollback

HONEST LABELS -- what is real here and what is not:

  REAL   release_guard.py (unmodified, imported from this tree), the bundle
         tarball and manifest, the per-file sha256 verification, the staged
         layout, the atomic switch, the rollback logic, the state records.
  FAKE   systemd (a probe whose health answer depends on which release the
         service path points at), sudo, pip and venv creation (recorded, not
         run: an aarch64 lock cannot install on x86/macOS), config migration.
         The model-artifact check runs for REAL when mediapipe+yaml import in
         this interpreter (the Mac venv), otherwise it is recorded as FAKE.

  Hardware-only remainder: a real `pip install --require-hashes` of the aarch64
  lock, real systemd following the symlink, real sudo -- a Raspberry Pi.

    python scripts/release/transfer_install_check.py \
        --bundle .../sensordyme-<id>.tar.gz --manifest .../sensordyme-<id>.manifest.json \
        [--out DIR]     # default: $SDJOB_OUT, else ./transfer-install-report

Exit 0 only when every step PASSes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import release_guard as rg  # noqa: E402 - the REAL guard, path set above

SYNTHETIC_COMMIT = "b" * 40  # the deliberately broken second release (J5)
SYNTHETIC_FILES = {
    "main.py": b"raise SystemExit('this release is deliberately broken')\n",
    "requirements-device.lock": b"pyyaml==6.0.2 \\\n    --hash=sha256:" + b"0" * 64 + b"\n",
    "scripts/release/install_units.sh": b"#!/bin/sh\n",
    "scripts/migrate_config.py": b"# migrate: no-op for the disposable test\n",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _can_run_artifact_check() -> bool:
    try:
        import mediapipe  # noqa: F401
        import yaml  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means the real check cannot run here
        return False
    return True


class FakeDevice:
    """systemd + every subprocess the guard runs, in-process.

    Health depends on WHICH release the service path points at (`broken` names
    the releases whose unit never stays up). pip/venv/sudo are recorded, not
    run. The model-artifact check runs for real when this interpreter can."""

    def __init__(self, link: Path, *, real_artifact_check: bool) -> None:
        self.link = link
        self.broken: set[str] = set()
        self.commands: list[list[str]] = []
        self.pip_argv: list[list[str]] = []
        self.restarts = 0
        self.stopped = False
        self.now = 0.0
        self.real_artifact_check = real_artifact_check
        self.real_for = ""  # release id whose artifact check may run for real (the real bundle only)
        self.artifact_check_ran_real = False

    def live(self) -> str:
        return Path(os.path.realpath(self.link)).name

    def run(self, argv: list[str]) -> int:
        self.commands.append(list(argv))
        if argv[1:3] == ["-m", "venv"]:
            fake = Path(argv[-1]) / "bin"
            fake.mkdir(parents=True, exist_ok=True)
            (fake / "python").write_text("#!/bin/sh\n# FAKE venv python (disposable install test)\n")
            return 0
        if argv[1:4] == ["-m", "pip", "install"]:
            self.pip_argv.append(list(argv))
            return 0  # FAKE: the aarch64 lock cannot install here
        joined = " ".join(argv)
        if "verify-artifacts" in joined:
            if self.real_artifact_check and self.real_for and f"/{self.real_for}/" in joined:
                import subprocess
                self.artifact_check_ran_real = True
                return subprocess.run([sys.executable, *argv[1:]], check=False, timeout=300).returncode
            return 0  # FAKE (synthetic release, or mediapipe not importable here)
        if argv[:3] == ["sudo", "systemctl", "restart"]:
            self.restarts += 1
        return 0  # sudo -n true, install_units.sh, migrate_config.py, config check: FAKE ok

    def probe(self) -> dict:
        if self.stopped:
            return {"ActiveState": "inactive", "SubState": "dead", "NRestarts": "0"}
        if self.live() in self.broken:
            return {"ActiveState": "activating", "SubState": "auto-restart",
                    "NRestarts": str(self.restarts + int(self.now)), "ExecMainStatus": "1"}
        return {"ActiveState": "active", "SubState": "running", "NRestarts": "0", "ExecMainStatus": "0"}

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def make_legacy_tree(root: Path) -> None:
    """A pre-M8.5 in-place install with device state the test can track."""
    (root / "data").mkdir(parents=True)
    (root / "data" / "device_state.json").write_text('{"preset": "driver_safety"}')
    (root / "models").mkdir()
    (root / "models" / "damoyolo_tinynasL20_T.hef").write_bytes(b"FAKE-HEF-DEVICE-STATE")
    (root / ".secrets").write_text("fake-device-secret\n")
    (root / "main.py").write_text("print('the pre-M8.5 release')\n")
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    config = root / "config.yaml"
    config.write_text("module: driver_attention\ncloud:\n  device_key: dk_fake_disposable_test\n")
    os.chmod(config, 0o600)


def synthetic_bundle(out: Path) -> tuple[Path, Path, str]:
    """A minimal, deliberately broken second release (manifest + matching tarball)."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"commit": SYNTHETIC_COMMIT, "deploy_set": {"files": {
        p: {"sha256": hashlib.sha256(d).hexdigest()} for p, d in SYNTHETIC_FILES.items()}}}
    manifest_path = out / "broken.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    tar_path = out / "broken.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in SYNTHETIC_FILES.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return tar_path, manifest_path, _sha256(manifest_path)


def tampered_copy(bundle: Path, out: Path) -> Path:
    """The real tarball with one byte flipped in one member's contents."""
    with tarfile.open(bundle) as tar:
        members = []
        for member in tar:
            handle = tar.extractfile(member)
            members.append((member, handle.read() if handle else b""))
    bad = out / "tampered.tar.gz"
    with tarfile.open(bad, "w:gz", compresslevel=1) as tar:
        for index, (member, data) in enumerate(members):
            blob = (data[:-1] + bytes([data[-1] ^ 1]) if data else b"x") if index == len(members) // 2 else data
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    return bad


class Report:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def step(self, step_id: str, name: str, ok: bool, label: str, operator_command: str,
             detail: str, log: list[str]) -> bool:
        self.steps.append({"step": step_id, "name": name, "result": "PASS" if ok else "FAIL",
                           "components": label, "operator_command": operator_command,
                           "detail": detail, "guard_output": log[-6:]})
        print(f"[{'PASS' if ok else 'FAIL'}] {step_id} {name}: {detail}")
        return ok


def run_checks(bundle: Path, manifest: Path, work: Path, out_dir: Path) -> int:
    manifest_sha = _sha256(manifest)
    rid = str(json.loads(manifest.read_text())["commit"])[:12]
    real_artifacts = _can_run_artifact_check()
    link = work / "sensordyme"
    make_legacy_tree(link)
    device = FakeDevice(link, real_artifact_check=real_artifacts)
    device.real_for = rid
    report = Report()

    def guard(log: list[str]) -> rg.Guard:
        return rg.Guard(link, run=device.run, probe=device.probe, clock=device.clock,
                        sleep=device.sleep, user="sdpi", out=log.append)

    ok = True
    lay = rg.Layout(link)

    # J1 adopt ---------------------------------------------------------------
    log: list[str] = []
    device.stopped = True
    code = guard(log).adopt()
    device.stopped = False
    legacy = lay.current()
    j1 = code == 0 and link.is_symlink() and legacy is not None and legacy.name.startswith("legacy-") \
        and (lay.state / "data" / "device_state.json").exists() and (lay.state / ".secrets").exists()
    ok &= report.step("J1", "adopt the legacy tree", j1,
                      "REAL guard; FAKE systemd (service reported stopped)",
                      "sudo systemctl stop sensordyme.service && "
                      "python3 - adopt --root ~/sensordyme < release_guard.py",
                      f"exit {code}; {link.name} -> {legacy.name if legacy else None}; "
                      "state moved to the state dir", log)

    # J2 stage the real bundle ------------------------------------------------
    log = []
    code = guard(log).stage(bundle, manifest, manifest_sha)
    release = lay.release(rid)
    staged = json.loads((release / rg.STAGED_MARK).read_text()) if (release / rg.STAGED_MARK).exists() else {}
    expected = json.loads(manifest.read_text())["deploy_set"]["files"]
    drift = rg.Guard._verify_tree(release, expected) if release.is_dir() else ["release missing"]
    pip = device.pip_argv[-1] if device.pip_argv else []
    pip_ok = "--require-hashes" in pip and "--only-binary=:all:" in pip and \
        pip[-1].endswith(f"{rid}/requirements-device.lock")
    artifact_label = ("REAL model-artifact check (core.model_registry verify-artifacts)"
                      if device.artifact_check_ran_real else "FAKE model-artifact check (mediapipe not importable here)")
    j2 = code == 0 and not drift and staged.get("files") == len(expected) and pip_ok \
        and device.live() != rid
    ok &= report.step("J2", f"stage the real bundle ({len(expected)} files verified)", j2,
                      f"REAL guard + REAL tarball + REAL per-file sha256 verification; FAKE venv/pip; {artifact_label}",
                      f"python3 - stage --root ~/sensordyme --bundle …/{bundle.name} "
                      f"--manifest …/{manifest.name} --manifest-sha256 {manifest_sha} "
                      "[--find-links ~/sensordyme.wheels] < release_guard.py",
                      f"exit {code}; {len(expected)} files byte-identical to the manifest; "
                      f"pip argv hashed+wheels-only: {pip_ok}; nothing switched yet", log)

    # J3 activate + confirm ---------------------------------------------------
    log = []
    code = guard(log).activate(rid, timeout_s=60, settle_s=5, attempt="pkg-j3")
    confirmed = rg._read_json(lay.state / rg.STATE_DIR / "confirmed.json") or {}
    carried = (release / "config.yaml").is_file() and \
        (release / "config.yaml").read_text().endswith("dk_fake_disposable_test\n") and \
        oct((release / "config.yaml").stat().st_mode & 0o777) == "0o600"
    j3 = code == 0 and device.live() == rid and confirmed.get("release") == rid \
        and str(confirmed.get("previous", "")).startswith("legacy-") and carried \
        and (link / "models" / "damoyolo_tinynasL20_T.hef").read_bytes() == b"FAKE-HEF-DEVICE-STATE"
    ok &= report.step("J3", "activate: switch atomically, confirm", j3,
                      "REAL guard (atomic rename(2) switch, config carry, records); "
                      "FAKE sudo/systemd/units/migration",
                      f"python3 ~/sensordyme.releases/{rid}/scripts/release/release_guard.py activate "
                      f"--root ~/sensordyme --id {rid} --user \"$(id -un)\" --attempt pkg-j3 --detach",
                      f"exit {code}; live={device.live()}; config carried 0600; device state intact", log)

    # J4 tampered bundle refused ---------------------------------------------
    log = []
    bad = tampered_copy(bundle, work)
    pip_before = len(device.pip_argv)
    # Staging the SAME commit over the running release is refused earlier (it is
    # protected), so the refusal under test is the per-file verification: run it
    # on a fresh layout, exactly as a first install of a tampered upload would.
    link2 = work / "sensordyme2"
    make_legacy_tree(link2)
    device2 = FakeDevice(link2, real_artifact_check=False)
    device2.stopped = True
    g2 = rg.Guard(link2, run=device2.run, probe=device2.probe, clock=device2.clock,
                  sleep=device2.sleep, user="sdpi", out=log.append)
    assert g2.adopt() == 0
    device2.stopped = False
    code = g2.stage(bad, manifest, manifest_sha)
    lay2 = rg.Layout(link2)
    j4 = code == 1 and any("REFUSED" in line and "sha256 does not match" in line for line in log) \
        and not lay2.release(rid).exists() and not list(lay2.releases.glob(".partial-*")) \
        and not device2.pip_argv and len(device.pip_argv) == pip_before
    ok &= report.step("J4", "a tampered bundle is refused, nothing installed", j4,
                      "REAL guard + REAL tampered tarball (one byte flipped) + REAL verification",
                      f"python3 - stage --root ~/sensordyme --bundle <tampered>.tar.gz "
                      f"--manifest …/{manifest.name} --manifest-sha256 {manifest_sha} < release_guard.py",
                      f"exit {code}; refused before anything was written; no partial dir; no pip", log)

    # J5 crash-looping release rolls back automatically -----------------------
    log = []
    broken_tar, broken_manifest, broken_sha = synthetic_bundle(work / "broken")
    broken_id = SYNTHETIC_COMMIT[:12]
    code = guard(log).stage(broken_tar, broken_manifest, broken_sha)
    device.broken = {broken_id}
    code2 = guard(log).activate(broken_id, timeout_s=60, settle_s=5, attempt="pkg-j5")
    rolled = rg._read_json(lay.state / rg.STATE_DIR / "rolled_back.json") or {}
    j5 = code == 0 and code2 == 1 and device.live() == rid and rolled.get("release") == broken_id \
        and rolled.get("previous") == rid and rolled.get("previous_healthy") is True \
        and not (lay.state / rg.STATE_DIR / "pending.json").exists()
    ok &= report.step("J5", "a crash-looping deploy is rolled back automatically", j5,
                      "REAL guard rollback-by-switch-back (no pip, no network); FAKE systemd crash-loop; "
                      "synthetic broken bundle (REAL verification applied to it)",
                      f"… activate --root ~/sensordyme --id {broken_id} …  "
                      "# then: status --wait exits 1; rolled_back.json holds the reason",
                      f"stage exit {code}, activate exit {code2}; live back on {device.live()}; "
                      f"reason: {rolled.get('reason', '')[:60]}", log)

    # J6 operator rollback: switch --to previous -------------------------------
    log = []
    device.broken = set()
    code = guard(log).switch_to("previous", timeout_s=60, settle_s=5)
    j6 = code == 0 and device.live().startswith("legacy-") \
        and (link / "data" / "device_state.json").read_text() == '{"preset": "driver_safety"}'
    ok &= report.step("J6", "operator rollback: switch --to previous", j6,
                      "REAL guard; FAKE systemd/sudo",
                      "python3 ~/sensordyme/scripts/release/release_guard.py switch "
                      "--root ~/sensordyme --to previous --user \"$(id -un)\"",
                      f"exit {code}; live={device.live()}; shared state intact", log)

    # ------------------------------------------------------------------ report
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = "PASS" if ok else "FAIL"
    payload = {
        "format": "sensordyme-transfer-install-check/1",
        "verdict": verdict,
        "bundle": {"path": str(bundle), "sha256": _sha256(bundle)},
        "manifest": {"path": str(manifest), "sha256": manifest_sha, "release": rid},
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0],
                        "machine": platform.machine(), "sdjob_id": os.environ.get("SDJOB_ID"),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "honesty": {
            "real": "release_guard.py, the bundle tarball + manifest, per-file sha256 verification, "
                    "atomic switch, rollback logic, state records"
                    + (", model-artifact check" if device.artifact_check_ran_real else ""),
            "fake": "systemd, sudo, pip, venv creation, config migration"
                    + ("" if device.artifact_check_ran_real else ", model-artifact check"),
            "hardware_only": "real pip install of the aarch64 lock, real systemd following the "
                             "symlink, real sudo, the Hailo model on a Pi (docs/DEPLOYMENT.md §8)",
        },
        "steps": report.steps,
    }
    (out_dir / "install_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"# Disposable install verification — release `{rid}`: **{verdict}**", "",
             f"- bundle `{bundle.name}` sha256 `{payload['bundle']['sha256']}`",
             f"- manifest sha256 `{manifest_sha}`",
             f"- environment: {payload['environment']['platform']} / Python {payload['environment']['python']}"
             + (f" / job {payload['environment']['sdjob_id']}" if payload['environment']['sdjob_id'] else ""),
             "", f"REAL: {payload['honesty']['real']}", "", f"FAKE: {payload['honesty']['fake']}", "",
             f"Hardware-only remainder: {payload['honesty']['hardware_only']}", "",
             "| Step | Result | What ran | Operator command |", "|---|---|---|---|"]
    lines += [f"| {s['step']} {s['name']} | **{s['result']}** | {s['components']} | `{s['operator_command']}` |"
              for s in report.steps]
    lines.append("")
    (out_dir / "install_verification.md").write_text("\n".join(lines))
    print(f"INSTALL VERIFICATION: {verdict} — report in {out_dir}/install_verification.{{json,md}}")
    return 0 if ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Disposable install test of a release bundle (checkpoint J)")
    parser.add_argument("--bundle", required=True, help="the release tarball (sensordyme-<id>.tar.gz)")
    parser.add_argument("--manifest", required=True, help="the release manifest (sensordyme-<id>.manifest.json)")
    parser.add_argument("--out", default=os.environ.get("SDJOB_OUT") or "transfer-install-report")
    parser.add_argument("--work", default=None, help="throwaway tree (default: a temp dir, removed afterwards)")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args(argv)
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="sensordyme-install-check-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        return run_checks(Path(args.bundle), Path(args.manifest), work, Path(args.out))
    finally:
        if not args.keep_work and not args.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
