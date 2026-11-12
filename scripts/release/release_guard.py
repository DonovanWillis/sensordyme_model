#!/usr/bin/env python3
"""Release rollback for `make deploy` (M8): a release that does not come up is undone.

`make deploy` used to rsync the new code over the old, install, restart, and
stop there. A release that passed the deploy preflight but could not start (an
ImportError, a crash before READY, a crash loop after it) left the device on the
broken release; recovery meant a person, SSH and a working tree of the old
code. Now:

  snapshot              BEFORE the rsync. Copies the code the device is running
                        (everything the deploy rsync would overwrite: never data/,
                        .venv, models, the .hef, secrets or logs) to
                        data/release/previous/, and config.yaml beside it (0600:
                        it holds the device key). Run from stdin with the NEW
                        guard (`python - snapshot < release_guard.py`), so the
                        device needs no copy of this file beforehand.
  confirm-or-rollback   AFTER the restart. Waits for systemd to report the unit
                        active (Type=notify: only after READY=1) and still
                        active, with no automatic restart, for --settle seconds.
                        Then the release is confirmed. Otherwise the snapshot is
                        restored, the units re-installed, the service restarted
                        on the previous release, and the deploy exits 1.
  rollback              the restore on its own (also what make runs when a deploy
                        step before the restart fails). No-op without a pending
                        snapshot, so running it twice is safe.
  status                prints the guard's state.

A second snapshot is refused while an earlier deploy is still pending (neither
confirmed nor rolled back): snapshotting a half-deployed tree would make the
broken release the one to roll back to.

Identity is untouched: config.yaml is restored only to the device's own
earlier copy; nothing here creates, rotates or reads out a credential.
Tested against fake systemd and a temp tree (tests/test_m8_lifecycle_release.py);
it never runs on this Mac against a real unit.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

#: Never part of a release: device state, the venv, models, secrets and logs. The
#: same set `make deploy` excludes from its rsync, so a snapshot holds exactly
#: what a deploy can overwrite. config.yaml is snapshotted separately.
EXCLUDED_DIRS = {".venv", "data", ".git", "__pycache__", "models", "logs", ".secrets",
                 ".pytest_cache", ".ruff_cache", "ios", ".claude"}
EXCLUDED_DIR_PATHS = {"benchmarks/results"}
EXCLUDED_FILES = {"config.yaml", "cloud/.jwt_secret", "cloud/.backend_id"}
EXCLUDED_PATTERNS = ("config.yaml", "*.hef", "*.pt", "*.db", "*.sqlite3", "*.csv", "*.log", "*.pem", "*.key",
                     ".env", ".env.*", "*.yaml.bak-*", ".claude*")

STATE_DIR = Path("data") / "release"
UNIT = "sensordyme.service"


def _excluded_dir(rel: Path) -> bool:
    return (rel.name in EXCLUDED_DIRS or rel.name.startswith(".claude")
            or rel.as_posix() in EXCLUDED_DIR_PATHS)


def _excluded(rel: Path) -> bool:
    if any(_excluded_dir(Path(*rel.parts[: i + 1])) for i in range(len(rel.parts) - 1)):
        return True
    if rel.as_posix() in EXCLUDED_FILES:
        return True
    return any(fnmatch.fnmatch(rel.name, pattern) for pattern in EXCLUDED_PATTERNS)


def release_files(root: Path) -> list[Path]:
    """Every file in `root` a deploy may overwrite, relative, sorted."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(d for d in dirnames if not _excluded_dir(rel_dir / d))
        for name in filenames:
            rel = rel_dir / name
            if not _excluded(rel) and not os.path.islink(Path(dirpath) / name):
                found.append(rel)
    return sorted(found)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


class Guard:
    def __init__(self, root: Path, *, run: Optional[Callable[[list[str]], int]] = None,
                 probe: Optional[Callable[[], dict]] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, unit: str = UNIT,
                 user: Optional[str] = None, out: Callable[[str], None] = print) -> None:
        self.root = Path(root).resolve()
        self.state = self.root / STATE_DIR
        self.previous = self.state / "previous"
        self.pending = self.state / "pending.json"
        self.config_copy = self.state / "config.yaml.previous"
        self.unit = unit
        self.user = user
        self._run = run or self._subprocess
        self._probe = probe or self._systemctl_probe
        self._clock = clock
        self._sleep = sleep
        self._out = out

    # ------------------------------------------------------------------ system edges
    def _subprocess(self, argv: list[str]) -> int:
        return subprocess.run(argv, cwd=self.root, check=False).returncode

    def _systemctl_probe(self) -> dict:
        try:
            output = subprocess.run(
                ["systemctl", "show", self.unit, "-p", "ActiveState,SubState,NRestarts,ExecMainStatus"],
                capture_output=True, text=True, timeout=10, check=False).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ActiveState": "unknown", "error": str(exc)}
        return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, force: bool = False) -> int:
        if self.pending.exists() and not force:
            self._out(f"release_guard: an earlier deploy is still pending ({self.pending}); refusing to "
                      "snapshot a half-deployed tree. Run `rollback` (or `confirm-or-rollback`) first.")
            return 2
        files = release_files(self.root)
        staging = self.state / "previous.new"
        shutil.rmtree(staging, ignore_errors=True)
        for rel in files:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, target)
        _write_json(staging / "MANIFEST.json", {"files": [p.as_posix() for p in files]})
        shutil.rmtree(self.previous, ignore_errors=True)
        os.replace(staging, self.previous)
        config = self.root / "config.yaml"
        if config.exists():
            shutil.copy2(config, self.config_copy)
            os.chmod(self.config_copy, 0o600)
        else:
            self.config_copy.unlink(missing_ok=True)
        _write_json(self.pending, {"snapshot_at": time.time(), "files": len(files),
                                   "config": config.exists()})
        self._out(f"release_guard: snapshot of the running release ({len(files)} files) in {self.previous}")
        return 0

    # ------------------------------------------------------------------ health
    def wait_healthy(self, timeout_s: float, settle_s: float) -> tuple[bool, str]:
        """Active (after READY), and no automatic restart, for settle_s, within timeout_s."""
        deadline = self._clock() + timeout_s
        active_since: Optional[float] = None
        baseline: Optional[str] = None
        last = {}
        while True:
            last = self._probe()
            state = last.get("ActiveState", "unknown")
            restarts = last.get("NRestarts")
            if baseline is None:
                baseline = restarts
            if state == "failed":
                return False, f"{self.unit} failed ({last.get('SubState')}, exit {last.get('ExecMainStatus')})"
            if restarts != baseline:
                return False, f"{self.unit} restarted on its own (NRestarts {baseline} -> {restarts})"
            if state == "active":
                if active_since is None:
                    active_since = self._clock()
                if self._clock() - active_since >= settle_s:
                    return True, f"{self.unit} active for {settle_s:.0f} s with no restart"
            else:
                active_since = None
            if self._clock() >= deadline:
                return False, f"{self.unit} not stably active within {timeout_s:.0f} s (last: {state}/{last.get('SubState')})"
            self._sleep(1.0)

    def confirm_or_rollback(self, timeout_s: float = 180.0, settle_s: float = 30.0) -> int:
        if not self.pending.exists():
            self._out("release_guard: no pending deploy to confirm")
            return 0
        healthy, why = self.wait_healthy(timeout_s, settle_s)
        if healthy and self._run(self._config_check()) != 0:
            # The runtime would still come up -- on the last known good config, with
            # a fault (core/config_lkg.py) -- so "active" alone would confirm a
            # release whose migration broke config.yaml.
            healthy, why = False, "config.yaml does not load under the new release (running on the fallback)"
        if healthy:
            record = _read_json(self.pending) or {}
            _write_json(self.state / "confirmed.json", {**record, "confirmed_at": time.time(), "detail": why})
            self.pending.unlink(missing_ok=True)
            self._out(f"release_guard: release CONFIRMED — {why}")
            return 0
        self._out(f"release_guard: release FAILED — {why}; rolling back to the previous release")
        self.rollback(reason=why, timeout_s=timeout_s, settle_s=settle_s)
        return 1

    def _config_check(self) -> list[str]:
        """The device's config.yaml, judged by the new release's own loader rules."""
        code = ("import sys; from core import config_lkg; from core.config_schema import validate; "
                "c, p = config_lkg.read('config.yaml'); "
                "sys.exit(1 if p or validate(c)[0] or config_lkg.torn_write('config.yaml') else 0)")
        return [str(self.root / ".venv" / "bin" / "python"), "-c", code]

    # ------------------------------------------------------------------ rollback
    def rollback(self, reason: str = "requested", timeout_s: float = 180.0, settle_s: float = 30.0) -> int:
        record = _read_json(self.pending)
        manifest = _read_json(self.previous / "MANIFEST.json")
        if record is None or manifest is None:
            self._out("release_guard: nothing to roll back (no pending snapshot)")
            return 0
        snapshot_at = float(record.get("snapshot_at") or 0.0)
        kept = {Path(p) for p in manifest.get("files", [])}
        # 1. Files the failed release added: removed, but only ones that appeared
        #    after the snapshot -- never anything that was already there.
        removed = 0
        for rel in release_files(self.root):
            if rel in kept:
                continue
            path = self.root / rel
            try:
                if path.stat().st_mtime >= snapshot_at - 1.0:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        # 2. Every file of the previous release, byte for byte.
        for rel in kept:
            source = self.previous / rel
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        # 3. The device's own config.yaml as it was before the deploy (a migration
        #    may have added keys the previous code refuses).
        if record.get("config") and self.config_copy.exists():
            config = self.root / "config.yaml"
            tmp = config.with_suffix(".yaml.rollback.tmp")
            shutil.copy2(self.config_copy, tmp)
            os.replace(tmp, config)
        self._out(f"release_guard: restored {len(kept)} files, removed {removed} added by the failed release")
        # 4. Same steps as a deploy, for the previous release.
        steps = [
            [str(self.root / ".venv" / "bin" / "pip"), "install", "-q", "-r", "requirements-device.txt"],
            ["sudo", "bash", "scripts/release/install_units.sh"] + (["--user", self.user] if self.user else []),
            ["sudo", "systemctl", "restart", self.unit],
        ]
        failed_steps = []
        for argv in steps:
            if self._run(argv) != 0:
                failed_steps.append(" ".join(argv[:4]))
        healthy, why = self.wait_healthy(timeout_s, settle_s)
        _write_json(self.state / "rolled_back.json", {**record, "rolled_back_at": time.time(), "reason": reason,
                                                       "previous_healthy": healthy, "detail": why,
                                                       "failed_steps": failed_steps})
        self.pending.unlink(missing_ok=True)
        if healthy:
            self._out(f"release_guard: ROLLED BACK — the previous release is running ({why})")
        else:
            self._out(f"release_guard: ROLLED BACK, but the previous release is not healthy either ({why}); "
                      f"failed steps: {failed_steps or 'none'}. Needs a person (docs/MONDAY_RUNBOOK.md).")
        return 0

    def status(self) -> int:
        for name in ("pending.json", "confirmed.json", "rolled_back.json"):
            payload = _read_json(self.state / name)
            if payload is not None:
                self._out(f"{name}: {json.dumps(payload, sort_keys=True)}")
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("command", choices=["snapshot", "confirm-or-rollback", "rollback", "status"])
    parser.add_argument("--root", default=".", help="the device checkout (default: current directory)")
    parser.add_argument("--unit", default=UNIT)
    parser.add_argument("--user", default=None, help="service user, passed to install_units.sh")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to reach stable active")
    parser.add_argument("--settle", type=float, default=30.0, help="seconds active with no restart")
    parser.add_argument("--force", action="store_true", help="snapshot even over a pending deploy")
    args = parser.parse_args(argv)
    guard = Guard(Path(args.root), unit=args.unit, user=args.user)
    if args.command == "snapshot":
        return guard.snapshot(force=args.force)
    if args.command == "confirm-or-rollback":
        return guard.confirm_or_rollback(args.timeout, args.settle)
    if args.command == "rollback":
        return guard.rollback(timeout_s=args.timeout, settle_s=args.settle)
    return guard.status()


if __name__ == "__main__":
    sys.exit(main())
