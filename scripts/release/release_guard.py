#!/usr/bin/env python3
"""Release installs and rollback on the device: staged, verified, switched atomically (M8.5).

M8.5 (M84-6) replaces the in-place rsync with STAGED RELEASES. The service's
directory (the path the systemd unit names, e.g. ~/sensordyme) becomes a
symlink into a releases directory beside it, and device state lives in a
state directory that no release ever contains:

  ~/sensordyme                 -> sensordyme.releases/<id>   (atomic rename(2) to switch)
  ~/sensordyme.releases/<id>/  one release: the verified code, its own .venv, STAGED.json,
                               config.yaml (carried over, see activate), and symlinks
                               data -> ../../sensordyme.state/data (models, logs, ...)
  ~/sensordyme.state/          data/, models/ (the .hef), logs/, .secrets: shared, never
                               part of a release, never overwritten by one
  ~/sensordyme.incoming/       uploaded bundles (tarball + manifest)

  adopt                 ONE TIME, service stopped: turn a legacy checkout into that
                        layout (the old tree becomes release `legacy-<time>`; its
                        state moves to the state dir and is symlinked back).
  stage                 extract a release bundle (`make_release_manifest.py bundle`)
                        into releases/<id>/, verifying EVERY file against the
                        manifest (sha256, nothing missing, nothing extra, no
                        device state, no links, no traversal) BEFORE it is written;
                        the manifest itself must hash to --manifest-sha256 (passed
                        by the operator over SSH). Then a fresh venv, the hashed
                        wheels-only dependency install from the release's lock
                        (pip_install_argv), and the MediaPipe model-file check
                        (core/model_registry.py). Any failure: the release is
                        removed and nothing live has changed.
  activate              carry config.yaml over (never overwriting), migrate it in
                        the NEW release, record the deploy as pending, switch the
                        symlink atomically, install the units, restart, and then
                        confirm-or-rollback. --detach runs it in its own session
                        so a dropped SSH connection cannot interrupt it.
  confirm-or-rollback   waits for systemd to report the unit active (Type=notify:
                        only after READY=1) and still active, with no automatic
                        restart, for --settle seconds, and the device config to
                        load under the new release. Otherwise: rollback.
  rollback              staged: switch back to the previous release (its venv and
                        its pre-migration config.yaml are still there: no network,
                        no pip), re-install its units, restart. Legacy (M8
                        snapshot) deploys are restored from the snapshot.
  switch --to ID        operator switch to any staged release (`previous` = the one
                        before the last confirmed deploy): the manual rollback.
  status [--wait ID]    the guard's state; --wait blocks until that release is
                        confirmed (exit 0) or rolled back (exit 1).
  snapshot              M8 legacy only (an in-place tree): refused on a staged layout.

A second deploy is refused while an earlier one is still pending (neither
confirmed nor rolled back). Identity is untouched: config.yaml only ever moves
between this device's own copies; nothing here creates, rotates or reads out a
credential. Standard library only: it runs with the device's system python3,
streamed over SSH (`python3 - stage ... < release_guard.py`) before a release
exists. Tested against fake systemd/pip and temp trees
(tests/test_m8_lifecycle_release.py, tests/test_m85_deploy_*.py); it has not
yet run on a Pi.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Optional

#: Never part of a release: device state, the venv, models, secrets and logs. The
#: same set the release manifest excludes, so a release holds exactly the code.
#: config.yaml is carried separately.
EXCLUDED_DIRS = {".venv", "data", ".git", "__pycache__", "models", "logs", ".secrets",
                 ".pytest_cache", ".ruff_cache", "ios", ".claude"}
EXCLUDED_DIR_PATHS = {"benchmarks/results"}
EXCLUDED_FILES = {"config.yaml", "cloud/.jwt_secret", "cloud/.backend_id"}
EXCLUDED_PATTERNS = ("config.yaml", "*.hef", "*.pt", "*.db", "*.sqlite3", "*.csv", "*.log", "*.pem", "*.key",
                     ".env", ".env.*", "*.yaml.bak-*", ".claude*")

STATE_DIR = Path("data") / "release"
UNIT = "sensordyme.service"

# ---------------------------------------------------------------------- staged layout (M8.5)
RELEASES_SUFFIX = ".releases"
STATE_SUFFIX = ".state"
#: Device state shared by every release, kept in the state dir and symlinked into
#: each release (directories, and the secret files a device may hold at its root).
SHARED_STATE = ("data", "models", "logs", ".secrets", ".env")
#: Device files COPIED into a new release at activation, never overwritten: the
#: previous release keeps its own copy, which is what a rollback runs on again
#: (a migration may add keys the previous code refuses). Not a symlink because
#: every writer replaces config.yaml with os.replace(), which would replace a
#: symlink with a plain file inside one release.
CARRIED_FILES = ("config.yaml",)
STAGED_MARK = "STAGED.json"
KEEP_RELEASES = 3

# ---------------------------------------------------------------------- dependencies (M8.5, M84-4)
LOCK = "requirements-device.lock"
#: THE device dependency install: every package hash-checked against the lock,
#: wheels only (no package's build code runs at install time). setup_pi.sh runs
#: the same flags; the release gate (make_release_manifest.py install_paths)
#: refuses a device install path that does not.
PIP_FLAGS = ("--require-hashes", "--only-binary=:all:", "--disable-pip-version-check", "--no-input")


def pip_install_argv(python: Path, lock: Path, find_links: Optional[str] = None) -> list[str]:
    """`python -m pip install --require-hashes --only-binary=:all: ... -r <lock>`.

    `find_links`: a directory of wheels shipped with the release (offline install,
    still hash-checked against the lock); the index is then not used at all."""
    argv = [str(python), "-m", "pip", "install", *PIP_FLAGS, "-r", str(lock)]
    if find_links:
        argv += ["--no-index", "--find-links", str(find_links)]
    return argv


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(target: Path, start: Path) -> str:
    """A relative symlink target from directory `start` to `target`, computed on
    the PHYSICAL paths (the kernel resolves `..` physically, so a lexical relpath
    through a symlinked parent such as macOS /var would point elsewhere)."""
    return os.path.relpath(os.path.realpath(target), os.path.realpath(start))


class StageError(Exception):
    """A release bundle that must not be installed; nothing live was changed."""


def extract_verified(bundle: Path, dest: Path, expected: dict[str, dict]) -> int:
    """Extract `bundle` into the empty directory `dest`, writing a file only after
    its bytes match the manifest's sha256. Refuses (StageError) anything that is
    not a regular file, an absolute or `..` path, device state or a secret, a file
    the manifest does not list, a duplicate, and finally any listed file missing."""
    seen: set[str] = set()
    try:
        tar = tarfile.open(bundle, "r:*")  # noqa: SIM115 - closed below; streaming read
    except (OSError, tarfile.TarError) as exc:
        raise StageError(f"cannot read the bundle {bundle}: {exc}") from exc
    with tar:
        for member in tar:
            name = member.name[2:] if member.name.startswith("./") else member.name
            if member.isdir():
                continue
            path = PurePosixPath(name)
            if not member.isfile():
                raise StageError(f"refusing {name!r}: not a regular file (links and devices never ship)")
            if path.is_absolute() or ".." in path.parts or not name:
                raise StageError(f"refusing {name!r}: path escapes the release directory")
            if _excluded(Path(name)):
                raise StageError(f"refusing {name!r}: device state or a secret is never part of a release")
            want = expected.get(name)
            if want is None:
                raise StageError(f"refusing {name!r}: not in the manifest")
            if name in seen:
                raise StageError(f"refusing {name!r}: listed twice in the bundle")
            handle = tar.extractfile(member)
            data = handle.read() if handle is not None else b""
            if _sha256(data) != str(want.get("sha256", "")).lower():
                raise StageError(f"refusing {name!r}: sha256 does not match the manifest (tampered or corrupt)")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "xb") as out:  # x: never follow or overwrite anything
                out.write(data)
            os.chmod(target, 0o755 if member.mode & 0o111 else 0o644)
            seen.add(name)
    missing = sorted(set(expected) - seen)
    if missing:
        raise StageError(f"the bundle lacks {len(missing)} file(s) the manifest lists: {missing[:10]}")
    return len(seen)


class Layout:
    """Where a staged install keeps its releases and state, from the service path."""

    def __init__(self, link: Path) -> None:
        self.link = Path(os.path.abspath(link))
        self.releases = self.link.with_name(self.link.name + RELEASES_SUFFIX)
        self.state = self.link.with_name(self.link.name + STATE_SUFFIX)

    def adopted(self) -> bool:
        return self.link.is_symlink() and self.releases.is_dir()

    def current(self) -> Optional[Path]:
        return Path(os.path.realpath(self.link)) if self.link.is_symlink() else None

    def release(self, release_id: str) -> Path:
        if not release_id or "/" in release_id or release_id.startswith("."):
            raise StageError(f"not a release id: {release_id!r}")
        return self.releases / release_id

    def staged(self) -> list[Path]:
        if not self.releases.is_dir():
            return []
        return sorted((p for p in self.releases.iterdir() if (p / STAGED_MARK).exists()),
                      key=lambda p: (_read_json(p / STAGED_MARK) or {}).get("staged_at", 0.0))

    def switch(self, target: Path) -> None:
        """Point the service path at `target` with one rename(2): atomic, never absent."""
        tmp = self.link.with_name(f".{self.link.name}.switch-{os.getpid()}")
        tmp.unlink(missing_ok=True)
        os.symlink(_relative(target, self.link.parent), tmp)
        os.replace(tmp, self.link)

    def link_state(self, release: Path) -> list[str]:
        linked = []
        for name in SHARED_STATE:
            if (self.state / name).exists() and not (release / name).exists():
                os.symlink(_relative(self.state / name, release), release / name)
                linked.append(name)
        return linked


class Guard:
    def __init__(self, root: Path, *, run: Optional[Callable[[list[str]], int]] = None,
                 probe: Optional[Callable[[], dict]] = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, unit: str = UNIT,
                 user: Optional[str] = None, out: Callable[[str], None] = print) -> None:
        self.layout = Layout(root)
        self.root = Path(root).resolve()
        self.state = self.root / STATE_DIR
        if self.layout.adopted():
            # Staged layout: the guard's records live in the shared state, whichever
            # release is current (and stay put across a switch).
            self.state = self.layout.state / STATE_DIR
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

    def _units_argv(self, release: Path) -> list[str]:
        """Install `release`'s units, rendered for the SERVICE PATH (the symlink),
        never for the release directory itself: the unit must follow a switch."""
        argv = ["sudo", "bash", str(release / "scripts" / "release" / "install_units.sh")]
        if self.user:
            argv += ["--user", self.user]
        return [*argv, "--app-dir", str(self.layout.link)]

    # ------------------------------------------------------------------ snapshot (M8 legacy)
    def snapshot(self, force: bool = False) -> int:
        if self.layout.adopted():
            self._out("release_guard: this device uses staged releases; `snapshot` is the legacy in-place "
                      "mechanism. Deploy with `make deploy` (stage + activate); see docs/DEPLOYMENT.md.")
            return 2
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

    # ------------------------------------------------------------------ adopt (M8.5, one time)
    def adopt(self) -> int:
        """Legacy checkout -> staged layout. The old tree becomes release
        `legacy-<time>` (its .venv keeps working through the service path), its
        state moves to the state dir and is symlinked back. Service must be stopped."""
        lay = self.layout
        if lay.adopted():
            self._out(f"release_guard: already a staged layout ({lay.link} -> {lay.current()})")
            return 0
        if lay.link.is_symlink() or not lay.link.is_dir():
            self._out(f"release_guard: {lay.link} is not a checkout directory; nothing to adopt")
            return 2
        state = self._probe().get("ActiveState", "unknown")
        if state in ("active", "activating", "reloading"):
            self._out(f"release_guard: {self.unit} is {state}; stop it first (sudo systemctl stop {self.unit})")
            return 2
        if self.pending.exists():
            self._out("release_guard: an M8 deploy is still pending; run `rollback` or `confirm-or-rollback` first")
            return 2
        clash = [n for n in SHARED_STATE if (lay.state / n).exists() and (lay.link / n).exists()]
        if clash:
            self._out(f"release_guard: {lay.state} already holds {clash}; refusing to merge device state")
            return 2
        release = lay.releases / time.strftime("legacy-%Y%m%dT%H%M%S")
        lay.releases.mkdir(exist_ok=True)
        lay.state.mkdir(exist_ok=True)
        os.rename(lay.link, release)
        moved = []
        for name in SHARED_STATE:
            source = release / name
            if source.exists() and not source.is_symlink():
                os.rename(source, lay.state / name)
                moved.append(name)
        (lay.state / "data").mkdir(exist_ok=True)
        linked = lay.link_state(release)
        _write_json(release / STAGED_MARK, {"release": release.name, "adopted": True, "staged_at": time.time(),
                                            "note": "the pre-M8.5 in-place tree, adopted unchanged"})
        os.symlink(_relative(release, lay.link.parent), lay.link)
        self.state = lay.state / STATE_DIR
        self.pending = self.state / "pending.json"
        self._out(f"release_guard: ADOPTED — {lay.link} -> {release.name}; moved {moved} to {lay.state}; "
                  f"linked {linked}. Start the service: sudo systemctl start {self.unit}")
        return 0

    # ------------------------------------------------------------------ stage (M8.5)
    def stage(self, bundle: Path, manifest_path: Path, manifest_sha256: str, *,
              python: str = "python3", find_links: Optional[str] = None) -> int:
        lay = self.layout
        if not lay.adopted():
            self._out(f"release_guard: {lay.link} is not a staged layout yet; run `adopt` once "
                      "(make deploy-adopt; docs/DEPLOYMENT.md)")
            return 2
        try:
            raw = Path(manifest_path).read_bytes()
        except OSError as exc:
            self._out(f"release_guard: cannot read the manifest: {exc}")
            return 2
        if _sha256(raw) != manifest_sha256.strip().lower():
            self._out("release_guard: REFUSED — the manifest is not the one the operator built "
                      f"(sha256 {_sha256(raw)[:16]}…, expected {manifest_sha256[:16]}…)")
            return 2
        try:
            manifest = json.loads(raw)
            commit = str(manifest["commit"])
            expected = dict(manifest["deploy_set"]["files"])
            release = lay.release(commit[:12])
        except (ValueError, KeyError, TypeError, StageError) as exc:
            self._out(f"release_guard: REFUSED — not a release manifest ({exc})")
            return 2
        record = _read_json(release / STAGED_MARK)
        if record is not None and record.get("commit") == commit:
            # The same commit again (a re-run deploy): its files on disk are
            # re-verified against THIS manifest; the release is never rebuilt.
            drift = self._verify_tree(release, expected)
            if not drift:
                self._out(f"release_guard: {release.name} is already staged; its {len(expected)} files "
                          f"re-verified against manifest {manifest_sha256[:16]}…")
                return 0
            self._out(f"release_guard: staged {release.name} no longer matches its manifest: {drift[:5]}")
        protected = self._protected_releases()
        if release.name in protected:
            self._out(f"release_guard: REFUSED — {release.name} is the running or the rollback release and "
                      "is never rebuilt in place")
            return 2
        partial = lay.releases / f".partial-{release.name}"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        try:
            count = extract_verified(Path(bundle), partial, expected)
        except (StageError, OSError) as exc:
            shutil.rmtree(partial, ignore_errors=True)
            self._out(f"release_guard: REFUSED — {exc}. Nothing was installed; the running release is unchanged.")
            return 1
        # Only a verified tree replaces an earlier (never running, never rollback)
        # copy, and it is renamed BEFORE the venv is built: a venv records its path.
        shutil.rmtree(release, ignore_errors=True)
        os.replace(partial, release)
        self._out(f"release_guard: {count} files verified against manifest {manifest_sha256[:16]}… "
                  f"(commit {commit[:12]}) in {release}")
        venv_python = release / ".venv" / "bin" / "python"
        steps = [
            ("venv", [python, "-m", "venv", "--system-site-packages", str(release / ".venv")]),
            ("hashed dependency install", pip_install_argv(venv_python, release / LOCK, find_links)),
            ("model artifact check", self._artifact_check_argv(release, venv_python)),
        ]
        for label, argv in steps:
            if self._run(argv) != 0:
                shutil.rmtree(release, ignore_errors=True)
                self._out(f"release_guard: REFUSED — {label} failed ({' '.join(argv[:6])} …). The staged "
                          "release was removed; the running release is unchanged.")
                return 1
        linked = lay.link_state(release)
        _write_json(release / STAGED_MARK, {
            "release": release.name, "commit": commit, "manifest_sha256": manifest_sha256.lower(),
            "files": count, "staged_at": time.time(), "state_links": linked,
            "dependency_install": " ".join(pip_install_argv(Path(".venv/bin/python"), Path(LOCK), find_links)),
        })
        self._out(f"release_guard: STAGED {release.name} (not running yet). Activate: release_guard.py "
                  f"activate --root {lay.link} --id {release.name}")
        return 0

    @staticmethod
    def _verify_tree(release: Path, expected: dict[str, dict]) -> list[str]:
        """Listed files of an already staged release that are missing or differ."""
        drift = []
        for name, want in sorted(expected.items()):
            path = release / name
            try:
                ok = path.is_file() and not path.is_symlink() and \
                    _sha256(path.read_bytes()) == str(want.get("sha256", "")).lower()
            except OSError:
                ok = False
            if not ok:
                drift.append(name)
        return drift

    def _protected_releases(self) -> set[str]:
        """Never removed or rebuilt: the running release and the one a rollback needs."""
        names = {p.name for p in (self.layout.current(),) if p is not None}
        for record in ("pending.json", "confirmed.json"):
            previous = (_read_json(self.state / record) or {}).get("previous")
            if previous:
                names.add(str(previous))
        return names

    @staticmethod
    def _artifact_check_argv(release: Path, python: Path) -> list[str]:
        """The release's own MediaPipe model-file check, run in its own venv (M84-5)."""
        code = (f"import sys; sys.path.insert(0, {str(release)!r}); from core import model_registry as r; "
                f"sys.exit(r.main(['verify-artifacts', '--json-out', {str(release / 'ARTIFACTS.json')!r}]))")
        return [str(python), "-I", "-c", code]

    # ------------------------------------------------------------------ activate (M8.5)
    def activate(self, release_id: str, timeout_s: float = 180.0, settle_s: float = 30.0,
                 attempt: Optional[str] = None) -> int:
        """Switch to a staged release and confirm it or roll back. Every call, refused
        or not, ends in activation.json (what `status --wait` reads)."""
        lines: list[str] = []
        out = self._out

        def capture(line: str) -> None:
            lines.append(line)
            out(line)

        self._out = capture
        try:
            code = self._activate(release_id, timeout_s, settle_s, attempt)
        finally:
            self._out = out
        _write_json(self.state / "activation.json", {
            "release": release_id, "attempt": attempt, "exit": code, "finished_at": time.time(),
            "result": {0: "confirmed", 1: "failed"}.get(code, "refused"), "detail": lines[-1] if lines else ""})
        return code

    def _activate(self, release_id: str, timeout_s: float, settle_s: float, attempt: Optional[str]) -> int:
        lay = self.layout
        if not lay.adopted():
            self._out(f"release_guard: {lay.link} is not a staged layout; run `adopt` first")
            return 2
        try:
            release = lay.release(release_id)
        except StageError as exc:
            self._out(f"release_guard: {exc}")
            return 2
        if _read_json(release / STAGED_MARK) is None:
            self._out(f"release_guard: REFUSED — {release_id} is not a staged, verified release")
            return 2
        if self.pending.exists():
            self._out(f"release_guard: an earlier deploy is still pending ({self.pending}); refusing")
            return 2
        previous = lay.current()
        if previous is None or previous == release.resolve():
            self._out(f"release_guard: {release_id} is already the running release")
            return 0
        if self._run(["sudo", "-n", "true"]) != 0:
            # Detached, there is no terminal to ask for a password: without
            # non-interactive sudo neither the switch's units/restart nor a
            # rollback's could run. Refused before anything changes.
            self._out("release_guard: REFUSED — activation needs non-interactive sudo (systemctl, "
                      "install_units.sh) for this user; nothing switched")
            return 2
        carried = []
        for name in CARRIED_FILES:
            source, target = previous / name, release / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
                carried.append(name)
        venv_python = str(release / ".venv" / "bin" / "python")
        config = release / "config.yaml"
        if config.exists():
            os.chmod(config, 0o600)  # it holds the device key
            migrate = [venv_python, str(release / "scripts" / "migrate_config.py"), str(config)]
            if self._run(migrate) != 0:
                self._out("release_guard: REFUSED — config migration failed in the new release; nothing switched")
                return 1
        _write_json(self.pending, {"mode": "staged", "release": release.name, "previous": previous.name,
                                   "snapshot_at": time.time(), "carried": carried, "attempt": attempt})
        lay.switch(release)
        self._out(f"release_guard: switched {lay.link} -> {release.name} (previous {previous.name})")
        for argv in (self._units_argv(release), ["sudo", "systemctl", "restart", self.unit]):
            if self._run(argv) != 0:
                why = f"step failed: {' '.join(argv[:4])}"
                self._out(f"release_guard: release FAILED — {why}; rolling back to {previous.name}")
                self.rollback(reason=why, timeout_s=timeout_s, settle_s=settle_s)
                return 1
        return self.confirm_or_rollback(timeout_s, settle_s)

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
            if record.get("mode") == "staged":
                self.prune()
            return 0
        self._out(f"release_guard: release FAILED — {why}; rolling back to the previous release")
        self.rollback(reason=why, timeout_s=timeout_s, settle_s=settle_s)
        return 1

    def _config_check(self) -> list[str]:
        """The device's config.yaml, judged by the RUNNING release's own loader rules
        (through the service path, so a staged switch is followed)."""
        live = self.layout.link if self.layout.adopted() else self.root
        code = (f"import os, sys; os.chdir({str(live)!r}); sys.path.insert(0, os.getcwd()); "
                "from core import config_lkg; from core.config_schema import validate; "
                "c, p = config_lkg.read('config.yaml'); "
                "sys.exit(1 if p or validate(c)[0] or config_lkg.torn_write('config.yaml') else 0)")
        return [str(live / ".venv" / "bin" / "python"), "-c", code]

    # ------------------------------------------------------------------ rollback
    def rollback(self, reason: str = "requested", timeout_s: float = 180.0, settle_s: float = 30.0) -> int:
        record = _read_json(self.pending)
        if record is not None and record.get("mode") == "staged":
            return self._rollback_staged(record, reason, timeout_s, settle_s)
        manifest = _read_json(self.previous / "MANIFEST.json")
        if record is None or manifest is None:
            self._out("release_guard: nothing to roll back (no pending snapshot)")
            return 0
        snapshot_at = float(record.get("snapshot_at") or 0.0)
        kept = {Path(p) for p in manifest.get("files", [])}
        # 1. Files the failed release added: removed, but only ones that appeared
        #    after the snapshot -- never anything that was already there.
        #
        #    M8.4: judged by the inode's CHANGE time, not st_mtime alone. `make
        #    deploy` copied with `rsync -a`, which stamps every file with its
        #    mtime on the developer's machine -- a module written yesterday lands
        #    with yesterday's mtime, so the old test kept it, and modules/
        #    auto-imports every sibling: the rolled-back release imported the
        #    failed release's leftover and could crash-loop on it. ctime is set
        #    by the kernel when the file is created or written here and cannot
        #    be carried over by a copy.
        removed = 0
        for rel in release_files(self.root):
            if rel in kept:
                continue
            path = self.root / rel
            try:
                stat = path.stat()
                if max(stat.st_mtime, stat.st_ctime) >= snapshot_at - 1.0:
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
        # 4. Same steps as a deploy, for the previous release. M8.5 (M84-4): the
        #    dependencies come from the previous release's HASHED lock, or not at
        #    all -- never an unhashed install from the index.
        steps = []
        if (self.root / LOCK).exists():
            steps.append(pip_install_argv(self.root / ".venv" / "bin" / "python", self.root / LOCK))
        else:
            self._out(f"release_guard: the previous release has no {LOCK}; its packages are left as installed")
        steps += [
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

    def _rollback_staged(self, record: dict, reason: str, timeout_s: float, settle_s: float) -> int:
        """Switch back: the previous release still has its own code, venv and
        pre-migration config.yaml, so no file is copied and nothing is downloaded."""
        lay = self.layout
        previous = lay.releases / str(record.get("previous") or "")
        failed_steps: list[str] = []
        if not record.get("previous") or not previous.is_dir():
            healthy, why = False, f"the previous release {record.get('previous')!r} is missing"
        else:
            lay.switch(previous)
            self._out(f"release_guard: switched {lay.link} back -> {previous.name}")
            for argv in (self._units_argv(previous), ["sudo", "systemctl", "restart", self.unit]):
                if self._run(argv) != 0:
                    failed_steps.append(" ".join(argv[:4]))
            healthy, why = self.wait_healthy(timeout_s, settle_s)
        _write_json(self.state / "rolled_back.json", {**record, "rolled_back_at": time.time(), "reason": reason,
                                                       "previous_healthy": healthy, "detail": why,
                                                       "failed_steps": failed_steps})
        self.pending.unlink(missing_ok=True)
        if healthy:
            self._out(f"release_guard: ROLLED BACK — {previous.name} is running again ({why}); "
                      f"{record.get('release')} is kept for inspection")
        else:
            self._out(f"release_guard: ROLLED BACK, but {previous.name} is not healthy either ({why}); "
                      f"failed steps: {failed_steps or 'none'}. Needs a person (docs/DEPLOYMENT.md).")
        return 0

    # ------------------------------------------------------------------ switch / prune / status
    def switch_to(self, target: str, timeout_s: float = 180.0, settle_s: float = 30.0) -> int:
        """Operator switch to a staged release (`previous`: the one before the last
        confirmed deploy). No pip, no network; reports health, does not undo itself."""
        lay = self.layout
        if not lay.adopted():
            self._out("release_guard: not a staged layout")
            return 2
        if self.pending.exists():
            self._out("release_guard: a deploy is pending; let it confirm or roll back first")
            return 2
        if target == "previous":
            target = str((_read_json(self.state / "confirmed.json") or {}).get("previous") or "")
        try:
            release = lay.release(target)
        except StageError as exc:
            self._out(f"release_guard: {exc}")
            return 2
        if _read_json(release / STAGED_MARK) is None:
            self._out(f"release_guard: {target!r} is not a staged release")
            return 2
        current = lay.current()
        for name in CARRIED_FILES:  # a release never staged here may lack the device config
            if current is not None and (current / name).is_file() and not (release / name).exists():
                shutil.copy2(current / name, release / name)
        lay.switch(release)
        failed = [" ".join(a[:4]) for a in (self._units_argv(release), ["sudo", "systemctl", "restart", self.unit])
                  if self._run(a) != 0]
        healthy, why = self.wait_healthy(timeout_s, settle_s)
        _write_json(self.state / "switched.json", {"release": release.name,
                                                   "from": current.name if current else None,
                                                   "switched_at": time.time(), "healthy": healthy, "detail": why,
                                                   "failed_steps": failed})
        self._out(f"release_guard: SWITCHED to {release.name} — {'healthy' if healthy else 'NOT healthy'}: {why}")
        return 0 if healthy and not failed else 1

    def prune(self, keep: int = KEEP_RELEASES) -> list[str]:
        """Remove old releases: never the current one, the previous one, an adopted
        legacy tree, or the newest `keep`."""
        lay = self.layout
        protected = self._protected_releases()
        staged = lay.staged()
        protected |= {p.name for p in staged[-keep:]}
        removed = []
        for path in staged:
            if path.name in protected or path.name.startswith("legacy-"):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.name)
        for partial in lay.releases.glob(".partial-*") if lay.releases.is_dir() else ():
            shutil.rmtree(partial, ignore_errors=True)
        if removed:
            self._out(f"release_guard: pruned {removed}")
        return removed

    def status(self, wait_for: Optional[str] = None, timeout_s: float = 900.0,
               attempt: Optional[str] = None) -> int:
        if wait_for:
            # The outcome of THIS activation: matched by the operator's attempt token
            # (make deploy passes one), else by release id and a finish after now.
            started = time.time()
            deadline = self._clock() + timeout_s
            while True:
                done = _read_json(self.state / "activation.json") or {}
                mine = done.get("release") == wait_for and (
                    done.get("attempt") == attempt if attempt else float(done.get("finished_at") or 0) >= started)
                if mine:
                    self._out(f"release_guard: {wait_for} {str(done.get('result')).upper()}: {done.get('detail')}")
                    return {0: 0, 1: 1}.get(int(done.get("exit", 2)), 2)
                if self._clock() >= deadline:
                    break
                self._sleep(2.0)
            self._out(f"release_guard: {wait_for} has no outcome after {timeout_s:.0f} s; the device keeps "
                      "going on its own (confirm or roll back). Check: release_guard.py status")
            return 3
        lay = self.layout
        if lay.adopted():
            current = lay.current()
            self._out(f"layout: staged; {lay.link} -> {current.name if current is not None else None}")
            for path in lay.staged():
                self._out(f"  release {path.name}: {json.dumps(_read_json(path / STAGED_MARK), sort_keys=True)}")
        else:
            self._out(f"layout: legacy in-place tree at {lay.link} (adopt to use staged releases)")
        for name in ("pending.json", "activation.json", "confirmed.json", "rolled_back.json", "switched.json"):
            payload = _read_json(self.state / name)
            if payload is not None:
                self._out(f"{name}: {json.dumps(payload, sort_keys=True)}")
        return 0


def _detach(argv: list[str], log: Path) -> int:
    """Re-run this script with `argv` in its own session (survives an SSH hangup)."""
    script = Path(__file__)
    if not script.is_file():
        print("release_guard: --detach needs the guard as a file (the staged release's copy), not stdin")
        return 2
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as handle:
        child = subprocess.Popen([sys.executable, str(script.resolve()), *argv], stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"release_guard: running detached (pid {child.pid}); log {log}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("command", choices=["snapshot", "confirm-or-rollback", "rollback", "status",
                                            "adopt", "stage", "activate", "switch", "prune"])
    parser.add_argument("--root", default=".", help="the service path (default: current directory)")
    parser.add_argument("--unit", default=UNIT)
    parser.add_argument("--user", default=None, help="service user, passed to install_units.sh")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds to reach stable active")
    parser.add_argument("--settle", type=float, default=30.0, help="seconds active with no restart")
    parser.add_argument("--force", action="store_true", help="snapshot even over a pending deploy")
    parser.add_argument("--bundle", help="stage: the release tarball")
    parser.add_argument("--manifest", help="stage: the release manifest (JSON)")
    parser.add_argument("--manifest-sha256", help="stage: the manifest's sha256 as the operator built it")
    parser.add_argument("--find-links", default=None, help="stage: install wheels from this directory only")
    parser.add_argument("--python", default="python3", help="stage: the interpreter the venv is made from")
    parser.add_argument("--id", default=None, help="activate: the staged release id")
    parser.add_argument("--to", default=None, help="switch: a release id, or `previous`")
    parser.add_argument("--wait", default=None, help="status: block until this release is confirmed/rolled back")
    parser.add_argument("--detach", action="store_true", help="activate: run in its own session, log to state")
    parser.add_argument("--attempt", default=None, help="activate/status --wait: the operator's attempt token")
    args = parser.parse_args(argv)
    guard = Guard(Path(args.root), unit=args.unit, user=args.user)
    if args.command == "snapshot":
        return guard.snapshot(force=args.force)
    if args.command == "confirm-or-rollback":
        return guard.confirm_or_rollback(args.timeout, args.settle)
    if args.command == "rollback":
        return guard.rollback(timeout_s=args.timeout, settle_s=args.settle)
    if args.command == "adopt":
        return guard.adopt()
    if args.command == "stage":
        if not (args.bundle and args.manifest and args.manifest_sha256):
            parser.error("stage needs --bundle, --manifest and --manifest-sha256")
        return guard.stage(Path(args.bundle), Path(args.manifest), args.manifest_sha256,
                           python=args.python, find_links=args.find_links)
    if args.command == "activate":
        if not args.id:
            parser.error("activate needs --id")
        if args.detach:
            rest = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--detach"]
            return _detach(rest, guard.state / f"activate-{args.id}.log")
        return guard.activate(args.id, args.timeout, args.settle, attempt=args.attempt)
    if args.command == "switch":
        if not args.to:
            parser.error("switch needs --to <release id | previous>")
        return guard.switch_to(args.to, args.timeout, args.settle)
    if args.command == "prune":
        guard.prune()
        return 0
    return guard.status(wait_for=args.wait, timeout_s=args.timeout if args.wait else 0.0, attempt=args.attempt)


if __name__ == "__main__":
    sys.exit(main())
