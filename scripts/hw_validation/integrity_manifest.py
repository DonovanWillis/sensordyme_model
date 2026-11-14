#!/usr/bin/env python3
"""What exactly is installed on this device? A file-hash answer, because
/home/pi/sensordyme is NOT a git checkout: `make deploy` rsyncs a developer's
working tree (not a commit, no --delete), so the device can hold uncommitted
edits and files long since deleted from the repo.

  snapshot  (Pi, read-only) hash every deployable file under --root with the
            same excludes as `make deploy`; hash config.yaml and the .hef
            separately; check the .hef against the config pin; run this
            checkout's scripts/deploy_preflight.py against the device config
            (pure check, prints, edits nothing); record the service unit hash,
            systemd state and the service venv's `pip freeze`.
  compare   (worker/Mac) diff a snapshot against one or more git revisions
            (`git archive`, same excludes): identical / differ / differ only
            in line endings / only on device / only in the revision. The
            revision with the most identical files is the best match; anything
            else is drift to explain.

Contents are never copied: only paths and sha256 (config.yaml holds the
dashboard token, so only its hash is recorded).
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hwv_common as hc  # noqa: E402

#: `make deploy` rsync excludes (Makefile), plus models/ (provisioned by
#: setup_pi.sh, hashed separately) and runtime state the service writes.
EXCLUDE_DIRS = {".venv", "data", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "logs",
                "ios", "models", "node_modules"}
EXCLUDE_PATHS = {".git", "config.yaml", "cloud/.jwt_secret", "cloud/.backend_id", "benchmarks/results"}
EXCLUDE_GLOBS = ("*.log", ".secrets", "*.pt", "*.hef", "*.db", "*.sqlite3", "*.csv", ".claude*",
                 ".env", ".env.*", "*.pem", "*.key", "*.pyc")


def should_exclude(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return True
    if any(rel == p or rel.startswith(p + "/") for p in EXCLUDE_PATHS):
        return True
    return any(fnmatch.fnmatch(part, g) for part in parts for g in EXCLUDE_GLOBS)


def content_hashes(data: bytes) -> dict:
    return {"sha256": hc.sha256_bytes(data),
            "sha256_lf": hc.sha256_bytes(data.replace(b"\r\n", b"\n"))}


def tree_manifest(root: Path) -> dict[str, dict]:
    out = {}
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)  # prune .venv etc.
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if not should_exclude(rel):
                out[rel] = content_hashes(path.read_bytes())
    return out


def diff_manifests(device: dict[str, dict], ref: dict[str, dict]) -> dict:
    identical, differ, eol_only = [], [], []
    for rel in sorted(set(device) & set(ref)):
        if device[rel]["sha256"] == ref[rel]["sha256"]:
            identical.append(rel)
        elif device[rel]["sha256_lf"] == ref[rel]["sha256_lf"]:
            eol_only.append(rel)
        else:
            differ.append(rel)
    only_device = sorted(set(device) - set(ref))
    only_ref = sorted(set(ref) - set(device))
    return {"identical": len(identical), "differ": differ, "differ_only_line_endings": eol_only,
            "only_on_device": only_device, "only_in_revision": only_ref,
            "exact_match": not (differ or eol_only or only_device or only_ref)}


def git_manifest(repo: Path, rev: str) -> dict[str, dict]:
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev],
                          capture_output=True, check=True, timeout=120).stdout
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if member.isfile() and not should_exclude(member.name):
                fh = tar.extractfile(member)
                if fh is not None:
                    out[member.name] = content_hashes(fh.read())
    return out


def _run(cmd: list[str], timeout: float = 60) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<failed: {exc}>"
    return (out.stdout or out.stderr or "").strip()


def cmd_snapshot(args) -> int:
    import yaml

    root = Path(args.root).resolve()
    ok, why = hc.ensure_outside(Path(args.out), root)
    if not ok:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    accel = config.get("accelerator") or {}
    hef = accel.get("yolo_hef")
    hef_path = (root / hef) if hef and not Path(hef).is_absolute() else (Path(hef) if hef else None)
    hef_sha = hc.sha256_file(hef_path) if hef_path and hef_path.exists() else ""
    pin_ok, pin_why = hc.verify_pin(hef_sha, accel.get("yolo_hef_sha256")) if hef_sha else (False, "no .hef")
    preflight_cmd = [sys.executable, "-B", str(hc.REPO_ROOT / "scripts" / "deploy_preflight.py"),
                     "--config", str(config_path), "--hef-sha256", hef_sha, "--json"]
    if args.expect_role:
        preflight_cmd += ["--expect-role", args.expect_role]
    preflight = _run(preflight_cmd)
    unit = Path("/etc/systemd/system/sensordyme.service")
    venv_python = root / ".venv" / "bin" / "python"
    snapshot = {
        "utc": hc.utc_now(),
        "root": str(root),
        "environment": hc.environment(),
        "files": tree_manifest(root),
        "config": {"sha256": hc.sha256_file(config_path), "role": config.get("role"),
                   "module": config.get("module")},
        "model": {"path": str(hef_path) if hef_path else None, "sha256": hef_sha,
                  "pin": accel.get("yolo_hef_sha256"), "pin_ok": pin_ok, "detail": pin_why},
        "deploy_preflight": {"command": " ".join(preflight_cmd[1:]), "output": preflight},
        "service_unit_sha256": hc.sha256_file(unit) if unit.exists() else None,
        "systemd": _run(["systemctl", "show", "sensordyme.service", "-p",
                         "ActiveState,SubState,ExecMainStartTimestamp,NRestarts"]),
        "pip_freeze": (_run([str(venv_python), "-m", "pip", "freeze"], timeout=120)
                       if venv_python.exists() else None),
        "checkout_used": {"repo": str(hc.REPO_ROOT), "git_commit": hc.git_commit(hc.REPO_ROOT)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    hc.write_json(Path(args.out), snapshot)
    print(f"{len(snapshot['files'])} files hashed; model pin: {pin_why}")
    print(f"config sha256 {snapshot['config']['sha256']}  role {config.get('role')!r}")
    print(f"deploy preflight: {preflight}")
    print(f"-> {args.out}")
    return 0


def cmd_compare(args) -> int:
    snapshot = json.loads(Path(args.snapshot).read_text())
    results = {}
    for rev in args.rev:
        commit = _run(["git", "-C", args.repo, "rev-parse", rev])
        results[rev] = {"commit": commit, **diff_manifests(snapshot["files"], git_manifest(Path(args.repo), rev))}
    best = max(results, key=lambda r: results[r]["identical"]) if results else None
    out = {"snapshot": args.snapshot, "best_match": best, "revisions": results}
    print(json.dumps(out, indent=2))
    if args.out:
        hc.write_json(Path(args.out), out)
    return 0 if best and results[best]["exact_match"] else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Device file-hash manifest and git comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="(Pi) hash the installed tree, read-only")
    s.add_argument("--root", default=str(hc.DEFAULT_INSTALL_DIR))
    s.add_argument("--out", required=True)
    s.add_argument("--expect-role", default=None)
    s.set_defaults(func=cmd_snapshot)
    c = sub.add_parser("compare", help="(worker) diff a snapshot against git revisions")
    c.add_argument("--snapshot", required=True)
    c.add_argument("--repo", default=str(hc.REPO_ROOT))
    c.add_argument("--rev", action="append", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    # Lazily imported code may come from the INSTALLED tree (--code-root): leave no
    # __pycache__ behind in it. Set here, not at import, so importing this file has
    # no process-wide side effect.
    sys.dont_write_bytecode = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(hc.run_cli(main))
