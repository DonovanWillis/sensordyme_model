#!/usr/bin/env python3
"""Release manifest: exactly what a release candidate puts on a device, and a
check that a device holds exactly that.

  build   (Mac/worker) describe a git ref: commit, branch, the sha256 of every
          file in the deploy set (`git archive` with the same excludes as
          `make deploy`, via scripts/hw_validation/integrity_manifest.py), the
          repo config.yaml hash, the model pins (config, setup_pi.sh, model
          registry) and whether they agree, the decoder setting, the device
          Python requirements, the systemd units, and test-run references.
          Writes JSON (machine) and Markdown (people).

  verify  (Mac/worker) compare a DEVICE SNAPSHOT -- produced on the Pi, read-only,
          by `scripts/hw_validation/integrity_manifest.py snapshot` -- against a
          manifest: deployed files, model hash vs the pins, the device config's
          own pin, installed requirement versions, the rendered service unit,
          and the device-config preflight. Exit 0 = PASS, 1 = FAIL.

    python scripts/release/make_release_manifest.py build --ref m6/autonomy-security-runbook \\
        --out-json rc.json --out-md rc.md --test-ref "pytest tests/test_m6_*.py: 120 passed"
    python scripts/release/make_release_manifest.py verify --manifest rc.json --snapshot snap.json

Reads git objects only (never the working tree), so uncommitted edits can never
leak into a manifest. Contents are never copied: paths and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "hw_validation"))

import integrity_manifest as im  # noqa: E402 - path set above; reused, not duplicated

FORMAT = "sensordyme-release-manifest/1"
#: Files whose contents the manifest reads (not just hashes).
_CONFIG = "config.yaml"
_SETUP = "setup_pi.sh"
_REGISTRY = "core/model_registry.yaml"
_DEVICE_REQUIREMENTS = "requirements-device.txt"
_UNITS = ("sensordyme.service", "deploy/systemd/sensordyme-recover.service")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, timeout=60).stdout.strip()


def _archive(repo: Path, rev: str) -> dict[str, bytes]:
    """Every regular file at `rev`, by path (unfiltered)."""
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev],
                          capture_output=True, check=True, timeout=120).stdout
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                fh = tar.extractfile(member)
                if fh is not None:
                    out[member.name] = fh.read()
    return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aggregate_sha256(files: dict[str, dict]) -> str:
    """One hash for the whole deploy set: sha256 over sorted `path sha256` lines."""
    lines = "".join(f"{path} {files[path]['sha256']}\n" for path in sorted(files))
    return _sha(lines.encode())


def setup_model_pins(setup_text: str) -> dict:
    """MODEL_SHA256 / MODEL_SHA256_EVALUATED / MODEL_FILE / MODEL_URL from setup_pi.sh."""
    pins = {}
    for key in ("MODEL_FILE", "MODEL_URL", "MODEL_SHA256", "MODEL_SHA256_EVALUATED"):
        match = re.search(rf'^{key}="([^"]*)"', setup_text, re.MULTILINE)
        pins[key.lower()] = match.group(1) if match else None
    if pins["model_file"]:
        pins["model_file"] = pins["model_file"].replace("$MODEL_DIR", "models")
    return pins


def registry_artifacts(registry_text: str) -> list[dict]:
    try:
        registry = yaml.safe_load(registry_text) or {}
    except yaml.YAMLError:
        return []
    out = []
    for model in registry.get("models") or []:
        for artifact in (model or {}).get("artifacts") or []:
            if isinstance(artifact, dict) and artifact.get("sha256"):
                out.append({"model": model.get("id"), "path": artifact.get("path"),
                            "sha256": str(artifact["sha256"]).lower(), "format": artifact.get("format")})
    return out


def requirement_pins(text: str) -> dict:
    pins, other = {}, []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, version = line.partition("==")
        if sep:
            pins[_norm(name)] = version.strip()
        else:
            other.append(line)
    return {"pins": pins, "unpinned": other}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def build(repo: Path, ref: str, test_refs: list[str], branch: Optional[str] = None) -> dict:
    commit = _git(repo, "rev-parse", f"{ref}^{{commit}}")
    if branch is None:
        try:
            branch = _git(repo, "rev-parse", "--abbrev-ref", ref)
        except subprocess.CalledProcessError:
            branch = ""
        if branch in ("HEAD", commit):
            branch = ""
    everything = _archive(repo, commit)
    files = {path: im.content_hashes(data) for path, data in sorted(everything.items())
             if not im.should_exclude(path)}
    config_text = everything.get(_CONFIG, b"").decode("utf-8", "replace")
    config = (yaml.safe_load(config_text) or {}) if config_text else {}
    accel = config.get("accelerator") or {}
    setup = setup_model_pins(everything.get(_SETUP, b"").decode("utf-8", "replace"))
    registry = registry_artifacts(everything.get(_REGISTRY, b"").decode("utf-8", "replace"))
    config_pin = str(accel.get("yolo_hef_sha256") or "").lower()
    hef_path = accel.get("yolo_hef")
    registry_for_hef = [a for a in registry if a["path"] == hef_path]
    agree = bool(config_pin) and config_pin == (setup.get("model_sha256") or "").lower() and \
        all(a["sha256"] == config_pin for a in registry_for_hef) and bool(registry_for_hef)
    setup_text = everything.get(_SETUP, b"").decode("utf-8", "replace")
    extra = re.findall(r"pip install ((?:[A-Za-z0-9_.-]+==[^\s]+\s*)+)$", setup_text, re.MULTILINE)
    requirements = {}
    if _DEVICE_REQUIREMENTS in everything:
        req = requirement_pins(everything[_DEVICE_REQUIREMENTS].decode("utf-8", "replace"))
        requirements[_DEVICE_REQUIREMENTS] = {"sha256": _sha(everything[_DEVICE_REQUIREMENTS]), **req}
    if extra:
        requirements[f"{_SETUP} (pip install)"] = requirement_pins("\n".join(extra[0].split()))
    return {
        "format": FORMAT,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "ref": ref,
        "commit": commit,
        "branch": branch,
        "deploy_set": {
            "excludes": {"dirs": sorted(im.EXCLUDE_DIRS), "paths": sorted(im.EXCLUDE_PATHS),
                         "globs": list(im.EXCLUDE_GLOBS)},
            "file_count": len(files),
            "aggregate_sha256": aggregate_sha256(files),
            "files": files,
        },
        "config": {"path": _CONFIG, "sha256": _sha(everything[_CONFIG]) if _CONFIG in everything else None,
                   "role": config.get("role"), "module": config.get("module"), "profile": config.get("profile"),
                   "note": "repo default only: make deploy never overwrites a device's config.yaml"},
        "model_pins": {"config_yolo_hef": hef_path, "config_yolo_hef_sha256": config_pin or None,
                       "setup_pi": setup, "registry": registry, "consistent": agree},
        "decoder": {"yolo_decoder": accel.get("yolo_decoder", "legacy"),
                    "note": "reference decoder is opt-in; INT8 parity unvalidated (docs/DECODER_CANDIDATE.md)"},
        "requirements": requirements,
        "systemd_units": {u: _sha(everything[u]) for u in _UNITS if u in everything},
        "test_runs": list(test_refs),
    }


def to_markdown(manifest: dict) -> str:
    pins = manifest["model_pins"]
    setup = pins["setup_pi"]
    lines = [
        f"# Release manifest — `{manifest['commit'][:12]}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Ref | `{manifest['ref']}` |",
        f"| Commit | `{manifest['commit']}` |",
        f"| Branch | `{manifest['branch'] or '(detached)'}` |",
        f"| Generated (UTC) | {manifest['generated_utc']} |",
        f"| Deploy-set files | {manifest['deploy_set']['file_count']} |",
        f"| Deploy-set aggregate sha256 | `{manifest['deploy_set']['aggregate_sha256']}` |",
        f"| config.yaml sha256 (repo default) | `{manifest['config']['sha256']}` |",
        f"| Decoder | `{manifest['decoder']['yolo_decoder']}` |",
        "",
        "## Model pins",
        "",
        f"- config `accelerator.yolo_hef`: `{pins['config_yolo_hef']}`",
        f"- config `accelerator.yolo_hef_sha256`: `{pins['config_yolo_hef_sha256']}`",
        f"- setup_pi.sh pinned: `{setup.get('model_sha256')}`",
        f"- setup_pi.sh evaluated (provenance unverified; the runtime refuses it): `{setup.get('model_sha256_evaluated')}`",
        f"- registry artifacts: {len(pins['registry'])}",
        f"- **pins consistent: {'YES' if pins['consistent'] else 'NO'}**",
        "",
        "## Device requirements",
        "",
    ]
    for name, req in manifest["requirements"].items():
        lines.append(f"- `{name}`" + (f" sha256 `{req['sha256']}`" if req.get("sha256") else ""))
        lines += [f"  - {pkg}=={ver}" for pkg, ver in sorted(req["pins"].items())]
    lines += ["", "## systemd units (templates)", ""]
    lines += [f"- `{unit}` `{sha}`" for unit, sha in manifest["systemd_units"].items()]
    lines += ["", "## Test runs", ""]
    lines += [f"- {ref}" for ref in manifest["test_runs"]] or ["- (none recorded)"]
    lines += ["", "Per-file sha256 for every deployed file: the JSON manifest, `deploy_set.files`.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ verify
def _freeze_versions(freeze: Optional[str]) -> dict:
    out = {}
    for line in (freeze or "").splitlines():
        name, sep, version = line.partition("==")
        if sep:
            out[_norm(name)] = version.strip()
    return out


def _render_unit(template: bytes, root: str) -> list[str]:
    """Plausible rendered-unit hashes for an install at `root` (user from /home/<user>/...)."""
    parts = Path(root).parts
    users = [parts[2]] if len(parts) > 2 and parts[1] == "home" else []
    text = template.decode("utf-8", "replace")
    return [_sha(text.replace("__SENSORDYME_DIR__", root).replace("__SENSORDYME_USER__", u).encode())
            for u in users]


def verify(manifest: dict, snapshot: dict, unit_template: Optional[bytes] = None) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"check": name, "result": "PASS" if ok else ("INFO" if ok is None else "FAIL"),
                       "detail": detail})

    diff = im.diff_manifests(snapshot.get("files") or {}, manifest["deploy_set"]["files"])
    add("deployed files", diff["exact_match"],
        f"{diff['identical']} identical; differ {diff['differ'][:20]}; line-endings-only "
        f"{diff['differ_only_line_endings'][:20]}; only on device {diff['only_on_device'][:20]}; "
        f"missing on device {diff['only_in_revision'][:20]}")

    pins = manifest["model_pins"]
    pinned = (pins.get("config_yolo_hef_sha256") or "").lower()
    model = snapshot.get("model") or {}
    have = str(model.get("sha256") or "").lower()
    add("model hash = release pin", bool(have) and have == pinned,
        f"device {have or '(no .hef)'} vs pinned {pinned or '(none)'}")
    device_pin = str(model.get("pin") or "").lower()
    add("device config pins the release model", device_pin == pinned,
        f"device config yolo_hef_sha256 {device_pin or '(unset)'}")

    freeze = _freeze_versions(snapshot.get("pip_freeze"))
    wanted = {}
    for req in manifest["requirements"].values():
        wanted.update(req["pins"])
    if snapshot.get("pip_freeze") is None:
        add("installed requirements", False, "snapshot has no pip freeze (no .venv?)")
    else:
        wrong = {k: (v, freeze.get(k)) for k, v in wanted.items() if freeze.get(k) != v}
        add("installed requirements", not wrong,
            "all pinned versions installed" if not wrong else f"wanted vs installed: {wrong}")

    if unit_template is not None and snapshot.get("service_unit_sha256"):
        candidates = _render_unit(unit_template, str(snapshot.get("root", "")))
        add("installed sensordyme.service", snapshot["service_unit_sha256"] in candidates if candidates else None,
            "matches this release's unit rendered for this install" if snapshot["service_unit_sha256"] in candidates
            else "differs from this release's unit: run scripts/release/install_units.sh")

    preflight = (snapshot.get("deploy_preflight") or {}).get("output") or ""
    try:
        parsed = json.loads(preflight)
        add("device config preflight", bool(parsed.get("ok")),
            "; ".join(parsed.get("errors") or []) or f"ok ({len(parsed.get('warnings') or [])} warning(s))")
    except (ValueError, AttributeError):
        add("device config preflight", None, f"unparsed output: {preflight[:200]}")

    add("device config (informational)", None,
        f"sha256 {(snapshot.get('config') or {}).get('sha256')} role {(snapshot.get('config') or {}).get('role')!r}")
    failed = [c for c in checks if c["result"] == "FAIL"]
    return {"manifest_commit": manifest["commit"], "verdict": "FAIL" if failed else "PASS", "checks": checks}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Release manifest: build for a git ref, verify a device snapshot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", required=True)
    b.add_argument("--repo", default=str(REPO_ROOT))
    b.add_argument("--branch", default=None, help="branch name to record (default: derived from --ref)")
    b.add_argument("--test-ref", action="append", default=[], help="a test run backing this release (repeatable)")
    b.add_argument("--out-json", required=True)
    b.add_argument("--out-md", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--snapshot", required=True, help="from integrity_manifest.py snapshot, run on the device")
    v.add_argument("--repo", default=str(REPO_ROOT))
    v.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        manifest = build(Path(args.repo), args.ref, args.test_ref, args.branch)
        Path(args.out_json).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.out_md:
            Path(args.out_md).write_text(to_markdown(manifest))
        print(f"{manifest['commit']}  {manifest['deploy_set']['file_count']} files  "
              f"aggregate {manifest['deploy_set']['aggregate_sha256'][:16]}…  "
              f"model pins consistent: {manifest['model_pins']['consistent']}")
        return 0 if manifest["model_pins"]["consistent"] else 1

    manifest = json.loads(Path(args.manifest).read_text())
    snapshot = json.loads(Path(args.snapshot).read_text())
    template = None
    try:
        template = subprocess.run(["git", "-C", args.repo, "show", f"{manifest['commit']}:sensordyme.service"],
                                  capture_output=True, check=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        template = None
    result = verify(manifest, snapshot, template)
    for check in result["checks"]:
        print(f"  [{check['result']}] {check['check']}: {check['detail']}")
    print(f"VERDICT: {result['verdict']} against {manifest['commit'][:12]}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
