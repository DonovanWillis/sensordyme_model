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

  check   (Mac/worker) the release gates over a built manifest -- model pins
          consistent; decoder `legacy` (the reference decoder is never part of
          a known-good release); every /v1 route the device and the iOS app
          call exists in the backend; the hashed device lock agrees with the
          device pins; the repo config.yaml passes core.config_schema; the
          status-tier file is well formed. Exit 0 = all gates pass.

  verify  (Mac/worker) compare a DEVICE SNAPSHOT -- produced on the Pi, read-only,
          by `scripts/hw_validation/integrity_manifest.py snapshot` -- against a
          manifest: deployed files, model hash vs the pins, the device config's
          own pin, installed requirement versions, the rendered service unit,
          and the device-config preflight. Exit 0 = PASS, 1 = FAIL.

    python scripts/release/make_release_manifest.py build --ref m6/autonomy-security-runbook \\
        --out-json rc.json --out-md rc.md --test-ref "pytest tests/test_m6_*.py: 120 passed"
    python scripts/release/make_release_manifest.py check --manifest rc.json
    python scripts/release/make_release_manifest.py verify --manifest rc.json --snapshot snap.json

Reads git objects only (never the working tree), so uncommitted edits can never
leak into a manifest. Contents are never copied: paths and hashes only.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
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
sys.path.insert(0, str(HERE))

import check_device_deps as cdd  # noqa: E402 - path set above (marker evaluation, lock parsing)
import integrity_manifest as im  # noqa: E402 - path set above; reused, not duplicated

FORMAT = "sensordyme-release-manifest/1"
#: Files whose contents the manifest reads (not just hashes).
_CONFIG = "config.yaml"
_SETUP = "setup_pi.sh"
_REGISTRY = "core/model_registry.yaml"
_DEVICE_REQUIREMENTS = "requirements-device.txt"
_DEVICE_LOCK = "requirements-device.lock"
_STATUS = "scripts/release/release_status.yaml"
_UNITS = ("sensordyme.service", "deploy/systemd/sensordyme-recover.service",
          "deploy/systemd/sensordyme.service.hardened",
          "deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf")
TIERS = ("software-ready", "simulator-verified", "hardware-pending", "experiment-only", "production-blocked")
#: Files the device itself writes next to the code: scripts/migrate_config.py's
#: config backups. Reported, never counted as drift. Deliberately narrow.
DEVICE_STATE_GLOBS = ("config.yaml.bak-*",
                      # M8.5: a staged release's own records, written by release_guard.py stage
                      "STAGED.json", "ARTIFACTS.json")


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
    """`pins`: what the DEVICE installs (markers evaluated for the device, M8.5);
    `not_on_device`: pins whose marker keeps them off it (e.g. simpleaudio on Linux)."""
    pins, other, elsewhere = {}, [], {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        spec, _, marker = line.partition(";")
        name, sep, version = spec.partition("==")
        if not sep:
            other.append(line)
        elif cdd.marker_applies(marker.strip() or None):
            pins[_norm(name)] = version.strip()
        else:
            elsewhere[_norm(name)] = version.strip()
    return {"pins": pins, "unpinned": other, "not_on_device": elsewhere}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


# ------------------------------------------------------------------ release facts
def _text(everything: dict[str, bytes], path: str) -> str:
    return everything.get(path, b"").decode("utf-8", "replace")


def _first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


def route_key(path: str) -> str:
    r"""`/v1/devices/{device_id}/rules` (backend) and `v1/devices/\(id)/rules` (Swift) -> `v1/devices/{}/rules`."""
    path = re.sub(r"\\\([^)]*\)", "{}", path.strip().lstrip("/"))
    return re.sub(r"\{[^}]*\}", "{}", path).split("?", 1)[0].rstrip("/")


def versions_and_protocol(everything: dict[str, bytes]) -> dict:
    """App <-> backend <-> device: declared versions, and whether every /v1 route a
    client calls exists in the backend. A side whose source is absent is reported, not failed."""
    backend_src = _text(everything, "cloud/app.py")
    backend = {route_key(p) for p in re.findall(r'@app\.(?:get|post|put|patch|delete)\("(/v1/[^"]+)"', backend_src)}
    device = {route_key(p) for p in re.findall(r"(/v1/[A-Za-z0-9_{}/.-]+)", _text(everything, "core/uplink.py"))}
    app: set[str] = set()
    for path, data in everything.items():
        if path.startswith("ios/Sources/") and path.endswith(".swift"):
            app.update(route_key(p) for p in re.findall(r'"(/?v1/[^"\s]*)"', data.decode("utf-8", "replace")))
    device_missing = sorted(device - backend) if backend else []
    app_missing = sorted(app - backend) if backend else []
    return {
        "versions": {
            "device_runtime": _first(r'^_VERSION = "([^"]+)"', _text(everything, "services/runtime.py")),
            "backend_api": _first(r'FastAPI\([^)]*?version="([^"]+)"', backend_src),
            "ios_app": _first(r'MARKETING_VERSION: "([^"]+)"', _text(everything, "ios/project.yml")),
            "api_prefix": "/v1" if backend else None,
        },
        "backend_routes": len(backend),
        "device_routes": sorted(device),
        "app_routes": len(app),
        "device_missing_in_backend": device_missing,
        "app_missing_in_backend": app_missing,
        "compatible": (not device_missing and not app_missing) if backend else None,
    }


def lock_status(everything: dict[str, bytes]) -> Optional[dict]:
    """The hashed device lock: does it pin what requirements-device.txt pins, with a hash for every package?"""
    if _DEVICE_LOCK not in everything:
        return None
    lock = _text(everything, _DEVICE_LOCK)
    locked = {_norm(k): v for k, v in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)", lock, re.MULTILINE)}
    pins = requirement_pins(_text(everything, _DEVICE_REQUIREMENTS))["pins"]
    mismatched = {k: [v, locked.get(k)] for k, v in pins.items() if locked.get(k) != v}
    blocks = [b for b in re.split(r"\n(?=[A-Za-z0-9])", lock) if re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*==", b)]
    unhashed = [b.split("==", 1)[0] for b in blocks if "--hash=sha256:" not in b]
    return {"path": _DEVICE_LOCK, "sha256": _sha(everything[_DEVICE_LOCK]), "packages": len(locked),
            "pins_mismatched": mismatched, "unhashed": unhashed, "consistent": not mismatched and not unhashed,
            "locked": locked}


#: Where a device gets its Python packages (M8.5, M84-4). Every pip install of the
#: device set in these files must be the hashed, wheels-only lock install.
INSTALL_PATH_FILES = ("setup_pi.sh", "Makefile", "scripts/release/release_guard.py",
                      "scripts/release/install_units.sh")
#: The marker a deliberate, documented non-device fallback carries on its line.
UNHASHED_FALLBACK_MARK = "unhashed-fallback"


def install_paths(everything: dict[str, bytes]) -> dict:
    """Does every device install path use `--require-hashes` + the lock (M84-4)?

    A violation is a non-comment line that pip-installs requirements-device.txt,
    a bare `pkg==version`, or upgrades pip from the index, unless the line carries
    the UNHASHED_FALLBACK_MARK (a documented fallback that refuses on a device)."""
    violations: list[str] = []
    hashed: dict[str, bool] = {}
    present = [p for p in INSTALL_PATH_FILES if p in everything]
    for path in present:
        text = _text(everything, path)
        installs = bool(re.search(r"pip\b.*\binstall\b|\"install\"", text))
        hashed[path] = not installs or ("--require-hashes" in text and _DEVICE_LOCK in text)
        for number, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0] if not path.endswith(".py") else line
            stripped = code.strip()
            if not stripped or stripped.startswith("#") or UNHASHED_FALLBACK_MARK in line:
                continue
            unhashed_req = re.search(r"pip[\w.\"', ]*install.*requirements-device\.txt", code)
            bare = re.search(r"pip[\w.\"', ]*install\s+(?:-q\s+)?[A-Za-z0-9_.-]+==", code)
            upgrade = re.search(r"pip[\w.\"', ]*install\s+(?:-q\s+)?--upgrade\s+pip", code)
            if unhashed_req or bare or upgrade:
                violations.append(f"{path}:{number}: {stripped[:120]}")
    needs_lock = [p for p in present if not hashed.get(p)]
    return {"files": present, "violations": violations, "missing_hashed_install": needs_lock,
            "ok": bool(present) and not violations and not needs_lock}


def config_schema_status(config: dict) -> dict:
    """core.config_schema.validate on the release's default config.yaml (the validator is this checkout's)."""
    if not config:
        return {"checked": False, "errors": [], "warnings": [], "note": "no config.yaml"}
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from core.config_schema import validate
    except Exception as exc:  # noqa: BLE001 - report it; never crash a manifest build
        return {"checked": False, "errors": [], "warnings": [], "note": f"validator unavailable: {exc}"}
    errors, warnings = validate(config)
    return {"checked": True, "errors": list(errors), "warnings": list(warnings)}


def status_tiers(everything: dict[str, bytes]) -> Optional[dict]:
    """scripts/release/release_status.yaml at the ref, validated: known tiers, evidence, no hardware claim."""
    if _STATUS not in everything:
        return None
    try:
        data = yaml.safe_load(_text(everything, _STATUS)) or {}
    except yaml.YAMLError as exc:
        return {"path": _STATUS, "sha256": _sha(everything[_STATUS]), "as_of": None, "components": {},
                "problems": [f"unparseable: {exc}"]}
    problems: list[str] = []
    components = {}
    for name, raw in (data.get("components") or {}).items():
        entry = raw or {}
        tiers = list(entry.get("tiers") or [])
        evidence = str(entry.get("evidence") or "")
        if not tiers or any(t not in TIERS for t in tiers):
            problems.append(f"{name}: tiers {tiers} not within {list(TIERS)}")
        if not evidence:
            problems.append(f"{name}: no evidence")
        if re.search(r"\b(hardware|device|physically)[- ]validated\b", evidence, re.IGNORECASE):
            problems.append(f"{name}: claims hardware validation")
        components[name] = {"tiers": tiers, "evidence": evidence,
                            "pending_integration": bool(entry.get("pending_integration"))}
    return {"path": _STATUS, "sha256": _sha(everything[_STATUS]), "as_of": data.get("as_of"),
            "components": components, "problems": problems}


def release_gates(manifest: dict) -> list[dict]:
    """What a release must pass before it may go to a device. An input absent at the
    ref (an older release without a lock or status file) is INFO, never a silent PASS."""
    gates: list[dict] = []

    def gate(name: str, ok: Optional[bool], detail: str) -> None:
        gates.append({"gate": name, "result": "PASS" if ok else ("INFO" if ok is None else "FAIL"),
                      "detail": detail})

    pins = manifest["model_pins"]
    gate("model pins consistent (config = setup_pi.sh = registry)", pins["consistent"],
         f".hef {pins['config_yolo_hef']} sha256 {pins['config_yolo_hef_sha256']}")
    decoder = manifest["decoder"]["yolo_decoder"]
    gate("decoder is legacy (the reference decoder is not enabled)", decoder == "legacy", f"yolo_decoder: {decoder}")
    proto = manifest.get("protocol")
    if proto:
        gate("protocol: every /v1 route the device and app call exists in the backend", proto["compatible"],
             f"versions {proto['versions']}; device missing {proto['device_missing_in_backend']}; "
             f"app missing {proto['app_missing_in_backend']}")
    else:
        gate("protocol", None, "no protocol section (manifest predates M6.5)")
    lock = manifest.get("dependency_lock")
    if lock:
        gate("device lock agrees with the device pins, every package hashed", lock["consistent"],
             f"{lock['packages']} packages, sha256 {lock['sha256'][:16]}; mismatched {lock['pins_mismatched']}; "
             f"unhashed {lock['unhashed']}")
    else:
        gate("device lock", None, "no requirements-device.lock at this ref")
    paths = manifest.get("install_paths")
    if paths:
        gate("every device install path uses pip --require-hashes with the lock (M84-4)", paths["ok"],
             f"files {paths['files']}; unhashed installs {paths['violations'][:5]}; "
             f"no hashed install in {paths['missing_hashed_install']}")
    else:
        gate("device install paths", None, "no install-path facts (manifest predates M8.5)")
    schema = manifest.get("config_schema") or {}
    if schema.get("checked"):
        gate("repo config.yaml passes core.config_schema", not schema["errors"],
             f"{len(schema['errors'])} error(s), {len(schema['warnings'])} warning(s) {schema['errors'][:3]}")
    else:
        gate("repo config.yaml schema", None, schema.get("note", "not checked"))
    tiers = manifest.get("status_tiers")
    if tiers:
        gate("status tiers well formed, no hardware-validation claim", not tiers["problems"],
             f"{len(tiers['components'])} components; problems {tiers['problems']}")
    else:
        gate("status tiers", None, "no scripts/release/release_status.yaml at this ref")
    return gates


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
        "dependency_lock": lock_status(everything),
        "install_paths": install_paths(everything),
        "protocol": versions_and_protocol(everything),
        "config_schema": config_schema_status(config),
        "status_tiers": status_tiers(everything),
        "known_good": {
            "note": "the configuration this release was checked with; a device's own config.yaml is never overwritten",
            "config_sha256": _sha(everything[_CONFIG]) if _CONFIG in everything else None,
            "role": config.get("role"), "module": config.get("module"), "profile": config.get("profile"),
            "yolo_decoder": accel.get("yolo_decoder", "legacy"),
            "yolo_hef": hef_path, "yolo_hef_sha256": config_pin or None,
            "requirements_device_sha256": (requirements.get(_DEVICE_REQUIREMENTS) or {}).get("sha256"),
            "service_unit": "sensordyme.service (stock); the least-privilege unit is hardware-pending",
        },
        "procedures": {
            "rollback": "docs/MONDAY_RUNBOOK.md section O (O1 previous release, O2 pre-Monday state, "
                        "O3 SD image, O4 least-privilege unit)",
            "identity_migration": "none in this release: device id, device key and claim code carry over "
                                  "unchanged (runbook F1 copies config.yaml and data/); no key is rotated",
            "least_privilege_migration": "scripts/release/least_privilege_plan.sh (prints only; "
                                         "docs/LEAST_PRIVILEGE.md)",
        },
        "test_runs": list(test_refs),
    }


def bundle(repo: Path, ref: str, out_dir: Path, test_refs: Optional[list[str]] = None) -> dict:
    """M8.5 (M84-6): the release a device stages. From git objects only (never the
    working tree): a deterministic tar.gz of EXACTLY the manifest's deploy set
    (same excludes; regular files; mode 0644/0755; owner 0; mtime = commit time)
    plus the manifest it is verified against. release_guard.py `stage` refuses any
    file whose bytes differ, any extra or missing file, links and device state."""
    manifest = build(repo, ref, list(test_refs or []))
    commit = manifest["commit"]
    rid = commit[:12]
    when = int(_git(repo, "show", "-s", "--format=%ct", commit))
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit],
                          capture_output=True, check=True, timeout=120).stdout
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"sensordyme-{rid}.tar.gz"
    wanted = manifest["deploy_set"]["files"]
    written = set()
    with tarfile.open(fileobj=io.BytesIO(data)) as source, open(tar_path, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as out:
        for member in sorted(source.getmembers(), key=lambda m: m.name):
            if not member.isfile() or member.name not in wanted:
                continue
            handle = source.extractfile(member)
            blob = handle.read() if handle is not None else b""
            if _sha(blob) != wanted[member.name]["sha256"]:
                raise RuntimeError(f"{member.name}: archive bytes differ from the manifest")
            info = tarfile.TarInfo(member.name)
            info.size, info.mtime = len(blob), when
            info.mode = 0o755 if member.mode & 0o111 else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            out.addfile(info, io.BytesIO(blob))
            written.add(member.name)
    if written != set(wanted):
        raise RuntimeError(f"bundle and manifest disagree: {sorted(set(wanted) ^ written)[:10]}")
    manifest_path = out_dir / f"sensordyme-{rid}.manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    (out_dir / f"sensordyme-{rid}.manifest.md").write_text(to_markdown(manifest))
    sums = {tar_path.name: _sha(tar_path.read_bytes()), manifest_path.name: _sha(manifest_bytes)}
    (out_dir / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    (out_dir / "RELEASE_ID").write_text(rid + "\n")
    (out_dir / "COMMIT").write_text(commit + "\n")
    return {"release": rid, "commit": commit, "files": len(written), "tarball": str(tar_path),
            "manifest": str(manifest_path), "manifest_sha256": sums[manifest_path.name],
            "tarball_sha256": sums[tar_path.name],
            "gates_ok": not any(g["result"] == "FAIL" for g in release_gates(manifest))}


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
    proto = manifest.get("protocol") or {}
    if proto:
        lines += ["", "## Versions and protocol", ""]
        lines += [f"- {k}: `{v}`" for k, v in (proto.get("versions") or {}).items()]
        lines += [(f"- backend /v1 routes: {proto.get('backend_routes')}; device calls {proto.get('device_routes')}; "
                   f"app calls {proto.get('app_routes')} routes"),
                  f"- **compatible: {proto.get('compatible')}**"]
    lock = manifest.get("dependency_lock")
    if lock:
        lines += ["", "## Device dependency lock", "",
                  (f"- `{lock['path']}` sha256 `{lock['sha256']}`, {lock['packages']} packages, "
                   f"consistent with the pins: **{lock['consistent']}**")]
    tiers = manifest.get("status_tiers")
    if tiers:
        lines += ["", f"## Status tiers (as of {tiers.get('as_of')})", "",
                  "| Component | Tiers | Evidence |", "|---|---|---|"]
        for name, entry in tiers["components"].items():
            flag = " *(pending integration)*" if entry["pending_integration"] else ""
            lines.append(f"| {name}{flag} | {', '.join(entry['tiers'])} | {entry['evidence']} |")
    lines += ["", "## Release gates", ""]
    lines += [f"- [{g['result']}] {g['gate']}: {g['detail']}" for g in release_gates(manifest)]
    lines += ["", "## Procedures", ""] + [f"- {k}: {v}" for k, v in (manifest.get("procedures") or {}).items()]
    lines += ["", "## Test runs", ""]
    lines += [f"- {ref}" for ref in manifest["test_runs"]] or ["- (none recorded)"]
    lines += ["", "Per-file sha256 for every deployed file: the JSON manifest, `deploy_set.files`.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ verify
#: What `pip freeze` leaves out by default.
_FREEZE_HIDDEN = frozenset({"pip", "setuptools", "wheel", "distribute"})


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

    device_files = dict(snapshot.get("files") or {})
    # Device state the release legitimately leaves beside the code (not code drift).
    state = sorted(p for p in device_files if any(fnmatch.fnmatch(p, g) for g in DEVICE_STATE_GLOBS))
    for path in state:
        device_files.pop(path)
    diff = im.diff_manifests(device_files, manifest["deploy_set"]["files"])
    add("deployed files", diff["exact_match"],
        f"{diff['identical']} identical; differ {diff['differ'][:20]}; line-endings-only "
        f"{diff['differ_only_line_endings'][:20]}; only on device {diff['only_on_device'][:20]}; "
        f"missing on device {diff['only_in_revision'][:20]}")
    if state:
        add("device state files (informational)", None, f"not code, ignored: {state[:20]}")

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
        locked = (manifest.get("dependency_lock") or {}).get("locked")
        if locked:
            # M8.5: the hashed lock installs EVERY package at its locked version, so
            # a transitive dependency at another version means the venv did not come
            # from the lock. pip freeze never lists pip/setuptools/wheel.
            drift = {k: (v, freeze.get(k)) for k, v in locked.items()
                     if k not in _FREEZE_HIDDEN and freeze.get(k) != v}
            add("installed packages = the hashed lock", not drift,
                f"all {len(locked)} locked versions installed" if not drift else f"locked vs installed: {drift}")

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
    k = sub.add_parser("bundle", help="M8.5: the release tarball + manifest a device stages (make deploy)")
    k.add_argument("--ref", required=True)
    k.add_argument("--repo", default=str(REPO_ROOT))
    k.add_argument("--out-dir", required=True)
    k.add_argument("--test-ref", action="append", default=[])
    k.add_argument("--allow-failed-gates", action="store_true",
                   help="write the bundle even when a release gate FAILs (never for a device)")
    c = sub.add_parser("check", help="release gates over a built manifest")
    c.add_argument("--manifest", required=True)
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

    if args.cmd == "bundle":
        made = bundle(Path(args.repo), args.ref, Path(args.out_dir), args.test_ref)
        print(f"release {made['release']} ({made['commit']}): {made['files']} files\n"
              f"  {made['tarball']}  sha256 {made['tarball_sha256']}\n"
              f"  {made['manifest']}  sha256 {made['manifest_sha256']}")
        if not made["gates_ok"]:
            print("RELEASE GATES FAIL for this ref: `make_release_manifest.py check` says why. Not for a device.")
            return 0 if args.allow_failed_gates else 1
        return 0

    manifest = json.loads(Path(args.manifest).read_text())
    if args.cmd == "check":
        gates = release_gates(manifest)
        for g in gates:
            print(f"  [{g['result']}] {g['gate']}: {g['detail']}")
        failed = [g for g in gates if g["result"] == "FAIL"]
        print(f"GATES: {'FAIL' if failed else 'PASS'} for {manifest['commit'][:12]}")
        return 1 if failed else 0
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
