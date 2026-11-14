#!/usr/bin/env python3
"""The transfer package: everything a Raspberry Pi needs, in one directory (M8.5).

`make transfer-package REF=<rc>` builds `~/sensordyme-release/transfer-<id>/`
(`<id>` = commit[:12]) for an operator carrying a release to a device that may
have no internet:

  bundle/    the release tarball + manifest from `make_release_manifest.py
             bundle` (git objects only; refused if a release gate FAILs). This
             is what `release_guard.py stage` verifies file by file.
  wheels/    every locked device package as an aarch64/cp311 wheel, fetched with
             `pip download --require-hashes --only-binary=:all:` -- pip refuses
             any wheel whose sha256 is not in requirements-device.lock. A copy
             of the lock sits beside them. Used with `--find-links` for a fully
             offline staged install (docs/DEPLOYMENT.md §3).
  models/    the production .hef, downloaded from the URL pinned in setup_pi.sh
             and REFUSED unless its sha256 equals the pin -- verified before it
             enters the package, exactly as setup_pi.sh would on the device.
  config/    config.yaml.example: the repo default at the ref. A template only;
             a device's own config.yaml is state and never ships.
  install/   convenience copies (from git objects at the ref) of setup_pi.sh,
             release_guard.py, install_units.sh and the systemd unit templates.
             The canonical copies are inside the bundle tarball.
  INSTALL.md / MODELS.md   generated for this exact release (entrypoints,
             compatibility floor, rollback, offline install; model inventory).
  PACKAGE.json             what was built, with what tools, and every size.
  SHA256SUMS               sha256 of every file above (sha256sum -c compatible).

Secrets never enter the package: before SHA256SUMS is written, every file is
checked against a refusal list (.secrets, *.pem, *.key, .env, a real
config.yaml, factory.json, device_state*, databases) and the config example is
parsed and refused if any *key*/*token*/*secret*/*password*/*psk* field holds a
value. The one permitted download besides the hash-checked wheels is the .hef,
and only at its pinned sha256.

Deterministic where feasible: the bundle tarball is byte-stable per commit, file
order is sorted everywhere, no wall-clock timestamp enters any hashed file
(PACKAGE.json carries the commit time and the recorded tool versions), and the
wheels are hash-pinned, so rebuilding the same ref with the same toolchain gives
the same SHA256SUMS.

    python scripts/release/build_transfer_package.py --ref <rc> \
        [--out-root ~/sensordyme-release] [--skip-wheels] [--hef <file> | --skip-hef]

`--skip-wheels` / `--skip-hef` build an INCOMPLETE package (clearly marked; for
tests and dry runs only -- never carry one to a device). `--hef <file>` reuses
an already-downloaded model file; it is verified against the pin all the same.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import make_release_manifest as rm  # noqa: E402 - path set above; reused, not duplicated

#: Never in a transfer package, wherever it sits. The task of this list is the
#: same as release_guard's EXCLUDED_* (device state and secrets never ship), but
#: applied to the PACKAGE directory, where the only sanctioned config file is
#: the example. `models/*.hef` is deliberately shippable here: the package
#: exists to carry the verified model.
FORBIDDEN_GLOBS = (".secrets", "*.pem", "*.key", ".env", ".env.*", "config.yaml",
                   "factory.json", "device_state*", "*.db", "*.sqlite3",
                   ".jwt_secret", ".backend_id", "*.yaml.bak-*")
#: A config value that looks like a credential: any mapping key naming one of
#: these, holding a non-empty scalar, refuses the package.
CREDENTIAL_KEY_WORDS = ("key", "token", "secret", "password", "psk", "bearer")
#: Allowed empty/placeholder values for such keys in the repo default config.
_EMPTY = (None, "", "null")

#: The aarch64 wheel platforms the Pi (Bookworm, CPython 3.11) accepts. The same
#: list as docs/DEPLOYMENT.md §3 and the validated m85 wheelhouse run.
WHEEL_PLATFORMS = (*(f"manylinux_2_{v}_aarch64" for v in (36, 35, 34, 33, 32, 31, 28, 27, 26, 24, 17)),
                   "manylinux2014_aarch64", "linux_aarch64")

#: Convenience copies placed under install/ (source path at the ref -> package name).
INSTALL_COPIES = (
    ("setup_pi.sh", "setup_pi.sh"),
    ("scripts/release/release_guard.py", "release_guard.py"),
    ("scripts/release/install_units.sh", "install_units.sh"),
    ("sensordyme.service", "sensordyme.service"),
    ("deploy/systemd/sensordyme-recover.service", "systemd/sensordyme-recover.service"),
    ("deploy/systemd/sensordyme.service.hardened", "systemd/sensordyme.service.hardened"),
    ("deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf",
     "systemd/sensordyme.service.d/btmgmt-advertising.conf"),
)

#: EXPERIMENTAL path B: the EXP-08 3-class (person/phone/hand) device-layout
#: export, oryn job 20261113-090349-6d22e9 (docs/research/M84_MODEL_AUDIT.md).
#: Inputs to a FUTURE Hailo Dataflow Compiler run -- NOT COMPILED, NOT PRODUCTION.
EXPERIMENTAL_EXPORT = {
    "job": "20261113-090349-6d22e9 (oryn; ~/sensordyme-fleet/runs/<job>/artifacts/)",
    "checkpoint_sha256": "baad3842943b6ebc4cd58ecb6eff0a6615e9238d92d115cee50ddca94f4ef37b",
    "onnx_sha256": "f591a6b6eca94bafc8f71f8cc1026b0b0ec06fa37c0424f465befa6c6040a609",
    "heads3_sha256": "9a03d0612994c548336e544e8b1a1651db9cbb57d8c80200150929830f1f047e",
    "heads81_sha256": "a0327a25da034e942bb1f6529b7747cb5ddef8c821dc9545ac40ccf394ed7aa1",
    "zoo_reference_onnx_sha256": "02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657",
    "float_fixtures": ("exp08_known_good_expected_float.json", "exp08_oi_val_expected_float.json"),
}


class PackageError(Exception):
    """A package that must not be carried to a device; the build stops."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _show(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                          capture_output=True, check=True, timeout=60).stdout


def _model_pins(setup_text: str) -> dict:
    """The .hef pins the package verifies against (a seam the tests re-point;
    the release gates inside rm.bundle always read the real parser)."""
    return rm.setup_model_pins(setup_text)


# ------------------------------------------------------------------ the .hef (the ONE download)
def _urllib_download(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise PackageError(f"refusing a non-https model URL: {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:  # noqa: S310 - https enforced above; the bytes are verified against the pin before use
        shutil.copyfileobj(response, out)


def fetch_hef(url: str, expected_sha256: str, dest: Path, *,
              download: Callable[[str, Path], None] = _urllib_download,
              local_file: Optional[Path] = None) -> dict:
    """The production model, verified against the pin BEFORE it enters the package.

    A mismatch removes the bytes and raises PackageError: a model file whose
    sha256 is not the pinned one is never packaged, whatever the URL served."""
    expected = expected_sha256.strip().lower()
    if not expected or len(expected) != 64:
        raise PackageError(f"no usable model pin at this ref (got {expected_sha256!r})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        if local_file is not None:
            shutil.copyfile(local_file, part)
            source = f"local file {local_file}"
        else:
            download(url, part)
            source = url
        got = _sha256_file(part)
        if got != expected:
            raise PackageError(f"REFUSED: the model from {source} hashes to {got}, not the pinned "
                               f"{expected}; it does not enter the package")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    return {"path": dest.name, "sha256": expected, "bytes": dest.stat().st_size, "url": url,
            "source": "local file (verified against the pin)" if local_file else "downloaded (verified)"}


# ------------------------------------------------------------------ the wheels
def download_wheels(lock: Path, dest: Path, python: str) -> dict:
    """Every locked package as an aarch64/cp311 wheel, hash-checked by pip itself
    (`--require-hashes`: a wheel whose sha256 is not in the lock fails the run)."""
    dest.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "pip", "download", "--require-hashes", "--only-binary=:all:",
            "--disable-pip-version-check", "--no-input"]
    for platform in WHEEL_PLATFORMS:
        argv += ["--platform", platform]
    argv += ["--python-version", "3.11", "--implementation", "cp", "--abi", "cp311",
             "-r", str(lock), "-d", str(dest)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800, check=False)
    if proc.returncode != 0:
        raise PackageError("REFUSED: the hash-checked wheel download failed "
                           f"(exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    wheels = sorted(dest.glob("*.whl"))
    if not wheels:
        raise PackageError("the wheel download produced no wheels")
    pip_version = subprocess.run([python, "-m", "pip", "--version"], capture_output=True,
                                 text=True, timeout=60, check=False).stdout.strip()
    return {"count": len(wheels), "bytes": sum(w.stat().st_size for w in wheels),
            "pip": pip_version, "command": " ".join(argv)}


# ------------------------------------------------------------------ refusals
def forbidden_files(root: Path) -> list[str]:
    """Package files that may never ship: secrets, device state, a real config."""
    bad = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            bad.append(f"{path.relative_to(root).as_posix()}: a symlink never ships")
            continue
        if not path.is_file():
            continue
        name = path.name
        for pattern in FORBIDDEN_GLOBS:
            if fnmatch.fnmatch(name, pattern):
                bad.append(f"{path.relative_to(root).as_posix()}: matches {pattern!r}")
                break
    return bad


def _credential_values(node: object, where: str, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            spot = f"{where}.{key}" if where else str(key)
            words = set(re.split(r"[^a-z0-9]+", str(key).lower()))
            # `*_env` names WHERE a credential lives (an env var name), never one.
            suspect = words & set(CREDENTIAL_KEY_WORDS) and not str(key).lower().endswith("_env")
            if suspect and isinstance(value, str) and value not in _EMPTY:
                found.append(f"{spot} holds a value")
            _credential_values(value, spot, found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _credential_values(value, f"{where}[{index}]", found)


def config_credential_check(config_text: str) -> list[str]:
    """Fields of the config example that look like a filled-in credential."""
    import yaml
    try:
        config = yaml.safe_load(config_text) or {}
    except yaml.YAMLError as exc:
        return [f"unparseable config example: {exc}"]
    found: list[str] = []
    _credential_values(config, "", found)
    return found


# ------------------------------------------------------------------ generated docs
def install_md(made: dict, pins: dict, wheels: Optional[dict], hef: Optional[dict]) -> str:
    rid, commit = made["release"], made["commit"]
    man = Path(made["manifest"]).name
    tgz = Path(made["tarball"]).name
    wheels_line = (f"{wheels['count']} wheels, {wheels['bytes'] / 1e6:.0f} MB (hash-checked against the lock)"
                   if wheels else "NOT INCLUDED (`--skip-wheels`): the staged install needs the index, "
                                  "or build the package again without the flag")
    hef_line = (f"`models/{hef['path']}` sha256 `{hef['sha256']}` ({hef['bytes']:,} bytes, verified)"
                if hef else "NOT INCLUDED (`--skip-hef`): run setup_pi.sh online, or build again without the flag")
    return f"""# Installing release `{rid}` from this transfer package

Release commit `{commit}`. Everything below is also in `docs/DEPLOYMENT.md`
(inside the bundle); this file only pins the exact commands for THIS release.

## Compatibility floor (check before anything else)

| Requirement | Why |
|---|---|
| Raspberry Pi OS **Bookworm**, **aarch64** | the wheels and the lock target this OS |
| **Python 3.11** (`python3 --version`) | the lock pins cp311 wheels |
| **HailoRT 4.x** via `hailo-all` (apt) | the .hef is compiled for Hailo-8L; no CPU fallback exists |
| non-interactive sudo for the installing user | the detached activation cannot ask for a password |

## Contents

- `bundle/{tgz}` + `bundle/{man}` -- the release (every file's sha256 in the manifest;
  manifest sha256 `{made['manifest_sha256']}`)
- `wheels/` -- {wheels_line}
- `models/` -- {hef_line}
- `config/config.yaml.example` -- the repo default. A device's own `config.yaml` is device
  state: never overwrite it with this file, and never copy a real one into this package.
- `install/` -- convenience copies of setup_pi.sh, release_guard.py, install_units.sh and
  the unit templates (the canonical copies are inside the bundle tarball)

Verify the package first: `shasum -a 256 -c SHA256SUMS` (Mac) or `sha256sum -c SHA256SUMS` (Pi).

## A. Fresh device (no SensorDyme install yet)

```bash
mkdir -p ~/sensordyme && tar -xzf bundle/{tgz} -C ~/sensordyme
mkdir -p ~/sensordyme/models && cp models/*.hef ~/sensordyme/models/   # setup_pi.sh verifies the pin again
cp -r wheels ~/sensordyme.wheels                                        # optional: offline installs
cd ~/sensordyme && bash setup_pi.sh                                     # the install entrypoint
```

`setup_pi.sh` installs apt packages, the hashed lock (wheels only), verifies the model
against `{pins.get('model_sha256')}`, and installs + starts the systemd service. Then adopt
the staged layout once (`make deploy-adopt`, or on the Pi:
`python3 install/release_guard.py adopt --root ~/sensordyme` with the service stopped).

## B. Staged install onto an adopted device (the normal update)

From the Mac with this repo: `make deploy DEVICE=<user>@<host> DIR=sensordyme REF={commit}`.

By hand, with only this package (copy `bundle/` and `wheels/` to the device first):

```bash
# on the device
mkdir -p ~/sensordyme.incoming && cp bundle/{tgz} bundle/{man} ~/sensordyme.incoming/
python3 install/release_guard.py stage --root ~/sensordyme \\
    --bundle ~/sensordyme.incoming/{tgz} \\
    --manifest ~/sensordyme.incoming/{man} \\
    --manifest-sha256 {made['manifest_sha256']} \\
    --find-links ~/sensordyme.wheels                      # offline: wheels/, still hash-checked
python3 ~/sensordyme.releases/{rid}/scripts/release/release_guard.py activate \\
    --root ~/sensordyme --id {rid} --user "$(id -un)" --attempt pkg1 --detach
python3 ~/sensordyme.releases/{rid}/scripts/release/release_guard.py status \\
    --root ~/sensordyme --wait {rid} --attempt pkg1 --timeout 900    # exit 0 = CONFIRMED
```

Without `--find-links` the stage step fetches the same hash-pinned wheels from PyPI.

## Preflight (before the switch)

- Mac: `make deploy-preflight DEVICE=<user>@<host> DIR=sensordyme ROLE=driver_safety`
  (the device's own config against this code), and `make verify-release REF={commit}`.
- Pi, read-only: `scripts/hw_validation/sunday_preflight.py --install-dir ~/sensordyme
  --expect-commit {commit}` (from the extracted bundle).

## Rollback

- Automatic: a release that does not stay healthy is switched back by the device itself
  (no network, no pip; the previous release keeps its venv and config).
- Manual: `python3 ~/sensordyme/scripts/release/release_guard.py switch --root ~/sensordyme
  --to previous --user "$(id -un)"` (or `make deploy-rollback DEVICE=… DIR=sensordyme`).

## What this package cannot prove (hardware-only)

Everything in `docs/DEPLOYMENT.md` §8: the hashed install under Bookworm's pip, systemd
following the symlink, the detached activation surviving an SSH drop, and the model on a
real Hailo-8L. This package was verified on a Mac and a Linux x86 worker with a real
guard and real per-file verification, but fake systemd and (on x86) fake pip.
"""


def models_md(pins: dict, registry: list[dict], hef: Optional[dict]) -> str:
    rows = "\n".join(f"| `{a['model']}` | `{a['path']}` | `{a['sha256']}` |"
                     for a in registry) or "| (none at this ref) | | |"
    hef_status = (f"included as `models/{hef['path']}`, verified" if hef
                  else "NOT in this package (`--skip-hef`); setup_pi.sh downloads and verifies it")
    exp = EXPERIMENTAL_EXPORT
    return f"""# Model inventory for this transfer package

## Path A -- PRODUCTION (what this release runs)

| Artifact | Where | sha256 |
|---|---|---|
| Hailo `.hef` `damoyolo_tinynasL20_T` | {hef_status}; pinned in setup_pi.sh, config.yaml and the registry | `{pins.get('model_sha256')}` |
| MediaPipe `.tflite` files | inside the `mediapipe` wheel the lock pins; verified at install (`core.model_registry verify-artifacts`) and at every boot (fail closed) | pins below |
| openpilot `dmonitoring_model.onnx` | benchmark module only; pinned at download and at load | see `core/model_registry.yaml` |

Registry artifact pins at this release:

| Model | Artifact | sha256 |
|---|---|---|
{rows}

The runtime refuses a `.hef` whose sha256 is not `accelerator.yolo_hef_sha256`
(`core/accel.py`), and a `.tflite` that does not match its pin is never loaded --
the capability is reported impaired instead (docs/DEPLOYMENT.md §6).

## Path B -- EXPERIMENTAL 3-class export: **NOT COMPILED, NOT PRODUCTION**

> The EXP-08 person/phone/hand detector exists only as a float export and host-side
> fixtures. **No compiled `.hef` of it exists**: the Hailo Dataflow Compiler is not on
> the fleet, and nothing below may be deployed to a device or wired into the
> production path. It is recorded here so the compile inputs are pinned.

| Input | sha256 / where |
|---|---|
| Training checkpoint (EXP-08, 60 epochs) | `{exp['checkpoint_sha256']}` |
| Device-layout ONNX export (RepConv fused, raw heads, opset 11, 640x640) | `{exp['onnx_sha256']}` |
| heads81 variant (zoo-layout 6 raw outputs) | `{exp['heads81_sha256']}` |
| heads3 variant | `{exp['heads3_sha256']}` |
| Zoo reference ONNX (structure comparison) | `{exp['zoo_reference_onnx_sha256']}` |
| Float fixtures | {', '.join(f'`{f}`' for f in exp['float_fixtures'])} |
| Export job | {exp['job']} |

To compile it one day (x86 Linux, Hailo Dataflow Compiler v3.x for hailo8l --
`docs/research/FINETUNE_EXP08_SPEC.md` and `docs/research/M8_CV_RESEARCH.md`):

1. Calibration set: at least **1,024 NoIR frames from the device camera**, collected
   only with consent under the M5 workflow (docs/MONDAY_RUNBOOK.md §R8). The known-good
   reference fixtures and their manifest are `scripts/hw_validation/known_good/`
   (`SOURCES.json`, `expected_float.json`) -- a parity reference, NOT a calibration set.
2. Compile with the DFC against the heads81 export; check host-decoder parity against
   the float fixtures above.
3. Validate on the Pi through the M5/MHS gates before the registry status may move.

Until all of that has happened, the production detector remains the pinned
`damoyolo_tinynasL20_T.hef` above.
"""


# ------------------------------------------------------------------ the build
def write_sha256sums(root: Path) -> tuple[Path, int, int]:
    """SHA256SUMS over every file (sorted, `sha256sum -c` compatible)."""
    entries = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and not (path.name == "SHA256SUMS" and path.parent == root):
            entries.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
            total += path.stat().st_size
    sums = root / "SHA256SUMS"
    sums.write_text("".join(entries))
    return sums, len(entries), total


def verify_sha256sums(root: Path) -> list[str]:
    """Re-hash everything SHA256SUMS lists; also flag files it does not cover."""
    sums = root / "SHA256SUMS"
    problems = []
    listed = set()
    for line in sums.read_text().splitlines():
        want, _, rel = line.partition("  ")
        listed.add(rel)
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: listed but missing")
        elif _sha256_file(path) != want:
            problems.append(f"{rel}: sha256 differs from SHA256SUMS")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    problems += [f"{rel}: on disk but not in SHA256SUMS" for rel in sorted(on_disk - listed)]
    return problems


def build_package(repo: Path, ref: str, out_root: Path, *, python: str = sys.executable,
                  skip_wheels: bool = False, skip_hef: bool = False, hef_file: Optional[Path] = None,
                  download: Callable[[str, Path], None] = _urllib_download,
                  force: bool = False, out: Callable[[str], None] = print) -> dict:
    commit = rm._git(repo, "rev-parse", f"{ref}^{{commit}}")
    rid = commit[:12]
    package = Path(out_root).expanduser() / f"transfer-{rid}"
    if package.exists():
        if not force:
            raise PackageError(f"{package} already exists; pass --force to rebuild it")
        shutil.rmtree(package)
    package.mkdir(parents=True)
    try:
        return _build_into(repo, ref, commit, rid, package, python=python, skip_wheels=skip_wheels,
                           skip_hef=skip_hef, hef_file=hef_file, download=download, out=out)
    except BaseException:
        shutil.rmtree(package, ignore_errors=True)  # never leave a half-built, unhashed package
        raise


def _build_into(repo: Path, ref: str, commit: str, rid: str, package: Path, *, python: str,
                skip_wheels: bool, skip_hef: bool, hef_file: Optional[Path],
                download: Callable[[str, Path], None], out: Callable[[str], None]) -> dict:
    out(f"==> transfer package for {ref} = {commit} -> {package}")

    out("==> 1/6 release bundle (git objects only; release gates)")
    made = rm.bundle(repo, commit, package / "bundle")
    if not made["gates_ok"]:
        raise PackageError("REFUSED: a release gate FAILs for this ref "
                           "(scripts/release/make_release_manifest.py check says why)")

    pins = _model_pins(_show(repo, commit, "setup_pi.sh").decode("utf-8", "replace"))
    registry = rm.registry_artifacts(_show(repo, commit, "core/model_registry.yaml").decode("utf-8", "replace"))

    wheels: Optional[dict] = None
    if skip_wheels:
        out("==> 2/6 wheels: SKIPPED (--skip-wheels) -- the package is INCOMPLETE")
    else:
        out("==> 2/6 hashed aarch64/cp311 wheel set (pip download --require-hashes)")
        lock = package / "wheels" / "requirements-device.lock"
        lock.parent.mkdir(parents=True)
        lock.write_bytes(_show(repo, commit, "requirements-device.lock"))
        wheels = download_wheels(lock, package / "wheels", python)
        out(f"    {wheels['count']} wheels, {wheels['bytes'] / 1e6:.1f} MB ({wheels['pip']})")

    hef: Optional[dict] = None
    if skip_hef:
        out("==> 3/6 production .hef: SKIPPED (--skip-hef) -- the package is INCOMPLETE")
    else:
        out(f"==> 3/6 production .hef, verified against {str(pins.get('model_sha256'))[:16]}… before inclusion")
        model_name = Path(str(pins.get("model_file") or "models/model.hef")).name
        hef = fetch_hef(str(pins.get("model_url") or ""), str(pins.get("model_sha256") or ""),
                        package / "models" / model_name, download=download, local_file=hef_file)
        out(f"    {hef['source']}: models/{hef['path']} ({hef['bytes']:,} bytes)")

    out("==> 4/6 config template, install scripts, systemd units (from git objects)")
    config_text = _show(repo, commit, "config.yaml").decode("utf-8", "replace")
    filled = config_credential_check(config_text)
    if filled:
        raise PackageError("REFUSED: the repo config.yaml at this ref holds credential-like "
                           f"values and cannot be a template: {filled}")
    (package / "config").mkdir()
    (package / "config" / "config.yaml.example").write_text(config_text)
    for source, name in INSTALL_COPIES:
        target = package / "install" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_show(repo, commit, source))
        if source.endswith((".sh", ".py")):
            target.chmod(0o755)

    out("==> 5/6 generated INSTALL.md, MODELS.md, PACKAGE.json; secrets refusal check")
    (package / "INSTALL.md").write_text(install_md(made, pins, wheels, hef))
    (package / "MODELS.md").write_text(models_md(pins, registry, hef))
    versions = {"python": sys.version.split()[0], "pip": (wheels or {}).get("pip"),
                "builder": "scripts/release/build_transfer_package.py"}
    complete = not skip_wheels and not skip_hef
    record = {
        "format": "sensordyme-transfer-package/1",
        "ref": ref, "commit": commit, "release": rid,
        "commit_time_utc": rm._git(repo, "show", "-s", "--format=%cI", commit),
        "bundle": {"tarball_sha256": made["tarball_sha256"], "manifest_sha256": made["manifest_sha256"],
                   "files": made["files"]},
        "wheels": wheels or {"skipped": True},
        "hef": hef or {"skipped": True},
        "complete": complete,
        "versions": versions,
    }
    (package / "PACKAGE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    bad = forbidden_files(package)
    if bad:
        raise PackageError(f"REFUSED: the package holds files that never ship: {bad}")

    out("==> 6/6 SHA256SUMS and final verification")
    _, count, total = write_sha256sums(package)
    problems = verify_sha256sums(package)
    if problems:
        raise PackageError(f"package self-verification failed: {problems}")
    sums_sha = _sha256_file(package / "SHA256SUMS")
    mark = "OK" if complete else "INCOMPLETE (skip flags used; never carry this to a device)"
    out(f"TRANSFER PACKAGE {mark}: {package} — release {rid}, {count} files, "
        f"{total / 1e6:.1f} MB, SHA256SUMS sha256 {sums_sha}")
    return {**record, "package": str(package), "files": count, "bytes": total,
            "sha256sums_sha256": sums_sha}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Raspberry Pi transfer package for a git ref")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--out-root", default=str(Path.home() / "sensordyme-release"))
    parser.add_argument("--python", default=sys.executable, help="interpreter whose pip downloads the wheels")
    parser.add_argument("--skip-wheels", action="store_true", help="INCOMPLETE package (tests/dry runs only)")
    parser.add_argument("--skip-hef", action="store_true", help="INCOMPLETE package (tests/dry runs only)")
    parser.add_argument("--hef", default=None, help="an already-downloaded .hef; verified against the pin")
    parser.add_argument("--force", action="store_true", help="rebuild over an existing package directory")
    args = parser.parse_args(argv)
    if args.hef and args.skip_hef:
        parser.error("--hef and --skip-hef are mutually exclusive")
    try:
        build_package(Path(args.repo), args.ref, Path(args.out_root), python=args.python,
                      skip_wheels=args.skip_wheels, skip_hef=args.skip_hef,
                      hef_file=Path(args.hef) if args.hef else None, force=args.force)
    except PackageError as exc:
        print(f"build_transfer_package: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"build_transfer_package: {exc} (is --ref a commit in --repo?)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
