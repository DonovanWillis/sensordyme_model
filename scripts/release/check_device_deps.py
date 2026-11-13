"""Check requirements-device.txt against what the device runtime actually imports.

    python3 scripts/release/check_device_deps.py [--json]

The device runtime is main.py, scripts/preflight.py and the core/, services/,
modules/ packages that main.py loads. Every third-party top-level import in it
must be satisfied by one of:
  * a pin in requirements-device.txt (the pip-installed set),
  * an apt package installed by setup_pi.sh (APT_PROVIDED: picamera2, hailo...),
  * an explicit, documented exclusion (DEV_ONLY_IMPORTS: imported only by a
    Mac/workstation code path that refuses to run on the device).
And requirements-device.txt must not carry test, development or research
tooling (FORBIDDEN_ON_DEVICE). Exit status 1 lists every violation.

M8.5 (M84-4): the hashed lock is what a device actually installs
(`pip install --require-hashes --only-binary=:all: -r requirements-device.lock`),
so it is checked too: every pin that applies on the device (environment markers
evaluated for DEVICE_ENV: Linux, aarch64, CPython 3.11) is in the lock at the
same version with a hash; every device import a pin covers is in the lock; a pin
whose marker keeps it off the device is NOT in the lock; nothing forbidden is.

Static (ast) and offline: nothing is imported or installed. Lazy imports inside
functions count, on purpose: a lazy import is still code the device can reach.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEVICE_REQUIREMENTS = REPO / "requirements-device.txt"
DEVICE_LOCK = REPO / "requirements-device.lock"

#: The one environment a device install targets (Pi OS Bookworm on a Pi 5). Markers
#: in requirements-device.txt are evaluated against it; the lock is compiled for it.
DEVICE_ENV = {"sys_platform": "linux", "platform_machine": "aarch64", "python_version": "3.11",
              "platform_system": "Linux", "os_name": "posix", "implementation_name": "cpython"}
DEVICE_ROOTS = ("main.py", "scripts/preflight.py", "core", "services", "modules")
FIRST_PARTY = {"core", "services", "modules", "hardware", "sim", "cloud", "research", "evaluation",
               "scripts", "benchmarks", "jupiter", "tests", "main", "third_party", "typings"}

# Vendored third-party code the device imports: import name -> its package dir.
# modules/driver_attention_dashsentinel.py puts third_party/DashSentinel on
# sys.path and imports its `src` package; its own imports are device imports too.
VENDORED = {"src": "third_party/DashSentinel/src"}

# import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "onnxruntime": "onnxruntime",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
}

# Installed by setup_pi.sh through apt into the system site-packages that the
# --system-site-packages venv sees; not pip-installable on the Pi.
APT_PROVIDED = {
    "picamera2": "python3-picamera2",
    "libcamera": "python3-libcamera",
    "hailo_platform": "hailo-all",
    "gpiozero": "python3-gpiozero",
    "dbus": "python3-dbus",
    "gi": "python3-gi",
    "systemd": "python3-systemd",
}

# Third-party imports that only Mac/workstation code paths reach. Each one names
# the guard that keeps it off the device; tests/test_device_dependencies.py
# checks that the importing files are the ones listed here.
DEV_ONLY_IMPORTS = {
    "torch": "modules/driver_attention_max.py: mac_max profile, require_mps() refuses non-Apple hardware",
    "ultralytics": "AGPL, developer-only (docs/LICENSING.md); core.detectors raises without an accelerator",
    "anthropic": "core/vlm_judge.py provider 'anthropic': the mac_max VLM judge, imported lazily",
    "mlx": "mac_max on Apple silicon only",
    "mlx_vlm": "mac_max on Apple silicon only",
    "coremltools": "Mac-only export path",
}

# Never on a device we convey: test runners, linters, notebooks, training stacks.
FORBIDDEN_ON_DEVICE = {
    "pytest", "pytest-cov", "pytest-timeout", "coverage", "ruff", "pyright", "mypy", "pip-audit",
    "mutmut", "pre-commit", "ipython", "jupyter", "notebook", "anthropic", "ultralytics", "torch",
    "torchvision", "tensorboard", "timm", "thop", "pycocotools", "fiftyone",
}

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)\s*(?:;([^#]*))?")
_MARKER_TERM = re.compile(r"""^\s*([a-z_]+)\s*(==|!=)\s*["']([^"']*)["']\s*$""")
_LOCKED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\]+)", re.MULTILINE)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def marker_applies(marker: str | None, env: dict[str, str] | None = None) -> bool:
    """Whether a requirement marker holds in `env` (default DEVICE_ENV).

    Deliberately tiny: `name ==|!= "value"` terms joined by `and`, which is all
    requirements-device.txt uses. Anything else raises, so a marker this cannot
    judge fails the check instead of being guessed."""
    if not marker or not marker.strip():
        return True
    env = DEVICE_ENV if env is None else env
    result = True
    for term in re.split(r"\s+and\s+", marker.strip()):
        match = _MARKER_TERM.match(term)
        if not match or match.group(1) not in env:
            raise ValueError(f"unsupported requirement marker: {marker.strip()!r}")
        name, op, value = match.groups()
        result = result and ((env[name] == value) if op == "==" else (env[name] != value))
    return result


def requirement_markers(path: Path = DEVICE_REQUIREMENTS) -> dict[str, str | None]:
    """Pinned distribution -> its environment marker (None when unconditional)."""
    out: dict[str, str | None] = {}
    for line in path.read_text().splitlines():
        match = _REQ.match(line)
        if match:
            marker = (match.group(3) or "").strip() or None
            out[normalise(match.group(1))] = marker
    return out


def requirement_pins(path: Path = DEVICE_REQUIREMENTS) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = _REQ.match(line)
        if match:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


def device_pins(path: Path = DEVICE_REQUIREMENTS) -> dict[str, str]:
    """The pins that apply on the device (markers evaluated for DEVICE_ENV)."""
    markers = requirement_markers(path)
    return {name: version for name, version in requirement_pins(path).items() if marker_applies(markers[name])}


def lock_entries(text: str) -> dict[str, dict]:
    """Locked distribution -> {version, hashes} from a `--generate-hashes` lock."""
    out: dict[str, dict] = {}
    for block in re.split(r"\n(?=[A-Za-z0-9])", text):
        match = _LOCKED.match(block)
        if match:
            out[normalise(match.group(1))] = {"version": match.group(2),
                                              "hashes": re.findall(r"--hash=sha256:([0-9a-f]{64})", block)}
    return out


def check_lock(requirements: Path = DEVICE_REQUIREMENTS, lock: Path = DEVICE_LOCK,
               covered: dict[str, str] | None = None) -> dict:
    """Is the hashed lock complete and consistent for what the device installs and imports?"""
    if not lock.exists():
        return {"ok": False, "problems": [f"{lock.name} is missing"]}
    entries = lock_entries(lock.read_text())
    markers = requirement_markers(requirements)
    pins = requirement_pins(requirements)
    problems: list[str] = []
    for name, version in sorted(pins.items()):
        try:
            on_device = marker_applies(markers[name])
        except ValueError as exc:
            problems.append(f"{name}: {exc}")
            continue
        entry = entries.get(name)
        if on_device and entry is None:
            problems.append(f"{name}=={version} is a device pin but is not in the lock")
        elif on_device and entry is not None and entry["version"] != version:
            problems.append(f"{name}: lock {entry['version']} != pin {version}")
        elif not on_device and entry is not None:
            problems.append(f"{name} is locked but its marker ({markers[name]}) keeps it off the device")
    for name, entry in sorted(entries.items()):
        if not entry["hashes"]:
            problems.append(f"{name} has no --hash in the lock")
        if name in FORBIDDEN_ON_DEVICE:
            problems.append(f"{name} is forbidden on the device but is in the lock")
    for top, dist in sorted((covered or {}).items()):
        if dist not in entries and marker_applies(markers.get(dist)):
            problems.append(f"device import {top} ({dist}) is not in the lock")
    return {"ok": not problems, "problems": problems, "packages": len(entries)}


def device_files(repo: Path = REPO) -> list[Path]:
    files: list[Path] = []
    for root in DEVICE_ROOTS:
        path = repo / root
        files.extend([path] if path.is_file() else sorted(path.rglob("*.py")))
    return files


def _imported_names(file: Path) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(("." * node.level) + node.module)
    return names


def third_party_imports(files: list[Path], repo: Path = REPO) -> dict[str, set[str]]:
    """Top-level third-party import -> the files importing it. Vendored packages
    (VENDORED) are followed into the files the device actually reaches."""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    queue, seen = list(files), set()
    while queue:
        file = queue.pop()
        if file in seen:
            continue
        seen.add(file)
        vendored_root = next((repo / root for root in VENDORED.values() if file.is_relative_to(repo / root)), None)
        for name in _imported_names(file):
            if name.startswith("."):  # relative import inside a vendored package
                if vendored_root is not None:
                    queue.extend(_vendored_files(vendored_root, name.lstrip("."), file.parent))
                continue
            top = name.split(".")[0]
            if top in VENDORED:
                queue.extend(_vendored_files(repo / VENDORED[top], name.partition(".")[2]))
            elif top not in stdlib and top not in FIRST_PARTY and top != "__future__":
                found.setdefault(top, set()).add(str(file.relative_to(repo)))
    return found


def _vendored_files(root: Path, dotted: str, base: Path | None = None) -> list[Path]:
    target = (base or root).joinpath(*dotted.split(".")) if dotted else (base or root)
    candidates = [target.with_suffix(".py"), target / "__init__.py"]
    return [path for path in candidates if path.is_file()]


def check(repo: Path = REPO, requirements: Path = DEVICE_REQUIREMENTS) -> dict:
    pins = requirement_pins(requirements)
    imports = third_party_imports(device_files(repo), repo)
    covered, apt, dev_only, missing = {}, {}, {}, {}
    for top, files in sorted(imports.items()):
        dist = normalise(IMPORT_TO_DIST.get(top, top))
        if dist in pins:
            covered[top] = dist
        elif top in APT_PROVIDED:
            apt[top] = APT_PROVIDED[top]
        elif top in DEV_ONLY_IMPORTS:
            dev_only[top] = sorted(files)
        else:
            missing[top] = sorted(files)
    forbidden = sorted(dist for dist in pins if dist in FORBIDDEN_ON_DEVICE)
    lock = check_lock(requirements, requirements.with_name(DEVICE_LOCK.name), covered)
    return {"pins": pins, "covered": covered, "apt": apt, "dev_only": dev_only,
            "missing": missing, "forbidden": forbidden, "lock": lock,
            "ok": not missing and not forbidden and lock["ok"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = check()
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        print(f"device pins: {len(result['pins'])}; imports covered by pins: {len(result['covered'])}, "
              f"by apt: {len(result['apt'])}, dev-only: {sorted(result['dev_only'])}")
        for top, files in result["missing"].items():
            print(f"MISSING  {top}: imported by {', '.join(files[:4])} but not in requirements-device.txt")
        for dist in result["forbidden"]:
            print(f"FORBIDDEN  {dist} is pinned in requirements-device.txt")
        lock = result["lock"]
        print(f"device lock: {lock.get('packages', 0)} hashed packages; "
              f"{'complete and consistent' if lock['ok'] else 'PROBLEMS'}")
        for problem in lock["problems"]:
            print(f"LOCK  {problem}")
        print("device dependencies OK" if result["ok"] else "device dependencies FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
