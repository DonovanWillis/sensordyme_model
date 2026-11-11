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

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_pins(path: Path = DEVICE_REQUIREMENTS) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = _REQ.match(line)
        if match:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


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
    return {"pins": pins, "covered": covered, "apt": apt, "dev_only": dev_only,
            "missing": missing, "forbidden": forbidden, "ok": not missing and not forbidden}


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
        print("device dependencies OK" if result["ok"] else "device dependencies FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
