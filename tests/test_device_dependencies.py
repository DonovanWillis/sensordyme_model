"""4E: the device runtime's dependencies are exactly what it imports, and nothing for tests or research."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from scripts.release import check_device_deps as deps

REPO = Path(__file__).resolve().parent.parent

# Where each developer-only import is allowed to appear in device-tree code. Each of
# these files is reached only from a Mac/workstation path (see DEV_ONLY_IMPORTS).
DEV_ONLY_FILES = {
    "anthropic": {"core/vlm_judge.py"},
    "torch": {"modules/driver_attention_max.py"},
    "ultralytics": {"core/detectors.py"},
}


def test_every_device_import_is_pinned_or_apt_provided_and_nothing_dev_is_pinned():
    result = deps.check()
    assert result["missing"] == {}, f"device code imports packages requirements-device.txt lacks: {result['missing']}"
    assert result["forbidden"] == [], f"test/dev/research packages on the device: {result['forbidden']}"
    assert result["ok"]


def test_pytest_and_anthropic_left_the_device_but_ci_still_installs_them():
    device = deps.requirement_pins(REPO / "requirements-device.txt")
    assert "pytest" not in device and "anthropic" not in device
    ci = (REPO / "requirements.txt").read_text()
    assert "-r requirements-device.txt" in ci
    assert re.search(r"^pytest==", ci, re.M) and re.search(r"^anthropic==", ci, re.M)


def test_developer_only_imports_stay_in_their_mac_only_files():
    found = deps.third_party_imports(deps.device_files())
    for top, allowed in DEV_ONLY_FILES.items():
        assert found.get(top, set()) <= allowed, (top, found.get(top))
    assert set(deps.DEV_ONLY_IMPORTS) >= set(DEV_ONLY_FILES)


def test_the_vlm_judge_is_reached_only_from_the_mac_max_profile():
    importers = set()
    for file in deps.device_files():
        for node in ast.walk(ast.parse(file.read_text())):
            from_import = isinstance(node, ast.ImportFrom) and node.module == "core.vlm_judge"
            plain_import = isinstance(node, ast.Import) and any(a.name == "core.vlm_judge" for a in node.names)
            if from_import or plain_import:
                importers.add(str(file.relative_to(REPO)))
    assert importers == {"modules/driver_attention_max.py"}
    assert "require_mps" in (REPO / "modules/driver_attention_max.py").read_text()


def test_the_vendored_dashsentinel_package_is_followed_not_ignored():
    found = deps.third_party_imports(deps.device_files())
    assert "src" not in found, "the vendored package must be resolved, not reported as a dependency"
    vendored = {f for files in found.values() for f in files if f.startswith("third_party/DashSentinel/")}
    assert vendored, "imports inside the vendored package are device imports too"


def test_a_planted_violation_is_caught(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "x.py").write_text("import yaml\nimport leftpad\n")
    req = tmp_path / "requirements-device.txt"
    req.write_text("pyyaml==6.0.2\npytest==8.3.3\n")
    found = deps.third_party_imports([tmp_path / "core" / "x.py"], tmp_path)
    assert set(found) == {"yaml", "leftpad"}
    pins = deps.requirement_pins(req)
    assert pins == {"pyyaml": "6.0.2", "pytest": "8.3.3"}
    assert sorted(d for d in pins if d in deps.FORBIDDEN_ON_DEVICE) == ["pytest"]


def test_the_lock_matches_the_pins_and_hashes_everything():
    lock = (REPO / "requirements-device.lock").read_text()
    locked = dict(re.findall(r"^([a-z0-9][a-z0-9._-]*)==([^\s\\]+)", lock, re.M))
    for name, version in deps.requirement_pins(REPO / "requirements-device.txt").items():
        assert locked.get(name) == version, f"{name}: lock {locked.get(name)} != pin {version}"
    blocks = re.split(r"\n(?=[a-z0-9])", lock.split("\n\n", 1)[-1] if lock.startswith("#") else lock)
    for block in blocks:
        if re.match(r"^[a-z0-9]", block):
            assert "--hash=sha256:" in block, block.splitlines()[0]
    assert not set(locked) & deps.FORBIDDEN_ON_DEVICE
