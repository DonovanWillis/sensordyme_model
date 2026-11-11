"""M6 release tooling: a manifest pins exactly what a release puts on a device,
and `verify` proves (or refutes) that a device snapshot holds exactly that.

Built against a throwaway git repository so every hash is known.
"""

import io
import json
import subprocess
import tarfile

import pytest

from scripts.release import make_release_manifest as rm

PIN = "a" * 64
EVALUATED = "b" * 64
GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"]

FILES = {
    "main.py": "print('device')\n",
    "core/x.py": "X = 1\r\n",  # CRLF on purpose
    "config.yaml": (
        "role: driver_safety\nmodule: driver_attention\nprofile: pi\naccelerator:\n"
        f"  yolo_hef: models/m.hef\n  yolo_hef_sha256: {PIN}\n  yolo_decoder: legacy\n"
    ),
    "setup_pi.sh": (
        'MODEL_DIR="models"\nMODEL_FILE="$MODEL_DIR/m.hef"\nMODEL_URL="https://example.invalid/m.hef"\n'
        f'MODEL_SHA256="{PIN}"\nMODEL_SHA256_EVALUATED="{EVALUATED}"\n'
        "pip install gpiozero==2.0.1 lgpio==0.2.2.0\n"
    ),
    "core/model_registry.yaml": f'models:\n  - id: m\n    artifacts:\n      - path: models/m.hef\n        sha256: "{PIN}"\n',
    "requirements-device.txt": "# device\npyyaml==6.0.2\nrequests==2.34.2 # http\n",
    "sensordyme.service": "[Service]\nUser=__SENSORDYME_USER__\nWorkingDirectory=__SENSORDYME_DIR__\n",
    "deploy/systemd/sensordyme-recover.service": "[Service]\nType=oneshot\n",
    # never part of the deploy set:
    "models/m.hef": "binary",
    "data/events.csv": "t,fall\n",
    "ios/App.swift": "// app\n",
    "cloud/.jwt_secret": "secret\n",
    ".env": "KEY=1\n",
    "benchmarks/results/r.json": "{}\n",
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    subprocess.run(["git", "init", "-q", "-b", "rc", str(root)], check=True)  # noqa: S603, S607
    subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)  # noqa: S603
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "-m", "rc"], check=True)  # noqa: S603
    return root


@pytest.fixture
def manifest(repo):
    return rm.build(repo, "rc", ["pytest tests/test_m6_*.py: N passed"])


def _snapshot(manifest, **changes):
    snap = {
        "root": "/home/pi/sensordyme",
        "files": json.loads(json.dumps(manifest["deploy_set"]["files"])),
        "config": {"sha256": "c" * 64, "role": "driver_safety"},
        "model": {"sha256": PIN, "pin": PIN},
        "pip_freeze": "PyYAML==6.0.2\nrequests==2.34.2\ngpiozero==2.0.1\nlgpio==0.2.2.0\nnumpy==1.26.4\n",
        "deploy_preflight": {"output": json.dumps({"ok": True, "errors": [], "warnings": []})},
        "service_unit_sha256": rm._sha(FILES["sensordyme.service"].replace("__SENSORDYME_USER__", "pi")
                                       .replace("__SENSORDYME_DIR__", "/home/pi/sensordyme").encode()),
    }
    snap.update(changes)
    return snap


def _verify(manifest, snap):
    return rm.verify(manifest, snap, FILES["sensordyme.service"].encode())


def _failed(result):
    return {c["check"] for c in result["checks"] if c["result"] == "FAIL"}


# ----------------------------------------------------------------- build
def test_the_deploy_set_uses_the_make_deploy_excludes(manifest):
    files = set(manifest["deploy_set"]["files"])
    assert files == {"main.py", "core/x.py", "setup_pi.sh", "core/model_registry.yaml",
                     "requirements-device.txt", "sensordyme.service", "deploy/systemd/sensordyme-recover.service"}
    assert manifest["deploy_set"]["file_count"] == len(files)


def test_hashes_are_the_committed_bytes(manifest):
    entry = manifest["deploy_set"]["files"]["core/x.py"]
    assert entry["sha256"] == rm._sha(b"X = 1\r\n") and entry["sha256_lf"] == rm._sha(b"X = 1\n")
    assert manifest["config"]["sha256"] == rm._sha(FILES["config.yaml"].encode())
    assert manifest["deploy_set"]["aggregate_sha256"] == rm.aggregate_sha256(manifest["deploy_set"]["files"])


def test_commit_branch_decoder_requirements_and_tests_are_recorded(repo, manifest):
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,  # noqa: S603, S607
                          text=True, check=True).stdout.strip()
    assert manifest["commit"] == head and manifest["branch"] == "rc"
    assert manifest["decoder"]["yolo_decoder"] == "legacy"
    assert manifest["requirements"]["requirements-device.txt"]["pins"] == {"pyyaml": "6.0.2", "requests": "2.34.2"}
    assert manifest["requirements"]["setup_pi.sh (pip install)"]["pins"] == {"gpiozero": "2.0.1", "lgpio": "0.2.2.0"}
    assert manifest["test_runs"] == ["pytest tests/test_m6_*.py: N passed"]
    assert set(manifest["systemd_units"]) == {"sensordyme.service", "deploy/systemd/sensordyme-recover.service"}


def test_model_pins_agree(manifest):
    pins = manifest["model_pins"]
    assert pins["consistent"] is True
    assert pins["config_yolo_hef_sha256"] == PIN and pins["setup_pi"]["model_sha256"] == PIN
    assert pins["setup_pi"]["model_sha256_evaluated"] == EVALUATED
    assert pins["setup_pi"]["model_file"] == "models/m.hef"
    assert pins["registry"] == [{"model": "m", "path": "models/m.hef", "sha256": PIN, "format": None}]


def test_disagreeing_model_pins_are_flagged(repo):
    (repo / "core/model_registry.yaml").write_text(FILES["core/model_registry.yaml"].replace(PIN, "d" * 64))
    subprocess.run([*GIT, "-C", str(repo), "commit", "-qam", "drift"], check=True)  # noqa: S603
    assert rm.build(repo, "rc", [])["model_pins"]["consistent"] is False


def test_uncommitted_edits_never_enter_a_manifest(repo, manifest):
    (repo / "main.py").write_text("print('tampered')\n")
    assert rm.build(repo, "rc", [])["deploy_set"]["files"] == manifest["deploy_set"]["files"]


def test_markdown_names_commit_pins_and_tests(manifest):
    md = rm.to_markdown(manifest)
    assert manifest["commit"] in md and PIN in md and "pins consistent: YES" in md
    assert "pytest tests/test_m6_*.py" in md and "gpiozero==2.0.1" in md


def test_an_installed_tree_snapshots_to_the_manifest(repo, manifest, tmp_path):
    """The device-side hasher (integrity_manifest) and the manifest agree on the set."""
    from scripts.hw_validation import integrity_manifest as im

    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "rc"],  # noqa: S603, S607
                          capture_output=True, check=True).stdout
    install = tmp_path / "install"
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(install, filter="data")
    # Runtime state and the venv that a real device also has:
    (install / "data").mkdir(exist_ok=True)
    (install / "data" / "device_state.json").write_text("{}")
    (install / ".venv" / "bin").mkdir(parents=True)
    (install / ".venv" / "bin" / "python").write_text("")
    assert im.diff_manifests(im.tree_manifest(install), manifest["deploy_set"]["files"])["exact_match"]


# ----------------------------------------------------------------- verify
def test_an_exact_device_passes(manifest):
    result = _verify(manifest, _snapshot(manifest))
    assert result["verdict"] == "PASS", result


def test_a_modified_file_fails(manifest):
    snap = _snapshot(manifest)
    snap["files"]["main.py"] = {"sha256": "0" * 64, "sha256_lf": "0" * 64}
    result = _verify(manifest, snap)
    assert result["verdict"] == "FAIL" and _failed(result) == {"deployed files"}


def test_a_stale_extra_file_fails(manifest):
    """make deploy has no --delete: a file removed from the repo stays on the device."""
    snap = _snapshot(manifest)
    snap["files"]["modules/removed.py"] = {"sha256": "1" * 64, "sha256_lf": "1" * 64}
    assert _failed(_verify(manifest, snap)) == {"deployed files"}


def test_migrate_config_backups_are_device_state_not_drift(manifest):
    """scripts/migrate_config.py writes config.yaml.bak-<stamp> beside the code."""
    snap = _snapshot(manifest)
    snap["files"]["config.yaml.bak-20261116-101500"] = {"sha256": "2" * 64, "sha256_lf": "2" * 64}
    result = _verify(manifest, snap)
    assert result["verdict"] == "PASS"
    assert any(c["check"].startswith("device state files") for c in result["checks"])
    snap["files"]["core/config.yaml.bak-1"] = {"sha256": "3" * 64, "sha256_lf": "3" * 64}
    assert _failed(_verify(manifest, snap)) == {"deployed files"}, "only the root config backups are exempt"


def test_a_missing_file_fails(manifest):
    snap = _snapshot(manifest)
    del snap["files"]["main.py"]
    assert _failed(_verify(manifest, snap)) == {"deployed files"}


def test_a_line_ending_only_difference_still_fails(manifest):
    snap = _snapshot(manifest)
    entry = snap["files"]["core/x.py"]
    snap["files"]["core/x.py"] = {"sha256": entry["sha256_lf"], "sha256_lf": entry["sha256_lf"]}
    assert _failed(_verify(manifest, snap)) == {"deployed files"}


@pytest.mark.parametrize("model", [
    {"sha256": EVALUATED, "pin": PIN},  # the unverified "evaluated" build
    {"sha256": "", "pin": PIN},          # no model on the device
])
def test_the_wrong_or_missing_model_fails(manifest, model):
    assert "model hash = release pin" in _failed(_verify(manifest, _snapshot(manifest, model=model)))


def test_a_device_config_pinning_another_model_fails(manifest):
    snap = _snapshot(manifest, model={"sha256": PIN, "pin": ""})
    assert _failed(_verify(manifest, snap)) == {"device config pins the release model"}


def test_a_wrong_installed_version_fails(manifest):
    snap = _snapshot(manifest, pip_freeze="PyYAML==6.0.1\nrequests==2.34.2\ngpiozero==2.0.1\nlgpio==0.2.2.0\n")
    assert _failed(_verify(manifest, snap)) == {"installed requirements"}


def test_a_stale_service_unit_fails(manifest):
    snap = _snapshot(manifest, service_unit_sha256="e" * 64)
    assert _failed(_verify(manifest, snap)) == {"installed sensordyme.service"}


def test_a_failing_device_config_preflight_fails(manifest):
    snap = _snapshot(manifest, deploy_preflight={"output": json.dumps({"ok": False, "errors": ["profile"]})})
    assert _failed(_verify(manifest, snap)) == {"device config preflight"}


def test_the_cli_round_trip(repo, tmp_path):
    out_json, out_md = tmp_path / "rc.json", tmp_path / "rc.md"
    assert rm.main(["build", "--repo", str(repo), "--ref", "rc", "--out-json", str(out_json),
                    "--out-md", str(out_md), "--test-ref", "t1"]) == 0
    manifest = json.loads(out_json.read_text())
    snap_path = tmp_path / "snap.json"
    snap_path.write_text(json.dumps(_snapshot(manifest)))
    assert rm.main(["verify", "--repo", str(repo), "--manifest", str(out_json), "--snapshot", str(snap_path),
                    "--out", str(tmp_path / "v.json")]) == 0
    bad = _snapshot(manifest, model={"sha256": EVALUATED, "pin": PIN})
    snap_path.write_text(json.dumps(bad))
    assert rm.main(["verify", "--repo", str(repo), "--manifest", str(out_json), "--snapshot", str(snap_path)]) == 1
