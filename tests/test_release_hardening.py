"""M6.5 release hardening: protocol compatibility, the device lock, status tiers and the
release gates recorded in the manifest (scripts/release/make_release_manifest.py)."""

from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.release import make_release_manifest as rm

REPO = Path(__file__).resolve().parent.parent
PIN = "a" * 64
GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"]
H = "    --hash=sha256:" + "0" * 64

FILES = {
    "config.yaml": f"role: driver_safety\naccelerator:\n  yolo_hef: models/m.hef\n  yolo_hef_sha256: {PIN}\n",
    "setup_pi.sh": f'MODEL_DIR="models"\nMODEL_FILE="$MODEL_DIR/m.hef"\nMODEL_SHA256="{PIN}"\n',
    "core/model_registry.yaml": f'models:\n  - id: m\n    artifacts:\n      - path: models/m.hef\n        sha256: "{PIN}"\n',
    "requirements-device.txt": "pyyaml==6.0.2\nrequests==2.34.2\n",
    "requirements-device.lock": f"# header\npyyaml==6.0.2 \\\n{H}\nrequests==2.34.2 \\\n{H}\nidna==3.20 \\\n{H}\n",
    "services/runtime.py": '_VERSION = "0.2"\n',
    "core/uplink.py": 'URL = f"{base}/v1/ingest"\n',
    "cloud/app.py": ('app = FastAPI(\n    title="x",\n    version="0.1",\n)\n'
                     '@app.post("/v1/ingest")\n@app.get("/v1/devices/{device_id}/rules")\n'
                     '@app.patch("/v1/devices/{device_id}/rules/{rule_id}")\n@app.get("/v1/health")\n'),
    "ios/project.yml": 'settings:\n  MARKETING_VERSION: "1.0"\n',
    "ios/Sources/Api.swift": ('let a = "v1/devices/\\(deviceID)/rules"\nlet b = "v1/devices/\\(deviceID)/rules/\\(ruleID)"\n'
                              'let c = "v1/health"\n'),
    "scripts/release/release_status.yaml": (
        "format: sensordyme-release-status/1\nas_of: '2026-11-11'\ncomponents:\n"
        "  a:\n    tiers: [simulator-verified, hardware-pending]\n    evidence: sim e2e\n"
        "  b:\n    tiers: [experiment-only]\n    pending_integration: true\n    evidence: PLACEHOLDER\n"),
}


def _repo(tmp_path, **overrides):
    root = tmp_path / "repo"
    for rel, text in {**FILES, **overrides}.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q", "-b", "rc", str(root)], check=True)  # noqa: S603, S607
    subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)  # noqa: S603
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "-m", "rc"], check=True)  # noqa: S603
    return root


def _result(manifest, prefix):
    return next(g["result"] for g in rm.release_gates(manifest) if g["gate"].startswith(prefix))


@pytest.mark.parametrize(("raw", "key"), [
    ("/v1/devices/{device_id}/rules/{rule_id}", "v1/devices/{}/rules/{}"),
    ("v1/devices/\\(deviceID)/rules/\\(ruleID)", "v1/devices/{}/rules/{}"),
    ("/v1/health?x=1", "v1/health"), ("v1/devices/", "v1/devices"),
])
def test_route_keys_normalise_backend_and_swift_paths(raw, key):
    assert rm.route_key(raw) == key


def test_a_compatible_release_records_versions_lock_and_tiers(tmp_path):
    manifest = rm.build(_repo(tmp_path), "rc", [])
    proto = manifest["protocol"]
    assert proto["versions"] == {"device_runtime": "0.2", "backend_api": "0.1", "ios_app": "1.0", "api_prefix": "/v1"}
    assert proto["compatible"] is True and proto["device_routes"] == ["v1/ingest"] and proto["app_routes"] == 3
    assert manifest["dependency_lock"]["consistent"] is True and manifest["dependency_lock"]["packages"] == 3
    assert manifest["status_tiers"]["problems"] == []
    assert manifest["status_tiers"]["components"]["b"]["pending_integration"] is True
    assert manifest["known_good"]["yolo_decoder"] == "legacy" and manifest["known_good"]["yolo_hef_sha256"] == PIN
    for prefix in ("model pins", "decoder", "protocol", "device lock", "status tiers"):
        assert _result(manifest, prefix) == "PASS", prefix
    md = rm.to_markdown(manifest)
    assert "## Status tiers" in md and "*(pending integration)*" in md and "## Release gates" in md


def test_an_app_route_the_backend_lacks_fails_the_protocol_gate(tmp_path):
    swift = FILES["ios/Sources/Api.swift"] + 'let d = "v1/devices/\\(deviceID)/firmware"\n'
    manifest = rm.build(_repo(tmp_path, **{"ios/Sources/Api.swift": swift}), "rc", [])
    assert manifest["protocol"]["app_missing_in_backend"] == ["v1/devices/{}/firmware"]
    assert _result(manifest, "protocol") == "FAIL"


def test_a_lock_that_disagrees_with_the_pins_or_lacks_a_hash_fails(tmp_path):
    lock = FILES["requirements-device.lock"].replace("requests==2.34.2", "requests==2.32.0")
    lock += "chardet==7.7.0\n"
    manifest = rm.build(_repo(tmp_path, **{"requirements-device.lock": lock}), "rc", [])
    assert manifest["dependency_lock"]["pins_mismatched"] == {"requests": ["2.34.2", "2.32.0"]}
    assert manifest["dependency_lock"]["unhashed"] == ["chardet"]
    assert _result(manifest, "device lock") == "FAIL"


def test_an_older_ref_without_the_new_inputs_is_info_not_pass(tmp_path):
    manifest = rm.build(_repo(tmp_path, **{"requirements-device.lock": None,
                                            "scripts/release/release_status.yaml": None,
                                            "cloud/app.py": None}), "rc", [])
    assert _result(manifest, "device lock") == "INFO"
    assert _result(manifest, "status tiers") == "INFO"
    assert _result(manifest, "protocol") == "INFO"


def test_the_reference_decoder_is_never_a_known_good_release(tmp_path):
    config = FILES["config.yaml"] + "  yolo_decoder: reference\n"
    manifest = rm.build(_repo(tmp_path, **{"config.yaml": config}), "rc", [])
    assert _result(manifest, "decoder") == "FAIL"


@pytest.mark.parametrize(("entry", "problem"), [
    ("tiers: [production-ready]\n    evidence: x", "not within"),
    ("tiers: []\n    evidence: x", "not within"),
    ("tiers: [software-ready]\n    evidence: ''", "no evidence"),
    ("tiers: [software-ready]\n    evidence: hardware-validated on the bench", "claims hardware validation"),
])
def test_status_tier_problems_fail_the_gate(tmp_path, entry, problem):
    status = f"format: x\ncomponents:\n  c:\n    {entry}\n"
    manifest = rm.build(_repo(tmp_path, **{"scripts/release/release_status.yaml": status}), "rc", [])
    assert any(problem in p for p in manifest["status_tiers"]["problems"])
    assert _result(manifest, "status tiers") == "FAIL"


def test_a_config_the_schema_refuses_fails_its_gate(tmp_path):
    """The synthetic config.yaml is not a full driver_safety config: the schema gate must say so."""
    manifest = rm.build(_repo(tmp_path), "rc", [])
    assert manifest["config_schema"]["checked"] and manifest["config_schema"]["errors"]
    assert _result(manifest, "repo config.yaml") == "FAIL"


def test_check_exits_nonzero_on_a_failed_gate(tmp_path, capsys):
    manifest = rm.build(_repo(tmp_path), "rc", [])
    manifest["config_schema"] = {"checked": True, "errors": [], "warnings": []}  # see the test above
    bad = copy.deepcopy(manifest)
    bad["decoder"]["yolo_decoder"] = "reference"
    for name, data, code in (("ok.json", manifest, 0), ("bad.json", bad, 1)):
        path = tmp_path / name
        path.write_text(__import__("json").dumps(data))
        assert rm.main(["check", "--manifest", str(path)]) == code
    assert "GATES: FAIL" in capsys.readouterr().out


# ----------------------------------------------------------------- this repository
def test_the_repository_status_file_is_valid_and_claims_no_hardware_validation():
    status = yaml.safe_load((REPO / "scripts/release/release_status.yaml").read_text())
    for name, entry in status["components"].items():
        assert entry["tiers"] and set(entry["tiers"]) <= set(rm.TIERS), name
        assert entry["evidence"], name
    pending = {n for n, e in status["components"].items() if e.get("pending_integration")}
    assert {"m65_device_auth_signed_commands", "m65_ble_proof_of_possession", "m65_xcuitest_journeys",
            "m65_reliability_fault_scenarios", "m65_model_registry_evidence_binding",
            "m65_training_run_exp08_full60"} <= pending
    assert status["components"]["yolo_decoder_reference"]["tiers"] == ["experiment-only"]


@pytest.mark.skipif(not (REPO / ".git").exists() or not shutil.which("git"), reason="not a git checkout")
def test_this_commit_passes_its_release_gates():
    manifest = rm.build(REPO, "HEAD", [])
    failed = [g for g in rm.release_gates(manifest) if g["result"] == "FAIL"]
    assert failed == [], failed
    assert manifest["protocol"]["compatible"] is True
    assert set(rm._UNITS) <= set(manifest["systemd_units"])


def test_the_one_verification_command_is_wired():
    script = (REPO / "scripts/release/verify.sh").read_text()
    for step in ("check_device_deps.py", "make_release_manifest.py build", "make_release_manifest.py check",
                 "verify_systemd_units.sh", "tests/test_release_hardening.py"):
        assert step in script, step
    assert "verify-release:" in (REPO / "Makefile").read_text()
