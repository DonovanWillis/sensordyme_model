"""M8.5 PACKAGE: the transfer package is reproducible, verified, and never carries a secret.

scripts/release/build_transfer_package.py against this repo's HEAD, with the
network edges faked: the wheel download is skipped (its command is the one the
m85 wheelhouse evidence validated) and the .hef "download" writes known bytes,
so the refusal of a wrong hash is tested without touching the network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.release import build_transfer_package as btp
from scripts.release import make_release_manifest as rm

REPO = Path(__file__).resolve().parent.parent
PRODUCTION_HEF_SHA256 = "e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"

pytestmark = pytest.mark.skipif(not shutil.which("git") or not (REPO / ".git").exists(),
                                reason="needs a git checkout")

FAKE_HEF = b"FAKE-HEF-BYTES-FOR-THE-PACKAGE-TEST\n"
FAKE_HEF_SHA = hashlib.sha256(FAKE_HEF).hexdigest()


def fake_download(url: str, dest: Path) -> None:  # the url is recorded, not fetched
    dest.write_bytes(FAKE_HEF)


def fake_pins(setup_text: str) -> dict:
    """The real parser's shape, with a pin that matches the fake download."""
    pins = rm.setup_model_pins(setup_text)
    return {**pins, "model_url": "https://fake.invalid/model.hef", "model_sha256": FAKE_HEF_SHA}


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> dict:
    """One package build for the module: real bundle, fake .hef, no wheels."""
    out_root = tmp_path_factory.mktemp("pkg")
    real = btp._model_pins
    btp._model_pins = fake_pins
    try:
        made = btp.build_package(REPO, "HEAD", out_root, skip_wheels=True,
                                 download=fake_download, out=lambda _line: None)
    finally:
        btp._model_pins = real
    return made


# ---------------------------------------------------------------------- the .hef pin
def test_fetch_hef_accepts_only_the_pinned_bytes(tmp_path):
    dest = tmp_path / "models" / "m.hef"
    made = btp.fetch_hef("https://fake.invalid/m.hef", FAKE_HEF_SHA, dest, download=fake_download)
    assert dest.read_bytes() == FAKE_HEF and made["sha256"] == FAKE_HEF_SHA


@pytest.mark.parametrize("expected", ["0" * 64, PRODUCTION_HEF_SHA256])
def test_fetch_hef_refuses_a_wrong_hash_before_inclusion(tmp_path, expected):
    dest = tmp_path / "models" / "m.hef"
    with pytest.raises(btp.PackageError, match="REFUSED"):
        btp.fetch_hef("https://fake.invalid/m.hef", expected, dest, download=fake_download)
    assert not dest.exists() and not dest.with_suffix(".hef.part").exists(), \
        "a model that does not match the pin never enters the package"


def test_a_build_with_a_wrong_hef_download_leaves_no_package(tmp_path):
    # The REAL pin from setup_pi.sh at HEAD; the fake download serves other bytes.
    with pytest.raises(btp.PackageError, match="REFUSED"):
        btp.build_package(REPO, "HEAD", tmp_path, skip_wheels=True,
                          download=fake_download, out=lambda _line: None)
    assert list(tmp_path.iterdir()) == [], "a refused build removes the half-built package"


def test_the_real_pin_and_url_come_from_setup_pi_sh():
    pins = rm.setup_model_pins((REPO / "setup_pi.sh").read_text())
    assert pins["model_sha256"] == PRODUCTION_HEF_SHA256
    assert str(pins["model_url"]).startswith("https://hailo-model-zoo")


def test_fetch_hef_refuses_an_absent_pin(tmp_path):
    with pytest.raises(btp.PackageError, match="pin"):
        btp.fetch_hef("https://fake.invalid/m.hef", "", tmp_path / "m.hef", download=fake_download)


# ---------------------------------------------------------------------- secrets never ship
@pytest.mark.parametrize("name", [".secrets", "device.pem", "device.key", ".env", ".env.prod",
                                  "config.yaml", "factory.json", "device_state.json",
                                  "events.db", ".jwt_secret"])
def test_forbidden_files_flags_every_secret_and_state_name(tmp_path, name):
    (tmp_path / "ok.txt").write_text("fine")
    target = tmp_path / "sub" / name
    target.parent.mkdir()
    target.write_text("boom")
    flagged = btp.forbidden_files(tmp_path)
    assert flagged and name in flagged[0]
    target.unlink()
    assert btp.forbidden_files(tmp_path) == []


def test_forbidden_files_flags_symlinks(tmp_path):
    (tmp_path / "link.txt").symlink_to("/etc/hosts")
    assert "symlink" in btp.forbidden_files(tmp_path)[0]


def test_config_credential_check_refuses_filled_credentials():
    assert btp.config_credential_check("cloud:\n  device_key: dk_real_123\n")
    assert btp.config_credential_check("dashboard:\n  token: abc\n")
    assert btp.config_credential_check("wifi:\n  psk: hunter2\n")
    # The repo default is a template: empty key, null token, *_env pointers.
    assert btp.config_credential_check((REPO / "config.yaml").read_text()) == []
    assert btp.config_credential_check("a:\n  api_key_env: OPENAI_API_KEY\n  max_tokens: 64\n") == []


def test_the_built_package_holds_no_forbidden_file(package):
    assert btp.forbidden_files(Path(package["package"])) == []
    assert not (Path(package["package"]) / "config" / "config.yaml").exists()
    assert (Path(package["package"]) / "config" / "config.yaml.example").exists()


# ---------------------------------------------------------------------- hashes and the manifest
def test_sha256sums_cover_every_file_and_verify(package):
    root = Path(package["package"])
    assert btp.verify_sha256sums(root) == []
    listed = {line.split("  ", 1)[1] for line in (root / "SHA256SUMS").read_text().splitlines()}
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"SHA256SUMS"}
    assert listed == on_disk


def test_a_tampered_package_file_fails_verification(package, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(package["package"], root)
    target = root / "INSTALL.md"
    target.write_text(target.read_text() + "tampered\n")
    problems = btp.verify_sha256sums(root)
    assert problems and "INSTALL.md" in problems[0]
    (root / "planted.key").write_text("x")
    assert any("planted.key" in p for p in btp.verify_sha256sums(root))


def test_bundle_hashes_agree_everywhere(package):
    root = Path(package["package"])
    record = json.loads((root / "PACKAGE.json").read_text())
    tarball = root / "bundle" / f"sensordyme-{record['release']}.tar.gz"
    sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert sha == record["bundle"]["tarball_sha256"]
    assert f"{sha}  bundle/{tarball.name}" in (root / "SHA256SUMS").read_text()
    assert f"{sha}  {tarball.name}" in (root / "bundle" / "SHA256SUMS").read_text()
    manifest = json.loads((root / "bundle" / f"sensordyme-{record['release']}.manifest.json").read_text())
    assert manifest["commit"] == record["commit"]
    # The packaged fake .hef is in SHA256SUMS at the (test) pin it was verified against.
    assert f"{FAKE_HEF_SHA}  models/damoyolo_tinynasL20_T.hef" in (root / "SHA256SUMS").read_text()


def test_an_existing_package_dir_is_refused_without_force(package):
    with pytest.raises(btp.PackageError, match="--force"):
        btp.build_package(REPO, "HEAD", Path(package["package"]).parent,
                          skip_wheels=True, skip_hef=True, out=lambda _line: None)


# ---------------------------------------------------------------------- generated docs
def test_install_md_pins_this_release_and_names_real_entrypoints(package):
    root = Path(package["package"])
    text = (root / "INSTALL.md").read_text()
    record = json.loads((root / "PACKAGE.json").read_text())
    deploy_set = json.loads(
        (root / "bundle" / f"sensordyme-{record['release']}.manifest.json").read_text())["deploy_set"]["files"]
    # The entrypoints INSTALL.md names exist in the release it ships.
    for entry in ("setup_pi.sh", "scripts/release/release_guard.py", "scripts/release/install_units.sh",
                  "scripts/hw_validation/sunday_preflight.py", "docs/DEPLOYMENT.md"):
        assert entry in text.replace("\\\n", "") or entry.split("/")[-1] in text
        assert entry in deploy_set, f"{entry} is named but not in the release"
    assert record["bundle"]["manifest_sha256"] in text, "the stage command carries the real manifest sha256"
    assert record["commit"] in text
    for needed in ("--find-links", "switch", "--to previous", "Bookworm", "Python 3.11", "HailoRT 4.x",
                   "SHA256SUMS", "adopt"):
        assert needed in text, needed


def test_models_md_separates_production_from_the_experimental_export(package):
    text = (Path(package["package"]) / "MODELS.md").read_text()
    assert "NOT COMPILED, NOT PRODUCTION" in text
    assert "1,024 NoIR" in text
    assert "scripts/hw_validation/known_good" in text
    assert "Dataflow Compiler" in text
    assert btp.EXPERIMENTAL_EXPORT["onnx_sha256"] in text
    assert btp.EXPERIMENTAL_EXPORT["heads81_sha256"] in text
    assert "verify-artifacts" in text


def test_models_md_on_the_real_pins_names_the_production_hash():
    pins = rm.setup_model_pins((REPO / "setup_pi.sh").read_text())
    registry = rm.registry_artifacts((REPO / "core" / "model_registry.yaml").read_text())
    text = btp.models_md(pins, registry, None)
    assert PRODUCTION_HEF_SHA256 in text
    assert "e06a804e0144f9929eda782122916b35d60c697c3c9344013ca2bbe76a6ce2b4" in text, \
        "the MediaPipe face-landmark pin is listed"


# ---------------------------------------------------------------------- the one command
def test_makefile_has_the_transfer_package_target():
    text = (REPO / "Makefile").read_text()
    assert "transfer-package:" in text
    assert "build_transfer_package.py" in text


def test_the_disposable_install_check_passes_against_the_packaged_bundle(package, tmp_path):
    """Checkpoint J, in-process: the REAL guard over the REAL packaged bundle."""
    root = Path(package["package"])
    record = json.loads((root / "PACKAGE.json").read_text())
    import sys
    script = REPO / "scripts" / "release" / "transfer_install_check.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 2, "argparse: --bundle/--manifest are required"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(script),
         "--bundle", str(root / "bundle" / f"sensordyme-{record['release']}.tar.gz"),
         "--manifest", str(root / "bundle" / f"sensordyme-{record['release']}.manifest.json"),
         "--out", str(tmp_path / "report")],
        capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "report" / "install_verification.json").read_text())
    assert report["verdict"] == "PASS"
    assert {s["step"] for s in report["steps"]} == {"J1", "J2", "J3", "J4", "J5", "J6"}
    assert all(s["result"] == "PASS" for s in report["steps"])
    assert "fake" in report["honesty"] and "systemd" in report["honesty"]["fake"]
