"""M8.5 (M84-6): staged, verified, atomically switched releases with rollback by switching back.

The device side (scripts/release/release_guard.py: adopt, stage, activate,
confirm-or-rollback, switch) against temp trees, a fake systemd that answers by
WHICH release the service path points at, and a fake runner that records every
command (pip, venv, units, restart) instead of running it. Nothing here touches a
real unit, a real venv or the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.release import release_guard as rg

REPO = Path(__file__).resolve().parent.parent
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40

CODE_V1 = {"main.py": b"VERSION = 'v1'\n", "core/x.py": b"X = 1\n",
           "requirements-device.lock": b"pyyaml==6.0.2 \\\n    --hash=sha256:" + b"0" * 64 + b"\n",
           "scripts/release/install_units.sh": b"#!/bin/sh\n", "scripts/migrate_config.py": b"# migrate\n"}
CODE_V2 = {**CODE_V1, "main.py": b"VERSION = 'v2'\n", "core/new_module.py": b"Y = 2\n"}


# ---------------------------------------------------------------------- fakes
class FakeDevice:
    """systemd + every subprocess the guard runs. Health depends on the LIVE release."""

    def __init__(self, link: Path) -> None:
        self.link = link
        self.broken: set[str] = set()   # release names whose service never stays up
        self.fail: dict[str, int] = {}  # substring of a command -> exit status
        self.commands: list[list[str]] = []
        self.now = 0.0
        self.restarts = 0
        self.stopped = False

    def live(self) -> str:
        return Path(os.path.realpath(self.link)).name

    def run(self, argv: list[str]) -> int:
        self.commands.append(list(argv))
        joined = " ".join(argv)
        for needle, code in self.fail.items():
            if needle in joined:
                return code
        if argv[1:3] == ["-m", "venv"]:
            (Path(argv[-1]) / "bin").mkdir(parents=True, exist_ok=True)
            (Path(argv[-1]) / "bin" / "python").write_text("#!/bin/sh\n")
        if argv[:3] == ["sudo", "systemctl", "restart"]:
            self.restarts += 1
        return 0

    def probe(self) -> dict:
        if self.stopped:
            return {"ActiveState": "inactive", "SubState": "dead", "NRestarts": "0"}
        if self.live() in self.broken:
            return {"ActiveState": "activating", "SubState": "auto-restart", "NRestarts": str(self.restarts + int(self.now)),
                    "ExecMainStatus": "1"}
        return {"ActiveState": "active", "SubState": "running", "NRestarts": "0", "ExecMainStatus": "0"}

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def pip_calls(self) -> list[list[str]]:
        return [c for c in self.commands if c[1:4] == ["-m", "pip", "install"]]


def guard(link: Path, device: FakeDevice, out: list[str] | None = None) -> rg.Guard:
    log = out if out is not None else []
    return rg.Guard(link, run=device.run, probe=device.probe, clock=device.clock, sleep=device.sleep,
                    user="sdpi", out=log.append)


def legacy_device(tmp_path: Path) -> Path:
    """A pre-M8.5 in-place install: code, venv and device state in one directory."""
    root = tmp_path / "sensordyme"
    for rel, data in CODE_V1.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    (root / "data").mkdir()
    (root / "data" / "device_state.json").write_text('{"preset": "driver_safety"}')
    (root / "models").mkdir()
    (root / "models" / "m.hef").write_bytes(b"HEF")
    (root / ".secrets").write_text("device-secret\n")
    config = root / "config.yaml"
    config.write_text("module: driver_attention\ncloud:\n  device_key: dk_device_own_key\n")
    os.chmod(config, 0o600)
    return root


def bundle(tmp_path: Path, files: dict[str, bytes], commit: str = COMMIT_B, *,
           tar_files: dict[str, bytes] | None = None, extra_members: list[tarfile.TarInfo] | None = None,
           name: str = "b") -> tuple[Path, Path, str]:
    """A release bundle as make_release_manifest.py `bundle` writes it."""
    out = tmp_path / f"incoming-{name}"
    out.mkdir(exist_ok=True)
    manifest = {"commit": commit, "deploy_set": {"files": {
        p: {"sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()}}}
    manifest_path = out / "m.json"
    manifest_path.write_bytes(json.dumps(manifest).encode())
    tar_path = out / "r.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path, data in (tar_files if tar_files is not None else files).items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for member in extra_members or []:
            tar.addfile(member)
    return tar_path, manifest_path, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def adopted(tmp_path: Path) -> tuple[Path, FakeDevice]:
    link = legacy_device(tmp_path)
    device = FakeDevice(link)
    device.stopped = True  # the operator stops the service for the one-time adopt
    assert guard(link, device).adopt() == 0
    device.stopped = False
    return link, device


def staged(tmp_path: Path, files: dict[str, bytes] = CODE_V2, commit: str = COMMIT_B):
    link, device = adopted(tmp_path)
    tar, manifest, sha = bundle(tmp_path, files, commit)
    out: list[str] = []
    assert guard(link, device, out).stage(tar, manifest, sha) == 0, out
    return link, device, commit[:12]


def tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink()}


# ---------------------------------------------------------------------- adopt
def test_adopt_turns_the_legacy_tree_into_a_release_and_moves_state_out(tmp_path):
    link, _ = adopted(tmp_path)
    lay = rg.Layout(link)
    current = lay.current()
    assert current is not None and link.is_symlink() and current.name.startswith("legacy-")
    assert (link / "main.py").read_bytes() == CODE_V1["main.py"], "the running code is untouched"
    assert (link / ".venv" / "bin" / "python").exists(), "the legacy venv keeps working through the path"
    for name in ("data", "models", ".secrets"):
        assert (current / name).is_symlink(), name
        assert (lay.state / name).exists() and not (lay.state / name).is_symlink(), name
    assert (link / "data" / "device_state.json").read_text() == '{"preset": "driver_safety"}'
    assert (link / "config.yaml").read_text().endswith("dk_device_own_key\n"), "config.yaml stays with its release"


def test_adopt_refuses_while_the_service_runs(tmp_path):
    link = legacy_device(tmp_path)
    device = FakeDevice(link)
    out: list[str] = []
    g = rg.Guard(link, run=device.run, probe=lambda: {"ActiveState": "active"}, out=out.append)
    assert g.adopt() == 2 and not link.is_symlink()
    assert "stop it first" in out[-1]


# ---------------------------------------------------------------------- stage: verification before anything is written
def test_stage_verifies_every_file_and_builds_the_venv_from_the_hashed_lock(tmp_path):
    link, device, rid = staged(tmp_path)
    release = rg.Layout(link).release(rid)
    assert {k: v for k, v in tree(release).items() if k in CODE_V2} == CODE_V2
    record = json.loads((release / rg.STAGED_MARK).read_text())
    assert record["commit"] == COMMIT_B and record["files"] == len(CODE_V2)
    (pip,) = device.pip_calls()
    assert pip[0] == str(release / ".venv" / "bin" / "python")
    assert "--require-hashes" in pip and "--only-binary=:all:" in pip
    assert pip[pip.index("-r") + 1] == str(release / "requirements-device.lock")
    assert "requirements-device.txt" not in " ".join(pip)
    venv = next(c for c in device.commands if c[1:3] == ["-m", "venv"])
    assert "--system-site-packages" in venv, "picamera2 and hailo come from apt"
    artifacts = next(c for c in device.commands if "verify-artifacts" in " ".join(c))
    assert artifacts[0] == str(release / ".venv" / "bin" / "python"), "the release's own venv checks its model files"
    assert os.path.realpath(link) != str(release.resolve()), "staging never switches"


@pytest.mark.parametrize("case", ["tampered", "extra", "missing", "traversal", "symlink", "state", "secret"])
def test_stage_refuses_a_bundle_that_does_not_match_the_manifest(tmp_path, case):
    link, device = adopted(tmp_path)
    files = dict(CODE_V2)
    tar_files = dict(files)
    extra: list[tarfile.TarInfo] = []
    if case == "tampered":
        tar_files["core/x.py"] = b"X = 'evil'\n"
    elif case == "extra":
        tar_files["core/backdoor.py"] = b"import os\n"
    elif case == "missing":
        del tar_files["core/new_module.py"]
    elif case == "traversal":
        tar_files["../outside.py"] = b"x\n"
        files["../outside.py"] = b"x\n"
    elif case == "symlink":
        member = tarfile.TarInfo("core/link.py")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        extra.append(member)
    elif case == "state":
        tar_files["data/device_state.json"] = b"{}"
        files["data/device_state.json"] = b"{}"
    elif case == "secret":
        tar_files["config.yaml"] = b"cloud:\n  device_key: attacker\n"
        files["config.yaml"] = b"cloud:\n  device_key: attacker\n"
    tar, manifest, sha = bundle(tmp_path, files, tar_files=tar_files, extra_members=extra)
    before = tree(link.resolve())
    out: list[str] = []
    assert guard(link, device, out).stage(tar, manifest, sha) == 1, out
    assert "REFUSED" in out[-1]
    lay = rg.Layout(link)
    assert not lay.release(COMMIT_B[:12]).exists() and not list(lay.releases.glob(".partial-*"))
    assert not (tmp_path / "outside.py").exists()
    assert device.pip_calls() == [], "nothing is installed for a refused release"
    assert tree(link.resolve()) == before


def test_stage_refuses_a_manifest_that_is_not_the_one_the_operator_built(tmp_path):
    link, device = adopted(tmp_path)
    tar, manifest, sha = bundle(tmp_path, CODE_V2)
    # The attacker rewrites the manifest to match a tampered bundle.
    doctored = json.loads(manifest.read_text())
    doctored["deploy_set"]["files"]["core/x.py"]["sha256"] = hashlib.sha256(b"evil").hexdigest()
    manifest.write_text(json.dumps(doctored))
    out: list[str] = []
    assert guard(link, device, out).stage(tar, manifest, sha) == 2
    assert "not the one the operator built" in out[-1]


@pytest.mark.parametrize("step", ["--require-hashes", "verify-artifacts", "-m venv"])
def test_a_failed_install_or_model_check_removes_the_staged_release(tmp_path, step):
    link, device = adopted(tmp_path)
    device.fail[step] = 1
    tar, manifest, sha = bundle(tmp_path, CODE_V2)
    out: list[str] = []
    assert guard(link, device, out).stage(tar, manifest, sha) == 1
    assert not rg.Layout(link).release(COMMIT_B[:12]).exists()
    assert "running release is unchanged" in out[-1]


def test_stage_needs_the_staged_layout(tmp_path):
    link = legacy_device(tmp_path)
    tar, manifest, sha = bundle(tmp_path, CODE_V2)
    out: list[str] = []
    assert guard(link, FakeDevice(link), out).stage(tar, manifest, sha) == 2
    assert "adopt" in out[-1]


# ---------------------------------------------------------------------- activate: atomic switch, confirm or switch back
def test_a_healthy_release_is_switched_to_atomically_and_confirmed(tmp_path, monkeypatch):
    link, device, rid = staged(tmp_path)
    calls: list[tuple] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((Path(src).name, Path(dst).name, Path(src).is_symlink()))
        return real_replace(src, dst)

    monkeypatch.setattr(rg.os, "replace", recording_replace)
    out: list[str] = []
    assert guard(link, device, out).activate(rid, timeout_s=60, settle_s=5) == 0, out
    assert os.path.realpath(link) == str(rg.Layout(link).release(rid).resolve())
    switch = [c for c in calls if c[1] == link.name]
    assert switch == [(f".{link.name}.switch-{os.getpid()}", link.name, True)], \
        "the service path changes with exactly one rename(2) of a ready symlink"
    units = next(c for c in device.commands if c[:2] == ["sudo", "bash"])
    assert units[units.index("--app-dir") + 1] == str(link), "units always name the service path, never a release"
    assert device.restarts == 1
    confirmed = json.loads((rg.Layout(link).state / rg.STATE_DIR / "confirmed.json").read_text())
    assert confirmed["release"] == rid and confirmed["previous"].startswith("legacy-")
    assert (link / "main.py").read_bytes() == CODE_V2["main.py"]


def test_a_failing_start_is_rolled_back_by_switching_back_without_network(tmp_path):
    link, device, rid = staged(tmp_path)
    legacy = rg.Layout(link).current()
    assert legacy is not None
    before = tree(legacy)
    device.broken = {rid}
    pip_before = len(device.pip_calls())
    out: list[str] = []
    assert guard(link, device, out).activate(rid, timeout_s=60, settle_s=5) == 1
    assert Path(os.path.realpath(link)) == legacy, "the previous release is live again"
    assert tree(legacy) == before, "the previous release's files and config are byte-identical"
    assert len(device.pip_calls()) == pip_before, "rolling back installs nothing (its venv is still there)"
    assert device.restarts == 2
    rolled = json.loads((rg.Layout(link).state / rg.STATE_DIR / "rolled_back.json").read_text())
    assert rolled["release"] == rid and rolled["previous_healthy"] is True
    assert rg.Layout(link).release(rid).is_dir(), "the failed release is kept for inspection"
    assert not (rg.Layout(link).state / rg.STATE_DIR / "pending.json").exists()


def test_a_failed_unit_install_rolls_back_too(tmp_path):
    link, device, rid = staged(tmp_path)
    device.fail[f"{rid}/scripts/release/install_units.sh"] = 1
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5) == 1
    assert Path(os.path.realpath(link)).name.startswith("legacy-")


def test_device_state_is_carried_over_and_never_overwritten(tmp_path):
    link, device, rid = staged(tmp_path)
    lay = rg.Layout(link)
    release = lay.release(rid)
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5) == 0
    config = release / "config.yaml"
    assert config.read_text().endswith("dk_device_own_key\n"), "the device's own config, not a repo default"
    assert oct(config.stat().st_mode & 0o777) == "0o600"
    migrate = next(c for c in device.commands if c[1].endswith("scripts/migrate_config.py"))
    assert migrate == [str(release / ".venv/bin/python"), str(release / "scripts/migrate_config.py"), str(config)], \
        "the migration runs in the NEW release, on its copy"
    # Shared state: the same files through every release; a write after the switch survives a rollback.
    assert (release / "data").resolve() == (lay.state / "data").resolve()
    (link / "data" / "written_by_v2.json").write_text("{}")
    assert (link / "models" / "m.hef").read_bytes() == b"HEF"
    assert (link / ".secrets").read_text() == "device-secret\n"
    assert guard(link, device).switch_to("previous", timeout_s=60, settle_s=5) == 0
    assert (link / "data" / "written_by_v2.json").exists() and (link / "models" / "m.hef").read_bytes() == b"HEF"


def test_a_config_already_in_the_release_is_never_overwritten(tmp_path):
    link, device, rid = staged(tmp_path)
    release = rg.Layout(link).release(rid)
    (release / "config.yaml").write_text("operator: placed deliberately\n")
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5) == 0
    assert (release / "config.yaml").read_text() == "operator: placed deliberately\n"


def test_a_failed_migration_switches_nothing(tmp_path):
    link, device, rid = staged(tmp_path)
    device.fail["migrate_config.py"] = 1
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5) == 1
    assert Path(os.path.realpath(link)).name.startswith("legacy-") and device.restarts == 0
    assert not (rg.Layout(link).state / rg.STATE_DIR / "pending.json").exists()


def test_activation_without_non_interactive_sudo_is_refused_before_anything_changes(tmp_path):
    link, device, rid = staged(tmp_path)
    device.fail["sudo -n true"] = 1
    out: list[str] = []
    assert guard(link, device, out).activate(rid, timeout_s=60, settle_s=5) == 2
    assert "non-interactive sudo" in out[-1] and Path(os.path.realpath(link)).name.startswith("legacy-")
    assert not (rg.Layout(link).release(rid) / "config.yaml").exists() and device.restarts == 0


def test_only_a_staged_release_can_be_activated_and_never_over_a_pending_one(tmp_path):
    link, device, rid = staged(tmp_path)
    out: list[str] = []
    assert guard(link, device, out).activate("0123456789ab", timeout_s=60, settle_s=5) == 2
    (rg.Layout(link).release(rid) / rg.STAGED_MARK).unlink()
    assert guard(link, device, out).activate(rid, timeout_s=60, settle_s=5) == 2
    assert "not a staged, verified release" in out[-1]


def test_switch_to_previous_is_the_operator_rollback(tmp_path):
    link, device, rid = staged(tmp_path)
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5) == 0
    assert guard(link, device).switch_to("previous", timeout_s=60, settle_s=5) == 0
    assert Path(os.path.realpath(link)).name.startswith("legacy-")
    assert device.pip_calls()[-1][0].endswith(f"{rid}/.venv/bin/python"), "only the stage ever installed"


def test_activation_outcome_is_what_status_wait_reports(tmp_path):
    link, device, rid = staged(tmp_path)
    device.broken = {rid}
    assert guard(link, device).activate(rid, timeout_s=60, settle_s=5, attempt="t-1") == 1
    out: list[str] = []
    assert guard(link, device, out).status(wait_for=rid, timeout_s=10, attempt="t-1") == 1
    assert "FAILED" in out[-1]
    assert guard(link, device, out).status(wait_for=rid, timeout_s=4, attempt="t-other") == 3


def test_prune_keeps_current_previous_legacy_and_the_newest(tmp_path):
    link, device = adopted(tmp_path)
    lay = rg.Layout(link)
    ids = []
    for n in range(5):
        commit = f"{n}" * 40
        tar, manifest, sha = bundle(tmp_path, {**CODE_V2, "main.py": f"V = {n}\n".encode()}, commit, name=str(n))
        assert guard(link, device).stage(tar, manifest, sha) == 0
        assert guard(link, device).activate(commit[:12], timeout_s=60, settle_s=5) == 0
        ids.append(commit[:12])
    kept = {p.name for p in lay.staged()}
    assert ids[-1] in kept and ids[-2] in kept and any(k.startswith("legacy-") for k in kept)
    assert ids[0] not in kept and len(kept) <= rg.KEEP_RELEASES + 1


def test_detach_needs_the_guard_as_a_file_and_snapshot_refuses_on_a_staged_layout(tmp_path):
    link, device = adopted(tmp_path)
    script = (REPO / "scripts" / "release" / "release_guard.py").read_bytes()
    proc = subprocess.run([sys.executable, "-", "activate", "--root", str(link), "--id", "x", "--detach"],  # noqa: S603
                          input=script, capture_output=True, check=False, cwd=tmp_path)
    assert proc.returncode == 2 and b"not stdin" in proc.stdout
    out: list[str] = []
    assert guard(link, device, out).snapshot() == 2 and "staged releases" in out[-1]


# ---------------------------------------------------------------------- the bundle the operator builds
@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_the_real_bundle_stages_cleanly_and_a_tampered_copy_does_not(tmp_path):
    from scripts.release import make_release_manifest as rm

    made = rm.bundle(REPO, "HEAD", tmp_path / "out")
    files = json.loads(Path(made["manifest"]).read_text())["deploy_set"]["files"]
    dest = tmp_path / "x"
    dest.mkdir()
    assert rg.extract_verified(Path(made["tarball"]), dest, files) == made["files"] == len(files)
    assert not (dest / "config.yaml").exists() and not (dest / "data").exists() and not (dest / "ios").exists()
    # Same commit, same bytes: the operator can re-check a tarball by hash.
    again = rm.bundle(REPO, "HEAD", tmp_path / "again")
    assert again["tarball_sha256"] == made["tarball_sha256"]
    # One flipped byte in one file of the real tarball is refused.
    with tarfile.open(made["tarball"]) as tar:
        members = [(m, tar.extractfile(m).read()) for m in tar]  # type: ignore[union-attr] - streaming order
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz", compresslevel=1) as tar:
        for i, (member, data) in enumerate(members):
            blob = (data[:-1] + bytes([data[-1] ^ 1]) if data else b"x") if i == len(members) // 2 else data
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    dest2 = tmp_path / "y"
    dest2.mkdir()
    with pytest.raises(rg.StageError, match="sha256 does not match"):
        rg.extract_verified(bad, dest2, files)
