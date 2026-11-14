"""M8.5 (M84-6) end to end, on the Mac: `scripts/release/deploy.sh` (what `make deploy`
runs) against a FAKE device -- ssh/scp run locally in a temp home, sudo/systemctl are
recorders whose health answer depends on which release the service path points at,
and a release venv's pip records its argv instead of installing.

Everything else is real: the bundle from git objects of a temp copy of this tree,
the upload, the guard streamed from the same commit, per-file verification, the
release venv (fake) + the real MediaPipe model-file check, the detached activation
with config carry-over and migration, the atomic switch, the health wait, the
rollback by switching back, and `status --wait`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
GIT = ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"]

pytestmark = [pytest.mark.skipif(not shutil.which("git") or not (REPO / ".git").exists(), reason="needs git"),
              pytest.mark.timeout(240)]  # three real bundles of this tree and a detached activation each

FAKE_SSH = f"""#!{PY}
import os, subprocess, sys
home = os.environ["FAKE_HOME"]
sys.exit(subprocess.run(["bash", "-c", " ".join(sys.argv[2:])], cwd=home, env={{**os.environ, "HOME": home}}).returncode)
"""
FAKE_SCP = f"""#!{PY}
import os, shutil, sys
args = [a for a in sys.argv[1:] if a != "-q"]
dest = os.path.join(os.environ["FAKE_HOME"], args[-1].split(":", 1)[1])
for src in args[:-1]:
    shutil.copy(src, dest)
"""
FAKE_SUDO = f"""#!{PY}
import os, sys
home = os.environ["FAKE_HOME"]
args = [a for a in sys.argv[1:] if a != "-n"]
if args == ["true"]:
    sys.exit(0)
sys.argv[1:] = args
if sys.argv[1] == "bash" and sys.argv[2].endswith("install_units.sh"):
    with open(os.path.join(home, "units.log"), "a") as f:
        f.write(" ".join(sys.argv[2:]) + "\\n")
    sys.exit(0)
os.execvp(sys.argv[1], sys.argv[1:])
"""
FAKE_SYSTEMCTL = f"""#!{PY}
import os, sys
home = os.environ["FAKE_HOME"]
state = os.path.join(home, "systemd.state")
count = os.path.join(home, "restarts")
cmd = sys.argv[1]
if cmd in ("stop", "start", "restart"):
    open(state, "w").write("stopped" if cmd == "stop" else "running")
    if cmd == "restart":
        n = int(open(count).read()) if os.path.exists(count) else 0
        open(count, "w").write(str(n + 1))
    with open(os.path.join(home, "systemctl.log"), "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
    sys.exit(0)
if cmd == "show":
    stopped = os.path.exists(state) and open(state).read() == "stopped"
    live = os.path.realpath(os.path.join(home, "sensordyme"))
    if stopped:
        print("ActiveState=inactive\\nSubState=dead\\nNRestarts=0")
    elif os.path.exists(os.path.join(live, "BROKEN_RELEASE")):
        n = int(open(count).read()) if os.path.exists(count) else 0
        bump = os.path.join(home, "crashes")
        c = int(open(bump).read()) + 1 if os.path.exists(bump) else 1
        open(bump, "w").write(str(c))
        print(f"ActiveState=activating\\nSubState=auto-restart\\nNRestarts={{n + c}}\\nExecMainStatus=1")
    else:
        print("ActiveState=active\\nSubState=running\\nNRestarts=0\\nExecMainStatus=0")
    sys.exit(0)
sys.exit(0)
"""
# The device's system python3: real, except `-m venv` makes a release venv whose
# python records pip and runs everything else with the real interpreter.
FAKE_PYTHON3 = f"""#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  dir="${{@: -1}}"; mkdir -p "$dir/bin"
  cat > "$dir/bin/python" <<'EOF'
#!/bin/bash
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "$*" >> "$FAKE_HOME/pip.log"; exit 0; fi
exec {PY} "$@"
EOF
  chmod +x "$dir/bin/python"; exit 0
fi
exec {PY} "$@"
"""


def _fake_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ssh", FAKE_SSH), ("scp", FAKE_SCP), ("sudo", FAKE_SUDO), ("systemctl", FAKE_SYSTEMCTL),
                       ("python3", FAKE_PYTHON3)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    return bin_dir


def _repo_copy(tmp_path: Path) -> Path:
    """This tree as it is now (tracked + new files), committed in a throwaway repo."""
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard", "-z"],  # noqa: S603, S607
                            capture_output=True, check=True).stdout.decode().split("\0")
    root = tmp_path / "repo"
    for rel in filter(None, listed):
        source = REPO / rel
        if source.is_file() and not source.is_symlink():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, root / rel)
    subprocess.run(["git", "init", "-q", "-b", "rc", str(root)], check=True)  # noqa: S603, S607
    subprocess.run([*GIT, "-C", str(root), "add", "-A"], check=True)  # noqa: S603
    subprocess.run([*GIT, "-C", str(root), "commit", "-q", "-m", "good"], check=True)  # noqa: S603
    return root


def _commit(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S603, S607
                          check=True).stdout.strip()


def _legacy_device(home: Path) -> Path:
    device = home / "sensordyme"
    (device / "data").mkdir(parents=True)
    (device / "data" / "device_state.json").write_text('{"preset": "driver_safety"}')
    (device / "models").mkdir()
    (device / "models" / "damoyolo_tinynasL20_T.hef").write_bytes(b"HEF")
    (device / "main.py").write_text("print('the pre-M8.5 release')\n")
    (device / ".venv" / "bin").mkdir(parents=True)
    os.symlink(PY, device / ".venv" / "bin" / "python")
    shutil.copy2(REPO / "config.yaml", device / "config.yaml")
    os.chmod(device / "config.yaml", 0o600)
    return device


def _deploy(repo: Path, env: dict, ref: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(repo / "scripts/release/deploy.sh")], cwd=repo,  # noqa: S603, S607
                          env={**env, "REF": ref}, capture_output=True, text=True, timeout=280, check=False)


def test_deploy_stages_verifies_switches_confirms_and_rolls_back_a_broken_release(tmp_path):
    repo = _repo_copy(tmp_path)
    good = _commit(repo)
    home = tmp_path / "home"
    device = _legacy_device(home)
    env = {**os.environ, "PATH": f"{_fake_bin(tmp_path)}:{os.environ['PATH']}", "FAKE_HOME": str(home),
           "DEVICE": "op@fake-pi", "DIR": "sensordyme", "PY": PY, "DEPLOY_SETTLE": "1", "DEPLOY_TIMEOUT": "20",
           "DEPLOY_WAIT": "120", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("SENSORDYME_ALLOW_UNHASHED", None)

    # make deploy-adopt: stop, adopt (the guard streamed from the ref), start.
    subprocess.run(["systemctl", "stop", "sensordyme.service"], env=env, check=True)  # noqa: S607
    guard = subprocess.run(["git", "-C", str(repo), "show", f"{good}:scripts/release/release_guard.py"],  # noqa: S603, S607
                           capture_output=True, check=True).stdout
    adopt = subprocess.run(["ssh", "op@fake-pi", "python3 - adopt --root sensordyme"], input=guard, env=env,  # noqa: S607
                           capture_output=True, check=False)
    assert adopt.returncode == 0, adopt.stdout + adopt.stderr
    subprocess.run(["systemctl", "start", "sensordyme.service"], env=env, check=True)  # noqa: S607
    legacy = Path(os.path.realpath(device))
    assert legacy.name.startswith("legacy-")

    # 1. A healthy release: staged, verified, switched, confirmed.
    first = _deploy(repo, env, good)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "CONFIRMED" in first.stdout
    live = Path(os.path.realpath(device))
    assert live.name == good[:12] and (live / "STAGED.json").exists()
    assert (live / "ARTIFACTS.json").exists() and json.loads((live / "ARTIFACTS.json").read_text())["ok"]
    # config.yaml: the device's own copy, carried into the new release and migrated THERE;
    # the previous release keeps its untouched copy (what a rollback runs on).
    assert (legacy / "config.yaml").read_bytes() == (REPO / "config.yaml").read_bytes()
    backups = sorted(live.glob("config.yaml.bak-*"))
    assert backups and backups[0].read_bytes() == (legacy / "config.yaml").read_bytes(), "carried, then migrated"
    assert oct((live / "config.yaml").stat().st_mode & 0o777) == "0o600"
    assert (device / "data" / "device_state.json").read_text() == '{"preset": "driver_safety"}'
    assert (device / "models" / "damoyolo_tinynasL20_T.hef").read_bytes() == b"HEF"
    pip = (home / "pip.log").read_text().splitlines()
    assert len(pip) == 1 and "--require-hashes --only-binary=:all:" in pip[0]
    assert pip[0].endswith(f"-r {home.resolve()}/sensordyme.releases/{good[:12]}/requirements-device.lock") or \
        pip[0].endswith(f"/sensordyme.releases/{good[:12]}/requirements-device.lock")
    units = (home / "units.log").read_text()
    assert f"--app-dir {home}/sensordyme" in units or ("--app-dir " in units and "/sensordyme\n" in units)

    # 2. A release whose service crash-loops: rolled back by switching back, offline.
    (repo / "BROKEN_RELEASE").write_text("the fake systemd crash-loops this release\n")
    subprocess.run([*GIT, "-C", str(repo), "add", "-A"], check=True)  # noqa: S603
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "broken"], check=True)  # noqa: S603
    broken = _commit(repo)
    second = _deploy(repo, env, broken)
    assert second.returncode == 1, second.stdout + second.stderr
    assert "FAILED" in second.stdout
    assert Path(os.path.realpath(device)).name == good[:12], "the previous release is live again"
    assert len((home / "pip.log").read_text().splitlines()) == 2, "staging installed; the rollback did not"
    rolled = json.loads((home / "sensordyme.state/data/release/rolled_back.json").read_text())
    assert rolled["release"] == broken[:12] and rolled["previous"] == good[:12] and rolled["previous_healthy"]

    # 3. A tampered upload of a new release never gets staged: refused before a file is written.
    (repo / "BROKEN_RELEASE").unlink()
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-am", "fixed"], check=True)  # noqa: S603
    fixed = _commit(repo)
    out = tmp_path / "bundle"
    subprocess.run([PY, str(repo / "scripts/release/make_release_manifest.py"), "bundle", "--repo", str(repo),  # noqa: S603
                    "--ref", fixed, "--out-dir", str(out)], check=True, capture_output=True)
    tgz, man = out / f"sensordyme-{fixed[:12]}.tar.gz", out / f"sensordyme-{fixed[:12]}.manifest.json"
    with tarfile.open(tgz) as tar:
        members = [(m, tar.extractfile(m).read()) for m in tar]  # type: ignore[union-attr] - streaming order
    with tarfile.open(tgz, "w:gz", compresslevel=1) as tar:
        for member, data in members:
            blob = data + b"\nimport os  # injected\n" if member.name == "main.py" else data
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    subprocess.run(["scp", "-q", str(tgz), str(man), "op@fake-pi:sensordyme.incoming/"], env=env, check=True)  # noqa: S603, S607
    stage = (f"python3 - stage --root sensordyme --bundle sensordyme.incoming/{tgz.name} "
             f"--manifest sensordyme.incoming/{man.name} --manifest-sha256 {hashlib.sha256(man.read_bytes()).hexdigest()}")
    third = subprocess.run(["ssh", "op@fake-pi", stage],  # noqa: S603, S607
                           input=guard, env=env, capture_output=True, check=False)
    assert third.returncode == 1, third.stdout + third.stderr
    assert b"'main.py': sha256 does not match the manifest" in third.stdout
    assert not (home / "sensordyme.releases" / fixed[:12]).exists()
    assert Path(os.path.realpath(device)).name == good[:12]
    assert len((home / "pip.log").read_text().splitlines()) == 2, "nothing installed for it"

    # 4. Re-running the deploy of the running release is a no-op that re-verifies it.
    again = _deploy(repo, env, good)
    assert again.returncode == 0 and "re-verified" in again.stdout and "already the running release" in again.stdout, \
        again.stdout + again.stderr
    assert len((home / "pip.log").read_text().splitlines()) == 2
