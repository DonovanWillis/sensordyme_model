# Developer setup: three machines, one repository

SensorDyme is developed on three machines. Each does the work it is best at, and
they share code only through Git.

| Machine | Use it for | Do not use it for |
|---|---|---|
| **MacBook** (macOS, Apple silicon) | Xcode and the iOS app: Simulator, XCUITests, signing, installing on the iPhone. `make test-ios`, `make acceptance-full`. It is also the `sdrun` controller for the compute fleet. | Big venvs, datasets, checkpoints, training. It has very little free disk (about 7 GiB on 2026-11-15). |
| **quantcompute** (ASUS TUF laptop: Windows 11, 64 GB RAM, RX 7700S) | Python, backend and integration work: the backend, the device runtime, the virtual device, the test suite, lint. Best done in **WSL2** (Ubuntu-24.04 is already installed there). | Anything iOS. CUDA. |
| **oryn** (Windows desktop, RTX 4060, WSL2) | CUDA training and evaluation, through `sdrun` jobs submitted from the MacBook. | Interactive development. It runs restricted jobs only. |

The Mac mini is **not** part of this setup. Do not install anything on it.

How the fleet itself works (accounts, limits, recovery) is documented in
`~/sensordyme-fleet/docs/` on the MacBook. That infrastructure is frozen, and
nothing in this document changes it. The Mac-only development loop (run the
detector, lint, type-check) is in [DEVELOPMENT.md](DEVELOPMENT.md).

---

## 1. How source moves between machines

There is no file syncing, shared drive or rsync. **Git is the only channel.**

- **`origin`** is the owner's repository, `github.com/monishallada/sensordyme_model`.
  Every machine pushes and pulls branches here.
- **`upstream`** is a collaborator's repository. Fetch from it if you need to.
  **Never push to it.** Changes reach it only as a pull request the owner opens
  on purpose.
- Work on a branch (`feat/<topic>`). Commit on one machine, `git push origin
  <branch>`, then `git fetch origin && git switch <branch>` (or `git pull`) on
  the next machine.
- A branch that only one machine works on at a time never conflicts. If two
  machines committed to the same branch, `git pull --rebase` before you push.

### Never commit secrets

`.gitignore` already covers the files that hold them: `.secrets/`,
`cloud/.jwt_secret`, `cloud/*.db`, `*.local.yaml`, `data/`, `models/`, `*.pt` and
`*.hef`. Do not commit API keys, tokens, Wi-Fi passwords, the iPhone's UDID,
signing material, or Tailscale addresses, even in a doc or a test fixture. Use
the hostnames (`quantcompute`, `oryn`) in anything you write down. `sdrun`
refuses to upload a commit that contains credential-looking files, but that is
a last line of defence, not a reason to rely on it.

### The existing checkout on quantcompute points at the wrong remote

There is already a clean checkout in the owner's Windows profile, at `%USERPROFILE%\sensordyme\sensordyme_model`
(branch `integration/all`, HEAD `0cca747` on 2026-11-15). Its **`origin` is the
collaborator's repository**, so a plain `git push` from it would go to the
wrong place. Before you use it, the owner should re-point it. This was not done
for you:

```powershell
cd $HOME\sensordyme\sensordyme_model
git remote rename origin upstream
git remote add origin https://github.com/monishallada/sensordyme_model.git
git fetch origin
git remote -v        # origin = monishallada, upstream = DonovanWillis
```

If you follow the WSL setup below, you make a new clone inside WSL with the
right remotes from the start, and this Windows checkout can stay as it is.

---

## 2. First-time setup

### 2.1 quantcompute, in WSL2 (recommended)

WSL2 is installed and working on quantcompute: WSL 3.0.5, distributions
`Ubuntu-24.04` and `docker-desktop`. Use `Ubuntu-24.04`. It gives the same
Linux environment as oryn's job worker, where the whole suite passes (§4).

Open an Ubuntu-24.04 terminal (Windows Terminal, or `wsl -d Ubuntu-24.04`) and
run:

```bash
# 1. uv, installed for your user only (no sudo). https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh
exec "$SHELL" -l

# 2. Clone INSIDE the WSL filesystem, not under /mnt/c. That is much faster, and
#    it keeps Windows line-ending conversion away from the repository.
mkdir -p ~/src && cd ~/src
git clone https://github.com/monishallada/sensordyme_model.git
cd sensordyme_model
git remote add upstream https://github.com/DonovanWillis/sensordyme_model.git

# 3. Python environment and smoke test (see §2.4)
scripts/dev/bootstrap_python.sh
```

Inside the distro, `git` needs your name, email and GitHub credentials. Use
`gh auth login`, or Git Credential Manager from Git for Windows (2.55 is
installed on the Windows side).

If `import cv2` fails with `libGL.so.1: cannot open shared object file`, install
the two system libraries that CI installs. This is your own distro, so it is
your call:

```bash
sudo apt-get update && sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0 make
```

`make` is included because a minimal Ubuntu does not always have it.

**Not checked:** what is already installed inside `Ubuntu-24.04` (python3, git,
make, libGL). The distro was stopped during the inspection, and it was left
stopped rather than booted just to look.

### 2.2 quantcompute, Windows-native (works partly)

You can work without WSL. Measured on 2026-11-15 (§4), the backend tests mostly
pass, but not all of them:

- **Smart App Control** (on for the whole machine) blocks zeroconf's compiled
  extension: `ImportError: DLL load failed ... An Application Control policy has
  blocked this file`. The backend's mDNS announcement then degrades to a warning.
  Other compiled packages could be blocked the same way. Popular, signed wheels
  passed: pydantic-core, jiter, cryptography and numpy all imported.
- **6 tests fail** because they assume POSIX behaviour: `0o600` file modes, which
  NTFS reports as `0o666`, and glibc's `%-I` in `strftime`. These are test and
  platform gaps, not regressions.
- `simpleaudio` (Windows/macOS only in `requirements-device.txt`) and
  `mediapipe` were **not** tried on Windows.

If you still want it, in PowerShell (uv 0.13.7 and a uv-managed CPython 3.12
are already installed; Python 3.11.9 is available too, as `py -3.11`):

```powershell
cd $HOME\sensordyme\sensordyme_model     # after fixing its remotes (§1)
uv venv --python 3.12 $HOME\.venvs\sensordyme
$py = "$HOME\.venvs\sensordyme\Scripts\python.exe"
# The set that was measured; add packages from requirements-device.txt as needed.
uv pip install --python $py -r cloud/requirements.txt pytest==8.3.3 pytest-timeout==2.4.0 `
    anthropic==0.116.0 cryptography requests numpy pyyaml -c requirements-device.txt
$env:PYTHONDONTWRITEBYTECODE = "1"; $env:MPLBACKEND = "Agg"
& $py -m pytest -p no:cacheprovider -q tests/test_cloud_backend.py tests/test_phase1_cloud_auth.py
```

Notes:

- The bare `python` on PATH is the Microsoft Store alias, not a real
  interpreter. Use `py -3.11`, `uv`, or the venv's own `python.exe`.
- Python 3.13 and 3.14 are installed too. Do not use them for SensorDyme:
  mediapipe 0.10.14 has no wheel for them.
- `bootstrap_python.sh` does not run on native Windows; it is for Linux, WSL and
  macOS.
- No PowerShell bootstrap script is provided, because WSL is available. The
  commands above are what a `.ps1` would do.

### 2.3 Claude Code on quantcompute

Claude Code **2.1.311 is installed natively on Windows** (Node v26.8.1 is also
installed). Both setups are possible:

- **Native Windows:** run `claude` in PowerShell, in a Windows checkout.
- **Inside WSL:** install Claude Code inside `Ubuntu-24.04` and run it in the WSL
  checkout (`~/src/sensordyme_model`). This is the better fit for §2.1, because
  its shell commands then run in the same Linux environment as the tests.
  Whether it is already installed inside the distro was not checked.

### 2.4 The bootstrap script (Linux, WSL, macOS)

```bash
scripts/dev/bootstrap_python.sh                    # venv at ~/.venvs/sensordyme, backend profile, smoke test
scripts/dev/bootstrap_python.sh --profile full     # also torch + ultralytics (what CI installs)
scripts/dev/bootstrap_python.sh --smoke-only       # just re-run the smoke test
scripts/dev/bootstrap_python.sh -- tests/test_rules.py -k fall   # your own pytest arguments
scripts/dev/bootstrap_python.sh --help
```

What it does:

- **Where the venv goes:** it creates the venv **outside the checkout**, at
  `~/.venvs/sensordyme` (override with `--venv DIR` or `SENSORDYME_VENV`).
- **What it installs:** only the repository's own pin files.
  - The **backend** profile is `requirements.txt` without its torch and
    ultralytics lines, plus `cloud/requirements.txt` and `requirements-dev.txt`.
    That is everything the backend, the device runtime and the test suite need,
    in about 1.2 GB. Torch and ultralytics would add several GB on Linux, and
    only the `mac_max` profile, the benchmarks and training use them.
  - The **full** profile is exactly CI's test install.
- **Python version:** it uses uv when it is available, and falls back to
  `python3 -m venv` + pip. It requires Python 3.11 or 3.12.
- **Provenance:** it records the profile, interpreter, commit and the SHA-256 of
  each pin file in `<venv>/sensordyme-bootstrap.txt`.
- **Smoke test:** it runs the backend + device test files listed in
  `scripts/dev/backend_tests.txt`.
- **What it never does:** it never uses sudo, and it never writes into the
  checkout.

To use the venv with the Makefile, override `VENV`. It defaults to `.venv/bin`:

```bash
make test VENV="$HOME/.venvs/sensordyme/bin"
make test-backend VENV="$HOME/.venvs/sensordyme/bin"
make sim-e2e SIM_PY="$HOME/.venvs/sensordyme/bin/python"
```

`make typecheck` is the exception. `pyrightconfig.json` looks for `.venv` in the
checkout. On quantcompute, where disk is not a concern, you can create the venv
there instead: `scripts/dev/bootstrap_python.sh --venv "$PWD/.venv"`. `.venv/`
is git-ignored.

### 2.5 oryn

Nothing to install. Jobs run from the MacBook with `sdrun`, in a restricted
account inside the WSL distro `sdjobs` (Ubuntu 24.04, Python 3.12.16, git 2.43,
uv 0.13.7). The CUDA environment `sdgpu` (torch 2.14.1+cu126) is shared and
read-only to jobs. Datasets and checkpoints live in `$SDJOB_DATA` on oryn
(918 GB free on 2026-11-15) and never travel to the MacBook. See
`~/sensordyme-fleet/docs/README.md`, under "Handoff".

**Jobs can only be submitted from the MacBook.** The job accounts' SSH keys
only accept the MacBook. To train from quantcompute's code, push the branch,
then on the MacBook:

```bash
cd ~/sensordyme/sensordyme_model && git fetch origin
/usr/bin/python3 ~/sensordyme-fleet/sdrun submit -t oryn --repo ~/sensordyme/sensordyme_model \
    --ref origin/feat/<topic> --include <paths> --name <name> -- <command>
```

`--ref` takes any commit. `sdrun` still refuses to submit while the MacBook's
own working tree has uncommitted changes to tracked files.

### 2.6 MacBook

The existing setup stays as it is: the repo's `.venv` (from `setup_mac.sh`) and
Xcode. iOS build, signing and device install are in
[IOS_DEVICE_BUILD.md](IOS_DEVICE_BUILD.md). The Simulator and XCUITest journeys
are in [VIRTUAL_DEVICE.md](VIRTUAL_DEVICE.md) and [ACCEPTANCE.md](ACCEPTANCE.md).
Do not create a second venv here, and especially not the `full` profile.

---

## 3. The daily loop

1. **quantcompute (WSL):** `git switch -c feat/<topic>` (or `git pull`), edit,
   `make test-backend VENV=...`, then `make test VENV=...` before pushing.
   `make lint` works anywhere: it uses ruff from PATH or `uvx ruff@0.16.4`.
   Commit, then `git push -u origin feat/<topic>`.
2. **Training or evaluation:** on the MacBook, `git fetch origin`, then `sdrun
   submit -t oryn --ref origin/feat/<topic> ...` (§2.5). Fetch only small
   results: `sdrun fetch <id> --only '<regex>'`.
3. **iOS or app work:** on the MacBook, `git fetch origin && git switch
   feat/<topic>` (or `git pull`), then build in Xcode, run `make test-ios` and
   the Simulator journeys, and install on the phone.
4. Open pull requests from `origin` branches. Merge order and release records
   follow the existing process in the repository.

---

## 4. Which test commands run where

These are only targets that exist in the `Makefile` (`make help` lists them all).

| Command | quantcompute WSL / Linux | quantcompute Windows-native | MacBook |
|---|---|---|---|
| `make test-backend` (scripts/dev/backend_tests.txt) | **yes**, measured: 344 passed on oryn's Linux worker | no `make`; partly: 6 of its 12 files were tried directly with pytest, giving 188 passed, 6 failed and 1 skipped (§2.2) | yes: 344 passed |
| `make test` (full suite) | **yes**, see the measurement below | not tried | yes |
| `make sim-e2e` | yes: its files are part of the full suite that passed on Linux | not tried | yes |
| `make lint` | yes (ruff from PATH or uvx) | `uvx ruff@0.16.4 check .` should work, since uv is installed (not run) | yes |
| `make typecheck` | only with the venv at `./.venv` (§2.4); not run off the Mac | not tried | yes |
| `make audit` | yes, with uv (queries PyPI) | not tried | yes |
| `make acceptance` | **not verified** off the Mac | no | yes |
| `make test-ios`, `make acceptance-full` | no (needs Swift, Xcode and the Simulator) | no | **yes, only here** |
| `make deploy*`, `make demo-*` | need SSH to the Pi from that machine; not set up off the Mac | no | as documented in DEPLOYMENT.md |

**Measured on 2026-11-15.**

- **Linux:** the backend profile was built fresh in an `sdrun` job on oryn's
  Linux worker (`--no-gpu`, 8 CPUs). That job ran `scripts/dev/bootstrap_python.sh`
  itself. Job ids, logs and pytest summaries are in the M12 evidence folder,
  `~/sensordyme-audit/2026-11-10/m12-evidence/devenv/`.
  - Install: 163 s from a warm uv cache.
  - Smoke test: 344 passed.
  - Full suite (`pytest -q`, same venv): 4521 passed, 1 failed, 52 skipped and
    1 xfailed, in 1012 s.
  - The one failure is `test_m84_sunday_preflight.py::test_a_healthy_pi_double_passes_every_read_only_check`.
    Its `software_revision` check needs a `.git` directory, and `sdrun` jobs run
    a `git archive` export, which has none. It fails the same way on the Mac in
    an exported tree, and passes in a real clone.
  - Most of the skips have the same cause ("needs a git checkout"). The rest are
    Mac-only tools (swiftc, hdiutil), `visudo`, `onnx`, or gitignored model files.
  - A clone in WSL has a `.git` directory, so the git-dependent tests run there.
    That has not yet been measured on quantcompute itself.
- **Windows-native:** the run on quantcompute used the signed CPython 3.12.10 of
  the job account, with a venv and pip.
  - Install: 24 s.
  - Six backend files: 188 passed, 6 failed, 1 skipped. The causes are in §2.2.
  - Leaving out `anthropic` is not an option here either: without it the
    backend tests cannot be collected, because nothing else installs httpx.

---

## 5. Keep large artifacts off the Mac

- **Datasets, checkpoints, exported models, videos:** these go in `$SDJOB_DATA`
  on oryn, or on quantcompute's disk (about 546 GB free on C:). Never in the
  Mac's checkout. `data/`, `models/`, `*.pt` and `*.hef` are git-ignored, so
  they never travel through Git either.
- **`sdrun fetch`** has a 200 MiB cap by default. Use `--only '<regex>'` to fetch
  just the metrics or JSON you need. Fetched artifacts land in
  `~/sensordyme-fleet/runs/<id>/artifacts/`. Prune old ones there when disk is
  tight.
- **Venvs:** the MacBook has one, the repo's `.venv`. Everywhere else they live
  in `~/.venvs/`.
- **Worktrees:** each extra git worktree on the Mac is a full checkout. Remove
  the ones you are done with: `git worktree list`, then `git worktree remove
  <path>`.
- **Xcode:** DerivedData and old Simulator runtimes are the usual large items on
  the Mac (`~/Library/Developer/Xcode/DerivedData`, `xcrun simctl delete
  unavailable`). Cleaning them is the owner's call. Nothing here does it
  automatically.

---

## 6. Optional: building the iOS app on the Mac from quantcompute over SSH

> **Requires owner approval. Not enabled.** This means turning on macOS Remote
> Login (sshd) on the MacBook. That is a new way into the machine that holds the
> source, the signing identity and the fleet controller's keys. Nothing in this
> repository enables it, and nothing in this setup depends on it.

If the owner decides to enable it, this is the shape of a careful version:

1. **Turn on sshd for one account only.** On the MacBook: System Settings →
   General → Sharing → Remote Login, with "Allow access for: Only these users"
   set to the owner's own account.
2. **Key-only login.** Add the ASUS user's public key to
   `~/.ssh/authorized_keys`, restricted with `from="<quantcompute's tailnet
   address>"` plus `no-agent-forwarding,no-port-forwarding,no-X11-forwarding`.
   Then disable password authentication in `/etc/ssh/sshd_config`.
3. **Reachable over Tailscale only.** Keep the macOS firewall blocking inbound
   SSH on other interfaces. The fleet notes record that the tailnet ACL is
   currently permissive, which is a pending owner decision. Tighten it first, so
   that other tailnet devices cannot reach the Mac's port 22.
4. **Use it for builds only:**
   `ssh <mac> 'cd ~/sensordyme/sensordyme_model && git fetch origin && git switch feat/<topic> && make test-ios'`,
   plus Simulator builds with `xcodebuild`.
   - Signing and installing on a device over SSH needs the login keychain
     unlocked in that session, and a connected, trusted iPhone. Expect to do
     device installs at the Mac itself.
5. **To undo it,** turn Remote Login off and delete the key line.

None of these steps has been tried.

---

## 7. Open owner decisions

None of these is needed for the workflow above.

1. **Re-point the remotes of the existing Windows checkout on quantcompute**
   (§1). Or leave it alone and use a fresh clone in WSL.
2. **macOS Remote Login on the MacBook,** for building iOS from quantcompute
   (§6). This is new remote-access exposure.
3. **Letting quantcompute submit `sdrun` jobs.** That would need a new
   controller key, and a change to the job accounts' `from=` restriction on
   both workers. The fleet is frozen and this would be new remote access, so
   for now jobs are submitted from the MacBook (§2.5).
4. **An explicit `httpx` pin.** The backend tests' `starlette.testclient` needs
   httpx, and it arrives only through `anthropic` in `requirements.txt`. One
   line in `requirements-dev.txt` would make that dependency deliberate.
   `bootstrap_python.sh` works either way.
5. **Making the POSIX-only tests portable,** if native Windows is ever meant to
   be a supported development platform (§2.2). WSL makes this unnecessary.
