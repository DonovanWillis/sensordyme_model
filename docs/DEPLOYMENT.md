# Deploying a release to a device (M8.5)

M8.5 replaces `make deploy`'s in-place rsync with **staged releases**. It closes
the three M8.4 safe-update findings (`docs/SECURITY_REVIEW_M6.md` §7):

| Finding | Before | Now |
|---|---|---|
| M84-4 | Every device install was an unhashed `pip install -r requirements-device.txt`, plus an unhashed `pip install gpiozero lgpio` and `pip install --upgrade pip`. | Every device install is `pip install --require-hashes --only-binary=:all: -r requirements-device.lock`. This covers setup_pi.sh, the staged install and the legacy rollback. A release gate refuses a ref whose install paths are not hashed. |
| M84-5 | The MediaPipe `.tflite` files were never checked. The openpilot `.onnx` was downloaded with `urlretrieve` and no hash. | The `.tflite` files are checked against their registry pins at install time and at every boot. A file that does not match is never loaded, and the capability is reported impaired. The `.onnx` is pinned by sha256 at download and at load. |
| M84-6 | `rsync` wrote the working tree over the live tree. Nothing was verified on the device and it was not atomic. The `\|\|` rollback ran over the same SSH session. | The release is built from git objects only. On the device it is staged beside the live release, and every file is checked against the manifest before it is written. The switch is one `rename(2)` of a symlink. The switch and any rollback run on the device, detached from SSH. A rollback is a switch back: no network, no pip. |

Device side: `scripts/release/release_guard.py` (stdlib only, runs with the
device's system `python3`). Mac side: `scripts/release/make_release_manifest.py
bundle` and `scripts/release/deploy.sh`.

The procedures below are **tested on the Mac against fake devices only**. See
"What only a Pi can show" at the end.

## 1. The layout on the device

```
~/sensordyme                -> sensordyme.releases/<id>      the path the systemd unit names (a symlink)
~/sensordyme.releases/<id>/ one release: the verified code, its own .venv, STAGED.json,
                            ARTIFACTS.json, config.yaml (carried over), and symlinks:
                              data   -> ../../sensordyme.state/data
                              models -> ../../sensordyme.state/models   (the .hef)
                              logs, .secrets, .env (when the device has them)
~/sensordyme.releases/legacy-<time>/   the pre-M8.5 tree, adopted unchanged (§2)
~/sensordyme.state/         data/, models/, logs/, .secrets: device state, in no release
~/sensordyme.incoming/      uploaded bundles
```

`<id>` is the first 12 hex digits of the release commit.

Device state never enters a release, and a release never overwrites it:

* **Shared through symlinks.** `data/` (the guard's own records live in
  `data/release/`), `models/`, `logs/`, `.secrets` and `.env`. Every release sees
  the same files, so anything written after a switch survives a rollback.
* **`config.yaml` is copied, not linked.** Every writer replaces it with
  `os.replace()`, which would turn a symlink into a plain file. At activation the
  running release's copy is copied into the new release, but only if the new
  release has none. It is then migrated there (`scripts/migrate_config.py`, which
  also leaves a `config.yaml.bak-*`). The previous release keeps its own
  pre-migration copy, and a rollback runs on that copy. This matches the M8
  guard, which restored `config.yaml.previous`. A config change made after the
  switch, such as a token set in the app, is therefore not carried back by a
  rollback. Copy it back by hand if it is needed.
* **Excluded from bundles.** A bundle cannot carry state. The manifest excludes
  `config.yaml`, `data/`, `models/`, `*.hef`, secrets, `ios/` and so on (the same
  excludes as `integrity_manifest.py`). The stage step also refuses any such
  member outright.

## 2. One time per device: adopt the legacy tree

Prerequisites:

* The device runs the M8.4 candidate (or any earlier in-place install) at
  `~/sensordyme`, or at the directory found in runbook §C1.
* `make verify-release REF=$RC` passes (runbook A0).
* The deploying user has **non-interactive sudo** for `systemctl` and
  `install_units.sh`. This is the Pi OS default for the first user. The detached
  activation has no terminal to ask for a password, so `activate` refuses
  without it.

From the Mac:

```bash
make deploy-adopt DEVICE=<user>@<host> DIR=sensordyme     # DIR: relative to the login dir, or absolute
```

This is `ssh … 'sudo systemctl stop sensordyme.service'`, then the guard from
`REF` streamed over SSH (`python3 - adopt --root $DIR`), then `sudo systemctl
start` with an `is-active` check 30 s later. If the adopt is refused, the
service is started again unchanged.

The adopt step:

1. renames the tree to `sensordyme.releases/legacy-<time>`;
2. moves its `data/`, `models/`, `logs/`, `.secrets` and `.env` to
   `sensordyme.state/`;
3. symlinks them back;
4. points `~/sensordyme` at it.

The code, the `.venv` (reached through the same path) and `config.yaml` are
untouched, and the unit needs no change: it still names `~/sensordyme`.

The adopt step refuses to run while the unit is active, while an M8 deploy is
pending, or when the state directory already holds the same items.

**A fresh device.** First install from an approved release with
`setup_pi.sh`: extract the runbook A3 tarball to `~/sensordyme` and run
`bash setup_pi.sh`. Since M8.5 it installs only the hashed lock. Then run
`make deploy-adopt` once. Every update after that is `make deploy`.

To undo the adopt step (service stopped):

```bash
sudo systemctl stop sensordyme.service
rm ~/sensordyme                     # the symlink only; every release stays in sensordyme.releases/
mv ~/sensordyme.releases/legacy-<time> ~/sensordyme
for n in data models logs .secrets .env; do
  [ -L ~/sensordyme/$n ] && rm ~/sensordyme/$n && mv ~/sensordyme.state/$n ~/sensordyme/$n
done
sudo systemctl start sensordyme.service
```

## 3. Each deploy

```bash
make verify-release REF=$RC                                 # A0: must say RELEASE VERIFICATION: PASS
make deploy DEVICE=<user>@<host> DIR=sensordyme REF=$RC ROLE=driver_safety
```

`deploy-preflight` runs first (unchanged: the device's own config against this
code). Then `scripts/release/deploy.sh` runs these steps:

1. **bundle** (Mac). `make_release_manifest.py bundle --ref $RC` builds
   `sensordyme-<id>.tar.gz` and `sensordyme-<id>.manifest.json` from the **git
   objects of that commit**, never the working tree.
   * The tarball is deterministic (the same commit gives the same bytes) and holds
     exactly the manifest's deploy set.
   * The manifest lists every file's sha256.
   * The bundle is refused if any release gate FAILs.
2. **upload**. Both files go to `$DIR.incoming/`. Nothing live changes.
3. **stage** (device). The guard is streamed from the same commit (`git show
   <commit>:scripts/release/release_guard.py | ssh … python3 - stage …
   --manifest-sha256 <sha>`). In order, it:
   * refuses a manifest whose sha256 is not the one passed on its command line;
   * extracts into `releases/.partial-<id>`, writing each file only after its
     bytes match the manifest;
   * refuses anything that is not a regular file, an absolute or `..` path, device
     state or a secret, a file not in the manifest, a duplicate, and any listed
     file that is missing;
   * renames the result to `releases/<id>`;
   * creates `python3 -m venv --system-site-packages releases/<id>/.venv` (picamera2
     and hailo come from apt);
   * runs `releases/<id>/.venv/bin/python -m pip install --require-hashes
     --only-binary=:all: --disable-pip-version-check --no-input -r
     releases/<id>/requirements-device.lock`;
   * runs the release's own MediaPipe model-file check in that venv
     (`core.model_registry verify-artifacts`, which writes `ARTIFACTS.json`);
   * links the state and writes `STAGED.json`.

   On any failure the staged release is removed and the live release is
   unchanged. If the same commit is staged again, its files are re-verified and
   it is not rebuilt. The running release and the rollback release are never
   rebuilt in place.
4. **activate** (device, detached with `--detach`: its own session, logged to
   `~/sensordyme.state/data/release/activate-<id>.log`, so a dropped SSH session
   cannot interrupt it). It uses the staged release's own verified copy of the
   guard. In order, it:
   * checks non-interactive sudo;
   * copies `config.yaml` into the new release (never overwriting) and migrates it
     there;
   * records `pending.json`;
   * makes **one `rename(2)`** of a ready symlink onto `~/sensordyme`;
   * runs `sudo bash releases/<id>/scripts/release/install_units.sh --user <u>
     --app-dir ~/sensordyme` (the unit always names the service path, never a
     release);
   * runs `sudo systemctl restart sensordyme.service`;
   * runs **confirm-or-rollback** (the M8 semantics, unchanged): the unit must be
     `active`, with no automatic restart, for the settle time (default 30 s within
     180 s), and the device config must load under the new release.
   * On failure it switches back to the previous release, re-installs its units
     and restarts. There is no network and no pip, because that release still has
     its venv and its config.

   Every outcome, refused or not, is written to `activation.json`. After a
   confirmation, old releases are pruned. The guard keeps the current release,
   the previous one, any `legacy-*` and the 3 newest.
5. **wait** (Mac). `status --wait <id> --attempt <token>` exits:
   * `0`: confirmed;
   * `1`: rolled back;
   * `3`: no outcome yet. The device still finishes on its own; check with `make
     deploy-status`.

Optional variables:

* `FIND_LINKS=<dir on the device>`: install offline from a wheelhouse, still
  hash-checked against the lock.
* `DEPLOY_TIMEOUT` / `DEPLOY_SETTLE`: the health wait.
* `DEPLOY_WAIT`: how long the Mac waits for the outcome.
* `TESTS="…"`: recorded in the manifest.

**The same steps by hand**, for when make is not available or the device needs
a different path:

```bash
OUT=$(mktemp -d); C=$(git rev-parse "$RC^{commit}"); ID=${C:0:12}
.venv/bin/python scripts/release/make_release_manifest.py bundle --ref $C --out-dir $OUT
MSHA=$(shasum -a 256 $OUT/sensordyme-$ID.manifest.json | cut -d' ' -f1)
ssh $PI 'mkdir -p sensordyme.incoming'
scp $OUT/sensordyme-$ID.tar.gz $OUT/sensordyme-$ID.manifest.json $PI:sensordyme.incoming/
git show $C:scripts/release/release_guard.py | ssh $PI "python3 - stage --root sensordyme \
  --bundle sensordyme.incoming/sensordyme-$ID.tar.gz --manifest sensordyme.incoming/sensordyme-$ID.manifest.json \
  --manifest-sha256 $MSHA"                                                     # expect: STAGED <id>
ssh $PI "python3 sensordyme.releases/$ID/scripts/release/release_guard.py activate --root sensordyme \
  --id $ID --user \$(id -un) --attempt m1 --detach"
ssh $PI "python3 sensordyme.releases/$ID/scripts/release/release_guard.py status --root sensordyme \
  --wait $ID --attempt m1 --timeout 900"                                       # expect: CONFIRMED, exit 0
```

**Offline wheelhouse** (a device without internet). The Mac downloads exactly
the locked wheels for the Pi, and pip refuses any wheel whose hash is not in the
lock:

```bash
P=(); for v in 36 35 34 33 32 31 28 27 26 24 17; do P+=(--platform manylinux_2_${v}_aarch64); done
.venv/bin/pip download --require-hashes --only-binary=:all: "${P[@]}" --platform manylinux2014_aarch64 \
  --python-version 3.11 --implementation cp --abi cp311 -r requirements-device.lock -d wheels-aarch64
scp -r wheels-aarch64 $PI:sensordyme.wheels
make deploy … FIND_LINKS=sensordyme.wheels
```

## 4. Rollback

* **Automatic.** A release that does not stay up is switched back by the device
  itself (§3, step 4). Afterwards the release is still in `releases/` for
  inspection, and `rolled_back.json` holds the reason.
* **Manual, to the release before the last confirmed deploy** (no network, no
  pip):

  ```bash
  make deploy-rollback DEVICE=<user>@<host> DIR=sensordyme      # = release_guard.py switch --to previous
  make deploy-rollback … TO=<id>                              # any staged release, including legacy-<time>
  ```

  This switches the symlink, re-installs that release's units with `--app-dir
  ~/sensordyme`, restarts, and reports health in `switched.json`. A switch never
  undoes itself, so the operator decides what happens next.
* **On the device without the Mac:**

  ```bash
  python3 ~/sensordyme/scripts/release/release_guard.py switch --root ~/sensordyme --to previous --user $(id -un)
  ```
* **To the pre-M8.5 tree:** `TO=legacy-<time>`. To leave the staged layout
  entirely, undo the adopt step (§2).
* **To the pre-session state:** runbook §O2/O3 (the E1 archive, or the SD image).
* **A deploy left pending.** A power cut or a killed activation leaves
  `pending.json`, and the next `make deploy` refuses until the deploy is
  resolved. Resolve it on the device in one of two ways:
  * `python3 ~/sensordyme/scripts/release/release_guard.py confirm-or-rollback
    --root ~/sensordyme --user $(id -un)` judges the running release and confirms
    it or switches back;
  * `… rollback …` switches back unconditionally.

## 5. Dependencies (M84-4)

* `requirements-device.lock` is the device install. It targets CPython 3.11 on
  aarch64 (Pi OS Bookworm). Every entry carries sha256 hashes.
  `scripts/release/check_device_deps.py` (verify-release step 1) checks that:
  * every device pin is in the lock at its version;
  * every device import's distribution is locked;
  * a pin whose marker keeps it off the device is not locked;
  * nothing forbidden is locked.

  The release gate (step 3) checks that the lock agrees with the pins. It also
  checks that **every device install path is hashed**: setup_pi.sh, the
  Makefile, `release_guard.py` and `install_units.sh`. A non-comment unhashed
  install of `requirements-device.txt`, a bare `pkg==ver`, or `--upgrade pip`
  fails the gate, unless the line carries the `unhashed-fallback` mark.
* M8.5 folded `gpiozero==2.0.1` and `lgpio==0.2.2.0` into the lock, with
  `colorzero` and `setuptools` as gpiozero's dependencies. Their old separate
  unhashed install is gone from setup_pi.sh.
* M8.5 dropped `simpleaudio` on Linux (`; sys_platform != "linux"`):
  * it has no Linux wheel;
  * it was the only package pip had to build on the Pi (build code at install
    time, build dependencies fetched unhashed, ALSA headers setup_pi.sh never
    installed);
  * on the Pi, `core/alerts.py` SoundSink plays through `aplay` (alsa-utils,
    installed by setup_pi.sh), its documented Linux backend.

  The install is wheels-only (`--only-binary=:all:`), so no package's code runs
  at install time.
* setup_pi.sh's single fallback: on a machine that is **not a Raspberry Pi**
  (`/proc/device-tree/model`) and only with `SENSORDYME_ALLOW_UNHASHED=1`, a
  failed hashed install falls back to an unhashed `requirements-device.txt`
  install, loudly. On a Pi it stops instead.
* Validated on the Mac (evidence `m85-evidence/deploy/`):
  * `pip download --require-hashes --only-binary=:all:` for `manylinux_2_*_aarch64`
    / cp311 fetched every locked package as a wheel whose hash is in the lock;
  * an offline resolution **with dependencies** from only those wheels succeeded,
    so the lock is complete;
  * the base lock fails that check on simpleaudio.
* Regenerating the lock: see its header. Keep the other versions with `-c <old
  pins>`, then re-run `make verify-release`.

## 6. Model files (M84-5)

* **MediaPipe `.tflite`** (6 files in the mediapipe wheel; face, hands, pose).
  The pins are in `core/model_registry.yaml`.
  * The cp311 aarch64 wheel that the lock pins (sha256 `1061d499…4315`) carries
    byte-identical files to the macOS wheel. Evidence:
    `mediapipe_tflite_sha256_aarch64_wheel.txt`. One pin per file therefore
    serves both platforms, and the wheel hash binds them at install.
  * At install time, the staged release runs `verify-artifacts` in its own venv,
    and a mismatch refuses the release.
  * At every boot, `main.py` calls `enforce_bundled_artifacts()` before any
    module is built. A file that is missing or differs replaces that MediaPipe
    solution class (FaceMesh / Hands / Pose) with a stand-in that loads nothing
    and finds nothing.
  * The runtime then blocks that perception component. Health shows the
    capability **impaired** with the reason, for example `model integrity:
    face_landmark_with_attention.tflite sha256 … is not the pinned …; refused`,
    and the journal and event log record `model_integrity_fault`. For a role
    that requires face, monitoring is FAILED and the impaired alert sounds.
  * If the check itself cannot run, every MediaPipe capability is refused.
  * To recover: reinstall the release (stage again) or switch to a good release.
    Never edit the pin to match.
* **openpilot `dmonitoring_model.onnx`** (benchmark module only). It is pinned
  to `3dd39829…8683`, the v0.9.7 file and the repo copy. A download is verified
  before it is moved into place, and the file is verified before every load. A
  mismatch is refused, the module cannot be built, and the device reports why.
* **Hailo `.hef`.** Unchanged: pinned in setup_pi.sh at download and in
  `core/accel.py` at load. It lives in the shared state (`models/`) and never
  arrives with a release.

## 7. Why the manifest is not signed

The release rests on this trust chain:

1. The operator's verified commit (`make verify-release`).
2. The manifest built from that commit's git objects.
3. The manifest's sha256, passed **on the SSH command line** of the stage step,
   from the operator's authenticated session.
4. Every file checked against the manifest before it is written.

A tampered upload, or a manifest swapped at rest in `sensordyme.incoming/`, is
refused (tested).

An HMAC or minisign signature would add protection only against someone who can
change the SSH command line. That attacker already controls the device. It would
also bring a key to provision, store and rotate on every device. minisign is not
installed on the Mac or the Pi. Decision: no signature for now. Revisit if
releases are ever pulled by the device rather than pushed by an operator.

## 8. What only a Pi can show (hardware-pending)

* The hashed install on a real Pi:
  * Bookworm's pip 23.0.x with `--require-hashes --only-binary=:all:` against
    this lock;
  * the time and disk cost of a per-release venv (about 300 MB of wheels; pip's
    cache makes repeat stages cheap; three releases are kept);
  * the interaction with `--system-site-packages`: an apt package at the exact
    pinned version satisfies the pin without a reinstall. To see which packages
    come from apt, run `pip list --path …/site-packages`.
* That `lgpio`'s manylinux_2_34 wheel drives the buzzer on a Pi 5 (previously
  installed by the same version pin, unhashed).
* Whether simpleaudio is currently in the device venv. A fresh release venv will
  not have it, so SoundSink uses `aplay`. Check with `aplay -l` and the R1 sound
  test.
* systemd (252) following the `~/sensordyme` symlink for `WorkingDirectory`,
  `ExecStartPre` and `ExecStart`, and `install_units.sh --app-dir`. Check with
  `systemctl show -p WorkingDirectory,ExecStart`.
* The detached activation surviving an SSH drop: disconnect during step 5, then
  run `make deploy-status`.
* Non-interactive sudo for the deploying user.
* `adopt` on the real tree. In particular, check that no code writes outside
  `data/`, `models/` or `logs/` at the tree root (anything else stays with the
  release it was written in).
* The least-privilege unit (`docs/LEAST_PRIVILEGE.md`) copies a checkout to
  `/opt/sensordyme` and bind-mounts `data/`. Adopting it together with staged
  releases needs its own plan (state dir vs `StateDirectory=`); the two have not
  been combined.

## 9. Tests

| What | Where |
|---|---|
| Staged install: verification refuses tampered, extra, missing, traversal, symlink, state, secret and a doctored manifest; atomic switch (one `rename(2)`); failing start rolls back by switching without pip; state carried over and never overwritten; config migrated in the new release only; sudo precheck; prune; `status --wait`; the real bundle stages, and a one-byte change in it is refused | `tests/test_m85_deploy_staged.py` |
| Lock and install paths: setup_pi.sh block with a fake pip (hashed argv; no fallback on a Pi; dev-box fallback only on request); guard argv; legacy rollback installs the lock or nothing; gate on this tree and on `87d91b3`; lock completeness with markers; release-verify against the lock | `tests/test_m85_deploy_lock.py` |
| Model files: verification, fail closed at boot (never constructed), runtime health impaired with reason, openpilot pin at download and at load | `tests/test_m85_deploy_artifacts.py` |
| End to end on the Mac: `deploy.sh` → fake ssh/scp/sudo/systemctl → adopt, healthy deploy confirmed, crash-looping deploy rolled back, tampered upload refused, re-deploy is a no-op | `tests/test_m85_deploy_e2e.py` (about 30 s; not in verify-release) |
