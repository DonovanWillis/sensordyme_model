# Release manifest (M6)

`scripts/release/make_release_manifest.py` answers two questions with hashes,
not memory:

1. **What exactly does release candidate X put on a device?** (`build`)
2. **Does this device hold exactly that?** (`verify`)

Before M6, a deploy was `make deploy`. It rsynced a developer's working tree,
which could include uncommitted edits. It ran without `--delete`, so a file
removed from the repo stayed on the device. And nothing on the device checked
what arrived.

**M8.5:** `make deploy` no longer rsyncs anything. It ships a **bundle** built
by this tool from git objects (`bundle`, below) and the device stages it,
verifying every file against the manifest before it is written. The procedure
is `docs/DEPLOYMENT.md`.

## build (Mac / worker)

```bash
make release-manifest REF=m6/autonomy-security-runbook \
     TESTS="M6 focused: 26 files, N passed (2026-11-11, Mac)"
# -> ~/sensordyme-release/manifest.json and manifest.md
```

The tool reads git objects only (`git archive`), never the working tree, so an
uncommitted edit can never enter a manifest. It records:

| Field | Meaning |
|---|---|
| `commit`, `branch`, `ref` | What was described |
| `deploy_set.files` | `{path: {sha256, sha256_lf}}` for every file `make deploy` would send. Excludes are reused from `scripts/hw_validation/integrity_manifest.py`, which the device snapshot also uses (`.venv`, `data`, `.git`, `ios`, `models`, `config.yaml`, `*.hef`, `*.pt`, `*.db`, `*.csv`, secrets, …) |
| `deploy_set.aggregate_sha256` | One hash for the whole set: sha256 of the sorted `path sha256` lines |
| `config` | sha256 of the repo's default `config.yaml`. A deploy **never** overwrites a device's own config, so this is a reference, not something to install |
| `model_pins` | `accelerator.yolo_hef(_sha256)`; `setup_pi.sh` `MODEL_SHA256` and `MODEL_SHA256_EVALUATED`; the model-registry artifacts; and `consistent` (config = setup = registry for the `.hef`). `build` exits 1 when the pins disagree |
| `decoder` | `accelerator.yolo_decoder` (`legacy` is the default. `reference` is opt-in and not INT8-validated; see `docs/DECODER_CANDIDATE.md`) |
| `requirements` | `requirements-device.txt` sha256 and the pins that apply **on the device** (M8.5: markers evaluated for Linux/aarch64/CPython 3.11; e.g. `simpleaudio` is listed under `not_on_device`). gpiozero and lgpio are device pins since M8.5 (they used to be a separate `setup_pi.sh` install) |
| `systemd_units` | sha256 of the `sensordyme.service` template and `deploy/systemd/sensordyme-recover.service` |
| `test_runs` | Free-text references to the test runs that back this release (`--test-ref`, repeatable) |

## M6.5 additions: release facts, gates, and one verification command

`build` now also records:

| Field | Meaning |
|---|---|
| `protocol` | Declared versions (device runtime `_VERSION`, backend FastAPI `version`, iOS `MARKETING_VERSION`, API prefix `/v1`) and every `/v1` route the device (`core/uplink.py`) and the iOS app (`ios/Sources/*.swift`) call, matched against the backend's routes (`cloud/app.py`). `compatible` is false when a client calls a route the backend lacks |
| `dependency_lock` | `requirements-device.lock` sha256, package count, pins that disagree with `requirements-device.txt`, packages without a hash |
| `config_schema` | `core.config_schema.validate` on the repo `config.yaml` (errors and warnings) |
| `status_tiers` | `scripts/release/release_status.yaml`: per component, one or more of software-ready / simulator-verified / hardware-pending / experiment-only / production-blocked, with evidence. Six `pending_integration` slots wait for the other M6.5 workstreams (device auth, BLE PoP, XCUITest, reliability, registry binding, the EXP-08 training run) |
| `known_good` | The configuration the release was checked with: config hash, role/module/profile, decoder, `.hef` path and pin, device-requirements hash, which unit |
| `procedures` | Where rollback, identity migration (none in this release) and the least-privilege migration are documented |
| `systemd_units` | Now also the least-privilege unit and its optional drop-in |

`check --manifest m.json` runs the **release gates** (exit 1 on any FAIL): model pins
consistent; decoder is `legacy`; protocol compatible; lock consistent and fully hashed; config
passes the schema; status tiers valid and free of hardware-validation claims. An input that does
not exist at an older ref is INFO, never a silent PASS.

**One command** for all offline checks: `make verify-release [REF=…]`
(`scripts/release/verify.sh`): device dependencies → manifest build → gates → systemd units
(`systemd-analyze verify` + offline exposure score; SKIP on macOS) → focused release tests.
About 15 s on the Mac, no downloads, changes nothing. `KEEP_MANIFEST=<dir>` keeps the manifest.

## M8.5 additions: the bundle, hashed install paths, the lock on the device

`bundle --ref <ref> --out-dir <dir>` writes the release a device stages:

| File | Meaning |
|---|---|
| `sensordyme-<id>.tar.gz` | EXACTLY the manifest's deploy set, from git objects. Regular files only, mode 0644/0755, owner 0, mtime = commit time, so the same commit always gives the same bytes |
| `sensordyme-<id>.manifest.json` / `.md` | The manifest the device verifies every file against. Its sha256 travels on the stage command line, not in a file |
| `SHA256SUMS`, `RELEASE_ID`, `COMMIT` | The tarball's and manifest's sha256; `<id>` = commit[:12]; the full commit |

`bundle` exits 1 when a release gate FAILs (`--allow-failed-gates` writes it anyway; never for a device).

New in the manifest and the gates:

| Field / gate | Meaning |
|---|---|
| `install_paths` | Every pip install in `setup_pi.sh`, `Makefile`, `scripts/release/release_guard.py` and `scripts/release/install_units.sh`. Gate **"every device install path uses pip --require-hashes with the lock (M84-4)"**: FAIL on a non-comment unhashed install of `requirements-device.txt`, a bare `pkg==ver`, or `--upgrade pip` (a line marked `unhashed-fallback` is the documented dev-box exception), or a file that installs without the lock. INFO for a manifest older than M8.5 |
| `dependency_lock.locked` | Every locked package and version. `verify` adds the check **"installed packages = the hashed lock"**: the device venv's `pip freeze` must show every locked version (pip/setuptools/wheel excepted, freeze hides them). A venv that did not come from the lock FAILs |

`scripts/release/check_device_deps.py` (verify-release step 1) now also checks the lock against
the device pins and imports (`docs/DEPLOYMENT.md` §5).

## verify (device snapshot → Mac)

On the Pi, read-only, with the M5 integrity script (it never copies contents;
`config.yaml` is hashed, not read out):

```bash
$PY $V/integrity_manifest.py snapshot --root $D --expect-role driver_safety --out ~/hwval/snap.json
```

Then on the Mac:

```bash
scp <user>@<pi>:hwval/snap.json ~/sensordyme-release/
make release-verify SNAPSHOT=~/sensordyme-release/snap.json
```

| Check | FAIL when |
|---|---|
| deployed files | Any file differs, differs only in line endings, is only on the device (a stale file), or is missing on the device |
| model hash = release pin | The `.hef` on the device is not the pinned build (for example the "evaluated" build of unverified provenance), or there is none |
| device config pins the release model | The device's own `accelerator.yolo_hef_sha256` is not the release pin |
| installed requirements | A pinned package is missing or at another version in the service venv's `pip freeze` |
| installed sensordyme.service | The installed unit is not this release's template rendered for the install directory and user. Fix with `sudo bash scripts/release/install_units.sh --user <user>` |
| device config preflight | `scripts/deploy_preflight.py` refuses the device's config |

The device config itself is reported as INFO (hash and role). It is device
state and is not expected to match the repo default. So are the
`config.yaml.bak-<stamp>` backups that `scripts/migrate_config.py` writes beside
it: they are listed as INFO, never as drift. That exemption is exact: a backup
anywhere else is still drift.

Exit code: `0` = PASS, `1` = FAIL. Tests: `tests/test_m6_security_release_manifest.py`.

## Limits

* The manifest is not signed. It proves what the device holds only relative to a manifest you trust. Store it with the release record. For a staged deploy, the device checks that the manifest's sha256 equals the value the operator passed over SSH. `docs/DEPLOYMENT.md` §7 explains why that, and not a signature, is the trust anchor.
* On a staged device the snapshot root is `~/sensordyme`, which is a symlink to the current release. Snapshot it as before: `integrity_manifest.py snapshot --root ~/sensordyme`. The `STAGED.json` and `ARTIFACTS.json` beside the code are release records, not code. `verify` reports them as INFO, as it does `config.yaml.bak-*`.
* The unit check guesses the service user from `/home/<user>/…`. An install elsewhere reports INFO, not PASS.
* `pip freeze` in a `--system-site-packages` venv also lists apt packages. Only the pinned names are compared.
