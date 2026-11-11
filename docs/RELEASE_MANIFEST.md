# Release manifest (M6)

`scripts/release/make_release_manifest.py` answers two questions with hashes,
not memory:

1. **What exactly does release candidate X put on a device?** (`build`)
2. **Does this device hold exactly that?** (`verify`)

Before M6, a deploy was `make deploy`. It rsyncs a developer's working tree,
which may include uncommitted edits. It runs without `--delete`, so a file
removed from the repo stays on the device. And nothing on the device checks
what arrived.

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
| `requirements` | `requirements-device.txt` sha256 and its pins, plus the extra pins `setup_pi.sh` installs (gpiozero, lgpio) |
| `systemd_units` | sha256 of the `sensordyme.service` template and `deploy/systemd/sensordyme-recover.service` |
| `test_runs` | Free-text references to the test runs that back this release (`--test-ref`, repeatable) |

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

* The manifest is not signed. It proves what the device holds only relative to a manifest you trust. Store it with the release record.
* The unit check guesses the service user from `/home/<user>/…`. An install elsewhere reports INFO, not PASS.
* `pip freeze` in a `--system-site-packages` venv also lists apt packages. Only the pinned names are compared.
