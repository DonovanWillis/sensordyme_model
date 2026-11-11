# Device credential storage and migration plan

Mission 6.5, checkpoint B/4C. The code is `core/secret_store.py`, with
`scripts/migrate_device_key.py` and the tests in `tests/test_secret_store.py`.
This closes the device half of M6 blocker 3 ("plaintext device key, config.yaml
not 0600"). **No real device, config or key has been migrated, changed or
rotated.**

## 1. Today

`cloud.device_key` sits in `config.yaml` in plaintext. The file's mode follows
the umask (typically 0644), and `make deploy` never touches it. The dashboard
token lives there too.

## 2. The credentials file

`cloud.device_key_file: /etc/sensordyme/credentials.json`. Any path works if
the service user owns it.

```
{"format": "sensordyme-credentials", "version": 1,
 "secrets": {"device_key": "...", "previous_device_key": "..."},
 "sha256": "<SHA-256 of the canonical secrets>"}
```

| Property | How |
|---|---|
| Permissions | Written 0600 from the first byte (`mkstemp` + `fchmod` before any data). On load it must be a regular file, not a symlink, owned by the service user, with no group or other bits. Strict mode, which the runtime uses, refuses anything else |
| Atomic updates | Temp file in the same directory, `fsync`, `os.replace`, then a directory `fsync`. A crash leaves either the old or the new file, never a mix (tested by failing `os.replace`) |
| Backup | Before an update, the current file, if valid, is re-written as `<file>.bak` (0600). A damaged file is never copied over a good backup |
| Integrity | The checksum detects truncation, zero-length files, hand edits and non-UTF-8 content |
| Recovery | A damaged primary loads from `.bak`, with a warning that names the repair command. If both are bad, `CredentialError`: the runtime runs **standalone with the uplink off** and states the reason. It does not crash-loop, and it never falls back to "no key" |
| Redaction | Values are held in `Secret`; `repr`, `str`, `format`, `%s` and containers all show only `<redacted, kid …>`. A `Secret` cannot be pickled. `redact()` and `RedactingFilter` scrub known secrets from free text and log records |
| Config hygiene | At startup, `main.py` warns when `config.yaml` holds a device key or dashboard token and is group- or other-readable. It warns rather than refuses, so a working device is never taken offline over a file mode |
| Schema | `device_key` and `device_key_file` together is an error, because it is ambiguous. With a key file, an empty `device_key` is fine |

## 3. Migration plan (per device; operator-run; not done)

**Pre-checks:**
* The device runs a build that includes this change (the runtime resolves
  `device_key_file`).
* A backup of the device's `config.yaml` exists and is treated as a secret.
* Note the device's key id now; it is printed by the dry run.

**Steps:**
1. **Dry run, as the service user, on the device:**
   `python scripts/migrate_device_key.py --config /home/<svc>/sensordyme/config.yaml --store /etc/sensordyme/credentials.json`.
   It prints the planned steps and the kid, never the key. It writes nothing.
2. **Apply:** add `--apply`. This:
   * writes the store (0600, atomic) and reads it back;
   * keeps `config.yaml.pre-migration` (0600, still holds the key);
   * rewrites the config by line edit, so comments survive. It blanks
     `device_key` and adds `device_key_file`. The result must re-parse to
     exactly the original values except those keys, otherwise nothing is
     written. The config keeps its own mode.
3. **Verify:**
   * restart the service;
   * the journal shows `cloud uplink -> …` and no `credentials:` warning;
   * the app shows the device online.
4. **Clean up:** delete `config.yaml.pre-migration`.
5. **Optional, and recommended when the device moves to
   `cloud.command_auth: required`:** rotate the key, following
   `docs/DEVICE_AUTH_CONTRACT.md` §6, because the old key has travelled as a
   bearer.

**Rollback:** `cp config.yaml.pre-migration config.yaml`, then restart. Leave
the store in place, or delete it.

**Failure handling:**
* A config with no `cloud:` block, an already migrated config, an empty key,
  or a key that is not on a single line (for example a block scalar) is
  refused with a reason, and nothing is written.
* A damaged store after migration: `--restore-backup <store>`.

## 4. Still open

* **Where the store should live, and its owner.** This depends on the
  least-privilege unit (checkpoint 4D, another agent):
  * `/etc/sensordyme/` owned by the service user, mode 0700;
  * or systemd `LoadCredential=`, which would make the file readable only by
    the unit.
  The code accepts any path the service user owns.
* **Not encrypted at rest.** Physical access to the SD card still yields the
  key (M6 blocker 6, disk encryption). The file only stops other local users
  and accidental exposure in backups or diffs of `config.yaml`.
* **The dashboard token** stays in `config.yaml`. Only the startup warning
  covers it; moving it is a follow-up using the same store.
* **The backend side** (device keys in plaintext in SQLite) is unchanged; see
  `docs/DEVICE_AUTH_CONTRACT.md` §9.
