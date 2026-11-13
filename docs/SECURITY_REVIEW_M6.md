# Security review — Mission 6 (device and backend-link layer)

Branch `m6/autonomy-security-runbook`, based on the release candidate
`integration/m6-rc` (`c01e43b`). Scope: narrow hardening that can be verified in
tests at the **device / backend-link** layer. No UI, no iOS, no `cloud/app.py`
endpoints, nothing executed on a Pi. This does not redo the 2026-11-10 audit
(`~/sensordyme-audit/2026-11-10/07_SECURITY_PRIVACY_AND_THREAT_MODEL.md`); it
closes the parts of it that are fixable on the device side and states what is
left.

Every row marked **fixed** has a test that fails on the old code.

## 1. Findings addressed

| # | Finding (audit ref) | Change | Proving tests |
|---|---|---|---|
| S1 | Dashboard token accepted in the `?token=` query string on **every** path, POST included (07 "Additional findings"). A link or a browser-history entry could change the preset, stop monitoring (`/api/protection`) or re-provision Wi-Fi (`/setup/wifi`). | `core/dashboard.py` `_authorized`: the query fallback is honoured only for `GET /stream` (the MJPEG `<img>`, which cannot send a header). Everything else needs `X-Device-Token`. The page JS, the iOS app (`MJPEGStream.swift`, `DiagnosticsView.swift`) and the scripts (`demo_preflight.py`, `demo_cycle.sh`) already send the header, so nothing that pairs today breaks. | `tests/test_m6_security_device.py::test_a_query_token_no_longer_opens_the_api`, `::test_a_query_token_cannot_authorize_a_write` (3 write paths), `::test_a_query_token_still_opens_the_video_stream`; `tests/test_dashboard.py::test_the_query_token_is_url_decoded` now exercises `/stream` |
| S2 | Wi-Fi credentials from an unauthenticated BLE write (or the dashboard) went to nmcli argv and to the journal unvalidated (07 D12/F1): a NUL raised out of `subprocess`, a newline in an SSID forged journal lines, a 400-byte "SSID" reached NetworkManager. | `core/network.py` `credential_problem()`: SSID 1–32 UTF-8 bytes, no control characters; password ≤128 bytes, no control characters; `None` = open network. Called first in `WifiManager.connect()` and `.provision()`, so the refusal reaches the app as the BLE provisioning error. `services/ble_gatt.py` `dispatch_credentials()` logs the SSID with `%r`, clipped to 64 characters, and never logs the passphrase. | `tests/test_m6_security_device.py::test_credential_problem` (14 cases), `::test_bad_credentials_never_reach_nmcli`, `::test_the_join_reports_the_refusal_to_the_app`, `::test_a_credential_write_never_logs_the_passphrase_or_a_raw_newline`, `::test_a_malformed_credential_write_is_ignored` |
| S3 | Replayed or re-delivered `set_protection` (stop/resume) applied again (07 F4, "untrusted commands"). The backend re-sends until the device echoes the id, and a lost response or a replay delivers the same id twice. A stale **stop** after the owner resumed locally would silently stop monitoring again. | `core/uplink.py` `_accept_protection()`: each protection command id is applied at most once (bounded memory of 64 ids). Commands without an id (older backends) are still accepted. | `tests/test_m6_security_uplink.py::test_a_protection_command_id_is_applied_once`, `::test_distinct_ids_are_each_applied`, `::test_a_command_without_an_id_is_still_accepted_for_older_backends`, `::test_the_seen_id_memory_is_bounded` |
| S4 | Malformed remote command fields were handed to the main loop: a non-string `set_preset` (dict, list, number, 500-character string) and a non-mapping `command`. | `core/uplink.py`: `set_preset` must be a string of 1–128 characters; a non-mapping `command` is ignored. | `tests/test_m6_security_uplink.py::test_a_malformed_set_preset_is_ignored` (4 cases), `::test_a_valid_set_preset_still_arrives`, `::test_a_non_mapping_command_is_ignored` |
| S5 | Plain-`http://` uplink to a network host carries the device key and every command unauthenticated, silently (07 F3/F5). | `core/uplink.py` `cleartext_remote()`. The uplink announces it once at startup and the deploy preflight warns on every run. Not refused, because LAN development backends are `http://*.local`. | `tests/test_m6_security_uplink.py::test_cleartext_remote` (7 cases), `::test_a_cleartext_uplink_is_announced_once`; `tests/test_m6_security_preflight.py::test_exposures_are_warned` |
| S6 | The Mac-only `mac_max` profile, the one path that uploads frames (VLM judge, Gemini by default; 07 F2), could be configured on a device. On a Pi it also cannot start (no torch/MPS), so it would crash-loop the unit. | `scripts/deploy_preflight.py` `_device_profile_and_exposure()`: any `profile` other than `pi` is an **error** (no deploy). | `tests/test_m6_security_preflight.py::test_the_developer_profile_is_refused_on_a_device`, `::test_the_device_image_cannot_run_the_vlm_judge` (no torch in `requirements-device.txt`; the `_max` module is built only after `require_mps()`), `::test_the_uplink_payload_has_no_image_field`, `::test_the_repo_config_keeps_frame_capture_and_frame_upload_off` |
| S7 | Exposures that are legitimate on a development unit but must be seen at deploy time. | Deploy-preflight **warnings**: LAN dashboard without a token, or with a token under 16 characters; `http://` uplink; `tls_insecure_skip_verify: true`; on-disk frame capture (`alerts.save_evidence`, `training.capture.enabled`). | `tests/test_m6_security_preflight.py::test_exposures_are_warned` (6 cases), `::test_a_hardened_config_has_no_exposure_warnings`, `::test_warnings_do_not_refuse_a_deploy` |
| S8 | Unsigned, unverifiable deploys: `make deploy` rsyncs a working tree with no `--delete` and no on-device verification (07 F11). | `scripts/release/make_release_manifest.py`. `build` hashes exactly what a git ref deploys (git objects only, never the working tree; same excludes as `make deploy`) plus the model pins, decoder, requirements and units. `verify` compares a read-only device snapshot against it and FAILS on a modified, stale-extra or missing file, a line-ending-only difference, a wrong or missing model, a device config pinning another model, a wrong installed version, a stale unit, or a failing device-config preflight. | `tests/test_m6_security_release_manifest.py` (20 tests) |
| S9 | Root-executed recovery code must not be editable by the service user. | The new start-limit guard (see autonomy A2) runs as root from a **root-owned copy** at `/usr/local/lib/sensordyme/`, never from the service user's checkout. | `tests/test_m6_autonomy_boot.py::test_the_recovery_unit_runs_a_root_owned_copy_not_the_checkout` |
| S10 | Secrets in logs. | Regression tests for the existing controls. The device key never appears in uplink output on connect failure, 401, 500 or redirect, and the Wi-Fi passphrase never appears in BLE logs. | `tests/test_m6_security_uplink.py::test_the_device_key_never_reaches_the_log` (4 paths); `tests/test_m6_security_device.py::test_a_credential_write_never_logs_the_passphrase_or_a_raw_newline` |

Verified existing controls, re-asserted by the M6 tests:
* The dashboard fails closed on a LAN bind with no token, on all five protected GET paths and on POST.
* Token comparison uses `hmac.compare_digest`.
* The repo `config.yaml` ships `dashboard.token: null`, so no token is shared across units.
* Frame capture is off by default.
* The uplink payload has no image field.

## 2. iOS / pairing impact (migration note; no app change made)

* **S1.** No change is needed. The app sends `X-Device-Token` on `/stream` and on every `/api` call. A third-party client that put `?token=` on `/api/*` or on a POST now gets **401** and must switch to the header.
* **S2.** The app's credential payload `{"ssid","psk"}` is unchanged. An SSID over 32 bytes, or one containing control characters, is refused with a readable error on the status characteristic (`provision.state: "error"`, `error: "network name is longer than 32 bytes…"`). Any SSID the iPhone can see is valid by definition.
* **S3.** Backend ids are random per request (`secrets.token_hex(6)` in `cloud/app.py`), so ordinary use never repeats an id.
* A device whose dashboard token is under 16 characters still works. The deploy preflight warns about it. To rotate, generate a new token (`openssl rand -hex 16`), put it in the device's `config.yaml`, restart, and enter it in the app under Settings → Live feed. Rotation is an operator action, not done by this branch.

## 3. Remaining risks (not fixed here)

| Risk | Why not fixed in M6 | Mitigation in place / next step |
|---|---|---|
| **BLE provisioning is unauthenticated and unencrypted** ("Just Works"). Anyone in range during the window can move the device to their own AP (07 F1). The window is open whenever the device is offline, which is a whole drive for an in-car unit, plus 300 s after every boot. | Needs LE Secure Connections with passkey/OOB, or an app-level HMAC with a per-device claim secret. That is an iOS + BlueZ agent change and cannot be verified without the hardware. `services/provisioning.py` (the window policy) is outside this mission's file ownership. | Input is now validated (S2). Next: a passkey printed on the label, and opening BLE on a button press or first boot only. **Production blocker.** |
| **Commands are not signed end-to-end.** Over `http://` (the LAN default) a MITM can inject `set_preset` / `set_rules` / `set_protection` (07 F3/F4). | Signing needs a backend key-management change in `cloud/app.py` (another agent's file) and an app-side change. | HTTPS with verification (`ca_cert_path`) is supported; S3 stops duplicate application; S5 makes cleartext visible. **Production blocker:** HTTPS mandatory outside loopback, plus signed commands. |
| **Device key at rest in plaintext** on both ends, with no rotation (07 F5). Device `config.yaml` permissions follow the umask. | The backend hashing is in `cloud/app.py`; a device-side `chmod 0600` would change the file mode of a file `make deploy` deliberately never touches. | The runbook (§5, step B4) checks the mode and records it. Next: hash the key server-side, add rotation, `chmod 0600 config.yaml` in setup. |
| **Service hardening** (07 F7). The service runs as the interactive admin user, with no `ProtectSystem`/`NoNewPrivileges`. | `NoNewPrivileges` breaks the sudoers-based reboot and btmgmt. Sandboxing needs testing on the real image. | Next: a dedicated system user and a polkit/root helper for reboot and btmgmt. The runbook records `sudo -l -U <user>`. |
| **SD card unencrypted** (07 F8): config, NetworkManager PSKs, events, telemetry. | Needs LUKS plus a key store and a physical threat model. | The runbook's backup and image steps treat the backup itself as a secret. |
| **Persistent tracking id** in BLE name, mDNS TXT and health (07 F9). | Changing it breaks pairing identity (the app keys on `SD-XXXXXXXX`). | Next: rotating resolvable addresses, and disclose the id only after authenticated pairing. |
| Telemetry is always on, and `events.csv` grows without bound (07 F6). | Behaviour change across modules; out of scope. | Documented. |
| `anthropic==0.116.0` and `pytest==8.3.3` are in `requirements-device.txt` (07 F11): unused on the device, and pytest has an advisory. | `requirements-device.txt` is not in this mission's file set, and removing a pinned dependency needs a device run. | The manifest records both. Next: remove both from the device set. |
| The Wi-Fi PSK appears in nmcli argv, readable via `/proc/*/cmdline` during the join. | Needs `nmcli --ask` or a keyfile, verified on NetworkManager. | Documented (low). |
| The uplink `health` field is unbounded at ingest (07 DoS row). | Backend change. | Documented. |

## 4. Production blockers (security)

1. Authenticated BLE provisioning (passkey/OOB or app-level HMAC), and a provisioning window that does not open just because a car has no Wi-Fi.
2. HTTPS mandatory for the uplink outside loopback, with signed, or at least MAC'd, commands.
3. Device keys hashed at rest on the backend, with rotation; `config.yaml` mode 0600 on the device.
4. A dedicated, sandboxed service user.
5. Remove `pytest` and `anthropic` from the device requirements.
6. Disk encryption for secrets and personal data, or an explicit, documented acceptance of the physical-access risk.

None of these can be closed by a device-only software change verified on a Mac.

## 5. Mission 8 review (security hardening and compatibility)

Branch `m8/security`, from `11c524c` (M0.1…M7). Scope:

* device auth (`core/command_auth.py`, uplink verification);
* backend auth, claim and ingest verification (`cloud/app.py`, `cloud/manage.py`, `cloud/ratelimit.py`);
* credential storage;
* a review of BLE proof of possession, which belongs to the onboarding stream.

Each fix has a test that fails on `11c524c`. Evidence is in
`~/sensordyme-audit/2026-11-10/m8-evidence/security/`:

* the repro outputs;
* `compat_matrix.json`, with 56 live cells;
* `e2e_transitions.json`, the two scenarios against a live uvicorn backend.

| # | Finding | Severity | Fix | Proving tests |
|---|---|---|---|---|
| M8-1 | The backend kept sd-auth-v1 request nonces in process memory. A captured signed request was accepted again after a backend restart, within ±300 s. Reproduced: `200` after a restart, where the same process answers `401 replayed_nonce`. | Medium | Nonces are stored in the `auth_nonces` table. `INSERT OR IGNORE` on the primary key is the atomic test-and-set. Rows are pruned after 600 s, and the table is bounded at 1 M rows. The store is shared across processes. | `tests/test_m8_security_backend.py::test_a_captured_request_is_refused_after_a_backend_restart[permissive,required]`, `::test_stored_nonces_are_pruned…`, `::test_the_stored_nonces_are_bounded`; E2E 2d/2e |
| M8-2 | A `required` device adopted the unsigned `server_time` of any `clock_skew` answer. An on-path attacker (plain `http://`) could skew the device's request timestamps forward, hold the signed requests, and deliver them later as fresh, for example to fake heartbeats while the device is cut off. | Medium (http only; TLS covers it) | `clock_auth`: the backend signs its time with the command key, bound to the refused request's nonce. The device adopts only a verified time. | `tests/test_m8_security_device.py::test_an_unsigned_clock_answer_is_not_adopted`, `::test_a_forged_or_replayed_clock_answer_is_not_adopted` (4), `::test_a_signed_clock_answer_is_adopted…`; `tests/test_command_auth_uplink.py::test_a_clock_skew_refusal_teaches_the_request_clock_only` (updated to the signed form) |
| M8-3 | Revocation was not visible. A revoked previous key got `unknown_key` or `invalid device key`, and the app showed only "offline" after 30 s. There was no way to revoke a device's current credential short of rotating it to a key nobody holds. | Medium (truthfulness, ops) | Revoked keys are remembered, giving `device_auth_revoked_key`. `manage.py revoke-device` refuses every request, giving `device_auth_revoked_device`, and `rotate-key` re-instates the device. `GET /v1/devices` gains `credentials`, and a stale status names the refusal. Only refusals proven to come from the key holder are recorded. | `tests/test_m8_security_backend.py::test_a_revoked_previous_key_is_refused_as_revoked_and_the_app_is_told`, `::test_a_forged_tag_under_a_revoked_kid_is_not_reported_against_the_device`, `::test_a_revoked_device_cannot_ingest_or_receive_commands_until_reissued`; E2E 1e/1f |
| M8-4 | `required` could only be enforced for the whole backend. | Low (migration) | `devices.auth_mode`, set with `manage.py require-signed` or `allow-bearer`. It only tightens, and is refused until the device has been seen signing (`--force` overrides). | `::test_signed_requests_can_be_required_for_one_device`, `::test_a_device_cannot_loosen_a_required_backend`; matrix rows `new/permissive+device-required` |
| M8-5 | Claim-code guessing was limited per address only. One account rotating addresses had no limit, for example a botnet, or a trusted proxy header naming a new client each time. | Low (48-bit codes) | `CLAIM_FAILURE_LIMITER`: 20 failed claims per account per hour. Only unknown codes and devices owned by another account count. | `::test_claim_guessing_is_limited_per_account_across_addresses`, `::test_a_successful_claim_is_not_a_failure_and_another_account_is_refused` |
| M8-6 | An existing backend JWT secret file kept whatever mode it had (for example 0644 after a restore). Anyone who can read it can forge sessions. | Low (local) | The file is set to 0600 at startup, with a warning. | `::test_an_existing_jwt_secret_file_is_made_owner_only` |
| M8-7 | `scripts/migrate_device_key.py` printed PyYAML's traceback for a config that does not parse. The traceback quotes the offending line, which can be the device key. | Low (operator terminal) | The script refuses with line and column only. | `tests/test_m8_security_credentials.py` (3) |

Re-verified with tests that pass on both `11c524c` and M8, kept as regression pins:

* A captured signed command is refused after a device restart (`stale_response`), with no on-disk replay state. The `rn` binding and per-request random nonces make that state unnecessary.
* One device key cannot fetch or sign another device's commands.
* Old backends that never sign are still obeyed by `off` and `permissive` devices.

Unchanged by design, and documented:

* `permissive` obeys a stripped envelope. Use device `required` for that.
* Device keys are stored in plaintext in SQLite (symmetric scheme).
* The claim code is never rotated. No unclaim or transfer endpoint exists, so re-claim is limited to the same account (idempotent) and a 409 for another account. Any future transfer endpoint must rotate the claim code.

**Compatibility.** The full matrix is in `docs/DEVICE_AUTH_CONTRACT.md` §7. Old and new code behave identically in every cell except one. An M8 `required` device behind an 11c524c backend cannot recover from a clock more than 300 s off. This follows from M8-2 and is avoided by the backend-first migration order. No migration is enforced, no default changed, and no key rotated.

**Onboarding stream (review only, not changed here).** `services/ble_pop.py` `load_setup_code` turns only lstat/format problems into `SetupCodeError`. An unreadable code file (mode 0000 owned by the service) or a non-ASCII one raises `PermissionError` or `UnicodeDecodeError`. `services/provisioning.py._open_ble_window` catches only `(SetupCodeError, ImportError)`, so the provisioning thread dies (`_guarded_run`: "crashed") instead of retrying with Bluetooth closed. The failure is closed, so it is not a bypass, but it is an availability defect. The suggested fix is to wrap the open/read in `except (OSError, UnicodeDecodeError) as exc: raise SetupCodeError(...) from None`. The rest of the protocol review found no defect:

* one guess per challenge;
* a device-held lockout capped at 1 h;
* the device proves itself before credentials are sent;
* a fixed PBKDF2 cost;
* no code or tag in logs;
* a 0600 / no-symlink / owner check, and an O_EXCL+link write.

The documented limits stand: offline guessing from a recorded session, a lockout that resets on power cycle, and a default of `auth: legacy` outside `deployment: production`.

**Still needs hardware.** Everything in §4, plus the following:

* sd-auth-v1 and `clock_auth` on a Pi with no RTC and a real NTP-less boot;
* the `auth_nonces` write cost at fleet poll rates on the deployed backend;
* BLE PoP over a physical link;
* `revoke-device` and `require-signed` exercised on a deployed backend.

## 6. Mission 8.3 review (authorization, logs, client storage, evidence)

Base `66cf6b6`. Scope: every `/v1` route's authorization, stream access, credential storage on the
device, the backend and the phone, token handling, BLE provisioning logs, transport, log contents,
evidence retention and the in-app diagnostics. Replay resistance (§5) was spot-checked only: the M8
suites pass unchanged. Each fix has a test that fails on `66cf6b6`. Evidence is in
`~/sensordyme-audit/2026-11-10/m83-evidence/security/`, and it holds no secrets.

| # | Finding | Severity | Fix | Proving tests |
|---|---|---|---|---|
| M83-1 | A failed ntfy push logged `%r` of the `requests` exception. Its text is the request URL `https://ntfy.sh/<topic>`, and the topic is the whole credential of the account's alert channel. A failed Twilio send logged the account SID the same way. | Medium (backend log readers can subscribe to an account's alerts) | `core.logging_setup.describe_failure` logs only the exception type and HTTP status. | `tests/test_m83_security_logs.py::test_a_failed_push_never_logs_the_ntfy_topic`, `::test_a_failed_sms_never_logs_the_twilio_account` |
| M83-2 | With no Twilio configured, the backend's log-only SMS path printed every account's phone number on every serious alert. | Low (PII in logs) | `mask_phone` keeps the last two digits. | `::test_the_log_only_sms_masks_the_phone_number` |
| M83-3 | `subprocess.TimeoutExpired` quotes nmcli's argv. `WifiManager._roll_back` logged `could not restore the previous network: %s`, which put the restored Wi-Fi passphrase in the journal when `connection modify … wifi-sec.psk` stalled. | Medium (PSK in journald) | `core.network._run` re-raises the timeout with every passphrase argument replaced by `<redacted>`. The join paths already caught the timeout and are now pinned. | `::test_a_stalled_rollback_never_logs_the_restored_passphrase`, `::test_a_stalled_join_never_returns_or_logs_the_passphrase[connect,provision]`, `::test_the_redaction_keeps_the_rest_of_the_command` |
| M83-4 | `/v1` answers carried no `Cache-Control`. The app sends everything through `URLSession.shared`, whose default `URLCache` stores GET answers on disk. Reproduced on Foundation: the answer is cached without the header and not cached with `no-store`. A device list, an alert history or an explanation outlived sign-out and an account change on a shared phone, and an HTTP cache in front of the backend could replay them. | Low–Medium (privacy at rest) | `_NoStore`, a pure ASGI middleware, sets `Cache-Control: no-store` on every `/v1` response. | `tests/test_m83_security_routes.py::test_every_v1_answer_is_marked_no_store` |
| M83-5 | The phone's Keychain items (the 30-day session JWT, the device dashboard token that opens the camera, the ntfy topic) were `AfterFirstUnlock`. An encrypted iCloud or Finder backup restores that class onto any phone. | Low–Medium (backup holder gets a live session and the camera) | `AfterFirstUnlockThisDeviceOnly`, which background polling can still read. Never synchronizable. Updates carry the class, and `get()` re-classes an older item on first read. The cost: a restore to a new phone signs in once and re-enters the device token. | `ios/Tests/KeychainTests.swift` (attributes only, no real keychain) |

**Route-authorization matrix.** `tests/test_m83_security_routes.py` enumerates FastAPI's routes:
26 routes, 3 public, 1 device key, 9 account, and 13 device-scoped (12 by path, plus
`/v1/ai/configure` by body). A route that nobody has classified fails the suite. Every non-public
route must depend on `current_token` or `current_device`. Each device-scoped route is driven with a
valid body three ways:

* as a second account on the owner's device;
* as that account on a missing device;
* anonymously.

All 13 answer 404, 404 and 401. The two 404s carry identical details, so there is no ownership
oracle, and nothing is written. Further pins cover four cases:

* `explain` through the caller's own device id with another account's event;
* `GET /v1/devices` and `/v1/orders` show only the caller's rows, never a claim code or key;
* claiming another account's code is a 409 and ownership does not move;
* a device key reaches no account route.

**No defect found** in authorization. Every scoped handler goes through `_owned_device` in the
transaction it writes in. `ai_configure` re-checks `owner_id` in its UPDATE. No transfer or unclaim
endpoint exists, which leaves no TOCTOU window.

**Log sweep.** Signup, both login outcomes, claim (unknown and valid), ntfy and phone settings,
ingest (good key, wrong key, stale sd-auth-v1), logout and a revoked-session request are captured
from `uvicorn.error` and stdout. None of the following appears: the password, the email, the claim
code, the device key, the session JWT, the topic or the phone number (`::test_representative_backend_paths_log_no_credentials`).
The device side was reviewed by reading the code:

* the uplink prints reason codes, never keys or command bodies;
* BLE logs the SSID clipped with `%r` and never the passphrase, the setup code or the tag (M6 tests);
* the dashboard silences per-request logging, so `/stream?token=` is not logged on the device;
* `secret_store` messages name paths and fields, never values.

SSIDs are still logged at INFO, which is accepted: an SSID is not a credential.
`cloud/manage.py` and `scripts/ble_setup_code.py` print keys and codes by design. They are operator
provisioning output.

**Reviewed, unchanged by design (documented):**

* *JWT.* The TTL is 30 days with no refresh token. `exp`, `jti`, `uid` and `email` are required, and
  logout revokes the current `jti`. The app deletes `authToken` on sign-out and on any 401
  (`expireSession`). It keeps the device token and the ntfy topic until a *different* account signs
  in (`claimLocalState`), so that an expired session does not force a re-pair. On a shared phone,
  sign-out alone leaves the camera token in the Keychain.
* *Transport.* `NSAllowsArbitraryLoads: true` is unscoped, and it has to be: ATS has no key for LAN
  IP literals. Adding `NSAllowsLocalNetworking` makes iOS ignore `NSAllowsArbitraryLoads`, and the
  public-range campus addresses measured in `ios/project.yml` would then be blocked. The rule is enforced in the
  app: `BackendResolution.cleartextPermitted` refuses plain HTTP to any internet DNS name before a
  password or token leaves (`API.swift`, `BackendAPI.swift`, Settings).
* *MJPEG over http on the LAN.* The app sends `X-Device-Token` in a header, through an ephemeral
  session with no URL cache, and never in a URL. The token and the video still cross the Wi-Fi in
  cleartext. Anyone on the same network who can sniff it (open Wi-Fi, a shared PSK, a compromised
  AP) can take both. This needs device TLS, which is a production item.
* *`?token=` URLs.* `/stream` still accepts `?token=` (GET only, since M6) for the browser
  dashboard's `<img>`, and the browser dashboard and the legacy hotspot `/setup?token=` page put the
  token in the page URL, where it lands in browser history. The iOS app opens `/setup?token=` in
  Safari only on the legacy hotspot path. BLE is the default.
* *Device keys* are stored in plaintext in the backend's SQLite, which is 0600 (M8). The nmcli argv
  exposure via `/proc/*/cmdline` stands (§3).
* *Diagnostics.* `DiagnosticsView` shows the account email and backend address, and never a token.
  `SetupDebugPanel` shows the BLE phase, the device id and the credential *mode*, never a code,
  token or PSK. Neither copies or shares anything. No release-build change is needed.

**Evidence retention (report only; no user files changed).** A scan of the text files under
`~/sensordyme-audit/2026-11-10` found no real credentials. Counts and file names only are in
`evidence_retention_scan.json`. The findings:

* `scripts/sim/ios_uitest.sh` runs its throwaway stack *inside* the evidence directory
  (`--work "$OUT/stack"`) and leaves the following behind in `design-evidence/*`,
  `m8-evidence/ios/*` and `m81-evidence/ios/*`:
  * `backend/.jwt_secret` and `backend/sensordyme.db`, both 0600;
  * `backend/factory.json`, 0644, holding device keys, claim codes and the fixed demo password;
  * `device/config.yaml` and `identity.json`, 0644, holding the device key.

  These credentials are generated for each run and only ever served on a local sim backend, so
  they are worthless once it stops. If the evidence is ever shared, they are key material in
  readable files.
* The backend, device and xcodebuild logs hold sim claim codes. The xcresult screenshots show only
  `@sensordyme.local` demo accounts.
* A pytest failure trace in `m82-evidence` prints a fixture JWT signed by a temp test secret.
* `evidence/` and `m01-evidence/` (`freeze.txt`, `sdrun-state.json`) contain compute-fleet
  addresses (Tailscale 100.64/10 and campus addresses).
* The public `docs/audit` redaction (afee07a) holds: only placeholders (`$TOKEN`, `?token=…`)
  appear in `docs/`.

**For other streams (not changed here).** STREAM (`core/dashboard.py`): send
`Cache-Control: no-store` on `/api/*` JSON and `/stream`, and `Referrer-Policy: no-referrer` on the
`/` and `/setup` pages, whose URLs carry `?token=`. IOS-PRODUCT: refuse redirects that change host or
drop to `http` in `APIClient` (a `URLSessionTaskDelegate`), because `cleartextPermitted` checks only
the first URL. Optionally, clear `deviceToken` on an explicit `signOut()` but not on `expireSession`.
Sim tooling: run `uitest_control.py --work` under `mktemp -d` and copy only `*.log` into `$OUT`,
or delete `stack/backend/{.jwt_secret,sensordyme.db,factory.json}` and `stack/device/config.yaml`
at the end of `ios_uitest.sh`.
