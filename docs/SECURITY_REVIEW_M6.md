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
