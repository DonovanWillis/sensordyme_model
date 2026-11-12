# Device authentication contract — `sd-auth-v1`

Mission 6.5, checkpoint B/4A. This closes the M6 security blocker "commands are
not signed end-to-end" (`docs/SECURITY_REVIEW_M6.md` §3–4), in software, for
the uplink between a device (`core/uplink.py`) and the backend (`cloud/app.py`).
The code for both sides is `core/command_auth.py`.

**Status: simulator-verified only.** It is tested end to end on a Mac and on
oryn: the real uplink against the real backend app (`tests/test_command_auth_*.py`).
It has not run on a Pi or a deployed backend, and no key has been rotated.
M8 additions (replay state that survives a restart, signed clock answers,
revocation the app can see, and per-device enforcement) are tested in
`tests/test_m8_security_*.py`. They were also run live against uvicorn
backends (`docs/SECURITY_REVIEW_M6.md` §5).

## 1. Threats addressed

| Threat | Before | With `required` on both ends |
|---|---|---|
| Passive observer on a plain-`http://` path reads `X-Device-Key` and then impersonates the device or the backend | possible | the key never travels; only a per-request HMAC tag does |
| Active man-in-the-middle injects `set_preset` / `set_rules` / `set_protection` | possible | refused: no valid envelope (`missing_signature`, `bad_signature`) |
| Command substitution (valid envelope, different command) | n/a | refused: types and canonical body hash are signed (`type_mismatch`, `body_mismatch`) |
| Re-targeting (a command for device A delivered to device B) | n/a | refused: per-device keys plus the signed `dev` field (`unknown_key`, `wrong_device`) |
| Replay of an old response, including an old stop | partly: M6 S3 suppressed repeated protection ids | refused without a clock: the envelope is bound to the nonce of the request it answers (`stale_response`), and command ids are remembered (`duplicate_command`) |
| Replay of a captured device request | possible | refused: ±300 s window on the backend clock plus a nonce cache (`clock_skew`, `replayed_nonce`) |
| Malformed or future-version messages | M6 S4 for some fields | refused before interpretation (`malformed`, `unsupported_version`) |

Not addressed: a compromised backend or backend database (it holds the
symmetric key), denial of service (an on-path attacker can always drop
traffic), and BLE provisioning, which checkpoint 4B covers separately.

## 2. Keys

The per-device root secret is the existing device key (`cloud.device_key`,
issued by `cloud/manage.py`). Nothing new has to be provisioned. Derived
values use HKDF-SHA256 (RFC 5869, tested against its appendix A vectors), with
salt `"sensordyme/sd-auth-v1"`:

| info | length | use |
|---|---|---|
| `key-id` | 8 bytes → `kid`, 16 hex | public. Names the key in headers, logs and the `devices.auth_kid` column |
| `request-mac` | 32 | HMAC-SHA256 key for device → backend requests |
| `command-mac` | 32 | HMAC-SHA256 key for backend → device envelopes |

A key under 16 bytes cannot derive these keys and can only use legacy bearer
auth. Keys from `manage.py` are 51 bytes. Tags are compared with
`hmac.compare_digest`.

## 3. Requests (device → backend)

Header: `X-SD-Auth: sd-auth-v1 kid=<16 hex>,ts=<unix s>,nonce=<32 hex>,sig=<64 hex>`

`sig = HMAC(request-mac, "sd-auth-v1\nrequest\n<kid>\n<METHOD>\n<path>\n<sha256(body bytes)>\n<nonce>\n<ts>")`

* `path` is the **logical** endpoint, `/v1/ingest`, not the URL's own path, so a
  reverse proxy that strips a prefix does not break the tag. The backend
  removes any ASGI `root_path`.
* The device serialises the JSON body itself and signs those exact bytes.
* The backend looks the device up by `kid`, either current or previous key. It
  then checks the tag, then `|ts − backend clock| ≤ 300 s`, then that the nonce
  is new. Nonces are kept in the backend database (table `auth_nonces`, M8) for
  600 s and bounded at 1,000,000 rows. A backend restart therefore does not
  reopen the replay window, and every process on the same database shares them.
  (Until M8 they lived in process memory: a request captured up to 300 s before
  a restart was accepted again after it.)
* In `required` mode the request carries **no** `X-Device-Key`.

## 4. Command envelope (backend → device)

The ingest response is `{"ok", "stored", "command", "command_auth"}`.
`command_auth` is present only when the request asked for it, either with a
signed request or with `X-SD-Nonce` alongside the bearer key, and only when
there is a command.

```
{"v": "sd-auth-v1", "kid": …, "dev": "SD-XXXXXXXX" | "kid:<kid>",
 "types": ["set_protection", …], "body_sha256": …, "cid": <24 hex>,
 "nonce": <32 hex>, "rn": <the request's nonce>, "iat": <unix s>, "sig": <64 hex>}
sig = HMAC(command-mac, "sd-auth-v1\ncommand\nkid\ndev\ntypes(comma)\nbody_sha256\ncid\nnonce\nrn\niat")
```

* `body_sha256` is the SHA-256 of the canonical command JSON (sorted keys, no
  spaces, ASCII, no NaN). Both sides recompute it from the parsed object, so
  transport formatting does not matter. It covers every key in `command`,
  including `rules_version`, `rules_instruction` and `zones_version`.
* `dev` is the hardware id reported in **this** request's health when it is
  well formed, otherwise `kid:<kid>`. The device accepts its own reported id or
  `kid:` with its current or previous kid.
* The device verifies the checks in this order: shape and version, key id
  (revoked or unknown), tag, device, request nonce, types, body hash, then the
  duplicate `cid`. Nothing unauthenticated is interpreted beyond its shape. A
  refused command is dropped whole: none of its parts are applied. It is not a
  connectivity failure, because the events in that request were delivered.
* Field formats are matched in full (`fullmatch`), so a newline cannot make the
  newline-joined signing string ambiguous.

## 5. Replay and clocks

* **Device side: no clock needed.** A Pi has no RTC and may boot with an old
  clock until NTP syncs. Freshness of a command is proven by `rn`: an envelope
  answers exactly one request, whose nonce the device generated a moment ago.
  `iat` is signed and kept for audit, but never judged against the device clock.
* **Backend side: the backend clock is authoritative.** A request outside
  ±300 s is refused with `401 {"code": "clock_skew", "server_time": …}`. In
  `required` mode the device then timestamps its next requests with that
  offset. The offset is used for request timestamps only, never applied to the
  system clock, and never used to judge commands. A forged `server_time` can
  only cause refusals, which is denial of service, and an on-path attacker can
  do that anyway.
* **The time in a `clock_skew` answer is signed (M8).** The answer carries
  `clock_auth = {"v", "kid", "rn", "t", "sig"}`, where
  `sig = HMAC(command-mac, "sd-auth-v1\nclock\n<kid>\n<rn>\n<t>")`. `rn` is the
  refused request's nonce. The backend can always sign it, because the tag is
  checked before the clock. A `required` device adopts `t` only when this
  verifies. Otherwise it keeps its own clock and says why once
  (`unsigned_clock`, `bad_signature`, `stale_response`, …). Before M8, the
  plain `server_time` was adopted as is. An on-path attacker could then push
  the device's request timestamps into the future, hold the signed requests,
  and deliver them later inside the window, with nonces the backend had never
  seen. `server_time` is still sent, for older devices.
* M6's protection-id memory (`_accept_protection`, 64 ids) still applies, so a
  re-delivered stop is not applied twice.

## 6. Rotation, revocation, recovery

Rotation is an operator procedure. **No key has been rotated by this work.**

1. `python cloud/manage.py rotate-key <device id>` issues a new key. The old
   key moves to `previous_device_key`; it is still accepted for **signed**
   requests, and the backend signs replies with whichever key the request used.
   A second rotation is refused while one is open.
2. On the device, set `cloud.device_key: <new>` and
   `cloud.previous_device_key: <old>`, or rewrite the key file
   (`docs/CREDENTIAL_STORAGE.md`). Then restart.
3. `python cloud/manage.py revoke-previous-key <device id>` closes the
   rotation. A device can also refuse a key id locally with
   `cloud.revoked_key_ids: [<kid>]`.

**Revocation (M8).**
* A revoked key is remembered in `revoked_device_keys`. Nothing that grants
  access reads that table. A request made with a revoked key, signed or bearer,
  is refused as `device_auth_revoked_key` instead of `unknown_key` or
  `invalid device key`. The refusal is attributed to the device only when the
  request proves it holds that key, either a valid tag or the exact bearer key.
* `python cloud/manage.py revoke-device <device id>` refuses every request from
  the device (`device_auth_revoked_device`), and revokes its previous key too.
  It therefore cannot ingest, and it receives no command, because commands
  travel only in ingest answers.
* `rotate-key` on a revoked device re-instates it with a fresh key. The revoked
  key stays refused.
* **The app sees it.** `GET /v1/devices` carries
  `credentials: {state, auth, signature_required, refusal}`:
  * `state` is `accepted`, `refused`, `revoked` or `unknown`;
  * `auth` is how the device last authenticated, `signed` or `bearer`;
  * `refusal` is `{reason, detail, at}`.
  
  A stale device's `status.reasons` also names the refusal. Only refusals that
  prove the key holder is calling are recorded: `revoked_key`,
  `revoked_device`, `clock_skew` (the tag was valid) and `signature_required`
  (the bearer key matched). A forged tag under a public kid is never recorded.
  A refusal disappears once a later report is accepted. Writes are limited to
  one per device per reason per minute.

Legacy bearer auth (`off`/`permissive` devices) uses only the current key, so
for such a device steps 1–2 must be done together.

**Recovery:**
* **Device lost its key file:** the device runs standalone and the uplink is
  off with a reason. Restore `<file>.bak`
  (`scripts/migrate_device_key.py --restore-backup`) or re-issue with
  `rotate-key`. Re-issuing needs the device's owner or physical access.
* **Key suspected leaked:** `rotate-key`, update the device, then
  `revoke-previous-key` right away.
* **Device clock wildly wrong:** self-heals through `clock_skew` (§5).

## 7. Modes, defaults and compatibility

| Device `cloud.command_auth` | Sends | Applies |
|---|---|---|
| `off` | `X-Device-Key` only: byte-for-byte the pre-M6.5 request | everything, unsigned |
| `permissive` (**default**) | `X-Device-Key` + `X-SD-Nonce` | valid signed commands; unsigned commands from a backend that does not sign (said once). **Never** an envelope that fails verification |
| `required` | `X-SD-Auth` only, with the key never sent | valid signed commands only |

An unknown value fails closed: it is treated as `required`.

| Backend `SENSORDYME_DEVICE_AUTH` | Accepts | Signs |
|---|---|---|
| `off` | bearer only | never |
| `permissive` (**default**) | bearer and signed requests | whenever the request carries a nonce |
| `required` | signed requests only | always (when there is a command) |

An unknown value stops the server at import.

**Per device (M8).** Run `python cloud/manage.py require-signed <device id>`.
That device must then sign whatever the backend mode is. A per-device setting
can only tighten the backend mode, never loosen it. The command is refused
until the device has been seen signing (`credentials.auth == "signed"`),
unless `--force` is given. `allow-bearer <device id>` undoes it. Before M8,
enforcement was all or nothing on the backend.

**Compatibility:** every existing device and backend keeps working with both
defaults. A legacy request, with no nonce, gets the exact legacy response.

M8 checked this live: real uvicorn backends, with virtual devices over HTTP.
Every pairing was run: device {11c524c, M8} × {no setting, off, permissive,
required}, against backend {11c524c, M8} × {off, permissive, required}, plus
M8 permissive with a per-device `required`. Evidence is in
`~/sensordyme-audit/2026-11-10/m8-evidence/security/compat_matrix.json`.
Every cell behaves identically on old and new code except one, which is
deliberate:

* An M8 `required` device behind an 11c524c backend still works. But it can no
  longer recover from a clock more than 300 s off, because that backend's
  `clock_skew` answer is unsigned.
* Migration step 1 (backend first) avoids this pairing.
* An 11c524c `required` device behind an M8 backend recovers as before, using
  `server_time`.

**Production gate:** `scripts/deploy_preflight.py` refuses a device config
with `command_auth: off` and warns on `permissive`. The production target is
`required` on both ends plus HTTPS (M6 blocker 2). `permissive` does not stop
an active man-in-the-middle, who can strip the envelope.

**Migration order:**
1. Deploy the backend (default `permissive`).
2. Move devices to `required` one at a time. Rotate each device's key when it
   moves, because its old key has travelled as a bearer in the past.
3. Set the backend to `required`.

Rollback is the reverse; each step is a config value.

## 8. Refusal codes

On the backend these are answered as `401 {"code": "device_auth_<reason>"}`,
or `clock_skew`, and logged with the reason and kid only. On the device they
are printed as `refused a remote command (<reason>)`, once per change of
reason, and counted in `CloudUplink.command_auth_rejections`.

`malformed`, `unsupported_version`, `unknown_key`, `revoked_key`,
`bad_signature`, `wrong_device`, `stale_response`, `type_mismatch`,
`body_mismatch`, `duplicate_command`, `missing_signature`, `no_usable_key`,
`clock_skew`, `replayed_nonce`, `signature_required`, `disabled`,
`revoked_device` (M8, backend), and `unsigned_clock` (M8, device: a
`clock_skew` time it would not adopt).

**Logging rule** (enforced by tests): no device key, derived key or tag is
ever logged, printed, put in an exception message or shown in a repr.

## 9. Limitations

* The key is symmetric, so the backend database holds it in plaintext (M6
  blocker 3). Hashing it at rest is incompatible with HMAC verification. The
  next step is an asymmetric scheme (Ed25519 via `cryptography`, which is not
  a dependency today) or a KMS-held key.
* Request nonces are in the backend's SQLite database (M8), so they survive a
  restart and are shared by processes on one database. A backend on several
  hosts would need a shared store. At the 1,000,000 bound, the oldest unexpired
  nonces are evicted, with a warning. That takes more than 1,600 signed
  requests a second.
* The device keeps no replay state on disk, and needs none. A command envelope
  is bound to the nonce of the request it answers, and nonces are random per
  request, so an envelope captured before a device restart answers a request
  the restarted device never sends. Tested in
  `tests/test_m8_security_device.py` and `tests/test_m8_security_e2e.py`.
* `permissive` devices still obey an unsigned command. An on-path attacker can
  strip the envelope. Setting that device's own `cloud.command_auth: required`
  closes this. The backend's per-device `required` (§7) then also stops its
  bearer key from being accepted, one device at a time.
* The iOS app does not take part in this contract. App → backend stays JWT
  over HTTPS.
* `/v1/ingest` is the only device endpoint. Any new device endpoint must use
  the same `current_device` dependency.
* Not validated on a Pi, a deployed backend or real NTP-less boot conditions.
