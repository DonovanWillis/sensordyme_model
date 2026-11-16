# Local access: the iPhone talks to the camera directly (sd-local-v1)

**Status:** implemented and tested in software (M12). Nothing has been run on a
Pi or a physical iPhone yet. The hardware gates are in §10.

## 1. What problem this solves

The device already serves its live feed, state, history and stop/resume/preset
on its own LAN port (`core/dashboard.py`, port 8899). Before M12 the only way
in was **one shared `dashboard.token`**:

* someone had to copy it from `config.yaml` into the app by hand;
* it never expired;
* it could not be revoked for one lost phone;
* the shipped config leaves it unset, so a new customer's camera refused the
  app entirely.

The app also reached the camera only through the backend's device list
(`docs/PORTABLE_OPERATION.md` C1–C4). With the MacBook backend off, a cold
launch showed nothing.

sd-local-v1 gives **each phone its own credential**, enrolled through a path
that proves ownership. After that, the phone reaches the camera on any shared
network with no backend, no internet and no token.

## 2. Keys and identities

| Thing | Where | What |
|---|---|---|
| Phone key | iPhone Keychain. Secure Enclave when available (not in the Simulator) | P-256 signing key. **Never leaves the phone.** |
| `client_id` | derived | first 16 hex of SHA-256(phone public key, X9.63 uncompressed, 65 bytes). Nobody can choose an id for a key they do not hold. |
| Device identity | `data/local_identity.pem`, mode 0600, generated on first start (`core/local_auth.py` `DeviceIdentity`) | P-256 key. The phone learns its **public** key at enrolment and checks the device's signature on every session. A loose, linked or foreign-owned file is **refused, never replaced**: replacing it would silently break every phone's trust. |
| Enrolled phones | `data/local_clients.json`, mode 0600, atomic writes | public key, name, source (`ble`, `account`, `token`), enrolled_at, last_seen. At most 16. A row whose id does not match its key is dropped. A loose file trusts **no** phone. |
| `device_key_fp` | derived | 16 hex of SHA-256(device public key). Shown, never secret. |

Config (`local_access:` in `config.yaml`). All keys are optional; the defaults are shown:

```yaml
local_access:
  enabled: true            # false: only the legacy token opens /api and /stream
  idle_expiry_days: 180    # a phone not seen for this long must enrol again (0 = never)
  # dir / identity_path / clients_path / events_path / events_max: default beside data/events.csv
```

## 3. Wire protocol

All binary values are **base64url without padding**. Hex is lowercase.

### 3.1 Challenge (no auth)

`GET /api/local/hello` →

```json
{"contract":"sd-local-v1","device_id":"SD-1A2B3C4D","device_key_fp":"6fc66a90486171ca",
 "challenge":"<32 hex>","challenge_ttl_s":60,"token_enrollment":false}
```

* A challenge is **single-use** and lives 60 s (device monotonic clock).
* At most 256 are outstanding; the oldest is dropped.
* Every signature covers one challenge, so replay protection needs **no clock**.
  A Pi with no RTC and no NTP still refuses a replay.

### 3.2 Signed request (every write)

The phone signs, with ECDSA P-256 / SHA-256 (raw `r||s`, 64 bytes), the UTF-8
bytes of:

```
sd-local-v1\nrequest\n<device_id>\n<client_id>\n<challenge>\n<METHOD>\n<path>\n<sha256 hex of the exact body bytes>
```

* `path` is the URL path with no query string.
* An empty body hashes as `sha256("")`.

It sends this header:

```
X-SD-Local: sd-local-v1 client=<client_id>,challenge=<challenge>,sig=<b64url 64 bytes>
```

### 3.3 Session (reads and the stream)

1. The phone sends a signed `POST /api/local/session`, body `{"nonce":"<32 hex, phone-random>"}`.
2. The device answers:

   ```json
   {"contract":"sd-local-v1","session":"<43 chars>","expires_in_s":900,"client_id":"...",
    "device_id":"...","device_key_fp":"...","device_sig":"<b64url>"}
   ```

3. `device_sig` is the **device** key's signature over:

   ```
   sd-local-v1\nsession\n<device_id>\n<client_id>\n<challenge>\n<nonce>\n<sha256 hex of the session string>\n<expires_in_s>
   ```

   The phone **must** verify it against the device public key it stored at
   enrolment, and refuse the session if it fails (a look-alike on the LAN).
4. Reads send `X-SD-Session: <session>`. It works on GET `/api/*` and `/stream`
   **only**; a session never authorises a write.

Session rules:

* bound to the phone and to the peer IP address;
* 15 minutes; the app renews before expiry;
* expiry is checked when a stream connects;
* **revocation ends an open stream at once**.

### 3.4 Endpoints

| Endpoint | Auth | Notes |
|---|---|---|
| `GET /api/local/hello` | none | challenge + fingerprint only |
| `POST /api/local/session` | signed | opens a session |
| `POST /api/local/enroll` | legacy `X-Device-Token` only | migration path (§4.3); body `{"public_key","name"}` → 201 `{"client_id","device_id","device_public_key","device_key_fp","contract"}` |
| `GET /api/local/clients` | session or token | `{"clients":[{client_id,name,source,enrolled_at,last_seen,key_fp}],"this_client"}` |
| `POST /api/local/revoke` | signed or token | body `{"client_id"}`; any enrolled phone (they are all the owner's) |
| `GET /api/state`, `/api/health`, `/api/presets`, `/api/history`, `/api/stream/stats` | session or token | unchanged bodies |
| `GET /api/events` | session or token | §6 |
| `GET /stream` | session or token | unchanged MJPEG; camera privacy rules unchanged (camera off while stopped unless the owner turned on camera-only) |
| `POST /api/protection` | **signed** or token | unchanged body; a stop still needs `"confirm": true`; recorded as `by: "local_app"` |
| `POST /api/preset` | **signed** or token | unchanged; the device's **role floor still refuses** an unsafe preset (409) |
| `POST /setup/wifi` | signed or token | unchanged |
| `GET /api/diagnostics` | token only | unchanged (developer feature) |

Refusals are JSON `{"error", "code"}`. The codes:

| Status | Code | Meaning | What the app does |
|---|---|---|---|
| 401 | `missing_signature`, `malformed_signature`, `stale_challenge`, `bad_signature` | the request failed authentication | fetch a new challenge, retry once |
| 401 | `session_invalid` | session expired, wrong peer, or the phone was revoked | open a new session |
| 403 | `unknown_client` | not enrolled, revoked, or idle-expired | **forget the local credential** and say "this iPhone needs to be paired again" |
| 429 | `locked_out` (+ `retry_after_s`) | 10 failures from one address in 60 s; doubles per lockout up to 1 h | wait `retry_after_s` |
| 404 | `local_access_off` | the device has no local access | fall back to the account path |
| 409 | `too_many_clients` | the device already trusts 16 phones | revoke one first |
| 401 | `token_required` | `/api/local/enroll` without the legacy token | use Bluetooth or the account instead |

## 4. Enrolment: three authorised paths, never "being on the network"

### 4.1 Bluetooth, inside a proven PoP session

The phone proves the setup code printed on the label (`services/ble_pop.py`,
`docs/BLE_PROOF_OF_POSSESSION.md`). After `proven`, and before or instead of
Wi-Fi credentials:

```
app    -> {"op":"enroll","iv":<12 bytes b64>,"ct": AES-256-GCM(K_enc, iv, canonical{"name","public_key"}, aad = "local-enroll" || T)}
device -> {"op":"enrolled","iv":<12 bytes b64>,"ct": AES-256-GCM(K_enc, iv', canonical{"client_id","device_id","device_public_key"}, aad = "local-enrolled" || T)}
```

* Canonical JSON means sorted keys, no spaces, UTF-8. BLE messages use
  standard base64; the fields inside are base64url.
* One enrolment per session. The state stays `proven`, so credentials may follow.
* Refusals are `{"op":"error","error":"refused","code":<LocalAuthError reason>}`
  or `unsupported` (no local access).
* Works with no network at all. It needs the Bluetooth window to be open: the
  device is offline, within `reconfigure_window_s` after boot, or has
  `always_available` set.

### 4.2 The owner's account (backend relay)

For a phone that is signed in, and a device already claimed and reporting to
the backend:

1. The app sends `POST /v1/devices/{id}/local-clients {"public_key","name"}`.
   This is owner-only.
2. The backend stores the **public** key (`local_clients` table) and adds
   `set_local_clients: {"add":[{public_key,name}], "revoke":[client_id]}` to
   the next ingest response, under an sd-auth-v1 signed envelope.
3. The device **refuses an unsigned add**, whatever `cloud.command_auth` says.
   A revoke is the safe direction and is accepted like any other command.
4. Convergence: the device reports `health.local_access =
   {contract, device_public_key, device_key_fp, clients:[ids], fault}`. The
   backend resends until the report matches.
5. `GET /v1/devices` returns `local_identity: {contract, device_public_key,
   device_key_fp, reported_at}`. This is kept when health is stale, because it
   names the camera; it does not claim it is healthy.
6. `GET /v1/devices/{id}/local-clients` returns `on_device` per phone.
   `DELETE …/local-clients/{client_id}` revokes one.
7. A new owner's claim asks the device to revoke every phone it last reported.
   The claiming app then re-registers its own key (path 4.2).

The backend holds only public keys. **It cannot act on the LAN.**

### 4.3 The legacy token (migration)

An install that already set `dashboard.token` can enrol with
`POST /api/local/enroll` plus `X-Device-Token`. The token itself keeps
working exactly as before. The app should stop storing it once it has enrolled.

## 5. Credential lifecycle

| Stage | Behaviour |
|---|---|
| Provisioning | §4. The phone generates its key once per device pairing (or once per install; either is valid) |
| Storage | phone: Keychain (`kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`, never synced to iCloud). Device: 0600 files |
| Renewal | sessions last 15 min; the app opens a new one before expiry or on `session_invalid` |
| Expiration | sessions: 15 min. Phones: idle for `idle_expiry_days` (default 180) |
| Revocation | `POST /api/local/revoke` from any enrolled phone; the account `DELETE`; `scripts/local_clients.py revoke-all` on the device. Sessions and open streams end at once |
| Reconnection | the app finds the device by Bonjour (or its last-known address), fetches a challenge and opens a session. An IP change only needs a new session |

## 6. Local event history: `GET /api/events`

Schema `sd-local-events/1`, from `core/local_events.py`:

* `?limit=N` (≤500): the newest N rows, newest first.
* `?after=<seq>&limit=N`: the rows with `seq > after`, oldest first. This is
  what a foreground app polls for new alerts.
* Response: `{"schema","events":[…],"latest_seq","boot_id","retention":{"max_events","kept","oldest"},"fault"}`.
* Each event: `seq, event_key, type, timestamp, severity, confidence, signals,
  duration, experimental, simulated, alert, age_s, sync`.
  * `event_key` is the **same id the backend stores** (`Event.event_id`), so
    the app merges local and account history with no duplicates.
  * `age_s` is the device's monotonic age of the event. It is `null` for an
    event from a previous boot.
  * `sync` is `pending` (still in the uplink backlog), `sent`, `local_only`
    (no uplink configured), or `unknown`.
* Bounded: 2,000 rows, about 1 MB, compacted atomically.
* Privacy: the same metadata the uplink already carries. **No frame or image,
  ever.**

## 7. What the app must do (the truthful rules)

* A cached device is **never** shown as protecting from the cache. Only a
  fresh `/api/state` or `/api/health` read, or the backend's fresh status,
  says what it is doing. A device that does not answer is shown offline or
  stale, with the age of the last report.
* A local stop is confirmed from the device's next `/api/state`
  `policy.protection`, never assumed from the 202.
* A 403 `unknown_client` forgets the local credential. The app then says this
  iPhone must be paired again; it does not retry for ever.
* A local event plays a sound only under the M11 rules (`AlertSound.decide`):
  * never for a simulated, experimental or log-only event;
  * never for an event older than 60 s by `age_s`;
  * never twice for the same `event_key`, whether it arrived locally or from
    the account.

## 8. Security analysis and limits

* **In scope:**
  * A stranger on the same Wi-Fi gets nothing without an enrolled key: not the
    feed, not commands.
  * A sniffer cannot forge or replay a command.
  * A look-alike device cannot pose as the camera to an enrolled phone.
  * A lost phone can be revoked alone.
  * The backend cannot act locally.
* **Plain HTTP on the LAN.** Anyone on the same network can read a session
  token and the MJPEG frames in transit, for the session's lifetime (bound to
  the peer IP, so they must also spoof that address). They cannot command.
  TLS with a pinned device certificate is the fix; not built.
* **Unauthenticated hello.** A LAN host can flood it and evict challenges (a
  nuisance DoS). It learns only the device id (already in Bonjour) and a
  fingerprint.
* **Ownership transfer** clears phones only through a claim (§4.2 item 7) or
  `scripts/local_clients.py revoke-all`. A device moved between accounts
  without either keeps trusting the old phones.
* **The legacy token** remains a shared credential while it is set. Removing
  it from `config.yaml` once every phone has enrolled is the owner's choice.

## 9. Recovery on the device

```bash
cd ~/sensordyme && .venv/bin/python scripts/local_clients.py list
cd ~/sensordyme && .venv/bin/python scripts/local_clients.py revoke <client_id>
cd ~/sensordyme && .venv/bin/python scripts/local_clients.py revoke-all
```

The tool edits the store file. The running service notices the change at its next local
request: the revoked phone's sessions and open stream end. No restart is needed.

## 10. Tests and hardware gates

* Device: `tests/test_m12_local_access.py`. It exercises the real
  `DashboardServer` on a LAN bind with no token (the shipping default).
* BLE: `tests/test_m12_ble_enroll.py`.
* Events: `tests/test_m12_local_events.py`.
* Backend relay: `tests/test_m12_local_access_backend.py`.
* Vectors for Swift: `tests/local_access_vectors.json`
  (`scripts/local_access_vectors.py --check`).
* **Hardware gates, not yet run:**
  1. A physical iPhone enrols over real Bluetooth.
  2. It streams and stops/resumes on home Wi-Fi with the MacBook off.
  3. The same on an iPhone Personal Hotspot.
  4. An IP change and a device power cycle are survived.
  5. A revoked phone is cut off mid-stream.
  6. The Secure Enclave key works on the physical phone.
