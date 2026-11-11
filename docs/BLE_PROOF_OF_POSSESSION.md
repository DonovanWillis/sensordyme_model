# Bluetooth provisioning: proof of possession (M6.5, 4B)

Closes the "BLE provisioning is unauthenticated" production blocker in
`docs/SECURITY_REVIEW_M6.md` for the **credential write**, behind config. Status:
**software-ready, simulator/vector-verified, hardware-pending**. Nothing here was run
over a physical Bluetooth link.

## What changes

| | `auth: legacy` (default today) | `auth: pop` |
|---|---|---|
| Who can hand the device a network | anyone in range while the window is open | only a phone that proves it holds the device's setup code |
| Credentials on the air | cleartext JSON on 5d1e0003 | AES-256-GCM, per-session key, on 5d1e0007 |
| 5d1e0003 (plain write) | accepted | refused (logged) |
| Phone checks the device | no | yes: the device proves the code back before anything secret is sent |
| `deployment: production` | **refused** (schema error; the window stays shut) | allowed |

The setup code is 12 Crockford base32 symbols (60 bits), printed as `XXXX-XXXX-XXXX`
and as a QR code carrying `SDSETUP:1:<device id>:<code>`. The app connects only to
the device whose device-info `id` matches the label. Being discoverable proves nothing.

## Protocol v1

The full description is in `services/ble_pop.py`. It uses PBKDF2-HMAC-SHA256 (600 000
iterations, salted with the device id) to derive the code key, HKDF-SHA256 over both
nonces for the session keys, HMAC-SHA256 proofs over a transcript (version, device id,
both nonces), and AES-256-GCM for the credentials. Established primitives only:
`cryptography` on the device, CryptoKit and CommonCrypto on iOS.

The device-side state machine is explicit: idle → advertising → challenged → proven
→ credentials_received → applied/failed. `TRANSITIONS` lists every legal move; any
other move raises.

- **Timeouts:** challenge 30 s, proof 120 s, join outcome 120 s.
- **Retries:** a wrong code spends its challenge, so each challenge allows one guess.
- **Lockout:** 5 failures lock the device out for 30 s, doubling per further failure up
  to 1 h. The lockout is held by the device, so reconnecting does not reset it.

## Device setup (factory / bring-up, on the device as the service user)

```
python scripts/ble_setup_code.py generate --out /etc/sensordyme/ble_setup_code --device-id <SD-…>
```

- The command writes the file atomically with mode 0600 and prints the label text and
  QR payload. Send that output to the label printer only, never to tickets or logs.
- It refuses to overwrite an existing code. There is no rotation command.
- The device refuses a code file that is group- or world-readable, a symlink, or owned
  by another user, and keeps Bluetooth shut until the file is fixed (health shows why).

Config (`network.provisioning`):

```yaml
auth: pop
setup_code_path: /etc/sensordyme/ble_setup_code
```

Optionally also set top-level `deployment: production`, which makes `auth: legacy` a
schema error.

## Rollout order (do not flip a device before the app can follow)

1. Ship an app build whose setup screen scans the label and uses
   `BLEProofOfPossession.Session` over characteristic 5d1e0007.
   - Not done: views and `BLEProvisioning.swift` belong to the iOS work stream.
   - Device info now carries `"auth": "legacy" | "pop"`, so the app can choose the
     flow per device.
2. Generate and print the codes for existing units.
3. Set `auth: pop` per device, then verify on hardware:
   - a wrong code is refused and the lockout engages;
   - a plain write to 5d1e0003 is refused;
   - a correct code joins.
4. Set `deployment: production`.

Rollback: set `auth: legacy` and leave `deployment` unset (development). No data
migration is involved.

## Verification

- Python: `pytest tests/test_ble_pop.py`. It covers the protocol, replay, tampering,
  lockout, timeouts, every illegal transition, the config gate, GATT refusal, the
  provisioning service end-to-end with a fake radio, and no secrets in logs. It also
  checks the shared vectors against hashlib/hmac.
- Swift: `ios/Tests/run.sh` (or concatenate `ios/Sources/BLEProofOfPossession.swift`
  and `ios/Tests/BLEProofOfPossessionTests.swift` and run with `swift`). This checks
  the same vectors (`tests/ble_pop_vectors.json`, embedded verbatim) byte for byte.
- To regenerate the vectors, run `python scripts/ble_setup_code.py vectors --out
  tests/ble_pop_vectors.json`, then re-embed them in the Swift test. The Python test
  fails until both copies agree.

## Known limits

- **Offline guessing is still possible.** Someone who records a session, or runs a fake
  device to collect a proof, can try codes offline. The 60-bit code combined with
  600 000 PBKDF2 iterations is the defence. A PAKE (SPAKE2/OPAQUE) would remove the
  attack, but no vetted library is available on both CryptoKit and the device.
- **Lockout denial of service.** A stranger in range can trigger the lockout, which
  costs the owner at most an hour, or a power cycle (the counter is in memory).
- **Link-layer pairing is still BLE "Just Works".** The protection is at the
  application layer.
- **Dependency.** `cryptography` must be on the device (`python3-cryptography` on
  Raspberry Pi OS, or pinned in `requirements-device.txt`; that file is the release
  stream's). Without it, `auth: pop` fails closed: Bluetooth stays shut.
- **Other open items.** `scripts/ble_provision_client.py` (the bleak test client) still
  speaks only legacy. The persistent device id in the BLE name and the PSK in nmcli
  argv remain as documented in the security review.
